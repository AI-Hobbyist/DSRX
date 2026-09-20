import pathlib

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from inference.ds_acoustic import DiffSingerAcousticInfer
from inference.ds_variance import DiffSingerVarianceInfer
from modules.toplevel import DiffSingerAllInOne
from training.acoustic_task import AcousticDataset, AcousticTask
from training.variance_task import VarianceDataset, VarianceTask
from utils.hparams import hparams
from utils.plot import spec_to_figure
from utils.training_utils import DsBatchSampler
from utils.variance_validation import (
    load_validation_sources, validate_variance_validation_config
)


class AllInOneTask(AcousticTask):
    def __init__(self):
        all_in_one_config = hparams.get('all_in_one', {})
        if not isinstance(all_in_one_config, dict) or not all_in_one_config.get('enabled', False):
            raise ValueError('AllInOneTask requires all_in_one.enabled: true.')

        self.val_with_variance = hparams.get('val_with_variance', {})
        self.val_with_variance_enabled = (
            isinstance(self.val_with_variance, dict)
            and self.val_with_variance.get('enable', False)
        )
        if self.val_with_variance_enabled:
            validate_variance_validation_config(
                self.val_with_variance, hparams['dictionaries']
            )
            hparams['val_with_vocoder'] = True

        hparams['predict_dur'] = True
        hparams['predict_pitch'] = True
        hparams['predict_energy'] = True
        hparams['predict_breathiness'] = True
        hparams['predict_voicing'] = True
        hparams['predict_tension'] = True

        self.use_spk_id = hparams['use_spk_id']
        self.use_lang_id = hparams['use_lang_id']
        self.predict_dur = True
        self.lambda_dur_loss = hparams['lambda_dur_loss']
        self.predict_pitch = True
        self.lambda_pitch_loss = hparams['lambda_pitch_loss']
        self.variance_prediction_list = ['energy', 'breathiness', 'voicing', 'tension']
        self.predict_variances = True
        self.lambda_var_loss = hparams['lambda_var_loss']
        self.all_in_one_samplers = {}
        self.variance_validation_language = None
        self.variance_validation_speakers = []
        super().__init__()

    def _build_model(self):
        return DiffSingerAllInOne(
            vocab_size=len(self.phoneme_dictionary),
            out_dims=hparams['audio_num_mel_bins']
        )

    def build_losses_and_metrics(self):
        AcousticTask.build_losses_and_metrics(self)
        VarianceTask.build_losses_and_metrics(self, register_metrics=False)

    def setup(self, stage):
        binary_data_dir = pathlib.Path(hparams['binary_data_dir'])
        self.acoustic_train_dataset = AcousticDataset(
            'train', data_dir=binary_data_dir / 'acoustic'
        )
        self.acoustic_valid_dataset = AcousticDataset(
            'valid', data_dir=binary_data_dir / 'acoustic'
        )
        self.variance_train_dataset = VarianceDataset(
            'train', data_dir=binary_data_dir / 'variance'
        )
        self.variance_valid_dataset = VarianceDataset(
            'valid', data_dir=binary_data_dir / 'variance'
        )
        if self.val_with_variance_enabled:
            speaker_names = self.acoustic_valid_dataset.metadata.get('spk_names', [])
            speaker_ids = self.acoustic_valid_dataset.metadata.get('spk_ids', [])
            self.variance_validation_speakers = list(dict.fromkeys(
                zip(speaker_names, speaker_ids)
            ))
        self.num_replicas = (self.trainer.distributed_sampler_kwargs or {}).get('num_replicas', 1)

    def _build_dataloader(self, name, dataset, training):
        sampler = DsBatchSampler(
            dataset,
            max_batch_frames=self.max_batch_frames if training else self.max_val_batch_frames,
            max_batch_size=self.max_batch_size if training else self.max_val_batch_size,
            num_replicas=self.num_replicas,
            rank=self.global_rank,
            sort_by_similar_size=hparams['sort_by_len'] if training else False,
            size_reversed=training,
            required_batch_count_multiple=hparams['accumulate_grad_batches'] if training else 1,
            shuffle_sample=training,
            shuffle_batch=training,
            disallow_empty_batch=training,
            pad_batch_assignment=training
        )
        if training:
            self.all_in_one_samplers[name] = sampler
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collater,
            batch_sampler=sampler,
            num_workers=hparams['ds_workers'],
            prefetch_factor=(hparams['dataloader_prefetch_factor'] if hparams['ds_workers'] > 0 else None),
            pin_memory=True,
            persistent_workers=(hparams['ds_workers'] > 0)
        )

    def train_dataloader(self):
        loaders = {
            'acoustic': self._build_dataloader(
                'acoustic', self.acoustic_train_dataset, training=True
            ),
            'variance': self._build_dataloader(
                'variance', self.variance_train_dataset, training=True
            )
        }
        return CombinedLoader(loaders, mode='max_size_cycle')

    def val_dataloader(self):
        return [
            self._build_dataloader('acoustic', self.acoustic_valid_dataset, training=False),
            self._build_dataloader('variance', self.variance_valid_dataset, training=False)
        ]

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        for sampler in self.all_in_one_samplers.values():
            sampler.set_epoch(self.current_epoch)

    def _training_step(self, sample):
        acoustic_losses = AcousticTask.run_model(
            self, sample['acoustic'], model=self.model.acoustic
        )
        variance_losses = VarianceTask.run_model(
            self, sample['variance'], model=self.model.variance
        )
        losses = {**acoustic_losses, **variance_losses}
        batch_size = sample['acoustic']['size'] + sample['variance']['size']
        return sum(losses.values()), {**losses, 'batch_size': float(batch_size)}

    def validation_step(self, sample, batch_idx, dataloader_idx=0):
        if sample['size'] == 0:
            return
        with torch.autocast(self.device.type, enabled=False):
            if dataloader_idx == 0:
                losses = AcousticTask.run_model(self, sample, model=self.model.acoustic)
            else:
                losses = VarianceTask.run_model(self, sample, model=self.model.variance)
        losses = {'total_loss': sum(losses.values()), **losses}
        for name, value in losses.items():
            self.valid_losses[name].update(value, weight=sample['size'])

    def _on_validation_start(self):
        super()._on_validation_start()
        if not self.val_with_variance_enabled or self.global_rank != 0:
            return
        speaker_map = {
            name: speaker_id for name, speaker_id in self.variance_validation_speakers
        }
        lang_map = {
            language: index for index, language in enumerate(hparams['dictionaries'])
        }
        self.variance_validation_infer = DiffSingerVarianceInfer(
            device=self.device, predictions=set(), model=self.model.variance,
            spk_map=speaker_map, lang_map=lang_map
        )
        self.acoustic_validation_infer = DiffSingerAcousticInfer(
            device=self.device, load_vocoder=False, model=self.model.acoustic,
            spk_map=speaker_map, lang_map=lang_map
        )

    @torch.inference_mode()
    def _on_validation_epoch_end(self):
        if (
                not self.val_with_variance_enabled
                or self.global_rank != 0
                or not self.variance_validation_speakers
        ):
            return
        language, source = load_validation_sources(
            self.val_with_variance, hparams['dictionaries'],
            previous_language=self.variance_validation_language
        )
        self.variance_validation_language = language
        timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        for speaker_name, _ in self.variance_validation_speakers:
            param = dict(source)
            if hparams['use_spk_id']:
                param['ph_spk_mix'] = param['spk_mix'] = {speaker_name: 1.0}
            variance_batch = self.variance_validation_infer.preprocess_input(param)
            durations, pitch, variances = self.variance_validation_infer.forward_model(
                variance_batch
            )
            param['ph_dur'] = ' '.join(
                str(round(value * timestep, 6)) for value in durations[0].cpu().tolist()
            )
            f0 = 440.0 * torch.pow(2.0, (pitch[0] - 69.0) / 12.0)
            param['f0_seq'] = ' '.join(str(round(value, 1)) for value in f0.cpu().tolist())
            param['f0_timestep'] = str(timestep)
            for name, values in variances.items():
                param[name] = ' '.join(
                    str(round(value, 4)) for value in values[0].cpu().tolist()
                )
                param[f'{name}_timestep'] = str(timestep)

            acoustic_batch = self.acoustic_validation_infer.preprocess_input(param)
            mel = self.acoustic_validation_infer.forward_model(acoustic_batch)
            waveform = self.vocoder.spec2wav_torch(mel, f0=acoustic_batch['f0'])
            tag = f'variance_validate/{speaker_name}'
            title = f'{language} - {speaker_name} - Variance Validate'
            self.logger.all_rank_experiment.add_audio(
                f'{tag}/audio', waveform,
                sample_rate=hparams['audio_sample_rate'], global_step=self.global_step
            )
            figure = spec_to_figure(
                mel[0], hparams['mel_vmin'], hparams['mel_vmax'], title
            )
            self.logger.all_rank_experiment.add_figure(
                f'{tag}/mel', figure, global_step=self.global_step
            )
            plt.close(figure)