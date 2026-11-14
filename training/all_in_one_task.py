import pathlib
import warnings

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from inference.ds_acoustic import DiffSingerAcousticInfer
from inference.ds_variance import DiffSingerVarianceInfer
from modules.toplevel import DiffSingerAllInOne
from training.acoustic_task import AcousticDataset, AcousticTask
from training.variance_task import VarianceDataset, VarianceTask
from utils.hparams import hparams, mini_ddsp_config
from utils.plot import spec_to_figure
from utils.training_utils import DsBatchSampler
from utils.variance_validation import (
    iter_validation_sources, validate_variance_validation_config
)


class AllInOneTask(AcousticTask):
    training_stream_count = 2

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
            if not hparams['predict_dur'] or not hparams['predict_pitch']:
                raise ValueError(
                    'val_with_variance requires predict_dur and predict_pitch.'
                )
            hparams['val_with_vocoder'] = True

        self.use_spk_id = hparams['use_spk_id']
        self.use_lang_id = hparams['use_lang_id']
        self.predict_dur = hparams['predict_dur']
        if self.predict_dur:
            self.lambda_dur_loss = hparams['lambda_dur_loss']
        self.predict_pitch = hparams['predict_pitch']
        if self.predict_pitch:
            self.lambda_pitch_loss = hparams['lambda_pitch_loss']
        self.variance_prediction_list = [
            name for name in ('energy', 'breathiness', 'voicing', 'tension')
            if hparams[f'predict_{name}']
        ]
        self.predict_variances = bool(self.variance_prediction_list)
        self.lambda_var_loss = hparams['lambda_var_loss']
        self.all_in_one_samplers = {}
        self.variance_validation_language = None
        self.variance_validation_speakers = []
        self.mini_ddsp_settings = mini_ddsp_config(hparams)
        super().__init__()
        self._joint_log_sums = {}
        self._joint_log_count = 0
        self._joint_log_names = ["ddsp_joint_loss"]
        if self.use_shallow_diffusion and self.train_aux_decoder:
            self._joint_log_names.append("ddsp_aux_joint_loss")

    def _build_model(self):
        return DiffSingerAllInOne(
            vocab_size=len(self.phoneme_dictionary),
            out_dims=hparams['audio_num_mel_bins']
        )

    def load_finetune_ckpt(self, state_dict):
        if not self.mini_ddsp_settings.get("enabled", False):
            return super().load_finetune_ckpt(state_dict)
        expected = self.state_dict()
        missing = set(expected) - set(state_dict)
        unexpected = set(state_dict) - set(expected)
        vocoder_keys = {key for key in expected if key.startswith("model.vocoder.")}
        if unexpected or (missing and missing != vocoder_keys):
            raise RuntimeError(
                f"Mini-DDSP initialization only permits a complete new vocoder; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        initialized = dict(state_dict)
        for key in sorted(missing):
            initialized[key] = expected[key]
            print(f"| Initialized new Mini-DDSP state: {key}")
        self.load_state_dict(initialized, strict=True)

    def _build_vocoder(self):
        if self.mini_ddsp_settings.get("enabled", False):
            from modules.vocoders.all_in_one import build_all_in_one_vocoder

            return build_all_in_one_vocoder(
                self.model, hparams.get("all_in_one_vocoder_backend")
            )
        return super()._build_vocoder()

    def build_losses_and_metrics(self):
        AcousticTask.build_losses_and_metrics(self)
        VarianceTask.build_losses_and_metrics(self)
        if self.mini_ddsp_settings.get("enabled", False):
            from training.mini_ddsp import MiniDDSPLoss

            if self.use_shallow_diffusion and not self.train_diffusion:
                raise ValueError(
                    "Joint Mini-DDSP requires train_diffusion=true for the main predictor"
                )
            self.ddsp_loss = MiniDDSPLoss(hparams)
            names = ["ddsp_spectral_loss", "ddsp_amplitude_loss", "ddsp_joint_loss"]
            if self.use_shallow_diffusion and self.train_aux_decoder:
                names.append("ddsp_aux_joint_loss")
            for name in names:
                self.register_validation_loss(name)

    def setup(self, stage):
        binary_data_dir = pathlib.Path(hparams["binary_data_dir"])
        acoustic_dataset = AcousticDataset
        if self.mini_ddsp_settings.get("enabled", False):
            from training.mini_ddsp import PairedAcousticDataset

            acoustic_dataset = PairedAcousticDataset
        self.acoustic_train_dataset = acoustic_dataset(
            "train", data_dir=binary_data_dir / "acoustic"
        )
        self.acoustic_valid_dataset = acoustic_dataset(
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
        if training:
            max_batch_frames = self._split_training_batch_limit(
                self.max_batch_frames, 'max_batch_frames'
            )
            max_batch_size = self._split_training_batch_limit(
                self.max_batch_size, 'max_batch_size'
            )
        else:
            max_batch_frames = self.max_val_batch_frames
            max_batch_size = self.max_val_batch_size
        sampler = DsBatchSampler(
            dataset,
            max_batch_frames=max_batch_frames,
            max_batch_size=max_batch_size,
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

    @classmethod
    def _split_training_batch_limit(cls, total_limit, config_name):
        if total_limit < cls.training_stream_count:
            raise ValueError(
                f'All-in-one training requires {config_name} >= '
                f'{cls.training_stream_count}, got {total_limit}.'
            )
        return total_limit // cls.training_stream_count

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
        joint_metrics = {}
        if self.mini_ddsp_settings.get("enabled", False):
            acoustic_losses, output = AcousticTask.run_model(
                self, sample["acoustic"], model=self.model.acoustic, return_output=True
            )
            ddsp_losses, joint_metrics = self.ddsp_loss(
                self.model.vocoder,
                sample["acoustic"],
                output,
                self.model.acoustic,
                self.global_step,
                training=True,
                return_metrics=True,
            )
            acoustic_losses.update(ddsp_losses)
        else:
            acoustic_losses = AcousticTask.run_model(
                self, sample["acoustic"], model=self.model.acoustic
            )
        variance_losses = VarianceTask.run_model(
            self, sample["variance"], model=self.model.variance
        )
        losses = {**acoustic_losses, **variance_losses}
        batch_size = sample["acoustic"]["size"] + sample["variance"]["size"]
        # Weighted joint terms remain in the objective, never in the raw curves.
        logs = {
            name: value
            for name, value in losses.items()
            if name not in ("ddsp_joint_loss", "ddsp_aux_joint_loss")
        }
        joint_metrics = {
            name.replace("_raw_loss", "_loss"): value
            for name, value in joint_metrics.items()
        }
        return sum(losses.values()), {
            **logs,
            **joint_metrics,
            "batch_size": float(batch_size),
        }

    def _training_log_metrics(self, log_outputs):
        if not self.mini_ddsp_settings.get("enabled", False):
            return log_outputs
        count = log_outputs["ddsp_joint_sample_count"]
        for name in self._joint_log_names:
            if name in log_outputs:
                weighted = log_outputs[name].detach() * count
                self._joint_log_sums[name] = (
                    self._joint_log_sums.get(name, 0) + weighted
                )
        self._joint_log_count += count
        if self.global_step % hparams["log_interval"] != 0:
            return log_outputs
        # Include valid batches between log steps, even if this batch has none.
        metrics = dict(log_outputs)
        for name in self._joint_log_names:
            if name in self._joint_log_sums:
                metrics[name] = self._joint_log_sums[name] / self._joint_log_count
            else:
                metrics.pop(name, None)
        metrics["ddsp_joint_sample_count"] = self._joint_log_count
        self._joint_log_sums.clear()
        self._joint_log_count = 0
        return metrics

    @torch.no_grad()
    def _joint_validation(self, sample, batch_idx):
        # Keep validation noise fixed without changing subsequent training RNG.
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(1234 + batch_idx)
            if devices:
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(1234 + batch_idx)
            output = AcousticTask.run_model(
                self, {**sample, "mel": None}, model=self.model.acoustic, infer=True
            )
            return self.ddsp_loss.validation_joint(self.model.vocoder, sample, output)

    def validation_step(self, sample, batch_idx, dataloader_idx=0):
        if sample["size"] == 0:
            return
        joint_metrics = {}
        with torch.autocast(self.device.type, enabled=False):
            if dataloader_idx == 0:
                losses, weight = self._run_validation_step(
                    AcousticTask._validation_step,
                    self.acoustic_valid_dataset,
                    sample,
                    batch_idx,
                    model=self.model.acoustic,
                )
                if self.mini_ddsp_settings.get("enabled", False):
                    losses.update(
                        self.ddsp_loss(
                            self.model.vocoder,
                            sample,
                            None,
                            self.model.acoustic,
                            self.global_step,
                            training=False,
                            include_joint=False,
                        )
                    )
                    joint_metrics = self._joint_validation(sample, batch_idx)
                    if self.use_vocoder:
                        self._run_validation_step(
                            AllInOneTask._dataset_vocoder_validation,
                            self.acoustic_valid_dataset,
                            sample,
                            batch_idx,
                        )
            else:
                losses, weight = self._run_validation_step(
                    VarianceTask._validation_step,
                    self.variance_valid_dataset,
                    sample,
                    batch_idx,
                    model=self.model.variance,
                )
        # Inference quality metrics do not change the validation objective.
        losses = {'total_loss': sum(losses.values()), **losses, **joint_metrics}
        for name, value in losses.items():
            self.valid_losses[name].update(value, weight=weight)

    def _run_validation_step(self, validation_step, dataset, sample, batch_idx, **kwargs):
        previous_dataset = getattr(self, 'valid_dataset', None)
        self.valid_dataset = dataset
        try:
            return validation_step(self, sample, batch_idx, **kwargs)
        finally:
            self.valid_dataset = previous_dataset

    @torch.inference_mode()
    def _dataset_vocoder_validation(self, sample, batch_idx):
        from utils.binarizer_utils import get_mel_torch

        for index, data_index in enumerate(sample["indices"].tolist()):
            if data_index >= hparams["num_valid_plots"]:
                continue
            paired = sample["vocoder"]
            frames = int(paired["frame_lengths"][index])
            samples = int(paired["sample_lengths"][index])
            mel = sample["mel"][index, :frames]
            f0 = sample["f0"][index : index + 1, :frames]
            source = paired["waveform"][index, :samples]
            reconstructed = self.vocoder.spec2wav_torch(mel[None], f0=f0)[0, :samples]
            writer = self.logger.all_rank_experiment
            tag = f"dataset_vocoder/{data_index}"
            writer.add_audio(
                f"{tag}/source",
                source,
                sample_rate=hparams["audio_sample_rate"],
                global_step=self.global_step,
            )
            writer.add_audio(
                f"{tag}/reconstruction",
                reconstructed,
                sample_rate=hparams["audio_sample_rate"],
                global_step=self.global_step,
            )
            reconstructed_mel = get_mel_torch(
                reconstructed.cpu().numpy(),
                hparams["audio_sample_rate"],
                num_mel_bins=hparams["audio_num_mel_bins"],
                hop_size=hparams["hop_size"],
                win_size=hparams["win_size"],
                fft_size=hparams["fft_size"],
                fmin=hparams["fmin"],
                fmax=hparams["fmax"],
                device=self.device,
            )
            reconstructed_mel = torch.from_numpy(reconstructed_mel).to(mel)
            length = min(mel.shape[0], reconstructed_mel.shape[0])
            # Reuse DiffSinger's existing vertical prediction/target/error view.
            self.plot_mel(
                data_index, mel[:length], reconstructed_mel[:length], "dataset_vocoder"
            )

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
        timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        for speaker_name, _ in self.variance_validation_speakers:
            sources = iter_validation_sources(
                self.val_with_variance, hparams['dictionaries'],
                previous_language=self.variance_validation_language,
            )
            for language, source in sources:
                param = dict(source)
                if hparams['use_spk_id']:
                    param['ph_spk_mix'] = param['spk_mix'] = {speaker_name: 1.0}
                variance_batch = self.variance_validation_infer.preprocess_input(param)
                durations, pitch, variances = self.variance_validation_infer.forward_model(
                    variance_batch
                )
                if (not torch.any(durations > 0) or pitch.shape[-1] == 0
                        or any(value.shape[-1] == 0 for value in variances.values())):
                    continue
                self.variance_validation_language = language
                break
            else:
                warnings.warn(
                    f'Skipping variance validation audio for {speaker_name}: '
                    'all validation segments produced zero frames.',
                    UserWarning,
                )
                continue
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
