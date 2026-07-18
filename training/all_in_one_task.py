import librosa
import matplotlib
import torch
import torch.nn.functional as F
import torch.utils.data
from lightning.pytorch.utilities.rank_zero import rank_zero_only

import utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from basics.base_vocoder import BaseVocoder
from modules.all_in_one import DiffSingerAllInOne
from modules.aux_decoder import build_aux_loss
from modules.losses import DurationLoss, DiffusionLoss, RectifiedFlowLoss
from modules.metrics import (
    PhonemeDurationAccuracy, RawCurveAccuracy, RawCurveR2Score, RhythmCorrectness
)
from modules.fastspeech.tts_modules import LengthRegulator
from modules.toplevel import ShallowDiffusionOutput
from modules.vocoders.registry import get_vocoder_cls
from training.all_in_one_val_ds import AllInOneDsValidationRunner
from training.variance_task import random_retake_masks
from utils.hparams import hparams
from utils.model_summary import build_module_param_summary, print_param_summary
from utils.plot import curve_to_figure, dur_to_figure, pitch_note_to_figure, spec_to_figure
from utils.tensorboard_utils import validation_tb_tag

matplotlib.use('Agg')


class AllInOneDataset(BaseDataset):
    def __init__(self, prefix, preload=False):
        super(AllInOneDataset, self).__init__(prefix, hparams['dataset_size_key'], preload)
        self.required_variances = {}
        if hparams['use_energy_embed'] or hparams['predict_energy']:
            self.required_variances['energy'] = 0.0
        if hparams['use_breathiness_embed'] or hparams['predict_breathiness']:
            self.required_variances['breathiness'] = 0.0
        if hparams['use_voicing_embed'] or hparams['predict_voicing']:
            self.required_variances['voicing'] = 0.0
        if hparams['use_tension_embed'] or hparams['predict_tension']:
            self.required_variances['tension'] = 0.0
        self.predict_variances = any([
            hparams['predict_energy'],
            hparams['predict_breathiness'],
            hparams['predict_voicing'],
            hparams['predict_tension'],
        ])

    def collater(self, samples):
        batch = super().collater(samples)
        if batch['size'] == 0:
            return batch

        batch.update({
            'tokens': utils.collate_nd([s['tokens'] for s in samples], 0),
            'ph_dur': utils.collate_nd([s['ph_dur'] for s in samples], 0),
            'mel2ph': utils.collate_nd([s['mel2ph'] for s in samples], 0),
            'mel': utils.collate_nd([s['mel'] for s in samples], 0.0),
            'f0': utils.collate_nd([s['f0'] for s in samples], 0.0),
        })
        if hparams['use_spk_id']:
            batch['spk_ids'] = torch.LongTensor([s['spk_id'] for s in samples])
        if hparams['use_lang_id']:
            batch['languages'] = utils.collate_nd([s['languages'] for s in samples], 0)
        if hparams['predict_dur']:
            batch['ph2word'] = utils.collate_nd([s['ph2word'] for s in samples], 0)
            batch['midi'] = utils.collate_nd([s['midi'] for s in samples], 0)
        if hparams['predict_pitch']:
            batch['note_midi'] = utils.collate_nd([s['note_midi'] for s in samples], -1)
            batch['note_rest'] = utils.collate_nd([s['note_rest'] for s in samples], True)
            batch['note_dur'] = utils.collate_nd([s['note_dur'] for s in samples], 0)
            if hparams['use_glide_embed']:
                batch['note_glide'] = utils.collate_nd([s['note_glide'] for s in samples], 0)
            batch['mel2note'] = utils.collate_nd([s['mel2note'] for s in samples], 0)
            batch['base_pitch'] = utils.collate_nd([s['base_pitch'] for s in samples], 0)
        if hparams['predict_pitch'] or self.predict_variances:
            batch['pitch'] = utils.collate_nd([s['pitch'] for s in samples], 0)
            batch['uv'] = utils.collate_nd([s['uv'] for s in samples], True)
        for v_name, v_pad in self.required_variances.items():
            batch[v_name] = utils.collate_nd([s[v_name] for s in samples], v_pad)
        if hparams['use_key_shift_embed']:
            batch['key_shift'] = torch.FloatTensor([s.get('key_shift', 0.0) for s in samples])[:, None]
        if hparams['use_speed_embed']:
            batch['speed'] = torch.FloatTensor([s.get('speed', 1.0) for s in samples])[:, None]
        return batch


class AllInOneTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = AllInOneDataset
        self.diffusion_type = hparams['diffusion_type']
        assert self.diffusion_type in ['ddpm', 'reflow'], f"Unknown diffusion type: {self.diffusion_type}"

        self.use_spk_id = hparams['use_spk_id']
        self.use_lang_id = hparams['use_lang_id']
        self.predict_dur = hparams['predict_dur']
        self.predict_pitch = hparams['predict_pitch']
        all_in_one_cfg = hparams.get('all_in_one', {})
        self.train_dur = bool(all_in_one_cfg.get('train_dur', self.predict_dur))
        self.train_pitch = bool(all_in_one_cfg.get('train_pitch', self.predict_pitch))
        self.train_variance = bool(all_in_one_cfg.get('train_variance', True))
        self.train_acoustic = bool(all_in_one_cfg.get('train_acoustic', True))
        self.loss_weights = all_in_one_cfg.get('loss_weights', {}) or {}
        self.variance_prediction_list = [
            name for name in ['energy', 'breathiness', 'voicing', 'tension']
            if hparams[f'predict_{name}']
        ]
        self.predict_variances = len(self.variance_prediction_list) > 0
        self.required_variances = [
            name for name in ['energy', 'breathiness', 'voicing', 'tension']
            if hparams[f'use_{name}_embed']
        ]

        self.lambda_dur_loss = hparams['lambda_dur_loss']
        self.lambda_pitch_loss = hparams['lambda_pitch_loss']
        self.lambda_var_loss = hparams['lambda_var_loss']
        self.use_shallow_diffusion = hparams['use_shallow_diffusion']
        if self.use_shallow_diffusion:
            self.shallow_args = hparams['shallow_diffusion_args']
            self.train_aux_decoder = self.shallow_args['train_aux_decoder']
            self.train_diffusion = self.shallow_args['train_diffusion']
            self.lambda_aux_mel_loss = hparams['lambda_aux_mel_loss']

        self.use_vocoder = hparams['infer'] or hparams['val_with_vocoder']
        if self.use_vocoder:
            self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        self.lr = LengthRegulator()
        self.logged_gt_wav = set()
        self.val_ds_runner = None
        self._skip_val_ds_on_validation_end = False
        val_with_ds = (hparams.get('validation') or {}).get('val_with_ds') or hparams.get('val_with_ds')
        if AllInOneDsValidationRunner.is_enabled(val_with_ds):
            self.val_ds_runner = AllInOneDsValidationRunner(val_with_ds)
        super()._finish_init()

    def _build_model(self):
        return DiffSingerAllInOne(
            vocab_size=len(self.phoneme_dictionary),
            out_dims=hparams['audio_num_mel_bins']
        )

    @rank_zero_only
    def print_arch(self):
        utils.print_arch(self.model)
        if hasattr(self.model, 'named_submodules_for_summary'):
            rows = build_module_param_summary(self.model, self.model.named_submodules_for_summary())
            print_param_summary(rows, title='All-in-One Parameter Summary')

    def build_losses_and_metrics(self):
        if self.predict_dur and self.train_dur:
            dur_hparams = hparams['dur_prediction_args']
            self.dur_loss = DurationLoss(
                offset=dur_hparams['log_offset'],
                loss_type=dur_hparams['loss_type'],
                lambda_pdur=dur_hparams['lambda_pdur_loss'],
                lambda_wdur=dur_hparams['lambda_wdur_loss'],
                lambda_sdur=dur_hparams['lambda_sdur_loss']
            )
            self.register_validation_loss('dur_loss')
            self.register_validation_metric('rhythm_corr', RhythmCorrectness(tolerance=0.05))
            self.register_validation_metric('ph_dur_acc', PhonemeDurationAccuracy(tolerance=0.2))
        if self.predict_pitch and self.train_pitch:
            self.pitch_loss = self._build_diffusion_loss()
            self.register_validation_loss('pitch_loss')
            self.register_validation_metric('pitch_acc', RawCurveAccuracy(tolerance=0.5))
            self.register_validation_metric('pitch_r2', RawCurveR2Score())
        if self.predict_variances and self.train_variance:
            self.var_loss = self._build_diffusion_loss()
            self.register_validation_loss('var_loss')
            for name in self.variance_prediction_list:
                self.register_validation_metric(f'{name}_r2', RawCurveR2Score())
        if self.use_shallow_diffusion and self.train_acoustic:
            self.aux_mel_loss = build_aux_loss(self.shallow_args['aux_decoder_arch'])
            self.register_validation_loss('aux_mel_loss')
        if self.train_acoustic:
            self.mel_loss = self._build_diffusion_loss()
            self.register_validation_loss('mel_loss')
        self.register_validation_loss('joint_mel_loss')

    def _build_diffusion_loss(self):
        if self.diffusion_type == 'ddpm':
            return DiffusionLoss(loss_type=hparams['main_loss_type'])
        if self.diffusion_type == 'reflow':
            return RectifiedFlowLoss(
                loss_type=hparams['main_loss_type'], log_norm=hparams['main_loss_log_norm']
            )
        raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")

    def run_model(self, sample, infer=False, joint=False):
        if infer:
            variance_out = self.run_variance_model(sample, infer=True)
            acoustic_sample = self.build_joint_acoustic_sample(sample, variance_out) if joint else sample
            acoustic_out = self.run_acoustic_model(acoustic_sample, infer=True)
            return variance_out, acoustic_out

        losses = {}
        variance_out = self.run_variance_model(sample, infer=False)
        acoustic_out = self.run_acoustic_model(sample, infer=False)
        losses.update(self.compute_variance_losses(sample, variance_out))
        losses.update(self.compute_acoustic_losses(sample, acoustic_out))
        if joint:
            variance_infer_out = self.run_variance_model(sample, infer=True)
            joint_sample = self.build_joint_acoustic_sample(sample, variance_infer_out)
            joint_out = self.run_acoustic_model(joint_sample, infer=True)
            if joint_out is not None and joint_out.diff_out is not None:
                losses['joint_mel_loss'] = torch.nn.functional.l1_loss(
                    joint_out.diff_out, joint_sample['mel'], reduction='mean'
                )
        return self.apply_loss_weights(losses)

    def apply_loss_weights(self, losses):
        weighted = {}
        for name, value in losses.items():
            if name == 'dur_loss':
                weight = self.loss_weights.get('dur', 1.0)
            elif name == 'pitch_loss':
                weight = self.loss_weights.get('pitch', 1.0)
            elif name == 'var_loss':
                weight = self.loss_weights.get('variance', 1.0)
            elif name == 'mel_loss':
                weight = self.loss_weights.get('acoustic', 1.0)
            elif name == 'aux_mel_loss':
                weight = self.loss_weights.get('aux_mel', 1.0)
            elif name == 'joint_mel_loss':
                weight = self.loss_weights.get('joint', 1.0)
            else:
                weight = 1.0
            if weight != 0:
                weighted[name] = value * weight
        return weighted

    def run_variance_model(self, sample, infer=False):
        if self.model.variance is None:
            return None, None, None
        spk_ids = sample['spk_ids'] if self.use_spk_id else None
        languages = sample['languages'] if self.use_lang_id else None
        mel2ph = sample.get('mel2ph')
        pitch_retake = variance_retake = None
        if (self.predict_pitch or self.predict_variances) and not infer:
            b = sample['size']
            t = mel2ph.shape[1]
            device = mel2ph.device
            if self.predict_pitch:
                pitch_retake = random_retake_masks(b, t, device)
            if self.predict_variances:
                variance_retake = {
                    v_name: random_retake_masks(b, t, device)
                    for v_name in self.variance_prediction_list
                }
        output = self.model.forward_variance(
            sample['tokens'], languages=languages,
            midi=sample.get('midi'), ph2word=sample.get('ph2word'),
            ph_dur=sample['ph_dur'], mel2ph=mel2ph,
            note_midi=sample.get('note_midi'), note_rest=sample.get('note_rest'),
            note_dur=sample.get('note_dur'), note_glide=sample.get('note_glide'),
            mel2note=sample.get('mel2note'),
            base_pitch=sample.get('base_pitch'), pitch=sample.get('pitch'),
            energy=sample.get('energy'), breathiness=sample.get('breathiness'),
            voicing=sample.get('voicing'), tension=sample.get('tension'),
            pitch_retake=pitch_retake, variance_retake=variance_retake,
            spk_id=spk_ids, infer=infer
        )
        if infer and output[0] is not None:
            return output[0].round().long(), output[1], output[2]
        return output

    def build_joint_acoustic_sample(self, sample, variance_out):
        joint_sample = dict(sample)
        if variance_out is None:
            return joint_sample

        dur_pred, pitch_pred, variances_pred = variance_out
        if dur_pred is not None and self.train_dur:
            durations = dur_pred.clamp(min=0).round().long()
            mel2ph = self.lr(durations, sample['tokens'] == 0)
            if mel2ph.shape[1] > 0:
                joint_sample['mel2ph'] = mel2ph

        target_len = joint_sample['mel2ph'].shape[1]
        joint_sample['mel'] = self._align_time_tensor(sample['mel'], target_len, pad_value=0.0)

        if self.predict_pitch and pitch_pred is not None and sample.get('base_pitch') is not None:
            midi = self._align_time_tensor(sample['base_pitch'] + pitch_pred, target_len, pad_value=0.0)
            f0_np = librosa.midi_to_hz(midi.detach().cpu().numpy())
            joint_sample['f0'] = torch.from_numpy(f0_np).to(midi)
        else:
            joint_sample['f0'] = self._align_time_tensor(sample['f0'], target_len, pad_value=0.0)

        if isinstance(variances_pred, dict):
            for name in self.required_variances:
                if name in variances_pred:
                    joint_sample[name] = self._align_time_tensor(variances_pred[name], target_len, pad_value=0.0)
                elif name in sample:
                    joint_sample[name] = self._align_time_tensor(sample[name], target_len, pad_value=0.0)
        else:
            for name in self.required_variances:
                if name in sample:
                    joint_sample[name] = self._align_time_tensor(sample[name], target_len, pad_value=0.0)

        for name, pad_value in [('key_shift', 0.0), ('speed', 1.0)]:
            if name in sample and sample[name].dim() > 1 and sample[name].shape[1] > 1:
                joint_sample[name] = self._align_time_tensor(sample[name], target_len, pad_value=pad_value)
        return joint_sample

    @staticmethod
    def _align_time_tensor(tensor, target_len, pad_value=0.0):
        if tensor is None or tensor.dim() < 2:
            return tensor
        cur_len = tensor.shape[1]
        if cur_len == target_len:
            return tensor
        if cur_len > target_len:
            return tensor[:, :target_len, ...]
        pad_len = target_len - cur_len
        if tensor.dim() == 2:
            return F.pad(tensor, (0, pad_len), value=pad_value)
        if tensor.dim() == 3:
            return F.pad(tensor, (0, 0, 0, pad_len), value=pad_value)
        raise ValueError(f'Unsupported time-major tensor rank: {tensor.dim()}')

    def run_acoustic_model(self, sample, infer=False):
        if self.model.acoustic is None:
            return None
        variances = {
            v_name: sample[v_name]
            for v_name in self.required_variances
        }
        f0 = sample['f0']
        return self.model.forward_acoustic(
            sample['tokens'], mel2ph=sample['mel2ph'], f0=f0, **variances,
            key_shift=sample.get('key_shift'), speed=sample.get('speed'),
            spk_embed_id=sample['spk_ids'] if self.use_spk_id else None,
            languages=sample['languages'] if self.use_lang_id else None,
            gt_mel=sample['mel'], infer=infer
        )

    def compute_variance_losses(self, sample, variance_out):
        losses = {}
        if variance_out is None:
            return losses
        dur_pred, pitch_pred, variances_pred = variance_out
        if dur_pred is not None:
            losses['dur_loss'] = self.lambda_dur_loss * self.dur_loss(
                dur_pred, sample['ph_dur'], ph2word=sample.get('ph2word')
            )
        non_padding = (sample['mel2ph'] > 0).unsqueeze(-1)
        if pitch_pred is not None and self.train_pitch:
            losses['pitch_loss'] = self.lambda_pitch_loss * self._diffusion_loss_value(
                self.pitch_loss, pitch_pred, non_padding
            )
        if variances_pred is not None and self.train_variance:
            losses['var_loss'] = self.lambda_var_loss * self._diffusion_loss_value(
                self.var_loss, variances_pred, non_padding
            )
        return losses

    def compute_acoustic_losses(self, sample, output: ShallowDiffusionOutput):
        losses = {}
        if output is None:
            return losses
        if not self.train_acoustic:
            return losses
        if output.aux_out is not None:
            norm_gt = self.model.acoustic.aux_decoder.norm_spec(sample['mel'])
            losses['aux_mel_loss'] = self.lambda_aux_mel_loss * self.aux_mel_loss(output.aux_out, norm_gt)
        non_padding = (sample['mel2ph'] > 0).unsqueeze(-1).float()
        if output.diff_out is not None:
            losses['mel_loss'] = self._diffusion_loss_value(self.mel_loss, output.diff_out, non_padding)
        return losses

    def _diffusion_loss_value(self, loss_fn, pred, non_padding):
        if self.diffusion_type == 'ddpm':
            x_recon, x_noise = pred
            return loss_fn(x_recon, x_noise, non_padding=non_padding)
        if self.diffusion_type == 'reflow':
            v_pred, v_gt, t = pred
            return loss_fn(v_pred, v_gt, t=t, non_padding=non_padding)
        raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")

    def on_train_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)

    def _on_validation_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False, joint=True)
        if sample['size'] > 0 and min(sample['indices']) < hparams['num_valid_plots']:
            variance_out, acoustic_out = self.run_model(sample, infer=True, joint=False)
            _, joint_out = self.run_model(sample, infer=True, joint=True)
            self.update_variance_metrics_and_plots(sample, variance_out)
            self.plot_acoustic_outputs(sample, acoustic_out, prefix='all_in_one_acoustic')
            joint_sample = self.build_joint_acoustic_sample(sample, variance_out)
            self.plot_acoustic_outputs(joint_sample, joint_out, prefix='all_in_one_joint')
        return losses, sample['size']

    def on_validation_epoch_end(self):
        was_skipping = self.skip_immediate_validation
        super().on_validation_epoch_end()
        if was_skipping:
            self._skip_val_ds_on_validation_end = True

    def on_validation_end(self):
        if self.val_ds_runner is not None:
            if getattr(self.trainer, 'sanity_checking', False):
                return
            if getattr(self.trainer, 'testing', False):
                return
            if self._skip_val_ds_on_validation_end or self.skip_immediate_ckpt_save:
                self._skip_val_ds_on_validation_end = False
                return
            self.val_ds_runner.run(self)

    def update_variance_metrics_and_plots(self, sample, variance_out):
        if variance_out is None:
            return
        dur_preds, pitch_preds, variances_preds = variance_out
        for i in range(len(sample['indices'])):
            data_idx = sample['indices'][i].item()
            if data_idx >= hparams['num_valid_plots']:
                continue
            if dur_preds is not None:
                dur_len = self.valid_dataset.metadata['ph_dur'][data_idx]
                gt_dur = self._sample_get(sample, 'ph_dur', i, data_idx)
                pred_dur = dur_preds[i][:dur_len].unsqueeze(0)
                tokens = self._sample_get(sample, 'tokens', i, data_idx)
                ph2word = self._sample_get(sample, 'ph2word', i, data_idx)
                mask = tokens != 0
                if 'rhythm_corr' in self.valid_metrics:
                    self.valid_metrics['rhythm_corr'].update(
                        pdur_pred=pred_dur, pdur_target=gt_dur, ph2word=ph2word, mask=mask
                    )
                if 'ph_dur_acc' in self.valid_metrics:
                    self.valid_metrics['ph_dur_acc'].update(
                        pdur_pred=pred_dur, pdur_target=gt_dur, ph2word=ph2word, mask=mask
                    )
                self.plot_dur(data_idx, gt_dur, pred_dur)
            if pitch_preds is not None:
                pitch_len = self.valid_dataset.metadata['pitch'][data_idx]
                pred_pitch = self._sample_get(sample, 'base_pitch', i, data_idx) + pitch_preds[i][:pitch_len].unsqueeze(0)
                gt_pitch = self._sample_get(sample, 'pitch', i, data_idx)
                mask = (self._sample_get(sample, 'mel2ph', i, data_idx) > 0) & ~self._sample_get(sample, 'uv', i, data_idx)
                if 'pitch_acc' in self.valid_metrics:
                    self.valid_metrics['pitch_acc'].update(pred=pred_pitch, target=gt_pitch, mask=mask)
                if 'pitch_r2' in self.valid_metrics:
                    self.valid_metrics['pitch_r2'].update(pred=pred_pitch, target=gt_pitch, mask=mask)
                self.plot_pitch(
                    data_idx, gt_pitch, pred_pitch,
                    note_midi=self._sample_get(sample, 'note_midi', i, data_idx),
                    note_dur=self._sample_get(sample, 'note_dur', i, data_idx),
                    note_rest=self._sample_get(sample, 'note_rest', i, data_idx)
                )
            if isinstance(variances_preds, dict):
                for name in self.variance_prediction_list:
                    variance_len = self.valid_dataset.metadata[name][data_idx]
                    gt_variance = sample[name][i][:variance_len].unsqueeze(0)
                    pred_variance = variances_preds[name][i][:variance_len].unsqueeze(0)
                    mask = (self._sample_get(sample, 'mel2ph', i, data_idx) > 0) & ~self._sample_get(sample, 'uv', i, data_idx)
                    if f'{name}_r2' in self.valid_metrics:
                        self.valid_metrics[f'{name}_r2'].update(pred=pred_variance, target=gt_variance, mask=mask)
                    self.plot_curve(data_idx, gt_variance, pred_variance, curve_name=name)

    def _sample_get(self, sample, key, idx, abs_idx):
        return sample[key][idx][:self.valid_dataset.metadata[key][abs_idx]].unsqueeze(0)

    def _validation_tb_tag(self, prefix, data_idx):
        return validation_tb_tag(
            hparams.get('tb_layout', 'flat'),
            prefix,
            data_idx,
            speaker=self.valid_dataset.metadata['spk_names'][data_idx],
            item=self.valid_dataset.metadata['names'][data_idx]
        )

    def plot_acoustic_outputs(self, sample, output, prefix):
        if output is None:
            return
        for i in range(len(sample['indices'])):
            data_idx = sample['indices'][i].item()
            if data_idx >= hparams['num_valid_plots']:
                continue
            mel_len = self.valid_dataset.metadata['mel'][data_idx]
            f0_len = self.valid_dataset.metadata['f0'][data_idx]
            gt_mel = sample['mel'][i][:mel_len]
            f0 = sample['f0'][i][:f0_len].unsqueeze(0)
            if output.aux_out is not None:
                self.plot_mel(data_idx, gt_mel, output.aux_out[i], f'{prefix}_auxmel')
            if output.diff_out is not None:
                self.plot_mel(data_idx, gt_mel, output.diff_out[i], f'{prefix}_diffmel')
            if self.use_vocoder and output.diff_out is not None:
                wav = self.vocoder.spec2wav_torch(output.diff_out[i][:mel_len].unsqueeze(0), f0=f0)
                self.logger.all_rank_experiment.add_audio(
                    self._validation_tb_tag(f'{prefix}_audio', data_idx),
                    wav,
                    sample_rate=hparams['audio_sample_rate'],
                    global_step=self.global_step
                )
            if self.use_vocoder and data_idx not in self.logged_gt_wav:
                gt_wav = self.vocoder.spec2wav_torch(gt_mel.unsqueeze(0), f0=f0)
                self.logger.all_rank_experiment.add_audio(
                    self._validation_tb_tag('all_in_one_audio_gt', data_idx),
                    gt_wav,
                    sample_rate=hparams['audio_sample_rate'],
                    global_step=self.global_step
                )
                self.logged_gt_wav.add(data_idx)

    def plot_dur(self, data_idx, gt_dur, pred_dur):
        title = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        txt = self.valid_dataset.metadata['ph_texts'][data_idx].split()
        self.logger.all_rank_experiment.add_figure(
            self._validation_tb_tag('all_in_one_variance_dur', data_idx),
            dur_to_figure(gt_dur[0], pred_dur[0], txt, title),
            self.global_step
        )

    def plot_pitch(self, data_idx, gt_pitch, pred_pitch, note_midi=None, note_dur=None, note_rest=None):
        title = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        self.logger.all_rank_experiment.add_figure(
            self._validation_tb_tag('all_in_one_variance_pitch', data_idx),
            pitch_note_to_figure(
                gt_pitch[0], pred_pitch[0],
                note_midi[0] if note_midi is not None else None,
                note_dur[0] if note_dur is not None else None,
                note_rest[0] if note_rest is not None else None,
                title
            ),
            self.global_step
        )

    def plot_curve(self, data_idx, gt_curve, pred_curve, base_curve=None, grid=None, curve_name='curve'):
        title = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        self.logger.all_rank_experiment.add_figure(
            self._validation_tb_tag(f'all_in_one_variance_{curve_name}', data_idx),
            curve_to_figure(gt_curve[0], pred_curve[0], base_curve, grid=grid, title=title),
            self.global_step
        )

    def plot_mel(self, data_idx, gt_spec, out_spec, name_prefix='mel'):
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        mel_len = self.valid_dataset.metadata['mel'][data_idx]
        gt_spec = gt_spec[:mel_len]
        out_spec = out_spec[:mel_len]
        spec_cat = torch.cat([(out_spec - gt_spec).abs() + vmin, gt_spec, out_spec], -1)
        title = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        self.logger.all_rank_experiment.add_figure(
            self._validation_tb_tag(name_prefix, data_idx),
            spec_to_figure(spec_cat[:mel_len], vmin, vmax, title),
            global_step=self.global_step
        )
