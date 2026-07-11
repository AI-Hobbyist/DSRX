import json
import pathlib
from collections import OrderedDict
from typing import Dict
import contextlib

import numpy as np
import torch
import tqdm

from basics.base_svs_infer import BaseSVSInfer
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from modules.fastspeech.tts_modules import LengthRegulator
from modules.toplevel import DiffSingerAcoustic, ShallowDiffusionOutput
from modules.vocoders.registry import VOCODERS
from utils import load_ckpt
from utils.lora import inject_lora, load_lora_state_dict
from utils.hparams import hparams
from utils.infer_utils import cross_fade, resample_align_curve, save_wav
from utils.phoneme_utils import load_phoneme_dictionary


class DiffSingerAcousticInfer(BaseSVSInfer):
    def __init__(self, device=None, load_model=True, load_vocoder=True, ckpt_steps=None, dtype='fp32'):
        super().__init__(device=device)
        self.dtype = dtype
        self._precision_ctx = None
        if load_model:
            from utils.precision import PrecisionContext
            self._precision_ctx = PrecisionContext(dtype, device_type=self.device.type)

            self.variance_checklist = []
            self.variance_checklist = []

            self.variances_to_embed = set()

            if hparams.get('use_energy_embed', False):
                self.variances_to_embed.add('energy')
            if hparams.get('use_breathiness_embed', False):
                self.variances_to_embed.add('breathiness')
            if hparams.get('use_voicing_embed', False):
                self.variances_to_embed.add('voicing')
            if hparams.get('use_tension_embed', False):
                self.variances_to_embed.add('tension')

            self.phoneme_dictionary = load_phoneme_dictionary()
            if hparams['use_spk_id']:
                with open(pathlib.Path(hparams['work_dir']) / 'spk_map.json', 'r', encoding='utf8') as f:
                    self.spk_map = json.load(f)
                assert isinstance(self.spk_map, dict) and len(self.spk_map) > 0, 'Invalid or empty speaker map!'
                assert len(self.spk_map) == len(set(self.spk_map.values())), 'Duplicate speaker id in speaker map!'
            lang_map_fn = pathlib.Path(hparams['work_dir']) / 'lang_map.json'
            if lang_map_fn.exists():
                with open(lang_map_fn, 'r', encoding='utf8') as f:
                    self.lang_map = json.load(f)
            self.model = self.build_model(ckpt_steps=ckpt_steps)
            self.lr = LengthRegulator().to(self.device)
        if load_vocoder:
            self.vocoder = self.build_vocoder()

    def build_model(self, ckpt_steps=None):
        model = DiffSingerAcoustic(
            vocab_size=len(self.phoneme_dictionary),
            out_dims=hparams['audio_num_mel_bins']
        ).eval().to(self.device)
        lora_cfg = hparams.get('lora', {})
        if isinstance(lora_cfg, dict) and lora_cfg.get('enabled', False):
            # Inject LoRA
            rank = int(lora_cfg.get('rank', 8))
            alpha = int(lora_cfg.get('alpha', 16))
            targets = lora_cfg.get('target_modules', ['linear'])
            inject_lora(model, rank=rank, alpha=alpha, target_modules=targets)
            # Ensure newly injected LoRA layers are moved to the target device
            model = model.to(self.device)
            base_ckpt = lora_cfg.get('base_ckpt', None)
            if base_ckpt:
                # Load base checkpoint non-strictly to tolerate LoRA params and shape diffs (e.g., spk_embed)
                load_ckpt(model, base_ckpt, ckpt_steps=None, prefix_in_ckpt='model', strict=False, device=self.device)
            else:
                load_ckpt(model, hparams['work_dir'], ckpt_steps=ckpt_steps,
                          prefix_in_ckpt='model', strict=False, device=self.device)
            # Load latest checkpoint from work_dir if present (full or LoRA-only)
            try:
                import pathlib as _p
                from utils.training_utils import get_latest_checkpoint_path
                latest = get_latest_checkpoint_path(_p.Path(hparams['work_dir']))
                if latest:
                    try:
                        # Prefer loading full state dict (base + LoRA) non-strictly
                        load_ckpt(model, pathlib.Path(latest), ckpt_steps=None,
                                  prefix_in_ckpt='model', strict=False, device=self.device)
                    except Exception:
                        # Fallback: only apply LoRA params if checkpoint is LoRA-only
                        sd = torch.load(latest, map_location=self.device).get('state_dict', {})
                        load_lora_state_dict(model, sd, prefix='model', strict=False)
            except Exception as e:
                print(f'| warn: load lora weights failed: {e}')
            # One more time to guarantee all parameters are on the correct device after loading
            model = model.to(self.device)
        else:
            load_ckpt(model, hparams['work_dir'], ckpt_steps=ckpt_steps,
                      prefix_in_ckpt='model', strict=True, device=self.device)
        # Convert to target precision
        if self._precision_ctx is not None:
            model = self._precision_ctx.convert_model(model)
        return model

    def build_vocoder(self):
        if hparams['vocoder'] in VOCODERS:
            vocoder = VOCODERS[hparams['vocoder']]()
        else:
            vocoder = VOCODERS[hparams['vocoder'].split('.')[-1]]()
        vocoder.to_device(self.device)
        return vocoder

    def preprocess_input(self, param, idx=0):
        """
        :param param: one segment in the .ds file
        :param idx: index of the segment
        :return: batch of the model inputs
        """
        batch = {}
        summary = OrderedDict()

        tokens = param['ph_seq'].split()
        lang, phoneme_langs = self._resolve_language_config(tokens, param.get('lang'))
        if hparams.get('use_lang_id', False):
            lang_ids = []
            for phone, phone_lang in zip(tokens, phoneme_langs):
                if not self.phoneme_dictionary.is_cross_lingual(phone):
                    lang_ids.append(0)
                    continue
                assert self.lang_map, (
                    'Language map is required when language IDs are enabled.'
                )
                assert phone_lang is not None, (
                    f"Missing language tag for phoneme '{phone}' required by language id embedding."
                )
                assert phone_lang in self.lang_map, (
                    f"Phoneme '{phone}' uses unknown language '{phone_lang}'."
                )
                lang_ids.append(self.lang_map[phone_lang])
            batch['languages'] = torch.LongTensor(lang_ids).to(self.device)[None]  # => [B=1, T_txt]
        txt_tokens = torch.LongTensor([
            self.phoneme_dictionary.encode(param['ph_seq'], lang=lang)
        ]).to(self.device)  # => [B, T_txt]
        batch['tokens'] = txt_tokens

        ph_dur = torch.from_numpy(np.array(param['ph_dur'].split(), np.float32)).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))[None]  # => [B=1, T_txt]
        mel2ph = self.lr(durations, txt_tokens == 0)  # => [B=1, T]
        batch['mel2ph'] = mel2ph
        length = mel2ph.size(1)  # => T

        summary['tokens'] = txt_tokens.size(1)
        summary['frames'] = length
        summary['seconds'] = '%.2f' % (length * self.timestep)

        if hparams['use_spk_id']:
            # Allow direct injection of a custom speaker embedding vector
            custom_spk_embed = param.get('spk_embed')
            if custom_spk_embed is not None:
                try:
                    vec = np.array(custom_spk_embed, np.float32)
                    H = int(hparams['hidden_size'])
                    if vec.ndim == 0:
                        vec = vec[None]
                    if vec.shape[-1] < H:
                        vec = np.pad(vec, (0, H - vec.shape[-1]), mode='constant')
                    elif vec.shape[-1] > H:
                        vec = vec[:H]
                    spk_mix_embed = torch.from_numpy(vec).to(self.device)[None, None, :].repeat([1, length, 1])
                    batch['spk_mix_embed'] = spk_mix_embed
                    summary['spk'] = 'custom-embed'
                except Exception:
                    # Fallback to normal mixing when parsing fails
                    spk_mix_id, spk_mix_value = self.load_speaker_mix(
                        param_src=param, summary_dst=summary, mix_mode='frame', mix_length=length
                    )
                    batch['spk_mix_id'] = spk_mix_id
                    batch['spk_mix_value'] = spk_mix_value
            else:
                spk_mix_id, spk_mix_value = self.load_speaker_mix(
                    param_src=param, summary_dst=summary, mix_mode='frame', mix_length=length
                )
                batch['spk_mix_id'] = spk_mix_id
                batch['spk_mix_value'] = spk_mix_value

        batch['f0'] = torch.from_numpy(resample_align_curve(
            np.array(param['f0_seq'].split(), np.float32),
            original_timestep=float(param['f0_timestep']),
            target_timestep=self.timestep,
            align_length=length
        )).to(self.device)[None]

        for v_name in VARIANCE_CHECKLIST:
            if v_name in self.variances_to_embed:
                batch[v_name] = torch.from_numpy(resample_align_curve(
                    np.array(param[v_name].split(), np.float32),
                    original_timestep=float(param[f'{v_name}_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )).to(self.device)[None]
                summary[v_name] = 'manual'

        if hparams['use_key_shift_embed']:
            shift_min, shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
            gender = param.get('gender')
            if gender is None:
                gender = 0.
            if isinstance(gender, (int, float, bool)):  # static gender value
                summary['gender'] = f'static({gender:.3f})'
                key_shift_value = gender * shift_max if gender >= 0 else gender * abs(shift_min)
                batch['key_shift'] = torch.FloatTensor([key_shift_value]).to(self.device)[:, None]  # => [B=1, T=1]
            else:
                summary['gender'] = 'dynamic'
                gender_seq = resample_align_curve(
                    np.array(gender.split(), np.float32),
                    original_timestep=float(param['gender_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )
                gender_mask = gender_seq >= 0
                key_shift_seq = gender_seq * (gender_mask * shift_max + (1 - gender_mask) * abs(shift_min))
                batch['key_shift'] = torch.clip(
                    torch.from_numpy(key_shift_seq.astype(np.float32)).to(self.device)[None],  # => [B=1, T]
                    min=shift_min, max=shift_max
                )

        if hparams['use_speed_embed']:
            if param.get('velocity') is None:
                summary['velocity'] = 'default'
                batch['speed'] = torch.FloatTensor([1.]).to(self.device)[:, None]  # => [B=1, T=1]
            else:
                summary['velocity'] = 'manual'
                speed_min, speed_max = hparams['augmentation_args']['random_time_stretching']['range']
                speed_seq = resample_align_curve(
                    np.array(param['velocity'].split(), np.float32),
                    original_timestep=float(param['velocity_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )
                batch['speed'] = torch.clip(
                    torch.from_numpy(speed_seq.astype(np.float32)).to(self.device)[None],  # => [B=1, T]
                    min=speed_min, max=speed_max
                )

        print(f'[{idx}]\t' + ', '.join(f'{k}: {v}' for k, v in summary.items()))

        return batch

    @torch.no_grad()
    def forward_model(self, sample):
        txt_tokens = sample['tokens']
        variances = {
            v_name: sample.get(v_name)
            for v_name in self.variances_to_embed
        }
        if hparams['use_spk_id']:
            spk_mix_embed = sample.get('spk_mix_embed')
            if spk_mix_embed is None and 'spk_mix_id' in sample and 'spk_mix_value' in sample:
                spk_mix_id = sample['spk_mix_id']
                spk_mix_value = sample['spk_mix_value']
                # perform mixing on spk embed
                spk_mix_embed = torch.sum(
                    self.model.fs2.spk_embed(spk_mix_id) * spk_mix_value.unsqueeze(3),  # => [B, T, N, H]
                    dim=2, keepdim=False
                )  # => [B, T, H]
        else:
            spk_mix_embed = None
        # Run model under autocast if precision ctx is configured
        ctx = self._precision_ctx.autocast() if self._precision_ctx is not None else contextlib.nullcontext()
        with ctx:
            mel_pred: ShallowDiffusionOutput = self.model(
                txt_tokens,  languages=sample.get('languages'),
                mel2ph=sample['mel2ph'], f0=sample['f0'], **variances,
                key_shift=sample.get('key_shift'), speed=sample.get('speed'),
                spk_mix_embed=spk_mix_embed,
                infer=True
            )
        return mel_pred.diff_out

    @torch.no_grad()
    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder.spec2wav_torch(spec, **kwargs)
        return y[None]

    def run_inference(
            self, params,
            out_dir: pathlib.Path = None,
            title: str = None,
            num_runs: int = 1,
            spk_mix: Dict[str, float] = None,
            seed: int = -1,
            save_mel: bool = False
    ):
        batches = [self.preprocess_input(param, idx=i) for i, param in enumerate(params)]

        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = '.wav' if not save_mel else '.mel.pt'
        for i in range(num_runs):
            if save_mel:
                result = []
            else:
                result = np.zeros(0)
            current_length = 0

            for param, batch in tqdm.tqdm(
                    zip(params, batches), desc='infer segments', total=len(params)
            ):
                if 'seed' in param:
                    torch.manual_seed(param["seed"] & 0xffff_ffff)
                    torch.cuda.manual_seed_all(param["seed"] & 0xffff_ffff)
                elif seed >= 0:
                    torch.manual_seed(seed & 0xffff_ffff)
                    torch.cuda.manual_seed_all(seed & 0xffff_ffff)

                mel_pred = self.forward_model(batch)
                if save_mel:
                    result.append({
                        'offset': param.get('offset', 0.),
                        'mel': mel_pred.cpu(),
                        'f0': batch['f0'].cpu()
                    })
                else:
                    waveform_pred = self.run_vocoder(mel_pred, f0=batch['f0'])[0].cpu().numpy()
                    silent_length = round(param.get('offset', 0) * hparams['audio_sample_rate']) - current_length
                    if silent_length >= 0:
                        result = np.append(result, np.zeros(silent_length))
                        result = np.append(result, waveform_pred)
                    else:
                        result = cross_fade(result, waveform_pred, current_length + silent_length)
                    current_length = current_length + silent_length + waveform_pred.shape[0]

            if num_runs > 1:
                filename = f'{title}-{str(i).zfill(3)}{suffix}'
            else:
                filename = title + suffix
            save_path = out_dir / filename
            if save_mel:
                print(f'| save mel: {save_path}')
                torch.save(result, save_path)
            else:
                print(f'| save audio: {save_path}')
                save_wav(result, save_path, hparams['audio_sample_rate'])
