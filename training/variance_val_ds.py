import copy
import json
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import tqdm
import yaml
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from utils.hparams import hparams
from utils.plot import spec_to_figure


_VAL_WITH_DS_WORKER_CTX = None
_VAL_WITH_DS_WORKER_RUNNER = None
_VAL_WITH_DS_WORKER_TASK = None


@contextmanager
def temporary_hparams(new_hparams: dict):
    old_hparams = hparams.copy()
    hparams.clear()
    hparams.update(new_hparams)
    try:
        yield
    finally:
        hparams.clear()
        hparams.update(old_hparams)


def _init_val_with_ds_worker(ctx: dict):
    global _VAL_WITH_DS_WORKER_CTX, _VAL_WITH_DS_WORKER_RUNNER, _VAL_WITH_DS_WORKER_TASK
    _VAL_WITH_DS_WORKER_CTX = ctx
    _VAL_WITH_DS_WORKER_RUNNER = None
    _VAL_WITH_DS_WORKER_TASK = None


def _get_val_with_ds_worker_task():
    global _VAL_WITH_DS_WORKER_TASK
    if _VAL_WITH_DS_WORKER_TASK is not None:
        return _VAL_WITH_DS_WORKER_TASK

    ctx = _VAL_WITH_DS_WORKER_CTX
    train_hparams = ctx['train_hparams']
    device = torch.device(ctx['device'])
    with temporary_hparams(train_hparams):
        from modules.toplevel import DiffSingerVariance
        from utils.phoneme_utils import load_phoneme_dictionary

        model = DiffSingerVariance(vocab_size=len(load_phoneme_dictionary()))
        model.load_state_dict(ctx['model_state_dict'], strict=False)
        model.to(device)
        model.eval()

    _VAL_WITH_DS_WORKER_TASK = SimpleNamespace(
        model=model,
        device=device,
        global_rank=0,
        global_step=ctx['global_step']
    )
    return _VAL_WITH_DS_WORKER_TASK


def _run_val_with_ds_worker(spec):
    global _VAL_WITH_DS_WORKER_RUNNER
    ctx = _VAL_WITH_DS_WORKER_CTX
    task = _get_val_with_ds_worker_task()
    if _VAL_WITH_DS_WORKER_RUNNER is None:
        _VAL_WITH_DS_WORKER_RUNNER = VarianceDsValidationRunner(ctx['cfg'])
    runner = _VAL_WITH_DS_WORKER_RUNNER
    with torch.no_grad():
        if not runner._validated:
            runner._validate(task)
        return runner.run_spec(task, *spec)


class VarianceDsValidationRunner:
    def __init__(self, cfg: dict):
        if not isinstance(cfg, dict):
            raise ValueError('val_with_ds must be a mapping.')
        self.cfg = cfg
        self.acoustic_ckpt_dir = self._resolve_path(cfg.get('acoustic_ckpt_dir'))
        self.acoustic_ckpt_steps = cfg.get('acoustic_ckpt_steps')
        self.acoustic_vocoder = bool(cfg.get('acoustic_vocoder', True))
        self.release_acoustic_after_val = bool(cfg.get('release_acoustic_after_val', False))
        self.release_variance_after_val = bool(cfg.get('release_variance_after_val', False))
        self.overwrite_ds_dur = bool(cfg.get('overwrite_ds_dur', False))
        self.overwrite_ds_pitch = bool(cfg.get('overwrite_ds_pitch', False))
        self.overwrite_ds_var = bool(cfg.get('overwrite_ds_var', False))
        self.num_workers = max(0, int(cfg.get('num_workers', 0) or 0))
        self.worker_device = cfg.get('worker_device')
        self.mp_start_method = str(cfg.get('mp_start_method', 'spawn') or 'spawn')
        self.show_progress = bool(cfg.get('show_progress', True))
        self.variance_ckpts = self._normalize_variance_ckpts(cfg.get('variance_ckpts') or {})
        self.max_samples_per_val = cfg.get('max_samples_per_val')
        self.seed = int(cfg.get('seed', -1))
        self.spk_to_ds = self._build_spk_to_ds(
            ds_files=cfg.get('ds_files'),
            ds_val_spks=cfg.get('ds_val_spks')
        )
        self.acoustic_hparams = None
        self.acoustic_use_spk_id = False
        self.acoustic_infer = None
        self.variance_infers = {}
        self._validated = False

        if self.acoustic_ckpt_dir is None:
            raise ValueError('val_with_ds.acoustic_ckpt_dir is required when val_with_ds.ds_files is set.')
        if not self.spk_to_ds:
            raise ValueError(
                'val_with_ds.ds_files did not yield any valid speaker–.ds associations. '
                'Make sure at least one .ds file maps to a speaker that has test_prefixes in datasets.'
            )

    @staticmethod
    def is_enabled(cfg) -> bool:
        return isinstance(cfg, dict) and bool(cfg.get('acoustic_ckpt_dir')) and bool(cfg.get('ds_files'))

    @staticmethod
    def _resolve_path(value):
        if value is None or value == '':
            return None
        path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    # (path, lang) — lang is None if not specified
    DsEntry = Tuple[Path, Optional[str]]

    @staticmethod
    def _normalize_ds_entry(entry: Union[str, dict]) -> 'VarianceDsValidationRunner.DsEntry':
        """Normalize a single ds_files entry to (path_str, lang)."""
        if isinstance(entry, str):
            return (entry, None)
        if isinstance(entry, dict):
            return (entry.get('path'), entry.get('lang'))
        raise ValueError(
            f'val_with_ds.ds_files entry must be a string path or '
            f'{{path: ..., lang: ...}} dict, got {type(entry).__name__}.'
        )

    def _build_spk_to_ds(
        self, ds_files, ds_val_spks
    ) -> Dict[str, List[DsEntry]]:
        """Build speaker → .ds files mapping.

        If ``ds_val_spks`` is specified, those speakers are used directly.
        Otherwise, speakers are auto-detected from ``datasets`` entries that have
        non-empty ``test_prefixes``.

        Each speaker gets **all** .ds files from ``ds_files``.
        Each entry is ``(path, lang)`` where *lang* may be ``None``.
        Existence in the acoustic model's spk_map is validated later in ``_validate``.
        """
        if not isinstance(ds_files, list) or len(ds_files) == 0:
            raise ValueError('val_with_ds.ds_files must be a non-empty list.')

        # Auto-detect speakers from datasets with test_prefixes if not explicitly given
        if not ds_val_spks or (isinstance(ds_val_spks, list) and len(ds_val_spks) == 0):
            datasets = hparams.get('datasets', [])
            ds_val_spks = [
                ds['speaker'] for ds in datasets
                if ds.get('test_prefixes')
            ]
            if not ds_val_spks:
                raise ValueError(
                    'val_with_ds.ds_val_spks is empty and no datasets have test_prefixes. '
                    'Either specify ds_val_spks or add test_prefixes to at least one dataset.'
                )
        elif not isinstance(ds_val_spks, list):
            raise ValueError('val_with_ds.ds_val_spks must be a list of speaker names.')

        # Normalize + resolve all entries
        entries: List[VarianceDsValidationRunner.DsEntry] = []
        for raw in ds_files:
            path_str, lang = self._normalize_ds_entry(raw)
            entries.append((self._resolve_path(path_str), lang))

        spk_to_ds: Dict[str, List[VarianceDsValidationRunner.DsEntry]] = {}
        for spk in ds_val_spks:
            spk_to_ds[str(spk)] = list(entries)
        return spk_to_ds

    def _normalize_variance_ckpts(self, ckpt_cfg) -> dict:
        if not isinstance(ckpt_cfg, dict):
            raise ValueError('val_with_ds.variance_ckpts must be a mapping.')
        normalized = {}
        stage_aliases = {
            'dur': 'dur',
            'duration': 'dur',
            'pitch': 'pitch',
            'variance': 'variance',
            'var': 'variance'
        }
        for raw_stage, raw_value in ckpt_cfg.items():
            stage = stage_aliases.get(str(raw_stage))
            if stage is None:
                raise ValueError(f"Unknown val_with_ds.variance_ckpts stage: '{raw_stage}'.")
            if raw_value is None or raw_value == '':
                continue
            if isinstance(raw_value, (str, Path)):
                ckpt_dir = raw_value
                ckpt_steps = None
            elif isinstance(raw_value, dict):
                ckpt_dir = raw_value.get('ckpt_dir')
                ckpt_steps = raw_value.get('ckpt_steps')
            else:
                raise ValueError(f'val_with_ds.variance_ckpts.{raw_stage} must be a path or mapping.')
            if ckpt_dir is None or ckpt_dir == '':
                continue
            normalized[stage] = {
                'ckpt_dir': self._resolve_path(ckpt_dir),
                'ckpt_steps': ckpt_steps
            }
        return normalized

    def _load_acoustic_hparams(self):
        return self._load_ckpt_hparams(self.acoustic_ckpt_dir, 'acoustic')

    @staticmethod
    def _load_ckpt_hparams(ckpt_dir: Path, name: str):
        config_path = ckpt_dir / 'config.yaml'
        if not ckpt_dir.exists():
            raise FileNotFoundError(f'val_with_ds {name} ckpt dir does not exist: {ckpt_dir}')
        if not config_path.exists():
            raise FileNotFoundError(f'{name} config.yaml not found in {ckpt_dir}')
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_hparams = yaml.safe_load(f) or {}
        loaded_hparams['work_dir'] = str(ckpt_dir)
        loaded_hparams['infer'] = True
        loaded_hparams.setdefault('exp_name', ckpt_dir.name)
        return loaded_hparams

    @staticmethod
    def _load_json(path: Path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _iter_ds_specs(self) -> Iterable[Tuple[str, Path, Optional[str]]]:
        """Yield (speaker, ds_path, lang).  lang is None when not specified."""
        count = 0
        for spk, ds_entries in self.spk_to_ds.items():
            for ds_path, lang in ds_entries:
                if self.max_samples_per_val is not None and count >= int(self.max_samples_per_val):
                    return
                yield spk, ds_path, lang
                count += 1

    def _validate(self, task):
        self.acoustic_hparams = self._load_acoustic_hparams()
        self.acoustic_use_spk_id = bool(self.acoustic_hparams.get('use_spk_id', False))
        acoustic_spk_map_path = self.acoustic_ckpt_dir / 'spk_map.json'
        if self.acoustic_use_spk_id:
            if not acoustic_spk_map_path.exists():
                raise FileNotFoundError(f'Acoustic spk_map.json not found in {self.acoustic_ckpt_dir}')
            acoustic_spk_map = self._load_json(acoustic_spk_map_path)
            for spk in self.spk_to_ds:
                if spk not in acoustic_spk_map:
                    raise ValueError(f"Speaker '{spk}' (from ds_val_spks) is not in acoustic spk_map.json.")

        current_model_used = any(
            stage not in self.variance_ckpts and self._current_model_predicts(task, stage)
            for stage in ('dur', 'pitch', 'variance')
        )
        if current_model_used and hparams.get('use_spk_id', False):
            variance_spk_map_path = Path(hparams['work_dir']) / 'spk_map.json'
            if not variance_spk_map_path.exists():
                raise FileNotFoundError(f'Variance spk_map.json not found in {hparams["work_dir"]}')
            variance_spk_map = self._load_json(variance_spk_map_path)
            for spk in self.spk_to_ds:
                if spk not in variance_spk_map:
                    raise ValueError(f"Speaker '{spk}' (from ds_val_spks) is not in variance spk_map.json.")

        for stage, ckpt in self.variance_ckpts.items():
            ckpt_dir = ckpt['ckpt_dir']
            stage_hparams = self._load_ckpt_hparams(ckpt_dir, f'{stage} variance')
            ckpt['hparams'] = stage_hparams
            if stage_hparams.get('use_spk_id', False):
                spk_map_path = ckpt_dir / 'spk_map.json'
                if not spk_map_path.exists():
                    raise FileNotFoundError(f'{stage} variance spk_map.json not found in {ckpt_dir}')
                spk_map = self._load_json(spk_map_path)
                for spk in self.spk_to_ds:
                    if spk not in spk_map:
                        raise ValueError(
                            f"Speaker '{spk}' (from ds_val_spks) is not in {stage} variance spk_map.json."
                        )

        for spk, ds_path, _ in self._iter_ds_specs():
            if ds_path is None or not ds_path.exists():
                raise FileNotFoundError(f"val_with_ds.ds_files contains missing .ds file: {ds_path}")
            params = self._load_ds(ds_path)
            if len(params) == 0:
                raise ValueError(f'val_with_ds .ds file is empty: {ds_path}')

        if self.acoustic_vocoder:
            vocoder_ckpt = Path(self.acoustic_hparams.get('vocoder_ckpt', ''))
            if not vocoder_ckpt.is_absolute():
                vocoder_ckpt = Path.cwd() / vocoder_ckpt
            if not vocoder_ckpt.exists():
                raise FileNotFoundError(f'Acoustic vocoder ckpt not found: {vocoder_ckpt}')

        self._validated = True

    def _load_ds(self, ds_path: Path) -> List[OrderedDict]:
        data = self._load_json(ds_path)
        if not isinstance(data, list):
            data = [data]
        return [OrderedDict(item) for item in data]

    @staticmethod
    def _force_spk(params: List[OrderedDict], spk: str) -> List[OrderedDict]:
        forced = []
        spk_mix = {spk: 1.0}
        for param in params:
            param_copy = copy.deepcopy(param)
            param_copy['spk_mix'] = spk_mix
            param_copy['ph_spk_mix'] = spk_mix
            forced.append(param_copy)
        return forced

    @staticmethod
    def _apply_lang(params: List[OrderedDict], lang: Optional[str]) -> List[OrderedDict]:
        """Override lang field in all params if specified."""
        if lang is None:
            return params
        for param in params:
            param['lang'] = lang
        return params

    @staticmethod
    def _stage_predictions(stage: str) -> set:
        if stage == 'dur':
            return {'dur'}
        if stage == 'pitch':
            return {'pitch'}
        if stage == 'variance':
            return set(VARIANCE_CHECKLIST)
        raise ValueError(f'Unknown variance prediction stage: {stage}')

    @staticmethod
    def _current_model_predicts(task, stage: str) -> bool:
        if stage == 'dur':
            return bool(getattr(task.model.fs2, 'predict_dur', False))
        if stage == 'pitch':
            return bool(getattr(task.model, 'predict_pitch', False))
        if stage == 'variance':
            return bool(getattr(task.model, 'predict_variances', False))
        return False

    def _get_current_variance_infer(self, task, stage: str):
        from inference.ds_variance import DiffSingerVarianceInfer

        class TrainingVarianceInfer(DiffSingerVarianceInfer):
            def __init__(self, variance_task, predictions: set):
                self._training_model = variance_task.model
                super().__init__(device=variance_task.device, ckpt_steps=None, predictions=predictions)

            def build_model(self, ckpt_steps=None):
                return self._training_model

        return TrainingVarianceInfer(task, predictions=self._stage_predictions(stage))

    def _get_external_variance_infer(self, task, stage: str):
        if stage in self.variance_infers:
            return self.variance_infers[stage]
        from inference.ds_variance import DiffSingerVarianceInfer

        ckpt = self.variance_ckpts[stage]
        stage_hparams = ckpt.get('hparams') or self._load_ckpt_hparams(ckpt['ckpt_dir'], f'{stage} variance')
        ckpt['hparams'] = stage_hparams
        with temporary_hparams(stage_hparams):
            infer_ins = DiffSingerVarianceInfer(
                device=task.device,
                ckpt_steps=ckpt.get('ckpt_steps'),
                predictions=self._stage_predictions(stage)
            )
        self.variance_infers[stage] = (infer_ins, stage_hparams)
        return self.variance_infers[stage]

    def _iter_stage_infers(self, task):
        for stage in ('dur', 'pitch', 'variance'):
            if stage in self.variance_ckpts:
                yield stage, self._get_external_variance_infer(task, stage)
            elif self._current_model_predicts(task, stage):
                yield stage, (self._get_current_variance_infer(task, stage), None)

    def _complete_params(self, infer_ins, params: List[OrderedDict]) -> List[OrderedDict]:
        import librosa

        batches = []
        predictor_flags = []
        for i, param in enumerate(params):
            if infer_ins.auto_completion_mode:
                flag = (
                    infer_ins.model.fs2.predict_dur and param.get('ph_dur') is None,
                    infer_ins.model.predict_pitch and param.get('f0_seq') is None,
                    infer_ins.model.predict_variances and any(
                        param.get(v_name) is None for v_name in infer_ins.model.variance_prediction_list
                    )
                )
            else:
                predict_variances = infer_ins.model.predict_variances and infer_ins.global_predict_variances
                predict_pitch = infer_ins.model.predict_pitch and (
                    infer_ins.global_predict_pitch or (param.get('f0_seq') is None and predict_variances)
                )
                predict_dur = infer_ins.model.predict_dur and (
                    infer_ins.global_predict_dur or (param.get('ph_dur') is None and (predict_pitch or predict_variances))
                )
                flag = (predict_dur, predict_pitch, predict_variances)
            if param.get('ph_dur') is not None and not self.overwrite_ds_dur:
                flag = (False, flag[1], flag[2])
            if param.get('f0_seq') is not None and not self.overwrite_ds_pitch:
                flag = (flag[0], False, flag[2])
            # When dur is not predicted but needed for mel2ph alignment,
            # force dur prediction to avoid relying on .ds ph_dur which
            # may be incompatible (e.g. rounds to zero frames).
            need_dur_for_align = not flag[0] and (flag[1] or flag[2])
            if need_dur_for_align and infer_ins.model.fs2.predict_dur:
                flag = (True, flag[1], flag[2])
            predictor_flags.append(flag)
            batches.append(infer_ins.preprocess_input(
                param, idx=i,
                load_dur=not flag[0] and (flag[1] or flag[2]),
                load_pitch=not flag[1] and flag[2]
            ))

        results = []
        for param, flag, batch in zip(params, predictor_flags, batches):
            if 'seed' in param:
                torch.manual_seed(param['seed'] & 0xffff_ffff)
                torch.cuda.manual_seed_all(param['seed'] & 0xffff_ffff)
            elif self.seed >= 0:
                torch.manual_seed(self.seed & 0xffff_ffff)
                torch.cuda.manual_seed_all(self.seed & 0xffff_ffff)

            param_copy = copy.deepcopy(param)
            flag_saved = (
                infer_ins.model.fs2.predict_dur,
                infer_ins.model.predict_pitch,
                infer_ins.model.predict_variances
            )
            (
                infer_ins.model.fs2.predict_dur,
                infer_ins.model.predict_pitch,
                infer_ins.model.predict_variances
            ) = flag
            try:
                dur_pred, pitch_pred, variance_pred = infer_ins.forward_model(batch)
            finally:
                (
                    infer_ins.model.fs2.predict_dur,
                    infer_ins.model.predict_pitch,
                    infer_ins.model.predict_variances
                ) = flag_saved

            if dur_pred is not None:
                dur_pred = dur_pred[0].cpu().numpy()
                param_copy['ph_dur'] = ' '.join(str(round(dur, 6)) for dur in (dur_pred * infer_ins.timestep).tolist())
            if pitch_pred is not None:
                pitch_pred = pitch_pred[0].cpu().numpy()
                f0_pred = librosa.midi_to_hz(pitch_pred)
                param_copy['f0_seq'] = ' '.join(str(round(freq, 1)) for freq in f0_pred.tolist())
                param_copy['f0_timestep'] = str(infer_ins.timestep)
            if variance_pred is None:
                variance_pred = {}
            for v_name, v_tensor in variance_pred.items():
                if v_name not in VARIANCE_CHECKLIST:
                    continue
                if infer_ins.auto_completion_mode and param.get(v_name) is not None and not self.overwrite_ds_var:
                    continue
                if not infer_ins.auto_completion_mode and v_name not in infer_ins.variance_prediction_set:
                    continue
                v_pred = v_tensor[0].cpu().numpy()
                param_copy[v_name] = ' '.join(str(round(v, 4)) for v in v_pred.tolist())
                param_copy[f'{v_name}_timestep'] = str(infer_ins.timestep)
            results.append(param_copy)
        return results

    def _complete_params_with_stage(self, task, stage: str, params: List[OrderedDict]) -> List[OrderedDict]:
        for infer_stage, (infer_ins, stage_hparams) in self._iter_stage_infers(task):
            if infer_stage != stage:
                continue
            if stage_hparams is None:
                return self._complete_params(infer_ins, params)
            with temporary_hparams(stage_hparams):
                return self._complete_params(infer_ins, params)
        return params

    def _get_acoustic_infer(self, task):
        if self.acoustic_infer is not None:
            return self.acoustic_infer
        with temporary_hparams(self.acoustic_hparams):
            from inference.ds_acoustic import DiffSingerAcousticInfer
            import modules.vocoders  # noqa: F401
            self.acoustic_infer = DiffSingerAcousticInfer(
                device=task.device,
                load_vocoder=self.acoustic_vocoder,
                ckpt_steps=self.acoustic_ckpt_steps
            )
        return self.acoustic_infer

    def _release_acoustic_infer(self):
        if self.acoustic_infer is None:
            return
        del self.acoustic_infer
        self.acoustic_infer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_variance_infers(self):
        if not self.variance_infers:
            return
        self.variance_infers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _mel_to_time_major(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() != 2:
            return mel
        mel_bins = int(self.acoustic_hparams.get('audio_num_mel_bins') or 0)
        if mel_bins > 0:
            if mel.shape[-1] == mel_bins:
                return mel
            if mel.shape[0] == mel_bins:
                return mel.transpose(0, 1)
        return mel if mel.shape[0] >= mel.shape[1] else mel.transpose(0, 1)

    def _log_acoustic(self, task, spk: str, ds_path: Path, params: List[OrderedDict]):
        acoustic_infer = self._get_acoustic_infer(task)
        with temporary_hparams(self.acoustic_hparams):
            mels = []
            wavs = []
            for idx, param in enumerate(params):
                batch = acoustic_infer.preprocess_input(param, idx=idx)
                mel = acoustic_infer.forward_model(batch)
                mels.append(self._mel_to_time_major(mel[0]))
                if self.acoustic_vocoder:
                    wav = acoustic_infer.run_vocoder(mel, f0=batch['f0'])
                    wavs.append(wav[0].detach().cpu())
            # Concatenate segments → full-song mel
            full_mel = torch.cat(mels, dim=0)  # [T_total, n_mel]
            tag = f'val_with_ds/{ds_path.stem}/{spk}'
            task.logger.all_rank_experiment.add_figure(
                f'{tag}/mel',
                spec_to_figure(
                    full_mel,
                    self.acoustic_hparams.get('mel_vmin'),
                    self.acoustic_hparams.get('mel_vmax'),
                    f'{spk} - {ds_path.stem}'
                ),
                global_step=task.global_step
            )
            if self.acoustic_vocoder and wavs:
                full_wav = torch.cat(wavs, dim=-1)
                task.logger.all_rank_experiment.add_audio(
                    f'{tag}/audio',
                    full_wav,
                    sample_rate=self.acoustic_hparams['audio_sample_rate'],
                    global_step=task.global_step
                )

    def _synthesize_acoustic(self, task, spk: str, ds_path: Path, params: List[OrderedDict]):
        acoustic_infer = self._get_acoustic_infer(task)
        with temporary_hparams(self.acoustic_hparams):
            mels = []
            wavs = []
            for idx, param in enumerate(params):
                batch = acoustic_infer.preprocess_input(param, idx=idx)
                mel = acoustic_infer.forward_model(batch)
                mels.append(self._mel_to_time_major(mel[0]))
                if self.acoustic_vocoder:
                    wav = acoustic_infer.run_vocoder(mel, f0=batch['f0'])
                    wavs.append(wav[0].detach().cpu())
            full_mel = torch.cat(mels, dim=0)  # [T_total, n_mel]
            full_wav = torch.cat(wavs, dim=-1) if self.acoustic_vocoder and wavs else None
            return {
                'tag': f'val_with_ds/{ds_path.stem}/{spk}',
                'title': f'{spk} - {ds_path.stem}',
                'mel': full_mel.detach().cpu(),
                'wav': full_wav,
                'sample_rate': self.acoustic_hparams['audio_sample_rate'],
                'mel_vmin': self.acoustic_hparams.get('mel_vmin'),
                'mel_vmax': self.acoustic_hparams.get('mel_vmax')
            }

    def _log_acoustic_result(self, task, result: dict):
        task.logger.all_rank_experiment.add_figure(
            f"{result['tag']}/mel",
            spec_to_figure(
                result['mel'],
                result.get('mel_vmin'),
                result.get('mel_vmax'),
                result['title']
            ),
            global_step=task.global_step
        )
        if result.get('wav') is not None:
            task.logger.all_rank_experiment.add_audio(
                f"{result['tag']}/audio",
                result['wav'],
                sample_rate=result['sample_rate'],
                global_step=task.global_step
            )

    def run_spec(self, task, spk: str, ds_path: Path, lang: Optional[str]):
        params = self._load_ds(ds_path)
        if self.acoustic_use_spk_id:
            params = self._force_spk(params, spk)
        params = self._apply_lang(params, lang)
        completed_params = params
        for stage in ('dur', 'pitch', 'variance'):
            completed_params = self._complete_params_with_stage(task, stage, completed_params)
        return self._synthesize_acoustic(task, spk, ds_path, completed_params)

    def _iter_progress(self, items, desc, unit, total=None):
        return tqdm.tqdm(
            items,
            desc=desc,
            unit=unit,
            total=total,
            leave=True,
            dynamic_ncols=True,
            disable=not self.show_progress
        )

    @staticmethod
    def _spec_label(spec) -> str:
        spk, ds_path, _ = spec
        return f'{spk}/{ds_path.stem}'

    def _run_serial(self, task, specs):
        success_count = 0
        pbar = self._iter_progress(specs, desc='val_with_ds', unit='item')
        for spec in pbar:
            pbar.set_postfix_str(self._spec_label(spec))
            spk, ds_path, lang = spec
            try:
                result = self.run_spec(task, spk, ds_path, lang)
                self._log_acoustic_result(task, result)
                success_count += 1
            except Exception as exc:
                rank_zero_warn(f'val_with_ds failed for {spk}:{ds_path}: {exc}')
                rank_zero_warn(traceback.format_exc())
        return success_count

    def _run_multiprocess(self, task, specs):
        worker_count = min(self.num_workers, len(specs))
        ctx = {
            'cfg': self.cfg,
            'train_hparams': hparams.copy(),
            'model_state_dict': {
                k: v.detach().cpu()
                for k, v in task.model.state_dict().items()
            },
            'global_step': task.global_step,
            'device': str(self.worker_device or task.device)
        }
        mp_context = multiprocessing.get_context(self.mp_start_method)
        success_count = 0
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=_init_val_with_ds_worker,
            initargs=(ctx,)
        ) as executor:
            future_to_spec = {
                executor.submit(_run_val_with_ds_worker, spec): spec
                for spec in specs
            }
            pbar = self._iter_progress(
                as_completed(future_to_spec),
                desc=f'val_with_ds x{worker_count}',
                unit='item',
                total=len(future_to_spec)
            )
            for future in pbar:
                spec = future_to_spec[future]
                pbar.set_postfix_str(self._spec_label(spec))
                spk, ds_path, _ = spec
                try:
                    result = future.result()
                    self._log_acoustic_result(task, result)
                    success_count += 1
                except Exception as exc:
                    rank_zero_warn(f'val_with_ds failed for {spk}:{ds_path}: {exc}')
                    rank_zero_warn(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        return success_count

    @torch.no_grad()
    def run(self, task):
        if getattr(task, 'global_rank', 0) != 0:
            return
        if not self._validated:
            self._validate(task)

        was_training = task.model.training
        task.model.eval()
        try:
            specs = list(self._iter_ds_specs())
            if self.num_workers > 1 and len(specs) > 1:
                success_count = self._run_multiprocess(task, specs)
            else:
                success_count = self._run_serial(task, specs)
            if specs and success_count == 0:
                raise RuntimeError('val_with_ds failed for all configured .ds files.')
        finally:
            if self.release_variance_after_val:
                self._release_variance_infers()
            if self.release_acoustic_after_val:
                self._release_acoustic_infer()
            if was_training:
                task.model.train()
