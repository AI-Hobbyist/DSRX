import copy
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from training.variance_val_ds import VarianceDsValidationRunner, silence_output, temporary_hparams
from utils.hparams import hparams
from utils.plot import spec_to_figure
from utils.tensorboard_utils import validation_tb_tag


class AllInOneDsValidationRunner(VarianceDsValidationRunner):
    def __init__(self, cfg: dict):
        if not isinstance(cfg, dict):
            raise ValueError('validation.val_with_ds must be a mapping.')
        if cfg.get('acoustic_ckpt_dir'):
            raise ValueError('all-in-one .ds validation must not set acoustic_ckpt_dir.')
        if any((cfg.get('variance_ckpts') or {}).values()):
            raise ValueError('all-in-one .ds validation must not set variance_ckpts.')
        self.cfg = cfg
        self.acoustic_ckpt_dir = None
        self.acoustic_ckpt_steps = None
        self.acoustic_vocoder = bool(cfg.get('acoustic_vocoder', cfg.get('val_with_vocoder', True)))
        self.release_acoustic_after_val = False
        self.release_variance_after_val = False
        self.overwrite_ds_dur = bool(cfg.get('overwrite_ds_dur', False))
        self.overwrite_ds_pitch = bool(cfg.get('overwrite_ds_pitch', False))
        self.overwrite_ds_var = bool(cfg.get('overwrite_ds_var', False))
        self.show_progress = bool(cfg.get('show_progress', True))
        self.verbose = bool(cfg.get('verbose', False))
        self.infer_steps = self._normalize_infer_steps(cfg.get('infer_steps') or {})
        self.variance_ckpts = {}
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

    @staticmethod
    def is_enabled(cfg) -> bool:
        return isinstance(cfg, dict) and bool(cfg.get('enabled', False)) and bool(cfg.get('ds_files'))

    def _validate(self, task):
        if getattr(task.model, 'variance', None) is None:
            raise RuntimeError('all-in-one .ds validation requires the variance submodule.')
        if getattr(task.model, 'acoustic', None) is None:
            raise RuntimeError('all-in-one .ds validation requires the acoustic submodule.')
        self.acoustic_hparams = hparams.copy()
        self.acoustic_use_spk_id = bool(hparams.get('use_spk_id', False))

        if hparams.get('use_spk_id', False):
            spk_map_path = Path(hparams['work_dir']) / 'spk_map.json'
            if not spk_map_path.exists():
                raise FileNotFoundError(f'all-in-one spk_map.json not found in {hparams["work_dir"]}')
            spk_map = self._load_json(spk_map_path)
            for spk in self.spk_to_ds:
                if spk not in spk_map:
                    raise ValueError(f"Speaker '{spk}' (from ds_val_spks) is not in all-in-one spk_map.json.")

        for _, ds_path, _ in self._iter_ds_specs():
            if ds_path is None or not ds_path.exists():
                raise FileNotFoundError(f"validation.val_with_ds.ds_files contains missing .ds file: {ds_path}")
            if len(self._load_ds(ds_path)) == 0:
                raise ValueError(f'all-in-one validation .ds file is empty: {ds_path}')

        if self.acoustic_vocoder:
            vocoder_ckpt = Path(hparams.get('vocoder_ckpt', ''))
            if not vocoder_ckpt.is_absolute():
                vocoder_ckpt = Path.cwd() / vocoder_ckpt
            if not vocoder_ckpt.exists():
                raise FileNotFoundError(f'all-in-one vocoder ckpt not found: {vocoder_ckpt}')
        self._validated = True

    @staticmethod
    def _current_model_predicts(task, stage: str) -> bool:
        variance_model = getattr(task.model, 'variance', None)
        if variance_model is None:
            return False
        if stage == 'dur':
            return bool(getattr(variance_model.fs2, 'predict_dur', False))
        if stage == 'pitch':
            return bool(getattr(variance_model, 'predict_pitch', False))
        if stage == 'variance':
            return bool(getattr(variance_model, 'predict_variances', False))
        return False

    def _get_current_variance_infer(self, task, stage: str):
        from inference.ds_variance import DiffSingerVarianceInfer

        class TrainingAllInOneVarianceInfer(DiffSingerVarianceInfer):
            def __init__(self, all_in_one_task, predictions: set):
                self._training_model = all_in_one_task.model.variance
                super().__init__(device=all_in_one_task.device, ckpt_steps=None, predictions=predictions)

            def build_model(self, ckpt_steps=None):
                return self._training_model

        return TrainingAllInOneVarianceInfer(task, predictions=self._stage_predictions(stage))

    def _iter_stage_infers(self, task):
        for stage in ('dur', 'pitch', 'variance'):
            if self._current_model_predicts(task, stage):
                yield stage, (self._get_current_variance_infer(task, stage), None)

    def _complete_params_with_stage(self, task, stage: str, params: List[OrderedDict]) -> List[OrderedDict]:
        for infer_stage, (infer_ins, _) in self._iter_stage_infers(task):
            if infer_stage == stage:
                with temporary_hparams(self._apply_infer_steps(hparams.copy(), stage)):
                    return self._complete_params(infer_ins, params)
        return params

    def _get_acoustic_infer(self, task):
        if self.acoustic_infer is not None:
            return self.acoustic_infer
        from inference.ds_acoustic import DiffSingerAcousticInfer

        class TrainingAllInOneAcousticInfer(DiffSingerAcousticInfer):
            def __init__(self, all_in_one_task, load_vocoder=True):
                self._training_model = all_in_one_task.model.acoustic
                super().__init__(
                    device=all_in_one_task.device,
                    load_model=True,
                    load_vocoder=load_vocoder,
                    ckpt_steps=None
                )

            def build_model(self, ckpt_steps=None):
                return self._training_model

        with temporary_hparams(self._apply_infer_steps(hparams.copy(), 'acoustic')):
            self.acoustic_infer = TrainingAllInOneAcousticInfer(task, load_vocoder=self.acoustic_vocoder)
        return self.acoustic_infer

    def _add_ds_labels(self, specs):
        labeled = super()._add_ds_labels(specs)
        return [
            (tag.replace('var_mel', 'all_in_one_ds_mel', 1), ds_label, spk, ds_path, lang)
            for tag, ds_label, spk, ds_path, lang in labeled
        ]

    def _synthesize_acoustic(
        self, task, tag: str, ds_label: str, spk: str, ds_path: Path, params: List[OrderedDict]
    ):
        result = super()._synthesize_acoustic(task, tag, ds_label, spk, ds_path, params)
        result['audio_tag'] = tag.replace('all_in_one_ds_mel', 'all_in_one_ds_audio', 1)
        return result
