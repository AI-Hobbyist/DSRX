from pathlib import Path
from typing import Dict, List, Tuple

import torch

from deployment.exporters.acoustic_exporter import DiffSingerAcousticExporter
from deployment.exporters.variance_exporter import DiffSingerVarianceExporter
from utils.hparams import hparams
from utils.training_utils import get_latest_checkpoint_path


def _load_checkpoint_state_dict(work_dir, ckpt_steps=None, device='cpu'):
    work_dir = Path(work_dir)
    if work_dir.is_file():
        checkpoint_path = work_dir
    elif ckpt_steps is not None:
        checkpoint_path = work_dir / f'model_ckpt_steps_{int(ckpt_steps)}.ckpt'
    else:
        latest = get_latest_checkpoint_path(work_dir)
        if latest is None:
            raise RuntimeError(f"No checkpoint found in '{work_dir}'.")
        checkpoint_path = Path(latest)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"| load all-in-one checkpoint from '{checkpoint_path}'.")
    return checkpoint.get('state_dict', checkpoint)


def _extract_submodule_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str):
    prefix = prefix.rstrip('.')
    remapped = {
        key[len(prefix) + 1:]: value
        for key, value in state_dict.items()
        if key.startswith(prefix + '.')
    }
    if not remapped:
        raise RuntimeError(f"No parameters found with prefix '{prefix}' in all-in-one checkpoint.")
    return remapped


class AllInOneCompatExporter:
    def __init__(
            self,
            device,
            cache_dir: Path,
            ckpt_steps: int = None,
            precision: str = 'fp32',
            quantize: bool = False,
    ):
        self.device = device
        self.cache_dir = cache_dir
        self.ckpt_steps = ckpt_steps
        self.precision = precision
        self.quantize = quantize
        export_prefixes = hparams.get('all_in_one', {}).get('export_prefixes', {})
        self.acoustic_prefix = export_prefixes.get('acoustic', 'model.acoustic')
        self.variance_prefix = export_prefixes.get('variance', 'model.variance')
        self.state_dict = _load_checkpoint_state_dict(hparams['work_dir'], ckpt_steps=ckpt_steps, device=device)

    def export(
            self,
            path: Path,
            freeze_gender: float = 0.,
            freeze_velocity: bool = False,
            freeze_glide: bool = False,
            freeze_expr: bool = False,
            export_spk: List[Tuple[str, Dict[str, float]]] = None,
            freeze_spk: Tuple[str, Dict[str, float]] = None,
    ):
        acoustic_dir = path / 'acoustic'
        variance_dir = path / 'variance'
        acoustic_state = _extract_submodule_state_dict(self.state_dict, self.acoustic_prefix)
        variance_state = _extract_submodule_state_dict(self.state_dict, self.variance_prefix)

        acoustic_exporter = DiffSingerAcousticExporter(
            device=self.device,
            cache_dir=self.cache_dir,
            ckpt_steps=self.ckpt_steps,
            freeze_gender=freeze_gender,
            freeze_velocity=freeze_velocity,
            export_spk=export_spk,
            freeze_spk=freeze_spk,
            precision=self.precision,
            quantize=self.quantize,
            state_dict_override=acoustic_state,
        )
        acoustic_exporter.export(acoustic_dir)

        variance_exporter = DiffSingerVarianceExporter(
            device=self.device,
            cache_dir=self.cache_dir,
            ckpt_steps=self.ckpt_steps,
            freeze_glide=freeze_glide,
            freeze_expr=freeze_expr,
            export_spk=export_spk,
            freeze_spk=freeze_spk,
            state_dict_override=variance_state,
        )
        variance_exporter.export(variance_dir)
