import re
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        assert r > 0, 'LoRA rank r must be positive'
        self.lora_r = int(r)
        self.lora_alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        # LoRA parameters (A: in->r, B: r->out)
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.reset_lora_parameters()
        # By default, base weight frozen (training on LoRA only)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) if 'math' in globals() else nn.init.normal_(self.lora_A, 0, 0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, input):
        # base
        result = F.linear(input, self.weight, self.bias)
        # lora update: x @ A @ B^T
        lora_out = F.linear(F.linear(input, self.lora_A.t()), self.lora_B.t())
        return result + self.scaling * lora_out

    @torch.no_grad()
    def merge_lora_weights_(self):
        # W' = W + scale * (B^T @ A^T) -> [out_features, in_features]
        delta = (self.lora_B.t() @ self.lora_A.t()) * self.scaling
        self.weight.data += delta.to(self.weight.data.dtype)
        # zero LoRA params to avoid double counting if kept
        self.lora_A.zero_()
        self.lora_B.zero_()


def _match_module(name: str, patterns: Iterable[str]) -> bool:
    for p in patterns:
        if p == '*' or p.lower() == 'linear':
            return True
        if re.search(p, name):
            return True
    return False


def inject_lora(
    model: nn.Module, *, rank: int = 8, alpha: int = 16,
    target_modules: Iterable[str] = ('linear',), require_match: bool = False
) -> list[str]:
    """Recursively replace Linear layers with LoRA-augmented ones.
    target_modules: iterable of regex patterns matched against full module names or 'linear' for all linears.
    """
    from modules.commons.common_layers import XavierUniformInitLinear  # available in runtime

    matched_modules = []

    def replace(module: nn.Module, prefix: str = ''):
        for name, child in list(module.named_children()):
            full_name = f'{prefix}.{name}' if prefix else name
            # Recurse first
            replace(child, full_name)
            # Replace Linear-like
            if isinstance(child, (nn.Linear, XavierUniformInitLinear)) and _match_module(full_name, target_modules):
                lora = LoRALinear(child.in_features, child.out_features, r=rank, alpha=alpha, bias=child.bias is not None)
                # copy base weights
                lora.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora.bias.data.copy_(child.bias.data)
                    lora.bias.requires_grad = False
                setattr(module, name, lora)
                matched_modules.append(full_name)

    replace(model)
    if require_match and not matched_modules:
        raise ValueError(
            f'LoRA target patterns matched no linear modules: {list(target_modules)}'
        )
    return matched_modules


def mark_only_lora_as_trainable(model: nn.Module, train_bias: bool = False):
    for n, p in model.named_parameters():
        if ('.lora_A' in n) or ('.lora_B' in n) or (train_bias and n.endswith('.bias')):
            p.requires_grad = True
        else:
            p.requires_grad = False


def extract_lora_state_dict(model: nn.Module, prefix: str = 'model') -> Dict[str, torch.Tensor]:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f'{prefix}.{name}.lora_A'] = module.lora_A.detach().cpu()
            state[f'{prefix}.{name}.lora_B'] = module.lora_B.detach().cpu()
            state[f'{prefix}.{name}.lora_alpha'] = torch.tensor(module.lora_alpha)
            state[f'{prefix}.{name}.lora_r'] = torch.tensor(module.lora_r)
    return state


def load_lora_state_dict(
        model: nn.Module, state_dict: Dict[str, torch.Tensor],
        prefix: str = 'model', strict: bool = False
) -> list[str]:
    expected = {
        f'{prefix}.{name}.{suffix}'
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
        for suffix in ('lora_A', 'lora_B')
    }
    loaded = set()
    # Filter keys with given prefix
    for key, tensor in state_dict.items():
        if not key.startswith(prefix + '.'):
            continue
        subkey = key[len(prefix) + 1:]
        if subkey.endswith('.lora_A'):
            module_name = subkey[:-len('.lora_A')]
            module = dict(model.named_modules()).get(module_name)
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_A.copy_(tensor.to(module.lora_A.device, dtype=module.lora_A.dtype))
                loaded.add(key)
        elif subkey.endswith('.lora_B'):
            module_name = subkey[:-len('.lora_B')]
            module = dict(model.named_modules()).get(module_name)
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_B.copy_(tensor.to(module.lora_B.device, dtype=module.lora_B.dtype))
                loaded.add(key)
        elif subkey.endswith('.lora_alpha'):
            module_name = subkey[:-len('.lora_alpha')]
            module = dict(model.named_modules()).get(module_name)
            if isinstance(module, LoRALinear):
                module.lora_alpha = int(tensor.item())
                module.scaling = float(module.lora_alpha) / float(module.lora_r)
        elif subkey.endswith('.lora_r'):
            # rank stored for info; not changing structure dynamically
            pass
        else:
            if strict and '.lora_' in subkey:
                raise KeyError(f'Unexpected LoRA key: {key}')
    if strict:
        missing = sorted(expected - loaded)
        if missing:
            raise KeyError(f'Missing LoRA weights: {missing}')
    return sorted(loaded)


def uses_dit_backend(config: Dict) -> bool:
    backbone_types = [config.get('backbone_type')]
    for key in ('pitch_prediction_args', 'variances_prediction_args'):
        predictor_config = config.get(key, {})
        if isinstance(predictor_config, dict):
            backbone_types.append(predictor_config.get('backbone_type'))
    return 'dit' in backbone_types


def load_dit_lora_for_inference(
        model: nn.Module, lora_config: Dict, *, work_dir,
        device, prefix: str = 'model'
) -> list[str]:
    from utils import load_ckpt
    from utils.training_utils import get_latest_checkpoint_path

    base_checkpoint = lora_config.get('base_ckpt')
    if not base_checkpoint:
        raise ValueError('DiT LoRA inference requires lora.base_ckpt.')
    load_ckpt(
        model, base_checkpoint, prefix_in_ckpt=prefix,
        strict=True, device=device, prefer_inference=True
    )
    matched_modules = inject_lora(
        model,
        rank=int(lora_config.get('rank', 8)),
        alpha=int(lora_config.get('alpha', 16)),
        target_modules=lora_config.get('target_modules', ['linear']),
        require_match=True,
    )
    checkpoint_path = get_latest_checkpoint_path(Path(work_dir))
    if checkpoint_path is None:
        raise FileNotFoundError(f'No DiT LoRA checkpoint found in {work_dir}.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    loaded = load_lora_state_dict(model, state_dict, prefix=prefix, strict=True)
    print(
        f'| DiT LoRA inference loaded {len(matched_modules)} modules '
        f'and {len(loaded)} adapter tensors from {checkpoint_path}'
    )
    return matched_modules


def load_dit_lora_for_export(
        model: nn.Module, lora_config: Dict, *, work_dir, ckpt_steps,
        device, prefix: str = 'model'
) -> list[str]:
    from utils import load_ckpt
    from utils.training_utils import get_latest_checkpoint_path

    base_checkpoint = lora_config.get('base_ckpt')
    if not base_checkpoint:
        raise ValueError('DiT LoRA export requires lora.base_ckpt.')
    load_ckpt(
        model, base_checkpoint, prefix_in_ckpt=prefix,
        strict=True, device=device, prefer_inference=True
    )
    matched_modules = inject_lora(
        model,
        rank=int(lora_config.get('rank', 8)),
        alpha=int(lora_config.get('alpha', 16)),
        target_modules=lora_config.get('target_modules', ['linear']),
        require_match=True,
    )
    work_dir = Path(work_dir)
    checkpoint_path = (
        work_dir / f'model_ckpt_steps_{int(ckpt_steps)}.ckpt'
        if ckpt_steps is not None
        else get_latest_checkpoint_path(work_dir)
    )
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f'No DiT LoRA checkpoint found in {work_dir}'
            + (f' at step {ckpt_steps}.' if ckpt_steps is not None else '.')
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    loaded = load_lora_state_dict(model, state_dict, prefix=prefix, strict=True)
    print(
        f'| DiT LoRA export loaded {len(matched_modules)} modules '
        f'and {len(loaded)} adapter tensors from {checkpoint_path}'
    )
    return matched_modules


def merge_lora_into_model(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_lora_weights_()
