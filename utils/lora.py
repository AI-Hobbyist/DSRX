import re
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
        target_modules: Iterable[str] = ('linear',), module_prefixes: Iterable[str] = None
):
    """Recursively replace Linear layers with LoRA-augmented ones.
    target_modules: iterable of regex patterns matched against full module names or 'linear' for all linears.
    module_prefixes: optional full-name prefixes that limit where LoRA can be injected.
    """
    from modules.commons.common_layers import XavierUniformInitLinear  # available in runtime
    module_prefixes = tuple(module_prefixes or ())

    def allowed_by_prefix(name: str) -> bool:
        if not module_prefixes:
            return True
        return any(name == p or name.startswith(p + '.') for p in module_prefixes)

    def replace(module: nn.Module, prefix: str = ''):
        for name, child in list(module.named_children()):
            full_name = f'{prefix}.{name}' if prefix else name
            # Recurse first
            replace(child, full_name)
            # Replace Linear-like
            if (
                    isinstance(child, (nn.Linear, XavierUniformInitLinear))
                    and not isinstance(child, LoRALinear)
                    and allowed_by_prefix(full_name)
                    and _match_module(full_name, target_modules)
            ):
                lora = LoRALinear(child.in_features, child.out_features, r=rank, alpha=alpha, bias=child.bias is not None)
                # copy base weights
                lora.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora.bias.data.copy_(child.bias.data)
                    lora.bias.requires_grad = False
                setattr(module, name, lora)

    replace(model)


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


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], prefix: str = 'model', strict: bool = False):
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
        elif subkey.endswith('.lora_B'):
            module_name = subkey[:-len('.lora_B')]
            module = dict(model.named_modules()).get(module_name)
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_B.copy_(tensor.to(module.lora_B.device, dtype=module.lora_B.dtype))
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
            if strict:
                raise KeyError(f'Unexpected LoRA key: {key}')


def merge_lora_into_model(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_lora_weights_()
