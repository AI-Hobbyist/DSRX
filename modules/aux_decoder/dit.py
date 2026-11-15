import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from modules.backbones.dit import DiTBlock


class DiTAuxDecoder(nn.Module):
    """Single-pass mel predictor using DiT blocks conditioned on valid frames."""

    supports_valid_mask = True

    def __init__(
        self,
        in_dims,
        out_dims,
        /,
        *,
        num_channels=256,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4,
        rope_base=10000.0,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        use_gradient_checkpointing=True,
    ):
        super().__init__()
        for name, value in (
            ("num_channels", num_channels),
            ("num_layers", num_layers),
            ("num_heads", num_heads),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if num_channels % num_heads or (num_channels // num_heads) % 2:
            raise ValueError(
                "DiT AUX requires an even integer attention head dimension."
            )
        if mlp_ratio <= 0 or int(num_channels * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden dimension.")
        if rope_base <= 0 or layer_norm_eps <= 0:
            raise ValueError("rope_base and layer_norm_eps must be positive.")
        if not 0 <= attention_dropout < 1 or not 0 <= mlp_dropout < 1:
            raise ValueError("DiT AUX dropout must be in [0, 1).")
        if not isinstance(use_gradient_checkpointing, bool):
            raise ValueError("use_gradient_checkpointing must be a boolean.")

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.input_projection = nn.Linear(in_dims, num_channels)
        self.condition_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(num_channels, num_channels)
        )
        self.blocks = nn.ModuleList(
            DiTBlock(
                num_channels,
                num_heads,
                mlp_ratio,
                rope_base,
                layer_norm_eps,
                attention_dropout,
                mlp_dropout,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(num_channels, eps=layer_norm_eps)
        self.output_projection = nn.Linear(num_channels, out_dims)

    def forward(self, condition, infer=False, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones_like(condition[:, :, 0], dtype=torch.bool)
        if not torch.jit.is_tracing():
            if (
                valid_mask.dtype != torch.bool
                or valid_mask.shape != condition.shape[:2]
            ):
                raise ValueError("valid_mask must be boolean with shape [B, T].")
            if not torch.all(valid_mask.any(dim=1)):
                raise ValueError(
                    "Each DiT AUX sample requires at least one valid frame."
                )

        query_mask = valid_mask[:, :, None]
        x = self.input_projection(condition) * query_mask
        # AUX has no diffusion time: pooled valid conditioning drives adaLN.
        pooled = x.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).to(x.dtype)
        conditioning = self.condition_projection(pooled)
        for block in self.blocks:
            if (
                self.training
                and torch.is_grad_enabled()
                and self.use_gradient_checkpointing
            ):
                x = checkpoint(block, x, conditioning, valid_mask, use_reentrant=False)
            else:
                x = block(x, conditioning, valid_mask)
        return self.output_projection(self.final_norm(x)) * query_mask
