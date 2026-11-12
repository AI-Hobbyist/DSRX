import math

import torch
from torch import nn


class DSLightBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout_rate):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation, groups=channels,
        )
        self.expand = nn.Conv1d(channels, 2 * channels, 1)
        self.project = nn.Conv1d(2 * channels, channels, 1)
        self.condition = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, condition, time, condition_prepared=False):
        a, b = self.expand(self.depthwise(x)).chunk(2, dim=1)
        h = self.project(torch.cat((a * torch.sigmoid(b), a), dim=1))
        projected_condition = condition if condition_prepared else self.condition(condition)
        if self.training:
            h = h * (1 + time) + time + projected_condition
        else:
            h = torch.addcmul(time, h, 1 + time) + projected_condition
        return x + self.dropout(h)


class DSLightBackbone(nn.Module):
    supports_valid_mask = False

    def __init__(self, in_dims, n_feats, *, num_channels=192, num_layers=12,
                 kernel_size=7, dilation_cycle=(1, 3, 9, 27, 81),
                 cond_channels=256, time_embed_dim=64, dropout_rate=0.0):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd integer')
        if not dilation_cycle or any(d < 1 for d in dilation_cycle):
            raise ValueError('dilation_cycle must contain positive integers')
        if time_embed_dim < 4 or time_embed_dim % 2:
            raise ValueError('time_embed_dim must be an even integer of at least 4')
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.time_embed_dim = time_embed_dim
        half_dim = time_embed_dim // 2
        self.register_buffer(
            'time_frequencies',
            torch.exp(torch.arange(half_dim, dtype=torch.float32)
                      * (-math.log(10000.0) / (half_dim - 1))),
            persistent=False,
        )
        self.input_projection = nn.Conv1d(in_dims * n_feats, num_channels, 1)
        self.condition_projection = nn.Conv1d(cond_channels, num_channels, 1)
        self.time_projection = nn.Linear(time_embed_dim, num_channels)
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size,
                      padding=kernel_size // 2, groups=num_channels),
            nn.Conv1d(num_channels, num_channels, 1),
        )
        self.blocks = nn.ModuleList(
            DSLightBlock(num_channels, kernel_size, dilation_cycle[i % len(dilation_cycle)], dropout_rate)
            for i in range(num_layers)
        )
        self.output_projection = nn.Conv1d(num_channels, in_dims * n_feats, 1)

    def _apply(self, fn):
        result = super()._apply(fn)
        half_dim = self.time_embed_dim // 2
        self.time_frequencies = torch.exp(
            torch.arange(half_dim, device=self.time_projection.weight.device,
                         dtype=torch.float32)
            * (-math.log(10000.0) / (half_dim - 1))
        )
        return result

    @torch.no_grad()
    def prepare_condition(self, cond):
        """Return per-layer projections owned by one sampling invocation."""
        condition = self.condition_projection(cond)
        if len(self.blocks) == 0:
            return cond
        return torch.stack([block.condition(condition) for block in self.blocks])

    def forward(self, spec, diffusion_step, cond):
        half_dim = self.time_embed_dim // 2
        if spec.dtype == torch.float32 and self.time_frequencies.dtype == torch.float32:
            frequencies = self.time_frequencies
        else:
            # Preserve the original low-precision arange/exp rounding.
            frequencies = torch.exp(
                torch.arange(half_dim, device=diffusion_step.device, dtype=spec.dtype)
                * (-math.log(10000.0) / (half_dim - 1))
            )
        phase = diffusion_step.reshape(-1, 1) * frequencies.unsqueeze(0)
        time = self.time_projection(torch.cat((phase.sin(), phase.cos()), dim=1)).unsqueeze(-1)
        condition_prepared = cond.dim() == 4
        condition = cond if condition_prepared else self.condition_projection(cond)
        x = self.stem(self.input_projection(spec.flatten(1, 2)))
        for index, block in enumerate(self.blocks):
            layer_condition = condition[index] if condition_prepared else condition
            x = block(x, layer_condition, time, condition_prepared)
        x = self.output_projection(x)
        return x.reshape(spec.shape[0], self.n_feats, self.in_dims, spec.shape[-1])
