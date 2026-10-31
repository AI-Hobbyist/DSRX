import torch
from torch import nn


class DSLightAuxBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout_rate):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation, groups=channels,
        )
        self.expand = nn.Conv1d(channels, 2 * channels, 1)
        self.project = nn.Conv1d(2 * channels, channels, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        a, b = self.expand(self.depthwise(x)).chunk(2, dim=1)
        h = self.project(torch.cat((a * torch.sigmoid(b), a), dim=1))
        return x + self.dropout(h)


class DSLightAuxDecoder(nn.Module):
    def __init__(self, in_dims, out_dims, /, *, num_channels=128, num_layers=4,
                 kernel_size=7, dilation_cycle=(1, 3, 9), dropout_rate=0.0):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd integer')
        if not dilation_cycle or any(d < 1 for d in dilation_cycle):
            raise ValueError('dilation_cycle must contain positive integers')
        self.input_projection = nn.Conv1d(in_dims, num_channels, 1)
        self.blocks = nn.ModuleList(
            DSLightAuxBlock(num_channels, kernel_size, dilation_cycle[i % len(dilation_cycle)], dropout_rate)
            for i in range(num_layers)
        )
        self.output_projection = nn.Conv1d(
            num_channels, out_dims, kernel_size, padding=kernel_size // 2,
        )

    def forward(self, condition, infer=False):
        x = self.input_projection(condition.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        return self.output_projection(x).transpose(1, 2)
