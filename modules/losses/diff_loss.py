import torch.nn as nn
from torch import Tensor


class DiffusionLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _get_mask(non_padding, loss):
        if non_padding is None:
            return None
        mask = non_padding.transpose(1, 2).unsqueeze(1).to(loss)
        return mask.expand_as(loss)

    def _forward(self, x_recon, noise):
        return self.loss(x_recon, noise)

    def forward(self, x_recon: Tensor, noise: Tensor, non_padding: Tensor = None) -> Tensor:
        """
        :param x_recon: [B, 1, M, T]
        :param noise: [B, 1, M, T]
        :param non_padding: [B, T, M]
        """
        loss = self._forward(x_recon, noise)
        mask = self._get_mask(non_padding, loss)
        if mask is None:
            return loss.mean()
        return (loss * mask).sum() / mask.sum().clamp_min(1)
