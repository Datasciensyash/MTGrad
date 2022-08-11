import torch
import torch.nn as nn

from mtgrad.data_types import MTModel


class MTCriterion(nn.Module):
    def __init__(self):
        super(MTCriterion, self).__init__()
        self._criterion = nn.MSELoss()

    def forward(self, real_model: MTModel, synt_model: MTModel) -> torch.Tensor:
        return self._criterion(real_model.resistivity, synt_model.resistivity)
