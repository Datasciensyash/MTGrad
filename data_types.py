from dataclasses import dataclass

import torch


@dataclass
class MTDataSample:
    resistivity: torch.Tensor
    apparent_resistivity: torch.Tensor
    impedance_phase: torch.Tensor
    periods: torch.Tensor
    layer_powers: torch.Tensor

    def __post_init__(self):
        if len(self.resistivity.shape) != 3:
            raise ValueError(
                "MTDataSample must be used only for 2D MT Model with shape (B, H, W)"
            )
