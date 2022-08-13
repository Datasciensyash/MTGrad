from dataclasses import dataclass

import torch
from pymt.direct_task import direct_task_2d
import numpy as np
import typing as tp


@dataclass
class MTField:
    apparent_resistivity: torch.Tensor
    impedance_phase: torch.Tensor


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


@dataclass
class MTModel:
    resistivity: torch.Tensor
    pixel_size: float
    default_periods: np.ndarray

    def get_field(self, periods: tp.Optional[np.ndarray] = None):
        periods = periods if periods is not None else self.default_periods
        resistivity = self.resistivity.cpu().numpy()
        powers = np.empty(resistivity.shape)
        powers.fill(self.pixel_size)
        shape = (resistivity.shape[0], resistivity.shape[2], periods.shape[0])
        apparent_resistivity = np.empty(shape)
        impedance_phase = np.empty(shape)
        for i in range(self.resistivity.shape[0]):
            app_res, imp_ph = direct_task_2d(
                periods, resistivity[i, 0, :, :], powers[i, 0, :, :]
            )
            apparent_resistivity[i, :, :] = app_res
            impedance_phase[i, :, :] = imp_ph

        return MTField(
            apparent_resistivity=torch.from_numpy(apparent_resistivity)
            .unsqueeze(1)
            .to(self.resistivity.device),
            impedance_phase=torch.from_numpy(impedance_phase)
            .unsqueeze(1)
            .to(self.resistivity.device),
        )