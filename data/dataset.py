import random
import typing as tp

import numpy as np
import torch
from pymt.microgrid import ResistivityMicrogrid
from pymt.modeling import RandomLayerModel
from torch.utils.data import Dataset

from data_types import MTDataSample


class RandomLayerDataset(Dataset):
    def __init__(
        self,
        layer_power_max: tp.List[float],
        layer_power_min: tp.List[float],
        layer_resistivity_max: tp.List[float],
        layer_resistivity_min: tp.List[float],
        period_range: tp.Tuple[float, float],
        period_count_range: tp.Tuple[int, int],
        size: int,
        pixel_size: int,
        layer_exist_probability: tp.Optional[tp.List[float]] = None,
        batch_size: int = 64,
        epoch_size: int = 1,
    ):
        self._layer_model = RandomLayerModel(
            layer_power_max,
            layer_power_min,
            layer_resistivity_max,
            layer_resistivity_min,
            layer_exist_probability,
        )
        self._size = size
        self._pixel_size = pixel_size
        self._period_range = period_range
        self._period_count_range = period_count_range

        self._batch_size = batch_size
        self._epoch_size = epoch_size

    def sample_periods(self) -> np.ndarray:
        return np.clip(
            np.random.exponential(
                0.6, size=(random.randint(*self._period_count_range),)
            ),
            *self._period_range
        )

    def __getitem__(self, index: int) -> ResistivityMicrogrid:
        random_mgrid = self._layer_model.to_microgrid(self._size, self._pixel_size)
        return random_mgrid

    def __len__(self):
        return self._batch_size * self._epoch_size

    def collate_function(
        self,
        data: tp.List[ResistivityMicrogrid],
    ) -> MTDataSample:
        periods = self.sample_periods()

        print(data[0].resistivity.shape)

        app_res = torch.empty(
            (self._batch_size, len(periods), self._size), dtype=torch.float32
        )
        imp_phs = torch.empty(
            (self._batch_size, len(periods), self._size), dtype=torch.float32
        )
        resistivity = torch.empty(
            (self._batch_size, data[0].resistivity.shape[1], self._size),
            dtype=torch.float32,
        )
        layer_powers = torch.ones_like(resistivity) * self._pixel_size

        for i, mgrid in enumerate(data):
            mgrid.compute_direct_task(periods)
            resistivity[i, :] = torch.Tensor(mgrid.resistivity).T
            app_res[i, :] = torch.Tensor(mgrid.apparent_resistivity).T
            imp_phs[i, :] = torch.Tensor(mgrid.impedance_phase).T

        periods = torch.ones((self._batch_size, len(periods))) * torch.Tensor(periods)

        return MTDataSample(
            resistivity=resistivity,
            apparent_resistivity=app_res,
            impedance_phase=imp_phs,
            periods=periods,
            layer_powers=layer_powers,
        )
