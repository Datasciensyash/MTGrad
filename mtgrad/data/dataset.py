import random
import typing as tp

import numpy as np
import torch
from pymt.microgrid import ResistivityMicrogrid
from pymt.modeling import RandomLayerModel
from torch.utils.data import Dataset

from mtgrad.data_types import MTDataSample


class RandomLayerDataset(Dataset):
    def __init__(
        self,
        size: tp.Tuple[int, int],
        num_layers_range: tp.Tuple[int, int],
        powers_range: tp.Tuple[float, float],
        resistivity_range: tp.Tuple[float, float],
        alpha_range: tp.Tuple[float, float],
        period_count_range: tp.Tuple[int, int],
        batch_size: int = 64,
        epoch_size: int = 1,
    ):
        self._layer_model = RandomLayerModel(
            alpha_range=tuple(alpha_range),
            resistivity_range=tuple(resistivity_range),
            powers_range=tuple(powers_range),
            num_layers_range=tuple(num_layers_range),
        )
        self._period_count_range = period_count_range

        self._size = size
        self._batch_size = batch_size
        self._epoch_size = epoch_size

    def sample_periods(self) -> np.ndarray:
        start = random.uniform(0.001, 0.002)
        exp_step = random.uniform(1.5, 2.5)
        count = random.randint(*self._period_count_range)
        return np.array([start * exp_step ** i for i in range(count)])

    def __getitem__(self, index: int) -> ResistivityMicrogrid:
        random_mgrid = self._layer_model.to_microgrid(self._size)
        return random_mgrid

    def __len__(self):
        return self._batch_size * self._epoch_size

    def collate_function(
        self,
        data: tp.List[ResistivityMicrogrid],
    ) -> MTDataSample:
        periods = self.sample_periods()

        app_res = torch.empty(
            (self._batch_size, len(periods), self._size[0]), dtype=torch.float32
        )
        imp_phs = torch.empty(
            (self._batch_size, len(periods), self._size[0]), dtype=torch.float32
        )
        resistivity = torch.empty(
            (self._batch_size, data[0].resistivity.shape[1], self._size[0]),
            dtype=torch.float32,
        )
        layer_powers = torch.ones_like(resistivity)

        for i, mgrid in enumerate(data):
            mgrid.compute_direct_task(periods)
            resistivity[i, :] = torch.Tensor(mgrid.resistivity).T
            app_res[i, :] = torch.Tensor(mgrid.apparent_resistivity).T
            imp_phs[i, :] = torch.Tensor(mgrid.impedance_phase).T
            layer_powers[i, :] = torch.Tensor(mgrid.layer_power).T

        periods = torch.ones((self._batch_size, len(periods))) * torch.Tensor(
            periods
        )

        return MTDataSample(
            resistivity=resistivity,
            apparent_resistivity=app_res,
            impedance_phase=imp_phs,
            periods=periods,
            layer_powers=layer_powers,
        )
