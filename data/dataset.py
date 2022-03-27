import random
import numpy as np
import torch
from pymt.microgrid import ResistivityMicrogrid
from torch.utils.data import Dataset
from pymt.modeling import RandomLayerModel
import typing as tp

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
        return np.random.uniform(
            *self._period_range, size=(random.randint(*self._period_count_range),)
        )

    @staticmethod
    def apply_noise(mgrid: ResistivityMicrogrid, signal_weight: float) -> ResistivityMicrogrid:
        noise = np.random.uniform(0, np.max(mgrid.resistivity), mgrid.resistivity.shape)
        mgrid.resistivity = np.clip(mgrid.resistivity * signal_weight + noise, 0, np.inf)
        return mgrid

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
        signal_weight = random.random()

        resistivity = []
        resistivity_noisy = []
        app_res, im_ph = [], []
        app_res_noisy, imp_ph_noisy = [], []
        periods_list = []
        for mgrid in data:
            mgrid.compute_direct_task(periods)
            app_res.append(torch.from_numpy(mgrid.apparent_resistivity).float())
            im_ph.append(torch.from_numpy(mgrid.impedance_phase).float())
            resistivity.append(torch.from_numpy(mgrid.resistivity).float())

            mgrid = self.apply_noise(mgrid, signal_weight)
            mgrid.compute_direct_task(periods)
            app_res_noisy.append(torch.from_numpy(mgrid.apparent_resistivity).float())
            imp_ph_noisy.append(torch.from_numpy(mgrid.impedance_phase).float())
            resistivity_noisy.append(torch.from_numpy(mgrid.resistivity).float())

            periods_list.append(torch.from_numpy(periods))

        return MTDataSample(
            resistivity=torch.stack(resistivity).unsqueeze(1).transpose(-1, -2),
            resistivity_noisy=torch.stack(resistivity_noisy).unsqueeze(1).transpose(-1, -2),
            apparent_resistivity=torch.stack(app_res).unsqueeze(1).transpose(-1, -2),
            apparent_resistivity_noisy=torch.stack(app_res_noisy).unsqueeze(1).transpose(-1, -2),
            impedance_phase=torch.stack(im_ph).unsqueeze(1).transpose(-1, -2),
            impedance_phase_noisy=torch.stack(imp_ph_noisy).unsqueeze(1).transpose(-1, -2),
            periods=torch.stack(periods_list).float(),
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = RandomLayerDataset(
        [1000, 2000, 3000],
        [800, 1800, 2500],
        [1000, 1000, 1000],
        [0, 0, 0],
        64,
        100,
        [0.01 * (2 ** i) for i in range(13)],
    )
    import tqdm

    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_function)
    for i in tqdm.tqdm(dataloader):
        pass

    print(next(iter(dataloader)).apparent_resistivity.shape)
