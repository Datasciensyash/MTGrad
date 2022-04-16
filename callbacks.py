import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from data_types import MTDataSample


class VisualizationCallback(pl.Callback):
    def __init__(
            self,
            figsize: tp.Tuple[float, float] = (8, 5),
            epoch_period: int = 10
    ):
        self._figsize = figsize
        self._epoch_period = epoch_period

    def _visualize_tensor(
            self,
            tensor: torch.Tensor,
            cbar: bool = True
    ) -> np.ndarray:

        if tensor.ndim != 2:
            raise ValueError(
                "visualize_tensor function argument tensor "
                f"must have exactly 2 dims, got tensor.ndim = {tensor.ndim}"
            )

        fig = plt.figure(figsize=self._figsize)
        plt.imshow(tensor.cpu().detach().numpy())
        plt.colorbar() if cbar else None
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: torch.Tensor,
        batch: MTDataSample,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        if batch_idx != 0:
            return None

        if trainer.current_epoch % self._epoch_period != 0:
            return None

        for i in range(batch.resistivity.shape[0]):
            resistivity = self._visualize_tensor(
                batch.resistivity[i], cbar=True
            )
            resistivity_pred = self._visualize_tensor(
                outputs[0][i], cbar=True
            )

            pl_module.logger.experiment.add_image(
                f"resistivity_{i}/target",
                resistivity,
                pl_module.current_epoch,
                dataformats="HWC"
            )
            pl_module.logger.experiment.add_image(
                f"resistivity_{i}/predicted",
                resistivity_pred,
                pl_module.current_epoch,
                dataformats="HWC"
            )


