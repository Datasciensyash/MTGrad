import pytorch_lightning as pl
import torch.nn as nn
import torch.optim
import typing as tp

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from data_types import MTDataSample
from model.mtformer import MTFormer


class TrainingEngine(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        model: MTFormer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,
        criterion: nn.Module,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.save_hyperparameters("config")

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.criterion(y_pred, y_true)

    def training_step(self, batch: MTDataSample, batch_idx):
        predicted_resistivity = self.model(
            base_resistivity=batch.resistivity_noisy,
            base_apparent_resistivity=batch.apparent_resistivity_noisy,
            base_impedance_phase=batch.impedance_phase_noisy,
            target_apparent_resistivity=batch.apparent_resistivity,
            target_impedance_phase=batch.impedance_phase,
            periods=batch.periods,
        )
        loss = self._compute_loss(predicted_resistivity, batch.resistivity)
        self.log("train/loss", loss.cpu().item())
        self.lr_schedulers().step()
        return loss

    def validation_step(self, batch: MTDataSample, batch_idx):
        predicted_resistivity = self.model(
            base_resistivity=batch.resistivity_noisy,
            base_apparent_resistivity=batch.apparent_resistivity_noisy,
            base_impedance_phase=batch.impedance_phase_noisy,
            target_apparent_resistivity=batch.apparent_resistivity,
            target_impedance_phase=batch.impedance_phase,
            periods=batch.periods,
        )
        loss = self._compute_loss(predicted_resistivity, batch.resistivity)
        self.log("valid/loss", loss.cpu().item())
        return loss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
