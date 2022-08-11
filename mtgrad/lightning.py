import pytorch_lightning as pl
import torch.nn as nn
import torch.optim
from data_types import MTDataSample
from torch.nn import functional

from mtgrad.model import MTFormer


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

    def training_step(self, batch: MTDataSample, batch_idx):
        resistivity_predicted = self.model(
            apparent_resistivity=batch.apparent_resistivity,
            impedance_phase=batch.impedance_phase,
            periods=batch.periods,
            layer_powers=batch.layer_powers,
        )
        resistivity = self.model.normalize_resistivity(batch.resistivity)
        loss = functional.mse_loss(resistivity, resistivity_predicted)
        self.log(
            "train/loss",
            loss.cpu().item(),
            batch_size=resistivity_predicted.shape[0],
        )
        self.lr_schedulers().step()
        return loss

    def validation_step(self, batch: MTDataSample, batch_idx):
        resistivity_predicted = self.model(
            apparent_resistivity=batch.apparent_resistivity,
            impedance_phase=batch.impedance_phase,
            periods=batch.periods,
            layer_powers=batch.layer_powers,
        )
        resistivity = self.model.normalize_resistivity(batch.resistivity)
        loss = functional.mse_loss(resistivity, resistivity_predicted)
        self.log(
            "valid/loss",
            loss.cpu().item(),
            batch_size=resistivity_predicted.shape[0],
        )
        return (
            self.model.denormalize_resistivity(resistivity_predicted),
            resistivity,
        )

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
