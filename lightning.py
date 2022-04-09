import pytorch_lightning as pl
import torch.nn as nn
import torch.optim
import typing as tp

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from data_types import MTDataSample
from model.diffusion import GaussianDiffusion
from model.mtformer import MTFormer


class TrainingEngine(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        model: GaussianDiffusion,
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
        loss = self.model(**batch.__dict__)
        self.log("train/loss", loss.cpu().item())
        self.lr_schedulers().step()
        return loss

    def validation_step(self, batch: MTDataSample, batch_idx):
        loss = self.model(**batch.__dict__)
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
