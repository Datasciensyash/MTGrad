import argparse
import time
from pathlib import Path

import torch
import torch.optim as optimizers
import torch.optim.lr_scheduler as lr_schedulers
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from mtgrad.callbacks import VisualizationCallback
from mtgrad.data import RandomLayerDataset
from mtgrad.lightning import TrainingEngine
from mtgrad.model import MTFormer


def parse_args() -> argparse.Namespace:
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-c", "--config", help="path to yaml config", type=str, required=True
    )
    args = arguments_parser.parse_args()
    return args


def prepare_experiment(exp_path: Path, exp_name: str):
    exp_path.mkdir(exist_ok=True)
    experiment_path = (
        exp_path
        / exp_name
        / str(time.ctime()).replace(" ", "_").replace(":", "_")
    )
    logs_dir = experiment_path / "logs"
    checkpoint_dir = experiment_path / "_checkpoints"
    experiment_path.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    return experiment_path, logs_dir, checkpoint_dir


def train(config: dict):

    experiment_path = Path(config["experiments_path"])
    experiment_path, logs_dir, checkpoint_dir = prepare_experiment(
        experiment_path, config["experiment_name"]
    )

    seed_everything(config.get("seed", 1024))

    # Initialize model
    model = MTFormer(**config["model_params"])

    # Initialize criterion
    criterion = torch.nn.MSELoss()

    # Initialize optimizer
    optimizer = getattr(optimizers, config["optimizer_name"])(
        model.parameters(), **config["optimizer_params"]
    )

    # Initialize scheduler
    scheduler = getattr(lr_schedulers, config["scheduler_name"])(
        optimizer, **config["scheduler_params"]
    )

    # Initialize train / val / test datasets
    train_dataset = RandomLayerDataset(**config["train_dataset_params"])
    valid_dataset = RandomLayerDataset(**config["valid_dataset_params"])

    # Initialize train & validation dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["dataloader_params"]["train_batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=config["dataloader_params"]["num_workers"],
        collate_fn=train_dataset.collate_function,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["dataloader_params"]["val_batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config["dataloader_params"]["num_workers"],
        collate_fn=train_dataset.collate_function,
    )

    # Initialize Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, **config["model_checkpoint_params"]
    )
    logger = TensorBoardLogger(
        name=config["experiment_name"], save_dir=str(logs_dir)
    )

    # Initialize lightning module
    module = TrainingEngine(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )

    # Initialize trainer
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, VisualizationCallback()],
        **config["trainer_params"]  # , visualize],
    )

    trainer.fit(module, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    train(config)
