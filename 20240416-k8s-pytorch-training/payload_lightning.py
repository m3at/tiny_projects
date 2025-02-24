#!/usr/bin/env python3


import argparse
import logging
import os
import platform

# from time import sleep
from datetime import datetime
from pathlib import Path
from typing import Callable

# from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.loggers import CSVLogger
# from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
# from pytorch_lightning.loggers import TensorBoardLogger
import lightning as L
import torch
import torchvision
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F

# from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchmetrics.classification.accuracy import Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10

# from fsspec import fuse
# import gcsfs


logger = logging.getLogger("base")

NUM_WORKERS = int(os.cpu_count() or 8 / 2)


def p(message: str, log: Callable = logger.info) -> None:
    log(message)


class SimpleModel(L.LightningModule):
    def __init__(self, learning_rate=0.1):
        super().__init__()
        # MNIST
        # self.l1 = torch.nn.Linear(28 * 28, 10)
        # self.loss_fn = F.cross_entropy

        self.save_hyperparameters()

        # CIFAR
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model.maxpool = nn.Identity()  # type:ignore
        self.loss_fn = F.nll_loss

        self.val_acc = Accuracy(task="multiclass")
        self.test_acc = Accuracy(task="multiclass")

        # Log hyper parameters to tensorboard
        # tensorboard = self.logger.experiment  # type:ignore
        # tensorboard.log_hyperparams(self.hparams)

    def forward(self, x):
        # return torch.relu(self.l1(x.view(x.size(0), -1)))
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)

        self.log("train/loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("val/loss", loss.item())

        self.val_acc.update(pred, y)
        # Doesn't appear to play well with tensorboard
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Explicit
        # tensorboard: SummaryWriter = self.logger.experiment  # type:ignore
        # tensorboard.add_scalar("val/loss", float(loss.item()), self.global_step)

    def validation_epoch_end(self, outputs):
        acc = float(self.val_acc.compute())
        self.val_acc.reset()
        self.log("val/accuracy", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("test/loss", loss.item())

        self.test_acc.update(pred, y)
        self.log("test/accuracy", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr: float = self.hparams.learning_rate  # type:ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=int(45_000 * 30 // 256),
                # epochs=self.trainer.max_epochs,  # type:ignore
                # steps_per_epoch=45000 // 256,
            ),
            # "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


def main() -> None:
    ####
    # Check GPU
    ####
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "N/A"
    p(
        "[Pre-flight check] CUDA: {} , GPU count: {}, device: {}; home: {}".format(
            has_cuda,
            torch.cuda.device_count(),
            device_name,
            # /root
            Path.home(),
        )
    )

    gcsfuse_dir = Path("/usr/share/mapped_fuse")

    # p(f"Files root: {list(mapped_dir.glob('*'))}")
    # Only visible dirs: '/usr/share/mapped_fuse/poc_data_embeddings_included', '/usr/share/mapped_fuse/redis_cache_export'

    exp_dir = gcsfuse_dir / "subdir/tmp_fuse"
    p(f"GCP FUSE check, saw files: {list(exp_dir.glob('*'))}")

    # with (exp_dir / "to_read.txt").open("r") as f:
    #     p(f"Reading content: {f.read()}")

    payload = "{}-{}-{}".format(platform.node(), platform.machine(), datetime.now())
    with (exp_dir / "test.txt").open("w+") as f:
        f.write(payload)

    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#configure-console-logging
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    p("[Liftoff]", logger.debug)

    # Init our model
    mnist_model = SimpleModel()

    PATH_DATASETS = "."
    BATCH_SIZE = 256 if has_cuda else 64

    # Init DataLoader from MNIST Dataset
    # D = MNIST
    D = CIFAR10
    train_ds = D(
        PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor()
    )
    val_ds = D(
        PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # This works, not the CSVLogger for some reason
    # "works" as in log something, but overwrite at each logging step, instead of accumulating
    tb_logger = TensorBoardLogger(save_dir=str(exp_dir))

    # Initialize a trainer
    trainer = L.Trainer(
        accelerator="auto",
        # devices=1 if has_cuda else None,
        devices=1,
        max_epochs=30,
        precision=16 if has_cuda else 32,
        # callbacks=[TQDMProgressBar(refresh_rate=50)],
        # Disable
        callbacks=[TQDMProgressBar(refresh_rate=0)],
        # Never seems to work…
        # default_root_dir=str(exp_dir),
        # logger=CSVLogger(save_dir="logs/"),
        logger=tb_logger,
        # logger=CSVLogger(save_dir="dir"),
    )

    # Train the model ⚡
    trainer.fit(mnist_model, train_loader, val_dataloaders=val_loader)

    # Test
    # logs/lightning_logs/version_0/checkpoints/epoch=1-step=1876.ckpt
    test_metrics = trainer.test(ckpt_path="best", dataloaders=val_loader)
    p(f"[MECO] Test metrics: {test_metrics}")

    # jp-item-embeddings/test-job-k8s/trainings

    # p(f"[Payload deployment] Wrote content into")

    p("[Mission success]", logger.debug)


if __name__ == "__main__":
    # Get system arguments
    parser = argparse.ArgumentParser(
        description="replace_me",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="DEBUG",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )
    args = vars(parser.parse_args())
    log_level = getattr(logging, args.pop("log_level").upper())

    # Setup logging
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(
        logging.Formatter(
            "{asctime} {levelname}│ {message}", datefmt="%H:%M:%S", style="{"
        )
    )
    logger.addHandler(ch)

    # Add colors
    _levels = [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]
    for color, lvl in _levels:
        _l = getattr(logging, lvl)
        logging.addLevelName(
            _l, "\x1b[38;5;{}m{:<7}\x1b[0m".format(color, logging.getLevelName(_l))
        )

    main()
