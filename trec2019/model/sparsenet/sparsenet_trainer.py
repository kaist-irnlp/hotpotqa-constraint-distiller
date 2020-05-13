"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter

import torch
from torch.backends import cudnn
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from test_tube import HyperOptArgumentParser
from test_tube import Experiment
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler, PassThroughProfiler
import pytorch_lightning as pl
from trec2019.utils.dense import *
from trec2019.model.sparsenet import SparseNet
import hydra
from omegaconf import DictConfig
import os


# cudnn.benchmark = True
root_dir = Path(__file__).parent.absolute()


@hydra.main(config_path="conf/config.yaml")
def main_hydra(cfg: DictConfig) -> None:
    print(cfg.pretty())
    print(os.getcwd())


class DummyCallback(pl.Callback):
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("trainer is init now")

    def on_train_end(self, trainer, pl_module):
        print("do something when training ends")


def main(hparams):
    # init model
    model = SparseNet(hparams)

    # early stop
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=10, verbose=True, mode="min"
    )

    # logger
    # log_dir = str(root_dir / "lightning_logs")/
    tt_logger = loggers.TestTubeLogger("tb_logs", name=hparams.experiment_name)
    tb_logger = loggers.TensorBoardLogger("tb_logs", name=hparams.experiment_name)
    neptune_logger = NeptuneLogger(
        project_name="kjang0517/sparsenet",
        experiment_name=hparams.experiment_name,  # Optional,
        params=vars(hparams),  # Optional,
        tags=hparams.tags,  # Optional,
        close_after_fit=False,
    )
    logger_list = [neptune_logger]

    # custom callbacks
    callbacks = [DummyCallback()]

    # profile
    if hparams.profile:
        profiler = AdvancedProfiler()
    else:
        profiler = None

    # train
    # trainer = Trainer.from_argparse_args(hparams)
    trainer = Trainer(
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        fast_dev_run=hparams.fast_dev_run,
        amp_level=hparams.amp_level,
        precision=hparams.precision,
        benchmark=True,
        profiler=profiler,
        logger=logger_list,
        # early_stop_callback=early_stop_callback,
        callbacks=callbacks,
    )
    trainer.fit(model)

    return neptune_logger


# if __name__ == "__main__":
#     main_hydra()
#     sys.exit(-1)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", "-e", type=str, default="default")
    parser.add_argument("--tags", "-t", type=str, nargs="+")
    parser.add_argument("--distributed_backend", "-d", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--gpus", default=None, type=str)
    parser.add_argument(
        "--check_grad_nans", dest="check_grad_nans", action="store_true"
    )
    parser.add_argument("--amp_level", default=None, type=str)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument(
        "--fast_dev_run",
        dest="fast_dev_run",
        default=False,
        action="store_true",
        help="runs validation after 1 training step",
    )

    # model params
    parser.add_argument("--data_dir", type=str, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", dest="max_nb_epochs", default=1000, type=int)
    parser.add_argument("--learning_rate", "-lr", default=0.0002, type=float)

    # add default & model params
    parser = SparseNet.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    # run fixed params
    neptune_logger = main(hparams)
