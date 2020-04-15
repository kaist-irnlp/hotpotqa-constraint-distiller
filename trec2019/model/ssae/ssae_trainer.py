import os
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter

import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from test_tube import HyperOptArgumentParser
from test_tube import Experiment
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler, PassThroughProfiler
from trec2019.utils.dense import *
from trec2019.model.ssae import SSAE

_root_dir = str(Path(__file__).parent.absolute())


def main(hparams):
    pass


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", dest="max_nb_epochs", default=500, type=int)
    parser.add_argument("--learning_rate", "-lr", default=0.0002, type=float)

    # add default & model params
    parser = SSAE.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    # run with fixed params
    main(hparams)
