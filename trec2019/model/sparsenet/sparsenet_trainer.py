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
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser
from test_tube import Experiment
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler, PassThroughProfiler
from trec2019.utils.dense import *
from trec2019.model.sparsenet import SparseNet

# cudnn.benchmark = True
root_dir = str(Path(__file__).parent.absolute())


# def get_dense(dense_name, embedding_path):
#     dense_cls = {"bow": BowEmbedding, "disc": DiscEmbedding, "bert": BertEmbedding}[
#         dense_name
#     ]
#     if issubclass(dense_cls, BasePretrainedEmbedding):
#         dense = dense_cls(embedding_path)
#     else:
#         dense = dense_cls()
#     return dense


def main(hparams):
    # init model
    model = SparseNet(hparams)

    # early stop
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )
    # logger
    tt_logger = loggers.TestTubeLogger(root_dir)

    # profile
    if hparams.profile:
        profiler = AdvancedProfiler()
    else:
        profiler = None

    # train
    # trainer = Trainer.from_argparse_args(hparams)
    trainer = Trainer(
        logger=tt_logger,
        default_save_path=root_dir,
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        nb_gpu_nodes=hparams.nodes,
        fast_dev_run=hparams.fast_dev_run,
        amp_level=hparams.amp_level,
        precision=hparams.precision,
        early_stop_callback=early_stop_callback,
        benchmark=True,
        profiler=profiler,
    )
    trainer.fit(model)


if __name__ == "__main__":
    # parser = HyperOptArgumentParser(strategy="grid_search", add_help=False)
    parser = ArgumentParser(add_help=False)

    # parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--nodes", type=int, default=1)
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

    # searchable params
    # parser.opt_list("--n", type=int, tunable=True, options=[10000])
    # parser.opt_list("--k", type=int, tunable=True, options=[500, 1000])
    # parser.opt_list("--batch_size", type=int, tunable=True, options=[64])

    # model params
    parser.add_argument("--data_path", type=str, default=None, required=True)
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument("--k", type=int, nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", dest="max_nb_epochs", default=500, type=int)
    parser.add_argument("--learning_rate", "-lr", default=0.0002, type=float)
    parser.add_argument("--dense", type=str, choices=["bow", "bert"], default="bow")

    # add default & model params
    # add_default_args(parser, root_dir)
    parser = SparseNet.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    # run fixed params
    main(hparams)

    # run trials of random search over the hyperparams
    # N_TRIALS = 3
    # for hparam_trial in hparams.trials(N_TRIALS):
    #     main(hparam_trial)
