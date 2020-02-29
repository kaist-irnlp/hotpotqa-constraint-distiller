"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.backends import cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser
from test_tube import Experiment
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from trec2019.sparse.sparsenet import SparseNet

cudnn.benchmark = True
root_dir = str(Path(__file__).parent.absolute())


def main(hparams):
    # init module
    model = SparseNet(hparams)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )

    trainer = Trainer(
        default_save_path=root_dir,
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        nb_gpu_nodes=hparams.nodes,
        fast_dev_run=hparams.fast_dev_run,
        use_amp=hparams.use_amp,
        amp_level="O2",
        early_stop_callback=early_stop_callback,
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = HyperOptArgumentParser(strategy="grid_search", add_help=False)
    parser.add_argument("--data_dir", type=str, default=None, required=True)
    parser.add_argument("--epochs", dest="max_nb_epochs", default=500, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--distributed_backend", type=str, default=None)

    add_default_args(parser, root_dir)
    parser = SparseNet.add_model_specific_args(parser)

    # param search
    parser.opt_list("--n", type=int, tunable=True, options=[8000, 10000])
    parser.opt_list("--k", type=int, tunable=True, options=[30, 60, 120, 240, 300])
    parser.opt_list("--batch_size", type=int, tunable=True, options=[512])

    # parse params
    hparams = parser.parse_args()

    # run trials of random search over the hyperparams
    N_TRIALS = 50
    for hparam_trial in hparams.trials(N_TRIALS):
        main(hparam_trial)
