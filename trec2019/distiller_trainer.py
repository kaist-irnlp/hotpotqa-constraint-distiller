"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
import os
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter
from pprint import pprint

import torch
from torch.backends import cudnn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler, PassThroughProfiler
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from test_tube import HyperOptArgumentParser
from test_tube import Experiment
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

# project specific
from trec2019.model.sparsenet import SparseNet
from trec2019.utils.dataset import EmbeddingLabelDataset


root_dir = Path(__file__).parent.absolute()


class UploadFinalCheckpointCallback(pl.Callback):
    def on_init_start(self, trainer):
        pass

    def on_init_end(self, trainer):
        pass

    def on_train_end(self, trainer, pl_module):
        # save the last checkpoint
        trainer.save_checkpoint(
            Path(trainer.ckpt_path) / f"last_epoch={trainer.current_epoch}.ckpt"
        )


def gather_tags(hparams):
    tags = []
    for grp, val in hparams.items():
        if isinstance(val, DictConfig):
            _tags = hparams[grp].get("tags", [])
            tags += list(_tags)
    return tags


def flatten_params(hparams):
    params = {}
    for grp, val in hparams.items():
        if isinstance(val, DictConfig):
            for k, v in val.items():
                params[f"{grp}.{k}"] = v
    return params


def get_data_cls(name):
    if name in ("news20"):
        return EmbeddingLabelDataset
    else:
        raise ValueError("Unkonwn dataset")


def get_sparse_cls(name):
    if name == "sparsenet":
        return SparseNet
    else:
        raise ValueError("Unknown sparse model")


@hydra.main(config_path="conf/config.yaml")
def main_hydra(cfg: DictConfig) -> None:
    # print(cfg.pretty())
    # print(os.getcwd())
    # hparams = Namespace(**cfg)
    hparams = cfg
    main(hparams)


def main(hparams):
    # init model
    model = SparseNet(hparams)

    # early stop
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=20, verbose=True, mode="min"
    )

    # logger
    # log_dir = str(root_dir / "lightning_logs")/
    # tt_logger = loggers.TestTubeLogger("tb_logs", name=hparams.experiment_name)
    # tb_logger = loggers.TensorBoardLogger("tb_logs")

    # init logger
    source_files_path = str(Path(hydra.utils.get_original_cwd()) / "*.py")
    tags = gather_tags(hparams)
    log_params = flatten_params(hparams)
    neptune_logger = NeptuneLogger(
        project_name=hparams.project,
        experiment_name=hparams.experiment.name,  # Optional,
        params=log_params,  # Optional,
        tags=tags,  # Optional,
        close_after_fit=False,
        upload_source_files=[source_files_path],
    )
    # logger_list = [neptune_logger, tb_logger]

    # checkpoint
    # checkpoint_dir = "sparsenet/checkpoints"
    # checkpoint_callback = ModelCheckpoint(filepath=checkpoint_dir)

    # custom callbacks
    callbacks = [UploadFinalCheckpointCallback()]

    # use profiler
    profiler = AdvancedProfiler() if hparams.train.profile else None

    # train
    # trainer = Trainer.from_argparse_args(hparams)
    trainer = Trainer(
        # default_root_dir=root_dir,
        max_nb_epochs=hparams.train.max_nb_epochs,
        gpus=str(hparams.train.gpus),
        distributed_backend=hparams.train.distributed_backend,
        fast_dev_run=hparams.train.fast_dev_run,
        amp_level=hparams.train.amp_level,
        precision=hparams.train.precision,
        benchmark=True,
        profiler=profiler,
        logger=neptune_logger,
        # early_stop_callback=early_stop_callback,
        # checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
    )
    trainer.fit(model)

    # upload the best model and configs
    checkpoints_dir = os.getcwd()
    neptune_logger.experiment.log_artifact(str(checkpoints_dir))
    neptune_logger.experiment.stop()


if __name__ == "__main__":
    main_hydra()
