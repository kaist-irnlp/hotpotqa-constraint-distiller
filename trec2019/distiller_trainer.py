"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter
from pprint import pprint
from io import StringIO

import torch
from torch.backends import cudnn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
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
from trec2019.distiller import Distiller
from trec2019.model import SparseNetModel, WTAModel
from trec2019.utils.dataset import EmbeddingLabelDataset
from trec2019.task import ClassificationTask, RankingTask


root_dir = Path(__file__).parent.absolute()

# seed_everything(2020)


class UploadFinalCheckpointCallback(pl.Callback):
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
        return SparseNetModel
    elif name == "wta":
        return WTAModel
    else:
        raise ValueError("Unknown sparse model")


def get_task_cls(tp):
    if tp == "classify":
        return ClassificationTask
    elif tp == "ranking":
        return RankingTask
    else:
        raise ValueError("Unknown task")


@hydra.main(config_path="conf", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    print(cfg.pretty())
    hparams = cfg
    if hparams.train.gpus is not None:
        hparams.train.gpus = str(hparams.train.gpus)
    main(hparams)


def main(hparams):
    # # init model
    # ## dataset
    # data_cls = get_data_cls(hparams.dataset.name)
    # data_path = hparams.dataset.path
    # arr_path = hparams.dataset.arr_path
    # ## sparse model
    # sparse_cls = get_sparse_cls(hparams.model.name)
    # ## task model
    # task_cls = get_task_cls(hparams.task.type)
    ## init Distiller
    model = Distiller(hparams)

    # early stop

    # logger
    # log_dir = str(root_dir / "lightning_logs")/
    # tt_logger = loggers.TestTubeLogger("tb_logs", name=hparams.experiment_name)
    # tb_logger = loggers.TensorBoardLogger("tb_logs")

    # init logger
    source_files_path = str(Path(hydra.utils.get_original_cwd()) / "**/*.py")
    tags = gather_tags(hparams)
    log_params = flatten_params(hparams)
    close_after_fit = not hparams.train.upload_checkpoints
    neptune_logger = NeptuneLogger(
        project_name=hparams.project,
        experiment_name=hparams.experiment.name,  # Optional,
        params=log_params,  # Optional,
        tags=tags,  # Optional,
        close_after_fit=close_after_fit,
        upload_source_files=[source_files_path],
    )

    # log hparams
    hparams_io = StringIO(hparams.pretty())
    neptune_logger.log_artifact(hparams_io)

    # class SaveHparamsCallback(pl.Callback):
    #     def on_sanity_check_start(self, trainer, pl_module):
    #         # save hparams
    #         hparams_str = trainer.model.hparams.pretty()
    #         hparams_path = Path(trainer.ckpt_path) / "hparams.yaml"
    #         with hparams_path.open("w", encoding="utf-8") as f:
    #             f.write(hparams_str)
    #         trainer.logger.experiment.log_artifact(str(hparams_path))

    # Callbacks
    callbacks = []

    # Callbacks: Early stop
    if hparams.train.use_early_stop:
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=50, verbose=True, mode="min"
        )
    else:
        callbacks.append(UploadFinalCheckpointCallback())
        early_stop_callback = None

    # use profiler
    profiler = AdvancedProfiler() if hparams.train.profile else None

    # train
    # trainer = Trainer.from_argparse_args(hparams)
    trainer = Trainer(
        # default_root_dir=root_dir,
        max_epochs=hparams.train.max_epochs,
        gpus=hparams.train.gpus,
        distributed_backend=hparams.train.distributed_backend,
        fast_dev_run=hparams.train.fast_dev_run,
        amp_level=hparams.train.amp_level,
        precision=hparams.train.precision,
        val_percent_check=hparams.train.val_percent_check,
        # **hparams.train,
        benchmark=True,
        profiler=profiler,
        logger=neptune_logger,
        early_stop_callback=early_stop_callback,
        callbacks=callbacks,
        # deterministic=True,
    )

    # train
    trainer.fit(model)

    # upload the best model and configs
    # checkpoints_dir = os.getcwd()
    if hparams.train.upload_checkpoints:
        checkpoints_dir = Path(trainer.ckpt_path)
        neptune_logger.experiment.log_artifact(str(checkpoints_dir))
        neptune_logger.experiment.stop()


if __name__ == "__main__":
    main_hydra()
