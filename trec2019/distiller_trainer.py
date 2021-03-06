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
import tarfile

import torch
from torch.backends import cudnn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateLogger
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
from trec2019.model import WTAModel
from trec2019.utils.dataset import EmbeddingLabelDataset, TripleEmbeddingDataModule
from trec2019.task import ClassificationTask, RankingTask


root_dir = Path(__file__).parent.absolute()

# seed_everything(2020)


class PostTrainCallback(pl.Callback):
    # def on_sanity_check_end(self, trainer, pl_module):
    #     ckpt_path = Path(trainer.ckpt_path)
    #     # save hparams
    #     hparams_str = pl_module.hparams.pretty()
    #     hparams_path = ckpt_path / "hparams.yaml"
    #     with hparams_path.open("w", encoding="utf-8") as f:
    #         f.write(hparams_str)
    #     # upload hparams
    #     pl_module.logger.log_artifact(str(hparams_path))

    def on_train_end(self, trainer, pl_module):
        ckpt_path = Path(trainer.ckpt_path)
        exp_name = pl_module.logger.experiment_name
        # save hparams
        hparams_str = pl_module.hparams.pretty()
        if exp_name:
            hparams_path = ckpt_path / exp_name / "hparams.yaml"
        else:
            hparams_path = ckpt_path / "hparams.yaml"
        with hparams_path.open("w", encoding="utf-8") as f:
            f.write(hparams_str)

        # compress
        ckpt_path_compressed = ckpt_path.with_suffix(".tar.gz")
        with tarfile.open(ckpt_path_compressed, "w:gz") as tar:
            tar.add(ckpt_path, arcname=hparams_path.parent)

        # upload artifacts
        try:
            pl_module.logger.log_artifact(str(ckpt_path_compressed))
        except:
            pl_module.logger.log_text("error", "Error uploading checkpoints.")


def generate_tags(hparams):
    tags = []

    # dataset
    # tags.append(hparams.dataset.name)

    # emb model
    tags.append(hparams.dataset.emb_path.split("/")[1])

    # model
    tags.append(hparams.model.name)

    # n, k
    tags.append(",".join([str(n) for n in hparams.model_n.n]))
    tags.append(",".join([str(k) for k in hparams.model_k.k]))

    # disc
    # tags.append(f"disc-{'_'.join([str(h) for h in hparams.disc.hidden])}")

    # noise, dropout
    # tags.append(f"noise:{hparams.noise.ratio}")
    # tags.append(f"dropout:{hparams.model.dropout}")

    # batch_size, lr
    tags.append(f"bsz-{hparams.train.batch_size}")
    tags.append(f"lr-{hparams.train.learning_rate}")

    # explicit tags
    # for grp, val in hparams.items():
    #     if isinstance(val, DictConfig):
    #         _tags = hparams[grp].get("tags", [])
    #         tags += list(_tags)
    # # emb model
    # emb = hparams.dataset.emb_path.split("/")[1]
    # tags.append(emb)
    print(tags)
    return tags


def flatten_params(hparams):
    params = {}
    for grp, val in hparams.items():
        if isinstance(val, DictConfig):
            for k, v in val.items():
                params[f"{grp}.{k}"] = v
    return params


@hydra.main(config_path="conf", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    print(cfg.pretty())
    hparams = cfg
    if hparams.train.gpus is not None:
        hparams.train.gpus = str(hparams.train.gpus)
    main(hparams)


def main(hparams):
    # init data
    dm = TripleEmbeddingDataModule(hparams)
    ## identify input_size
    hparams.model.input_size = dm.dim

    # model
    model = Distiller(hparams)

    # early stop

    # logger
    # log_dir = str(root_dir / "lightning_logs")/
    # tt_logger = loggers.TestTubeLogger("tb_logs", name=hparams.experiment_name)
    # tb_logger = loggers.TensorBoardLogger("tb_logs")

    # init logger
    source_files_path = str(Path(hydra.utils.get_original_cwd()) / "**/*.py")
    tags = generate_tags(hparams)
    log_params = flatten_params(hparams)
    close_after_fit = not hparams.train.upload_checkpoints
    neptune_logger = NeptuneLogger(
        experiment_name="_".join(tags),
        project_name=f"kjang0517/{hparams.dataset.name}",
        params=log_params,  # Optional,
        tags=tags,  # Optional,
        close_after_fit=close_after_fit,
        upload_source_files=[source_files_path],
    )

    # class SaveHparamsCallback(pl.Callback):
    #     def on_sanity_check_start(self, trainer, pl_module):
    #         # save hparams
    #         hparams_str = trainer.model.hparams.pretty()
    #         hparams_path = Path(trainer.ckpt_path) / "hparams.yaml"
    #         with hparams_path.open("w", encoding="utf-8") as f:
    #             f.write(hparams_str)
    #         trainer.logger.experiment.log_artifact(str(hparams_path))

    # Callbacks
    callbacks = [PostTrainCallback(), LearningRateLogger()]

    # Callbacks: Early stop
    early_stop_callback = True
    # if hparams.train.use_early_stop:
    #     patience = hparams.train.early_stop_patience
    #     early_stop_callback = EarlyStopping(
    #         monitor="val_early_stop_on", patience=patience, verbose=True, mode="min"
    #     )
    # else:
    #     early_stop_callback = None

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
        train_percent_check=hparams.train.train_percent_check,
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
    trainer.fit(model, dm)

    # upload the best model and configs
    # checkpoints_dir = os.getcwd()
    # if hparams.train.upload_checkpoints:
    #     checkpoints_dir = Path(trainer.ckpt_path)
    #     neptune_logger.experiment.log_artifact(str(checkpoints_dir))
    #     neptune_logger.experiment.stop()


if __name__ == "__main__":
    main_hydra()
