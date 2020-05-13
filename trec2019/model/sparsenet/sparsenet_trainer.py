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
        pass

    def on_init_end(self, trainer):
        pass

    def on_train_end(self, trainer, pl_module):
        # save the last checkpoint
        trainer.save_checkpoint(
            Path(trainer.ckpt_path) / f"last_epoch={trainer.current_epoch}.ckpt"
        )


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
    neptune_logger = NeptuneLogger(
        project_name="kjang0517/news20",
        experiment_name=hparams.experiment_name,  # Optional,
        params=vars(hparams),  # Optional,
        tags=hparams.tags,  # Optional,
        close_after_fit=False,
        upload_source_files=["*.py"],
    )
    # logger_list = [neptune_logger, tb_logger]

    # checkpoint
    # checkpoint_dir = "sparsenet/checkpoints"
    # checkpoint_callback = ModelCheckpoint(filepath=checkpoint_dir)

    #     >>> class LitModel(LightningModule):
    # ...     def training_step(self, batch, batch_idx):
    # ...         # log metrics
    # ...         self.logger.experiment.log_metric('acc_train', ...)
    # ...         # log images
    # ...         self.logger.experiment.log_image('worse_predictions', ...)
    # ...         # log model checkpoint
    # ...         self.logger.experiment.log_artifact('model_checkpoint.pt', ...)
    # ...         self.logger.experiment.whatever_neptune_supports(...)
    # ...
    # ...     def any_lightning_module_function_or_hook(self):
    # ...         self.logger.experiment.log_metric('acc_train', ...)
    # ...         self.logger.experiment.log_image('worse_predictions', ...)
    # ...         self.logger.experiment.log_artifact('model_checkpoint.pt', ...)
    # ...         self.logger.experiment.whatever_neptune_supports(...)

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
        default_root_dir=root_dir,
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        fast_dev_run=hparams.fast_dev_run,
        amp_level=hparams.amp_level,
        precision=hparams.precision,
        benchmark=True,
        profiler=profiler,
        logger=neptune_logger,
        early_stop_callback=early_stop_callback,
        # checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
    )
    trainer.fit(model)

    # upload the best model then stop
    checkpoints_dir = (
        root_dir / "default" / neptune_logger.experiment.id / "checkpoints"
    )
    neptune_logger.experiment.log_artifact(str(checkpoints_dir))
    neptune_logger.experiment.stop()


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
    main(hparams)
