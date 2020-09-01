from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
import torch
from numcodecs import Zstd
from tqdm import tqdm

from trec2019.distiller import Distiller
from trec2019.utils.dataset import TripleEmbeddingDataModule


parser = ArgumentParser()
parser.add_argument("--exp_dir", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"


def load_model(exp_dir):
    exp_dir = Path(exp_dir)
    model_path = str(list(exp_dir.glob("**/*.ckpt"))[0])
    hparams_path = str(exp_dir / "hparams.yaml")

    # load model
    model = Distiller.load_from_checkpoint(
        model_path, hparams_file=hparams_path, map_location=device,
    )
    model.eval()
    model.freeze()
    return model


def load_data(data_dir, num_workers, hparams):
    hparams.dataset.path = data_dir
    hparams.dataset.on_memory = True
    hparams.dataset.num_workers = num_workers
    return TripleEmbeddingDataModule(hparams)


if __name__ == "__main__":
    model = load_model(args.exp_dir)
    dm = load_data(args.data_dir, args.num_workers, model.hparams)
    # dm.prepare_data()
    dm.setup("test")

    # trainer for test
    trainer = Trainer()
    trainer.test(model, datamodule=dm)
