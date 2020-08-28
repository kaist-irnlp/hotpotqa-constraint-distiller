from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path

import pytorch_lightning as pl
import torch
from numcodecs import Zstd
from tqdm import tqdm

from trec2019.distiller import Distiller
from trec2019.utils.dataset import TripleEmbeddingDataModule


parser = ArgumentParser()
parser.add_argument("--exp_dir", type=str, required=True)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"


def load_model(exp_dir):
    exp_dir = Path(exp_dir)
    model_path = list(exp_dir.glob("**/*.ckpt"))[0]
    hparams_path = exp_dir / "hparams.yaml"

    # load model
    model = Distiller.load_from_checkpoint(
        model_path, hparams_file=hparams_path, map_location=device,
    )
    model.eval()
    model.freeze()
    return model


if __name__ == "__main__":
    model = load_model(args.exp_dir)
