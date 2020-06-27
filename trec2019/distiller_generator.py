from pathlib import Path
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from trec2019.distiller import Distiller
from trec2019.utils.dataset import EmbeddingDataset

parser = ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("hparams_path", type=str)
parser.add_argument("data_path", type=str)
parser.add_argument("--emb_path", type=str, default="dense/bert")
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    model = Distiller.load_from_checkpoint(
        args.model_path, hparams_file=args.hparams_path, map_location=device,
    )
    model.eval()
    model.freeze()

    # data
    dataset = EmbeddingDataset(args.data_path, args.emb_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # encode
    for data in loader:
        out = model.forward_sparse(data["orig_data"])
        print(out.shape)

    # encode
    # enc = model(x)

