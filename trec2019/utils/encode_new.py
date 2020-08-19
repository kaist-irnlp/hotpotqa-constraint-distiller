import logging
from argparse import ArgumentParser
from pathlib import Path

import zarr
from torch.utils.data import DataLoader, Dataset

from trec2019.utils.dense import DenseSIF

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__file__)

parser = ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model", type=str, choices=("sif", "bert"), required=True)
parser.add_argument("--batch_size", type=int, default=512)
known_args, extra_args = parser.parse_known_args()


class TextDataset(Dataset):
    def __init__(self, data_path):
        self.data = zarr.open(str(data_path), "r").text[:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    # identify model
    MODEL = {"sif": DenseSIF}[known_args.model]

    # add model-specific args
    # and initialize
    parser = MODEL.add_argparse_args(parser)
    args = parser.parse_args()
    model = MODEL(args)

    # load data
    data_path = Path(args.data_path)
    dset = TextDataset(data_path)
    loader = DataLoader(
        dset, batch_size=args.batch_size, num_workers=0, pin_memory=True
    )

    # open data to save
    emb_path = f"dense/{args.model}"
    z = zarr.open(str(data_path), "r+")
    z_embs = z.zeros(emb_path, shape=(len(z.text), model.dim), dtype="f4", overwrite=True)

    # infer & save
    for i, batch in enumerate(loader):
        embs = model.encode_batch(batch)
        # save
        start = i * args.batch_size
        end = start + embs.shape[0]
        z_embs[start:end] = embs[:]
