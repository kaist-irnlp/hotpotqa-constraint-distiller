from pathlib import Path
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import yaml

from trec2019.distiller import Distiller
from trec2019.utils.dataset import EmbeddingDataset

parser = ArgumentParser()
parser.add_argument("ckpt_dir", type=str)
parser.add_argument("dataset_path", type=str)
parser.add_argument("--use_last", action="store_true")
parser.add_argument("--emb_path", type=str, default="dense/bert")
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--num_workers", type=int, default=0)
args = parser.parse_args()


def get_model_path(ckpt_dir, use_last):
    ckpt_files = ckpt_dir.glob("*.ckpt")
    if use_last:
        f = [f for f in ckpt_files if f.name.startswith("last_")][0]
    else:
        f = [f for f in ckpt_files if not f.name.startswith("last_")][0]
    return str(f)


def modify_dataset_path(hparams_path, dataset_path):
    with open(hparams_path, encoding="utf-8") as fp:
        y = yaml.load(fp, Loader=yaml.SafeLoader)
    # modify and save
    y["dataset"]["path"] = dataset_path
    with open(hparams_path, "w", encoding="utf-8") as fp:
        yaml.dump(y, fp)


if __name__ == "__main__":
    # get paths
    ckpt_dir = Path(args.ckpt_dir)
    model_path = get_model_path(ckpt_dir, args.use_last)
    hparams_path = str(ckpt_dir / "hparams.yaml")
    modify_dataset_path(hparams_path, args.dataset_path)
    # hparam_overrides = {"dataset": {"path": args.dataset_path}}

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Distiller.load_from_checkpoint(
        model_path,
        hparams_file=hparams_path,
        map_location=device,
        # hparam_overrides=hparam_overrides,
    )
    model.eval()
    model.freeze()

    # data
    dataset = EmbeddingDataset(
        Path(args.dataset_path) / "queries.all.zarr", args.emb_path
    )
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
        break

