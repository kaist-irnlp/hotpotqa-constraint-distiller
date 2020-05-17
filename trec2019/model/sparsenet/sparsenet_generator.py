from pathlib import Path
import torch
import json
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import zarr

from trec2019.model.sparsenet import SparseNet
from trec2019.utils.dataset import *
from trec2019.utils.dense import *

OUT_DIR = "./generated"

parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=8192)
args = parser.parse_args()


def load_datasets(data_dir, ext=".zarr"):
    data_dir = Path(data_dir)
    fnames = ["train", "test", "val"]
    datasets = {}
    for f in fnames:
        fpath = (data_dir / f).with_suffix(ext)
        datasets[f] = EmbeddingLabelDataset(fpath)
    return datasets


def get_out_dir(model_name, out_dir=OUT_DIR):
    _out_dir = Path(out_dir) / model_name
    if not _out_dir.exists():
        _out_dir.mkdir(parents=True)
    return _out_dir


def open_zarr(model_name, fname, out_dir=OUT_DIR):
    _out_dir = get_out_dir(model_name, out_dir)
    out_path = (_out_dir / fname).with_suffix(".zarr")
    store = zarr.DirectoryStore(str(out_path))
    z = zarr.group(store=store)
    return z


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_dir):
    model_dir = Path(model_dir)
    tags_path = model_dir / "meta_tags.csv"
    model_paths = list((model_dir / "checkpoints").glob("*.ckpt"))
    assert len(model_paths) == 1
    model_path = model_paths[0]
    # load model
    model = SparseNet.load_from_checkpoint(
        model_path, tags_csv=tags_path, map_location=device
    )
    # make eval mode
    model.eval()
    return model


if __name__ == "__main__":
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    # load model
    model = get_model(model_dir)
    model_name = model_dir.stem

    # get out dir
    out_dir = get_out_dir(model_name)

    # copy hparams
    hparams = vars(model.hparams)
    with open(out_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=4, ensure_ascii=False)

    # generate
    datasets = load_datasets(data_dir)
    for dset_name, dset in datasets.items():
        print("Processing", dset_name)
        # init storage
        init_shape = (args.batch_size, model.hparams.n[-1])
        chunks = (1024, None)
        z = open_zarr(model_name, dset_name)
        z_index = z.zeros("index", shape=init_shape[0], chunks=chunks[0])
        z_target = z.zeros("target", shape=init_shape[0], chunks=chunks[0])
        z_out = z.zeros("out", shape=init_shape, chunks=chunks)

        # save
        loader = DataLoader(dset, batch_size=args.batch_size)
        for i, row in enumerate(tqdm(loader)):
            index, data, target = (
                row["index"].numpy(),
                row["data"],
                row["target"].numpy(),
            )
            # infer
            with torch.no_grad():
                out = model.sparse(data).numpy()
            # write when batch_idx = 0 else append
            if i == 0:
                n_rows = len(index)
                # resize to fit the batch size
                z_index.resize(n_rows)
                z_target.resize(n_rows)
                z_out.resize(n_rows, None)
                # write the first row
                z_index[:] = index
                z_target[:] = target
                z_out[:] = out
            else:
                # append afterward
                z_index.append(index, axis=0)
                z_target.append(target, axis=0)
                z_out.append(out, axis=0)
