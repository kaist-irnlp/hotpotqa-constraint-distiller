from pathlib import Path
import torch
import json
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import zarr
from omegaconf import OmegaConf

from trec2019.distiller import Distiller
from trec2019.utils.dataset import *

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--arr_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="generated")
parser.add_argument("--batch_size", type=int, default=8192)
args = parser.parse_args()


def load_datasets(data_dir, arr_path):
    data_dir = Path(data_dir)
    fnames = ["train", "test", "val"]
    datasets = {}
    for f in fnames:
        fpath = (data_dir / f).with_suffix(".zarr")
        datasets[f] = EmbeddingDataset(fpath, arr_path)
    return datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_path):
    # load model
    model = Distiller.load_from_checkpoint(
        model_path, map_location=device, data_path=data_path
    )
    # make eval mode
    model.eval()
    return model


# def open_zarr(model_name, fname, out_dir):
#     _out_dir = get_out_dir(model_name, out_dir)
#     out_path = (_out_dir / fname).with_suffix(".zarr")
#     store = zarr.DirectoryStore(str(out_path))
#     z = zarr.group(store=store)
#     return z


if __name__ == "__main__":
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    arr_path = args.arr_path
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # load model
    model = get_model(model_path)
    model_name = model_path.stem

    # copy hparams
    hparams = OmegaConf.to_container(model.hparams)
    with open(out_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=4, ensure_ascii=False)

    # generate
    datasets = load_datasets(data_path, arr_path)
    for dset_name, dset in datasets.items():
        print("Processing", dset_name)
        # init storage
        init_shape = (args.batch_size, model.hparams.model.n[-1])
        chunks = (1024, None)
        z = zarr.open(str(out_dir / dset_name), mode="w")
        z_index = z.zeros("index", shape=init_shape[0], chunks=chunks[0])
        # z_target = z.zeros("target", shape=init_shape[0], chunks=chunks[0])
        z_out = z.zeros("out", shape=init_shape, chunks=chunks)

        # save
        loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False)
        for i, row in enumerate(tqdm(loader)):
            index, data = (
                row["index"].numpy(),
                row["data"],
                # row["target"].numpy(),
            )
            # infer
            with torch.no_grad():
                out = model.sparse(data).numpy()
            # write when batch_idx = 0 else append
            if i == 0:
                n_rows = len(index)
                # resize to fit the batch size
                z_index.resize(n_rows)
                # z_target.resize(n_rows)
                z_out.resize(n_rows, None)
                # write the first row
                z_index[:] = index
                # z_target[:] = target
                z_out[:] = out
            else:
                # append afterward
                z_index.append(index, axis=0)
                # z_target.append(target, axis=0)
                z_out.append(out, axis=0)
