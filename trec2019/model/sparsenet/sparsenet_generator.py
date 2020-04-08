from pathlib import Path
import torch
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import zarr

from trec2019.model.sparsenet import SparseNet
from trec2019.utils.dataset import *
from trec2019.utils.dense import *


def load_datasets(data_dir, ext=".zarr.zip"):
    data_dir = Path(data_dir)
    fnames = ["train", "test", "val"]
    datasets = {}
    for f in fnames:
        fpath = (data_dir / f).with_suffix(ext)
        datasets[f] = News20EmbeddingDataset(fpath)
    return datasets


def get_zarr(fname, out_dir="output/"):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    out_path = (out_dir / fname).with_suffix(".zarr.zip")
    store = zarr.ZipStore(str(out_path), mode="w")
    z = zarr.group(store=store)
    return z


parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=8192)
args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/Users/kyoungrok/Dropbox/Project/naver/experiment/200324_SparseNet/result/fse/2000/version_0/checkpoints/epoch=28.ckpt"
    tags_csv = "/Users/kyoungrok/Dropbox/Project/naver/experiment/200324_SparseNet/result/fse/2000/version_0/meta_tags.csv"
    data_dir = "/Users/kyoungrok/Dropbox/Project/naver/data/news20/"

    # load model
    model = SparseNet.load_from_checkpoint(
        model_path, tags_csv=tags_csv, map_location=device
    )
    model.freeze()

    # generate
    datasets = load_datasets(data_dir)
    for dset_name, dset in datasets.items():
        print("Processing", dset_name)
        # init storage
        init_shape = (args.batch_size, model.hparams.n[-1])
        chunks = (1024, None)
        z = get_zarr(dset_name)
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
            out = model.forward_sparse(data).numpy()
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
