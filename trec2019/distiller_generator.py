from pathlib import Path
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from argparse import Namespace
import yaml
import zarr
import numpy as np
from tqdm import tqdm
from numcodecs import Zstd

from trec2019.distiller import Distiller
from trec2019.utils.dataset import EmbeddingDataset, EmbeddingLabelDataset

parser = ArgumentParser()
parser.add_argument("exp_dir", type=str)
parser.add_argument("dataset_dir", type=str)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("-p", "--pattern", type=str, default="*.zarr")
args = parser.parse_args()


def get_model_path(ckpt_dir, use_last=False):
    ckpt_files = ckpt_dir.glob("*.ckpt")
    try:
        if use_last:
            f = [f for f in ckpt_files if f.name.startswith("last_")][0]
        else:
            f = [f for f in ckpt_files if not f.name.startswith("last_")][0]

        return str(f)
    except:
        return None


def override_hparams(hparams_path, dataset_dir):
    with open(hparams_path, encoding="utf-8") as fp:
        y = yaml.load(fp, Loader=yaml.SafeLoader)
    # modify and save
    y["dataset"]["path"] = dataset_dir
    y["dataset"]["on_memory"] = False
    with open(hparams_path, "w", encoding="utf-8") as fp:
        yaml.dump(y, fp)


def load_hparams(hparams_path):
    with open(hparams_path, encoding="utf-8") as fp:
        hparams = yaml.load(fp, Loader=yaml.SafeLoader)
        return hparams


def get_model_properties(hparams):
    # properties
    properties = []
    ## base_emb
    properties.append(hparams["dataset"]["emb_path"].split("/")[1])
    ## base_emb_dim
    properties.append(hparams["model"]["input_size"])
    ## model_name
    properties.append(hparams["model"]["name"])
    ## model_arch
    properties.append(
        f"{'_'.join([str(v) for v in hparams['model_n']['n']])}_{'_'.join((str(v) for v in hparams['model_k']['k']))}"
    )
    ## losses
    losses = []
    if hparams["loss"]["use_recovery_loss"]:
        losses.append("recover")
    if hparams["loss"]["use_task_loss"]:
        losses.append("task")
    properties.append("-".join(losses))
    ## projection
    if hparams["loss"]["use_task_projection"]:
        properties.append("projection")
    ## bs & lr
    properties.append("batch-" + str(hparams["train"]["batch_size"]))
    properties.append("lr-" + str(hparams["train"]["learning_rate"]))
    # return as a single string
    return "_".join([str(p) for p in properties])


def encode_dset(model, hparams, dset_path, emb_path):
    dataset = EmbeddingLabelDataset(dset_path, emb_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # model props
    model_props = get_model_properties(hparams)
    print(model_props)
    # encode
    z = zarr.open(str(dset_path))
    out_shape = (len(z.id), int(hparams["model_n"]["n"][-1]))
    z_out = z.zeros(
        f"result/{model_props}",
        shape=out_shape,
        chunks=(args.batch_size, None),
        overwrite=True,
        compressor=Zstd(),
    )
    for i, data in enumerate(tqdm(loader)):
        out = model.forward_sparse(data["data"].permute(0, 2, 1)).cpu().numpy()[:, 0, :]
        start = args.batch_size * i
        end = start + out.shape[0]
        z_out[start:end] = out[:]
        # sparsity
        # n_filled = len(out.nonzero()[0])
        # print("AVG sparsity", n_filled / out.size)


def main(ckpt_dir, dataset_dir):
    # get paths
    print(ckpt_dir)
    ckpt_dir = Path(ckpt_dir)
    model_path = get_model_path(ckpt_dir)
    hparams_path = ckpt_dir / "hparams.yaml"
    if (model_path is None) or (not hparams_path.exists()):
        print(f"checkpoint or hparams does not exist: {ckpt_dir}")
        return None
    hparams_path = str(hparams_path)
    hparams = load_hparams(hparams_path)

    model_props = get_model_properties(hparams)
    # check if ckpt exists
    override_hparams(hparams_path, dataset_dir)
    # hparam_overrides = {"dataset": {"path": dataset_dir}}

    # load model
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model = Distiller.load_from_checkpoint(
        model_path,
        hparams_file=hparams_path,
        map_location=device,
        # hparam_overrides=hparam_overrides,
    )
    model.eval()
    model.freeze()

    # load hparams & parse model properties
    emb_path = hparams["dataset"]["emb_path"]

    # encode: all
    for f in Path(dataset_dir).glob(args.pattern):
        print(f)
        encode_dset(model, hparams, f, emb_path)


if __name__ == "__main__":
    # main(args.exp_dir, args.dataset_dir)
    out_dirs = list(Path(args.exp_dir).glob("*"))
    for out_dir in tqdm(out_dirs):
        main(out_dir, args.dataset_dir)
