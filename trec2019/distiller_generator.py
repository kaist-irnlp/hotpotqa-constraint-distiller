from pathlib import Path
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from argparse import Namespace
import yaml
import zarr
from tqdm import tqdm

from trec2019.distiller import Distiller
from trec2019.utils.dataset import EmbeddingDataset

parser = ArgumentParser()
parser.add_argument("ckpt_dir", type=str)
parser.add_argument("dataset_path", type=str)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--use_last", action="store_true")
parser.add_argument("--emb_path", type=str, default="dense/bert")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=4)
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
        f"{'_'.join([str(v) for v in hparams['model']['n']])}_{'_'.join((str(v) for v in hparams['model']['k']))}"
    )
    ## losses
    losses = []
    if hparams["loss"]["use_recovery_loss"]:
        losses.append("recover")
    if hparams["loss"]["use_task_loss"]:
        losses.append("task")
    properties.append("-".join(losses))
    ## bs & lr
    properties.append(hparams["train"]["batch_size"])
    properties.append(hparams["train"]["learning_rate"])
    # return as a single string
    return "_".join([str(p) for p in properties])


def encode_dset(dset_path):
    dataset = EmbeddingDataset(dset_path, args.emb_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # encode
    z = zarr.open(str(dset_path))
    out_shape = (len(z.id), int(hparams["model"]["n"][-1]))
    z_out = z.zeros(
        f"result/{model_props}", shape=out_shape, chunks=(64, None), overwrite=True,
    )
    for i, data in enumerate(tqdm(loader)):
        out = model.forward_sparse(data["orig_data"]).cpu().numpy()
        start = args.batch_size * i
        end = start + out.shape[0]
        z_out[start:end] = out[:]


if __name__ == "__main__":
    # get paths
    ckpt_dir = Path(args.ckpt_dir)
    model_path = get_model_path(ckpt_dir, args.use_last)
    hparams_path = str(ckpt_dir / "hparams.yaml")
    modify_dataset_path(hparams_path, args.dataset_path)
    # hparam_overrides = {"dataset": {"path": args.dataset_path}}

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
    hparams = load_hparams(hparams_path)
    model_props = get_model_properties(hparams)
    print(model_props)

    # encode: queries
    encode_dset(Path(args.dataset_path) / "queries.eval.zarr")
    encode_dset(Path(args.dataset_path) / "docs.eval.zarr")
