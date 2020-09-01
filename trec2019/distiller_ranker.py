import logging
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path
import pickle

import pytorch_lightning as pl
import torch
import zarr
from numcodecs import Zstd
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.sparse import csr_matrix

from trec2019.distiller import Distiller
from trec2019.utils.dataset import EmbeddingDataset

logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


parser = ArgumentParser()
parser.add_argument("--exp_dir", type=str, required=True)
parser.add_argument("--query_dir", type=str, required=True)
parser.add_argument("--doc_dir", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
print("Running on", device)


def load_model(exp_dir):
    exp_dir = Path(exp_dir)
    model_path = str(list(exp_dir.glob("**/*.ckpt"))[0])
    hparams_path = str(exp_dir / "hparams.yaml")

    # load model
    model = Distiller.load_from_checkpoint(
        model_path, hparams_file=hparams_path, map_location=device,
    )
    model.eval()
    return model


def model_desc(hparams):
    tags = []

    # emb model
    tags.append(hparams.dataset.emb_path.split("/")[1])

    # model
    tags.append(hparams.model.name)

    # n, k
    tags.append("n-" + ",".join([str(n) for n in hparams.model_n.n]))
    tags.append("k-" + ",".join([str(k) for k in hparams.model_k.k]))

    # disc
    # tags.append(f"disc-{'_'.join([str(h) for h in hparams.disc.hidden])}")

    # batch_size, lr
    tags.append(f"bsz-{hparams.train.batch_size}")
    tags.append(f"lr-{hparams.train.learning_rate}")

    return "_".join(tags)


def encode(model, data_dir, batch_size=8192, num_workers=4):
    emb_path = model.hparams.dataset.emb_path
    dataset = EmbeddingDataset(data_dir, emb_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    )

    encoded = {}
    for i, batch in enumerate(tqdm(loader)):
        ids, encs = batch["id"], model(batch["emb"]).detach().cpu().numpy()
        encoded.update({k: v for (k, v) in zip(ids, encs)})
    return encoded


output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    logging.info("Loading checkpoint...")
    model = load_model(args.exp_dir)
    model_desc = model_desc(model.hparams)

    # encode query
    logging.info("Encoding query...")
    queries = encode(
        model, args.query_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    with open(output_dir / f"{model_desc}_queries.pkl", "wb") as f:
        pickle.dump(queries, f)

    # # encode doc
    logging.info("Encoding doc...")
    docs = encode(
        model, args.model_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    with open(output_dir / f"{model_desc}_docs.pkl", "wb") as f:
        pickle.dump(docs, f)
