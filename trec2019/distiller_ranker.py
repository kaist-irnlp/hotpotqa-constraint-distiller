import logging
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path
import pickle
from collections import defaultdict

import pytorch_lightning as pl
import torch
import zarr
from numcodecs import Zstd
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.sparse import vstack
from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

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
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=0)

args = parser.parse_args()

device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
print("Running on", device)


output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


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


def encode(model, data_dir, batch_size, num_workers):
    emb_path = model.hparams.dataset.emb_path
    dataset = EmbeddingDataset(data_dir, emb_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    )

    ids = []
    embs = []
    for i, batch in enumerate(tqdm(loader)):
        _ids, _embs = (
            batch["id"],
            csr_matrix(model(batch["emb"]).detach().cpu().numpy()),
        )
        ids.append(_ids)
        embs.append(_embs)
    # pad to mamtch the shape
    # diff_n_rows = embs[0].shape[0] - embs[-1].shape[0]
    # dim = embs[0].shape[1]
    # ## pad embs
    # padding = csr_matrix(np.zeros((diff_n_rows, dim)))
    # embs[-1] = vstack([embs[-1], padding])
    # merge all
    ids, embs = np.concatenate(ids), vstack(embs)
    # remove pad from embs
    # embs = embs[:-diff_n_rows]
    assert ids.shape[0] == embs.shape[0]
    return ids, embs


def load_encoded(model_desc, data_type):
    out_name = f"{model_desc}_{data_type}"
    ids_path = output_dir / f"{out_name}_ids.npy"
    embs_path = output_dir / f"{out_name}_embs.npz"
    if ids_path.exists() and embs_path.exists():
        logging.info(f"Loading {data_type}...")
        ids = np.load(ids_path)
        embs = load_npz(embs_path)
        return ids, embs
    else:
        raise IOError


def save_encoded(model_desc, data_type, ids, embs):
    out_name = f"{model_desc}_{data_type}"
    ids_path = output_dir / f"{out_name}_ids.npy"
    embs_path = output_dir / f"{out_name}_embs.npz"
    np.save(ids_path, ids)
    save_npz(embs_path, embs)


def load_or_encode(model_desc, data_type, dirs):
    logging.info(f"Trying to load {data_type}...")
    try:
        ids, embs = load_encoded(model_desc, data_type)
        logging.info(f"Loaded {data_type}...")
    except IOError:
        logging.info(f"Encoding {data_type}...")
        ids, embs = encode(
            model,
            dirs[data_type],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        save_encoded(model_desc, data_type, ids, embs)
    return ids, embs


if __name__ == "__main__":
    logging.info("Loading checkpoint...")
    model = load_model(args.exp_dir)
    model_desc = model_desc(model.hparams)

    # dict for results
    dirs = {"query": args.query_dir, "doc": args.doc_dir}
    ids = {}
    embs = {}

    # encode or load query & doc
    for data_type in ["query", "doc"]:
        ids[data_type], embs[data_type] = load_or_encode(model_desc, data_type, dirs)
        assert ids[data_type].shape[0] == embs[data_type].shape[0]

    # generate runs
    K = 1000
    ranks = {}
    for (q_id, q_emb) in tqdm(zip(ids["query"], embs["query"])):
        scores = np.squeeze(cosine_distances(q_emb, embs["doc"]))
        ind = np.argsort(scores)
        top_ids = ids["doc"][ind][:K]
        # save ranks
        ranks[q_id] = top_ids
    with open(output_dir / f"runs.{model_desc}.txt", "w", encoding="utf-8") as f:
        for q_id, p_ids in ranks.items():
            for r, p_id in enumerate(p_ids):
                f.write(f"{q_id}\t{p_id}\t{r+1}\n")

    # build inverted index
