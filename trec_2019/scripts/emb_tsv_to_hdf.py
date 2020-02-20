import dask.dataframe as dd
import dask.bag as db
from dask.diagnostics import ProgressBar
from argparse import ArgumentParser
from pathlib import Path
import gzip
import numpy as np
import math
import h5py
import logging
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_path", type=str, default=None)
# parser.add_argument("--low_memory", action="store_true")
args = parser.parse_args()

COMP_LEVEL = 9
HDF_PATH = "/data"
fields = ["id", "vector"]


def parse_id_and_vector(tsv_line):
    uid, vector = tsv_line.split("\t")
    uid = int(uid)
    vector = np.array(vector.split(), np.float32)
    return (uid, vector)


def write_id_and_vector():
    pass


def get_n_rows(data_path):
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return sum(1 for _ in f)


def get_n_dims(data_path):
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        line = f.readline()
        return len(parse_id_and_vector(line)[1])


if __name__ == "__main__":
    data_path = Path(args.data_path)
    n_rows = get_n_rows(data_path)
    n_dims = get_n_dims(data_path)

    # save
    fname = f"{data_path.stem.replace('.tsv', '')}.hdf5"
    out_path = args.out_path or (data_path.parent / fname)
    opts = {
        "compression": "lzf"
    }
    with h5py.File(out_path, 'w') as f:
        dset_id = f.create_dataset('id', shape=(
            n_rows,), dtype=np.int32, **opts)
        dset_vector = f.create_dataset('vector', shape=(
            n_rows, n_dims), dtype=np.float32, **opts)

        BUFFER_SIZE = 20000
        row_start = 0
        with gzip.open(data_path, 'rt', encoding='utf-8') as fin:
            ids, vectors = [], []
            for i, line in enumerate(tqdm(fin, total=n_rows)):
                uid, vector = parse_id_and_vector(line)
                ids.append(uid)
                vectors.append(vector)
                if (i % BUFFER_SIZE) == 0:
                    buffer_size = len(ids)
                    row_end = (row_start + buffer_size)
                    dset_id[row_start:row_end] = np.array(ids, dtype=np.int32)
                    dset_vector[row_start:row_end, :] = np.array(vectors, dtype=np.float32)
                    row_start += buffer_size
                    ids, vectors = [], []
                    continue

            # flush buffer
            buffer_size = len(ids)
            row_end = (row_start + buffer_size)
            dset_id[row_start:row_end] = ids
            dset_vector[row_start:row_end, :] = vectors
            row_start += buffer_size
            ids, vectors = [], []

        assert n_rows == dset_id.shape[0] == dset_vector.shape[0]
