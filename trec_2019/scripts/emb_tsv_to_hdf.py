import dask.dataframe as dd
import dask.bag as db
from dask.diagnostics import ProgressBar
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import math
import h5py
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_path", type=str, default=None)
parser.add_argument("--low_memory", action="store_true")
args = parser.parse_args()

COMP_LEVEL = 9
HDF_PATH = "/data"
fields = ["id", "vector"]


def parse_id_and_vector(tsv_line):
    uid, vector = tsv_line.split("\t")
    uid = int(uid)
    vector = np.array(vector.split(), np.float32)
    return (uid, *vector)


def make_meta(data_path):
    with open(data_path, encoding="utf-8") as f:
        line = f.readline()
        uid, *vector = parse_id_and_vector(line)
        return [("id", np.int32)] + [
            (f"d{i+1}", np.float32) for i, val in enumerate(vector)
        ]


if __name__ == "__main__":
    with ProgressBar():
        # read
        data_path = Path(args.data_path)
        meta = make_meta(data_path)
        d = db.read_text(data_path).map(parse_id_and_vector).to_dataframe(meta=meta)
        # .drop_duplicates(subset=["id"]).set_index("id")
        num_samples = len(d)

        # SAVE
        fname = (
            Path(data_path.stem) / f"{data_path.stem}.*.hdf5" if args.low_memory else f"{data_path.stem}.hdf5"
        )
        out_path = data_path.parent / fname if not args.out_path else args.out_path
        data_columns = ["id"]
        d.to_hdf(
            out_path,
            HDF_PATH,
            mode="w",
            data_columns=data_columns,
            complevel=COMP_LEVEL,
        )
