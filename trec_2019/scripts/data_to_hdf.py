import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from argparse import ArgumentParser
from pathlib import Path
import math
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_path", type=str, default=None)
args = parser.parse_args()

COMP_LEVEL = 9
HDF_PATH = "/data"
fields = ["id", "text"]

if __name__ == "__main__":
    with ProgressBar():
        # read
        data_path = Path(args.data_path)
        d = dd.read_csv(data_path, sep='\t', names=fields)
        num_samples = len(d)

        # SAVE
        out_path = data_path.parent / \
            f"{data_path.stem}.hdf5" if not args.out_path else args.out_path
        data_columns = ["id"]
        d.to_hdf(
            out_path,
            HDF_PATH,
            mode="w",
            data_columns=data_columns,
            index=True,
            complevel=COMP_LEVEL
        )
