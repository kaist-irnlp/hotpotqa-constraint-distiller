from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--out_dir", type=str, default="./dataset")
    args = parser.parse_args()

    # out_dir
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # split & save
    split_ratio = {"train": 0.8, "valid": 0.1, "test": 0.1}
    with ProgressBar():
        d = dd.read_parquet(args.data_path).repartition(npartitions=4)
        splits = d.random_split(split_ratio.values())
        for split_name, split in zip(split_ratio.keys(), splits):
            out_path = out_dir / f"{split_name}.parquet"
            split.compute().to_parquet(out_path, compression="snappy")
