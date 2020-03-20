import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from argparse import ArgumentParser
from pathlib import Path
import numcodecs
import zarr

parser = ArgumentParser()
parser.add_argument("fin", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    fout = Path(args.fin).with_suffix(".tsv.zarr.zip")

    with ProgressBar():
        d = dd.read_csv(
            args.fin,
            sep="\t",
            encoding="utf-8",
            blocksize="1MB",
            names=["query", "doc_pos", "doc_neg"],
        )
        arr = d.to_dask_array(lengths=True)
        arr.rechunk({0: 32}).to_zarr(
            zarr.ZipStore(str(fout)), object_codec=numcodecs.VLenUTF8()
        )

