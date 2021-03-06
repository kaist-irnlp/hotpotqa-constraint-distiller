import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from torch.nn import functional as F
from pathlib import Path
import zarr
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser

from trec2019.utils.dense import *
from torch import optim
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel


class TripleDataset(Dataset):
    def __init__(self, fpath):
        self.f = zarr.open(fpath, "r")

    def __len__(self):
        return len(self.f)

    def __getitem__(self, index):
        row = self.f[index]
        return {"query": row[0], "doc_pos": row[1], "doc_neg": row[2]}


class DensifyModel(pl.LightningModule):
    def __init__(self, hparams, dense):
        super().__init__()
        self.hparams = hparams
        self.dense = dense
        self.dummy = torch.nn.Linear(1, 1)
        # output
        out_path = self._get_output_path()
        init_shape = chunks = (self.hparams.batch_size * 2, self.dense.get_dim())
        self.batch_idx = 0
        # self.fout = zarr.open(zarr.ZipStore(out_path), mode="w")
        self.fout = zarr.open(out_path, mode="w")
        q_sync = zarr.ProcessSynchronizer("sync/query.sync")
        pd_sync = zarr.ProcessSynchronizer("sync/doc_pos.sync")
        nd_sync = zarr.ProcessSynchronizer("sync/doc_neg.sync")

        self.out_query = self.fout.zeros(
            "query", shape=init_shape, chunks=chunks, synchronizer=q_sync,
        )
        self.out_doc_pos = self.fout.zeros(
            "doc_pos", shape=init_shape, chunks=chunks, synchronizer=pd_sync,
        )
        self.out_doc_neg = self.fout.zeros(
            "doc_neg", shape=init_shape, chunks=chunks, synchronizer=nd_sync,
        )

    def _write_outputs(self, outputs, batch_idx):
        dense_query = outputs["dense_query"]
        dense_doc_pos = outputs["dense_doc_pos"]
        dense_doc_neg = outputs["dense_doc_neg"]

        # write when batch_idx = 0 else append
        if batch_idx == 0:
            # resize to fit the batch size
            n_rows = len(dense_query)
            self.out_query.resize(n_rows, None)
            self.out_doc_pos.resize(n_rows, None)
            self.out_doc_neg.resize(n_rows, None)
            # write 1-st data
            self.out_query[:] = dense_query
            self.out_doc_pos[:] = dense_doc_pos
            self.out_doc_neg[:] = dense_doc_neg
        else:
            # append afterward
            self.out_query.append(dense_query, axis=0)
            self.out_doc_pos.append(dense_doc_pos, axis=0)
            self.out_doc_neg.append(dense_doc_neg, axis=0)

    def _get_output_path(self):
        hparams = self.hparams
        data_name = Path(hparams.data_path).name
        fout_name = (
            data_name.replace(".tsv", "").replace(".zarr", "").replace(".zip", "")
        )
        fout_path = (
            Path("output") / f"{fout_name}_{hparams.model}_{self.dense.weights}.zarr"
        )
        if not fout_path.parent.exists():
            fout_path.parent.mkdir(parents=True)
        return str(fout_path)

    # def configure_ddp(self, model, device_ids):
    #     # Lightning DDP simply routes to test_step, val_step, etc...
    #     model = LightningDistributedDataParallel(
    #         model, device_ids=device_ids, find_unused_parameters=False
    #     )
    #     return model

    def prepare_data(self):
        self.train_dataset = TripleDataset(self.hparams.data_path)

    def forward(self, query, doc_pos, doc_neg, dummy_val):
        with torch.no_grad():
            Qe = self.dense(query).detach().cpu().numpy()
            PDe = self.dense(doc_pos).detach().cpu().numpy()
            NDe = self.dense(doc_neg).detach().cpu().numpy()
        return {
            "dense_query": Qe,
            "dense_doc_pos": PDe,
            "dense_doc_neg": NDe,
            "dummy": self.dummy(dummy_val),
        }

    def training_step(self, batch, batch_idx):
        query, doc_pos, doc_neg = batch["query"], batch["doc_pos"], batch["doc_neg"]
        dummy_val = torch.randn(5, 1, dtype=torch.float).to(self.dummy.weight.device)
        return self.forward(query, doc_pos, doc_neg, dummy_val)

    def training_step_end(self, outputs):
        # save
        self._write_outputs(outputs, self.batch_idx)
        self.batch_idx += 1

        # return dummy
        dummy = outputs['dummy']
        loss = F.mse_loss(dummy, dummy)
        return {"loss": loss}

    def train_dataloader(self):
        N_WORKERS = 0

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=N_WORKERS,
            pin_memory=True,
            shuffle=False,
        )

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


def get_dense_model(model, weights):
    if model == "bert":
        return BertEmbedding(weights=weights)
    elif model == "bow":
        raise NotImplementedError


root_dir = str(Path(__file__).parent.absolute())


def main(hparams):
    dense = get_dense_model(hparams.model, hparams.weights)
    model = DensifyModel(hparams, dense)
    tt_logger = loggers.TestTubeLogger(root_dir)
    trainer = Trainer(
        logger=tt_logger,
        default_save_path=root_dir,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_amp,
        amp_level=hparams.amp_level,
        benchmark=True,
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("model", type=str, choices=["bow", "bert"])
    parser.add_argument("--weights", type=str, default="bert-base-uncased")
    parser.add_argument("--distributed_backend", "-d", type=str, default=None)
    parser.add_argument("--gpus", default=None, type=str)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--amp_level", default="O1", type=str)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    hparams = parser.parse_args()

    # run
    print(vars(hparams))
    main(hparams)
