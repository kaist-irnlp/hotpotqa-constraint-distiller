import torch
import pandas as pd
import gzip
from torch.utils.data import Dataset, ConcatDataset, IterableDataset


class TRECTripleEmbeddingDataset(Dataset):
    HDF_KEY = '/data'
    IDX_UID = 1
    IDX_EMB = 2

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = None
        self._data_shape = self._read_data_shape()

    def _read_data_shape(self):
        with h5py.File(self.data_path, "r") as fin:
            dataset = fin[self.HDF_KEY]["table"]
            num_samples = dataset.shape[0]
            dimension = dataset[0][self.vector_idx].shape[0]
            return (num_samples, dimension)

    def get_data_shape(self):
        return self._data_shape

    def __len__(self):
        return self._data_shape[0]

    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.data_path, "r")[self.HDF_KEY]["table"]
        sample = self.data[index]
        uid = sample[self.IDX_UID][0].decode(encoding="utf-8")
        vector = np.array(sample[self.IDX_EMB], dtype=np.float32)
        return {"id": uid, "vector": vector}


class TRECTripleIdDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = pd.read_csv(data_path, sep='\t', names=[
                                'q_id', 'pos_id', 'neg_id'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index].to_dict()
