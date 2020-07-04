#!/usr/bin/env python
# coding: utf-8

# In[7]:


from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import zarr
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer

torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("fpath", type=str)
parser.add_argument("gpu", type=int)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--emb_dim", type=int, default=768)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()

# # Load

# ## Model

# In[2]:


# Store the model we want to use
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-cased"

# We need to create the model and tokenizer
config = AutoConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
model = AutoModel.from_pretrained(MODEL_NAME, config=config).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,)


# # Encode

# In[5]:


def batch_encode(texts):
    # tokenize
    tokens = tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        pad_to_max_length=True,
        max_length=200,
        return_attention_masks=True,
        return_special_tokens_masks=True,
    )
    # encode
    inputs = tokens["input_ids"].to(device)
    outputs = model(inputs)
    return outputs[1].cpu()
    # hidden_states = outputs[2]
    # pooling_layer = hidden_states[-2]
    # return pooling_layer.mean(1)


# In[ ]:


class TextDataset(Dataset):
    def __init__(self, data_path):
        self.data = zarr.open(str(data_path)).text[:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# load data
fpath = Path(args.fpath)
dset = TextDataset(fpath)
loader = DataLoader(dset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

# save to
emb_path = "dense/bert_new"
emb_dim = args.emb_dim
z = zarr.open(str(fpath))
if emb_path not in z:
    z_embs = z.zeros(
        emb_path, shape=(len(z.text), emb_dim), chunks=(64, None), dtype="f4"
    )
else:
    z_embs = z[emb_path]


# encode & save
for i, batch in enumerate(tqdm(loader)):
    # encode
    embs = batch_encode(batch).cpu()
    # save
    start = i * args.batch_size
    end = start + embs.shape[0]
    z_embs[start:end] = embs[:]

