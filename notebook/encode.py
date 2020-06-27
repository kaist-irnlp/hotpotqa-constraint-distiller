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
from transformers import AutoModel, AutoConfig, AutoTokenizer

torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("fpath", type=str)
parser.add_argument("gpu", type=int, default=0)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()

# # Load

# ## Model

# In[2]:


# Store the model we want to use
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-uncased"

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
        return_attention_masks=True,
        pad_to_max_length=True,
        max_length=200,
        return_special_tokens_masks=True,
    )
    # encode
    inputs = tokens["input_ids"].to(device)
    outputs = model(inputs)
    del inputs
    torch.cuda.empty_cache()
    hidden_states = outputs[2]
    pooling_layer = hidden_states[-2]
    # pooling
    return pooling_layer.mean(1)


# In[ ]:


data_dir = Path("../msmarco-passages/")
output_dir = Path("./output")
if not output_dir.exists():
    output_dir.mkdir(parents=True)

fpath = Path(args.fpath)
z = zarr.open(str(fpath))
docs = z.text[args.start : args.end][:]
real_shape = z.text.shape
num_digits = real_shape[0] // 10

# save to
emb_path = "dense/bert"
emb_dim = 768
if emb_path not in z:
    z.zeros("dense/bert", shape=(len(z.text), emb_dim), chunks=(64, None), dtype="f4")
z_embs = z[emb_path]


# In[6]:

CHUNK_SIZE = 8
i = 0
num_chunks = int(len(docs) / CHUNK_SIZE)
print("# of chunks", num_chunks)
while True:
    print(f"{i} / {num_chunks}")
    start, end = i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE
    real_start, real_end = start + args.start, end + args.start
    embs = batch_encode(docs[start:end])
    # save
    embs = embs.cpu()
    z_embs[real_start:real_end] = embs[:]
    # embs_list.append(embs)
    np.save(
        output_dir / f"{real_start:07d}_{(real_start + embs.shape[0]):07d}.npy", embs,
    )
    # check exit condition
    if embs.shape[0] < CHUNK_SIZE:
        break
    i += 1
# embs = torch.cat(embs_list, dim=0).cpu().numpy()
