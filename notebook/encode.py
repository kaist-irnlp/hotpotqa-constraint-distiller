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
from fse import SplitIndexedList, IndexedList
from textblob import TextBlob
import gensim.downloader as api
from fse.models import SIF

torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("fpath", type=str)
parser.add_argument("gpu", type=int)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_len", type=int, default=512)
parser.add_argument("--model", type=str, default="nboost/pt-bert-large-msmarco")
parser.add_argument("--pooling", type=str, default="cls")
parser.add_argument("--pooling_layer", type=int, default=0)
args = parser.parse_args()

# encoder functions
batch_encode = None


def _batch_encode_fse(texts, model):
    return model.infer(texts)


def _pooling_bert(outputs, pooling):
    output = None
    last_hidden_state, pooler_output, hidden_states = (
        outputs[0],
        outputs[1],
        outputs[2][1:],
    )
    pooling_layer = args.pooling_layer
    if pooling == "cls":  # [CLS] of the last layer
        # output = last_hidden_state[0]
        output = pooler_output
    elif pooling == "mean":  # mean pooling
        output = hidden_states[pooling_layer].mean(1)
    elif pooling == "max":  # max pooling
        output = hidden_states[pooling_layer].max(1)
    return output


def _batch_encode_bert(texts, tokenizer, model):
    # tokenize
    tokens = tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        pad_to_max_length=True,
        max_length=args.max_len,
        return_attention_masks=True,
        return_special_tokens_masks=True,
    )
    # encode
    input_ids = tokens["input_ids"].to(device)
    token_type_ids = tokens["token_type_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    output = _pooling_bert(outputs, args.pooling)
    return output


# Store the model we want to use
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
model_name = args.model

# We need to create the model and tokenizer
if "bert" in model_name:
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    model = AutoModel.from_pretrained(model_name, config=config).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name,)
    batch_encode = _batch_encode_bert
    emb_dim = config.hidden_size
elif "fse" in model_name:
    model = api.load("glove-wiki-gigaword-300")
    batch_encode = _batch_encode_fse
    emb_dim = model.vector_size


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
emb_path = f"dense/{args.model}"
if args.model != "fse":
    if args.pooling == "cls":
        emb_path = f"{emb_path}_{args.pooling}"
    else:
        emb_path = f"{emb_path}_{args.pooling}_{args.pooling_layer}"
z = zarr.open(str(fpath), "r+")
if emb_path not in z:
    z_embs = z.zeros(
        emb_path, shape=(len(z.text), emb_dim), chunks=(256, None), dtype="f4"
    )
else:
    z_embs = z[emb_path]


# encode & save
if "bert" in model_name:
    for i, batch in enumerate(tqdm(loader)):
        # encode
        embs = batch_encode(batch, tokenizer, model).cpu().numpy()
        # save
        start = i * args.batch_size
        end = start + embs.shape[0]
        z_embs[start:end] = embs[:]
elif "fse" in model_name:
    sent_model = SIF(model, workers=8, lang_freq="en")
    # train
    for i, batch in enumerate(loader):
        sentences = IndexedList([TextBlob(s).tokens for s in batch])
        sent_model.train(sentences)
    sent_model.save(fpath.parent / "fse.model")
    # infer
    for i, batch in enumerate(loader):
        sentences = IndexedList([TextBlob(s).tokens for s in batch])
        # encode
        embs = batch_encode(sentences, sent_model)
        # save
        start = i * args.batch_size
        end = start + embs.shape[0]
        z_embs[start:end] = embs[:]
