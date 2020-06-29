#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import zarr
from pathlib import Path
from argparse import ArgumentParser

try:
    import cupy as np
except:
    import numpy as np


# In[22]:

parser = ArgumentParser()
parser.add_argument("--data_dir", default="d:/Data/msmarco-passages")
parser.add_argument("--emb_path", default="dense/fse")
args = parser.parse_args()


data_dir = Path(args.data_dir)
emb_path = args.emb_path


# # Query

# ## Target IDs

# In[4]:


q_ids = pd.read_csv(
    data_dir / "runs/queries.dev.small.tsv", sep="\t", names=["id", "query"]
).id.values


# ## Target Embs

# In[14]:


z = zarr.open(str(data_dir / "queries.eval.zarr"))
q_embs = z[emb_path][:]


# In[7]:


# z = zarr.open('../msmarco-passages/queries.eval.zarr')
# q_ids_all = z.id[:]
# sorter = np.argsort(q_ids_all)
# indices = sorter[np.searchsorted(q_ids_all, q_ids, sorter=sorter)]
# q_embs = z.dense["bert"].oindex[indices.tolist(), :][:]


# # Passages

# In[17]:


z = zarr.open(str(data_dir / "docs.eval.zarr"))
p_embs = z[emb_path][:]
p_ids = z.id[:]


# # Retrieve

# In[19]:


from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def search(q_emb, p_embs, p_ids, topk=1000):
    scores = cosine_similarity(p_embs, q_emb.reshape(1, -1))
    indices = np.argsort(scores.squeeze())[::-1][:topk]
    return p_ids[indices]


# In[20]:

out_dir = Path("runs")
if not out_dir.exists():
    out_dir.mkdir(parents=True)

emb_name = args.emb_path.split("/")[-1]
with open(
    out_dir / f"run.msmarco-passage.dev.small.{emb_name}.tsv", "w", encoding="utf-8"
) as f:
    for q_id, q_emb in tqdm(zip(q_ids, q_embs)):
        topk = search(q_emb, p_embs, p_ids)
        for i, tk in enumerate(topk):
            f.write(f"{q_id}\t{tk}\t{i+1}\n")


# In[ ]:

