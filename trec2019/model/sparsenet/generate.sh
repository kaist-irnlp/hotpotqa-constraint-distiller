#!/usr/bin/env bash

MODEL_DIR="/Users/kyoungrok/Dropbox/Project/naver/experiment/200324_SparseNet/result/200508_news20"
DATA_DIR="/Users/kyoungrok/Dropbox/Project/naver/experiment/200324_SparseNet/data/news20"

for i in {0..7}
do
    python sparsenet_generator.py \
        --model_dir "${MODEL_DIR}/version_${i}" \
        --data_dir "${DATA_DIR}"
done