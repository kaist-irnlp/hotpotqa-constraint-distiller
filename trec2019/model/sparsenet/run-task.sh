#!/usr/bin/env bash

GPUS=$1
OPT_LOSS=""

python sparsenet_trainer.py --data_dir /home/kyoungrok/data/news20 --gpus "${GPUS}" --n 2000 --k 40 --input_size 300 --output_size 20 ${OPT_LOSS}
