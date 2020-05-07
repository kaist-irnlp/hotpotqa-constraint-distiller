#!/usr/bin/env sh

GPUS=$1

python sparsenet_trainer.py --data_dir /home/kyoungrok/data/news20 --gpus "${GPUS}" --n 2000 2000 2000 --k 200 100 40 --input_size 300 --output_size 20
