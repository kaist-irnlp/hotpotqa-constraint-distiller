#!/usr/bin/env bash

python sparsenet_trainer.py --data_dir /home/kyoungrok/data/news20 --gpus "${GPUS}" --n 5000 --k 50 --input_size 300 --output_size 20 ${OPT_LOSS} ${OPT_DIST}
