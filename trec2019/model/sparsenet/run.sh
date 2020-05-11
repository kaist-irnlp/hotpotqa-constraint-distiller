#!/usr/bin/env bash

python sparsenet_trainer.py --data_dir /home/kyoungrok/data/news20 --gpus "${GPUS}" --n 2000 2000 --k 200 40 --input_size 300 --output_size 20 ${OPT_LOSS}
