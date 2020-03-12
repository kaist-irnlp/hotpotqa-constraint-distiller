#!/usr/bin/env sh

python sparsenet_trainer.py --data_dir ../../data --dense bert --gpus -1 -d ddp --n 2000 10000 2000 --k 40 200 40 --use_amp --amp_level O1
