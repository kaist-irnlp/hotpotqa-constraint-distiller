#!/usr/bin/env sh

python sparsenet_trainer.py --data_dir ../../data --dense bert --gpus -1 -d ddp --n 10000 --k 200 --use_amp --amp_level O1
