#!/usr/bin/env sh

DATA_DIR=$1
N="1000 3000 5000 10000"
K="50 100 200 300 500"

for n in $N
do
    for k in $K
    do
        python sparsenet_trainer.py --data_dir ${DATA_DIR} --gpus -1 --n ${n} --k ${k} -d ddp --fine_tune
    done
done