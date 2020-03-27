#!/usr/bin/env sh

DATA_DIR=$1
GPU=$2
N=$3
K=$4

for n in $N
do
    for k in $K
    do
        python sparsenet_trainer.py --data_dir ${DATA_DIR} --gpus ${GPU} --n ${n} --k ${k}
    done
done
