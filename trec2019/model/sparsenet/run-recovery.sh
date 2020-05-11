#!/usr/bin/env bash

export GPUS=$1
export OPT_LOSS="--use_recovery_loss --no_task_loss"

bash ./run.sh
