#!/usr/bin/env bash
export OMP_NUM_THREADS=1
PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=${3:-"0"}
export CUDA_VISIBLE_DEVICES="$GPUS"
GPU_NUM=$(((${#GPUS}+1)/2))
echo "using $GPU_NUM gpus with number $GPUS"


$PYTHON -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$RANDOM \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
