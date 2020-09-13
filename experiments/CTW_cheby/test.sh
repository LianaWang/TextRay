#!/bin/bash
# usage1: ./test.sh 0,1,2,3 
# usage2: ./test.sh 0,1,2,3 custom_pkl_filename
cur_path=$(dirname $(readlink -f $0))
cd $cur_path

if [ -z "$1" ]; then
    gpus=0
else
    gpus=$1
fi

if [ -z "$2" ]; then
    exp_name=$(basename $cur_path)_$(date +"%m%d%H%M")
else
    exp_name=$2
fi
echo "test on gpus=${gpus} and output '${exp_name}.pkl/json' files"
PYTHONPATH=../../:$PATHPATH ../../curve/tools/dist_test.sh ./config.py work_dirs/latest.pth $gpus --out $exp_name.pkl --fuse_conv_bn
