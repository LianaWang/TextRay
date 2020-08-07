#!/bin/bash
# kill last traininig processes within current bash
pids=$(ps -f| grep "../../curve/tools/train.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$pids" ]; then
    kill $pids;
fi
# make sure we are under current exp directory
cur_path=$(dirname $(readlink -f $0))
cd $cur_path
PYTHONPATH=../../:$PATHPATH ../../curve/tools/dist_train.sh ./config.py $1
