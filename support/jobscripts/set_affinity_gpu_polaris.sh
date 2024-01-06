#!/bin/bash

num_gpus=4

# need to assign GPUs in reverse order due to topology
# See Polaris Device Affinity Information https://www.alcf.anl.gov/support/user-guides/polaris/hardware-overview/machine-overview/index.html

#Usage: mpiexec -n 5 --ppn 5 --cpu-bind list:24:16:8:0:23 --env OMP_NUM_THREADS=1 $PWD/set_affinity_gpu_polaris.sh

gpu=$((${num_gpus} - 1 - ${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu

if [ $PALS_LOCAL_RANKID -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

#echo "RANK= ${PMI_RANK} LOCAL_RANK= ${PALS_LOCAL_RANKID} gpu= ${gpu}"
exec "$@"
