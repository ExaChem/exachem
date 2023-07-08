#!/bin/bash

num_gpus=4

# https://docs.nersc.gov/jobs/affinity/
# Usage: srun --gpus-per-node 4 --ntasks-per-node 5 -c 24 --cpu-bind=cores ./set_affinity_gpu_perlmutter.sh

# need to assign GPUs in reverse order due to topology ?
gpu=$((${num_gpus} - 1 - ${SLURM_LOCALID} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$((${SLURM_LOCALID} % ${num_gpus}))
# export CUDA_VISIBLE_DEVICES=$gpu

if [ $SLURM_LOCALID -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "RANK=${SLURM_PROCID} LOCAL_RANK=${SLURM_LOCALID} gpu=${gpu}"
exec "$@"
