#!/bin/bash

# Add the number of GPUs per node here
num_gpus=

if test -z "$num_gpus"
then
    echo "num_gpus is not set"
    exit 1
fi

if [ "$num_gpus" -lt 1 ];
then
    echo "num_gpus cannot be < 1"
    exit 1
fi

#Usage: mpiexec -n X --ppn Y --env OMP_NUM_THREADS=1 $PWD/sample_mpich_gpu_bind.sh $EXEC ....
export CUDA_VISIBLE_DEVICES=$((${OMPI_COMM_WORLD_LOCAL_RANK} % ${num_gpus}))

echo "RANK=${OMPI_COMM_WORLD_RANK} LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK} gpu=${CUDA_VISIBLE_DEVICES}"
exec $*
