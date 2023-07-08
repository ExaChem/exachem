#!/bin/bash

#SBATCH -A m3946_g   -q regular
#SBATCH -C gpu
#SBATCH -t 00:30:00
#SBATCH -N 10
#SBATCH --gpus-per-node 4
#SBATCH -o my_app_10n.%j


module purge
module load PrgEnv-gnu
module load craype-x86-milan
module load cpe gpu
module load cpe-cuda
module load cmake
module unload cray-libsci
module list

set -x

export CRAYPE_LINK_TYPE=dynamic

export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=0

export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=1
export GA_NUM_PROGRESS_RANKS_PER_NODE=1

# nvidia-cuda-mps-control -d
srun -u -c 14 --cpu_bind=cores --gpu-bind=closest --ntasks-per-node 9 $TAMM_EXE $TAMM_INP
# echo quit | nvidia-cuda-mps-control
