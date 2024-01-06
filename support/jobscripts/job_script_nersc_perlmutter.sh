#!/bin/bash

#SBATCH -A m3196_g   -q regular
#SBATCH -C gpu
#SBATCH -t 00:15:00
#SBATCH -N 4
#SBATCH -J pt
#SBATCH --gpus-per-node 4
#SBATCH -o test_4n.%j

set -x

module load PrgEnv-gnu
module load cudatoolkit/11.7
module load cpe-cuda
module load gcc/11.2.0
module load cmake
module unload cray-libsci
module unload craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=0
export CRAYPE_LINK_TYPE=dynamic

ppn=4

export OMP_NUM_THREADS=1
export GA_NUM_PROGRESS_RANKS_PER_NODE=1
export MPICH_OFI_SKIP_NIC_SYMMETRY_TEST=1

export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=1

TAMM_EXE=/$PSCRATCH/TAMM/build/test_stage/$PSCRATCH/install/tamm/tests/Test_Mult_Ops
TAMM_INPUT="100 40"

cd /$PSCRATCH/output

srun -u --cpu_bind=map_cpu:0,16,32,48,64 --gpu-bind=closest --gpus-per-node 4 --ntasks-per-node $(( ppn + GA_NUM_PROGRESS_RANKS_PER_NODE )) $TAMM_EXE $TAMM_INP
