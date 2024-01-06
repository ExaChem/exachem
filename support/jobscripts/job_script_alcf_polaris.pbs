#!/bin/bash

#PBS -N run_ci
#PBS -l select=1:system=polaris,place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -A gpu_hack

module load PrgEnv-gnu cpe-cuda cudatoolkit-standalone/11.8.0 cmake
module list

export GA_NUM_PROGRESS_RANKS_PER_NODE=1
export CRAYPE_LINK_TYPE=dynamic
export MPICH_GPU_SUPPORT_ENABLED=0

cd ${PBS_O_WORKDIR}

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=5
NDEPTH=1
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"
env

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:24:16:8:0:23 --env OMP_NUM_THREADS=${NTHREADS} ./set_affinity_gpu_polaris.sh ${TAMM_EXE} ${INPUT}
