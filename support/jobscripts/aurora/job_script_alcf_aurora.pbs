#!/bin/bash

#PBS -N tamm_test
#PBS -l select=5
#PBS -A <project>
#PBS -l walltime=01:00:00
#PBS -m abe
#PBS -k doe
#PBS -q <queue>
#PBS -j oe

#env
set -x

module use /soft/modulefiles/
module load spack-pe-gcc/0.4-rc1 numactl/2.0.14-gcc-testing cmake
module load oneapi/release/2023.12.15.001
#module load intel_compute_runtime/release/stable-736.25
module list

export TZ='/usr/share/zoneinfo/US/Central'

cd $HOME/output
pwd

export MPIR_CVAR_ENABLE_GPU=0
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_CQ_FILL_PERCENT=20
export ZE_FLAT_DEVICE_HIERARCHY=flat
export MEMKIND_HBW_THRESHOLD=402400
export SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0x001
export ZES_ENABLE_SYSMAN=1
export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=1
export GA_PROGRESS_RANKS_DISTRIBUTION_CYCLIC=0

unset MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE

NNODES=`wc -l < $PBS_NODEFILE`
NTHREADS=1

export LD_LIBRARY_PATH=$HOME/install/tamm/lib64:$LD_LIBRARY_PATH

EXE=<tamm-exe>
INPUT=<args>

#1pr

# export GA_NUM_PROGRESS_RANKS_PER_NODE=1
# NRANKS_PER_NODE=13

# NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
# echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

# mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --spindle --no-vni --pmi=pmix --cpu-bind list:2:10:18:26:34:42:58:66:74:82:90:98:99 --mem-bind list:0:0:0:0:0:0:1:1:1:1:1:1:1 --env TAMM_USE_MEMKIND=1 --env OMP_NUM_THREADS=${NTHREADS} ./aurora_bind_tiles_closest.sh $EXE $INPUT

#2pr

export GA_NUM_PROGRESS_RANKS_PER_NODE=2
NRANKS_PER_NODE=14
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --spindle --no-vni --pmi=pmix --cpu-bind list:0-7:8-15:16-23:24-31:32-39:40-47:50:52-59:60-67:68-75:76-83:84-91:92-99:100 --mem-bind list:0:0:0:0:0:0:0:1:1:1:1:1:1:1 --env TAMM_USE_MEMKIND=1 ./aurora_bind_tiles_closest_2pr.sh $EXE $INPUT

