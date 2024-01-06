#!/bin/bash

if [ $PALS_LOCAL_RANKID -eq 0 ]; then
    export ZE_AFFINITY_MASK=0
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 1 ]; then
    export ZE_AFFINITY_MASK=1
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 2 ]; then
    export ZE_AFFINITY_MASK=2
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 3 ]; then
    export ZE_AFFINITY_MASK=3
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 4 ]; then
    export ZE_AFFINITY_MASK=4
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 5 ]; then
    export ZE_AFFINITY_MASK=5
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 7 ]; then
    export ZE_AFFINITY_MASK=6
    export MEMKIND_HBW_NODES=3
elif [ $PALS_LOCAL_RANKID -eq 8 ]; then
    export ZE_AFFINITY_MASK=7
    export MEMKIND_HBW_NODES=3
elif [ $PALS_LOCAL_RANKID -eq 9 ]; then
    export ZE_AFFINITY_MASK=8
    export MEMKIND_HBW_NODES=3
elif [ $PALS_LOCAL_RANKID -eq 10 ]; then
    export ZE_AFFINITY_MASK=9
    export MEMKIND_HBW_NODES=3
elif [ $PALS_LOCAL_RANKID -eq 11 ]; then
    export ZE_AFFINITY_MASK=10
    export MEMKIND_HBW_NODES=3
elif [ $PALS_LOCAL_RANKID -eq 12 ]; then
    export ZE_AFFINITY_MASK=11
    export MEMKIND_HBW_NODES=3
fi

#echo "[I am rank $PMIX_RANK on node `hostname`]  Localrank=$PALS_LOCAL_RANKID, ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK, HBW_NODES=$MEMKIND_HBW_NODES, ZEX_CONFIG=$ZEX_NUMBER_OF_CCS"

# Launch the executable:
$*
