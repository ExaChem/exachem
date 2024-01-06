#!/bin/bash

if [ $PALS_LOCAL_RANKID -eq 0 ]; then
    export ZE_AFFINITY_MASK=0.0
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 1 ]; then
    export ZE_AFFINITY_MASK=0.1
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 2 ]; then
    export ZE_AFFINITY_MASK=1.0
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 3 ]; then
    export ZE_AFFINITY_MASK=1.1
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 4 ]; then
    export ZE_AFFINITY_MASK=2.0
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 5 ]; then
    export ZE_AFFINITY_MASK=2.1
    export MEMKIND_HBW_NODES=2
elif [ $PALS_LOCAL_RANKID -eq 6 ]; then
    export ZE_AFFINITY_MASK=3.0
    export MEMKIND_HBW_NODES=3
elif [ $PALS_LOCAL_RANKID -eq 7 ]; then
    export ZE_AFFINITY_MASK=3.1
    export MEMKIND_HBW_NODES=3
elif [ $PALS_LOCAL_RANKID -eq 8 ]; then
    export ZE_AFFINITY_MASK=4.0
    export MEMKIND_HBW_NODES=3    
elif [ $PALS_LOCAL_RANKID -eq 9 ]; then
    export ZE_AFFINITY_MASK=4.1
    export MEMKIND_HBW_NODES=3    
elif [ $PALS_LOCAL_RANKID -eq 10 ]; then
    export ZE_AFFINITY_MASK=5.0
    export MEMKIND_HBW_NODES=3
elif [ $PALS_LOCAL_RANKID -eq 11 ]; then
    export ZE_AFFINITY_MASK=5.1
    export MEMKIND_HBW_NODES=3
fi
# Launch the executable:
$*
