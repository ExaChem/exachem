
include(TargetMacros)

set(TASK_SRCDIR task)

set(TASK_INCLUDES
    ${TASK_SRCDIR}/ec_task.hpp
    ${TASK_SRCDIR}/geom_analysis.hpp
    )

set(TASK_SRCS
    ${TASK_SRCDIR}/ec_task.cpp
    ${TASK_SRCDIR}/geom_analysis.cpp
   )

