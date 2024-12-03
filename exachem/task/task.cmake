
include(TargetMacros)

set(TASK_SRCDIR task)

set(TASK_INCLUDES
    ${TASK_SRCDIR}/ec_task.hpp
    ${TASK_SRCDIR}/geometry_analysis.hpp
    ${TASK_SRCDIR}/geometry_optimizer.hpp
    ${TASK_SRCDIR}/numerical_gradients.hpp
    )

set(TASK_SRCS
    ${TASK_SRCDIR}/ec_task.cpp
    ${TASK_SRCDIR}/geometry_analysis.cpp
    ${TASK_SRCDIR}/geometry_optimizer.cpp
    ${TASK_SRCDIR}/numerical_gradients.cpp
   )

