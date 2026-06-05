
include(TargetMacros)

set(OPTIMIZERS_SRCDIR optimizers)

set(OPTIMIZERS_INCLUDES
    ${OPTIMIZERS_SRCDIR}/pyberny_impl.hpp
    )

set(OPTIMIZERS_SRCS
    ${OPTIMIZERS_SRCDIR}/pyberny_impl.cpp
   )

if (EXACHEM_HAS_PYTHON)
  list(APPEND OPTIMIZERS_INCLUDES ${OPTIMIZERS_SRCDIR}/geometric.hpp)
  list(APPEND OPTIMIZERS_SRCS ${OPTIMIZERS_SRCDIR}/geometric.cpp)
endif()
