
include(TargetMacros)

set(CD_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/)
set(CD_INCLUDES
    ${CD_SRCDIR}/cholesky/cholesky_2e.hpp
    ${CD_SRCDIR}/cholesky/v2tensors.cpp
    ${CD_SRCDIR}/cholesky/cholesky_2e_driver.hpp
    )
set(CD_SRCS
    ${CD_SRCDIR}/cholesky/cholesky_2e.cpp
    ${CD_SRCDIR}/cholesky/v2tensors.cpp
    ${CD_SRCDIR}/cholesky/cholesky_2e_driver.cpp
    )

