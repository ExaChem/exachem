
include(TargetMacros)

set(CD_SRCDIR cholesky)
set(CD_INCLUDES
    ${CD_SRCDIR}/cholesky_2e.hpp
    ${CD_SRCDIR}/v2tensors.hpp
    ${CD_SRCDIR}/two_index_transform.hpp
    ${CD_SRCDIR}/cholesky_2e_driver.hpp
    )
set(CD_SRCS
    ${CD_SRCDIR}/cholesky_2e.cpp
    ${CD_SRCDIR}/v2tensors.cpp
    ${CD_SRCDIR}/cholesky_2e_driver.cpp
    )

