
include(TargetMacros)

set(DUCC_SRCDIR cc/ducc)
set(DUCC_SRCS
    ${DUCC_SRCDIR}/ducc_driver.cpp
    ${DUCC_SRCDIR}/ducc-t_ccsd.cpp
    )

set(DUCC_INCLUDES ${DUCC_SRCDIR}/ducc-t_ccsd.hpp)
