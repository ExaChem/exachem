
include(TargetMacros)

set(DUCC_SRCDIR cc/ducc)
set(DUCC_SRCS
    ${DUCC_SRCDIR}/ducc_driver.cpp
    ${DUCC_SRCDIR}/ducc-t_ccsd.cpp
    )

if(USE_NWQSIM)
    list(APPEND DUCC_SRCS ${DUCC_SRCDIR}/qflow-t_ccsd.cpp)
endif()

set(DUCC_INCLUDES ${DUCC_SRCDIR}/ducc-t_ccsd.hpp)
