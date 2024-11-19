
include(TargetMacros)

set(CCSD_SRCDIR cc/ccsd)
set(CCSD_SRCS
    ${CCSD_SRCDIR}/ccsd_util.cpp
    ${CCSD_SRCDIR}/cd_ccsd.cpp
    ${CCSD_SRCDIR}/cd_ccsd_cs_ann.cpp
    ${CCSD_SRCDIR}/cd_ccsd_os_ann.cpp
    )

set(CCSD_INCLUDES
    ${CCSD_SRCDIR}/ccsd_util.hpp
    ${CCSD_SRCDIR}/cd_ccsd.hpp
    ${CCSD_SRCDIR}/cd_ccsd_cs_ann.hpp
    ${CCSD_SRCDIR}/cd_ccsd_os_ann.hpp
)    

