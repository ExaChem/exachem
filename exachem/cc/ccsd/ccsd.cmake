
include(TargetMacros)

set(CCSD_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/cc/ccsd)
set(CCSD_SRCS
    ${CCSD_SRCDIR}/ccsd_util.cpp
    ${CCSD_SRCDIR}/cd_ccsd.cpp
    ${CCSD_SRCDIR}/cd_ccsd_cs_ann.cpp
    ${CCSD_SRCDIR}/cd_ccsd_os_ann.cpp
    )

