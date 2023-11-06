
include(TargetMacros)

set(COMMON_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/common)
set(COMMON_SRCS
    ${COMMON_SRCDIR}/sinfo.cpp
    ${COMMON_SRCDIR}/cutils.cpp
    ${COMMON_SRCDIR}/molden.cpp
    ${COMMON_SRCDIR}/fcidump.cpp
    ${COMMON_SRCDIR}/json_data.cpp
    )

