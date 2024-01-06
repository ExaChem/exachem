
include(TargetMacros)

set(COMMON_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/common)
set(COMMON_SRCS
    ${COMMON_SRCDIR}/sinfo.cpp
    ${COMMON_SRCDIR}/cutils.cpp
    ${COMMON_SRCDIR}/molden.cpp
    ${COMMON_SRCDIR}/ecatom.cpp
    ${COMMON_SRCDIR}/fcidump.cpp
    ${COMMON_SRCDIR}/parser_options.cpp
    ${COMMON_SRCDIR}/system_data.cpp
    )

