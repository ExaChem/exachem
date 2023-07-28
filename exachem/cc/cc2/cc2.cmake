
include(TargetMacros)

set(CC2_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/cc/cc2)
set(CC2_SRCS
    ${CC2_SRCDIR}/cd_cc2.cpp    
    ${CC2_SRCDIR}/cd_cc2_cs.cpp
    ${CC2_SRCDIR}/cd_cc2_os.cpp    
    #${CC2_SRCDIR}/cc2_canonical.cpp
    )

