
include(TargetMacros)

set(CC2_SRCDIR cc/cc2)
set(CC2_SRCS
    ${CC2_SRCDIR}/cd_cc2.cpp    
    ${CC2_SRCDIR}/cd_cc2_cs.cpp
    ${CC2_SRCDIR}/cd_cc2_os.cpp    
    #${CC2_SRCDIR}/cc2_canonical.cpp
    )

set(CC2_INCLUDES
    ${CC2_SRCDIR}/cd_cc2_cs.hpp
    ${CC2_SRCDIR}/cd_cc2_os.hpp
)
