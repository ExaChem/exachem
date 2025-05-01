
include(TargetMacros)

set(GFCC_SRCDIR cc/gfcc)
set(GFCC_SRCS
    ${GFCC_SRCDIR}/gfccsd_driver.cpp
    ${GFCC_SRCDIR}/gfccsd_ip_a.cpp
    ${GFCC_SRCDIR}/gfccsd_ip_b.cpp
    ${GFCC_SRCDIR}/gfccsd_ea_a.cpp
    ${GFCC_SRCDIR}/gfccsd_ea_b.cpp
    )
set(GFCC_INCLUDES
    ${GFCC_SRCDIR}/gfccsd.hpp
    )

