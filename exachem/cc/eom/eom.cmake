
include(TargetMacros)

set(CC_EOM_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/cc/eom)
set(CC_EOM_SRCS
    ${CC_EOM_SRCDIR}/eomccsd_opt.cpp    
    ${CC_EOM_SRCDIR}/eomccsd_driver.cpp
    # ${CC_EOM_SRCDIR}/eomccsd.cpp
    # ${CC_EOM_SRCDIR}/eom_gradients.cpp
    )

