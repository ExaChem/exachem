
include(TargetMacros)

set(CC_EOM_SRCDIR cc/eom)
set(CC_EOM_SRCS
    ${CC_EOM_SRCDIR}/eomccsd_opt.cpp    
    ${CC_EOM_SRCDIR}/eomccsd_driver.cpp
    # ${CC_EOM_SRCDIR}/eomccsd.cpp
    # ${CC_EOM_SRCDIR}/eom_gradients.cpp
    )

set(CC_EOM_INCLUDES
    ${CC_EOM_SRCDIR}/eomccsd_opt.hpp
    ${CC_EOM_SRCDIR}/eomguess_opt.hpp
    )

