
include(TargetMacros)

set(CC_LAMBDA_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/cc/lambda)
set(CC_LAMBDA_SRCS
    ${CC_LAMBDA_SRCDIR}/ccsd_rdm.cpp    
    ${CC_LAMBDA_SRCDIR}/ccsd_lambda.cpp
    ${CC_LAMBDA_SRCDIR}/ccsd_lambda_driver.cpp
    ${CC_LAMBDA_SRCDIR}/ccsd_natural_orbitals.cpp
    )

