
include(TargetMacros)

set(CD_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/cc)
set(CD_INCLUDES
    ${CD_SRCDIR}/cd_svd/cd_svd.hpp
    )
set(CD_SRCS
    ${CD_SRCDIR}/cd_svd/cd_svd.cpp
    )

