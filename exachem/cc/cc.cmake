include(TargetMacros)

include(${CMAKE_CURRENT_SOURCE_DIR}/cc/cc2/cc2.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cc/ccsd/ccsd.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cc/ccsd_t/ccsd_t.cmake)

set(CC_INCLUDES
    cc/diis.hpp
    cc/ccse_tensors.hpp
)
