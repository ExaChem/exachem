cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(ExaChem-Binary VERSION 0.0.0 LANGUAGES CXX)
find_package(EXACHEM REQUIRED)
include(TargetMacros)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
set(EXACHEM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem)

option(EC_COMPLEX "enable complex codes" ON)

#TODO: cleanup
include_directories(${EXACHEM_SRC_DIR})

list(FIND MODULES "CC" _cc_index)

if(NOT _cc_index EQUAL -1)
  message(STATUS "Compiling CC modules")
  include(${EXACHEM_SRC_DIR}/cc/eom/eom.cmake)
  include(${EXACHEM_SRC_DIR}/cc/ccsd_t/ccsd_t.cmake)
  include(${EXACHEM_SRC_DIR}/cc/lambda/lambda.cmake)
  include(${EXACHEM_SRC_DIR}/cc/ducc/ducc.cmake)

  if(NOT USE_UPCXX AND EC_COMPLEX)
    include(${EXACHEM_SRC_DIR}/cc/gfcc/gfcc.cmake)
    include(${EXACHEM_SRC_DIR}/cc/rteom/rteom.cmake)
  endif()

  set(CoupledCluster_SRCS ${CCSDT_SRCS} ${CCSD_T_SRCS} 
  ${CC_LAMBDA_SRCS} ${CC_EOM_SRCS} ${GFCC_SRCS} ${RTEOM_SRCS} ${DUCC_SRCS})

endif()

set(EXACHEM_SRCS ${CoupledCluster_SRCS})

add_mpi_gpu_unit_test(ExaChem "${EXACHEM_SRCS}" 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")

