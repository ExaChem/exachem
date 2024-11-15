cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(EXACHEM VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros)

set(EXACHEM_CFLAGS )
set(EXACHEM_LFLAGS )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(${CMAKE_CURRENT_SOURCE_DIR}/scf/scf.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/common/common.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/mp2/mp2.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cholesky/cholesky.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/task/task.cmake)

option(EC_COMPLEX "enable complex codes" ON)

list(FIND MODULES "CC" _cc_index)

if(NOT _cc_index EQUAL -1)
  message(STATUS "Compiling CC modules")

  include(${CMAKE_CURRENT_SOURCE_DIR}/fci/fci.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cc/cc.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cc/eom/eom.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cc/lambda/lambda.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cc/ducc/ducc.cmake)
  
  if(EC_COMPLEX)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cc/gfcc/gfcc.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cc/rteom/rteom.cmake)
  endif()
  
endif()

#Add the various modules sources to the list
set(EXACHEM_SRCS ${SCF_SRCS} ${COMMON_SRCS} ${MP2_SRCS} ${CD_SRCS} ${CC2_SRCS} ${CCSD_SRCS} ${CCSD_T_SRCS} ${FCI_SRCS}
${CCSDT_SRCS} ${CC_LAMBDA_SRCS} ${DUCC_SRCS} ${CC_EOM_SRCS} ${RTEOM_SRCS} ${GFCC_SRCS} ${TASK_SRCS})

#Add the various modules header files to the list
set(EXACHEM_INCLUDES ${SCF_INCLUDES} ${COMMON_INCLUDES} ${CD_INCLUDES} ${MP2_INCLUDES} ${DUCC_INCLUDES}
    ${CC_INCLUDES} ${CC2_INCLUDES} ${CCSD_INCLUDES} ${CCSD_T_INCLUDES} ${CC_EOM_INCLUDES} ${FCI_INCLUDES}
    ${CC_LAMBDA_INCLUDES} ${RTEOM_INCLUDES} ${GFCC_INCLUDES} ${CCSDT_INCLUDES} ${TASK_INCLUDES})

cmsb_add_library(exachem EXACHEM_SRCS EXACHEM_INCLUDES EXACHEM_CFLAGS EXACHEM_LFLAGS)
target_include_directories(exachem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


