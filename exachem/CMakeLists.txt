cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(EXACHEM VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros)

set(EXACHEM_CFLAGS )
set(EXACHEM_LFLAGS )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# include(${CMAKE_CURRENT_SOURCE_DIR}/scf/scf.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/common/common.cmake)
# include(${CMAKE_CURRENT_SOURCE_DIR}/mp2/mp2.cmake)
# include(${CMAKE_CURRENT_SOURCE_DIR}/cc/cd.cmake)

#Add the current directory's sources to the list
set(EXACHEM_SRCS ${SCF_SRCS} ${COMMON_SRCS} ${MP2_SRCS} ${CD_SRCS})

#Add the current directory's header files to the list
set(EXACHEM_INCLUDES ${SCF_INCLUDES} ${COMMON_INCLUDES} ${CD_INCLUDES})

cmsb_add_library(exachem EXACHEM_SRCS EXACHEM_INCLUDES EXACHEM_CFLAGS EXACHEM_LFLAGS)
target_include_directories(exachem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


