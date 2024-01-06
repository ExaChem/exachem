cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(EXACHEM VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros)

#Add the current directory's sources to the list
set(EXACHEM_SRCS exachem.cpp)

#Add the current directory's header files to the list
set(EXACHEM_INCLUDES exachem.hpp)

set(EXACHEM_CFLAGS )
set(EXACHEM_LFLAGS )

cmsb_add_library(exachem EXACHEM_SRCS EXACHEM_INCLUDES EXACHEM_CFLAGS EXACHEM_LFLAGS)
target_include_directories(exachem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


