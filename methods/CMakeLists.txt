cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(ExaChem-Binary VERSION 0.0.0 LANGUAGES CXX)
find_package(EXACHEM REQUIRED)
include(TargetMacros)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
set(EXACHEM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem)
set(EXACHEM_SRCS ${EXACHEM_SRC_DIR}/exachem.cpp)

add_mpi_gpu_unit_test(ExaChem "${EXACHEM_SRCS}" 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")

