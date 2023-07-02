cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(ExaChem-Binary VERSION 0.0.0 LANGUAGES CXX)
find_package(EXACHEM REQUIRED)
include(TargetMacros)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
#TODO: cleanup
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../exachem)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../exachem/util/external)
include(${CMAKE_CURRENT_SOURCE_DIR}/../exachem/cc/ccsd_t/ccsd_t.cmake)

add_mpi_gpu_unit_test(ExaChem "${CCSD_T_FUSED_SRCS}" 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")



