cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(ExaChem-Binary VERSION 0.0.0 LANGUAGES CXX)
find_package(EXACHEM REQUIRED)
include(TargetMacros)

add_mpi_unit_test(ExaChem 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")

