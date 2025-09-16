if(CMAKE_CXX_COMPILER_ID STREQUAL "XL"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "Cray"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel" 
    OR CMAKE_CXX_COMPILER_ID STREQUAL "PGI"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        message(FATAL_ERROR "ExaChem cannot be currently built with ${CMAKE_CXX_COMPILER_ID} compilers.")
endif()

if (DEFINED ENV{CONDA_PREFIX}) #VIRTUAL_ENV
  message(FATAL_ERROR "ExaChem cannot be currently built with CONDA. \
          Please deactivate CONDA environment.")
endif()

if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
    if (USE_CUDA)
        message(FATAL_ERROR "ExaChem does not support building with GPU support \
        on MACOSX. Please use -DUSE_CUDA=OFF for MACOSX builds.")
    endif()
    
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" 
        OR CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
        message(FATAL_ERROR "ExaChem does not support ${CMAKE_CXX_COMPILER_ID} compilers on MACOSX.")
    endif()
endif()

macro(get_compiler_exec_name comp_exec_path)
    get_filename_component(comp_exec_name ${comp_exec_path} NAME_WE)
endmacro()

macro(check_compiler_version lang_arg comp_type comp_version)
    if(CMAKE_${lang_arg}_COMPILER_ID STREQUAL "${comp_type}")
        if(CMAKE_${lang_arg}_COMPILER_VERSION VERSION_LESS "${comp_version}")
            get_compiler_exec_name("${CMAKE_${lang_arg}_COMPILER}")
            message(FATAL_ERROR "${comp_exec_name} version provided (${CMAKE_${lang_arg}_COMPILER_VERSION}) \
            is insufficient. Need ${comp_exec_name} >= ${comp_version} for building ExaChem.")
        endif()
    endif()
endmacro()

set(GA_RUNTIME_TAMM MPI_RMA OPENIB MPI-PR MPI-TS MPI_2SIDED MPI_PROGRESS_RANK)
if(DEFINED GA_RUNTIME)
    list(FIND GA_RUNTIME_TAMM ${GA_RUNTIME} _index)
    if(${_index} EQUAL -1)
        message(FATAL_ERROR "ExaChem only supports building GA using one of ${GA_RUNTIME_TAMM}, default is MPI-PR")
    endif()
endif()

check_compiler_version(C Clang 9)
check_compiler_version(CXX Clang 9)

check_compiler_version(C AppleClang 15)
check_compiler_version(CXX AppleClang 15)

check_compiler_version(C GNU 9.1)
check_compiler_version(CXX GNU 9.1)
check_compiler_version(Fortran GNU 9.1)

#TODO:Check for GCC>=9 compatibility
# check_compiler_version(C Intel 19)
# check_compiler_version(CXX Intel 19)
# check_compiler_version(Fortran Intel 19)

#TODO:Check for GCC>=9 compatibility
check_compiler_version(C PGI 20)
check_compiler_version(CXX PGI 20)
check_compiler_version(Fortran PGI 20)

find_package(MPI REQUIRED)

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_C_COMPILER_VERSION VERSION_EQUAL CMAKE_CXX_COMPILER_VERSION
    AND CMAKE_C_COMPILER_VERSION VERSION_EQUAL CMAKE_Fortran_COMPILER_VERSION)
    message(STATUS "Check GNU compiler versions.")
  else()
    message(STATUS "GNU C,CXX,Fortran compiler versions do not match")
    message(FATAL_ERROR "GNU Compiler versions provided: gcc: ${CMAKE_C_COMPILER_VERSION}, 
    g++: ${CMAKE_CXX_COMPILER_VERSION}, gfortran version: ${CMAKE_Fortran_COMPILER_VERSION}")
  endif()
endif()

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_ARCHITECTURES)
        set(GPU_ARCH ${CMAKE_CUDA_ARCHITECTURES} CACHE STRING "CUDA ARCH" FORCE)
    elseif(GPU_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH} CACHE STRING "GPU ARCH" FORCE)
    else()
        message(FATAL_ERROR "One of CMAKE_CUDA_ARCHITECTURES or GPU_ARCH options needs to be provided")
    endif()

    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
            message(STATUS "CUDA Architecture provided: ${GPU_ARCH}")
    else()
        message(FATAL_ERROR "CUDA Toolkit not found.")
    endif()

    set(_CUDA_MIN "11.7")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${_CUDA_MIN})
        message(FATAL_ERROR "CUDA version provided \
        (${CMAKE_CUDA_COMPILER_VERSION}) \
        is insufficient. Need CUDA >= ${_CUDA_MIN})")
    endif()
    
endif()


