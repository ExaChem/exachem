
include(TargetMacros)

set(CCSD_T_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/cc/ccsd_t)
set(CCSD_T_COMMON_SRCS
    ${CCSD_T_SRCDIR}/ccsd_t.cpp    
    ${CCSD_T_SRCDIR}/memory.cpp
    ${CCSD_T_SRCDIR}/ccsd_t_common.hpp
    ${CCSD_T_SRCDIR}/hybrid.cpp
    ${CCSD_T_SRCDIR}/ccsd_t_fused_driver.hpp
    ${CCSD_T_SRCDIR}/fused_common.hpp
    )

if(TAMM_HAS_CUDA)
    set(CCSD_T_SRCS ${CCSD_T_COMMON_SRCS}
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused.hpp
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_gpu.cu
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_nontcCuda_Hip_Sycl.cpp)

    set_source_files_properties(${CCSD_T_SRCDIR}/ccsd_t_all_fused_nontcCuda_Hip_Sycl.cpp PROPERTIES LANGUAGE CUDA)
elseif(TAMM_HAS_HIP)
    set(CCSD_T_SRCS ${CCSD_T_COMMON_SRCS}
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused.hpp
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_nontcCuda_Hip_Sycl.cpp)

    set_source_files_properties(${CCSD_T_SRCDIR}/ccsd_t_all_fused_nontcCuda_Hip_Sycl.cpp PROPERTIES LANGUAGE HIP)
elseif(TAMM_HAS_DPCPP)
    set(CCSD_T_SRCS ${CCSD_T_COMMON_SRCS}
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused.hpp
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_nontcCuda_Hip_Sycl.cpp)
else()
    set(CCSD_T_SRCS ${CCSD_T_COMMON_SRCS}
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_cpu.hpp)
endif()

