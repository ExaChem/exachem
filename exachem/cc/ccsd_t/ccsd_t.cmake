
include(TargetMacros)

set(CCSD_T_SRCDIR cc/ccsd_t)

set(CCSD_T_INCLUDES
    ${CCSD_T_SRCDIR}/ccsd_t_common.hpp
    ${CCSD_T_SRCDIR}/ccsd_t_fused_driver.hpp
    ${CCSD_T_SRCDIR}/fused_common.hpp
    ${CCSD_T_SRCDIR}/ccsd_t.hpp
    ${CCSD_T_SRCDIR}/ccsd_t_all_fused_singles.hpp
    ${CCSD_T_SRCDIR}/ccsd_t_all_fused_doubles1.hpp
    ${CCSD_T_SRCDIR}/ccsd_t_all_fused_doubles2.hpp
)

if(EXACHEM_HAS_CUDA OR EXACHEM_HAS_HIP OR EXACHEM_HAS_DPCPP)
    list(APPEND CCSD_T_INCLUDES ${CCSD_T_SRCDIR}/ccsd_t_all_fused.hpp)
else()
    list(APPEND CCSD_T_INCLUDES ${CCSD_T_SRCDIR}/ccsd_t_all_fused_cpu.hpp)
endif()

set(CCSD_T_COMMON_SRCS
    ${CCSD_T_SRCDIR}/ccsd_t.cpp    
    ${CCSD_T_SRCDIR}/hybrid.cpp
    )

if(EXACHEM_HAS_CUDA)
    set(CCSD_T_SRCS ${CCSD_T_COMMON_SRCS}
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_gpu.cu
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_nontcCuda_Hip_Sycl.cpp)

    set_source_files_properties(${CCSD_T_SRCDIR}/ccsd_t_all_fused_nontcCuda_Hip_Sycl.cpp PROPERTIES LANGUAGE CUDA)
elseif(EXACHEM_HAS_HIP)
    set(CCSD_T_SRCS ${CCSD_T_COMMON_SRCS}
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_nontcCuda_Hip_Sycl.cpp)

    set_source_files_properties(${CCSD_T_SRCDIR}/ccsd_t_all_fused_nontcCuda_Hip_Sycl.cpp PROPERTIES LANGUAGE HIP)
elseif(EXACHEM_HAS_DPCPP)
    set(CCSD_T_SRCS ${CCSD_T_COMMON_SRCS}
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_nontcCuda_Hip_Sycl.cpp)
else()
    set(CCSD_T_SRCS ${CCSD_T_COMMON_SRCS})
endif()

