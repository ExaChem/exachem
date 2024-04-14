
include(TargetMacros)

set(COMMON_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/common)
set(COMMON_SRCS
    ${COMMON_SRCDIR}/cutils.cpp
    ${COMMON_SRCDIR}/ecatom.cpp
    ${COMMON_SRCDIR}/fcidump.cpp
    ${COMMON_SRCDIR}/txt_utils.cpp    
    ${COMMON_SRCDIR}/system_data.cpp
    ${COMMON_SRCDIR}/chemenv.cpp
    ${COMMON_SRCDIR}/ec_basis.cpp
    ${COMMON_SRCDIR}/options/parse_options.cpp
    ${COMMON_SRCDIR}/options/parse_common_options.cpp
    ${COMMON_SRCDIR}/options/parse_scf_options.cpp
    ${COMMON_SRCDIR}/options/parse_ccsd_options.cpp    
    ${COMMON_SRCDIR}/options/parse_cd_options.cpp        
    ${COMMON_SRCDIR}/options/parse_fci_options.cpp 
    ${COMMON_SRCDIR}/options/parse_gw_options.cpp                
    ${COMMON_SRCDIR}/options/parse_task_options.cpp  
    ${COMMON_SRCDIR}/options/input_options.cpp    
    ${COMMON_SRCDIR}/initialize_system_data.cpp
    ${COMMON_SRCDIR}/ec_molden.cpp                   
    )

