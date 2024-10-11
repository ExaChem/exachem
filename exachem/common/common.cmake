
include(TargetMacros)

set(COMMON_SRCDIR common)
set(COMMON_INCLUDES
    ${COMMON_SRCDIR}/cutils.hpp
    ${COMMON_SRCDIR}/ecatom.hpp
    ${COMMON_SRCDIR}/fcidump.hpp
    ${COMMON_SRCDIR}/atom_info.hpp
    ${COMMON_SRCDIR}/txt_utils.hpp
    ${COMMON_SRCDIR}/system_data.hpp
    ${COMMON_SRCDIR}/chemenv.hpp
    ${COMMON_SRCDIR}/ec_basis.hpp
    ${COMMON_SRCDIR}/ec_dplot.hpp
    ${COMMON_SRCDIR}/ec_molden.hpp
    ${COMMON_SRCDIR}/libint2_includes.hpp
    ${COMMON_SRCDIR}/context/cd_context.hpp
    ${COMMON_SRCDIR}/context/cc_context.hpp
    ${COMMON_SRCDIR}/options/parse_options.hpp
    ${COMMON_SRCDIR}/options/parse_common_options.hpp
    ${COMMON_SRCDIR}/options/parse_scf_options.hpp
    ${COMMON_SRCDIR}/options/parse_ccsd_options.hpp    
    ${COMMON_SRCDIR}/options/parse_cd_options.hpp        
    ${COMMON_SRCDIR}/options/parse_fci_options.hpp 
    ${COMMON_SRCDIR}/options/parse_gw_options.hpp                
    ${COMMON_SRCDIR}/options/parse_task_options.hpp  
    ${COMMON_SRCDIR}/options/input_options.hpp
    ${COMMON_SRCDIR}/options/parser_utils.hpp    
    ${COMMON_SRCDIR}/initialize_system_data.hpp
)

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

