
include(TargetMacros)

set(SCF_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/scf)
set(SCF_SRCS
    ${SCF_SRCDIR}/scf_iter.cpp        
    ${SCF_SRCDIR}/scf_main.cpp
    ${SCF_SRCDIR}/scf_gauxc.cpp            
    ${SCF_SRCDIR}/scf_guess.cpp    
    ${SCF_SRCDIR}/scf_common.cpp
    ${SCF_SRCDIR}/scf_compute.cpp        
    ${SCF_SRCDIR}/scf_taskmap.cpp
    ${SCF_SRCDIR}/scf_matrix.cpp 
    ${SCF_SRCDIR}/scf_restart.cpp         
    ${SCF_SRCDIR}/scf_outputs.cpp
    ${SCF_SRCDIR}/scf_hartree_fock.cpp    
    )

