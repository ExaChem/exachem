
include(TargetMacros)

set(SCF_SRCDIR scf)

set(SCF_INCLUDES
    ${SCF_SRCDIR}/scf_qed.hpp
    ${SCF_SRCDIR}/scf_vars.hpp
    ${SCF_SRCDIR}/scf_iter.hpp
    ${SCF_SRCDIR}/scf_main.hpp
    ${SCF_SRCDIR}/scf_gauxc.hpp
    ${SCF_SRCDIR}/scf_guess.hpp
    ${SCF_SRCDIR}/scf_common.hpp
    ${SCF_SRCDIR}/scf_compute.hpp
    ${SCF_SRCDIR}/scf_taskmap.hpp
    ${SCF_SRCDIR}/scf_restart.hpp
    ${SCF_SRCDIR}/scf_outputs.hpp
    ${SCF_SRCDIR}/scf_engine.hpp
    ${SCF_SRCDIR}/scf_tamm_tensors.hpp
    ${SCF_SRCDIR}/scf_eigen_tensors.hpp
    )

set(SCF_SRCS
    ${SCF_SRCDIR}/scf_qed.cpp    
    ${SCF_SRCDIR}/scf_iter.cpp        
    ${SCF_SRCDIR}/scf_main.cpp
    ${SCF_SRCDIR}/scf_gauxc.cpp            
    ${SCF_SRCDIR}/scf_guess.cpp    
    ${SCF_SRCDIR}/scf_common.cpp
    ${SCF_SRCDIR}/scf_compute.cpp        
    ${SCF_SRCDIR}/scf_taskmap.cpp
    ${SCF_SRCDIR}/scf_restart.cpp         
    ${SCF_SRCDIR}/scf_outputs.cpp
    ${SCF_SRCDIR}/scf_engine.cpp
    )

