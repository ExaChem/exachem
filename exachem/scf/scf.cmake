
include(TargetMacros)

set(SCF_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../exachem/scf)
set(SCF_SRCS
    ${SCF_SRCDIR}/scf_iter.cpp
    ${SCF_SRCDIR}/scf_main.cpp    
    ${SCF_SRCDIR}/scf_atom.cpp
    ${SCF_SRCDIR}/scf_guess.cpp
    ${SCF_SRCDIR}/scf_common.cpp
    ${SCF_SRCDIR}/scf_taskmap.cpp
    )

