
include(TargetMacros)

set(INTEGRATIONS_SRCDIR integrations)

set(INTEGRATIONS_INCLUDES
    ${INTEGRATIONS_SRCDIR}/ipi/sockets.hpp
    ${INTEGRATIONS_SRCDIR}/ipi/ipi_backend.hpp
    ${INTEGRATIONS_SRCDIR}/ase/ase_backend.hpp
    )

set(INTEGRATIONS_SRCS
    ${INTEGRATIONS_SRCDIR}/ipi/sockets.cpp
    ${INTEGRATIONS_SRCDIR}/ipi/ipi_backend.cpp
    ${INTEGRATIONS_SRCDIR}/ase/ase_backend.cpp
   )
