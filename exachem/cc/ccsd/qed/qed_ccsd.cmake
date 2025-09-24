
include(TargetMacros)

set(QEDCCSD_SRCDIR cc/ccsd/qed)


set(QEDCCSD_OS_RESIDS_SRCS
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_1.cpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_2.cpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_3.cpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_4.cpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_5.cpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_6.cpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_7.cpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_8.cpp
)

set(QEDCCSD_CS_RESIDS_SRCS
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_cs/qed_ccsd_cs_resid_1.cpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_cs/qed_ccsd_cs_resid_2.cpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_cs/qed_ccsd_cs_resid_3.cpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_cs/qed_ccsd_cs_resid_4.cpp
)

set(QEDCCSD_SRCS
    ${QEDCCSD_SRCDIR}/qed_ccsd_cs.cpp
    ${QEDCCSD_SRCDIR}/qed_ccsd_os.cpp
    ${QEDCCSD_OS_RESIDS_SRCS}
    ${QEDCCSD_CS_RESIDS_SRCS}
    )

set(QEDCCSD_OS_RESIDS_INCLUDES
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_tmps.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_1.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_2.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_3.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_4.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_5.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_6.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_7.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_os/qed_ccsd_os_resid_8.hpp
)

set(QEDCCSD_CS_RESIDS_INCLUDES
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_cs/qed_ccsd_cs_tmps.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_cs/qed_ccsd_cs_resid_1.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_cs/qed_ccsd_cs_resid_2.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_cs/qed_ccsd_cs_resid_3.hpp
    ${QEDCCSD_SRCDIR}/residuals/qed_ccsd_cs/qed_ccsd_cs_resid_4.hpp
)

set(QEDCCSD_INCLUDES
    ${QEDCCSD_SRCDIR}/qed_ccsd_cs.hpp
    ${QEDCCSD_SRCDIR}/qed_ccsd_os.hpp
    ${QEDCCSD_OS_RESIDS_INCLUDES}
    ${QEDCCSD_CS_RESIDS_INCLUDES}
)