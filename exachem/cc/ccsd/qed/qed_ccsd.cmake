
include(TargetMacros)

set(QEDCCSD_SRCDIR cc/ccsd/qed)


set(CD_QEDCCSD_CS_RESIDS_SRCS
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_cs/cd_qed_ccsd_cs_resid_1.cpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_cs/cd_qed_ccsd_cs_resid_2.cpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_cs/cd_qed_ccsd_cs_resid_3.cpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_cs/cd_qed_ccsd_cs_resid_4.cpp
)
set(CD_QEDCCSD_OS_RESIDS_SRCS
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_1.cpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_2.cpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_3.cpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_4.cpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_5.cpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_6.cpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_7.cpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_8.cpp
)

set(QEDCCSD_SRCS
    ${QEDCCSD_SRCDIR}/cd_qed_ccsd_cs.cpp
    ${QEDCCSD_SRCDIR}/cd_qed_ccsd_os.cpp
    ${CD_QEDCCSD_CS_RESIDS_SRCS}
    ${CD_QEDCCSD_OS_RESIDS_SRCS}
    )

set(CD_QEDCCSD_CS_RESIDS_INCLUDES
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_cs/cd_qed_ccsd_cs_tmps.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_cs/cd_qed_ccsd_cs_resid_1.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_cs/cd_qed_ccsd_cs_resid_2.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_cs/cd_qed_ccsd_cs_resid_3.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_cs/cd_qed_ccsd_cs_resid_4.hpp
)
set(CD_QEDCCSD_OS_RESIDS_INCLUDES
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_tmps.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_1.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_2.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_3.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_4.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_5.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_6.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_7.hpp
    ${QEDCCSD_SRCDIR}/residuals/cd_qed_ccsd_os/cd_qed_ccsd_os_resid_8.hpp
)

set(QEDCCSD_INCLUDES
    ${QEDCCSD_SRCDIR}/cd_qed_ccsd_cs.hpp
    ${QEDCCSD_SRCDIR}/cd_qed_ccsd_os.hpp
    ${CD_QEDCCSD_CS_RESIDS_INCLUDES}
    ${CD_QEDCCSD_OS_RESIDS_INCLUDES}
)
