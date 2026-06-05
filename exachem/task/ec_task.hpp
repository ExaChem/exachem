/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#define EC_COMPLEX

// clang-format off
#if defined(ENABLE_CC)
#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"
#include "exachem/cc/ccsd/canonical/ccsd_canonical.hpp"
#include "exachem/cc/ccsd/canonical/ccsd_cs.hpp"
#include "exachem/cc/ccsd/canonical/ccsd_os.hpp"
// #include "exachem/cc/ccsd/qed/qed_ccsd_cs.hpp"
#include "exachem/cc/ccsd/qed/cd_qed_ccsd_cs.hpp"
#include "exachem/cc/ccsd/qed/qed_ccsd_os.hpp"
#include "exachem/cc/ccsd_t/ccsd_t_fused_driver.hpp"
#include "exachem/cc/lambda/ccsd_lambda.hpp"
#include "exachem/cc/eom/eomccsd_opt.hpp"
#include "exachem/cc/ducc/ducc-t_ccsd.hpp"
#include "exachem/cc/cc2/cd_cc2.hpp"
#endif

#include "exachem/task/task_interface.hpp"
#include "exachem/common/chemenv.hpp"
#include "exachem/common/options/parse_options.hpp"
#include "exachem/scf/scf_main.hpp"
#include "exachem/mp2/cd_mp2.hpp"
#include "exachem/gradients/ec_gradients.hpp"

#if defined(TAMM_USE_PYTHON)
#include "exachem/optimizers/geometric.hpp"
#endif
#include "exachem/optimizers/pyberny_impl.hpp"
// clang-format on

using namespace exachem;

#if defined(EC_COMPLEX) and defined(ENABLE_CC)
#include "exachem/cc/gfcc/gfccsd.hpp"
#include "exachem/cc/rteom/rt_eom_cd_ccsd.hpp"
#include "exachem/fci/fci.hpp"
#endif
