/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/cc2/cd_cc2.hpp"
#include "exachem/cc/ccsd/cd_ccsd.hpp"

#include <filesystem>
namespace fs = std::filesystem;

namespace exachem::cc2 {
void cd_cc2_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  // does not suffice to just check input task cc2 since multiple tasks can be specified.
  chem_env.cc_context.task_cc2 = true;
  exachem::cc::ccsd::cd_ccsd_driver(ec, chem_env);
}
} // namespace exachem::cc2
