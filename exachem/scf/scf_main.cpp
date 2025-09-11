/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_main.hpp"
#include "exachem/common/termcolor.hpp"
#include "exachem/scf/scf_common.hpp"
#include "exachem/scf/scf_engine.hpp"
#include "exachem/scf/scf_outputs.hpp"
#include <string_view>

namespace exachem::scf {

void scf_driver(ExecutionContext& ec, ChemEnv& chem_env) { scf(ec, chem_env); }

void scf(ExecutionContext& ec, ChemEnv& chem_env) {
  const auto rank = ec.pg().rank();

  const auto hf_t1 = std::chrono::high_resolution_clock::now();

  SCFEngine scf_engine(ec, chem_env);
  scf_engine.run(ec, chem_env);

  const auto hf_t2 = std::chrono::high_resolution_clock::now();

  const double hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

  ec.flush_and_sync();

  if(rank == 0) {
    chem_env.sys_data.results["output"]["SCF"]["performance"] = {{"total_time", hf_time}};
    if(chem_env.ioptions.task_options.scf) chem_env.write_json_data();
    std::cout << std::endl
              << "Total Time taken for Hartree-Fock: " << std::fixed << std::setprecision(2)
              << hf_time << " secs" << std::endl;
  }
}
} // namespace exachem::scf
