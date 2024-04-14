#include "scf/scf_main.hpp"
#include "common/termcolor.hpp"
#include "scf/scf_common.hpp"
#include "scf/scf_hartree_fock.hpp"
#include "scf/scf_outputs.hpp"
#include <string_view>

void scf(ExecutionContext& ec, ChemEnv& chem_env) {
  auto rank = ec.pg().rank();

  auto hf_t1 = std::chrono::high_resolution_clock::now();

  SCFHartreeFock SCFHF(ec, chem_env);

  if(rank == 0 && chem_env.ioptions.task_options.scf) chem_env.write_json_data("SCF");

  auto hf_t2 = std::chrono::high_resolution_clock::now();

  double hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

  ec.flush_and_sync();

  if(rank == 0)
    std::cout << std::endl
              << "Total Time taken for Hartree-Fock: " << std::fixed << std::setprecision(2)
              << hf_time << " secs" << std::endl;
}
