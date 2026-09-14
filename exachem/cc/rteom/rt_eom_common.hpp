/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cholesky/cholesky_2e_driver.hpp"
#include "exachem/common/chemenv.hpp"
#include "exachem/common/initialize_system_data.hpp"
#include <fstream>

namespace exachem::rteom_cc {

// RT-EOMCC describes a core-hole state: the closed-shell (RHF) reference with one beta electron
// removed from orbital `pcore`, keeping the reference orbitals. The input describes only the
// reference SCF; this runs it, then rewrites the SCF options for the core-hole (N-1, unrestricted)
// system and stages the reference orbitals so that the second SCF pass (noscf) only sets up the
// environment.
// Returns the reference SCF results (json), to be put back into sys_data.results["output"]["SCF"]
// once the core-hole SCF pass has run.
inline json rteom_reference_scf(ExecutionContext& ec, ChemEnv& chem_env) {
  const auto   rank         = ec.pg().rank();
  SCFOptions&  scf_options  = chem_env.ioptions.scf_options;
  CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;

  if(!chem_env.sys_data.is_restricted)
    tamm_terminate("[RT-EOMCC] scf_type must be restricted: the input describes the closed-shell "
                   "reference; the core-hole (charge+1, multiplicity 2, unrestricted) SCF settings "
                   "are derived automatically");
  if(ccsd_options.pcore <= 0)
    tamm_terminate("[RT-EOMCC] pcore (1-based index of the core orbital) must be > 0");

  // reference SCF movecs/density and the MSO fock/movecs written by two_index_transform
  const std::string src_prefix = chem_env.get_files_prefix("restricted", "scf");
  const std::string ref_prefix = chem_env.get_files_prefix("restricted");
  const std::string ref_f1file = ref_prefix + ".td.f1_mo";
  const std::string ref_c1file = ref_prefix + ".td.movecs_so";

  // restart: everything the reference phase produces already exists; skip it (same rule as the
  // cholesky and CC restarts).
  bool restart = false;
  if(rank == 0)
    restart = (ccsd_options.readt || ccsd_options.writet) && fs::exists(ref_f1file) &&
              fs::exists(ref_c1file) && fs::exists(src_prefix + ".alpha.movecs") &&
              fs::exists(src_prefix + ".alpha.density");
  ec.pg().broadcast(&restart, 0);

  // reference SCF record: <workspace>/restricted/json/<prefix>.<task>.json
  const std::string ref_json_file = chem_env.get_files_prefix("restricted", "json") + "." +
                                    txt_utils::to_lower(chem_env.task_string) + ".json";
  json ref_scf_results;

  if(!restart) {
    // phase 1: reference SCF; cholesky_2e_driver returns right after two_index_transform has
    // written the reference fock and movecs (MSO basis) to disk.
    cholesky_2e::cholesky_2e_driver(ec, chem_env);

    ref_scf_results = chem_env.sys_data.results["output"]["SCF"];
    if(rank == 0) chem_env.write_json_data();
  }
  else {
    if(rank == 0)
      std::cout << std::endl
                << "[RT-EOMCC] restart: reference fock and movecs found on disk, skipping the "
                   "reference SCF"
                << std::endl;
    if(fs::exists(ref_json_file)) {
      std::ifstream jread(ref_json_file);
      json          ref_results;
      jread >> ref_results;
      ref_scf_results = ref_results["output"]["SCF"];
    }
    else if(rank == 0)
      std::cout << "[RT-EOMCC] warning: reference SCF record " << ref_json_file << " not found"
                << std::endl;
  }

  // the core-hole SCF pass restarts (noscf) from the reference orbitals: stage the RHF movecs and
  // density as both alpha and beta under the unrestricted SCF directory.
  if(rank == 0) {
    const std::string dst_prefix = chem_env.get_files_prefix("unrestricted", "scf");
    fs::create_directories(chem_env.get_files_dir("unrestricted", "scf"));
    for(const std::string ext: {".movecs", ".density"}) {
      const std::string src = src_prefix + ".alpha" + ext;
      for(const std::string spin: {".alpha", ".beta"})
        fs::copy_file(src, dst_prefix + spin + ext, fs::copy_options::overwrite_existing);
    }
  }
  ec.pg().barrier();

  // phase 2 settings: core-hole system, environment setup only
  scf_options.charge += 1;
  scf_options.multiplicity = 2;
  scf_options.scf_type     = "unrestricted";
  scf_options.noscf        = true;
  IniSystemData ini_sys_data(chem_env); // sets is_unrestricted, focc, ...

  // the run context (tilesizes, num_chol_vecs, ...) of the core-hole phase lives under the
  // unrestricted directory; on a restart pick it up from the previous run.
  if(restart) chem_env.read_run_context();

  if(rank == 0)
    std::cout << std::endl
              << "[RT-EOMCC] core-hole system: charge = " << scf_options.charge
              << ", multiplicity = " << scf_options.multiplicity
              << ", scf_type = unrestricted (noscf)" << std::endl;

  return ref_scf_results;
}

} // namespace exachem::rteom_cc
