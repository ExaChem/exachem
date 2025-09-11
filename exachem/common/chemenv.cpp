/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/chemenv.hpp"

int ChemEnv::get_nfcore() {
  if(ioptions.ccsd_options.freeze_atomic) {
    int nfcore = 0;
    for(size_t i = 0; i < atoms.size(); ++i) {
      const auto Z = atoms[i].atomic_number;
      if(Z >= 3 && Z <= 10) nfcore += 1;
      else if(Z >= 11 && Z <= 18) nfcore += 5;
      else if(Z >= 19 && Z <= 36) nfcore += 9;
      else if(Z >= 37 && Z <= 54) nfcore += 18;
      else if(Z >= 55 && Z <= 86) nfcore += 27;
      else if(Z >= 87 && Z <= 118) nfcore += 43;
    }
    return nfcore;
  }

  return ioptions.ccsd_options.freeze_core;
}

void ChemEnv::read_run_context() {
  std::string files_prefix = get_files_prefix();
  std::string json_file    = files_prefix + ".runcontext.json";
  bool        json_exists  = std::filesystem::exists(json_file);
  if(json_exists) {
    std::ifstream jread(json_file);
    jread >> run_context;
  }
  // else {
  //   std::string err_msg = "\n[ERROR] " + json_file + " does not exist!"
  //   tamm_terminate(err_msg);
  // }
}

void ChemEnv::write_run_context() {
  std::string files_dir = get_files_dir();
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);
  std::string files_prefix = get_files_prefix();
  std::string json_file    = files_prefix + ".runcontext.json";
  bool        json_exists  = std::filesystem::exists(json_file);
  if(json_exists) {
    // std::ifstream jread(json_file);
    // jread >> run_context;
    std::filesystem::remove(json_file);
  }

  // std::cout << std::endl << std::endl << run_context.dump() << std::endl;
  std::ofstream res_file(json_file);
  res_file << std::setw(2) << run_context << std::endl;
}

void ChemEnv::write_sinfo() {
  std::string basis = ioptions.scf_options.basis;

  std::string out_fp       = workspace_dir;
  std::string files_dir    = out_fp + ioptions.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + sys_data.output_file_prefix;
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);

  json& results = sys_data.results;

  results["molecule"]["name"]  = sys_data.input_molecule;
  results["molecule"]["basis"] = jinput["basis"];

  // for(size_t i = 0; i < atoms.size(); i++) {
  //   ECAtom& iatom = ec_atoms[i];
  //   if(iatom.basis != basis) results["molecule"]["basis"][iatom.esymbol] = iatom.basis;
  // }

  results["molecule"]["nbf"]              = shells.nbf();
  results["molecule"]["nshells"]          = shells.size();
  results["molecule"]["nelectrons"]       = sys_data.nelectrons;
  results["molecule"]["nelectrons_alpha"] = sys_data.nelectrons_alpha;
  results["molecule"]["nelectrons_beta"]  = sys_data.nelectrons_beta;

  std::string json_file   = files_prefix + ".sinfo.json";
  bool        json_exists = std::filesystem::exists(json_file);
  if(json_exists) std::filesystem::remove(json_file);

  std::ofstream res_file(json_file);
  res_file << std::setw(2) << results << std::endl;
}

void ChemEnv::write_json_data() {
  json& results = sys_data.results;

  results["input"] = jinput;

  std::string l_module = task_string; // cmodule
  txt_utils::to_lower(l_module);

  std::string files_dir = workspace_dir + ioptions.scf_options.scf_type + "/json";
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);
  std::string files_prefix = files_dir + "/" + sys_data.output_file_prefix;
  std::string json_file    = files_prefix + "." + l_module + ".json";
  bool        json_exists  = std::filesystem::exists(json_file);
  if(json_exists) {
    // std::ifstream jread(json_file);
    // jread >> results;
    std::filesystem::remove(json_file);
  }

  // std::cout << std::endl << std::endl << results.dump() << std::endl;
  std::ofstream res_file(json_file);
  res_file << std::setw(2) << results << std::endl;
}

void ChemEnv::sinfo() {
  SCFOptions& scf_options = ioptions.scf_options;

  // TODO: dont create ec
  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  auto             rank   = ec.pg().rank();
  std::string      basis  = scf_options.basis;
  int              charge = scf_options.charge;

  const int N = shells.nbf();

  auto nelectrons = 0;
  for(size_t i = 0; i < atoms.size(); ++i) nelectrons += atoms[i].atomic_number;
  nelectrons -= charge;

  // sys_data.nelectrons = nelectrons;
  if((nelectrons + scf_options.multiplicity - 1) % 2 != 0) {
    std::string err_msg = "[ERROR] Number of electrons (" + std::to_string(nelectrons) + ") " +
                          "and multiplicity (" + std::to_string(scf_options.multiplicity) + ") " +
                          " not compatible!";
    tamm_terminate(err_msg);
  }

  int nelectrons_alpha = (nelectrons + scf_options.multiplicity - 1) / 2;
  int nelectrons_beta  = nelectrons - nelectrons_alpha;

  if(rank == 0) {
    std::cout << std::endl << "Number of basis functions = " << N << std::endl;
    std::cout << std::endl << "Total number of shells = " << shells.size() << std::endl;
    std::cout << std::endl << "Total number of electrons = " << nelectrons << std::endl;
    std::cout << "  # of alpha electrons    = " << nelectrons_alpha << std::endl;
    std::cout << "  # of beta electons      = " << nelectrons_beta << std::endl << std::flush;
  }

  if(rank == 0) write_sinfo();

  ec.flush_and_sync();
}

Matrix ChemEnv::compute_shellblock_norm(const libint2::BasisSet& obs, const Matrix& A) const {
  const auto nsh = obs.size();
  Matrix     Ash = Matrix::Zero(nsh, nsh);

  auto shell2bf = obs.shell2bf();
  for(size_t s1 = 0; s1 != nsh; ++s1) {
    const auto& s1_first = shell2bf[s1];
    const auto& s1_size  = obs[s1].size();
    for(size_t s2 = 0; s2 != nsh; ++s2) {
      const auto& s2_first = shell2bf[s2];
      const auto& s2_size  = obs[s2].size();

      Ash(s1, s2) = A.block(s1_first, s2_first, s1_size, s2_size).lpNorm<Eigen::Infinity>();
    }
  }

  return Ash;
}
