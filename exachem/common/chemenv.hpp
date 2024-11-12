/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "exachem/common/ec_basis.hpp"
#include "exachem/common/ecatom.hpp"
#include "exachem/common/system_data.hpp"
#include "options/input_options.hpp"
// #include "libint2_includes.hpp"
#include "exachem/common/context/cc_context.hpp"
#include "exachem/common/context/cd_context.hpp"
#include "exachem/common/context/is_context.hpp"
#include "exachem/common/context/scf_context.hpp"
#include "exachem/common/txt_utils.hpp"

#include "tamm/tamm.hpp"
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

class ChemEnv {
public:
  using TensorType = double;

  void write_sinfo();
  // void write_json_data(const std::string cmodule);
  void sinfo();
  int  get_nfcore();

  ChemEnv() = default;

  json        jinput; //!< container for the parsed input data.
  std::string input_file;
  SystemData  sys_data;
  ECOptions   ioptions;
  ECBasis     ec_basis;
  json        run_context;

  std::vector<Atom>   atoms;
  std::vector<ECAtom> ec_atoms;
  libint2::BasisSet   shells;

  std::string workspace_dir{};

  SCFContext scf_context;
  CDContext  cd_context;
  CCContext  cc_context;
  ISContext  is_context; // IndexSpaces

  void read_run_context();
  void write_run_context();
  void write_json_data(const std::string cmodule);

  Matrix compute_shellblock_norm(const libint2::BasisSet& obs, const Matrix& A);

  std::string get_files_dir(std::string scf_type = "") {
    if(scf_type.empty()) scf_type = ioptions.scf_options.scf_type;
    const std::string files_dir = workspace_dir + scf_type;
    return files_dir;
  }

  std::string get_files_prefix(std::string scf_type = "") {
    if(scf_type.empty()) scf_type = ioptions.scf_options.scf_type;

    const std::string out_fp       = workspace_dir;
    const std::string files_dir    = out_fp + scf_type;
    const std::string files_prefix = files_dir + "/" + sys_data.output_file_prefix;
    return files_prefix;
  }
};
