/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/txt_utils.hpp"
#include "options/input_options.hpp"
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

class SystemData {
public:
  int  n_occ_alpha{};
  int  n_vir_alpha{};
  int  n_occ_beta{};
  int  n_vir_beta{};
  int  n_lindep{};
  int  ndf{};
  int  nbf{};
  int  nbf_orig{};
  int  nelectrons{};
  int  nelectrons_alpha{};
  int  nelectrons_beta{};
  int  nelectrons_active{};
  int  n_frozen_core{};
  int  n_frozen_virtual{};
  int  nmo{};
  int  nocc{};
  int  nvir{};
  int  nact{};
  int  focc{};
  int  qed_nmodes{};
  bool freeze_atomic{};
  bool is_restricted{};
  bool is_unrestricted{};
  bool is_restricted_os{};
  bool is_ks{};
  bool is_qed{};
  bool do_qed{};
  bool do_snK{};
  bool has_ecp{};
  // bool       is_cas{};???

  std::string scf_type_string;
  std::string input_molecule;
  std::string output_file_prefix;

  // json data
  nlohmann::ordered_json results;
  void                   print();
  void                   update(bool spin_orbital = true);

  SystemData() = default;
};
