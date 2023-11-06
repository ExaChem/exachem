/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "input_parser.hpp"

#include <filesystem>
namespace fs = std::filesystem;

struct SystemData {
  OptionsMap options_map;
  int        n_occ_alpha{};
  int        n_vir_alpha{};
  int        n_occ_beta{};
  int        n_vir_beta{};
  int        n_lindep;
  int        ndf{};
  int        nbf{};
  int        nbf_orig{};
  int        nelectrons{};
  int        nelectrons_alpha{};
  int        nelectrons_beta{};
  int        nelectrons_active{};
  int        n_frozen_core{};
  int        n_frozen_virtual{};
  int        nmo{};
  int        nocc{};
  int        nvir{};
  int        nact{};
  int        focc{};
  int        qed_nmodes{};
  bool       freeze_atomic{};
  bool       is_restricted{};
  bool       is_unrestricted{};
  bool       is_restricted_os{};
  bool       is_ks{};
  bool       is_qed{};
  bool       do_qed{};
  bool       has_ecp{};
  // bool       is_cas{};???

  std::string scf_type_string;
  std::string input_molecule;
  std::string output_file_prefix;

  // output data
  double scf_energy{};
  int    num_chol_vectors{};
  double cc2_corr_energy{};
  double ccsd_corr_energy{};

  // json data
  json results;

  void print();

  void update(bool spin_orbital = true);

  SystemData(OptionsMap options_map_, const std::string scf_type_string);
};

inline void check_json(std::string filename) {
  std::string get_ext = fs::path(filename).extension();
  const bool  is_json = (get_ext == ".json");
  if(!is_json) tamm_terminate("ERROR: Input file provided [" + filename + "] must be a json file");
}

inline json json_from_file(std::string jfile) {
  json jdata;
  check_json(jfile);

  auto                  is = std::ifstream(jfile);
  json_sax_no_exception jsax(jdata);
  bool                  parse_result = json::sax_parse(is, &jsax);
  if(!parse_result) tamm_terminate("Error parsing file: " + jfile);

  return jdata;
}

inline void json_to_file(json jdata, std::string jfile) {
  std::ofstream res_file(jfile);
  res_file << std::setw(2) << jdata << std::endl;
}

inline std::string getfilename(std::string filename) {
  size_t lastindex = filename.find_last_of(".");
  auto   fname     = filename.substr(0, lastindex);
  return fname.substr(fname.find_last_of("/") + 1, fname.length());
}

void write_sinfo(SystemData& sys_data, libint2::BasisSet& shells);

void write_json_data(SystemData& sys_data, const std::string cmodule);
