/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/initialize_system_data.hpp"

void IniSystemData::initialize(ChemEnv& chem_env) {
  chem_env.sys_data.results          = json::object();
  chem_env.sys_data.is_restricted    = false;
  chem_env.sys_data.is_unrestricted  = false;
  chem_env.sys_data.is_restricted_os = false;
  chem_env.sys_data.is_ks            = false;
  chem_env.sys_data.is_qed           = false;
  chem_env.sys_data.do_qed           = false;
  chem_env.sys_data.do_snK           = false;
  chem_env.sys_data.freeze_atomic    = false;

  SCFOptions& scf_options           = chem_env.ioptions.scf_options;
  chem_env.sys_data.scf_type_string = scf_options.scf_type;

  if(scf_options.scf_type == "restricted") {
    chem_env.sys_data.focc          = 1;
    chem_env.sys_data.is_restricted = true;
  }
  else if(scf_options.scf_type == "unrestricted") {
    chem_env.sys_data.focc            = 2;
    chem_env.sys_data.is_unrestricted = true;
  }
  else if(scf_options.scf_type == "restricted_os") {
    chem_env.sys_data.focc             = -1;
    chem_env.sys_data.is_restricted_os = true;
  }
  else tamm_terminate("ERROR: unrecognized scf_type [" + scf_options.scf_type + "] provided");
  if(!scf_options.xc_type.empty()) { chem_env.sys_data.is_ks = true; }

  if(scf_options.snK) { chem_env.sys_data.do_snK = true; }

  if(!scf_options.qed_volumes.empty()) {
    for(auto x: scf_options.qed_volumes) { scf_options.qed_lambdas.push_back(sqrt(1.0 / x)); }
  }
  chem_env.sys_data.qed_nmodes = scf_options.qed_lambdas.size();
  if(chem_env.sys_data.qed_nmodes > 0) {
    chem_env.sys_data.is_qed = true;
    chem_env.sys_data.do_qed = true;
  }

  if(chem_env.sys_data.is_qed && chem_env.sys_data.is_ks) {
    for(auto x: scf_options.xc_type) {
      if(txt_utils::strequal_case(x, "gga_xc_qed") || txt_utils::strequal_case(x, "mgga_xc_qed") ||
         txt_utils::strequal_case(x, "hyb_gga_xc_qed") ||
         txt_utils::strequal_case(x, "hyb_mgga_xc_qed")) {
        if(scf_options.qed_omegas.size() != (size_t) chem_env.sys_data.qed_nmodes) {
          tamm_terminate("ERROR: (m)gga_xc_qed needs qed_omegas for each qed_nmode");
        }
        break;
      }
      else if(txt_utils::strequal_case(x, "gga_x_qed") ||
              txt_utils::strequal_case(x, "mgga_x_qed") ||
              txt_utils::strequal_case(x, "hyb_gga_x_qed") ||
              txt_utils::strequal_case(x, "hyb_mgga_x_qed")) {
        chem_env.sys_data.do_qed = false;
        break;
      }
    }
  }
}
