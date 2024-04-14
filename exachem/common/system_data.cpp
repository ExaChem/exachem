#include "system_data.hpp"

void SystemData::print() {
  std::cout << "scf_type = " << scf_type_string << std::endl;
  if(is_restricted) std::cout << "Closed-Shell SCF" << std::endl;
  if(is_unrestricted) std::cout << "Open-Shell SCF" << std::endl;
  if(is_restricted_os) std::cout << "Restricted Open-Shell SCF" << std::endl;
  if(is_ks) std::cout << "KS-DFT Enabled" << std::endl;
  if(is_qed && do_qed && is_ks) { std::cout << "QED-KS Enabled" << std::endl; }
  else if(is_qed && is_ks) { std::cout << "QEDFT Enabled" << std::endl; }
  else if(is_qed) { std::cout << "QED-HF Enabled" << std::endl; }

  std::cout << "nbf = " << nbf << std::endl;
  std::cout << "nbf_orig = " << nbf_orig << std::endl;
  std::cout << "n_lindep = " << n_lindep << std::endl;

  std::cout << "focc = " << focc << std::endl;
  std::cout << "nmo = " << nmo << std::endl;
  std::cout << "nocc = " << nocc << std::endl;
  if(nact) std::cout << "nact = " << nact << std::endl;
  std::cout << "nvir = " << nvir << std::endl;

  std::cout << "n_occ_alpha = " << n_occ_alpha << std::endl;
  std::cout << "n_vir_alpha = " << n_vir_alpha << std::endl;
  std::cout << "n_occ_beta = " << n_occ_beta << std::endl;
  std::cout << "n_vir_beta = " << n_vir_beta << std::endl;

  std::cout << "nelectrons = " << nelectrons << std::endl;
  if(nelectrons_active) std::cout << "nelectrons_active = " << nelectrons_active << std::endl;
  std::cout << "nelectrons_alpha = " << nelectrons_alpha << std::endl;
  std::cout << "nelectrons_beta = " << nelectrons_beta << std::endl;
  if(freeze_atomic) std::cout << "freeze atomic = true" << std::endl;
  std::cout << "n_frozen_core = " << n_frozen_core << std::endl;
  std::cout << "n_frozen_virtual = " << n_frozen_virtual << std::endl;
  std::cout << "----------------------------" << std::endl;
}

void SystemData::update(bool spin_orbital) {
  EXPECTS(nbf == n_occ_alpha + n_vir_alpha); // lin-deps
  // EXPECTS(nbf_orig == n_occ_alpha + n_vir_alpha + n_lindep + n_frozen_core + n_frozen_virtual);
  nocc = n_occ_alpha + n_occ_beta;
  nvir = n_vir_alpha + n_vir_beta;
  // EXPECTS(nelectrons == n_occ_alpha + n_occ_beta);
  // EXPECTS(nelectrons == nelectrons_alpha+nelectrons_beta);
  if(spin_orbital) nmo = n_occ_alpha + n_vir_alpha + n_occ_beta + n_vir_beta; // lin-deps
  else nmo = n_occ_alpha + n_vir_alpha;
}

SystemData::SystemData(ECOptions options_map_, const std::string scf_type_string):
  options_map(options_map_), scf_type_string(scf_type_string) {
  results          = nlohmann::ordered_json::object();
  is_restricted    = false;
  is_unrestricted  = false;
  is_restricted_os = false;
  is_ks            = false;
  is_qed           = false;
  do_qed           = false;
  freeze_atomic    = false;
  if(scf_type_string == "restricted") {
    focc          = 1;
    is_restricted = true;
  }
  else if(scf_type_string == "unrestricted") {
    focc            = 2;
    is_unrestricted = true;
  }
  else if(scf_type_string == "restricted_os") {
    focc             = -1;
    is_restricted_os = true;
  }
  else tamm_terminate("ERROR: unrecognized scf_type [" + scf_type_string + "] provided");
  if(!options_map_.scf_options.xc_type.empty()) { is_ks = true; }

  if(!options_map_.scf_options.qed_volumes.empty()) {
    for(auto x: options_map_.scf_options.qed_volumes) {
      options_map_.scf_options.qed_lambdas.push_back(sqrt(1.0 / x));
    }
  }
  qed_nmodes = options_map_.scf_options.qed_lambdas.size();
  if(qed_nmodes > 0) {
    is_qed = true;
    do_qed = true;
  }
  if(is_qed && is_ks) {
    for(auto x: options_map_.scf_options.xc_type) {
      if(txt_utils::strequal_case(x, "gga_xc_qed") || txt_utils::strequal_case(x, "mgga_xc_qed") ||
         txt_utils::strequal_case(x, "hyb_gga_xc_qed") ||
         txt_utils::strequal_case(x, "hyb_mgga_xc_qed")) {
        if(options_map_.scf_options.qed_omegas.size() != (size_t) qed_nmodes) {
          tamm_terminate("ERROR: (m)gga_xc_qed needs qed_omegas for each qed_nmode");
        }
        break;
      }
      else if(txt_utils::strequal_case(x, "gga_x_qed") ||
              txt_utils::strequal_case(x, "mgga_x_qed") ||
              txt_utils::strequal_case(x, "hyb_gga_x_qed") ||
              txt_utils::strequal_case(x, "hyb_mgga_x_qed")) {
        do_qed = false;
        break;
      }
    }
  }
}
