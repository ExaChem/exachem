/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/system_data.hpp"

void SystemData::print() {
  std::cout << "scf_type = " << scf_type_string << std::endl;
  if(is_restricted) std::cout << "Closed-Shell SCF" << std::endl;
  if(is_unrestricted) {
    if(is_cuscf) { std::cout << "Constrained Unrestricted SCF" << std::endl; }
    else { std::cout << "Open-Shell SCF" << std::endl; }
  }
  if(is_restricted_os) std::cout << "Restricted Open-Shell SCF" << std::endl;
  if(is_ks) std::cout << "KS-DFT Enabled" << std::endl;
  if(do_snK) std::cout << "snK Enabled" << std::endl;
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
