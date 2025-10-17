/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "input_options.hpp"

void SCFOptions::print() {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "SCF Options" << std::endl;
  std::cout << "{" << std::endl;
  print_option("charge", charge, 18);
  print_option("multiplicity", multiplicity, 18);
  print_option("level shift", lshift, 18);
  print_option("tol_int", tol_int, 18);
  print_option("tol_sch", tol_sch, 18);
  print_option("tol_lindep", tol_lindep, 18);
  print_option("conve", conve, 18);
  print_option("convd", convd, 18);
  print_option("diis_hist", diis_hist, 18);
  print_option("AO_tilesize", AO_tilesize, 18);
  print_option("writem", writem, 18);
  print_option("damp", damp, 18);
  print_option("n_lindep", n_lindep, 18);

  if(!moldenfile.empty()) {
    print_option("moldenfile", moldenfile, 18);
    // std::cout << " n_lindep = " << n_lindep <<  std::endl;
  }

  print_option("scf_type", scf_type, 18);
  print_option("cuscf", cuscf, 18);

  // QED
  print_vec("qed_omegas", qed_omegas, 17);
  print_vec("qed_lambdas", qed_lambdas, 17);
  print_vec("qed_volumes", qed_volumes, 17);
  print_vec2d("qed_polvecs", qed_polvecs, 17);
  print_option("direct_df", direct_df, 18);

  if(!xc_type.empty() || snK) {
    std::cout << " DFT" << std::endl << " {" << std::endl;
    // Print xc_type as a custom formatted vector
    std::cout << "  " << std::setw(20) << std::left << "xc_type"
              << "= [ ";
    for(const auto& xcfunc: xc_type) { std::cout << "\"" << xcfunc << "\", "; }
    std::cout << "\b\b ]" << std::endl; // Remove last comma and space

    print_option("xc_grid_type", xc_grid_type, 20);
    print_option("xc_pruning_scheme", xc_pruning_scheme, 20);
    print_option("xc_weight_scheme", xc_weight_scheme, 20);
    print_option("xc_exec_space", xc_exec_space, 20);
    print_option("xc_basis_tol", xc_basis_tol, 20);
    print_option("xc_batch_size", xc_batch_size, 20);
    print_option("xc_lb_kernel", xc_lb_kernel, 20);
    print_option("xc_mw_kernel", xc_mw_kernel, 20);
    print_option("xc_int_kernel", xc_int_kernel, 20);
    print_option("xc_red_kernel", xc_red_kernel, 20);
    print_option("xc_lwd_kernel", xc_lwd_kernel, 20);

    std::cout << " }" << std::endl;

    if(xc_radang_size.first > 0 && xc_radang_size.second > 0) {
      print_pair("xc_radang_size", xc_radang_size, 20);
    }
    else { print_option("xc_rad_quad", xc_rad_quad, 20); }

    print_option("snK", snK, 20);
    print_option("xc_snK_etol", xc_snK_etol, 20);
    print_option("xc_snK_ktol", xc_snK_ktol, 20);
    std::cout << " }" << std::endl;
  }

  if(scalapack_np_row > 0 && scalapack_np_col > 0) {
    print_option("scalapack_np_row", scalapack_np_row, 20);
    print_option("scalapack_np_col", scalapack_np_col, 20);
    if(scalapack_nb > 1) print_option("scalapack_nb", scalapack_nb, 20);
  }
  print_option("restart_size", restart_size, 18);
  print_option("restart", restart, 18);
  print_option("debug", debug, 18);
  if(restart) print_option("noscf", noscf, 18);
  // if(sad) print_option("sad", sad, 18);
  if(mulliken_analysis || mos_txt || mo_vectors_analysis.first) {
    std::cout << " PRINT {" << std::endl;

    if(mos_txt) print_option("mos_txt", mos_txt, 20);
    if(mulliken_analysis) print_option("mulliken_analysis", mulliken_analysis, 20);

    if(mo_vectors_analysis.first) { print_pair("mo_vectors_analysis", mo_vectors_analysis, 20); }
    std::cout << " }" << std::endl;
  }

  std::cout << "}" << std::endl << std::flush;
} // END of SCFOptions::print

void CCSDOptions::print() {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "CCSD Options" << std::endl;
  std::cout << "{" << std::endl;

  print_option("cache_size", cache_size, 22);
  print_option("ccsdt_tilesize", ccsdt_tilesize, 22);
  print_option("ndiis", ndiis, 22);
  print_option("threshold", threshold, 22);
  print_option("tilesize", tilesize, 22);
  // if(nactive > 0)
  print_option("ccsd_maxiter", ccsd_maxiter, 22);

  if(pcore > 0) print_option("pcore", pcore, 22);
  print_option("freeze_atomic", freeze_atomic, 22);
  print_option("freeze_core", freeze_core, 22);
  print_option("freeze_virtual", freeze_virtual, 22);
  if(lshift != 0) print_option("lshift", lshift, 22);
  if(gf_nprocs_poi > 0) print_option("gf_nprocs_poi", gf_nprocs_poi, 22);

  print_option("readt", readt, 22);
  print_option("writet", writet, 22);
  print_option("writet_iter", writet_iter, 22);
  print_option("profile_ccsd", profile_ccsd, 22);
  print_option("balance_tiles", balance_tiles, 22);

  if(!dlpno_dfbasis.empty()) print_option("dlpno_dfbasis", dlpno_dfbasis, 22);

  if(!doubles_opt_eqns.empty()) print_vec("doubles_opt_eqns", doubles_opt_eqns);
  if(!ext_data_path.empty()) { print_option("ext_data_path", ext_data_path, 18); }

  if(eom_nroots > 0) {
    print_option("eom_nroots", eom_nroots, 22);
    print_option("eom_microiter", eom_microiter, 22);
    print_option("eom_threshold", eom_threshold, 22);
  }

  if(gf_p_oi_range > 0) {
    print_option("gf_p_oi_range", gf_p_oi_range, 22);
    print_option("gf_ip", gf_ip, 22);
    print_option("gf_ea", gf_ea, 22);
    print_option("gf_os", gf_os, 22);
    print_option("gf_cs", gf_cs, 22);
    print_option("gf_restart", gf_restart, 22);
    print_option("gf_profile", gf_profile, 22);
    print_option("gf_itriples", gf_itriples, 22);
    print_option("gf_ndiis", gf_ndiis, 22);
    print_option("gf_ngmres", gf_ngmres, 22);
    print_option("gf_maxiter", gf_maxiter, 22);
    print_option("gf_eta", gf_eta, 22);
    print_option("gf_lshift", gf_lshift, 22);
    print_option("gf_preconditioning", gf_preconditioning, 22);
    print_option("gf_damping_factor", gf_damping_factor, 22);
    // print_option("gf_omega",          gf_omega);
    print_option("gf_threshold", gf_threshold, 22);
    print_option("gf_omega_min_ip", gf_omega_min_ip, 22);
    print_option("gf_omega_max_ip", gf_omega_max_ip, 22);
    print_option("gf_omega_min_ip_e", gf_omega_min_ip_e, 22);
    print_option("gf_omega_max_ip_e", gf_omega_max_ip_e, 22);
    print_option("gf_omega_min_ea", gf_omega_min_ea, 22);
    print_option("gf_omega_max_ea", gf_omega_max_ea, 22);
    print_option("gf_omega_min_ea_e", gf_omega_min_ea_e, 22);
    print_option("gf_omega_max_ea_e", gf_omega_max_ea_e, 22);
    print_option("gf_omega_delta", gf_omega_delta, 22);
    print_option("gf_omega_delta_e", gf_omega_delta_e, 22);
    if(!gf_orbitals.empty()) {
      if(!gf_orbitals.empty()) print_vec("gf_orbitals", gf_orbitals, 22);
    }
    if(gf_analyze_level > 0) {
      print_option("gf_analyze_level", gf_analyze_level, 22);
      print_option("gf_analyze_num_omega", gf_analyze_num_omega, 22);
      print_vec("gf_analyze_omega", gf_analyze_omega, 22);
    }
    if(gf_extrapolate_level > 0) print_option("gf_extrapolate_level", gf_extrapolate_level, 22);
  }

  print_option("debug", debug, 22);
  std::cout << "}" << std::endl;
} // END of CCSDOptions::print()

void CommonOptions::print() {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "Common Options" << std::endl;
  std::cout << "{" << std::endl;

  print_option("maxiter", maxiter, 14);

  // Combine basis and gaussian_type on one line
  std::cout << "  " << std::setw(14) << std::left << "basis"
            << "= " << basis << " " << gaussian_type << std::endl;

  if(!dfbasis.empty()) print_option("dfbasis", dfbasis, 14);
  print_option("geom_units", geom_units, 14);
  print_option("ang2au factor", exachem::constants::ang2bohr, 14, 10);
  // print_option("natoms_max", natoms_max, 14);
  print_option("debug", debug, 14);
  if(!file_prefix.empty()) print_option("file_prefix", file_prefix, 14);
  if(!output_dir.empty()) print_option("output_dir", output_dir, 14);

  std::cout << "}" << std::endl;
}

// initialize() functions for CCSDOptions and FCIOptions removed; defaults now set inline.

void CDOptions::print() {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "CD Options" << std::endl;
  std::cout << "{" << std::endl;
  print_option("debug", debug, 18);
  print_pair("skip_cd", skip_cd, 18);
  print_pair("write_cv", write_cv, 18);
  print_option("diagtol", diagtol, 18);
  print_option("itilesize", itilesize, 18);
  print_option("max_cvecs_factor", max_cvecs_factor, 18);
  std::cout << "}" << std::endl;
}

void GWOptions::print() {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "GW Options" << std::endl;
  std::cout << "{" << std::endl;
  print_option("ngl", ngl, 12);
  print_option("noqpa", noqpa, 12);
  print_option("noqpb", noqpb, 12);
  print_option("nvqpa", nvqpa, 12);
  print_option("nvqp/b", nvqpb, 12);
  print_option("ieta", ieta, 12);
  print_option("maxnewton", maxnewton, 12);
  print_option("maxev", maxev, 12);
  print_option("method", method, 12);
  print_option("cdbasis", cdbasis, 12);
  print_option("evgw", evgw, 12);
  print_option("evgw0", evgw0, 12);
  print_option("core", core, 12);
  print_option("minres", minres, 12);
  print_option("debug", debug, 12);
  std::cout << "}" << std::endl;
}

void TaskOptions::print() {
  CommonOptions::print();

  std::cout << std::endl << "Task Options" << std::endl;
  std::cout << "{" << std::endl;
  print_option("sinfo", sinfo, 18);
  print_option("scf", scf, 18);
  print_option("mp2", mp2, 18);
  print_option("gw", gw, 18);
  print_option("cc2", cc2, 18);
  print_option("fci", fci, 18);
  print_option("fcidump", fcidump, 18);
  print_option("cd_2e", cd_2e, 18);
  print_option("ccsd", ccsd, 18);
  print_option("ccsd_sf", ccsd_sf, 18);
  print_option("ccsd_lambda", ccsd_lambda, 18);
  print_option("eom_ccsd", eom_ccsd, 18);
  print_option("rteom_cc2", rteom_cc2, 18);
  print_option("rteom_ccsd", rteom_ccsd, 18);
  print_option("gfccsd", gfccsd, 18);
  print_pair("ducc", ducc, 18);
  print_pair("dlpno_ccsd", dlpno_ccsd, 18);
  print_pair("dlpno_ccsd_t", dlpno_ccsd_t, 18);
  std::cout << "}" << std::endl;
}
