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
  std::cout << " charge            = " << charge << std::endl;
  std::cout << " multiplicity      = " << multiplicity << std::endl;
  std::cout << " level shift       = " << lshift << std::endl;
  std::cout << " tol_int           = " << tol_int << std::endl;
  std::cout << " tol_sch           = " << tol_sch << std::endl;
  std::cout << " tol_lindep        = " << tol_lindep << std::endl;
  std::cout << " conve             = " << conve << std::endl;
  std::cout << " convd             = " << convd << std::endl;
  std::cout << " diis_hist         = " << diis_hist << std::endl;
  std::cout << " AO_tilesize       = " << AO_tilesize << std::endl;
  std::cout << " writem            = " << writem << std::endl;
  std::cout << " damp              = " << damp << std::endl;
  std::cout << " n_lindep          = " << n_lindep << std::endl;
  if(!moldenfile.empty()) {
    std::cout << " moldenfile        = " << moldenfile << std::endl;
    // std::cout << " n_lindep = " << n_lindep <<  std::endl;
  }

  std::cout << " scf_type          = " << scf_type << std::endl;

  // QED
  if(!qed_omegas.empty()) {
    std::cout << " qed_omegas       = [";
    for(auto x: qed_omegas) { std::cout << x << ","; }
    std::cout << "\b]" << std::endl;
  }

  if(!qed_lambdas.empty()) {
    std::cout << " qed_lambdas       = [";
    for(auto x: qed_lambdas) { std::cout << x << ","; }
    std::cout << "\b]" << std::endl;
  }

  if(!qed_volumes.empty()) {
    std::cout << " qed_volumes       = [";
    for(auto x: qed_volumes) { std::cout << x << ","; }
    std::cout << "\b]" << std::endl;
  }

  if(!qed_polvecs.empty()) {
    std::cout << " qed_polvecs       = [";
    for(auto x: qed_polvecs) {
      std::cout << "[";
      for(auto y: x) { std::cout << y << ","; }
      std::cout << "\b],";
    }
    std::cout << "\b]" << std::endl;
  }

  txt_utils::print_bool(" direct_df        ", direct_df);

  if(!xc_type.empty() || snK) {
    std::cout << " DFT " << std::endl << " {" << std::endl;
    std::cout << "  xc_type           = [ ";
    for(auto xcfunc: xc_type) { std::cout << " \"" << xcfunc << "\","; }
    std::cout << "\b ]" << std::endl;

    std::cout << "  xc_grid_type      = " << xc_grid_type << std::endl;
    std::cout << "  xc_pruning_scheme = " << xc_pruning_scheme << std::endl;
    std::cout << "  xc_weight_scheme  = " << xc_weight_scheme << std::endl;
    std::cout << "  xc_exec_space     = " << xc_exec_space << std::endl;
    std::cout << "  xc_basis_tol      = " << xc_basis_tol << std::endl;
    std::cout << "  xc_batch_size     = " << xc_batch_size << std::endl;
    std::cout << "  xc_lb_kernel      = " << xc_lb_kernel << std::endl;
    std::cout << "  xc_mw_kernel      = " << xc_mw_kernel << std::endl;
    std::cout << "  xc_int_kernel     = " << xc_int_kernel << std::endl;
    std::cout << "  xc_red_kernel     = " << xc_red_kernel << std::endl;
    std::cout << "  xc_lwd_kernel     = " << xc_lwd_kernel << std::endl;
    if(xc_radang_size.first > 0 && xc_radang_size.second > 0) {
      std::cout << "  xc_radang_size    = " << xc_radang_size.first << ", " << xc_radang_size.second
                << std::endl;
    }
    else { std::cout << "  xc_rad_quad       = " << xc_rad_quad << std::endl; }

    txt_utils::print_bool("  snK              ", snK);
    std::cout << "  xc_snK_etol       = " << xc_snK_etol << std::endl;
    std::cout << "  xc_snK_ktol       = " << xc_snK_ktol << std::endl;
    std::cout << " }" << std::endl;
  }

  if(scalapack_np_row > 0 && scalapack_np_col > 0) {
    std::cout << " scalapack_np_row  = " << scalapack_np_row << std::endl;
    std::cout << " scalapack_np_col  = " << scalapack_np_col << std::endl;
    if(scalapack_nb > 1) std::cout << " scalapack_nb      = " << scalapack_nb << std::endl;
  }
  std::cout << " restart_size      = " << restart_size << std::endl;
  txt_utils::print_bool(" restart          ", restart);
  txt_utils::print_bool(" debug            ", debug);
  if(restart) txt_utils::print_bool(" noscf            ", noscf);
  // txt_utils::print_bool(" sad         ", sad);
  if(mulliken_analysis || mos_txt || mo_vectors_analysis.first) {
    std::cout << " PRINT {" << std::endl;
    if(mos_txt) std::cout << std::boolalpha << "  mos_txt             = " << mos_txt << std::endl;
    if(mulliken_analysis)
      std::cout << std::boolalpha << "  mulliken_analysis   = " << mulliken_analysis << std::endl;
    if(mo_vectors_analysis.first) {
      std::cout << "  mo_vectors_analysis = [" << std::boolalpha << mo_vectors_analysis.first;
      std::cout << "," << mo_vectors_analysis.second << "]" << std::endl;
    }
    std::cout << " }" << std::endl;
  }
  std::cout << "}" << std::endl << std::flush;
} // END of SCFOptions::print

void CCSDOptions::print() {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "CCSD Options" << std::endl;
  std::cout << "{" << std::endl;
  std::cout << " cache_size           = " << cache_size << std::endl;
  std::cout << " ccsdt_tilesize       = " << ccsdt_tilesize << std::endl;

  std::cout << " ndiis                = " << ndiis << std::endl;
  std::cout << " threshold            = " << threshold << std::endl;
  std::cout << " tilesize             = " << tilesize << std::endl;
  // if(nactive > 0) std::cout << " nactive              = " << nactive << std::endl;
  if(pcore > 0) std::cout << " pcore                = " << pcore << std::endl;
  std::cout << " ccsd_maxiter         = " << ccsd_maxiter << std::endl;
  txt_utils::print_bool(" freeze_atomic       ", freeze_atomic);
  std::cout << " freeze_core          = " << freeze_core << std::endl;
  std::cout << " freeze_virtual       = " << freeze_virtual << std::endl;
  if(lshift != 0) std::cout << " lshift               = " << lshift << std::endl;
  if(gf_nprocs_poi > 0) std::cout << " gf_nprocs_poi        = " << gf_nprocs_poi << std::endl;
  txt_utils::print_bool(" readt               ", readt);
  txt_utils::print_bool(" writet              ", writet);
  std::cout << " writet_iter          = " << writet_iter << std::endl;
  txt_utils::print_bool(" profile_ccsd        ", profile_ccsd);
  txt_utils::print_bool(" balance_tiles       ", balance_tiles);

  if(!dlpno_dfbasis.empty()) std::cout << " dlpno_dfbasis        = " << dlpno_dfbasis << std::endl;
  if(!doubles_opt_eqns.empty()) {
    std::cout << " doubles_opt_eqns        = [";
    for(auto x: doubles_opt_eqns) std::cout << x << ",";
    std::cout << "]" << std::endl;
  }

  if(!ext_data_path.empty()) { std::cout << " ext_data_path   = " << ext_data_path << std::endl; }

  if(eom_nroots > 0) {
    std::cout << " eom_nroots           = " << eom_nroots << std::endl;
    std::cout << " eom_microiter        = " << eom_microiter << std::endl;
    std::cout << " eom_threshold        = " << eom_threshold << std::endl;
  }

  if(gf_p_oi_range > 0) {
    std::cout << " gf_p_oi_range        = " << gf_p_oi_range << std::endl;
    txt_utils::print_bool(" gf_ip               ", gf_ip);
    txt_utils::print_bool(" gf_ea               ", gf_ea);
    txt_utils::print_bool(" gf_os               ", gf_os);
    txt_utils::print_bool(" gf_cs               ", gf_cs);
    txt_utils::print_bool(" gf_restart          ", gf_restart);
    txt_utils::print_bool(" gf_profile          ", gf_profile);
    txt_utils::print_bool(" gf_itriples         ", gf_itriples);
    std::cout << " gf_ndiis             = " << gf_ndiis << std::endl;
    std::cout << " gf_ngmres            = " << gf_ngmres << std::endl;
    std::cout << " gf_maxiter           = " << gf_maxiter << std::endl;
    std::cout << " gf_eta               = " << gf_eta << std::endl;
    std::cout << " gf_lshift            = " << gf_lshift << std::endl;
    std::cout << " gf_preconditioning   = " << gf_preconditioning << std::endl;
    std::cout << " gf_damping_factor    = " << gf_damping_factor << std::endl;

    // std::cout << " gf_omega       = " << gf_omega <<std::endl;
    std::cout << " gf_threshold         = " << gf_threshold << std::endl;
    std::cout << " gf_omega_min_ip      = " << gf_omega_min_ip << std::endl;
    std::cout << " gf_omega_max_ip      = " << gf_omega_max_ip << std::endl;
    std::cout << " gf_omega_min_ip_e    = " << gf_omega_min_ip_e << std::endl;
    std::cout << " gf_omega_max_ip_e    = " << gf_omega_max_ip_e << std::endl;
    std::cout << " gf_omega_min_ea      = " << gf_omega_min_ea << std::endl;
    std::cout << " gf_omega_max_ea      = " << gf_omega_max_ea << std::endl;
    std::cout << " gf_omega_min_ea_e    = " << gf_omega_min_ea_e << std::endl;
    std::cout << " gf_omega_max_ea_e    = " << gf_omega_max_ea_e << std::endl;
    std::cout << " gf_omega_delta       = " << gf_omega_delta << std::endl;
    std::cout << " gf_omega_delta_e     = " << gf_omega_delta_e << std::endl;
    if(!gf_orbitals.empty()) {
      std::cout << " gf_orbitals        = [";
      for(auto x: gf_orbitals) std::cout << x << ",";
      std::cout << "]" << std::endl;
    }
    if(gf_analyze_level > 0) {
      std::cout << " gf_analyze_level     = " << gf_analyze_level << std::endl;
      std::cout << " gf_analyze_num_omega = " << gf_analyze_num_omega << std::endl;
      std::cout << " gf_analyze_omega     = [";
      for(auto x: gf_analyze_omega) std::cout << x << ",";
      std::cout << "]" << std::endl;
    }
    if(gf_extrapolate_level > 0)
      std::cout << " gf_extrapolate_level = " << gf_extrapolate_level << std::endl;
  }

  txt_utils::print_bool(" debug               ", debug);
  std::cout << "}" << std::endl;
} // END of CCSDOptions::print()

void CommonOptions::print() {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "Common Options" << std::endl;
  std::cout << "{" << std::endl;
  std::cout << " maxiter    = " << maxiter << std::endl;
  std::cout << " basis      = " << basis << " ";
  std::cout << gaussian_type;
  std::cout << std::endl;
  if(!dfbasis.empty()) std::cout << " dfbasis    = " << dfbasis << std::endl;
  if(!basisfile.empty()) std::cout << " basisfile  = " << basisfile << std::endl;
  std::cout << " geom_units = " << geom_units << std::endl;
  std::cout << " natoms_max = " << natoms_max << std::endl;
  txt_utils::print_bool(" debug     ", debug);
  if(!file_prefix.empty()) std::cout << " file_prefix    = " << file_prefix << std::endl;
  if(!output_dir.empty()) std::cout << " output_dir    = " << output_dir << std::endl;
  std::cout << "}" << std::endl;
}

void CCSDOptions::initialize() {
  threshold      = 1e-6;
  tilesize       = 40;
  ndiis          = 5;
  lshift         = 0;
  nactive_oa     = 0;
  nactive_ob     = 0;
  nactive_va     = 0;
  nactive_vb     = 0;
  ducc_lvl       = 2;
  ccsd_maxiter   = 100;
  freeze_core    = 0;
  freeze_virtual = 0;
  balance_tiles  = true;
  profile_ccsd   = false;

  writet      = false;
  writet_iter = ndiis;
  readt       = false;

  localize      = false;
  skip_dlpno    = false;
  keep_npairs   = 1;
  max_pnos      = 1;
  dlpno_dfbasis = "";
  TCutEN        = 0.97;
  TCutPNO       = 1.0e-6;
  TCutTNO       = 1.0e-6;
  TCutPre       = 1.0e-3;
  TCutPairs     = 1.0e-3;
  TCutDO        = 1e-2;
  TCutDOij      = 1e-7;
  TCutDOPre     = 3e-2;

  cache_size     = 8;
  skip_ccsd      = false;
  ccsdt_tilesize = 40;

  eom_nroots    = 1;
  eom_threshold = 1e-6;
  eom_type      = "right";
  eom_microiter = ccsd_maxiter;

  pcore         = 0;
  ntimesteps    = 10;
  rt_microiter  = 20;
  rt_threshold  = 1e-6;
  rt_multiplier = 0.5;
  rt_step_size  = 0.025;
  secent_x      = 0.1;
  h_red         = 0.5;
  h_inc         = 1.2;
  h_max         = 0.25;

  gf_ip       = true;
  gf_ea       = false;
  gf_os       = false;
  gf_cs       = true;
  gf_restart  = false;
  gf_itriples = false;
  gf_profile  = false;

  gf_p_oi_range      = 0; // 1-number of occupied, 2-all MOs
  gf_ndiis           = 10;
  gf_ngmres          = 10;
  gf_maxiter         = 500;
  gf_eta             = 0.01;
  gf_lshift          = 1.0;
  gf_preconditioning = true;
  gf_damping_factor  = 1.0;
  gf_nprocs_poi      = 0;
  // gf_omega          = -0.4; //a.u (range min to max)
  gf_threshold         = 1e-2;
  gf_omega_min_ip      = -0.8;
  gf_omega_max_ip      = -0.4;
  gf_omega_min_ip_e    = -2.0;
  gf_omega_max_ip_e    = 0;
  gf_omega_min_ea      = 0.0;
  gf_omega_max_ea      = 0.1;
  gf_omega_min_ea_e    = 0.0;
  gf_omega_max_ea_e    = 2.0;
  gf_omega_delta       = 0.01;
  gf_omega_delta_e     = 0.002;
  gf_extrapolate_level = 0;
  gf_analyze_level     = 0;
  gf_analyze_num_omega = 0;
} // end of CCSDOptions::initialize()

void FCIOptions::initialize() {
  job       = "CI";
  expansion = "CAS";

  // MCSCF
  max_macro_iter     = 100;
  max_orbital_step   = 0.5;
  orb_grad_tol_mcscf = 5e-6;
  enable_diis        = true;
  diis_start_iter    = 3;
  diis_nkeep         = 10;
  ci_res_tol         = 1e-8;
  ci_max_subspace    = 20;
  ci_matel_tol       = std::numeric_limits<double>::epsilon();

  // PRINT
  print_mcscf = true;
}

void CDOptions::print() {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "CD Options" << std::endl;
  std::cout << "{" << std::endl;
  std::cout << std::boolalpha << " debug            = " << debug << std::endl;
  std::cout << std::boolalpha << " skip_cd          = [" << skip_cd.first << "," << skip_cd.second
            << "]" << std::endl;
  std::cout << std::boolalpha << " write_cv         = [" << write_cv.first << "," << write_cv.second
            << "]" << std::endl;
  std::cout << " diagtol          = " << diagtol << std::endl;
  std::cout << " itilesize        = " << itilesize << std::endl;
  std::cout << " max_cvecs_factor = " << max_cvecs_factor << std::endl;
  std::cout << "}" << std::endl;
}

void GWOptions::print() {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "GW Options" << std::endl;
  std::cout << "{" << std::endl;
  std::cout << " ngl       = " << ngl << std::endl;
  std::cout << " noqpa     = " << noqpa << std::endl;
  std::cout << " noqpb     = " << noqpb << std::endl;
  std::cout << " nvqpa     = " << nvqpa << std::endl;
  std::cout << " nvqp/b     = " << nvqpb << std::endl;
  std::cout << " ieta      = " << ieta << std::endl;
  std::cout << " maxnewton = " << maxnewton << std::endl;
  std::cout << " maxev     = " << maxev << std::endl;
  std::cout << " method    = " << method << std::endl;
  std::cout << " cdbasis   = " << cdbasis << std::endl;
  txt_utils::print_bool(" evgw     ", evgw);
  txt_utils::print_bool(" evgw0    ", evgw0);
  txt_utils::print_bool(" core     ", core);
  txt_utils::print_bool(" minres   ", minres);
  txt_utils::print_bool(" debug    ", debug);
  std::cout << "}" << std::endl;
}

void TaskOptions::print() {
  CommonOptions::print();

  std::cout << std::endl << "Task Options" << std::endl;
  std::cout << "{" << std::endl;
  txt_utils::print_bool(" sinfo        ", sinfo);
  txt_utils::print_bool(" scf          ", scf);
  txt_utils::print_bool(" mp2          ", mp2);
  txt_utils::print_bool(" gw           ", gw);
  txt_utils::print_bool(" cc2          ", cc2);
  txt_utils::print_bool(" fci          ", fci);
  txt_utils::print_bool(" fcidump      ", fcidump);
  txt_utils::print_bool(" cd_2e        ", cd_2e);
  txt_utils::print_bool(" ducc         ", ducc);
  txt_utils::print_bool(" ccsd         ", ccsd);
  txt_utils::print_bool(" ccsd_sf      ", ccsd_sf);
  txt_utils::print_bool(" ccsd_lambda  ", ccsd_lambda);
  txt_utils::print_bool(" eom_ccsd     ", eom_ccsd);
  txt_utils::print_bool(" rteom_cc2    ", rteom_cc2);
  txt_utils::print_bool(" rteom_ccsd   ", rteom_ccsd);
  txt_utils::print_bool(" gfccsd       ", gfccsd);
  std::cout << " dlpno_ccsd:  " << dlpno_ccsd.first << ", " << dlpno_ccsd.second << "\n";
  std::cout << " dlpno_ccsd_t " << dlpno_ccsd_t.first << ", " << dlpno_ccsd_t.second << "\n";
  std::cout << "}" << std::endl;
}
