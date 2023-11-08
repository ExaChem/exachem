/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cctype>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"

using namespace tamm;

#include "libint2_includes.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

using libint2::Atom;
using std::cerr;
using std::cout;
using std::endl;
using std::string;

inline bool strequal_case(const std::string& a, const std::string& b) {
  return a.size() == b.size() and
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const char a, const char b) { return std::tolower(a) == std::tolower(b); });
}

inline void print_bool(std::string str, bool val) {
  if(val) cout << str << " = true" << endl;
  else cout << str << " = false" << endl;
}

struct ECAtom {
  Atom                atom;
  std::string         esymbol;
  std::string         basis;
  bool                has_ecp{false};
  int                 ecp_nelec{};
  std::vector<double> ecp_coeffs{};
  std::vector<double> ecp_exps{};
  std::vector<int>    ecp_ams{};
  std::vector<int>    ecp_ns{};
};

class Options {
public:
  Options() {
    maxiter       = 50;
    debug         = false;
    basis         = "sto-3g";
    geom_units    = "angstrom";
    gaussian_type = "spherical";
  }

  bool                       debug;
  int                        maxiter;
  std::string                basis;
  std::string                dfbasis{};
  std::string                basisfile{}; // supports only ECPs for now
  std::string                gaussian_type;
  std::string                geom_units;
  std::string                output_file_prefix{};
  std::string                ext_data_path{};
  std::vector<libint2::Atom> atoms;
  std::vector<ECAtom>        ec_atoms;

  void print() {
    std::cout << std::defaultfloat;
    cout << endl << "Common Options" << endl;
    cout << "{" << endl;
    cout << " maxiter    = " << maxiter << endl;
    cout << " basis      = " << basis << " ";
    cout << gaussian_type;
    cout << endl;
    if(!dfbasis.empty()) cout << " dfbasis    = " << dfbasis << endl;
    if(!basisfile.empty()) cout << " basisfile  = " << basisfile << endl;
    cout << " geom_units = " << geom_units << endl;
    print_bool(" debug     ", debug);
    if(!output_file_prefix.empty())
      cout << " output_file_prefix    = " << output_file_prefix << endl;
    cout << "}" << endl;
  }
};

class TaskOptions: public Options {
public:
  TaskOptions() = default;
  TaskOptions(Options o): Options(o) {}
  bool                         sinfo{false};
  bool                         scf{false};
  bool                         mp2{false};
  bool                         gw{false};
  bool                         cc2{false};
  bool                         fci{false};
  bool                         fcidump{false};
  bool                         ducc{false};
  bool                         cd_2e{false};
  bool                         ccsd{false};
  bool                         ccsd_sf{false};
  bool                         ccsd_t{false};
  bool                         ccsd_lambda{false};
  bool                         eom_ccsd{false};
  bool                         rteom_cc2{false};
  bool                         rteom_ccsd{false};
  bool                         gfccsd{false};
  std::pair<bool, std::string> dlpno_ccsd{false, ""};
  std::pair<bool, std::string> dlpno_ccsd_t{false, ""};
};

class SCFOptions: public Options {
public:
  SCFOptions() = default;
  SCFOptions(Options o): Options(o) {
    charge           = 0;
    multiplicity     = 1;
    lshift           = 0;
    tol_int          = 1e-22;
    tol_sch          = 1e-10;
    tol_lindep       = 1e-5;
    conve            = 1e-8;
    convd            = 1e-7;
    diis_hist        = 10;
    AO_tilesize      = 30;
    dfAO_tilesize    = 50;
    restart_size     = 2000;
    restart          = false;
    noscf            = false;
    sad              = true;
    force_tilesize   = false;
    riscf            = 0; // 0 for JK, 1 for J, 2 for K
    riscf_str        = "JK";
    moldenfile       = "";
    n_lindep         = 0;
    scf_type         = "restricted";
    xc_type          = {}; // pbe0
    xc_grid_type     = "UltraFine";
    damp             = 100;
    nnodes           = 1;
    writem           = 1;
    scalapack_nb     = 256;
    scalapack_np_row = 0;
    scalapack_np_col = 0;
    qed_omegas       = {};
    qed_volumes      = {};
    qed_lambdas      = {};
    qed_polvecs      = {};
  }

  int         charge;
  int         multiplicity;
  double      lshift;     // level shift factor, +ve value b/w 0 and 1
  double      tol_int;    // tolerance for integral primitive screening
  double      tol_sch;    // tolerance for schwarz screening
  double      tol_lindep; // tolerance for linear dependencies
  double      conve;      // energy convergence
  double      convd;      // density convergence
  int         diis_hist;  // number of diis history entries
  uint32_t    AO_tilesize;
  uint32_t    dfAO_tilesize;
  bool        restart; // Read movecs from disk
  bool        noscf;   // only recompute energy from movecs
  bool        sad;
  bool        force_tilesize;
  int         restart_size; // read/write orthogonalizer, schwarz, etc matrices when N>=restart_size
  int         scalapack_nb;
  int         riscf;
  int         nnodes;
  int         scalapack_np_row;
  int         scalapack_np_col;
  std::string riscf_str;
  std::string moldenfile;
  int         n_lindep;
  int         writem;
  int         damp; // density mixing parameter
  std::string scf_type;
  std::string xc_grid_type;

  std::map<std::string, std::tuple<int, int>> guess_atom_options;

  std::vector<std::string> xc_type;
  // mos_txt: write lcao, mo transformed core H, fock, and v2 to disk as text files.
  bool                             mos_txt{false};
  bool                             mulliken_analysis{false};
  std::pair<bool, double>          mo_vectors_analysis{false, 0.15};
  std::vector<double>              qed_omegas;
  std::vector<double>              qed_volumes;
  std::vector<double>              qed_lambdas;
  std::vector<std::vector<double>> qed_polvecs;

  void print() {
    std::cout << std::defaultfloat;
    cout << endl << "SCF Options" << endl;
    cout << "{" << endl;
    cout << " charge       = " << charge << endl;
    cout << " multiplicity = " << multiplicity << endl;
    cout << " level shift  = " << lshift << endl;
    cout << " tol_int      = " << tol_int << endl;
    cout << " tol_sch      = " << tol_sch << endl;
    cout << " tol_lindep   = " << tol_lindep << endl;
    cout << " conve        = " << conve << endl;
    cout << " convd        = " << convd << endl;
    cout << " diis_hist    = " << diis_hist << endl;
    cout << " AO_tilesize  = " << AO_tilesize << endl;
    cout << " writem       = " << writem << endl;
    cout << " damp         = " << damp << endl;
    if(!moldenfile.empty()) {
      cout << " moldenfile   = " << moldenfile << endl;
      // cout << " n_lindep = " << n_lindep << endl;
    }

    cout << " scf_type     = " << scf_type << endl;

    // QED
    if(!qed_omegas.empty()) {
      cout << " qed_omegas  = [";
      for(auto x: qed_omegas) { cout << x << ","; }
      cout << "\b]" << endl;
    }

    if(!qed_lambdas.empty()) {
      cout << " qed_lambdas  = [";
      for(auto x: qed_lambdas) { cout << x << ","; }
      cout << "\b]" << endl;
    }

    if(!qed_volumes.empty()) {
      cout << " qed_volumes  = [";
      for(auto x: qed_volumes) { cout << x << ","; }
      cout << "\b]" << endl;
    }

    if(!qed_polvecs.empty()) {
      cout << " qed_polvecs  = [";
      for(auto x: qed_polvecs) {
        cout << "[";
        for(auto y: x) { cout << y << ","; }
        cout << "\b],";
      }
      cout << "\b]" << endl;
    }

    if(!xc_type.empty()) {
      cout << " xc_type      = [ ";
      for(auto xcfunc: xc_type) { cout << " \"" << xcfunc << "\","; }
      cout << "\b ]" << endl;
      cout << " xc_grid_type = " << xc_grid_type << std::endl;
    }

    if(scalapack_np_row > 0 && scalapack_np_col > 0) {
      cout << " scalapack_np_row = " << scalapack_np_row << endl;
      cout << " scalapack_np_col = " << scalapack_np_col << endl;
      if(scalapack_nb > 1) cout << " scalapack_nb = " << scalapack_nb << endl;
    }
    cout << " restart_size = " << restart_size << endl;
    print_bool(" restart     ", restart);
    print_bool(" debug       ", debug);
    if(restart) print_bool(" noscf       ", noscf);
    // print_bool(" sad         ", sad);
    if(mulliken_analysis || mos_txt || mo_vectors_analysis.first) {
      cout << " PRINT {" << endl;
      if(mos_txt) cout << std::boolalpha << "  mos_txt             = " << mos_txt << endl;
      if(mulliken_analysis)
        cout << std::boolalpha << "  mulliken_analysis   = " << mulliken_analysis << endl;
      if(mo_vectors_analysis.first) {
        cout << "  mo_vectors_analysis = [" << std::boolalpha << mo_vectors_analysis.first;
        cout << "," << mo_vectors_analysis.second << "]" << endl;
      }
      cout << " }" << endl;
    }
    cout << "}" << endl;
  }
};

class CDOptions: public Options {
public:
  CDOptions() = default;
  CDOptions(Options o): Options(o) {
    diagtol      = 1e-5;
    write_cv     = false;
    write_vcount = 5000;
    // At most 8*ao CholVec's. For vast majority cases, this is way
    // more than enough. For very large basis, it can be increased.
    max_cvecs_factor = 12;
    itilesize        = 1000;
  }

  double diagtol;
  int    itilesize;
  int    max_cvecs_factor;
  // write to disk after every count number of vectors are computed.
  // enabled only if write_cv=true and nbf>1000
  bool write_cv;
  int  write_vcount;

  void print() {
    std::cout << std::defaultfloat;
    cout << endl << "CD Options" << endl;
    cout << "{" << endl;
    cout << std::boolalpha << " debug            = " << debug << endl;
    cout << std::boolalpha << " write_cv         = " << write_cv << endl;
    cout << " diagtol          = " << diagtol << endl;
    cout << " write_vcount     = " << write_vcount << endl;
    cout << " itilesize        = " << itilesize << endl;
    cout << " max_cvecs_factor = " << max_cvecs_factor << endl;
    cout << "}" << endl;
  }
};

class GWOptions: public Options {
public:
  GWOptions() = default;
  GWOptions(Options o): Options(o) {
    cdbasis   = "";
    ngl       = 200;
    noqpa     = 1;
    noqpb     = 1;
    nvqpa     = 0;
    nvqpb     = 0;
    ieta      = 0.01;
    evgw      = false;
    evgw0     = false;
    core      = false;
    maxnewton = 15;
    maxev     = 0;
    minres    = false;
    method    = "sdgw";
  }

  int    ngl;       // Number of Gauss-Legendre quadrature points
  int    noqpa;     // Number of Occupied QP energies ALPHA spi
  int    noqpb;     // Number of Occupied QP energies BETA spin
  int    nvqpa;     // Number of Virtual QP energies ALPHA spin
  int    nvqpb;     // Number of Virtual QP energies BETA spin
  double ieta;      // Imaginary infinitesimal value
  bool   evgw;      // Do an evGW self-consistent calculation
  bool   evgw0;     // Do an evGW_0 self-consistent calculation
  bool   core;      // If true, start counting from the core
  int    maxnewton; // Maximum number of Newton steps per QP
  int    maxev;     // Maximum number of evGW or evGW_0 cycles
  bool   minres;    // Use MINRES solver
  string method;    // Method to use [cdgw,sdgw]
  string cdbasis;   // Name of the CD basis set

  void print() {
    std::cout << std::defaultfloat;
    cout << endl << "GW Options" << endl;
    cout << "{" << endl;
    cout << " ngl       = " << ngl << endl;
    cout << " noqpa     = " << noqpa << endl;
    cout << " noqpb     = " << noqpb << endl;
    cout << " nvqpa     = " << nvqpa << endl;
    cout << " nvqp/b     = " << nvqpb << endl;
    cout << " ieta      = " << ieta << endl;
    cout << " maxnewton = " << maxnewton << endl;
    cout << " maxev     = " << maxev << endl;
    cout << " method    = " << method << endl;
    cout << " cdbasis   = " << cdbasis << endl;
    print_bool(" evgw     ", evgw);
    print_bool(" evgw0    ", evgw0);
    print_bool(" core     ", core);
    print_bool(" minres   ", minres);
    print_bool(" debug    ", debug);
    cout << "}" << endl;
  }
};

class CCSDOptions: public Options {
public:
  CCSDOptions() = default;
  CCSDOptions(Options o): Options(o) {
    threshold      = 1e-6;
    force_tilesize = false;
    tilesize       = 40;
    ndiis          = 5;
    lshift         = 0;
    nactive        = 0;
    ccsd_maxiter   = 50;
    freeze_core    = 0;
    freeze_virtual = 0;
    balance_tiles  = true;
    profile_ccsd   = false;

    writet       = false;
    writev       = false;
    writet_iter  = ndiis;
    readt        = false;
    computeTData = false;

    localize      = false;
    skip_dlpno    = false;
    keep_npairs   = 1;
    max_pnos      = 1;
    dlpno_dfbasis = "";
    TCutEN        = 0.97;
    TCutPNO       = 1.0e-6;
    TCutPre       = 1.0e-3;
    TCutPairs     = 1.0e-3;
    TCutDO        = 1e-2;
    TCutDOij      = 1e-7;
    TCutDOPre     = 3e-2;

    cache_size     = 8;
    skip_ccsd      = false;
    ccsdt_tilesize = 32;

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
  }

  int  tilesize;
  bool force_tilesize;
  int  ndiis;
  int  writet_iter;
  bool readt, writet, writev, gf_restart, gf_ip, gf_ea, gf_os, gf_cs, gf_itriples, gf_profile,
    balance_tiles, computeTData;
  bool                    profile_ccsd;
  double                  lshift;
  double                  threshold;
  bool                    ccsd_diagnostics{false};
  std::pair<bool, double> tamplitudes{false, 0.05};
  std::vector<int>        cc_rdm{};

  int  nactive;
  int  ccsd_maxiter;
  bool freeze_atomic{false};
  int  freeze_core;
  int  freeze_virtual;

  // RT-EOMCC
  int    pcore;      // pth core orbital
  int    ntimesteps; // number of time steps
  int    rt_microiter;
  double rt_threshold;
  double rt_step_size;
  double rt_multiplier;
  double secent_x; // secent scale factor
  double h_red;    // time-step reduction factor
  double h_inc;    // time-step increase factor
  double h_max;    // max time-step factor

  // CCSD(T)
  bool skip_ccsd;
  int  cache_size;
  int  ccsdt_tilesize;

  // DLPNO
  bool             localize;
  bool             skip_dlpno;
  int              max_pnos;
  size_t           keep_npairs;
  double           TCutEN;
  double           TCutPNO;
  double           TCutPre;
  double           TCutPairs;
  double           TCutDO;
  double           TCutDOij;
  double           TCutDOPre;
  std::string      dlpno_dfbasis;
  std::vector<int> doubles_opt_eqns;

  // EOM
  int    eom_nroots;
  int    eom_microiter;
  string eom_type;
  double eom_threshold;

  // GF
  int    gf_p_oi_range;
  int    gf_ndiis;
  int    gf_ngmres;
  int    gf_maxiter;
  double gf_eta;
  double gf_lshift;
  bool   gf_preconditioning;
  int    gf_nprocs_poi;
  double gf_damping_factor;
  // double gf_omega;
  double              gf_threshold;
  double              gf_omega_min_ip;
  double              gf_omega_max_ip;
  double              gf_omega_min_ip_e;
  double              gf_omega_max_ip_e;
  double              gf_omega_min_ea;
  double              gf_omega_max_ea;
  double              gf_omega_min_ea_e;
  double              gf_omega_max_ea_e;
  double              gf_omega_delta;
  double              gf_omega_delta_e;
  int                 gf_extrapolate_level;
  int                 gf_analyze_level;
  int                 gf_analyze_num_omega;
  std::vector<double> gf_analyze_omega;
  // Force processing of specified orbitals first
  std::vector<size_t> gf_orbitals;

  void print() {
    std::cout << std::defaultfloat;
    cout << endl << "CCSD Options" << endl;
    cout << "{" << endl;
    cout << " cache_size           = " << cache_size << endl;
    cout << " ccsdt_tilesize       = " << ccsdt_tilesize << endl;

    cout << " ndiis                = " << ndiis << endl;
    cout << " threshold            = " << threshold << endl;
    cout << " tilesize             = " << tilesize << endl;
    if(nactive > 0) cout << " nactive              = " << nactive << endl;
    if(pcore > 0) cout << " pcore                = " << pcore << endl;
    cout << " ccsd_maxiter         = " << ccsd_maxiter << endl;
    print_bool(" freeze_atomic        ", freeze_atomic);
    cout << " freeze_core          = " << freeze_core << endl;
    cout << " freeze_virtual       = " << freeze_virtual << endl;
    if(lshift != 0) cout << " lshift               = " << lshift << endl;
    if(gf_nprocs_poi > 0) cout << " gf_nprocs_poi        = " << gf_nprocs_poi << endl;
    print_bool(" readt               ", readt);
    print_bool(" writet              ", writet);
    print_bool(" writev              ", writev);
    // print_bool(" computeTData        ", computeTData);
    cout << " writet_iter          = " << writet_iter << endl;
    print_bool(" profile_ccsd        ", profile_ccsd);
    print_bool(" balance_tiles       ", balance_tiles);

    if(!dlpno_dfbasis.empty()) cout << " dlpno_dfbasis        = " << dlpno_dfbasis << endl;
    if(!doubles_opt_eqns.empty()) {
      cout << " doubles_opt_eqns        = [";
      for(auto x: doubles_opt_eqns) cout << x << ",";
      cout << "]" << endl;
    }

    if(!ext_data_path.empty()) { cout << " ext_data_path   = " << ext_data_path << endl; }

    if(eom_nroots > 0) {
      cout << " eom_nroots           = " << eom_nroots << endl;
      cout << " eom_microiter        = " << eom_microiter << endl;
      cout << " eom_threshold        = " << eom_threshold << endl;
    }

    if(gf_p_oi_range > 0) {
      cout << " gf_p_oi_range        = " << gf_p_oi_range << endl;
      print_bool(" gf_ip               ", gf_ip);
      print_bool(" gf_ea               ", gf_ea);
      print_bool(" gf_os               ", gf_os);
      print_bool(" gf_cs               ", gf_cs);
      print_bool(" gf_restart          ", gf_restart);
      print_bool(" gf_profile          ", gf_profile);
      print_bool(" gf_itriples         ", gf_itriples);
      cout << " gf_ndiis             = " << gf_ndiis << endl;
      cout << " gf_ngmres            = " << gf_ngmres << endl;
      cout << " gf_maxiter           = " << gf_maxiter << endl;
      cout << " gf_eta               = " << gf_eta << endl;
      cout << " gf_lshift            = " << gf_lshift << endl;
      cout << " gf_preconditioning   = " << gf_preconditioning << endl;
      cout << " gf_damping_factor    = " << gf_damping_factor << endl;

      // cout << " gf_omega       = " << gf_omega << endl;
      cout << " gf_threshold         = " << gf_threshold << endl;
      cout << " gf_omega_min_ip      = " << gf_omega_min_ip << endl;
      cout << " gf_omega_max_ip      = " << gf_omega_max_ip << endl;
      cout << " gf_omega_min_ip_e    = " << gf_omega_min_ip_e << endl;
      cout << " gf_omega_max_ip_e    = " << gf_omega_max_ip_e << endl;
      cout << " gf_omega_min_ea      = " << gf_omega_min_ea << endl;
      cout << " gf_omega_max_ea      = " << gf_omega_max_ea << endl;
      cout << " gf_omega_min_ea_e    = " << gf_omega_min_ea_e << endl;
      cout << " gf_omega_max_ea_e    = " << gf_omega_max_ea_e << endl;
      cout << " gf_omega_delta       = " << gf_omega_delta << endl;
      cout << " gf_omega_delta_e     = " << gf_omega_delta_e << endl;
      if(!gf_orbitals.empty()) {
        cout << " gf_orbitals        = [";
        for(auto x: gf_orbitals) cout << x << ",";
        cout << "]" << endl;
      }
      if(gf_analyze_level > 0) {
        cout << " gf_analyze_level     = " << gf_analyze_level << endl;
        cout << " gf_analyze_num_omega = " << gf_analyze_num_omega << endl;
        cout << " gf_analyze_omega     = [";
        for(auto x: gf_analyze_omega) cout << x << ",";
        cout << "]" << endl;
      }
      if(gf_extrapolate_level > 0)
        cout << " gf_extrapolate_level = " << gf_extrapolate_level << endl;
    }

    print_bool(" debug               ", debug);
    cout << "}" << endl;
  }
};

class FCIOptions: public Options {
public:
  FCIOptions() = default;
  FCIOptions(Options o): Options(o) {
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

  int         nalpha{}, nbeta{}, nactive{}, ninactive{};
  std::string job, expansion;

  // MCSCF
  bool   enable_diis;
  int    max_macro_iter, diis_start_iter, diis_nkeep, ci_max_subspace;
  double max_orbital_step, orb_grad_tol_mcscf, ci_res_tol, ci_matel_tol;

  // FCIDUMP

  // PRINT
  bool print_davidson{}, print_ci{}, print_mcscf{}, print_diis{}, print_asci_search{};
  std::pair<bool, double> print_state_char{false, 1e-2};
};

class OptionsMap {
public:
  OptionsMap() = default;
  Options     options;
  SCFOptions  scf_options;
  CDOptions   cd_options;
  GWOptions   gw_options;
  CCSDOptions ccsd_options;
  FCIOptions  fci_options;
  TaskOptions task_options;
};

inline void to_upper(std::string& str) {
  std::transform(str.begin(), str.end(), str.begin(), ::toupper);
}
inline void to_lower(std::string& str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
}
inline string str_upper(std::string str) {
  string ustr = str;
  std::transform(ustr.begin(), ustr.end(), ustr.begin(), ::toupper);
  return ustr;
}

template<typename T>
void parse_option(T& val, json j, string key, bool optional = true) {
  if(j.contains(key)) val = j[key].get<T>();
  else if(!optional) {
    tamm_terminate("INPUT FILE ERROR: " + key + " not specified. Please specify the " + key +
                   " option!");
  }
}

inline int get_atomic_number(std::string element_symbol) {
  int Z = -1;
  for(const auto& e: libint2::chemistry::get_element_info()) {
    auto es = element_symbol;
    es.erase(std::remove_if(std::begin(es), std::end(es), [](auto d) { return std::isdigit(d); }),
             es.end());
    if(strequal_case(e.symbol, es)) {
      Z = e.Z;
      break;
    }
  }
  if(Z == -1) {
    tamm_terminate("INPUT FILE ERROR: element symbol \"" + element_symbol + "\" is not recognized");
  }

  return Z;
}

inline std::tuple<Options, SCFOptions, CDOptions, GWOptions, CCSDOptions, FCIOptions, TaskOptions>
parse_json(json& jinput, std::vector<ECAtom>& ec_atoms) {
  Options options;

  parse_option<string>(options.geom_units, jinput["geometry"], "units");

  const std::vector<string> valid_sections{"geometry", "basis", "common", "SCF",  "CD",
                                           "GW",       "CC",    "FCI",    "TASK", "comments"};
  for(auto& el: jinput.items()) {
    if(std::find(valid_sections.begin(), valid_sections.end(), el.key()) == valid_sections.end())
      tamm_terminate("INPUT FILE ERROR: Invalid section [" + el.key() + "] in the input file");
  }

  // basis
  json jbasis = jinput["basis"];
  parse_option<string>(options.basis, jbasis, "basisset", false);
  parse_option<string>(options.basisfile, jbasis, "basisfile");
  // parse_option<string>(options.gaussian_type, jbasis, "gaussian_type");
  parse_option<string>(options.dfbasis, jbasis, "df_basisset");

  to_lower(options.basis);
  const std::vector<string> valid_basis{"comments", "basisset", "basisfile",
                                        /*"gaussian_type",*/ "df_basisset", "atom_basis"};
  for(auto& el: jbasis.items()) {
    if(std::find(valid_basis.begin(), valid_basis.end(), el.key()) == valid_basis.end())
      tamm_terminate("INPUT FILE ERROR: Invalid basis section option [" + el.key() +
                     "] in the input file");
  }

  json                               jatom_basis = jinput["basis"]["atom_basis"];
  std::map<std::string, std::string> atom_basis_map;
  for(auto& [element_symbol, basis_string]: jatom_basis.items()) {
    atom_basis_map[element_symbol] = basis_string;
  }

  for(size_t i = 0; i < ec_atoms.size(); i++) {
    const auto es     = ec_atoms[i].esymbol; // element_symbol
    ec_atoms[i].basis = options.basis;
    if(atom_basis_map.find(es) != atom_basis_map.end()) ec_atoms[i].basis = atom_basis_map[es];
  }

  // common
  json jcommon = jinput["common"];
  parse_option<int>(options.maxiter, jcommon, "maxiter");
  parse_option<bool>(options.debug, jcommon, "debug");
  parse_option<string>(options.output_file_prefix, jcommon, "output_file_prefix");

  const std::vector<string> valid_common{"comments", "maxiter", "debug", "output_file_prefix"};
  for(auto& el: jcommon.items()) {
    if(std::find(valid_common.begin(), valid_common.end(), el.key()) == valid_common.end()) {
      tamm_terminate("INPUT FILE ERROR: Invalid common section option [" + el.key() +
                     "] in the input file");
    }
  }

  SCFOptions  scf_options(options);
  CDOptions   cd_options(options);
  GWOptions   gw_options(options);
  CCSDOptions ccsd_options(options);
  FCIOptions  fci_options(options);
  TaskOptions task_options(options);

  // SCF
  // clang-format off
  json jscf = jinput["SCF"];
  parse_option<int>   (scf_options.charge          , jscf, "charge");
  parse_option<int>   (scf_options.multiplicity    , jscf, "multiplicity");
  parse_option<double>(scf_options.lshift          , jscf, "lshift");
  parse_option<double>(scf_options.tol_int         , jscf, "tol_int");
  parse_option<double>(scf_options.tol_sch         , jscf, "tol_sch");
  parse_option<double>(scf_options.tol_lindep      , jscf, "tol_lindep");
  parse_option<double>(scf_options.conve           , jscf, "conve");
  parse_option<double>(scf_options.convd           , jscf, "convd");
  parse_option<int>   (scf_options.diis_hist       , jscf, "diis_hist");
  parse_option<bool>  (scf_options.force_tilesize  , jscf, "force_tilesize");
  parse_option<uint32_t>(scf_options.AO_tilesize   , jscf, "tilesize");
  parse_option<uint32_t>(scf_options.dfAO_tilesize , jscf, "df_tilesize");
  parse_option<int>   (scf_options.damp            , jscf, "damp");
  parse_option<int>   (scf_options.writem          , jscf, "writem");
  parse_option<int>   (scf_options.nnodes          , jscf, "nnodes");
  parse_option<bool>  (scf_options.restart         , jscf, "restart");
  parse_option<bool>  (scf_options.noscf           , jscf, "noscf");
  parse_option<bool>  (scf_options.debug           , jscf, "debug");
  parse_option<string>(scf_options.moldenfile      , jscf, "moldenfile");
  parse_option<string>(scf_options.scf_type        , jscf, "scf_type");
  parse_option<string>(scf_options.xc_grid_type    , jscf, "xc_grid_type");
  parse_option<std::vector<string>>(scf_options.xc_type, jscf, "xc_type");
  parse_option<int>   (scf_options.n_lindep        , jscf, "n_lindep");
  parse_option<int>   (scf_options.restart_size    , jscf, "restart_size");
  parse_option<int>   (scf_options.scalapack_nb    , jscf, "scalapack_nb");
  parse_option<int>   (scf_options.scalapack_np_row, jscf, "scalapack_np_row");
  parse_option<int>   (scf_options.scalapack_np_col, jscf, "scalapack_np_col");
  parse_option<string>(scf_options.ext_data_path   , jscf, "ext_data_path");
  parse_option<std::vector<double>>(scf_options.qed_omegas , jscf, "qed_omegas");
  parse_option<std::vector<double>>(scf_options.qed_lambdas, jscf, "qed_lambdas");
  parse_option<std::vector<double>>(scf_options.qed_volumes, jscf, "qed_volumes");
  parse_option<std::vector<std::vector<double>>>(scf_options.qed_polvecs, jscf, "qed_polvecs");


  json jscf_guess  = jscf["guess"];
  json jguess_atom_options = jscf_guess["atom_options"];
  parse_option<bool>  (scf_options.sad, jscf_guess, "sad");

  for(auto& [element_symbol, atom_opt]: jguess_atom_options.items()) {
    scf_options.guess_atom_options[element_symbol] = atom_opt;
  }  

  json jscf_analysis = jscf["PRINT"];
  parse_option<bool> (scf_options.mos_txt          , jscf_analysis, "mos_txt");
  parse_option<bool> (scf_options.mulliken_analysis, jscf_analysis, "mulliken");
  parse_option<std::pair<bool, double>>(scf_options.mo_vectors_analysis, jscf_analysis, "mo_vectors");
  // clang-format on
  std::string riscf_str;
  parse_option<string>(riscf_str, jscf, "riscf");
  if(riscf_str == "J") scf_options.riscf = 1;
  else if(riscf_str == "K") scf_options.riscf = 2;
  // clang-format off
  const std::vector<string> valid_scf{"charge", "multiplicity", "lshift", "tol_int", "tol_sch",
    "tol_lindep", "conve", "convd", "diis_hist","force_tilesize","tilesize","df_tilesize",
    "damp","writem","nnodes","restart","noscf","moldenfile", "guess",
    "debug","scf_type","xc_type", "xc_grid_type", "n_lindep","restart_size","scalapack_nb","riscf",
    "scalapack_np_row","scalapack_np_col","ext_data_path","PRINT",
    "qed_omegas","qed_lambdas","qed_volumes","qed_polvecs","comments"};
  // clang-format on

  for(auto& el: jscf.items()) {
    if(std::find(valid_scf.begin(), valid_scf.end(), el.key()) == valid_scf.end())
      tamm_terminate("INPUT FILE ERROR: Invalid SCF option [" + el.key() + "] in the input file");
  }
  if(scf_options.nnodes < 1 || scf_options.nnodes > 100) {
    tamm_terminate("INPUT FILE ERROR: SCF option nnodes should be a number between 1 and 100");
  }
  {
    auto xc_grid_str = scf_options.xc_grid_type;
    xc_grid_str.erase(remove_if(xc_grid_str.begin(), xc_grid_str.end(), isspace),
                      xc_grid_str.end());
    scf_options.xc_grid_type = xc_grid_str;
    std::transform(xc_grid_str.begin(), xc_grid_str.end(), xc_grid_str.begin(), ::tolower);
    if(xc_grid_str != "fine" && xc_grid_str != "ultrafine" && xc_grid_str != "superfine")
      tamm_terminate(
        "INPUT FILE ERROR: SCF option xc_grid_type should be one of [Fine, UltraFine, SuperFine]");
  }

  // CD
  json jcd = jinput["CD"];
  parse_option<bool>(cd_options.debug, jcd, "debug");
  parse_option<int>(cd_options.itilesize, jcd, "itilesize");
  parse_option<double>(cd_options.diagtol, jcd, "diagtol");
  parse_option<bool>(cd_options.write_cv, jcd, "write_cv");
  parse_option<int>(cd_options.write_vcount, jcd, "write_vcount");
  parse_option<int>(cd_options.max_cvecs_factor, jcd, "max_cvecs");

  parse_option<string>(cd_options.ext_data_path, jcd, "ext_data_path");

  const std::vector<string> valid_cd{"comments", "debug",        "itilesize", "diagtol",
                                     "write_cv", "write_vcount", "max_cvecs", "ext_data_path"};
  for(auto& el: jcd.items()) {
    if(std::find(valid_cd.begin(), valid_cd.end(), el.key()) == valid_cd.end())
      tamm_terminate("INPUT FILE ERROR: Invalid CD option [" + el.key() + "] in the input file");
  }

  // GW
  // clang-format off
  json jgw = jinput["GW"];
  parse_option<bool>  (gw_options.debug,     jgw, "debug");
  parse_option<int>   (gw_options.ngl,       jgw, "ngl");
  parse_option<int>   (gw_options.noqpa,     jgw, "noqpa");
  parse_option<int>   (gw_options.noqpb,     jgw, "noqpb");
  parse_option<int>   (gw_options.nvqpa,     jgw, "nvqpa");
  parse_option<int>   (gw_options.nvqpb,     jgw, "nvqpb");
  parse_option<double>(gw_options.ieta,      jgw, "ieta");
  parse_option<bool>  (gw_options.evgw,      jgw, "evgw");
  parse_option<bool>  (gw_options.evgw0,     jgw, "evgw0");
  parse_option<bool>  (gw_options.core,      jgw, "core");
  parse_option<int>   (gw_options.maxev,     jgw, "maxev");
  parse_option<bool>  (gw_options.minres,    jgw, "minres");
  parse_option<int>   (gw_options.maxnewton, jgw, "maxnewton");
  parse_option<string>(gw_options.method,    jgw, "method");
  parse_option<string>(gw_options.cdbasis,   jgw, "cdbasis");
  parse_option<string>(gw_options.ext_data_path, jgw, "ext_data_path");
  // clang-format on
  std::vector<string> gwlist{"cdgw", "sdgw", "CDGW", "SDGW"};
  if(std::find(std::begin(gwlist), std::end(gwlist), string(gw_options.method)) == std::end(gwlist))
    tamm_terminate("INPUT FILE ERROR: GW method can only be one of [sdgw,cdgw]");

  // CC
  json                      jcc = jinput["CC"];
  const std::vector<string> valid_cc{
    "CCSD(T)",      "DLPNO",          "EOMCCSD",  "RT-EOMCC",     "GFCCSD",        "comments",
    "threshold",    "force_tilesize", "tilesize", "computeTData", "lshift",        "ndiis",
    "ccsd_maxiter", "freeze",         "PRINT",    "readt",        "writet",        "writev",
    "writet_iter",  "debug",          "nactive",  "profile_ccsd", "balance_tiles", "ext_data_path"};
  for(auto& el: jcc.items()) {
    if(std::find(valid_cc.begin(), valid_cc.end(), el.key()) == valid_cc.end())
      tamm_terminate("INPUT FILE ERROR: Invalid CC option [" + el.key() + "] in the input file");
  }
  // clang-format off
  parse_option<int>   (ccsd_options.ndiis         , jcc, "ndiis");
  parse_option<int>   (ccsd_options.nactive       , jcc, "nactive");
  parse_option<int>   (ccsd_options.ccsd_maxiter  , jcc, "ccsd_maxiter");
  parse_option<double>(ccsd_options.lshift        , jcc, "lshift");
  parse_option<double>(ccsd_options.threshold     , jcc, "threshold");
  parse_option<int>   (ccsd_options.tilesize      , jcc, "tilesize");
  parse_option<bool>  (ccsd_options.debug         , jcc, "debug");
  parse_option<bool>  (ccsd_options.readt         , jcc, "readt");
  parse_option<bool>  (ccsd_options.writet        , jcc, "writet");
  parse_option<bool>  (ccsd_options.writev        , jcc, "writev");
  parse_option<int>   (ccsd_options.writet_iter   , jcc, "writet_iter");
  parse_option<bool>  (ccsd_options.balance_tiles , jcc, "balance_tiles");
  parse_option<bool>  (ccsd_options.profile_ccsd  , jcc, "profile_ccsd");
  parse_option<bool>  (ccsd_options.force_tilesize, jcc, "force_tilesize");
  parse_option<string>(ccsd_options.ext_data_path , jcc, "ext_data_path");
  parse_option<bool>  (ccsd_options.computeTData  , jcc, "computeTData");

  json jcc_print = jcc["PRINT"];
  parse_option<bool> (ccsd_options.ccsd_diagnostics, jcc_print, "ccsd_diagnostics");
  parse_option<std::vector<int>>(ccsd_options.cc_rdm, jcc_print, "rdm");
  parse_option<std::pair<bool, double>>(ccsd_options.tamplitudes, jcc_print, "tamplitudes");

  json jcc_freeze = jcc["freeze"];
  parse_option<bool>(ccsd_options.freeze_atomic,  jcc_freeze, "atomic");
  parse_option<int> (ccsd_options.freeze_core,    jcc_freeze, "core");
  parse_option<int> (ccsd_options.freeze_virtual, jcc_freeze, "virtual");

  //RT-EOMCC
  json jrt_eom = jcc["RT-EOMCC"];
  parse_option<int>   (ccsd_options.pcore        , jrt_eom, "pcore");
  parse_option<int>   (ccsd_options.ntimesteps   , jrt_eom, "ntimesteps");
  parse_option<int>   (ccsd_options.rt_microiter , jrt_eom, "rt_microiter");
  parse_option<double>(ccsd_options.rt_threshold , jrt_eom, "rt_threshold");
  parse_option<double>(ccsd_options.rt_step_size , jrt_eom, "rt_step_size");
  parse_option<double>(ccsd_options.rt_multiplier, jrt_eom, "rt_multiplier");
  parse_option<double>(ccsd_options.secent_x     , jrt_eom, "secent_x");
  parse_option<double>(ccsd_options.h_red        , jrt_eom, "h_red");
  parse_option<double>(ccsd_options.h_inc        , jrt_eom, "h_inc");
  parse_option<double>(ccsd_options.h_max        , jrt_eom, "h_max");

  // DLPNO
  json jdlpno = jcc["DLPNO"];
  parse_option<int>   (ccsd_options.max_pnos     , jdlpno, "max_pnos");
  parse_option<size_t>(ccsd_options.keep_npairs  , jdlpno, "keep_npairs");
  parse_option<bool>  (ccsd_options.localize     , jdlpno, "localize");
  parse_option<bool>  (ccsd_options.skip_dlpno   , jdlpno, "skip_dlpno");
  parse_option<string>(ccsd_options.dlpno_dfbasis, jdlpno, "df_basisset");
  parse_option<double>(ccsd_options.TCutDO       , jdlpno, "TCutDO");
  parse_option<double>(ccsd_options.TCutEN       , jdlpno, "TCutEN");
  parse_option<double>(ccsd_options.TCutPNO      , jdlpno, "TCutPNO");
  parse_option<double>(ccsd_options.TCutPre      , jdlpno, "TCutPre");
  parse_option<double>(ccsd_options.TCutPairs    , jdlpno, "TCutPairs");
  parse_option<double>(ccsd_options.TCutDOij     , jdlpno, "TCutDOij");
  parse_option<double>(ccsd_options.TCutDOPre    , jdlpno, "TCutDOPre");
  parse_option<std::vector<int>>(ccsd_options.doubles_opt_eqns, jdlpno, "doubles_opt_eqns");

  json jccsd_t = jcc["CCSD(T)"];
  parse_option<bool>(ccsd_options.skip_ccsd    , jccsd_t, "skip_ccsd");
  parse_option<int>(ccsd_options.cache_size    , jccsd_t, "cache_size");
  parse_option<int>(ccsd_options.ccsdt_tilesize, jccsd_t, "ccsdt_tilesize");

  json jeomccsd = jcc["EOMCCSD"];
  parse_option<int>   (ccsd_options.eom_nroots   , jeomccsd, "eom_nroots");
  parse_option<int>   (ccsd_options.eom_microiter, jeomccsd, "eom_microiter");
  parse_option<string>(ccsd_options.eom_type     , jeomccsd, "eom_type");
  parse_option<double>(ccsd_options.eom_threshold, jeomccsd, "eom_threshold");
  // clang-format on
  std::vector<string> etlist{"right", "left", "RIGHT", "LEFT"};
  if(std::find(std::begin(etlist), std::end(etlist), string(ccsd_options.eom_type)) ==
     std::end(etlist))
    tamm_terminate("INPUT FILE ERROR: EOMCC type can only be one of [left,right]");

  json jgfcc = jcc["GFCCSD"];
  // clang-format off
  parse_option<bool>(ccsd_options.gf_ip      , jgfcc, "gf_ip");
  parse_option<bool>(ccsd_options.gf_ea      , jgfcc, "gf_ea");
  parse_option<bool>(ccsd_options.gf_os      , jgfcc, "gf_os");
  parse_option<bool>(ccsd_options.gf_cs      , jgfcc, "gf_cs");
  parse_option<bool>(ccsd_options.gf_restart , jgfcc, "gf_restart");
  parse_option<bool>(ccsd_options.gf_profile , jgfcc, "gf_profile");
  parse_option<bool>(ccsd_options.gf_itriples, jgfcc, "gf_itriples");

  parse_option<int>   (ccsd_options.gf_ndiis            , jgfcc, "gf_ndiis");
  parse_option<int>   (ccsd_options.gf_ngmres           , jgfcc, "gf_ngmres");
  parse_option<int>   (ccsd_options.gf_maxiter          , jgfcc, "gf_maxiter");
  parse_option<int>   (ccsd_options.gf_nprocs_poi       , jgfcc, "gf_nprocs_poi");
  parse_option<double>(ccsd_options.gf_damping_factor   , jgfcc, "gf_damping_factor");
  parse_option<double>(ccsd_options.gf_eta              , jgfcc, "gf_eta");
  parse_option<double>(ccsd_options.gf_lshift           , jgfcc, "gf_lshift");
  parse_option<bool>  (ccsd_options.gf_preconditioning  , jgfcc, "gf_preconditioning");
  parse_option<double>(ccsd_options.gf_threshold        , jgfcc, "gf_threshold");
  parse_option<double>(ccsd_options.gf_omega_min_ip     , jgfcc, "gf_omega_min_ip");
  parse_option<double>(ccsd_options.gf_omega_max_ip     , jgfcc, "gf_omega_max_ip");
  parse_option<double>(ccsd_options.gf_omega_min_ip_e   , jgfcc, "gf_omega_min_ip_e");
  parse_option<double>(ccsd_options.gf_omega_max_ip_e   , jgfcc, "gf_omega_max_ip_e");
  parse_option<double>(ccsd_options.gf_omega_min_ea     , jgfcc, "gf_omega_min_ea");
  parse_option<double>(ccsd_options.gf_omega_max_ea     , jgfcc, "gf_omega_max_ea");
  parse_option<double>(ccsd_options.gf_omega_min_ea_e   , jgfcc, "gf_omega_min_ea_e");
  parse_option<double>(ccsd_options.gf_omega_max_ea_e   , jgfcc, "gf_omega_max_ea_e");
  parse_option<double>(ccsd_options.gf_omega_delta      , jgfcc, "gf_omega_delta");
  parse_option<double>(ccsd_options.gf_omega_delta_e    , jgfcc, "gf_omega_delta_e");
  parse_option<int>   (ccsd_options.gf_extrapolate_level, jgfcc, "gf_extrapolate_level");
  parse_option<int>   (ccsd_options.gf_analyze_level    , jgfcc, "gf_analyze_level");
  parse_option<int>   (ccsd_options.gf_analyze_num_omega, jgfcc, "gf_analyze_num_omega");
  parse_option<int>   (ccsd_options.gf_p_oi_range       , jgfcc, "gf_p_oi_range");

  parse_option<std::vector<size_t>>(ccsd_options.gf_orbitals     , jgfcc, "gf_orbitals");
  parse_option<std::vector<double>>(ccsd_options.gf_analyze_omega, jgfcc, "gf_analyze_omega");
  // clang-format on
  if(ccsd_options.gf_p_oi_range != 0) {
    if(ccsd_options.gf_p_oi_range != 1 && ccsd_options.gf_p_oi_range != 2)
      tamm_terminate("gf_p_oi_range can only be one of 1 or 2");
  }

  // FCI
  json                      jfci = jinput["FCI"];
  const std::vector<string> valid_fci{"nalpha", "nbeta",     "nactive", "ninactive", "comments",
                                      "job",    "expansion", "FCIDUMP", "MCSCF",     "PRINT"};

  for(auto& el: jfci.items()) {
    if(std::find(valid_fci.begin(), valid_fci.end(), el.key()) == valid_fci.end())
      tamm_terminate("INPUT FILE ERROR: Invalid FCI option [" + el.key() + "] in the input file");
  }

  // clang-format off
  parse_option<int>(fci_options.nalpha,       jfci, "nalpha");
  parse_option<int>(fci_options.nbeta,        jfci, "nbeta");
  parse_option<int>(fci_options.nactive,      jfci, "nactive");
  parse_option<int>(fci_options.ninactive,    jfci, "ninactive");
  parse_option<string>(fci_options.job,       jfci, "job");
  parse_option<string>(fci_options.expansion, jfci, "expansion");

  // MCSCF
  json jmcscf = jfci["MCSCF"];
  parse_option<int>   (fci_options.max_macro_iter,     jmcscf, "max_macro_iter");
  parse_option<double>(fci_options.max_orbital_step,   jmcscf, "max_orbital_step");
  parse_option<double>(fci_options.orb_grad_tol_mcscf, jmcscf, "orb_grad_tol_mcscf");
  parse_option<bool>  (fci_options.enable_diis,        jmcscf, "enable_diis");
  parse_option<int>   (fci_options.diis_start_iter,    jmcscf, "diis_start_iter");
  parse_option<int>   (fci_options.diis_nkeep,         jmcscf, "diis_nkeep");
  parse_option<double>(fci_options.ci_res_tol,         jmcscf, "ci_res_tol");
  parse_option<int>   (fci_options.ci_max_subspace,    jmcscf, "ci_max_subspace");
  parse_option<double>(fci_options.ci_matel_tol,       jmcscf, "ci_matel_tol");

  // FCIDUMP
  json jfcidump = jfci["FCIDUMP"];
  // parse_option<bool>(fci_options.fcidump,    jfcidump, "fcidump");

  // PRINT
  json jprint = jfci["PRINT"];
  parse_option<bool>(fci_options.print_davidson,    jprint, "davidson");
  parse_option<bool>(fci_options.print_ci,          jprint, "ci");
  parse_option<bool>(fci_options.print_mcscf,       jprint, "mcscf");
  parse_option<bool>(fci_options.print_diis,        jprint, "diis");
  parse_option<bool>(fci_options.print_asci_search, jprint, "asci_search");
  parse_option<std::pair<bool, double>>(fci_options.print_state_char, jprint, "state_char");
  // clang-format on

  // TASK
  json                      jtask = jinput["TASK"];
  const std::vector<string> valid_tasks{
    "sinfo",  "scf",         "fci",          "fcidump",   "mp2",        "gw",      "cd_2e",
    "cc2",    "dlpno_ccsd",  "dlpno_ccsd_t", "ducc",      "ccsd",       "ccsd_sf", "ccsd_t",
    "gfccsd", "ccsd_lambda", "eom_ccsd",     "rteom_cc2", "rteom_ccsd", "comments"};

  for(auto& el: jtask.items()) {
    if(std::find(valid_tasks.begin(), valid_tasks.end(), el.key()) == valid_tasks.end())
      tamm_terminate("INPUT FILE ERROR: Invalid TASK option [" + el.key() + "] in the input file");
  }

  // clang-format off
  parse_option<bool>(task_options.sinfo       , jtask, "sinfo");
  parse_option<bool>(task_options.scf         , jtask, "scf");
  parse_option<bool>(task_options.mp2         , jtask, "mp2");
  parse_option<bool>(task_options.gw          , jtask, "gw");
  parse_option<bool>(task_options.cc2         , jtask, "cc2");
  parse_option<bool>(task_options.fci         , jtask, "fci");  
  parse_option<bool>(task_options.fcidump     , jtask, "fcidump");  
  parse_option<bool>(task_options.cd_2e       , jtask, "cd_2e");
  parse_option<bool>(task_options.ducc        , jtask, "ducc");
  parse_option<bool>(task_options.ccsd        , jtask, "ccsd");
  parse_option<bool>(task_options.ccsd_sf     , jtask, "ccsd_sf");
  parse_option<bool>(task_options.ccsd_t      , jtask, "ccsd_t");
  parse_option<bool>(task_options.ccsd_lambda , jtask, "ccsd_lambda");
  parse_option<bool>(task_options.eom_ccsd    , jtask, "eom_ccsd");
  parse_option<bool>(task_options.rteom_cc2   , jtask, "rteom_cc2");
  parse_option<bool>(task_options.rteom_ccsd  , jtask, "rteom_ccsd");
  parse_option<bool>(task_options.gfccsd      , jtask, "gfccsd");

  parse_option<std::pair<bool, std::string>>(task_options.dlpno_ccsd  , jtask, "dlpno_ccsd");
  parse_option<std::pair<bool, std::string>>(task_options.dlpno_ccsd_t, jtask, "dlpno_ccsd_t");
  // clang-format on

  // options.print();
  // scf_options.print();
  // ccsd_options.print();

  return std::make_tuple(options, scf_options, cd_options, gw_options, ccsd_options, fci_options,
                         task_options);
}

class json_sax_no_exception: public nlohmann::detail::json_sax_dom_parser<json> {
public:
  json_sax_no_exception(json& j): nlohmann::detail::json_sax_dom_parser<json>(j, false) {}

  bool parse_error(std::size_t position, const std::string& last_token, const json::exception& ex) {
    if(ProcGroup::world_rank() == 0) {
      std::cerr << std::endl << ex.what() << std::endl << "last read: " << last_token << std::endl;
    }
    return false;
  }
};

inline std::tuple<OptionsMap, json> parse_input(std::istream& is) {
  const double angstrom_to_bohr = 1.8897259878858;

  json                  jinput;
  json_sax_no_exception jsax(jinput);
  bool                  parse_result = json::sax_parse(is, &jsax);
  if(!parse_result) tamm_terminate("Error parsing input file");

  std::vector<string> geometry;
  parse_option<std::vector<string>>(geometry, jinput["geometry"], "coordinates", false);
  size_t natom = geometry.size();

  std::vector<ECAtom> ec_atoms(natom);
  std::vector<Atom>   atoms(natom);
  std::vector<string> geom_bohr(natom);

  for(size_t i = 0; i < natom; i++) {
    std::string        line = geometry[i];
    std::istringstream iss(line);
    std::string        element_symbol;
    double             x, y, z;
    iss >> element_symbol >> x >> y >> z;
    geom_bohr[i] = element_symbol;

    const auto Z = get_atomic_number(element_symbol);

    atoms[i].atomic_number = Z;
    atoms[i].x             = x;
    atoms[i].y             = y;
    atoms[i].z             = z;

    ec_atoms[i].atom    = atoms[i];
    ec_atoms[i].esymbol = element_symbol;
  }

  auto [options, scf_options, cd_options, gw_options, ccsd_options, fci_options, task_options] =
    parse_json(jinput, ec_atoms);

  json jgeom_bohr;
  bool nw_units_bohr = true;
  // Done parsing input file
  {
    // If geometry units specified are angstrom, convert to bohr
    if(options.geom_units == "angstrom") nw_units_bohr = false;

    if(!nw_units_bohr) {
      // .xyz files report Cartesian coordinates in angstroms;
      // convert to bohr
      for(auto i = 0U; i < atoms.size(); i++) {
        std::ostringstream ss_bohr;
        atoms[i].x *= angstrom_to_bohr;
        atoms[i].y *= angstrom_to_bohr;
        atoms[i].z *= angstrom_to_bohr;
        ss_bohr << std::setw(3) << std::left << geom_bohr[i] << " " << std::right << std::setw(14)
                << std::fixed << std::setprecision(10) << atoms[i].x << " " << std::right
                << std::setw(14) << std::fixed << std::setprecision(10) << atoms[i].y << " "
                << std::right << std::setw(14) << std::fixed << std::setprecision(10) << atoms[i].z;
        geom_bohr[i]     = ss_bohr.str();
        ec_atoms[i].atom = atoms[i];
      }
      jgeom_bohr["geometry_bohr"] = geom_bohr;
    }
  }

  if(ProcGroup::world_rank() == 0) {
    jinput.erase(std::remove(jinput.begin(), jinput.end(), nullptr), jinput.end());

    // std::cout << jinput.dump(2) << std::endl;
    // if(!nw_units_bohr) {
    //   std::cout << "Geometry in bohr as follows:" << std::endl;
    //   std::cout << jgeom_bohr.dump(2) << std::endl;
    // }
    // options.print();
    // scf_options.print();
    // cd_options.print();
    // ccsd_options.print();
  }

  OptionsMap options_map;
  options_map.options          = options;
  options_map.options.atoms    = atoms;
  options_map.options.ec_atoms = ec_atoms;

  options_map.scf_options  = scf_options;
  options_map.cd_options   = cd_options;
  options_map.gw_options   = gw_options;
  options_map.ccsd_options = ccsd_options;
  options_map.fci_options  = fci_options;
  options_map.task_options = task_options;

  return std::make_tuple(options_map, jinput);
}
