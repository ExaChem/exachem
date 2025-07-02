/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/constants.hpp"
#include "exachem/common/ecatom.hpp"
// #include "exachem/common/txt_utils.hpp"
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

class PrintOptions {
protected:
  template<typename T>
  static void print_vec(const std::string& label, const std::vector<T>& vec, int width = 20) {
    if(!vec.empty()) {
      std::cout << "  " << std::setw(width) << label << " = [";
      for(const auto& x: vec) std::cout << x << ",";
      std::cout << "\b]" << std::endl;
    }
  }

  template<typename T>
  static void print_vec2d(const std::string& label, const std::vector<std::vector<T>>& vec2d,
                          int width = 20) {
    if(!vec2d.empty()) {
      std::cout << "  " << std::setw(width) << label << " = [";
      for(const auto& x: vec2d) {
        std::cout << "[";
        for(const auto& y: x) std::cout << y << ",";
        std::cout << "\b],";
      }
      std::cout << "\b]" << std::endl;
    }
  }

  // Stream precision is typically a positive integer, so using -1 signals “do nothing”
  template<typename T>
  static void print_option(const std::string& label, const T& value, int width = 20,
                           int precision = -1) {
    std::cout << "  " << std::setw(width) << std::left << label << "= ";
    if constexpr(std::is_same_v<T, bool>)
      std::cout << std::boolalpha << value << std::noboolalpha << std::endl;
    else if constexpr(std::is_floating_point_v<T>)
      if(precision > 0) std::cout << std::setprecision(precision) << value << std::endl;
      else std::cout << value << std::endl;
    else std::cout << value << std::endl;
  }

  template<typename T, typename U>
  static void print_pair(const std::string& label, const std::pair<T, U>& p, int width = 18) {
    std::cout << "  " << std::setw(width) << std::left << label << "= [";
    if constexpr(std::is_same_v<T, bool>)
      std::cout << std::boolalpha << p.first << std::noboolalpha;
    else std::cout << p.first;
    std::cout << ", ";
    if constexpr(std::is_same_v<U, bool>)
      std::cout << std::boolalpha << p.second << std::noboolalpha;
    else std::cout << p.second;
    std::cout << "]" << std::endl;
  }
};

class CommonOptions: public PrintOptions {
public:
  bool         debug{false};
  int          maxiter{100};
  std::string  basis{"sto-3g"};
  std::string  dfbasis{};
  std::string  gaussian_type{"spherical"};
  std::string  geom_units{"angstrom"};
  int          natoms_max{30}; // max natoms for geometry analysis
  std::string  file_prefix{};
  std::string  output_dir{};
  std::string  ext_data_path{};
  virtual void print();
};

class DPlotOptions: public CommonOptions {
public:
  bool        cube{false};
  std::string density{"total"}; // or spin
  int         orbitals{0};      // highest occupied orbitals
};

class SCFOptions: public CommonOptions {
public:
  int      charge{0};
  int      multiplicity{1};
  double   lshift{0};        // level shift factor, +ve value b/w 0 and 1
  double   tol_int{1e-22};   // tolerance for integral primitive screening
  double   tol_sch{1e-12};   // tolerance for schwarz screening
  double   tol_lindep{1e-5}; // tolerance for linear dependencies
  double   conve{1e-8};      // energy convergence
  double   convd{1e-7};      // density convergence
  int      diis_hist{10};    // number of diis history entries
  uint32_t AO_tilesize{30};
  uint32_t dfAO_tilesize{50};
  bool     restart{false}; // Read movecs from disk
  bool     noscf{false};   // only recompute energy from movecs
  bool     sad{true};
  bool     direct_df{false};
  bool     snK{false};
  int  restart_size{2000}; // read/write orthogonalizer, schwarz, etc matrices when N>=restart_size
  int  scalapack_nb{256};
  int  nnodes{1};
  int  scalapack_np_row{0};
  int  scalapack_np_col{0};
  bool molden{false};
  std::string         moldenfile{""};
  int                 n_lindep{0};
  int                 writem{1};
  int                 damp{100}; // density mixing parameter
  std::string         scf_type{"restricted"};
  std::string         xc_grid_type{"UltraFine"};
  std::string         xc_pruning_scheme{"Robust"};
  std::string         xc_rad_quad{"MK"};
  std::string         xc_weight_scheme{"SSF"};
  std::string         xc_exec_space{"Host"};
  std::string         xc_lb_kernel{"Default"};
  std::string         xc_mw_kernel{"Default"};
  std::string         xc_int_kernel{"Default"};
  std::string         xc_red_kernel{"Default"};
  std::string         xc_lwd_kernel{"Default"};
  std::pair<int, int> xc_radang_size{0, 0};
  int                 xc_batch_size{2048};
  double              xc_basis_tol{1e-8};
  double              xc_snK_etol{1e-10};
  double              xc_snK_ktol{1e-10};

  std::map<std::string, std::tuple<int, int>> guess_atom_options;

  std::vector<std::string> xc_type;
  // mos_txt: write lcao, mo transformed core H, fock, and v2 to disk as text files.
  bool                             mos_txt{false};
  bool                             mulliken_analysis{false};
  std::pair<bool, double>          mo_vectors_analysis{false, 0.15};
  std::vector<double>              qed_omegas{};
  std::vector<double>              qed_volumes{};
  std::vector<double>              qed_lambdas{};
  std::vector<std::vector<double>> qed_polvecs{};
  void                             print() override;
};

class CCSDOptions: public CommonOptions {
private:
  void initialize();

public:
  CCSDOptions(): CommonOptions() { initialize(); }

  int  tilesize;
  int  ndiis;
  int  writet_iter;
  bool readt, writet, gf_restart, gf_ip, gf_ea, gf_os, gf_cs, gf_itriples, gf_profile,
    balance_tiles;
  bool                    profile_ccsd;
  double                  lshift;
  double                  threshold;
  bool                    ccsd_diagnostics{false};
  std::pair<bool, double> tamplitudes{false, 0.05};
  std::vector<int>        cc_rdm{};

  int    nactive_oa, nactive_ob, nactive_va, nactive_vb;
  int    ducc_lvl, qflow_cycles;
  double qflow_threshold;
  int    ccsd_maxiter;
  bool   freeze_atomic{false};
  int    freeze_core;
  int    freeze_virtual;

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
  double           TCutTNO;
  double           TCutPre;
  double           TCutPairs;
  double           TCutDO;
  double           TCutDOij;
  double           TCutDOPre;
  std::string      dlpno_dfbasis;
  std::vector<int> doubles_opt_eqns;

  // EOM
  int         eom_nroots;
  int         eom_microiter;
  std::string eom_type;
  double      eom_threshold;

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
  void                print() override;
};

class CDOptions: public CommonOptions {
public:
  CDOptions() = default;
  double diagtol{1e-5};
  int    itilesize{1000};
  int    max_cvecs_factor{12};

  // skip cholesky and use the value specified as the cholesky vector count.
  std::pair<bool, int> skip_cd{false, 100};
  // enabled only if set to true and nbf > 1000
  // write to disk after every count number of vectors are computed.
  std::pair<bool, int> write_cv{false, 5000};
  void                 print() override;
};

class FCIOptions: public CommonOptions {
private:
  void initialize();

public:
  FCIOptions(): CommonOptions() { initialize(); }
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

class GWOptions: public CommonOptions {
public:
  int         ngl{200};       // Number of Gauss-Legendre quadrature points
  int         noqpa{1};       // Number of Occupied QP energies ALPHA spi
  int         noqpb{1};       // Number of Occupied QP energies BETA spin
  int         nvqpa{0};       // Number of Virtual QP energies ALPHA spin
  int         nvqpb{0};       // Number of Virtual QP energies BETA spin
  double      ieta{0.01};     // Imaginary infinitesimal value
  bool        evgw{false};    // Do an evGW self-consistent calculation
  bool        evgw0{false};   // Do an evGW_0 self-consistent calculation
  bool        core{false};    // If true, start counting from the core
  int         maxnewton{15};  // Maximum number of Newton steps per QP
  int         maxev{0};       // Maximum number of evGW or evGW_0 cycles
  bool        minres{false};  // Use MINRES solver
  std::string method{"sdgw"}; // Method to use [cdgw,sdgw]
  std::string cdbasis{""};    // Name of the CD basis set
  void        print() override;
};

class TaskOptions: public CommonOptions {
public:
  bool                         sinfo{false};
  bool                         scf{false};
  bool                         mp2{false};
  bool                         gw{false};
  bool                         cc2{false};
  bool                         fci{false};
  bool                         fcidump{false};
  bool                         cd_2e{false};
  bool                         ccsd{false};
  bool                         ccsd_sf{false};
  bool                         ccsd_t{false};
  bool                         ccsdt{false};
  bool                         ccsd_lambda{false};
  bool                         eom_ccsd{false};
  bool                         rteom_cc2{false};
  bool                         rteom_ccsd{false};
  bool                         gfccsd{false};
  std::pair<bool, std::string> ducc{false, "default"};

  std::pair<bool, std::string> dlpno_ccsd{false, ""};
  std::pair<bool, std::string> dlpno_ccsd_t{false, ""};
  std::vector<std::string>     operation{"energy"};
  void                         print() override;
};

class ECOptions {
public:
  ECOptions() = default;

  CommonOptions common_options;
  DPlotOptions  dplot_options;
  SCFOptions    scf_options;
  CDOptions     cd_options;
  GWOptions     gw_options;
  CCSDOptions   ccsd_options;
  FCIOptions    fci_options;
  TaskOptions   task_options;
};
