/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cctype>

#include "common/cutils.hpp"
#include "common/misc.hpp"
#include "common/molden.hpp"

#if defined(USE_GAUXC)
#include <gauxc/xc_integrator.hpp>
#include <gauxc/xc_integrator/impl.hpp>
#endif

#include <libecpint.hpp>

using namespace tamm;
using libint2::Atom;
using std::cerr;
using std::cout;
using std::endl;
using std::string;

using TensorType       = double;
using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
using shellpair_data_t =
  std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>; // in same order as
                                                                 // shellpair_list_t

// Tensor<TensorType> vxc_tamm; // TODO: cleanup

struct SCFVars {
  // diis
  int                      idiis        = 0;
  bool                     switch_diis  = false;
  double                   exc          = 0.0;
  double                   eqed         = 0.0;
  bool                     do_dens_fit  = false;
  bool                     do_load_bal  = false;
  bool                     lshift_reset = false;
  bool                     lshift       = 0;
  libecpint::ECPIntegrator ecp_factory;

  // AO spaces
  tamm::TiledIndexSpace   tAO, tAO_ortho, tAO_occ_a, tAO_occ_b, tAOt; // tAO_ld
  std::vector<tamm::Tile> AO_tiles;
  std::vector<tamm::Tile> AO_opttiles;
  std::vector<size_t>     shell_tile_map;

  // AO spaces for BC representation
  tamm::TiledIndexSpace tN_bc;
  tamm::TiledIndexSpace tNortho_bc;

  // AO labels
  // tamm::TiledIndexLabel mu_ld, nu_ld;
  tamm::TiledIndexLabel mu, nu, ku;
  tamm::TiledIndexLabel mup, nup, kup;

  tamm::TiledIndexLabel mu_oa, nu_oa;
  tamm::TiledIndexLabel mu_ob, nu_ob;

  // DF spaces
  libint2::BasisSet     dfbs;
  tamm::IndexSpace      dfAO;
  std::vector<Tile>     dfAO_tiles;
  std::vector<Tile>     dfAO_opttiles;
  std::vector<size_t>   df_shell_tile_map;
  tamm::TiledIndexSpace tdfAO, tdfAOt;
  tamm::TiledIndexSpace tdfCocc;
  tamm::TiledIndexLabel dCocc_til;

  // DF labels
  tamm::TiledIndexLabel d_mu, d_nu, d_ku;
  tamm::TiledIndexLabel d_mup, d_nup, d_kup;

  // shellpair list
  shellpair_list_t obs_shellpair_list;        // shellpair list for OBS
  shellpair_list_t dfbs_shellpair_list;       // shellpair list for DFBS
  shellpair_list_t minbs_shellpair_list;      // shellpair list for minBS
  shellpair_list_t obs_shellpair_list_atom;   // shellpair list for OBS for specfied atom
  shellpair_list_t minbs_shellpair_list_atom; // shellpair list for minBS for specfied atom

  // shellpair data
  shellpair_data_t obs_shellpair_data;        // shellpair data for OBS
  shellpair_data_t dfbs_shellpair_data;       // shellpair data for DFBS
  shellpair_data_t minbs_shellpair_data;      // shellpair data for minBS
  shellpair_data_t obs_shellpair_data_atom;   // shellpair data for OBS for specfied atom
  shellpair_data_t minbs_shellpair_data_atom; // shellpair data for minBS for specfied atom
};

struct EigenTensors {
  Matrix C_alpha, C_beta, C_occ; // allocated only on rank 0 when scalapack is not used
  Matrix VXC_alpha, VXC_beta;    // allocated only on rank 0 when DFT is enabled
  Matrix G_alpha, D_alpha;       // allocated on all ranks for 4c HF, only on rank 0 otherwise.
  Matrix G_beta, D_beta; // allocated on all ranks for 4c HF, only D_beta on rank 0 otherwise.
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    taskmap; // on all ranks for 4c HF only
};

struct TAMMTensors {
  std::vector<Tensor<TensorType>> ehf_tamm_hist;

  std::vector<Tensor<TensorType>> diis_hist;
  std::vector<Tensor<TensorType>> fock_hist;
  std::vector<Tensor<TensorType>> D_hist;

  std::vector<Tensor<TensorType>> diis_beta_hist;
  std::vector<Tensor<TensorType>> fock_beta_hist;
  std::vector<Tensor<TensorType>> D_beta_hist;

  Tensor<TensorType> ehf_tamm;
  Tensor<TensorType> ehf_tmp;
  Tensor<TensorType> ehf_beta_tmp;

  Tensor<TensorType> H1; // core hamiltonian
  Tensor<TensorType> S1; // overlap ints
  Tensor<TensorType> T1; // kinetic ints
  Tensor<TensorType> V1; // nuclear ints

  Tensor<TensorType> X_alpha;
  Tensor<TensorType> F_alpha; // H1+F_alpha_tmp
  Tensor<TensorType> F_beta;
  Tensor<TensorType> F_alpha_tmp; // computed via call to compute_2bf(...)
  Tensor<TensorType> F_beta_tmp;
  Tensor<TensorType> F_BC; // block-cyclic Fock matrix used in the scalapack code path
  // not allocated, shell tiled. tensor structure used to identify shell blocks in compute_2bf
  Tensor<TensorType> F_dummy;
  Tensor<TensorType> VXC_alpha;
  Tensor<TensorType> VXC_beta;

  Tensor<TensorType> C_alpha;
  Tensor<TensorType> C_beta;
  Tensor<TensorType> C_occ_a;
  Tensor<TensorType> C_occ_b;
  Tensor<TensorType> C_occ_aT;
  Tensor<TensorType> C_occ_bT;

  Tensor<TensorType> C_alpha_BC;
  Tensor<TensorType> C_beta_BC;

  Tensor<TensorType> D_alpha;
  Tensor<TensorType> D_beta;
  Tensor<TensorType> D_diff;
  Tensor<TensorType> D_last_alpha;
  Tensor<TensorType> D_last_beta;

  Tensor<TensorType> FD_alpha;
  Tensor<TensorType> FDS_alpha;
  Tensor<TensorType> FD_beta;
  Tensor<TensorType> FDS_beta;

  // DF
  Tensor<TensorType> xyK; // n,n,ndf
  Tensor<TensorType> Zxy; // ndf,n,n
};

// DENSITY FITTING
struct DFFockEngine {
  const libint2::BasisSet& obs;
  const libint2::BasisSet& dfbs;
  DFFockEngine(const libint2::BasisSet& _obs, const libint2::BasisSet& _dfbs):
    obs(_obs), dfbs(_dfbs) {}
};

template<typename TensorType>
void compute_1body_ints(ExecutionContext& ec, const SCFVars&, Tensor<TensorType>& tensor1e,
                        std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                        libint2::Operator otype);

Matrix compute_soad(const std::vector<libint2::Atom>& atoms);

// computes norm of shell-blocks of A
Matrix compute_shellblock_norm(const libint2::BasisSet& obs, const Matrix& A);

std::tuple<shellpair_list_t, shellpair_data_t>
compute_shellpairs(const libint2::BasisSet& bs1, const libint2::BasisSet& bs2 = libint2::BasisSet(),
                   double threshold = 1e-16);

template<libint2::Operator Kernel = libint2::Operator::coulomb>
Matrix compute_schwarz_ints(ExecutionContext& ec, const SCFVars& scf_vars,
                            const libint2::BasisSet& bs1,
                            const libint2::BasisSet& bs2       = libint2::BasisSet(),
                            bool                     use_2norm = false, // use infty norm by default
                            typename libint2::operator_traits<Kernel>::oper_params_type params =
                              libint2::operator_traits<Kernel>::default_params());

std::tuple<size_t, double, double> gensqrtinv(ExecutionContext& ec, SystemData& sys_data,
                                              SCFVars& scf_vars, ScalapackInfo& scalapack_info,
                                              TAMMTensors& ttensors, bool symmetric = false,
                                              double threshold = 1e-5);

inline size_t max_nprim(const libint2::BasisSet& shells) {
  size_t n = 0;
  for(auto shell: shells) n = std::max(shell.nprim(), n);
  return n;
}

inline int max_l(const libint2::BasisSet& shells) {
  int l = 0;
  for(auto shell: shells)
    for(auto c: shell.contr) l = std::max(c.l, l);
  return l;
}

inline std::vector<size_t> map_shell_to_basis_function(const libint2::BasisSet& shells) {
  std::vector<size_t> result;
  result.reserve(shells.size());

  size_t n = 0;
  for(auto shell: shells) {
    result.push_back(n);
    n += shell.size();
  }

  return result;
}

inline std::vector<size_t> map_basis_function_to_shell(const libint2::BasisSet& shells) {
  std::vector<size_t> result(shells.nbf());

  auto shell2bf = map_shell_to_basis_function(shells);
  for(size_t s1 = 0; s1 != shells.size(); ++s1) {
    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1        = shells[s1].size();
    for(size_t f1 = 0; f1 != n1; ++f1) {
      const auto bf1 = f1 + bf1_first;
      result[bf1]    = s1;
    }
  }
  return result;
}

inline void recompute_tilesize(tamm::Tile& tile_size, const int N, const bool force_ts,
                               const bool rank0) {
  // heuristic to set tilesize to atleast 5% of nbf
  if(tile_size < N * 0.05 && !force_ts) {
    tile_size = std::ceil(N * 0.05);
    if(rank0) cout << "***** Reset tilesize to nbf*5% = " << tile_size << endl;
  }
}

int get_nfcore(SystemData& sys_data);

BasisSetMap construct_basisset_maps(std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                                    bool is_spherical = true);

template<typename T>
Matrix read_scf_mat(std::string matfile);

template<typename T>
void write_scf_mat(Matrix& C, std::string matfile);

template<typename T>
std::vector<size_t> sort_indexes(std::vector<T>& v);

template<typename T, int ndim>
void t2e_hf_helper(const ExecutionContext& ec, tamm::Tensor<T>& ttensor, Matrix& etensor,
                   const std::string& ustr = "");

void compute_shellpair_list(const ExecutionContext& ec, const libint2::BasisSet& shells,
                            SCFVars& scf_vars);

std::tuple<int, double> compute_NRE(const ExecutionContext& ec, std::vector<libint2::Atom>& atoms);

std::tuple<std::vector<size_t>, std::vector<Tile>, std::vector<Tile>>
compute_AO_tiles(const ExecutionContext& ec, const SystemData& sys_data, libint2::BasisSet& shells,
                 const bool is_df = false);

// returns {X,X^{-1},S_condition_number_after_conditioning}, where
// X is the generalized square-root-inverse such that X.transpose() * S * X = I
// columns of Xinv is the basis conditioned such that
// the condition number of its metric (Xinv.transpose . Xinv) <
// S_condition_number_threshold
void compute_orthogonalizer(ExecutionContext& ec, SystemData& sys_data, SCFVars& scf_vars,
                            ScalapackInfo& scalapack_info, TAMMTensors& ttensors);
template<typename TensorType>
void compute_hamiltonian(ExecutionContext& ec, const SCFVars& scf_vars,
                         std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                         TAMMTensors& ttensors, EigenTensors& etensors);

template<typename TensorType>
void compute_density(ExecutionContext& ec, const SystemData& sys_data, const SCFVars& scf_vars,
                     ScalapackInfo& scalapack_info, TAMMTensors& ttensors, EigenTensors& etensors);

void scf_restart_test(const ExecutionContext& ec, const SystemData& sys_data, bool restart,
                      std::string files_prefix);

void scf_restart(ExecutionContext& ec, ScalapackInfo& scalapack_info, const SystemData& sys_data,
                 TAMMTensors& ttensors, EigenTensors& etensors, std::string files_prefix);

void rw_md_disk(ExecutionContext& ec, ScalapackInfo& scalapack_info, const SystemData& sys_data,
                TAMMTensors& ttensors, EigenTensors& etensors, std::string files_prefix,
                bool read = false);

template<typename T>
void rw_mat_disk(Tensor<T> tensor, std::string tfilename, bool profile, bool read = false);

template<typename TensorType>
double tt_trace(ExecutionContext& ec, Tensor<TensorType>& T1, Tensor<TensorType>& T2);

void print_energies(ExecutionContext& ec, TAMMTensors& ttensors, EigenTensors& etensors,
                    const SystemData& sys_data, SCFVars& scf_vars, ScalapackInfo& scalapack_info,
                    bool debug = false);

// returns {X,X^{-1},rank,A_condition_number,result_A_condition_number}, where
// X is the generalized square-root-inverse such that X.transpose() * A * X = I
//
// if symmetric is true, produce "symmetric" sqrtinv: X = U . A_evals_sqrtinv .
// U.transpose()),
// else produce "canonical" sqrtinv: X = U . A_evals_sqrtinv
// where U are eigenvectors of A
// rows and cols of symmetric X are equivalent; for canonical X the rows are
// original basis (AO),
// cols are transformed basis ("orthogonal" AO)
//
// A is conditioned to max_condition_number
std::tuple<size_t, double, double> gensqrtinv(ExecutionContext& ec, SystemData& sys_data,
                                              SCFVars& scf_vars, ScalapackInfo& scalapack_info,
                                              TAMMTensors& ttensors, bool symmetric,
                                              double threshold);

std::tuple<Matrix, size_t, double, double>
gensqrtinv_atscf(ExecutionContext& ec, SystemData& sys_data, SCFVars& scf_vars,
                 ScalapackInfo& scalapack_info, Tensor<double> S1, TiledIndexSpace& tao_atom,
                 bool symmetric, double threshold);

std::tuple<shellpair_list_t, shellpair_data_t> compute_shellpairs(const libint2::BasisSet& bs1,
                                                                  const libint2::BasisSet& _bs2,
                                                                  const double threshold);

template<libint2::Operator Kernel>
Matrix compute_schwarz_ints(ExecutionContext& ec, const SCFVars& scf_vars,
                            const libint2::BasisSet& bs1, const libint2::BasisSet& _bs2,
                            bool                                                        use_2norm,
                            typename libint2::operator_traits<Kernel>::oper_params_type params);

Matrix compute_shellblock_norm(const libint2::BasisSet& obs, const Matrix& A);

void print_mulliken(OptionsMap& options_map, libint2::BasisSet& shells, Matrix& D, Matrix& D_beta,
                    Matrix& S, bool is_uhf);

template<typename TensorType>
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
gather_task_vectors(ExecutionContext& ec, std::vector<int>& s1vec, std::vector<int>& s2vec,
                    std::vector<int>& ntask_vec);

#if defined(USE_GAUXC)

namespace gauxc_util {

GauXC::Molecule make_gauxc_molecule(const std::vector<libint2::Atom>& atoms);

GauXC::BasisSet<double> make_gauxc_basis(const libint2::BasisSet& basis);

template<typename TensorType>
TensorType compute_xcf(ExecutionContext& ec, const SystemData& sys_data, TAMMTensors& ttensors,
                       EigenTensors& etensors, GauXC::XCIntegrator<Matrix>& xc_integrator);

} // namespace gauxc_util
#endif
