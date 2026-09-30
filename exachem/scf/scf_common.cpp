/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_common.hpp"

template<typename T>
std::vector<size_t> exachem::scf::SCFUtil::sort_indexes(const std::vector<T>& v, bool reverse) {
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&v](size_t x, size_t y) { return v[x] < v[y]; });

  if(reverse) std::reverse(idx.begin(), idx.end());

  return idx;
}

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
template<typename T>
std::tuple<size_t, double, double>
exachem::scf::SCFUtil::gensqrtinv(ExecutionContext& ec, ChemEnv& chem_env, SCFData& scf_data,
                                  exachem::scf::TAMMTensors<T>& ttensors, bool symmetric,
                                  double threshold) {
  SystemData& sys_data    = chem_env.sys_data;
  SCFOptions& scf_options = chem_env.ioptions.scf_options;

  Scheduler sch{ec};
  // auto world = ec.pg().comm();
  const int world_rank = ec.pg().rank().value();
  const int world_size = ec.pg().size().value();

  int64_t       n_cond{}, n_illcond{};
  double        condition_number{}, result_condition_number{};
  const int64_t N = sys_data.nbf_orig;

  // TODO: avoid eigen matrices
  Matrix         X;
  std::vector<T> eps(N);

  // Eigen decompose S -> V s V**T, row i of V = eigenvector i
#if defined(USE_SCALAPACK)
  // V is TAMM-dense (for tensor_block below) and lives on the ScaLAPACK grid's ranks
  const tamm::ScalapackGrid& grid = tamm::find_scalapack_grid(ec);
  scf_data.tN_bc                  = grid.index_space(sys_data.nbf_orig);
  Tensor<T> V                     = grid.allocate_dense<T>(scf_data.tN_bc, scf_data.tN_bc);
#else
  Tensor<T> V{scf_data.tAO, scf_data.tAO};
  sch.allocate(V).execute();
#endif
  tamm::eigensolve(ec, ttensors.S1, V, eps, ec.exhw());

  typename std::vector<T>::iterator first_above_thresh;
  if(world_rank == 0) {
    // condition_number = std::min(
    //   eps.back() / std::max( eps.front(), std::numeric_limits<double>::min() ),
    //   1.       / std::numeric_limits<double>::epsilon()
    // );

    // const auto threshold = eps.back() / max_condition_number;
    first_above_thresh =
      std::find_if(eps.begin(), eps.end(), [&](const auto& x) { return x >= threshold; });
    result_condition_number = eps.back() / *first_above_thresh;

    n_illcond = std::distance(eps.begin(), first_above_thresh);
    n_cond    = N - n_illcond;

    if(n_illcond > 0) {
      std::cout << std::endl
                << "WARNING: Found " << n_illcond << " linear dependencies" << std::endl;
      cout << std::defaultfloat << "First eigen value above tol_lindep = " << *first_above_thresh
           << endl;
      std::cout << "The overlap matrix has " << n_illcond
                << " vectors deemed linearly dependent with eigenvalues:" << std::endl;

      for(int64_t i = 0; i < n_illcond; i++)
        cout << std::defaultfloat << i + 1 << ": " << eps[i] << endl;
    }
  }

  if(world_size > 1) ec.pg().broadcast(&n_illcond, 0);
  n_cond = N - n_illcond;

  sys_data.n_lindep = n_illcond;
  sys_data.nbf      = n_cond;

  scf_data.tAO_ortho =
    TiledIndexSpace{IndexSpace{range((size_t) sys_data.nbf)}, scf_options.AO_tilesize};

  Tensor<T> X_tmp{scf_data.tAO, scf_data.tAO_ortho};
  Tensor<T> eps_tamm{scf_data.tAO_ortho};
  Tensor<T>::allocate(&ec, X_tmp, eps_tamm);

  if(world_rank == 0) {
    std::vector<T> epso(first_above_thresh, eps.end());
    std::transform(epso.begin(), epso.end(), epso.begin(),
                   [](auto& c) { return 1.0 / std::sqrt(c); });
    tamm::vector_to_tamm_tensor(eps_tamm, epso);
  }
  ec.pg().barrier();

#if defined(USE_SCALAPACK)
  scf_data.tNortho_bc = grid.index_space(sys_data.nbf);
  ttensors.X_alpha    = grid.allocate<T>(scf_data.tN_bc, scf_data.tNortho_bc);
#else
  ttensors.X_alpha = {scf_data.tAO, scf_data.tAO_ortho};
  sch.allocate(ttensors.X_alpha).execute();
#endif

#if defined(USE_SCALAPACK)
  if(grid.participates()) {
    Tensor<T> X_t = tensor_block(V, {n_illcond, 0}, {N, N}, {1, 0});
    tamm::from_dense_tensor(X_t, X_tmp);
    Tensor<T>::deallocate(V, X_t);
  }
#else
  if(world_rank == 0) {
    Matrix Vm     = tamm_to_eigen_matrix(V);
    Matrix V_cond = Vm.block(n_illcond, 0, N - n_illcond, N);
    Vm.resize(0, 0);
    X.resize(N, n_cond);
    X = V_cond.transpose();
    V_cond.resize(0, 0);
    eigen_to_tamm_tensor(X_tmp, X);
    X.resize(0, 0);
  }
  ec.pg().barrier();
  sch.deallocate(V).execute();
#endif

  Tensor<T> X_comp{scf_data.tAO, scf_data.tAO_ortho};
  auto [mu, nu] = scf_data.tAO.labels<2>("all");
  auto mu_o     = scf_data.tAO_ortho.label("all");

#if defined(USE_SCALAPACK)
  sch.allocate(X_comp).execute();
#else
  X_comp = ttensors.X_alpha;
#endif

  sch(X_comp(mu, mu_o) = X_tmp(mu, mu_o) * eps_tamm(mu_o)).deallocate(X_tmp, eps_tamm).execute();

#if defined(USE_SCALAPACK)
  grid.to_block_cyclic(X_comp, ttensors.X_alpha);
#endif

  if(sys_data.is_cuscf) {
    ttensors.Xm1 = {scf_data.tAO, scf_data.tAO_ortho};
    // clang-format off
    sch.allocate(ttensors.Xm1)
      (ttensors.Xm1(mu, mu_o) = ttensors.S1(mu, nu) * X_comp(nu, mu_o)).execute();
    // clang-format on
  }

#if defined(USE_SCALAPACK)
  sch.deallocate(X_comp).execute();
#endif

  return std::make_tuple(size_t(n_cond), condition_number, result_condition_number);
}
template<typename T>
std::tuple<Matrix, size_t, double, double> exachem::scf::SCFUtil::gensqrtinv_atscf(
  ExecutionContext& ec, const ChemEnv& chem_env, const SCFData& scf_data, Tensor<T> S1,
  TiledIndexSpace& tao_atom, bool symmetric, double threshold) {
  const SCFOptions& scf_options = chem_env.ioptions.scf_options;

  Scheduler sch{ec};
  // auto world = ec.pg().comm();
  const int world_rank = ec.pg().rank().value();
  const int world_size = ec.pg().size().value();

  int64_t       n_cond{}, n_illcond{};
  double        condition_number{}, result_condition_number{};
  const int64_t N = tao_atom.index_space().num_indices();

  // TODO: avoid eigen matrices
  Matrix         X, V;
  std::vector<T> eps(N);

  if(world_rank == 0) {
    // Eigen decompose S -> VsV**T
    V.resize(N, N);
    tamm_to_eigen_tensor(S1, V);
    tamm::eigensolve(N, V.data(), eps);
  }

  typename std::vector<T>::iterator first_above_thresh;
  if(world_rank == 0) {
    // condition_number = std::min(
    //   eps.back() / std::max( eps.front(), std::numeric_limits<double>::min() ),
    //   1.       / std::numeric_limits<double>::epsilon()
    // );

    // const auto threshold = eps.back() / max_condition_number;
    first_above_thresh =
      std::find_if(eps.begin(), eps.end(), [&](const auto& x) { return x >= threshold; });
    result_condition_number = eps.back() / *first_above_thresh;

    n_illcond = std::distance(eps.begin(), first_above_thresh);
    n_cond    = N - n_illcond;

    if(n_illcond > 0) {
      std::cout << std::endl
                << "WARNING: Found " << n_illcond << " linear dependencies" << std::endl;
      cout << std::defaultfloat << "First eigen value above tol_lindep = " << *first_above_thresh
           << endl;
      std::cout << "The overlap matrix has " << n_illcond
                << " vectors deemed linearly dependent with eigenvalues:" << std::endl;

      for(int64_t i = 0; i < n_illcond; i++)
        cout << std::defaultfloat << i + 1 << ": " << eps[i] << endl;
    }
  }

  if(world_size > 1) { ec.pg().broadcast(&n_illcond, 0); }
  n_cond = N - n_illcond;

  if(world_rank == 0) {
    // auto* V_cond = Vbuf + n_illcond * N;
    Matrix V_cond = V.block(n_illcond, 0, N - n_illcond, N);
    V.resize(0, 0);
    X.resize(N, n_cond);
    X = V_cond.transpose();
    V_cond.resize(0, 0);
  }

  if(world_rank == 0) {
    // Form canonical X/Xinv
    for(auto i = 0; i < n_cond; ++i) {
      const double srt = std::sqrt(*(first_above_thresh + i));

      // X is row major...
      auto* X_col = X.data() + i;
      // auto* Xinv_col = Xinv.data() + i;

      blas::scal(N, 1. / srt, X_col, n_cond);
      // blas::scal( N, srt, Xinv_col, n_cond );
    }

  } // compute on root

  TiledIndexSpace tAO_atom_ortho{IndexSpace{range((size_t) n_cond)}, scf_options.AO_tilesize};

  Tensor<T> x_tamm{tao_atom, tAO_atom_ortho};
  sch.allocate(x_tamm).execute();

  if(world_rank == 0) eigen_to_tamm_tensor(x_tamm, X);
  ec.pg().barrier();

  X = tamm_to_eigen_matrix(x_tamm);
  sch.deallocate(x_tamm).execute();

  return std::make_tuple(X, size_t(n_cond), condition_number, result_condition_number);
}

template<typename T>
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
exachem::scf::SCFUtil::gather_task_vectors(ExecutionContext& ec, const std::vector<int>& s1vec,
                                           const std::vector<int>& s2vec,
                                           const std::vector<int>& ntask_vec) {
  const int rank   = ec.pg().rank().value();
  const int nranks = ec.pg().size().value();

  std::vector<int> s1_count(nranks);
  std::vector<int> s2_count(nranks);
  std::vector<int> nt_count(nranks);

  const int s1vec_size = static_cast<int>(s1vec.size());
  const int s2vec_size = static_cast<int>(s2vec.size());
  const int ntvec_size = static_cast<int>(ntask_vec.size());

  // Root gathers number of elements at each rank.
  ec.pg().gather(&s1vec_size, s1_count.data(), 0);
  ec.pg().gather(&s2vec_size, s2_count.data(), 0);
  ec.pg().gather(&ntvec_size, nt_count.data(), 0);

  // Displacements in the receive buffer for GATHERV
  std::vector<int> disps_s1(nranks);
  std::vector<int> disps_s2(nranks);
  std::vector<int> disps_nt(nranks);
  for(int i = 0; i < nranks; i++) {
    disps_s1[i] = (i > 0) ? (disps_s1[i - 1] + s1_count[i - 1]) : 0;
    disps_s2[i] = (i > 0) ? (disps_s2[i - 1] + s2_count[i - 1]) : 0;
    disps_nt[i] = (i > 0) ? (disps_nt[i - 1] + nt_count[i - 1]) : 0;
  }

  // Allocate vectors to gather data at root
  std::vector<int> s1_all;
  std::vector<int> s2_all;
  std::vector<int> ntasks_all;
  if(rank == 0) {
    s1_all.resize(disps_s1[nranks - 1] + s1_count[nranks - 1]);
    s2_all.resize(disps_s2[nranks - 1] + s2_count[nranks - 1]);
    ntasks_all.resize(disps_nt[nranks - 1] + nt_count[nranks - 1]);
  }

  // Gather at root
  ec.pg().gatherv(s1vec.data(), s1vec_size, s1_all.data(), s1_count.data(), disps_s1.data(), 0);
  ec.pg().gatherv(s2vec.data(), s2vec_size, s2_all.data(), s2_count.data(), disps_s2.data(), 0);
  ec.pg().gatherv(ntask_vec.data(), ntvec_size, ntasks_all.data(), nt_count.data(), disps_nt.data(),
                  0);

  EXPECTS(s1_all.size() == s2_all.size());
  EXPECTS(s1_all.size() == ntasks_all.size());
  return std::make_tuple(s1_all, s2_all, ntasks_all);
}

template std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
exachem::scf::SCFUtil::gather_task_vectors<double>(ExecutionContext&       ec,
                                                   const std::vector<int>& s1vec,
                                                   const std::vector<int>& s2vec,
                                                   const std::vector<int>& ntask_vec);

template std::vector<size_t>
exachem::scf::SCFUtil::sort_indexes<double>(const std::vector<double>& v, bool reverse);
template std::tuple<Matrix, size_t, double, double> exachem::scf::SCFUtil::gensqrtinv_atscf<double>(
  ExecutionContext& ec, const ChemEnv& chem_env, const SCFData& scf_data, Tensor<double> S1,
  TiledIndexSpace& tao_atom, bool symmetric, double threshold);
template std::tuple<size_t, double, double>
exachem::scf::SCFUtil::gensqrtinv<double>(ExecutionContext& ec, ChemEnv& chem_env,
                                          SCFData& scf_data, TAMMTensors<double>& ttensors,
                                          bool symmetric, double threshold);
