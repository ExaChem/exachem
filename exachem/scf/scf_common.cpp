/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "scf/scf_common.hpp"

template<typename T>
std::vector<size_t> sort_indexes(std::vector<T>& v) {
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&v](size_t x, size_t y) { return v[x] < v[y]; });

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
std::tuple<size_t, double, double> gensqrtinv(ExecutionContext& ec, ChemEnv& chem_env,
                                              SCFVars& scf_vars, ScalapackInfo& scalapack_info,
                                              TAMMTensors& ttensors, bool symmetric,
                                              double threshold) {
  using T = TensorType;

  SystemData& sys_data    = chem_env.sys_data;
  SCFOptions& scf_options = chem_env.ioptions.scf_options;

  Scheduler sch{ec};
  // auto world = ec.pg().comm();
  int world_rank = ec.pg().rank().value();
  int world_size = ec.pg().size().value();

  int64_t       n_cond{}, n_illcond{};
  double        condition_number{}, result_condition_number{};
  const int64_t N = sys_data.nbf_orig;

  // TODO: avoid eigen matrices
  Matrix         X, V;
  std::vector<T> eps(N);

#if defined(USE_SCALAPACK)
  Tensor<T> V_sca;
  if(scalapack_info.comm != MPI_COMM_NULL) {
    blacspp::Grid*                  blacs_grid       = scalapack_info.blacs_grid.get();
    const auto&                     grid             = *blacs_grid;
    scalapackpp::BlockCyclicDist2D* blockcyclic_dist = scalapack_info.blockcyclic_dist.get();
    const tamm::Tile                mb               = blockcyclic_dist->mb();

    scf_vars.tN_bc         = TiledIndexSpace{IndexSpace{range(sys_data.nbf_orig)}, mb};
    TiledIndexSpace& tN_bc = scf_vars.tN_bc;
    Tensor<T>        S_BC{tN_bc, tN_bc};
    V_sca = {tN_bc, tN_bc};
    S_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
    V_sca.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
    Tensor<T>::allocate(&scalapack_info.ec, S_BC, V_sca);

    tamm::to_block_cyclic_tensor(ttensors.S1, S_BC);

    auto desc_lambda = [&](const int64_t M, const int64_t N) {
      auto [M_loc, N_loc] = (*blockcyclic_dist).get_local_dims(M, N);
      return (*blockcyclic_dist).descinit_noerror(M, N, M_loc);
    };

    if(grid.ipr() >= 0 and grid.ipc() >= 0) {
      auto desc_S = desc_lambda(N, N);
      auto desc_V = desc_lambda(N, N);

      /*info=*/scalapackpp::hereig(scalapackpp::Job::Vec, scalapackpp::Uplo::Lower, desc_S[2],
                                   S_BC.access_local_buf(), 1, 1, desc_S, eps.data(),
                                   V_sca.access_local_buf(), 1, 1, desc_V);
    }

    Tensor<T>::deallocate(S_BC);
  }

#else

  if(world_rank == 0) {
    // Eigen decompose S -> VsV**T
    V.resize(N, N);
    tamm_to_eigen_tensor(ttensors.S1, V);
    lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, N, V.data(), N, eps.data());
  }

#endif

  std::vector<T>::iterator first_above_thresh;
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

  scf_vars.tAO_ortho =
    TiledIndexSpace{IndexSpace{range((size_t) sys_data.nbf)}, scf_options.AO_tilesize};

  Tensor<T> X_tmp{scf_vars.tAO, scf_vars.tAO_ortho};
  Tensor<T> eps_tamm{scf_vars.tAO_ortho};
  Tensor<T>::allocate(&ec, X_tmp, eps_tamm);

  if(world_rank == 0) {
    std::vector<T> epso(first_above_thresh, eps.end());
    std::transform(epso.begin(), epso.end(), epso.begin(),
                   [](auto& c) { return 1.0 / std::sqrt(c); });
    tamm::vector_to_tamm_tensor(eps_tamm, epso);
  }
  ec.pg().barrier();

#if defined(USE_SCALAPACK)
  if(scalapack_info.comm != MPI_COMM_NULL) {
    const tamm::Tile _mb = (scalapack_info.blockcyclic_dist.get())->mb();
    scf_vars.tNortho_bc  = TiledIndexSpace{IndexSpace{range(sys_data.nbf)}, _mb};
    ttensors.X_alpha     = {scf_vars.tN_bc, scf_vars.tNortho_bc};
    ttensors.X_alpha.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
    Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.X_alpha);
  }
#else
  ttensors.X_alpha = {scf_vars.tAO, scf_vars.tAO_ortho};
  sch.allocate(ttensors.X_alpha).execute();
#endif

#if defined(USE_SCALAPACK)
  if(scalapack_info.comm != MPI_COMM_NULL) {
    Tensor<T> V_t = from_block_cyclic_tensor(V_sca);
    Tensor<T> X_t = tensor_block(V_t, {n_illcond, 0}, {N, N}, {1, 0});
    tamm::from_dense_tensor(X_t, X_tmp);
    Tensor<T>::deallocate(V_sca, V_t, X_t);
  }
#else
  if(world_rank == 0) {
    // auto* V_cond = Vbuf + n_illcond * N;
    Matrix V_cond = V.block(n_illcond, 0, N - n_illcond, N);
    V.resize(0, 0);
    X.resize(N, n_cond);
    X = V_cond.transpose();
    V_cond.resize(0, 0);
    eigen_to_tamm_tensor(X_tmp, X);
    X.resize(0, 0);
  }
  ec.pg().barrier();
#endif

  Tensor<T> X_comp{scf_vars.tAO, scf_vars.tAO_ortho};
  auto      mu   = scf_vars.tAO.label("all");
  auto      mu_o = scf_vars.tAO_ortho.label("all");

#if defined(USE_SCALAPACK)
  sch.allocate(X_comp).execute();
#else
  X_comp = ttensors.X_alpha;
#endif

  sch(X_comp(mu, mu_o) = X_tmp(mu, mu_o) * eps_tamm(mu_o)).deallocate(X_tmp, eps_tamm).execute();

#if defined(USE_SCALAPACK)
  if(scalapack_info.comm != MPI_COMM_NULL) {
    tamm::to_block_cyclic_tensor(X_comp, ttensors.X_alpha);
  }
  sch.deallocate(X_comp).execute();
#endif

  return std::make_tuple(size_t(n_cond), condition_number, result_condition_number);
}

// template<typename TensorType>
// std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
// gather_task_vectors(ExecutionContext& ec, std::vector<int>& s1vec, std::vector<int>& s2vec,
//                     std::vector<int>& ntask_vec) {
//   const int rank   = ec.pg().rank().value();
//   const int nranks = ec.pg().size().value();

// #ifdef USE_UPCXX
//   upcxx::global_ptr<int> s1_count = upcxx::new_array<int>(nranks);
//   upcxx::global_ptr<int> s2_count = upcxx::new_array<int>(nranks);
//   upcxx::global_ptr<int> nt_count = upcxx::new_array<int>(nranks);
//   assert(s1_count && s2_count && nt_count);

//   upcxx::dist_object<upcxx::global_ptr<int>> s1_count_dobj(s1_count, *ec.pg().team());
//   upcxx::dist_object<upcxx::global_ptr<int>> s2_count_dobj(s2_count, *ec.pg().team());
//   upcxx::dist_object<upcxx::global_ptr<int>> nt_count_dobj(nt_count, *ec.pg().team());
// #else
//   std::vector<int> s1_count(nranks);
//   std::vector<int> s2_count(nranks);
//   std::vector<int> nt_count(nranks);
// #endif

//   int s1vec_size = (int) s1vec.size();
//   int s2vec_size = (int) s2vec.size();
//   int ntvec_size = (int) ntask_vec.size();

//   // Root gathers number of elements at each rank.
// #ifdef USE_UPCXX
//   ec.pg().gather(&s1vec_size, s1_count_dobj.fetch(0).wait());
//   ec.pg().gather(&s2vec_size, s2_count_dobj.fetch(0).wait());
//   ec.pg().gather(&ntvec_size, nt_count_dobj.fetch(0).wait());
// #else
//   ec.pg().gather(&s1vec_size, s1_count.data(), 0);
//   ec.pg().gather(&s2vec_size, s2_count.data(), 0);
//   ec.pg().gather(&ntvec_size, nt_count.data(), 0);
// #endif

//   // Displacements in the receive buffer for GATHERV
// #ifdef USE_UPCXX
//   upcxx::global_ptr<int>                     disps_s1 = upcxx::new_array<int>(nranks);
//   upcxx::global_ptr<int>                     disps_s2 = upcxx::new_array<int>(nranks);
//   upcxx::global_ptr<int>                     disps_nt = upcxx::new_array<int>(nranks);
//   upcxx::dist_object<upcxx::global_ptr<int>> disps_s1_dobj(disps_s1, *ec.pg().team());
//   upcxx::dist_object<upcxx::global_ptr<int>> disps_s2_dobj(disps_s2, *ec.pg().team());
//   upcxx::dist_object<upcxx::global_ptr<int>> disps_nt_dobj(disps_nt, *ec.pg().team());
// #else
//   std::vector<int> disps_s1(nranks);
//   std::vector<int> disps_s2(nranks);
//   std::vector<int> disps_nt(nranks);
// #endif
//   for(int i = 0; i < nranks; i++) {
// #ifdef USE_UPCXX
//     disps_s1.local()[i] = (i > 0) ? (disps_s1.local()[i - 1] + s1_count.local()[i - 1]) : 0;
//     disps_s2.local()[i] = (i > 0) ? (disps_s2.local()[i - 1] + s2_count.local()[i - 1]) : 0;
//     disps_nt.local()[i] = (i > 0) ? (disps_nt.local()[i - 1] + nt_count.local()[i - 1]) : 0;
// #else
//     disps_s1[i] = (i > 0) ? (disps_s1[i - 1] + s1_count[i - 1]) : 0;
//     disps_s2[i] = (i > 0) ? (disps_s2[i - 1] + s2_count[i - 1]) : 0;
//     disps_nt[i] = (i > 0) ? (disps_nt[i - 1] + nt_count[i - 1]) : 0;
// #endif
//   }

//   // Allocate vectors to gather data at root
// #ifdef USE_UPCXX
//   upcxx::global_ptr<int> s1_all;
//   upcxx::global_ptr<int> s2_all;
//   upcxx::global_ptr<int> ntasks_all;
//   std::vector<int>       s1_all_v;
//   std::vector<int>       s2_all_v;
//   std::vector<int>       ntasks_all_v;
// #else
//   std::vector<int> s1_all;
//   std::vector<int> s2_all;
//   std::vector<int> ntasks_all;
// #endif
//   if(rank == 0) {
// #ifdef USE_UPCXX
//     s1_all     = upcxx::new_array<int>(disps_s1.local()[nranks - 1] + s1_count.local()[nranks -
//     1]); s2_all     = upcxx::new_array<int>(disps_s2.local()[nranks - 1] +
//     s2_count.local()[nranks - 1]); ntasks_all = upcxx::new_array<int>(disps_nt.local()[nranks -
//     1] + nt_count.local()[nranks - 1]);

//     s1_all_v.resize(disps_s1.local()[nranks - 1] + s1_count.local()[nranks - 1]);
//     s2_all_v.resize(disps_s2.local()[nranks - 1] + s2_count.local()[nranks - 1]);
//     ntasks_all_v.resize(disps_nt.local()[nranks - 1] + nt_count.local()[nranks - 1]);
// #else
//     s1_all.resize(disps_s1[nranks - 1] + s1_count[nranks - 1]);
//     s2_all.resize(disps_s2[nranks - 1] + s2_count[nranks - 1]);
//     ntasks_all.resize(disps_nt[nranks - 1] + nt_count[nranks - 1]);
// #endif
//   }

// #ifdef USE_UPCXX
//   upcxx::dist_object<upcxx::global_ptr<int>> s1_all_dobj(s1_all, *ec.pg().team());
//   upcxx::dist_object<upcxx::global_ptr<int>> s2_all_dobj(s2_all, *ec.pg().team());
//   upcxx::dist_object<upcxx::global_ptr<int>> ntasks_all_dobj(ntasks_all, *ec.pg().team());
// #endif

//   // Gather at root
// #ifdef USE_UPCXX
//   ec.pg().gatherv(s1vec.data(), s1vec_size, s1_all_dobj.fetch(0).wait(), s1_count.local(),
//                   disps_s1_dobj.fetch(0).wait());
//   ec.pg().gatherv(s2vec.data(), s2vec_size, s2_all_dobj.fetch(0).wait(), s2_count.local(),
//                   disps_s2_dobj.fetch(0).wait());
//   ec.pg().gatherv(ntask_vec.data(), ntvec_size, ntasks_all_dobj.fetch(0).wait(),
//   nt_count.local(),
//                   disps_nt_dobj.fetch(0).wait());
// #else
//   ec.pg().gatherv(s1vec.data(), s1vec_size, s1_all.data(), s1_count.data(), disps_s1.data(), 0);
//   ec.pg().gatherv(s2vec.data(), s2vec_size, s2_all.data(), s2_count.data(), disps_s2.data(), 0);
//   ec.pg().gatherv(ntask_vec.data(), ntvec_size, ntasks_all.data(), nt_count.data(),
//   disps_nt.data(),
//                   0);
// #endif

// #ifdef USE_UPCXX
//   if(rank == 0) {
//     memcpy(s1_all_v.data(), s1_all.local(),
//            (disps_s1.local()[nranks - 1] + s1_count.local()[nranks - 1]) * sizeof(int));
//     memcpy(s2_all_v.data(), s2_all.local(),
//            (disps_s2.local()[nranks - 1] + s2_count.local()[nranks - 1]) * sizeof(int));
//     memcpy(ntasks_all_v.data(), ntasks_all.local(),
//            (disps_nt.local()[nranks - 1] + nt_count.local()[nranks - 1]) * sizeof(int));
//   }

//   upcxx::delete_array(s1_count);
//   upcxx::delete_array(s2_count);
//   upcxx::delete_array(nt_count);
//   upcxx::delete_array(disps_s1);
//   upcxx::delete_array(disps_s2);
//   upcxx::delete_array(disps_nt);
//   if(rank == 0) {
//     upcxx::delete_array(s1_all);
//     upcxx::delete_array(s2_all);
//     upcxx::delete_array(ntasks_all);
//   }
//   return std::make_tuple(s1_all_v, s2_all_v, ntasks_all_v);
// #else
//   EXPECTS(s1_all.size() == s2_all.size());
//   EXPECTS(s1_all.size() == ntasks_all.size());
//   return std::make_tuple(s1_all, s2_all, ntasks_all);
// #endif
// }

std::tuple<Matrix, size_t, double, double>
gensqrtinv_atscf(ExecutionContext& ec, ChemEnv& chem_env, SCFVars& scf_vars,
                 ScalapackInfo& scalapack_info, Tensor<double> S1, TiledIndexSpace& tao_atom,
                 bool symmetric, double threshold) {
  using T = double;

  SCFOptions& scf_options = chem_env.ioptions.scf_options;

  Scheduler sch{ec};
  // auto world = ec.pg().comm();
  int world_rank = ec.pg().rank().value();
  int world_size = ec.pg().size().value();

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
    lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, N, V.data(), N, eps.data());
  }

  std::vector<T>::iterator first_above_thresh;
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

template<typename TensorType>
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
gather_task_vectors(ExecutionContext& ec, std::vector<int>& s1vec, std::vector<int>& s2vec,
                    std::vector<int>& ntask_vec) {
  const int rank   = ec.pg().rank().value();
  const int nranks = ec.pg().size().value();

#ifdef USE_UPCXX
  upcxx::global_ptr<int> s1_count = upcxx::new_array<int>(nranks);
  upcxx::global_ptr<int> s2_count = upcxx::new_array<int>(nranks);
  upcxx::global_ptr<int> nt_count = upcxx::new_array<int>(nranks);
  assert(s1_count && s2_count && nt_count);

  upcxx::dist_object<upcxx::global_ptr<int>> s1_count_dobj(s1_count, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> s2_count_dobj(s2_count, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> nt_count_dobj(nt_count, *ec.pg().team());
#else
  std::vector<int> s1_count(nranks);
  std::vector<int> s2_count(nranks);
  std::vector<int> nt_count(nranks);
#endif

  int s1vec_size = (int) s1vec.size();
  int s2vec_size = (int) s2vec.size();
  int ntvec_size = (int) ntask_vec.size();

  // Root gathers number of elements at each rank.
#ifdef USE_UPCXX
  ec.pg().gather(&s1vec_size, s1_count_dobj.fetch(0).wait());
  ec.pg().gather(&s2vec_size, s2_count_dobj.fetch(0).wait());
  ec.pg().gather(&ntvec_size, nt_count_dobj.fetch(0).wait());
#else
  ec.pg().gather(&s1vec_size, s1_count.data(), 0);
  ec.pg().gather(&s2vec_size, s2_count.data(), 0);
  ec.pg().gather(&ntvec_size, nt_count.data(), 0);
#endif

  // Displacements in the receive buffer for GATHERV
#ifdef USE_UPCXX
  upcxx::global_ptr<int>                     disps_s1 = upcxx::new_array<int>(nranks);
  upcxx::global_ptr<int>                     disps_s2 = upcxx::new_array<int>(nranks);
  upcxx::global_ptr<int>                     disps_nt = upcxx::new_array<int>(nranks);
  upcxx::dist_object<upcxx::global_ptr<int>> disps_s1_dobj(disps_s1, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> disps_s2_dobj(disps_s2, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> disps_nt_dobj(disps_nt, *ec.pg().team());
#else
  std::vector<int> disps_s1(nranks);
  std::vector<int> disps_s2(nranks);
  std::vector<int> disps_nt(nranks);
#endif
  for(int i = 0; i < nranks; i++) {
#ifdef USE_UPCXX
    disps_s1.local()[i] = (i > 0) ? (disps_s1.local()[i - 1] + s1_count.local()[i - 1]) : 0;
    disps_s2.local()[i] = (i > 0) ? (disps_s2.local()[i - 1] + s2_count.local()[i - 1]) : 0;
    disps_nt.local()[i] = (i > 0) ? (disps_nt.local()[i - 1] + nt_count.local()[i - 1]) : 0;
#else
    disps_s1[i] = (i > 0) ? (disps_s1[i - 1] + s1_count[i - 1]) : 0;
    disps_s2[i] = (i > 0) ? (disps_s2[i - 1] + s2_count[i - 1]) : 0;
    disps_nt[i] = (i > 0) ? (disps_nt[i - 1] + nt_count[i - 1]) : 0;
#endif
  }

  // Allocate vectors to gather data at root
#ifdef USE_UPCXX
  upcxx::global_ptr<int> s1_all;
  upcxx::global_ptr<int> s2_all;
  upcxx::global_ptr<int> ntasks_all;
  std::vector<int>       s1_all_v;
  std::vector<int>       s2_all_v;
  std::vector<int>       ntasks_all_v;
#else
  std::vector<int> s1_all;
  std::vector<int> s2_all;
  std::vector<int> ntasks_all;
#endif
  if(rank == 0) {
#ifdef USE_UPCXX
    s1_all     = upcxx::new_array<int>(disps_s1.local()[nranks - 1] + s1_count.local()[nranks - 1]);
    s2_all     = upcxx::new_array<int>(disps_s2.local()[nranks - 1] + s2_count.local()[nranks - 1]);
    ntasks_all = upcxx::new_array<int>(disps_nt.local()[nranks - 1] + nt_count.local()[nranks - 1]);

    s1_all_v.resize(disps_s1.local()[nranks - 1] + s1_count.local()[nranks - 1]);
    s2_all_v.resize(disps_s2.local()[nranks - 1] + s2_count.local()[nranks - 1]);
    ntasks_all_v.resize(disps_nt.local()[nranks - 1] + nt_count.local()[nranks - 1]);
#else
    s1_all.resize(disps_s1[nranks - 1] + s1_count[nranks - 1]);
    s2_all.resize(disps_s2[nranks - 1] + s2_count[nranks - 1]);
    ntasks_all.resize(disps_nt[nranks - 1] + nt_count[nranks - 1]);
#endif
  }

#ifdef USE_UPCXX
  upcxx::dist_object<upcxx::global_ptr<int>> s1_all_dobj(s1_all, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> s2_all_dobj(s2_all, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> ntasks_all_dobj(ntasks_all, *ec.pg().team());
#endif

  // Gather at root
#ifdef USE_UPCXX
  ec.pg().gatherv(s1vec.data(), s1vec_size, s1_all_dobj.fetch(0).wait(), s1_count.local(),
                  disps_s1_dobj.fetch(0).wait());
  ec.pg().gatherv(s2vec.data(), s2vec_size, s2_all_dobj.fetch(0).wait(), s2_count.local(),
                  disps_s2_dobj.fetch(0).wait());
  ec.pg().gatherv(ntask_vec.data(), ntvec_size, ntasks_all_dobj.fetch(0).wait(), nt_count.local(),
                  disps_nt_dobj.fetch(0).wait());
#else
  ec.pg().gatherv(s1vec.data(), s1vec_size, s1_all.data(), s1_count.data(), disps_s1.data(), 0);
  ec.pg().gatherv(s2vec.data(), s2vec_size, s2_all.data(), s2_count.data(), disps_s2.data(), 0);
  ec.pg().gatherv(ntask_vec.data(), ntvec_size, ntasks_all.data(), nt_count.data(), disps_nt.data(),
                  0);
#endif

#ifdef USE_UPCXX
  if(rank == 0) {
    memcpy(s1_all_v.data(), s1_all.local(),
           (disps_s1.local()[nranks - 1] + s1_count.local()[nranks - 1]) * sizeof(int));
    memcpy(s2_all_v.data(), s2_all.local(),
           (disps_s2.local()[nranks - 1] + s2_count.local()[nranks - 1]) * sizeof(int));
    memcpy(ntasks_all_v.data(), ntasks_all.local(),
           (disps_nt.local()[nranks - 1] + nt_count.local()[nranks - 1]) * sizeof(int));
  }

  upcxx::delete_array(s1_count);
  upcxx::delete_array(s2_count);
  upcxx::delete_array(nt_count);
  upcxx::delete_array(disps_s1);
  upcxx::delete_array(disps_s2);
  upcxx::delete_array(disps_nt);
  if(rank == 0) {
    upcxx::delete_array(s1_all);
    upcxx::delete_array(s2_all);
    upcxx::delete_array(ntasks_all);
  }
  return std::make_tuple(s1_all_v, s2_all_v, ntasks_all_v);
#else
  EXPECTS(s1_all.size() == s2_all.size());
  EXPECTS(s1_all.size() == ntasks_all.size());
  return std::make_tuple(s1_all, s2_all, ntasks_all);
#endif
}

template std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
gather_task_vectors<double>(ExecutionContext& ec, std::vector<int>& s1vec, std::vector<int>& s2vec,
                            std::vector<int>& ntask_vec);

template std::vector<size_t> sort_indexes<double>(std::vector<double>& v);
