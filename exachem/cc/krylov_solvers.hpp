/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccsd/ccsd_util.hpp"

namespace exachem::cc::solvers {

using namespace tamm;

/**
 * @brief Computes the Euclidean norm of a packed vector of TAMM tensors.
 *
 * Treats the entire vector-of-tensors as a single flat vector and returns
 * sqrt( sum_i ||v[i]||_F^2 ), where ||.||_F is the Frobenius norm of each
 * individual tensor block.
 *
 * @tparam T Scalar type (e.g. double)
 * @param v Input: vector of TAMM tensors (e.g. {t1, t2} for CCSD amplitudes)
 * @return double  Euclidean norm of the packed vector
 */
template<typename T>
double compute_norm(const std::vector<Tensor<T>>& v) {
  double norm_tensor  = 0.0;
  double norm_squared = 0.0;

  for(const auto& tensor: v) {
    norm_tensor = norm(tensor);
    norm_squared += norm_tensor * norm_tensor;
  }

  return std::sqrt(norm_squared);
}

/**
 * @brief Scales every tensor in a packed vector in-place by a scalar factor.
 *
 * Performs v[i] *= c for all i, using tamm::scale_ip on each tensor in the vector of tensors.
 *
 * @tparam T Scalar type
 * @param v  In/out: vector of TAMM tensors to be scaled
 * @param c  Scalar factor
 */
template<typename T>
void scale_tensors(std::vector<Tensor<T>>& v, double c) {
  for(auto& tensor: v) { tamm::scale_ip(tensor, c); }
}

/**
 * @brief Allocates a new vector of TAMM tensors with the same structure as ref,
 *        initialized to zero.
 *
 * For each tensor in ref, a new tensor with identical TiledIndexSpaces is
 * allocated and zeroed. The caller takes ownership of the allocated tensors.
 *
 * @tparam T Scalar type
 * @param ec  ExecutionContext used for tensor allocation
 * @param ref Input: reference vector whose tensor shapes are mirrored
 * @return std::vector<Tensor<T>>  Freshly allocated, zero-initialized tensors
 */
template<typename T>
std::vector<Tensor<T>> zeroslike(ExecutionContext& ec, const std::vector<Tensor<T>>& ref) {
  std::vector<Tensor<T>> out;
  out.reserve(ref.size());
  for(const auto& t: ref) {
    Tensor<T> z{t.tiled_index_spaces()};
    Tensor<T>::allocate(&ec, z);
    out.push_back(z);
  }
  scale_tensors(out, 0.0);
  return out;
}

/**
 * @brief Accumulates c*w into v in-place: v[i] += c * w[i] for all i.
 *
 * @tparam T Scalar type
 * @param sch  TAMM Scheduler
 * @param v    In/out: vector of destination tensors (must be pre-allocated)
 * @param w    Input: vector of source tensors (same shapes as v)
 * @param c    Scalar coefficient applied to w before accumulation
 */
template<typename T>
void tensor_arithmetic(Scheduler& sch, std::vector<Tensor<T>>& v, const std::vector<Tensor<T>>& w,
                       double c) {
  for(size_t i = 0; i < v.size(); ++i) { sch(v[i]() += c * w[i]()); }
  sch.execute();
}

/**
 * @brief Creates an independent deep copy of a packed vector of TAMM tensors.
 *
 * Allocates fresh tensors with the same TiledIndexSpaces as src, then copies
 * the data via NK_accumulate. This avoids the TAMM self-assignment error that
 * arises from shallow handle copies (auto dst = src copies tensor handles,
 * not data).
 *
 * @tparam T Scalar type
 * @param ec  ExecutionContext used for allocation
 * @param src Input: vector of source tensors to copy
 * @return std::vector<Tensor<T>>  Newly allocated tensors containing a copy of src
 */
template<typename T>
std::vector<Tensor<T>> tensor_copy(ExecutionContext& ec, const std::vector<Tensor<T>>& src) {
  auto out = zeroslike(ec, src);
  auto sch = Scheduler{ec};
  tensor_arithmetic(sch, out, src, 1.0);
  return out;
}

/**
 * @brief Computes the inner product of two packed vectors of TAMM tensors.
 *
 * Evaluates sum_i <v[i], w[i]> using TAMM:
 *   d_s() = v[i]() * w[i]() (element-wise product summed over all indices)
 *
 * @tparam T Scalar type
 * @param ec  ExecutionContext used for temporary scalar tensor allocation
 * @param v   Input: first vector of tensors
 * @param w   Input: second vector of tensors (same shapes as v)
 * @return double  Inner product sum_i trace(v[i]^T * w[i])
 */
template<typename T>
double inner_prod(ExecutionContext& ec, const std::vector<Tensor<T>>& v,
                  const std::vector<Tensor<T>>& w) {
  double innerp = 0.0;

  for(size_t i = 0; i < v.size(); ++i) {
    Tensor<T> d_s{};
    Tensor<T>::allocate(&ec, d_s);
    Scheduler{ec}(d_s() = v[i]() * w[i]()).execute();
    innerp += get_scalar(d_s);
    Tensor<T>::deallocate(d_s);
  }

  return innerp;
}
/**
 * @brief Newton-Krylov solver for nonlinear systems using GMRES with Arnoldi iteration.
 *
 *  Solves F(x) = 0,
 *  Newton step: x_{n+1} = x_n - J^{-1}(x_n) F(x_n)
 *  where J is the Jacobian of F at x_n. Let delta_x = x_{n+1} - x_n, then
 *  J*delta_x = -F(x_n). The linear system is solved using GMRES with Arnoldi iteration.
 *
 * This contains 3 main components:
 * 1. Arnoldi iteration to build an orthonormal Krylov basis and Hessenberg matrix.
 * 2. GMRES solver that uses the Arnoldi basis to solve the linearized system.
 * 3. Newton-Krylov iteration that updates the solution based on the GMRES
 *    solution and checks for convergence.
 */

/**
 * @brief Arnoldi iteration:
 *
 * Purpose: Given a linear system A*x (Matrix vector product) = b (right handside of the linear
 * system) , constructs an orthonormal basis Q for the Krylov subspace and an upper Hessenberg
 * matrix H. A = Q*H*Q^T
 *
 * Assumes that the matrix-vector product A*x is provided as a callable function. (Note: Can be used
 * with the explicit matrix as well)
 *
 * Inputs: Matrix vector product function, right hand side of the linear system b, and the maximum
 * Krylov dimension. Outputs: Q (orthonormal basis) and H (upper Hessenberg matrix).
 *
 * Procedure:
 *  Starting from the vector b, applies modified Gram-Schmidt to produce:
 *  H[j,k] = <Q[:,j], A*Q[:,k]>
 *  Q[:,k+1] = (mvp(Q[:,k]) - sum_{j<=k} H[j,k]*Q[:,j]) / H[k+1,k]
 *
 * @tparam MVPFun Callable: (const vector<Tensor<T>>&) -> vector<Tensor<T>>
 * @tparam T   Scalar type
 *
 * Input variables:
 * @param ec   ExecutionContext for tensor allocation and scheduling
 * @param mvpfn  Matrix vector product A*x callback (e.g. the Jacobian-vector product Jv in
 * Newton-Krylov)
 * @param rhs  Right handside vector (used as the initial vector after normalization)
 * @param krylov_dim Maximum number of Arnoldi steps (Krylov subspace dimension)
 *
 * Outputs variables:
 * @return pair of:
 *   - Q: vector of (up to krylov_dim+1) Krylov basis vectors, each a vector<Tensor<T>>
 *        of the same shape as b. Q[0] = b/||b||
 *   - H: (krylov_dim+1) x krylov_dim (or smaller on breakdown) upper Hessenberg Eigen matrix
 */

// template<typename MVPFun, typename T>
// auto arnoldi_iteration(ExecutionContext& ec, MVPFun&& mvpfn, const std::vector<Tensor<T>>& rhs,
//                        int krylov_dims)
//   -> std::pair<std::vector<std::vector<Tensor<T>>>,
//                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {
//   auto sch     = Scheduler{ec};
//   using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

//   std::vector<std::vector<Tensor<T>>> Q;
//   for(int j = 0; j <= krylov_dims; ++j) { Q.push_back(zeroslike(ec, rhs)); }

//   Matrix H = Matrix::Zero(krylov_dims + 1, krylov_dims);

//   auto rhs_norm = compute_norm(rhs);

//   // This is a safety check to avoid division by zero in the case of a zero rhs.
//   if(rhs_norm < 1e-14) { return {Q, H}; }

//   // This is building the first Krylov vector Q[0] = rhs / ||rhs||, which is the normalized
//   // right-hand side.
//   tensor_arithmetic(sch, Q[0], rhs, 1.0 / rhs_norm);

//   // This is the main Arnoldi iteration loop, which constructs the Krylov basis and Hessenberg
//   // matrix.
//   for(int k = 0; k < krylov_dims; ++k) {
//     auto mvp_of_Qk = mvpfn(Q[k]);

//     for(int j = 0; j <= k; ++j) {
//       H(j, k) = inner_prod(ec, Q[j], mvp_of_Qk);
//       tensor_arithmetic(sch, mvp_of_Qk, Q[j], -H(j, k));
//     }

//     H(k + 1, k) = compute_norm(mvp_of_Qk);

//     if(H(k + 1, k) < 1e-14) {
//       std::vector<std::vector<Tensor<T>>> Q_computed(Q.begin(), Q.begin() + (k + 1));
//       Matrix                              H_computed = H.topLeftCorner(k + 2, k + 1).eval();
//       return {std::move(Q_computed), std::move(H_computed)};
//     }

//     scale_tensors(Q[k + 1], 0.0);
//     tensor_arithmetic(sch, Q[k + 1], mvp_of_Qk, 1.0 / H(k + 1, k));
//     for(auto& t: mvp_of_Qk) sch.deallocate(t).execute();
//   }

//   return {std::move(Q), std::move(H)};
// }

/**
 * @brief GMRES (Generalized Minimum Residual):
 *
 * Note: This is via Arnoldi iteration.
 *
 * Purpose: Solves the linear system A*x = b approximately in the Krylov subspace.
 * Note: This requires the Arnoldi iteration to build the Krylov basis and Hessenberg matrix.
 *
 * Assumes that the matrix-vector product A*x is provided as a callable function. (Note: Can be used
 * with the explicit matrix as well)
 *
 * Main inputs: Matrix vector product function, right hand side of the linear system b, and the
 * maximum Krylov dimension. Other inputs:
 *  1. Convergence threshold to GMRES - To avoid unnecessary iterations if GMRES converges before
 * reaching the maximum krylov subspace dimension.
 *  2. Eisenstat-Walker forcing term patameters - To avoid over-solving the linear system in the
 * early Newton iterations. a) Initial forcing term. b) gamma, alpha - user-defined parameters for
 * the Eisenstat-Walker forcing term.
 *
 * Outputs: Solution of the linear system A*x = b, which is x (Define as the solution).
 *
 * Procedure:
 *  1. Call Arnoldi iteration to build the Krylov basis Q and Hessenberg matrix H.
 *  2. Solve the least-squares problem min || H*y - ||b||*e1 ||, e1 is the first canonical vector
 * and y in the Krylov subspace.
 *     [- min || A*x - b || (in original space)
 *       = min || A*Q*y - b || (in Krylov subspace) - By letting x = Q*y, we can express the
 * solution in the Krylov subspace. = min || Q*H*y - b || (Note: A*Q = Q*H) = min || H*y - Q^T*b ||
 *       = min || H*y - ||b||*e1 || (Note: Q^T*b = ||b||*e1, since Q is orthonormal and the first
 * column of Q is b/||b||)]
 *  3. Compute the solution x = Q*y in the original space.
 *
 * @tparam MVPFun Callable: (const vector<Tensor<T>>&) -> vector<Tensor<T>>
 * @tparam T   Scalar type
 *
 * Input variables:
 * @param ec   ExecutionContext for tensor allocation and scheduling
 * @param mvpfn  Matrix vector product A*x callback (e.g. the Jacobian-vector product Jv in
 * Newton-Krylov)
 * @param rhs  Right handside vector (used as the initial vector after normalization)
 * @param krylov_dim Maximum number of Arnoldi steps (Krylov subspace dimension)
 *
 * Other input variables:
 * @param gmres_tol Convergence threshold to GMRES: GMRES iterations stops when ||F(x)|| < gmres_tol
 * @param eta Initial Eisenstat-Walker forcing term
 *  User-defined parameter, set to 0.5 by default.
 * @param gamma, alpha - user-defined parameters for the Eisenstat-Walker forcing term.
 *  Eisenstat-Walker forcing term is evaluated as follows:
 *  eta = gamma * (||F(x_n)|| / ||F(x_{n-1})||)^alpha
 *  where alpha [0,1] and beta (1,2] are user-defined parameters (default: gamma=0.5, alpha=1.5)
 *
 * I make the function signature more genral,
 *  - Adaptive forcing term: If true, the Eisenstat-Walker forcing term is used to avoid
 * over-solving the linear system in the early Newton iterations.
 *  - Verbose: If true, prints the GMRES iteration number and the residual norm
 *    Both set to false by default.
 *
 * Outputs variables:
 * @return x  Solution of the linear system A*x = b, which is x (Define as the solution).
 */

// template<typename MVPFun, typename T>
// std::vector<Tensor<T>> gmres_solver(ExecutionContext& ec, ChemEnv& chem_env, Tensor<T>& d_e,
//                                     MVPFun&& mvpfn, const std::vector<Tensor<T>>& rhs,
//                                     int krylov_dims, double gmres_tol, double eta = 0.5,
//                                     double gamma = 0.5, double alpha = 1.5,
//                                     bool adaptive_forcing = false, bool verbose = false) {
//   using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
//   auto sch     = Scheduler{ec};

//   // Time the Arnoldi phase: this is where the matrix-vector products (the dominant
//   // GMRES cost) happen, up front, before the per-iteration reporting loop below.
//   const auto arnoldi_start = std::chrono::high_resolution_clock::now();
//   auto [Q, H]              = arnoldi_iteration(ec, std::forward<MVPFun>(mvpfn), rhs,
//   krylov_dims); const auto   arnoldi_end = std::chrono::high_resolution_clock::now(); const
//   double arnoldi_time =
//     std::chrono::duration_cast<std::chrono::duration<double>>(arnoldi_end -
//     arnoldi_start).count();
//   double rhs_norm = compute_norm(rhs);

//   if(verbose && ec.pg().rank() == 0) {
//     std::cout << "Starting GMRES " << std::endl;
//     std::cout << " Arnoldi time: " << arnoldi_time << " secs" << std::endl;
//   }

//   for(int k = 0; k < H.cols(); ++k) {
//     const auto iter_start = std::chrono::high_resolution_clock::now();
//     // This part is related to printing GMRES iteration information.
//     auto   Hk  = H.topLeftCorner(k + 2, k + 1);
//     Vector e1k = Vector::Zero(k + 2);
//     e1k(0)     = rhs_norm;

//     Vector       yk            = Hk.colPivHouseholderQr().solve(e1k);
//     Vector       residual      = e1k - Hk * yk;
//     double       residual_norm = residual.norm();
//     const auto   iter_end      = std::chrono::high_resolution_clock::now();
//     const double iter_time =
//       std::chrono::duration_cast<std::chrono::duration<double>>(iter_end - iter_start).count();

//     if(verbose) {
//       chem_env.cc_context.ccsd_iter++;
//       const double energy = get_scalar(d_e); // collective: all ranks
//       if(ec.pg().rank() == 0) {
//         iteration_print(chem_env, ec.pg(), chem_env.cc_context.ccsd_iter, residual_norm, energy,
//                         iter_time);
//       }
//     }

//     // This part is related to the convergence check for GMRES.
//     if(verbose && residual_norm <= gmres_tol) {
//       if(ec.pg().rank() == 0) std::cout << "GMRES converged." << std::endl;
//       break;
//     }

//     // This part is related to the Eisenstat-Walker forcing term for adaptive convergence.
//     if(adaptive_forcing && residual_norm <= eta * rhs_norm) {
//       if(ec.pg().rank() == 0) std::cout << "GMRES met forcing threshold." << std::endl;
//       break;
//     }
//   }

//   // This part is related to solving the least-squares problem and computing the solution in the
//   // original space.
//   Vector e1       = Vector::Zero(H.rows());
//   e1(0)           = rhs_norm;
//   Vector y        = H.colPivHouseholderQr().solve(e1);
//   auto   solution = zeroslike(ec, rhs);
//   for(int j = 0; j < H.cols(); ++j) { tensor_arithmetic(sch, solution, Q[j], y(j)); }
//   for(auto& q_vec: Q) {
//     for(auto& t: q_vec) { sch.deallocate(t); }
//   }
//   sch.execute();

//   return solution;
// }

/**
 * @brief GMRES (Generalized Minimum Residual) solver.
 *
 * Purpose:
 * Solves the linear system A*x = b approximately using the GMRES algorithm,
 * where the matrix A is accessed only through a user-provided matrix-vector
 * product. The Arnoldi iteration is performed internally while simultaneously
 * monitoring the least-squares residual at every Krylov iteration.
 *
 * Assumes that the matrix-vector product A*x is provided as a callable
 * function (i.e., the matrix A does not need to be formed explicitly).
 *
 * Main inputs:
 *  1. Matrix-vector product callback.
 *  2. Right-hand side vector b.
 *  3. Maximum Krylov subspace dimension.
 *
 * Additional inputs:
 *  1. GMRES convergence tolerance.
 *  2. Eisenstat-Walker forcing term parameters for inexact Newton methods.
 *  3. Flags for adaptive forcing and iteration printing.
 *
 * Procedure:
 *  1. Normalize the right-hand side to obtain the first Krylov basis vector.
 *  2. Perform the Arnoldi iteration:
 *       - Compute A*Q[:,k].
 *       - Orthogonalize against the existing Krylov basis using modified
 *         Gram-Schmidt.
 *       - Store the projection coefficients in the Hessenberg matrix H.
 *       - Normalize the resulting vector to obtain the next Krylov basis
 *         vector.
 *  3. After each Arnoldi step, solve the least-squares problem
 *
 *         min || H*y - ||b||*e1 ||
 *
 *     to obtain the current GMRES residual.
 *
 *  4. Stop if either
 *       - the GMRES residual satisfies the prescribed tolerance, or
 *       - the Eisenstat-Walker forcing criterion is satisfied (when adaptive
 *         forcing is enabled), or
 *       - the Krylov subspace reaches the prescribed dimension.
 *
 *  5. Solve the final least-squares problem and recover the solution
 *
 *         x = Q*y.
 *
 * @tparam MVPFun Callable:
 *         (const std::vector<Tensor<T>>&) -> std::vector<Tensor<T>>
 * @tparam T Scalar type.
 *
 * Input variables:
 * @param ec ExecutionContext used for tensor allocation and scheduling.
 * @param chem_env Chemical environment used for iteration bookkeeping and
 *        printing.
 * @param d_e Tensor containing the current CC energy (used only for verbose
 *        iteration output).
 * @param mvpfn Matrix-vector product callback representing A*x (for example,
 *        the Jacobian-vector product in Newton-Krylov methods).
 * @param rhs Right-hand side vector b.
 * @param krylov_dims Maximum Krylov subspace dimension.
 *
 * Additional input variables:
 * @param gmres_tol Absolute convergence tolerance for the GMRES residual.
 * @param eta Initial Eisenstat-Walker forcing term (default = 0.5).
 * @param gamma User-defined Eisenstat-Walker parameter (default = 0.5).
 * @param alpha User-defined Eisenstat-Walker exponent (default = 1.5).
 *
 * The Eisenstat-Walker forcing term is typically updated externally as
 *
 *     eta = gamma * (||F(x_n)|| / ||F(x_{n-1})||)^alpha,
 *
 * where gamma and alpha are user-defined parameters.
 *
 * @param adaptive_forcing
 *        If true, terminates GMRES once the residual satisfies
 *
 *            ||r|| <= eta ||b||,
 *
 *        thereby avoiding over-solving the linear system during early Newton
 *        iterations.
 *
 * @param verbose
 *        If true, prints the GMRES iteration number, residual norm, current
 *        energy, and elapsed iteration time.
 *
 * Output variables:
 * @return Approximate solution x of the linear system A*x = b represented as
 *         a vector of tensors having the same layout as rhs.
 */
template<typename MVPFun, typename T>
std::vector<Tensor<T>> gmres_solver(ExecutionContext& ec, ChemEnv& chem_env, Tensor<T>& d_e,
                                    MVPFun&& mvpfn, const std::vector<Tensor<T>>& rhs,
                                    int krylov_dims, double gmres_tol, double eta = 0.5,
                                    double gamma = 0.9, double alpha = 1.5,
                                    bool adaptive_forcing = false, bool verbose = false) {
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

  auto   sch      = Scheduler{ec};
  double rhs_norm = compute_norm(rhs);
  auto   solution = zeroslike(ec, rhs);
  if(rhs_norm < 1e-14) return solution;

  std::vector<std::vector<Tensor<T>>> Q(krylov_dims + 1);

  for(auto& q: Q) q = zeroslike(ec, rhs);

  Matrix H = Matrix::Zero(krylov_dims + 1, krylov_dims);
  tensor_arithmetic(sch, Q[0], rhs, 1.0 / rhs_norm);
  sch.execute();

  if(verbose && ec.pg().rank() == 0) { std::cout << "Starting GMRES " << std::endl; }

  int final_k = 0;
  for(int k = 0; k < krylov_dims; k++) {
    final_k         = k;
    auto iter_start = std::chrono::high_resolution_clock::now();
    auto v          = mvpfn(Q[k]);
    for(int j = 0; j <= k; j++) {
      H(j, k) = inner_prod(ec, Q[j], v);
      tensor_arithmetic(sch, v, Q[j], -H(j, k));
    }
    sch.execute();
    H(k + 1, k) = compute_norm(v);

    if(H(k + 1, k) < 1e-14) {
      for(auto& t: v) sch.deallocate(t);
      sch.execute();

      final_k = k;
      break;
    }

    tensor_arithmetic(sch, Q[k + 1], v, 1.0 / H(k + 1, k));
    sch.execute();

    for(auto& t: v) sch.deallocate(t);
    sch.execute();

    Matrix Hk = H.topLeftCorner(k + 2, k + 1);

    Vector e1 = Vector::Zero(k + 2);
    e1(0)     = rhs_norm;

    Vector y = Hk.colPivHouseholderQr().solve(e1);

    Vector residual      = e1 - Hk * y;
    double residual_norm = residual.norm();

    auto   iter_end  = std::chrono::high_resolution_clock::now();
    double iter_time = std::chrono::duration<double>(iter_end - iter_start).count();

    if(verbose) {
      chem_env.cc_context.ccsd_iter++;
      double energy = get_scalar(d_e);
      if(ec.pg().rank() == 0) {
        iteration_print(chem_env, ec.pg(), chem_env.cc_context.ccsd_iter, residual_norm, energy,
                        iter_time);
      }
    }

    if(verbose && residual_norm <= gmres_tol) {
      if(verbose && ec.pg().rank() == 0) std::cout << "GMRES converged.\n";
      final_k = k;
      break;
    }

    if(adaptive_forcing && residual_norm <= eta * rhs_norm) {
      if(verbose && ec.pg().rank() == 0) std::cout << "GMRES met forcing threshold.\n";
      final_k = k;

      break;
    }
  }

  Matrix Hfinal = H.topLeftCorner(final_k + 2, final_k + 1);

  Vector e1 = Vector::Zero(final_k + 2);

  e1(0) = rhs_norm;

  Vector y = Hfinal.colPivHouseholderQr().solve(e1);

  scale_tensors(solution, 0.0);

  for(int j = 0; j <= final_k; j++) tensor_arithmetic(sch, solution, Q[j], y(j));

  for(auto& qvec: Q)
    for(auto& t: qvec) sch.deallocate(t);

  sch.execute();

  return solution;
}

/**
 * @brief Newton-Krylov solver:
 *
 * Purpose: Solves a nonlinear system F(x) (Define as the residual function) = 0 in a Jacobian-free
 * manner. Solves F(x) = 0, Newton step: x_{n+1} = x_n - J^{-1}(x_n) F(x_n) where J is the Jacobian
 * of F at x_n. Let delta_x = x_{n+1} - x_n, then J*delta_x = -F(x_n). The linear system is solved
 * using GMRES with Arnoldi iteration.
 *
 * Assumes that the matrix-vector product A*x is provided as a callable function. (Note: Prohibitted
 * to form the Jacobian explicitly)
 *
 * Main solvers include:
 *  1. Newton method for solving nonlinear system F(x) = 0.
 *  2. GMRES for solving the linearized system J*delta_x = -F(x).
 *
 * Main inputs:
 *  - Related to Newton: A function evaluates the residual F(x) at a given x, initail guess x,
 * convergence threshold for Newton iterations.
 *  - Related to GMRES: Krylov subspace dimension.
 *
 * Other inputs:
 *  - Related to Newton: Maximum number of outer Newton iterations - To avoid infinite loops in case
 * of non-convergence.
 *  - Related to GMRES:
 *    - Convergence threshold to GMRES - To avoid unnecessary iterations if GMRES converges before
 * reaching the maximum krylov subspace dimension.
 *    - Eisenstat-Walker forcing term patameters - To avoid over-solving the linear system in the
 * early Newton iterations. a) Initial forcing term. b) gamma, alpha - user-defined parameters for
 * the Eisenstat-Walker forcing term.
 *
 * Outputs: Solution of the nonlinear system F(x) = 0, which is x (Define as the solution).
 *
 * Procedure:
 *  1. Evaluate the residual F(x) at the current iterate x.
 *  2. Check for convergence: if ||F(x)|| < n_tol, then the solution has converged.
 *  3. Build a Jacobian-vector product via finite differences.
 *    - J*delta_x ~ ( F(x + pert*v) - F(x) ) / pert, v is the perturbation vector, pert is the
 * finite difference perturbation size.
 *  4. Call GMRES to solve the linearized system.
 *    - As inputs GMRES takes:
 *     - The Jacobian-vector product function (from step 3)
 *     - Minus right-hand side -F(x) (from step 1)
 *    - Outputs the solution delta_x in the Krylov subspace.
 *  5. Update the iterate: x (Define as the solution) += delta_x, where delta_x is the solution from
 * GMRES.
 *
 * @tparam ResidualFun Callable: (const vector<Tensor<T>>&) -> vector<Tensor<T>>&
 * @tparam T  Scalar type
 *
 * Input variables:
 * @param ec  ExecutionContext for all tensor operations
 * @param residual_fun  Residual function F(x); evaluates the nonlinear equations
 * @param initial_guess Pass by reference: In/out: initial guess on input, solution on output.
 * @param maxiter Maximum number of outer Newton iterations
 * @param krylov_dims Krylov subspace dimension for inner GMRES solve
 *
 * Other input variables:
 * @param newton_tol Convergence threshold: Newton iterations stops when ||F(x)|| < newton_tol
 * @param gmres_tol Convergence threshold: GMRES iterations stops when ||F(x)|| < gmres_tol
 * @param eta Initial Eisenstat-Walker forcing term
 * @param gamma, alpha - user-defined parameters for the Eisenstat-Walker forcing term.
 *
 *
 * Outputs variables:
 * @return x  Solution of the nonlinear system F(x) = 0, which is x (Define as the solution).
 *
 */

/**
 * @brief Write the current CC amplitudes {t1, t2} to disk for restart, every writet_iter Newton
 * steps (when writet is enabled). amplitudes[0]/amplitudes[1] are the T1/T2 tensors.
 */
template<typename T>
void write_amplitudes(ChemEnv& chem_env, int iter, const std::vector<Tensor<T>>& amplitudes) {
  const auto& cc_options = chem_env.ioptions.ccsd_options;
  if(cc_options.writet && ((iter + 1) % cc_options.writet_iter == 0)) {
    write_to_disk(amplitudes[0], chem_env.cc_context.t1file);
    write_to_disk(amplitudes[1], chem_env.cc_context.t2file);
  }
}

template<typename ResidualFun, typename T>
std::pair<double, double> newton_krylov_solver(ExecutionContext& ec, ChemEnv& chem_env,
                                               ResidualFun&& residual_fun, Tensor<T>& d_e,
                                               std::vector<Tensor<T>>& initial_guess) {
  const auto&  nk               = chem_env.ioptions.ccsd_options.solvers.newton_krylov;
  const int    maxiter          = chem_env.ioptions.ccsd_options.ccsd_maxiter;
  const int    krylov_dims      = nk.krylov_dims;
  const double newton_tol       = nk.newton_tol;
  const double gmres_tol        = nk.gmres_tol;
  const double eta              = nk.eta;
  const double gamma            = nk.gamma;
  const double alpha            = nk.alpha;
  const bool   adaptive_forcing = nk.adaptive_forcing;

  auto   sch                             = Scheduler{ec};
  auto&  iterative_solution              = initial_guess;
  double previous_residual_function_norm = 0.0;
  double residual                        = 0.0;
  double energy                          = 0.0;

  for(int iter = 0; iter < maxiter; ++iter) {
    const auto   step_start   = std::chrono::high_resolution_clock::now();
    auto         res_fun      = tensor_copy(ec, residual_fun(iterative_solution));
    double       res_fun_norm = compute_norm(res_fun);
    const auto   step_end     = std::chrono::high_resolution_clock::now();
    const double step_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(step_end - step_start).count();

    chem_env.cc_context.ccsd_iter++;
    energy   = get_scalar(d_e); // collective: all ranks
    residual = res_fun_norm;
    if(ec.print()) {
      std::cout << "Newton step" << std::endl;
      iteration_print(chem_env, ec.pg(), chem_env.cc_context.ccsd_iter, res_fun_norm, energy,
                      step_time);
    }

    // Write amplitudes to disk after each Newton step so the run can be restarted.
    write_amplitudes(chem_env, iter, iterative_solution);

    if(res_fun_norm < newton_tol) {
      if(ec.print()) { std::cout << "Newton-Krylov is converged!" << std::endl; }
      for(auto& t: res_fun) sch.deallocate(t);
      sch.execute();
      break;
    }

    // Eisenstat-Walker forcing term evalutation.
    double current_eta = eta;
    if(iter == 0) { current_eta = eta; }
    else {
      double ratio = res_fun_norm / (previous_residual_function_norm);
      current_eta  = std::min(0.8, gamma * std::pow(ratio, alpha));
      // if(current_eta < 1e-3) current_eta = 1e-3;
      current_eta = std::max(current_eta, 1e-8);
    }

    previous_residual_function_norm = res_fun_norm;

    // Jacobian-vector product via finite differences.
    double norm_iterative_solution = compute_norm(iterative_solution);
    auto   jvp                     = [&](const std::vector<Tensor<T>>& vector) {
      const double norm_vector = compute_norm(vector);
      double       perturbation =
        sqrt(1e-14) * (1.0 + norm_iterative_solution) / (norm_vector > 1e-12 ? norm_vector : 1e-12);
      auto perturbed_iterative_solution = tensor_copy(ec, iterative_solution);
      tensor_arithmetic(sch, perturbed_iterative_solution, vector, perturbation);

      auto perturbed_residual_function =
        tensor_copy(ec, residual_fun(perturbed_iterative_solution));
      tensor_arithmetic(sch, perturbed_residual_function, res_fun, -1.0);
      scale_tensors(perturbed_residual_function, 1.0 / perturbation);
      for(auto& t: perturbed_iterative_solution) sch.deallocate(t).execute();
      return perturbed_residual_function;
    };

    auto neg_res_fun = tensor_copy(ec, res_fun);
    scale_tensors(neg_res_fun, -1.0);

    // GMRES inner loop to solve the linearized system J*delta_x = -F(x)
    auto delta_x = gmres_solver(ec, chem_env, d_e, jvp, neg_res_fun, krylov_dims, gmres_tol,
                                current_eta, gamma, alpha, adaptive_forcing, true);
    tensor_arithmetic(sch, iterative_solution, delta_x, 1.0);

    for(auto& t: res_fun) sch.deallocate(t);
    for(auto& t: neg_res_fun) sch.deallocate(t);
    for(auto& t: delta_x) sch.deallocate(t);
    sch.execute();
  }

  return {residual, energy};
}

/**
 * @brief Preconditioned Newton-Krylov solver:
 *
 * Purpose: Solves a nonlinear system F(x) (Define as the residual function) = 0 in a Jacobian-free
 * manner. Solves F(x) = 0, Newton step: x_{n+1} = x_n - J^{-1}(x_n) F(x_n) where J is the Jacobian
 * of F at x_n. Let delta_x = x_{n+1} - x_n, then J*delta_x = -F(x_n) Then we apply the
 * preconditioner M^{-1} to both sides of the equation: M^{-1}*J*delta_x = -M^{-1}*F(x_n) We solve
 * lhs and rhs using GMRES with Arnoldi iteration. * The resulting linear system is solved using
 * GMRES with Arnoldi iteration.
 *
 * Assumes that the matrix-vector product A*x is provided as a callable function. (Note: Prohibitted
 * to form the Jacobian explicitly)
 *
 * Main solvers include:
 *  1. Newton method for solving nonlinear system F(x) = 0.
 *  2. GMRES for solving the linearized systems
 *    - M*y = J*delta_x
 *    - M*y = residual function
 *  3. GMRES for solving the linearized system M^{-1}*J*delta_x = -M^{-1}*F(x).
 *
 * Main inputs:
 *  - Related to Newton: A function evaluates the residual F(x) at a given x, initail guess x,
 * convergence threshold for Newton iterations.
 *  - Related to GMRES: Krylov subspace dimension.
 *  - Related to preconditioner: Preconditoner inverse application function, krylov subspace
 * dimension for the preconditioned GMRES solve.
 *
 * Other inputs:
 *  - Related to Newton: Maximum number of outer Newton iterations - To avoid infinite loops in case
 * of non-convergence.
 *  - Related to GMRES:
 *    - Convergence threshold to GMRES - To avoid unnecessary iterations if GMRES converges before
 * reaching the maximum krylov subspace dimension.
 *    - Eisenstat-Walker forcing term patameters - To avoid over-solving the linear system in the
 * early Newton iterations. a) Initial forcing term. b) gamma, alpha - user-defined parameters for
 * the Eisenstat-Walker forcing term.
 *
 * Outputs: Solution of the nonlinear system F(x) = 0, which is x (Define as the solution).
 *
 * Procedure:
 *  1. Evaluate the residual F(x) at the current iterate x.
 *  2. Check for convergence: if ||F(x)|| < n_tol, then the solution has converged.
 *  3. Call GMRES to compute the precondtioned right handside, M^{-1}*F(x).
 *    - As inputs GMRES takes:
 *     - The preconditioner action function
 *     - As the rhs right-hand side F(x) (from step 1)
 *    - Outputs the preconditioned right-hand side M^{-1}*F(x) in the Krylov subspace.
 *  4. Build a Jacobian-vector product via finite differences.
 *    - J*delta_x ~ ( F(x + pert*v) - F(x) ) / pert, v is the perturbation vector, pert is the
 * finite difference perturbation size.
 *  5. Call GMRES to compute the precondtioned left handside, M^{-1}*J*delta_x.
 *    - As inputs GMRES takes:
 *     - The preconditioner action function
 *     - As the rhs Jacobian-vector product function (from step 4)
 *    - Outputs the preconditioned left-hand side M^{-1}*J*delta_x in the Krylov subspace.
 *  6. Call GMRES to solve the linearized system.
 *    - As inputs GMRES takes:
 *     - The preconditioned Jacobian-vector product function (from step 5)
 *     - Minus preconditioned right-hand side (from step 3)
 *    - Outputs the solution delta_x in the Krylov subspace
 *  5. Update the iterate: x (Define as the solution) += delta_x, where delta_x is the solution from
 * GMRES.
 *
 * @tparam ResidualFun Callable: (const vector<Tensor<T>>&) -> vector<Tensor<T>>&
 * @tparam PreconditionedInverseApplication Callable: (const vector<Tensor<T>>&) ->
 * vector<Tensor<T>>&
 * @tparam T  Scalar type
 *
 * Input variables:
 * @param ec  ExecutionContext for all tensor operations
 * @param residual_fun  Residual function F(x); evaluates the nonlinear equations
 * @param preconditioner_inverse_application  Preconditioner inverse application function M^{-1}*v;
 * evaluates the preconditioner action on a given vector.
 * @param initial_guess Pass by reference: In/out: initial guess on input, solution on output
 * @param maxiter Maximum number of outer Newton iterations
 * @param krylov_dims Krylov subspace dimension for inner GMRES solve
 * @param krylov_dims_precond Krylov subspace dimension for preconditioned GMRES solve
 *
 * Other input variables:
 * @param newton_tol Convergence threshold: Newton iterations stops when ||F(x)|| < newton_tol
 * @param gmres_tol Convergence threshold: GMRES iterations stops when ||F(x)|| < gmres_tol
 * @param eta Initial Eisenstat-Walker forcing term
 * @param gamma, alpha - user-defined parameters for the Eisenstat-Walker forcing term.
 *
 * Outputs variables:
 * @return x  Solution of the nonlinear system F(x) = 0, which is x (Define as the solution).
 *
 */

template<typename ResidualFun, typename PreconditionedInverseApplication, typename T>
std::pair<double, double> preconditioned_newton_krylov_solver(
  ExecutionContext& ec, ChemEnv& chem_env, ResidualFun&& residual_fun,
  PreconditionedInverseApplication&& preconditioner_inverse_application, Tensor<T>& d_e,
  std::vector<Tensor<T>>& initial_guess) {
  const auto&  nk               = chem_env.ioptions.ccsd_options.solvers.newton_krylov;
  const int    maxiter          = chem_env.ioptions.ccsd_options.ccsd_maxiter;
  const int    krylov_dims      = nk.krylov_dims;
  const double newton_tol       = nk.newton_tol;
  const double gmres_tol        = nk.gmres_tol;
  const double eta              = nk.eta;
  const double gamma            = nk.gamma;
  const double alpha            = nk.alpha;
  const bool   adaptive_forcing = nk.adaptive_forcing;

  auto  sch                = Scheduler{ec};
  auto& iterative_solution = initial_guess;

  double previous_residual_norm = 0.0;
  double residual               = 0.0;
  double energy                 = 0.0;

  for(int iter = 0; iter < maxiter; ++iter) {
    const auto step_start   = std::chrono::high_resolution_clock::now();
    auto       res_fun      = tensor_copy(ec, residual_fun(iterative_solution));
    double     res_fun_norm = compute_norm(res_fun);

    auto M_inv_res_fun_temp = preconditioner_inverse_application(res_fun);
    auto M_inv_res_fun      = tensor_copy(ec, M_inv_res_fun_temp);

    for(auto& t: M_inv_res_fun_temp) { sch.deallocate(t).execute(); }

    const auto   step_end = std::chrono::high_resolution_clock::now();
    const double step_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(step_end - step_start).count();

    chem_env.cc_context.ccsd_iter++;
    energy   = get_scalar(d_e);
    residual = res_fun_norm;
    if(ec.pg().rank() == 0) {
      std::cout << "Newton step " << std::endl;

      iteration_print(chem_env, ec.pg(), chem_env.cc_context.ccsd_iter, res_fun_norm, energy,
                      step_time);
    }

    // Write amplitudes to disk after each Newton step so the run can be restarted.
    write_amplitudes(chem_env, iter, iterative_solution);

    if(res_fun_norm < newton_tol) {
      if(ec.pg().rank() == 0) {
        std::cout << "Preconditioned Newton-Krylov is converged!" << std::endl;
      }
      for(auto& t: res_fun) sch.deallocate(t);
      for(auto& t: M_inv_res_fun) sch.deallocate(t);
      sch.execute();

      break;
    }

    // Eisenstat-Walker forcing term evalutation.
    double current_eta;
    if(iter == 0) { current_eta = eta; }
    else {
      double ratio = res_fun_norm / previous_residual_norm;
      current_eta  = std::min(0.9, gamma * std::pow(ratio, alpha));
      current_eta  = std::max(current_eta, 1e-8);
    }

    previous_residual_norm = res_fun_norm;

    // Jacobian-vector product using finite difference.
    double norm_iterative_solution = compute_norm(iterative_solution);
    auto   jvp                     = [&](const std::vector<Tensor<T>>& vector) {
      const double norm_vector  = compute_norm(vector);
      double       perturbation = std::sqrt(1e-14) * (1.0 + norm_iterative_solution) /
                            (norm_vector > 1e-12 ? norm_vector : 1e-12);
      auto perturbed_iterative_solution = tensor_copy(ec, iterative_solution);
      tensor_arithmetic(sch, perturbed_iterative_solution, vector, perturbation);

      auto perturbed_residual_function =
        tensor_copy(ec, residual_fun(perturbed_iterative_solution));
      tensor_arithmetic(sch, perturbed_residual_function, res_fun, -1.0);
      scale_tensors(perturbed_residual_function, 1.0 / perturbation);
      for(auto& t: perturbed_iterative_solution) sch.deallocate(t).execute();
      return perturbed_residual_function;
    };

    auto M_inv_jvp = [&](const std::vector<Tensor<T>>& vector) {
      auto jv = jvp(vector);

      auto result      = preconditioner_inverse_application(jv);
      auto result_copy = tensor_copy(ec, result);

      for(auto& t: jv) sch.deallocate(t);
      for(auto& t: result) sch.deallocate(t);
      sch.execute();
      return result_copy;
    };

    auto neg_M_inv_res_fun = tensor_copy(ec, M_inv_res_fun);
    scale_tensors(neg_M_inv_res_fun, -1.0);

    // GMRES inner loop to solve the linearized system M^{-1}*J*delta_x = -M^{-1}*F(x)
    auto delta_x = gmres_solver(ec, chem_env, d_e, M_inv_jvp, neg_M_inv_res_fun, krylov_dims,
                                gmres_tol, current_eta, gamma, alpha, adaptive_forcing, true);
    tensor_arithmetic(sch, iterative_solution, delta_x, 1.0);

    for(auto& t: res_fun) sch.deallocate(t);
    for(auto& t: neg_M_inv_res_fun) sch.deallocate(t);
    for(auto& t: M_inv_res_fun) sch.deallocate(t);
    for(auto& t: delta_x) sch.deallocate(t);
    sch.execute();
  }

  return {residual, energy};
}

/**
 * @brief Inexact Newton-Krylov solver:
 *
 * System: M*delta_t = -residual_fun(t_n)
 *
 * Note:
 *  - This is nearly identical to fixed-point iteration solver.
 *  - I have showed that in the MO basis, M reduced to diag(Fock matrix), we call it as energy
 * denominator.
 *  - In fixed-point iteration, we solve delta_t = -residual_fun(t_n) / diag(Fock matrix).
 *  - In inexact Newton-Krylov, we solve M*delta_t = -residual_fun(t_n) using GMRES.
 *
 * Purpose: Avoid zero denominator divisions.
 *
 * Main solvers include:
 *  1. Newton method for solving nonlinear system F(x) = 0.
 *  2. GMRES for solving the linearized system M*delta_x = -F(x).
 *
 * Main inputs:
 *  - Related to Newton: A function evaluates the residual F(x) at a given x, initail guess x,
 * convergence threshold for Newton iterations.
 *  - Related to preconditioner: Preconditoner inverse application function, krylov subspace
 * dimension for the preconditioned GMRES solve.
 *  - Related to GMRES: Krylov subspace dimension for the preconditioned GMRES solve.
 *
 * Other inputs:
 *  - Related to Newton: Maximum number of outer Newton iterations - To avoid infinite loops in case
 * of non-convergence.
 *  - Related to GMRES:
 *    - Convergence threshold to GMRES - To avoid unnecessary iterations if GMRES converges before
 * reaching the maximum krylov subspace dimension.
 *
 * Outputs: Solution of the nonlinear system F(x) = 0, which is x (Define as the solution).
 *
 * Procedure:
 *  1. Evaluate the residual F(x) at the current iterate x.
 *  2. Check for convergence: if ||F(x)|| < n_tol, then the solution has converged.
 *  3. Build the precinditioner.
 *    - M*delta_x
 *  4. Call GMRES to solve the linearized system.
 *    - As inputs GMRES takes:
 *     - The precontioner action(from step 3)
 *     - Minus right-hand side -F(x) (from step 1)
 *    - Outputs the solution delta_x in the Krylov subspace.
 *  5. Update the iterate: x (Define as the solution) += delta_x, where delta_x is the solution from
 * GMRES.
 *
 * @tparam ResidualFun Callable: (const vector<Tensor<T>>&) -> vector<Tensor<T>>&
 * @tparam PreconditionerAction Callable: (const vector<Tensor<T>>&) -> vector<Tensor<T>>&
 * @tparam T  Scalar type
 *
 * Input variables:
 * @param ec  ExecutionContext for all tensor operations
 * @param residual_fun  Residual function F(x); evaluates the nonlinear equations
 * @param preconditioner_action  Preconditioner action function M*v; evaluates the preconditioner
 * action on a given vector.
 * @param initial_guess Pass by reference: In/out: initial guess on input, solution on output.
 * @param maxiter Maximum number of outer Newton iterations
 * @param krylov_dims_precond Krylov subspace dimension for preconditioned GMRES solve
 *
 * Other input variables:
 * @param newton_tol Convergence threshold: Newton iterations stops when ||F(x)|| < newton_tol
 * @param gmres_tol Convergence threshold: GMRES iterations stops when ||F(x)|| < gmres_tol *
 *
 * Outputs variables:
 * @return x  Solution of the nonlinear system F(x) = 0, which is x (Define as the solution).
 *
 */
template<typename ResidualFun, typename PreconditionerAction, typename T>
std::pair<double, double>
inexact_newton_krylov_solver(ExecutionContext& ec, ChemEnv& chem_env, ResidualFun&& residual_fun,
                             PreconditionerAction&& precond_action, Tensor<T>& d_e,
                             std::vector<Tensor<T>>& initial_guess) {
  const auto&  nk                  = chem_env.ioptions.ccsd_options.solvers.newton_krylov;
  const int    maxiter             = chem_env.ioptions.ccsd_options.ccsd_maxiter;
  const int    krylov_dims_precond = nk.krylov_dims_precond;
  const double newton_tol          = nk.newton_tol;
  const double gmres_tol           = nk.gmres_tol;

  auto   sch                = Scheduler{ec};
  auto&  iterative_solution = initial_guess;
  double residual           = 0.0;
  double energy             = 0.0;

  for(int iter = 0; iter < maxiter; ++iter) {
    const auto   step_start   = std::chrono::high_resolution_clock::now();
    auto         res_fun      = tensor_copy(ec, residual_fun(iterative_solution));
    double       res_fun_norm = compute_norm(res_fun);
    const auto   step_end     = std::chrono::high_resolution_clock::now();
    const double step_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(step_end - step_start).count();

    energy   = get_scalar(d_e); // collective: all ranks
    residual = res_fun_norm;
    if(ec.pg().rank() == 0) {
      iteration_print(chem_env, ec.pg(), iter, res_fun_norm, energy, step_time);
    }

    // Write amplitudes to disk after each Newton step so the run can be restarted.
    write_amplitudes(chem_env, iter, iterative_solution);

    if(res_fun_norm < newton_tol) {
      if(ec.pg().rank() == 0) { std::cout << "Inexact Newton-Krylov is converged!" << std::endl; }
      for(auto& t: res_fun) sch.deallocate(t);
      sch.execute();
      break;
    }

    auto neg_res_fun = tensor_copy(ec, res_fun);
    scale_tensors(neg_res_fun, -1.0);

    // GMRES inner loop to solve the linearized system M*delta_x = -F(x)
    auto delta_x =
      gmres_solver(ec, chem_env, d_e, precond_action, neg_res_fun, krylov_dims_precond, gmres_tol);
    tensor_arithmetic(sch, iterative_solution, delta_x, 1.0);

    for(auto& t: res_fun) sch.deallocate(t);
    for(auto& t: neg_res_fun) sch.deallocate(t);
    for(auto& t: delta_x) sch.deallocate(t);
    sch.execute();
  }

  return {residual, energy};
}

} // namespace exachem::cc::solvers
