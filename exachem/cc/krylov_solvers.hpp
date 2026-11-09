/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccsd/ccsd_util.hpp"
#include <limits>
#include <stdexcept>

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
 * @brief Allocates zero-initialized TAMM tensors with the same structure as ref.
 *
 * The caller takes ownership of the allocated tensors.
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
    Tensor<T> z = t.has_spin() ? Tensor<T>{t.tiled_index_spaces(), t.spin_mask()}
                               : Tensor<T>{t.tiled_index_spaces()};
    Tensor<T>::allocate(&ec, z);
    out.push_back(z);
  }
  auto sch = Scheduler{ec};
  for(auto& tensor: out) sch(tensor() = T{0});
  sch.execute();
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
 * Allocates fresh tensors with the same structure as src, then assigns
 * the data directly. This avoids the TAMM self-assignment error that
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
  for(size_t i = 0; i < src.size(); ++i) sch(out[i]() = src[i]());
  sch.execute();
  return out;
}

/**
 * @brief Computes the inner product of two packed vectors of TAMM tensors.
 *
 * Evaluates sum_i <v[i], w[i]> using TAMM:
 *   d_s() = v[i]() * w[i]() (element-wise product summed over all indices)
 *
 * @tparam T Scalar type
 * @param ec  ExecutionContext used for tensor operations
 * @param v   Input: first vector of tensors
 * @param w   Input: second vector of tensors (same shapes as v)
 * @param d_s Caller-owned allocated scalar scratch tensor
 * @return double  Inner product sum_i trace(v[i]^T * w[i])
 */
template<typename T>
double inner_prod(ExecutionContext& ec, const std::vector<Tensor<T>>& v,
                  const std::vector<Tensor<T>>& w, Tensor<T>& d_s) {
  if(v.empty()) return 0.0;
  auto sch = Scheduler{ec};
  sch(d_s() = v[0]() * w[0]());
  for(size_t i = 1; i < v.size(); ++i) sch(d_s() += v[i]() * w[i]());
  sch.execute();
  const double innerp = get_scalar(d_s);
  return innerp;
}

// Apply the forward Jacobi operator using the orbital energies and level shift.
template<typename T>
void apply_jacobi_forward(ExecutionContext& ec, const Tensor<T>& eps,
                          const std::vector<Tensor<T>>& v, std::vector<Tensor<T>>& out,
                          double shift) {
  const auto& spaces = v[1].tiled_index_spaces();
  auto [a]           = spaces[0].template labels<1>("all");
  auto [b]           = spaces[1].template labels<1>("all");
  auto [i]           = spaces[2].template labels<1>("all");
  auto [j]           = spaces[3].template labels<1>("all");
  Scheduler{ec}(out[0](a, i) = eps(a) * v[0](a, i))(out[0](a, i) -= eps(i) * v[0](a, i))(
    out[0](a, i) += shift * v[0](a, i))(out[1](a, b, i, j) = eps(a) * v[1](a, b, i, j))(
    out[1](a, b, i, j) += eps(b) * v[1](a, b, i, j))(out[1](a, b, i, j) -=
                                                     eps(i) * v[1](a, b, i, j))(
    out[1](a, b, i, j) -= eps(j) * v[1](a, b, i, j))(out[1](a, b, i, j) +=
                                                     (2.0 * shift) * v[1](a, b, i, j))
    .execute(ec.exhw());
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
 *  2. Eisenstat-Walker forcing term for inexact Newton methods.
 *  3. Iteration printing and preconditioned residual handling.
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
 *       - the Eisenstat-Walker forcing criterion is satisfied, or
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
 * @param rhs Temporary right-hand side b; nonzero input is moved into the Krylov basis.
 *        Callers release any unconsumed tensors after return.
 * @param krylov_dims Maximum Krylov subspace dimension.
 *
 * Additional input variables:
 * @param gmres_tol Absolute convergence tolerance for the unpreconditioned GMRES residual.
 * @param eta Eisenstat-Walker forcing term supplied by the outer iteration (default = 0.1).
 *
 * The Eisenstat-Walker forcing term is typically updated externally as
 *
 *     eta = gamma * (||F(x_n)|| / ||F(x_{n-1})||)^alpha,
 *
 * where gamma and alpha are user-defined parameters.
 *
 * GMRES terminates once the residual satisfies
 *
 *            ||r|| <= eta ||b||,
 *
 * thereby avoiding over-solving the linear system during early Newton iterations.
 *
 * @param verbose
 *        If true, prints the GMRES iteration number, residual norm, current
 *        energy, and elapsed iteration time.
 * @param preconditioned
 *        Preconditioner name ("jacobi" or "identity") for testing ||M*r|| <= gmres_tol.
 *        An empty string tests ||r|| <= gmres_tol directly.
 *
 * Output variables:
 * With krylov_dims == 0, return rhs unchanged as a direct preconditioner correction.
 * @return Approximate solution x of the linear system A*x = b represented as
 *         a vector of tensors having the same layout as rhs.
 */
template<typename MVPFun, typename T>
std::vector<Tensor<T>> gmres_solver(ExecutionContext& ec, ChemEnv& chem_env, Tensor<T>& d_e,
                                    MVPFun&& mvpfn, std::vector<Tensor<T>>&& rhs, int krylov_dims,
                                    double gmres_tol, double eta = 0.1, bool verbose = false,
                                    const std::string& preconditioned = "") {
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

  auto   sch      = Scheduler{ec};
  double rhs_norm = compute_norm(rhs);
  if(!std::isfinite(rhs_norm))
    throw std::invalid_argument("GMRES right-hand side norm must be finite");
  if(rhs_norm == 0.0) return zeroslike(ec, rhs);

  Tensor<T>              eps{};
  std::vector<Tensor<T>> linear_residual, unpreconditioned;
  if(!preconditioned.empty()) {
    linear_residual  = zeroslike(ec, rhs);
    unpreconditioned = preconditioned == "jacobi" ? zeroslike(ec, rhs) : linear_residual;
    if(preconditioned == "jacobi") {
      eps      = Tensor<T>{chem_env.is_context.MSO};
      auto [p] = chem_env.is_context.MSO.labels<1>("all");
      sch.allocate(eps)(eps(p) = chem_env.cd_context.d_f1(p, p)).execute();
    }
  }

  std::vector<std::vector<Tensor<T>>> Q(krylov_dims + 1);

  Matrix H = Matrix::Zero(krylov_dims + 1, krylov_dims);

  if(verbose && ec.pg().rank() == 0) { std::cout << "Starting GMRES " << std::endl; }

  int       final_k    = -1;
  double    basis_norm = rhs_norm;
  Vector    y;
  auto      solution = std::move(rhs);
  Tensor<T> dot_scalar{};
  Tensor<T>::allocate(&ec, dot_scalar);
  for(int k = 0; k < krylov_dims; k++) {
    final_k = k;
    Q[k]    = std::move(solution);
    scale_tensors(Q[k], 1.0 / basis_norm);
    auto         iter_start  = std::chrono::high_resolution_clock::now();
    auto         v           = mvpfn(Q[k]);
    const double action_norm = compute_norm(v);
    for(int j = 0; j <= k; j++) {
      const double hij = inner_prod(ec, Q[j], v, dot_scalar);
      H(j, k) += hij;
      tensor_arithmetic(sch, v, Q[j], -hij);
    }
    sch.execute();
    H(k + 1, k) = compute_norm(v);

    const bool breakdown = H(k + 1, k) <=
                           10.0 * std::numeric_limits<double>::epsilon() * action_norm;

    Matrix Hk = H.topLeftCorner(k + 2, k + 1);

    Vector e1 = Vector::Zero(k + 2);
    e1(0)     = rhs_norm;

    y = Hk.colPivHouseholderQr().solve(e1);

    Vector residual      = e1 - Hk * y;
    double residual_norm = residual.norm();

    if(!preconditioned.empty() && !(residual_norm <= eta * rhs_norm)) {
      for(size_t n = 0; n < linear_residual.size(); ++n) {
        sch(linear_residual[n]() = -y(k) * v[n]());
        for(int q = 0; q <= k; ++q) sch(linear_residual[n]() += residual(q) * Q[q][n]());
      }
      sch.execute();
      if(preconditioned == "jacobi")
        apply_jacobi_forward(ec, eps, linear_residual, unpreconditioned,
                             chem_env.ioptions.ccsd_options.lshift);
    }
    const bool converged =
      residual_norm <= eta * rhs_norm ||
      (!preconditioned.empty() ? compute_norm(unpreconditioned) : residual_norm) <= gmres_tol;

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

    if(converged && verbose && ec.pg().rank() == 0) std::cout << "GMRES converged.\n";

    solution = std::move(v);
    if(converged || breakdown || k + 1 == krylov_dims) {
      for(auto& t: solution) sch(t() = T{0});
      sch.execute();
      break;
    }

    basis_norm = H(k + 1, k);
  }

  for(int j = 0; j <= final_k; j++)
    for(size_t i = 0; i < solution.size(); ++i) sch(solution[i]() += y(j) * Q[j][i]());
  sch.execute();

  for(auto& qvec: Q)
    for(auto& t: qvec) sch.deallocate(t);
  sch.deallocate(dot_scalar);
  for(auto& t: linear_residual) sch.deallocate(t);
  if(preconditioned == "jacobi") {
    for(auto& t: unpreconditioned) sch.deallocate(t);
    sch.deallocate(eps);
  }

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
void write_amplitudes(ExecutionContext& ec, ChemEnv& chem_env, int iter,
                      const std::vector<Tensor<T>>& amplitudes) {
  const auto& cc_options = chem_env.ioptions.ccsd_options;
  if(cc_options.writet && ((iter + 1) % cc_options.writet_iter == 0)) {
    write_to_disk(ec, amplitudes[0], chem_env.cc_context.t1file);
    write_to_disk(ec, amplitudes[1], chem_env.cc_context.t2file);
  }
}

/**
 * @brief Left-preconditioned Jacobian-free Newton-Krylov solver.
 *
 * Solves M^{-1} J delta_x = -M^{-1} F(x) with GMRES, then updates x += delta_x.
 * The caller supplies the residual and preconditioner.
 * Jacobian-vector products use finite differences of F(x).
 * Eisenstat-Walker forcing controls the inner GMRES tolerance, with krylov_dims
 * limiting the number of Arnoldi steps. Outer convergence uses ||F(x)||.
 * With krylov_dims == 0, directly apply -M^{-1} F(x) without calling GMRES.
 *
 * @tparam ResidualFun Callable returning writable residual scratch tensors. PNK may overwrite
 * their data; each evaluation must regenerate the residual. Returned tensor storage remains
 * owned by the caller and must not alias the input amplitudes.
 * @tparam Preconditioner Callable applying M^{-1} and returning newly allocated tensors.
 * @param initial_guess In/out: initial amplitudes on input, solution on output.
 * @return Final nonlinear residual norm and energy.
 */

template<typename ResidualFun, typename Preconditioner, typename T>
std::pair<double, double>
preconditioned_newton_krylov_solver(ExecutionContext& ec, ChemEnv& chem_env,
                                    ResidualFun&& residual_fun, Preconditioner&& preconditioner,
                                    Tensor<T>& d_e, std::vector<Tensor<T>>& initial_guess) {
  const auto&                   nk          = chem_env.ioptions.ccsd_options.solvers.newton_krylov;
  const int                     maxiter     = chem_env.ioptions.ccsd_options.ccsd_maxiter;
  const int                     krylov_dims = nk.krylov_dims;
  const double                  newton_tol  = nk.newton_tol;
  const double                  gmres_tol   = nk.gmres_tol;
  [[maybe_unused]] const double eta         = nk.eta;
  const double                  gamma       = nk.gamma;
  const double                  alpha       = nk.alpha;

  auto  sch                = Scheduler{ec};
  auto& iterative_solution = initial_guess;

  double previous_residual_norm = 0.0;
  double residual               = 0.0;
  double energy                 = 0.0;

  double precond_time  = 0.0;
  long   precond_calls = 0;
  auto   timed_precond = [&](const std::vector<Tensor<T>>& v) {
    const auto t0     = std::chrono::high_resolution_clock::now();
    auto       result = nk.preconditioner == "identity" ? tensor_copy(ec, v) : preconditioner(v);
    const auto t1     = std::chrono::high_resolution_clock::now();
    precond_time += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    ++precond_calls;
    return result;
  };

  auto perturbed_iterative_solution = zeroslike(ec, iterative_solution);
  for(int iter = 0; iter <= maxiter; ++iter) {
    const auto  step_start         = std::chrono::high_resolution_clock::now();
    const auto& evaluated_residual = residual_fun(iterative_solution);
    double      res_fun_norm       = compute_norm(evaluated_residual);

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
    write_amplitudes(ec, chem_env, iter, iterative_solution);

    if(!std::isfinite(res_fun_norm) || res_fun_norm <= newton_tol || iter == maxiter) {
      if(std::isfinite(res_fun_norm) && res_fun_norm <= newton_tol && ec.pg().rank() == 0) {
        std::cout << "Preconditioned Newton-Krylov is converged!" << std::endl;
      }
      break;
    }

    auto res_fun       = tensor_copy(ec, evaluated_residual);
    auto M_inv_res_fun = timed_precond(res_fun);

    // Eisenstat-Walker forcing term evaluation.
    double current_eta = eta;
    if(iter > 0) {
      double ratio = res_fun_norm / previous_residual_norm;
      current_eta  = std::min(0.9, gamma * std::pow(ratio, alpha));
      current_eta  = std::max(current_eta, 1e-8);
    }

    previous_residual_norm = res_fun_norm;

    // Jacobian-vector product using finite difference.
    double norm_iterative_solution = compute_norm(iterative_solution);
    auto   jvp                     = [&](const std::vector<Tensor<T>>& vector) {
      const double norm_vector = compute_norm(vector);
      const double h           = std::sqrt(std::numeric_limits<double>::epsilon()) *
                       (1.0 + norm_iterative_solution) / std::max(norm_vector, 1e-12);
      for(size_t i = 0; i < iterative_solution.size(); ++i)
        sch(perturbed_iterative_solution[i]() =
                                    iterative_solution[i]())(perturbed_iterative_solution[i]() += h * vector[i]());
      sch.execute();

      // Copy handles only; use the callback's scratch data for the JVP.
      auto perturbed_residual = residual_fun(perturbed_iterative_solution);
      tensor_arithmetic(sch, perturbed_residual, res_fun, -1.0);
      scale_tensors(perturbed_residual, 1.0 / h);
      return perturbed_residual;
    };

    auto M_inv_jvp = [&](const std::vector<Tensor<T>>& vector) {
      return timed_precond(jvp(vector));
    };

    scale_tensors(M_inv_res_fun, -1.0);

    // GMRES inner loop to solve the linearized system M^{-1}*J*delta_x = -M^{-1}*F(x)
    auto delta_x = krylov_dims == 0
                     ? std::move(M_inv_res_fun)
                     : gmres_solver(ec, chem_env, d_e, M_inv_jvp, std::move(M_inv_res_fun),
                                    krylov_dims, gmres_tol, current_eta, true, nk.preconditioner);
    tensor_arithmetic(sch, iterative_solution, delta_x, 1.0);

    for(auto& t: res_fun) sch.deallocate(t);
    for(auto& t: M_inv_res_fun) sch.deallocate(t);
    for(auto& t: delta_x) sch.deallocate(t);
    sch.execute();
  }

  for(auto& t: perturbed_iterative_solution) sch.deallocate(t);
  sch.execute();

  if(ec.pg().rank() == 0) {
    std::cout << "Preconditioner applications: " << precond_calls
              << ", total preconditioner time (s): " << std::to_string(precond_time) << std::endl;
  }

  return {residual, energy};
}

} // namespace exachem::cc::solvers
