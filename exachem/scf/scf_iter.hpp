/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/scf/scf_compute.hpp"
#include "exachem/scf/scf_gauxc.hpp"
#include "exachem/scf/scf_guess.hpp"

namespace exachem::scf {

template<typename T>
class DefaultSCFIter {
protected:
  virtual void scf_diis(ExecutionContext& ec, ChemEnv& chem_env, const TiledIndexSpace& tAO,
                        Tensor<T> F_alpha, Tensor<T> F_beta, Tensor<T> err_mat_alpha,
                        Tensor<T> err_mat_beta, int iter, int max_hist, const SCFVars& scf_vars,
                        const int n_lindep, std::vector<Tensor<T>>& diis_hist_alpha,
                        std::vector<Tensor<T>>& diis_hist_beta,
                        std::vector<Tensor<T>>& fock_hist_alpha,
                        std::vector<Tensor<T>>& fock_hist_beta);

  virtual void compute_2bf_ri(ExecutionContext& ec, ChemEnv& chem_env,
                              ScalapackInfo& scalapack_info, const SCFVars& scf_vars,
                              const std::vector<size_t>& shell2bf, TAMMTensors& ttensors,
                              EigenTensors& etensors, bool& is_3c_init, double xHF);

  virtual void compute_3c_ints(ExecutionContext& ec, ChemEnv& chem_env, const SCFVars& scf_vars,
                               Tensor<T>& xyZ);

  virtual void compute_2c_ints(ExecutionContext& ec, ChemEnv& chem_env, EigenTensors& etensors,
                               const SCFVars& scf_vars, TAMMTensors& ttensors);

  virtual void compute_2bf_ri_direct(ExecutionContext& ec, ChemEnv& chem_env,
                                     const SCFVars& scf_vars, const std::vector<size_t>& shell2bf,
                                     TAMMTensors& ttensors, EigenTensors& etensors,
                                     const Matrix& SchwarzK);

public:
  virtual void init_ri(ExecutionContext& ec, ChemEnv& chem_env, ScalapackInfo& scalapack_info,
                       const SCFVars& scf_vars, EigenTensors& etensors, TAMMTensors& ttensors);

  virtual void compute_2bf(ExecutionContext& ec, ChemEnv& chem_env, ScalapackInfo& scalapack_info,
                           const SCFVars& scf_vars, const bool do_schwarz_screen,
                           const std::vector<size_t>& shell2bf, const Matrix& SchwarzK,
                           const size_t& max_nprim4, TAMMTensors& ttensors, EigenTensors& etensors,
                           bool& is_3c_init, const bool do_density_fitting, double xHF);

  virtual std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
  compute_2bf_taskinfo(ExecutionContext& ec, ChemEnv& chem_env, const SCFVars& scf_vars,
                       const bool do_schwarz_screen, const std::vector<size_t>& shell2bf,
                       const Matrix& SchwarzK, const size_t& max_nprim4, TAMMTensors& ttensors,
                       EigenTensors& etensors, const bool cs1s2 = false);

  virtual std::tuple<T, T> scf_iter_body(ExecutionContext& ec, ChemEnv& chem_env,
                                         ScalapackInfo& scalapack_info, const int& iter,
                                         SCFVars& scf_vars, TAMMTensors& ttensors,
                                         EigenTensors& etensors
#if defined(USE_GAUXC)
                                         ,
                                         GauXC::XCIntegrator<Matrix>& gauxc_integrator
#endif
  );
};

template<typename T>
class SCFIter: public DefaultSCFIter<T> {
  // Add/override methods here if needed in the future
};

} // namespace exachem::scf
