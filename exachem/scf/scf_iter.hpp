/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "scf/scf_compute.hpp"
#include "scf/scf_gauxc.hpp"
#include "scf/scf_guess.hpp"

class SCFIter: public SCFCompute {
private:
  template<typename TensorType>
  void scf_diis(ExecutionContext& ec, ChemEnv& chem_env, const TiledIndexSpace& tAO,
                Tensor<TensorType> F_alpha, Tensor<TensorType> F_beta,
                Tensor<TensorType> err_mat_alpha, Tensor<TensorType> err_mat_beta, int iter,
                int max_hist, const SCFVars& scf_vars, const int n_lindep,
                std::vector<Tensor<TensorType>>& diis_hist_alpha,
                std::vector<Tensor<TensorType>>& diis_hist_beta,
                std::vector<Tensor<TensorType>>& fock_hist_alpha,
                std::vector<Tensor<TensorType>>& fock_hist_beta);

  template<typename TensorType>
  void compute_2bf_ri(ExecutionContext& ec, ChemEnv& chem_env, ScalapackInfo& scalapack_info,
                      const SCFVars& scf_vars, const std::vector<size_t>& shell2bf,
                      TAMMTensors& ttensors, EigenTensors& etensors, bool& is_3c_init, double xHF);

  template<typename TensorType>
  void compute_3c_ints(ExecutionContext& ec, ChemEnv& chem_env, const SCFVars& scf_vars,
                       Tensor<TensorType>& xyZ);

  template<typename TensorType>
  void compute_2c_ints(ExecutionContext& ec, ChemEnv& chem_env, EigenTensors& etensors,
                       const SCFVars& scf_vars, TAMMTensors& ttensors);

  template<typename TensorType>
  void compute_2bf_ri_direct(ExecutionContext& ec, ChemEnv& chem_env, const SCFVars& scf_vars,
                             const std::vector<size_t>& shell2bf, TAMMTensors& ttensors,
                             EigenTensors& etensors, const Matrix& SchwarzK);

public:
  template<typename TensorType>
  void init_ri(ExecutionContext& ec, ChemEnv& chem_env, ScalapackInfo& scalapack_info,
               const SCFVars& scf_vars, EigenTensors& etensors, TAMMTensors& ttensors);

  template<typename TensorType>
  void compute_2bf(ExecutionContext& ec, ChemEnv& chem_env, ScalapackInfo& scalapack_info,
                   const SCFVars& scf_vars, const bool do_schwarz_screen,
                   const std::vector<size_t>& shell2bf, const Matrix& SchwarzK,
                   const size_t& max_nprim4, TAMMTensors& ttensors, EigenTensors& etensors,
                   bool& is_3c_init, const bool do_density_fitting, double xHF);

  template<typename TensorType>
  std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
  compute_2bf_taskinfo(ExecutionContext& ec, ChemEnv& chem_env, const SCFVars& scf_vars,
                       const bool do_schwarz_screen, const std::vector<size_t>& shell2bf,
                       const Matrix& SchwarzK, const size_t& max_nprim4, TAMMTensors& ttensors,
                       EigenTensors& etensors, const bool cs1s2 = false);

  template<typename TensorType>
  std::tuple<TensorType, TensorType>
  scf_iter_body(ExecutionContext& ec, ChemEnv& chem_env, ScalapackInfo& scalapack_info,
                const int& iter, SCFVars& scf_vars, TAMMTensors& ttensors, EigenTensors& etensors
#if defined(USE_GAUXC)
                ,
                GauXC::XCIntegrator<Matrix>& gauxc_integrator
#endif
  );
};
