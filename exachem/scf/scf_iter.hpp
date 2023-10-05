/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "scf_guess.hpp"

template<typename TensorType>
void scf_diis(ExecutionContext& ec, const TiledIndexSpace& tAO, Tensor<TensorType> D,
              Tensor<TensorType> F, Tensor<TensorType> err_mat, int iter, int max_hist,
              const SCFVars& scf_vars, const int n_lindep,
              std::vector<Tensor<TensorType>>& diis_hist,
              std::vector<Tensor<TensorType>>& fock_hist);

template<typename TensorType>
std::tuple<TensorType, TensorType> scf_iter_body(ExecutionContext& ec,
                                                 ScalapackInfo& scalapack_info, const int& iter,
                                                 const SystemData& sys_data, SCFVars& scf_vars,
                                                 TAMMTensors& ttensors, EigenTensors& etensors,
#if defined(USE_GAUXC)
                                                 GauXC::XCIntegrator<Matrix>& gauxc_integrator,
#endif
                                                 bool scf_restart = false);

template<typename TensorType>
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
compute_2bf_taskinfo(ExecutionContext& ec, const SystemData& sys_data, const SCFVars& scf_vars,
                     const libint2::BasisSet& obs, const bool do_schwarz_screen,
                     const std::vector<size_t>& shell2bf, const Matrix& SchwarzK,
                     const size_t& max_nprim4, libint2::BasisSet& shells, TAMMTensors& ttensors,
                     EigenTensors& etensors, const bool cs1s2 = false);

template<typename TensorType>
void compute_2bf(ExecutionContext& ec, ScalapackInfo& scalapack_info, const SystemData& sys_data,
                 const SCFVars& scf_vars, const libint2::BasisSet& obs,
                 const bool do_schwarz_screen, const std::vector<size_t>& shell2bf,
                 const Matrix& SchwarzK, const size_t& max_nprim4, libint2::BasisSet& shells,
                 TAMMTensors& ttensors, EigenTensors& etensors, bool& is_3c_init,
                 const bool do_density_fitting = false, double xHF = 1.);
