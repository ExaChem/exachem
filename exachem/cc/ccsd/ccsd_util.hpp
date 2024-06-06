/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
// clang-format off
#include "cc/ccse_tensors.hpp"
#include "cc/diis.hpp"
#include "scf/scf_main.hpp"
#include "cholesky/cholesky_2e_driver.hpp"
// clang-format on

template<typename T>
void setup_full_t1t2(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& dt1_full,
                     Tensor<T>& dt2_full);

template<typename TensorType>
void update_r2(ExecutionContext& ec, LabeledTensor<TensorType> ltensor);

template<typename TensorType>
void init_diagonal(ExecutionContext& ec, LabeledTensor<TensorType> ltensor);

void iteration_print(ChemEnv& chem_env, const ProcGroup& pg, int iter, double residual,
                     double energy, double time, string cmethod = "CCSD");

/**
 *
 * @tparam T
 * @param MO
 * @param p_evl_sorted
 * @return pair of residual and energy
 */
template<typename T>
std::pair<double, double> rest(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1,
                               Tensor<T>& d_r2, Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& de,
                               Tensor<T>& d_r1_residual, Tensor<T>& d_r2_residual,
                               std::vector<T>& p_evl_sorted, T zshiftl, const TAMM_SIZE& noa,
                               const TAMM_SIZE& nob, bool transpose = false);

template<typename T>
std::pair<double, double>
rest_cs(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1, Tensor<T>& d_r2,
        Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& de, Tensor<T>& d_r1_residual,
        Tensor<T>& d_r2_residual, std::vector<T>& p_evl_sorted, T zshiftl, const TAMM_SIZE& noa,
        const TAMM_SIZE& nva, bool transpose = false, const bool not_spin_orbital = false);

void print_ccsd_header(const bool do_print, std::string mname = "");

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
             bool ccsd_restart = false);

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors_cs(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
                bool ccsd_restart = false);

void ccsd_stats(ExecutionContext& ec, double hf_energy, double residual, double energy,
                double thresh);

template<typename T>
void cc_print(ChemEnv& chem_env, Tensor<T> d_t1, Tensor<T> d_t2, std::string files_prefix);
