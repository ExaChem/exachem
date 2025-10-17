/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <vector>

#include "exachem/cholesky/cholesky_2e.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"
#include "exachem/common/chemenv.hpp"
#include "exachem/common/system_data.hpp"
#include "tamm/eigen_utils.hpp"

namespace exachem::mp2 {
using T = double;
class CDMP2 {
protected:
  // Encapsulated state type
  struct CDMP2State {
    TiledIndexSpace O, V;
    TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;
    std::vector<T>  p_evl_sorted_occ;
    std::vector<T>  p_evl_sorted_virt;

    TiledIndexLabel p1, p2, h1, h2;
    TiledIndexLabel cind;
    Tensor<T>       dtmp;
    Tensor<T>       v2ijab;
    T               mp2_energy{0};
    T               mp2_alpha_energy{0};
    T               mp2_beta_energy{0};
    double          hf_energy{0.0};
    double          mp2_total_energy{0.0};
  };

  virtual void initialize_index_spaces(const ChemEnv& chem_env, CDMP2State& state);
  virtual void initialize_eigenvalues(const ChemEnv& chem_env, CDMP2State& state);
  virtual void compute_dtmp(ExecutionContext& ec, CDMP2State& state);
  virtual void compute_closed_shell_mp2(ExecutionContext& ec, Scheduler& sch, CDMP2State& state);
  virtual void compute_open_shell_mp2(ExecutionContext& ec, Scheduler& sch, CDMP2State& state);

  virtual void update_chemenv(ChemEnv& chem_env, const CDMP2State& state) {
    chem_env.mp2_context.mp2_correlation_energy = state.mp2_energy;
    chem_env.mp2_context.mp2_total_energy       = state.mp2_total_energy;
  }

  CDMP2(const CDMP2&)                = default;
  CDMP2& operator=(const CDMP2&)     = default;
  CDMP2(CDMP2&&) noexcept            = default;
  CDMP2& operator=(CDMP2&&) noexcept = default;

public:
  CDMP2() = default;

  virtual ~CDMP2() noexcept = default;

  virtual void run(ExecutionContext& ec, ChemEnv& chem_env);
};

void cd_mp2(ExecutionContext& ec, ChemEnv& chem_env);
} // namespace exachem::mp2
