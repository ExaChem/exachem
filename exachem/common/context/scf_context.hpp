/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "tamm/tamm.hpp"

class SCFContext {
public:
  SCFContext() = default;

  using TensorType = double;
  double                   hf_energy{0.0};
  double                   nuc_repl_energy{0.0};
  std::vector<size_t>      shell_tile_map;
  tamm::Tensor<TensorType> C_AO;
  tamm::Tensor<TensorType> F_AO;
  tamm::Tensor<TensorType> C_beta_AO;
  tamm::Tensor<TensorType> F_beta_AO;
  bool                     no_scf;
  bool                     do_df;

  // bool scf_converged{false};
  bool skip_scf{false};

  void update(double hf_energy, double nuc_repl_energy, std::vector<size_t> shell_tile_map,
              tamm::Tensor<TensorType> C_AO, tamm::Tensor<TensorType> F_AO,
              tamm::Tensor<TensorType> C_beta_AO, tamm::Tensor<TensorType> F_beta_AO, bool no_scf) {
    this->hf_energy       = hf_energy;
    this->nuc_repl_energy = nuc_repl_energy;
    this->shell_tile_map  = shell_tile_map;
    this->C_AO            = C_AO;
    this->F_AO            = F_AO;
    this->C_beta_AO       = C_beta_AO;
    this->F_beta_AO       = F_beta_AO;
    this->no_scf          = no_scf;
  }
};
