/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "exachem/cholesky/v2tensors.hpp"

class CDContext {
public:
  CDContext() = default;

  int                                     num_chol_vecs{0};
  int                                     max_cvecs{0};
  Tensor<double>                          movecs_so; // spin-orbital movecs
  Tensor<double>                          d_f1;      // Fock in MO
  Tensor<double>                          cholV2;    // 2e ints in MO
  Tensor<double>                          d_v2;      // full V2 in MO
  std::vector<double>                     p_evl_sorted;
  exachem::cholesky_2e::V2Tensors<double> v2tensors; // v2 tensor blocks in MO

  std::string movecs_so_file;
  std::string f1file;
  std::string v2file;
  std::string cv_count_file;
  std::string fullV2file;

  bool readv2   = false;
  bool is_dlpno = false;
  bool is_mso   = true;

  bool keep_movecs_so{false};

  void init_filenames(std::string files_prefix) {
    this->f1file         = files_prefix + ".f1_mo";
    this->v2file         = files_prefix + ".cholv2";
    this->fullV2file     = files_prefix + ".fullV2";
    this->cv_count_file  = files_prefix + ".cholcount";
    this->movecs_so_file = files_prefix + ".movecs_so";
  }

  void read_movecs_so() { tamm::read_from_disk(movecs_so, movecs_so_file); }
};
