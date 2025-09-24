/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "tamm/tamm.hpp"

class CCContext {
public:
  CCContext() = default;

  // subgroups for CCSD restarts
  // CCSD module intializes these, calling method is responsible for destroying it
  ProcGroup         sub_pg;
  ExecutionContext* sub_ec{nullptr};
  bool              use_subgroup{false};

  // CC2
  bool   task_cc2{false};
  double cc2_correlation_energy{0};
  double cc2_total_energy{0};

  double ccsd_correlation_energy{0};
  double ccsd_total_energy{0};

  double ccsdt_correlation_energy{0};
  double ccsdt_total_energy{0};

  Tensor<double> d_t1;
  Tensor<double> d_t2;
  Tensor<double> d_t3; // always full for now

  Tensor<double> d_t1_full;
  Tensor<double> d_t2_full;

  // CCSD(T)
  double ccsd_pt_correction_energy{0};
  double ccsd_pt_correlation_energy{0};
  double ccsd_pt_total_energy{0};

  // CCSD[T]
  double ccsd_st_correction_energy{0};
  double ccsd_st_correlation_energy{0};
  double ccsd_st_total_energy{0};

  // tensor files
  std::string t1file;
  std::string t2file;
  std::string t3file;
  std::string t1_1pfile;
  std::string t2_1pfile;
  std::string t1_2pfile;
  std::string t2_2pfile;

  std::string full_t1file;
  std::string full_t2file;

  bool is_converged(nlohmann::ordered_json& j, std::string task_str) {
    if(j.contains(task_str)) { return j[task_str]["converged"].get<bool>(); }
    return false;
  }

  void init_filenames(std::string files_prefix) {
    this->t1file = files_prefix + ".t1amp";
    this->t2file = files_prefix + ".t2amp";
    this->t3file = files_prefix + ".t3amp";

    this->full_t1file = files_prefix + ".full_t1amp";
    this->full_t2file = files_prefix + ".full_t2amp";

    this->t1_1pfile = files_prefix + ".t1_1pamp";
    this->t2_1pfile = files_prefix + ".t2_1pamp";
    this->t1_2pfile = files_prefix + ".t1_2pamp";
    this->t2_2pfile = files_prefix + ".t2_2pamp";
  }

  struct Keep {
    // bool fvt12 = false;
    bool fvt12_full = false;
  };

  struct Compute {
    bool fvt12_full = false;
    bool v2_full    = false;
    void set(bool ft, bool v2) {
      fvt12_full = ft;
      v2_full    = v2;
    }
  };

  Keep    keep;
  Compute compute;

  void destroy_subgroup() {
    if(sub_pg.is_valid()) {
      (*sub_ec).flush_and_sync();
      sub_pg.destroy_coll();
      delete sub_ec;
      // MemoryManagerGA::destroy_coll(sub_mgr);
    }
  }
};
