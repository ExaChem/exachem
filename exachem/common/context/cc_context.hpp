/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
// #include "tamm/tamm.hpp"

class CCContext {
public:
  CCContext() = default;

  double cc2_correlation_energy{0};
  double cc2_total_energy{0};

  double ccsd_correlation_energy{0};
  double ccsd_total_energy{0};

  double ccsdt_correlation_energy{0};
  double ccsdt_total_energy{0};

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
  std::string t2_11file;
  std::string t2_21file;
  std::string t2_12file;
  std::string t2_22file;
  std::string ccsdstatus;

  void init_filenames(std::string files_prefix) {
    this->t1file = files_prefix + ".t1amp";
    this->t2file = files_prefix + ".t2amp";

    this->t2_11file = files_prefix + ".t2_11amp";
    this->t2_21file = files_prefix + ".t2_21amp";
    this->t2_12file = files_prefix + ".t2_12amp";
    this->t2_22file = files_prefix + ".t2_22amp";

    this->ccsdstatus = files_prefix + ".ccsdstatus";
  }
};