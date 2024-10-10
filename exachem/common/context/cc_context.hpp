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
};