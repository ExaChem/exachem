/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

class MP2Context {
public:
  MP2Context() = default;

  double mp2_total_energy{0};
  double mp2_correlation_energy{0};
};