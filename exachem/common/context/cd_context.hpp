/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
// #include "tamm/tamm.hpp"

class CDContext {
public:
  CDContext() = default;

  int num_chol_vectors{0};
};