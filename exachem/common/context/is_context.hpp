/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "tamm/tamm.hpp"

class ISContext {
public:
  ISContext() = default;

  TiledIndexSpace AO_opt;   // large tilesize based
  TiledIndexSpace AO_tis;   // shell tiles
  TiledIndexSpace AO_ortho; // opt for Northo dim
  TiledIndexSpace MO;
  TiledIndexSpace MSO; // MSO used in all CC methods
  TiledIndexSpace CI;  // cholesky vectors space

  int ao_tilesize{30};
  int dfao_tilesize{30};
  int mso_tilesize{40};
  int mso_tilesize_triples{40};
};