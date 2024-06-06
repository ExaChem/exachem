/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "v2tensors.hpp"

// ---------------------------------- class V2TensorSetup -------------------------------

template<typename T>
exachem::cholesky_2e::V2Tensors<T>
exachem::cholesky_2e::setupV2Tensors(ExecutionContext& ec, Tensor<T> cholVpr, ExecutionHW ex_hw,
                                     std::vector<std::string> blocks) {
  TiledIndexSpace MO    = cholVpr.tiled_index_spaces()[0]; // MO
  TiledIndexSpace CI    = cholVpr.tiled_index_spaces()[2]; // CI
  auto [cind]           = CI.labels<1>("all");
  auto [h1, h2, h3, h4] = MO.labels<4>("occ");
  auto [p1, p2, p3, p4] = MO.labels<4>("virt");

  V2Tensors<T> v2tensors(blocks);
  v2tensors.allocate(ec, MO);
  Scheduler sch{ec};

  for(auto x: blocks) {
    // clang-format off
    if (x == "ijab") {
      sch( v2tensors.v2ijab(h1,h2,p1,p2)      =   1.0 * cholVpr(h1,p1,cind) * cholVpr(h2,p2,cind) )
         ( v2tensors.v2ijab(h1,h2,p1,p2)     +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) );
    }

    else if (x == "iajb") {
      sch( v2tensors.v2iajb(h1,p1,h2,p2)      =   1.0 * cholVpr(h1,h2,cind) * cholVpr(p1,p2,cind) )
         ( v2tensors.v2iajb(h1,p1,h2,p2)     +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) );
    }
    else if (x == "ijka") {
      sch( v2tensors.v2ijka(h1,h2,h3,p1)      =   1.0 * cholVpr(h1,h3,cind) * cholVpr(h2,p1,cind) )
         ( v2tensors.v2ijka(h1,h2,h3,p1)     +=  -1.0 * cholVpr(h2,h3,cind) * cholVpr(h1,p1,cind) );
    }
    else if (x == "ijkl") {
      sch( v2tensors.v2ijkl(h1,h2,h3,h4)      =   1.0 * cholVpr(h1,h3,cind) * cholVpr(h2,h4,cind) )
         ( v2tensors.v2ijkl(h1,h2,h3,h4)     +=  -1.0 * cholVpr(h1,h4,cind) * cholVpr(h2,h3,cind) );
    }
    else if (x == "iabc") {
      sch( v2tensors.v2iabc(h1,p1,p2,p3)      =   1.0 * cholVpr(h1,p2,cind) * cholVpr(p1,p3,cind) )
         ( v2tensors.v2iabc(h1,p1,p2,p3)     +=  -1.0 * cholVpr(h1,p3,cind) * cholVpr(p1,p2,cind) );
    }
    else if (x == "abcd") {
      sch( v2tensors.v2abcd(p1,p2,p3,p4)      =   1.0 * cholVpr(p1,p3,cind) * cholVpr(p2,p4,cind) )
         ( v2tensors.v2abcd(p1,p2,p3,p4)     +=  -1.0 * cholVpr(p1,p4,cind) * cholVpr(p2,p3,cind) );
    }
    // clang-format on
  }

  sch.execute(ex_hw);

  return v2tensors;
}

template Tensor<T> exachem::cholesky_2e::setupV2<T>(ExecutionContext& ec, TiledIndexSpace& MO,
                                                    TiledIndexSpace& CI, Tensor<T> cholVpr,
                                                    const tamm::Tile chol_count, ExecutionHW hw,
                                                    bool anti_sym);

template exachem::cholesky_2e::V2Tensors<T>
exachem::cholesky_2e::setupV2Tensors<T>(ExecutionContext& ec, Tensor<T> cholVpr, ExecutionHW ex_hw,
                                        std::vector<std::string> blocks);
