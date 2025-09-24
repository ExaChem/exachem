/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccsd/ccsd_util.hpp"

using namespace tamm;

namespace exachem::cc::qed_ccsd_os {

template<typename T>
using TensorMap = std::map<std::string, Tensor<T>>;

template<typename T>
extern void resid_4(Scheduler& sch, const TiledIndexSpace& MO, TensorMap<T>& tmps,
                    TensorMap<T>& scalars, const TensorMap<T>& f, const TensorMap<T>& eri,
                    const TensorMap<T>& dp, const double w0, const TensorMap<T>& t1,
                    const TensorMap<T>& t2, const double t0_1p, const TensorMap<T>& t1_1p,
                    const TensorMap<T>& t2_1p, const double t0_2p, const TensorMap<T>& t1_2p,
                    const TensorMap<T>& t2_2p, Tensor<T>& energy, TensorMap<T>& r1,
                    TensorMap<T>& r2, Tensor<T>& r0_1p, TensorMap<T>& r1_1p, TensorMap<T>& r2_1p,
                    Tensor<T>& r0_2p, TensorMap<T>& r1_2p, TensorMap<T>& r2_2p);

}; // namespace exachem::cc::qed_ccsd_os
