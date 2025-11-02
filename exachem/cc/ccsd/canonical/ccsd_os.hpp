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

namespace exachem::cc::ccsd_os {

template<typename T>
void build_tmps(Scheduler& sch, ChemEnv& chem_env, TensorMap<T>& tmps, TensorMap<T>& scalars,
                const TensorMap<T>& f, const TensorMap<T>& eri, const TensorMap<T>& t1,
                const TensorMap<T>& t2);

template<typename T>
std::tuple<TensorMap<T>, // fock
           TensorMap<T>  // eri
           >
extract_spin_blocks(Scheduler& sch, ChemEnv& chem_env, const Tensor<T>& d_f1,
                    const Tensor<T>& cholVpr);

template<typename T>
void residuals(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO, const TensorMap<T>& f,
               const TensorMap<T>& eri, const TensorMap<T>& t1, const TensorMap<T>& t2,
               Tensor<T>& energy, TensorMap<T>& r1, TensorMap<T>& r2);

void ccsd_os_driver(ExecutionContext& ec, ChemEnv& chem_env);

}; // namespace exachem::cc::ccsd_os
