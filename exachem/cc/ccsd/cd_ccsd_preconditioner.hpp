/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccse_tensors.hpp"

namespace exachem::cc::ccsd {

// Cholesky-CCSD diagonal (Fock) preconditioner action M*y for the Newton-Krylov solver, built from
// the spin-block Fock CCSE tensors (f1_oo, f1_vv):
//   singles: M_1*y(a,i) = f_vv(a,c) y(c,i) - f_oo(k,i) y(a,k)
//   doubles: M_2*y(a,b,i,j) = f_vv(a,c) y(c,b,i,j) + f_vv(b,d) y(a,d,i,j)
//                           - f_oo(i,k) y(a,b,k,j) - f_oo(j,l) y(a,b,i,l)
//
// _os acts on the full {V,O}/{V,V,O,O} amplitude vector (open-shell, setupTensors); _cs on the
// reduced aa singles ({Va,Oa}) and abab doubles ({Va,Vb,Oa,Ob}) (closed-shell, setupTensors_cs).
// preconditoned_vector = {y_1 (singles), y_2 (doubles)}; returns {M*y_1, M*y_2}.

template<typename T>
std::vector<Tensor<T>> preconditioner_action_os(Scheduler& sch, const TiledIndexSpace& MO,
                                                CCSE_Tensors<T>& f1_oo, CCSE_Tensors<T>& f1_vv,
                                                const std::vector<Tensor<T>>& preconditoned_vector,
                                                ExecutionHW                   exhw);

template<typename T>
std::vector<Tensor<T>> preconditioner_action_cs(Scheduler& sch, const TiledIndexSpace& MO,
                                                CCSE_Tensors<T>& f1_oo, CCSE_Tensors<T>& f1_vv,
                                                const std::vector<Tensor<T>>& preconditoned_vector,
                                                ExecutionHW                   exhw);

} // namespace exachem::cc::ccsd
