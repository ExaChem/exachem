/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cholesky/cholesky_2e.hpp"
#include "exachem/cholesky/v2tensors.hpp"
#include <tamm/op_executor.hpp>

using namespace tamm;
using namespace tamm::new_ops;

namespace exachem::cc::ducc {
void ducc_driver(ExecutionContext& ec, ChemEnv& chem_env);

template<typename T>
void DUCC_T_CCSD_Driver(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO,
                        Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& f1,
                        cholesky_2e::V2Tensors<T>& v2tensors, IndexVector& occ_int_vec,
                        IndexVector& virt_int_vec, string& pos_str);

#if defined(USE_NWQSIM)
void ducc_qflow_driver(ExecutionContext& ec, ChemEnv& chem_env);

template<typename T>
void DUCC_T_QFLOW_Driver(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
                         const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
                         const Tensor<T>& vtijkl, const Tensor<T>& vtijka, const Tensor<T>& vtaijb,
                         const Tensor<T>& vtijab, const Tensor<T>& vtiabc, const Tensor<T>& vtabcd,
                         ExecutionHW ex_hw, T shift, IndexVector& occ_int_vec,
                         IndexVector& virt_int_vec, string& pos_str);
#endif

} // namespace exachem::cc::ducc

namespace exachem::cc::ducc::internal {

inline void reset_ducc_runcontext(ExecutionContext& ec, ChemEnv& chem_env) {
  CCSDOptions& ccsd_options                  = chem_env.ioptions.ccsd_options;
  chem_env.run_context["ducc"]["nactive_oa"] = ccsd_options.nactive_oa;
  chem_env.run_context["ducc"]["nactive_ob"] = ccsd_options.nactive_ob;
  chem_env.run_context["ducc"]["nactive_va"] = ccsd_options.nactive_va;
  chem_env.run_context["ducc"]["nactive_vb"] = ccsd_options.nactive_vb;
  chem_env.run_context["ducc"]["ducc_lvl"]   = ccsd_options.ducc_lvl;
  chem_env.run_context["ducc"]["level0"]     = false;
  chem_env.run_context["ducc"]["level1"]     = false;
  chem_env.run_context["ducc"]["level2"]     = false;
  if(ec.print()) chem_env.write_run_context();
}

// clang-format off
template<typename T>
void H_0(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
            const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
            const Tensor<T>& vtijkl, const Tensor<T>& vtijka,
            const Tensor<T>& vtaijb, const Tensor<T>& vtijab, const Tensor<T>& vtiabc, const Tensor<T>& vtabcd,
            const Tensor<T>& f1, exachem::cholesky_2e::V2Tensors<T>&v2tensors, 
            ExecutionHW ex_hw = ExecutionHW::CPU){

  const size_t nactva = chem_env.ioptions.ccsd_options.nactive_va;

  auto [i, j, k, l] = MO.labels<4>("occ_int");
  auto [a, b, c, d] = MO.labels<4>("virt_int");

  sch(ftij(i, j) = f1(i, j))
     (vtijkl(i, j, k, l) = v2tensors.v2ijkl(i, j, k, l))
     .execute(ex_hw);
  if (nactva > 0) {
  sch(ftia(i, a ) = f1(i, a))
     (ftab(a, b) = f1(a, b))
     (vtabcd(a, b, c, d) = v2tensors.v2abcd(a, b, c, d))
     (vtijka(i, j, k, a) = v2tensors.v2ijka(i, j, k, a))
     (vtiabc(i, a, b, c) = v2tensors.v2iabc(i, a, b, c))
     (vtijab(i, j, a, b) = v2tensors.v2ijab(i, j, a, b))
     (vtaijb(a, i, j, b) = -1.0 * v2tensors.v2iajb(i, a, j, b))
     .execute(ex_hw);
  }
}

template<typename T>
void F_1(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
            const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
            const Tensor<T>& vtijkl, const Tensor<T>& vtijka,
            const Tensor<T>& vtaijb, const Tensor<T>& vtijab, const Tensor<T>& vtiabc,
            const Tensor<T>& vtabcd, const Tensor<T>& f1,
            const Tensor<T>& t1, const Tensor<T>& t2, 
            ExecutionHW ex_hw = ExecutionHW::CPU){

  const size_t nactva = chem_env.ioptions.ccsd_options.nactive_va;
  const size_t nactvb = chem_env.ioptions.ccsd_options.nactive_vb;

  TiledIndexLabel a, b, c, d;
  TiledIndexLabel i, j, k, l;
  TiledIndexLabel e, f, g, h;
  TiledIndexLabel m, n, o, p;

  std::tie(a, b, c, d) = MO.labels<4>("virt_int");
  std::tie(e, f, g, h) = MO.labels<4>("virt");
  std::tie(i, j, k, l) = MO.labels<4>("occ_int");
  std::tie(m, n, o, p) = MO.labels<4>("occ");

  SymbolTable symbol_table;
  OpExecutor op_exec{sch, symbol_table};
  TAMM_REGISTER_SYMBOLS(symbol_table, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
  TAMM_REGISTER_SYMBOLS(symbol_table, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1, t1, t2);

//HH

auto op1 = (
           1.0 * (LTOp)t1(e, j) * (LTOp)f1(e, i) +    // PT ORDER = 2
           1.0 * (LTOp)f1(j, e) * (LTOp)t1(e, i)      // PT ORDER = 2
           );

ftij(i, j).update(op1);
// op_exec.print_op_binarized((LTOp)ftij(i, j), op1.clone(), true);
op_exec.execute(ftij, true, ex_hw);

  if (nactva > 0 || nactvb > 0) {
//PP

auto op2 = (
          -1.0 * (LTOp)f1(m, b) * (LTOp)t1(a, m) +    // PT ORDER = 2, Zero because of T_int
          -1.0 * (LTOp)t1(b, m) * (LTOp)f1(a, m)      // PT ORDER = 2, Zero because of T_int
           );

ftab(a, b).update(op2);
// op_exec.print_op_binarized((LTOp)ftab(a, b), op2.clone(), true);
op_exec.execute(ftab, true, ex_hw);

//HP/PH

auto op3 = (
           1.0 * (LTOp)f1(m, e) * (LTOp)t2(a, e, i, m) +    // PT ORDER = 1
          -1.0 * (LTOp)f1(m, i) * (LTOp)t1(a, m) +    // PT ORDER = 2, Zero because of T_int
           1.0 * (LTOp)f1(a, e) * (LTOp)t1(e, i)      // PT ORDER = 2
           );

ftia(i, a).update(op3);
// op_exec.print_op_binarized((LTOp)ftia(i, a), op3.clone(), true);
op_exec.execute(ftia, true, ex_hw);

//HHHH

// NO TERMS

//PPPP

// NO TERMS

//HHHP/HPHH

auto op6 = (
          -1.0 * (LTOp)f1(k, e) * (LTOp)t2(a, e, i, j)      // PT ORDER = 1
           );

vtijka(i, j, k, a).update(op6);
// op_exec.print_op_binarized((LTOp)vtijka(i, j, k, a), op6.clone(), true);
op_exec.execute(vtijka, true, ex_hw);

//PPPH/PHPP

auto op7 = (
           1.0 * (LTOp)f1(m, a) * (LTOp)t2(c, b, i, m)      // PT ORDER = 1, Zero because of T_int
           );

vtiabc(i, a, b, c).update(op7);
// op_exec.print_op_binarized((LTOp)vtiabc(i, a, b, c), op7.clone(), true);
op_exec.execute(vtiabc, true, ex_hw);

//HHPP/PPHH

auto op8 = (
           1.0 * (LTOp)f1(b, e) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 1
          -1.0 * (LTOp)f1(m, j) * (LTOp)t2(a, b, i, m) +    // PT ORDER = 1, Zero because of T_int
           1.0 * (LTOp)f1(m, i) * (LTOp)t2(a, b, j, m) +    // PT ORDER = 1, Zero because of T_int
          -1.0 * (LTOp)f1(a, e) * (LTOp)t2(b, e, i, j)      // PT ORDER = 1
           );

vtijab(i, j, a, b).update(op8);
// op_exec.print_op_binarized((LTOp)vtijab(i, j, a, b), op8.clone(), true);
op_exec.execute(vtijab, true, ex_hw);

//PHHP

// NO TERMS
  }
}

template<typename T>
void V_1(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
            const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
            const Tensor<T>& vtijkl, const Tensor<T>& vtijka,
            const Tensor<T>& vtaijb, const Tensor<T>& vtijab, const Tensor<T>& vtiabc,
            const Tensor<T>& vtabcd, exachem::cholesky_2e::V2Tensors<T>& v2tensors,
            const Tensor<T>& t1, const Tensor<T>& t2, 
            ExecutionHW ex_hw = ExecutionHW::CPU){

  const size_t nactva = chem_env.ioptions.ccsd_options.nactive_va;
  const size_t nactvb = chem_env.ioptions.ccsd_options.nactive_vb;

  TiledIndexLabel a, b, c, d;
  TiledIndexLabel i, j, k, l;
  TiledIndexLabel e, f, g, h;
  TiledIndexLabel m, n, o, p;

  std::tie(a, b, c, d) = MO.labels<4>("virt_int");
  std::tie(e, f, g, h) = MO.labels<4>("virt");
  std::tie(i, j, k, l) = MO.labels<4>("occ_int");
  std::tie(m, n, o, p) = MO.labels<4>("occ");

  SymbolTable symbol_table;
  OpExecutor op_exec{sch, symbol_table};
  TAMM_REGISTER_SYMBOLS(symbol_table, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
  TAMM_REGISTER_SYMBOLS(symbol_table, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd,
                        v2tensors.v2ijkl, v2tensors.v2ijka, v2tensors.v2iajb, v2tensors.v2ijab,
                        v2tensors.v2iabc, v2tensors.v2abcd, t1, t2);

//HH

auto op1 = (
          -(1.0/2.0) * (LTOp)v2tensors.v2ijab(m, j, e, f) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 2
          -(1.0/2.0) * (LTOp)t2(e, f, m, j) * (LTOp)v2tensors.v2ijab(i, m, e, f) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2ijka(m, j, i, e) * (LTOp)t1(e, m) +    // PT ORDER = 3
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, i, j, e)      // PT ORDER = 3
           );

ftij(i, j).update(op1);
// op_exec.print_op_binarized((LTOp)ftij(i, j), op1.clone(), true);
op_exec.execute(ftij, true, ex_hw);

  if (nactva > 0 || nactvb > 0) {
//PP

auto op2 = (
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2ijab(m, n, a, e) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, e, b) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2iabc(m, a, b, e) * (LTOp)t1(e, m) +    // PT ORDER = 3
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(m, b, a, e)      // PT ORDER = 3
           );

ftab(a, b).update(op2);
// op_exec.print_op_binarized((LTOp)ftab(a, b), op2.clone(), true);
op_exec.execute(ftab, true, ex_hw);

//HP/PH

auto op3 = (
           (1.0/2.0) * (LTOp)v2tensors.v2iabc(m, a, f, e) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 2
          -(1.0/2.0) * (LTOp)v2tensors.v2ijka(m, n, i, e) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 2
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijab(i, m, a, e) +    // PT ORDER = 3
          -1.0 * (LTOp)v2tensors.v2iajb(m, a, i, e) * (LTOp)t1(e, m)      // PT ORDER = 3
           );

ftia(i, a).update(op3);
// op_exec.print_op_binarized((LTOp)ftia(i, a), op3.clone(), true);
op_exec.execute(ftia, true, ex_hw);
  }

//HHHH

auto op4 = (
          -(1.0/2.0) * (LTOp)v2tensors.v2ijab(l, k, e, f) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 2
          -(1.0/2.0) * (LTOp)t2(e, f, l, k) * (LTOp)v2tensors.v2ijab(i, j, e, f) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2ijka(l, k, i, e) * (LTOp)t1(e, j) +    // PT ORDER = 3
           1.0 * (LTOp)v2tensors.v2ijka(l, k, j, e) * (LTOp)t1(e, i) +    // PT ORDER = 3
           1.0 * (LTOp)t1(e, k) * (LTOp)v2tensors.v2ijka(j, i, l, e) +    // PT ORDER = 3
          -1.0 * (LTOp)t1(e, l) * (LTOp)v2tensors.v2ijka(j, i, k, e)      // PT ORDER = 3
           );

vtijkl(i, j, k, l).update(op4);
// op_exec.print_op_binarized((LTOp)vtijkl(i, j, k, l), op4.clone(), true);
op_exec.execute(vtijkl, true, ex_hw);

  if (nactva > 0 || nactvb > 0) {
//PPPP

auto op5 = (
          -(1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, d, c) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 2, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(d, c, m, n) * (LTOp)v2tensors.v2ijab(m, n, a, b) +    // PT ORDER = 2, Zero because of T_int
           1.0 * (LTOp)t1(d, m) * (LTOp)v2tensors.v2iabc(m, c, a, b) +    // PT ORDER = 3, Zero because of T_int
          -1.0 * (LTOp)t1(c, m) * (LTOp)v2tensors.v2iabc(m, d, a, b) +    // PT ORDER = 3, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2iabc(m, b, c, d) * (LTOp)t1(a, m) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2iabc(m, a, c, d) * (LTOp)t1(b, m)      // PT ORDER = 3, Zero because of T_int
           );

vtabcd(a, b, c, d).update(op5);
// op_exec.print_op_binarized((LTOp)vtabcd(a, b, c, d), op5.clone(), true);
op_exec.execute(vtabcd, true, ex_hw);

//HHHP/HPHH

auto op6 = (
           1.0 * (LTOp)v2tensors.v2ijka(m, k, j, e) * (LTOp)t2(a, e, i, m) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2ijka(m, k, i, e) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 2
          -(1.0/2.0) * (LTOp)v2tensors.v2iabc(k, a, f, e) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 2
           1.0 * (LTOp)v2tensors.v2ijkl(m, k, i, j) * (LTOp)t1(a, m) +    // PT ORDER = 3, Zero because of T_int
          -1.0 * (LTOp)t1(e, k) * (LTOp)v2tensors.v2ijab(i, j, a, e) +    // PT ORDER = 3
          -1.0 * (LTOp)v2tensors.v2iajb(k, a, j, e) * (LTOp)t1(e, i) +    // PT ORDER = 3
           1.0 * (LTOp)v2tensors.v2iajb(k, a, i, e) * (LTOp)t1(e, j)      // PT ORDER = 3
           );

vtijka(i, j, k, a).update(op6);
// op_exec.print_op_binarized((LTOp)vtijka(i, j, k, a), op6.clone(), true);
op_exec.execute(vtijka, true, ex_hw);

//PPPH/PHPP

auto op7 = (
           1.0 * (LTOp)v2tensors.v2iabc(m, b, a, e) * (LTOp)t2(c, e, i, m) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2iabc(m, c, a, e) * (LTOp)t2(b, e, i, m) +    // PT ORDER = 2
          -(1.0/2.0) * (LTOp)v2tensors.v2ijka(m, n, i, a) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 2, Zero because of T_int
           1.0 * (LTOp)t1(a, m) * (LTOp)v2tensors.v2ijab(i, m, c, b) +    // PT ORDER = 3, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2abcd(c, b, e, a) * (LTOp)t1(e, i)      // PT ORDER = 3
          -1.0 * (LTOp)v2tensors.v2iajb(m, c, i, a) * (LTOp)t1(b, m) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2iajb(m, b, i, a) * (LTOp)t1(c, m)      // PT ORDER = 3, Zero because of T_int
           );

vtiabc(i, a, b, c).update(op7);
// op_exec.print_op_binarized((LTOp)vtiabc(i, a, b, c), op7.clone(), true);
op_exec.execute(vtiabc, true, ex_hw);

//HHPP/PPHH

auto op8 = (
           1.0 * (LTOp)v2tensors.v2iajb(m, a, j, e) * (LTOp)t2(b, e, i, m) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)v2tensors.v2ijkl(m, n, i, j) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 2, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2iajb(m, b, i, e) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2iajb(m, a, i, e) * (LTOp)t2(b, e, j, m) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2iajb(m, b, j, e) * (LTOp)t2(a, e, i, m) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)v2tensors.v2abcd(a, b, e, f) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 2
           1.0 * (LTOp)v2tensors.v2iabc(i, e, a, b) * (LTOp)t1(e, j) +    // PT ORDER = 3
          -1.0 * (LTOp)v2tensors.v2ijka(j, i, m, a) * (LTOp)t1(b, m) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2ijka(j, i, m, b) * (LTOp)t1(a, m) +    // PT ORDER = 3, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2iabc(j, e, a, b) * (LTOp)t1(e, i)      // PT ORDER = 3
           );

vtijab(i, j, a, b).update(op8);
// op_exec.print_op_binarized((LTOp)vtijab(i, j, a, b), op8.clone(), true);
op_exec.execute(vtijab, true, ex_hw);

//PHHP

auto op9 = (
           1.0 * (LTOp)v2tensors.v2ijab(m, i, e, b) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 2
           1.0 * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2ijab(j, m, a, e) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2ijka(m, i, j, b) * (LTOp)t1(a, m) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2iabc(i, a, b, e) * (LTOp)t1(e, j) +    // PT ORDER = 3
          -1.0 * (LTOp)t1(b, m) * (LTOp)v2tensors.v2ijka(m, j, i, a) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)t1(e, i) * (LTOp)v2tensors.v2iabc(j, b, a, e)      // PT ORDER = 3
           );

vtaijb(a, i, j, b).update(op9);
// op_exec.print_op_binarized((LTOp)vtaijb(a, i, j, b), op9.clone(), true);
op_exec.execute(vtaijb, true, ex_hw);

  }
}


template<typename T>
void F_2(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
            const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
            const Tensor<T>& vtijkl, const Tensor<T>& vtijka,
            const Tensor<T>& vtaijb, const Tensor<T>& vtijab, const Tensor<T>& vtiabc,
            const Tensor<T>& vtabcd, const Tensor<T>& f1,
            const Tensor<T>& t1, const Tensor<T>& t2, 
            const Tensor<T>& adj_scalar, ExecutionHW ex_hw = ExecutionHW::CPU){

  const size_t nactva = chem_env.ioptions.ccsd_options.nactive_va;
  const size_t nactvb = chem_env.ioptions.ccsd_options.nactive_vb;

  TiledIndexLabel a, b, c, d;
  TiledIndexLabel i, j, k, l;
  TiledIndexLabel e, f, g, h;
  TiledIndexLabel m, n, o, p;

  std::tie(a, b, c, d) = MO.labels<4>("virt_int");
  std::tie(e, f, g, h) = MO.labels<4>("virt");
  std::tie(i, j, k, l) = MO.labels<4>("occ_int");
  std::tie(m, n, o, p) = MO.labels<4>("occ");

  SymbolTable symbol_table;
  OpExecutor op_exec{sch, symbol_table};
  TAMM_REGISTER_SYMBOLS(symbol_table, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
  TAMM_REGISTER_SYMBOLS(symbol_table, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1, t1, t2);

//FC

auto opF2 = (
              (1.0/2.0) * (LTOp)f1(m, e) * (LTOp)t1(f, n) * (LTOp)t2(e, f, m, n) +
              -1.0 * (LTOp)t1(e, m) * (LTOp)f1(n, m) * (LTOp)t1(e, n) +
              1.0 * (LTOp)t1(e, m) * (LTOp)f1(e, f) * (LTOp)t1(f, m) +
              (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(m, e) * (LTOp)t1(f, n) +
              (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(e, f, n, o) +
              (-1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(f, g, m, n)
           );

adj_scalar().update(opF2);
op_exec.execute(adj_scalar, true, ex_hw);

//HH

auto op1 = (
           (1.0/2.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(n, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 2
           1.0 * (LTOp)t2(e, f, m, j) * (LTOp)f1(e, g) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 2
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(j, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 2
          -(1.0/4.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(n, i) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(e, m) * (LTOp)t1(f, i) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(e, i) * (LTOp)t1(f, m) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(j, f) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t1(e, j) * (LTOp)f1(m, f) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t1(e, j) * (LTOp)f1(m, i) * (LTOp)t1(e, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(j, m) * (LTOp)t1(e, i) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, j) * (LTOp)f1(e, f) * (LTOp)t1(f, i)      // PT ORDER = 4
           );

ftij(i, j).update(op1);
// op_exec.print_op_binarized((LTOp)ftij(i, j), op1.clone(), true);
op_exec.execute(ftij, true, ex_hw);

  if (nactva > 0 || nactvb > 0) {
//PP

auto op2 = (
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(e, f) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 2
           1.0 * (LTOp)t2(e, b, m, n) * (LTOp)f1(o, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 2
          -(1.0/4.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(a, f) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 2
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, b) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(a, m) * (LTOp)t1(e, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(e, m) * (LTOp)t1(a, n) +    // PT ORDER = 3, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, b) * (LTOp)t2(a, e, m, n)      // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t1(b, m) * (LTOp)f1(n, e) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)t1(b, m) * (LTOp)f1(n, m) * (LTOp)t1(a, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(b, m) * (LTOp)f1(a, e) * (LTOp)t1(e, m) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, b) * (LTOp)t1(a, m)      // PT ORDER = 4, Zero because of T_int
           );

ftab(a, b).update(op2);
// op_exec.print_op_binarized((LTOp)ftab(a, b), op2.clone(), true);
op_exec.execute(ftab, true, ex_hw);

//HP/PH

auto op3 = (
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, m) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 2
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 2
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, i) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 2
          -1.0 * (LTOp)t1(e, m) * (LTOp)f1(n, m) * (LTOp)t2(a, e, i, n) +    // PT ORDER = 3
           1.0 * (LTOp)t1(e, m) * (LTOp)f1(e, f) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, i) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(a, f) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(a, m) * (LTOp)t1(e, i)      // PT ORDER = 4
          -1.0 * (LTOp)f1(m, e) * (LTOp)t1(e, i) * (LTOp)t1(a, m) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, i) * (LTOp)t1(a, m)      // PT ORDER = 4, Zero because of T_int
           );

ftia(i, a).update(op3);
// op_exec.print_op_binarized((LTOp)ftia(i, a), op3.clone(), true);
op_exec.execute(ftia, true, ex_hw);
  }

//HHHH

auto op4 = (
          -(1.0/4.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(m, i) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 2
           1.0 * (LTOp)t2(e, f, l, k) * (LTOp)f1(e, g) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 2
           (1.0/4.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(m, j) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 2
          -(1.0/4.0) * (LTOp)t2(e, f, m, l) * (LTOp)f1(k, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 2
           (1.0/4.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(l, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(e, j) * (LTOp)t1(f, i) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(e, i) * (LTOp)t1(f, j) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t1(e, k) * (LTOp)f1(l, f) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t1(e, l) * (LTOp)f1(k, f) * (LTOp)t2(e, f, i, j)      // PT ORDER = 3
           );

vtijkl(i, j, k, l).update(op4);
// op_exec.print_op_binarized((LTOp)vtijkl(i, j, k, l), op4.clone(), true);
op_exec.execute(vtijkl, true, ex_hw);

  if (nactva > 0 || nactvb > 0) {
//PPPP

auto op5 = (
          -1.0 * (LTOp)t2(d, c, m, n) * (LTOp)f1(o, m) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 2, Zero because of T_int
           (1.0/4.0) * (LTOp)t2(e, d, m, n) * (LTOp)f1(e, c) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 2, Zero because of T_int
          -(1.0/4.0) * (LTOp)t2(e, c, m, n) * (LTOp)f1(e, d) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 2, Zero because of T_int
           (1.0/4.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(a, e) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 2, Zero because of T_int
          -(1.0/4.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(b, e) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 2, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(b, m) * (LTOp)t1(a, n) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(d, m) * (LTOp)f1(n, c) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 3, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(c, m) * (LTOp)f1(n, d) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(a, m) * (LTOp)t1(b, n)      // PT ORDER = 3, Zero because of T_int
           );

vtabcd(a, b, c, d).update(op5);
// op_exec.print_op_binarized((LTOp)vtabcd(a, b, c, d), op5.clone(), true);
op_exec.execute(vtabcd, true, ex_hw);

//HHHP/HPHH

auto op6 = (
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, m) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 2
           (1.0/4.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(a, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 2
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, i) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, j) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 2
          -1.0 * (LTOp)t1(e, k) * (LTOp)f1(e, f) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t1(e, k) * (LTOp)f1(m, j) * (LTOp)t2(a, e, i, m) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t1(e, k) * (LTOp)f1(m, i) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(k, m) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t1(e, k) * (LTOp)f1(a, f) * (LTOp)t2(e, f, i, j)      // PT ORDER = 3
           );

vtijka(i, j, k, a).update(op6);
// op_exec.print_op_binarized((LTOp)vtijka(i, j, k, a), op6.clone(), true);
op_exec.execute(vtijka, true, ex_hw);

//PPPH/PHPP

auto op7 = (
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(e, m) * (LTOp)t2(c, b, i, n) +    // PT ORDER = 2
          -(1.0/4.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(e, i) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 2, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(b, m) * (LTOp)t2(c, e, i, n) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(c, m) * (LTOp)t2(b, e, i, n) +      // PT ORDER = 2
           (1.0/2.0) * (LTOp)t1(a, m) * (LTOp)f1(n, i) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 3, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(a, m) * (LTOp)f1(b, e) * (LTOp)t2(c, e, i, m) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(a, m) * (LTOp)f1(c, e) * (LTOp)t2(b, e, i, m) +    // PT ORDER = 3, Zero because of T_int
          -1.0 * (LTOp)t1(a, m) * (LTOp)f1(n, m) * (LTOp)t2(c, b, i, n) +    // PT ORDER = 3, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, a) * (LTOp)t2(c, b, i, m)      // PT ORDER = 3, Zero because of T_int
           );

vtiabc(i, a, b, c).update(op7);
// op_exec.print_op_binarized((LTOp)vtiabc(i, a, b, c), op7.clone(), true);
op_exec.execute(vtiabc, true, ex_hw);

//HHPP/PPHH

auto op8 = (
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(b, m) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, i) * (LTOp)t2(a, b, j, m) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, j) * (LTOp)t2(a, b, i, m) +    // PT ORDER = 3, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(a, m) * (LTOp)t2(b, e, i, j) +    // PT ORDER = 3
          -1.0 * (LTOp)f1(m, e) * (LTOp)t1(e, j) * (LTOp)t2(a, b, i, m) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)f1(m, e) * (LTOp)t1(a, m) * (LTOp)t2(b, e, i, j) +    // PT ORDER = 3, Zero because of T_int
          -1.0 * (LTOp)f1(m, e) * (LTOp)t1(b, m) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)f1(m, e) * (LTOp)t1(e, i) * (LTOp)t2(a, b, j, m)      // PT ORDER = 3, Zero because of T_int
           );

vtijab(i, j, a, b).update(op8);
// op_exec.print_op_binarized((LTOp)vtijab(i, j, a, b), op8.clone(), true);
op_exec.execute(vtijab, true, ex_hw);

//PHHP

auto op9 = (
           1.0 * (LTOp)t2(e, b, m, i) * (LTOp)f1(e, f) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 2
          -1.0 * (LTOp)t2(e, b, m, i) * (LTOp)f1(n, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(n, j) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(i, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 2
          -(1.0/2.0) * (LTOp)t2(e, f, m, i) * (LTOp)f1(e, b) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 2
          -(1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(a, f) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 2
          -(1.0/2.0) * (LTOp)t1(b, m) * (LTOp)f1(i, e) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, i) * (LTOp)f1(m, b) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(a, m) * (LTOp)t1(e, j) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(e, j) * (LTOp)t1(a, m)      // PT ORDER = 3, Zero because of T_int
           );

vtaijb(a, i, j, b).update(op9);
// op_exec.print_op_binarized((LTOp)vtaijb(a, i, j, b), op9.clone(), true);
op_exec.execute(vtaijb, true, ex_hw);

  }
}


template<typename T>
void V_2(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
            const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
            const Tensor<T>& vtijkl, const Tensor<T>& vtijka,
            const Tensor<T>& vtaijb, const Tensor<T>& vtijab, const Tensor<T>& vtiabc,
            const Tensor<T>& vtabcd, exachem::cholesky_2e::V2Tensors<T>& v2tensors,
            const Tensor<T>& t1, const Tensor<T>& t2, const Tensor<T>& adj_scalar, ExecutionHW ex_hw = ExecutionHW::CPU){

  const size_t nactva = chem_env.ioptions.ccsd_options.nactive_va;
  const size_t nactvb = chem_env.ioptions.ccsd_options.nactive_vb;

  TiledIndexLabel a, b, c, d;
  TiledIndexLabel i, j, k, l;
  TiledIndexLabel e, f, g, h;
  TiledIndexLabel m, n, o, p;

  std::tie(a, b, c, d) = MO.labels<4>("virt_int");
  std::tie(e, f, g, h) = MO.labels<4>("virt");
  std::tie(i, j, k, l) = MO.labels<4>("occ_int");
  std::tie(m, n, o, p) = MO.labels<4>("occ");

  SymbolTable symbol_table;
  OpExecutor op_exec{sch, symbol_table};
  TAMM_REGISTER_SYMBOLS(symbol_table, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
  TAMM_REGISTER_SYMBOLS(symbol_table, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd,
                        v2tensors.v2ijkl, v2tensors.v2ijka, v2tensors.v2iajb, v2tensors.v2ijab,
                        v2tensors.v2iabc, v2tensors.v2abcd, t1, t2);

//FC

auto opV2 = (
              (1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t1(e, m) * (LTOp)t1(f, n) +
              -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, e, m, f) * (LTOp)t1(f, n) +
              (-1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(n, o, m, f) * (LTOp)t2(e, f, n, o) +
              (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(n, e, g, f) * (LTOp)t2(f, g, m, n) +
              (-1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(n, m, o, e) * (LTOp)t1(f, o) +
              (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(m, g, e, f) * (LTOp)t1(g, n) +
              -1.0 * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iajb(o, e, m, h) * (LTOp)t2(f, h, n, o) +
              (1.0/8.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijkl(o, p, m, n) * (LTOp)t2(e, f, o, p) +
              (1.0/8.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2abcd(e, f, g, h) * (LTOp)t2(g, h, m, n) +
              (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)t1(f, n) * (LTOp)v2tensors.v2ijab(m, n, e, f)
           );

adj_scalar().update(opV2);
op_exec.execute(adj_scalar, true, ex_hw);

//HH

auto op1 = (
          -1.0 * (LTOp)t2(e, f, m, j) * (LTOp)v2tensors.v2iajb(n, e, m, g) * (LTOp)t2(f, g, i, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijkl(o, j, i, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iajb(j, e, m, g) * (LTOp)t2(f, g, i, n) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, f, m, j) * (LTOp)v2tensors.v2iajb(n, e, i, g) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijkl(o, j, m, n) * (LTOp)t2(e, f, i, o) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, j) * (LTOp)v2tensors.v2ijkl(n, o, i, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, j) * (LTOp)v2tensors.v2abcd(e, f, g, h) * (LTOp)t2(g, h, i, m) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iajb(j, e, i, g) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(m, i, j, e) * (LTOp)t1(f, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, j) * (LTOp)v2tensors.v2iabc(m, g, e, f) * (LTOp)t1(g, i) +    // PT ORDER = 4
           1.0 * (LTOp)t2(e, f, m, j) * (LTOp)v2tensors.v2ijka(m, i, n, e) * (LTOp)t1(f, n) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(n, j, m, f) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(n, m, j, e) * (LTOp)t1(f, i) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(n, j, i, f) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, j) * (LTOp)v2tensors.v2iabc(i, g, e, f) * (LTOp)t1(g, m) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t1(e, j) * (LTOp)v2tensors.v2ijka(m, n, i, f) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, j) * (LTOp)v2tensors.v2iabc(m, e, g, f) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(j, e, g, f) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 4
          -1.0 * (LTOp)v2tensors.v2ijab(m, j, e, f) * (LTOp)t1(e, i) * (LTOp)t1(f, m) +    // PT ORDER = 5
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijkl(n, j, i, m) * (LTOp)t1(e, n) +    // PT ORDER = 5
          -1.0 * (LTOp)t1(e, m) * (LTOp)t1(f, j) * (LTOp)v2tensors.v2ijab(i, m, e, f) +    // PT ORDER = 5
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(j, e, m, f) * (LTOp)t1(f, i) +    // PT ORDER = 5
          -1.0 * (LTOp)t1(e, j) * (LTOp)v2tensors.v2iajb(m, e, i, f) * (LTOp)t1(f, m) +    // PT ORDER = 5
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(j, e, i, f) * (LTOp)t1(f, m)      // PT ORDER = 5
           );

ftij(i, j).update(op1);
// op_exec.print_op_binarized((LTOp)ftij(i, j), op1.clone(), true);
op_exec.execute(ftij, true, ex_hw);

  if (nactva > 0 || nactvb > 0) {
//PP

auto op2 = (
           1.0 * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2iajb(o, e, m, f) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2abcd(a, e, g, b) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2iajb(o, a, m, f) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iajb(o, e, m, b) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2ijkl(o, p, m, n) * (LTOp)t2(a, e, o, p) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2abcd(a, e, f, g) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2abcd(e, f, g, b) * (LTOp)t2(a, g, m, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iajb(o, a, m, b) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2ijka(n, m, o, e) * (LTOp)t1(a, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(n, a, b, f) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 4
           1.0 * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2iabc(m, f, a, e) * (LTOp)t1(f, n) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t1(b, m) * (LTOp)v2tensors.v2iabc(n, a, f, e) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(m, b, a, e) * (LTOp)t1(f, n) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(n, e, b, f) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2ijka(n, m, o, a) * (LTOp)t1(e, o) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(n, o, m, b) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(b, m) * (LTOp)v2tensors.v2ijka(n, o, m, e) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(m, b, e, f) * (LTOp)t1(a, n) +    // PT ORDER = 4, Zero because of T_int
           1.0 * (LTOp)t1(e, m) * (LTOp)t1(b, n) * (LTOp)v2tensors.v2ijab(m, n, a, e) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2abcd(a, e, f, b) * (LTOp)t1(f, m) +    // PT ORDER = 5
           1.0 * (LTOp)v2tensors.v2ijab(m, n, e, b) * (LTOp)t1(a, m) * (LTOp)t1(e, n) +    // PT ORDER = 5, Zero because of T_int
           1.0 * (LTOp)t1(b, m) * (LTOp)v2tensors.v2iajb(n, a, m, e) * (LTOp)t1(e, n) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, a, m, b) * (LTOp)t1(e, n) +    // PT ORDER = 5
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, e, m, b) * (LTOp)t1(a, n)      // PT ORDER = 5, Zero because of T_int
           );

ftab(a, b).update(op2);
// op_exec.print_op_binarized((LTOp)ftab(a, b), op2.clone(), true);
op_exec.execute(ftab, true, ex_hw);

//HP/PH

auto op3 = (
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(m, i, o, a) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(n, m, o, a) * (LTOp)t2(e, f, i, o) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(m, i, o, e) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(n, m, o, e) * (LTOp)t2(a, f, i, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(m, g, e, f) * (LTOp)t2(a, g, i, n) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(m, g, a, e) * (LTOp)t2(f, g, i, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(i, g, e, f) * (LTOp)t2(a, g, m, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(i, g, a, e) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 3
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, a, m, f) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, e, m, f) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, e, i, f) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijkl(n, o, i, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, a, i, f) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2abcd(a, e, f, g) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(i, m, a, e) * (LTOp)t1(f, n) +    // PT ORDER = 4
           1.0 * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t1(e, m) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 4
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(i, m, e, f) * (LTOp)t1(a, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(m, n, a, e) * (LTOp)t1(f, i) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t1(a, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t1(e, i) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, i, n, a) * (LTOp)t1(e, n) +    // PT ORDER = 5
           1.0 * (LTOp)v2tensors.v2iabc(m, a, f, e) * (LTOp)t1(e, i) * (LTOp)t1(f, m) +    // PT ORDER = 5
          -1.0 * (LTOp)v2tensors.v2ijka(m, n, i, e) * (LTOp)t1(a, m) * (LTOp)t1(e, n) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(m, f, a, e) * (LTOp)t1(f, i) +    // PT ORDER = 5
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, i, n, e) * (LTOp)t1(a, n) +    // PT ORDER = 5, Zero because of T_int
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(i, f, a, e) * (LTOp)t1(f, m)      // PT ORDER = 5
           );

ftia(i, a).update(op3);
// op_exec.print_op_binarized((LTOp)ftia(i, a), op3.clone(), true);
op_exec.execute(ftia, true, ex_hw);
  }

//HHHH

auto op4 = (
           1.0 * (LTOp)t2(e, f, l, k) * (LTOp)v2tensors.v2iajb(m, e, i, g) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2iajb(l, e, m, g) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijkl(n, l, i, j) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, f, m, l) * (LTOp)v2tensors.v2iajb(k, e, m, g) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijkl(l, k, j, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijkl(l, k, i, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, l) * (LTOp)v2tensors.v2ijkl(n, k, i, j) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, l, k) * (LTOp)v2tensors.v2iajb(m, e, j, g) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 3
          -(1.0/8.0) * (LTOp)t2(e, f, l, k) * (LTOp)v2tensors.v2ijkl(m, n, i, j) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
          -(1.0/8.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijkl(l, k, m, n) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, l, k) * (LTOp)v2tensors.v2abcd(e, f, g, h) * (LTOp)t2(g, h, i, j) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijkl(n, l, j, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, l) * (LTOp)v2tensors.v2ijkl(n, k, i, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, l) * (LTOp)v2tensors.v2ijkl(n, k, j, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijkl(n, l, i, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, m, l) * (LTOp)v2tensors.v2iajb(k, e, j, g) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2iajb(l, e, j, g) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2iajb(l, e, i, g) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, f, m, l) * (LTOp)v2tensors.v2iajb(k, e, i, g) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(l, k, j, f) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijka(j, i, l, e) * (LTOp)t1(f, m) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(l, k, m, f) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(l, k, i, f) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, l) * (LTOp)v2tensors.v2ijka(m, k, i, f) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, k) * (LTOp)v2tensors.v2ijka(m, l, i, f) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, k) * (LTOp)v2tensors.v2ijka(m, l, j, f) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, l) * (LTOp)v2tensors.v2ijka(j, i, k, e) * (LTOp)t1(f, m) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, l) * (LTOp)v2tensors.v2ijka(m, k, j, f) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 4
           1.0 * (LTOp)t2(e, f, l, k) * (LTOp)v2tensors.v2ijka(j, i, m, e) * (LTOp)t1(f, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, l) * (LTOp)v2tensors.v2ijka(m, i, k, e) * (LTOp)t1(f, j) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijka(m, j, l, e) * (LTOp)t1(f, i) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijka(m, i, l, e) * (LTOp)t1(f, j) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, l) * (LTOp)v2tensors.v2ijka(m, j, k, e) * (LTOp)t1(f, i) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, l, k) * (LTOp)v2tensors.v2iabc(i, g, e, f) * (LTOp)t1(g, j) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, l, k) * (LTOp)v2tensors.v2iabc(j, g, e, f) * (LTOp)t1(g, i) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, k) * (LTOp)v2tensors.v2iabc(l, e, g, f) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, l) * (LTOp)v2tensors.v2iabc(k, e, g, f) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, l) * (LTOp)v2tensors.v2ijkl(m, k, i, j) * (LTOp)t1(e, m) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijkl(l, k, j, m) * (LTOp)t1(e, i) +    // PT ORDER = 5
          -1.0 * (LTOp)t1(e, l) * (LTOp)t1(f, k) * (LTOp)v2tensors.v2ijab(i, j, e, f) +    // PT ORDER = 5
          -1.0 * (LTOp)v2tensors.v2ijab(l, k, e, f) * (LTOp)t1(e, i) * (LTOp)t1(f, j) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t1(e, k) * (LTOp)v2tensors.v2ijkl(m, l, i, j) * (LTOp)t1(e, m) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijkl(l, k, i, m) * (LTOp)t1(e, j) +    // PT ORDER = 5
          -1.0 * (LTOp)t1(e, k) * (LTOp)v2tensors.v2iajb(l, e, i, f) * (LTOp)t1(f, j) +    // PT ORDER = 5
           1.0 * (LTOp)t1(e, l) * (LTOp)v2tensors.v2iajb(k, e, i, f) * (LTOp)t1(f, j) +    // PT ORDER = 5
           1.0 * (LTOp)t1(e, k) * (LTOp)v2tensors.v2iajb(l, e, j, f) * (LTOp)t1(f, i) +    // PT ORDER = 5
          -1.0 * (LTOp)t1(e, l) * (LTOp)v2tensors.v2iajb(k, e, j, f) * (LTOp)t1(f, i)      // PT ORDER = 5
           );

vtijkl(i, j, k, l).update(op4);
// op_exec.print_op_binarized((LTOp)vtijkl(i, j, k, l), op4.clone(), true);
op_exec.execute(vtijkl, true, ex_hw);

  if (nactva > 0 || nactvb > 0) {
//PPPP

auto op5 = (
          -(1.0/4.0) * (LTOp)t2(e, c, m, n) * (LTOp)v2tensors.v2abcd(a, b, f, d) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(d, c, m, n) * (LTOp)v2tensors.v2ijkl(o, p, m, n) * (LTOp)t2(a, b, o, p) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/8.0) * (LTOp)t2(d, c, m, n) * (LTOp)v2tensors.v2abcd(a, b, e, f) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3, Zero because of T_int
          -1.0 * (LTOp)t2(e, c, m, n) * (LTOp)v2tensors.v2iajb(o, e, m, d) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 3
          -(1.0/8.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2abcd(e, f, d, c) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)t2(e, d, m, n) * (LTOp)v2tensors.v2iajb(o, e, m, c) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, d, m, n) * (LTOp)v2tensors.v2abcd(a, b, f, c) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2abcd(b, e, d, c) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
           1.0 * (LTOp)t2(d, c, m, n) * (LTOp)v2tensors.v2iajb(o, a, m, e) * (LTOp)t2(b, e, n, o) +    // PT ORDER = 3, Zero because of T_int
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2abcd(a, e, d, c) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(d, c, m, n) * (LTOp)v2tensors.v2iajb(o, b, m, e) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)t2(e, c, m, n) * (LTOp)v2tensors.v2iajb(o, b, m, d) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, d, m, n) * (LTOp)v2tensors.v2abcd(a, e, f, c) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, c, m, n) * (LTOp)v2tensors.v2abcd(a, e, f, d) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, c, m, n) * (LTOp)v2tensors.v2abcd(b, e, f, d) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, d, m, n) * (LTOp)v2tensors.v2iajb(o, b, m, c) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, d, m, n) * (LTOp)v2tensors.v2abcd(b, e, f, c) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, c, m, n) * (LTOp)v2tensors.v2iajb(o, a, m, d) * (LTOp)t2(b, e, n, o) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, d, m, n) * (LTOp)v2tensors.v2iajb(o, a, m, c) * (LTOp)t2(b, e, n, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t1(c, m) * (LTOp)v2tensors.v2iabc(n, b, d, e) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)t2(d, c, m, n) * (LTOp)v2tensors.v2iabc(m, e, a, b) * (LTOp)t1(e, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, c, m, n) * (LTOp)v2tensors.v2iabc(m, d, a, b) * (LTOp)t1(e, n) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(n, e, c, d) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, d, m, n) * (LTOp)v2tensors.v2iabc(m, c, b, e) * (LTOp)t1(a, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, d, m, n) * (LTOp)v2tensors.v2iabc(m, c, a, b) * (LTOp)t1(e, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(d, m) * (LTOp)v2tensors.v2ijka(n, o, m, c) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(c, m) * (LTOp)v2tensors.v2ijka(n, o, m, d) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(c, m) * (LTOp)v2tensors.v2iabc(n, a, d, e) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, c, m, n) * (LTOp)v2tensors.v2iabc(m, d, b, e) * (LTOp)t1(a, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, d, m, n) * (LTOp)v2tensors.v2iabc(m, c, a, e) * (LTOp)t1(b, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, c, m, n) * (LTOp)v2tensors.v2iabc(m, d, a, e) * (LTOp)t1(b, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(n, b, c, d) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(n, a, c, d) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(d, m) * (LTOp)v2tensors.v2iabc(n, b, c, e) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(d, m) * (LTOp)v2tensors.v2iabc(n, a, c, e) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(d, c, m, n) * (LTOp)v2tensors.v2ijka(n, m, o, a) * (LTOp)t1(b, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(d, c, m, n) * (LTOp)v2tensors.v2ijka(n, m, o, b) * (LTOp)t1(a, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(c, m) * (LTOp)v2tensors.v2abcd(a, b, e, d) * (LTOp)t1(e, m) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2ijab(m, n, d, c) * (LTOp)t1(a, m) * (LTOp)t1(b, n) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)t1(d, m) * (LTOp)t1(c, n) * (LTOp)v2tensors.v2ijab(m, n, a, b) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(d, m) * (LTOp)v2tensors.v2abcd(a, b, e, c) * (LTOp)t1(e, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2abcd(a, e, d, c) * (LTOp)t1(b, m) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2abcd(b, e, d, c) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
           1.0 * (LTOp)t1(c, m) * (LTOp)v2tensors.v2iajb(n, b, m, d) * (LTOp)t1(a, n) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)t1(c, m) * (LTOp)v2tensors.v2iajb(n, a, m, d) * (LTOp)t1(b, n) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)t1(d, m) * (LTOp)v2tensors.v2iajb(n, b, m, c) * (LTOp)t1(a, n) +    // PT ORDER = 5, Zero because of T_int
           1.0 * (LTOp)t1(d, m) * (LTOp)v2tensors.v2iajb(n, a, m, c) * (LTOp)t1(b, n)      // PT ORDER = 5, Zero because of T_int
           );

vtabcd(a, b, c, d).update(op5);
// op_exec.print_op_binarized((LTOp)vtabcd(a, b, c, d), op5.clone(), true);
op_exec.execute(vtabcd, true, ex_hw);

//HHHP/HPHH

auto op6 = (
           1.0 * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijka(m, i, n, e) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2iabc(m, g, e, f) * (LTOp)t2(a, g, i, j) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijka(m, i, n, a) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2iabc(m, g, a, e) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijka(j, i, n, a) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijka(j, i, n, e) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(m, i, k, e) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijka(m, j, n, e) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(m, j, k, e) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(m, i, k, a) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijka(m, j, n, a) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(m, j, k, a) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(j, i, k, e) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(n, m, k, e) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2iabc(i, g, e, f) * (LTOp)t2(a, g, j, m) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2iabc(i, g, a, e) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2iabc(j, g, a, e) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 3
          -(1.0/8.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijka(n, m, k, a) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2iabc(j, g, e, f) * (LTOp)t2(a, g, i, m) +    // PT ORDER = 3
          -1.0 * (LTOp)v2tensors.v2ijab(m, k, e, f) * (LTOp)t1(e, m) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijkl(n, k, i, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijab(i, j, a, e) * (LTOp)t1(f, m) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)v2tensors.v2ijab(m, k, e, f) * (LTOp)t1(a, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(k, a, m, f) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(k, a, j, f) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 4
           (1.0/4.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijab(i, j, e, f) * (LTOp)t1(a, m) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(k, e, j, f) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijkl(n, k, j, m) * (LTOp)t2(a, e, i, n) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, k) * (LTOp)v2tensors.v2iajb(m, e, i, f) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(k, e, i, f) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijkl(n, k, i, j) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, k) * (LTOp)v2tensors.v2iajb(m, a, i, f) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, k) * (LTOp)v2tensors.v2iajb(m, a, j, f) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, k) * (LTOp)v2tensors.v2iajb(m, e, j, f) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(k, a, i, f) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t1(e, k) * (LTOp)v2tensors.v2ijkl(m, n, i, j) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(k, e, m, f) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, k) * (LTOp)v2tensors.v2abcd(a, e, f, g) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijab(j, m, a, e) * (LTOp)t1(f, i) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)v2tensors.v2ijab(i, m, a, e) * (LTOp)t1(f, j) +    // PT ORDER = 4
          -1.0 * (LTOp)v2tensors.v2ijab(m, k, e, f) * (LTOp)t1(e, i) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
           1.0 * (LTOp)v2tensors.v2ijab(m, k, e, f) * (LTOp)t1(e, j) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 4
          -1.0 * (LTOp)v2tensors.v2iabc(k, a, f, e) * (LTOp)t1(e, i) * (LTOp)t1(f, j) +    // PT ORDER = 5
          -1.0 * (LTOp)v2tensors.v2ijka(m, k, j, e) * (LTOp)t1(e, i) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2ijka(m, k, i, e) * (LTOp)t1(a, m) * (LTOp)t1(e, j) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, k) * (LTOp)v2tensors.v2ijka(j, i, m, a) * (LTOp)t1(e, m) +    // PT ORDER = 5
          -1.0 * (LTOp)t1(e, k) * (LTOp)v2tensors.v2iabc(i, f, a, e) * (LTOp)t1(f, j) +    // PT ORDER = 5
          -1.0 * (LTOp)t1(e, k) * (LTOp)v2tensors.v2ijka(j, i, m, e) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(j, i, k, e) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, j, k, a) * (LTOp)t1(e, i) +    // PT ORDER = 5
           1.0 * (LTOp)t1(e, k) * (LTOp)v2tensors.v2iabc(j, f, a, e) * (LTOp)t1(f, i) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, i, k, a) * (LTOp)t1(e, j)      // PT ORDER = 5
           );

vtijka(i, j, k, a).update(op6);
// op_exec.print_op_binarized((LTOp)vtijka(i, j, k, a), op6.clone(), true);
op_exec.execute(vtijka, true, ex_hw);

//PPPH/PHPP

auto op7 = (
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(m, a, c, e) * (LTOp)t2(b, f, i, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2ijka(m, i, o, e) * (LTOp)t2(c, b, n, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2iabc(i, f, b, e) * (LTOp)t2(c, f, m, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2ijka(m, i, o, c) * (LTOp)t2(b, e, n, o) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2ijka(n, m, o, c) * (LTOp)t2(b, e, i, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2ijka(n, m, o, b) * (LTOp)t2(c, e, i, o) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(i, a, c, e) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2iabc(m, f, b, e) * (LTOp)t2(c, f, i, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2ijka(n, m, o, e) * (LTOp)t2(c, b, i, o) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2iabc(m, f, c, b) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2ijka(m, i, o, b) * (LTOp)t2(c, e, n, o) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2iabc(m, f, c, e) * (LTOp)t2(b, f, i, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2iabc(i, f, c, b) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(m, a, c, b) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(m, a, b, e) * (LTOp)t2(c, f, i, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2iabc(i, f, c, e) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(m, a, e, f) * (LTOp)t2(c, b, i, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(i, a, b, e) * (LTOp)t2(c, f, m, n) +    // PT ORDER = 3
          -(1.0/8.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iabc(i, a, e, f) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 3, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2abcd(c, b, f, a) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 4
           1.0 * (LTOp)v2tensors.v2ijab(m, n, e, a) * (LTOp)t1(e, m) * (LTOp)t2(c, b, i, n) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2ijab(m, n, c, b) * (LTOp)t1(e, i) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2ijab(i, m, c, b) * (LTOp)t1(e, n) +    // PT ORDER = 4
           (1.0/4.0) * (LTOp)t1(a, m) * (LTOp)v2tensors.v2abcd(c, b, e, f) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, e, a) * (LTOp)t1(e, i) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 4, Zero because of T_int
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, e, i, a) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(a, m) * (LTOp)v2tensors.v2iajb(n, b, i, e) * (LTOp)t2(c, e, m, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(a, m) * (LTOp)v2tensors.v2iajb(n, c, i, e) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 4, Zero because of T_int
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2abcd(b, e, f, a) * (LTOp)t2(c, f, i, m) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(a, m) * (LTOp)v2tensors.v2ijkl(n, o, i, m) * (LTOp)t2(c, b, n, o) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, c, m, a) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 4
           1.0 * (LTOp)t1(a, m) * (LTOp)v2tensors.v2iajb(n, c, m, e) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, e, m, a) * (LTOp)t2(c, b, i, n) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(a, m) * (LTOp)v2tensors.v2iajb(n, b, m, e) * (LTOp)t2(c, e, i, n) +    // PT ORDER = 4, Zero because of T_int
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, b, m, a) * (LTOp)t2(c, e, i, n) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2abcd(c, e, f, a) * (LTOp)t2(b, f, i, m) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, c, i, a) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(n, b, i, a) * (LTOp)t2(c, e, m, n) +    // PT ORDER = 4
           1.0 * (LTOp)v2tensors.v2ijab(m, n, e, a) * (LTOp)t1(c, m) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2ijab(m, n, e, a) * (LTOp)t1(b, m) * (LTOp)t2(c, e, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2ijab(i, m, c, e) * (LTOp)t1(b, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)v2tensors.v2ijab(i, m, b, e) * (LTOp)t1(c, n) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2ijka(m, n, i, a) * (LTOp)t1(c, m) * (LTOp)t1(b, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(i, a, b, e) * (LTOp)t1(c, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(i, a, c, e) * (LTOp)t1(b, m) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2iabc(m, b, a, e) * (LTOp)t1(c, m) * (LTOp)t1(e, i) +    // PT ORDER = 5, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2iabc(m, c, a, e) * (LTOp)t1(e, i) * (LTOp)t1(b, m) +    // PT ORDER = 5, Zero because of T_int
           1.0 * (LTOp)t1(a, m) * (LTOp)v2tensors.v2ijka(m, i, n, b) * (LTOp)t1(c, n) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)t1(a, m) * (LTOp)v2tensors.v2ijka(m, i, n, c) * (LTOp)t1(b, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(m, a, c, b) * (LTOp)t1(e, i) +    // PT ORDER = 5
          -1.0 * (LTOp)t1(a, m) * (LTOp)v2tensors.v2iabc(m, e, c, b) * (LTOp)t1(e, i) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(a, m) * (LTOp)v2tensors.v2iabc(i, e, c, b) * (LTOp)t1(e, m)      // PT ORDER = 5, Zero because of T_int
           );

vtiabc(i, a, b, c).update(op7);
// op_exec.print_op_binarized((LTOp)vtiabc(i, a, b, c), op7.clone(), true);
op_exec.execute(vtiabc, true, ex_hw);

//HHPP/PPHH

auto op8 = (
          -1.0 * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t2(b, e, i, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(i, m, a, e) * (LTOp)t2(b, f, j, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(j, m, b, e) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t2(a, b, i, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(i, m, b, e) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t2(e, f, i, m) * (LTOp)t2(a, b, j, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(i, m, e, f) * (LTOp)t2(a, b, j, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(j, m, a, e) * (LTOp)t2(b, f, i, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(m, n, a, e) * (LTOp)t2(b, f, i, j) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t2(b, e, i, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(m, n, b, e) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t2(a, e, i, j) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(j, m, e, f) * (LTOp)t2(a, b, i, n) +    // PT ORDER = 3
           1.0 * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t2(a, e, i, m) * (LTOp)t2(b, f, j, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(i, m, a, b) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(j, m, a, b) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(i, j, a, e) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(i, j, b, e) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)v2tensors.v2ijab(m, n, e, f) * (LTOp)t2(e, f, i, j) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 3, Zero because of T_int
           (1.0/8.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(m, n, a, b) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 3
           (1.0/8.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2ijab(i, j, e, f) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(i, f, a, e) * (LTOp)t2(b, f, j, m) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(i, f, b, e) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
          -1.0 * (LTOp)v2tensors.v2ijka(m, n, i, e) * (LTOp)t1(e, m) * (LTOp)t2(a, b, j, n) +    // PT ORDER = 4
           1.0 * (LTOp)v2tensors.v2ijka(m, n, j, e) * (LTOp)t1(e, m) * (LTOp)t2(a, b, i, n) +    // PT ORDER = 4
           1.0 * (LTOp)v2tensors.v2ijka(m, n, i, e) * (LTOp)t1(b, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 4, Zero because of T_int
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(m, f, a, e) * (LTOp)t2(b, f, i, j) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(j, i, n, a) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, j, n, e) * (LTOp)t2(a, b, i, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(j, i, n, b) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)v2tensors.v2iabc(m, a, f, e) * (LTOp)t1(b, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2ijka(m, n, i, e) * (LTOp)t1(a, m) * (LTOp)t2(b, e, j, n) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(m, f, b, e) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)v2tensors.v2ijka(m, n, j, e) * (LTOp)t1(e, i) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2iabc(m, b, f, e) * (LTOp)t1(e, i) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
           1.0 * (LTOp)v2tensors.v2ijka(m, n, j, e) * (LTOp)t1(a, m) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(j, f, a, b) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(j, i, n, e) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(j, f, a, e) * (LTOp)t2(b, f, i, m) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, j, n, a) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(m, f, a, b) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, i, n, e) * (LTOp)t2(a, b, j, n) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, i, n, a) * (LTOp)t2(b, e, j, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(i, f, a, b) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)v2tensors.v2iabc(m, b, f, e) * (LTOp)t1(a, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2iabc(m, a, f, e) * (LTOp)t1(e, m) * (LTOp)t2(b, f, i, j) +    // PT ORDER = 4
          -1.0 * (LTOp)v2tensors.v2ijka(m, n, j, e) * (LTOp)t1(b, m) * (LTOp)t2(a, e, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)v2tensors.v2ijka(m, n, i, e) * (LTOp)t1(e, j) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, j, n, b) * (LTOp)t2(a, e, i, n) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(m, i, n, b) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(j, f, b, e) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 4
           1.0 * (LTOp)v2tensors.v2iabc(m, a, f, e) * (LTOp)t1(e, i) * (LTOp)t2(b, f, j, m) +    // PT ORDER = 4
           1.0 * (LTOp)v2tensors.v2iabc(m, b, f, e) * (LTOp)t1(e, j) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 4
          -1.0 * (LTOp)v2tensors.v2iabc(m, b, f, e) * (LTOp)t1(e, m) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 4
          -1.0 * (LTOp)v2tensors.v2iabc(m, a, f, e) * (LTOp)t1(e, j) * (LTOp)t2(b, f, i, m) +    // PT ORDER = 4
           1.0 * (LTOp)v2tensors.v2ijkl(m, n, i, j) * (LTOp)t1(a, m) * (LTOp)t1(b, n) +    // PT ORDER = 5, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2abcd(a, b, e, f) * (LTOp)t1(e, i) * (LTOp)t1(f, j) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijab(i, j, a, e) * (LTOp)t1(b, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijab(i, j, b, e) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijab(j, m, a, b) * (LTOp)t1(e, i) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijab(i, m, a, b) * (LTOp)t1(e, j) +    // PT ORDER = 5
           1.0 * (LTOp)v2tensors.v2iajb(m, a, i, e) * (LTOp)t1(e, j) * (LTOp)t1(b, m) +    // PT ORDER = 5, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2iajb(m, b, j, e) * (LTOp)t1(e, i) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2iajb(m, b, i, e) * (LTOp)t1(a, m) * (LTOp)t1(e, j) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2iajb(m, a, j, e) * (LTOp)t1(e, i) * (LTOp)t1(b, m)      // PT ORDER = 5, Zero because of T_int
           );

vtijab(i, j, a, b).update(op8);
// op_exec.print_op_binarized((LTOp)vtijab(i, j, a, b), op8.clone(), true);
op_exec.execute(vtijab, true, ex_hw);

//PHHP

auto op9 = (
          -1.0 * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2iajb(n, e, m, f) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2iajb(n, e, j, f) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2iajb(i, e, m, f) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, f, m, i) * (LTOp)v2tensors.v2iajb(n, e, m, b) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2iajb(i, e, j, f) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2iajb(n, a, m, f) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, i) * (LTOp)v2tensors.v2iajb(n, a, m, b) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2ijkl(o, i, j, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2ijkl(o, i, m, n) * (LTOp)t2(a, e, j, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2ijkl(n, o, j, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 3
           1.0 * (LTOp)t2(e, f, m, i) * (LTOp)v2tensors.v2abcd(a, e, g, b) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, i) * (LTOp)v2tensors.v2abcd(e, f, g, b) * (LTOp)t2(a, g, j, m) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, f, m, i) * (LTOp)v2tensors.v2iajb(n, e, j, b) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iajb(i, e, m, b) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2iajb(n, a, j, f) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
          -1.0 * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2iajb(i, a, m, f) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2abcd(a, e, f, g) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iajb(i, e, j, b) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, i) * (LTOp)v2tensors.v2iajb(n, a, j, b) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2iajb(i, a, j, f) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)v2tensors.v2iajb(i, a, m, b) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t1(e, i) * (LTOp)v2tensors.v2iabc(m, a, b, f) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, i) * (LTOp)v2tensors.v2iabc(m, e, b, f) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t2(e, f, m, i) * (LTOp)v2tensors.v2iabc(j, b, e, f) * (LTOp)t1(a, m) +    // PT ORDER = 4, Zero because of T_int
           1.0 * (LTOp)t1(b, m) * (LTOp)v2tensors.v2ijka(n, i, m, e) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(b, m) * (LTOp)v2tensors.v2ijka(n, i, j, e) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2ijka(m, j, n, a) * (LTOp)t1(e, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, i) * (LTOp)v2tensors.v2iabc(j, b, a, e) * (LTOp)t1(f, m) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2ijka(m, j, i, a) * (LTOp)t1(e, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(n, i, j, b) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4
           (1.0/4.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2ijka(n, m, i, a) * (LTOp)t1(e, j) +    // PT ORDER = 4
          -1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2ijka(n, i, m, b) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 4
           1.0 * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(i, e, b, f) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
          -1.0 * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2iabc(m, f, a, e) * (LTOp)t1(f, j) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)v2tensors.v2ijka(m, j, i, e) * (LTOp)t1(a, n) +    // PT ORDER = 4, Zero because of T_int
           1.0 * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2iabc(j, f, a, e) * (LTOp)t1(f, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iabc(i, a, b, f) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
           1.0 * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2ijka(m, j, n, e) * (LTOp)t1(a, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, f, m, i) * (LTOp)v2tensors.v2iabc(m, b, a, e) * (LTOp)t1(f, j) +    // PT ORDER = 4
           (1.0/4.0) * (LTOp)t1(e, i) * (LTOp)v2tensors.v2ijka(m, n, j, b) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t1(b, m) * (LTOp)v2tensors.v2iabc(i, a, f, e) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4, Zero because of T_int
          -1.0 * (LTOp)t1(b, m) * (LTOp)t1(e, i) * (LTOp)v2tensors.v2ijab(j, m, a, e) +    // PT ORDER = 5, Zero because of T_int
          1.0 * (LTOp)t1(b, m) * (LTOp)v2tensors.v2ijkl(n, i, j, m) * (LTOp)t1(a, n) +    // PT ORDER = 5, Zero because of T_int
           1.0 * (LTOp)t1(e, i) * (LTOp)v2tensors.v2abcd(a, e, f, b) * (LTOp)t1(f, j) +    // PT ORDER = 5
          -1.0 * (LTOp)v2tensors.v2ijab(m, i, e, b) * (LTOp)t1(e, j) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(i, e, j, b) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)t1(e, i) * (LTOp)v2tensors.v2iajb(m, e, j, b) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, i) * (LTOp)v2tensors.v2iajb(m, a, j, b) * (LTOp)t1(e, m) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)v2tensors.v2iajb(i, a, m, b) * (LTOp)t1(e, j) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(b, m) * (LTOp)v2tensors.v2iajb(i, a, j, e) * (LTOp)t1(e, m) +    // PT ORDER = 5, Zero because of T_int
          -1.0 * (LTOp)t1(b, m) * (LTOp)v2tensors.v2iajb(i, a, m, e) * (LTOp)t1(e, j)      // PT ORDER = 5, Zero because of T_int
           );

vtaijb(a, i, j, b).update(op9);
// op_exec.print_op_binarized((LTOp)vtaijb(a, i, j, b), op9.clone(), true);
op_exec.execute(vtaijb, true, ex_hw);

  }
}


template<typename T>
void F_3(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
            const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
            const Tensor<T>& vtijkl, const Tensor<T>& vtijka,
            const Tensor<T>& vtaijb, const Tensor<T>& vtijab, const Tensor<T>& vtiabc,
            const Tensor<T>& vtabcd, const Tensor<T>& f1,
            const Tensor<T>& t1, const Tensor<T>& t2, const Tensor<T>& adj_scalar, ExecutionHW ex_hw = ExecutionHW::CPU){

  const size_t nactva = chem_env.ioptions.ccsd_options.nactive_va;
  const size_t nactvb = chem_env.ioptions.ccsd_options.nactive_vb;

  TiledIndexLabel a, b, c, d;
  TiledIndexLabel i, j, k, l;
  TiledIndexLabel e, f, g, h;
  TiledIndexLabel m, n, o, p;

  std::tie(a, b, c, d) = MO.labels<4>("virt_int");
  std::tie(e, f, g, h) = MO.labels<4>("virt");
  std::tie(i, j, k, l) = MO.labels<4>("occ_int");
  std::tie(m, n, o, p) = MO.labels<4>("occ");

  SymbolTable symbol_table;
  OpExecutor op_exec{sch, symbol_table};
  TAMM_REGISTER_SYMBOLS(symbol_table, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
  TAMM_REGISTER_SYMBOLS(symbol_table, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1, t1, t2);

//FC

auto opF3 = (
              (-2.0/3.0) * (LTOp)f1(m, e) * (LTOp)t1(f, n) * (LTOp)t1(f, m) * (LTOp)t1(e, n) +
              (-2.0/3.0) * (LTOp)t1(e, m) * (LTOp)t1(f, n) * (LTOp)f1(m, f) * (LTOp)t1(e, n) +
              (-1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, m) * (LTOp)t1(f, o) * (LTOp)t2(e, f, n, o) +
              (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, f) * (LTOp)t1(g, n) * (LTOp)t2(f, g, m, n) +
              (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t1(e, n) * (LTOp)t1(f, o) +
              (-1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t1(f, m) * (LTOp)t1(g, n) +
              (-1.0/3.0) * (LTOp)f1(m, e) * (LTOp)t2(f, g, n, o) * (LTOp)t1(f, m) * (LTOp)t2(e, g, n, o) +
              (-1.0/3.0) * (LTOp)f1(m, e) * (LTOp)t2(f, g, n, o) * (LTOp)t1(e, n) * (LTOp)t2(f, g, m, o) +
              (1.0/6.0) * (LTOp)f1(m, e) * (LTOp)t2(f, g, n, o) * (LTOp)t1(f, n) * (LTOp)t2(e, g, m, o) +
              (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(m, e) * (LTOp)t1(g, o) * (LTOp)t2(f, g, n, o) +
              (-1.0/3.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, n, o) * (LTOp)f1(m, f) * (LTOp)t2(e, g, n, o) +
              (-1.0/3.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, n, o) * (LTOp)f1(n, e) * (LTOp)t2(f, g, m, o)
           );

adj_scalar().update(opF3);
op_exec.execute(adj_scalar, true, ex_hw);

//HH

auto op1 = (
           (1.0/6.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(n, g) * (LTOp)t1(e, m) * (LTOp)t2(f, g, i, n) +    // PT ORDER = 4
           (1.0/3.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(n, g) * (LTOp)t1(g, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, n, j) * (LTOp)f1(f, n) * (LTOp)t2(e, g, i, m) +    // PT ORDER = 4
           (1.0/3.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, n, j) * (LTOp)f1(e, n) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, n, j) * (LTOp)f1(f, i) * (LTOp)t2(e, g, m, n) +    // PT ORDER = 4
           (2.0/3.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, n, j) * (LTOp)f1(f, m) * (LTOp)t2(e, g, i, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, j) * (LTOp)t2(f, g, m, n) * (LTOp)f1(f, m) * (LTOp)t2(e, g, i, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(j, g) * (LTOp)t1(e, m) * (LTOp)t2(f, g, i, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(n, g) * (LTOp)t1(e, i) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 4
          -(2.0/3.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(n, g) * (LTOp)t1(e, n) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 4
          -(1.0/12.0) * (LTOp)t1(e, j) * (LTOp)t2(f, g, m, n) * (LTOp)f1(f, i) * (LTOp)t2(e, g, m, n) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, n, j) * (LTOp)f1(e, i) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t1(e, j) * (LTOp)t2(f, g, m, n) * (LTOp)f1(e, m) * (LTOp)t2(f, g, i, n) +    // PT ORDER = 4
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(j, g) * (LTOp)t1(g, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(j, g) * (LTOp)t1(e, i) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(n, g) * (LTOp)t1(g, i) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(n, m) * (LTOp)t1(e, i) * (LTOp)t1(f, n) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)t1(f, j) * (LTOp)f1(e, g) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(e, g) * (LTOp)t1(f, i) * (LTOp)t1(g, m) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(e, g) * (LTOp)t1(g, i) * (LTOp)t1(f, m) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)t1(f, j) * (LTOp)f1(n, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t2(e, f, m, j) * (LTOp)f1(n, i) * (LTOp)t1(e, m) * (LTOp)t1(f, n) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)t1(f, j) * (LTOp)f1(f, g) * (LTOp)t2(e, g, i, m) +    // PT ORDER = 5
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(j, m) * (LTOp)t1(e, i) * (LTOp)t1(f, n) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, n) * (LTOp)f1(j, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 5
          -(1.0/3.0) * (LTOp)t1(e, m) * (LTOp)t1(f, j) * (LTOp)f1(n, i) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, j) * (LTOp)f1(e, i) * (LTOp)t1(f, m) +    // PT ORDER = 6
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)t1(f, j) * (LTOp)f1(f, m) * (LTOp)t1(e, i) +    // PT ORDER = 6
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(j, f) * (LTOp)t1(e, i) * (LTOp)t1(f, m) +    // PT ORDER = 6
          -(1.0/2.0) * (LTOp)t1(e, j) * (LTOp)f1(m, f) * (LTOp)t1(f, i) * (LTOp)t1(e, m)      // PT ORDER = 6
           );

ftij(i, j).update(op1);
// op_exec.print_op_binarized((LTOp)ftij(i, j), op1.clone(), true);
op_exec.execute(ftij, true, ex_hw);

  if (nactva > 0 || nactvb > 0) {
//PP

auto op2 = (
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, o) * (LTOp)f1(f, n) * (LTOp)t2(a, e, m, o) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(o, f) * (LTOp)t1(e, m) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 4
          -(1.0/3.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, o) * (LTOp)f1(f, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 4
          -(1.0/3.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(o, f) * (LTOp)t1(e, o) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, b) * (LTOp)t1(e, m) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(o, f) * (LTOp)t1(a, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 4, Zero because of T_int
           (2.0/3.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(o, f) * (LTOp)t1(f, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 4
          -(2.0/3.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, o) * (LTOp)f1(e, n) * (LTOp)t2(a, f, m, o) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(b, m) * (LTOp)t2(e, f, n, o) * (LTOp)f1(e, n) * (LTOp)t2(a, f, m, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, o) * (LTOp)f1(a, n) * (LTOp)t2(e, f, m, o) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, b) * (LTOp)t1(e, o) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, b) * (LTOp)t1(a, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/4.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(o, f) * (LTOp)t1(a, o) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/4.0) * (LTOp)t1(b, m) * (LTOp)t2(e, f, n, o) * (LTOp)f1(e, m) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/12.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, o) * (LTOp)f1(a, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t1(b, m) * (LTOp)t2(e, f, n, o) * (LTOp)f1(a, n) * (LTOp)t2(e, f, m, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, n) * (LTOp)f1(e, b) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)t1(b, n) * (LTOp)f1(e, f) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/3.0) * (LTOp)t1(e, m) * (LTOp)t1(b, n) * (LTOp)f1(a, f) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(e, f) * (LTOp)t1(a, m) * (LTOp)t1(f, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, b) * (LTOp)t1(a, m) * (LTOp)t1(f, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(a, f) * (LTOp)t1(e, m) * (LTOp)t1(f, n) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(o, m) * (LTOp)t1(a, n) * (LTOp)t1(e, o) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)t1(b, n) * (LTOp)f1(o, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(o, m) * (LTOp)t1(a, o) * (LTOp)t1(e, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(b, m) * (LTOp)t1(e, n) * (LTOp)f1(o, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, b) * (LTOp)t1(a, m) * (LTOp)t1(e, n) +    // PT ORDER = 6, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(b, m) * (LTOp)f1(n, e) * (LTOp)t1(a, n) * (LTOp)t1(e, m) +    // PT ORDER = 6, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(b, n) * (LTOp)f1(a, m) * (LTOp)t1(e, n) +    // PT ORDER = 6, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(b, m) * (LTOp)t1(e, n) * (LTOp)f1(e, m) * (LTOp)t1(a, n)      // PT ORDER = 6, Zero because of T_int
           );

ftab(a, b).update(op2);
// op_exec.print_op_binarized((LTOp)ftab(a, b), op2.clone(), true);
op_exec.execute(ftab, true, ex_hw);

//HP/PH

auto op3 = (
          -(2.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, g) * (LTOp)t2(e, g, i, m) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 3
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, g) * (LTOp)t2(a, e, i, m) * (LTOp)t2(f, g, n, o) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, g) * (LTOp)t2(e, g, i, o) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, g) * (LTOp)t2(a, g, i, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 3
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, g) * (LTOp)t2(a, e, i, o) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, g) * (LTOp)t2(e, f, i, m) * (LTOp)t2(a, g, n, o) +    // PT ORDER = 3
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, g) * (LTOp)t2(e, f, i, o) * (LTOp)t2(a, g, m, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t1(e, n) * (LTOp)t2(a, f, i, o) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t1(f, m) * (LTOp)t2(a, g, i, n) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t1(a, n) * (LTOp)t2(e, f, i, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t1(f, i) * (LTOp)t2(a, g, m, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, i) * (LTOp)t1(e, m) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t1(e, o) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t1(e, i) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, g) * (LTOp)t1(e, m) * (LTOp)t2(f, g, i, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t1(a, m) * (LTOp)t2(f, g, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t1(g, m) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, i) * (LTOp)t1(e, o) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, i) * (LTOp)t1(a, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t1(a, o) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, g) * (LTOp)t1(e, i) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 4
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, g) * (LTOp)t1(g, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t1(g, i) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
          -(2.0/3.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t1(f, m) * (LTOp)t2(a, e, i, n) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t1(e, i) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, n) * (LTOp)f1(a, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 5
          -(2.0/3.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t1(e, n) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, n) * (LTOp)f1(e, i) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 5
          -(2.0/3.0) * (LTOp)t1(e, m) * (LTOp)t1(f, n) * (LTOp)f1(f, m) * (LTOp)t2(a, e, i, n) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t1(a, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t1(f, i) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 5
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, m) * (LTOp)t1(f, i) * (LTOp)t1(a, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t1(a, n) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, m) * (LTOp)t1(e, i) * (LTOp)t1(f, n) +    // PT ORDER = 5
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, i) * (LTOp)t1(a, m) * (LTOp)t1(f, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, m) * (LTOp)t1(e, i) * (LTOp)t1(a, n) +    // PT ORDER = 6, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, i) * (LTOp)t1(a, m) * (LTOp)t1(e, n) +    // PT ORDER = 6, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(a, f) * (LTOp)t1(e, i) * (LTOp)t1(f, m)      // PT ORDER = 6
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, f) * (LTOp)t1(f, i) * (LTOp)t1(a, m)      // PT ORDER = 6, Zero because of T_int
           );

ftia(i, a).update(op3);
// op_exec.print_op_binarized((LTOp)ftia(i, a), op3.clone(), true);
op_exec.execute(ftia, true, ex_hw);
  }

//HHHH

auto op4 = (
          -(1.0/4.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(m, g) * (LTOp)t1(g, i) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
          -(1.0/12.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, l, k) * (LTOp)f1(e, i) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(m, g) * (LTOp)t1(e, i) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, k) * (LTOp)t2(f, g, m, l) * (LTOp)f1(f, m) * (LTOp)t2(e, g, i, j) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, l, k) * (LTOp)f1(f, i) * (LTOp)t2(e, g, j, m) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, l) * (LTOp)t2(f, g, m, k) * (LTOp)f1(f, m) * (LTOp)t2(e, g, i, j) +    // PT ORDER = 4
           (2.0/3.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, l, k) * (LTOp)f1(f, m) * (LTOp)t2(e, g, i, j) +    // PT ORDER = 4
          -(2.0/3.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(m, g) * (LTOp)t1(e, m) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, l, k) * (LTOp)f1(f, j) * (LTOp)t2(e, g, i, m) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(m, g) * (LTOp)t1(e, j) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t1(e, m) * (LTOp)t2(f, g, l, k) * (LTOp)f1(e, j) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 4
           (1.0/4.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(m, g) * (LTOp)t1(g, j) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t1(e, k) * (LTOp)t2(f, g, m, l) * (LTOp)f1(e, m) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 4
           (1.0/4.0) * (LTOp)t1(e, l) * (LTOp)t2(f, g, m, k) * (LTOp)f1(e, m) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(l, g) * (LTOp)t1(e, m) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, l) * (LTOp)f1(k, g) * (LTOp)t1(e, m) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 4
          -(1.0/12.0) * (LTOp)t2(e, f, m, l) * (LTOp)f1(k, g) * (LTOp)t1(g, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(l, g) * (LTOp)t1(g, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, k) * (LTOp)t2(f, g, m, l) * (LTOp)f1(f, j) * (LTOp)t2(e, g, i, m) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, k) * (LTOp)t2(f, g, m, l) * (LTOp)f1(f, i) * (LTOp)t2(e, g, j, m) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, l) * (LTOp)t2(f, g, m, k) * (LTOp)f1(f, j) * (LTOp)t2(e, g, i, m) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, l) * (LTOp)t2(f, g, m, k) * (LTOp)f1(f, i) * (LTOp)t2(e, g, j, m) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, l) * (LTOp)f1(k, g) * (LTOp)t1(e, i) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(l, g) * (LTOp)t1(e, i) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, l) * (LTOp)f1(k, g) * (LTOp)t1(e, j) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(l, g) * (LTOp)t1(e, j) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 4
          -(1.0/3.0) * (LTOp)t1(e, l) * (LTOp)t1(f, k) * (LTOp)f1(m, i) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(e, g) * (LTOp)t1(g, i) * (LTOp)t1(f, j) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(m, i) * (LTOp)t1(e, j) * (LTOp)t1(f, m) +    // PT ORDER = 5
           (1.0/3.0) * (LTOp)t1(e, l) * (LTOp)t1(f, k) * (LTOp)f1(m, j) * (LTOp)t2(e, f, i, m) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, l) * (LTOp)f1(k, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 5
           (1.0/6.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(m, j) * (LTOp)t1(e, i) * (LTOp)t1(f, m) +    // PT ORDER = 5
           (1.0/3.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(l, m) * (LTOp)t1(e, i) * (LTOp)t1(f, j) +    // PT ORDER = 5
          -(1.0/3.0) * (LTOp)t2(e, f, m, l) * (LTOp)f1(k, m) * (LTOp)t1(e, i) * (LTOp)t1(f, j) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t1(e, l) * (LTOp)t1(f, k) * (LTOp)f1(f, g) * (LTOp)t2(e, g, i, j) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t2(e, f, l, k) * (LTOp)f1(e, g) * (LTOp)t1(f, i) * (LTOp)t1(g, j) +    // PT ORDER = 5
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, k) * (LTOp)f1(l, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, l) * (LTOp)t1(f, k) * (LTOp)f1(e, g) * (LTOp)t2(f, g, i, j)      // PT ORDER = 5
           );

vtijkl(i, j, k, l).update(op4);
// op_exec.print_op_binarized((LTOp)vtijkl(i, j, k, l), op4.clone(), true);
op_exec.execute(vtijkl, true, ex_hw);

  if (nactva > 0) {
//PPPP

auto op5 = (
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(d, c, n, o) * (LTOp)f1(b, n) * (LTOp)t2(a, e, m, o) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(c, m) * (LTOp)t2(e, d, n, o) * (LTOp)f1(e, n) * (LTOp)t2(a, b, m, o) +    // PT ORDER = 4, Zero because of T_int
           (2.0/3.0) * (LTOp)t1(e, m) * (LTOp)t2(d, c, n, o) * (LTOp)f1(e, n) * (LTOp)t2(a, b, m, o) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(d, c, n, o) * (LTOp)f1(a, n) * (LTOp)t2(b, e, m, o) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, d, m, n) * (LTOp)f1(o, c) * (LTOp)t1(e, m) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(d, m) * (LTOp)t2(e, c, n, o) * (LTOp)f1(e, n) * (LTOp)t2(a, b, m, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/12.0) * (LTOp)t2(e, c, m, n) * (LTOp)f1(o, d) * (LTOp)t1(e, o) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/4.0) * (LTOp)t1(c, m) * (LTOp)t2(e, d, n, o) * (LTOp)f1(e, m) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, c, m, n) * (LTOp)f1(o, d) * (LTOp)t1(e, m) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 4
          -(2.0/3.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(o, e) * (LTOp)t1(e, m) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/4.0) * (LTOp)t1(d, m) * (LTOp)t2(e, c, n, o) * (LTOp)f1(e, m) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/12.0) * (LTOp)t1(e, m) * (LTOp)t2(d, c, n, o) * (LTOp)f1(b, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 4
          -(1.0/12.0) * (LTOp)t2(e, d, m, n) * (LTOp)f1(o, c) * (LTOp)t1(e, o) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(o, e) * (LTOp)t1(a, m) * (LTOp)t2(b, e, n, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/12.0) * (LTOp)t1(e, m) * (LTOp)t2(d, c, n, o) * (LTOp)f1(a, m) * (LTOp)t2(b, e, n, o) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(o, e) * (LTOp)t1(b, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/4.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(o, e) * (LTOp)t1(a, o) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/4.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(o, e) * (LTOp)t1(b, o) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(c, m) * (LTOp)t2(e, d, n, o) * (LTOp)f1(b, n) * (LTOp)t2(a, e, m, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(c, m) * (LTOp)t2(e, d, n, o) * (LTOp)f1(a, n) * (LTOp)t2(b, e, m, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(d, m) * (LTOp)t2(e, c, n, o) * (LTOp)f1(a, n) * (LTOp)t2(b, e, m, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(d, m) * (LTOp)t2(e, c, n, o) * (LTOp)f1(b, n) * (LTOp)t2(a, e, m, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(e, c, m, n) * (LTOp)f1(o, d) * (LTOp)t1(b, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, c, m, n) * (LTOp)f1(o, d) * (LTOp)t1(a, m) * (LTOp)t2(b, e, n, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, d, m, n) * (LTOp)f1(o, c) * (LTOp)t1(b, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(e, d, m, n) * (LTOp)f1(o, c) * (LTOp)t1(a, m) * (LTOp)t2(b, e, n, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(d, n) * (LTOp)f1(e, c) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/3.0) * (LTOp)t2(e, d, m, n) * (LTOp)f1(e, c) * (LTOp)t1(a, m) * (LTOp)t1(b, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(d, m) * (LTOp)t1(c, n) * (LTOp)f1(o, m) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(c, m) * (LTOp)t1(d, n) * (LTOp)f1(o, m) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/3.0) * (LTOp)t2(e, c, m, n) * (LTOp)f1(e, d) * (LTOp)t1(a, m) * (LTOp)t1(b, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(c, n) * (LTOp)f1(e, d) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(a, e) * (LTOp)t1(b, m) * (LTOp)t1(e, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(b, e) * (LTOp)t1(a, m) * (LTOp)t1(e, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(o, m) * (LTOp)t1(a, n) * (LTOp)t1(b, o) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(o, m) * (LTOp)t1(a, o) * (LTOp)t1(b, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/3.0) * (LTOp)t1(d, m) * (LTOp)t1(c, n) * (LTOp)f1(a, e) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/3.0) * (LTOp)t1(d, m) * (LTOp)t1(c, n) * (LTOp)f1(b, e) * (LTOp)t2(a, e, m, n)      // PT ORDER = 5, Zero because of T_int
           );

vtabcd(a, b, c, d).update(op5);
// op_exec.print_op_binarized((LTOp)vtabcd(a, b, c, d), op5.clone(), true);
op_exec.execute(vtabcd, true, ex_hw);

//HHHP/HPHH

auto op6 = (
           (1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, g) * (LTOp)t2(a, e, i, j) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, g) * (LTOp)t2(a, g, i, j) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, g) * (LTOp)t2(a, e, i, j) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 3
          -(2.0/3.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, g) * (LTOp)t2(e, g, i, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
          -(1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, g) * (LTOp)t2(e, g, i, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, g) * (LTOp)t2(e, f, i, j) * (LTOp)t2(a, g, m, n) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, g) * (LTOp)t2(e, f, i, m) * (LTOp)t2(a, g, j, n) +    // PT ORDER = 3
           (2.0/3.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, g) * (LTOp)t2(a, e, i, n) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, g) * (LTOp)t2(e, g, i, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
          -(1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, g) * (LTOp)t2(e, g, i, n) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 3
           (2.0/3.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, g) * (LTOp)t2(e, g, i, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, g) * (LTOp)t2(a, e, i, m) * (LTOp)t2(f, g, j, n) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, g) * (LTOp)t2(a, g, i, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
          -(1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, g) * (LTOp)t2(a, e, i, m) * (LTOp)t2(f, g, j, n) +    // PT ORDER = 3
          -(1.0/24.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, g) * (LTOp)t2(e, f, i, j) * (LTOp)t2(a, g, m, n) +    // PT ORDER = 3
           (1.0/3.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, g) * (LTOp)t2(e, f, i, n) * (LTOp)t2(a, g, j, m) +    // PT ORDER = 3
          -(1.0/3.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, g) * (LTOp)t2(a, g, i, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, i) * (LTOp)t1(a, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, i) * (LTOp)t1(e, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, g) * (LTOp)t1(f, m) * (LTOp)t2(a, g, i, j) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(a, g) * (LTOp)t1(e, m) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(a, g) * (LTOp)t1(g, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4
          -(1.0/12.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, j) * (LTOp)t1(a, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, j) * (LTOp)t1(e, m) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, g) * (LTOp)t1(a, m) * (LTOp)t2(f, g, i, j) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, g) * (LTOp)t1(g, m) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, m) * (LTOp)t1(e, n) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, m) * (LTOp)t1(a, n) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, m) * (LTOp)t1(e, n) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, m) * (LTOp)t1(a, n) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, m) * (LTOp)t1(e, i) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, m) * (LTOp)t1(e, j) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, g) * (LTOp)t1(f, i) * (LTOp)t2(a, g, j, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, g) * (LTOp)t1(f, j) * (LTOp)t2(a, g, i, m) +    // PT ORDER = 4
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, m) * (LTOp)t1(e, i) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, j) * (LTOp)t1(e, i) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, i) * (LTOp)t1(e, n) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(k, m) * (LTOp)t1(e, j) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, i) * (LTOp)t1(e, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(n, j) * (LTOp)t1(e, n) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, g) * (LTOp)t1(g, i) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(a, g) * (LTOp)t1(e, i) * (LTOp)t2(f, g, j, m) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, g) * (LTOp)t1(g, j) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(a, g) * (LTOp)t1(e, j) * (LTOp)t2(f, g, i, m) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(k, f) * (LTOp)t1(e, i) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 5
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(k, f) * (LTOp)t1(f, m) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, k) * (LTOp)f1(e, i) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 5
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(k, f) * (LTOp)t1(e, j) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 5
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(k, f) * (LTOp)t1(a, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, k) * (LTOp)f1(m, f) * (LTOp)t1(a, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, k) * (LTOp)f1(e, j) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)t1(f, k) * (LTOp)f1(f, m) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 5
           (1.0/3.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(a, m) * (LTOp)t1(e, i) * (LTOp)t1(f, j) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t1(e, k) * (LTOp)f1(m, f) * (LTOp)t1(f, i) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 5
          -(1.0/3.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, j) * (LTOp)t1(f, i) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/3.0) * (LTOp)t2(e, f, m, k) * (LTOp)f1(e, i) * (LTOp)t1(a, m) * (LTOp)t1(f, j) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, k) * (LTOp)f1(a, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, k) * (LTOp)f1(m, f) * (LTOp)t1(e, m) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, k) * (LTOp)f1(m, f) * (LTOp)t1(f, j) * (LTOp)t2(a, e, i, m)      // PT ORDER = 5
           );

vtijka(i, j, k, a).update(op6);
// op_exec.print_op_binarized((LTOp)vtijka(i, j, k, a), op6.clone(), true);
op_exec.execute(vtijka, true, ex_hw);

//PPPH/PHPP

auto op7 = (
           (1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, f) * (LTOp)t2(c, e, i, m) * (LTOp)t2(b, f, n, o) +    // PT ORDER = 3
          -(2.0/3.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, f) * (LTOp)t2(c, e, m, o) * (LTOp)t2(b, f, i, n) +    // PT ORDER = 3
           (1.0/3.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, f) * (LTOp)t2(c, e, i, o) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
           (1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, f) * (LTOp)t2(c, f, m, o) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 3
          -(2.0/3.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, f) * (LTOp)t2(c, f, i, m) * (LTOp)t2(b, e, n, o) +    // PT ORDER = 3
          -(1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, f) * (LTOp)t2(c, b, i, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 3, Zero because of T_int
           (2.0/3.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, f) * (LTOp)t2(c, b, m, o) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 3
          -(1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, a) * (LTOp)t2(c, e, i, m) * (LTOp)t2(b, f, n, o) +    // PT ORDER = 3
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, a) * (LTOp)t2(c, e, m, o) * (LTOp)t2(b, f, i, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, f) * (LTOp)t2(c, b, i, o) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 3
          -(1.0/3.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, f) * (LTOp)t2(c, f, m, n) * (LTOp)t2(b, e, i, o) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, f) * (LTOp)t2(c, b, m, n) * (LTOp)t2(e, f, i, o) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, a) * (LTOp)t2(c, e, i, o) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, a) * (LTOp)t2(c, e, m, n) * (LTOp)t2(b, f, i, o) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, a) * (LTOp)t2(c, b, i, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, a) * (LTOp)t2(c, b, m, o) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 3
           (1.0/24.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, a) * (LTOp)t2(c, b, m, n) * (LTOp)t2(e, f, i, o) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(e, f) * (LTOp)t1(b, m) * (LTOp)t2(c, f, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, i) * (LTOp)t1(e, m) * (LTOp)t2(c, b, n, o) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(e, f) * (LTOp)t1(f, m) * (LTOp)t2(c, b, i, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(b, f) * (LTOp)t1(e, m) * (LTOp)t2(c, f, i, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, m) * (LTOp)t1(e, o) * (LTOp)t2(c, b, i, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, m) * (LTOp)t1(e, n) * (LTOp)t2(c, b, i, o) +    // PT ORDER = 4
          -(1.0/4.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(e, f) * (LTOp)t1(f, i) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/12.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, i) * (LTOp)t1(e, o) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, m) * (LTOp)t1(e, i) * (LTOp)t2(c, b, n, o) +    // PT ORDER = 4
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, a) * (LTOp)t1(f, m) * (LTOp)t2(c, b, i, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(c, f) * (LTOp)t1(e, m) * (LTOp)t2(b, f, i, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, a) * (LTOp)t1(f, i) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/12.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(b, f) * (LTOp)t1(e, i) * (LTOp)t2(c, f, m, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(e, f) * (LTOp)t1(c, m) * (LTOp)t2(b, f, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/12.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(c, f) * (LTOp)t1(e, i) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, m) * (LTOp)t1(b, n) * (LTOp)t2(c, e, i, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, m) * (LTOp)t1(c, n) * (LTOp)t2(b, e, i, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, i) * (LTOp)t1(b, m) * (LTOp)t2(c, e, n, o) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, m) * (LTOp)t1(b, o) * (LTOp)t2(c, e, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, i) * (LTOp)t1(c, m) * (LTOp)t2(b, e, n, o) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(o, m) * (LTOp)t1(c, o) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, a) * (LTOp)t1(b, m) * (LTOp)t2(c, f, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(c, f) * (LTOp)t1(b, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(b, f) * (LTOp)t1(f, m) * (LTOp)t2(c, e, i, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(b, f) * (LTOp)t1(c, m) * (LTOp)t2(e, f, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(c, f) * (LTOp)t1(f, m) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 4
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, a) * (LTOp)t1(c, m) * (LTOp)t2(b, f, i, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(a, m) * (LTOp)f1(n, e) * (LTOp)t1(b, n) * (LTOp)t2(c, e, i, m) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(a, n) * (LTOp)f1(e, i) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(a, m) * (LTOp)f1(n, e) * (LTOp)t1(c, n) * (LTOp)t2(b, e, i, m) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, a) * (LTOp)t1(e, i) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/3.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(e, i) * (LTOp)t1(c, m) * (LTOp)t1(b, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(a, n) * (LTOp)f1(b, m) * (LTOp)t2(c, e, i, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/3.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(c, m) * (LTOp)t1(e, i) * (LTOp)t1(b, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, a) * (LTOp)t1(c, m) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(a, m) * (LTOp)f1(n, e) * (LTOp)t1(e, i) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/3.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(b, m) * (LTOp)t1(c, n) * (LTOp)t1(e, i) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, a) * (LTOp)t1(b, m) * (LTOp)t2(c, e, i, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(a, n) * (LTOp)f1(c, m) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(a, m) * (LTOp)f1(n, e) * (LTOp)t1(e, m) * (LTOp)t2(c, b, i, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, a) * (LTOp)t1(e, n) * (LTOp)t2(c, b, i, m) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(a, m) * (LTOp)t1(e, n) * (LTOp)f1(e, m) * (LTOp)t2(c, b, i, n)      // PT ORDER = 5, Zero because of T_int
           );

vtiabc(i, a, b, c).update(op7);
// op_exec.print_op_binarized((LTOp)vtiabc(i, a, b, c), op7.clone(), true);
op_exec.execute(vtiabc, true, ex_hw);

//HHPP/PPHH

auto op8 = (
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(a, e, i, j) * (LTOp)t2(b, f, n, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(a, b, i, m) * (LTOp)t2(f, g, j, n) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(a, f, i, m) * (LTOp)t2(b, g, j, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(e, f, i, j) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(b, e, i, j) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(b, g, i, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(a, g, i, m) * (LTOp)t2(b, f, j, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(f, g, i, m) * (LTOp)t2(a, b, j, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(b, g, i, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
          -(1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, j) * (LTOp)t2(b, e, i, m) * (LTOp)t2(a, f, n, o) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(f, g, i, j) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, g) * (LTOp)t2(a, e, i, m) * (LTOp)t2(f, g, j, n) +    // PT ORDER = 3
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, g) * (LTOp)t2(e, g, i, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(b, f, i, m) * (LTOp)t2(a, g, j, n) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(b, e, i, n) * (LTOp)t2(a, f, j, o) +    // PT ORDER = 3
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, i) * (LTOp)t2(a, e, m, o) * (LTOp)t2(b, f, j, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(a, f, i, j) * (LTOp)t2(b, g, m, n) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, g) * (LTOp)t2(a, g, i, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, g) * (LTOp)t2(a, e, i, j) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(a, g, i, j) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, j) * (LTOp)t2(a, e, i, o) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
          -(1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, i) * (LTOp)t2(a, e, j, m) * (LTOp)t2(b, f, n, o) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, i) * (LTOp)t2(a, e, j, o) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, g) * (LTOp)t2(b, f, i, j) * (LTOp)t2(a, g, m, n) +    // PT ORDER = 3
          -(1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, g) * (LTOp)t2(e, g, i, m) * (LTOp)t2(b, f, j, n) +    // PT ORDER = 3
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, j) * (LTOp)t2(a, e, i, m) * (LTOp)t2(b, f, n, o) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(a, e, i, o) * (LTOp)t2(b, f, j, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(b, e, i, o) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 3
           (1.0/2.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(a, e, i, n) * (LTOp)t2(b, f, j, o) +    // PT ORDER = 3
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, g) * (LTOp)t2(b, e, i, m) * (LTOp)t2(f, g, j, n) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, g) * (LTOp)t2(e, g, i, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, i) * (LTOp)t2(a, b, m, o) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, j) * (LTOp)t2(a, b, i, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, g) * (LTOp)t2(e, f, i, m) * (LTOp)t2(a, g, j, n) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(a, b, i, n) * (LTOp)t2(e, f, j, o) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(a, b, i, o) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, j) * (LTOp)t2(e, f, i, m) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, i) * (LTOp)t2(a, b, j, m) * (LTOp)t2(e, f, n, o) +    // PT ORDER = 3, Zero because of T_int
           (1.0/24.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, g) * (LTOp)t2(e, f, i, j) * (LTOp)t2(a, g, m, n) +    // PT ORDER = 3
           (1.0/24.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, i) * (LTOp)t2(a, b, m, n) * (LTOp)t2(e, f, j, o) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, j) * (LTOp)t2(b, e, i, o) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, g) * (LTOp)t2(b, e, i, j) * (LTOp)t2(f, g, m, n) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, g) * (LTOp)t2(e, f, i, m) * (LTOp)t2(b, g, j, n) +    // PT ORDER = 3
          -(1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, i) * (LTOp)t2(a, e, m, n) * (LTOp)t2(b, f, j, o) +    // PT ORDER = 3
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(e, f, i, n) * (LTOp)t2(a, b, j, o) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, g) * (LTOp)t2(b, g, i, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 3
           (1.0/12.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, g) * (LTOp)t2(e, g, i, j) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 3
           (1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, m) * (LTOp)t2(e, f, i, o) * (LTOp)t2(a, b, j, n) +    // PT ORDER = 3
          -(1.0/24.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(o, j) * (LTOp)t2(e, f, i, o) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/24.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, g) * (LTOp)t2(e, f, i, j) * (LTOp)t2(b, g, m, n) +    // PT ORDER = 3
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, m) * (LTOp)t1(b, n) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 4, Zero because of T_int
           (2.0/3.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(e, f, i, j) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, m) * (LTOp)t1(a, n) * (LTOp)t2(b, f, i, j) +    // PT ORDER = 4, Zero because of T_int
           (2.0/3.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(b, f, i, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 4
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(e, f, i, m) * (LTOp)t2(a, b, j, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(e, f, i, n) * (LTOp)t2(a, b, j, m) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(b, f, i, j) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, m) * (LTOp)t1(f, j) * (LTOp)t2(a, b, i, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(a, b, i, n) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(a, b, i, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, m) * (LTOp)t1(e, n) * (LTOp)t2(b, f, i, j) +    // PT ORDER = 4
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, j) * (LTOp)t1(f, m) * (LTOp)t2(a, b, i, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(a, f, i, j) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 4
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, m) * (LTOp)t1(e, n) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 4
          -(2.0/3.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(a, f, i, m) * (LTOp)t2(b, e, j, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(b, e, i, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(a, e, i, j) * (LTOp)t2(b, f, m, n) +    // PT ORDER = 4
          -(2.0/3.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(a, e, i, n) * (LTOp)t2(b, f, j, m) +    // PT ORDER = 4
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, m) * (LTOp)t1(f, i) * (LTOp)t2(a, b, j, n) +    // PT ORDER = 4
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, i) * (LTOp)t1(f, m) * (LTOp)t2(a, b, j, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, j) * (LTOp)t1(f, i) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, i) * (LTOp)t1(f, j) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 4, Zero because of T_int
           (2.0/3.0) * (LTOp)t1(e, m) * (LTOp)f1(n, f) * (LTOp)t2(b, e, i, n) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, m) * (LTOp)t1(b, n) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, m) * (LTOp)t1(a, n) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, i) * (LTOp)t1(a, m) * (LTOp)t2(b, f, j, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, j) * (LTOp)t1(a, m) * (LTOp)t2(b, f, i, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, j) * (LTOp)t1(b, m) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, i) * (LTOp)t1(b, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, m) * (LTOp)t1(e, j) * (LTOp)t2(a, f, i, n) +    // PT ORDER = 4
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(b, m) * (LTOp)t1(e, i) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 4
           (1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, m) * (LTOp)t1(e, j) * (LTOp)t2(b, f, i, n) +    // PT ORDER = 4
          -(1.0/3.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(a, m) * (LTOp)t1(e, i) * (LTOp)t2(b, f, j, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, j) * (LTOp)t1(b, m) * (LTOp)t2(a, e, i, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(b, f) * (LTOp)t1(a, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(a, f) * (LTOp)t1(e, j) * (LTOp)t2(b, f, i, m) +    // PT ORDER = 5
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(b, f) * (LTOp)t1(e, i) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(b, f) * (LTOp)t1(e, j) * (LTOp)t2(a, f, i, m) +    // PT ORDER = 5
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, j) * (LTOp)t1(e, i) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, i) * (LTOp)t1(e, j) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, i) * (LTOp)t1(b, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, i) * (LTOp)t1(e, n) * (LTOp)t2(a, b, j, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(a, f) * (LTOp)t1(b, m) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, j) * (LTOp)t1(e, n) * (LTOp)t2(a, b, i, m) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(b, f) * (LTOp)t1(f, m) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, j) * (LTOp)t1(a, m) * (LTOp)t2(b, e, i, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, f) * (LTOp)t1(b, m) * (LTOp)t2(a, f, i, j) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, m) * (LTOp)t1(b, n) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, m) * (LTOp)t1(a, n) * (LTOp)t2(b, e, i, j) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, f) * (LTOp)t1(a, m) * (LTOp)t2(b, f, i, j) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, m) * (LTOp)t1(e, j) * (LTOp)t2(a, b, i, n) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, f) * (LTOp)t1(f, j) * (LTOp)t2(a, b, i, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(a, f) * (LTOp)t1(f, m) * (LTOp)t2(b, e, i, j) +    // PT ORDER = 5
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(a, f) * (LTOp)t1(e, i) * (LTOp)t2(b, f, j, m) +    // PT ORDER = 5
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)f1(n, i) * (LTOp)t1(a, m) * (LTOp)t2(b, e, j, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, m) * (LTOp)t1(e, i) * (LTOp)t2(a, b, j, n) +    // PT ORDER = 5
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, f) * (LTOp)t1(f, i) * (LTOp)t2(a, b, j, m)      // PT ORDER = 5, Zero because of T_int
           );

vtijab(i, j, a, b).update(op8);
// op_exec.print_op_binarized((LTOp)vtijab(i, j, a, b), op8.clone(), true);
op_exec.execute(vtijab, true, ex_hw);

//PHHP

auto op9 = (
          -(1.0/6.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(n, f) * (LTOp)t1(a, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 4, Zero because of T_int
          -(2.0/3.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(n, f) * (LTOp)t1(f, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, f, m, i) * (LTOp)f1(n, b) * (LTOp)t1(e, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, i) * (LTOp)f1(a, n) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4
          -(2.0/3.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(n, f) * (LTOp)t1(e, n) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t2(e, f, m, i) * (LTOp)f1(n, b) * (LTOp)t1(a, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(i, f) * (LTOp)t1(e, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 4
          -(2.0/3.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, i) * (LTOp)f1(e, n) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, i) * (LTOp)f1(f, j) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(n, f) * (LTOp)t1(e, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(i, f) * (LTOp)t1(e, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(b, m) * (LTOp)t2(e, f, n, i) * (LTOp)f1(e, n) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, i) * (LTOp)t2(f, b, m, n) * (LTOp)f1(f, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 4
          -(2.0/3.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, i) * (LTOp)f1(f, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 4
           (1.0/12.0) * (LTOp)t1(b, m) * (LTOp)t2(e, f, n, i) * (LTOp)f1(a, n) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4, Zero because of T_int
           (1.0/12.0) * (LTOp)t1(e, i) * (LTOp)t2(f, b, m, n) * (LTOp)f1(f, j) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, i) * (LTOp)f1(e, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(b, m) * (LTOp)t2(e, f, n, i) * (LTOp)f1(e, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(b, m) * (LTOp)t2(e, f, n, i) * (LTOp)f1(e, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t2(f, b, n, i) * (LTOp)f1(a, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t1(e, i) * (LTOp)t2(f, b, m, n) * (LTOp)f1(e, m) * (LTOp)t2(a, f, j, n) +    // PT ORDER = 4
          -(1.0/6.0) * (LTOp)t1(e, i) * (LTOp)t2(f, b, m, n) * (LTOp)f1(a, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(i, f) * (LTOp)t1(f, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(i, f) * (LTOp)t1(a, m) * (LTOp)t2(e, f, j, n) +    // PT ORDER = 4, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, f, m, i) * (LTOp)f1(n, b) * (LTOp)t1(e, n) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t2(e, f, m, i) * (LTOp)f1(n, b) * (LTOp)t1(e, j) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 4
           (1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(n, f) * (LTOp)t1(a, n) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 4, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(n, f) * (LTOp)t1(f, j) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 4
           (1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(b, n) * (LTOp)f1(i, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(b, m) * (LTOp)t1(e, i) * (LTOp)f1(e, f) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/3.0) * (LTOp)t1(b, m) * (LTOp)t1(e, i) * (LTOp)f1(a, f) * (LTOp)t2(e, f, j, m) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/3.0) * (LTOp)t1(b, m) * (LTOp)t1(e, i) * (LTOp)f1(n, j) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(b, m) * (LTOp)t1(e, i) * (LTOp)f1(n, m) * (LTOp)t2(a, e, j, n) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t1(e, m) * (LTOp)t1(f, i) * (LTOp)f1(e, b) * (LTOp)t2(a, f, j, m) +    // PT ORDER = 5
          -(1.0/3.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(i, m) * (LTOp)t1(e, j) * (LTOp)t1(a, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/3.0) * (LTOp)t2(e, f, m, i) * (LTOp)f1(e, b) * (LTOp)t1(f, j) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
          -(1.0/6.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(a, f) * (LTOp)t1(e, j) * (LTOp)t1(f, m) +    // PT ORDER = 5
          -(1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(e, f) * (LTOp)t1(f, j) * (LTOp)t1(a, m) +    // PT ORDER = 5, Zero because of T_int
           (1.0/6.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(n, j) * (LTOp)t1(a, m) * (LTOp)t1(e, n) +    // PT ORDER = 5, Zero because of T_int
           (1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(n, m) * (LTOp)t1(e, j) * (LTOp)t1(a, n)      // PT ORDER = 5, Zero because of T_int
           );

vtaijb(a, i, j, b).update(op9);
// op_exec.print_op_binarized((LTOp)vtaijb(a, i, j, b), op9.clone(), true);
op_exec.execute(vtaijb, true, ex_hw);

  }
}


template<typename T>
bool ducc_tensors_exist(std::vector<Tensor<T>>& rwtensors, std::vector<std::string> tfiles,
                        bool do_io = false) {
  if(!do_io) return false;


  return (std::all_of(tfiles.begin(), tfiles.end(), [](std::string x) { return fs::exists(x); }));
}

template<typename T>
void ducc_tensors_io(ExecutionContext& ec, ChemEnv& chem_env, std::vector<Tensor<T>>& rwtensors,
                     std::vector<std::string> tfiles, int dlvl, bool writet = false,
                     bool read = false) {
  if(!writet) return;
  // read only if writet is enabled

  if(read) {
    if(ec.pg().rank() == 0)
      std::cout << std::endl << "DUCC: [Level " << dlvl << "] Reading ducc tensors from disk..." << std::endl;
    read_from_disk_group(ec, rwtensors, tfiles);
  }
  else {
    if(ec.pg().rank() == 0)
      std::cout << std::endl << "DUCC: [Level " << dlvl << "] Writing ducc tensors to disk..." << std::endl;
    write_to_disk_group(ec, rwtensors, tfiles);
  }
}

} //namespace exachem::cc::ducc::internal
