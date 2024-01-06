/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cc/ccsd/cd_ccsd_os_ann.hpp"
#include <tamm/op_executor.hpp>

using namespace tamm;
using namespace tamm::new_ops;

// clang-format off
template<typename T>
void H_0(Scheduler& sch, const TiledIndexSpace& MO,
            const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
            const Tensor<T>& vtijkl, const Tensor<T>& vtijka,
            const Tensor<T>& vtaijb, const Tensor<T>& vtijab, const Tensor<T>& vtiabc, const Tensor<T>& vtabcd,
            const Tensor<T>& f1, V2Tensors<T>& v2tensors, size_t nactv, ExecutionHW ex_hw = ExecutionHW::CPU){

  auto [i, j, k, l] = MO.labels<4>("occ");
  auto [a, b, c, d] = MO.labels<4>("virt_int");

  sch(ftij(i, j) = f1(i, j))
     (vtijkl(i, j, k, l) = v2tensors.v2ijkl(i, j, k, l))
     .execute(ex_hw);
  if (nactv > 0) {
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
void F_1(Scheduler& sch, const TiledIndexSpace& MO,
            const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
            const Tensor<T>& vtijkl, const Tensor<T>& vtijka,
            const Tensor<T>& vtaijb, const Tensor<T>& vtijab, const Tensor<T>& vtiabc,
            const Tensor<T>& vtabcd, const Tensor<T>& f1,
            const Tensor<T>& t1, const Tensor<T>& t2, size_t nactv, ExecutionHW ex_hw = ExecutionHW::CPU){

  TiledIndexLabel a, b, c, d;
  TiledIndexLabel i, j, k, l;
  TiledIndexLabel e, f, g, h;
  TiledIndexLabel m, n, o, p;

  std::tie(a, b, c, d) = MO.labels<4>("virt_int");
  std::tie(e, f, g, h) = MO.labels<4>("virt");
  std::tie(i, j, k, l, m, n, o, p) = MO.labels<8>("occ");

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

  if (nactv > 0) {
//PP

// auto op2 = (
          // -1.0 * (LTOp)f1(m, b) * (LTOp)t1(a, m) +    // PT ORDER = 2, Zero because of T_int
          // -1.0 * (LTOp)t1(b, m) * (LTOp)f1(a, m)      // PT ORDER = 2, Zero because of T_int
          //  );

// ftab(a, b).update(op2);
// op_exec.print_op_binarized((LTOp)ftab(a, b), op2.clone(), true);
// op_exec.execute(ftab, true, ex_hw);

//HP/PH

auto op3 = (
           1.0 * (LTOp)f1(m, e) * (LTOp)t2(a, e, i, m) +    // PT ORDER = 1
          // -1.0 * (LTOp)f1(m, i) * (LTOp)t1(a, m) +    // PT ORDER = 2, Zero because of T_int
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

// auto op7 = (
          //  1.0 * (LTOp)f1(m, a) * (LTOp)t2(c, b, i, m)      // PT ORDER = 1, Zero because of T_int
          //  );

// vtiabc(i, a, b, c).update(op7);
// op_exec.print_op_binarized((LTOp)vtiabc(i, a, b, c), op7.clone(), true);
// op_exec.execute(vtiabc, true, ex_hw);

//HHPP/PPHH

auto op8 = (
           1.0 * (LTOp)f1(b, e) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 1
          // -1.0 * (LTOp)f1(m, j) * (LTOp)t2(a, b, i, m) +    // PT ORDER = 1, Zero because of T_int
          //  1.0 * (LTOp)f1(m, i) * (LTOp)t2(a, b, j, m) +    // PT ORDER = 1, Zero because of T_int
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
void V_1(Scheduler& sch, const TiledIndexSpace& MO,
            const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
            const Tensor<T>& vtijkl, const Tensor<T>& vtijka,
            const Tensor<T>& vtaijb, const Tensor<T>& vtijab, const Tensor<T>& vtiabc,
            const Tensor<T>& vtabcd, V2Tensors<T>& v2tensors,
            const Tensor<T>& t1, const Tensor<T>& t2, size_t nactv, ExecutionHW ex_hw = ExecutionHW::CPU){

  TiledIndexLabel a, b, c, d;
  TiledIndexLabel i, j, k, l;
  TiledIndexLabel e, f, g, h;
  TiledIndexLabel m, n, o, p;

  std::tie(a, b, c, d) = MO.labels<4>("virt_int");
  std::tie(e, f, g, h) = MO.labels<4>("virt");
  std::tie(i, j, k, l, m, n, o, p) = MO.labels<8>("occ");

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

  if (nactv > 0) {
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

  if (nactv > 0) {
//PPPP

// auto op5 = (
          // -(1.0/2.0) * (LTOp)v2tensors.v2ijab(m, n, d, c) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 2, Zero because of T_int
          // -(1.0/2.0) * (LTOp)t2(d, c, m, n) * (LTOp)v2tensors.v2ijab(m, n, a, b) +    // PT ORDER = 2, Zero because of T_int
          //  1.0 * (LTOp)t1(d, m) * (LTOp)v2tensors.v2iabc(m, c, a, b) +    // PT ORDER = 3, Zero because of T_int
          // -1.0 * (LTOp)t1(c, m) * (LTOp)v2tensors.v2iabc(m, d, a, b) +    // PT ORDER = 3, Zero because of T_int
          // -1.0 * (LTOp)v2tensors.v2iabc(m, b, c, d) * (LTOp)t1(a, m) +    // PT ORDER = 3, Zero because of T_int
          //  1.0 * (LTOp)v2tensors.v2iabc(m, a, c, d) * (LTOp)t1(b, m)      // PT ORDER = 3, Zero because of T_int
          //  );

// vtabcd(a, b, c, d).update(op5);
// op_exec.print_op_binarized((LTOp)vtabcd(a, b, c, d), op5.clone(), true);
// op_exec.execute(vtabcd, true, ex_hw);

//HHHP/HPHH

auto op6 = (
           1.0 * (LTOp)v2tensors.v2ijka(m, k, j, e) * (LTOp)t2(a, e, i, m) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2ijka(m, k, i, e) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 2
          -(1.0/2.0) * (LTOp)v2tensors.v2iabc(k, a, f, e) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 2
          //  1.0 * (LTOp)v2tensors.v2ijkl(m, k, i, j) * (LTOp)t1(a, m) +    // PT ORDER = 3, Zero because of T_int
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
          // -(1.0/2.0) * (LTOp)v2tensors.v2ijka(m, n, i, a) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 2, Zero because of T_int
          //  1.0 * (LTOp)t1(a, m) * (LTOp)v2tensors.v2ijab(i, m, c, b) +    // PT ORDER = 3, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2abcd(c, b, e, a) * (LTOp)t1(e, i)      // PT ORDER = 3
          // -1.0 * (LTOp)v2tensors.v2iajb(m, c, i, a) * (LTOp)t1(b, m) +    // PT ORDER = 3, Zero because of T_int
          //  1.0 * (LTOp)v2tensors.v2iajb(m, b, i, a) * (LTOp)t1(c, m)      // PT ORDER = 3, Zero because of T_int
           );

vtiabc(i, a, b, c).update(op7);
// op_exec.print_op_binarized((LTOp)vtiabc(i, a, b, c), op7.clone(), true);
op_exec.execute(vtiabc, true, ex_hw);

//HHPP/PPHH

auto op8 = (
           1.0 * (LTOp)v2tensors.v2iajb(m, a, j, e) * (LTOp)t2(b, e, i, m) +    // PT ORDER = 2
          //  (1.0/2.0) * (LTOp)v2tensors.v2ijkl(m, n, i, j) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 2, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2iajb(m, b, i, e) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2iajb(m, a, i, e) * (LTOp)t2(b, e, j, m) +    // PT ORDER = 2
          -1.0 * (LTOp)v2tensors.v2iajb(m, b, j, e) * (LTOp)t2(a, e, i, m) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)v2tensors.v2abcd(a, b, e, f) * (LTOp)t2(e, f, i, j) +    // PT ORDER = 2
           1.0 * (LTOp)v2tensors.v2iabc(i, e, a, b) * (LTOp)t1(e, j) +    // PT ORDER = 3
          // -1.0 * (LTOp)v2tensors.v2ijka(j, i, m, a) * (LTOp)t1(b, m) +    // PT ORDER = 3, Zero because of T_int
          //  1.0 * (LTOp)v2tensors.v2ijka(j, i, m, b) * (LTOp)t1(a, m) +    // PT ORDER = 3, Zero because of T_int
          -1.0 * (LTOp)v2tensors.v2iabc(j, e, a, b) * (LTOp)t1(e, i)      // PT ORDER = 3
           );

vtijab(i, j, a, b).update(op8);
// op_exec.print_op_binarized((LTOp)vtijab(i, j, a, b), op8.clone(), true);
op_exec.execute(vtijab, true, ex_hw);

//PHHP

auto op9 = (
           1.0 * (LTOp)v2tensors.v2ijab(m, i, e, b) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 2
           1.0 * (LTOp)t2(e, b, m, i) * (LTOp)v2tensors.v2ijab(j, m, a, e) +    // PT ORDER = 2
          // -1.0 * (LTOp)v2tensors.v2ijka(m, i, j, b) * (LTOp)t1(a, m) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)v2tensors.v2iabc(i, a, b, e) * (LTOp)t1(e, j) +    // PT ORDER = 3
          // -1.0 * (LTOp)t1(b, m) * (LTOp)v2tensors.v2ijka(m, j, i, a) +    // PT ORDER = 3, Zero because of T_int
           1.0 * (LTOp)t1(e, i) * (LTOp)v2tensors.v2iabc(j, b, a, e)      // PT ORDER = 3
           );

vtaijb(a, i, j, b).update(op9);
// op_exec.print_op_binarized((LTOp)vtaijb(a, i, j, b), op9.clone(), true);
op_exec.execute(vtaijb, true, ex_hw);

  }
}


template<typename T>
void F_2(Scheduler& sch, const TiledIndexSpace& MO,
            const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
            const Tensor<T>& vtijkl, const Tensor<T>& vtijka,
            const Tensor<T>& vtaijb, const Tensor<T>& vtijab, const Tensor<T>& vtiabc,
            const Tensor<T>& vtabcd, const Tensor<T>& f1,
            const Tensor<T>& t1, const Tensor<T>& t2, size_t nactv, ExecutionHW ex_hw = ExecutionHW::CPU){

  TiledIndexLabel a, b, c, d;
  TiledIndexLabel i, j, k, l;
  TiledIndexLabel e, f, g, h;
  TiledIndexLabel m, n, o, p;

  std::tie(a, b, c, d) = MO.labels<4>("virt_int");
  std::tie(e, f, g, h) = MO.labels<4>("virt");
  std::tie(i, j, k, l, m, n, o, p) = MO.labels<8>("occ");

  SymbolTable symbol_table;
  OpExecutor op_exec{sch, symbol_table};
  TAMM_REGISTER_SYMBOLS(symbol_table, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
  TAMM_REGISTER_SYMBOLS(symbol_table, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1, t1, t2);

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

  if (nactv > 0) {
//PP

auto op2 = (
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(e, f) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 2
           1.0 * (LTOp)t2(e, b, m, n) * (LTOp)f1(o, m) * (LTOp)t2(a, e, n, o) +    // PT ORDER = 2
          -(1.0/4.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(a, f) * (LTOp)t2(e, f, m, n) +    // PT ORDER = 2
          -(1.0/4.0) * (LTOp)t2(e, f, m, n) * (LTOp)f1(e, b) * (LTOp)t2(a, f, m, n) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(a, m) * (LTOp)t1(e, n) +    // PT ORDER = 3
          // -(1.0/2.0) * (LTOp)t2(e, b, m, n) * (LTOp)f1(e, m) * (LTOp)t1(a, n) +    // PT ORDER = 3, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(n, b) * (LTOp)t2(a, e, m, n)      // PT ORDER = 3
          // -(1.0/2.0) * (LTOp)t1(b, m) * (LTOp)f1(n, e) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 3, Zero because of T_int
          //  1.0 * (LTOp)t1(b, m) * (LTOp)f1(n, m) * (LTOp)t1(a, n) +    // PT ORDER = 4, Zero because of T_int
          // -(1.0/2.0) * (LTOp)t1(b, m) * (LTOp)f1(a, e) * (LTOp)t1(e, m) +    // PT ORDER = 4, Zero because of T_int
          // -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, b) * (LTOp)t1(a, m)      // PT ORDER = 4, Zero because of T_int
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
          // -1.0 * (LTOp)f1(m, e) * (LTOp)t1(e, i) * (LTOp)t1(a, m) +    // PT ORDER = 4, Zero because of T_int
          // -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, i) * (LTOp)t1(a, m)      // PT ORDER = 4, Zero because of T_int
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

  if (nactv > 0) {
//PPPP

// auto op5 = (
          // -1.0 * (LTOp)t2(d, c, m, n) * (LTOp)f1(o, m) * (LTOp)t2(a, b, n, o) +    // PT ORDER = 2, Zero because of T_int
          //  (1.0/4.0) * (LTOp)t2(e, d, m, n) * (LTOp)f1(e, c) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 2, Zero because of T_int
          // -(1.0/4.0) * (LTOp)t2(e, c, m, n) * (LTOp)f1(e, d) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 2, Zero because of T_int
          //  (1.0/4.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(a, e) * (LTOp)t2(b, e, m, n) +    // PT ORDER = 2, Zero because of T_int
          // -(1.0/4.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(b, e) * (LTOp)t2(a, e, m, n) +    // PT ORDER = 2, Zero because of T_int
          //  (1.0/2.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(b, m) * (LTOp)t1(a, n) +    // PT ORDER = 3, Zero because of T_int
          // -(1.0/2.0) * (LTOp)t1(d, m) * (LTOp)f1(n, c) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 3, Zero because of T_int
          //  (1.0/2.0) * (LTOp)t1(c, m) * (LTOp)f1(n, d) * (LTOp)t2(a, b, m, n) +    // PT ORDER = 3, Zero because of T_int
          // -(1.0/2.0) * (LTOp)t2(d, c, m, n) * (LTOp)f1(a, m) * (LTOp)t1(b, n)      // PT ORDER = 3, Zero because of T_int
          //  );

// vtabcd(a, b, c, d).update(op5);
// op_exec.print_op_binarized((LTOp)vtabcd(a, b, c, d), op5.clone(), true);
// op_exec.execute(vtabcd, true, ex_hw);

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
          // -(1.0/4.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(e, i) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 2, Zero because of T_int
          -(1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(b, m) * (LTOp)t2(c, e, i, n) +    // PT ORDER = 2
           (1.0/2.0) * (LTOp)t2(e, a, m, n) * (LTOp)f1(c, m) * (LTOp)t2(b, e, i, n)      // PT ORDER = 2
          //  (1.0/2.0) * (LTOp)t1(a, m) * (LTOp)f1(n, i) * (LTOp)t2(c, b, m, n) +    // PT ORDER = 3, Zero because of T_int
          //  (1.0/2.0) * (LTOp)t1(a, m) * (LTOp)f1(b, e) * (LTOp)t2(c, e, i, m) +    // PT ORDER = 3, Zero because of T_int
          // -(1.0/2.0) * (LTOp)t1(a, m) * (LTOp)f1(c, e) * (LTOp)t2(b, e, i, m) +    // PT ORDER = 3, Zero because of T_int
          // -1.0 * (LTOp)t1(a, m) * (LTOp)f1(n, m) * (LTOp)t2(c, b, i, n) +    // PT ORDER = 3, Zero because of T_int
          //  (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, a) * (LTOp)t2(c, b, i, m)      // PT ORDER = 3, Zero because of T_int
           );

vtiabc(i, a, b, c).update(op7);
// op_exec.print_op_binarized((LTOp)vtiabc(i, a, b, c), op7.clone(), true);
op_exec.execute(vtiabc, true, ex_hw);

//HHPP/PPHH

auto op8 = (
          -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(b, m) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 3
          //  (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, i) * (LTOp)t2(a, b, j, m) +    // PT ORDER = 3, Zero because of T_int
          // -(1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(e, j) * (LTOp)t2(a, b, i, m) +    // PT ORDER = 3, Zero because of T_int
           (1.0/2.0) * (LTOp)t1(e, m) * (LTOp)f1(a, m) * (LTOp)t2(b, e, i, j)      // PT ORDER = 3
          // -1.0 * (LTOp)f1(m, e) * (LTOp)t1(e, j) * (LTOp)t2(a, b, i, m) +    // PT ORDER = 3, Zero because of T_int
          //  1.0 * (LTOp)f1(m, e) * (LTOp)t1(a, m) * (LTOp)t2(b, e, i, j) +    // PT ORDER = 3, Zero because of T_int
          // -1.0 * (LTOp)f1(m, e) * (LTOp)t1(b, m) * (LTOp)t2(a, e, i, j) +    // PT ORDER = 3, Zero because of T_int
          //  1.0 * (LTOp)f1(m, e) * (LTOp)t1(e, i) * (LTOp)t2(a, b, j, m)      // PT ORDER = 3, Zero because of T_int
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
          // -(1.0/2.0) * (LTOp)t1(b, m) * (LTOp)f1(i, e) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 3, Zero because of T_int
          -(1.0/2.0) * (LTOp)t1(e, i) * (LTOp)f1(m, b) * (LTOp)t2(a, e, j, m) +    // PT ORDER = 3
          -(1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(a, m) * (LTOp)t1(e, j)      // PT ORDER = 3
          // -(1.0/2.0) * (LTOp)t2(e, b, m, i) * (LTOp)f1(e, j) * (LTOp)t1(a, m)      // PT ORDER = 3, Zero because of T_int
           );

vtaijb(a, i, j, b).update(op9);
// op_exec.print_op_binarized((LTOp)vtaijb(a, i, j, b), op9.clone(), true);
op_exec.execute(vtaijb, true, ex_hw);

  }
}
// clang-format on

template<typename T>
bool ducc_tensors_exist(std::vector<Tensor<T>>& rwtensors, std::vector<std::string> tfiles,
                        std::string comp, bool do_io = false) {
  if(!do_io) return false;

  std::transform(tfiles.begin(), tfiles.end(), tfiles.begin(),
                 [&](string tname) -> std::string { return tname + "_" + comp; });

  return (std::all_of(tfiles.begin(), tfiles.end(), [](std::string x) { return fs::exists(x); }));
}

template<typename T>
void ducc_tensors_io(ExecutionContext& ec, std::vector<Tensor<T>>& rwtensors,
                     std::vector<std::string> tfiles, std::string comp, bool writet = false,
                     bool read = false) {
  if(!writet) return;
  // read only if writet is enabled
  std::transform(tfiles.begin(), tfiles.end(), tfiles.begin(),
                 [&](string tname) -> std::string { return tname + "_" + comp; });
  if(read) {
    if(ec.pg().rank() == 0)
      std::cout << std::endl << "Reading " << comp << " tensors from disk..." << std::endl;
    read_from_disk_group(ec, rwtensors, tfiles);
  }
  else {
    if(ec.pg().rank() == 0)
      std::cout << std::endl << "Writing " << comp << " tensors to disk..." << std::endl;
    write_to_disk_group(ec, rwtensors, tfiles);
  }
}

template<typename T>
void DUCC_T_CCSD_Driver(SystemData sys_data, ExecutionContext& ec, const TiledIndexSpace& MO,
                        Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& f1, V2Tensors<T>& v2tensors,
                        size_t nactv, ExecutionHW ex_hw = ExecutionHW::CPU) {
  Scheduler sch{ec};

  const TiledIndexSpace& O = MO("occ");
  // const TiledIndexSpace& V  = MO("virt");
  const TiledIndexSpace& Vi = MO("virt_int");
  // const TiledIndexSpace& N = MO("all");
  // const TiledIndexSpace& Vai = MO("virt_alpha_int");
  // const TiledIndexSpace& Vbi = MO("virt_beta_int");
  // const TiledIndexSpace& Vae = MO("virt_alpha_ext");
  // const TiledIndexSpace& Vbe = MO("virt_beta_ext");

  auto [z1, z2, z3, z4]     = MO.labels<4>("all");
  auto [h1, h2, h3, h4]     = MO.labels<4>("occ");
  auto [p1, p2, p3, p4]     = MO.labels<4>("virt");
  auto [p1i, p2i, p3i, p4i] = MO.labels<4>("virt_int");
  // auto [p1ai,p2ai] = MO.labels<2>("virt_alpha_int");
  // auto [p1bi,p2bi] = MO.labels<2>("virt_beta_int");

  std::cout.precision(15);

  std::string out_fp = sys_data.output_file_prefix + "." + sys_data.options_map.ccsd_options.basis;
  std::string files_dir = out_fp + "_files/" + sys_data.options_map.scf_options.scf_type + "/ducc";
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + out_fp;
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);
  std::string ftij_file   = files_prefix + ".ftij";
  std::string ftia_file   = files_prefix + ".ftia";
  std::string ftab_file   = files_prefix + ".ftab";
  std::string vtijkl_file = files_prefix + ".vtijkl";
  std::string vtijka_file = files_prefix + ".vtijka";
  std::string vtaijb_file = files_prefix + ".vtaijb";
  std::string vtijab_file = files_prefix + ".vtijab";
  std::string vtiabc_file = files_prefix + ".vtiabc";
  std::string vtabcd_file = files_prefix + ".vtabcd";
  const bool  drestart    = sys_data.options_map.ccsd_options.writet;

  // Allocate the transformed arrays
  Tensor<T> ftij{{O, O}, {1, 1}};
  Tensor<T> ftia{{O, Vi}, {1, 1}};
  Tensor<T> ftab{{Vi, Vi}, {1, 1}};
  Tensor<T> vtijkl{{O, O, O, O}, {2, 2}};
  Tensor<T> vtijka{{O, O, O, Vi}, {2, 2}};
  Tensor<T> vtaijb{{Vi, O, O, Vi}, {2, 2}};
  Tensor<T> vtijab{{O, O, Vi, Vi}, {2, 2}};
  Tensor<T> vtiabc{{O, Vi, Vi, Vi}, {2, 2}};
  Tensor<T> vtabcd{{Vi, Vi, Vi, Vi}, {2, 2}};
  sch.allocate(ftij, vtijkl).execute();
  if(nactv > 0) { sch.allocate(ftia, ftab, vtijka, vtaijb, vtijab, vtiabc, vtabcd).execute(); }

  // Zero the transformed arrays
  sch(ftij(h1, h2) = 0)(vtijkl(h1, h2, h3, h4) = 0).execute();
  if(nactv > 0) {
    // clang-format off
    sch (ftia(h1,p1i) = 0)
        (ftab(p1i,p2i) = 0)
        (vtijka(h1,h2,h3,p1i) = 0)
        (vtaijb(p1i,h1,h2,p2i) = 0)
        (vtijab(h1,h2,p1i,p2i) = 0)
        (vtiabc(h1,p1i,p2i,p3i) = 0)
        (vtabcd(p1i,p2i,p3i,p4i) = 0).execute();
    // clang-format on
  }

  // Zero T_int components
  // This means that T1 and T2 are no longer the full T1 and T2.
  sch(t1(p1i, h1) = 0)(t2(p1i, p2i, h1, h2) = 0).execute();

  // Tensors to read/write from/to disk
  std::vector<Tensor<T>>   ducc_tensors = {ftij,   ftia,   ftab,   vtijkl, vtijka,
                                           vtaijb, vtijab, vtiabc, vtabcd};
  std::vector<std::string> dt_files     = {ftij_file,   ftia_file,   ftab_file,
                                           vtijkl_file, vtijka_file, vtaijb_file,
                                           vtijab_file, vtiabc_file, vtabcd_file};

  const auto  rank  = ec.pg().rank();
  std::string ctype = "bareH";
  auto        cc_t1 = std::chrono::high_resolution_clock::now();
  // Bare Hamiltonian
  if(ducc_tensors_exist(ducc_tensors, dt_files, ctype, drestart))
    ducc_tensors_io(ec, ducc_tensors, dt_files, ctype, drestart, true);
  else {
    H_0(sch, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1, v2tensors,
        nactv, ex_hw);
    ducc_tensors_io(ec, ducc_tensors, dt_files, ctype, drestart);
  }
  // TODO Setup a print statement here.
  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ducc_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << std::endl
              << "DUCC: Time taken to compute Bare Hamiltonian: " << std::fixed
              << std::setprecision(2) << ducc_time << " secs" << std::endl;

  double ducc_total_time = ducc_time;

  cc_t1 = std::chrono::high_resolution_clock::now();
  // Single Commutator
  ctype = "singleC";
  if(ducc_tensors_exist(ducc_tensors, dt_files, ctype, drestart))
    ducc_tensors_io(ec, ducc_tensors, dt_files, ctype, drestart, true);
  else {
    F_1(sch, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1, t1, t2,
        nactv, ex_hw);
    V_1(sch, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, v2tensors, t1,
        t2, nactv, ex_hw);
    ducc_tensors_io(ec, ducc_tensors, dt_files, ctype, drestart);
  }
  cc_t2     = std::chrono::high_resolution_clock::now();
  ducc_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  ducc_total_time += ducc_time;
  if(rank == 0)
    std::cout << std::endl
              << "DUCC: Time taken to compute Single Commutator: " << std::fixed
              << std::setprecision(2) << ducc_time << " secs" << std::endl;

  cc_t1 = std::chrono::high_resolution_clock::now();
  // Double Commutator
  ctype = "doubleC";
  if(ducc_tensors_exist(ducc_tensors, dt_files, ctype, drestart))
    ducc_tensors_io(ec, ducc_tensors, dt_files, ctype, drestart, true);
  else {
    F_2(sch, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1, t1, t2,
        nactv, ex_hw);
    // V_2(sch, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, v2tensors, t1,
    //     t2, nactv, ex_hw);
    ducc_tensors_io(ec, ducc_tensors, dt_files, ctype, drestart);
  }
  cc_t2     = std::chrono::high_resolution_clock::now();
  ducc_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  ducc_total_time += ducc_time;
  if(rank == 0)
    std::cout << std::endl
              << "DUCC: Time taken to compute Double Commutator: " << std::fixed
              << std::setprecision(2) << ducc_time << " secs" << std::endl;

  // Transform ft from Fock operator to one-electron operator.
  Tensor<T> deltaoo{{O, O}, {1, 1}};
  sch.allocate(deltaoo).execute();
  sch(deltaoo() = 0).execute();
  init_diagonal(ec, deltaoo());

  // clang-format off
  if (nactv > 0) {
    sch(ftab(p1i,p2i) +=  1.0 * deltaoo(h1,h2) * vtaijb(p1i,h1,h2,p2i))
       (ftia(h3,p1i)  += -1.0 * deltaoo(h1,h2) * vtijka(h1,h3,h2,p1i))
       (ftij(h3,h4)   += -1.0 * deltaoo(h1,h2) * vtijkl(h3,h1,h4,h2))
       .deallocate(deltaoo).execute(ex_hw);
  } else {
    sch(ftij(h3,h4)   += -1.0 * deltaoo(h1,h2) * vtijkl(h3,h1,h4,h2))
       .deallocate(deltaoo).execute(ex_hw);
  }
  // clang-format on

  if(rank == 0) {
    sys_data.results["output"]["DUCC"]["performance"]["total_time"] = ducc_total_time;
    std::cout << std::endl
              << "DUCC: Total Compute Time: " << std::fixed << std::setprecision(2)
              << ducc_total_time << " secs" << std::endl
              << std::endl;
  }

  // PRINT STATEMENTS
  // TODO: Everything is assuming closed shell. For open shell calculations,
  //       formats starting from the tensor contractions to printing must be reconsidered.
  cc_t1 = std::chrono::high_resolution_clock::now();
  ExecutionContext ec_dense{ec.pg(), DistributionKind::dense,
                            MemoryManagerKind::ga}; // create ec_dense once
  // const auto       nelectrons       = sys_data.nelectrons;
  const size_t nelectrons_alpha = sys_data.nelectrons_alpha;
  auto         print_blockstr   = [](std::string filename, std::string val, bool append = false) {
    if(!filename.empty()) {
      std::ofstream tos;
      if(append) tos.open(filename + ".txt", std::ios::app);
      else tos.open(filename + ".txt", std::ios::out);
      if(!tos) std::cerr << "Error opening file " << filename << std::endl;
      tos << val << std::endl;
      tos.close();
    }
  };
  const std::string results_file = files_prefix + ".ducc.results";

  if(rank == 0) {
    std::cout << "Number of active orbitals = " << nactv + nelectrons_alpha << std::endl
              << std::endl;
    print_blockstr(results_file, "Begin IJ Block");
    // std::cout << "Begin IJ Block" << std::endl;
  }

  Tensor<T>                                X1       = to_dense_tensor(ec_dense, ftij);
  std::function<bool(std::vector<size_t>)> dp_cond1 = [&](std::vector<size_t> cond) {
    if(cond[0] < nelectrons_alpha && cond[1] < nelectrons_alpha && cond[0] <= cond[1]) return true;
    return false;
  };
  print_dense_tensor(X1, dp_cond1, results_file, true);
  if(rank == 0) {
    print_blockstr(results_file, "End IJ Block", true);
    T first_val                                         = tamm::get_tensor_element(X1, {0, 0});
    sys_data.results["output"]["DUCC"]["results"]["X1"] = first_val;
  }
  Tensor<T>::deallocate(X1);

  if(nactv > 0) {
    if(rank == 0) print_blockstr(results_file, "Begin IA Block", true);
    Tensor<T>                                X2       = to_dense_tensor(ec_dense, ftia);
    std::function<bool(std::vector<size_t>)> dp_cond2 = [&](std::vector<size_t> cond) {
      if(cond[0] < nelectrons_alpha && cond[1] < nactv) return true;
      return false;
    };
    print_dense_tensor(X2, dp_cond2, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End IA Block", true);
      T first_val                                         = tamm::get_tensor_element(X2, {0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X2"] = first_val;
    }
    Tensor<T>::deallocate(X2);

    if(rank == 0) print_blockstr(results_file, "Begin AB Block", true);
    Tensor<T>                                X3       = to_dense_tensor(ec_dense, ftab);
    std::function<bool(std::vector<size_t>)> dp_cond3 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactv && cond[1] < nactv && cond[0] <= cond[1]) return true;
      return false;
    };
    print_dense_tensor(X3, dp_cond3, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End AB Block", true);
      T first_val                                         = tamm::get_tensor_element(X3, {0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X3"] = first_val;
    }
    Tensor<T>::deallocate(X3);

    if(rank == 0) print_blockstr(results_file, "Begin IJAB Block", true);
    Tensor<T>                                X4       = to_dense_tensor(ec_dense, vtijab);
    std::function<bool(std::vector<size_t>)> dp_cond4 = [&](std::vector<size_t> cond) {
      if(cond[0] < nelectrons_alpha && nelectrons_alpha <= cond[1] && cond[2] < nactv &&
         nactv <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X4, dp_cond4, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End IJAB Block", true);
      T first_val = tamm::get_tensor_element(X4, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X4"] = first_val;
    }
    Tensor<T>::deallocate(X4);
  }

  if(rank == 0) print_blockstr(results_file, "Begin IJKL Block", true);
  Tensor<T>                                X5       = to_dense_tensor(ec_dense, vtijkl);
  std::function<bool(std::vector<size_t>)> dp_cond5 = [&](std::vector<size_t> cond) {
    if(cond[0] < nelectrons_alpha && cond[2] < nelectrons_alpha && nelectrons_alpha <= cond[1] &&
       nelectrons_alpha <= cond[3])
      return true;
    return false;
  };
  print_dense_tensor(X5, dp_cond5, results_file, true);
  if(rank == 0) {
    print_blockstr(results_file, "End IJKL Block", true);
    T first_val = tamm::get_tensor_element(X5, {0, 0, 0, 0});
    sys_data.results["output"]["DUCC"]["results"]["X5"] = first_val;
  }
  Tensor<T>::deallocate(X5);

  if(nactv > 0) {
    if(rank == 0) print_blockstr(results_file, "Begin ABCD Block", true);
    Tensor<T>                                X6       = to_dense_tensor(ec_dense, vtabcd);
    std::function<bool(std::vector<size_t>)> dp_cond6 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactv && cond[2] < nactv && nactv <= cond[1] && nactv <= cond[3]) return true;
      return false;
    };
    print_dense_tensor(X6, dp_cond6, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End ABCD Block", true);
      T first_val = tamm::get_tensor_element(X6, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X6"] = first_val;
    }
    Tensor<T>::deallocate(X6);

    if(rank == 0) print_blockstr(results_file, "Begin AIJB Block", true);
    Tensor<T>                                X7         = to_dense_tensor(ec_dense, vtaijb);
    std::function<bool(std::vector<size_t>)> dp_cond7_1 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactv && cond[2] < nelectrons_alpha && nelectrons_alpha <= cond[1] &&
         nactv <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X7, dp_cond7_1, results_file, true);

    std::function<bool(std::vector<size_t>)> dp_cond7_2 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactv && nelectrons_alpha <= cond[1] && nelectrons_alpha <= cond[2] &&
         cond[3] < nactv)
        return true;
      return false;
    };
    print_dense_tensor(X7, dp_cond7_2, results_file, true);
    if(rank == 0) print_blockstr(results_file, "End AIJB Block", true);
    if(rank == 0) {
      T first_val = tamm::get_tensor_element(X7, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X7"] = first_val;
    }
    Tensor<T>::deallocate(X7);

    if(rank == 0) print_blockstr(results_file, "Begin IJKA Block", true);
    Tensor<T>                                X8       = to_dense_tensor(ec_dense, vtijka);
    std::function<bool(std::vector<size_t>)> dp_cond8 = [&](std::vector<size_t> cond) {
      if(cond[0] < nelectrons_alpha && cond[2] < nelectrons_alpha && nelectrons_alpha <= cond[1] &&
         nactv <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X8, dp_cond8, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End IJKA Block", true);
      T first_val = tamm::get_tensor_element(X8, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X8"] = first_val;
    }
    Tensor<T>::deallocate(X8);

    if(rank == 0) print_blockstr(results_file, "Begin IABC Block", true);
    Tensor<T>                                X9       = to_dense_tensor(ec_dense, vtiabc);
    std::function<bool(std::vector<size_t>)> dp_cond9 = [&](std::vector<size_t> cond) {
      if(cond[0] < nelectrons_alpha && cond[2] < nactv && nactv <= cond[1] && nactv <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X9, dp_cond9, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End IABC Block", true);
      T first_val = tamm::get_tensor_element(X9, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X9"] = first_val;
    }
    Tensor<T>::deallocate(X9);
  }

  cc_t2     = std::chrono::high_resolution_clock::now();
  ducc_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << std::endl
              << "DUCC: Time to write results: " << std::fixed << std::setprecision(2) << ducc_time
              << " secs" << std::endl;

  free_tensors(ftij, vtijkl);
  if(nactv > 0) { free_tensors(ftia, ftab, vtijka, vtaijb, vtijab, vtiabc, vtabcd); }

  if(rank == 0) sys_data.write_json_data("DUCC");
}

using T = double;
template void DUCC_T_CCSD_Driver<T>(SystemData sys_data, ExecutionContext& ec,
                                    const TiledIndexSpace& MO, Tensor<T>& t1, Tensor<T>& t2,
                                    Tensor<T>& f1, V2Tensors<T>& v2tensors, size_t nactv,
                                    ExecutionHW ex_hw);
