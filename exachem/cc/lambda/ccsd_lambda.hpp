/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"
#include "exachem/scf/scf_main.hpp"

using namespace tamm;

namespace exachem::cc::ccsd_lambda {

// Forward declarations for tensor helper aggregates used in CCSDLambda_Engine static methods.
template<typename T>
struct Y1Tensors;
template<typename T>
struct Y2Tensors;

// Non-template wrapper function for external interface
void ccsd_lambda_driver(ExecutionContext& ec, ChemEnv& chem_env);

// Refactored: encapsulate CCSD Lambda routines inside a class interface.
template<typename T>
class CCSDLambda_Engine {
public:
  // Special member functions
  CCSDLambda_Engine()                                    = default;
  virtual ~CCSDLambda_Engine()                           = default;
  CCSDLambda_Engine(const CCSDLambda_Engine&)            = default;
  CCSDLambda_Engine(CCSDLambda_Engine&&)                 = default;
  CCSDLambda_Engine& operator=(const CCSDLambda_Engine&) = default;
  CCSDLambda_Engine& operator=(CCSDLambda_Engine&&)      = default;

  // Main driver method that orchestrates the entire CCSD Lambda computation
  void run(ExecutionContext& ec, ChemEnv& chem_env);

  // Printing helper (formerly free function).
  void iteration_print_lambda(ChemEnv& chem_env, const ProcGroup& pg, int iter, double residual,
                              double time);

  // Allocate and initialize lambda tensors (formerly free function setupLambdaTensors).
  static std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
                    std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
  setupLambdaTensors(ExecutionContext& ec, TiledIndexSpace& MO, size_t ndiis);

  // y1 / y2 kernel builders.
  void lambda_ccsd_y1(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                      Tensor<T>& i0, const Tensor<T>& t1, const Tensor<T>& t2, const Tensor<T>& y1,
                      const Tensor<T>& y2, const Tensor<T>& f1,
                      cholesky_2e::V2Tensors<T>& v2tensors, Tensor<T>& cv3d,
                      Y1Tensors<T>& y1tensors);

  void lambda_ccsd_y2(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                      Tensor<T>& i0, const Tensor<T>& t1, Tensor<T>& t2, const Tensor<T>& y1,
                      Tensor<T>& y2, const Tensor<T>& f1, cholesky_2e::V2Tensors<T>& v2tensors,
                      Tensor<T>& cv3d, Y2Tensors<T>& y2tensors);

  std::tuple<double, double> lambda_ccsd_driver(
    ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
    Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1, cholesky_2e::V2Tensors<T>& v2tensors,
    Tensor<T>& cv3d, Tensor<T>& d_r1, Tensor<T>& d_r2, Tensor<T>& d_y1, Tensor<T>& d_y2,
    std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_y1s,
    std::vector<Tensor<T>>& d_y2s, std::vector<T>& p_evl_sorted);
};

template<typename T>
struct Y1Tensors {
  Tensor<T> i_1;
  Tensor<T> i_1_1;
  Tensor<T> i_2;
  Tensor<T> i_2_1;
  Tensor<T> i_3;
  Tensor<T> i_3_1;
  Tensor<T> i_3_1_1;
  Tensor<T> i_3_2;
  Tensor<T> i_3_3;
  Tensor<T> i_3_4;
  Tensor<T> i_4;
  Tensor<T> i_4_1;
  Tensor<T> i_4_1_1;
  Tensor<T> i_4_2;
  Tensor<T> i_4_3;
  Tensor<T> i_4_4;
  Tensor<T> i_5;
  Tensor<T> i_6;
  Tensor<T> i_6_1;
  Tensor<T> i_6_2;
  Tensor<T> i_7;
  Tensor<T> i_8;
  Tensor<T> i_9;
  Tensor<T> i_10;
  Tensor<T> i_11;
  Tensor<T> i_11_1;
  Tensor<T> i_11_1_1;
  Tensor<T> i_11_2;
  Tensor<T> i_11_3;
  Tensor<T> i_12;
  Tensor<T> i_12_1;
  Tensor<T> i_13;
  Tensor<T> i_13_1;

  Tensor<T> tmp_1;
  Tensor<T> tmp_2;
  Tensor<T> tmp_4;
  Tensor<T> tmp_5;
  Tensor<T> tmp_6;
  Tensor<T> tmp_7;
  Tensor<T> tmp_8;
  Tensor<T> tmp_9;
  Tensor<T> tmp_11;
  Tensor<T> tmp_12;
  Tensor<T> tmp_13;
  Tensor<T> tmp_14;
  Tensor<T> tmp_15;
  Tensor<T> tmp_16;
  Tensor<T> tmp_19;
  Tensor<T> tmp_20;
  // Tensor<T> tmp_3 {{CI},{1,1}};
  // Tensor<T> tmp_10 {{O, V, V, V},{2,2}};
  // Tensor<T> tmp_17 {{O, O, V, V, CI},{2,2}};
  // Tensor<T> tmp_18 {{O, O, V, V},{2,2}};

  void allocate(ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");

    i_1      = Tensor<T>{{O, O}, {1, 1}};
    i_1_1    = Tensor<T>{{O, V}, {1, 1}};
    i_2      = Tensor<T>{{V, V}, {1, 1}};
    i_2_1    = Tensor<T>{{O, V}, {1, 1}};
    i_3      = Tensor<T>{{V, O}, {1, 1}};
    i_3_1    = Tensor<T>{{O, O}, {1, 1}};
    i_3_1_1  = Tensor<T>{{O, V}, {1, 1}};
    i_3_2    = Tensor<T>{{V, V}, {1, 1}};
    i_3_3    = Tensor<T>{{O, V}, {1, 1}};
    i_3_4    = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_4      = Tensor<T>{{O, V, O, O}, {2, 2}};
    i_4_1    = Tensor<T>{{O, O, O, O}, {2, 2}};
    i_4_1_1  = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_4_2    = Tensor<T>{{O, V, O, V}, {2, 2}};
    i_4_3    = Tensor<T>{{O, V}, {1, 1}};
    i_4_4    = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_5      = Tensor<T>{{V, V, O, V}, {2, 2}};
    i_6      = Tensor<T>{{V, O}, {1, 1}};
    i_6_1    = Tensor<T>{{O, O}, {1, 1}};
    i_6_2    = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_7      = Tensor<T>{{O, O}, {1, 1}};
    i_8      = Tensor<T>{{O, O}, {1, 1}};
    i_9      = Tensor<T>{{V, V}, {1, 1}};
    i_10     = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_11     = Tensor<T>{{O, V, O, O}, {2, 2}};
    i_11_1   = Tensor<T>{{O, O, O, O}, {2, 2}};
    i_11_1_1 = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_11_2   = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_11_3   = Tensor<T>{{O, O}, {1, 1}};
    i_12     = Tensor<T>{{O, O, O, O}, {2, 2}};
    i_12_1   = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_13     = Tensor<T>{{O, V, O, V}, {2, 2}};
    i_13_1   = Tensor<T>{{O, O, O, V}, {2, 2}};

    tmp_1  = Tensor<T>{{V, O, CI}, {1, 1}};
    tmp_2  = Tensor<T>{CI};
    tmp_4  = Tensor<T>{{V, V, CI}, {1, 1}};
    tmp_5  = Tensor<T>{{V, O, CI}, {1, 1}};
    tmp_6  = Tensor<T>{{V, O, CI}, {1, 1}};
    tmp_7  = Tensor<T>{{V, O, CI}, {1, 1}};
    tmp_8  = Tensor<T>{{O, O, CI}, {1, 1}};
    tmp_9  = Tensor<T>{{O, V, V, V}, {2, 2}};
    tmp_11 = Tensor<T>{{V, V, V, V}, {2, 2}};
    tmp_12 = Tensor<T>{{O, O, V, V}, {2, 2}};
    tmp_13 = Tensor<T>{CI};
    tmp_14 = Tensor<T>{{O, V, CI}, {1, 1}};
    tmp_15 = Tensor<T>{{O, O, CI}, {1, 1}};
    tmp_16 = Tensor<T>{{O, V, CI}, {1, 1}};
    tmp_19 = Tensor<T>{{O, V, CI}, {1, 1}};
    tmp_20 = Tensor<T>{{O, V, CI}, {1, 1}};

    Tensor<T>::allocate(&ec, i_1, i_1_1, i_2, i_2_1, i_3, i_3_1, i_3_1_1, i_3_2, i_3_3, i_3_4, i_4,
                        i_4_1, i_4_1_1, i_4_2, i_4_3, i_4_4, i_5, i_6, i_6_1, i_6_2, i_7, i_8, i_9,
                        i_10, i_11, i_11_1, i_11_1_1, i_11_2, i_11_3, i_12, i_12_1, i_13, i_13_1,
                        tmp_1, tmp_2, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9, tmp_11, tmp_12,
                        tmp_13, tmp_14, tmp_15, tmp_16, tmp_19, tmp_20);
  }

  void deallocate() {
    Tensor<T>::deallocate(i_1, i_1_1, i_2, i_2_1, i_3, i_3_1, i_3_1_1, i_3_2, i_3_3, i_3_4, i_4,
                          i_4_1, i_4_1_1, i_4_2, i_4_3, i_4_4, i_5, i_6, i_6_1, i_6_2, i_7, i_8,
                          i_9, i_10, i_11, i_11_1, i_11_1_1, i_11_2, i_11_3, i_12, i_12_1, i_13,
                          i_13_1, tmp_1, tmp_2, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9, tmp_11,
                          tmp_12, tmp_13, tmp_14, tmp_15, tmp_16, tmp_19, tmp_20);
  }
};

template<typename T>
struct Y2Tensors {
  Tensor<T> i_1;
  Tensor<T> i_2;
  Tensor<T> i_3;
  Tensor<T> i_3_1;
  Tensor<T> i_4;
  Tensor<T> i_4_1;
  Tensor<T> i_5;
  Tensor<T> i_5_1;
  Tensor<T> i_6;
  Tensor<T> i_7;
  Tensor<T> i_8;
  Tensor<T> i_9;
  Tensor<T> i_10;
  Tensor<T> i_11;
  Tensor<T> i_12;
  Tensor<T> i_12_1;
  Tensor<T> i_13;
  Tensor<T> i_13_1;

  Tensor<T> tmp_1;
  Tensor<T> tmp_2;
  Tensor<T> tmp_3;
  Tensor<T> tmp_4;
  Tensor<T> tmp_5;
  Tensor<T> tmp_6;
  Tensor<T> tmp_7;
  Tensor<T> tmp_8;
  Tensor<T> tmp_9;
  Tensor<T> tmp_10;
  Tensor<T> tmp_11;
  Tensor<T> tmp_12;
  Tensor<T> tmp_13;

  void allocate(ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");

    i_1    = Tensor<T>{{O, V}, {1, 1}};
    i_2    = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_3    = Tensor<T>{{O, O}, {1, 1}};
    i_3_1  = Tensor<T>{{O, V}, {1, 1}};
    i_4    = Tensor<T>{{V, V}, {1, 1}};
    i_4_1  = Tensor<T>{{O, V}, {1, 1}};
    i_5    = Tensor<T>{{O, O, O, O}, {2, 2}};
    i_5_1  = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_6    = Tensor<T>{{O, V, O, V}, {2, 2}};
    i_7    = Tensor<T>{{O, O}, {1, 1}};
    i_8    = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_9    = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_10   = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_11   = Tensor<T>{{V, V}, {1, 1}};
    i_12   = Tensor<T>{{O, O, O, O}, {2, 2}};
    i_12_1 = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_13   = Tensor<T>{{O, V, O, V}, {2, 2}};
    i_13_1 = Tensor<T>{{O, O, O, V}, {2, 2}};

    tmp_1  = Tensor<T>{{O, V, CI}, {1, 1}};
    tmp_2  = Tensor<T>{{O, V, CI}, {1, 1}};
    tmp_3  = Tensor<T>{{O, V, CI}, {1, 1}};
    tmp_4  = Tensor<T>{{O, V, CI}, {1, 1}};
    tmp_5  = Tensor<T>{{O, V, CI}, {1, 1}};
    tmp_6  = Tensor<T>{CI};
    tmp_7  = Tensor<T>{{O, V, CI}, {1, 1}};
    tmp_8  = Tensor<T>{{O, O, CI}, {1, 1}};
    tmp_9  = Tensor<T>{{V, V, V, V}, {2, 2}};
    tmp_10 = Tensor<T>{{O, O, V, V}, {2, 2}};
    tmp_11 = Tensor<T>{{O, V, V, V}, {2, 2}};
    tmp_12 = Tensor<T>{{O, O, V, V}, {2, 2}};
    tmp_13 = Tensor<T>{{O, O, V, V}, {2, 2}};

    Tensor<T>::allocate(&ec, i_1, i_2, i_3, i_3_1, i_4, i_4_1, i_5, i_5_1, i_6, i_7, i_8, i_9, i_10,
                        i_11, i_12, i_12_1, i_13, i_13_1, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6,
                        tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12, tmp_13);
  }

  void deallocate() {
    Tensor<T>::deallocate(i_1, i_2, i_3, i_3_1, i_4, i_4_1, i_5, i_5_1, i_6, i_7, i_8, i_9, i_10,
                          i_11, i_12, i_12_1, i_13, i_13_1, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5,
                          tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12, tmp_13);
  }
};

// (All former free function templates now live as static methods on CCSDLambda.)
} // namespace exachem::cc::ccsd_lambda
