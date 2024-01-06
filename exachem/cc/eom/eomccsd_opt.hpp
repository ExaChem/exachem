/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "eomguess_opt.hpp"

using namespace tamm;

void eom_ccsd_driver(std::string filename, ECOptions options_map);

template<typename T>
struct EOM_X1Tensors {
  Tensor<T> i_1;
  Tensor<T> i_1_1; // deallocated early
  Tensor<T> i_2;
  Tensor<T> i_3;
  Tensor<T> i_4;
  Tensor<T> i_5_1;
  Tensor<T> i_5;
  Tensor<T> i_5_2;
  Tensor<T> i_6;
  Tensor<T> i_7;
  Tensor<T> i_8;

  void allocate(ExecutionContext& ec, const TiledIndexSpace& MO) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");

    i_1   = Tensor<T>{{O, O}, {1, 1}};
    i_1_1 = Tensor<T>{{O, V}, {1, 1}};
    i_2   = Tensor<T>{{V, V}, {1, 1}};
    i_3   = Tensor<T>{{O, V}, {1, 1}};
    i_4   = Tensor<T>{{O, O, O, V}, {2, 2}};
    i_5_1 = Tensor<T>{{O, V}, {1, 1}};
    i_5   = Tensor<T>{{O, O}, {1, 1}};
    i_5_2 = Tensor<T>{{O, V}, {1, 1}};
    i_6   = Tensor<T>{{V, V}, {1, 1}};
    i_7   = Tensor<T>{{O, V}, {1, 1}};
    i_8   = Tensor<T>{{O, O, O, V}, {2, 2}};

    Tensor<T>::allocate(&ec, i_1, i_2, i_3, i_4, i_5_1, i_5_2, i_5, i_6, i_7, i_8);
  }

  void deallocate() { Tensor<T>::deallocate(i_1, i_2, i_3, i_4, i_5_1, i_5_2, i_5, i_6, i_7, i_8); }
};

template<typename T>
struct EOM_X2Tensors {
  Tensor<T> i_1;
  Tensor<T> i_2;
  Tensor<T> i_3;
  Tensor<T> i_4;
  Tensor<T> i_5;
  Tensor<T> i_6_1;
  Tensor<T> i_6;
  Tensor<T> i_6_2;
  Tensor<T> i_6_3;
  Tensor<T> i_6_4;
  Tensor<T> i_6_4_1;
  Tensor<T> i_6_5;
  Tensor<T> i_6_6;
  Tensor<T> i_6_7;
  Tensor<T> i_7;
  Tensor<T> i_8_1;
  Tensor<T> i_8;
  Tensor<T> i_8_2;
  Tensor<T> i_9;
  Tensor<T> i_9_1;
  Tensor<T> i_10;
  Tensor<T> i_11;
  Tensor<T> i0_temp;
  Tensor<T> i_6_temp;
  Tensor<T> i_6_4_temp;
  Tensor<T> i_9_temp;
  // deallocated early
  Tensor<T> i_1_1;
  Tensor<T> i_1_2;
  Tensor<T> i_1_3;
  Tensor<T> i_2_1;
  Tensor<T> i_4_1;
  Tensor<T> i_6_1_1;
  Tensor<T> i_1_temp;
  Tensor<T> i_4_temp;
  Tensor<T> i_6_1_temp;

  void allocate(ExecutionContext& ec, const TiledIndexSpace& MO) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");

    i_1        = {{O, V, O, O}, {2, 2}};
    i_1_1      = {{O, V, O, V}, {2, 2}};
    i_1_2      = {{O, V}, {1, 1}};
    i_1_3      = {{O, O, O, V}, {2, 2}};
    i_2        = {{O, O}, {1, 1}};
    i_2_1      = {{O, V}, {1, 1}};
    i_3        = {{V, V}, {1, 1}};
    i_4        = {{O, O, O, O}, {2, 2}};
    i_4_1      = {{O, O, O, V}, {2, 2}};
    i_5        = {{O, V, O, V}, {2, 2}};
    i_6_1      = {{O, O, O, O}, {2, 2}};
    i_6_1_1    = {{O, O, O, V}, {2, 2}};
    i_6        = {{O, V, O, O}, {2, 2}};
    i_6_2      = {{O, V}, {1, 1}};
    i_6_3      = {{O, O, O, V}, {2, 2}};
    i_6_4      = {{O, O, O, O}, {2, 2}};
    i_6_4_1    = {{O, O, O, V}, {2, 2}};
    i_6_5      = {{O, V, O, V}, {2, 2}};
    i_6_6      = {{O, V}, {1, 1}};
    i_6_7      = {{O, O, O, V}, {2, 2}};
    i_7        = {{V, V, O, V}, {2, 2}};
    i_8_1      = {{O, V}, {1, 1}};
    i_8        = {{O, O}, {1, 1}};
    i_8_2      = {{O, V}, {1, 1}};
    i_9        = {{O, O, O, O}, {2, 2}};
    i_9_1      = {{O, O, O, V}, {2, 2}};
    i_10       = {{V, V}, {1, 1}};
    i_11       = {{O, V, O, V}, {2, 2}};
    i0_temp    = {{V, V, O, O}, {2, 2}};
    i_1_temp   = {{O, V, O, O}, {2, 2}};
    i_4_temp   = {{O, O, O, O}, {2, 2}};
    i_6_1_temp = {{O, O, O, O}, {2, 2}};
    i_6_temp   = {{O, V, O, O}, {2, 2}};
    i_6_4_temp = {{O, O, O, O}, {2, 2}};
    i_9_temp   = {{O, O, O, O}, {2, 2}};

    Tensor<T>::allocate(&ec, i_1, i_2, i_3, i_4, i_5, i_6_1, i_6, i_6_2, i_6_3, i_6_4, i_6_4_1,
                        i_6_5, i_6_6, i_6_7, i_7, i_8_1, i_8, i_8_2, i_9, i_9_1, i_10, i_11,
                        i0_temp, i_6_temp, i_6_4_temp, i_9_temp);
  }

  void deallocate() {
    Tensor<T>::deallocate(i_1, i_2, i_3, i_4, i_5, i_6_1, i_6, i_6_2, i_6_3, i_6_4, i_6_4_1, i_6_5,
                          i_6_6, i_6_7, i_7, i_8_1, i_8, i_8_2, i_9, i_9_1, i_10, i_11, i0_temp,
                          i_6_temp, i_6_4_temp, i_9_temp);
  }
};

template<typename T>
void eomccsd_x1(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0, const Tensor<T>& t1,
                const Tensor<T>& t2, const Tensor<T>& x1, const Tensor<T>& x2, const Tensor<T>& f1,
                V2Tensors<T>& v2tensors, EOM_X1Tensors<T>& x1tensors);

template<typename T>
void eomccsd_x2(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0, const Tensor<T>& t1,
                const Tensor<T>& t2, const Tensor<T>& x1, const Tensor<T>& x2, const Tensor<T>& f1,
                V2Tensors<T>& v2tensors, EOM_X2Tensors<T>& x2tensors);

template<typename T>
void right_eomccsd_driver(SystemData sys_data, ExecutionContext& ec, const TiledIndexSpace& MO,
                          Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& f1, V2Tensors<T>& v2tensors,
                          std::vector<T> p_evl_sorted);
