/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "cd_ccsd_cs_ann.hpp"

template<typename T>
void ccsd_e_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de,
               CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2, std::vector<CCSE_Tensors<T>>& f1_se,
               std::vector<CCSE_Tensors<T>>& chol3d_se);

template<typename T>
void ccsd_t1_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                CCSE_Tensors<T>& r1_vo, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se);

template<typename T>
void ccsd_t2_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                CCSE_Tensors<T>& r2, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se,
                CCSE_Tensors<T>& i0tmp);

template<typename T>
std::tuple<double, double>
cd_ccsd_os_driver(SystemData& sys_data, ExecutionContext& ec, const TiledIndexSpace& MO,
                  const TiledIndexSpace& CI, Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1,
                  Tensor<T>& d_r1, Tensor<T>& d_r2, std::vector<Tensor<T>>& d_r1s,
                  std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
                  std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d,
                  bool ccsd_restart = false, std::string out_fp = "", bool computeTData = false);
