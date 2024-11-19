/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/cc2/cd_cc2_cs.hpp"

namespace exachem::cc2::cc2_os {

template<typename T>
void cc2_e_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de,
              CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2, std::vector<CCSE_Tensors<T>>& f1_se,
              std::vector<CCSE_Tensors<T>>& chol3d_se);

template<typename T>
void cc2_t1_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
               CCSE_Tensors<T>& r1_vo, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
               std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se);

template<typename T>
void cc2_t2_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
               CCSE_Tensors<T>& r2, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
               std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se,
               CCSE_Tensors<T>& i0tmp, Tensor<T>& d_f1, Tensor<T>& cv3d, Tensor<T>& res_2);

template<typename T>
std::tuple<double, double>
cd_cc2_os_driver(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO,
                 const TiledIndexSpace& CI, Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1,
                 Tensor<T>& d_r1, Tensor<T>& d_r2, std::vector<Tensor<T>>& d_r1s,
                 std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
                 std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d,
                 bool cc2_restart = false, std::string out_fp = "", bool computeTData = false);

}; // namespace exachem::cc2::cc2_os
