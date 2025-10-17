/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/cc2/cd_cc2_cs.hpp"

namespace exachem::cc2 {

// Template class encapsulating Open-Shell CC2 routines (previously free functions)
// Variable names in signatures are preserved exactly.
template<typename T>
class CD_CC2_OS_Engine {
protected:
  CCSE_Tensors<T> _a021_os;
  TiledIndexSpace o_alpha_os, v_alpha_os, o_beta_os, v_beta_os;

  Tensor<T>       _a01V_os, _a02V_os, _a007V_os;
  CCSE_Tensors<T> _a01_os, _a02_os, _a03_os, _a04_os, _a05_os, _a06_os, _a001_os, _a004_os,
    _a006_os, _a008_os, _a009_os, _a017_os, _a019_os, _a020_os; //_a022

  virtual void cc2_e_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                        Tensor<T>& de, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                        std::vector<CCSE_Tensors<T>>& f1_se,
                        std::vector<CCSE_Tensors<T>>& chol3d_se);

  virtual void cc2_t1_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                         CCSE_Tensors<T>& r1_vo, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                         std::vector<CCSE_Tensors<T>>& f1_se,
                         std::vector<CCSE_Tensors<T>>& chol3d_se);

  virtual void cc2_t2_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                         CCSE_Tensors<T>& r2, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                         std::vector<CCSE_Tensors<T>>& f1_se,
                         std::vector<CCSE_Tensors<T>>& chol3d_se, CCSE_Tensors<T>& i0tmp,
                         Tensor<T>& d_f1, Tensor<T>& cv3d, Tensor<T>& res_2);

public:
  // Constructor
  CD_CC2_OS_Engine() = default;

  CD_CC2_OS_Engine(const CD_CC2_OS_Engine&)            = default;
  CD_CC2_OS_Engine(CD_CC2_OS_Engine&&)                 = default;
  CD_CC2_OS_Engine& operator=(const CD_CC2_OS_Engine&) = default;
  CD_CC2_OS_Engine& operator=(CD_CC2_OS_Engine&&)      = default;

  // Destructor
  virtual ~CD_CC2_OS_Engine() = default;

  virtual std::tuple<double, double>
  run(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
      Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1, Tensor<T>& d_r1, Tensor<T>& d_r2,
      std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
      std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d,
      bool cc2_restart = false, std::string out_fp = "", bool computeTData = false);
};

}; // namespace exachem::cc2
