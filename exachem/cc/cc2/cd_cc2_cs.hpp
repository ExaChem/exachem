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

namespace exachem::cc2 {

// Template class encapsulating CC2 Closed-Shell routines (was previously free functions)
// Variable names are preserved exactly as in original free function interfaces.
template<typename T>
class CD_CC2_CS_Engine {
protected:
  using CCEType = double;
  CCSE_Tensors<CCEType> _a021; // reused in driver logic
  TiledIndexSpace       o_alpha, v_alpha, o_beta, v_beta;

  Tensor<CCEType>       _a01V, _a02V, _a007V;
  CCSE_Tensors<CCEType> _a01, _a02, _a03, _a04, _a05, _a06, _a001, _a004, _a006, _a008, _a009,
    _a017, _a019, _a020;                 //_a022
  Tensor<CCEType> i0_temp, t2_aaaa_temp; // CS only

  virtual void cc2_e_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                        Tensor<T>& de, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                        const Tensor<T>& t2_aaaa, std::vector<CCSE_Tensors<T>>& f1_se,
                        std::vector<CCSE_Tensors<T>>& chol3d_se);

  virtual void cc2_t1_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                         Tensor<T>& i0_aa, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                         std::vector<CCSE_Tensors<T>>& f1_se,
                         std::vector<CCSE_Tensors<T>>& chol3d_se);

  virtual void cc2_t2_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                         Tensor<T>& i0_abab, const Tensor<T>& t1_aa, Tensor<T>& t2_abab,
                         Tensor<T>& t2_aaaa, std::vector<CCSE_Tensors<T>>& f1_se,
                         std::vector<CCSE_Tensors<T>>& chol3d_se, Tensor<T>& d_f1, Tensor<T>& cv3d,
                         Tensor<T>& res_2);

public:
  virtual std::tuple<double, double>
  run(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
      Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& d_f1, Tensor<T>& r1_aa, Tensor<T>& r2_abab,
      std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
      std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d,
      Tensor<T> dt1_full, Tensor<T> dt2_full, bool cc2_restart = false, std::string out_fp = "",
      bool computeTData = false);

  virtual ~CD_CC2_CS_Engine()                              = default;
  CD_CC2_CS_Engine()                                       = default;
  CD_CC2_CS_Engine(const CD_CC2_CS_Engine&)                = default;
  CD_CC2_CS_Engine(CD_CC2_CS_Engine&&) noexcept            = default;
  CD_CC2_CS_Engine& operator=(const CD_CC2_CS_Engine&)     = default;
  CD_CC2_CS_Engine& operator=(CD_CC2_CS_Engine&&) noexcept = default;
};

}; // namespace exachem::cc2
