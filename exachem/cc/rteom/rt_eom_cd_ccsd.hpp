/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "exachem/cc/ccsd/ccsd_util.hpp"
#include "exachem/common/chemenv.hpp"

namespace exachem::rteom_cc::ccsd {
void rt_eom_cd_ccsd_driver(ExecutionContext& ec, ChemEnv& chem_env);

// Implementation detail class wrapping free functions from rt_eom_cd_ccsd.cpp
template<typename T>
class RT_EOM_CD_CCSD {
public:
  using CCEType = std::complex<double>;

  // Constructor
  RT_EOM_CD_CCSD() = default;

  // Destructor
  virtual ~RT_EOM_CD_CCSD() = default;

  RT_EOM_CD_CCSD(const RT_EOM_CD_CCSD&)                = default;
  RT_EOM_CD_CCSD& operator=(const RT_EOM_CD_CCSD&)     = default;
  RT_EOM_CD_CCSD(RT_EOM_CD_CCSD&&) noexcept            = default;
  RT_EOM_CD_CCSD& operator=(RT_EOM_CD_CCSD&&) noexcept = default;

  bool            debug = false;
  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;

  Tensor<CCEType>       _a01V, _a02V, _a007V; // vector intermediates
  CCSE_Tensors<CCEType> _a01, _a02, _a03, _a04, _a05, _a06, _a001, _a004, _a006, _a008, _a009,
    _a017, _a019, _a020, _a021, _a022;

  // Public high-level driver
  virtual void rt_eom_cd_ccsd(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO,
                              const TiledIndexSpace& CI, Tensor<T>& d_f1,
                              std::vector<T>& p_evl_sorted, Tensor<T>& cv3d, bool cc_restart,
                              std::string rt_eom_fp);

protected:
  // energy evaluation
  virtual void ccsd_e_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                         Tensor<T>& de, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                         std::vector<CCSE_Tensors<T>>& f1_se,
                         std::vector<CCSE_Tensors<T>>& chol3d_se);

  // t1 residual
  virtual void ccsd_t1_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                          CCSE_Tensors<T>& r1_vo, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                          std::vector<CCSE_Tensors<T>>& f1_se,
                          std::vector<CCSE_Tensors<T>>& chol3d_se);

  // t2 residual
  virtual void ccsd_t2_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                          CCSE_Tensors<T>& r2, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                          std::vector<CCSE_Tensors<T>>& f1_se,
                          std::vector<CCSE_Tensors<T>>& chol3d_se, CCSE_Tensors<T>& i0tmp);

  // Scale complex tensor in-place: (re,im) -> (alpha_real*re, alpha_imag*im)
  virtual void scale_complex_ip(Tensor<T> tensor, double alpha_real, double alpha_imag);

  virtual void debug_full_rt(Scheduler& sch, const TiledIndexSpace& MO, CCSE_Tensors<T>& r1_vo,
                             CCSE_Tensors<T>& r2_vvoo);

  // Copy src into dest swapping real/imag (val = (imag, real)); optionally accumulate
  virtual void complex_copy_swap(ExecutionContext& ec, Tensor<T> src, Tensor<T> dest,
                                 bool update = true);

  // Iteration print helper
  virtual void td_iteration_print(ChemEnv& chem_env, int iter, CCEType energy, CCEType x1_1,
                                  CCEType x1_2, CCEType x2_1, CCEType x2_2, CCEType x2_3,
                                  double time);
};

} // namespace exachem::rteom_cc::ccsd
