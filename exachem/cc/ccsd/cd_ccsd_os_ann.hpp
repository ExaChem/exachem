/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccsd/cd_ccsd.hpp"
#include "exachem/cc/ccsd/cd_ccsd_cs_ann.hpp"

template<typename T>
class CD_CCSD_OS {
protected:
  // Inline static members (previously file-scope globals) holding
  // TiledIndexSpaces, intermediates, and scratch tensors for OS-ANN CCSD.
  // They are default-constructed and sized/allocated within driver methods.
  TiledIndexSpace o_alpha_os, v_alpha_os, o_beta_os, v_beta_os;

  // V^4 contraction scratch (may be used conditionally by variants)
  Tensor<T> a22_abab_os, a22_aaaa_os, a22_bbbb_os;

  // Scalar (vector over Cholesky index) temporaries
  Tensor<T> _a01V_os, _a02V_os, _a007V_os;

  // Labeled CCSE_Tensors intermediates
  CCSE_Tensors<T> _a01_os, _a02_os, _a03_os, _a04_os, _a05_os, _a06_os, _a001_os, _a004_os,
    _a006_os, _a008_os, _a009_os, _a017_os, _a019_os, _a020_os, _a021_os, _a022_os;

  virtual void ccsd_e_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                         Tensor<T>& de, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                         std::vector<CCSE_Tensors<T>>& f1_se,
                         std::vector<CCSE_Tensors<T>>& chol3d_se);

  virtual void ccsd_t1_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                          CCSE_Tensors<T>& r1_vo, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                          std::vector<CCSE_Tensors<T>>& f1_se,
                          std::vector<CCSE_Tensors<T>>& chol3d_se);

  virtual void ccsd_t2_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                          CCSE_Tensors<T>& r2, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                          std::vector<CCSE_Tensors<T>>& f1_se,
                          std::vector<CCSE_Tensors<T>>& chol3d_se, CCSE_Tensors<T>& i0tmp);

public:
  // Constructor
  CD_CCSD_OS() = default;

  // Destructor
  virtual ~CD_CCSD_OS() = default;

  CD_CCSD_OS(const CD_CCSD_OS& other)                = default;
  CD_CCSD_OS& operator=(const CD_CCSD_OS& other)     = default;
  CD_CCSD_OS(CD_CCSD_OS&& other) noexcept            = default;
  CD_CCSD_OS& operator=(CD_CCSD_OS&& other) noexcept = default;

  virtual std::tuple<double, double>
  cd_ccsd_os_driver(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO,
                    const TiledIndexSpace& CI, Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1,
                    Tensor<T>& d_r1, Tensor<T>& d_r2, std::vector<Tensor<T>>& d_r1s,
                    std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
                    std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d,
                    bool ccsd_restart = false, std::string out_fp = "", bool computeTData = false);
};
