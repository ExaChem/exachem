/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccsd/ccsd_util.hpp"

using namespace tamm;

namespace exachem::cc::ccsd {

void cd_ccsd_driver(ExecutionContext& ec, ChemEnv& chem_env);

// Encapsulated CCSD (closed-shell) ANN routines in a template utility class.
template<typename T>
class CD_CCSD_CS {
protected:
  // Former free-floating globals moved into class scope as inline static members
  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;

  Tensor<T> a22_abab, a22_aaaa, a22_bbbb;

  Tensor<T> _a01V, _a02V, _a007V;  // scalar (Cholesky index) temporaries
  Tensor<T> i0_temp, t2_aaaa_temp; // CS-only scratch

  CCSE_Tensors<T> _a01, _a02, _a03, _a04, _a05, _a06;
  CCSE_Tensors<T> _a001, _a004, _a006, _a008, _a009, _a017, _a019, _a020,
    _a021; // _a022 unused

  virtual void ccsd_e_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                         Tensor<T>& de, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                         const Tensor<T>& t2_aaaa, std::vector<CCSE_Tensors<T>>& f1_se,
                         std::vector<CCSE_Tensors<T>>& chol3d_se);

  virtual void ccsd_t1_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                          Tensor<T>& i0_aa, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                          std::vector<CCSE_Tensors<T>>& f1_se,
                          std::vector<CCSE_Tensors<T>>& chol3d_se);

  virtual void ccsd_t2_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                          Tensor<T>& i0_abab, const Tensor<T>& t1_aa, Tensor<T>& t2_abab,
                          Tensor<T>& t2_aaaa, std::vector<CCSE_Tensors<T>>& f1_se,
                          std::vector<CCSE_Tensors<T>>& chol3d_se);

public:
  // Default constructor
  CD_CCSD_CS() = default;

  // Destructor
  ~CD_CCSD_CS() = default;

  CD_CCSD_CS(const CD_CCSD_CS& other)                = default;
  CD_CCSD_CS(CD_CCSD_CS&& other) noexcept            = default;
  CD_CCSD_CS& operator=(const CD_CCSD_CS& other)     = default;
  CD_CCSD_CS& operator=(CD_CCSD_CS&& other) noexcept = default;

  virtual std::tuple<double, double> cd_ccsd_cs_driver(
    ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
    Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& d_f1, Tensor<T>& r1_aa, Tensor<T>& r2_abab,
    std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
    std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d,
    Tensor<T> dt1_full, Tensor<T> dt2_full, bool ccsd_restart = false, std::string out_fp = "",
    bool computeTData = false);
};

} // namespace exachem::cc::ccsd
