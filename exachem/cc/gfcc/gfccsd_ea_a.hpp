/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/gfcc/gfccsd_ea.hpp"

namespace exachem::cc::gfcc {

template<typename T>
class GFCCSD_EA_A_Driver {
public:
  // Constructor
  GFCCSD_EA_A_Driver() = default;

  // Destructor
  virtual ~GFCCSD_EA_A_Driver() = default;

  // Copy constructor
  GFCCSD_EA_A_Driver(const GFCCSD_EA_A_Driver&)            = default;
  GFCCSD_EA_A_Driver& operator=(const GFCCSD_EA_A_Driver&) = default;
  GFCCSD_EA_A_Driver(GFCCSD_EA_A_Driver&&)                 = default;
  GFCCSD_EA_A_Driver& operator=(GFCCSD_EA_A_Driver&&)      = default;

  /**
   * @brief Driver function for electron affinity GF-CCSD (alpha spin)
   */
  virtual void gfccsd_driver_ea_a(
    ExecutionContext& gec, ChemEnv& chem_env, const TiledIndexSpace& MO, Tensor<T>& t1_a,
    Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab, Tensor<T>& f1,
    Tensor<T>& t2v2_v, Tensor<T>& lt12_v_a, Tensor<T>& lt12_v_b, Tensor<T>& iy1_1_a,
    Tensor<T>& iy1_1_b, Tensor<T>& iy1_2_1_a, Tensor<T>& iy1_2_1_b, Tensor<T>& iy1_a,
    Tensor<T>& iy1_b, Tensor<T>& iy2_a, Tensor<T>& iy2_b, Tensor<T>& iy3_1_aaaa,
    Tensor<T>& iy3_1_bbbb, Tensor<T>& iy3_1_abab, Tensor<T>& iy3_1_baba, Tensor<T>& iy3_1_baab,
    Tensor<T>& iy3_1_abba, Tensor<T>& iy3_1_2_a, Tensor<T>& iy3_1_2_b, Tensor<T>& iy3_aaaa,
    Tensor<T>& iy3_bbbb, Tensor<T>& iy3_abab, Tensor<T>& iy3_baba, Tensor<T>& iy3_baab,
    Tensor<T>& iy3_abba, Tensor<T>& iy4_1_aaaa, Tensor<T>& iy4_1_baab, Tensor<T>& iy4_1_baba,
    Tensor<T>& iy4_1_bbbb, Tensor<T>& iy4_1_abba, Tensor<T>& iy4_1_abab, Tensor<T>& iy4_2_aaaa,
    Tensor<T>& iy4_2_baab, Tensor<T>& iy4_2_bbbb, Tensor<T>& iy4_2_abba, Tensor<T>& iy5_aaaa,
    Tensor<T>& iy5_abab, Tensor<T>& iy5_baab, Tensor<T>& iy5_bbbb, Tensor<T>& iy5_baba,
    Tensor<T>& iy5_abba, Tensor<T>& iy6_a, Tensor<T>& iy6_b, Tensor<T>& v2ijab_aaaa,
    Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, Tensor<T>& cholOO_a, Tensor<T>& cholOO_b,
    Tensor<T>& cholOV_a, Tensor<T>& cholOV_b, Tensor<T>& cholVV_a, Tensor<T>& cholVV_b,
    std::vector<T>& p_evl_sorted_occ, std::vector<T>& p_evl_sorted_virt, const TAMM_SIZE nocc,
    const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& CI, const TiledIndexSpace& unit_tis,
    std::string files_prefix, std::string levelstr, double gf_omega);
};

} // namespace exachem::cc::gfcc
