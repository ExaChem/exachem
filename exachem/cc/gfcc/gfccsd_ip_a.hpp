/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/gfcc/gfccsd_ip.hpp"

namespace exachem::cc::gfcc {

template<typename T>
class GFCCSD_IP_A_Driver {
public:
  // Constructor
  GFCCSD_IP_A_Driver() = default;

  // Destructor
  virtual ~GFCCSD_IP_A_Driver() = default;

  // Copy constructor and assignment
  GFCCSD_IP_A_Driver(const GFCCSD_IP_A_Driver&)            = default;
  GFCCSD_IP_A_Driver& operator=(const GFCCSD_IP_A_Driver&) = default;
  GFCCSD_IP_A_Driver(GFCCSD_IP_A_Driver&&)                 = default;
  GFCCSD_IP_A_Driver& operator=(GFCCSD_IP_A_Driver&&)      = default;

  /**
   * @brief Driver function for ionization potential GF-CCSD (alpha spin)
   */
  virtual void gfccsd_driver_ip_a(
    ExecutionContext& gec, ChemEnv& chem_env, const TiledIndexSpace& MO, Tensor<T>& t1_a,
    Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab, Tensor<T>& f1,
    Tensor<T>& t2v2_o, Tensor<T>& lt12_o_a, Tensor<T>& lt12_o_b, Tensor<T>& ix1_1_1_a,
    Tensor<T>& ix1_1_1_b, Tensor<T>& ix2_1_aaaa, Tensor<T>& ix2_1_abab, Tensor<T>& ix2_1_bbbb,
    Tensor<T>& ix2_1_baba, Tensor<T>& ix2_2_a, Tensor<T>& ix2_2_b, Tensor<T>& ix2_3_a,
    Tensor<T>& ix2_3_b, Tensor<T>& ix2_4_aaaa, Tensor<T>& ix2_4_abab, Tensor<T>& ix2_4_bbbb,
    Tensor<T>& ix2_5_aaaa, Tensor<T>& ix2_5_abba, Tensor<T>& ix2_5_abab, Tensor<T>& ix2_5_bbbb,
    Tensor<T>& ix2_5_baab, Tensor<T>& ix2_5_baba, Tensor<T>& ix2_6_2_a, Tensor<T>& ix2_6_2_b,
    Tensor<T>& ix2_6_3_aaaa, Tensor<T>& ix2_6_3_abba, Tensor<T>& ix2_6_3_abab,
    Tensor<T>& ix2_6_3_bbbb, Tensor<T>& ix2_6_3_baab, Tensor<T>& ix2_6_3_baba,
    Tensor<T>& v2ijab_aaaa, Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb,
    std::vector<T>& p_evl_sorted_occ, std::vector<T>& p_evl_sorted_virt, const TAMM_SIZE nocc,
    const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& unit_tis, std::string files_prefix,
    std::string levelstr, double gf_omega);
};

} // namespace exachem::cc::gfcc
