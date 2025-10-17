/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <tamm/tamm.hpp>

using namespace tamm;

namespace exachem::cc::ccsd_lambda {

template<typename T>
class CCSD_RDM {
public:
  // Constructor
  CCSD_RDM() = default;

  // Destructor
  virtual ~CCSD_RDM() = default;

  // Copy constructor and assignment
  CCSD_RDM(const CCSD_RDM&)            = default;
  CCSD_RDM& operator=(const CCSD_RDM&) = default;
  CCSD_RDM(CCSD_RDM&&)                 = default;
  CCSD_RDM& operator=(CCSD_RDM&&)      = default;

  /**
   * @brief Compute 1-electron reduced density matrix (1-RDM)
   */
  virtual Tensor<T> compute_1rdm(std::vector<int>& cc_rdm, std::string files_prefix, Scheduler& sch,
                                 TiledIndexSpace& MO, Tensor<T> d_t1, Tensor<T> d_t2,
                                 Tensor<T> d_y1, Tensor<T> d_y2);

  /**
   * @brief Compute 2-electron reduced density matrix (2-RDM)
   */
  virtual Tensor<T> compute_2rdm(std::vector<int>& cc_rdm, std::string files_prefix, Scheduler& sch,
                                 TiledIndexSpace& MO, Tensor<T> d_t1, Tensor<T> d_t2,
                                 Tensor<T> d_y1, Tensor<T> d_y2);
};

// Explicit instantiation declaration (definition in .cpp) - prevents implicit instantiation
extern template class CCSD_RDM<double>;

} // namespace exachem::cc::ccsd_lambda
