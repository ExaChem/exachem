/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/gfcc/gf_diis.hpp"
#include "exachem/cc/gfcc/gf_guess.hpp"
#include "exachem/cc/gfcc/gfccsd_ea.hpp"
#include "exachem/cc/gfcc/gfccsd_ea_a.hpp"
#include "exachem/cc/gfcc/gfccsd_ea_b.hpp"
#include "exachem/cc/gfcc/gfccsd_ip.hpp"
#include "exachem/cc/gfcc/gfccsd_ip_a.hpp"
#include "exachem/cc/gfcc/gfccsd_ip_b.hpp"
#include "exachem/cc/lambda/ccsd_lambda.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"

namespace exachem::cc::gfcc {

template<typename T>
class GFCCSD_Driver {
public:
  // Constructor
  GFCCSD_Driver() = default;

  // Destructor
  virtual ~GFCCSD_Driver() = default;

  // Copy constructor and assignment
  GFCCSD_Driver(const GFCCSD_Driver&)            = default;
  GFCCSD_Driver& operator=(const GFCCSD_Driver&) = default;
  GFCCSD_Driver(GFCCSD_Driver&&)                 = default;
  GFCCSD_Driver& operator=(GFCCSD_Driver&&)      = default;

  /**
   * @brief Main GF-CCSD driver function
   */
  virtual void gfccsd_driver(ExecutionContext& ec, ChemEnv& chem_env);

  /**
   * @brief Write GF-CCSD results to JSON file
   */
  virtual void write_results_to_json(ExecutionContext& ec, ChemEnv& chem_env, int level,
                                     std::vector<T>& ni_w, std::vector<T>& ni_A,
                                     std::string gfcc_type);

  /**
   * @brief Write string to disk from all MPI ranks
   */
  virtual void write_string_to_disk(ExecutionContext& ec, const std::string& tstring,
                                    const std::string& filename);
};

// Wrapper function for backward compatibility
void gfccsd_driver(ExecutionContext& ec, ChemEnv& chem_env);

} // namespace exachem::cc::gfcc
