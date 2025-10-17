/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cholesky/cholesky_2e_driver.hpp"
#include "exachem/common/ec_basis.hpp"
#include "exachem/common/system_data.hpp"
#include "exachem/scf/scf_compute.hpp"
#include "tamm/eigen_utils.hpp"
#if defined(USE_UPCXX)
#include "tamm/ga_over_upcxx.hpp"
#endif

using namespace tamm;

using TAMM_GA_SIZE = int64_t;

namespace exachem::cholesky_2e {

class Cholesky_2E_Util {
public:
  Cholesky_2E_Util()          = default;
  virtual ~Cholesky_2E_Util() = default;

  // Copy constructor and copy assignment operator
  Cholesky_2E_Util(const Cholesky_2E_Util&)            = default;
  Cholesky_2E_Util& operator=(const Cholesky_2E_Util&) = default;

  // Move constructor and move assignment operator
  Cholesky_2E_Util(Cholesky_2E_Util&&)            = default;
  Cholesky_2E_Util& operator=(Cholesky_2E_Util&&) = default;

  virtual int get_ts_recommendation(ExecutionContext& ec, ChemEnv& chem_env);

  virtual std::tuple<TiledIndexSpace, TAMM_SIZE>
  setup_mo_red(ExecutionContext& ec, ChemEnv& chem_env, bool triples = false);

  virtual std::tuple<TiledIndexSpace, TAMM_SIZE> setupMOIS(ExecutionContext& ec, ChemEnv& chem_env,
                                                           bool triples = false);

  virtual void update_sysdata(ExecutionContext& ec, ChemEnv& chem_env, TiledIndexSpace& MO,
                              bool is_mso = true);
  // reshape F/lcao after freezing
  virtual Matrix reshape_mo_matrix(ChemEnv& chem_env, Matrix& emat, bool is_lcao = false);
};

// Backward compatibility wrapper functions
int get_ts_recommendation(ExecutionContext& ec, ChemEnv& chem_env);

std::tuple<TiledIndexSpace, TAMM_SIZE> setup_mo_red(ExecutionContext& ec, ChemEnv& chem_env,
                                                    bool triples = false);

std::tuple<TiledIndexSpace, TAMM_SIZE> setupMOIS(ExecutionContext& ec, ChemEnv& chem_env,
                                                 bool triples = false);

void update_sysdata(ExecutionContext& ec, ChemEnv& chem_env, TiledIndexSpace& MO,
                    bool is_mso = true);

// reshape F/lcao after freezing
Matrix reshape_mo_matrix(ChemEnv& chem_env, Matrix& emat, bool is_lcao = false);

template<typename T>
class Cholesky_2E {
public:
  Cholesky_2E()          = default;
  virtual ~Cholesky_2E() = default;

  // Copy constructor and copy assignment operator
  Cholesky_2E(const Cholesky_2E&)            = default;
  Cholesky_2E& operator=(const Cholesky_2E&) = default;

  // Move constructor and move assignment operator
  Cholesky_2E(Cholesky_2E&&)            = default;
  Cholesky_2E& operator=(Cholesky_2E&&) = default;

  virtual void cholesky_2e(ExecutionContext& ec, ChemEnv& chem_env);
};

// Backward compatibility wrapper function
template<typename T>
void cholesky_2e(ExecutionContext& ec, ChemEnv& chem_env);

} // namespace exachem::cholesky_2e
