
// clang-format off
#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"
#include "exachem/cc/ccsd_t/ccsd_t_fused_driver.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"
// clang-format on

double ccsdt_s1_t1_GetTime  = 0;
double ccsdt_s1_v2_GetTime  = 0;
double ccsdt_d1_t2_GetTime  = 0;
double ccsdt_d1_v2_GetTime  = 0;
double ccsdt_d2_t2_GetTime  = 0;
double ccsdt_d2_v2_GetTime  = 0;
double genTime              = 0;
double ccsd_t_data_per_rank = 0; // in GB

namespace exachem::cc::ccsd_t {
template<typename T>
class CCSD_T_Driver {
public:
  CCSD_T_Driver()          = default;
  virtual ~CCSD_T_Driver() = default;

  // Copy constructor and copy assignment operator
  CCSD_T_Driver(const CCSD_T_Driver&)            = default;
  CCSD_T_Driver& operator=(const CCSD_T_Driver&) = default;

  // Move constructor and move assignment operator
  CCSD_T_Driver(CCSD_T_Driver&&)            = default;
  CCSD_T_Driver& operator=(CCSD_T_Driver&&) = default;

  void ccsd_t_driver(ExecutionContext& ec, ChemEnv& chem_env);
};

} // namespace exachem::cc::ccsd_t
