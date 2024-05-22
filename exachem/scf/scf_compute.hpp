#pragma once

#include "common/chemenv.hpp"
#include "common/cutils.hpp"
#include "scf/scf_common.hpp"
#include "scf/scf_guess.hpp"
using libint2::Atom;
using TensorType       = double;
using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
using shellpair_data_t =
  std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>; // in same order as
                                                                 // shellpair_list_t

#pragma once

namespace exachem::scf {

class SCFCompute {
public:
  void compute_shellpair_list(const ExecutionContext& ec, const libint2::BasisSet& shells,
                              SCFVars& scf_vars);
  void compute_trafo(const libint2::BasisSet& shells, EigenTensors& etensors);
  std::tuple<int, double> compute_NRE(const ExecutionContext&     ec,
                                      std::vector<libint2::Atom>& atoms);
  std::tuple<shellpair_list_t, shellpair_data_t>
       compute_shellpairs(const libint2::BasisSet& bs1,
                          const libint2::BasisSet& bs2 = libint2::BasisSet(), double threshold = 1e-16);
  void compute_orthogonalizer(ExecutionContext& ec, ChemEnv& chem_env, SCFVars& scf_vars,
                              ScalapackInfo& scalapack_info, TAMMTensors& ttensors);

  std::tuple<std::vector<size_t>, std::vector<Tile>, std::vector<Tile>>
  compute_AO_tiles(const ExecutionContext& ec, ChemEnv& chem_env, libint2::BasisSet& shells,
                   const bool is_df = false);

  void recompute_tilesize(tamm::Tile& tile_size, const int N, const bool force_ts,
                          const bool rank0);

  template<typename TensorType>
  void compute_sdens_to_cdens(const libint2::BasisSet& shells, Matrix& Spherical, Matrix& Cartesian,
                              EigenTensors& etensors);

  template<typename TensorType>
  void compute_cpot_to_spot(const libint2::BasisSet& shells, Matrix& Spherical, Matrix& Cartesian,
                            EigenTensors& etensors);

  template<typename TensorType>
  void compute_hamiltonian(ExecutionContext& ec, const SCFVars& scf_vars, ChemEnv& chem_env,
                           TAMMTensors& ttensors, EigenTensors& etensors);

  template<typename TensorType>
  void compute_density(ExecutionContext& ec, ChemEnv& chem_env, const SCFVars& scf_vars,
                       ScalapackInfo& scalapack_info, TAMMTensors& ttensors,
                       EigenTensors& etensors);

  template<libint2::Operator Kernel = libint2::Operator::coulomb>
  Matrix compute_schwarz_ints(ExecutionContext& ec, const SCFVars& scf_vars,
                              const libint2::BasisSet& bs1,
                              const libint2::BasisSet& bs2 = libint2::BasisSet(),
                              bool use_2norm               = false, // use infty norm by default
                              typename libint2::operator_traits<Kernel>::oper_params_type params =
                                libint2::operator_traits<Kernel>::default_params());
};
} // namespace exachem::scf
