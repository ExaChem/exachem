/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/chemenv.hpp"
#include "exachem/common/cutils.hpp"
#include "exachem/scf/scf_common.hpp"
#include "exachem/scf/scf_guess.hpp"
using libint2::Atom;
using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
using shellpair_data_t =
  std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>; // in same order as
                                                                 // shellpair_list_t

namespace exachem::scf {

template<typename T>
class SCFCompute {
public:
  SCFCompute()                                 = default;
  virtual ~SCFCompute()                        = default;
  SCFCompute(const SCFCompute&)                = default;
  SCFCompute& operator=(const SCFCompute&)     = default;
  SCFCompute(SCFCompute&&) noexcept            = default;
  SCFCompute& operator=(SCFCompute&&) noexcept = default;

  virtual void compute_shellpair_list(const ExecutionContext& ec, const libint2::BasisSet& shells,
                                      SCFData& scf_data) const;
  virtual void compute_trafo(const libint2::BasisSet& shells, EigenTensors& etensors) const;
  virtual std::tuple<int, double> compute_NRE(const ExecutionContext&           ec,
                                              const std::vector<libint2::Atom>& atoms) const;
  virtual std::tuple<shellpair_list_t, shellpair_data_t>
               compute_shellpairs(const libint2::BasisSet& bs1,
                                  const libint2::BasisSet& bs2       = libint2::BasisSet(),
                                  double                   threshold = 1e-16) const;
  virtual void compute_orthogonalizer(ExecutionContext& ec, ChemEnv& chem_env, SCFData& scf_data,
                                      ScalapackInfo&  scalapack_info,
                                      TAMMTensors<T>& ttensors) const;

  virtual std::tuple<std::vector<size_t>, std::vector<Tile>, std::vector<Tile>>
  compute_AO_tiles(const ExecutionContext& ec, const ChemEnv& chem_env,
                   const libint2::BasisSet& shells, const bool is_df = false) const;

  virtual void recompute_tilesize(ExecutionContext& ec, ChemEnv& chem_env,
                                  bool is_df = false) const;

  virtual void compute_sdens_to_cdens(const libint2::BasisSet& shells, Matrix& Spherical,
                                      Matrix& Cartesian, EigenTensors& etensors) const;

  virtual void compute_cpot_to_spot(const libint2::BasisSet& shells, Matrix& Spherical,
                                    Matrix& Cartesian, EigenTensors& etensors) const;

  virtual void compute_hamiltonian(ExecutionContext& ec, const SCFData& scf_data,
                                   const ChemEnv& chem_env, TAMMTensors<T>& ttensors,
                                   EigenTensors& etensors) const;

  virtual void compute_density(ExecutionContext& ec, const ChemEnv& chem_env,
                               const SCFData& scf_data, ScalapackInfo& scalapack_info,
                               TAMMTensors<T>& ttensors, EigenTensors& etensors) const;

  virtual std::pair<double, double> compute_s2(ExecutionContext& ec, const ChemEnv& chem_env,
                                               const SCFData& scf_data) const;

  template<libint2::Operator Kernel = libint2::Operator::coulomb>
  Matrix
  compute_schwarz_ints(ExecutionContext& ec, const SCFData& scf_data, const libint2::BasisSet& bs1,
                       const libint2::BasisSet& bs2 = libint2::BasisSet(), bool use_2norm = false,
                       typename libint2::operator_traits<Kernel>::oper_params_type params =
                         libint2::operator_traits<Kernel>::default_params()) const;
};

} // namespace exachem::scf
