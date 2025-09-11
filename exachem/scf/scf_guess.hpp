/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/scf/scf_common.hpp"
#include "exachem/scf/scf_compute.hpp"

#if defined(TAMM_USE_ELPA)
#include <elpa/elpa.h>
#endif

namespace exachem::scf {

template<typename T>
class SCFGuess {
private:
  /// computes orbital occupation numbers for a subshell of size \c size created
  /// by smearing no more than \c ne electrons (corresponds to spherical averaging)
  ///
  /// @param[in,out] occvec occupation vector, increments by \c size on return
  /// @param[in] size the size of the subshell
  /// @param[in,out] ne the number of electrons, on return contains the number of
  /// "remaining" electrons
  void subshell_occvec(T& occvec, size_t size, size_t& ne);

  /// @brief computes average orbital occupancies in the ground state of a neutral
  ///        atoms
  /// @return occupation vector corresponding to the ground state electronic
  ///         configuration of a neutral atom with atomic number \c Z
  ///         corresponding to the orbital ordering.

  const std::vector<T> compute_ao_occupation_vector(size_t Z);

public:
  SCFGuess()          = default;
  virtual ~SCFGuess() = default;

  SCFGuess(const SCFGuess&)                = default;
  SCFGuess& operator=(const SCFGuess&)     = default;
  SCFGuess(SCFGuess&&) noexcept            = default;
  SCFGuess& operator=(SCFGuess&&) noexcept = default;

  virtual Matrix compute_soad(const std::vector<Atom>& atoms);

  virtual void compute_dipole_ints(ExecutionContext& ec, const SCFData& spvars, Tensor<T>& tensorX,
                                   Tensor<T>& tensorY, Tensor<T>& tensorZ,
                                   const std::vector<libint2::Atom>& atoms,
                                   const libint2::BasisSet& shells, libint2::Operator otype);
  virtual void compute_1body_ints(ExecutionContext& ec, const SCFData& scf_data,
                                  Tensor<T>& tensor1e, const std::vector<libint2::Atom>& atoms,
                                  const libint2::BasisSet& shells, libint2::Operator otype) const;
  virtual void compute_ecp_ints(ExecutionContext& ec, const SCFData& scf_data, Tensor<T>& tensor1e,
                                const std::vector<libecpint::GaussianShell>& shells,
                                const std::vector<libecpint::ECP>&           ecps);
  virtual void compute_pchg_ints(ExecutionContext& ec, const SCFData& scf_data, Tensor<T>& tensor1e,
                                 const std::vector<std::pair<double, std::array<double, 3>>>& q,
                                 const libint2::BasisSet& shells, libint2::Operator otype);
  virtual void scf_diagonalize(Scheduler& sch, const ChemEnv& chem_env, SCFData& scf_data,
                               ScalapackInfo& scalapack_info, TAMMTensors<T>& ttensors,
                               EigenTensors& etensors);

  virtual void compute_sad_guess(ExecutionContext& ec, ChemEnv& chem_env, SCFData& scf_data,
                                 ScalapackInfo& scalapack_info, EigenTensors& etensors,
                                 TAMMTensors<T>& ttensors);

  template<int ndim>
  void t2e_hf_helper(const ExecutionContext& ec, tamm::Tensor<T>& ttensor, Matrix& etensor,
                     const std::string& ustr = "");
};

} // namespace exachem::scf
