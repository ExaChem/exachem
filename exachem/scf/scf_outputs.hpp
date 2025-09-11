/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/cutils.hpp"
#include "exachem/common/ec_basis.hpp"
#include "exachem/scf/scf_compute.hpp"
#include "exachem/scf/scf_data.hpp"
#include "exachem/scf/scf_tensors.hpp"
#include <string>

namespace exachem::scf {

template<typename T>
class SCFIO {
protected:
  virtual double tt_trace(ExecutionContext& ec, const Tensor<T>& T1, const Tensor<T>& T2) const;

public:
  SCFIO()                            = default;
  virtual ~SCFIO()                   = default;
  SCFIO(const SCFIO&)                = default;
  SCFIO& operator=(const SCFIO&)     = default;
  SCFIO(SCFIO&&) noexcept            = default;
  SCFIO& operator=(SCFIO&&) noexcept = default;

  // Matrix I/O routines
  virtual Matrix read_scf_mat(const std::string& matfile) const;
  virtual void   write_scf_mat(const Matrix& C, const std::string& matfile) const;

  // SCF I/O routines
  virtual void rw_md_disk(ExecutionContext& ec, const ChemEnv& chem_env,
                          ScalapackInfo& scalapack_info, TAMMTensors<T>& ttensors,
                          EigenTensors& etensors, const std::string& files_prefix,
                          bool read = false) const;

  virtual void rw_mat_disk(Tensor<T> tensor, const std::string& tfilename, bool profile,
                           bool read = false) const;
  virtual void print_mulliken(ChemEnv& chem_env, const Matrix& D, const Matrix& D_beta,
                              const Matrix& S) const;
  virtual void print_energies(ExecutionContext& ec, ChemEnv& chem_env, TAMMTensors<T>& ttensors,
                              EigenTensors& etensors, const SCFData& scf_data,
                              ScalapackInfo& scalapack_info) const;
};

} // namespace exachem::scf
