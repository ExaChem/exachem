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
class DefaultSCFIO {
protected:
  double tt_trace(ExecutionContext& ec, Tensor<T>& T1, Tensor<T>& T2);

public:
  virtual ~DefaultSCFIO() = default;

  // Matrix I/O routines
  virtual Matrix read_scf_mat(const std::string& matfile);
  virtual void   write_scf_mat(Matrix& C, const std::string& matfile);

  // SCF I/O routines
  virtual void rw_md_disk(ExecutionContext& ec, const ChemEnv& chem_env,
                          ScalapackInfo& scalapack_info, TAMMTensors<T>& ttensors,
                          EigenTensors& etensors, std::string files_prefix, bool read = false);

  virtual void rw_mat_disk(Tensor<T> tensor, std::string tfilename, bool profile,
                           bool read = false);
  virtual void print_mulliken(ChemEnv& chem_env, Matrix& D, Matrix& D_beta, Matrix& S);
  virtual void print_energies(ExecutionContext& ec, ChemEnv& chem_env, TAMMTensors<T>& ttensors,
                              EigenTensors& etensors, SCFData& scf_data,
                              ScalapackInfo& scalapack_info);
};

template<typename T>
class SCFIO: public DefaultSCFIO<T> {
  // Optionally override methods here
};

} // namespace exachem::scf
