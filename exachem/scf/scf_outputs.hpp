#pragma once

#include "common/chemenv.hpp"
#include "common/cutils.hpp"
#include "common/ec_basis.hpp"

#include "scf/scf_eigen_tensors.hpp"
#include "scf/scf_matrix.hpp"
#include "scf/scf_tamm_tensors.hpp"
#include "scf/scf_vars.hpp"

namespace exachem::scf {

class SCFIO: public SCFMatrix {
private:
  template<typename TensorType>
  double tt_trace(ExecutionContext& ec, Tensor<TensorType>& T1, Tensor<TensorType>& T2);

public:
  void rw_md_disk(ExecutionContext& ec, const ChemEnv& chem_env, ScalapackInfo& scalapack_info,
                  TAMMTensors& ttensors, EigenTensors& etensors, std::string files_prefix,
                  bool read = false);
  template<typename T>
  void rw_mat_disk(Tensor<T> tensor, std::string tfilename, bool profile, bool read = false);
  void print_mulliken(ChemEnv& chem_env, Matrix& D, Matrix& D_beta, Matrix& S);
  void print_energies(ExecutionContext& ec, const ChemEnv& chem_env, TAMMTensors& ttensors,
                      EigenTensors& etensors, SCFVars& scf_vars, ScalapackInfo& scalapack_info);
};
} // namespace exachem::scf
