/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cctype>

#include "exachem/common/chemenv.hpp"
#include "exachem/common/cutils.hpp"
#include "exachem/scf/scf_eigen_tensors.hpp"
#include "exachem/scf/scf_tamm_tensors.hpp"
#include "exachem/scf/scf_vars.hpp"

using namespace tamm;
using libint2::Atom;
using std::cerr;
using std::cout;
using std::endl;
using std::string;

using TensorType = double;

namespace exachem::scf {

// DENSITY FITTING
class DFFockEngine {
public:
  const libint2::BasisSet& obs;
  const libint2::BasisSet& dfbs;
  DFFockEngine(const libint2::BasisSet& _obs, const libint2::BasisSet& _dfbs):
    obs(_obs), dfbs(_dfbs) {}
};

std::tuple<size_t, double, double> gensqrtinv(ExecutionContext& ec, ChemEnv& chem_env,
                                              SCFVars& scf_vars, ScalapackInfo& scalapack_info,
                                              TAMMTensors& ttensors, bool symmetric = false,
                                              double threshold = 1e-5);

inline size_t max_nprim(const libint2::BasisSet& shells) {
  size_t n = 0;
  for(auto shell: shells) n = std::max(shell.nprim(), n);
  return n;
}

inline int max_l(const libint2::BasisSet& shells) {
  int l = 0;
  for(auto shell: shells)
    for(auto c: shell.contr) l = std::max(c.l, l);
  return l;
}

template<typename T>
std::vector<size_t> sort_indexes(std::vector<T>& v, bool reverse = false);

// returns {X,X^{-1},rank,A_condition_number,result_A_condition_number}, where
// X is the generalized square-root-inverse such that X.transpose() * A * X = I
//
// if symmetric is true, produce "symmetric" sqrtinv: X = U . A_evals_sqrtinv .
// U.transpose()),
// else produce "canonical" sqrtinv: X = U . A_evals_sqrtinv
// where U are eigenvectors of A
// rows and cols of symmetric X are equivalent; for canonical X the rows are
// original basis (AO),
// cols are transformed basis ("orthogonal" AO)
//
// A is conditioned to max_condition_number

std::tuple<size_t, double, double> gensqrtinv(ExecutionContext& ec, ChemEnv& chem_env,
                                              SCFVars& scf_vars, ScalapackInfo& scalapack_info,
                                              TAMMTensors& ttensors, bool symmetric,
                                              double threshold);

std::tuple<Matrix, size_t, double, double>
gensqrtinv_atscf(ExecutionContext& ec, ChemEnv& chem_env, SCFVars& scf_vars,
                 ScalapackInfo& scalapack_info, Tensor<double> S1, TiledIndexSpace& tao_atom,
                 bool symmetric, double threshold);

template<typename TensorType>
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
gather_task_vectors(ExecutionContext& ec, std::vector<int>& s1vec, std::vector<int>& s2vec,
                    std::vector<int>& ntask_vec);
} // namespace exachem::scf
