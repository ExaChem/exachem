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
#include "exachem/scf/scf_data.hpp"
#include "exachem/scf/scf_tensors.hpp"

using namespace tamm;
using libint2::Atom;
using std::cerr;
using std::cout;
using std::endl;
using std::string;

namespace exachem::scf {

// DENSITY FITTING
class DFFockEngine {
public:
  const libint2::BasisSet& obs;
  const libint2::BasisSet& dfbs;
  DFFockEngine(const libint2::BasisSet& _obs, const libint2::BasisSet& _dfbs):
    obs(_obs), dfbs(_dfbs) {}
};

class SCFUtil {
public:
  SCFUtil()                              = delete;
  SCFUtil(const SCFUtil&)                = delete;
  SCFUtil& operator=(const SCFUtil&)     = delete;
  SCFUtil(SCFUtil&&) noexcept            = delete;
  SCFUtil& operator=(SCFUtil&&) noexcept = delete;

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
  template<typename T>
  static std::tuple<size_t, double, double>
  gensqrtinv(ExecutionContext& ec, ChemEnv& chem_env, SCFData& scf_data,
             ScalapackInfo& scalapack_info, TAMMTensors<T>& ttensors, bool symmetric = false,
             double threshold = 1e-5);
  template<typename T>
  static std::tuple<Matrix, size_t, double, double>
  gensqrtinv_atscf(ExecutionContext& ec, const ChemEnv& chem_env, const SCFData& scf_data,
                   ScalapackInfo& scalapack_info, Tensor<T> S1, TiledIndexSpace& tao_atom,
                   bool symmetric, double threshold);

  static size_t max_nprim(const libint2::BasisSet& shells) {
    size_t n = 0;
    for(const auto& shell: shells) n = std::max(shell.nprim(), n);
    return n;
  }

  static int max_l(const libint2::BasisSet& shells) {
    int l = 0;
    for(const auto& shell: shells)
      for(const auto& c: shell.contr) l = std::max(c.l, l);
    return l;
  }
  template<typename T>
  static std::vector<size_t> sort_indexes(const std::vector<T>& v, bool reverse = false);
  template<typename T>
  static std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
  gather_task_vectors(ExecutionContext& ec, const std::vector<int>& s1vec,
                      const std::vector<int>& s2vec, const std::vector<int>& ntask_vec);
};
} // namespace exachem::scf
