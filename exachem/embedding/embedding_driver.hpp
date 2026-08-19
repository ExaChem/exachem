#pragma once

// standard C++ headers
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

#include "exachem/common/chemenv.hpp"
#include "tamm/tamm.hpp"

using libint2::BasisSet;

#include <filesystem>
namespace fs = std::filesystem;

namespace exachem::embedding {
void   embedding_driver(ExecutionContext& ec, ChemEnv& chem_env);
void   embedding(ExecutionContext& ec, ChemEnv& chem_env);
Matrix spade(const Matrix& S12, const Matrix& C_occ, const std::vector<int> indicesToKeep,
             const int n_acc_mos, bool rank0);
Matrix paos(const Matrix& S, const Matrix& C_occ, const std::vector<int> indicesToKeep, bool rank0,
            double pao_thresh1 = 0.05, double pao_thresh2 = 0.0001);
Matrix sqrtm(const Matrix& mat);
void permute_orbitals(ChemEnv& chem_env, const Matrix& C_occ, const Matrix& S, Tensor<double>& C_AO,
                      const int nocc, const int nvirtual);
void print_eigs(ExecutionContext& ec, Tensor<double>& F_tamm, Tensor<double>& C_tamm, size_t N,
                size_t NMO);

} // namespace exachem::embedding
