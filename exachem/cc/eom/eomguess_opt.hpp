/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "cc/ccsd/cd_ccsd_os_ann.hpp"
#include <algorithm>
#include <complex>
using namespace tamm;

template<typename T>
void eom_guess_opt(ExecutionContext& ec, const TiledIndexSpace& MO, TiledIndexSpace& hbar_tis,
                   int& nroots, const TAMM_SIZE n_occ_alpha, const TAMM_SIZE n_occ_beta,
                   std::vector<T>& p_evl_sorted, std::vector<Tensor<T>>& x1, bool left = false) {
  // Scheduler sch{ec};
  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  auto [h1, h2]            = O.labels<2>("all");
  auto [p1, p2]            = V.labels<2>("all");

  std::vector<T> minlist(nroots);
  for(auto root = 0; root < nroots; root++) {
    minlist[root] = 1000 + root; // large positive value
  }

  const TAMM_SIZE nbf         = static_cast<TAMM_SIZE>(p_evl_sorted.size() / 2);
  const TAMM_SIZE noab        = n_occ_alpha + n_occ_beta;
  const TAMM_SIZE n_vir_alpha = nbf - n_occ_alpha;
  const TAMM_SIZE n_vir_beta  = nbf - n_occ_beta;
  const TAMM_SIZE nvab        = n_vir_alpha + n_vir_beta;

  std::vector<T> p_evl_sorted_occ(noab);
  std::vector<T> p_evl_sorted_virt(nvab);
  std::copy(p_evl_sorted.begin(), p_evl_sorted.begin() + noab, p_evl_sorted_occ.begin());
  std::copy(p_evl_sorted.begin() + noab, p_evl_sorted.end(), p_evl_sorted_virt.begin());
  Matrix eom_diff(noab, nvab);
  eom_diff.setZero();

  for(TAMM_SIZE x = 0; x < n_occ_alpha; x++) {
    for(TAMM_SIZE y = 0; y < n_vir_alpha; y++) {
      eom_diff(x, y) = p_evl_sorted_virt[y] - p_evl_sorted_occ[x];
      auto max_ml    = std::max_element(minlist.begin(), minlist.end());
      if(eom_diff(x, y) < *max_ml) minlist[std::distance(minlist.begin(), max_ml)] = eom_diff(x, y);
    }
  }

  for(TAMM_SIZE x = n_occ_alpha; x < noab; x++) {
    for(TAMM_SIZE y = n_vir_alpha; y < nvab; y++) {
      eom_diff(x, y) = p_evl_sorted_virt[y] - p_evl_sorted_occ[x];
      auto max_ml    = std::max_element(minlist.begin(), minlist.end());
      if(eom_diff(x, y) < *max_ml) minlist[std::distance(minlist.begin(), max_ml)] = eom_diff(x, y);
    }
  }

  auto root   = 0;
  auto max_ml = std::max_element(minlist.begin(), minlist.end());

  for(TAMM_SIZE x = 0; x < n_occ_alpha; x++) {
    for(TAMM_SIZE y = 0; y < n_vir_alpha; y++) {
      if(eom_diff(x, y) <= *max_ml) {
        if(left) update_tensor_val(x1.at(root), {{(size_t) x, (size_t) y}}, 1.0);
        else update_tensor_val(x1.at(root), {{(size_t) y, (size_t) x}}, 1.0);
        root++;
      }
    }
  }

  for(auto x = n_occ_alpha; x < noab; x++) {
    for(auto y = n_vir_alpha; y < nvab; y++) {
      if(eom_diff(x, y) <= *max_ml) {
        if(left) update_tensor_val(x1.at(root), {{(size_t) x, (size_t) y}}, 1.0);
        else update_tensor_val(x1.at(root), {{(size_t) y, (size_t) x}}, 1.0);
        root++;
      }
    }
  }

  // NOTE: All that is important is which pairs of indices {a,i} give
  //      the 'nroots'-number of lowest energy differences. The differences don't matter
  //      only what pair of indices {a,i} give the lowest energy differences.
  //
  //      The above algorithm creates the DIFF matrix and a list of the lowest energy differences
  //      in the first loop and then goes through the DIFF array and check if a given value is
  //      below or equal to the largest values in minlist. If it is then the pair of {a,i} indices
  //      is used to create an initial guess x1 vector.
  //
  //      If in the first loop minlist could not only store the lowest values, but also which pair
  //      of indices {a,i} give the lowest energy differences, then there would be no need for the
  //      second loop to search all values of DIFF. Instead, for each pair {a,i} in the minlist,
  //      create the initial guess vector x1.
}
