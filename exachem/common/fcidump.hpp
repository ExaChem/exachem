/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "cc/ccsd_util.hpp"
#include <fstream>

namespace fcidump {

template<typename T>
bool nonzero(T value) {
  const double tresh = 1e-10;
  if(std::fabs(value) > tresh) return true;
  else return false;
}

bool nonredundant(int i, int j, int k, int l, bool is_uhf) {
  if(!is_uhf) {
    if(i >= j /*&& i*(i+1)/2+j >= k*(k+1)/2+l */ && k >= l) return true;
    return false;
  }
  else {
    if(i >= j && k >= l) return true;
    return false;
  }
}

template<typename T>
void write_2el_ints(std::ofstream& file, SystemData& sys_data, Tensor<T> V, int norb, bool is_uhf,
                    int offset = 0) {
  EXPECTS(V.num_modes() == 4);

  const size_t noa  = sys_data.n_occ_alpha;
  const size_t nob  = sys_data.n_occ_beta;
  const size_t nva  = sys_data.n_vir_alpha;
  const size_t nocc = sys_data.nocc;

  for(auto it: V.loop_nest()) {
    auto blockid = internal::translate_blockid(it, V());
    if(!V.is_non_zero(blockid)) continue;

    TAMM_SIZE      size = V.block_size(blockid);
    std::vector<T> buf(size);
    V.get(blockid, buf);

    auto block_dims   = V.block_dims(blockid);
    auto block_offset = V.block_offsets(blockid);

    size_t c{}, ix{}, jx{}, kx{}, lx{};
    int    factor = 1;
    if(is_uhf) factor = 2;

    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
        for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
          for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++, c++) {
            // if((i + 1 > norb || j + 1 > norb || k + 1 > norb || l + 1 > norb) && !is_uhf)
            // continue;
            if(!is_uhf) {
              if(i >= noa && i < nocc) continue;
              if(j >= noa && j < nocc) continue;
              if(!((k >= noa && k < nocc) || (k >= nocc + nva))) continue;
              if(!((l >= noa && l < nocc) || (l >= nocc + nva))) continue;
              if(i >= nocc + nva) continue;
              if(j >= nocc + nva) continue;
            }

            if(i < noa) ix = factor * i + 1;
            if(j < noa) jx = factor * j + 1;
            if(k < noa) kx = factor * k + 1;
            if(l < noa) lx = factor * l + 1;

            if(i >= noa && i < nocc) ix = factor * (i - noa + 1);
            if(j >= noa && j < nocc) jx = factor * (j - noa + 1);
            if(k >= noa && k < nocc) kx = factor * (k - noa + 1);
            if(l >= noa && l < nocc) lx = factor * (l - noa + 1);

            if(i >= nocc && i < nocc + nva) ix = factor * (i - nob) + 1;
            if(j >= nocc && j < nocc + nva) jx = factor * (j - nob) + 1;
            if(k >= nocc && k < nocc + nva) kx = factor * (k - nob) + 1;
            if(l >= nocc && l < nocc + nva) lx = factor * (l - nob) + 1;

            if(i >= nocc + nva) ix = factor * (i - nva - noa + 1);
            if(j >= nocc + nva) jx = factor * (j - nva - noa + 1);
            if(k >= nocc + nva) kx = factor * (k - nva - noa + 1);
            if(l >= nocc + nva) lx = factor * (l - nva - noa + 1);

            if(nonredundant(ix, jx, kx, lx, is_uhf) && nonzero(buf[c])) {
              file << std::setw(16) << buf[c] << std::setw(6) << offset + ix << std::setw(4)
                   << offset + jx << std::setw(4) << offset + kx << std::setw(4) << offset + lx
                   << std::endl;
            }
          }
        }
      }
    }
  }
}

template<typename T>
void write_1el_ints(std::ofstream& file, SystemData& sys_data, Tensor<T> h, int norb, bool is_uhf) {
  EXPECTS(h.num_modes() == 2);

  const size_t noa  = sys_data.n_occ_alpha;
  const size_t nob  = sys_data.n_occ_beta;
  const size_t nva  = sys_data.n_vir_alpha;
  const size_t nocc = sys_data.nocc;

  for(auto it: h.loop_nest()) {
    auto blockid = internal::translate_blockid(it, h());
    if(!h.is_non_zero(blockid)) continue;

    TAMM_SIZE      size = h.block_size(blockid);
    std::vector<T> buf(size);
    h.get(blockid, buf);

    auto block_dims   = h.block_dims(blockid);
    auto block_offset = h.block_offsets(blockid);

    size_t c{}, ix{}, jx{};
    int    factor = 1;
    if(is_uhf) factor = 2;

    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++, c++) {
        // if((i + 1 > norb || j + 1 > norb) && !is_uhf) continue;
        if(!is_uhf) {
          if(i >= noa && i < nocc) continue;
          if(j >= noa && j < nocc) continue;
          if(i >= nocc + nva) continue;
          if(j >= nocc + nva) continue;
        }

        if(i < noa) ix = factor * i + 1;
        if(j < noa) jx = factor * j + 1;

        if(i >= noa && i < nocc) ix = factor * (i - noa + 1);
        if(j >= noa && j < nocc) jx = factor * (j - noa + 1);

        if(i >= nocc && i < nocc + nva) ix = factor * (i - nob) + 1;
        if(j >= nocc && j < nocc + nva) jx = factor * (j - nob) + 1;

        if(i >= nocc + nva) ix = factor * (i - nva - noa + 1);
        if(j >= nocc + nva) jx = factor * (j - nva - noa + 1);

        if((ix >= jx) && nonzero(buf[c])) {
          file << std::setw(16) << buf[c] << std::setw(6) << ix << std::setw(4) << jx
               << std::setw(4) << 0 << std::setw(4) << 0 << std::setw(4) << std::endl;
        }
      }
    }
  }
}

std::string symmetry_string(std::vector<int>& sym) {
  std::stringstream out;
  for(auto& i: sym) out << i << ",";
  return out.str();
}

template<typename T>
void write_fcidump_file(SystemData& sys_data, Tensor<T> H_MO, Tensor<T> full_v2,
                        std::vector<int> orbsym, std::string filename) {
  double nuc_rep = sys_data.results["output"]["SCF"]["nucl_rep_energy"];
  bool   is_uhf  = sys_data.is_unrestricted;
  int    norbs   = is_uhf ? sys_data.nbf_orig * 2 : sys_data.nbf_orig;
  int    nelec   = sys_data.nelectrons;
  int    spin    = 0; // sys_data.options_map.scf_options.multiplicity;
  int    isym    = 1;

  ExecutionContext& ec = get_ec(full_v2());

  if(ec.pg().rank() == 0) {
    std::ofstream file(filename);

    auto orbsym_str = symmetry_string(orbsym);
    if(orbsym_str.empty()) orbsym_str = ",";

    file << "&FCI NORB=" << norbs << ",NELEC=" << nelec << ",MS2=" << spin
         << ",\nORBSYM=" << orbsym_str << "\nISYM=" << isym << ",IUHF=" << is_uhf << std::endl
         << "&END" << std::endl;

    file << std::fixed << std::setprecision(10);

    write_2el_ints(file, sys_data, full_v2, norbs, is_uhf);
    write_1el_ints(file, sys_data, H_MO, norbs, is_uhf);
    file << std::setw(16) << nuc_rep << "     " << 0 << "   " << 0 << "   " << 0 << "   " << 0
         << std::endl;

    std::cout << std::endl << "Integral file written to: " << filename << std::endl;
    file.close();
  }
}

}; // namespace fcidump
