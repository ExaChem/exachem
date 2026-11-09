/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/fcidump.hpp"

template<typename T = double>
bool nonzero(T value) {
  const T tresh = 1e-10;
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

std::string symmetry_string(std::vector<int>& sym) {
  std::stringstream out;
  for(auto& i: sym) out << i << ",";
  return out.str();
}

template<typename T>
void fcidump::write_2el_ints(std::ofstream& file, SystemData& sys_data, Tensor<T> V, int norb,
                             bool is_uhf, int offset) {
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
                   << '\n';
            }
          }
        }
      }
    }
  }
}

template<typename T>
void fcidump::write_1el_ints(std::ofstream& file, SystemData& sys_data, Tensor<T> h, int norb,
                             bool is_uhf) {
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
               << std::setw(4) << 0 << std::setw(4) << 0 << std::setw(4) << '\n';
        }
      }
    }
  }
}

template<typename T>
T fcidump::one_shot_hf_energy(ChemEnv& chem_env, Tensor<T> H_MO, Tensor<T> full_v2, T core_energy) {
  ExecutionContext&      ec = get_ec(full_v2());
  const TiledIndexSpace& MO = chem_env.is_context.MSO;
  const TiledIndexSpace  O  = MO("occ");
  const auto [i, j, k, l]   = MO.labels<4>("occ");

  // In the MSO basis the density is the identity over the occupied orbitals
  Tensor<T> D{{O, O}, {1, 1}};
  Tensor<T> G{{O, O}, {1, 1}};
  Tensor<T> energy{};

  Scheduler sch{ec};
  sch.allocate(D, G, energy)(D(i, j) = 0).execute();

  auto set_identity = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, D());
    if(blockid[0] != blockid[1]) return;

    const TAMM_SIZE size = D.block_size(blockid);
    const size_t    dim  = D.block_dims(blockid)[0];
    std::vector<T>  buf(size, 0);
    for(size_t x = 0; x < dim; x++) buf[x * dim + x] = 1;
    D.put(blockid, buf);
  };
  block_for(ec, D(), set_identity);

  // clang-format off
  sch
    (G(i,j)    = full_v2(i,j,k,l) * D(k,l))
    (G(i,j)   -= full_v2(i,k,j,l) * D(k,l))
    (energy()  = H_MO(i,j) * D(i,j))
    (energy() += 0.5 * G(i,j) * D(i,j))
    .execute(ec.exhw());
  // clang-format on

  const T e_total = core_energy + get_scalar(energy);

  sch.deallocate(D, G, energy).execute();

  return e_total;
}

template<typename T>
void fcidump::write_fcidump_file(ChemEnv& chem_env, Tensor<T> H_MO, Tensor<T> full_v2,
                                 std::vector<int> orbsym, std::string filename) {
  SystemData& sys_data = chem_env.sys_data;
  double      nuc_rep  = sys_data.results["output"]["SCF"]["nucl_rep_energy"];
  bool        is_uhf   = sys_data.is_unrestricted;
  int         norbs    = static_cast<int>(orbsym.size());
  int         nelec    = sys_data.nelectrons;
  int         spin     = sys_data.nelectrons_alpha - sys_data.nelectrons_beta;
  int         isym     = 1;

  ExecutionContext& ec = get_ec(full_v2());

  // With frozen core orbitals, H_MO already includes their potential, so the scalar term only
  // needs their energy. frozen_core_energy is zero when no core orbitals are frozen.
  const double core_energy = chem_env.cd_context.frozen_core_energy;
  const double scalar      = nuc_rep + core_energy;

  const double e_one_shot = one_shot_hf_energy<T>(chem_env, H_MO, full_v2, scalar);

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
    file << std::setw(16) << scalar << "     " << 0 << "   " << 0 << "   " << 0 << "   " << 0
         << std::endl;

    std::cout << std::endl << "FCIDUMP file written to: " << filename << std::endl;
    file.close();
  }

  if(ec.pg().rank() == 0) {
    std::cout << std::fixed << std::setprecision(10);
    if(sys_data.n_frozen_core > 0) std::cout << "Frozen core energy: " << core_energy << std::endl;
    if(sys_data.is_ks) {
      std::cout << "One-shot HF energy (on Kohn-Sham orbitals): " << e_one_shot << std::endl;
    }
    else {
      const double e_scf = chem_env.scf_context.hf_energy;
      const double diff  = e_one_shot - e_scf;
      std::cout << "One-shot HF energy: " << e_one_shot << std::endl
                << "SCF energy        : " << e_scf << std::endl
                << "Difference        : " << std::scientific << std::setprecision(4) << diff
                << std::endl
                << std::defaultfloat;
    }
  }
}

template<typename T>
void fcidump::generate_fcidump(ChemEnv& chem_env, ExecutionContext& ec, const Tensor<T>& hcore,
                               const Tensor<T>& lcao, const Tensor<T>& full_v2, ExecutionHW ex_hw,
                               std::string filename) {
  SystemData& sys_data = chem_env.sys_data;

  if(filename.empty()) filename = chem_env.get_files_prefix("", "fci") + ".fcidump";
  const fs::path files_dir = fs::path(filename).parent_path();
  if(!files_dir.empty() && !fs::exists(files_dir)) fs::create_directories(files_dir);

  const TiledIndexSpace& AO  = lcao.tiled_index_spaces()[0];
  const TiledIndexSpace& MSO = lcao.tiled_index_spaces()[1];
  const auto [mu, nu]        = AO.labels<2>("all");
  const auto [z1, z2]        = MSO.labels<2>("all");

  // Transform the core Hamiltonian to the one-electron integrals in the MO basis
  Tensor<T> hcore_mo{{MSO, MSO}, {1, 1}};
  Tensor<T> tmp{MSO, AO};
  Scheduler sch{ec};
  // clang-format off
  sch.allocate(hcore_mo, tmp)
    (tmp(z1,nu) = lcao(mu,z1) * hcore(mu,nu))
    (hcore_mo(z1,z2) = tmp(z1,nu) * lcao(nu,z2))
    .deallocate(tmp).execute(ex_hw);
  // clang-format on

  if(sys_data.n_frozen_core > 0 && chem_env.cd_context.keep_frozen_core)
    sch(hcore_mo(z1, z2) += chem_env.cd_context.frozen_core_potential(z1, z2)).execute(ex_hw);

  // One entry per orbital in the MO space (linear dependencies and frozen orbitals excluded)
  const size_t     nmso = MSO.max_num_indices();
  std::vector<int> symvec(sys_data.is_unrestricted ? nmso : nmso / 2, 1);

  write_fcidump_file(chem_env, hcore_mo, full_v2, symvec, filename);

  free_tensors(hcore_mo);
}

template void fcidump::write_2el_ints<double>(std::ofstream& file, SystemData& sys_data,
                                              Tensor<double> V, int norb, bool is_uhf, int offset);

template void fcidump::write_1el_ints<double>(std::ofstream& file, SystemData& sys_data,
                                              Tensor<double> h, int norb, bool is_uhf);

template double fcidump::one_shot_hf_energy<double>(ChemEnv& chem_env, Tensor<double> H_MO,
                                                    Tensor<double> full_v2, double core_energy);

template void fcidump::write_fcidump_file<double>(ChemEnv& chem_env, Tensor<double> H_MO,
                                                  Tensor<double> full_v2, std::vector<int> orbsym,
                                                  std::string filename);

template void fcidump::generate_fcidump<double>(ChemEnv& chem_env, ExecutionContext& ec,
                                                const Tensor<double>& hcore,
                                                const Tensor<double>& lcao,
                                                const Tensor<double>& full_v2, ExecutionHW ex_hw,
                                                std::string filename);
