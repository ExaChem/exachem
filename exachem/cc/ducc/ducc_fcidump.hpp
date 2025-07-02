/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include "exachem/scf/scf_main.hpp"

namespace ducc {

enum TwoElSym { None, RHF, UHF };

template<typename T>
struct DUCC_ints {
  int       npart; // I seem to have switched Particles and Holes in the whole code -Mik
  int       nhole;
  Tensor<T> hij;
  Tensor<T> hia;
  Tensor<T> hab;
  Tensor<T> Vijkl;
  Tensor<T> Vijka;
  Tensor<T> Vijab;
  Tensor<T> Vaijb;
  Tensor<T> Viabc;
  Tensor<T> Vabcd;
};

template<typename T>
struct UHF_ints {
  Tensor<T> Vaaaa;
  Tensor<T> Vbbbb;
  Tensor<T> Vaabb;
  Tensor<T> haa;
  Tensor<T> hbb;
  T         nucl_rep;
};

template<typename T>
bool nonzero(T value) {
  const double tresh = 1e-12;
  if(value > tresh || value < -tresh) return true;
  else return false;
}

bool nonredundant(int i, int j, int k, int l, TwoElSym sym = None) {
  switch(sym) {
    case None: return true;
    case RHF:
      if(i >= j && i >= k && k >= l) return true;
      return false;
    case UHF:
      if(i >= j && k >= l) return true;
      return false;
    default:
      std::cout << "Error: Unrecognized type of symmetry for FCIDUMP! Aborting." << std::endl;
      exit(1);
  }
}

template<typename T>
void write_2el_ints(std::ofstream& file, Tensor<T> V, TwoElSym sym, int offset = 0) {
  EXPECTS(V.num_modes() == 4);

  for(auto it: V.loop_nest()) {
    auto blockid = internal::translate_blockid(it, V());
    if(!V.is_non_zero(blockid)) continue;

    TAMM_SIZE      size = V.block_size(blockid);
    std::vector<T> buf(size);
    V.get(blockid, buf);

    auto block_dims   = V.block_dims(blockid);
    auto block_offset = V.block_offsets(blockid);

    size_t c = 0;

    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
        for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
          for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++, c++) {
            if(nonredundant(i, j, k, l, sym) && nonzero(buf[c]))
              file << std::setw(16) << buf[c] << std::setw(6) << offset + i + 1 << std::setw(4)
                   << offset + j + 1 << std::setw(4) << offset + k + 1 << std::setw(4)
                   << offset + l + 1 << std::endl;
          }
        }
      }
    }
  }
}

template<typename T>
void write_1el_ints(std::ofstream& file, Tensor<T> h, bool symmetric = true) {
  EXPECTS(h.num_modes() == 2);

  for(auto it: h.loop_nest()) {
    auto blockid = internal::translate_blockid(it, h());
    if(!h.is_non_zero(blockid)) continue;

    TAMM_SIZE      size = h.block_size(blockid);
    std::vector<T> buf(size);
    h.get(blockid, buf);

    auto block_dims   = h.block_dims(blockid);
    auto block_offset = h.block_offsets(blockid);

    size_t c = 0;

    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++, c++) {
        if((i <= j || !symmetric) && nonzero(buf[c])) {
          file << std::setw(16) << buf[c] << std::setw(6) << i + 1 << std::setw(4) << j + 1
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
void writeFCIDUMPfile(Tensor<T> V, Tensor<T> h, T nuc_rep, int norbs, int nelec, int spin,
                      std::vector<int> orbsym, int isym, std::string filename,
                      TwoElSym sym = None) {
  ExecutionContext& ec = get_ec(V());
  EXPECTS(V.num_modes() == 4);
  EXPECTS(h.num_modes() == 2);

  if(ec.pg().rank() == 0) {
    std::ofstream file(filename);

    auto orbsym_str = symmetry_string(orbsym);
    if(orbsym_str.empty()) orbsym_str = ",";

    file << "&FCI NORB=" << norbs << ",NELEC=" << nelec << ",MS2=" << spin
         << ",\nORBSYM=" << orbsym_str << "\nISYM=" << isym << std::endl
         << "&END" << std::endl;

    file << std::fixed << std::setprecision(10);

    write_2el_ints(file, V, sym);
    write_1el_ints(file, h);
    file << std::setw(16) << nuc_rep << "     " << 0 << "   " << 0 << "   " << 0 << "   " << 0
         << std::endl;

    std::cout << std::endl << "Written file:  " << filename << std::endl;
    file.close();
  }
}

template<typename T>
void writeFCIDUMPfileUHF(Tensor<T> Vaaaa_inp, Tensor<T> Vbbbb, Tensor<T> Vaabb, Tensor<T> haa,
                         Tensor<T> hbb, T nuc_rep, int norbs, int nelec, int spin,
                         std::vector<int> orbsym, int isym, std::string filename,
                         TwoElSym sym = None) {
  ExecutionContext& ec = get_ec(Vaaaa_inp());
  EXPECTS(Vaaaa_inp.num_modes() == 4);
  EXPECTS(Vbbbb.num_modes() == 4);
  EXPECTS(Vaabb.num_modes() == 4);
  EXPECTS(haa.num_modes() == 2);
  EXPECTS(hbb.num_modes() == 2);

  ExecutionContext ec_dense{ec.pg(), DistributionKind::dense,
                            MemoryManagerKind::ga}; // create ec_dense once
  Tensor<T>        Vaaaa = to_dense_tensor(ec_dense, Vaaaa_inp);

  if(ec.pg().rank() == 0) {
    std::ofstream file(filename);

    auto orbsym_str = symmetry_string(orbsym);
    if(orbsym_str.empty()) orbsym_str = ",";

    file << "&FCI NORB=" << norbs << ",NELEC=" << nelec << ",MS2=" << spin
         << ",\nORBSYM=" << orbsym_str << "\nISYM=" << isym << ",IUHF=1" << std::endl
         << "&END" << std::endl;

    file << std::fixed << std::setprecision(10);

    write_2el_ints(file, Vaaaa, sym);
    file << std::setw(16) << 0.0 << "     " << 0 << "   " << 0 << "   " << 0 << "   " << 0
         << std::endl;
    write_2el_ints(file, Vbbbb, sym);
    file << std::setw(16) << 0.0 << "     " << 0 << "   " << 0 << "   " << 0 << "   " << 0
         << std::endl;
    write_2el_ints(file, Vaabb, sym);
    file << std::setw(16) << 0.0 << "     " << 0 << "   " << 0 << "   " << 0 << "   " << 0
         << std::endl;
    write_1el_ints(file, haa);
    file << std::setw(16) << 0.0 << "     " << 0 << "   " << 0 << "   " << 0 << "   " << 0
         << std::endl;
    write_1el_ints(file, hbb);
    file << std::setw(16) << nuc_rep << "     " << 0 << "   " << 0 << "   " << 0 << "   " << 0
         << std::endl;

    std::cout << std::endl << "Integral file written to: " << filename << std::endl;
    file.close();
  }
  Tensor<T>::deallocate(Vaaaa);
}

// Nasty way of doing this, kinda
template<typename T>
UHF_ints<T> FCIDUMPintsfromDUCC(ExecutionContext& ec, DUCC_ints<T>& ints, int nfrzc) {
  using Tensor4D = Eigen::Tensor<T, 4, Eigen::RowMajor>;
  using Tensor2D = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  Scheduler sch{ec};
  auto      hw       = ec.exhw();
  int       npart    = ints.npart;
  int       nhole    = ints.nhole;
  int       norb     = npart + nhole;
  Tile      tilesize = 50; // TODO

  if(ec.print()) std::cout << "Preparing downfolded Hamiltonian for FCIDUMP ... ";

  IndexSpace MO_IS{range(0, 2 * norb),
                   {{"alpha", {range(0, norb)}},
                    {"beta", {range(norb, 2 * norb)}},
                    {"occ", {range(0, npart), range(norb, norb + npart)}},
                    {"occ_alpha", {range(0, npart)}},
                    {"occ_beta", {range(norb, norb + npart)}},
                    {"frzc", {range(0, nfrzc), range(norb, norb + nfrzc)}},
                    {"frzc_alpha", {range(0, nfrzc)}},
                    {"frzc_beta", {range(norb, norb + nfrzc)}},
                    {"part", {range(nfrzc, npart), range(norb + nfrzc, norb + npart)}},
                    {"part_alpha", {range(nfrzc, npart)}},
                    {"part_beta", {range(norb + nfrzc, norb + npart)}},
                    {"hole_alpha", {range(npart, norb)}},
                    {"hole_beta", {range(norb + npart, 2 * norb)}},
                    {"hole", {range(npart, norb), range(norb + npart, 2 * norb)}},
                    {"act", {range(nfrzc, norb), range(norb + nfrzc, 2 * norb)}},
                    {"act_alpha", {range(nfrzc, norb)}},
                    {"act_beta", {range(norb + nfrzc, 2 * norb)}}}};

  TiledIndexSpace MO{MO_IS, tilesize};
  TiledIndexSpace OCC = MO("occ");
  TiledIndexSpace ACT = MO("part");
  TiledIndexSpace H   = MO("hole");
  TiledIndexSpace HA  = MO("hole_alpha");
  TiledIndexSpace HB  = MO("hole_beta");
  TiledIndexSpace OA  = MO("occ_alpha");
  TiledIndexSpace OB  = MO("occ_beta");
  TiledIndexSpace A   = MO("act_alpha");
  TiledIndexSpace B   = MO("act_beta");

  auto [o1, o2, o3, o4]         = MO.labels<4>("part");
  auto [oa1, oa2, oa3, oa4]     = MO.labels<4>("part_alpha");
  auto [oca1, oca2, oca3, oca4] = MO.labels<4>("occ_alpha");
  auto [ob1, ob2, ob3, ob4]     = MO.labels<4>("part_beta");
  auto [ocb1, ocb2, ocb3, ocb4] = MO.labels<4>("occ_beta");

  auto [f1, f2, f3]    = MO.labels<3>("frzc");
  auto [fa1, fa2, fa3] = MO.labels<3>("frzc_alpha");
  auto [fb1, fb2, fb3] = MO.labels<3>("frzc_beta");

  auto [v1, v2, v3]         = MO.labels<3>("hole");
  auto [va1, va2, va3, va4] = MO.labels<4>("hole_alpha");
  auto [vb1, vb2, vb3, vb4] = MO.labels<4>("hole_beta");

  Tensor<T> Enuc{}, Hij{OCC, OCC}, Hia{OCC, H}, Hab{H, H}, Vijkl{OCC, OCC, OCC, OCC},
    Vijka{OCC, OCC, OCC, H}, Vijab{OCC, OCC, H, H}, Vaijb{H, OCC, OCC, H}, Viabc{OCC, H, H, H},
    Vabcd{H, H, H, H}, Viajb_orig{OB, HA, OB, HA}, Viajb_aaaa{OA, HA, OA, HA},
    Viajb_bbbb{OB, HB, OB, HB}, Viajb_aabb{OA, HB, OA, HB}, Vaijb_orig{HB, OA, OB, HA},
    Vaijb_aaaa{HA, OA, OA, HA}, Vaijb_bbbb{HB, OB, OB, HB}, Vaijb_aabb{HA, OB, OA, HB}, Haa{A, A},
    Hbb{B, B}, Vaaaa{A, A, A, A}, Vbbbb{B, B, B, B}, Vaabb{A, A, B, B};

  sch
    .allocate(Enuc, Hij, Hia, Hab, Vijkl, Vijka, Vijab, Vaijb, Viabc, Vabcd, Viajb_orig, Viajb_aaaa,
              Viajb_bbbb, Viajb_aabb, Vaijb_orig, Vaijb_aaaa, Vaijb_bbbb, Vaijb_aabb, Haa, Hbb,
              Vaaaa, Vbbbb, Vaabb)
    .execute(hw);

  ec.pg().barrier();

  if(ec.pg().rank() == 0) {
    Tensor2D Hij_tmp(2 * npart, 2 * npart);
    Tensor2D Hia_tmp(2 * npart, 2 * nhole);
    Tensor2D Hab_tmp(2 * nhole, 2 * nhole);

    Tensor4D Vijkl_tmp(2 * npart, 2 * npart, 2 * npart, 2 * npart);
    Tensor4D Vijka_tmp(2 * npart, 2 * npart, 2 * npart, 2 * nhole);
    Tensor4D Vijab_tmp(2 * npart, 2 * npart, 2 * nhole, 2 * nhole);
    Tensor4D Vaijb_tmp(2 * nhole, 2 * npart, 2 * npart, 2 * nhole);
    Tensor4D Viabc_tmp(2 * npart, 2 * nhole, 2 * nhole, 2 * nhole);
    Tensor4D Vabcd_tmp(2 * nhole, 2 * nhole, 2 * nhole, 2 * nhole);

    tamm_to_eigen_tensor(ints.hij, Hij_tmp);
    eigen_to_tamm_tensor(Hij, Hij_tmp);
    tamm_to_eigen_tensor(ints.hia, Hia_tmp);
    eigen_to_tamm_tensor(Hia, Hia_tmp);
    tamm_to_eigen_tensor(ints.hab, Hab_tmp);
    eigen_to_tamm_tensor(Hab, Hab_tmp);

    tamm_to_eigen_tensor(ints.Vijkl, Vijkl_tmp);
    eigen_to_tamm_tensor(Vijkl, Vijkl_tmp);
    tamm_to_eigen_tensor(ints.Vijka, Vijka_tmp);
    eigen_to_tamm_tensor(Vijka, Vijka_tmp);
    tamm_to_eigen_tensor(ints.Vijab, Vijab_tmp);
    eigen_to_tamm_tensor(Vijab, Vijab_tmp);
    tamm_to_eigen_tensor(ints.Vaijb, Vaijb_tmp);
    eigen_to_tamm_tensor(Vaijb, Vaijb_tmp);
    tamm_to_eigen_tensor(ints.Viabc, Viabc_tmp);
    eigen_to_tamm_tensor(Viabc, Viabc_tmp);
    tamm_to_eigen_tensor(ints.Vabcd, Vabcd_tmp);
    eigen_to_tamm_tensor(Vabcd, Vabcd_tmp);
  }

  ec.pg().barrier();

  sch(Viajb_orig(ocb1, va1, ocb2, va2) = -1 * Vaijb(va1, ocb1, ocb2, va2))(
    Vaijb_orig(vb1, oca1, ocb1, va1) = Vaijb(vb1, oca1, ocb1, va1))

    .execute(hw);

  ec.pg().barrier();

  if(ec.pg().rank() == 0) {
    Tensor4D Viajb_tmp(npart, nhole, npart, nhole);
    Tensor4D Vaijb_tmp(nhole, npart, npart, nhole);

    tamm_to_eigen_tensor(Viajb_orig, Viajb_tmp);
    tamm_to_eigen_tensor(Vaijb_orig, Vaijb_tmp);

    eigen_to_tamm_tensor(Viajb_aaaa, Viajb_tmp);
    eigen_to_tamm_tensor(Viajb_bbbb, Viajb_tmp);
    eigen_to_tamm_tensor(Viajb_aabb, Viajb_tmp);

    eigen_to_tamm_tensor(Vaijb_aaaa, Vaijb_tmp);
    eigen_to_tamm_tensor(Vaijb_bbbb, Vaijb_tmp);
    eigen_to_tamm_tensor(Vaijb_aabb, Vaijb_tmp);
  }

  ec.pg().barrier();

  // clang-format off

  sch
    (Enuc() = 0.0)
    (Haa(oa1, oa2) = Hij(oa1, oa2))
    (Haa(oa1, va1) = Hia(oa1, va1))
    (Haa(va1, va2) = Hab(va1, va2))
          
    (Hbb(ob1, ob2) = Hij(ob1, ob2))
    (Hbb(ob1, vb1) = Hia(ob1, vb1))
    (Hbb(vb1, vb2) = Hab(vb1, vb2))
    .execute(hw);
      
  if(nfrzc > 0) {
    sch
      (Haa(oa1, oa2) += Vijkl(oa1, f1, oa2, f1))
      (Haa(oa1, oa2) -= Vijkl(oa1, oa2, fa1, fa1))
      
      (Haa(oa1, va1) += Vijka(f1, oa1, f1, va1))
      (Haa(oa1, va1) -= Vijka(fa1, fa1, oa1, va1))
      (Haa(va1, oa1) += Vijka(f1, oa1, f1, va1))
      (Haa(va1, oa1) -= Vijka( fa1, fa1, oa1, va1)).execute(hw);

     sch
        (Haa(va1, va2) += Viajb_aaaa(fa1, va1, fa1, va2))
        (Haa(va1, va2) += Viajb_orig(fb1, va1, fb1, va2))
        (Haa(va1, va2) -= Vaijb_aaaa(va1, fa1, fa1, va2))
        
        (Hbb(ob1, ob2) += Vijkl(ob1, f1, ob2, f1))
        (Hbb(ob1, ob2) -= Vijkl(ob1, ob2, fb1, fb1))

        (Hbb(ob1, vb1) += Vijka(f1, ob1, f1, vb1))
        (Hbb(ob1, vb1) -= Vijka(fb1, fb1, ob1, vb1))
        (Hbb(vb1, ob1) += Vijka(f1, ob1, f1, vb1))
        (Hbb(vb1, ob1) -= Vijka(fb1, fb1, ob1, vb1))
        
        (Hbb(vb1, vb2) += Viajb_bbbb(fb1, vb1, fb1, vb2))
        (Hbb(vb1, vb2) += Viajb_aabb(fa1, vb1, fa1, vb2))
        (Hbb(vb1, vb2) -= Vaijb_bbbb(vb1, fb1, fb1, vb2))
 
        (Enuc() += Hij(f1,f1))
        (Enuc() += Vijkl(fa1,fb2,fa1,fb2))
        (Enuc() += 0.5*Vijkl(fa1,fa2,fa1,fa2))
        (Enuc() += 0.5*Vijkl(fb1,fb2,fb1,fb2))
        (Enuc() -= 0.5*Vijkl(fa1,fa1,fa2,fa2))
        (Enuc() -= 0.5*Vijkl(fb1,fb1,fb2,fb2))
        
        .execute(hw);
  }

  sch
     (Vaaaa(oa1, oa2, oa3, oa4)  = Vijkl(oa1, oa3, oa2, oa4))
     (Vbbbb(ob1, ob2, ob3, ob4)  = Vijkl(ob1, ob3, ob2, ob4))
     (Vaabb(oa1, oa2, ob1, ob2)  = Vijkl(oa1, ob1, oa2, ob2))

     (Vaaaa(oa1, oa2, oa3, va1)  = Vijka(oa1, oa3, oa2, va1))
     (Vaaaa(oa2, oa1, va1, oa3)  = Vijka(oa1, oa3, oa2, va1))
     (Vaaaa(oa3, va1, oa1, oa2)  = Vijka(oa1, oa3, oa2, va1))
     (Vaaaa(va1, oa3, oa2, oa1)  = Vijka(oa1, oa3, oa2, va1))

     (Vbbbb(ob1, ob2, ob3, vb1)  = Vijka(ob1, ob3, ob2, vb1))
     (Vbbbb(ob2, ob1, vb1, ob3)  = Vijka(ob1, ob3, ob2, vb1))
     (Vbbbb(ob3, vb1, ob1, ob2)  = Vijka(ob1, ob3, ob2, vb1))
     (Vbbbb(vb1, ob3, ob2, ob1)  = Vijka(ob1, ob3, ob2, vb1))

     (Vaabb(oa1, oa2, ob3, vb1)  = Vijka(oa1, ob3, oa2, vb1))
     (Vaabb(oa2, oa1, vb1, ob3)  = Vijka(oa1, ob3, oa2, vb1))
     (Vaabb(oa3, va1, ob1, ob2)  = Vijka(ob1, oa3, ob2, va1))
     (Vaabb(va1, oa3, ob2, ob1)  = Vijka(ob1, oa3, ob2, va1))
 
     (Vaaaa(va1, oa1, oa2, va2)  = Vaijb_aaaa(va1, oa2, oa1, va2))
     (Vaaaa(va1, oa1, oa2, va2) -= Viajb_aaaa(oa2, va1, oa1, va2))
     (Vaaaa(oa1, va1, va2, oa2)  = Vaijb_aaaa(va1, oa2, oa1, va2))
     (Vaaaa(oa1, va1, va2, oa2) -= Viajb_aaaa(oa2, va1, oa1, va2))
  
     (Vaaaa(va1, oa1, va2, oa2)  = Vijab(oa1, oa2, va1, va2))
     (Vaaaa(oa1, va1, oa2, va2)  = Vijab(oa1, oa2, va1, va2))
     
     (Vaaaa(va1, va2, oa1, oa2)  = Viajb_aaaa(oa1, va1, oa2, va2))
     (Vaaaa(va1, va2, oa1, oa2) -= Vaijb_aaaa(va1, oa1, oa2, va2))
     (Vaaaa(oa1, oa2, va1, va2)  = Viajb_aaaa(oa1, va1, oa2, va2))
     (Vaaaa(oa1, oa2, va1, va2) -= Vaijb_aaaa(va1, oa1, oa2, va2))

     (Vbbbb(vb1, ob1, ob2, vb2)  = Vaijb_bbbb(vb1, ob2, ob1, vb2))
     (Vbbbb(vb1, ob1, ob2, vb2) -= Viajb_bbbb(ob2, vb1, ob1, vb2))
     (Vbbbb(ob1, vb1, vb2, ob2)  = Vaijb_bbbb(vb1, ob2, ob1, vb2))
     (Vbbbb(ob1, vb1, vb2, ob2) -= Viajb_bbbb(ob2, vb1, ob1, vb2))

     (Vbbbb(vb1, ob1, vb2, ob2)  = Vijab(ob1, ob2, vb1, vb2))
     (Vbbbb(ob1, vb1, ob2, vb2)  = Vijab(ob1, ob2, vb1, vb2))
     
     (Vbbbb(vb1, vb2, ob1, ob2)  = Viajb_bbbb(ob1, vb1, ob2, vb2))
     (Vbbbb(vb1, vb2, ob1, ob2) -= Vaijb_bbbb(vb1, ob1, ob2, vb2))
     (Vbbbb(ob1, ob2, vb1, vb2)  = Viajb_bbbb(ob1, vb1, ob2, vb2))
     (Vbbbb(ob1, ob2, vb1, vb2) -= Vaijb_bbbb(vb1, ob1, ob2, vb2))

     (Vaabb(va1, oa1, ob2, vb2)  =      Vaijb(va1, ob2, oa1, vb2))
     (Vaabb(oa1, va1, vb2, ob2)  =      Vaijb(va1, ob2, oa1, vb2))
     (Vaabb(oa1, oa2, vb1, vb2)  = Viajb_aabb(oa1, vb1, oa2, vb2))
     (Vaabb(va1, va2, ob1, ob2)  = Viajb_orig(ob1, va1, ob2, va2))
     (Vaabb(oa1, va1, ob2, vb2)  =      Vijab(oa1, ob2, va1, vb2))
     (Vaabb(va1, oa1, vb2, ob2)  =      Vijab(oa1, ob2, va1, vb2))

     (Vaaaa(oa1, va2, va1, va3)  = Viabc(oa1, va1, va2, va3))
     (Vaaaa(va2, oa1, va3, va1)  = Viabc(oa1, va1, va2, va3))
     (Vaaaa(va1, va3, oa1, va2)  = Viabc(oa1, va1, va2, va3))
     (Vaaaa(va3, va1, va2, oa1)  = Viabc(oa1, va1, va2, va3))

     (Vbbbb(ob1, vb2, vb1, vb3)  = Viabc(ob1, vb1, vb2, vb3))
     (Vbbbb(vb2, ob1, vb3, vb1)  = Viabc(ob1, vb1, vb2, vb3))
     (Vbbbb(vb1, vb3, ob1, vb2)  = Viabc(ob1, vb1, vb2, vb3))
     (Vbbbb(vb3, vb1, vb2, ob1)  = Viabc(ob1, vb1, vb2, vb3))

     (Vaabb(oa1, va2, vb1, vb3)  = Viabc(oa1, vb1, va2, vb3))
     (Vaabb(va2, oa1, vb3, vb1)  = Viabc(oa1, vb1, va2, vb3))
     (Vaabb(va1, va3, ob1, vb2)  = Viabc(ob1, va1, vb2, va3))
     (Vaabb(va3, va1, vb2, ob1)  = Viabc(ob1, va1, vb2, va3))
 
     (Vaaaa(va1, va2, va3, va4)  = Vabcd(va1, va3, va2, va4))
     (Vbbbb(vb1, vb2, vb3, vb4)  = Vabcd(vb1, vb3, vb2, vb4))
     (Vaabb(va1, va2, vb3, vb4)  = Vabcd(va1, vb3, va2, vb4))
     
  .deallocate(Hij,Hia,Hab,Vijkl,Vijka,Vijab,Vaijb,Viabc,Vabcd,
              Viajb_orig,Viajb_aaaa,Viajb_bbbb,Viajb_aabb,
              Vaijb_orig,Vaijb_aaaa,Vaijb_bbbb,Vaijb_aabb)
  .execute(hw);

  // clang-format on

  if(ec.print()) std::cout << "done" << std::endl;
  ec.pg().barrier();

  return {Vaaaa, Vbbbb, Vaabb, Haa, Hbb, get_scalar(Enuc)};
}

} // namespace ducc