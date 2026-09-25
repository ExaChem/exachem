/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, cholattelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_qed_ccsd_os_resid_1.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_os::resid_part1(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<T>& tmps, TensorMap<T>& scalars,
  const TensorMap<T>& f, const TensorMap<T>& chol, const TensorMap<T>& dp, const double w0,
  const TensorMap<T>& t1, const TensorMap<T>& t2, const double t0_1p, const TensorMap<T>& t1_1p,
  const TensorMap<T>& t2_1p, const double t0_2p, const TensorMap<T>& t1_2p,
  const TensorMap<T>& t2_2p, Tensor<T>& energy, TensorMap<T>& r1, TensorMap<T>& r2,
  Tensor<T>& r0_1p, TensorMap<T>& r1_1p, TensorMap<T>& r2_1p, Tensor<T>& r0_2p, TensorMap<T>& r1_2p,
  TensorMap<T>& r2_2p) {
  const TiledIndexSpace& MO      = chem_env.is_context.MSO;
  const TiledIndexSpace& O       = MO("occ");
  const TiledIndexSpace& V       = MO("virt");
  const TiledIndexSpace& CI      = chem_env.is_context.CI;
  const int              otiles  = O.num_tiles();
  const int              vtiles  = V.num_tiles();
  const int              oatiles = MO("occ_alpha").num_tiles();
  const int              vatiles = MO("virt_alpha").num_tiles();

  const TiledIndexSpace Oa = {MO("occ"), range(oatiles)};
  const TiledIndexSpace Va = {MO("virt"), range(vatiles)};
  const TiledIndexSpace Ob = {MO("occ"), range(oatiles, otiles)};
  const TiledIndexSpace Vb = {MO("virt"), range(vatiles, vtiles)};

  TiledIndexLabel aa, ba, ca, da;
  TiledIndexLabel ia, ja, ka, la;
  TiledIndexLabel ab, bb, cb, db;
  TiledIndexLabel ib, jb, kb, lb;
  TiledIndexLabel Q;

  std::tie(aa, ba, ca, da) = Va.labels<4>("all");
  std::tie(ab, bb, cb, db) = Vb.labels<4>("all");
  std::tie(ia, ja, ka, la) = Oa.labels<4>("all");
  std::tie(ib, jb, kb, lb) = Ob.labels<4>("all");
  std::tie(Q)              = CI.labels<1>("all");

  // clang-format off
  {
    sch
    ( scalars.at("0001")()  = dp.at("bb_ov")(ib,ab) * t1.at("bb")(ab,ib) )
    ( scalars.at("0002")()  = dp.at("aa_ov")(ia,aa) * t1.at("aa")(aa,ia) )
    ( scalars.at("0003")()  = scalars.at("0001")() )
    ( scalars.at("0003")() += scalars.at("0002")() )
    ( scalars.at("0004")()  = dp.at("aa_ov")(ia,aa) * t1_1p.at("aa")(aa,ia) )
    ( scalars.at("0005")()  = dp.at("bb_ov")(ib,ab) * t1_1p.at("bb")(ab,ib) )
    ( scalars.at("0006")()  = dp.at("bb_ov")(ib,ab) * t1_2p.at("bb")(ab,ib) )
    ( scalars.at("0007")()  = dp.at("aa_ov")(ia,aa) * t1_2p.at("aa")(aa,ia) )
    ( scalars.at("0008")()  = scalars.at("0006")() )
    ( scalars.at("0008")() += scalars.at("0007")() )
    ( scalars.at("0009")()  = scalars.at("0004")() )
    ( scalars.at("0009")() += scalars.at("0005")() )
    ( scalars.at("0009")()  = scalars.at("0004")() )
    ( scalars.at("0009")() += scalars.at("0005")() )
        
    // r1_2p[bb]  = +4.000 t1_2p_bb(a,i) w0 
    // flops: o1v1  = o1v1
    //  mems: o1v1  = o1v1
    ( r1_2p.at("bb")(ab,ib)  = 4.000 * w0 * t1_2p.at("bb")(ab,ib) )
    
    // r2_1p[bbbb]  = +1.000 t2_1p_bbbb(a,b,i,j) w0 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb)  = w0 * t2_1p.at("bbbb")(ab,bb,ib,jb) )
    
    // r1[bb]  = +1.000 f_bb(a,i) 
    ( r1.at("bb")(ab,ib)  = f.at("bb_vo")(ab,ib) )
    
    // r2_2p[aaaa]  = +4.000 t2_2p_aaaa(a,b,i,j) w0 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja)  = 4.000 * w0 * t2_2p.at("aaaa")(aa,ba,ia,ja) )
    
    // r2_1p[abab]  = +1.000 t2_1p_abab(a,b,i,j) w0 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb)  = w0 * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2_1p[aaaa]  = +1.000 t2_1p_aaaa(a,b,i,j) w0 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja)  = w0 * t2_1p.at("aaaa")(aa,ba,ia,ja) )
    
    // r2_2p[abab]  = +4.000 t2_2p_abab(a,b,i,j) w0 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb)  = 4.000 * w0 * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r2[bbbb]  = +1.000 d-_aa(k,c) t1_aa(c,k) t2_1p_bbbb(a,b,i,j) 
    //            += +1.000 d-_bb(k,c) t1_bb(c,k) t2_1p_bbbb(a,b,i,j) 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb)  = scalars.at("0003")() * t2_1p.at("bbbb")(ab,bb,ib,jb) )
    
    // r2[abab]  = +1.000 d-_aa(a,i) t1_1p_bb(b,j) 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2.at("abab")(aa,bb,ia,jb)  = dp.at("aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    
    // r2[aaaa]  = +1.000 d-_aa(k,c) t1_aa(c,k) t2_1p_aaaa(a,b,i,j) 
    //            += +1.000 d-_bb(k,c) t1_bb(c,k) t2_1p_aaaa(a,b,i,j) 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja)  = scalars.at("0003")() * t2_1p.at("aaaa")(aa,ba,ia,ja) )
    
    // r1_1p[aa]  = +1.000 d+_aa(a,i) 
    ( r1_1p.at("aa")(aa,ia)  = dp.at("aa_vo")(aa,ia) )
    
    // r1_2p[aa]  = +4.000 t1_2p_aa(a,i) w0 
    // flops: o1v1  = o1v1
    //  mems: o1v1  = o1v1
    ( r1_2p.at("aa")(aa,ia)  = 4.000 * w0 * t1_2p.at("aa")(aa,ia) )
    
    // r0_1p()  = +1.000 d+_aa(i,a) t1_aa(a,i) 
    //       += +1.000 d+_bb(i,a) t1_bb(a,i) 
    ( r0_1p()  = scalars.at("0003")() )
    
    // r2_2p[bbbb]  = +4.000 t2_2p_bbbb(a,b,i,j) w0 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb)  = 4.000 * w0 * t2_2p.at("bbbb")(ab,bb,ib,jb) )
    
    // r0_2p()  = +2.000 d+_aa(i,a) t1_1p_aa(a,i) 
    //       += +2.000 d+_bb(i,a) t1_1p_bb(a,i) 
    ( r0_2p()  = 2.000 * scalars.at("0009")() )
    
    // r1_1p[bb]  = +1.000 d+_bb(a,i) 
    ( r1_1p.at("bb")(ab,ib)  = dp.at("bb_vo")(ab,ib) )
    
    // r1[aa]  = +1.000 f_aa(a,i) 
    ( r1.at("aa")(aa,ia)  = f.at("aa_vo")(aa,ia) )
    
    // r0_1p() += +2.000 d-_aa(i,a) t1_2p_aa(a,i) 
    //       += +2.000 d-_bb(i,a) t1_2p_bb(a,i) 
    ( r0_1p() += 2.000 * scalars.at("0008")() )
    
    // r1[aa] += +1.000 d-_aa(a,i) t0_1p 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += t0_1p * dp.at("aa_vo")(aa,ia) )
    
    // r1[aa] += +1.000 d-_aa(j,b) t1_1p_aa(a,i) t1_aa(b,j) 
    //          += +1.000 d-_bb(j,b) t1_1p_aa(a,i) t1_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += scalars.at("0003")() * t1_1p.at("aa")(aa,ia) )
    
    // r1[aa] += -1.000 f_aa(j,i) t1_aa(a,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= f.at("aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    
    // r1[aa] += +1.000 f_aa(a,b) t1_aa(b,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += f.at("aa_vv")(aa,ba) * t1.at("aa")(ba,ia) )
    
    // r1[aa] += -1.000 f_aa(j,b) t2_aaaa(b,a,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= f.at("aa_ov")(ja,ba) * t2.at("aaaa")(ba,aa,ia,ja) )
    
    // r1[aa] += +1.000 f_bb(j,b) t2_abab(a,b,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += f.at("bb_ov")(jb,bb) * t2.at("abab")(aa,bb,ia,jb) )
    
    // r1[bb] += +1.000 d-_bb(a,i) t0_1p 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += t0_1p * dp.at("bb_vo")(ab,ib) )
    
    // r1[bb] += +1.000 d-_aa(j,b) t1_1p_bb(a,i) t1_aa(b,j) 
    //          += +1.000 d-_bb(j,b) t1_1p_bb(a,i) t1_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += scalars.at("0003")() * t1_1p.at("bb")(ab,ib) )
    
    // r1[bb] += -1.000 f_bb(j,i) t1_bb(a,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= f.at("bb_oo")(jb,ib) * t1.at("bb")(ab,jb) )
    
    // r1[bb] += +1.000 f_bb(a,b) t1_bb(b,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += f.at("bb_vv")(ab,bb) * t1.at("bb")(bb,ib) )
    
    // r1[bb] += +1.000 f_aa(j,b) t2_abab(b,a,j,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += f.at("aa_ov")(ja,ba) * t2.at("abab")(ba,ab,ja,ib) )
    
    // r1[bb] += -1.000 f_bb(j,b) t2_bbbb(b,a,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= f.at("bb_ov")(jb,bb) * t2.at("bbbb")(bb,ab,ib,jb) )
    
    // r1_1p[aa] += +1.000 t1_1p_aa(a,i) w0 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += w0 * t1_1p.at("aa")(aa,ia) )
    
    // r1_1p[aa] += +2.000 d-_aa(a,i) t0_2p 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 2.000 * t0_2p * dp.at("aa_vo")(aa,ia) )
    
    // r1_1p[aa] += +2.000 d-_aa(j,b) t1_2p_aa(a,i) t1_aa(b,j) 
    //             += +2.000 d-_bb(j,b) t1_2p_aa(a,i) t1_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 2.000 * scalars.at("0003")() * t1_2p.at("aa")(aa,ia) )
    
    // r1_1p[aa] += +1.000 d-_aa(j,b) t1_1p_aa(a,i) t1_1p_aa(b,j) 
    //             += +1.000 d-_bb(j,b) t1_1p_aa(a,i) t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += scalars.at("0009")() * t1_1p.at("aa")(aa,ia) )
    
    // r1_1p[aa] += -1.000 f_aa(j,i) t1_1p_aa(a,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= f.at("aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    
    // r1_1p[aa] += +1.000 f_aa(a,b) t1_1p_aa(b,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += f.at("aa_vv")(aa,ba) * t1_1p.at("aa")(ba,ia) )
    
    // r1_1p[aa] += -1.000 f_aa(j,b) t2_1p_aaaa(b,a,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= f.at("aa_ov")(ja,ba) * t2_1p.at("aaaa")(ba,aa,ia,ja) )
    
    // r1_1p[aa] += +1.000 f_bb(j,b) t2_1p_abab(a,b,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += f.at("bb_ov")(jb,bb) * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r1_1p[bb] += +1.000 t1_1p_bb(a,i) w0 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += w0 * t1_1p.at("bb")(ab,ib) )
    
    // r1_1p[bb] += +2.000 d-_bb(a,i) t0_2p 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += 2.000 * t0_2p * dp.at("bb_vo")(ab,ib) )
    
    // r1_1p[bb] += +2.000 d-_aa(j,b) t1_2p_bb(a,i) t1_aa(b,j) 
    //             += +2.000 d-_bb(j,b) t1_2p_bb(a,i) t1_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += 2.000 * scalars.at("0003")() * t1_2p.at("bb")(ab,ib) )
    
    // r1_1p[bb] += +1.000 d-_aa(j,b) t1_1p_bb(a,i) t1_1p_aa(b,j) 
    //             += +1.000 d-_bb(j,b) t1_1p_bb(a,i) t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += scalars.at("0009")() * t1_1p.at("bb")(ab,ib) )
    
    // r1_1p[bb] += -1.000 f_bb(j,i) t1_1p_bb(a,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= f.at("bb_oo")(jb,ib) * t1_1p.at("bb")(ab,jb) )
    
    // r1_1p[bb] += +1.000 f_bb(a,b) t1_1p_bb(b,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += f.at("bb_vv")(ab,bb) * t1_1p.at("bb")(bb,ib) )
    
    // r1_1p[bb] += +1.000 f_aa(j,b) t2_1p_abab(b,a,j,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += f.at("aa_ov")(ja,ba) * t2_1p.at("abab")(ba,ab,ja,ib) )
    
    // r1_1p[bb] += -1.000 f_bb(j,b) t2_1p_bbbb(b,a,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= f.at("bb_ov")(jb,bb) * t2_1p.at("bbbb")(bb,ab,ib,jb) )
    
    // r1_2p[aa] += +2.000 d-_aa(j,b) t1_1p_aa(a,i) t1_2p_aa(b,j) 
    //             += +2.000 d-_bb(j,b) t1_1p_aa(a,i) t1_2p_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * scalars.at("0008")() * t1_1p.at("aa")(aa,ia) )
    
    // r1_2p[aa] += +4.000 d-_aa(j,b) t1_2p_aa(a,i) t1_1p_aa(b,j) 
    //             += +4.000 d-_bb(j,b) t1_2p_aa(a,i) t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 4.000 * scalars.at("0009")() * t1_2p.at("aa")(aa,ia) )
    
    // r1_2p[aa] += -2.000 f_aa(j,i) t1_2p_aa(a,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * f.at("aa_oo")(ja,ia) * t1_2p.at("aa")(aa,ja) )
    
    // r1_2p[aa] += +2.000 f_aa(a,b) t1_2p_aa(b,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * f.at("aa_vv")(aa,ba) * t1_2p.at("aa")(ba,ia) )
    
    // r1_2p[aa] += -2.000 f_aa(j,b) t2_2p_aaaa(b,a,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * f.at("aa_ov")(ja,ba) * t2_2p.at("aaaa")(ba,aa,ia,ja) )
    
    // r1_2p[aa] += +2.000 f_bb(j,b) t2_2p_abab(a,b,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * f.at("bb_ov")(jb,bb) * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r1_2p[bb] += +2.000 d-_aa(j,b) t1_1p_bb(a,i) t1_2p_aa(b,j) 
    //             += +2.000 d-_bb(j,b) t1_1p_bb(a,i) t1_2p_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * scalars.at("0008")() * t1_1p.at("bb")(ab,ib) )
    
    // r1_2p[bb] += +4.000 d-_aa(j,b) t1_2p_bb(a,i) t1_1p_aa(b,j) 
    //             += +4.000 d-_bb(j,b) t1_2p_bb(a,i) t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 4.000 * scalars.at("0009")() * t1_2p.at("bb")(ab,ib) )
    
    // r1_2p[bb] += -2.000 f_bb(j,i) t1_2p_bb(a,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * f.at("bb_oo")(jb,ib) * t1_2p.at("bb")(ab,jb) )
    
    // r1_2p[bb] += +2.000 f_bb(a,b) t1_2p_bb(b,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * f.at("bb_vv")(ab,bb) * t1_2p.at("bb")(bb,ib) )
    
    // r1_2p[bb] += +2.000 f_aa(j,b) t2_2p_abab(b,a,j,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * f.at("aa_ov")(ja,ba) * t2_2p.at("abab")(ba,ab,ja,ib) )
    
    // r1_2p[bb] += -2.000 f_bb(j,b) t2_2p_bbbb(b,a,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * f.at("bb_ov")(jb,bb) * t2_2p.at("bbbb")(bb,ab,ib,jb) )
    
    // r2[abab] += +1.000 d-_bb(b,j) t1_1p_aa(a,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,ia) * dp.at("bb_vo")(bb,jb) )
    
    // r2[abab] += +1.000 d-_aa(k,c) t1_aa(c,k) t2_1p_abab(a,b,i,j) 
    //            += +1.000 d-_bb(k,c) t1_bb(c,k) t2_1p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += scalars.at("0003")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2[abab] += -1.000 f_aa(k,i) t2_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= f.at("aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2[abab] += -1.000 f_bb(k,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,kb) * f.at("bb_oo")(kb,jb) )
    
    // r2[abab] += +1.000 f_aa(a,c) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += f.at("aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2[abab] += +1.000 f_bb(b,c) t2_abab(a,c,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,cb,ia,jb) * f.at("bb_vv")(bb,cb) )
    
    // r2_1p[aaaa] += +2.000 d-_aa(k,c) t1_aa(c,k) t2_2p_aaaa(a,b,i,j) 
    //               += +2.000 d-_bb(k,c) t1_bb(c,k) t2_2p_aaaa(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 2.000 * scalars.at("0003")() * t2_2p.at("aaaa")(aa,ba,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 d-_aa(k,c) t1_1p_aa(c,k) t2_1p_aaaa(a,b,i,j) 
    //               += +1.000 d-_bb(k,c) t1_1p_bb(c,k) t2_1p_aaaa(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += scalars.at("0009")() * t2_1p.at("aaaa")(aa,ba,ia,ja) )
    
    // r2_1p[abab] += +2.000 d-_aa(a,i) t1_2p_bb(b,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * dp.at("aa_vo")(aa,ia) * t1_2p.at("bb")(bb,jb) )
    
    // r2_1p[abab] += +2.000 d-_bb(b,j) t1_2p_aa(a,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_2p.at("aa")(aa,ia) * dp.at("bb_vo")(bb,jb) )
    
    // r2_1p[abab] += +2.000 d-_aa(k,c) t1_aa(c,k) t2_2p_abab(a,b,i,j) 
    //               += +2.000 d-_bb(k,c) t1_bb(c,k) t2_2p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * scalars.at("0003")() * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r2_1p[abab] += +1.000 d-_aa(k,c) t1_1p_aa(c,k) t2_1p_abab(a,b,i,j) 
    //               += +1.000 d-_bb(k,c) t1_1p_bb(c,k) t2_1p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += scalars.at("0009")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2_1p[abab] += -1.000 f_aa(k,i) t2_1p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= f.at("aa_oo")(ka,ia) * t2_1p.at("abab")(aa,bb,ka,jb) )
    
    // r2_1p[abab] += -1.000 f_bb(k,j) t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,bb,ia,kb) * f.at("bb_oo")(kb,jb) )
    
    // r2_1p[abab] += +1.000 f_aa(a,c) t2_1p_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += f.at("aa_vv")(aa,ca) * t2_1p.at("abab")(ca,bb,ia,jb) )
    
    // r2_1p[abab] += +1.000 f_bb(b,c) t2_1p_abab(a,c,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,cb,ia,jb) * f.at("bb_vv")(bb,cb) )
    
    // r2_1p[bbbb] += +2.000 d-_aa(k,c) t1_aa(c,k) t2_2p_bbbb(a,b,i,j) 
    //               += +2.000 d-_bb(k,c) t1_bb(c,k) t2_2p_bbbb(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 2.000 * scalars.at("0003")() * t2_2p.at("bbbb")(ab,bb,ib,jb) )
    
    // r2_1p[bbbb] += +1.000 d-_aa(k,c) t1_1p_aa(c,k) t2_1p_bbbb(a,b,i,j) 
    //               += +1.000 d-_bb(k,c) t1_1p_bb(c,k) t2_1p_bbbb(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += scalars.at("0009")() * t2_1p.at("bbbb")(ab,bb,ib,jb) )
    
    // r2_2p[aaaa] += +2.000 d-_aa(k,c) t1_2p_aa(c,k) t2_1p_aaaa(a,b,i,j) 
    //               += +2.000 d-_bb(k,c) t1_2p_bb(c,k) t2_1p_aaaa(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * scalars.at("0008")() * t2_1p.at("aaaa")(aa,ba,ia,ja) )
    
    // r2_2p[aaaa] += +4.000 d-_aa(k,c) t1_1p_aa(c,k) t2_2p_aaaa(a,b,i,j) 
    //               += +4.000 d-_bb(k,c) t1_1p_bb(c,k) t2_2p_aaaa(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 4.000 * scalars.at("0009")() * t2_2p.at("aaaa")(aa,ba,ia,ja) )
    
    // r2_2p[abab] += +2.000 d-_aa(k,c) t1_2p_aa(c,k) t2_1p_abab(a,b,i,j) 
    //               += +2.000 d-_bb(k,c) t1_2p_bb(c,k) t2_1p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * scalars.at("0008")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2_2p[abab] += +4.000 d-_aa(k,c) t1_1p_aa(c,k) t2_2p_abab(a,b,i,j) 
    //               += +4.000 d-_bb(k,c) t1_1p_bb(c,k) t2_2p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.000 * scalars.at("0009")() * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 f_aa(k,i) t2_2p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * f.at("aa_oo")(ka,ia) * t2_2p.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p[abab] += -2.000 f_bb(k,j) t2_2p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_2p.at("abab")(aa,bb,ia,kb) * f.at("bb_oo")(kb,jb) )
    
    // r2_2p[abab] += -2.000 f_aa(k,c) t1_aa(a,k) t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = f.at("aa_ov")(ka,ca) * t2_2p.at("abab")(ca,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -6.000 d-_aa(k,c) t1_2p_aa(a,k) t2_1p_abab(c,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_1p.at("abab")(ca,bb,ia,jb) * dp.at("aa_ov")(ka,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * t1_2p.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -6.000 d-_aa(k,c) t1_1p_aa(a,k) t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(ca,bb,ia,jb) * dp.at("aa_ov")(ka,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * t1_1p.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 f_aa(a,c) t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * f.at("aa_vv")(aa,ca) * t2_2p.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 f_bb(b,c) t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("abab")(aa,cb,ia,jb) * f.at("bb_vv")(bb,cb) )
    
    // r2_2p[abab] += -2.000 f_bb(k,c) t1_bb(b,k) t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = f.at("bb_ov")(kb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -6.000 d-_bb(k,c) t1_2p_bb(b,k) t2_1p_abab(a,c,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1_2p.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -6.000 d-_bb(k,c) t1_1p_bb(b,k) t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p[bbbb] += +2.000 d-_aa(k,c) t1_2p_aa(c,k) t2_1p_bbbb(a,b,i,j) 
    //               += +2.000 d-_bb(k,c) t1_2p_bb(c,k) t2_1p_bbbb(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * scalars.at("0008")() * t2_1p.at("bbbb")(ab,bb,ib,jb) )
    
    // r2_2p[bbbb] += +4.000 d-_aa(k,c) t1_1p_aa(c,k) t2_2p_bbbb(a,b,i,j) 
    //               += +4.000 d-_bb(k,c) t1_1p_bb(c,k) t2_2p_bbbb(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 4.000 * scalars.at("0009")() * t2_2p.at("bbbb")(ab,bb,ib,jb) )
    .allocate(tmps.at("0001_bbbb_vvvv"))
    
    // flops: o0v4  = o0v4Q1
    //  mems: o0v4  = o0v4
    ( tmps.at("0001_bbbb_vvvv")(ab,cb,bb,db)  = chol.at("bb_vvQ")(ab,cb,Q) * chol.at("bb_vvQ")(bb,db,Q) )
    
    // r2[bbbb] += +0.500 <a,b||d,c>_bbbb t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) += 0.500 * tmps.at("0001_bbbb_vvvv")(ab,db,bb,cb) * t2.at("bbbb")(db,cb,ib,jb) )
    
    // r2[bbbb] += +0.500 <a,b||d,c>_bbbb t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) -= 0.500 * tmps.at("0001_bbbb_vvvv")(ab,cb,bb,db) * t2.at("bbbb")(db,cb,ib,jb) )
    
    // r2_1p[bbbb] += +0.500 <a,b||d,c>_bbbb t2_1p_bbbb(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 0.500 * tmps.at("0001_bbbb_vvvv")(ab,db,bb,cb) * t2_1p.at("bbbb")(db,cb,ib,jb) )
    
    // r2_1p[bbbb] += +0.500 <a,b||d,c>_bbbb t2_1p_bbbb(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 0.500 * tmps.at("0001_bbbb_vvvv")(ab,cb,bb,db) * t2_1p.at("bbbb")(db,cb,ib,jb) )
    
    // r2_2p[bbbb] += +1.000 <a,b||d,c>_bbbb t2_2p_bbbb(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0001_bbbb_vvvv")(ab,db,bb,cb) * t2_2p.at("bbbb")(db,cb,ib,jb) )
    
    // r2_2p[bbbb] += +1.000 <a,b||d,c>_bbbb t2_2p_bbbb(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0001_bbbb_vvvv")(ab,cb,bb,db) * t2_2p.at("bbbb")(db,cb,ib,jb) )
    .deallocate(tmps.at("0001_bbbb_vvvv"))
    .allocate(tmps.at("0002_aaaa_vvvv"))
    
    // flops: o0v4  = o0v4Q1
    //  mems: o0v4  = o0v4
    ( tmps.at("0002_aaaa_vvvv")(aa,ca,ba,da)  = chol.at("aa_vvQ")(aa,ca,Q) * chol.at("aa_vvQ")(ba,da,Q) )
    
    // r2[aaaa] += +0.500 <a,b||d,c>_aaaa t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) += 0.500 * tmps.at("0002_aaaa_vvvv")(aa,da,ba,ca) * t2.at("aaaa")(da,ca,ia,ja) )
    
    // r2[aaaa] += +0.500 <a,b||d,c>_aaaa t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) -= 0.500 * tmps.at("0002_aaaa_vvvv")(aa,ca,ba,da) * t2.at("aaaa")(da,ca,ia,ja) )
    
    // r2_1p[aaaa] += +0.500 <a,b||d,c>_aaaa t2_1p_aaaa(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 0.500 * tmps.at("0002_aaaa_vvvv")(aa,da,ba,ca) * t2_1p.at("aaaa")(da,ca,ia,ja) )
    
    // r2_1p[aaaa] += +0.500 <a,b||d,c>_aaaa t2_1p_aaaa(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 0.500 * tmps.at("0002_aaaa_vvvv")(aa,ca,ba,da) * t2_1p.at("aaaa")(da,ca,ia,ja) )
    
    // r2_2p[aaaa] += +1.000 <a,b||d,c>_aaaa t2_2p_aaaa(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0002_aaaa_vvvv")(aa,da,ba,ca) * t2_2p.at("aaaa")(da,ca,ia,ja) )
    
    // r2_2p[aaaa] += +1.000 <a,b||d,c>_aaaa t2_2p_aaaa(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0002_aaaa_vvvv")(aa,ca,ba,da) * t2_2p.at("aaaa")(da,ca,ia,ja) )
    .deallocate(tmps.at("0002_aaaa_vvvv"))
    .allocate(tmps.at("0003_aabb_vvvv"))
    
    // flops: o0v4  = o0v4Q1
    //  mems: o0v4  = o0v4
    ( tmps.at("0003_aabb_vvvv")(aa,da,bb,cb)  = chol.at("aa_vvQ")(aa,da,Q) * chol.at("bb_vvQ")(bb,cb,Q) )
    
    // r2[abab] += +0.500 <a,b||d,c>_abab t2_abab(d,c,i,j) 
    //            += +0.500 <a,b||c,d>_abab t2_abab(c,d,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0003_aabb_vvvv")(aa,da,bb,cb) * t2.at("abab")(da,cb,ia,jb) )
    
    // r2_1p[abab] += +0.500 <a,b||d,c>_abab t2_1p_abab(d,c,i,j) 
    //               += +0.500 <a,b||c,d>_abab t2_1p_abab(c,d,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0003_aabb_vvvv")(aa,da,bb,cb) * t2_1p.at("abab")(da,cb,ia,jb) )
    
    // r2_2p[abab] += +1.000 <a,b||d,c>_abab t2_2p_abab(d,c,i,j) 
    //               += +1.000 <a,b||c,d>_abab t2_2p_abab(c,d,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0003_aabb_vvvv")(aa,da,bb,cb) * t2_2p.at("abab")(da,cb,ia,jb) )
    .deallocate(tmps.at("0003_aabb_vvvv"))
    .allocate(tmps.at("0004_bbbb_ovoo"))
    
    // flops: o3v1  = o1v3Q1 o3v3
    //  mems: o3v1  = o1v3 o3v1
    ( tmps.at("bin1_bbbb_vvvo")(bb,cb,db,kb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("bb_vvQ")(bb,db,Q) )
    ( tmps.at("0004_bbbb_ovoo")(kb,bb,ib,jb)  = tmps.at("bin1_bbbb_vvvo")(bb,cb,db,kb) * t2.at("bbbb")(db,cb,ib,jb) )
    
    // r2[bbbb] += -0.500 P(a,b) <a,k||d,c>_bbbb t1_bb(b,k) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) += 0.500 * t1.at("bb")(ab,kb) * tmps.at("0004_bbbb_ovoo")(kb,bb,ib,jb) )
    
    // r2[bbbb] += -0.500 P(a,b) <a,k||d,c>_bbbb t1_bb(b,k) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) -= 0.500 * t1.at("bb")(bb,kb) * tmps.at("0004_bbbb_ovoo")(kb,ab,ib,jb) )
    
    // r2_1p[bbbb] += -0.500 P(a,b) <a,k||d,c>_bbbb t1_1p_bb(b,k) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 0.500 * t1_1p.at("bb")(ab,kb) * tmps.at("0004_bbbb_ovoo")(kb,bb,ib,jb) )
    
    // r2_1p[bbbb] += -0.500 P(a,b) <a,k||d,c>_bbbb t1_1p_bb(b,k) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 0.500 * t1_1p.at("bb")(bb,kb) * tmps.at("0004_bbbb_ovoo")(kb,ab,ib,jb) )
    
    // r2_2p[bbbb] += -1.000 P(a,b) <a,k||d,c>_bbbb t1_2p_bb(b,k) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += t1_2p.at("bb")(ab,kb) * tmps.at("0004_bbbb_ovoo")(kb,bb,ib,jb) )
    
    // r2_2p[bbbb] += -1.000 P(a,b) <a,k||d,c>_bbbb t1_2p_bb(b,k) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= t1_2p.at("bb")(bb,kb) * tmps.at("0004_bbbb_ovoo")(kb,ab,ib,jb) )
    .deallocate(tmps.at("0004_bbbb_ovoo"))
    .allocate(tmps.at("0005_aaaa_ovoo"))
    
    // flops: o3v1  = o1v3Q1 o3v3
    //  mems: o3v1  = o1v3 o3v1
    ( tmps.at("bin1_aaaa_vvvo")(ba,ca,da,ka)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("aa_vvQ")(ba,da,Q) )
    ( tmps.at("0005_aaaa_ovoo")(ka,ba,ia,ja)  = tmps.at("bin1_aaaa_vvvo")(ba,ca,da,ka) * t2.at("aaaa")(da,ca,ia,ja) )
    
    // r2[aaaa] += -0.500 P(a,b) <a,k||d,c>_aaaa t1_aa(b,k) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) += 0.500 * t1.at("aa")(aa,ka) * tmps.at("0005_aaaa_ovoo")(ka,ba,ia,ja) )
    
    // r2[aaaa] += -0.500 P(a,b) <a,k||d,c>_aaaa t1_aa(b,k) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) -= 0.500 * t1.at("aa")(ba,ka) * tmps.at("0005_aaaa_ovoo")(ka,aa,ia,ja) )
    
    // r2_1p[aaaa] += -0.500 P(a,b) <a,k||d,c>_aaaa t1_1p_aa(b,k) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 0.500 * t1_1p.at("aa")(aa,ka) * tmps.at("0005_aaaa_ovoo")(ka,ba,ia,ja) )
    
    // r2_1p[aaaa] += -0.500 P(a,b) <a,k||d,c>_aaaa t1_1p_aa(b,k) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 0.500 * t1_1p.at("aa")(ba,ka) * tmps.at("0005_aaaa_ovoo")(ka,aa,ia,ja) )
    
    // r2_2p[aaaa] += -1.000 P(a,b) <a,k||d,c>_aaaa t1_2p_aa(b,k) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += t1_2p.at("aa")(aa,ka) * tmps.at("0005_aaaa_ovoo")(ka,ba,ia,ja) )
    
    // r2_2p[aaaa] += -1.000 P(a,b) <a,k||d,c>_aaaa t1_2p_aa(b,k) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= t1_2p.at("aa")(ba,ka) * tmps.at("0005_aaaa_ovoo")(ka,aa,ia,ja) )
    .deallocate(tmps.at("0005_aaaa_ovoo"))
    .allocate(tmps.at("0006_bbbb_ovoo"))
    
    // flops: o3v1  = o1v3Q1 o3v3
    //  mems: o3v1  = o1v3 o3v1
    ( tmps.at("bin1_bbbb_vvvo")(bb,cb,db,kb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("bb_vvQ")(bb,db,Q) )
    ( tmps.at("0006_bbbb_ovoo")(kb,bb,ib,jb)  = tmps.at("bin1_bbbb_vvvo")(bb,cb,db,kb) * t2_1p.at("bbbb")(db,cb,ib,jb) )
    
    // r2_1p[bbbb] += -0.500 P(a,b) <a,k||d,c>_bbbb t1_bb(b,k) t2_1p_bbbb(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 0.500 * t1.at("bb")(ab,kb) * tmps.at("0006_bbbb_ovoo")(kb,bb,ib,jb) )
    
    // r2_1p[bbbb] += -0.500 P(a,b) <a,k||d,c>_bbbb t1_bb(b,k) t2_1p_bbbb(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 0.500 * t1.at("bb")(bb,kb) * tmps.at("0006_bbbb_ovoo")(kb,ab,ib,jb) )
    
    // r2_2p[bbbb] += -1.000 P(a,b) <a,k||d,c>_bbbb t1_1p_bb(b,k) t2_1p_bbbb(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += t1_1p.at("bb")(ab,kb) * tmps.at("0006_bbbb_ovoo")(kb,bb,ib,jb) )
    
    // r2_2p[bbbb] += -1.000 P(a,b) <a,k||d,c>_bbbb t1_1p_bb(b,k) t2_1p_bbbb(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= t1_1p.at("bb")(bb,kb) * tmps.at("0006_bbbb_ovoo")(kb,ab,ib,jb) )
    .deallocate(tmps.at("0006_bbbb_ovoo"))
    .allocate(tmps.at("0007_aaaa_ovoo"))
    
    // flops: o3v1  = o1v3Q1 o3v3
    //  mems: o3v1  = o1v3 o3v1
    ( tmps.at("bin1_aaaa_vvvo")(ba,ca,da,ka)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("aa_vvQ")(ba,da,Q) )
    ( tmps.at("0007_aaaa_ovoo")(ka,ba,ia,ja)  = tmps.at("bin1_aaaa_vvvo")(ba,ca,da,ka) * t2_1p.at("aaaa")(da,ca,ia,ja) )
    
    // r2_1p[aaaa] += -0.500 P(a,b) <a,k||d,c>_aaaa t1_aa(b,k) t2_1p_aaaa(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 0.500 * t1.at("aa")(aa,ka) * tmps.at("0007_aaaa_ovoo")(ka,ba,ia,ja) )
    
    // r2_1p[aaaa] += -0.500 P(a,b) <a,k||d,c>_aaaa t1_aa(b,k) t2_1p_aaaa(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 0.500 * t1.at("aa")(ba,ka) * tmps.at("0007_aaaa_ovoo")(ka,aa,ia,ja) )
    
    // r2_2p[aaaa] += -1.000 P(a,b) <a,k||d,c>_aaaa t1_1p_aa(b,k) t2_1p_aaaa(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += t1_1p.at("aa")(aa,ka) * tmps.at("0007_aaaa_ovoo")(ka,ba,ia,ja) )
    
    // r2_2p[aaaa] += -1.000 P(a,b) <a,k||d,c>_aaaa t1_1p_aa(b,k) t2_1p_aaaa(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= t1_1p.at("aa")(ba,ka) * tmps.at("0007_aaaa_ovoo")(ka,aa,ia,ja) )
    .deallocate(tmps.at("0007_aaaa_ovoo"))
    .allocate(tmps.at("0008_baba_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_bbbb_vooo")(cb,jb,kb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("0008_baba_ovoo")(jb,aa,kb,ia)  = tmps.at("bin1_bbbb_vooo")(cb,jb,kb,lb) * t2.at("abab")(aa,cb,ia,lb) )
    
    // r2[abab] += +1.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0008_baba_ovoo")(jb,aa,kb,ia) * t1.at("bb")(bb,kb) )
    
    // r2_1p[abab] += -1.000 <k,l||j,c>_bbbb t1_1p_bb(b,k) t2_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("bb")(bb,kb) * tmps.at("0008_baba_ovoo")(jb,aa,kb,ia) )
    
    // r2_2p[abab] += -2.000 <k,l||j,c>_bbbb t1_2p_bb(b,k) t2_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_2p.at("bb")(bb,kb) * tmps.at("0008_baba_ovoo")(jb,aa,kb,ia) )
    .deallocate(tmps.at("0008_baba_ovoo"))
    .allocate(tmps.at("0009_baab_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_baab_vooo")(cb,ia,la,kb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("aa_ooQ")(la,ia,Q) )
    ( tmps.at("0009_baab_ovoo")(kb,aa,ia,jb)  = tmps.at("bin1_baab_vooo")(cb,ia,la,kb) * t2.at("abab")(aa,cb,la,jb) )
    
    // r2[abab] += +1.000 <l,k||i,c>_abab t1_bb(b,k) t2_abab(a,c,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0009_baab_ovoo")(kb,aa,ia,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||i,c>_abab t1_1p_bb(b,k) t2_abab(a,c,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("bb")(bb,kb) * tmps.at("0009_baab_ovoo")(kb,aa,ia,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||i,c>_abab t1_2p_bb(b,k) t2_abab(a,c,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_2p.at("bb")(bb,kb) * tmps.at("0009_baab_ovoo")(kb,aa,ia,jb) )
    .deallocate(tmps.at("0009_baab_ovoo"))
    .allocate(tmps.at("0010_bbaa_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aabb_vooo")(ca,ka,jb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("0010_bbaa_ovoo")(jb,bb,ka,ia)  = tmps.at("bin1_aabb_vooo")(ca,ka,jb,lb) * t2.at("abab")(ca,bb,ia,lb) )
    
    // r2[abab] += +1.000 <k,l||c,j>_abab t1_aa(a,k) t2_abab(c,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0010_bbaa_ovoo")(jb,bb,ka,ia) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += +1.000 <k,l||c,j>_abab t1_1p_aa(a,k) t2_abab(c,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0010_bbaa_ovoo")(jb,bb,ka,ia) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||c,j>_abab t1_2p_aa(a,k) t2_abab(c,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0010_bbaa_ovoo")(jb,bb,ka,ia) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0010_bbaa_ovoo"))
    .allocate(tmps.at("0011_abab_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la)  = chol.at("aa_ooQ")(la,ia,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("0011_abab_ovoo")(ia,bb,ka,jb)  = tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la) * t2.at("abab")(ca,bb,la,jb) )
    
    // r2[abab] += +1.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0011_abab_ovoo")(ia,bb,ka,jb) )
    
    // r2_1p[abab] += -1.000 <k,l||i,c>_aaaa t1_1p_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,ka) * tmps.at("0011_abab_ovoo")(ia,bb,ka,jb) )
    
    // r2_2p[abab] += -2.000 <k,l||i,c>_aaaa t1_2p_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_2p.at("aa")(aa,ka) * tmps.at("0011_abab_ovoo")(ia,bb,ka,jb) )
    .deallocate(tmps.at("0011_abab_ovoo"))
    .allocate(tmps.at("0012_baba_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_bbbb_vooo")(cb,jb,kb,lb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0012_baba_ovoo")(kb,aa,jb,ia)  = tmps.at("bin1_bbbb_vooo")(cb,jb,kb,lb) * t2_1p.at("abab")(aa,cb,ia,lb) )
    
    // r2_1p[abab] += +1.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_1p_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0012_baba_ovoo")(kb,aa,jb,ia) )
    
    // r2_2p[abab] += +2.000 <l,k||j,c>_bbbb t1_1p_bb(b,k) t2_1p_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_1p.at("bb")(bb,kb) * tmps.at("0012_baba_ovoo")(kb,aa,jb,ia) )
    .deallocate(tmps.at("0012_baba_ovoo"))
    .allocate(tmps.at("0013_baab_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_baab_vooo")(cb,ia,la,kb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("aa_ooQ")(la,ia,Q) )
    ( tmps.at("0013_baab_ovoo")(kb,aa,ia,jb)  = tmps.at("bin1_baab_vooo")(cb,ia,la,kb) * t2_1p.at("abab")(aa,cb,la,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||i,c>_abab t1_bb(b,k) t2_1p_abab(a,c,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0013_baab_ovoo")(kb,aa,ia,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||i,c>_abab t1_1p_bb(b,k) t2_1p_abab(a,c,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_1p.at("bb")(bb,kb) * tmps.at("0013_baab_ovoo")(kb,aa,ia,jb) )
    .deallocate(tmps.at("0013_baab_ovoo"))
    .allocate(tmps.at("0014_bbaa_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aabb_vooo")(ca,ka,jb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("0014_bbaa_ovoo")(jb,bb,ka,ia)  = tmps.at("bin1_aabb_vooo")(ca,ka,jb,lb) * t2_1p.at("abab")(ca,bb,ia,lb) )
    
    // r2_1p[abab] += +1.000 <k,l||c,j>_abab t1_aa(a,k) t2_1p_abab(c,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0014_bbaa_ovoo")(jb,bb,ka,ia) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||c,j>_abab t1_1p_aa(a,k) t2_1p_abab(c,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0014_bbaa_ovoo")(jb,bb,ka,ia) * t1_1p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0014_bbaa_ovoo"))
    .allocate(tmps.at("0015_bbbb_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0015_bbbb_vvoo")(ab,bb,ib,jb)  = f.at("bb_vv")(ab,cb) * t2.at("bbbb")(cb,bb,ib,jb) )
    
    // r2[bbbb] += +1.000 P(a,b) f_bb(a,c) t2_bbbb(c,b,i,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0015_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2[bbbb] += +1.000 P(a,b) f_bb(a,c) t2_bbbb(c,b,i,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0015_bbbb_vvoo")(bb,ab,ib,jb) )
    .deallocate(tmps.at("0015_bbbb_vvoo"))
    .allocate(tmps.at("0016_bbbb_vovo"))
    
    // flops: o2v2  = o4v0Q1 o4v1 o3v2
    //  mems: o2v2  = o4v0 o3v1 o2v2
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = chol.at("bb_ooQ")(lb,ib,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t1.at("bb")(bb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("0016_bbbb_vovo")(bb,ib,ab,jb)  = tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1.at("bb")(ab,kb) )
    
    // r2[bbbb] += -1.000 <l,k||i,j>_bbbb t1_bb(a,k) t1_bb(b,l) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0016_bbbb_vovo")(bb,ib,ab,jb) )
    
    // r2[bbbb] += -1.000 <l,k||i,j>_bbbb t1_bb(a,k) t1_bb(b,l) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0016_bbbb_vovo")(bb,jb,ab,ib) )
    .deallocate(tmps.at("0016_bbbb_vovo"))
    .allocate(tmps.at("0017_bbbb_vovo"))
    
    // flops: o2v2  = o4v0Q1 o4v1 o3v2
    //  mems: o2v2  = o4v0 o3v1 o2v2
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = chol.at("bb_ooQ")(lb,ib,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t1_1p.at("bb")(bb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("0017_bbbb_vovo")(bb,ib,ab,jb)  = tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1_1p.at("bb")(ab,kb) )
    
    // r2_2p[bbbb] += -2.000 <l,k||i,j>_bbbb t1_1p_bb(a,k) t1_1p_bb(b,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0017_bbbb_vovo")(bb,ib,ab,jb) )
    
    // r2_2p[bbbb] += -2.000 <l,k||i,j>_bbbb t1_1p_bb(a,k) t1_1p_bb(b,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0017_bbbb_vovo")(bb,jb,ab,ib) )
    .deallocate(tmps.at("0017_bbbb_vovo"))
    .allocate(tmps.at("0018_aaaa_vovo"))
    
    // flops: o2v2  = o4v0Q1 o4v1 o3v2
    //  mems: o2v2  = o4v0 o3v1 o2v2
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = chol.at("aa_ooQ")(la,ia,Q) * chol.at("aa_ooQ")(ka,ja,Q) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1.at("aa")(ba,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0018_aaaa_vovo")(ba,ia,aa,ja)  = tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1.at("aa")(aa,ka) )
    
    // r2[aaaa] += -1.000 <l,k||i,j>_aaaa t1_aa(a,k) t1_aa(b,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0018_aaaa_vovo")(ba,ia,aa,ja) )
    
    // r2[aaaa] += -1.000 <l,k||i,j>_aaaa t1_aa(a,k) t1_aa(b,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0018_aaaa_vovo")(ba,ja,aa,ia) )
    .deallocate(tmps.at("0018_aaaa_vovo"))
    .allocate(tmps.at("0019_aaaa_vovo"))
    
    // flops: o2v2  = o4v0Q1 o4v1 o3v2
    //  mems: o2v2  = o4v0 o3v1 o2v2
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = chol.at("aa_ooQ")(la,ia,Q) * chol.at("aa_ooQ")(ka,ja,Q) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1_1p.at("aa")(ba,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0019_aaaa_vovo")(ba,ia,aa,ja)  = tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[aaaa] += -2.000 <l,k||i,j>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0019_aaaa_vovo")(ba,ia,aa,ja) )
    
    // r2_2p[aaaa] += -2.000 <l,k||i,j>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0019_aaaa_vovo")(ba,ja,aa,ia) )
    .deallocate(tmps.at("0019_aaaa_vovo"))
    .allocate(tmps.at("0020_bb_oo"))
    
    // flops: o2v0  = o2v2Q1 o2v1Q1
    //  mems: o2v0  = o1v1Q1 o2v0
    ( tmps.at("bin1_bb_voQ")(bb,ib,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t2.at("abab")(ca,bb,ka,ib) )
    ( tmps.at("0020_bb_oo")(ib,jb)  = tmps.at("bin1_bb_voQ")(bb,ib,Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    
    // r1[bb] += -0.500 <k,j||c,b>_abab t1_bb(a,j) t2_abab(c,b,k,i) 
    //          += -0.500 <k,j||b,c>_abab t1_bb(a,j) t2_abab(b,c,k,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= tmps.at("0020_bb_oo")(ib,jb) * t1.at("bb")(ab,jb) )
    
    // r1_1p[bb] += -0.500 <k,j||c,b>_abab t1_1p_bb(a,j) t2_abab(c,b,k,i) 
    //             += -0.500 <k,j||b,c>_abab t1_1p_bb(a,j) t2_abab(b,c,k,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0020_bb_oo")(ib,jb) * t1_1p.at("bb")(ab,jb) )
    
    // r1_2p[bb] += -1.000 <k,j||c,b>_abab t1_2p_bb(a,j) t2_abab(c,b,k,i) 
    //             += -1.000 <k,j||b,c>_abab t1_2p_bb(a,j) t2_abab(b,c,k,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0020_bb_oo")(ib,jb) * t1_2p.at("bb")(ab,jb) )
    
    // r2[abab] += -0.500 <k,l||d,c>_abab t2_abab(a,b,i,l) t2_abab(d,c,k,j) 
    //            += -0.500 <k,l||c,d>_abab t2_abab(a,b,i,l) t2_abab(c,d,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,lb) * tmps.at("0020_bb_oo")(jb,lb) )
    
    // r2_1p[abab] += -0.500 <k,l||d,c>_abab t2_1p_abab(a,b,i,l) t2_abab(d,c,k,j) 
    //               += -0.500 <k,l||c,d>_abab t2_1p_abab(a,b,i,l) t2_abab(c,d,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0020_bb_oo")(jb,lb) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_2p_abab(a,b,i,l) t2_abab(d,c,k,j) 
    //               += -1.000 <k,l||c,d>_abab t2_2p_abab(a,b,i,l) t2_abab(c,d,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_2p.at("abab")(aa,bb,ia,lb) * tmps.at("0020_bb_oo")(jb,lb) )
    .deallocate(tmps.at("0020_bb_oo"))
    .allocate(tmps.at("0021_bb_oo"))
    
    // flops: o2v0  = o2v2Q1 o2v1Q1
    //  mems: o2v0  = o1v1Q1 o2v0
    ( tmps.at("bin1_bb_voQ")(bb,ib,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t2_1p.at("abab")(ca,bb,ka,ib) )
    ( tmps.at("0021_bb_oo")(ib,jb)  = tmps.at("bin1_bb_voQ")(bb,ib,Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    
    // r1_1p[bb] += -0.500 <k,j||c,b>_abab t1_bb(a,j) t2_1p_abab(c,b,k,i) 
    //             += -0.500 <k,j||b,c>_abab t1_bb(a,j) t2_1p_abab(b,c,k,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0021_bb_oo")(ib,jb) * t1.at("bb")(ab,jb) )
    
    // r1_2p[bb] += -1.000 <k,j||c,b>_abab t1_1p_bb(a,j) t2_1p_abab(c,b,k,i) 
    //             += -1.000 <k,j||b,c>_abab t1_1p_bb(a,j) t2_1p_abab(b,c,k,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0021_bb_oo")(ib,jb) * t1_1p.at("bb")(ab,jb) )
    
    // r2_1p[abab] += -0.500 <l,k||d,c>_abab t2_abab(a,b,i,k) t2_1p_abab(d,c,l,j) 
    //               += -0.500 <l,k||c,d>_abab t2_abab(a,b,i,k) t2_1p_abab(c,d,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,kb) * tmps.at("0021_bb_oo")(jb,kb) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_1p_abab(a,b,i,l) t2_1p_abab(d,c,k,j) 
    //               += -1.000 <k,l||c,d>_abab t2_1p_abab(a,b,i,l) t2_1p_abab(c,d,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0021_bb_oo")(jb,lb) )
    .deallocate(tmps.at("0021_bb_oo"))
    .allocate(tmps.at("0022_abab_ovoo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v1
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1 o0v0Q1 o1v1 o3v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1_1p.at("aa")(ca,la) )
    ( tmps.at("bin1_aa_vo")(da,ka)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("0022_abab_ovoo")(ka,bb,ia,jb)  = t2.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,ka) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t1_1p.at("bb")(cb,lb) )
    ( tmps.at("bin1_aa_vo")(da,ka)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("0022_abab_ovoo")(ka,bb,ia,jb) += t2.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,ka) )
    
    // r2_1p[abab] += +1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_abab(d,b,i,j) 
    //               += -1.000 <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(c,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("0022_abab_ovoo")(ka,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_aaaa t1_1p_aa(a,l) t1_1p_aa(c,k) t2_abab(d,b,i,j) 
    //               += -2.000 <l,k||d,c>_abab t1_1p_aa(a,l) t1_1p_bb(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_1p.at("aa")(aa,la) * tmps.at("0022_abab_ovoo")(la,bb,ia,jb) )
    .deallocate(tmps.at("0022_abab_ovoo"))
    .allocate(tmps.at("0023_abab_ovoo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v1
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1 o0v0Q1 o1v1 o3v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("0023_abab_ovoo")(la,bb,ia,jb)  = t2.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,la) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("0023_abab_ovoo")(la,bb,ia,jb) += t2.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,la) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    //            += -1.000 <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,la) * tmps.at("0023_abab_ovoo")(la,bb,ia,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    //               += -1.000 <l,k||d,c>_abab t1_1p_aa(a,l) t1_bb(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("aa")(aa,la) * tmps.at("0023_abab_ovoo")(la,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_2p_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    //               += -2.000 <l,k||d,c>_abab t1_2p_aa(a,l) t1_bb(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_2p.at("aa")(aa,la) * tmps.at("0023_abab_ovoo")(la,bb,ia,jb) )
    .deallocate(tmps.at("0023_abab_ovoo"))
    .allocate(tmps.at("0024_abab_ovoo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v1
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1 o0v0Q1 o1v1 o3v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("0024_abab_ovoo")(la,bb,ia,jb)  = t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,la) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("0024_abab_ovoo")(la,bb,ia,jb) += t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,la) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    //               += -1.000 <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,la) * tmps.at("0024_abab_ovoo")(la,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    //               += -2.000 <l,k||d,c>_abab t1_1p_aa(a,l) t1_bb(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_1p.at("aa")(aa,la) * tmps.at("0024_abab_ovoo")(la,bb,ia,jb) )
    .deallocate(tmps.at("0024_abab_ovoo"))
    .allocate(tmps.at("0025_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0025_aa_voQ")(ca,ia,Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r1[aa] += -0.500 <j,k||i,b>_abab t2_abab(a,b,j,k) 
    //          += -0.500 <k,j||i,b>_abab t2_abab(a,b,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0025_aa_voQ")(aa,ja,Q) )
    
    // r2[abab] += +1.000 <b,k||j,c>_bbbb t2_abab(a,c,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0025_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0026_bb_ooQ"))
    
    // flops: o2v0Q1  = o2v1Q1
    //  mems: o2v0Q1  = o2v0Q1
    ( tmps.at("0026_bb_ooQ")(jb,ib,Q)  = chol.at("bb_ovQ")(jb,ab,Q) * t1.at("bb")(ab,ib) )
    .allocate(tmps.at("0027_bbaa_oovo"))
    
    // flops: o3v1  = o3v1Q1 o3v1Q1 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1
    ( tmps.at("0027_bbaa_oovo")(kb,jb,aa,ia)  = tmps.at("0025_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0027_bbaa_oovo")(kb,jb,aa,ia) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0026_bb_ooQ")(kb,jb,Q) )
    
    // r2[abab] += +1.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_abab(a,c,i,l) 
    //            += -1.000 <a,k||i,c>_abab t1_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0027_bbaa_oovo")(kb,jb,aa,ia) * t1.at("bb")(bb,kb) )
    
    // r2_1p[abab] += -1.000 <k,l||j,c>_bbbb t1_1p_bb(b,k) t2_abab(a,c,i,l) 
    //               += -1.000 <a,k||i,c>_abab t1_1p_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0027_bbaa_oovo")(kb,jb,aa,ia) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -2.000 <k,l||j,c>_bbbb t1_2p_bb(b,k) t2_abab(a,c,i,l) 
    //               += -2.000 <a,k||i,c>_abab t1_2p_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0027_bbaa_oovo")(kb,jb,aa,ia) * t1_2p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0027_bbaa_oovo"))
    .allocate(tmps.at("0028_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0028_aa_voQ")(ba,ia,Q)  = chol.at("bb_ovQ")(jb,ab,Q) * t2_1p.at("abab")(ba,ab,ia,jb) )
    
    // r1_1p[aa] += -0.500 <j,k||i,b>_abab t2_1p_abab(a,b,j,k) 
    //             += -0.500 <k,j||i,b>_abab t2_1p_abab(a,b,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0028_aa_voQ")(aa,ja,Q) )
    
    // r2_1p[abab] += +1.000 <b,k||j,c>_bbbb t2_1p_abab(a,c,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0028_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0029_bb_ooQ"))
    
    // flops: o2v0Q1  = o2v1Q1
    //  mems: o2v0Q1  = o2v0Q1
    ( tmps.at("0029_bb_ooQ")(jb,ib,Q)  = chol.at("bb_ovQ")(jb,ab,Q) * t1_1p.at("bb")(ab,ib) )
    .allocate(tmps.at("0030_bbaa_oovo"))
    
    // flops: o3v1  = o3v1Q1 o3v1Q1 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1
    ( tmps.at("0030_bbaa_oovo")(kb,jb,aa,ia)  = tmps.at("0028_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0030_bbaa_oovo")(kb,jb,aa,ia) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0029_bb_ooQ")(kb,jb,Q) )
    
    // r2_1p[abab] += +1.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_1p_abab(a,c,i,l) 
    //               += -1.000 <a,k||i,c>_abab t1_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0030_bbaa_oovo")(kb,jb,aa,ia) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||j,c>_bbbb t1_1p_bb(b,k) t2_1p_abab(a,c,i,l) 
    //               += -2.000 <a,k||i,c>_abab t1_1p_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0030_bbaa_oovo")(kb,jb,aa,ia) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0030_bbaa_oovo"))
    .allocate(tmps.at("0031_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0031_aa_oo")(ja,ia)  = dp.at("aa_ov")(ja,ba) * t1_1p.at("aa")(ba,ia) )
    
    // r1_2p[aa] += -6.000 d-_aa(j,b) t1_2p_aa(a,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 6.000 * tmps.at("0031_aa_oo")(ja,ia) * t1_2p.at("aa")(aa,ja) )
    
    // r2_2p[abab] += -6.000 d-_aa(k,c) t1_1p_aa(c,i) t2_2p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * tmps.at("0031_aa_oo")(ka,ia) * t2_2p.at("abab")(aa,bb,ka,jb) )
    .allocate(tmps.at("0032_aaaa_vvoo"))
    
    // flops: o2v2  = o3v2 o2v1 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v0 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0032_aaaa_vvoo")(aa,ba,ia,ja)  = dp.at("aa_oo")(ka,ia) * t2_1p.at("aaaa")(aa,ba,ja,ka) )
    ( tmps.at("bin1_aa_oo")(ia,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("0032_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_oo")(ia,ka) * t2_1p.at("aaaa")(aa,ba,ja,ka) )
    ( tmps.at("0032_aaaa_vvoo")(aa,ba,ia,ja) += t2.at("aaaa")(aa,ba,ja,ka) * tmps.at("0031_aa_oo")(ka,ia) )
    
    // r2[aaaa] += +1.000 P(i,j) d-_aa(k,i) t2_1p_aaaa(a,b,j,k) 
    //            += +1.000 P(i,j) d-_aa(k,c) t1_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //            += +1.000 P(i,j) d-_aa(k,c) t1_1p_aa(c,i) t2_aaaa(a,b,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0032_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2[aaaa] += +1.000 P(i,j) d-_aa(k,i) t2_1p_aaaa(a,b,j,k) 
    //            += +1.000 P(i,j) d-_aa(k,c) t1_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //            += +1.000 P(i,j) d-_aa(k,c) t1_1p_aa(c,i) t2_aaaa(a,b,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0032_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) d-_aa(k,i) t0_1p t2_1p_aaaa(a,b,j,k) 
    //               += +1.000 P(i,j) d-_aa(k,c) t0_1p t1_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //               += +1.000 P(i,j) d-_aa(k,c) t0_1p t1_1p_aa(c,i) t2_aaaa(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= t0_1p * tmps.at("0032_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) d-_aa(k,i) t0_1p t2_1p_aaaa(a,b,j,k) 
    //               += +1.000 P(i,j) d-_aa(k,c) t0_1p t1_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //               += +1.000 P(i,j) d-_aa(k,c) t0_1p t1_1p_aa(c,i) t2_aaaa(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += t0_1p * tmps.at("0032_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) d+_aa(k,i) t2_1p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d+_aa(k,c) t1_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d+_aa(k,c) t1_1p_aa(c,i) t2_aaaa(a,b,j,k) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0032_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) d+_aa(k,i) t2_1p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d+_aa(k,c) t1_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d+_aa(k,c) t1_1p_aa(c,i) t2_aaaa(a,b,j,k) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0032_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_2p[aaaa] += +4.000 P(i,j) d-_aa(k,i) t0_2p t2_1p_aaaa(a,b,j,k) 
    //               += +4.000 P(i,j) d-_aa(k,c) t0_2p t1_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //               += +4.000 P(i,j) d-_aa(k,c) t0_2p t1_1p_aa(c,i) t2_aaaa(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 4.000 * t0_2p * tmps.at("0032_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_2p[aaaa] += +4.000 P(i,j) d-_aa(k,i) t0_2p t2_1p_aaaa(a,b,j,k) 
    //               += +4.000 P(i,j) d-_aa(k,c) t0_2p t1_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //               += +4.000 P(i,j) d-_aa(k,c) t0_2p t1_1p_aa(c,i) t2_aaaa(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 4.000 * t0_2p * tmps.at("0032_aaaa_vvoo")(aa,ba,ia,ja) )
    .deallocate(tmps.at("0032_aaaa_vvoo"))
    .allocate(tmps.at("0033_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0033_aa_oo")(ja,ia)  = dp.at("aa_ov")(ja,ba) * t1_2p.at("aa")(ba,ia) )
    
    // r1_2p[aa] += -6.000 d-_aa(j,b) t1_1p_aa(a,j) t1_2p_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 6.000 * tmps.at("0033_aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    
    // r2_2p[abab] += -6.000 d-_aa(k,c) t1_2p_aa(c,i) t2_1p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * tmps.at("0033_aa_oo")(ka,ia) * t2_1p.at("abab")(aa,bb,ka,jb) )
    .allocate(tmps.at("0034_aaaa_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o2v1 o3v2 o2v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0034_aaaa_vvoo")(aa,ba,ia,ja)  = t2.at("aaaa")(aa,ba,ja,ka) * tmps.at("0033_aa_oo")(ka,ia) )
    ( tmps.at("0034_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("0031_aa_oo")(ka,ia) * t2_1p.at("aaaa")(aa,ba,ja,ka) )
    ( tmps.at("bin1_aa_oo")(ia,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("0034_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_oo")(ia,ka) * t2_2p.at("aaaa")(aa,ba,ja,ka) )
    ( tmps.at("0034_aaaa_vvoo")(aa,ba,ia,ja) += dp.at("aa_oo")(ka,ia) * t2_2p.at("aaaa")(aa,ba,ja,ka) )
    
    // r2_1p[aaaa] += +2.000 P(i,j) d-_aa(k,i) t2_2p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t1_aa(c,i) t2_2p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t1_1p_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t1_2p_aa(c,i) t2_aaaa(a,b,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0034_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_1p[aaaa] += +2.000 P(i,j) d-_aa(k,i) t2_2p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t1_aa(c,i) t2_2p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t1_1p_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t1_2p_aa(c,i) t2_aaaa(a,b,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0034_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) d-_aa(k,i) t0_1p t2_2p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t0_1p t1_aa(c,i) t2_2p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t0_1p t1_1p_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t0_1p t1_2p_aa(c,i) t2_aaaa(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * t0_1p * tmps.at("0034_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) d-_aa(k,i) t0_1p t2_2p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t0_1p t1_aa(c,i) t2_2p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t0_1p t1_1p_aa(c,i) t2_1p_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t0_1p t1_2p_aa(c,i) t2_aaaa(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * t0_1p * tmps.at("0034_aaaa_vvoo")(aa,ba,ia,ja) )
    .deallocate(tmps.at("0034_aaaa_vvoo"))
    .allocate(tmps.at("0035_aaaa_vvoo"))
    
    // flops: o2v2  = o3v2 o2v1 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v0 o2v2 o2v2
    ( tmps.at("0035_aaaa_vvoo")(aa,ba,ia,ja)  = dp.at("aa_oo")(ka,ia) * t2.at("aaaa")(aa,ba,ja,ka) )
    ( tmps.at("bin1_aa_oo")(ia,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("0035_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_oo")(ia,ka) * t2.at("aaaa")(aa,ba,ja,ka) )
    
    // r2[aaaa] += +1.000 P(i,j) d-_aa(k,i) t0_1p t2_aaaa(a,b,j,k) 
    //            += +1.000 P(i,j) d-_aa(k,c) t0_1p t1_aa(c,i) t2_aaaa(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) -= t0_1p * tmps.at("0035_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2[aaaa] += +1.000 P(i,j) d-_aa(k,i) t0_1p t2_aaaa(a,b,j,k) 
    //            += +1.000 P(i,j) d-_aa(k,c) t0_1p t1_aa(c,i) t2_aaaa(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) += t0_1p * tmps.at("0035_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) d+_aa(k,i) t2_aaaa(a,b,j,k) 
    //               += +1.000 P(i,j) d+_aa(k,c) t1_aa(c,i) t2_aaaa(a,b,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0035_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) d+_aa(k,i) t2_aaaa(a,b,j,k) 
    //               += +1.000 P(i,j) d+_aa(k,c) t1_aa(c,i) t2_aaaa(a,b,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0035_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_1p[aaaa] += +2.000 P(i,j) d-_aa(k,i) t0_2p t2_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t0_2p t1_aa(c,i) t2_aaaa(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * t0_2p * tmps.at("0035_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_1p[aaaa] += +2.000 P(i,j) d-_aa(k,i) t0_2p t2_aaaa(a,b,j,k) 
    //               += +2.000 P(i,j) d-_aa(k,c) t0_2p t1_aa(c,i) t2_aaaa(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 2.000 * t0_2p * tmps.at("0035_aaaa_vvoo")(aa,ba,ia,ja) )
    .deallocate(tmps.at("0035_aaaa_vvoo"))
    .allocate(tmps.at("0036_aaaa_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o3v2 o3v2 o2v2 o2v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = dp.at("aa_ov")(ka,ca) * t2.at("aaaa")(ca,aa,ia,ja) )
    ( tmps.at("0036_aaaa_vvoo")(aa,ba,ia,ja)  = tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1_1p.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = dp.at("aa_ov")(ka,ca) * t2_1p.at("aaaa")(ca,aa,ia,ja) )
    ( tmps.at("0036_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("0036_aaaa_vvoo")(aa,ba,ia,ja) += dp.at("aa_vv")(aa,ca) * t2_1p.at("aaaa")(ca,ba,ia,ja) )
    
    // r2[aaaa] += +1.000 P(a,b) d-_aa(a,c) t2_1p_aaaa(c,b,i,j) 
    //            += -1.000 P(a,b) d-_aa(k,c) t1_1p_aa(a,k) t2_aaaa(c,b,i,j) 
    //            += -1.000 P(a,b) d-_aa(k,c) t1_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0036_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2[aaaa] += +1.000 P(a,b) d-_aa(a,c) t2_1p_aaaa(c,b,i,j) 
    //            += -1.000 P(a,b) d-_aa(k,c) t1_1p_aa(a,k) t2_aaaa(c,b,i,j) 
    //            += -1.000 P(a,b) d-_aa(k,c) t1_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0036_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 P(a,b) d-_aa(a,c) t0_1p t2_1p_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) d-_aa(k,c) t0_1p t1_1p_aa(a,k) t2_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) d-_aa(k,c) t0_1p t1_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= t0_1p * tmps.at("0036_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 P(a,b) d-_aa(a,c) t0_1p t2_1p_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) d-_aa(k,c) t0_1p t1_1p_aa(a,k) t2_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) d-_aa(k,c) t0_1p t1_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += t0_1p * tmps.at("0036_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_2p[aaaa] += +2.000 P(a,b) d+_aa(a,c) t2_1p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d+_aa(k,c) t1_1p_aa(a,k) t2_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d+_aa(k,c) t1_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0036_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_2p[aaaa] += +2.000 P(a,b) d+_aa(a,c) t2_1p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d+_aa(k,c) t1_1p_aa(a,k) t2_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d+_aa(k,c) t1_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0036_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_2p[aaaa] += +4.000 P(a,b) d-_aa(a,c) t0_2p t2_1p_aaaa(c,b,i,j) 
    //               += -4.000 P(a,b) d-_aa(k,c) t0_2p t1_1p_aa(a,k) t2_aaaa(c,b,i,j) 
    //               += -4.000 P(a,b) d-_aa(k,c) t0_2p t1_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 4.000 * t0_2p * tmps.at("0036_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_2p[aaaa] += +4.000 P(a,b) d-_aa(a,c) t0_2p t2_1p_aaaa(c,b,i,j) 
    //               += -4.000 P(a,b) d-_aa(k,c) t0_2p t1_1p_aa(a,k) t2_aaaa(c,b,i,j) 
    //               += -4.000 P(a,b) d-_aa(k,c) t0_2p t1_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 4.000 * t0_2p * tmps.at("0036_aaaa_vvoo")(aa,ba,ia,ja) )
    .deallocate(tmps.at("0036_aaaa_vvoo"))
    .allocate(tmps.at("0037_aaaa_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o3v2 o3v2 o2v2 o3v2 o3v2 o2v2 o2v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = dp.at("aa_ov")(ka,ca) * t2.at("aaaa")(ca,aa,ia,ja) )
    ( tmps.at("0037_aaaa_vvoo")(aa,ba,ia,ja)  = tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1_2p.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = dp.at("aa_ov")(ka,ca) * t2_1p.at("aaaa")(ca,aa,ia,ja) )
    ( tmps.at("0037_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1_1p.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = dp.at("aa_ov")(ka,ca) * t2_2p.at("aaaa")(ca,aa,ia,ja) )
    ( tmps.at("0037_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("0037_aaaa_vvoo")(aa,ba,ia,ja) += dp.at("aa_vv")(aa,ca) * t2_2p.at("aaaa")(ca,ba,ia,ja) )
    
    // r2_1p[aaaa] += +2.000 P(a,b) d-_aa(a,c) t2_2p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t1_2p_aa(a,k) t2_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t1_1p_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t1_aa(a,k) t2_2p_aaaa(c,b,i,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0037_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_1p[aaaa] += +2.000 P(a,b) d-_aa(a,c) t2_2p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t1_2p_aa(a,k) t2_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t1_1p_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t1_aa(a,k) t2_2p_aaaa(c,b,i,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0037_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_2p[aaaa] += +2.000 P(a,b) d-_aa(a,c) t0_1p t2_2p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t0_1p t1_2p_aa(a,k) t2_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t0_1p t1_1p_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t0_1p t1_aa(a,k) t2_2p_aaaa(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * t0_1p * tmps.at("0037_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_2p[aaaa] += +2.000 P(a,b) d-_aa(a,c) t0_1p t2_2p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t0_1p t1_2p_aa(a,k) t2_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t0_1p t1_1p_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t0_1p t1_aa(a,k) t2_2p_aaaa(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * t0_1p * tmps.at("0037_aaaa_vvoo")(aa,ba,ia,ja) )
    .deallocate(tmps.at("0037_aaaa_vvoo"))
    .allocate(tmps.at("0038_aaaa_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o2v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = dp.at("aa_ov")(ka,ca) * t2.at("aaaa")(ca,aa,ia,ja) )
    ( tmps.at("0038_aaaa_vvoo")(aa,ba,ia,ja)  = tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("0038_aaaa_vvoo")(aa,ba,ia,ja) += dp.at("aa_vv")(aa,ca) * t2.at("aaaa")(ca,ba,ia,ja) )
    
    // r2[aaaa] += +1.000 P(a,b) d-_aa(a,c) t0_1p t2_aaaa(c,b,i,j) 
    //            += -1.000 P(a,b) d-_aa(k,c) t0_1p t1_aa(a,k) t2_aaaa(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) -= t0_1p * tmps.at("0038_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2[aaaa] += +1.000 P(a,b) d-_aa(a,c) t0_1p t2_aaaa(c,b,i,j) 
    //            += -1.000 P(a,b) d-_aa(k,c) t0_1p t1_aa(a,k) t2_aaaa(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) += t0_1p * tmps.at("0038_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 P(a,b) d+_aa(a,c) t2_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) d+_aa(k,c) t1_aa(a,k) t2_aaaa(c,b,i,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0038_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 P(a,b) d+_aa(a,c) t2_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) d+_aa(k,c) t1_aa(a,k) t2_aaaa(c,b,i,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0038_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_1p[aaaa] += +2.000 P(a,b) d-_aa(a,c) t0_2p t2_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t0_2p t1_aa(a,k) t2_aaaa(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * t0_2p * tmps.at("0038_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_1p[aaaa] += +2.000 P(a,b) d-_aa(a,c) t0_2p t2_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) d-_aa(k,c) t0_2p t1_aa(a,k) t2_aaaa(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 2.000 * t0_2p * tmps.at("0038_aaaa_vvoo")(aa,ba,ia,ja) )
    .deallocate(tmps.at("0038_aaaa_vvoo"))
    .allocate(tmps.at("0039_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0039_bb_oo")(jb,ib)  = dp.at("bb_ov")(jb,bb) * t1_1p.at("bb")(bb,ib) )
    
    // r1_2p[bb] += -6.000 d-_bb(j,b) t1_2p_bb(a,j) t1_1p_bb(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 6.000 * tmps.at("0039_bb_oo")(jb,ib) * t1_2p.at("bb")(ab,jb) )
    
    // r2_2p[abab] += -6.000 d-_bb(k,c) t1_1p_bb(c,j) t2_2p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * tmps.at("0039_bb_oo")(kb,jb) * t2_2p.at("abab")(aa,bb,ia,kb) )
    .allocate(tmps.at("0040_bbbb_vvoo"))
    
    // flops: o2v2  = o3v2 o2v1 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v0 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0040_bbbb_vvoo")(ab,bb,ib,jb)  = dp.at("bb_oo")(kb,ib) * t2_1p.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("bin1_bb_oo")(ib,kb)  = dp.at("bb_ov")(kb,cb) * t1.at("bb")(cb,ib) )
    ( tmps.at("0040_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_oo")(ib,kb) * t2_1p.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("0040_bbbb_vvoo")(ab,bb,ib,jb) += t2.at("bbbb")(ab,bb,jb,kb) * tmps.at("0039_bb_oo")(kb,ib) )
    
    // r2[bbbb] += +1.000 P(i,j) d-_bb(k,i) t2_1p_bbbb(a,b,j,k) 
    //            += +1.000 P(i,j) d-_bb(k,c) t1_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //            += +1.000 P(i,j) d-_bb(k,c) t1_1p_bb(c,i) t2_bbbb(a,b,j,k) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0040_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2[bbbb] += +1.000 P(i,j) d-_bb(k,i) t2_1p_bbbb(a,b,j,k) 
    //            += +1.000 P(i,j) d-_bb(k,c) t1_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //            += +1.000 P(i,j) d-_bb(k,c) t1_1p_bb(c,i) t2_bbbb(a,b,j,k) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0040_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2_1p[bbbb] += +1.000 P(i,j) d-_bb(k,i) t0_1p t2_1p_bbbb(a,b,j,k) 
    //               += +1.000 P(i,j) d-_bb(k,c) t0_1p t1_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +1.000 P(i,j) d-_bb(k,c) t0_1p t1_1p_bb(c,i) t2_bbbb(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= t0_1p * tmps.at("0040_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2_1p[bbbb] += +1.000 P(i,j) d-_bb(k,i) t0_1p t2_1p_bbbb(a,b,j,k) 
    //               += +1.000 P(i,j) d-_bb(k,c) t0_1p t1_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +1.000 P(i,j) d-_bb(k,c) t0_1p t1_1p_bb(c,i) t2_bbbb(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += t0_1p * tmps.at("0040_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) d+_bb(k,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d+_bb(k,c) t1_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d+_bb(k,c) t1_1p_bb(c,i) t2_bbbb(a,b,j,k) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0040_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) d+_bb(k,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d+_bb(k,c) t1_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d+_bb(k,c) t1_1p_bb(c,i) t2_bbbb(a,b,j,k) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0040_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2_2p[bbbb] += +4.000 P(i,j) d-_bb(k,i) t0_2p t2_1p_bbbb(a,b,j,k) 
    //               += +4.000 P(i,j) d-_bb(k,c) t0_2p t1_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +4.000 P(i,j) d-_bb(k,c) t0_2p t1_1p_bb(c,i) t2_bbbb(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 4.000 * t0_2p * tmps.at("0040_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2_2p[bbbb] += +4.000 P(i,j) d-_bb(k,i) t0_2p t2_1p_bbbb(a,b,j,k) 
    //               += +4.000 P(i,j) d-_bb(k,c) t0_2p t1_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +4.000 P(i,j) d-_bb(k,c) t0_2p t1_1p_bb(c,i) t2_bbbb(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 4.000 * t0_2p * tmps.at("0040_bbbb_vvoo")(ab,bb,ib,jb) )
    .deallocate(tmps.at("0040_bbbb_vvoo"))
    .allocate(tmps.at("0041_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0041_bb_oo")(jb,ib)  = dp.at("bb_ov")(jb,bb) * t1_2p.at("bb")(bb,ib) )
    
    // r1_2p[bb] += -6.000 d-_bb(j,b) t1_1p_bb(a,j) t1_2p_bb(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 6.000 * tmps.at("0041_bb_oo")(jb,ib) * t1_1p.at("bb")(ab,jb) )
    
    // r2_2p[abab] += -6.000 d-_bb(k,c) t1_2p_bb(c,j) t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * tmps.at("0041_bb_oo")(kb,jb) * t2_1p.at("abab")(aa,bb,ia,kb) )
    .allocate(tmps.at("0042_bbbb_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o2v1 o3v2 o2v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0042_bbbb_vvoo")(ab,bb,ib,jb)  = t2.at("bbbb")(ab,bb,jb,kb) * tmps.at("0041_bb_oo")(kb,ib) )
    ( tmps.at("0042_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("0039_bb_oo")(kb,ib) * t2_1p.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("bin1_bb_oo")(ib,kb)  = dp.at("bb_ov")(kb,cb) * t1.at("bb")(cb,ib) )
    ( tmps.at("0042_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_oo")(ib,kb) * t2_2p.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("0042_bbbb_vvoo")(ab,bb,ib,jb) += dp.at("bb_oo")(kb,ib) * t2_2p.at("bbbb")(ab,bb,jb,kb) )
    
    // r2_1p[bbbb] += +2.000 P(i,j) d-_bb(k,i) t2_2p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t1_bb(c,i) t2_2p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t1_1p_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t1_2p_bb(c,i) t2_bbbb(a,b,j,k) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0042_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_1p[bbbb] += +2.000 P(i,j) d-_bb(k,i) t2_2p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t1_bb(c,i) t2_2p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t1_1p_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t1_2p_bb(c,i) t2_bbbb(a,b,j,k) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0042_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) d-_bb(k,i) t0_1p t2_2p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t0_1p t1_bb(c,i) t2_2p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t0_1p t1_1p_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t0_1p t1_2p_bb(c,i) t2_bbbb(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * t0_1p * tmps.at("0042_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) d-_bb(k,i) t0_1p t2_2p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t0_1p t1_bb(c,i) t2_2p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t0_1p t1_1p_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t0_1p t1_2p_bb(c,i) t2_bbbb(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * t0_1p * tmps.at("0042_bbbb_vvoo")(ab,bb,ib,jb) )
    .deallocate(tmps.at("0042_bbbb_vvoo"))
    .allocate(tmps.at("0043_bbbb_vvoo"))
    ;
  }
  // clang-format on
}

template void exachem::cc::cd_qed_ccsd_os::resid_part1<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);