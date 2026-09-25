/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_qed_ccsd_os_resid_4.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_os::resid_part4(
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
        
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0151_Q")(Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t1_1p.at("aa")(aa,ia) )
    
    // r1_1p[aa] += +1.000 <a,j||i,b>_aaaa t1_1p_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0151_Q")(Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r1_1p[bb] += +1.000 <j,a||b,i>_abab t1_1p_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0151_Q")(Q) * chol.at("bb_voQ")(ab,ib,Q) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,ka)  = chol.at("aa_ovQ")(ka,da,Q) * tmps.at("0151_Q")(Q) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin1_aa_vo")(da,ka) * t2_1p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r0_1p() += -1.000 <j,i||a,b>_aaaa t1_aa(a,i) t1_1p_aa(b,j) 
    //       += +1.000 <i,j||a,b>_abab t1_aa(a,i) t1_1p_bb(b,j) 
    //       += +0.250 <i,j||b,a>_aaaa t2_1p_aaaa(b,a,i,j) 
    //       += -1.000 <j,i||a,b>_aaaa t1_aa(a,i) t1_1p_aa(b,j) 
    //       += +0.250 <i,j||b,a>_aaaa t2_1p_aaaa(b,a,i,j) 
    //       += +0.250 <i,j||b,a>_abab t2_1p_abab(b,a,i,j) 
    //       += +0.250 <j,i||b,a>_abab t2_1p_abab(b,a,j,i) 
    //       += +0.250 <i,j||a,b>_abab t2_1p_abab(a,b,i,j) 
    //       += +0.250 <j,i||a,b>_abab t2_1p_abab(a,b,j,i) 
    //       += +1.000 <j,i||b,a>_abab t1_bb(a,i) t1_1p_aa(b,j) 
    //       += -1.000 <j,i||a,b>_bbbb t1_bb(a,i) t1_1p_bb(b,j) 
    //       += +0.250 <i,j||b,a>_bbbb t2_1p_bbbb(b,a,i,j) 
    //       += -1.000 <j,i||a,b>_bbbb t1_bb(a,i) t1_1p_bb(b,j) 
    //       += +0.250 <i,j||b,a>_bbbb t2_1p_bbbb(b,a,i,j) 
    //       += +1.000 f_aa(i,a) t1_1p_aa(a,i) 
    //       += +1.000 f_bb(i,a) t1_1p_bb(a,i) 
    //       += +2.000 d-_aa(i,a) t0_2p t1_aa(a,i) 
    //       += +2.000 d-_bb(i,a) t0_2p t1_bb(a,i) 
    //       += +1.000 d-_aa(i,a) t0_1p t1_1p_aa(a,i) 
    //       += +1.000 d-_bb(i,a) t0_1p t1_1p_bb(a,i) 
    //       += +1.000 t0_1p w0 
    // flops: 0 += 0 o2v2Q1 o1v1Q1 o2v2Q1 o1v1Q1 o2v2Q1 o1v1Q1 o2v1Q1 o1v1 0 0 o0v0Q1 0 o0v0Q1 0 0 o0v0Q1 o1v1Q1 0 0 o2v1Q1 o1v1 0 o2v2Q1 o1v1Q1 0 0 o1v1 o0v0Q1 0 0 0 0 o1v1 0 0 0
    //  mems: 0 += 0 o1v1Q1 0 o1v1Q1 0 o1v1Q1 0 o1v1 0 0 0 0 0 0 0 0 0 0 0 0 o1v1 0 0 o1v1Q1 0 0 0 0 0 0 0 0 0 0 0 0 0
    ( r0_1p() += 2.000 * t0_2p * scalars.at("0003")() )
    ( tmps.at("bin1_bb_voQ")(ab,ib,Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t2_1p.at("bbbb")(bb,ab,ib,jb) )
    ( r0_1p() -= 0.250 * tmps.at("bin1_bb_voQ")(ab,ib,Q) * chol.at("bb_ovQ")(ib,ab,Q) )
    ( tmps.at("bin1_aa_voQ")(ba,ia,Q)  = chol.at("aa_ovQ")(ja,aa,Q) * t2_1p.at("aaaa")(ba,aa,ia,ja) )
    ( r0_1p() += 0.250 * tmps.at("bin1_aa_voQ")(ba,ia,Q) * chol.at("aa_ovQ")(ia,ba,Q) )
    ( tmps.at("bin1_aa_voQ")(aa,ia,Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t2_1p.at("aaaa")(ba,aa,ia,ja) )
    ( r0_1p() -= 0.250 * tmps.at("bin1_aa_voQ")(aa,ia,Q) * chol.at("aa_ovQ")(ia,aa,Q) )
    ( tmps.at("bin1_aa_vo")(ba,ja)  = tmps.at("0053_aa_ooQ")(ja,ia,Q) * chol.at("aa_ovQ")(ia,ba,Q) )
    ( r0_1p() -= t1_1p.at("aa")(ba,ja) * tmps.at("bin1_aa_vo")(ba,ja) )
    ( r0_1p() += tmps.at("0049_Q")(Q) * tmps.at("0150_Q")(Q) )
    ( r0_1p() += tmps.at("0049_Q")(Q) * tmps.at("0151_Q")(Q) )
    ( r0_1p() += tmps.at("0148_Q")(Q) * tmps.at("0151_Q")(Q) )
    ( r0_1p() += tmps.at("0028_aa_voQ")(ba,ia,Q) * chol.at("aa_ovQ")(ia,ba,Q) )
    ( tmps.at("bin1_bb_vo")(bb,jb)  = tmps.at("0026_bb_ooQ")(jb,ib,Q) * chol.at("bb_ovQ")(ib,bb,Q) )
    ( r0_1p() -= t1_1p.at("bb")(bb,jb) * tmps.at("bin1_bb_vo")(bb,jb) )
    ( tmps.at("bin1_bb_voQ")(bb,ib,Q)  = chol.at("bb_ovQ")(jb,ab,Q) * t2_1p.at("bbbb")(bb,ab,ib,jb) )
    ( r0_1p() += 0.250 * tmps.at("bin1_bb_voQ")(bb,ib,Q) * chol.at("bb_ovQ")(ib,bb,Q) )
    ( r0_1p() += f.at("aa_ov")(ia,aa) * t1_1p.at("aa")(aa,ia) )
    ( r0_1p() += tmps.at("0148_Q")(Q) * tmps.at("0150_Q")(Q) )
    ( r0_1p() += t0_1p * scalars.at("0009")() )
    ( r0_1p() += f.at("bb_ov")(ib,ab) * t1_1p.at("bb")(ab,ib) )
    ( r0_1p() += t0_1p * w0 )
    
    // r1_2p[bb] += -2.000 <j,k||b,c>_abab t1_1p_aa(b,j) t2_1p_bbbb(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0136_bb_voQ")(ab,ib,Q) * tmps.at("0151_Q")(Q) )
    
    // r1_1p[bb] += +1.000 <j,a||c,b>_abab t1_bb(b,i) t1_1p_aa(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0060_bb_voQ")(ab,ib,Q) * tmps.at("0151_Q")(Q) )
    
    // r1_2p[aa] += +2.000 <j,k||b,c>_abab t1_1p_aa(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0028_aa_voQ")(aa,ia,Q) * tmps.at("0151_Q")(Q) )
    
    // r1_2p[bb] += -2.000 <k,j||b,c>_aaaa t1_1p_aa(b,j) t2_1p_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0135_bb_voQ")(ab,ib,Q) * tmps.at("0151_Q")(Q) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_1p_aa(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0140_aa_voQ")(aa,ia,Q) * tmps.at("0151_Q")(Q) )
    
    // r1_2p[aa] += -2.000 <a,j||b,c>_aaaa t1_1p_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0141_aa_voQ")(aa,ia,Q) * tmps.at("0151_Q")(Q) )
    
    // r1_1p[bb] += -1.000 <j,k||c,b>_aaaa t1_1p_aa(b,j) t2_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0058_bb_voQ")(ab,ib,Q) * tmps.at("0151_Q")(Q) )
    
    // r1_1p[aa] += +1.000 <j,k||c,b>_aaaa t1_1p_aa(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0151_Q")(Q) )
    
    // r1_1p[aa] += +1.000 <j,k||b,c>_abab t1_1p_aa(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0151_Q")(Q) )
    
    // r1_1p[aa] += +1.000 <a,j||b,c>_aaaa t1_aa(b,i) t1_1p_aa(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0151_Q")(Q) )
    
    // r1_1p[bb] += -1.000 <j,k||b,c>_abab t1_1p_aa(b,j) t2_bbbb(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0059_bb_voQ")(ab,ib,Q) * tmps.at("0151_Q")(Q) )
    .allocate(tmps.at("0152_aa_vv"))
    
    // flops: o0v2  = o0v2Q1 o0v2Q1 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0152_aa_vv")(aa,da)  = tmps.at("0150_Q")(Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( tmps.at("0152_aa_vv")(aa,da) += chol.at("aa_vvQ")(aa,da,Q) * tmps.at("0151_Q")(Q) )
    
    // r2_1p[abab] += +1.000 <a,k||d,c>_abab t1_1p_bb(c,k) t2_abab(d,b,i,j) 
    //               += +1.000 <a,k||d,c>_aaaa t1_1p_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0152_aa_vv")(aa,da) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <a,k||d,c>_abab t1_1p_bb(c,k) t2_1p_abab(d,b,i,j) 
    //               += -2.000 <a,k||c,d>_aaaa t1_1p_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0152_aa_vv")(aa,da) * t2_1p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0152_aa_vv"))
    .allocate(tmps.at("0153_aa_vo"))
    
    // flops: o1v1  = o1v2 o2v2 o1v1
    //  mems: o1v1  = o1v1 o1v1 o1v1
    ( tmps.at("0153_aa_vo")(aa,ia)  = dp.at("aa_vv")(aa,ba) * t1.at("aa")(ba,ia) )
    ( tmps.at("0153_aa_vo")(aa,ia) += dp.at("bb_ov")(jb,bb) * t2.at("abab")(aa,bb,ia,jb) )
    .allocate(tmps.at("0154_aaaa_vvoo"))
    
    // flops: o2v2  = o2v2 o2v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0154_aaaa_vvoo")(aa,ba,ia,ja)  = dp.at("aa_vo")(aa,ia) * t1_1p.at("aa")(ba,ja) )
    ( tmps.at("0154_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("0153_aa_vo")(ba,ja) * t1_1p.at("aa")(aa,ia) )
    
    // r2[aaaa] += +1.000 P(i,j) P(a,b) d-_aa(a,i) t1_1p_aa(b,j) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_aa(c,j) 
    //            += +1.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_abab(b,c,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0154_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2[aaaa] += +1.000 P(i,j) P(a,b) d-_aa(a,i) t1_1p_aa(b,j) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_aa(c,j) 
    //            += +1.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_abab(b,c,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0154_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2[aaaa] += +1.000 P(i,j) P(a,b) d-_aa(a,i) t1_1p_aa(b,j) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_aa(c,j) 
    //            += +1.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_abab(b,c,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0154_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2[aaaa] += +1.000 P(i,j) P(a,b) d-_aa(a,i) t1_1p_aa(b,j) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_aa(c,j) 
    //            += +1.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_abab(b,c,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0154_aaaa_vvoo")(ba,aa,ja,ia) )
    .deallocate(tmps.at("0154_aaaa_vvoo"))
    .allocate(tmps.at("0155_bb_vo"))
    
    // flops: o1v1  = o2v2 o1v2 o1v1
    //  mems: o1v1  = o1v1 o1v1 o1v1
    ( tmps.at("0155_bb_vo")(ab,ib)  = dp.at("aa_ov")(ja,ba) * t2.at("abab")(ba,ab,ja,ib) )
    ( tmps.at("0155_bb_vo")(ab,ib) += dp.at("bb_vv")(ab,bb) * t1.at("bb")(bb,ib) )
    .allocate(tmps.at("0156_bbbb_vvoo"))
    
    // flops: o2v2  = o2v2 o3v1Q1 o3v1Q1 o3v1 o3v2 o2v2 o2v2 o2v2
    //  mems: o2v2  = o2v2 o3v1 o3v1 o3v1 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0156_bbbb_vvoo")(ab,bb,ib,jb)  = dp.at("bb_vo")(ab,ib) * t1_1p.at("bb")(bb,jb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = chol.at("bb_ooQ")(kb,jb,Q) * tmps.at("0058_bb_voQ")(bb,ib,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) += tmps.at("0058_bb_voQ")(bb,ib,Q) * tmps.at("0026_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0156_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1.at("bb")(ab,kb) )
    ( tmps.at("0156_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("0155_bb_vo")(bb,jb) * t1_1p.at("bb")(ab,ib) )
    
    // r2[bbbb] += +1.000 P(i,j) P(a,b) d-_bb(a,i) t1_1p_bb(b,j) 
    //            += -1.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_bb(c,i) t2_abab(d,b,l,j) 
    //            += -1.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_bb(a,k) t2_abab(c,b,l,j) 
    //            += +1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_bb(a,i) t2_abab(c,b,k,j) 
    //            += -1.000 P(i,j) P(a,b) d-_bb(a,c) t1_1p_bb(b,i) t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0156_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2[bbbb] += +1.000 P(i,j) P(a,b) d-_bb(a,i) t1_1p_bb(b,j) 
    //            += -1.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_bb(c,i) t2_abab(d,b,l,j) 
    //            += -1.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_bb(a,k) t2_abab(c,b,l,j) 
    //            += +1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_bb(a,i) t2_abab(c,b,k,j) 
    //            += -1.000 P(i,j) P(a,b) d-_bb(a,c) t1_1p_bb(b,i) t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0156_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2[bbbb] += +1.000 P(i,j) P(a,b) d-_bb(a,i) t1_1p_bb(b,j) 
    //            += -1.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_bb(c,i) t2_abab(d,b,l,j) 
    //            += -1.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_bb(a,k) t2_abab(c,b,l,j) 
    //            += +1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_bb(a,i) t2_abab(c,b,k,j) 
    //            += -1.000 P(i,j) P(a,b) d-_bb(a,c) t1_1p_bb(b,i) t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0156_bbbb_vvoo")(bb,ab,ib,jb) )
    
    // r2[bbbb] += +1.000 P(i,j) P(a,b) d-_bb(a,i) t1_1p_bb(b,j) 
    //            += -1.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_bb(c,i) t2_abab(d,b,l,j) 
    //            += -1.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_bb(a,k) t2_abab(c,b,l,j) 
    //            += +1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_bb(a,i) t2_abab(c,b,k,j) 
    //            += -1.000 P(i,j) P(a,b) d-_bb(a,c) t1_1p_bb(b,i) t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0156_bbbb_vvoo")(bb,ab,jb,ib) )
    .deallocate(tmps.at("0156_bbbb_vvoo"))
    .allocate(tmps.at("0157_bb_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0157_bb_vv")(cb,ab)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t2_1p.at("bbbb")(db,ab,kb,lb) )
    
    // r1_1p[bb] += +0.500 <j,k||b,c>_bbbb t1_bb(b,i) t2_1p_bbbb(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= 0.500 * t1.at("bb")(bb,ib) * tmps.at("0157_bb_vv")(bb,ab) )
    
    // r1_2p[bb] += +1.000 <j,k||b,c>_bbbb t1_1p_bb(b,i) t2_1p_bbbb(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= t1_1p.at("bb")(bb,ib) * tmps.at("0157_bb_vv")(bb,ab) )
    
    // r2_2p[bbbb] += -1.000 P(a,b) <k,l||d,c>_bbbb t2_1p_bbbb(d,a,i,j) t2_1p_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0157_bb_vv")(db,ab) * t2_1p.at("bbbb")(db,bb,ib,jb) )
    .allocate(tmps.at("0158_bb_vo"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0158_bb_vo")(cb,kb)  = tmps.at("0075_bbbb_ovov")(jb,cb,kb,bb) * t1_1p.at("bb")(bb,jb) )
    
    // r1_2p[aa] += -2.000 <k,j||b,c>_bbbb t1_1p_bb(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0158_bb_vo")(cb,kb) * t2_1p.at("abab")(aa,cb,ia,kb) )
    
    // r1_2p[bb] += -2.000 <k,j||b,c>_bbbb t1_bb(a,j) t1_1p_bb(b,k) t1_1p_bb(c,i) 
    // flops: o1v1 += o2v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_bb_oo")(ib,jb)  = t1_1p.at("bb")(cb,ib) * tmps.at("0158_bb_vo")(cb,jb) )
    ( r1_2p.at("bb")(ab,ib) += 2.000 * t1.at("bb")(ab,jb) * tmps.at("bin1_bb_oo")(ib,jb) )
    
    // r1_2p[bb] += +2.000 <k,j||b,c>_bbbb t1_1p_bb(b,j) t2_1p_bbbb(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0158_bb_vo")(cb,kb) * t2_1p.at("bbbb")(cb,ab,ib,kb) )
    
    // r2_1p[abab] += +1.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_1p_bb(c,l) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t2.at("abab")(aa,db,ia,jb) * tmps.at("0158_bb_vo")(db,kb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_1p_bb(c,l) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t2_1p.at("abab")(aa,db,ia,jb) * tmps.at("0158_bb_vo")(db,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    .allocate(tmps.at("0159_bbbb_vvoo"))
    
    // flops: o2v2  = o4v2 o4v1 o3v2 o3v2 o3v2 o2v2 o2v3 o2v2 o2v3 o2v2
    //  mems: o2v2  = o4v0 o3v1 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) * t2.at("bbbb")(db,cb,ib,jb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = t1_1p.at("bb")(ab,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("0159_bbbb_vvoo")(ab,bb,ib,jb)  = 0.500 * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t2.at("bbbb")(db,bb,ib,jb) * tmps.at("0158_bb_vo")(db,kb) )
    ( tmps.at("0159_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1.at("bb")(ab,kb) )
    ( tmps.at("0159_bbbb_vvoo")(ab,bb,ib,jb) += 0.500 * t2.at("bbbb")(cb,ab,ib,jb) * tmps.at("0157_bb_vv")(cb,bb) )
    ( tmps.at("0159_bbbb_vvoo")(ab,bb,ib,jb) += f.at("bb_vv")(ab,cb) * t2_1p.at("bbbb")(cb,bb,ib,jb) )
    .deallocate(tmps.at("0157_bb_vv"))
    
    // r2_1p[bbbb] += +1.000 P(a,b) f_bb(a,c) t2_1p_bbbb(c,b,i,j) 
    //               += +1.000 P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_1p_bb(c,l) t2_bbbb(d,b,i,j) 
    //               += -0.500 P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_1p_bb(b,l) t2_bbbb(d,c,i,j) 
    //               += +0.500 P(a,b) <k,l||c,d>_bbbb t2_1p_bbbb(d,a,k,l) t2_bbbb(c,b,i,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0159_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_1p[bbbb] += +1.000 P(a,b) f_bb(a,c) t2_1p_bbbb(c,b,i,j) 
    //               += +1.000 P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_1p_bb(c,l) t2_bbbb(d,b,i,j) 
    //               += -0.500 P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_1p_bb(b,l) t2_bbbb(d,c,i,j) 
    //               += +0.500 P(a,b) <k,l||c,d>_bbbb t2_1p_bbbb(d,a,k,l) t2_bbbb(c,b,i,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0159_bbbb_vvoo")(bb,ab,ib,jb) )
    .deallocate(tmps.at("0159_bbbb_vvoo"))
    .allocate(tmps.at("0160_aa_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0160_aa_vv")(aa,da)  = t2_1p.at("aaaa")(ca,aa,ka,la) * tmps.at("0073_aaaa_ovov")(la,ca,ka,da) )
    
    // r1_1p[aa] += +0.500 <j,k||b,c>_aaaa t1_aa(b,i) t2_1p_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 0.500 * tmps.at("0160_aa_vv")(aa,ba) * t1.at("aa")(ba,ia) )
    
    // r1_2p[aa] += +1.000 <j,k||b,c>_aaaa t1_1p_aa(b,i) t2_1p_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += tmps.at("0160_aa_vv")(aa,ba) * t1_1p.at("aa")(ba,ia) )
    
    // r2_1p[abab] += +0.500 <k,l||c,d>_aaaa t2_1p_aaaa(d,a,k,l) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.500 * tmps.at("0160_aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p[abab] += +1.000 <k,l||d,c>_aaaa t2_1p_aaaa(c,a,k,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("0160_aa_vv")(aa,da) )
    .allocate(tmps.at("0161_aaaa_vvoo"))
    
    // flops: o2v2  = o2v3 o1v1Q1 o1v1Q1 o3v2 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o2v2 o2v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o2v2 o4v2 o4v1 o3v2 o2v2 o2v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o2v2 o2v2 o3v2 o3v2 o2v2 o3v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o0v0Q1 o1v1 o3v1 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o2v2 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o2v2 o4v0 o3v1 o2v2 o2v2 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o2v2 o2v2 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2
    ( tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja)  = 0.500 * tmps.at("0160_aa_vv")(aa,ca) * t2.at("aaaa")(ca,ba,ia,ja) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = tmps.at("bin1_aa_vo")(da,la) * t2.at("aaaa")(da,aa,ia,ja) )
    ( tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1_1p.at("aa")(ba,la) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = tmps.at("bin1_aa_vo")(da,la) * t2_1p.at("aaaa")(da,aa,ia,ja) )
    ( tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1.at("aa")(ba,la) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = tmps.at("bin1_aa_vo")(da,la) * t2_1p.at("aaaa")(da,aa,ia,ja) )
    ( tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1.at("aa")(ba,la) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t1_1p.at("bb")(cb,lb) )
    ( tmps.at("bin1_aa_vo")(da,ka)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("bin1_aa_vo")(da,ka) * t2.at("aaaa")(da,aa,ia,ja) )
    ( tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0073_aaaa_ovov")(la,ca,ka,da) * t2.at("aaaa")(da,ca,ia,ja) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1_1p.at("aa")(ba,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja) += 0.500 * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1.at("aa")(aa,ka) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = tmps.at("bin1_aa_vo")(da,la) * t2.at("aaaa")(da,aa,ia,ja) )
    ( tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1_1p.at("aa")(ba,la) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1_1p.at("aa")(ca,la) )
    ( tmps.at("bin1_aa_vo")(da,ka)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("bin1_aa_vo")(da,ka) * t2.at("aaaa")(da,aa,ia,ja) )
    ( tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = t2.at("aaaa")(da,ba,ia,ja) * tmps.at("0105_aa_ov")(la,da) )
    ( tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja) += t1_1p.at("aa")(aa,la) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = t2_1p.at("aaaa")(da,ba,ia,ja) * tmps.at("0105_aa_ov")(la,da) )
    ( tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja) += t1.at("aa")(aa,la) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) )
    .deallocate(tmps.at("0160_aa_vv"))
    
    // r2_1p[aaaa] += +1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_aaaa(d,b,i,j) 
    //               += -1.000 P(a,b) <l,k||d,c>_abab t1_1p_aa(a,l) t1_bb(c,k) t2_aaaa(d,b,i,j) 
    //               += -0.500 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t2_aaaa(d,c,i,j) 
    //               += +0.500 P(a,b) <k,l||c,d>_aaaa t2_1p_aaaa(d,a,k,l) t2_aaaa(c,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += -1.000 P(a,b) <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(c,l) t2_aaaa(d,b,i,j) 
    //               += -1.000 P(a,b) <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_aaaa(d,b,i,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0161_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_aaaa(d,b,i,j) 
    //               += -1.000 P(a,b) <l,k||d,c>_abab t1_1p_aa(a,l) t1_bb(c,k) t2_aaaa(d,b,i,j) 
    //               += -0.500 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t2_aaaa(d,c,i,j) 
    //               += +0.500 P(a,b) <k,l||c,d>_aaaa t2_1p_aaaa(d,a,k,l) t2_aaaa(c,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += -1.000 P(a,b) <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(c,l) t2_aaaa(d,b,i,j) 
    //               += -1.000 P(a,b) <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_aaaa(d,b,i,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0161_aaaa_vvoo")(ba,aa,ia,ja) )
    .deallocate(tmps.at("0161_aaaa_vvoo"))
    .allocate(tmps.at("0162_aaaa_voov"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin1_aaaa_vvoo")(ca,da,ka,la)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("0162_aaaa_voov")(aa,la,ia,da)  = t2.at("aaaa")(ca,aa,ia,ka) * tmps.at("bin1_aaaa_vvoo")(ca,da,ka,la) )
    
    // r1[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += t1.at("aa")(ba,ja) * tmps.at("0162_aaaa_voov")(aa,ja,ia,ba) )
    
    // r1_1p[aa] += +1.000 <j,k||c,b>_aaaa t1_1p_aa(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1_1p.at("aa")(ba,ja) * tmps.at("0162_aaaa_voov")(aa,ja,ia,ba) )
    
    // r1_2p[aa] += +2.000 <j,k||c,b>_aaaa t1_2p_aa(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1_2p.at("aa")(ba,ja) * tmps.at("0162_aaaa_voov")(aa,ja,ia,ba) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(da,bb,la,jb) * tmps.at("0162_aaaa_voov")(aa,la,ia,da) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(da,bb,la,jb) * tmps.at("0162_aaaa_voov")(aa,la,ia,da) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("abab")(da,bb,la,jb) * tmps.at("0162_aaaa_voov")(aa,la,ia,da) )
    .allocate(tmps.at("0163_abab_voov"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin1_bbbb_vvoo")(cb,db,kb,lb)  = chol.at("bb_ovQ")(lb,cb,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0163_abab_voov")(aa,lb,ia,db)  = t2.at("abab")(aa,cb,ia,kb) * tmps.at("bin1_bbbb_vvoo")(cb,db,kb,lb) )
    
    // r1[aa] += -1.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= t1.at("bb")(bb,jb) * tmps.at("0163_abab_voov")(aa,jb,ia,bb) )
    
    // r1_1p[aa] += -1.000 <j,k||c,b>_bbbb t1_1p_bb(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1_1p.at("bb")(bb,jb) * tmps.at("0163_abab_voov")(aa,jb,ia,bb) )
    
    // r1_2p[aa] += -2.000 <j,k||c,b>_bbbb t1_2p_bb(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * t1_2p.at("bb")(bb,jb) * tmps.at("0163_abab_voov")(aa,jb,ia,bb) )
    
    // r2[abab] += +1.000 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("bbbb")(db,bb,jb,lb) * tmps.at("0163_abab_voov")(aa,lb,ia,db) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("bbbb")(db,bb,jb,lb) * tmps.at("0163_abab_voov")(aa,lb,ia,db) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_2p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("bbbb")(db,bb,jb,lb) * tmps.at("0163_abab_voov")(aa,lb,ia,db) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_2p_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t1_2p.at("bb")(cb,jb) * tmps.at("0163_abab_voov")(aa,kb,ia,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0164_aaaa_ooov"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0164_aaaa_ooov")(ka,ia,ja,ca)  = t1.at("aa")(ba,ia) * tmps.at("0073_aaaa_ovov")(ka,ba,ja,ca) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,la,jb) * tmps.at("0164_aaaa_ooov")(ka,ia,la,da) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0164_aaaa_ooov")(la,ia,ka,da) * t2_2p.at("abab")(da,bb,la,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(da,bb,la,jb) * tmps.at("0164_aaaa_ooov")(ka,ia,la,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0165_aaaa_vvoo"))
    
    // flops: o2v2  = o2v1 o2v1 o2v2 o2v2 o2v2 o2v1Q1 o4v0Q1 o4v1 o3v2 o2v1Q1 o4v0Q1 o4v1 o3v2 o2v2 o2v2 o2v1Q1 o4v0Q1 o4v1 o3v2 o2v2 o2v2 o2v2 o2v2 o2v1 o2v2 o2v2 o2v2 o2v1 o2v2 o2v2 o4v0Q1 o4v1 o3v2 o2v2 o2v1 o2v1 o2v2 o2v2 o4v2 o3v2 o2v2 o2v1 o2v2 o2v1 o2v2 o2v2 o2v2 o3v2 o3v2 o2v2 o3v2 o3v2 o2v2 o3v3 o2v2 o3v3 o2v2 o2v1 o2v2 o2v2
    //  mems: o2v2  = o2v0 o1v1 o2v2 o1v1 o2v2 o2v0Q1 o4v0 o3v1 o2v2 o2v0Q1 o4v0 o3v1 o2v2 o2v2 o2v2 o2v0Q1 o4v0 o3v1 o2v2 o2v2 o2v2 o1v1 o2v2 o1v1 o2v2 o2v2 o2v2 o1v1 o2v2 o2v2 o4v0 o3v1 o2v2 o2v2 o2v0 o1v1 o2v2 o2v2 o3v1 o2v2 o2v2 o1v1 o2v2 o1v1 o2v2 o2v2 o2v2 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o1v1 o2v2 o2v2
    ( tmps.at("bin1_aa_oo")(ja,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ja) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = tmps.at("bin1_aa_oo")(ja,ka) * t1_1p.at("aa")(aa,ka) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia)  = 2.000 * tmps.at("bin1_aa_vo")(aa,ja) * t1_2p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = dp.at("aa_ov")(ka,ca) * t2_1p.at("aaaa")(ca,aa,ja,ka) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += 2.000 * tmps.at("bin1_aa_vo")(aa,ja) * t1_2p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aa_ooQ")(ia,ka,Q)  = t1_1p.at("aa")(da,ia) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0053_aa_ooQ")(la,ja,Q) * tmps.at("bin1_aa_ooQ")(ia,ka,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1_1p.at("aa")(ba,la) )
    ( tmps.at("bin1_aa_ooQ")(ia,ka,Q)  = t1_2p.at("aa")(da,ia) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0053_aa_ooQ")(la,ja,Q) * tmps.at("bin1_aa_ooQ")(ia,ka,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1.at("aa")(ba,la) )
    ( tmps.at("bin1_aa_ooQ")(ia,ka,Q)  = t1.at("aa")(ca,ia) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0137_aa_ooQ")(la,ja,Q) * tmps.at("bin1_aa_ooQ")(ia,ka,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1_1p.at("aa")(ba,la) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = dp.at("aa_ov")(ka,ca) * t2_2p.at("aaaa")(ca,aa,ja,ka) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aa_vo")(aa,ja) * t1_1p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = dp.at("aa_oo")(ka,ja) * t1_1p.at("aa")(aa,ka) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += 2.000 * tmps.at("bin1_aa_vo")(aa,ja) * t1_2p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = tmps.at("0031_aa_oo")(ka,ja) * t1.at("aa")(aa,ka) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += 2.000 * tmps.at("bin1_aa_vo")(aa,ja) * t1_2p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0053_aa_ooQ")(la,ja,Q) * tmps.at("0053_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1_2p.at("aa")(ba,la) )
    ( tmps.at("bin1_aa_oo")(ja,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ja) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = tmps.at("bin1_aa_oo")(ja,ka) * t1_2p.at("aa")(aa,ka) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aa_vo")(aa,ja) * t1_1p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t2_2p.at("aaaa")(da,ba,ia,la) * tmps.at("0164_aaaa_ooov")(la,ja,ka,da) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = t1.at("aa")(aa,ka) * tmps.at("0033_aa_oo")(ka,ja) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aa_vo")(aa,ja) * t1_1p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = tmps.at("0031_aa_oo")(ka,ja) * t1_1p.at("aa")(aa,ka) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += t1_1p.at("aa")(ba,ia) * tmps.at("bin1_aa_vo")(aa,ja) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1_2p.at("aa")(ca,ja) * tmps.at("0162_aaaa_voov")(ba,ka,ia,ca) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1_1p.at("aa")(ca,ja) * tmps.at("0162_aaaa_voov")(ba,ka,ia,ca) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += t1_1p.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += t2_2p.at("aaaa")(da,aa,ja,la) * tmps.at("0162_aaaa_voov")(ba,la,ia,da) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += t2_2p.at("abab")(aa,db,ja,lb) * tmps.at("0163_abab_voov")(ba,lb,ia,db) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = dp.at("aa_oo")(ka,ja) * t1_2p.at("aa")(aa,ka) )
    ( tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aa_vo")(aa,ja) * t1_1p.at("aa")(ba,ia) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_aaaa(d,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_2p_aaaa(d,a,i,l) t2_aaaa(c,b,j,k) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_2p_abab(a,d,i,l) t2_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(b,l) t1_aa(c,i) t1_2p_aa(d,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +2.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t2_1p_aaaa(c,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_2p_aaaa(c,b,j,k) 
    //               += +4.000 P(i,j) P(a,b) d-_aa(k,i) t1_2p_aa(a,j) t1_1p_aa(b,k) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_aa(b,k) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_2p_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_2p_aa(b,k) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_1p_aa(b,k) t1_1p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_2p_aa(c,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0165_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_aaaa(d,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_2p_aaaa(d,a,i,l) t2_aaaa(c,b,j,k) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_2p_abab(a,d,i,l) t2_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(b,l) t1_aa(c,i) t1_2p_aa(d,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +2.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t2_1p_aaaa(c,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_2p_aaaa(c,b,j,k) 
    //               += +4.000 P(i,j) P(a,b) d-_aa(k,i) t1_2p_aa(a,j) t1_1p_aa(b,k) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_aa(b,k) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_2p_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_2p_aa(b,k) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_1p_aa(b,k) t1_1p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_2p_aa(c,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0165_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_aaaa(d,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_2p_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_2p_abab(b,d,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(b,l) t1_aa(c,i) t1_2p_aa(d,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +2.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t2_1p_aaaa(c,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_2p_aaaa(c,b,j,k) 
    //               += +4.000 P(i,j) P(a,b) d-_aa(k,i) t1_2p_aa(a,j) t1_1p_aa(b,k) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_aa(b,k) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_2p_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_2p_aa(b,k) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_1p_aa(b,k) t1_1p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_2p_aa(c,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0165_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_aaaa(d,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_2p_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_2p_abab(b,d,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(b,l) t1_aa(c,i) t1_2p_aa(d,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +2.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t2_1p_aaaa(c,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_2p_aaaa(c,b,j,k) 
    //               += +4.000 P(i,j) P(a,b) d-_aa(k,i) t1_2p_aa(a,j) t1_1p_aa(b,k) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_aa(b,k) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_2p_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_2p_aa(b,k) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_1p_aa(b,k) t1_1p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_2p_aa(c,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0165_aaaa_vvoo")(ba,aa,ja,ia) )
    .deallocate(tmps.at("0165_aaaa_vvoo"))
    .allocate(tmps.at("0166_aaaa_vvoo"))
    
    // flops: o2v2  = o2v1 o2v1 o2v1 o1v1 o2v2 o1v1 o2v2 o3v2 o3v2 o2v2
    //  mems: o2v2  = o2v0 o1v1 o1v1 o1v1 o1v1 o1v1 o2v2 o3v1 o2v2 o2v2
    ( tmps.at("bin1_aa_oo")(ia,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin1_aa_vo")(ba,ia)  = tmps.at("bin1_aa_oo")(ia,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("bin1_aa_vo")(ba,ia) += dp.at("aa_oo")(ka,ia) * t1.at("aa")(ba,ka) )
    ( tmps.at("bin1_aa_vo")(ba,ia) += dp.at("aa_ov")(ka,ca) * t2.at("aaaa")(ca,ba,ia,ka) )
    ( tmps.at("0166_aaaa_vvoo")(aa,ba,ja,ia)  = tmps.at("bin1_aa_vo")(ba,ia) * t1_1p.at("aa")(aa,ja) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = tmps.at("0162_aaaa_voov")(ba,ka,ia,ca) * t1.at("aa")(ca,ja) )
    ( tmps.at("0166_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1.at("aa")(aa,ka) )
    
    // r2[aaaa] += -1.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_aa(b,k) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_aaaa(c,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_aa(c,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0166_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2[aaaa] += -1.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_aa(b,k) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_aaaa(c,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_aa(c,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0166_aaaa_vvoo")(ba,aa,ja,ia) )
    
    // r2[aaaa] += -1.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_aa(b,k) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_aaaa(c,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_aa(c,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0166_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2[aaaa] += -1.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_aa(b,k) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_aaaa(c,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_aa(c,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0166_aaaa_vvoo")(aa,ba,ja,ia) )
    .deallocate(tmps.at("0166_aaaa_vvoo"))
    .allocate(tmps.at("0167_aaaa_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o4v0Q1 o4v1 o3v2 o2v1Q1 o4v0Q1 o4v1 o3v2 o2v2 o2v2 o2v1 o2v1 o2v2 o2v2 o2v1 o2v2 o1v1 o2v2 o2v2 o2v1 o1v1 o2v1 o2v1 o1v1 o2v2 o2v2 o2v2 o3v3 o2v1 o2v2 o2v2 o2v2 o3v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o4v0 o3v1 o2v2 o2v0Q1 o4v0 o3v1 o2v2 o2v2 o2v2 o2v0 o1v1 o2v2 o2v2 o1v1 o1v1 o1v1 o2v2 o1v1 o1v1 o1v1 o2v0 o1v1 o1v1 o2v2 o2v2 o2v2 o2v2 o1v1 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1_1p.at("aa")(ca,ja) * tmps.at("0162_aaaa_voov")(ba,ka,ia,ca) )
    ( tmps.at("0167_aaaa_vvoo")(aa,ba,ja,ia)  = t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0053_aa_ooQ")(la,ja,Q) * tmps.at("0053_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1_1p.at("aa")(ba,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0167_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1.at("aa")(aa,ka) )
    ( tmps.at("bin1_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,da,Q) * t1_1p.at("aa")(da,ia) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("bin1_aa_ooQ")(ia,ka,Q) * tmps.at("0053_aa_ooQ")(la,ja,Q) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1.at("aa")(ba,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0167_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1.at("aa")(aa,ka) )
    ( tmps.at("bin1_aa_oo")(ja,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ja) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = tmps.at("bin1_aa_oo")(ja,ka) * t1_1p.at("aa")(aa,ka) )
    ( tmps.at("0167_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aa_vo")(aa,ja) * t1_1p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aa_vo")(ba,ia)  = dp.at("aa_oo")(ka,ia) * t1_1p.at("aa")(ba,ka) )
    ( tmps.at("bin1_aa_vo")(ba,ia) += dp.at("aa_ov")(ka,ca) * t2_1p.at("aaaa")(ca,ba,ia,ka) )
    ( tmps.at("0167_aaaa_vvoo")(aa,ba,ja,ia) += t1_1p.at("aa")(aa,ja) * tmps.at("bin1_aa_vo")(ba,ia) )
    ( tmps.at("bin1_aa_vo")(ba,ia)  = 2.000 * dp.at("aa_ov")(ka,ca) * t2.at("aaaa")(ca,ba,ia,ka) )
    ( tmps.at("bin1_aa_vo")(ba,ia) += 2.000 * dp.at("aa_oo")(ka,ia) * t1.at("aa")(ba,ka) )
    ( tmps.at("bin1_aa_oo")(ia,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin1_aa_vo")(ba,ia) += 2.000 * tmps.at("bin1_aa_oo")(ia,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("0167_aaaa_vvoo")(aa,ba,ja,ia) += t1_2p.at("aa")(aa,ja) * tmps.at("bin1_aa_vo")(ba,ia) )
    ( tmps.at("0167_aaaa_vvoo")(aa,ba,ja,ia) += t2_1p.at("aaaa")(da,aa,ja,la) * tmps.at("0162_aaaa_voov")(ba,la,ia,da) )
    ( tmps.at("bin1_aa_vo")(aa,ja)  = tmps.at("0031_aa_oo")(ka,ja) * t1.at("aa")(aa,ka) )
    ( tmps.at("0167_aaaa_vvoo")(aa,ba,ja,ia) += t1_1p.at("aa")(ba,ia) * tmps.at("bin1_aa_vo")(aa,ja) )
    ( tmps.at("0167_aaaa_vvoo")(aa,ba,ja,ia) += t2_1p.at("abab")(aa,db,ja,lb) * tmps.at("0163_abab_voov")(ba,lb,ia,db) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += +1.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_1p_aa(b,k) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_1p_aaaa(c,b,j,k) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,i) t1_2p_aa(a,j) t1_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t2_aaaa(c,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_1p_aaaa(d,b,j,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_1p_abab(b,d,j,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0167_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += +1.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_1p_aa(b,k) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_1p_aaaa(c,b,j,k) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,i) t1_2p_aa(a,j) t1_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t2_aaaa(c,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_1p_aaaa(d,b,j,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_1p_abab(b,d,j,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0167_aaaa_vvoo")(ba,aa,ja,ia) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += +1.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_1p_aa(b,k) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_1p_aaaa(c,b,j,k) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,i) t1_2p_aa(a,j) t1_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t2_aaaa(c,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t2_1p_aaaa(d,a,i,l) t2_aaaa(c,b,j,k) 
    //               += -1.000 P(i,j) <l,k||c,d>_bbbb t2_1p_abab(a,d,i,l) t2_abab(b,c,j,k) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0167_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += +1.000 P(i,j) P(a,b) d-_aa(k,i) t1_1p_aa(a,j) t1_1p_aa(b,k) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t2_1p_aaaa(c,b,j,k) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,i) t1_2p_aa(a,j) t1_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t2_aaaa(c,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_aa(a,i) t1_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t2_1p_aaaa(d,a,i,l) t2_aaaa(c,b,j,k) 
    //               += -1.000 P(i,j) <l,k||c,d>_bbbb t2_1p_abab(a,d,i,l) t2_abab(b,c,j,k) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(b,l) t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_aa(a,i) t1_aa(b,k) t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0167_aaaa_vvoo")(aa,ba,ja,ia) )
    .deallocate(tmps.at("0167_aaaa_vvoo"))
    .allocate(tmps.at("0168_aa_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0168_aa_vv")(ca,aa)  = tmps.at("0073_aaaa_ovov")(la,ca,ka,da) * t2_1p.at("aaaa")(da,aa,ka,la) )
    
    // r1_1p[aa] += +0.500 <j,k||b,c>_aaaa t1_aa(b,i) t2_1p_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.500 * t1.at("aa")(ba,ia) * tmps.at("0168_aa_vv")(ba,aa) )
    
    // r1_2p[aa] += +1.000 <j,k||b,c>_aaaa t1_1p_aa(b,i) t2_1p_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= t1_1p.at("aa")(ba,ia) * tmps.at("0168_aa_vv")(ba,aa) )
    
    // r2_1p[abab] += +0.500 <k,l||c,d>_aaaa t2_1p_aaaa(d,a,k,l) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.500 * t2.at("abab")(ca,bb,ia,jb) * tmps.at("0168_aa_vv")(ca,aa) )
    
    // r2_2p[aaaa] += -1.000 P(a,b) <k,l||d,c>_aaaa t2_1p_aaaa(d,a,i,j) t2_1p_aaaa(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0168_aa_vv")(da,aa) * t2_1p.at("aaaa")(da,ba,ia,ja) )
    
    // r2_2p[abab] += +1.000 <k,l||d,c>_aaaa t2_1p_aaaa(c,a,k,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("0168_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    .allocate(tmps.at("0169_aa_vv"))
    
    // flops: o0v2  = o2v2Q1 o1v2Q1
    //  mems: o0v2  = o1v1Q1 o0v2
    ( tmps.at("bin1_aa_voQ")(aa,ka,Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t2.at("abab")(aa,cb,ka,lb) )
    ( tmps.at("0169_aa_vv")(da,aa)  = chol.at("aa_ovQ")(ka,da,Q) * tmps.at("bin1_aa_voQ")(aa,ka,Q) )
    
    // r2[abab] += -0.500 <k,l||d,c>_abab t2_abab(a,c,k,l) t2_abab(d,b,i,j) 
    //            += -0.500 <l,k||d,c>_abab t2_abab(a,c,l,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(da,bb,ia,jb) * tmps.at("0169_aa_vv")(da,aa) )
    
    // r2_1p[abab] += -0.500 <k,l||d,c>_abab t2_abab(a,c,k,l) t2_1p_abab(d,b,i,j) 
    //               += -0.500 <l,k||d,c>_abab t2_abab(a,c,l,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("0169_aa_vv")(da,aa) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_abab(a,c,k,l) t2_2p_abab(d,b,i,j) 
    //               += -1.000 <l,k||d,c>_abab t2_abab(a,c,l,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_2p.at("abab")(da,bb,ia,jb) * tmps.at("0169_aa_vv")(da,aa) )
    .allocate(tmps.at("0170_aa_vv"))
    
    // flops: o0v2  = o1v2Q1
    //  mems: o0v2  = o0v2
    ( tmps.at("0170_aa_vv")(ca,aa)  = chol.at("aa_ovQ")(ka,ca,Q) * tmps.at("0028_aa_voQ")(aa,ka,Q) )
    
    // r2_1p[abab] += -0.500 <k,l||c,d>_abab t2_1p_abab(a,d,k,l) t2_abab(c,b,i,j) 
    //               += -0.500 <l,k||c,d>_abab t2_1p_abab(a,d,l,k) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(ca,bb,ia,jb) * tmps.at("0170_aa_vv")(ca,aa) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_1p_abab(a,c,k,l) t2_1p_abab(d,b,i,j) 
    //               += -1.000 <l,k||d,c>_abab t2_1p_abab(a,c,l,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("0170_aa_vv")(da,aa) )
    .allocate(tmps.at("0171_aa_vo"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0171_aa_vo")(ca,ka)  = tmps.at("0073_aaaa_ovov")(ja,ca,ka,ba) * t1_1p.at("aa")(ba,ja) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_1p_aa(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0171_aa_vo")(ca,ka) * t2_1p.at("aaaa")(ca,aa,ia,ka) )
    
    // r1_2p[bb] += -2.000 <k,j||b,c>_aaaa t1_1p_aa(b,j) t2_1p_abab(c,a,k,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0171_aa_vo")(ca,ka) * t2_1p.at("abab")(ca,ab,ka,ib) )
    
    // r2_1p[abab] += +1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,ia,jb) * tmps.at("0171_aa_vo")(da,ka) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("0171_aa_vo")(da,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0172_aaaa_vvoo"))
    
    // flops: o2v2  = o4v2 o4v1 o3v2 o1v1Q1 o0v2Q1 o2v3 o0v2Q1 o2v3 o2v2 o0v2Q1 o2v3 o2v2 o2v2 o2v3 o1v1Q1 o0v2Q1 o2v3 o2v3 o2v2 o2v2 o2v2 o3v2 o3v2 o2v2 o2v3 o2v2 o2v3 o2v2
    //  mems: o2v2  = o4v0 o3v1 o2v2 o0v0Q1 o0v2 o2v2 o0v2 o2v2 o2v2 o0v2 o2v2 o2v2 o2v2 o2v2 o0v0Q1 o0v2 o2v2 o2v2 o2v2 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0073_aaaa_ovov")(la,da,ka,ca) * t2.at("aaaa")(da,ca,ia,ja) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) * t1.at("aa")(ba,ka) )
    ( tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja)  = 0.500 * t1_1p.at("aa")(aa,la) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vv")(aa,da) * t2_1p.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = tmps.at("0049_Q")(Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja) += t2_1p.at("aaaa")(da,ba,ia,ja) * tmps.at("bin1_aa_vv")(aa,da) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = tmps.at("0151_Q")(Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja) += t2.at("aaaa")(da,ba,ia,ja) * tmps.at("bin1_aa_vv")(aa,da) )
    ( tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja) += t2_1p.at("aaaa")(da,aa,ia,ja) * tmps.at("0169_aa_vv")(da,ba) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1_1p.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vv")(aa,da) * t2.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja) += 0.500 * t2.at("aaaa")(ca,aa,ia,ja) * tmps.at("0168_aa_vv")(ca,ba) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t2.at("aaaa")(da,ba,ia,ja) * tmps.at("0171_aa_vo")(da,ka) )
    ( tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1.at("aa")(aa,ka) )
    ( tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja) += t2.at("aaaa")(ca,aa,ia,ja) * tmps.at("0170_aa_vv")(ca,ba) )
    ( tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja) += f.at("aa_vv")(aa,ca) * t2_1p.at("aaaa")(ca,ba,ia,ja) )
    .deallocate(tmps.at("0168_aa_vv"))
    
    // r2_1p[aaaa] += +1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_aaaa(d,b,i,j) 
    //               += -0.500 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t2_aaaa(d,c,i,j) 
    //               += +1.000 P(a,b) f_aa(a,c) t2_1p_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) <a,k||c,d>_aaaa t1_aa(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <a,k||d,c>_aaaa t1_1p_aa(c,k) t2_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <a,k||d,c>_abab t1_bb(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <a,k||d,c>_abab t1_1p_bb(c,k) t2_aaaa(d,b,i,j) 
    //               += +0.500 P(a,b) <k,l||d,c>_abab t2_1p_aaaa(d,a,i,j) t2_abab(b,c,k,l) 
    //               += +0.500 P(a,b) <l,k||d,c>_abab t2_1p_aaaa(d,a,i,j) t2_abab(b,c,l,k) 
    //               += +0.500 P(a,b) <k,l||c,d>_aaaa t2_1p_aaaa(d,a,k,l) t2_aaaa(c,b,i,j) 
    //               += -0.500 P(a,b) <k,l||c,d>_abab t2_1p_abab(a,d,k,l) t2_aaaa(c,b,i,j) 
    //               += -0.500 P(a,b) <l,k||c,d>_abab t2_1p_abab(a,d,l,k) t2_aaaa(c,b,i,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0172_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_1p[aaaa] += +1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_aaaa(d,b,i,j) 
    //               += -0.500 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t2_aaaa(d,c,i,j) 
    //               += +1.000 P(a,b) f_aa(a,c) t2_1p_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) <a,k||c,d>_aaaa t1_aa(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <a,k||d,c>_aaaa t1_1p_aa(c,k) t2_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <a,k||d,c>_abab t1_bb(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +1.000 P(a,b) <a,k||d,c>_abab t1_1p_bb(c,k) t2_aaaa(d,b,i,j) 
    //               += +0.500 P(a,b) <k,l||d,c>_abab t2_1p_aaaa(d,a,i,j) t2_abab(b,c,k,l) 
    //               += +0.500 P(a,b) <l,k||d,c>_abab t2_1p_aaaa(d,a,i,j) t2_abab(b,c,l,k) 
    //               += +0.500 P(a,b) <k,l||c,d>_aaaa t2_1p_aaaa(d,a,k,l) t2_aaaa(c,b,i,j) 
    //               += -0.500 P(a,b) <k,l||c,d>_abab t2_1p_abab(a,d,k,l) t2_aaaa(c,b,i,j) 
    //               += -0.500 P(a,b) <l,k||c,d>_abab t2_1p_abab(a,d,l,k) t2_aaaa(c,b,i,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0172_aaaa_vvoo")(aa,ba,ia,ja) )
    .deallocate(tmps.at("0172_aaaa_vvoo"))
    .allocate(tmps.at("0173_aa_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0173_aa_vv")(ca,aa)  = tmps.at("0073_aaaa_ovov")(la,ca,ka,da) * t2_2p.at("aaaa")(da,aa,ka,la) )
    
    // r1_2p[aa] += +1.000 <j,k||b,c>_aaaa t1_aa(b,i) t2_2p_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= t1.at("aa")(ba,ia) * tmps.at("0173_aa_vv")(ba,aa) )
    
    // r2_2p[abab] += +1.000 <k,l||c,d>_aaaa t2_2p_aaaa(d,a,k,l) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(ca,bb,ia,jb) * tmps.at("0173_aa_vv")(ca,aa) )
    .allocate(tmps.at("0174_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0174_Q")(Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t1_2p.at("aa")(ba,ja) )
    
    // r1_2p[aa] += +2.000 <a,j||i,b>_aaaa t1_2p_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0174_Q")(Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,ka)  = chol.at("aa_ovQ")(ka,da,Q) * tmps.at("0174_Q")(Q) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin1_aa_vo")(da,ka) * t2.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 <a,k||d,c>_aaaa t1_2p_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o0v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_aa_vv")(aa,da)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("0174_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aa_vv")(aa,da) * t2.at("abab")(da,bb,ia,jb) )
    
    // r1_2p[aa] += +2.000 <a,j||b,c>_aaaa t1_aa(b,i) t1_2p_aa(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0174_Q")(Q) )
    
    // r1_2p[bb] += -2.000 <j,k||b,c>_abab t1_2p_aa(b,j) t2_bbbb(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0059_bb_voQ")(ab,ib,Q) * tmps.at("0174_Q")(Q) )
    
    // r1_2p[bb] += +2.000 <j,a||b,i>_abab t1_2p_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0174_Q")(Q) * chol.at("bb_voQ")(ab,ib,Q) )
    
    // r1_2p[aa] += +2.000 <j,k||b,c>_abab t1_2p_aa(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0174_Q")(Q) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_abab t1_bb(b,k) t1_2p_aa(c,l) t2_abab(a,d,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,kb)  = chol.at("bb_ovQ")(kb,db,Q) * tmps.at("0174_Q")(Q) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin1_bb_vo")(db,kb) * t2.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r1_2p[bb] += -2.000 <j,k||c,b>_aaaa t1_2p_aa(b,j) t2_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0058_bb_voQ")(ab,ib,Q) * tmps.at("0174_Q")(Q) )
    
    // r1_2p[bb] += +2.000 <j,a||c,b>_abab t1_bb(b,i) t1_2p_aa(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0060_bb_voQ")(ab,ib,Q) * tmps.at("0174_Q")(Q) )
    
    // r1_2p[aa] += +2.000 <j,k||c,b>_aaaa t1_2p_aa(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0174_Q")(Q) )
    .allocate(tmps.at("0175_aa_ov"))
    
    // flops: o1v1  = o1v1Q1 o1v1Q1
    //  mems: o1v1  = o0v0Q1 o1v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t1.at("aa")(ba,ja) )
    ( tmps.at("0175_aa_ov")(ka,ca)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,la,jb)  = tmps.at("0175_aa_ov")(la,da) * t2_2p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,la) * tmps.at("bin1_baab_vooo")(bb,ia,la,jb) )
    .allocate(tmps.at("0176_aaaa_vvoo"))
    
    // flops: o2v2  = o3v3 o3v2 o0v2Q1 o2v3 o2v2 o0v2Q1 o2v3 o3v2 o3v2 o2v2 o2v2 o0v2Q1 o2v3 o2v2 o1v1Q1 o0v2Q1 o2v3 o2v2 o1v1Q1 o0v2Q1 o2v3 o2v2 o1v1Q1 o0v2Q1 o2v3 o2v2 o3v2 o3v2 o2v2 o4v2 o4v1 o3v2 o2v2 o3v2 o3v2 o2v3 o2v2 o2v2 o4v2 o4v1 o3v2 o2v2 o3v2 o3v2 o2v2 o2v2 o3v2 o3v2 o2v2 o2v3 o2v2 o1v2Q1 o2v3 o2v2 o2v3 o2v2 o3v2 o3v2 o2v2 o2v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o0v2 o2v2 o2v2 o0v2 o2v2 o3v1 o2v2 o2v2 o2v2 o0v2 o2v2 o2v2 o0v0Q1 o0v2 o2v2 o2v2 o0v0Q1 o0v2 o2v2 o2v2 o0v0Q1 o0v2 o2v2 o2v2 o3v1 o2v2 o2v2 o4v0 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o4v0 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o1v1 o3v1 o2v2 o2v2 o2v2 o2v2 o0v2 o2v2 o2v2 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("0090_aaaa_ovvv")(ka,da,aa,ca) * t2_2p.at("aaaa")(da,ca,ia,ja) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja)  = 0.500 * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("0151_Q")(Q) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vv")(aa,da) * t2_1p.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = tmps.at("0049_Q")(Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += t2_2p.at("aaaa")(da,ba,ia,ja) * tmps.at("bin1_aa_vv")(aa,da) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = t2_2p.at("aaaa")(da,aa,ia,ja) * tmps.at("0175_aa_ov")(la,da) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += t1.at("aa")(ba,la) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("0174_Q")(Q) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vv")(aa,da) * t2.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vv")(aa,da) * t2_2p.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1_1p.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vv")(aa,da) * t2_1p.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1_2p.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vv")(aa,da) * t2.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = dp.at("aa_ov")(ka,ca) * t2_1p.at("aaaa")(ca,aa,ia,ja) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += 3.000 * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1_2p.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0073_aaaa_ovov")(la,da,ka,ca) * t2.at("aaaa")(da,ca,ia,ja) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) * t1.at("aa")(ba,ka) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += 0.500 * t1_2p.at("aa")(aa,la) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = dp.at("aa_ov")(ka,ca) * t2_2p.at("aaaa")(ca,aa,ia,ja) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += 3.000 * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1_1p.at("aa")(ba,ka) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += t2_2p.at("aaaa")(da,aa,ia,ja) * tmps.at("0169_aa_vv")(da,ba) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0073_aaaa_ovov")(la,da,ka,ca) * t2_1p.at("aaaa")(da,ca,ia,ja) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) * t1.at("aa")(ba,ka) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += 0.500 * t1_1p.at("aa")(aa,la) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = tmps.at("0171_aa_vo")(da,ka) * t2_1p.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("bin1_aa_vo")(da,ka)  = tmps.at("0073_aaaa_ovov")(la,da,ka,ca) * t1_2p.at("aa")(ca,la) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = tmps.at("bin1_aa_vo")(da,ka) * t2.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += 0.500 * t2.at("aaaa")(ca,aa,ia,ja) * tmps.at("0173_aa_vv")(ca,ba) )
    ( tmps.at("bin1_aa_vv")(ba,ca)  = chol.at("aa_ovQ")(ka,ca,Q) * tmps.at("0144_aa_voQ")(ba,ka,Q) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vv")(ba,ca) * t2.at("aaaa")(ca,aa,ia,ja) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += t2_1p.at("aaaa")(da,aa,ia,ja) * tmps.at("0170_aa_vv")(da,ba) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = f.at("aa_ov")(ka,ca) * t2_2p.at("aaaa")(ca,aa,ia,ja) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) += f.at("aa_vv")(aa,ca) * t2_2p.at("aaaa")(ca,ba,ia,ja) )
    .deallocate(tmps.at("0173_aa_vv"))
    .deallocate(tmps.at("0171_aa_vo"))
    .deallocate(tmps.at("0170_aa_vv"))
    
    // r2_2p[aaaa] += -2.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_1p_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,l) t2_aaaa(d,b,i,j) 
    //               += -1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t2_1p_aaaa(d,c,i,j) 
    //               += -1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t2_aaaa(d,c,i,j) 
    //               += +2.000 P(a,b) f_aa(a,c) t2_2p_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) <a,k||d,c>_aaaa t1_aa(b,k) t2_2p_aaaa(d,c,i,j) 
    //               += -2.000 P(a,b) <a,k||c,d>_aaaa t1_aa(c,k) t2_2p_aaaa(d,b,i,j) 
    //               += -2.000 P(a,b) <a,k||c,d>_aaaa t1_1p_aa(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_2p_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <a,k||d,c>_aaaa t1_2p_aa(c,k) t2_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <a,k||d,c>_abab t1_bb(c,k) t2_2p_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <a,k||d,c>_abab t1_1p_bb(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <a,k||d,c>_abab t1_2p_bb(c,k) t2_aaaa(d,b,i,j) 
    //               += -6.000 P(a,b) d-_aa(k,c) t1_2p_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    //               += -6.000 P(a,b) d-_aa(k,c) t1_1p_aa(a,k) t2_2p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) f_aa(k,c) t1_aa(a,k) t2_2p_aaaa(c,b,i,j) 
    //               += +1.000 P(a,b) <k,l||d,c>_abab t2_2p_aaaa(d,a,i,j) t2_abab(b,c,k,l) 
    //               += +1.000 P(a,b) <l,k||d,c>_abab t2_2p_aaaa(d,a,i,j) t2_abab(b,c,l,k) 
    //               += +1.000 P(a,b) <k,l||c,d>_aaaa t2_2p_aaaa(d,a,k,l) t2_aaaa(c,b,i,j) 
    //               += +1.000 P(a,b) <k,l||d,c>_abab t2_1p_aaaa(d,a,i,j) t2_1p_abab(b,c,k,l) 
    //               += +1.000 P(a,b) <l,k||d,c>_abab t2_1p_aaaa(d,a,i,j) t2_1p_abab(b,c,l,k) 
    //               += -1.000 P(a,b) <k,l||c,d>_abab t2_2p_abab(a,d,k,l) t2_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) <l,k||c,d>_abab t2_2p_abab(a,d,l,k) t2_aaaa(c,b,i,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0176_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_2p[aaaa] += -2.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_1p_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,l) t2_aaaa(d,b,i,j) 
    //               += -1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t2_1p_aaaa(d,c,i,j) 
    //               += -1.000 P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t2_aaaa(d,c,i,j) 
    //               += +2.000 P(a,b) f_aa(a,c) t2_2p_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) <a,k||d,c>_aaaa t1_aa(b,k) t2_2p_aaaa(d,c,i,j) 
    //               += -2.000 P(a,b) <a,k||c,d>_aaaa t1_aa(c,k) t2_2p_aaaa(d,b,i,j) 
    //               += -2.000 P(a,b) <a,k||c,d>_aaaa t1_1p_aa(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_2p_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <a,k||d,c>_aaaa t1_2p_aa(c,k) t2_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <a,k||d,c>_abab t1_bb(c,k) t2_2p_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <a,k||d,c>_abab t1_1p_bb(c,k) t2_1p_aaaa(d,b,i,j) 
    //               += +2.000 P(a,b) <a,k||d,c>_abab t1_2p_bb(c,k) t2_aaaa(d,b,i,j) 
    //               += -6.000 P(a,b) d-_aa(k,c) t1_2p_aa(a,k) t2_1p_aaaa(c,b,i,j) 
    //               += -6.000 P(a,b) d-_aa(k,c) t1_1p_aa(a,k) t2_2p_aaaa(c,b,i,j) 
    //               += -2.000 P(a,b) f_aa(k,c) t1_aa(a,k) t2_2p_aaaa(c,b,i,j) 
    //               += +1.000 P(a,b) <k,l||d,c>_abab t2_2p_aaaa(d,a,i,j) t2_abab(b,c,k,l) 
    //               += +1.000 P(a,b) <l,k||d,c>_abab t2_2p_aaaa(d,a,i,j) t2_abab(b,c,l,k) 
    //               += +1.000 P(a,b) <k,l||c,d>_aaaa t2_2p_aaaa(d,a,k,l) t2_aaaa(c,b,i,j) 
    //               += +1.000 P(a,b) <k,l||d,c>_abab t2_1p_aaaa(d,a,i,j) t2_1p_abab(b,c,k,l) 
    //               += +1.000 P(a,b) <l,k||d,c>_abab t2_1p_aaaa(d,a,i,j) t2_1p_abab(b,c,l,k) 
    //               += -1.000 P(a,b) <k,l||c,d>_abab t2_2p_abab(a,d,k,l) t2_aaaa(c,b,i,j) 
    //               += -1.000 P(a,b) <l,k||c,d>_abab t2_2p_abab(a,d,l,k) t2_aaaa(c,b,i,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0176_aaaa_vvoo")(aa,ba,ia,ja) )
    .deallocate(tmps.at("0176_aaaa_vvoo"))
    .allocate(tmps.at("0177_baba_voov"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin1_aaaa_vvoo")(ca,da,ka,la)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("0177_baba_voov")(ab,la,ib,da)  = t2.at("abab")(ca,ab,ka,ib) * tmps.at("bin1_aaaa_vvoo")(ca,da,ka,la) )
    
    // r1[bb] += -1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_abab(c,a,k,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= t1.at("aa")(ba,ja) * tmps.at("0177_baba_voov")(ab,ja,ib,ba) )
    
    // r1_1p[bb] += -1.000 <j,k||c,b>_aaaa t1_1p_aa(b,j) t2_abab(c,a,k,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= t1_1p.at("aa")(ba,ja) * tmps.at("0177_baba_voov")(ab,ja,ib,ba) )
    
    // r1_2p[bb] += -2.000 <j,k||c,b>_aaaa t1_2p_aa(b,j) t2_abab(c,a,k,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * t1_2p.at("aa")(ba,ja) * tmps.at("0177_baba_voov")(ab,ja,ib,ba) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t2_1p_aaaa(d,a,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("aaaa")(da,aa,ia,la) * tmps.at("0177_baba_voov")(bb,la,jb,da) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t1_2p.at("aa")(ca,ia) * tmps.at("0177_baba_voov")(bb,ka,jb,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_2p_aaaa(d,a,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("aaaa")(da,aa,ia,la) * tmps.at("0177_baba_voov")(bb,la,jb,da) )
    .allocate(tmps.at("0178_baab_vooo"))
    
    // flops: o3v1  = o3v2 o4v2 o3v2 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1
    ( tmps.at("bin1_aabb_vooo")(da,ka,jb,lb)  = t1.at("bb")(cb,jb) * tmps.at("0083_aabb_ovov")(ka,da,lb,cb) )
    ( tmps.at("0178_baab_vooo")(bb,ka,ia,jb)  = tmps.at("bin1_aabb_vooo")(da,ka,jb,lb) * t2.at("abab")(da,bb,ia,lb) )
    ( tmps.at("0178_baab_vooo")(bb,ka,ia,jb) += tmps.at("0177_baba_voov")(bb,ka,jb,ca) * t1.at("aa")(ca,ia) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    //            += +1.000 <k,l||d,c>_abab t1_aa(a,k) t1_bb(c,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0178_baab_vooo")(bb,ka,ia,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += -1.000 <k,l||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    //               += +1.000 <k,l||d,c>_abab t1_1p_aa(a,k) t1_bb(c,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0178_baab_vooo")(bb,ka,ia,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_aaaa t1_2p_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    //               += +2.000 <k,l||d,c>_abab t1_2p_aa(a,k) t1_bb(c,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0178_baab_vooo")(bb,ka,ia,jb) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0178_baab_vooo"))
    .allocate(tmps.at("0179_aabb_vvoo"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0179_aabb_vvoo")(aa,ca,kb,jb)  = chol.at("aa_vvQ")(aa,ca,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    
    // r2[abab] += -1.000 <a,k||c,j>_abab t2_abab(c,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0179_aabb_vvoo")(aa,ca,kb,jb) * t2.at("abab")(ca,bb,ia,kb) )
    
    // r2_1p[abab] += -1.000 <a,k||c,j>_abab t2_1p_abab(c,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0179_aabb_vvoo")(aa,ca,kb,jb) * t2_1p.at("abab")(ca,bb,ia,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||c,j>_abab t2_2p_abab(c,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0179_aabb_vvoo")(aa,ca,kb,jb) * t2_2p.at("abab")(ca,bb,ia,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||c,j>_abab t1_bb(b,k) t1_2p_aa(c,i) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t1_2p.at("aa")(ca,ia) * tmps.at("0179_aabb_vvoo")(aa,ca,kb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0180_aabb_vooo"))
    
    // flops: o3v1  = o3v1Q1 o3v1Q1 o3v1 o3v1Q1 o3v1 o3v2 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("0180_aabb_vooo")(aa,ia,kb,jb)  = tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0026_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0180_aabb_vooo")(aa,ia,kb,jb) -= tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0026_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0180_aabb_vooo")(aa,ia,kb,jb) -= tmps.at("0051_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0180_aabb_vooo")(aa,ia,kb,jb) += t1.at("aa")(ca,ia) * tmps.at("0179_aabb_vvoo")(aa,ca,kb,jb) )
    
    // r2[abab] += -1.000 <a,k||c,j>_abab t1_bb(b,k) t1_aa(c,i) 
    //            += +1.000 <l,k||c,j>_abab t1_bb(b,k) t2_aaaa(c,a,i,l) 
    //            += +1.000 <l,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t2_aaaa(d,a,i,l) 
    //            += -1.000 <a,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t1_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0180_aabb_vooo")(aa,ia,kb,jb) * t1.at("bb")(bb,kb) )
    
    // r2_1p[abab] += -1.000 <a,k||c,j>_abab t1_1p_bb(b,k) t1_aa(c,i) 
    //               += +1.000 <l,k||c,j>_abab t1_1p_bb(b,k) t2_aaaa(c,a,i,l) 
    //               += +1.000 <l,k||d,c>_abab t1_1p_bb(b,k) t1_bb(c,j) t2_aaaa(d,a,i,l) 
    //               += -1.000 <a,k||c,d>_abab t1_1p_bb(b,k) t1_aa(c,i) t1_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0180_aabb_vooo")(aa,ia,kb,jb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||c,j>_abab t1_2p_bb(b,k) t1_aa(c,i) 
    //               += +2.000 <l,k||c,j>_abab t1_2p_bb(b,k) t2_aaaa(c,a,i,l) 
    //               += +2.000 <l,k||d,c>_abab t1_2p_bb(b,k) t1_bb(c,j) t2_aaaa(d,a,i,l) 
    //               += -2.000 <a,k||c,d>_abab t1_2p_bb(b,k) t1_aa(c,i) t1_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0180_aabb_vooo")(aa,ia,kb,jb) * t1_2p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0180_aabb_vooo"))
    .allocate(tmps.at("0181_bb_vv"))
    
    // flops: o0v2  = o2v2Q1 o1v2Q1
    //  mems: o0v2  = o1v1Q1 o0v2
    ( tmps.at("bin1_bb_voQ")(ab,lb,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t2_1p.at("abab")(ca,ab,ka,lb) )
    ( tmps.at("0181_bb_vv")(ab,db)  = tmps.at("bin1_bb_voQ")(ab,lb,Q) * chol.at("bb_ovQ")(lb,db,Q) )
    
    // r2_1p[abab] += -0.500 <k,l||d,c>_abab t2_abab(a,c,i,j) t2_1p_abab(d,b,k,l) 
    //               += -0.500 <l,k||d,c>_abab t2_abab(a,c,i,j) t2_1p_abab(d,b,l,k) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0181_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p[abab] += -1.000 <k,l||c,d>_abab t2_1p_abab(a,d,i,j) t2_1p_abab(c,b,k,l) 
    //               += -1.000 <l,k||c,d>_abab t2_1p_abab(a,d,i,j) t2_1p_abab(c,b,l,k) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0181_bb_vv")(bb,db) * t2_1p.at("abab")(aa,db,ia,jb) )
    .allocate(tmps.at("0183_bb_vv"))
    
    // flops: o0v2  = o0v2Q1 o0v2Q1 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0183_bb_vv")(ab,db)  = chol.at("bb_vvQ")(ab,db,Q) * tmps.at("0150_Q")(Q) )
    ( tmps.at("0183_bb_vv")(ab,db) += chol.at("bb_vvQ")(ab,db,Q) * tmps.at("0151_Q")(Q) )
    
    // r2_1p[abab] += +1.000 <b,k||d,c>_bbbb t1_1p_bb(c,k) t2_abab(a,d,i,j) 
    //               += +1.000 <k,b||c,d>_abab t1_1p_aa(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0183_bb_vv")(bb,db) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_2p[abab] += -2.000 <b,k||c,d>_bbbb t1_1p_bb(c,k) t2_1p_abab(a,d,i,j) 
    //               += +2.000 <k,b||c,d>_abab t1_1p_aa(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0183_bb_vv")(bb,db) * t2_1p.at("abab")(aa,db,ia,jb) )
    .allocate(tmps.at("0182_bb_vv"))
    
    // flops: o0v2  = o0v2Q1 o0v2Q1 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0182_bb_vv")(ab,db)  = chol.at("bb_vvQ")(ab,db,Q) * tmps.at("0148_Q")(Q) )
    ( tmps.at("0182_bb_vv")(ab,db) += chol.at("bb_vvQ")(ab,db,Q) * tmps.at("0049_Q")(Q) )
    .allocate(tmps.at("0184_bbbb_vvoo"))
    
    // flops: o2v2  = o1v1Q1 o1v1Q1 o3v2 o3v2 o2v3 o2v3 o4v2 o4v1 o3v2 o2v2 o2v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o2v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o2v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o2v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o2v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o2v2 o2v3 o2v2 o2v3 o2v2 o2v2 o3v2 o3v2 o2v2 o2v2 o3v2 o3v2 o2v2 o1v2Q1 o2v3 o2v2 o2v3 o2v2
    //  mems: o2v2  = o0v0Q1 o1v1 o3v1 o2v2 o0v2 o2v2 o4v0 o3v1 o2v2 o2v2 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o1v1 o3v1 o2v2 o2v2 o1v1 o3v1 o2v2 o2v2 o0v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = tmps.at("bin1_bb_vo")(db,lb) * t2.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb)  = tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) * t1_1p.at("bb")(bb,lb) )
    ( tmps.at("bin1_bb_vv")(ab,cb)  = tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) * t2_1p.at("bbbb")(db,ab,kb,lb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += 0.500 * tmps.at("bin1_bb_vv")(ab,cb) * t2.at("bbbb")(cb,bb,ib,jb) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t2.at("bbbb")(db,cb,ib,jb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t1_1p.at("bb")(bb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += 0.500 * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1.at("bb")(ab,kb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = tmps.at("bin1_bb_vo")(db,lb) * t2_1p.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) * t1.at("bb")(bb,lb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1_1p.at("aa")(ca,la) )
    ( tmps.at("bin1_bb_vo")(db,kb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = tmps.at("bin1_bb_vo")(db,kb) * t2.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = tmps.at("bin1_bb_vo")(db,lb) * t2.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) * t1_1p.at("bb")(bb,lb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = tmps.at("bin1_bb_vo")(db,lb) * t2_1p.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) * t1.at("bb")(bb,lb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t1_1p.at("bb")(cb,lb) )
    ( tmps.at("bin1_bb_vo")(db,kb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = tmps.at("bin1_bb_vo")(db,kb) * t2.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("0183_bb_vv")(ab,db) * t2.at("bbbb")(db,bb,ib,jb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("0182_bb_vv")(ab,db) * t2_1p.at("bbbb")(db,bb,ib,jb) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,lb)  = t2.at("bbbb")(db,bb,ib,jb) * tmps.at("bin1_bb_vo")(db,lb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += t1_1p.at("bb")(ab,lb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,lb) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,lb)  = t2_1p.at("bbbb")(db,bb,ib,jb) * tmps.at("bin1_bb_vo")(db,lb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += t1.at("bb")(ab,lb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,lb) )
    ( tmps.at("bin1_bb_vv")(bb,db)  = chol.at("bb_ovQ")(lb,db,Q) * tmps.at("0058_bb_voQ")(bb,lb,Q) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("bin1_bb_vv")(bb,db) * t2_1p.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) += t2.at("bbbb")(cb,ab,ib,jb) * tmps.at("0181_bb_vv")(bb,cb) )
    
    // r2_1p[bbbb] += +1.000 P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_1p_bb(c,l) t2_bbbb(d,b,i,j) 
    //               += -0.500 P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_1p_bb(b,l) t2_bbbb(d,c,i,j) 
    //               += +0.500 P(a,b) <k,l||c,d>_bbbb t2_1p_bbbb(d,a,k,l) t2_bbbb(c,b,i,j) 
    //               += +1.000 P(a,b) <a,k||d,c>_bbbb t1_1p_bb(c,k) t2_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <k,a||c,d>_abab t1_1p_aa(c,k) t2_bbbb(d,b,i,j) 
    //               += -1.000 P(a,b) <a,k||c,d>_bbbb t1_bb(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <k,a||c,d>_abab t1_aa(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,l) t1_bb(c,k) t2_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_bbbb t1_bb(a,l) t1_bb(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += -1.000 P(a,b) <l,k||c,d>_abab t1_bb(a,k) t1_1p_aa(c,l) t2_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_bbbb t1_bb(a,l) t1_bb(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += -1.000 P(a,b) <k,l||c,d>_abab t1_bb(a,l) t1_aa(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,l) t1_bb(c,k) t2_bbbb(d,b,i,j) 
    //               += -1.000 P(a,b) <k,l||c,d>_abab t1_1p_bb(a,l) t1_aa(c,k) t2_bbbb(d,b,i,j) 
    //               += +0.500 P(a,b) <k,l||c,d>_abab t2_1p_bbbb(d,a,i,j) t2_abab(c,b,k,l) 
    //               += +0.500 P(a,b) <l,k||c,d>_abab t2_1p_bbbb(d,a,i,j) t2_abab(c,b,l,k) 
    //               += -0.500 P(a,b) <k,l||d,c>_abab t2_1p_abab(d,a,k,l) t2_bbbb(c,b,i,j) 
    //               += -0.500 P(a,b) <l,k||d,c>_abab t2_1p_abab(d,a,l,k) t2_bbbb(c,b,i,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0184_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_1p[bbbb] += +1.000 P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_1p_bb(c,l) t2_bbbb(d,b,i,j) 
    //               += -0.500 P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_1p_bb(b,l) t2_bbbb(d,c,i,j) 
    //               += +0.500 P(a,b) <k,l||c,d>_bbbb t2_1p_bbbb(d,a,k,l) t2_bbbb(c,b,i,j) 
    //               += +1.000 P(a,b) <a,k||d,c>_bbbb t1_1p_bb(c,k) t2_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <k,a||c,d>_abab t1_1p_aa(c,k) t2_bbbb(d,b,i,j) 
    //               += -1.000 P(a,b) <a,k||c,d>_bbbb t1_bb(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <k,a||c,d>_abab t1_aa(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,l) t1_bb(c,k) t2_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_bbbb t1_bb(a,l) t1_bb(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += -1.000 P(a,b) <l,k||c,d>_abab t1_bb(a,k) t1_1p_aa(c,l) t2_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_bbbb t1_bb(a,l) t1_bb(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += -1.000 P(a,b) <k,l||c,d>_abab t1_bb(a,l) t1_aa(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += +1.000 P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,l) t1_bb(c,k) t2_bbbb(d,b,i,j) 
    //               += -1.000 P(a,b) <k,l||c,d>_abab t1_1p_bb(a,l) t1_aa(c,k) t2_bbbb(d,b,i,j) 
    //               += +0.500 P(a,b) <k,l||c,d>_abab t2_1p_bbbb(d,a,i,j) t2_abab(c,b,k,l) 
    //               += +0.500 P(a,b) <l,k||c,d>_abab t2_1p_bbbb(d,a,i,j) t2_abab(c,b,l,k) 
    //               += -0.500 P(a,b) <k,l||d,c>_abab t2_1p_abab(d,a,k,l) t2_bbbb(c,b,i,j) 
    //               += -0.500 P(a,b) <l,k||d,c>_abab t2_1p_abab(d,a,l,k) t2_bbbb(c,b,i,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0184_bbbb_vvoo")(bb,ab,ib,jb) )
    .deallocate(tmps.at("0184_bbbb_vvoo"))
    .allocate(tmps.at("0185_bbbb_voov"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin1_bbbb_vvoo")(cb,db,kb,lb)  = chol.at("bb_ovQ")(lb,cb,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0185_bbbb_voov")(ab,lb,ib,db)  = t2.at("bbbb")(cb,ab,ib,kb) * tmps.at("bin1_bbbb_vvoo")(cb,db,kb,lb) )
    
    // r1[bb] += +1.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_bbbb(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += t1.at("bb")(bb,jb) * tmps.at("0185_bbbb_voov")(ab,jb,ib,bb) )
    
    // r1_1p[bb] += +1.000 <j,k||c,b>_bbbb t1_1p_bb(b,j) t2_bbbb(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += t1_1p.at("bb")(bb,jb) * tmps.at("0185_bbbb_voov")(ab,jb,ib,bb) )
    
    // r1_2p[bb] += +2.000 <j,k||c,b>_bbbb t1_2p_bb(b,j) t2_bbbb(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * t1_2p.at("bb")(bb,jb) * tmps.at("0185_bbbb_voov")(ab,jb,ib,bb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t2_1p_abab(a,d,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,db,ia,lb) * tmps.at("0185_bbbb_voov")(bb,lb,jb,db) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t2_2p_abab(a,d,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("abab")(aa,db,ia,lb) * tmps.at("0185_bbbb_voov")(bb,lb,jb,db) )
    .allocate(tmps.at("0186_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0186_bb_voQ")(bb,ib,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t2_2p.at("abab")(ca,bb,ja,ib) )
    
    // r1_2p[bb] += -1.000 <j,k||b,i>_abab t2_2p_abab(b,a,j,k) 
    //             += -1.000 <k,j||b,i>_abab t2_2p_abab(b,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * chol.at("bb_ooQ")(kb,ib,Q) * tmps.at("0186_bb_voQ")(ab,kb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_abab t2_abab(a,c,i,k) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0186_bb_voQ")(bb,jb,Q) )
    
    // r1_2p[bb] += -1.000 <j,k||c,b>_abab t1_bb(b,i) t2_2p_abab(c,a,j,k) 
    //             += -1.000 <k,j||c,b>_abab t1_bb(b,i) t2_2p_abab(c,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0186_bb_voQ")(ab,kb,Q) * tmps.at("0026_bb_ooQ")(kb,ib,Q) )
    
    // r2_2p[abab] += +2.000 <a,k||i,c>_aaaa t2_2p_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0186_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_abab(c,b,l,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0186_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0186_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_abab(a,c,i,j) t2_2p_abab(d,b,k,l) 
    //               += -1.000 <l,k||d,c>_abab t2_abab(a,c,i,j) t2_2p_abab(d,b,l,k) 
    // flops: o2v2 += o1v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,cb)  = chol.at("bb_ovQ")(lb,cb,Q) * tmps.at("0186_bb_voQ")(bb,lb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <a,k||c,d>_aaaa t1_aa(c,i) t2_2p_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0186_bb_voQ")(bb,jb,Q) )
    
    // r1_2p[bb] += -2.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_2p_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0186_bb_voQ")(ab,ib,Q) * tmps.at("0049_Q")(Q) )
    
    // r1_2p[bb] += +2.000 <k,j||c,b>_abab t1_bb(b,j) t2_2p_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0186_bb_voQ")(ab,ib,Q) * tmps.at("0148_Q")(Q) )
    .allocate(tmps.at("0187_bb_ooQ"))
    
    // flops: o2v0Q1  = o2v1Q1
    //  mems: o2v0Q1  = o2v0Q1
    ( tmps.at("0187_bb_ooQ")(jb,ib,Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t1_2p.at("bb")(bb,ib) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_abab t1_bb(b,k) t1_2p_bb(c,j) t2_aaaa(d,a,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0187_bb_ooQ")(kb,jb,Q) * tmps.at("0051_aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t1_2p_bb(d,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0187_bb_ooQ")(kb,jb,Q) * tmps.at("0052_aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r1_2p[bb] += -1.000 <j,k||c,b>_abab t1_2p_bb(b,i) t2_abab(c,a,j,k) 
    //             += -1.000 <k,j||c,b>_abab t1_2p_bb(b,i) t2_abab(c,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0058_bb_voQ")(ab,kb,Q) * tmps.at("0187_bb_ooQ")(kb,ib,Q) )
    
    // r2_2p[abab] += -2.000 <a,k||i,c>_abab t1_bb(b,k) t1_2p_bb(c,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0187_bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0188_bb_vo"))
    
    // flops: o1v1  = o2v2 o1v2 o1v1
    //  mems: o1v1  = o1v1 o1v1 o1v1
    ( tmps.at("0188_bb_vo")(ab,ib)  = dp.at("aa_ov")(ja,ba) * t2_1p.at("abab")(ba,ab,ja,ib) )
    ( tmps.at("0188_bb_vo")(ab,ib) += dp.at("bb_vv")(ab,bb) * t1_1p.at("bb")(bb,ib) )
    .allocate(tmps.at("0189_bbbb_vvoo"))
    
    // flops: o2v2  = o3v2 o4v2 o3v2 o3v1Q1 o3v1Q1 o3v1 o3v1Q1 o3v1 o3v2 o2v2 o2v2 o2v2 o3v1Q1 o4v2 o3v2 o2v2 o3v1Q1 o3v1Q1 o3v1 o3v1Q1 o3v1 o3v2 o2v2 o2v2 o2v2 o2v2 o1v2 o2v2 o3v1Q1 o4v2 o3v2 o2v2 o2v2 o3v1Q1 o3v1Q1 o3v1 o3v2 o2v2 o3v2 o3v2 o2v2 o3v2 o4v2 o3v2 o2v2
    //  mems: o2v2  = o3v1 o3v1 o2v2 o3v1 o3v1 o3v1 o3v1 o3v1 o2v2 o2v2 o1v1 o2v2 o3v1 o3v1 o2v2 o2v2 o3v1 o3v1 o3v1 o3v1 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o1v1 o2v2 o3v1 o3v1 o2v2 o2v2 o2v2 o3v1 o3v1 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o3v1 o3v1 o2v2 o2v2
    ( tmps.at("bin2_bbbb_vooo")(db,jb,kb,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t1_1p.at("bb")(cb,jb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t2_1p.at("bbbb")(db,bb,ib,lb) * tmps.at("bin2_bbbb_vooo")(db,jb,kb,lb) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb)  = t1.at("bb")(ab,kb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = tmps.at("0135_bb_voQ")(bb,ib,Q) * tmps.at("0029_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) += tmps.at("0058_bb_voQ")(bb,ib,Q) * tmps.at("0187_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) += tmps.at("0186_bb_voQ")(bb,ib,Q) * tmps.at("0026_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) += t1.at("bb")(ab,kb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) += 2.000 * t1_2p.at("bb")(ab,ib) * tmps.at("0188_bb_vo")(bb,jb) )
    ( tmps.at("bin1_bb_vo")(ab,ib)  = dp.at("aa_ov")(ka,ca) * t2_2p.at("abab")(ca,ab,ka,ib) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_vo")(ab,ib) * t1_1p.at("bb")(bb,jb) )
    ( tmps.at("bin1_bbbb_vooo")(cb,ib,kb,lb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("bb_ooQ")(lb,ib,Q) )
    ( tmps.at("bin2_bbbb_vooo")(ab,ib,jb,kb)  = tmps.at("bin1_bbbb_vooo")(cb,ib,kb,lb) * t2_2p.at("bbbb")(cb,ab,jb,lb) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin2_bbbb_vooo")(ab,ib,jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = tmps.at("0058_bb_voQ")(bb,ib,Q) * tmps.at("0029_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) += tmps.at("0135_bb_voQ")(bb,ib,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) += tmps.at("0135_bb_voQ")(bb,ib,Q) * tmps.at("0026_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) += t1_1p.at("bb")(ab,kb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) )
    ( tmps.at("bin1_bb_vo")(ab,ib)  = dp.at("bb_vv")(ab,cb) * t1_2p.at("bb")(cb,ib) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_vo")(ab,ib) * t1_1p.at("bb")(bb,jb) )
    ( tmps.at("bin1_bbbb_vooo")(cb,ib,kb,lb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("bb_ooQ")(lb,ib,Q) )
    ( tmps.at("bin2_bbbb_vooo")(ab,ib,jb,kb)  = tmps.at("bin1_bbbb_vooo")(cb,ib,kb,lb) * t2_1p.at("bbbb")(cb,ab,jb,lb) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin2_bbbb_vooo")(ab,ib,jb,kb) * t1_1p.at("bb")(bb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = tmps.at("0058_bb_voQ")(bb,ib,Q) * tmps.at("0026_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) += tmps.at("0058_bb_voQ")(bb,ib,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) += t1_2p.at("bb")(ab,kb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = t1.at("bb")(cb,ib) * tmps.at("0185_bbbb_voov")(ab,kb,jb,cb) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1_2p.at("bb")(bb,kb) )
    ( tmps.at("bin2_bbbb_vooo")(db,jb,kb,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t1.at("bb")(cb,jb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t2_1p.at("bbbb")(db,bb,ib,lb) * tmps.at("bin2_bbbb_vooo")(db,jb,kb,lb) )
    ( tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) += t1_1p.at("bb")(ab,kb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_2p_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_1p_bb(c,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_bb(c,i) t2_2p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_1p_bb(a,k) t1_1p_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_1p_bb(a,k) t2_1p_abab(c,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_1p_bb(a,k) t1_bb(c,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,k) t1_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_bb(a,i) t2_1p_abab(c,b,k,j) 
    //               += -4.000 P(i,j) P(a,b) d-_bb(a,c) t1_2p_bb(b,i) t1_1p_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_2p_bb(a,k) t1_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_2p_bb(a,k) t2_abab(c,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_1p_bb(a,k) t2_1p_bbbb(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_bb(a,i) t2_2p_abab(c,b,k,j) 
    //               += -2.000 P(i,j) P(a,b) d-_bb(a,c) t1_1p_bb(b,i) t1_2p_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_bb(a,k) t2_2p_bbbb(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <k,l||c,d>_bbbb t1_2p_bb(a,k) t1_bb(c,i) t2_bbbb(d,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0189_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_2p_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_1p_bb(c,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_bb(c,i) t2_2p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_1p_bb(a,k) t1_1p_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_1p_bb(a,k) t2_1p_abab(c,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_1p_bb(a,k) t1_bb(c,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,k) t1_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_bb(a,i) t2_1p_abab(c,b,k,j) 
    //               += -4.000 P(i,j) P(a,b) d-_bb(a,c) t1_2p_bb(b,i) t1_1p_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_2p_bb(a,k) t1_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_2p_bb(a,k) t2_abab(c,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_1p_bb(a,k) t2_1p_bbbb(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_bb(a,i) t2_2p_abab(c,b,k,j) 
    //               += -2.000 P(i,j) P(a,b) d-_bb(a,c) t1_1p_bb(b,i) t1_2p_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_bb(a,k) t2_2p_bbbb(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <k,l||c,d>_bbbb t1_2p_bb(a,k) t1_bb(c,i) t2_bbbb(d,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0189_bbbb_vvoo")(bb,ab,jb,ib) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_2p_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_1p_bb(c,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_bb(c,i) t2_2p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_1p_bb(a,k) t1_1p_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_1p_bb(a,k) t2_1p_abab(c,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_1p_bb(a,k) t1_bb(c,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,k) t1_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_bb(a,i) t2_1p_abab(c,b,k,j) 
    //               += -4.000 P(i,j) P(a,b) d-_bb(a,c) t1_2p_bb(b,i) t1_1p_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_2p_bb(a,k) t1_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_2p_bb(a,k) t2_abab(c,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_1p_bb(a,k) t2_1p_bbbb(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_bb(a,i) t2_2p_abab(c,b,k,j) 
    //               += -2.000 P(i,j) P(a,b) d-_bb(a,c) t1_1p_bb(b,i) t1_2p_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_bb(a,k) t2_2p_bbbb(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <k,l||c,d>_bbbb t1_2p_bb(a,k) t1_bb(c,i) t2_bbbb(d,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0189_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_2p_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_1p_bb(c,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_bb(c,i) t2_2p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_1p_bb(a,k) t1_1p_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_1p_bb(a,k) t2_1p_abab(c,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_1p_bb(a,k) t1_bb(c,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,k) t1_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +4.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_bb(a,i) t2_1p_abab(c,b,k,j) 
    //               += -4.000 P(i,j) P(a,b) d-_bb(a,c) t1_2p_bb(b,i) t1_1p_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_2p_bb(a,k) t1_bb(c,i) t2_abab(d,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_2p_bb(a,k) t2_abab(c,b,l,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_1p_bb(a,k) t2_1p_bbbb(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_bb(a,i) t2_2p_abab(c,b,k,j) 
    //               += -2.000 P(i,j) P(a,b) d-_bb(a,c) t1_1p_bb(b,i) t1_2p_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_bb(a,k) t2_2p_bbbb(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <k,l||c,d>_bbbb t1_2p_bb(a,k) t1_bb(c,i) t2_bbbb(d,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0189_bbbb_vvoo")(bb,ab,ib,jb) )
    .deallocate(tmps.at("0189_bbbb_vvoo"))
    .allocate(tmps.at("0190_bbbb_vvoo"))
    
    // flops: o2v2  = o3v1Q1 o4v2 o3v2 o3v1Q1 o3v1Q1 o3v1 o3v1Q1 o3v1 o3v2 o2v2 o2v2 o2v2 o3v2 o4v2 o3v2 o2v2 o3v1Q1 o3v1Q1 o3v1 o3v2 o2v2 o3v2 o3v2 o2v2 o2v2 o2v2 o2v2 o2v2
    //  mems: o2v2  = o3v1 o3v1 o2v2 o3v1 o3v1 o3v1 o3v1 o3v1 o2v2 o2v2 o2v2 o2v2 o3v1 o3v1 o2v2 o2v2 o3v1 o3v1 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bbbb_vooo")(cb,ib,kb,lb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("bb_ooQ")(lb,ib,Q) )
    ( tmps.at("bin2_bbbb_vooo")(ab,ib,jb,kb)  = tmps.at("bin1_bbbb_vooo")(cb,ib,kb,lb) * t2_1p.at("bbbb")(cb,ab,jb,lb) )
    ( tmps.at("0190_bbbb_vvoo")(ab,bb,ib,jb)  = tmps.at("bin2_bbbb_vooo")(ab,ib,jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = tmps.at("0058_bb_voQ")(bb,ib,Q) * tmps.at("0029_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) += tmps.at("0135_bb_voQ")(bb,ib,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) += tmps.at("0135_bb_voQ")(bb,ib,Q) * tmps.at("0026_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0190_bbbb_vvoo")(ab,bb,ib,jb) += t1.at("bb")(ab,kb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) )
    ( tmps.at("0190_bbbb_vvoo")(ab,bb,ib,jb) += 2.000 * t1_2p.at("bb")(ab,ib) * tmps.at("0155_bb_vo")(bb,jb) )
    ( tmps.at("bin2_bbbb_vooo")(db,jb,kb,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t1.at("bb")(cb,jb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t2_1p.at("bbbb")(db,bb,ib,lb) * tmps.at("bin2_bbbb_vooo")(db,jb,kb,lb) )
    ( tmps.at("0190_bbbb_vvoo")(ab,bb,ib,jb) += t1.at("bb")(ab,kb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = tmps.at("0058_bb_voQ")(bb,ib,Q) * tmps.at("0026_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) += tmps.at("0058_bb_voQ")(bb,ib,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0190_bbbb_vvoo")(ab,bb,ib,jb) += t1_1p.at("bb")(ab,kb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = t1.at("bb")(cb,ib) * tmps.at("0185_bbbb_voov")(ab,kb,jb,cb) )
    ( tmps.at("0190_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1_1p.at("bb")(bb,kb) )
    ( tmps.at("0190_bbbb_vvoo")(ab,bb,ib,jb) += t1_1p.at("bb")(ab,ib) * tmps.at("0188_bb_vo")(bb,jb) )
    ( tmps.at("0190_bbbb_vvoo")(ab,bb,ib,jb) += 2.000 * dp.at("bb_vo")(ab,ib) * t1_2p.at("bb")(bb,jb) )
    
    // r2_1p[bbbb] += -1.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_1p_bb(c,i) t2_abab(d,b,l,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_bb(a,k) t2_1p_abab(c,b,l,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_bb(a,k) t1_bb(c,i) t2_1p_abab(d,b,l,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +1.000 P(i,j) P(a,b) d-_aa(k,c) t1_1p_bb(a,i) t2_1p_abab(c,b,k,j) 
    //               += -1.000 P(i,j) P(a,b) d-_bb(a,c) t1_1p_bb(b,i) t1_1p_bb(c,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||d,c>_abab t1_1p_bb(a,k) t1_bb(c,i) t2_abab(d,b,l,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||c,i>_abab t1_1p_bb(a,k) t2_abab(c,b,l,j) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(k,c) t1_2p_bb(a,i) t2_abab(c,b,k,j) 
    //               += -2.000 P(i,j) P(a,b) d-_bb(a,c) t1_2p_bb(b,i) t1_bb(c,j) 
    //               += +2.000 P(i,j) P(a,b) d-_bb(a,i) t1_2p_bb(b,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_bb(a,k) t2_1p_bbbb(c,b,j,l) 
    //               += +1.000 P(i,j) P(a,b) <k,l||c,d>_bbbb t1_1p_bb(a,k) t1_bb(c,i) t2_bbbb(d,b,j,l) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0190_bbbb_vvoo")(ab,bb,jb,ib) )
    ;
  }
  // clang-format on
}

template void exachem::cc::cd_qed_ccsd_os::resid_part4<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);