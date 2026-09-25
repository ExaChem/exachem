/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_qed_ccsd_os_resid_3.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_os::resid_part3(
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
        
    // r1_1p[bb] += -0.500 <j,k||c,b>_bbbb t1_1p_bb(b,i) t2_bbbb(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= 0.500 * t1_1p.at("bb")(bb,ib) * tmps.at("0095_bb_vv")(bb,ab) )
    
    // r1_2p[bb] += -1.000 <j,k||c,b>_bbbb t1_2p_bb(b,i) t2_bbbb(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= t1_2p.at("bb")(bb,ib) * tmps.at("0095_bb_vv")(bb,ab) )
    
    // r2[bbbb] += -0.500 P(a,b) <k,l||d,c>_bbbb t2_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) -= 0.500 * tmps.at("0095_bb_vv")(db,ab) * t2.at("bbbb")(db,bb,ib,jb) )
    
    // r2_1p[bbbb] += +0.500 P(a,b) <k,l||c,d>_bbbb t2_1p_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 0.500 * tmps.at("0095_bb_vv")(db,ab) * t2_1p.at("bbbb")(db,bb,ib,jb) )
    
    // r2_2p[bbbb] += +1.000 P(a,b) <k,l||c,d>_bbbb t2_2p_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0095_bb_vv")(db,ab) * t2_2p.at("bbbb")(db,bb,ib,jb) )
    .deallocate(tmps.at("0095_bb_vv"))
    .allocate(tmps.at("0096_bb_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0096_bb_vv")(bb,db)  = t2_1p.at("bbbb")(cb,bb,kb,lb) * tmps.at("0075_bbbb_ovov")(kb,cb,lb,db) )
    
    // r2_1p[abab] += +0.500 <k,l||c,d>_bbbb t2_abab(a,c,i,j) t2_1p_bbbb(d,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.500 * tmps.at("0096_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p[abab] += +1.000 <k,l||d,c>_bbbb t2_1p_abab(a,d,i,j) t2_1p_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("0096_bb_vv")(bb,db) * t2_1p.at("abab")(aa,db,ia,jb) )
    
    // r2_2p[bbbb] += -1.000 P(a,b) <k,l||d,c>_bbbb t2_1p_bbbb(d,a,i,j) t2_1p_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0096_bb_vv")(bb,db) * t2_1p.at("bbbb")(db,ab,ib,jb) )
    .deallocate(tmps.at("0096_bb_vv"))
    .allocate(tmps.at("0097_aa_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0097_aa_vv")(ba,da)  = t2.at("aaaa")(ca,ba,ka,la) * tmps.at("0073_aaaa_ovov")(ka,ca,la,da) )
    
    // r2[aaaa] += -0.500 P(a,b) <k,l||d,c>_aaaa t2_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) += 0.500 * tmps.at("0097_aa_vv")(ba,da) * t2.at("aaaa")(da,aa,ia,ja) )
    
    // r2_1p[aaaa] += +0.500 P(a,b) <k,l||c,d>_aaaa t2_1p_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 0.500 * tmps.at("0097_aa_vv")(ba,da) * t2_1p.at("aaaa")(da,aa,ia,ja) )
    
    // r2_2p[aaaa] += +1.000 P(a,b) <k,l||c,d>_aaaa t2_2p_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0097_aa_vv")(ba,da) * t2_2p.at("aaaa")(da,aa,ia,ja) )
    .deallocate(tmps.at("0097_aa_vv"))
    .allocate(tmps.at("0098_aabb_oovv"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0098_aabb_oovv")(ka,ia,bb,db)  = t1.at("aa")(ca,ia) * tmps.at("0047_aabb_ovvv")(ka,ca,bb,db) )
    
    // r2[abab] += -1.000 <k,b||c,d>_abab t1_aa(c,i) t2_abab(a,d,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,db,ka,jb) * tmps.at("0098_aabb_oovv")(ka,ia,bb,db) )
    
    // r2_1p[abab] += -1.000 <k,b||c,d>_abab t1_aa(c,i) t2_1p_abab(a,d,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,db,ka,jb) * tmps.at("0098_aabb_oovv")(ka,ia,bb,db) )
    
    // r2_2p[abab] += -2.000 <k,b||c,d>_abab t1_aa(c,i) t2_2p_abab(a,d,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_2p.at("abab")(aa,db,ka,jb) * tmps.at("0098_aabb_oovv")(ka,ia,bb,db) )
    .deallocate(tmps.at("0098_aabb_oovv"))
    .allocate(tmps.at("0099_aabb_vvov"))
    
    // flops: o1v3  = o1v3Q1
    //  mems: o1v3  = o1v3
    ( tmps.at("0099_aabb_vvov")(aa,da,kb,cb)  = chol.at("aa_vvQ")(aa,da,Q) * chol.at("bb_ovQ")(kb,cb,Q) )
    
    // r1[aa] += +0.500 <a,j||c,b>_abab t2_abab(c,b,i,j) 
    //          += +0.500 <a,j||b,c>_abab t2_abab(b,c,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0099_aabb_vvov")(aa,ca,jb,bb) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r1_1p[aa] += +0.500 <a,j||c,b>_abab t2_1p_abab(c,b,i,j) 
    //             += +0.500 <a,j||b,c>_abab t2_1p_abab(b,c,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0099_aabb_vvov")(aa,ca,jb,bb) * t2_1p.at("abab")(ca,bb,ia,jb) )
    
    // r1_2p[aa] += +1.000 <a,j||c,b>_abab t2_2p_abab(c,b,i,j) 
    //             += +1.000 <a,j||b,c>_abab t2_2p_abab(b,c,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0099_aabb_vvov")(aa,ca,jb,bb) * t2_2p.at("abab")(ca,bb,ia,jb) )
    
    // r2[abab] += +1.000 <a,b||c,j>_abab t1_aa(c,i) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(ca,ia) * tmps.at("0099_aabb_vvov")(ca,aa,jb,bb) )
    
    // r2_1p[abab] += +1.000 <a,b||c,j>_abab t1_1p_aa(c,i) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(ca,ia) * tmps.at("0099_aabb_vvov")(ca,aa,jb,bb) )
    
    // r2_2p[abab] += +2.000 <a,b||c,j>_abab t1_2p_aa(c,i) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_2p.at("aa")(ca,ia) * tmps.at("0099_aabb_vvov")(ca,aa,jb,bb) )
    
    // r2_2p[abab] += -2.000 <a,k||d,c>_abab t1_2p_bb(c,j) t2_abab(d,b,i,k) 
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_aabb_vvoo")(aa,da,jb,kb)  = tmps.at("0099_aabb_vvov")(aa,da,kb,cb) * t1_2p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2.at("abab")(da,bb,ia,kb) * tmps.at("bin1_aabb_vvoo")(aa,da,jb,kb) )
    
    // r2_2p[abab] += -1.000 <a,k||d,c>_abab t1_bb(b,k) t2_2p_abab(d,c,i,j) 
    //               += -1.000 <a,k||c,d>_abab t1_bb(b,k) t2_2p_abab(c,d,i,j) 
    // flops: o2v2 += o3v3 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0099_aabb_vvov")(aa,da,kb,cb) * t2_2p.at("abab")(da,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0100_aabb_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0100_aabb_vvoo")(aa,da,kb,jb)  = tmps.at("0099_aabb_vvov")(aa,da,kb,cb) * t1.at("bb")(cb,jb) )
    
    // r2[abab] += -1.000 <a,k||d,c>_abab t1_bb(c,j) t2_abab(d,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0100_aabb_vvoo")(aa,da,kb,jb) * t2.at("abab")(da,bb,ia,kb) )
    
    // r2_1p[abab] += -1.000 <a,k||d,c>_abab t1_bb(c,j) t2_1p_abab(d,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0100_aabb_vvoo")(aa,da,kb,jb) * t2_1p.at("abab")(da,bb,ia,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||d,c>_abab t1_bb(c,j) t2_2p_abab(d,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0100_aabb_vvoo")(aa,da,kb,jb) * t2_2p.at("abab")(da,bb,ia,kb) )
    .deallocate(tmps.at("0100_aabb_vvoo"))
    .allocate(tmps.at("0101_bbbb_voov"))
    
    // flops: o2v2  = o1v3 o2v3
    //  mems: o2v2  = o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,db)  = tmps.at("0086_bbbb_ovvv")(kb,db,bb,cb) * t1.at("bb")(cb,kb) )
    ( tmps.at("0101_bbbb_voov")(ab,ib,jb,bb)  = t2.at("bbbb")(db,ab,ib,jb) * tmps.at("bin1_bb_vv")(bb,db) )
    
    // r2[bbbb] += -1.000 P(a,b) <a,k||c,d>_bbbb t1_bb(c,k) t2_bbbb(d,b,i,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0101_bbbb_voov")(ab,ib,jb,bb) )
    
    // r2[bbbb] += -1.000 P(a,b) <a,k||c,d>_bbbb t1_bb(c,k) t2_bbbb(d,b,i,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0101_bbbb_voov")(bb,ib,jb,ab) )
    .deallocate(tmps.at("0101_bbbb_voov"))
    .allocate(tmps.at("0102_aabb_oovv"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0102_aabb_oovv")(ka,ia,bb,db)  = t1_1p.at("aa")(ca,ia) * tmps.at("0047_aabb_ovvv")(ka,ca,bb,db) )
    
    // r2_1p[abab] += -1.000 <k,b||c,d>_abab t1_1p_aa(c,i) t2_abab(a,d,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,db,ka,jb) * tmps.at("0102_aabb_oovv")(ka,ia,bb,db) )
    
    // r2_2p[abab] += -2.000 <k,b||c,d>_abab t1_1p_aa(c,i) t2_1p_abab(a,d,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_1p.at("abab")(aa,db,ka,jb) * tmps.at("0102_aabb_oovv")(ka,ia,bb,db) )
    .deallocate(tmps.at("0102_aabb_oovv"))
    .allocate(tmps.at("0103_aabb_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0103_aabb_vvoo")(aa,da,kb,jb)  = tmps.at("0099_aabb_vvov")(aa,da,kb,cb) * t1_1p.at("bb")(cb,jb) )
    
    // r2_1p[abab] += -1.000 <a,k||d,c>_abab t1_1p_bb(c,j) t2_abab(d,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0103_aabb_vvoo")(aa,da,kb,jb) * t2.at("abab")(da,bb,ia,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||d,c>_abab t1_1p_bb(c,j) t2_1p_abab(d,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0103_aabb_vvoo")(aa,da,kb,jb) * t2_1p.at("abab")(da,bb,ia,kb) )
    .deallocate(tmps.at("0103_aabb_vvoo"))
    .allocate(tmps.at("0104_baab_vooo"))
    
    // flops: o3v1  = o2v2 o3v2
    //  mems: o3v1  = o1v1 o3v1
    ( tmps.at("bin1_aa_vo")(da,la)  = t1.at("aa")(ca,ka) * tmps.at("0073_aaaa_ovov")(la,ca,ka,da) )
    ( tmps.at("0104_baab_vooo")(bb,la,ia,jb)  = t2.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,la) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0104_baab_vooo")(bb,la,ia,jb) * t1.at("aa")(aa,la) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0104_baab_vooo")(bb,la,ia,jb) * t1_1p.at("aa")(aa,la) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_2p_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0104_baab_vooo")(bb,la,ia,jb) * t1_2p.at("aa")(aa,la) )
    .deallocate(tmps.at("0104_baab_vooo"))
    .allocate(tmps.at("0105_aa_ov"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0105_aa_ov")(ka,ca)  = t1.at("aa")(ba,ja) * tmps.at("0073_aaaa_ovov")(ka,ba,ja,ca) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_aa_oo")(ia,ka)  = t1_2p.at("aa")(ca,ia) * tmps.at("0105_aa_ov")(ka,ca) )
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_aa_oo")(ia,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,la,jb)  = tmps.at("0105_aa_ov")(la,da) * t2_2p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,la) * tmps.at("bin1_baab_vooo")(bb,ia,la,jb) )
    .allocate(tmps.at("0106_baab_ovoo"))
    
    // flops: o3v1  = o2v2 o3v2
    //  mems: o3v1  = o1v1 o3v1
    ( tmps.at("bin1_bb_vo")(db,lb)  = t1.at("bb")(cb,kb) * tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) )
    ( tmps.at("0106_baab_ovoo")(lb,aa,ia,jb)  = tmps.at("bin1_bb_vo")(db,lb) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,lb) * tmps.at("0106_baab_ovoo")(lb,aa,ia,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t1_1p_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("bb")(bb,lb) * tmps.at("0106_baab_ovoo")(lb,aa,ia,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_2p_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_2p.at("bb")(bb,lb) * tmps.at("0106_baab_ovoo")(lb,aa,ia,jb) )
    .deallocate(tmps.at("0106_baab_ovoo"))
    .allocate(tmps.at("0107_baab_ovoo"))
    
    // flops: o3v1  = o2v2 o3v2
    //  mems: o3v1  = o1v1 o3v1
    ( tmps.at("bin1_bb_vo")(db,lb)  = t1.at("bb")(cb,kb) * tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) )
    ( tmps.at("0107_baab_ovoo")(lb,aa,ia,jb)  = tmps.at("bin1_bb_vo")(db,lb) * t2_1p.at("abab")(aa,db,ia,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,lb) * tmps.at("0107_baab_ovoo")(lb,aa,ia,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_1p_bb(b,l) t1_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_1p.at("bb")(bb,lb) * tmps.at("0107_baab_ovoo")(lb,aa,ia,jb) )
    .deallocate(tmps.at("0107_baab_ovoo"))
    .allocate(tmps.at("0108_baab_vooo"))
    
    // flops: o3v1  = o2v2 o3v2
    //  mems: o3v1  = o1v1 o3v1
    ( tmps.at("bin1_aa_vo")(da,la)  = t1.at("aa")(ca,ka) * tmps.at("0073_aaaa_ovov")(la,ca,ka,da) )
    ( tmps.at("0108_baab_vooo")(bb,la,ia,jb)  = t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,la) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0108_baab_vooo")(bb,la,ia,jb) * t1.at("aa")(aa,la) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0108_baab_vooo")(bb,la,ia,jb) * t1_1p.at("aa")(aa,la) )
    .deallocate(tmps.at("0108_baab_vooo"))
    .allocate(tmps.at("0109_abab_vvoo"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin1_aaaa_vvoo")(ca,da,ka,la)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("0109_abab_vvoo")(ca,ab,ka,ib)  = tmps.at("bin1_aaaa_vvoo")(ca,da,ka,la) * t2_1p.at("abab")(da,ab,la,ib) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t1_1p.at("aa")(ca,ia) * tmps.at("0109_abab_vvoo")(ca,bb,ka,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_1p_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_1p.at("aaaa")(ca,aa,ia,ka) * tmps.at("0109_abab_vvoo")(ca,bb,ka,jb) )
    .allocate(tmps.at("0110_baab_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0110_baab_vooo")(bb,ia,ka,jb)  = t1.at("aa")(ca,ia) * tmps.at("0109_abab_vvoo")(ca,bb,ka,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0110_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0110_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0110_baab_vooo"))
    .allocate(tmps.at("0111_bbbb_ovvo"))
    
    // flops: o2v2  = o3v1 o3v2
    //  mems: o2v2  = o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(ib,lb)  = tmps.at("0064_bbbb_ooov")(kb,ib,lb,cb) * t1.at("bb")(cb,kb) )
    ( tmps.at("0111_bbbb_ovvo")(ib,ab,bb,jb)  = tmps.at("bin1_bb_oo")(ib,lb) * t2.at("bbbb")(ab,bb,jb,lb) )
    
    // r2[bbbb] += +1.000 P(i,j) <l,k||i,c>_bbbb t1_bb(c,k) t2_bbbb(a,b,j,l) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0111_bbbb_ovvo")(ib,ab,bb,jb) )
    
    // r2[bbbb] += +1.000 P(i,j) <l,k||i,c>_bbbb t1_bb(c,k) t2_bbbb(a,b,j,l) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0111_bbbb_ovvo")(jb,ab,bb,ib) )
    .deallocate(tmps.at("0111_bbbb_ovvo"))
    .allocate(tmps.at("0112_bbbb_ovvo"))
    
    // flops: o2v2  = o3v1 o3v2
    //  mems: o2v2  = o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(ib,lb)  = tmps.at("0064_bbbb_ooov")(kb,ib,lb,cb) * t1.at("bb")(cb,kb) )
    ( tmps.at("0112_bbbb_ovvo")(ib,ab,bb,jb)  = tmps.at("bin1_bb_oo")(ib,lb) * t2_1p.at("bbbb")(ab,bb,jb,lb) )
    
    // r2_1p[bbbb] += +1.000 P(i,j) <l,k||i,c>_bbbb t1_bb(c,k) t2_1p_bbbb(a,b,j,l) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0112_bbbb_ovvo")(ib,ab,bb,jb) )
    
    // r2_1p[bbbb] += +1.000 P(i,j) <l,k||i,c>_bbbb t1_bb(c,k) t2_1p_bbbb(a,b,j,l) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0112_bbbb_ovvo")(jb,ab,bb,ib) )
    .deallocate(tmps.at("0112_bbbb_ovvo"))
    .allocate(tmps.at("0113_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0113_bb_oo")(ib,kb)  = tmps.at("0064_bbbb_ooov")(jb,ib,kb,bb) * t1.at("bb")(bb,jb) )
    
    // r1[bb] += -1.000 <k,j||i,b>_bbbb t1_bb(a,k) t1_bb(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += tmps.at("0113_bb_oo")(ib,kb) * t1.at("bb")(ab,kb) )
    
    // r1_1p[bb] += -1.000 <k,j||i,b>_bbbb t1_1p_bb(a,k) t1_bb(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0113_bb_oo")(ib,kb) * t1_1p.at("bb")(ab,kb) )
    
    // r1_2p[bb] += -2.000 <k,j||i,b>_bbbb t1_2p_bb(a,k) t1_bb(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0113_bb_oo")(ib,kb) * t1_2p.at("bb")(ab,kb) )
    
    // r2[abab] += -1.000 <l,k||j,c>_bbbb t1_bb(c,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ia,lb) * tmps.at("0113_bb_oo")(jb,lb) )
    
    // r2_1p[abab] += -1.000 <l,k||j,c>_bbbb t1_bb(c,k) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0113_bb_oo")(jb,lb) )
    
    // r2_2p[abab] += -2.000 <l,k||j,c>_bbbb t1_bb(c,k) t2_2p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("abab")(aa,bb,ia,lb) * tmps.at("0113_bb_oo")(jb,lb) )
    .allocate(tmps.at("0114_aaaa_ovvo"))
    
    // flops: o2v2  = o3v1 o3v2
    //  mems: o2v2  = o2v0 o2v2
    ( tmps.at("bin1_aa_oo")(ia,la)  = tmps.at("0056_aaaa_ooov")(ka,ia,la,ca) * t1.at("aa")(ca,ka) )
    ( tmps.at("0114_aaaa_ovvo")(ia,aa,ba,ja)  = tmps.at("bin1_aa_oo")(ia,la) * t2.at("aaaa")(aa,ba,ja,la) )
    
    // r2[aaaa] += +1.000 P(i,j) <l,k||i,c>_aaaa t1_aa(c,k) t2_aaaa(a,b,j,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0114_aaaa_ovvo")(ia,aa,ba,ja) )
    
    // r2[aaaa] += +1.000 P(i,j) <l,k||i,c>_aaaa t1_aa(c,k) t2_aaaa(a,b,j,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0114_aaaa_ovvo")(ja,aa,ba,ia) )
    .deallocate(tmps.at("0114_aaaa_ovvo"))
    .allocate(tmps.at("0115_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0115_aa_oo")(ia,ka)  = tmps.at("0056_aaaa_ooov")(ja,ia,ka,ba) * t1.at("aa")(ba,ja) )
    
    // r1[aa] += -1.000 <k,j||i,b>_aaaa t1_aa(a,k) t1_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0115_aa_oo")(ia,ka) * t1.at("aa")(aa,ka) )
    
    // r1_1p[aa] += -1.000 <k,j||i,b>_aaaa t1_1p_aa(a,k) t1_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0115_aa_oo")(ia,ka) * t1_1p.at("aa")(aa,ka) )
    
    // r1_2p[aa] += -2.000 <k,j||i,b>_aaaa t1_2p_aa(a,k) t1_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0115_aa_oo")(ia,ka) * t1_2p.at("aa")(aa,ka) )
    
    // r2[abab] += -1.000 <l,k||i,c>_aaaa t1_aa(c,k) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,jb) * tmps.at("0115_aa_oo")(ia,la) )
    
    // r2_1p[abab] += -1.000 <l,k||i,c>_aaaa t1_aa(c,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,la,jb) * tmps.at("0115_aa_oo")(ia,la) )
    
    // r2_2p[abab] += -2.000 <l,k||i,c>_aaaa t1_aa(c,k) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("abab")(aa,bb,la,jb) * tmps.at("0115_aa_oo")(ia,la) )
    .allocate(tmps.at("0116_aaaa_ovvo"))
    
    // flops: o2v2  = o3v1 o3v2
    //  mems: o2v2  = o2v0 o2v2
    ( tmps.at("bin1_aa_oo")(ia,la)  = tmps.at("0056_aaaa_ooov")(ka,ia,la,ca) * t1.at("aa")(ca,ka) )
    ( tmps.at("0116_aaaa_ovvo")(ia,aa,ba,ja)  = tmps.at("bin1_aa_oo")(ia,la) * t2_1p.at("aaaa")(aa,ba,ja,la) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) <l,k||i,c>_aaaa t1_aa(c,k) t2_1p_aaaa(a,b,j,l) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0116_aaaa_ovvo")(ia,aa,ba,ja) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) <l,k||i,c>_aaaa t1_aa(c,k) t2_1p_aaaa(a,b,j,l) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0116_aaaa_ovvo")(ja,aa,ba,ia) )
    .deallocate(tmps.at("0116_aaaa_ovvo"))
    .allocate(tmps.at("0117_bb_vv"))
    
    // flops: o0v2  = o1v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0117_bb_vv")(cb,ab)  = t1.at("bb")(bb,jb) * tmps.at("0086_bbbb_ovvv")(jb,cb,ab,bb) )
    
    // r1_2p[bb] += -2.000 <a,j||b,c>_bbbb t1_bb(b,j) t1_2p_bb(c,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0117_bb_vv")(cb,ab) * t1_2p.at("bb")(cb,ib) )
    
    // r2[abab] += -1.000 <b,k||c,d>_bbbb t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0117_bb_vv")(db,bb) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_1p[abab] += -1.000 <b,k||c,d>_bbbb t1_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0117_bb_vv")(db,bb) * t2_1p.at("abab")(aa,db,ia,jb) )
    
    // r2_2p[abab] += -2.000 <b,k||c,d>_bbbb t1_bb(c,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0117_bb_vv")(db,bb) * t2_2p.at("abab")(aa,db,ia,jb) )
    .deallocate(tmps.at("0117_bb_vv"))
    .allocate(tmps.at("0118_aa_vv"))
    
    // flops: o0v2  = o1v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0118_aa_vv")(ca,aa)  = t1.at("aa")(ba,ja) * tmps.at("0090_aaaa_ovvv")(ja,ca,aa,ba) )
    
    // r1_2p[aa] += -2.000 <a,j||b,c>_aaaa t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0118_aa_vv")(ca,aa) * t1_2p.at("aa")(ca,ia) )
    
    // r2[abab] += -1.000 <a,k||c,d>_aaaa t1_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0118_aa_vv")(da,aa) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2_1p[abab] += -1.000 <a,k||c,d>_aaaa t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0118_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 <a,k||c,d>_aaaa t1_aa(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0118_aa_vv")(da,aa) * t2_2p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0118_aa_vv"))
    .allocate(tmps.at("0119_bb_vv"))
    
    // flops: o0v2  = o1v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0119_bb_vv")(cb,ab)  = t1_1p.at("bb")(bb,jb) * tmps.at("0086_bbbb_ovvv")(jb,cb,ab,bb) )
    
    // r2_1p[abab] += +1.000 <b,k||d,c>_bbbb t1_1p_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0119_bb_vv")(db,bb) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_2p[abab] += -2.000 <b,k||c,d>_bbbb t1_1p_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0119_bb_vv")(db,bb) * t2_1p.at("abab")(aa,db,ia,jb) )
    .deallocate(tmps.at("0119_bb_vv"))
    .allocate(tmps.at("0120_aa_vv"))
    
    // flops: o0v2  = o1v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0120_aa_vv")(ca,aa)  = t1_1p.at("aa")(ba,ja) * tmps.at("0090_aaaa_ovvv")(ja,ca,aa,ba) )
    
    // r2_1p[abab] += +1.000 <a,k||d,c>_aaaa t1_1p_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0120_aa_vv")(da,aa) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 <a,k||c,d>_aaaa t1_1p_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0120_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0120_aa_vv"))
    .allocate(tmps.at("0121_aabb_ooov"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0121_aabb_ooov")(ja,ia,kb,bb)  = chol.at("aa_ooQ")(ja,ia,Q) * chol.at("bb_ovQ")(kb,bb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||i,c>_abab t1_bb(b,k) t2_2p_abab(a,c,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t2_2p.at("abab")(aa,cb,la,jb) * tmps.at("0121_aabb_ooov")(la,ia,kb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    .allocate(tmps.at("0122_aabb_oooo"))
    
    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    ( tmps.at("0122_aabb_oooo")(ka,ia,lb,jb)  = tmps.at("0121_aabb_ooov")(ka,ia,lb,cb) * t1_2p.at("bb")(cb,jb) )
    
    // r2_2p[abab] += +1.000 <k,l||i,c>_abab t1_2p_bb(c,j) t2_abab(a,b,k,l) 
    //               += +1.000 <l,k||i,c>_abab t1_2p_bb(c,j) t2_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,bb,ka,lb) * tmps.at("0122_aabb_oooo")(ka,ia,lb,jb) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_abab t1_aa(a,k) t1_bb(b,l) t1_2p_bb(c,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t1.at("bb")(bb,lb) * tmps.at("0122_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0122_aabb_oooo"))
    .allocate(tmps.at("0123_aabb_oooo"))
    
    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    ( tmps.at("0123_aabb_oooo")(ka,ia,lb,jb)  = tmps.at("0121_aabb_ooov")(ka,ia,lb,cb) * t1.at("bb")(cb,jb) )
    
    // r2[abab] += +0.500 <k,l||i,c>_abab t1_bb(c,j) t2_abab(a,b,k,l) 
    //            += +0.500 <l,k||i,c>_abab t1_bb(c,j) t2_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ka,lb) * tmps.at("0123_aabb_oooo")(ka,ia,lb,jb) )
    
    // r2_1p[abab] += +0.500 <k,l||i,c>_abab t1_bb(c,j) t2_1p_abab(a,b,k,l) 
    //               += +0.500 <l,k||i,c>_abab t1_bb(c,j) t2_1p_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,ka,lb) * tmps.at("0123_aabb_oooo")(ka,ia,lb,jb) )
    
    // r2_2p[abab] += +1.000 <k,l||i,c>_abab t1_bb(c,j) t2_2p_abab(a,b,k,l) 
    //               += +1.000 <l,k||i,c>_abab t1_bb(c,j) t2_2p_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("abab")(aa,bb,ka,lb) * tmps.at("0123_aabb_oooo")(ka,ia,lb,jb) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_abab t1_aa(a,k) t1_2p_bb(b,l) t1_bb(c,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t1_2p.at("bb")(bb,lb) * tmps.at("0123_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0124_aabb_oovo"))
    
    // flops: o3v1  = o4v1 o4v1
    //  mems: o3v1  = o4v0 o3v1
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb)  = tmps.at("0121_aabb_ooov")(ka,ia,lb,cb) * t1.at("bb")(cb,jb) )
    ( tmps.at("0124_aabb_oovo")(ka,ia,bb,jb)  = tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) * t1.at("bb")(bb,lb) )
    
    // r2[abab] += +1.000 <k,l||i,c>_abab t1_aa(a,k) t1_bb(b,l) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0124_aabb_oovo")(ka,ia,bb,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||i,c>_abab t1_1p_aa(a,l) t1_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,la) * tmps.at("0124_aabb_oovo")(la,ia,bb,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||i,c>_abab t1_2p_aa(a,l) t1_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_2p.at("aa")(aa,la) * tmps.at("0124_aabb_oovo")(la,ia,bb,jb) )
    .deallocate(tmps.at("0124_aabb_oovo"))
    .allocate(tmps.at("0125_aa_oo"))
    
    // flops: o2v0  = o2v2 o2v1
    //  mems: o2v0  = o1v1 o2v0
    ( tmps.at("bin1_aa_vo")(ca,ka)  = t1.at("aa")(ba,ja) * tmps.at("0073_aaaa_ovov")(ka,ba,ja,ca) )
    ( tmps.at("0125_aa_oo")(ka,ia)  = tmps.at("bin1_aa_vo")(ca,ka) * t1.at("aa")(ca,ia) )
    
    // r1[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0125_aa_oo")(ka,ia) * t1.at("aa")(aa,ka) )
    
    // r1_1p[aa] += +1.000 <k,j||b,c>_aaaa t1_1p_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1_1p.at("aa")(aa,ka) * tmps.at("0125_aa_oo")(ka,ia) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_2p_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1_2p.at("aa")(aa,ka) * tmps.at("0125_aa_oo")(ka,ia) )
    .deallocate(tmps.at("0125_aa_oo"))
    .allocate(tmps.at("0126_aa_oo"))
    
    // flops: o2v0  = o2v2 o2v1
    //  mems: o2v0  = o1v1 o2v0
    ( tmps.at("bin1_aa_vo")(ca,ka)  = t1.at("aa")(ba,ja) * tmps.at("0073_aaaa_ovov")(ka,ba,ja,ca) )
    ( tmps.at("0126_aa_oo")(ka,ia)  = tmps.at("bin1_aa_vo")(ca,ka) * t1_1p.at("aa")(ca,ia) )
    
    // r1_1p[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0126_aa_oo")(ka,ia) * t1.at("aa")(aa,ka) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_1p_aa(a,k) t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0126_aa_oo")(ka,ia) * t1_1p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0126_aa_oo"))
    .allocate(tmps.at("0127_aa_oo"))
    
    // flops: o2v0  = o2v2 o2v1
    //  mems: o2v0  = o1v1 o2v0
    ( tmps.at("bin1_aa_vo")(ba,ka)  = t1_1p.at("aa")(ca,ja) * tmps.at("0073_aaaa_ovov")(ka,ca,ja,ba) )
    ( tmps.at("0127_aa_oo")(ka,ia)  = tmps.at("bin1_aa_vo")(ba,ka) * t1.at("aa")(ba,ia) )
    
    // r1_2p[aa] += -2.000 <k,j||b,c>_aaaa t1_1p_aa(a,k) t1_aa(b,i) t1_1p_aa(c,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0127_aa_oo")(ka,ia) * t1_1p.at("aa")(aa,ka) )
    
    // r2_1p[aaaa] += -1.000 P(i,j) <k,l||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,k) t2_aaaa(a,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += t2.at("aaaa")(aa,ba,ia,la) * tmps.at("0127_aa_oo")(la,ja) )
    .deallocate(tmps.at("0127_aa_oo"))
    .allocate(tmps.at("0128_aaaa_voov"))
    
    // flops: o2v2  = o2v3 o2v3
    //  mems: o2v2  = o0v2 o2v2
    ( tmps.at("bin1_aa_vv")(aa,da)  = tmps.at("0073_aaaa_ovov")(ka,da,la,ca) * t2.at("aaaa")(ca,aa,ka,la) )
    ( tmps.at("0128_aaaa_voov")(ba,ia,ja,aa)  = t2.at("aaaa")(da,ba,ia,ja) * tmps.at("bin1_aa_vv")(aa,da) )
    
    // r2[aaaa] += -0.500 P(a,b) <k,l||d,c>_aaaa t2_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= 0.500 * tmps.at("0128_aaaa_voov")(aa,ia,ja,ba) )
    
    // r2[aaaa] += -0.500 P(a,b) <k,l||d,c>_aaaa t2_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += 0.500 * tmps.at("0128_aaaa_voov")(ba,ia,ja,aa) )
    .deallocate(tmps.at("0128_aaaa_voov"))
    .allocate(tmps.at("0129_aaaa_voov"))
    
    // flops: o2v2  = o2v3 o2v3
    //  mems: o2v2  = o0v2 o2v2
    ( tmps.at("bin1_aa_vv")(aa,da)  = tmps.at("0073_aaaa_ovov")(ka,da,la,ca) * t2.at("aaaa")(ca,aa,ka,la) )
    ( tmps.at("0129_aaaa_voov")(ba,ia,ja,aa)  = t2_1p.at("aaaa")(da,ba,ia,ja) * tmps.at("bin1_aa_vv")(aa,da) )
    
    // r2_1p[aaaa] += +0.500 P(a,b) <k,l||c,d>_aaaa t2_1p_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 0.500 * tmps.at("0129_aaaa_voov")(aa,ia,ja,ba) )
    
    // r2_1p[aaaa] += +0.500 P(a,b) <k,l||c,d>_aaaa t2_1p_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 0.500 * tmps.at("0129_aaaa_voov")(ba,ia,ja,aa) )
    .deallocate(tmps.at("0129_aaaa_voov"))
    .allocate(tmps.at("0130_bbbb_voov"))
    
    // flops: o2v2  = o2v3 o2v3
    //  mems: o2v2  = o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(ab,db)  = tmps.at("0075_bbbb_ovov")(kb,db,lb,cb) * t2.at("bbbb")(cb,ab,kb,lb) )
    ( tmps.at("0130_bbbb_voov")(bb,ib,jb,ab)  = t2.at("bbbb")(db,bb,ib,jb) * tmps.at("bin1_bb_vv")(ab,db) )
    
    // r2[bbbb] += -0.500 P(a,b) <k,l||d,c>_bbbb t2_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= 0.500 * tmps.at("0130_bbbb_voov")(ab,ib,jb,bb) )
    
    // r2[bbbb] += -0.500 P(a,b) <k,l||d,c>_bbbb t2_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += 0.500 * tmps.at("0130_bbbb_voov")(bb,ib,jb,ab) )
    .deallocate(tmps.at("0130_bbbb_voov"))
    .allocate(tmps.at("0131_bb_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0131_bb_vv")(db,ab)  = tmps.at("0075_bbbb_ovov")(kb,db,lb,cb) * t2.at("bbbb")(cb,ab,kb,lb) )
    
    // r2[abab] += +0.500 <k,l||d,c>_bbbb t2_abab(a,d,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += 0.500 * t2.at("abab")(aa,db,ia,jb) * tmps.at("0131_bb_vv")(db,bb) )
    
    // r2_1p[abab] += -0.500 <k,l||c,d>_bbbb t2_1p_abab(a,d,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.500 * t2_1p.at("abab")(aa,db,ia,jb) * tmps.at("0131_bb_vv")(db,bb) )
    
    // r2_2p[abab] += -1.000 <k,l||c,d>_bbbb t2_2p_abab(a,d,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += t2_2p.at("abab")(aa,db,ia,jb) * tmps.at("0131_bb_vv")(db,bb) )
    .allocate(tmps.at("0132_bbbb_voov"))
    
    // flops: o2v2  = o2v3 o2v3
    //  mems: o2v2  = o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(ab,db)  = tmps.at("0075_bbbb_ovov")(kb,db,lb,cb) * t2.at("bbbb")(cb,ab,kb,lb) )
    ( tmps.at("0132_bbbb_voov")(bb,ib,jb,ab)  = t2_1p.at("bbbb")(db,bb,ib,jb) * tmps.at("bin1_bb_vv")(ab,db) )
    
    // r2_1p[bbbb] += +0.500 P(a,b) <k,l||c,d>_bbbb t2_1p_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 0.500 * tmps.at("0132_bbbb_voov")(ab,ib,jb,bb) )
    
    // r2_1p[bbbb] += +0.500 P(a,b) <k,l||c,d>_bbbb t2_1p_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 0.500 * tmps.at("0132_bbbb_voov")(bb,ib,jb,ab) )
    .deallocate(tmps.at("0132_bbbb_voov"))
    .allocate(tmps.at("0133_aabb_oovv"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0133_aabb_oovv")(ka,ia,bb,cb)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("bb_vvQ")(bb,cb,Q) )
    
    // r2[abab] += -1.000 <k,b||i,c>_abab t2_abab(a,c,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0133_aabb_oovv")(ka,ia,bb,cb) * t2.at("abab")(aa,cb,ka,jb) )
    
    // r2_1p[abab] += -1.000 <k,b||i,c>_abab t2_1p_abab(a,c,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0133_aabb_oovv")(ka,ia,bb,cb) * t2_1p.at("abab")(aa,cb,ka,jb) )
    
    // r2_2p[abab] += -2.000 <k,b||i,c>_abab t2_2p_abab(a,c,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0133_aabb_oovv")(ka,ia,bb,cb) * t2_2p.at("abab")(aa,cb,ka,jb) )
    
    // r2_2p[abab] += -2.000 <k,b||i,c>_abab t1_aa(a,k) t1_2p_bb(c,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0133_aabb_oovv")(ka,ia,bb,cb) * t1_2p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0134_aabb_oovo"))
    
    // flops: o3v1  = o3v1Q1 o3v1Q1 o3v1 o3v1Q1 o3v1 o3v2 o3v1 o3v1Q1 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("0134_aabb_oovo")(ka,ia,bb,jb)  = -1.000 * chol.at("bb_voQ")(bb,jb,Q) * tmps.at("0053_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0134_aabb_oovo")(ka,ia,bb,jb) -= tmps.at("0058_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0134_aabb_oovo")(ka,ia,bb,jb) -= chol.at("bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0134_aabb_oovo")(ka,ia,bb,jb) -= t1.at("bb")(cb,jb) * tmps.at("0133_aabb_oovv")(ka,ia,bb,cb) )
    ( tmps.at("0134_aabb_oovo")(ka,ia,bb,jb) += tmps.at("0059_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    
    // r2[abab] += +1.000 <k,l||i,c>_abab t1_aa(a,k) t2_bbbb(c,b,j,l) 
    //            += -1.000 <k,b||i,c>_abab t1_aa(a,k) t1_bb(c,j) 
    //            += -1.000 <k,b||i,j>_abab t1_aa(a,k) 
    //            += +1.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_abab(c,b,l,j) 
    //            += -1.000 <k,b||c,j>_abab t1_aa(a,k) t1_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0134_aabb_oovo")(ka,ia,bb,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += +1.000 <k,l||i,c>_abab t1_1p_aa(a,k) t2_bbbb(c,b,j,l) 
    //               += -1.000 <k,b||i,c>_abab t1_1p_aa(a,k) t1_bb(c,j) 
    //               += -1.000 <k,b||i,j>_abab t1_1p_aa(a,k) 
    //               += -1.000 <k,l||i,c>_aaaa t1_1p_aa(a,k) t2_abab(c,b,l,j) 
    //               += -1.000 <k,b||c,j>_abab t1_1p_aa(a,k) t1_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0134_aabb_oovo")(ka,ia,bb,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_abab t1_2p_aa(a,k) t2_bbbb(c,b,j,l) 
    //               += -2.000 <k,b||i,c>_abab t1_2p_aa(a,k) t1_bb(c,j) 
    //               += -2.000 <k,b||i,j>_abab t1_2p_aa(a,k) 
    //               += -2.000 <k,l||i,c>_aaaa t1_2p_aa(a,k) t2_abab(c,b,l,j) 
    //               += -2.000 <k,b||c,j>_abab t1_2p_aa(a,k) t1_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0134_aabb_oovo")(ka,ia,bb,jb) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0134_aabb_oovo"))
    .allocate(tmps.at("0135_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0135_bb_voQ")(bb,ib,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t2_1p.at("abab")(ca,bb,ja,ib) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_abab t2_1p_abab(a,c,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0028_aa_voQ")(aa,ia,Q) * tmps.at("0135_bb_voQ")(bb,jb,Q) )
    
    // r1_1p[bb] += -1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_1p_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0135_bb_voQ")(ab,ib,Q) * tmps.at("0049_Q")(Q) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0135_bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += +1.000 <l,k||d,c>_abab t2_abab(a,c,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0135_bb_voQ")(bb,jb,Q) )
    
    // r1_1p[bb] += -0.500 <j,k||b,i>_abab t2_1p_abab(b,a,j,k) 
    //             += -0.500 <k,j||b,i>_abab t2_1p_abab(b,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= chol.at("bb_ooQ")(kb,ib,Q) * tmps.at("0135_bb_voQ")(ab,kb,Q) )
    
    // r2_1p[abab] += +1.000 <a,k||i,c>_aaaa t2_1p_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0135_bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += +1.000 <a,k||c,d>_aaaa t1_aa(c,i) t2_1p_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0135_bb_voQ")(bb,jb,Q) )
    
    // r1_2p[bb] += -1.000 <j,k||c,b>_abab t1_1p_bb(b,i) t2_1p_abab(c,a,j,k) 
    //             += -1.000 <k,j||c,b>_abab t1_1p_bb(b,i) t2_1p_abab(c,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0135_bb_voQ")(ab,kb,Q) * tmps.at("0029_bb_ooQ")(kb,ib,Q) )
    
    // r1_1p[bb] += -0.500 <j,k||c,b>_abab t1_bb(b,i) t2_1p_abab(c,a,j,k) 
    //             += -0.500 <k,j||c,b>_abab t1_bb(b,i) t2_1p_abab(c,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0135_bb_voQ")(ab,kb,Q) * tmps.at("0026_bb_ooQ")(kb,ib,Q) )
    .allocate(tmps.at("0136_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0136_bb_voQ")(bb,ib,Q)  = chol.at("bb_ovQ")(jb,cb,Q) * t2_1p.at("bbbb")(cb,bb,ib,jb) )
    
    // r1_1p[bb] += +0.500 <j,k||i,b>_bbbb t2_1p_bbbb(b,a,j,k) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += 0.500 * chol.at("bb_ooQ")(jb,ib,Q) * tmps.at("0136_bb_voQ")(ab,jb,Q) )
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_abab t2_aaaa(c,a,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0136_bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += -1.000 <a,k||i,c>_abab t2_1p_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0136_bb_voQ")(bb,jb,Q) )
    
    // r1_1p[bb] += -1.000 <j,k||b,c>_abab t1_aa(b,j) t2_1p_bbbb(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0136_bb_voQ")(ab,ib,Q) * tmps.at("0049_Q")(Q) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0136_bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += -1.000 <a,k||c,d>_abab t1_aa(c,i) t2_1p_bbbb(d,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0136_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t2_1p_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0028_aa_voQ")(aa,ia,Q) * tmps.at("0136_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0137_aa_ooQ"))
    
    // flops: o2v0Q1  = o2v1Q1
    //  mems: o2v0Q1  = o2v0Q1
    ( tmps.at("0137_aa_ooQ")(ja,ia,Q)  = chol.at("aa_ovQ")(ja,aa,Q) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t1_aa(a,k) t1_1p_aa(c,i) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0137_aa_ooQ")(ka,ia,Q) * tmps.at("0136_bb_voQ")(bb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r1_2p[aa] += -1.000 <j,k||b,c>_abab t1_1p_aa(b,i) t2_1p_abab(a,c,j,k) 
    //             += -1.000 <k,j||b,c>_abab t1_1p_aa(b,i) t2_1p_abab(a,c,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0028_aa_voQ")(aa,ja,Q) * tmps.at("0137_aa_ooQ")(ja,ia,Q) )
    
    // r1_1p[aa] += -0.500 <j,k||b,c>_abab t1_1p_aa(b,i) t2_abab(a,c,j,k) 
    //             += -0.500 <k,j||b,c>_abab t1_1p_aa(b,i) t2_abab(a,c,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0025_aa_voQ")(aa,ja,Q) * tmps.at("0137_aa_ooQ")(ja,ia,Q) )
    .allocate(tmps.at("0138_aabb_oovo"))
    
    // flops: o3v1  = o3v1Q1 o3v2 o4v1 o3v1 o3v1 o3v1Q1 o3v1Q1 o3v1 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("0138_aabb_oovo")(ka,ia,bb,jb)  = tmps.at("0136_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0138_aabb_oovo")(ka,ia,bb,jb) -= t1_1p.at("bb")(cb,jb) * tmps.at("0133_aabb_oovv")(ka,ia,bb,cb) )
    ( tmps.at("0138_aabb_oovo")(ka,ia,bb,jb) += t1_1p.at("bb")(bb,lb) * tmps.at("0123_aabb_oooo")(ka,ia,lb,jb) )
    ( tmps.at("0138_aabb_oovo")(ka,ia,bb,jb) -= chol.at("bb_voQ")(bb,jb,Q) * tmps.at("0137_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0138_aabb_oovo")(ka,ia,bb,jb) -= tmps.at("0135_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    .deallocate(tmps.at("0133_aabb_oovv"))
    .deallocate(tmps.at("0123_aabb_oooo"))
    
    // r2_1p[abab] += +1.000 <k,l||i,c>_abab t1_aa(a,k) t2_1p_bbbb(c,b,j,l) 
    //               += -1.000 <k,b||i,c>_abab t1_aa(a,k) t1_1p_bb(c,j) 
    //               += +1.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_abab(c,b,l,j) 
    //               += -1.000 <k,b||c,j>_abab t1_aa(a,k) t1_1p_aa(c,i) 
    //               += +1.000 <k,l||i,c>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0138_aabb_oovo")(ka,ia,bb,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_abab t1_1p_aa(a,k) t2_1p_bbbb(c,b,j,l) 
    //               += -2.000 <k,b||i,c>_abab t1_1p_aa(a,k) t1_1p_bb(c,j) 
    //               += +2.000 <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_abab(c,b,l,j) 
    //               += -2.000 <k,b||c,j>_abab t1_1p_aa(a,k) t1_1p_aa(c,i) 
    //               += +2.000 <k,l||i,c>_abab t1_1p_aa(a,k) t1_1p_bb(b,l) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0138_aabb_oovo")(ka,ia,bb,jb) * t1_1p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0138_aabb_oovo"))
    .allocate(tmps.at("0139_aaaa_vovo"))
    
    // flops: o2v2  = o3v1Q1 o3v2 o2v2Q1 o2v2 o2v2Q1 o2v2
    //  mems: o2v2  = o3v1 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0051_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0139_aaaa_vovo")(aa,ia,ba,ja)  = t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("0139_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0025_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0139_aaaa_vovo")(aa,ia,ba,ja) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0025_aa_voQ")(ba,ja,Q) )
    
    // r2[aaaa] += +1.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_abab(b,c,j,k) 
    //            += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_aaaa(c,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_abab(b,d,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0139_aaaa_vovo")(aa,ia,ba,ja) )
    
    // r2[aaaa] += +1.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_abab(b,c,j,k) 
    //            += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_aaaa(c,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_abab(b,d,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0139_aaaa_vovo")(aa,ja,ba,ia) )
    
    // r2[aaaa] += +1.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_abab(b,c,j,k) 
    //            += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_aaaa(c,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_abab(b,d,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0139_aaaa_vovo")(ba,ia,aa,ja) )
    
    // r2[aaaa] += +1.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_abab(b,c,j,k) 
    //            += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_aaaa(c,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_abab(b,d,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0139_aaaa_vovo")(ba,ja,aa,ia) )
    .deallocate(tmps.at("0139_aaaa_vovo"))
    .allocate(tmps.at("0140_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0140_aa_voQ")(ba,ia,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t2_1p.at("aaaa")(ca,ba,ia,ja) )
    
    // r2_1p[abab] += -1.000 <k,b||d,c>_abab t1_bb(c,j) t2_1p_aaaa(d,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0140_aa_voQ")(aa,ia,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    
    // r1_1p[aa] += +0.500 <j,k||i,b>_aaaa t2_1p_aaaa(b,a,j,k) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 0.500 * chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0140_aa_voQ")(aa,ja,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_abab t1_bb(b,k) t1_1p_bb(c,j) t2_1p_aaaa(d,a,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0029_bb_ooQ")(kb,jb,Q) * tmps.at("0140_aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r1_1p[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0140_aa_voQ")(aa,ia,Q) * tmps.at("0049_Q")(Q) )
    
    // r2_1p[abab] += -1.000 <k,b||c,j>_abab t2_1p_aaaa(c,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0140_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t2_1p_aaaa(d,a,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0140_aa_voQ")(aa,ia,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += +1.000 <l,k||d,c>_abab t2_1p_aaaa(d,a,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0140_aa_voQ")(aa,ia,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t2_1p_aaaa(c,a,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0140_aa_voQ")(aa,ia,Q) * tmps.at("0136_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_1p_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0140_aa_voQ")(aa,ia,Q) * tmps.at("0135_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0141_aa_voQ"))
    
    // flops: o1v1Q1  = o1v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0141_aa_voQ")(aa,ja,Q)  = chol.at("aa_vvQ")(aa,ba,Q) * t1_1p.at("aa")(ba,ja) )
    
    // r2_2p[abab] += -2.000 <a,k||c,d>_abab t1_1p_aa(c,i) t2_1p_bbbb(d,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0141_aa_voQ")(aa,ia,Q) * tmps.at("0136_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <a,k||c,d>_aaaa t1_1p_aa(c,i) t2_1p_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0141_aa_voQ")(aa,ia,Q) * tmps.at("0135_bb_voQ")(bb,jb,Q) )
    
    // r1_1p[aa] += -1.000 <a,j||b,c>_aaaa t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0141_aa_voQ")(aa,ia,Q) * tmps.at("0049_Q")(Q) )
    
    // r2_2p[abab] += -2.000 <a,k||c,d>_abab t1_bb(b,k) t1_1p_aa(c,i) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0029_bb_ooQ")(kb,jb,Q) * tmps.at("0141_aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_1p[abab] += -1.000 <a,k||c,d>_abab t1_1p_aa(c,i) t2_bbbb(d,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0141_aa_voQ")(aa,ia,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += +1.000 <a,b||d,c>_abab t1_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0141_aa_voQ")(aa,ia,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += -1.000 <a,k||d,c>_aaaa t1_1p_aa(c,i) t2_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0141_aa_voQ")(aa,ia,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0142_aaaa_oooo"))
    
    // flops: o4v0  = o4v0Q1
    //  mems: o4v0  = o4v0
    ( tmps.at("0142_aaaa_oooo")(ka,ia,la,ja)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("aa_ooQ")(la,ja,Q) )
    .allocate(tmps.at("0143_aaaa_vovo"))
    
    // flops: o2v2  = o4v1 o4v1 o3v2 o4v1 o3v2 o2v2 o4v1 o4v1 o3v2 o3v1Q1 o3v2 o2v2 o2v2 o2v2Q1 o2v2Q1 o2v2 o2v2 o3v1Q1 o3v2 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2
    //  mems: o2v2  = o4v0 o3v1 o2v2 o3v1 o2v2 o2v2 o4v0 o3v1 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = t1.at("aa")(ca,ja) * tmps.at("0056_aaaa_ooov")(la,ia,ka,ca) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = t1.at("aa")(ba,ka) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja)  = tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) * t1_1p.at("aa")(aa,la) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = t1.at("aa")(ba,ka) * tmps.at("0142_aaaa_oooo")(la,ia,ka,ja) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) * t1_1p.at("aa")(aa,la) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = t1.at("aa")(ca,ja) * tmps.at("0056_aaaa_ooov")(ka,ia,la,ca) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1_1p.at("aa")(ba,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1.at("aa")(aa,ka) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = tmps.at("0140_aa_voQ")(ba,ja,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1.at("aa")(aa,ka) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0028_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0140_aa_voQ")(ba,ja,Q) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = tmps.at("0051_aa_voQ")(ba,ja,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1_1p.at("aa")(aa,ka) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0028_aa_voQ")(ba,ja,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0028_aa_voQ")(ba,ja,Q) * tmps.at("0052_aa_voQ")(aa,ia,Q) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0141_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0025_aa_voQ")(ba,ja,Q) * tmps.at("0141_aa_voQ")(aa,ia,Q) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_1p_abab(b,c,j,k) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += +1.000 P(i,j) P(a,b) <k,l||i,c>_aaaa t1_1p_aa(a,k) t2_aaaa(c,b,j,l) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += -1.000 P(a,b) <l,k||i,j>_aaaa t1_aa(a,k) t1_1p_aa(b,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_1p_abab(b,d,j,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_1p_aaaa(d,b,j,l) 
    //               += +1.000 P(i,j) <a,b||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_1p_abab(b,d,j,k) 
    //               += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_1p_aa(c,i) t2_abab(b,d,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0143_aaaa_vovo")(aa,ia,ba,ja) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_1p_abab(b,c,j,k) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += +1.000 P(i,j) P(a,b) <k,l||i,c>_aaaa t1_1p_aa(a,k) t2_aaaa(c,b,j,l) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += -1.000 P(a,b) <l,k||i,j>_aaaa t1_aa(a,k) t1_1p_aa(b,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_1p_abab(b,d,j,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_1p_aaaa(d,b,j,l) 
    //               += +1.000 P(i,j) <a,b||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_1p_abab(b,d,j,k) 
    //               += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_1p_aa(c,i) t2_abab(b,d,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0143_aaaa_vovo")(aa,ja,ba,ia) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_1p_abab(b,c,j,k) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += +1.000 P(i,j) P(a,b) <k,l||i,c>_aaaa t1_1p_aa(a,k) t2_aaaa(c,b,j,l) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += -1.000 P(a,b) <l,k||i,j>_aaaa t1_aa(a,k) t1_1p_aa(b,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_bbbb t2_1p_abab(a,d,i,l) t2_abab(b,c,j,k) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t2_1p_aaaa(d,a,i,l) t2_aaaa(c,b,j,k) 
    //               += +1.000 P(i,j) <a,b||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_1p_abab(b,d,j,k) 
    //               += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_1p_aa(c,i) t2_abab(b,d,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0143_aaaa_vovo")(ba,ia,aa,ja) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_1p_abab(b,c,j,k) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += +1.000 P(i,j) P(a,b) <k,l||i,c>_aaaa t1_1p_aa(a,k) t2_aaaa(c,b,j,l) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += -1.000 P(a,b) <l,k||i,j>_aaaa t1_aa(a,k) t1_1p_aa(b,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_bbbb t2_1p_abab(a,d,i,l) t2_abab(b,c,j,k) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t2_1p_aaaa(d,a,i,l) t2_aaaa(c,b,j,k) 
    //               += +1.000 P(i,j) <a,b||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,j) 
    //               += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_1p_abab(b,d,j,k) 
    //               += +1.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_1p_aa(c,i) t2_abab(b,d,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0143_aaaa_vovo")(ba,ja,aa,ia) )
    .deallocate(tmps.at("0143_aaaa_vovo"))
    .allocate(tmps.at("0144_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0144_aa_voQ")(ba,ia,Q)  = chol.at("bb_ovQ")(jb,ab,Q) * t2_2p.at("abab")(ba,ab,ia,jb) )
    
    // r1_2p[aa] += +2.000 <j,k||b,c>_abab t1_aa(b,j) t2_2p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0144_aa_voQ")(aa,ia,Q) * tmps.at("0049_Q")(Q) )
    
    // r1_2p[aa] += -1.000 <j,k||i,b>_abab t2_2p_abab(a,b,j,k) 
    //             += -1.000 <k,j||i,b>_abab t2_2p_abab(a,b,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0144_aa_voQ")(aa,ja,Q) )
    
    // r2_2p[abab] += -1.000 <k,l||c,d>_abab t2_2p_abab(a,d,k,l) t2_abab(c,b,i,j) 
    //               += -1.000 <l,k||c,d>_abab t2_2p_abab(a,d,l,k) t2_abab(c,b,i,j) 
    // flops: o2v2 += o1v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_aa_vv")(aa,ca)  = tmps.at("0144_aa_voQ")(aa,ka,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2.at("abab")(ca,bb,ia,jb) * tmps.at("bin1_aa_vv")(aa,ca) )
    
    // r1_2p[aa] += -1.000 <j,k||b,c>_abab t1_aa(b,i) t2_2p_abab(a,c,j,k) 
    //             += -1.000 <k,j||b,c>_abab t1_aa(b,i) t2_2p_abab(a,c,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0144_aa_voQ")(aa,ja,Q) * tmps.at("0053_aa_ooQ")(ja,ia,Q) )
    
    // r2_2p[abab] += +2.000 <b,k||j,c>_bbbb t2_2p_abab(a,c,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0144_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t2_2p_abab(a,d,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0144_aa_voQ")(aa,ia,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_2p_abab(a,d,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0144_aa_voQ")(aa,ia,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_2p_abab(a,c,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0144_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t2_2p_abab(a,d,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0144_aa_voQ")(aa,ia,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0145_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0145_aa_voQ")(ba,ia,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t2_2p.at("aaaa")(ca,ba,ia,ja) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t2_2p_aaaa(d,a,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0026_bb_ooQ")(kb,jb,Q) * tmps.at("0145_aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += -2.000 <k,b||c,j>_abab t2_2p_aaaa(c,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0145_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,j>_abab t1_bb(b,k) t2_2p_aaaa(c,a,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0145_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_2p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0145_aa_voQ")(aa,ia,Q) * tmps.at("0049_Q")(Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_2p_aaaa(d,a,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0145_aa_voQ")(aa,ia,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_abab t2_2p_aaaa(d,a,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0145_aa_voQ")(aa,ia,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r1_2p[aa] += +1.000 <j,k||i,b>_aaaa t2_2p_aaaa(b,a,j,k) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0145_aa_voQ")(aa,ja,Q) )
    
    // r2_2p[abab] += -2.000 <k,b||d,c>_abab t1_bb(c,j) t2_2p_aaaa(d,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0145_aa_voQ")(aa,ia,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0146_aa_voQ"))
    
    // flops: o1v1Q1  = o1v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0146_aa_voQ")(aa,ja,Q)  = chol.at("aa_vvQ")(aa,ba,Q) * t1_2p.at("aa")(ba,ja) )
    
    // r2_2p[abab] += +2.000 <a,b||d,c>_abab t1_bb(c,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0146_aa_voQ")(aa,ia,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += -2.000 <a,k||c,d>_abab t1_2p_aa(c,i) t2_bbbb(d,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0146_aa_voQ")(aa,ia,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += -2.000 <a,k||d,c>_aaaa t1_2p_aa(c,i) t2_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0146_aa_voQ")(aa,ia,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    
    // r1_2p[aa] += -2.000 <a,j||b,c>_aaaa t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0146_aa_voQ")(aa,ia,Q) * tmps.at("0049_Q")(Q) )
    
    // r2_2p[abab] += -2.000 <a,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0026_bb_ooQ")(kb,jb,Q) * tmps.at("0146_aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    .allocate(tmps.at("0147_aaaa_vovo"))
    
    // flops: o2v2  = o2v2Q1 o3v1Q1 o3v2 o3v1Q1 o3v2 o4v1 o3v2 o2v2 o3v1Q1 o3v2 o4v1 o4v1 o3v2 o2v2 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2 o2v2Q1 o4v1 o4v1 o3v2 o2v2Q1 o2v2 o4v1 o4v1 o3v2 o4v1 o4v1 o3v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2Q1 o2v2
    //  mems: o2v2  = o2v2 o3v1 o2v2 o3v1 o2v2 o3v1 o2v2 o2v2 o3v1 o2v2 o4v0 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o4v0 o3v1 o2v2 o2v2 o2v2 o4v0 o3v1 o2v2 o4v0 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja)  = tmps.at("0025_aa_voQ")(ba,ja,Q) * tmps.at("0146_aa_voQ")(aa,ia,Q) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = tmps.at("0051_aa_voQ")(ba,ja,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1_2p.at("aa")(aa,ka) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0145_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = t1.at("aa")(ba,ka) * tmps.at("0142_aaaa_oooo")(la,ia,ka,ja) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) * t1_2p.at("aa")(aa,la) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = tmps.at("0140_aa_voQ")(ba,ja,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1_1p.at("aa")(aa,ka) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = t1.at("aa")(ca,ja) * tmps.at("0056_aaaa_ooov")(ka,ia,la,ca) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1_2p.at("aa")(ba,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1.at("aa")(aa,ka) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0144_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0144_aa_voQ")(ba,ja,Q) * tmps.at("0052_aa_voQ")(aa,ia,Q) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0146_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0028_aa_voQ")(ba,ja,Q) * tmps.at("0141_aa_voQ")(aa,ia,Q) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = t1_1p.at("aa")(ca,ja) * tmps.at("0056_aaaa_ooov")(la,ia,ka,ca) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = t1.at("aa")(ba,ka) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) * t1_1p.at("aa")(aa,la) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0145_aa_voQ")(ba,ja,Q) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = t1.at("aa")(ca,ja) * tmps.at("0056_aaaa_ooov")(ka,ia,la,ca) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t1_1p.at("aa")(ba,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) * t1_1p.at("aa")(aa,ka) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = t1.at("aa")(ca,ja) * tmps.at("0056_aaaa_ooov")(la,ia,ka,ca) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = t1.at("aa")(ba,ka) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) * t1_2p.at("aa")(aa,la) )
    ( tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0144_aa_voQ")(ba,ja,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_2p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) <l,k||i,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) <k,l||i,c>_aaaa t1_2p_aa(a,k) t2_aaaa(c,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_aaaa(c,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += -2.000 P(a,b) <l,k||i,j>_aaaa t1_aa(a,k) t1_2p_aa(b,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_2p_abab(b,d,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_2p_aaaa(d,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_2p_abab(b,d,j,k) 
    //               += +2.000 P(i,j) <a,b||c,d>_aaaa t1_aa(c,i) t1_2p_aa(d,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_1p_aa(c,i) t2_1p_abab(b,d,j,k) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_2p_aa(c,i) t2_abab(b,d,j,k) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0147_aaaa_vovo")(aa,ja,ba,ia) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_2p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) <l,k||i,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) <k,l||i,c>_aaaa t1_2p_aa(a,k) t2_aaaa(c,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_aaaa(c,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += -2.000 P(a,b) <l,k||i,j>_aaaa t1_aa(a,k) t1_2p_aa(b,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_2p_abab(b,d,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_2p_aaaa(d,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_2p_abab(b,d,j,k) 
    //               += +2.000 P(i,j) <a,b||c,d>_aaaa t1_aa(c,i) t1_2p_aa(d,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_1p_aa(c,i) t2_1p_abab(b,d,j,k) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_2p_aa(c,i) t2_abab(b,d,j,k) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0147_aaaa_vovo")(aa,ia,ba,ja) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_2p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) <l,k||i,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) <k,l||i,c>_aaaa t1_2p_aa(a,k) t2_aaaa(c,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_aaaa(c,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += -2.000 P(a,b) <l,k||i,j>_aaaa t1_aa(a,k) t1_2p_aa(b,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_2p_abab(a,d,i,l) t2_abab(b,c,j,k) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_2p_aaaa(d,a,i,l) t2_aaaa(c,b,j,k) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_2p_abab(b,d,j,k) 
    //               += +2.000 P(i,j) <a,b||c,d>_aaaa t1_aa(c,i) t1_2p_aa(d,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_1p_aa(c,i) t2_1p_abab(b,d,j,k) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_2p_aa(c,i) t2_abab(b,d,j,k) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0147_aaaa_vovo")(ba,ia,aa,ja) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) P(a,b) <a,k||i,c>_abab t2_2p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) <l,k||i,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_1p_aa(b,l) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) <k,l||i,c>_aaaa t1_2p_aa(a,k) t2_aaaa(c,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t1_2p_aa(b,l) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_aaaa(c,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += -2.000 P(a,b) <l,k||i,j>_aaaa t1_aa(a,k) t1_2p_aa(b,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_2p_abab(a,d,i,l) t2_abab(b,c,j,k) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_2p_aaaa(d,a,i,l) t2_aaaa(c,b,j,k) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_aa(c,i) t2_2p_abab(b,d,j,k) 
    //               += +2.000 P(i,j) <a,b||c,d>_aaaa t1_aa(c,i) t1_2p_aa(d,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_1p_aa(c,i) t2_1p_abab(b,d,j,k) 
    //               += +2.000 P(i,j) P(a,b) <a,k||c,d>_abab t1_2p_aa(c,i) t2_abab(b,d,j,k) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0147_aaaa_vovo")(ba,ja,aa,ia) )
    .deallocate(tmps.at("0147_aaaa_vovo"))
    .allocate(tmps.at("0148_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0148_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1.at("bb")(ab,ib) )
    
    // r1_2p[aa] += +2.000 <a,j||c,b>_abab t1_bb(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0146_aa_voQ")(aa,ia,Q) * tmps.at("0148_Q")(Q) )
    
    // r1_1p[aa] += -1.000 <k,j||c,b>_abab t1_bb(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0140_aa_voQ")(aa,ia,Q) * tmps.at("0148_Q")(Q) )
    
    // r1_1p[aa] += +1.000 <a,j||c,b>_abab t1_bb(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0141_aa_voQ")(aa,ia,Q) * tmps.at("0148_Q")(Q) )
    
    // r1[aa] += +1.000 <a,j||c,b>_abab t1_bb(b,j) t1_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0148_Q")(Q) )
    
    // r1_2p[aa] += -2.000 <k,j||c,b>_abab t1_bb(b,j) t2_2p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0145_aa_voQ")(aa,ia,Q) * tmps.at("0148_Q")(Q) )
    
    // r1_2p[aa] += -2.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_2p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0144_aa_voQ")(aa,ia,Q) * tmps.at("0148_Q")(Q) )
    
    // r1_1p[aa] += -1.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0028_aa_voQ")(aa,ia,Q) * tmps.at("0148_Q")(Q) )
    
    // r1[aa] += -1.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0148_Q")(Q) )
    
    // r1[aa] += +1.000 <a,j||i,b>_abab t1_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0148_Q")(Q) )
    
    // r1[aa] += -1.000 <k,j||c,b>_abab t1_bb(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0148_Q")(Q) )
    
    // r1[bb] += +1.000 <a,j||i,b>_bbbb t1_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += tmps.at("0148_Q")(Q) * chol.at("bb_voQ")(ab,ib,Q) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("0148_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,la,jb)  = t2_2p.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,la,jb) * t1.at("aa")(aa,la) )
    
    // r1[bb] += +1.000 <k,j||c,b>_abab t1_bb(b,j) t2_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += tmps.at("0058_bb_voQ")(ab,ib,Q) * tmps.at("0148_Q")(Q) )
    
    // r1[bb] += -1.000 <a,j||b,c>_bbbb t1_bb(b,j) t1_bb(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += tmps.at("0060_bb_voQ")(ab,ib,Q) * tmps.at("0148_Q")(Q) )
    
    // r1[bb] += +1.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_bbbb(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= tmps.at("0059_bb_voQ")(ab,ib,Q) * tmps.at("0148_Q")(Q) )
    
    // r1_1p[bb] += +1.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_1p_bbbb(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0136_bb_voQ")(ab,ib,Q) * tmps.at("0148_Q")(Q) )
    
    // r1_1p[bb] += +1.000 <k,j||c,b>_abab t1_bb(b,j) t2_1p_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0135_bb_voQ")(ab,ib,Q) * tmps.at("0148_Q")(Q) )
    .allocate(tmps.at("0149_aa_vv"))
    
    // flops: o0v2  = o0v2Q1 o0v2Q1 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0149_aa_vv")(aa,da)  = tmps.at("0148_Q")(Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( tmps.at("0149_aa_vv")(aa,da) += chol.at("aa_vvQ")(aa,da,Q) * tmps.at("0049_Q")(Q) )
    
    // r2[abab] += +1.000 <a,k||d,c>_abab t1_bb(c,k) t2_abab(d,b,i,j) 
    //            += -1.000 <a,k||c,d>_aaaa t1_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0149_aa_vv")(aa,da) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2_1p[abab] += +1.000 <a,k||d,c>_abab t1_bb(c,k) t2_1p_abab(d,b,i,j) 
    //               += -1.000 <a,k||c,d>_aaaa t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0149_aa_vv")(aa,da) * t2_1p.at("abab")(da,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <a,k||d,c>_abab t1_bb(c,k) t2_2p_abab(d,b,i,j) 
    //               += -2.000 <a,k||c,d>_aaaa t1_aa(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0149_aa_vv")(aa,da) * t2_2p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0149_aa_vv"))
    .allocate(tmps.at("0150_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0150_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1_1p.at("bb")(ab,ib) )
    
    // r2_2p[abab] += -2.000 <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(c,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,ka)  = tmps.at("0150_Q")(Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_1p_bb(c,l) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,kb)  = chol.at("bb_ovQ")(kb,db,Q) * tmps.at("0150_Q")(Q) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin1_bb_vo")(db,kb) * t2_1p.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r1_2p[bb] += +2.000 <k,j||c,b>_abab t1_1p_bb(b,j) t2_1p_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0135_bb_voQ")(ab,ib,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_2p[aa] += -2.000 <k,j||b,c>_bbbb t1_1p_bb(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0028_aa_voQ")(aa,ia,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_2p[aa] += +2.000 <a,j||c,b>_abab t1_1p_bb(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0141_aa_voQ")(aa,ia,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_2p[aa] += -2.000 <k,j||c,b>_abab t1_1p_bb(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0140_aa_voQ")(aa,ia,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_1p[aa] += +1.000 <a,j||i,b>_abab t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_1p[aa] += +1.000 <a,j||b,c>_abab t1_aa(b,i) t1_1p_bb(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_2p[bb] += +2.000 <k,j||b,c>_bbbb t1_1p_bb(b,j) t2_1p_bbbb(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0136_bb_voQ")(ab,ib,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_1p[aa] += -1.000 <j,k||c,b>_bbbb t1_1p_bb(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_1p[aa] += -1.000 <k,j||c,b>_abab t1_1p_bb(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_1p[bb] += +1.000 <j,k||c,b>_bbbb t1_1p_bb(b,j) t2_bbbb(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0059_bb_voQ")(ab,ib,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_1p[bb] += +1.000 <a,j||i,b>_bbbb t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0150_Q")(Q) * chol.at("bb_voQ")(ab,ib,Q) )
    
    // r1_1p[bb] += +1.000 <k,j||c,b>_abab t1_1p_bb(b,j) t2_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0058_bb_voQ")(ab,ib,Q) * tmps.at("0150_Q")(Q) )
    
    // r1_1p[bb] += +1.000 <a,j||b,c>_bbbb t1_bb(b,i) t1_1p_bb(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0060_bb_voQ")(ab,ib,Q) * tmps.at("0150_Q")(Q) )
    .allocate(tmps.at("0151_Q"))
    ;
  }
  // clang-format on
}

template void exachem::cc::cd_qed_ccsd_os::resid_part3<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);