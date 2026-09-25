/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_qed_ccsd_cs_resid_4.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_cs::resid_part4(
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
        
    // r2_2p[abab] += -1.000 <k,l||c,d>_abab t2_2p_abab(a,d,i,j) t2_abab(c,b,k,l) 
    //               += -1.000 <l,k||c,d>_abab t2_2p_abab(a,d,i,j) t2_abab(c,b,l,k) 
    //               += -1.000 <k,l||c,d>_bbbb t2_2p_abab(a,d,i,j) t2_bbbb(c,b,k,l) 
    //               += -2.000 <b,k||c,d>_bbbb t1_bb(c,k) t2_2p_abab(a,d,i,j) 
    //               += +2.000 <k,b||c,d>_abab t1_aa(c,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0145_bb_vv")(bb,db) * t2_2p.at("abab")(aa,db,ia,jb) )
    .deallocate(tmps.at("0145_bb_vv"))
    .allocate(tmps.at("0146_baab_vooo"))
    
    // flops: o3v1  = o1v1Q1 o3v2 o3v2 o3v1
    //  mems: o3v1  = o1v1 o3v1 o3v1 o3v1
    ( tmps.at("bin1_aa_vo")(da,la)  = chol.at("aa_ovQ")(la,da,Q) * tmps.at("0030_Q")(Q) )
    ( tmps.at("0146_baab_vooo")(bb,la,ia,jb)  = tmps.at("bin1_aa_vo")(da,la) * t2.at("abab")(da,bb,ia,jb) )
    ( tmps.at("0146_baab_vooo")(bb,la,ia,jb) -= t2.at("abab")(da,bb,ia,jb) * tmps.at("0093_aa_vo")(da,la) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    //            += +1.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0146_baab_vooo")(bb,la,ia,jb) * t1.at("aa")(aa,la) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    //               += +1.000 <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0146_baab_vooo")(bb,la,ia,jb) * t1_1p.at("aa")(aa,la) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_2p_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    //               += +2.000 <l,k||c,d>_aaaa t1_2p_aa(a,l) t1_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0146_baab_vooo")(bb,la,ia,jb) * t1_2p.at("aa")(aa,la) )
    .deallocate(tmps.at("0146_baab_vooo"))
    .allocate(tmps.at("0147_bb_oo"))
    
    // flops: o2v0  = o3v2 o3v1 o2v0 o3v2 o2v0 o3v2 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0 o2v0 o2v0 o2v0 o2v0
    ( tmps.at("0147_bb_oo")(jb,lb)  = 0.500 * tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) * t2.at("bbbb")(db,cb,jb,kb) )
    ( tmps.at("0147_bb_oo")(jb,lb) -= t1.at("bb")(cb,kb) * tmps.at("0091_bbbb_ooov")(kb,jb,lb,cb) )
    ( tmps.at("0147_bb_oo")(jb,lb) -= 0.500 * tmps.at("0055_bbbb_ovov")(kb,db,lb,cb) * t2.at("bbbb")(db,cb,jb,kb) )
    ( tmps.at("0147_bb_oo")(jb,lb) += tmps.at("0053_aabb_ovov")(ka,da,lb,cb) * t2.at("abab")(da,cb,ka,jb) )
    .deallocate(tmps.at("0091_bbbb_ooov"))
    
    // r2[abab] += -0.500 <k,l||d,c>_abab t2_abab(a,b,i,l) t2_abab(d,c,k,j) 
    //            += -0.500 <k,l||c,d>_abab t2_abab(a,b,i,l) t2_abab(c,d,k,j) 
    //            += -0.500 <l,k||d,c>_bbbb t2_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    //            += -0.500 <l,k||d,c>_bbbb t2_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    //            += -1.000 <l,k||j,c>_bbbb t1_bb(c,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0147_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_1p[abab] += -0.500 <k,l||d,c>_abab t2_1p_abab(a,b,i,l) t2_abab(d,c,k,j) 
    //               += -0.500 <k,l||c,d>_abab t2_1p_abab(a,b,i,l) t2_abab(c,d,k,j) 
    //               += -0.500 <l,k||d,c>_bbbb t2_1p_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    //               += -0.500 <l,k||d,c>_bbbb t2_1p_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    //               += -1.000 <l,k||j,c>_bbbb t1_bb(c,k) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0147_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_2p_abab(a,b,i,l) t2_abab(d,c,k,j) 
    //               += -1.000 <k,l||c,d>_abab t2_2p_abab(a,b,i,l) t2_abab(c,d,k,j) 
    //               += -1.000 <l,k||d,c>_bbbb t2_2p_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    //               += -1.000 <l,k||d,c>_bbbb t2_2p_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    //               += -2.000 <l,k||j,c>_bbbb t1_bb(c,k) t2_2p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0147_bb_oo")(jb,lb) * t2_2p.at("abab")(aa,bb,ia,lb) )
    .deallocate(tmps.at("0147_bb_oo"))
    .allocate(tmps.at("0148_abab_voov"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin1_bbbb_vvoo")(cb,db,kb,lb)  = chol.at("bb_ovQ")(lb,cb,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0148_abab_voov")(aa,lb,ia,db)  = t2_1p.at("abab")(aa,cb,ia,kb) * tmps.at("bin1_bbbb_vvoo")(cb,db,kb,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_1p_bb(c,j) t2_1p_abab(a,d,i,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0148_abab_voov")(aa,kb,ia,cb) * t1_1p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t2_1p_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0148_abab_voov")(aa,lb,ia,db) * t2_1p.at("bbbb")(db,bb,jb,lb) )
    .allocate(tmps.at("0149_abab_vooo"))
    
    // flops: o3v1  = o3v2 o3v3 o3v1 o3v2 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("0149_abab_vooo")(aa,kb,ia,jb)  = f.at("bb_ov")(kb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    ( tmps.at("0149_abab_vooo")(aa,kb,ia,jb) += t2_1p.at("abab")(da,cb,ia,jb) * tmps.at("0065_aabb_vvov")(aa,da,kb,cb) )
    ( tmps.at("0149_abab_vooo")(aa,kb,ia,jb) -= t1.at("bb")(cb,jb) * tmps.at("0148_abab_voov")(aa,kb,ia,cb) )
    .deallocate(tmps.at("0148_abab_voov"))
    .deallocate(tmps.at("0065_aabb_vvov"))
    
    // r2_1p[abab] += -1.000 f_bb(k,c) t1_bb(b,k) t2_1p_abab(a,c,i,j) 
    //               += -0.500 <a,k||d,c>_abab t1_bb(b,k) t2_1p_abab(d,c,i,j) 
    //               += -0.500 <a,k||c,d>_abab t1_bb(b,k) t2_1p_abab(c,d,i,j) 
    //               += +1.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_1p_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0149_abab_vooo")(aa,kb,ia,jb) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -2.000 f_bb(k,c) t1_1p_bb(b,k) t2_1p_abab(a,c,i,j) 
    //               += -1.000 <a,k||d,c>_abab t1_1p_bb(b,k) t2_1p_abab(d,c,i,j) 
    //               += -1.000 <a,k||c,d>_abab t1_1p_bb(b,k) t2_1p_abab(c,d,i,j) 
    //               += +2.000 <l,k||c,d>_bbbb t1_1p_bb(b,k) t1_bb(c,j) t2_1p_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0149_abab_vooo")(aa,kb,ia,jb) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0149_abab_vooo"))
    .allocate(tmps.at("0150_baab_vooo"))
    
    // flops: o3v1  = o1v1Q1 o3v2 o3v2 o3v1
    //  mems: o3v1  = o1v1 o3v1 o3v1 o3v1
    ( tmps.at("bin1_aa_vo")(da,la)  = chol.at("aa_ovQ")(la,da,Q) * tmps.at("0030_Q")(Q) )
    ( tmps.at("0150_baab_vooo")(bb,la,ia,jb)  = tmps.at("bin1_aa_vo")(da,la) * t2_1p.at("abab")(da,bb,ia,jb) )
    ( tmps.at("0150_baab_vooo")(bb,la,ia,jb) -= t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("0093_aa_vo")(da,la) )
    .deallocate(tmps.at("0093_aa_vo"))
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    //               += +1.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0150_baab_vooo")(bb,la,ia,jb) * t1.at("aa")(aa,la) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    //               += +2.000 <l,k||c,d>_aaaa t1_1p_aa(a,l) t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0150_baab_vooo")(bb,la,ia,jb) * t1_1p.at("aa")(aa,la) )
    .deallocate(tmps.at("0150_baab_vooo"))
    .allocate(tmps.at("0151_bb_vv"))
    
    // flops: o0v2  = o2v3 o2v3 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0151_bb_vv")(bb,cb)  = -0.500 * t2_1p.at("bbbb")(db,bb,kb,lb) * tmps.at("0055_bbbb_ovov")(lb,db,kb,cb) )
    ( tmps.at("0151_bb_vv")(bb,cb) += t2_1p.at("abab")(da,bb,ka,lb) * tmps.at("0053_aabb_ovov")(ka,da,lb,cb) )
    
    // r2_1p[abab] += -0.500 <k,l||d,c>_abab t2_abab(a,c,i,j) t2_1p_abab(d,b,k,l) 
    //               += -0.500 <l,k||d,c>_abab t2_abab(a,c,i,j) t2_1p_abab(d,b,l,k) 
    //               += +0.500 <k,l||c,d>_bbbb t2_abab(a,c,i,j) t2_1p_bbbb(d,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0151_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p[abab] += -1.000 <k,l||c,d>_abab t2_1p_abab(a,d,i,j) t2_1p_abab(c,b,k,l) 
    //               += -1.000 <l,k||c,d>_abab t2_1p_abab(a,d,i,j) t2_1p_abab(c,b,l,k) 
    //               += +1.000 <k,l||d,c>_bbbb t2_1p_abab(a,d,i,j) t2_1p_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0151_bb_vv")(bb,db) * t2_1p.at("abab")(aa,db,ia,jb) )
    .deallocate(tmps.at("0151_bb_vv"))
    .allocate(tmps.at("0152_aa_vv"))
    
    // flops: o0v2  = o2v3 o2v3 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0152_aa_vv")(ba,aa)  = -0.500 * t2.at("aaaa")(ca,aa,ja,ka) * tmps.at("0058_aaaa_ovov")(ja,ba,ka,ca) )
    ( tmps.at("0152_aa_vv")(ba,aa) += t2.at("abab")(aa,cb,ja,kb) * tmps.at("0053_aabb_ovov")(ja,ba,kb,cb) )
    
    // r1[aa] += -0.500 <j,k||b,c>_abab t1_aa(b,i) t2_abab(a,c,j,k) 
    //          += -0.500 <k,j||b,c>_abab t1_aa(b,i) t2_abab(a,c,k,j) 
    //          += +0.500 <j,k||b,c>_aaaa t1_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0152_aa_vv")(ba,aa) * t1.at("aa")(ba,ia) )
    
    // r1_1p[aa] += -0.500 <j,k||b,c>_abab t1_1p_aa(b,i) t2_abab(a,c,j,k) 
    //             += -0.500 <k,j||b,c>_abab t1_1p_aa(b,i) t2_abab(a,c,k,j) 
    //             += -0.500 <j,k||c,b>_aaaa t1_1p_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0152_aa_vv")(ba,aa) * t1_1p.at("aa")(ba,ia) )
    
    // r1_2p[aa] += -1.000 <j,k||b,c>_abab t1_2p_aa(b,i) t2_abab(a,c,j,k) 
    //             += -1.000 <k,j||b,c>_abab t1_2p_aa(b,i) t2_abab(a,c,k,j) 
    //             += -1.000 <j,k||c,b>_aaaa t1_2p_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0152_aa_vv")(ba,aa) * t1_2p.at("aa")(ba,ia) )
    
    // r2[abab] += -0.500 <k,l||d,c>_abab t2_abab(a,c,k,l) t2_abab(d,b,i,j) 
    //            += -0.500 <l,k||d,c>_abab t2_abab(a,c,l,k) t2_abab(d,b,i,j) 
    //            += +0.500 <k,l||d,c>_aaaa t2_aaaa(c,a,k,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0152_aa_vv")(da,aa) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2_1p[abab] += -0.500 <k,l||d,c>_abab t2_abab(a,c,k,l) t2_1p_abab(d,b,i,j) 
    //               += -0.500 <l,k||d,c>_abab t2_abab(a,c,l,k) t2_1p_abab(d,b,i,j) 
    //               += -0.500 <k,l||c,d>_aaaa t2_aaaa(c,a,k,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0152_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_abab(a,c,k,l) t2_2p_abab(d,b,i,j) 
    //               += -1.000 <l,k||d,c>_abab t2_abab(a,c,l,k) t2_2p_abab(d,b,i,j) 
    //               += -1.000 <k,l||c,d>_aaaa t2_aaaa(c,a,k,l) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0152_aa_vv")(da,aa) * t2_2p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0152_aa_vv"))
    .allocate(tmps.at("0153_aa_vv"))
    
    // flops: o0v2  = o2v3 o2v3 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0153_aa_vv")(ba,aa)  = -0.500 * t2_1p.at("aaaa")(ca,aa,ja,ka) * tmps.at("0058_aaaa_ovov")(ja,ba,ka,ca) )
    ( tmps.at("0153_aa_vv")(ba,aa) += t2_1p.at("abab")(aa,cb,ja,kb) * tmps.at("0053_aabb_ovov")(ja,ba,kb,cb) )
    
    // r1_1p[aa] += -0.500 <j,k||b,c>_abab t1_aa(b,i) t2_1p_abab(a,c,j,k) 
    //             += -0.500 <k,j||b,c>_abab t1_aa(b,i) t2_1p_abab(a,c,k,j) 
    //             += +0.500 <j,k||b,c>_aaaa t1_aa(b,i) t2_1p_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0153_aa_vv")(ba,aa) * t1.at("aa")(ba,ia) )
    
    // r1_2p[aa] += -1.000 <j,k||b,c>_abab t1_1p_aa(b,i) t2_1p_abab(a,c,j,k) 
    //             += -1.000 <k,j||b,c>_abab t1_1p_aa(b,i) t2_1p_abab(a,c,k,j) 
    //             += +1.000 <j,k||b,c>_aaaa t1_1p_aa(b,i) t2_1p_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0153_aa_vv")(ba,aa) * t1_1p.at("aa")(ba,ia) )
    
    // r2_1p[abab] += -0.500 <k,l||c,d>_abab t2_1p_abab(a,d,k,l) t2_abab(c,b,i,j) 
    //               += -0.500 <l,k||c,d>_abab t2_1p_abab(a,d,l,k) t2_abab(c,b,i,j) 
    //               += +0.500 <k,l||c,d>_aaaa t2_1p_aaaa(d,a,k,l) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0153_aa_vv")(ca,aa) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_1p_abab(a,c,k,l) t2_1p_abab(d,b,i,j) 
    //               += -1.000 <l,k||d,c>_abab t2_1p_abab(a,c,l,k) t2_1p_abab(d,b,i,j) 
    //               += +1.000 <k,l||d,c>_aaaa t2_1p_aaaa(c,a,k,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0153_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0153_aa_vv"))
    .allocate(tmps.at("0154_aa_vv"))
    
    // flops: o0v2  = o2v3 o2v3 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0154_aa_vv")(ba,aa)  = -0.500 * t2_2p.at("aaaa")(ca,aa,ja,ka) * tmps.at("0058_aaaa_ovov")(ja,ba,ka,ca) )
    ( tmps.at("0154_aa_vv")(ba,aa) += t2_2p.at("abab")(aa,cb,ja,kb) * tmps.at("0053_aabb_ovov")(ja,ba,kb,cb) )
    
    // r1_2p[aa] += -1.000 <j,k||b,c>_abab t1_aa(b,i) t2_2p_abab(a,c,j,k) 
    //             += -1.000 <k,j||b,c>_abab t1_aa(b,i) t2_2p_abab(a,c,k,j) 
    //             += +1.000 <j,k||b,c>_aaaa t1_aa(b,i) t2_2p_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0154_aa_vv")(ba,aa) * t1.at("aa")(ba,ia) )
    
    // r2_2p[abab] += -1.000 <k,l||c,d>_abab t2_2p_abab(a,d,k,l) t2_abab(c,b,i,j) 
    //               += -1.000 <l,k||c,d>_abab t2_2p_abab(a,d,l,k) t2_abab(c,b,i,j) 
    //               += +1.000 <k,l||c,d>_aaaa t2_2p_aaaa(d,a,k,l) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0154_aa_vv")(ca,aa) * t2.at("abab")(ca,bb,ia,jb) )
    .deallocate(tmps.at("0154_aa_vv"))
    .allocate(tmps.at("0155_aa_oo"))
    
    // flops: o2v0  = o3v2 o3v2 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0155_aa_oo")(ja,ia)  = -1.000 * t2_1p.at("aaaa")(ca,ba,ia,ka) * tmps.at("0058_aaaa_ovov")(ja,ca,ka,ba) )
    ( tmps.at("0155_aa_oo")(ja,ia) += t2_1p.at("aaaa")(ca,ba,ia,ka) * tmps.at("0058_aaaa_ovov")(ja,ba,ka,ca) )
    
    // r1_1p[aa] += +0.500 <k,j||c,b>_aaaa t1_aa(a,j) t2_1p_aaaa(c,b,i,k) 
    //             += +0.500 <k,j||c,b>_aaaa t1_aa(a,j) t2_1p_aaaa(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 0.500 * tmps.at("0155_aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    
    // r1_2p[aa] += +1.000 <k,j||c,b>_aaaa t1_1p_aa(a,j) t2_1p_aaaa(c,b,i,k) 
    //             += +1.000 <k,j||c,b>_aaaa t1_1p_aa(a,j) t2_1p_aaaa(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += tmps.at("0155_aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    
    // r2_1p[abab] += +0.500 <l,k||d,c>_aaaa t2_abab(a,b,k,j) t2_1p_aaaa(d,c,i,l) 
    //               += +0.500 <l,k||d,c>_aaaa t2_abab(a,b,k,j) t2_1p_aaaa(d,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.500 * tmps.at("0155_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    .deallocate(tmps.at("0155_aa_oo"))
    .allocate(tmps.at("0156_aa_oo"))
    
    // flops: o2v0  = o3v1 o3v1 o2v0 o2v0Q1 o2v0 o2v0Q1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0 o2v0 o2v0 o2v0 o2v0
    ( tmps.at("0156_aa_oo")(ja,ia)  = t1_1p.at("aa")(ba,ka) * tmps.at("0088_aaaa_ooov")(ja,ia,ka,ba) )
    ( tmps.at("0156_aa_oo")(ja,ia) += t1_1p.at("bb")(bb,kb) * tmps.at("0122_aabb_ooov")(ja,ia,kb,bb) )
    ( tmps.at("0156_aa_oo")(ja,ia) += tmps.at("0098_aa_ooQ")(ja,ia,Q) * tmps.at("0032_Q")(Q) )
    ( tmps.at("0156_aa_oo")(ja,ia) += tmps.at("0098_aa_ooQ")(ja,ia,Q) * tmps.at("0033_Q")(Q) )
    
    // r1_1p[aa] += +1.000 <k,j||i,b>_aaaa t1_aa(a,j) t1_1p_aa(b,k) 
    //             += -1.000 <j,k||i,b>_abab t1_aa(a,j) t1_1p_bb(b,k) 
    //             += -1.000 <j,k||b,c>_abab t1_aa(a,j) t1_aa(b,i) t1_1p_bb(c,k) 
    //             += +1.000 <k,j||b,c>_aaaa t1_aa(a,j) t1_aa(b,i) t1_1p_aa(c,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0156_aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    
    // r1_2p[aa] += -2.000 <k,j||i,b>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,j) 
    //             += -2.000 <k,j||i,b>_abab t1_1p_aa(a,k) t1_1p_bb(b,j) 
    //             += -2.000 <k,j||b,c>_abab t1_1p_aa(a,k) t1_aa(b,i) t1_1p_bb(c,j) 
    //             += -2.000 <k,j||b,c>_aaaa t1_1p_aa(a,k) t1_aa(b,i) t1_1p_aa(c,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0156_aa_oo")(ka,ia) * t1_1p.at("aa")(aa,ka) )
    
    // r2_1p[abab] += +1.000 <k,l||i,c>_aaaa t1_1p_aa(c,k) t2_abab(a,b,l,j) 
    //               += -1.000 <l,k||i,c>_abab t1_1p_bb(c,k) t2_abab(a,b,l,j) 
    //               += -1.000 <l,k||c,d>_abab t1_aa(c,i) t1_1p_bb(d,k) t2_abab(a,b,l,j) 
    //               += +1.000 <k,l||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,k) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0156_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||i,c>_aaaa t1_1p_aa(c,k) t2_1p_abab(a,b,l,j) 
    //               += -2.000 <l,k||i,c>_abab t1_1p_bb(c,k) t2_1p_abab(a,b,l,j) 
    //               += -2.000 <l,k||c,d>_abab t1_aa(c,i) t1_1p_bb(d,k) t2_1p_abab(a,b,l,j) 
    //               += -2.000 <l,k||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0156_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0156_aa_oo"))
    .allocate(tmps.at("0157_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0157_bb_voQ")(bb,jb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t2_2p.at("bbbb")(cb,bb,jb,kb) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_abab t1_aa(a,k) t2_2p_bbbb(c,b,j,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0157_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -2.000 <a,k||i,c>_abab t2_2p_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0157_bb_voQ")(bb,jb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t2_aaaa(c,a,i,k) t2_2p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0116_aa_voQ")(aa,ia,Q) * tmps.at("0157_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += -2.000 <a,k||c,d>_abab t1_aa(c,i) t2_2p_bbbb(d,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0102_aa_voQ")(aa,ia,Q) * tmps.at("0157_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0158_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0158_bb_voQ")(bb,jb,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t2_2p.at("abab")(ca,bb,ka,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_abab(c,b,l,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0158_bb_voQ")(bb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 <a,k||i,c>_aaaa t2_2p_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0158_bb_voQ")(bb,jb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[abab] += +2.000 <a,k||c,d>_aaaa t1_aa(c,i) t2_2p_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0102_aa_voQ")(aa,ia,Q) * tmps.at("0158_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0116_aa_voQ")(aa,ia,Q) * tmps.at("0158_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0159_bb_voQ"))
    
    // flops: o1v1Q1  = o1v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0159_bb_voQ")(bb,jb,Q)  = chol.at("bb_vvQ")(bb,cb,Q) * t1_2p.at("bb")(cb,jb) )
    
    // r2_2p[abab] += -2.000 <k,b||c,d>_abab t1_aa(a,k) t1_aa(c,i) t1_2p_bb(d,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0159_bb_voQ")(bb,jb,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 <k,b||d,c>_abab t1_2p_bb(c,j) t2_aaaa(d,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0116_aa_voQ")(aa,ia,Q) * tmps.at("0159_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <a,b||c,d>_abab t1_aa(c,i) t1_2p_bb(d,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0102_aa_voQ")(aa,ia,Q) * tmps.at("0159_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <a,b||i,c>_abab t1_2p_bb(c,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0159_bb_voQ")(bb,jb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[abab] += +2.000 <b,k||d,c>_bbbb t1_2p_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o1v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,db)  = chol.at("bb_ovQ")(kb,db,Q) * tmps.at("0159_bb_voQ")(bb,kb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_vv")(bb,db) * t2.at("abab")(aa,db,ia,jb) )
    .allocate(tmps.at("0160_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0160_aa_voQ")(ba,ia,Q)  = chol.at("bb_ovQ")(jb,ab,Q) * t2_2p.at("abab")(ba,ab,ia,jb) )
    
    // r1_2p[aa] += +2.000 <j,k||b,c>_abab t1_aa(b,j) t2_2p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0160_aa_voQ")(aa,ia,Q) * tmps.at("0030_Q")(Q) )
    
    // r1_2p[aa] += -1.000 <j,k||i,b>_abab t2_2p_abab(a,b,j,k) 
    //             += -1.000 <k,j||i,b>_abab t2_2p_abab(a,b,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0160_aa_voQ")(aa,ja,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_2p_abab(a,c,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = chol.at("bb_ooQ")(kb,jb,Q) * tmps.at("0160_aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +2.000 <b,k||j,c>_bbbb t2_2p_abab(a,c,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0160_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_2p_abab(a,d,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0160_aa_voQ")(aa,ia,Q) * tmps.at("0106_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t2_2p_abab(a,d,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0160_aa_voQ")(aa,ia,Q) * tmps.at("0026_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t2_2p_abab(a,d,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0160_aa_voQ")(aa,ia,Q) * tmps.at("0027_bb_voQ")(bb,jb,Q) )
    
    // r1_2p[aa] += -2.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_2p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0160_aa_voQ")(aa,ia,Q) * tmps.at("0029_Q")(Q) )
    .allocate(tmps.at("0161_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0161_aa_voQ")(ca,ia,Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||d,c>_abab t2_abab(a,c,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0097_bb_voQ")(bb,jb,Q) )
    
    // r1[aa] += -1.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0029_Q")(Q) )
    
    // r1[aa] += +1.000 <j,k||b,c>_abab t1_aa(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0030_Q")(Q) )
    
    // r1[aa] += -0.500 <j,k||i,b>_abab t2_abab(a,b,j,k) 
    //          += -0.500 <k,j||i,b>_abab t2_abab(a,b,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0161_aa_voQ")(aa,ja,Q) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0096_bb_voQ")(bb,jb,Q) )
    
    // r1_1p[aa] += -1.000 <j,k||c,b>_bbbb t1_1p_bb(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0032_Q")(Q) )
    
    // r2[abab] += +1.000 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0026_bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += -1.000 <b,k||d,c>_bbbb t1_1p_bb(c,j) t2_abab(a,d,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0079_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_abab t2_abab(a,c,i,k) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0158_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0158_bb_voQ"))
    
    // r2[abab] += +1.000 <b,k||j,c>_bbbb t2_abab(a,c,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0161_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += +1.000 <l,k||d,c>_abab t2_abab(a,c,i,k) t2_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0027_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += +1.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_abab(a,d,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0106_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += -2.000 <b,k||d,c>_bbbb t1_2p_bb(c,j) t2_abab(a,d,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0159_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0159_bb_voQ"))
    
    // r1_1p[aa] += +1.000 <j,k||b,c>_abab t1_1p_aa(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0033_Q")(Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_2p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0157_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0157_bb_voQ"))
    .allocate(tmps.at("0162_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0162_Q")(Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t1_2p.at("bb")(bb,jb) )
    
    // r1_2p[aa] += +2.000 <a,j||i,b>_abab t1_2p_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0162_Q")(Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r1_2p[aa] += +2.000 <a,j||b,c>_abab t1_aa(b,i) t1_2p_bb(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0102_aa_voQ")(aa,ia,Q) * tmps.at("0162_Q")(Q) )
    
    // r1_2p[aa] += -2.000 <k,j||c,b>_abab t1_2p_bb(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0116_aa_voQ")(aa,ia,Q) * tmps.at("0162_Q")(Q) )
    
    // r1_2p[aa] += -2.000 <j,k||c,b>_bbbb t1_2p_bb(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0162_Q")(Q) )
    
    // r2_2p[abab] += -2.000 <k,l||d,c>_abab t1_aa(a,k) t1_2p_bb(c,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,ka)  = chol.at("aa_ovQ")(ka,da,Q) * tmps.at("0162_Q")(Q) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin1_aa_vo")(da,ka) * t2.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_2p_bb(c,l) t2_abab(a,d,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,kb)  = tmps.at("0162_Q")(Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t2.at("abab")(aa,db,ia,jb) * tmps.at("bin1_bb_vo")(db,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +2.000 <a,k||d,c>_abab t1_2p_bb(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o0v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_aa_vv")(aa,da)  = tmps.at("0162_Q")(Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vv")(aa,da) )
    
    // r2_2p[abab] += +2.000 <b,k||d,c>_bbbb t1_2p_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o0v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,db)  = tmps.at("0162_Q")(Q) * chol.at("bb_vvQ")(bb,db,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,db,ia,jb) * tmps.at("bin1_bb_vv")(bb,db) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_bbbb t1_bb(c,j) t1_2p_bb(d,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = tmps.at("0099_bb_ooQ")(lb,jb,Q) * tmps.at("0162_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    .allocate(tmps.at("0163_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0163_Q")(Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t1_2p.at("aa")(ba,ja) )
    
    // r1_2p[aa] += +2.000 <a,j||b,c>_aaaa t1_aa(b,i) t1_2p_aa(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0102_aa_voQ")(aa,ia,Q) * tmps.at("0163_Q")(Q) )
    
    // r1_2p[aa] += +2.000 <a,j||i,b>_aaaa t1_2p_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0163_Q")(Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r1_2p[aa] += +2.000 <j,k||b,c>_abab t1_2p_aa(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0161_aa_voQ")(aa,ia,Q) * tmps.at("0163_Q")(Q) )
    .deallocate(tmps.at("0161_aa_voQ"))
    
    // r1_2p[aa] += +2.000 <j,k||c,b>_aaaa t1_2p_aa(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0116_aa_voQ")(aa,ia,Q) * tmps.at("0163_Q")(Q) )
    .deallocate(tmps.at("0116_aa_voQ"))
    
    // r2_2p[abab] += -2.000 <k,l||d,c>_abab t1_bb(c,j) t1_2p_aa(d,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = tmps.at("0099_bb_ooQ")(lb,jb,Q) * tmps.at("0163_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_abab t1_bb(b,k) t1_2p_aa(c,l) t2_abab(a,d,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,kb)  = tmps.at("0163_Q")(Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t2.at("abab")(aa,db,ia,jb) * tmps.at("bin1_bb_vo")(db,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +2.000 <a,k||d,c>_aaaa t1_2p_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o0v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_aa_vv")(aa,da)  = tmps.at("0163_Q")(Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vv")(aa,da) )
    
    // r2_2p[abab] += +2.000 <k,b||c,d>_abab t1_2p_aa(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o0v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,db)  = tmps.at("0163_Q")(Q) * chol.at("bb_vvQ")(bb,db,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,db,ia,jb) * tmps.at("bin1_bb_vv")(bb,db) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,ka)  = tmps.at("0163_Q")(Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r0_2p() += -2.000 <j,i||a,b>_aaaa t1_aa(a,i) t1_2p_aa(b,j) 
    //       += +2.000 <i,j||a,b>_abab t1_aa(a,i) t1_2p_bb(b,j) 
    //       += -1.000 <i,j||b,a>_aaaa t1_1p_aa(a,i) t1_1p_aa(b,j) 
    //       += +1.000 <i,j||a,b>_abab t1_1p_aa(a,i) t1_1p_bb(b,j) 
    //       += +1.000 <j,i||b,a>_abab t1_1p_bb(a,i) t1_1p_aa(b,j) 
    //       += +0.500 <i,j||b,a>_aaaa t2_2p_aaaa(b,a,i,j) 
    //       += -2.000 <j,i||a,b>_aaaa t1_aa(a,i) t1_2p_aa(b,j) 
    //       += +0.500 <i,j||b,a>_aaaa t2_2p_aaaa(b,a,i,j) 
    //       += +0.500 <i,j||b,a>_abab t2_2p_abab(b,a,i,j) 
    //       += +0.500 <j,i||b,a>_abab t2_2p_abab(b,a,j,i) 
    //       += +0.500 <i,j||a,b>_abab t2_2p_abab(a,b,i,j) 
    //       += +0.500 <j,i||a,b>_abab t2_2p_abab(a,b,j,i) 
    //       += +2.000 <j,i||b,a>_abab t1_bb(a,i) t1_2p_aa(b,j) 
    //       += -2.000 <j,i||a,b>_bbbb t1_bb(a,i) t1_2p_bb(b,j) 
    //       += -1.000 <i,j||b,a>_bbbb t1_1p_bb(a,i) t1_1p_bb(b,j) 
    //       += +0.500 <i,j||b,a>_bbbb t2_2p_bbbb(b,a,i,j) 
    //       += -2.000 <j,i||a,b>_bbbb t1_bb(a,i) t1_2p_bb(b,j) 
    //       += +0.500 <i,j||b,a>_bbbb t2_2p_bbbb(b,a,i,j) 
    //       += +2.000 f_aa(i,a) t1_2p_aa(a,i) 
    //       += +2.000 f_bb(i,a) t1_2p_bb(a,i) 
    //       += -1.000 <i,j||b,a>_aaaa t1_1p_aa(a,i) t1_1p_aa(b,j) 
    //       += -1.000 <i,j||b,a>_bbbb t1_1p_bb(a,i) t1_1p_bb(b,j) 
    //       += +4.000 d-_aa(i,a) t0_2p t1_1p_aa(a,i) 
    //       += +4.000 d-_bb(i,a) t0_2p t1_1p_bb(a,i) 
    //       += +2.000 d-_aa(i,a) t0_1p t1_2p_aa(a,i) 
    //       += +2.000 d-_bb(i,a) t0_1p t1_2p_bb(a,i) 
    //       += +4.000 t0_2p w0 
    // flops: 0 += 0 o2v2Q1 o1v1Q1 o0v0Q1 0 o0v0Q1 0 o0v0Q1 o2v1Q1 o1v1 0 0 o2v2Q1 o1v1Q1 0 o0v0Q1 0 o0v0Q1 0 o2v2Q1 o1v1Q1 0 o1v1Q1 o0v0Q1 0 o2v1Q1 o1v1 0 0 o2v2Q1 o1v1Q1 0 o2v1Q1 o2v1Q1 o1v1 0 0 o0v0Q1 o1v1 0 0 o2v1Q1 o2v1Q1 o1v1 0 0 0 0 0 o1v1 0
    //  mems: 0 += 0 o1v1Q1 0 0 0 0 0 0 o1v1 0 0 0 o1v1Q1 0 0 0 0 0 0 o1v1Q1 0 0 0 0 0 o1v1 0 0 0 o1v1Q1 0 0 o2v0Q1 o1v1 0 0 0 0 0 0 0 o2v0Q1 o1v1 0 0 0 0 0 0 0 0
    ( r0_2p() += 4.000 * t0_2p * scalars.at("0008")() )
    ( tmps.at("bin1_aa_voQ")(aa,ia,Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t2_2p.at("aaaa")(ba,aa,ia,ja) )
    ( r0_2p() -= 0.500 * tmps.at("bin1_aa_voQ")(aa,ia,Q) * chol.at("aa_ovQ")(ia,aa,Q) )
    ( r0_2p() += 2.000 * tmps.at("0030_Q")(Q) * tmps.at("0162_Q")(Q) )
    ( r0_2p() += 2.000 * tmps.at("0030_Q")(Q) * tmps.at("0163_Q")(Q) )
    ( r0_2p() += tmps.at("0033_Q")(Q) * tmps.at("0033_Q")(Q) )
    ( tmps.at("bin1_aa_vo")(ba,ja)  = tmps.at("0098_aa_ooQ")(ja,ia,Q) * chol.at("aa_ovQ")(ia,ba,Q) )
    ( r0_2p() -= 2.000 * t1_2p.at("aa")(ba,ja) * tmps.at("bin1_aa_vo")(ba,ja) )
    ( tmps.at("bin1_aa_voQ")(ba,ia,Q)  = chol.at("aa_ovQ")(ja,aa,Q) * t2_2p.at("aaaa")(ba,aa,ia,ja) )
    ( r0_2p() += 0.500 * tmps.at("bin1_aa_voQ")(ba,ia,Q) * chol.at("aa_ovQ")(ia,ba,Q) )
    ( r0_2p() += 2.000 * tmps.at("0032_Q")(Q) * tmps.at("0033_Q")(Q) )
    ( r0_2p() += tmps.at("0032_Q")(Q) * tmps.at("0032_Q")(Q) )
    ( tmps.at("bin1_bb_voQ")(ab,ib,Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t2_2p.at("bbbb")(bb,ab,ib,jb) )
    ( r0_2p() -= 0.500 * tmps.at("bin1_bb_voQ")(ab,ib,Q) * chol.at("bb_ovQ")(ib,ab,Q) )
    ( r0_2p() += 2.000 * tmps.at("0160_aa_voQ")(ba,ia,Q) * chol.at("aa_ovQ")(ia,ba,Q) )
    ( r0_2p() += 2.000 * tmps.at("0029_Q")(Q) * tmps.at("0163_Q")(Q) )
    ( tmps.at("bin1_bb_vo")(bb,jb)  = tmps.at("0099_bb_ooQ")(jb,ib,Q) * chol.at("bb_ovQ")(ib,bb,Q) )
    ( r0_2p() -= 2.000 * t1_2p.at("bb")(bb,jb) * tmps.at("bin1_bb_vo")(bb,jb) )
    ( tmps.at("bin1_bb_voQ")(bb,ib,Q)  = chol.at("bb_ovQ")(jb,ab,Q) * t2_2p.at("bbbb")(bb,ab,ib,jb) )
    ( r0_2p() += 0.500 * tmps.at("bin1_bb_voQ")(bb,ib,Q) * chol.at("bb_ovQ")(ib,bb,Q) )
    ( tmps.at("bin1_aa_ooQ")(ia,ja,Q)  = chol.at("aa_ovQ")(ja,aa,Q) * t1_1p.at("aa")(aa,ia) )
    ( tmps.at("bin1_aa_vo")(ba,ja)  = chol.at("aa_ovQ")(ia,ba,Q) * tmps.at("bin1_aa_ooQ")(ia,ja,Q) )
    ( r0_2p() -= tmps.at("bin1_aa_vo")(ba,ja) * t1_1p.at("aa")(ba,ja) )
    ( r0_2p() += 2.000 * tmps.at("0029_Q")(Q) * tmps.at("0162_Q")(Q) )
    ( r0_2p() += 2.000 * f.at("aa_ov")(ia,aa) * t1_2p.at("aa")(aa,ia) )
    ( tmps.at("bin1_bb_ooQ")(ib,jb,Q)  = chol.at("bb_ovQ")(jb,ab,Q) * t1_1p.at("bb")(ab,ib) )
    ( tmps.at("bin1_bb_vo")(bb,jb)  = chol.at("bb_ovQ")(ib,bb,Q) * tmps.at("bin1_bb_ooQ")(ib,jb,Q) )
    ( r0_2p() -= tmps.at("bin1_bb_vo")(bb,jb) * t1_1p.at("bb")(bb,jb) )
    ( r0_2p() += 2.000 * t0_1p * scalars.at("0009")() )
    ( r0_2p() += 4.000 * w0 * t0_2p )
    ( r0_2p() += 2.000 * f.at("bb_ov")(ib,ab) * t1_2p.at("bb")(ab,ib) )
    .deallocate(tmps.at("0160_aa_voQ"))
    .allocate(tmps.at("0164_aa_oo"))
    
    // flops: o2v0  = o3v1 o3v1 o2v0 o2v0Q1 o2v0 o2v0Q1 o2v0 o2v0Q1 o2v0 o2v0Q1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0 o2v0 o2v0 o2v0 o2v0 o2v0 o2v0 o2v0 o2v0
    ( tmps.at("0164_aa_oo")(ja,ia)  = tmps.at("0088_aaaa_ooov")(ja,ia,ka,ba) * t1_2p.at("aa")(ba,ka) )
    ( tmps.at("0164_aa_oo")(ja,ia) += tmps.at("0122_aabb_ooov")(ja,ia,kb,bb) * t1_2p.at("bb")(bb,kb) )
    ( tmps.at("0164_aa_oo")(ja,ia) += tmps.at("0098_aa_ooQ")(ja,ia,Q) * tmps.at("0162_Q")(Q) )
    ( tmps.at("0164_aa_oo")(ja,ia) += tmps.at("0098_aa_ooQ")(ja,ia,Q) * tmps.at("0163_Q")(Q) )
    ( tmps.at("0164_aa_oo")(ja,ia) += tmps.at("0032_Q")(Q) * tmps.at("0123_aa_ooQ")(ja,ia,Q) )
    ( tmps.at("0164_aa_oo")(ja,ia) += tmps.at("0033_Q")(Q) * tmps.at("0123_aa_ooQ")(ja,ia,Q) )
    .deallocate(tmps.at("0163_Q"))
    .deallocate(tmps.at("0162_Q"))
    
    // r1_2p[aa] += +2.000 <k,j||i,b>_aaaa t1_aa(a,j) t1_2p_aa(b,k) 
    //             += -2.000 <j,k||i,b>_abab t1_aa(a,j) t1_2p_bb(b,k) 
    //             += -2.000 <j,k||b,c>_abab t1_aa(a,j) t1_aa(b,i) t1_2p_bb(c,k) 
    //             += +2.000 <k,j||b,c>_aaaa t1_aa(a,j) t1_aa(b,i) t1_2p_aa(c,k) 
    //             += -2.000 <j,k||c,b>_abab t1_aa(a,j) t1_1p_bb(b,k) t1_1p_aa(c,i) 
    //             += -2.000 <k,j||b,c>_aaaa t1_aa(a,j) t1_1p_aa(b,k) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * t1.at("aa")(aa,ja) * tmps.at("0164_aa_oo")(ja,ia) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_aaaa t1_2p_aa(c,k) t2_abab(a,b,l,j) 
    //               += -2.000 <l,k||i,c>_abab t1_2p_bb(c,k) t2_abab(a,b,l,j) 
    //               += -2.000 <l,k||c,d>_abab t1_aa(c,i) t1_2p_bb(d,k) t2_abab(a,b,l,j) 
    //               += +2.000 <k,l||c,d>_aaaa t1_aa(c,i) t1_2p_aa(d,k) t2_abab(a,b,l,j) 
    //               += -2.000 <l,k||d,c>_abab t1_1p_bb(c,k) t1_1p_aa(d,i) t2_abab(a,b,l,j) 
    //               += -2.000 <k,l||c,d>_aaaa t1_1p_aa(c,k) t1_1p_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2.at("abab")(aa,bb,la,jb) * tmps.at("0164_aa_oo")(la,ia) )
    .deallocate(tmps.at("0164_aa_oo"))
    .allocate(tmps.at("0165_aa_oo"))
    
    // flops: o2v0  = o3v1 o3v1 o2v0 o2v0Q1 o2v0 o2v0Q1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0 o2v0 o2v0 o2v0 o2v0
    ( tmps.at("0165_aa_oo")(ka,ia)  = t1.at("aa")(ba,ja) * tmps.at("0088_aaaa_ooov")(ka,ia,ja,ba) )
    ( tmps.at("0165_aa_oo")(ka,ia) += t1.at("bb")(bb,jb) * tmps.at("0122_aabb_ooov")(ka,ia,jb,bb) )
    ( tmps.at("0165_aa_oo")(ka,ia) += tmps.at("0029_Q")(Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0165_aa_oo")(ka,ia) += tmps.at("0030_Q")(Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    
    // r1[aa] += -1.000 <k,j||i,b>_aaaa t1_aa(a,k) t1_aa(b,j) 
    //          += -1.000 <k,j||i,b>_abab t1_aa(a,k) t1_bb(b,j) 
    //          += -1.000 <k,j||c,b>_abab t1_aa(a,k) t1_bb(b,j) t1_aa(c,i) 
    //          += +1.000 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0165_aa_oo")(ka,ia) * t1.at("aa")(aa,ka) )
    
    // r1_1p[aa] += -1.000 <k,j||i,b>_aaaa t1_1p_aa(a,k) t1_aa(b,j) 
    //             += -1.000 <k,j||i,b>_abab t1_1p_aa(a,k) t1_bb(b,j) 
    //             += -1.000 <k,j||c,b>_abab t1_1p_aa(a,k) t1_bb(b,j) t1_aa(c,i) 
    //             += +1.000 <k,j||b,c>_aaaa t1_1p_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0165_aa_oo")(ka,ia) * t1_1p.at("aa")(aa,ka) )
    
    // r1_2p[aa] += -2.000 <k,j||i,b>_aaaa t1_2p_aa(a,k) t1_aa(b,j) 
    //             += -2.000 <k,j||i,b>_abab t1_2p_aa(a,k) t1_bb(b,j) 
    //             += -2.000 <k,j||c,b>_abab t1_2p_aa(a,k) t1_bb(b,j) t1_aa(c,i) 
    //             += +2.000 <k,j||b,c>_aaaa t1_2p_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0165_aa_oo")(ka,ia) * t1_2p.at("aa")(aa,ka) )
    
    // r2[abab] += -1.000 <l,k||i,c>_aaaa t1_aa(c,k) t2_abab(a,b,l,j) 
    //            += -1.000 <l,k||i,c>_abab t1_bb(c,k) t2_abab(a,b,l,j) 
    //            += -1.000 <l,k||d,c>_abab t1_bb(c,k) t1_aa(d,i) t2_abab(a,b,l,j) 
    //            += +1.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0165_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_1p[abab] += -1.000 <l,k||i,c>_aaaa t1_aa(c,k) t2_1p_abab(a,b,l,j) 
    //               += -1.000 <l,k||i,c>_abab t1_bb(c,k) t2_1p_abab(a,b,l,j) 
    //               += -1.000 <l,k||d,c>_abab t1_bb(c,k) t1_aa(d,i) t2_1p_abab(a,b,l,j) 
    //               += +1.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0165_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||i,c>_aaaa t1_aa(c,k) t2_2p_abab(a,b,l,j) 
    //               += -2.000 <l,k||i,c>_abab t1_bb(c,k) t2_2p_abab(a,b,l,j) 
    //               += -2.000 <l,k||d,c>_abab t1_bb(c,k) t1_aa(d,i) t2_2p_abab(a,b,l,j) 
    //               += +2.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0165_aa_oo")(la,ia) * t2_2p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0165_aa_oo"))
    .allocate(tmps.at("0166_aabb_oovo"))
    
    // flops: o3v1  = o4v1 o4v0Q1 o4v1 o3v1
    //  mems: o3v1  = o3v1 o4v0 o3v1 o3v1
    ( tmps.at("0166_aabb_oovo")(ka,ia,bb,jb)  = t1.at("bb")(bb,lb) * tmps.at("0100_aabb_oooo")(ka,ia,lb,jb) )
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb)  = tmps.at("0099_bb_ooQ")(lb,jb,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0166_aabb_oovo")(ka,ia,bb,jb) += t1.at("bb")(bb,lb) * tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) )
    .deallocate(tmps.at("0100_aabb_oooo"))
    
    // r2[abab] += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t1_aa(c,i) t1_bb(d,j) 
    //            += +1.000 <k,l||c,j>_abab t1_aa(a,k) t1_bb(b,l) t1_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0166_aabb_oovo")(ka,ia,bb,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_abab t1_1p_aa(a,l) t1_bb(b,k) t1_aa(c,i) t1_bb(d,j) 
    //               += +1.000 <l,k||c,j>_abab t1_1p_aa(a,l) t1_bb(b,k) t1_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0166_aabb_oovo")(la,ia,bb,jb) * t1_1p.at("aa")(aa,la) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_abab t1_2p_aa(a,l) t1_bb(b,k) t1_aa(c,i) t1_bb(d,j) 
    //               += +2.000 <l,k||c,j>_abab t1_2p_aa(a,l) t1_bb(b,k) t1_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0166_aabb_oovo")(la,ia,bb,jb) * t1_2p.at("aa")(aa,la) )
    .deallocate(tmps.at("0166_aabb_oovo"))
    .allocate(tmps.at("0167_aabb_oovo"))
    
    // flops: o3v1  = o4v1 o4v1 o4v0Q1 o4v1 o3v1
    //  mems: o3v1  = o4v0 o3v1 o4v0 o3v1 o3v1
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb)  = tmps.at("0049_aabb_ovoo")(ka,ca,lb,jb) * t1_1p.at("aa")(ca,ia) )
    ( tmps.at("0167_aabb_oovo")(ka,ia,bb,jb)  = t1.at("bb")(bb,lb) * tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) )
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb)  = tmps.at("0132_bb_ooQ")(lb,jb,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0167_aabb_oovo")(ka,ia,bb,jb) += t1.at("bb")(bb,lb) * tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) )
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t1_aa(c,i) t1_1p_bb(d,j) 
    //               += +1.000 <k,l||c,j>_abab t1_aa(a,k) t1_bb(b,l) t1_1p_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0167_aabb_oovo")(ka,ia,bb,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_abab t1_1p_aa(a,l) t1_bb(b,k) t1_aa(c,i) t1_1p_bb(d,j) 
    //               += +2.000 <l,k||c,j>_abab t1_1p_aa(a,l) t1_bb(b,k) t1_1p_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0167_aabb_oovo")(la,ia,bb,jb) * t1_1p.at("aa")(aa,la) )
    .deallocate(tmps.at("0167_aabb_oovo"))
    .allocate(tmps.at("0168_aa_ooQ"))
    
    // flops: o2v0Q1  = o2v1Q1
    //  mems: o2v0Q1  = o2v0Q1
    ( tmps.at("0168_aa_ooQ")(ja,ia,Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t1_2p.at("aa")(ba,ia) )
    
    // r2_2p[abab] += -2.000 <k,b||c,j>_abab t1_aa(a,k) t1_2p_aa(c,i) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0168_aa_ooQ")(ka,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -2.000 <k,b||d,c>_abab t1_aa(a,k) t1_bb(c,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0106_bb_voQ")(bb,jb,Q) * tmps.at("0168_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r1_2p[aa] += -2.000 <a,j||b,c>_aaaa t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0102_aa_voQ")(aa,ja,Q) * tmps.at("0168_aa_ooQ")(ja,ia,Q) )
    
    // r2_2p[abab] += -2.000 <a,k||d,c>_aaaa t1_2p_aa(c,i) t2_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_aaaa_vvoo")(aa,da,ia,ka)  = tmps.at("0168_aa_ooQ")(ka,ia,Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2.at("abab")(da,bb,ka,jb) * tmps.at("bin1_aaaa_vvoo")(aa,da,ia,ka) )
    .allocate(tmps.at("0169_aa_oo"))
    
    // flops: o2v0  = o2v0Q1 o2v0Q1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0169_aa_oo")(ka,ia)  = tmps.at("0168_aa_ooQ")(ka,ia,Q) * tmps.at("0029_Q")(Q) )
    ( tmps.at("0169_aa_oo")(ka,ia) += tmps.at("0168_aa_ooQ")(ka,ia,Q) * tmps.at("0030_Q")(Q) )
    
    // r1_2p[aa] += -2.000 <k,j||c,b>_abab t1_aa(a,k) t1_bb(b,j) t1_2p_aa(c,i) 
    //             += +2.000 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0169_aa_oo")(ka,ia) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_abab t1_bb(c,k) t1_2p_aa(d,i) t2_abab(a,b,l,j) 
    //               += +2.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_2p_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0169_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0169_aa_oo"))
    .allocate(tmps.at("0170_aa_oo"))
    
    // flops: o2v0  = o2v0Q1 o2v0Q1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0170_aa_oo")(ka,ia)  = tmps.at("0029_Q")(Q) * tmps.at("0123_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0170_aa_oo")(ka,ia) += tmps.at("0030_Q")(Q) * tmps.at("0123_aa_ooQ")(ka,ia,Q) )
    
    // r1_1p[aa] += -1.000 <k,j||c,b>_abab t1_aa(a,k) t1_bb(b,j) t1_1p_aa(c,i) 
    //             += +1.000 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0170_aa_oo")(ka,ia) * t1.at("aa")(aa,ka) )
    
    // r1_2p[aa] += -2.000 <k,j||c,b>_abab t1_1p_aa(a,k) t1_bb(b,j) t1_1p_aa(c,i) 
    //             += +2.000 <k,j||b,c>_aaaa t1_1p_aa(a,k) t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0170_aa_oo")(ka,ia) * t1_1p.at("aa")(aa,ka) )
    
    // r2_1p[abab] += -1.000 <l,k||d,c>_abab t1_bb(c,k) t1_1p_aa(d,i) t2_abab(a,b,l,j) 
    //               += +1.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_1p_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0170_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_abab t1_bb(c,k) t1_1p_aa(d,i) t2_1p_abab(a,b,l,j) 
    //               += +2.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_1p_aa(d,i) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0170_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0170_aa_oo"))
    .allocate(tmps.at("0171_baba_ovoo"))
    
    // flops: o3v1  = o3v2 o4v2 o3v2 o4v2 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb)  = t1_1p.at("bb")(cb,jb) * tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) )
    ( tmps.at("0171_baba_ovoo")(kb,aa,jb,ia)  = t2.at("abab")(aa,db,ia,lb) * tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb) )
    ( tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb)  = t1.at("bb")(cb,jb) * tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) )
    ( tmps.at("0171_baba_ovoo")(kb,aa,jb,ia) += t2_1p.at("abab")(aa,db,ia,lb) * tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb) )
    .deallocate(tmps.at("0055_bbbb_ovov"))
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_1p_abab(a,d,i,l) 
    //               += -1.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_1p_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0171_baba_ovoo")(kb,aa,jb,ia) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_1p_bb(b,k) t1_bb(c,j) t2_1p_abab(a,d,i,l) 
    //               += +2.000 <k,l||d,c>_bbbb t1_1p_bb(b,k) t1_1p_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0171_baba_ovoo")(kb,aa,jb,ia) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0171_baba_ovoo"))
    .allocate(tmps.at("0172_abab_ovoo"))
    
    // flops: o3v1  = o3v2 o4v2 o3v2 o4v2 o3v1 o3v2 o4v2 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("bin1_baab_vooo")(db,ia,ka,lb)  = tmps.at("0053_aabb_ovov")(ka,ca,lb,db) * t1.at("aa")(ca,ia) )
    ( tmps.at("0172_abab_ovoo")(ka,bb,ia,jb)  = tmps.at("bin1_baab_vooo")(db,ia,ka,lb) * t2_1p.at("bbbb")(db,bb,jb,lb) )
    ( tmps.at("bin1_baab_vooo")(db,ia,ka,lb)  = tmps.at("0053_aabb_ovov")(ka,ca,lb,db) * t1_1p.at("aa")(ca,ia) )
    ( tmps.at("0172_abab_ovoo")(ka,bb,ia,jb) += tmps.at("bin1_baab_vooo")(db,ia,ka,lb) * t2.at("bbbb")(db,bb,jb,lb) )
    ( tmps.at("bin1_aaaa_vooo")(da,ia,ka,la)  = tmps.at("0058_aaaa_ovov")(ka,da,la,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("0172_abab_ovoo")(ka,bb,ia,jb) += tmps.at("bin1_aaaa_vooo")(da,ia,ka,la) * t2_1p.at("abab")(da,bb,la,jb) )
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_1p_aa(c,i) t2_bbbb(d,b,j,l) 
    //               += +1.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0172_abab_ovoo")(ka,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t1_1p_aa(a,k) t1_aa(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +2.000 <k,l||c,d>_abab t1_1p_aa(a,k) t1_1p_aa(c,i) t2_bbbb(d,b,j,l) 
    //               += +2.000 <l,k||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_1p.at("aa")(aa,ka) * tmps.at("0172_abab_ovoo")(ka,bb,ia,jb) )
    .deallocate(tmps.at("0172_abab_ovoo"))
    .allocate(tmps.at("0173_abba_ovoo"))
    
    // flops: o3v1  = o3v2 o4v2 o3v2 o4v2 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("bin1_aabb_vooo")(da,ka,jb,lb)  = t1_1p.at("bb")(cb,jb) * tmps.at("0053_aabb_ovov")(ka,da,lb,cb) )
    ( tmps.at("0173_abba_ovoo")(ka,bb,jb,ia)  = t2.at("abab")(da,bb,ia,lb) * tmps.at("bin1_aabb_vooo")(da,ka,jb,lb) )
    ( tmps.at("bin1_aabb_vooo")(da,ka,jb,lb)  = t1.at("bb")(cb,jb) * tmps.at("0053_aabb_ovov")(ka,da,lb,cb) )
    ( tmps.at("0173_abba_ovoo")(ka,bb,jb,ia) += t2_1p.at("abab")(da,bb,ia,lb) * tmps.at("bin1_aabb_vooo")(da,ka,jb,lb) )
    
    // r2_1p[abab] += +1.000 <k,l||d,c>_abab t1_aa(a,k) t1_bb(c,j) t2_1p_abab(d,b,i,l) 
    //               += +1.000 <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(c,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0173_abba_ovoo")(ka,bb,jb,ia) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_abab t1_1p_aa(a,k) t1_bb(c,j) t2_1p_abab(d,b,i,l) 
    //               += +2.000 <k,l||d,c>_abab t1_1p_aa(a,k) t1_1p_bb(c,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_1p.at("aa")(aa,ka) * tmps.at("0173_abba_ovoo")(ka,bb,jb,ia) )
    .deallocate(tmps.at("0173_abba_ovoo"))
    .allocate(tmps.at("0174_aabb_oooo"))
    
    // flops: o4v0  = o4v0Q1 o4v1 o4v0 o4v0Q1 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0 o4v0 o4v0
    ( tmps.at("0174_aabb_oooo")(ka,ia,lb,jb)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0174_aabb_oooo")(ka,ia,lb,jb) += t1.at("bb")(cb,jb) * tmps.at("0122_aabb_ooov")(ka,ia,lb,cb) )
    ( tmps.at("0174_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0098_aa_ooQ")(ka,ia,Q) * tmps.at("0099_bb_ooQ")(lb,jb,Q) )
    
    // r2[abab] += +0.500 <k,l||i,j>_abab t2_abab(a,b,k,l) 
    //            += +0.500 <l,k||i,j>_abab t2_abab(a,b,l,k) 
    //            += +0.500 <k,l||i,c>_abab t1_bb(c,j) t2_abab(a,b,k,l) 
    //            += +0.500 <l,k||i,c>_abab t1_bb(c,j) t2_abab(a,b,l,k) 
    //            += +0.500 <k,l||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_abab(a,b,k,l) 
    //            += +0.500 <l,k||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0174_aabb_oooo")(ka,ia,lb,jb) * t2.at("abab")(aa,bb,ka,lb) )
    
    // r2_1p[abab] += +0.500 <k,l||i,j>_abab t2_1p_abab(a,b,k,l) 
    //               += +0.500 <l,k||i,j>_abab t2_1p_abab(a,b,l,k) 
    //               += +0.500 <k,l||i,c>_abab t1_bb(c,j) t2_1p_abab(a,b,k,l) 
    //               += +0.500 <l,k||i,c>_abab t1_bb(c,j) t2_1p_abab(a,b,l,k) 
    //               += +0.500 <k,l||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_1p_abab(a,b,k,l) 
    //               += +0.500 <l,k||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_1p_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0174_aabb_oooo")(ka,ia,lb,jb) * t2_1p.at("abab")(aa,bb,ka,lb) )
    
    // r2_2p[abab] += +1.000 <k,l||i,j>_abab t2_2p_abab(a,b,k,l) 
    //               += +1.000 <l,k||i,j>_abab t2_2p_abab(a,b,l,k) 
    //               += +1.000 <k,l||i,c>_abab t1_bb(c,j) t2_2p_abab(a,b,k,l) 
    //               += +1.000 <l,k||i,c>_abab t1_bb(c,j) t2_2p_abab(a,b,l,k) 
    //               += +1.000 <k,l||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_2p_abab(a,b,k,l) 
    //               += +1.000 <l,k||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_2p_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0174_aabb_oooo")(ka,ia,lb,jb) * t2_2p.at("abab")(aa,bb,ka,lb) )
    
    // r2_2p[abab] += +2.000 <k,l||i,j>_abab t1_aa(a,k) t1_2p_bb(b,l) 
    //               += +2.000 <k,l||i,c>_abab t1_aa(a,k) t1_2p_bb(b,l) t1_bb(c,j) 
    //               += +2.000 <k,l||c,d>_abab t1_aa(a,k) t1_2p_bb(b,l) t1_aa(c,i) t1_bb(d,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t1_2p.at("bb")(bb,lb) * tmps.at("0174_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0174_aabb_oooo"))
    .allocate(tmps.at("0175_bb_ooQ"))
    
    // flops: o2v0Q1  = o2v1Q1
    //  mems: o2v0Q1  = o2v0Q1
    ( tmps.at("0175_bb_ooQ")(kb,jb,Q)  = chol.at("bb_ovQ")(kb,bb,Q) * t1_2p.at("bb")(bb,jb) )
    
    // r2_2p[abab] += -2.000 <a,k||i,c>_abab t1_bb(b,k) t1_2p_bb(c,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0175_bb_ooQ")(kb,jb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(c,k) t1_2p_bb(d,j) t2_abab(a,b,i,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = tmps.at("0175_bb_ooQ")(lb,jb,Q) * tmps.at("0029_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_abab t1_aa(c,k) t1_2p_bb(d,j) t2_abab(a,b,i,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = tmps.at("0175_bb_ooQ")(lb,jb,Q) * tmps.at("0030_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[abab] += -2.000 <b,k||d,c>_bbbb t1_2p_bb(c,j) t2_abab(a,d,i,k) 
    // flops: o2v2 += o2v2Q1 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_bbbb_vvoo")(bb,db,jb,kb)  = tmps.at("0175_bb_ooQ")(kb,jb,Q) * chol.at("bb_vvQ")(bb,db,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2.at("abab")(aa,db,ia,kb) * tmps.at("bin1_bbbb_vvoo")(bb,db,jb,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t1_2p_bb(d,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0175_bb_ooQ")(kb,jb,Q) * tmps.at("0102_aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    .allocate(tmps.at("0176_aabb_oooo"))
    
    // flops: o4v0  = o4v1 o4v1 o4v0 o4v0Q1 o4v0 o4v0Q1 o4v0 o4v0Q1 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0 o4v0 o4v0 o4v0 o4v0 o4v0 o4v0
    ( tmps.at("0176_aabb_oooo")(ka,ia,lb,jb)  = t1_2p.at("bb")(cb,jb) * tmps.at("0122_aabb_ooov")(ka,ia,lb,cb) )
    ( tmps.at("0176_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0049_aabb_ovoo")(ka,ca,lb,jb) * t1_2p.at("aa")(ca,ia) )
    ( tmps.at("0176_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0168_aa_ooQ")(ka,ia,Q) * tmps.at("0099_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0176_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0175_bb_ooQ")(lb,jb,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0176_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0123_aa_ooQ")(ka,ia,Q) * tmps.at("0132_bb_ooQ")(lb,jb,Q) )
    .deallocate(tmps.at("0175_bb_ooQ"))
    .deallocate(tmps.at("0168_aa_ooQ"))
    
    // r2_2p[abab] += +1.000 <k,l||i,c>_abab t1_2p_bb(c,j) t2_abab(a,b,k,l) 
    //               += +1.000 <l,k||i,c>_abab t1_2p_bb(c,j) t2_abab(a,b,l,k) 
    //               += +1.000 <k,l||c,d>_abab t1_aa(c,i) t1_2p_bb(d,j) t2_abab(a,b,k,l) 
    //               += +1.000 <l,k||c,d>_abab t1_aa(c,i) t1_2p_bb(d,j) t2_abab(a,b,l,k) 
    //               += +1.000 <k,l||c,d>_abab t1_1p_aa(c,i) t1_1p_bb(d,j) t2_abab(a,b,k,l) 
    //               += +1.000 <l,k||c,d>_abab t1_1p_aa(c,i) t1_1p_bb(d,j) t2_abab(a,b,l,k) 
    //               += +1.000 <k,l||c,j>_abab t1_2p_aa(c,i) t2_abab(a,b,k,l) 
    //               += +1.000 <l,k||c,j>_abab t1_2p_aa(c,i) t2_abab(a,b,l,k) 
    //               += +1.000 <k,l||d,c>_abab t1_bb(c,j) t1_2p_aa(d,i) t2_abab(a,b,k,l) 
    //               += +1.000 <l,k||d,c>_abab t1_bb(c,j) t1_2p_aa(d,i) t2_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0176_aabb_oooo")(ka,ia,lb,jb) * t2.at("abab")(aa,bb,ka,lb) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_abab t1_aa(a,k) t1_bb(b,l) t1_2p_bb(c,j) 
    //               += +2.000 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t1_aa(c,i) t1_2p_bb(d,j) 
    //               += +2.000 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t1_1p_aa(c,i) t1_1p_bb(d,j) 
    //               += +2.000 <k,l||c,j>_abab t1_aa(a,k) t1_bb(b,l) t1_2p_aa(c,i) 
    //               += +2.000 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t1_bb(c,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t1.at("bb")(bb,lb) * tmps.at("0176_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0176_aabb_oooo"))
    .allocate(tmps.at("0177_aabb_oooo"))
    
    // flops: o4v0  = o4v1 o4v0Q1 o4v0 o4v0Q1 o4v1 o4v0 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0 o4v0 o4v0 o4v0 o4v0
    ( tmps.at("0177_aabb_oooo")(ka,ia,lb,jb)  = t1_1p.at("bb")(cb,jb) * tmps.at("0122_aabb_ooov")(ka,ia,lb,cb) )
    ( tmps.at("0177_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0099_bb_ooQ")(lb,jb,Q) * tmps.at("0123_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0177_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0098_aa_ooQ")(ka,ia,Q) * tmps.at("0132_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0177_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0049_aabb_ovoo")(ka,ca,lb,jb) * t1_1p.at("aa")(ca,ia) )
    .deallocate(tmps.at("0123_aa_ooQ"))
    .deallocate(tmps.at("0122_aabb_ooov"))
    .deallocate(tmps.at("0098_aa_ooQ"))
    .deallocate(tmps.at("0049_aabb_ovoo"))
    
    // r2_1p[abab] += +0.500 <k,l||i,c>_abab t1_1p_bb(c,j) t2_abab(a,b,k,l) 
    //               += +0.500 <l,k||i,c>_abab t1_1p_bb(c,j) t2_abab(a,b,l,k) 
    //               += +0.500 <k,l||c,d>_abab t1_aa(c,i) t1_1p_bb(d,j) t2_abab(a,b,k,l) 
    //               += +0.500 <l,k||c,d>_abab t1_aa(c,i) t1_1p_bb(d,j) t2_abab(a,b,l,k) 
    //               += +0.500 <k,l||c,j>_abab t1_1p_aa(c,i) t2_abab(a,b,k,l) 
    //               += +0.500 <l,k||c,j>_abab t1_1p_aa(c,i) t2_abab(a,b,l,k) 
    //               += +0.500 <k,l||d,c>_abab t1_bb(c,j) t1_1p_aa(d,i) t2_abab(a,b,k,l) 
    //               += +0.500 <l,k||d,c>_abab t1_bb(c,j) t1_1p_aa(d,i) t2_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0177_aabb_oooo")(ka,ia,lb,jb) * t2.at("abab")(aa,bb,ka,lb) )
    
    // r2_2p[abab] += +1.000 <k,l||i,c>_abab t1_1p_bb(c,j) t2_1p_abab(a,b,k,l) 
    //               += +1.000 <l,k||i,c>_abab t1_1p_bb(c,j) t2_1p_abab(a,b,l,k) 
    //               += +1.000 <k,l||c,d>_abab t1_aa(c,i) t1_1p_bb(d,j) t2_1p_abab(a,b,k,l) 
    //               += +1.000 <l,k||c,d>_abab t1_aa(c,i) t1_1p_bb(d,j) t2_1p_abab(a,b,l,k) 
    //               += +1.000 <k,l||c,j>_abab t1_1p_aa(c,i) t2_1p_abab(a,b,k,l) 
    //               += +1.000 <l,k||c,j>_abab t1_1p_aa(c,i) t2_1p_abab(a,b,l,k) 
    //               += +1.000 <k,l||d,c>_abab t1_bb(c,j) t1_1p_aa(d,i) t2_1p_abab(a,b,k,l) 
    //               += +1.000 <l,k||d,c>_abab t1_bb(c,j) t1_1p_aa(d,i) t2_1p_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0177_aabb_oooo")(ka,ia,lb,jb) * t2_1p.at("abab")(aa,bb,ka,lb) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_1p_bb(c,j) 
    //               += +2.000 <k,l||c,d>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_aa(c,i) t1_1p_bb(d,j) 
    //               += +2.000 <k,l||c,j>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_1p_aa(c,i) 
    //               += +2.000 <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t1_1p.at("bb")(bb,lb) * tmps.at("0177_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0177_aabb_oooo"))
    .allocate(tmps.at("0178_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0178_aa_oo")(la,ia)  = tmps.at("0061_aaaa_ovoo")(la,ca,ia,ka) * t1.at("aa")(ca,ka) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0178_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0178_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0178_aa_oo")(la,ia) * t2_2p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0178_aa_oo"))
    .allocate(tmps.at("0179_aaaa_ooov"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0179_aaaa_ooov")(ja,ia,ka,ca)  = t1_1p.at("aa")(ba,ia) * tmps.at("0058_aaaa_ovov")(ja,ba,ka,ca) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,la,jb) * tmps.at("0179_aaaa_ooov")(ka,ia,la,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,la,jb) * tmps.at("0179_aaaa_ooov")(la,ia,ka,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    .allocate(tmps.at("0180_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0180_aa_oo")(ia,ka)  = tmps.at("0179_aaaa_ooov")(ja,ia,ka,ba) * t1.at("aa")(ba,ja) )
    
    // r1_1p[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0180_aa_oo")(ia,ka) * t1.at("aa")(aa,ka) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_1p_aa(a,k) t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1_1p.at("aa")(aa,ka) * tmps.at("0180_aa_oo")(ia,ka) )
    .deallocate(tmps.at("0180_aa_oo"))
    .allocate(tmps.at("0181_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0181_aa_voQ")(ba,ia,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t2_1p.at("aaaa")(ca,ba,ia,ja) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t2_1p_aaaa(c,a,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0181_aa_voQ")(aa,ia,Q) * tmps.at("0096_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0096_bb_voQ"))
    
    // r2_1p[abab] += -1.000 <k,b||c,j>_abab t2_1p_aaaa(c,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0181_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r1_2p[aa] += -2.000 <k,j||c,b>_abab t1_1p_bb(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0181_aa_voQ")(aa,ia,Q) * tmps.at("0032_Q")(Q) )
    .deallocate(tmps.at("0032_Q"))
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_1p_aa(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0181_aa_voQ")(aa,ia,Q) * tmps.at("0033_Q")(Q) )
    .deallocate(tmps.at("0033_Q"))
    
    // r1_1p[aa] += +0.500 <a,j||c,b>_aaaa t2_1p_aaaa(c,b,i,j) 
    // flops: o1v1 += o1v2Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.500 * tmps.at("0181_aa_voQ")(ba,ia,Q) * chol.at("aa_vvQ")(aa,ba,Q) )
    
    // r2_1p[abab] += +1.000 <l,k||d,c>_abab t2_1p_aaaa(d,a,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0181_aa_voQ")(aa,ia,Q) * tmps.at("0026_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0026_bb_voQ"))
    
    // r1_1p[aa] += +0.500 <j,k||i,b>_aaaa t2_1p_aaaa(b,a,j,k) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 0.500 * chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0181_aa_voQ")(aa,ja,Q) )
    
    // r2_1p[abab] += -1.000 <k,b||d,c>_abab t1_bb(c,j) t2_1p_aaaa(d,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0181_aa_voQ")(aa,ia,Q) * tmps.at("0106_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0106_bb_voQ"))
    
    // r2_2p[abab] += -2.000 <k,b||d,c>_abab t1_1p_bb(c,j) t2_1p_aaaa(d,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0181_aa_voQ")(aa,ia,Q) * tmps.at("0079_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0079_bb_voQ"))
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_1p_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0181_aa_voQ")(aa,ia,Q) * tmps.at("0097_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0097_bb_voQ"))
    
    // r1_1p[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0181_aa_voQ")(aa,ia,Q) * tmps.at("0030_Q")(Q) )
    .deallocate(tmps.at("0030_Q"))
    
    // r1_1p[aa] += -1.000 <k,j||c,b>_abab t1_bb(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0181_aa_voQ")(aa,ia,Q) * tmps.at("0029_Q")(Q) )
    .deallocate(tmps.at("0029_Q"))
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t2_1p_aaaa(d,a,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0181_aa_voQ")(aa,ia,Q) * tmps.at("0027_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0027_bb_voQ"))
    .allocate(tmps.at("0182_aabb_vvoo"))
    
    // flops: o2v2  = o3v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0182_aabb_vvoo")(da,aa,lb,jb)  = tmps.at("0053_aabb_ovov")(ka,da,lb,cb) * t2_1p.at("abab")(aa,cb,ka,jb) )
    .deallocate(tmps.at("0053_aabb_ovov"))
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_abab t2_1p_abab(a,d,l,j) t2_abab(c,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(ca,bb,ia,kb) * tmps.at("0182_aabb_vvoo")(ca,aa,kb,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_abab t1_bb(b,k) t1_1p_aa(c,i) t2_1p_abab(a,d,l,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t1_1p.at("aa")(ca,ia) * tmps.at("0182_aabb_vvoo")(ca,aa,kb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_abab t2_1p_abab(a,c,k,j) t2_1p_abab(d,b,i,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0182_aabb_vvoo")(da,aa,lb,jb) * t2_1p.at("abab")(da,bb,ia,lb) )
    .allocate(tmps.at("0183_aabb_vooo"))
    
    // flops: o3v1  = o2v1Q1 o3v1Q1 o3v1Q1 o3v1 o3v1Q1 o3v1 o3v1Q1 o3v1 o3v2 o3v1 o3v2 o3v1
    //  mems: o3v1  = o2v0Q1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("bin1_bb_ooQ")(jb,kb,Q)  = t1_1p.at("bb")(cb,jb) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("0183_aabb_vooo")(aa,ia,kb,jb)  = -1.000 * chol.at("aa_voQ")(aa,ia,Q) * tmps.at("bin1_bb_ooQ")(jb,kb,Q) )
    ( tmps.at("0183_aabb_vooo")(aa,ia,kb,jb) += tmps.at("0181_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0183_aabb_vooo")(aa,ia,kb,jb) -= tmps.at("0102_aa_voQ")(aa,ia,Q) * tmps.at("0132_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0183_aabb_vooo")(aa,ia,kb,jb) -= tmps.at("0081_aa_voQ")(aa,ia,Q) * tmps.at("0099_bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0183_aabb_vooo")(aa,ia,kb,jb) -= t1_1p.at("aa")(ca,ia) * tmps.at("0117_aabb_vvoo")(aa,ca,kb,jb) )
    ( tmps.at("0183_aabb_vooo")(aa,ia,kb,jb) += t1.at("aa")(ca,ia) * tmps.at("0182_aabb_vvoo")(ca,aa,kb,jb) )
    .deallocate(tmps.at("0182_aabb_vvoo"))
    .deallocate(tmps.at("0181_aa_voQ"))
    .deallocate(tmps.at("0132_bb_ooQ"))
    .deallocate(tmps.at("0117_aabb_vvoo"))
    .deallocate(tmps.at("0102_aa_voQ"))
    .deallocate(tmps.at("0099_bb_ooQ"))
    .deallocate(tmps.at("0081_aa_voQ"))
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t2_1p_abab(a,d,l,j) 
    //               += -1.000 <a,k||c,j>_abab t1_bb(b,k) t1_1p_aa(c,i) 
    //               += -1.000 <a,k||i,c>_abab t1_bb(b,k) t1_1p_bb(c,j) 
    //               += +1.000 <l,k||c,j>_abab t1_bb(b,k) t2_1p_aaaa(c,a,i,l) 
    //               += -1.000 <a,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t1_1p_bb(d,j) 
    //               += -1.000 <a,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0183_aabb_vooo")(aa,ia,kb,jb) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_abab t1_1p_bb(b,k) t1_aa(c,i) t2_1p_abab(a,d,l,j) 
    //               += -2.000 <a,k||c,j>_abab t1_1p_bb(b,k) t1_1p_aa(c,i) 
    //               += -2.000 <a,k||i,c>_abab t1_1p_bb(b,k) t1_1p_bb(c,j) 
    //               += +2.000 <l,k||c,j>_abab t1_1p_bb(b,k) t2_1p_aaaa(c,a,i,l) 
    //               += -2.000 <a,k||c,d>_abab t1_1p_bb(b,k) t1_aa(c,i) t1_1p_bb(d,j) 
    //               += -2.000 <a,k||d,c>_abab t1_1p_bb(b,k) t1_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0183_aabb_vooo")(aa,ia,kb,jb) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0183_aabb_vooo"))
    .allocate(tmps.at("0184_aa_oo"))
    
    // flops: o2v0  = o3v1 o3v1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0184_aa_oo")(ia,ja)  = t1_1p.at("aa")(ba,ka) * tmps.at("0179_aaaa_ooov")(ka,ia,ja,ba) )
    ( tmps.at("0184_aa_oo")(ia,ja) += t1_2p.at("aa")(ba,ka) * tmps.at("0088_aaaa_ooov")(ka,ia,ja,ba) )
    .deallocate(tmps.at("0179_aaaa_ooov"))
    .deallocate(tmps.at("0088_aaaa_ooov"))
    
    // r1_2p[aa] += +2.000 <k,j||i,b>_aaaa t1_aa(a,j) t1_2p_aa(b,k) 
    //             += -2.000 <k,j||b,c>_aaaa t1_aa(a,j) t1_1p_aa(b,k) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0184_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_aaaa t1_2p_aa(c,k) t2_abab(a,b,l,j) 
    //               += -2.000 <k,l||c,d>_aaaa t1_1p_aa(c,k) t1_1p_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0184_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0184_aa_oo"))
    .allocate(tmps.at("0185_aaaa_ovoo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0185_aaaa_ovoo")(la,da,ia,ka)  = t1_1p.at("aa")(ca,ia) * tmps.at("0058_aaaa_ovov")(la,da,ka,ca) )
    .deallocate(tmps.at("0058_aaaa_ovov"))
    
    // r1_1p[aa] += -0.500 <j,k||c,b>_aaaa t1_1p_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.500 * tmps.at("0185_aaaa_ovoo")(ja,ca,ia,ka) * t2.at("aaaa")(ca,aa,ja,ka) )
    
    // r1_2p[aa] += +1.000 <j,k||b,c>_aaaa t1_1p_aa(b,i) t2_1p_aaaa(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= tmps.at("0185_aaaa_ovoo")(ja,ca,ia,ka) * t2_1p.at("aaaa")(ca,aa,ja,ka) )
    
    // r2_1p[abab] += -1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0185_aaaa_ovoo")(la,da,ia,ka) * t2.at("abab")(da,bb,la,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_1p[abab] += -1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,la,jb) * tmps.at("0185_aaaa_ovoo")(ka,da,ia,la) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0185_aaaa_ovoo")(la,da,ia,ka) * t2_1p.at("abab")(da,bb,la,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_1p.at("abab")(da,bb,la,jb) * tmps.at("0185_aaaa_ovoo")(ka,da,ia,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0187_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0187_aa_oo")(la,ia)  = tmps.at("0185_aaaa_ovoo")(la,ca,ia,ka) * t1.at("aa")(ca,ka) )
    .deallocate(tmps.at("0185_aaaa_ovoo"))
    .allocate(tmps.at("0186_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0186_aa_oo")(la,ia)  = tmps.at("0061_aaaa_ovoo")(la,da,ia,ka) * t1_1p.at("aa")(da,ka) )
    .deallocate(tmps.at("0061_aaaa_ovoo"))
    .allocate(tmps.at("0188_aa_oo"))
    
    // flops: o2v0  = o2v0
    //  mems: o2v0  = o2v0
    ( tmps.at("0188_aa_oo")(la,ia)  = tmps.at("0186_aa_oo")(la,ia) )
    ( tmps.at("0188_aa_oo")(la,ia) += tmps.at("0187_aa_oo")(la,ia) )
    .deallocate(tmps.at("0187_aa_oo"))
    .deallocate(tmps.at("0186_aa_oo"))
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,k) t2_abab(a,b,l,j) 
    //               += +1.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_1p_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0188_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,k) t2_1p_abab(a,b,l,j) 
    //               += +2.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_1p_aa(d,i) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0188_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0188_aa_oo"))
    ;
  }
  // clang-format on
}

template void exachem::cc::cd_qed_ccsd_cs::resid_part4<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);