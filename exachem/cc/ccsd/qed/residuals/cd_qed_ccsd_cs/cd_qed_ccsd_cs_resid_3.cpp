/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_qed_ccsd_cs_resid_3.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_cs::resid_part3(
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
        
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,b,i,k) t2_2p_bbbb(c,d,j,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = t2_2p.at("bbbb")(cb,db,jb,lb) * tmps.at("0117_bbbb_ovov")(lb,cb,kb,db) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,b,i,k) t2_2p_bbbb(c,d,j,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = t2_2p.at("bbbb")(cb,db,jb,lb) * tmps.at("0117_bbbb_ovov")(lb,db,kb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_1p_abab(a,c,i,j) t2_1p_bbbb(d,b,l,k) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = t2_1p.at("bbbb")(db,bb,lb,kb) * tmps.at("0117_bbbb_ovov")(kb,db,lb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_bb_vv")(bb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,j) t2_2p_bbbb(d,b,l,k) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = t2_2p.at("bbbb")(db,bb,lb,kb) * tmps.at("0117_bbbb_ovov")(kb,db,lb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_bbbb t2_bbbb(c,b,l,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,db)  = t2.at("bbbb")(cb,bb,lb,kb) * tmps.at("0117_bbbb_ovov")(lb,cb,kb,db) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_vv")(bb,db) * t2_2p.at("abab")(aa,db,ia,jb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_1p_abab(a,c,i,j) t2_1p_bbbb(d,b,l,k) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = t2_1p.at("bbbb")(db,bb,lb,kb) * tmps.at("0117_bbbb_ovov")(lb,db,kb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_vv")(bb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,j) t2_2p_bbbb(d,b,l,k) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = t2_2p.at("bbbb")(db,bb,lb,kb) * tmps.at("0117_bbbb_ovov")(lb,db,kb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,b,i,k) t1_bb(c,j) t1_2p_bb(d,l) 
    // flops: o2v2 += o2v2 o2v1 o3v2
    //  mems: o2v2 += o1v1 o2v0 o2v2
    ( tmps.at("bin_bb_vo")(cb,kb)  = t1_2p.at("bb")(db,lb) * tmps.at("0117_bbbb_ovov")(kb,db,lb,cb) )
    ( tmps.at("bin_bb_oo")(jb,kb)  = tmps.at("bin_bb_vo")(cb,kb) * t1.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(c,k) t2_1p_abab(a,b,i,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o2v2 o2v1 o3v2
    //  mems: o2v2 += o1v1 o2v0 o2v2
    ( tmps.at("bin_bb_vo")(db,lb)  = t1.at("bb")(cb,kb) * tmps.at("0117_bbbb_ovov")(lb,cb,kb,db) )
    ( tmps.at("bin_bb_oo")(jb,lb)  = tmps.at("bin_bb_vo")(db,lb) * t1_1p.at("bb")(db,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,b,i,l) t1_bb(c,k) t1_2p_bb(d,j) 
    // flops: o2v2 += o2v2 o2v1 o3v2
    //  mems: o2v2 += o1v1 o2v0 o2v2
    ( tmps.at("bin_bb_vo")(db,lb)  = t1.at("bb")(cb,kb) * tmps.at("0117_bbbb_ovov")(lb,cb,kb,db) )
    ( tmps.at("bin_bb_oo")(jb,lb)  = tmps.at("bin_bb_vo")(db,lb) * t1_2p.at("bb")(db,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(c,k) t2_1p_abab(a,d,i,j) t1_1p_bb(b,l) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_bb_vo")(db,lb)  = t1.at("bb")(cb,kb) * tmps.at("0117_bbbb_ovov")(lb,cb,kb,db) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("bin_bb_vo")(db,lb) * t2_1p.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,c,i,j) t1_bb(b,k) t1_2p_bb(d,l) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_bb_vo")(cb,kb)  = t1_2p.at("bb")(db,lb) * tmps.at("0117_bbbb_ovov")(kb,db,lb,cb) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bb_vo")(cb,kb) * t2.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_bb_vo")(db,lb)  = t1.at("bb")(cb,kb) * tmps.at("0117_bbbb_ovov")(lb,cb,kb,db) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("bin_bb_vo")(db,lb) * t2_2p.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1.at("bb")(bb,lb) )
    
    // r2.at("abab") += +0.50 <l,k||c,d>_bbbb t2_abab(a,c,i,j) t2_bbbb(d,b,l,k) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = t2.at("bbbb")(db,bb,lb,kb) * tmps.at("0117_bbbb_ovov")(lb,db,kb,cb) )
    ( r2.at("abab")(aa,bb,ia,jb) -= 0.50 * tmps.at("bin_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    .allocate(tmps.at("0118_bb_oo"))
    
    // flops: o2v0  = o3v2
    //  mems: o2v0  = o2v0
    ( tmps.at("0118_bb_oo")(jb,lb)  = t2.at("bbbb")(cb,db,jb,kb) * tmps.at("0117_bbbb_ovov")(kb,db,lb,cb) )
    
    // r2_1p.at("abab") += -0.50 <l,k||c,d>_bbbb t2_bbbb(c,d,j,k) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.50 * t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0118_bb_oo")(jb,lb) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_bbbb t2_bbbb(c,d,j,k) t2_2p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= t2_2p.at("abab")(aa,bb,ia,lb) * tmps.at("0118_bb_oo")(jb,lb) )
    
    // r2.at("abab") += -0.50 <l,k||c,d>_bbbb t2_abab(a,b,i,l) t2_bbbb(c,d,j,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= 0.50 * t2.at("abab")(aa,bb,ia,lb) * tmps.at("0118_bb_oo")(jb,lb) )
    .deallocate(tmps.at("0118_bb_oo"))
    .allocate(tmps.at("0119_bb_oo"))
    
    // flops: o2v0  = o3v2
    //  mems: o2v0  = o2v0
    ( tmps.at("0119_bb_oo")(jb,kb)  = t2_1p.at("bbbb")(cb,db,jb,lb) * tmps.at("0117_bbbb_ovov")(lb,cb,kb,db) )
    
    // r2_1p.at("abab") += +0.50 <l,k||c,d>_bbbb t2_abab(a,b,i,k) t2_1p_bbbb(c,d,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.50 * t2.at("abab")(aa,bb,ia,kb) * tmps.at("0119_bb_oo")(jb,kb) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_bbbb t2_1p_abab(a,b,i,l) t2_1p_bbbb(c,d,j,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0119_bb_oo")(jb,lb) )
    .deallocate(tmps.at("0119_bb_oo"))
    .allocate(tmps.at("0120_bb_oo"))
    
    // flops: o2v0  = o3v2
    //  mems: o2v0  = o2v0
    ( tmps.at("0120_bb_oo")(jb,kb)  = t2_1p.at("bbbb")(cb,db,jb,lb) * tmps.at("0117_bbbb_ovov")(lb,db,kb,cb) )
    
    // r2_1p.at("abab") += +0.50 <l,k||c,d>_bbbb t2_abab(a,b,i,k) t2_1p_bbbb(c,d,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.50 * t2.at("abab")(aa,bb,ia,kb) * tmps.at("0120_bb_oo")(jb,kb) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_bbbb t2_1p_abab(a,b,i,l) t2_1p_bbbb(c,d,j,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0120_bb_oo")(jb,lb) )
    .deallocate(tmps.at("0120_bb_oo"))
    .allocate(tmps.at("0121_bbaa_vvoo"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0121_bbaa_vvoo")(bb,cb,ka,ia)  = chol.at("bb_vvQ")(bb,cb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    
    // r2_1p.at("abab") += -1.00 <k,b||i,c>_abab t2_1p_abab(a,c,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,cb,ka,jb) * tmps.at("0121_bbaa_vvoo")(bb,cb,ka,ia) )
    
    // r2_2p.at("abab") += -2.00 <k,b||i,c>_abab t2_2p_abab(a,c,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_2p.at("abab")(aa,cb,ka,jb) * tmps.at("0121_bbaa_vvoo")(bb,cb,ka,ia) )
    
    // r2_2p.at("abab") += -2.00 <k,b||i,c>_abab t1_aa(a,k) t1_2p_bb(c,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0121_bbaa_vvoo")(bb,cb,ka,ia) * t1_2p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2.at("abab") += -1.00 <k,b||i,c>_abab t2_abab(a,c,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,cb,ka,jb) * tmps.at("0121_bbaa_vvoo")(bb,cb,ka,ia) )
    .allocate(tmps.at("0122_baab_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0122_baab_vooo")(bb,ka,ia,jb)  = tmps.at("0121_bbaa_vvoo")(bb,cb,ka,ia) * t1_1p.at("bb")(cb,jb) )
    .deallocate(tmps.at("0121_bbaa_vvoo"))
    
    // r2_1p.at("abab") += -1.00 <k,b||i,c>_abab t1_aa(a,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("0122_baab_vooo")(bb,ka,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 <k,b||i,c>_abab t1_1p_aa(a,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0122_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0122_baab_vooo"))
    .allocate(tmps.at("0123_abab_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0123_abab_vooo")(aa,kb,ia,jb)  = t2_1p.at("abab")(aa,cb,ia,jb) * dp.at("bb_ov")(kb,cb) )
    
    // r2_2p.at("abab") += -6.00 d-_bb(k,c) t2_1p_abab(a,c,i,j) t1_2p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.00 * t1_2p.at("bb")(bb,kb) * tmps.at("0123_abab_vooo")(aa,kb,ia,jb) )
    .allocate(tmps.at("0124_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0124_abab_vvoo")(aa,bb,ia,jb)  = tmps.at("0123_abab_vooo")(aa,kb,ia,jb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += -1.00 d-_bb(k,c) t1_bb(b,k) t0_1p t2_1p_abab(a,c,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0124_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 d+_bb(k,c) t1_bb(b,k) t2_1p_abab(a,c,i,j) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0124_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -4.00 d-_bb(k,c) t1_bb(b,k) t2_1p_abab(a,c,i,j) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * t0_2p * tmps.at("0124_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2.at("abab") += -1.00 d-_bb(k,c) t1_bb(b,k) t2_1p_abab(a,c,i,j) 
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0124_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0124_abab_vvoo"))
    .allocate(tmps.at("0125_abab_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0125_abab_vooo")(aa,kb,ia,jb)  = t2.at("abab")(aa,cb,ia,jb) * dp.at("bb_ov")(kb,cb) )
    .allocate(tmps.at("0126_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0126_abab_vvoo")(aa,bb,ia,jb)  = tmps.at("0125_abab_vooo")(aa,kb,ia,jb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += -1.00 d-_bb(k,c) t2_abab(a,c,i,j) t0_1p t1_1p_bb(b,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0126_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 d+_bb(k,c) t2_abab(a,c,i,j) t1_1p_bb(b,k) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0126_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -4.00 d-_bb(k,c) t2_abab(a,c,i,j) t1_1p_bb(b,k) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * t0_2p * tmps.at("0126_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2.at("abab") += -1.00 d-_bb(k,c) t2_abab(a,c,i,j) t1_1p_bb(b,k) 
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0126_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0126_abab_vvoo"))
    .allocate(tmps.at("0127_baab_ovoo"))
    
    // flops: o3v1  = o2v2 o3v2
    //  mems: o3v1  = o1v1 o3v1
    ( tmps.at("bin_bb_vo")(db,lb)  = t1.at("bb")(cb,kb) * tmps.at("0117_bbbb_ovov")(lb,cb,kb,db) )
    ( tmps.at("0127_baab_ovoo")(lb,aa,ia,jb)  = tmps.at("bin_bb_vo")(db,lb) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,d,i,j) t1_bb(c,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0127_baab_ovoo")(lb,aa,ia,jb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,d,i,j) t1_bb(c,k) t1_2p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0127_baab_ovoo")(lb,aa,ia,jb) * t1_2p.at("bb")(bb,lb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,d,i,j) t1_bb(b,l) t1_bb(c,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0127_baab_ovoo")(lb,aa,ia,jb) * t1.at("bb")(bb,lb) )
    .deallocate(tmps.at("0127_baab_ovoo"))
    .allocate(tmps.at("0128_baba_vvoo"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin_bbbb_vvoo")(cb,db,kb,lb)  = chol.at("bb_ovQ")(kb,db,Q) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("0128_baba_vvoo")(db,aa,lb,ia)  = tmps.at("bin_bbbb_vvoo")(cb,db,kb,lb) * t2.at("abab")(aa,cb,ia,kb) )
    
    // r1_1p.at("aa") += -1.00 <k,j||b,c>_bbbb t2_abab(a,b,i,j) t1_1p_bb(c,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1_1p.at("bb")(cb,kb) * tmps.at("0128_baba_vvoo")(cb,aa,kb,ia) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_bbbb t2_abab(a,b,i,j) t1_2p_bb(c,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1_2p.at("bb")(cb,kb) * tmps.at("0128_baba_vvoo")(cb,aa,kb,ia) )
    
    // r1.at("aa") += -1.00 <k,j||b,c>_bbbb t2_abab(a,c,i,k) t1_bb(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= t1.at("bb")(bb,jb) * tmps.at("0128_baba_vvoo")(bb,aa,jb,ia) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0128_baba_vvoo")(db,aa,lb,ia) * t2_1p.at("bbbb")(db,bb,jb,lb) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,l) t1_bb(b,k) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0128_baba_vvoo")(db,aa,kb,ia) * t1_1p.at("bb")(db,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_2p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0128_baba_vvoo")(db,aa,lb,ia) * t2_2p.at("bbbb")(db,bb,jb,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t1_1p_bb(b,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("0128_baba_vvoo")(db,aa,lb,ia) * t1_1p.at("bb")(db,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t2_abab(a,c,i,l) t1_bb(b,k) t1_2p_bb(d,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0128_baba_vvoo")(db,aa,kb,ia) * t1_2p.at("bb")(db,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0128_baba_vvoo")(db,aa,lb,ia) * t2.at("bbbb")(db,bb,jb,lb) )
    .allocate(tmps.at("0129_abba_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0129_abba_vooo")(aa,jb,lb,ia)  = t1.at("bb")(cb,jb) * tmps.at("0128_baba_vvoo")(cb,aa,lb,ia) )
    .deallocate(tmps.at("0128_baba_vvoo"))
    
    // r2_1p.at("abab") += -1.00 <l,k||c,d>_bbbb t2_abab(a,d,i,k) t1_bb(c,j) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("bb")(bb,lb) * tmps.at("0129_abba_vooo")(aa,jb,lb,ia) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t2_abab(a,d,i,k) t1_bb(c,j) t1_2p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("bb")(bb,lb) * tmps.at("0129_abba_vooo")(aa,jb,lb,ia) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,d,i,l) t1_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0129_abba_vooo")(aa,jb,kb,ia) )
    .deallocate(tmps.at("0129_abba_vooo"))
    .allocate(tmps.at("0130_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0130_abab_vvoo")(aa,bb,ia,jb)  = tmps.at("0125_abab_vooo")(aa,kb,ia,jb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += -1.00 d+_bb(k,c) t2_abab(a,c,i,j) t1_bb(b,k) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0130_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_1p.at("abab") += -2.00 d-_bb(k,c) t2_abab(a,c,i,j) t1_bb(b,k) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * t0_2p * tmps.at("0130_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2.at("abab") += -1.00 d-_bb(k,c) t2_abab(a,c,i,j) t1_bb(b,k) t0_1p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0130_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0130_abab_vvoo"))
    .allocate(tmps.at("0131_baab_ovoo"))
    
    // flops: o3v1  = o2v2 o3v2
    //  mems: o3v1  = o1v1 o3v1
    ( tmps.at("bin_bb_vo")(cb,kb)  = t1_1p.at("bb")(db,lb) * tmps.at("0117_bbbb_ovov")(kb,db,lb,cb) )
    ( tmps.at("0131_baab_ovoo")(kb,aa,ia,jb)  = tmps.at("bin_bb_vo")(cb,kb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,j) t1_bb(b,k) t1_1p_bb(d,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0131_baab_ovoo")(kb,aa,ia,jb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t2_abab(a,c,i,j) t1_1p_bb(b,l) t1_1p_bb(d,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0131_baab_ovoo")(lb,aa,ia,jb) * t1_1p.at("bb")(bb,lb) )
    .deallocate(tmps.at("0131_baab_ovoo"))
    .allocate(tmps.at("0132_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0132_abab_vvoo")(aa,bb,ia,jb)  = tmps.at("0125_abab_vooo")(aa,kb,ia,jb) * t1_2p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0125_abab_vooo"))
    
    // r2_1p.at("abab") += -2.00 d-_bb(k,c) t2_abab(a,c,i,j) t1_2p_bb(b,k) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0132_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 d-_bb(k,c) t2_abab(a,c,i,j) t0_1p t1_2p_bb(b,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t0_1p * tmps.at("0132_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0132_abab_vvoo"))
    .allocate(tmps.at("0133_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0133_abab_vvoo")(aa,bb,ia,jb)  = tmps.at("0123_abab_vooo")(aa,kb,ia,jb) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0123_abab_vooo"))
    
    // r2_1p.at("abab") += -2.00 d-_bb(k,c) t2_1p_abab(a,c,i,j) t1_1p_bb(b,k) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0133_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 d-_bb(k,c) t0_1p t2_1p_abab(a,c,i,j) t1_1p_bb(b,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t0_1p * tmps.at("0133_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0133_abab_vvoo"))
    .allocate(tmps.at("0134_abab_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2
    //  mems: o2v2  = o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    ( tmps.at("0134_abab_vvoo")(aa,bb,ia,jb)  = tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += -2.00 d-_bb(k,c) t1_bb(b,k) t2_2p_abab(a,c,i,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0134_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 d-_bb(k,c) t1_bb(b,k) t0_1p t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t0_1p * tmps.at("0134_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0134_abab_vvoo"))
    .allocate(tmps.at("0135_baab_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0135_baab_vooo")(bb,ka,ia,jb)  = t2.at("abab")(ca,bb,ia,jb) * dp.at("aa_ov")(ka,ca) )
    .allocate(tmps.at("0136_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0136_abab_vvoo")(aa,bb,ia,jb)  = t1_1p.at("aa")(aa,ka) * tmps.at("0135_baab_vooo")(bb,ka,ia,jb) )
    
    // r2_1p.at("abab") += -1.00 d-_aa(k,c) t2_abab(c,b,i,j) t0_1p t1_1p_aa(a,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0136_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 d+_aa(k,c) t2_abab(c,b,i,j) t1_1p_aa(a,k) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0136_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -4.00 d-_aa(k,c) t2_abab(c,b,i,j) t1_1p_aa(a,k) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * t0_2p * tmps.at("0136_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2.at("abab") += -1.00 d-_aa(k,c) t2_abab(c,b,i,j) t1_1p_aa(a,k) 
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0136_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0136_abab_vvoo"))
    .allocate(tmps.at("0137_baab_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0137_baab_vooo")(bb,ka,ia,jb)  = t2_1p.at("abab")(ca,bb,ia,jb) * dp.at("aa_ov")(ka,ca) )
    
    // r2_2p.at("abab") += -6.00 d-_aa(k,c) t2_1p_abab(c,b,i,j) t1_2p_aa(a,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.00 * t1_2p.at("aa")(aa,ka) * tmps.at("0137_baab_vooo")(bb,ka,ia,jb) )
    .allocate(tmps.at("0138_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0138_abab_vvoo")(aa,bb,ia,jb)  = t1.at("aa")(aa,ka) * tmps.at("0137_baab_vooo")(bb,ka,ia,jb) )
    
    // r2_1p.at("abab") += -1.00 d-_aa(k,c) t1_aa(a,k) t0_1p t2_1p_abab(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0138_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 d+_aa(k,c) t1_aa(a,k) t2_1p_abab(c,b,i,j) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0138_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -4.00 d-_aa(k,c) t1_aa(a,k) t2_1p_abab(c,b,i,j) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * t0_2p * tmps.at("0138_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2.at("abab") += -1.00 d-_aa(k,c) t1_aa(a,k) t2_1p_abab(c,b,i,j) 
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0138_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0138_abab_vvoo"))
    .allocate(tmps.at("0139_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0139_abab_vvoo")(aa,bb,ia,jb)  = t1.at("aa")(aa,ka) * tmps.at("0135_baab_vooo")(bb,ka,ia,jb) )
    
    // r2_1p.at("abab") += -1.00 d+_aa(k,c) t1_aa(a,k) t2_abab(c,b,i,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0139_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_1p.at("abab") += -2.00 d-_aa(k,c) t1_aa(a,k) t2_abab(c,b,i,j) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * t0_2p * tmps.at("0139_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2.at("abab") += -1.00 d-_aa(k,c) t1_aa(a,k) t2_abab(c,b,i,j) t0_1p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0139_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0139_abab_vvoo"))
    .allocate(tmps.at("0140_baab_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0140_baab_vooo")(bb,ka,ia,jb)  = t2_2p.at("abab")(ca,bb,ia,jb) * dp.at("aa_ov")(ka,ca) )
    
    // r2_2p.at("abab") += -6.00 d-_aa(k,c) t1_1p_aa(a,k) t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0140_baab_vooo")(bb,ka,ia,jb) )
    .allocate(tmps.at("0141_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0141_abab_vvoo")(aa,bb,ia,jb)  = t1.at("aa")(aa,ka) * tmps.at("0140_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0140_baab_vooo"))
    
    // r2_1p.at("abab") += -2.00 d-_aa(k,c) t1_aa(a,k) t2_2p_abab(c,b,i,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0141_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 d-_aa(k,c) t1_aa(a,k) t0_1p t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t0_1p * tmps.at("0141_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0141_abab_vvoo"))
    .allocate(tmps.at("0142_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0142_abab_vvoo")(aa,bb,ia,jb)  = t1_2p.at("aa")(aa,ka) * tmps.at("0135_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0135_baab_vooo"))
    
    // r2_1p.at("abab") += -2.00 d-_aa(k,c) t2_abab(c,b,i,j) t1_2p_aa(a,k) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0142_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 d-_aa(k,c) t2_abab(c,b,i,j) t0_1p t1_2p_aa(a,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t0_1p * tmps.at("0142_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0142_abab_vvoo"))
    .allocate(tmps.at("0143_abab_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0143_abab_vvoo")(aa,bb,ia,jb)  = t1_1p.at("aa")(aa,ka) * tmps.at("0137_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0137_baab_vooo"))
    
    // r2_1p.at("abab") += -2.00 d-_aa(k,c) t1_1p_aa(a,k) t2_1p_abab(c,b,i,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0143_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 d-_aa(k,c) t0_1p t1_1p_aa(a,k) t2_1p_abab(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t0_1p * tmps.at("0143_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0143_abab_vvoo"))
    .allocate(tmps.at("0144_aabb_vvoo"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin_abab_vvoo")(da,cb,ka,lb)  = chol.at("aa_ovQ")(ka,da,Q) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("0144_aabb_vvoo")(da,aa,lb,jb)  = tmps.at("bin_abab_vvoo")(da,cb,ka,lb) * t2.at("abab")(aa,cb,ka,jb) )
    
    // r2_1p.at("abab") += +1.00 <k,l||d,c>_abab t2_abab(a,c,k,j) t2_1p_abab(d,b,i,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0144_aabb_vvoo")(da,aa,lb,jb) * t2_1p.at("abab")(da,bb,ia,lb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t2_abab(a,c,k,j) t2_2p_abab(d,b,i,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0144_aabb_vvoo")(da,aa,lb,jb) * t2_2p.at("abab")(da,bb,ia,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||d,c>_abab t2_abab(a,c,l,j) t1_bb(b,k) t1_2p_aa(d,i) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0144_aabb_vvoo")(da,aa,kb,jb) * t1_2p.at("aa")(da,ia) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2.at("abab") += +1.00 <k,l||d,c>_abab t2_abab(a,c,k,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0144_aabb_vvoo")(da,aa,lb,jb) * t2.at("abab")(da,bb,ia,lb) )
    .allocate(tmps.at("0145_aabb_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0145_aabb_vooo")(aa,ia,lb,jb)  = t1.at("aa")(ca,ia) * tmps.at("0144_aabb_vvoo")(ca,aa,lb,jb) )
    
    // r2_1p.at("abab") += +1.00 <k,l||c,d>_abab t2_abab(a,d,k,j) t1_aa(c,i) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("bb")(bb,lb) * tmps.at("0145_aabb_vooo")(aa,ia,lb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,d>_abab t2_abab(a,d,k,j) t1_aa(c,i) t1_2p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("bb")(bb,lb) * tmps.at("0145_aabb_vooo")(aa,ia,lb,jb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_abab t2_abab(a,d,l,j) t1_bb(b,k) t1_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0145_aabb_vooo")(aa,ia,kb,jb) )
    .deallocate(tmps.at("0145_aabb_vooo"))
    .allocate(tmps.at("0146_aabb_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0146_aabb_vooo")(aa,ia,kb,jb)  = t1_1p.at("aa")(da,ia) * tmps.at("0144_aabb_vvoo")(da,aa,kb,jb) )
    .deallocate(tmps.at("0144_aabb_vvoo"))
    
    // r2_1p.at("abab") += +1.00 <l,k||d,c>_abab t2_abab(a,c,l,j) t1_bb(b,k) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0146_aabb_vooo")(aa,ia,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t2_abab(a,c,k,j) t1_1p_bb(b,l) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("bb")(bb,lb) * tmps.at("0146_aabb_vooo")(aa,ia,lb,jb) )
    .deallocate(tmps.at("0146_aabb_vooo"))
    .allocate(tmps.at("0147_aabb_vvoo"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin_abab_vvoo")(ca,db,la,kb)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0147_aabb_vvoo")(ca,aa,kb,jb)  = tmps.at("bin_abab_vvoo")(ca,db,la,kb) * t2_1p.at("abab")(aa,db,la,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_abab t2_abab(c,b,i,k) t2_1p_abab(a,d,l,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0147_aabb_vvoo")(ca,aa,kb,jb) * t2.at("abab")(ca,bb,ia,kb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t2_1p_abab(a,c,k,j) t2_1p_abab(d,b,i,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0147_aabb_vvoo")(da,aa,lb,jb) * t2_1p.at("abab")(da,bb,ia,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_abab t1_bb(b,k) t2_1p_abab(a,d,l,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0147_aabb_vvoo")(ca,aa,kb,jb) * t1_1p.at("aa")(ca,ia) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0148_aabb_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0148_aabb_vooo")(aa,ia,kb,jb)  = t1.at("aa")(ca,ia) * tmps.at("0147_aabb_vvoo")(ca,aa,kb,jb) )
    .deallocate(tmps.at("0147_aabb_vvoo"))
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t2_1p_abab(a,d,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0148_aabb_vooo")(aa,ia,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_abab t1_aa(c,i) t2_1p_abab(a,d,l,j) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("bb")(bb,kb) * tmps.at("0148_aabb_vooo")(aa,ia,kb,jb) )
    .deallocate(tmps.at("0148_aabb_vooo"))
    .allocate(tmps.at("0149_aa_oo"))
    
    // flops: o2v0  = o3v2
    //  mems: o2v0  = o2v0
    ( tmps.at("0149_aa_oo")(ia,ka)  = t2.at("aaaa")(ba,ca,ia,ja) * tmps.at("0094_aaaa_ovov")(ja,ca,ka,ba) )
    
    // r1_1p.at("aa") += -0.50 <k,j||b,c>_aaaa t2_aaaa(b,c,i,j) t1_1p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.50 * t1_1p.at("aa")(aa,ka) * tmps.at("0149_aa_oo")(ia,ka) )
    
    // r1_2p.at("aa") += -1.00 <k,j||b,c>_aaaa t2_aaaa(b,c,i,j) t1_2p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= t1_2p.at("aa")(aa,ka) * tmps.at("0149_aa_oo")(ia,ka) )
    
    // r1.at("aa") += +0.50 <k,j||b,c>_aaaa t1_aa(a,j) t2_aaaa(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= 0.50 * t1.at("aa")(aa,ja) * tmps.at("0149_aa_oo")(ia,ja) )
    
    // r2_1p.at("abab") += -0.50 <l,k||c,d>_aaaa t2_aaaa(c,d,i,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.50 * t2_1p.at("abab")(aa,bb,la,jb) * tmps.at("0149_aa_oo")(ia,la) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_aaaa t2_aaaa(c,d,i,k) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= t2_2p.at("abab")(aa,bb,la,jb) * tmps.at("0149_aa_oo")(ia,la) )
    
    // r2.at("abab") += -0.50 <l,k||c,d>_aaaa t2_abab(a,b,l,j) t2_aaaa(c,d,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= 0.50 * t2.at("abab")(aa,bb,la,jb) * tmps.at("0149_aa_oo")(ia,la) )
    .deallocate(tmps.at("0149_aa_oo"))
    .allocate(tmps.at("0150_aaaa_ooov"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0150_aaaa_ooov")(ja,ia,ka,ca)  = t1.at("aa")(ba,ia) * tmps.at("0094_aaaa_ovov")(ja,ba,ka,ca) )
    
    // r1_1p.at("aa") += +0.50 <k,j||b,c>_aaaa t1_aa(b,i) t2_1p_aaaa(c,a,k,j) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.50 * t2_1p.at("aaaa")(ca,aa,ka,ja) * tmps.at("0150_aaaa_ooov")(ja,ia,ka,ca) )
    
    // r1_2p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(b,i) t2_2p_aaaa(c,a,k,j) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += t2_2p.at("aaaa")(ca,aa,ka,ja) * tmps.at("0150_aaaa_ooov")(ka,ia,ja,ca) )
    
    // r1_2p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(b,i) t2_2p_aaaa(c,a,k,j) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= t2_2p.at("aaaa")(ca,aa,ka,ja) * tmps.at("0150_aaaa_ooov")(ja,ia,ka,ca) )
    
    // r1.at("aa") += +0.50 <k,j||b,c>_aaaa t2_aaaa(c,a,k,j) t1_aa(b,i) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= 0.50 * t2.at("aaaa")(ca,aa,ka,ja) * tmps.at("0150_aaaa_ooov")(ja,ia,ka,ca) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(da,bb,la,jb) * tmps.at("0150_aaaa_ooov")(la,ia,ka,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0150_aaaa_ooov"))
    .allocate(tmps.at("0151_aa_oo"))
    
    // flops: o2v0  = o3v2
    //  mems: o2v0  = o2v0
    ( tmps.at("0151_aa_oo")(ia,ja)  = t2_1p.at("aaaa")(ba,ca,ia,ka) * tmps.at("0094_aaaa_ovov")(ka,ca,ja,ba) )
    
    // r1_1p.at("aa") += +0.50 <k,j||b,c>_aaaa t1_aa(a,j) t2_1p_aaaa(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.50 * t1.at("aa")(aa,ja) * tmps.at("0151_aa_oo")(ia,ja) )
    
    // r1_2p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_1p_aa(a,j) t2_1p_aaaa(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= t1_1p.at("aa")(aa,ja) * tmps.at("0151_aa_oo")(ia,ja) )
    
    // r2_1p.at("abab") += +0.50 <l,k||c,d>_aaaa t2_abab(a,b,k,j) t2_1p_aaaa(c,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.50 * t2.at("abab")(aa,bb,ka,jb) * tmps.at("0151_aa_oo")(ia,ka) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_aaaa t2_1p_abab(a,b,l,j) t2_1p_aaaa(c,d,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,bb,la,jb) * tmps.at("0151_aa_oo")(ia,la) )
    .deallocate(tmps.at("0151_aa_oo"))
    .allocate(tmps.at("0152_aaaa_ooov"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0152_aaaa_ooov")(ja,ia,ka,ba)  = t1_1p.at("aa")(ca,ia) * tmps.at("0094_aaaa_ovov")(ja,ca,ka,ba) )
    
    // r1_1p.at("aa") += -0.50 <k,j||b,c>_aaaa t2_aaaa(b,a,k,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.50 * t2.at("aaaa")(ba,aa,ka,ja) * tmps.at("0152_aaaa_ooov")(ja,ia,ka,ba) )
    
    // r1_2p.at("aa") += +1.00 <k,j||b,c>_aaaa t2_1p_aaaa(c,a,k,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= t2_1p.at("aaaa")(ca,aa,ka,ja) * tmps.at("0152_aaaa_ooov")(ja,ia,ka,ca) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(a,k) t2_1p_abab(d,b,l,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t2_1p.at("abab")(da,bb,la,jb) * tmps.at("0152_aaaa_ooov")(la,ia,ka,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0152_aaaa_ooov"))
    .allocate(tmps.at("0153_aa_oo"))
    
    // flops: o2v0  = o3v2
    //  mems: o2v0  = o2v0
    ( tmps.at("0153_aa_oo")(ia,ja)  = t2_2p.at("aaaa")(ba,ca,ia,ka) * tmps.at("0094_aaaa_ovov")(ka,ba,ja,ca) )
    
    // r1_2p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(a,j) t2_2p_aaaa(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += t1.at("aa")(aa,ja) * tmps.at("0153_aa_oo")(ia,ja) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_aaaa t2_abab(a,b,k,j) t2_2p_aaaa(c,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ka,jb) * tmps.at("0153_aa_oo")(ia,ka) )
    .deallocate(tmps.at("0153_aa_oo"))
    .allocate(tmps.at("0154_aabb_ovoo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0154_aabb_ovoo")(ka,ca,lb,jb)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    
    // r2_1p.at("abab") += +1.00 <k,l||c,j>_abab t1_aa(a,k) t1_bb(b,l) t1_1p_aa(c,i) 
    // flops: o2v2 += o4v1 o4v1 o3v2
    //  mems: o2v2 += o4v0 o3v1 o2v2
    ( tmps.at("bin_aabb_oooo")(ia,ka,jb,lb)  = tmps.at("0154_aabb_ovoo")(ka,ca,lb,jb) * t1_1p.at("aa")(ca,ia) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aabb_oooo")(ia,ka,jb,lb) * t1.at("bb")(bb,lb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,j>_abab t1_aa(a,k) t2_2p_abab(c,b,i,l) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(ca,bb,ia,lb) * tmps.at("0154_aabb_ovoo")(ka,ca,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,j>_abab t1_bb(b,k) t1_1p_aa(a,l) t1_1p_aa(c,i) 
    // flops: o2v2 += o4v1 o4v1 o3v2
    //  mems: o2v2 += o4v0 o3v1 o2v2
    ( tmps.at("bin_aabb_oooo")(ia,la,jb,kb)  = tmps.at("0154_aabb_ovoo")(la,ca,kb,jb) * t1_1p.at("aa")(ca,ia) )
    ( tmps.at("bin_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin_aabb_oooo")(ia,la,jb,kb) * t1.at("bb")(bb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,la,jb) * t1_1p.at("aa")(aa,la) )
    .allocate(tmps.at("0155_abba_ovoo"))
    
    // flops: o3v1  = o4v1 o4v1
    //  mems: o3v1  = o4v0 o3v1
    ( tmps.at("bin_aabb_oooo")(ia,la,jb,kb)  = tmps.at("0154_aabb_ovoo")(la,ca,kb,jb) * t1.at("aa")(ca,ia) )
    ( tmps.at("0155_abba_ovoo")(la,bb,jb,ia)  = tmps.at("bin_aabb_oooo")(ia,la,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,j>_abab t1_bb(b,k) t1_aa(c,i) t1_1p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,la) * tmps.at("0155_abba_ovoo")(la,bb,jb,ia) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,j>_abab t1_bb(b,k) t1_aa(c,i) t1_2p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("aa")(aa,la) * tmps.at("0155_abba_ovoo")(la,bb,jb,ia) )
    
    // r2.at("abab") += +1.00 <k,l||c,j>_abab t1_aa(a,k) t1_bb(b,l) t1_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0155_abba_ovoo")(ka,bb,jb,ia) )
    .deallocate(tmps.at("0155_abba_ovoo"))
    .allocate(tmps.at("0156_abab_oooo"))
    
    // flops: o4v0  = o2v2Q1 o4v2
    //  mems: o4v0  = o2v2 o4v0
    ( tmps.at("bin_abab_vvoo")(ca,db,la,kb)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0156_abab_oooo")(la,kb,ia,jb)  = tmps.at("bin_abab_vvoo")(ca,db,la,kb) * t2.at("abab")(ca,db,ia,jb) )
    
    // r2_1p.at("abab") += +0.250 <l,k||c,d>_abab t2_abab(c,d,i,j) t2_1p_abab(a,b,l,k) 
    //            += +0.250 <k,l||c,d>_abab t2_abab(c,d,i,j) t2_1p_abab(a,b,k,l) 
    //            += +0.250 <l,k||d,c>_abab t2_abab(d,c,i,j) t2_1p_abab(a,b,l,k) 
    //            += +0.250 <k,l||d,c>_abab t2_abab(d,c,i,j) t2_1p_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,la,kb) * tmps.at("0156_abab_oooo")(la,kb,ia,jb) )
    
    // r2_2p.at("abab") += +0.50 <l,k||c,d>_abab t2_abab(c,d,i,j) t2_2p_abab(a,b,l,k) 
    //            += +0.50 <k,l||c,d>_abab t2_abab(c,d,i,j) t2_2p_abab(a,b,k,l) 
    //            += +0.50 <l,k||d,c>_abab t2_abab(d,c,i,j) t2_2p_abab(a,b,l,k) 
    //            += +0.50 <k,l||d,c>_abab t2_abab(d,c,i,j) t2_2p_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_2p.at("abab")(aa,bb,la,kb) * tmps.at("0156_abab_oooo")(la,kb,ia,jb) )
    
    // r2_2p.at("abab") += +1.00 <k,l||c,d>_abab t1_aa(a,k) t2_abab(c,d,i,j) t1_2p_bb(b,l) 
    //            += +1.00 <k,l||d,c>_abab t1_aa(a,k) t2_abab(d,c,i,j) t1_2p_bb(b,l) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_2p.at("bb")(bb,lb) * tmps.at("0156_abab_oooo")(ka,lb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2.at("abab") += +0.250 <l,k||c,d>_abab t2_abab(a,b,l,k) t2_abab(c,d,i,j) 
    //         += +0.250 <l,k||d,c>_abab t2_abab(a,b,l,k) t2_abab(d,c,i,j) 
    //         += +0.250 <k,l||c,d>_abab t2_abab(a,b,k,l) t2_abab(c,d,i,j) 
    //         += +0.250 <k,l||d,c>_abab t2_abab(a,b,k,l) t2_abab(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,kb) * tmps.at("0156_abab_oooo")(la,kb,ia,jb) )
    .allocate(tmps.at("0157_baab_vooo"))
    
    // flops: o3v1  = o4v1
    //  mems: o3v1  = o3v1
    ( tmps.at("0157_baab_vooo")(bb,la,ia,jb)  = t1.at("bb")(bb,kb) * tmps.at("0156_abab_oooo")(la,kb,ia,jb) )
    
    // r2_1p.at("abab") += +0.50 <l,k||c,d>_abab t1_bb(b,k) t2_abab(c,d,i,j) t1_1p_aa(a,l) 
    //            += +0.50 <l,k||d,c>_abab t1_bb(b,k) t2_abab(d,c,i,j) t1_1p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,la) * tmps.at("0157_baab_vooo")(bb,la,ia,jb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_abab t1_bb(b,k) t2_abab(c,d,i,j) t1_2p_aa(a,l) 
    //            += +1.00 <l,k||d,c>_abab t1_bb(b,k) t2_abab(d,c,i,j) t1_2p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("aa")(aa,la) * tmps.at("0157_baab_vooo")(bb,la,ia,jb) )
    
    // r2.at("abab") += +0.50 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t2_abab(c,d,i,j) 
    //         += +0.50 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t2_abab(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0157_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0157_baab_vooo"))
    .allocate(tmps.at("0158_aabb_oooo"))
    
    // flops: o4v0  = o4v0Q1
    //  mems: o4v0  = o4v0
    ( tmps.at("0158_aabb_oooo")(la,ia,kb,jb)  = chol.at("aa_ooQ")(la,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    
    // r2_1p.at("abab") += +0.50 <l,k||i,j>_abab t2_1p_abab(a,b,l,k) 
    //            += +0.50 <k,l||i,j>_abab t2_1p_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,la,kb) * tmps.at("0158_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||i,j>_abab t2_2p_abab(a,b,l,k) 
    //            += +1.00 <k,l||i,j>_abab t2_2p_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_2p.at("abab")(aa,bb,la,kb) * tmps.at("0158_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||i,j>_abab t1_aa(a,k) t1_2p_bb(b,l) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_2p.at("bb")(bb,lb) * tmps.at("0158_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2.at("abab") += +0.50 <l,k||i,j>_abab t2_abab(a,b,l,k) 
    //         += +0.50 <k,l||i,j>_abab t2_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,kb) * tmps.at("0158_aabb_oooo")(la,ia,kb,jb) )
    .allocate(tmps.at("0159_baab_vooo"))
    
    // flops: o3v1  = o4v1
    //  mems: o3v1  = o3v1
    ( tmps.at("0159_baab_vooo")(bb,la,ia,jb)  = t1.at("bb")(bb,kb) * tmps.at("0158_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||i,j>_abab t1_bb(b,k) t1_1p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,la) * tmps.at("0159_baab_vooo")(bb,la,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||i,j>_abab t1_bb(b,k) t1_2p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("aa")(aa,la) * tmps.at("0159_baab_vooo")(bb,la,ia,jb) )
    
    // r2.at("abab") += +1.00 <k,l||i,j>_abab t1_aa(a,k) t1_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0159_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0159_baab_vooo"))
    .allocate(tmps.at("0160_bbaa_ooov"))
    
    // flops: o3v1  = o2v1Q1 o3v1Q1
    //  mems: o3v1  = o2v0Q1 o3v1
    ( tmps.at("bin_bb_ooQ")(jb,lb,Q)  = chol.at("bb_ovQ")(lb,db,Q) * t1_1p.at("bb")(db,jb) )
    ( tmps.at("0160_bbaa_ooov")(lb,jb,ka,ca)  = tmps.at("bin_bb_ooQ")(jb,lb,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t1_aa(a,k) t2_1p_abab(d,b,i,l) t1_1p_bb(c,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t2_1p.at("abab")(da,bb,ia,lb) * tmps.at("0160_bbaa_ooov")(lb,jb,ka,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0161_bbaa_ovoo"))
    
    // flops: o3v1  = o4v1 o4v1
    //  mems: o3v1  = o4v0 o3v1
    ( tmps.at("bin_aabb_oooo")(ia,ka,jb,lb)  = t1.at("aa")(ca,ia) * tmps.at("0160_bbaa_ooov")(lb,jb,ka,ca) )
    ( tmps.at("0161_bbaa_ovoo")(jb,bb,ia,ka)  = tmps.at("bin_aabb_oooo")(ia,ka,jb,lb) * t1.at("bb")(bb,lb) )
    
    // r2_1p.at("abab") += +1.00 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t1_aa(c,i) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0161_bbaa_ovoo")(jb,bb,ia,ka) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t1_1p_aa(a,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,la) * tmps.at("0161_bbaa_ovoo")(jb,bb,ia,la) )
    .deallocate(tmps.at("0161_bbaa_ovoo"))
    .allocate(tmps.at("0162_abba_ovoo"))
    
    // flops: o3v1  = o4v1 o4v1
    //  mems: o3v1  = o4v0 o3v1
    ( tmps.at("bin_aabb_oooo")(ia,ka,jb,lb)  = tmps.at("0154_aabb_ovoo")(ka,ca,lb,jb) * t1.at("aa")(ca,ia) )
    ( tmps.at("0162_abba_ovoo")(ka,bb,jb,ia)  = tmps.at("bin_aabb_oooo")(ia,ka,jb,lb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_1p.at("abab") += +1.00 <k,l||c,j>_abab t1_aa(a,k) t1_aa(c,i) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0162_abba_ovoo")(ka,bb,jb,ia) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,j>_abab t1_aa(c,i) t1_1p_aa(a,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0162_abba_ovoo")(ka,bb,jb,ia) )
    .deallocate(tmps.at("0162_abba_ovoo"))
    .allocate(tmps.at("0163_aabb_ooov"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0163_aabb_ooov")(ka,ia,jb,bb)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||i,c>_abab t1_bb(b,k) t2_2p_abab(a,c,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = t2_2p.at("abab")(aa,cb,la,jb) * tmps.at("0163_aabb_ooov")(la,ia,kb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0164_aabb_oooo"))
    
    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    ( tmps.at("0164_aabb_oooo")(la,ia,jb,kb)  = t1_2p.at("bb")(cb,jb) * tmps.at("0163_aabb_ooov")(la,ia,kb,cb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||i,c>_abab t2_abab(a,b,l,k) t1_2p_bb(c,j) 
    //            += +1.00 <k,l||i,c>_abab t2_abab(a,b,k,l) t1_2p_bb(c,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2.at("abab")(aa,bb,la,kb) * tmps.at("0164_aabb_oooo")(la,ia,jb,kb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||i,c>_abab t1_aa(a,k) t1_bb(b,l) t1_2p_bb(c,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1.at("bb")(bb,lb) * tmps.at("0164_aabb_oooo")(ka,ia,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0164_aabb_oooo"))
    .allocate(tmps.at("0165_baab_vooo"))
    
    // flops: o3v1  = o4v1
    //  mems: o3v1  = o3v1
    ( tmps.at("0165_baab_vooo")(bb,ka,ia,jb)  = t1_1p.at("bb")(bb,lb) * tmps.at("0156_abab_oooo")(ka,lb,ia,jb) )
    .deallocate(tmps.at("0156_abab_oooo"))
    
    // r2_1p.at("abab") += +0.50 <k,l||c,d>_abab t1_aa(a,k) t2_abab(c,d,i,j) t1_1p_bb(b,l) 
    //            += +0.50 <k,l||d,c>_abab t1_aa(a,k) t2_abab(d,c,i,j) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0165_baab_vooo")(bb,ka,ia,jb) )
    
    // r2_2p.at("abab") += +1.00 <k,l||c,d>_abab t2_abab(c,d,i,j) t1_1p_aa(a,k) t1_1p_bb(b,l) 
    //            += +1.00 <k,l||d,c>_abab t2_abab(d,c,i,j) t1_1p_aa(a,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0165_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0165_baab_vooo"))
    .allocate(tmps.at("0166_baab_vooo"))
    
    // flops: o3v1  = o4v1
    //  mems: o3v1  = o3v1
    ( tmps.at("0166_baab_vooo")(bb,ka,ia,jb)  = t1_1p.at("bb")(bb,lb) * tmps.at("0158_aabb_oooo")(ka,ia,lb,jb) )
    .deallocate(tmps.at("0158_aabb_oooo"))
    
    // r2_1p.at("abab") += +1.00 <k,l||i,j>_abab t1_aa(a,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0166_baab_vooo")(bb,ka,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||i,j>_abab t1_1p_aa(a,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0166_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0166_baab_vooo"))
    .allocate(tmps.at("0167_aabb_oooo"))
    
    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    ( tmps.at("0167_aabb_oooo")(la,ia,kb,jb)  = t1.at("aa")(ca,ia) * tmps.at("0154_aabb_ovoo")(la,ca,kb,jb) )
    
    // r2_1p.at("abab") += +0.50 <l,k||c,j>_abab t1_aa(c,i) t2_1p_abab(a,b,l,k) 
    //            += +0.50 <k,l||c,j>_abab t1_aa(c,i) t2_1p_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,la,kb) * tmps.at("0167_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,j>_abab t1_aa(c,i) t2_2p_abab(a,b,l,k) 
    //            += +1.00 <k,l||c,j>_abab t1_aa(c,i) t2_2p_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_2p.at("abab")(aa,bb,la,kb) * tmps.at("0167_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,j>_abab t1_aa(a,k) t1_aa(c,i) t1_2p_bb(b,l) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_2p.at("bb")(bb,lb) * tmps.at("0167_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2.at("abab") += +0.50 <l,k||c,j>_abab t2_abab(a,b,l,k) t1_aa(c,i) 
    //         += +0.50 <k,l||c,j>_abab t2_abab(a,b,k,l) t1_aa(c,i) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,kb) * tmps.at("0167_aabb_oooo")(la,ia,kb,jb) )
    .deallocate(tmps.at("0167_aabb_oooo"))
    .allocate(tmps.at("0168_bbaa_oooo"))
    
    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    ( tmps.at("0168_bbaa_oooo")(kb,jb,ia,la)  = t1.at("aa")(ca,ia) * tmps.at("0160_bbaa_ooov")(kb,jb,la,ca) )
    
    // r2_1p.at("abab") += +0.50 <l,k||c,d>_abab t2_abab(a,b,l,k) t1_aa(c,i) t1_1p_bb(d,j) 
    //            += +0.50 <k,l||c,d>_abab t2_abab(a,b,k,l) t1_aa(c,i) t1_1p_bb(d,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,kb) * tmps.at("0168_bbaa_oooo")(kb,jb,ia,la) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_abab t1_aa(c,i) t2_1p_abab(a,b,l,k) t1_1p_bb(d,j) 
    //            += +1.00 <k,l||c,d>_abab t1_aa(c,i) t2_1p_abab(a,b,k,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_1p.at("abab")(aa,bb,la,kb) * tmps.at("0168_bbaa_oooo")(kb,jb,ia,la) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t1_1p_bb(b,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_1p.at("bb")(bb,lb) * tmps.at("0168_bbaa_oooo")(lb,jb,ia,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0168_bbaa_oooo"))
    .allocate(tmps.at("0169_aabb_oooo"))
    
    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    ( tmps.at("0169_aabb_oooo")(la,ia,kb,jb)  = t1_1p.at("aa")(ca,ia) * tmps.at("0154_aabb_ovoo")(la,ca,kb,jb) )
    
    // r2_1p.at("abab") += +0.50 <l,k||c,j>_abab t2_abab(a,b,l,k) t1_1p_aa(c,i) 
    //            += +0.50 <k,l||c,j>_abab t2_abab(a,b,k,l) t1_1p_aa(c,i) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,kb) * tmps.at("0169_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,j>_abab t2_1p_abab(a,b,l,k) t1_1p_aa(c,i) 
    //            += +1.00 <k,l||c,j>_abab t2_1p_abab(a,b,k,l) t1_1p_aa(c,i) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_1p.at("abab")(aa,bb,la,kb) * tmps.at("0169_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,j>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_1p_aa(c,i) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_1p.at("bb")(bb,lb) * tmps.at("0169_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0169_aabb_oooo"))
    .allocate(tmps.at("0170_bbaa_oooo"))
    
    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    ( tmps.at("0170_bbaa_oooo")(kb,jb,ia,la)  = t1_1p.at("aa")(da,ia) * tmps.at("0160_bbaa_ooov")(kb,jb,la,da) )
    .deallocate(tmps.at("0160_bbaa_ooov"))
    
    // r2_2p.at("abab") += +1.00 <l,k||d,c>_abab t2_abab(a,b,l,k) t1_1p_bb(c,j) t1_1p_aa(d,i) 
    //            += +1.00 <k,l||d,c>_abab t2_abab(a,b,k,l) t1_1p_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2.at("abab")(aa,bb,la,kb) * tmps.at("0170_bbaa_oooo")(kb,jb,ia,la) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t1_1p_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1.at("bb")(bb,lb) * tmps.at("0170_bbaa_oooo")(lb,jb,ia,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0170_bbaa_oooo"))
    .allocate(tmps.at("0171_aabb_oooo"))
    
    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    ( tmps.at("0171_aabb_oooo")(la,ia,kb,jb)  = t1_2p.at("aa")(ca,ia) * tmps.at("0154_aabb_ovoo")(la,ca,kb,jb) )
    .deallocate(tmps.at("0154_aabb_ovoo"))
    
    // r2_2p.at("abab") += +1.00 <l,k||c,j>_abab t2_abab(a,b,l,k) t1_2p_aa(c,i) 
    //            += +1.00 <k,l||c,j>_abab t2_abab(a,b,k,l) t1_2p_aa(c,i) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2.at("abab")(aa,bb,la,kb) * tmps.at("0171_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,j>_abab t1_aa(a,k) t1_bb(b,l) t1_2p_aa(c,i) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1.at("bb")(bb,lb) * tmps.at("0171_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0171_aabb_oooo"))
    .allocate(tmps.at("0172_bb_oo"))
    
    // flops: o2v0  = o3v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0172_bb_oo")(jb,lb)  = chol.at("bb_ooQ")(kb,jb,Q) * tmps.at("0107_bb_ooQ")(lb,kb,Q) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,j>_bbbb t1_bb(c,k) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0172_bb_oo")(jb,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,j>_bbbb t1_bb(c,k) t2_2p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_2p.at("abab")(aa,bb,ia,lb) * tmps.at("0172_bb_oo")(jb,lb) )
    
    // r2.at("abab") += +1.00 <l,k||c,j>_bbbb t2_abab(a,b,i,l) t1_bb(c,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ia,lb) * tmps.at("0172_bb_oo")(jb,lb) )
    .deallocate(tmps.at("0172_bb_oo"))
    .allocate(tmps.at("0173_aaaa_ooov"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0173_aaaa_ooov")(ja,ia,ka,ba)  = chol.at("aa_ooQ")(ja,ia,Q) * chol.at("aa_ovQ")(ka,ba,Q) )
    
    // r1_1p.at("aa") += -0.50 <k,j||b,i>_aaaa t2_1p_aaaa(b,a,k,j) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.50 * t2_1p.at("aaaa")(ba,aa,ka,ja) * tmps.at("0173_aaaa_ooov")(ja,ia,ka,ba) )
    
    // r1_1p.at("aa") += -1.00 <k,j||b,i>_aaaa t1_aa(a,j) t1_1p_aa(b,k) 
    // flops: o1v1 += o3v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ja)  = t1_1p.at("aa")(ba,ka) * tmps.at("0173_aaaa_ooov")(ka,ia,ja,ba) )
    ( r1_1p.at("aa")(aa,ia) += tmps.at("bin_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r1_2p.at("aa") += -1.00 <k,j||b,i>_aaaa t2_2p_aaaa(b,a,k,j) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= t2_2p.at("aaaa")(ba,aa,ka,ja) * tmps.at("0173_aaaa_ooov")(ja,ia,ka,ba) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,i>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,j) 
    // flops: o1v1 += o3v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ka)  = t1_1p.at("aa")(ba,ja) * tmps.at("0173_aaaa_ooov")(ja,ia,ka,ba) )
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("bin_aa_oo")(ia,ka) * t1_1p.at("aa")(aa,ka) )
    
    // r1.at("aa") += -0.50 <k,j||b,i>_aaaa t2_aaaa(b,a,k,j) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= 0.50 * t2.at("aaaa")(ba,aa,ka,ja) * tmps.at("0173_aaaa_ooov")(ja,ia,ka,ba) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,i>_aaaa t2_abab(a,b,k,j) t1_1p_aa(c,l) 
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,ka)  = t1_1p.at("aa")(ca,la) * tmps.at("0173_aaaa_ooov")(la,ia,ka,ca) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_aa_oo")(ia,ka) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,i>_aaaa t2_1p_abab(a,b,l,j) t1_1p_aa(c,k) 
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,la)  = t1_1p.at("aa")(ca,ka) * tmps.at("0173_aaaa_ooov")(ka,ia,la,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aa_oo")(ia,la) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,i>_aaaa t1_aa(a,k) t2_2p_abab(c,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(ca,bb,la,jb) * tmps.at("0173_aaaa_ooov")(la,ia,ka,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0174_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0174_aa_oo")(ia,ka)  = t1.at("aa")(ba,ja) * tmps.at("0173_aaaa_ooov")(ja,ia,ka,ba) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,i>_aaaa t1_aa(b,j) t1_1p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1_1p.at("aa")(aa,ka) * tmps.at("0174_aa_oo")(ia,ka) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,i>_aaaa t1_aa(b,j) t1_2p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * t1_2p.at("aa")(aa,ka) * tmps.at("0174_aa_oo")(ia,ka) )
    
    // r1.at("aa") += +1.00 <k,j||b,i>_aaaa t1_aa(a,k) t1_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += t1.at("aa")(aa,ka) * tmps.at("0174_aa_oo")(ia,ka) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,i>_aaaa t1_aa(c,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,la,jb) * tmps.at("0174_aa_oo")(ia,la) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,i>_aaaa t1_aa(c,k) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_2p.at("abab")(aa,bb,la,jb) * tmps.at("0174_aa_oo")(ia,la) )
    
    // r2.at("abab") += +1.00 <l,k||c,i>_aaaa t2_abab(a,b,l,j) t1_aa(c,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,jb) * tmps.at("0174_aa_oo")(ia,la) )
    .deallocate(tmps.at("0174_aa_oo"))
    .allocate(tmps.at("0175_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0175_aa_oo")(ia,ja)  = t1_2p.at("aa")(ba,ka) * tmps.at("0173_aaaa_ooov")(ka,ia,ja,ba) )
    .deallocate(tmps.at("0173_aaaa_ooov"))
    
    // r1_2p.at("aa") += -2.00 <k,j||b,i>_aaaa t1_aa(a,j) t1_2p_aa(b,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * t1.at("aa")(aa,ja) * tmps.at("0175_aa_oo")(ia,ja) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,i>_aaaa t2_abab(a,b,k,j) t1_2p_aa(c,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2.at("abab")(aa,bb,ka,jb) * tmps.at("0175_aa_oo")(ia,ka) )
    .deallocate(tmps.at("0175_aa_oo"))
    .allocate(tmps.at("0176_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0176_Q")(Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t1.at("aa")(ba,ja) )
    
    // r2_2p.at("abab") += -2.00 <k,l||c,j>_abab t1_aa(c,k) t2_2p_abab(a,b,i,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0176_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,lb) * t2_2p.at("abab")(aa,bb,ia,lb) )
    
    // r1_1p.at("aa") += +1.00 <j,k||b,c>_abab t1_aa(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0111_aa_voQ")(aa,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r1.at("aa") += +1.00 <j,a||b,i>_aaaa t1_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r1_1p.at("aa") += +1.00 <j,a||b,c>_aaaa t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0113_aa_voQ")(aa,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r2_1p.at("abab") += -1.00 <k,l||c,j>_abab t1_aa(c,k) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0176_Q")(Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    
    // r1.at("aa") += +1.00 <j,a||b,c>_aaaa t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_aa_vo")(da,la)  = chol.at("aa_ovQ")(la,da,Q) * tmps.at("0176_Q")(Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin_aa_vo")(da,la) * t2_2p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,la,jb) * t1.at("aa")(aa,la) )
    
    // r2_2p.at("abab") += -2.00 <k,l||c,d>_abab t1_bb(b,l) t1_aa(c,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_bb_vo")(db,lb)  = chol.at("bb_ovQ")(lb,db,Q) * tmps.at("0176_Q")(Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("bin_bb_vo")(db,lb) * t2_2p.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1.at("bb")(bb,lb) )
    
    // r2.at("abab") += -1.00 <k,l||c,j>_abab t2_abab(a,b,i,l) t1_aa(c,k) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0176_Q")(Q) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r1.at("aa") += +1.00 <k,j||b,c>_aaaa t2_aaaa(c,a,i,k) t1_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0176_Q")(Q) )
    .allocate(tmps.at("0177_bb_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0177_bb_oo")(lb,jb)  = tmps.at("0107_bb_ooQ")(lb,jb,Q) * tmps.at("0176_Q")(Q) )
    
    // r2_1p.at("abab") += -1.00 <k,l||c,d>_abab t1_aa(c,k) t1_bb(d,j) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0177_bb_oo")(lb,jb) )
    
    // r2_2p.at("abab") += -2.00 <k,l||c,d>_abab t1_aa(c,k) t1_bb(d,j) t2_2p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_2p.at("abab")(aa,bb,ia,lb) * tmps.at("0177_bb_oo")(lb,jb) )
    
    // r2.at("abab") += -1.00 <k,l||c,d>_abab t2_abab(a,b,i,l) t1_aa(c,k) t1_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,lb) * tmps.at("0177_bb_oo")(lb,jb) )
    .deallocate(tmps.at("0177_bb_oo"))
    .allocate(tmps.at("0178_aa_voQ"))
    
    // flops: o1v1Q1  = o1v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0178_aa_voQ")(aa,ka,Q)  = chol.at("aa_vvQ")(aa,da,Q) * t1_2p.at("aa")(da,ka) )
    
    // r2_2p.at("abab") += -2.00 <a,k||d,c>_abab t2_bbbb(c,b,j,k) t1_2p_aa(d,i) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0100_bb_voQ")(bb,jb,Q) * tmps.at("0178_aa_voQ")(aa,ia,Q) )
    
    // r2_2p.at("abab") += +2.00 <a,b||c,j>_abab t1_2p_aa(c,i) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0178_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += -2.00 <k,a||c,d>_aaaa t2_abab(c,b,i,j) t1_2p_aa(d,k) 
    // flops: o2v2 += o1v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_aa_vv")(aa,ca)  = tmps.at("0178_aa_voQ")(aa,ka,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r1_2p.at("aa") += +2.00 <j,a||b,c>_aaaa t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0178_aa_voQ")(aa,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r2_2p.at("abab") += +2.00 <k,a||c,d>_aaaa t2_abab(c,b,k,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0102_bb_voQ")(bb,jb,Q) * tmps.at("0178_aa_voQ")(aa,ia,Q) )
    
    // r2_2p.at("abab") += -2.00 <a,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0107_bb_ooQ")(kb,jb,Q) * tmps.at("0178_aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0179_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0179_aa_voQ")(ca,ia,Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t2_2p.at("aaaa")(ba,ca,ia,ja) )
    
    // r2_2p.at("abab") += +2.00 <l,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t2_2p_aaaa(d,a,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0107_bb_ooQ")(kb,jb,Q) * tmps.at("0179_aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r1_2p.at("aa") += -1.00 <k,j||b,i>_aaaa t2_2p_aaaa(b,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0179_aa_voQ")(aa,ka,Q) )
    
    // r1_2p.at("aa") += -1.00 <j,a||b,c>_aaaa t2_2p_aaaa(b,c,i,j) 
    // flops: o1v1 += o1v2Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= chol.at("aa_vvQ")(aa,ca,Q) * tmps.at("0179_aa_voQ")(ca,ia,Q) )
    
    // r2_2p.at("abab") += -2.00 <k,b||c,j>_abab t2_2p_aaaa(c,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0179_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,j>_abab t1_bb(b,k) t2_2p_aaaa(c,a,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0179_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t1_aa(b,j) t2_2p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0179_aa_voQ")(aa,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t2_abab(c,b,k,j) t2_2p_aaaa(d,a,i,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0102_bb_voQ")(bb,jb,Q) * tmps.at("0179_aa_voQ")(aa,ia,Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||d,c>_abab t2_bbbb(c,b,j,k) t2_2p_aaaa(d,a,i,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0100_bb_voQ")(bb,jb,Q) * tmps.at("0179_aa_voQ")(aa,ia,Q) )
    
    // r2_2p.at("abab") += -2.00 <k,b||d,c>_abab t1_bb(c,j) t2_2p_aaaa(d,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0097_bb_voQ")(bb,jb,Q) * tmps.at("0179_aa_voQ")(aa,ia,Q) )
    .allocate(tmps.at("0180_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0180_aa_voQ")(ba,ia,Q)  = chol.at("bb_ovQ")(jb,cb,Q) * t2_2p.at("abab")(ba,cb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_2p_abab(a,d,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0180_aa_voQ")(aa,ia,Q) * tmps.at("0107_bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r1_2p.at("aa") += -1.00 <k,j||i,b>_abab t2_2p_abab(a,b,k,j) 
    //          += -1.00 <j,k||i,b>_abab t2_2p_abab(a,b,j,k) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0180_aa_voQ")(aa,ka,Q) )
    
    // r1_2p.at("aa") += +1.00 <a,j||b,c>_abab t2_2p_abab(b,c,i,j) 
    //          += +1.00 <a,j||c,b>_abab t2_2p_abab(c,b,i,j) 
    // flops: o1v1 += o1v2Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * chol.at("aa_vvQ")(aa,ba,Q) * tmps.at("0180_aa_voQ")(ba,ia,Q) )
    
    // r1_2p.at("aa") += -1.00 <k,j||b,c>_abab t1_aa(b,i) t2_2p_abab(a,c,k,j) 
    //          += -1.00 <j,k||b,c>_abab t1_aa(b,i) t2_2p_abab(a,c,j,k) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0180_aa_voQ")(aa,ka,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    
    // r1_2p.at("aa") += -1.00 <j,k||b,c>_abab t1_aa(a,j) t2_2p_abab(b,c,i,k) 
    //          += -1.00 <j,k||c,b>_abab t1_aa(a,j) t2_2p_abab(c,b,i,k) 
    // flops: o1v1 += o2v1Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ja)  = chol.at("aa_ovQ")(ja,ba,Q) * tmps.at("0180_aa_voQ")(ba,ia,Q) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r2_2p.at("abab") += -1.00 <k,l||c,d>_abab t2_abab(a,b,k,j) t2_2p_abab(c,d,i,l) 
    //            += -1.00 <k,l||d,c>_abab t2_abab(a,b,k,j) t2_2p_abab(d,c,i,l) 
    // flops: o2v2 += o2v1Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,ka)  = chol.at("aa_ovQ")(ka,ca,Q) * tmps.at("0180_aa_voQ")(ca,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aa_oo")(ia,ka) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,j>_bbbb t1_bb(b,k) t2_2p_abab(a,c,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0180_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_abab t2_abab(c,b,i,j) t2_2p_abab(a,d,l,k) 
    //            += -1.00 <k,l||c,d>_abab t2_abab(c,b,i,j) t2_2p_abab(a,d,k,l) 
    // flops: o2v2 += o1v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_aa_vv")(aa,ca)  = tmps.at("0180_aa_voQ")(aa,la,Q) * chol.at("aa_ovQ")(la,ca,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_bbbb(c,b,j,k) t2_2p_abab(a,d,i,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0100_bb_voQ")(bb,jb,Q) * tmps.at("0180_aa_voQ")(aa,ia,Q) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,d>_abab t2_abab(c,b,k,j) t2_2p_abab(a,d,i,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0102_bb_voQ")(bb,jb,Q) * tmps.at("0180_aa_voQ")(aa,ia,Q) )
    
    // r2_2p.at("abab") += -2.00 <k,b||c,d>_bbbb t1_bb(c,j) t2_2p_abab(a,d,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0097_bb_voQ")(bb,jb,Q) * tmps.at("0180_aa_voQ")(aa,ia,Q) )
    
    // r2_2p.at("abab") += +2.00 <k,b||c,j>_bbbb t2_2p_abab(a,c,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0180_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r1_2p.at("aa") += +2.00 <j,k||b,c>_abab t1_aa(b,j) t2_2p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0180_aa_voQ")(aa,ia,Q) * tmps.at("0176_Q")(Q) )
    .allocate(tmps.at("0181_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0181_Q")(Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t1.at("bb")(bb,jb) )
    
    // r1_2p.at("aa") += +2.00 <a,j||c,b>_abab t1_bb(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0178_aa_voQ")(aa,ia,Q) * tmps.at("0181_Q")(Q) )
    .deallocate(tmps.at("0178_aa_voQ"))
    
    // r1_1p.at("aa") += -1.00 <k,j||i,b>_abab t1_bb(b,j) t1_1p_aa(a,k) 
    // flops: o1v1 += o2v0Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0181_Q")(Q) )
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("bin_aa_oo")(ia,ka) * t1_1p.at("aa")(aa,ka) )
    
    // r1_2p.at("aa") += -2.00 <k,j||i,b>_abab t1_bb(b,j) t1_2p_aa(a,k) 
    // flops: o1v1 += o2v0Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0181_Q")(Q) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_aa_oo")(ia,ka) * t1_2p.at("aa")(aa,ka) )
    
    // r1.at("aa") += +1.00 <a,j||i,b>_abab t1_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0181_Q")(Q) )
    
    // r1.at("aa") += -1.00 <k,j||i,b>_abab t1_aa(a,k) t1_bb(b,j) 
    // flops: o1v1 += o2v0Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0181_Q")(Q) )
    ( r1.at("aa")(aa,ia) -= tmps.at("bin_aa_oo")(ia,ka) * t1.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += -1.00 <l,k||i,c>_abab t1_bb(c,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,la)  = chol.at("aa_ooQ")(la,ia,Q) * tmps.at("0181_Q")(Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aa_oo")(ia,la) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,j>_bbbb t1_bb(c,k) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0181_Q")(Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||i,c>_abab t1_bb(c,k) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,la)  = chol.at("aa_ooQ")(la,ia,Q) * tmps.at("0181_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aa_oo")(ia,la) * t2_2p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,j>_bbbb t1_bb(c,k) t2_2p_abab(a,b,i,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0181_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,lb) * t2_2p.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_aa_vo")(da,la)  = chol.at("aa_ovQ")(la,da,Q) * tmps.at("0181_Q")(Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin_aa_vo")(da,la) * t2_2p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,la,jb) * t1.at("aa")(aa,la) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_bb_vo")(db,lb)  = chol.at("bb_ovQ")(lb,db,Q) * tmps.at("0181_Q")(Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("bin_bb_vo")(db,lb) * t2_2p.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1.at("bb")(bb,lb) )
    
    // r2.at("abab") += -1.00 <l,k||i,c>_abab t2_abab(a,b,l,j) t1_bb(c,k) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,la)  = chol.at("aa_ooQ")(la,ia,Q) * tmps.at("0181_Q")(Q) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2.at("abab") += +1.00 <l,k||c,j>_bbbb t2_abab(a,b,i,l) t1_bb(c,k) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0181_Q")(Q) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r1_2p.at("aa") += -2.00 <k,j||c,b>_abab t1_bb(b,j) t2_2p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0179_aa_voQ")(aa,ia,Q) * tmps.at("0181_Q")(Q) )
    .deallocate(tmps.at("0179_aa_voQ"))
    
    // r1_1p.at("aa") += -1.00 <k,j||b,c>_bbbb t1_bb(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0111_aa_voQ")(aa,ia,Q) * tmps.at("0181_Q")(Q) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_bbbb t1_bb(b,j) t2_2p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0180_aa_voQ")(aa,ia,Q) * tmps.at("0181_Q")(Q) )
    .deallocate(tmps.at("0180_aa_voQ"))
    
    // r1.at("aa") += +1.00 <a,j||c,b>_abab t1_bb(b,j) t1_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0181_Q")(Q) )
    
    // r1.at("aa") += -1.00 <k,j||c,b>_abab t2_aaaa(c,a,i,k) t1_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0181_Q")(Q) )
    
    // r1_1p.at("aa") += +1.00 <a,j||c,b>_abab t1_bb(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0113_aa_voQ")(aa,ia,Q) * tmps.at("0181_Q")(Q) )
    .allocate(tmps.at("0182_bb_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0182_bb_oo")(lb,jb)  = tmps.at("0107_bb_ooQ")(lb,jb,Q) * tmps.at("0181_Q")(Q) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,j) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0182_bb_oo")(lb,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,j) t2_2p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_2p.at("abab")(aa,bb,ia,lb) * tmps.at("0182_bb_oo")(lb,jb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,b,i,l) t1_bb(c,k) t1_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,lb) * tmps.at("0182_bb_oo")(lb,jb) )
    .deallocate(tmps.at("0182_bb_oo"))
    .allocate(tmps.at("0183_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0183_Q")(Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t1_1p.at("aa")(ba,ja) )
    
    // r1_1p.at("aa") += +1.00 <j,a||b,i>_aaaa t1_1p_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0183_Q")(Q) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,j>_abab t2_abab(a,b,i,k) t1_1p_aa(c,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = chol.at("bb_ooQ")(kb,jb,Q) * tmps.at("0183_Q")(Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r1_2p.at("aa") += +2.00 <j,a||b,c>_aaaa t1_1p_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0113_aa_voQ")(aa,ia,Q) * tmps.at("0183_Q")(Q) )
    
    // r2_2p.at("abab") += -2.00 <k,l||c,j>_abab t2_1p_abab(a,b,i,l) t1_1p_aa(c,k) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0183_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_abab t2_abab(a,b,i,k) t1_1p_aa(c,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = tmps.at("0115_bb_ooQ")(kb,jb,Q) * tmps.at("0183_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t2_abab(a,b,k,j) t1_aa(c,i) t1_1p_aa(d,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,ka)  = tmps.at("0098_aa_ooQ")(ka,ia,Q) * tmps.at("0183_Q")(Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aa_oo")(ia,ka) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r1_2p.at("aa") += +2.00 <j,k||b,c>_abab t2_1p_abab(a,c,i,k) t1_1p_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0111_aa_voQ")(aa,ia,Q) * tmps.at("0183_Q")(Q) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_aaaa t1_aa(a,k) t2_1p_abab(d,b,i,j) t1_1p_aa(c,l) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_aa_vo")(da,ka)  = chol.at("aa_ovQ")(ka,da,Q) * tmps.at("0183_Q")(Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aa_vo")(da,ka) * t2_1p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_aaaa t1_aa(a,j) t1_1p_aa(b,k) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v0Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ja)  = tmps.at("0104_aa_ooQ")(ja,ia,Q) * tmps.at("0183_Q")(Q) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_aaaa t2_abab(a,b,k,j) t1_1p_aa(c,l) t1_1p_aa(d,i) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,ka)  = tmps.at("0104_aa_ooQ")(ka,ia,Q) * tmps.at("0183_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aa_oo")(ia,ka) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_abab t1_bb(b,k) t2_1p_abab(a,d,i,j) t1_1p_aa(c,l) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_bb_vo")(db,kb)  = chol.at("bb_ovQ")(kb,db,Q) * tmps.at("0183_Q")(Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bb_vo")(db,kb) * t2_1p.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,c>_aaaa t2_aaaa(b,a,i,j) t1_1p_aa(c,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0183_Q")(Q) )
    
    // r1_1p.at("aa") += -1.00 <j,a||b,c>_aaaa t1_aa(b,i) t1_1p_aa(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0183_Q")(Q) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(a,j) t1_aa(b,i) t1_1p_aa(c,k) 
    // flops: o1v1 += o2v0Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ja)  = tmps.at("0098_aa_ooQ")(ja,ia,Q) * tmps.at("0183_Q")(Q) )
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("bin_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_aaaa t1_aa(b,i) t1_1p_aa(a,k) t1_1p_aa(c,j) 
    // flops: o1v1 += o2v0Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ka)  = tmps.at("0098_aa_ooQ")(ka,ia,Q) * tmps.at("0183_Q")(Q) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_aa_oo")(ia,ka) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_aaaa t1_aa(c,i) t2_1p_abab(a,b,l,j) t1_1p_aa(d,k) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,la)  = tmps.at("0098_aa_ooQ")(la,ia,Q) * tmps.at("0183_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aa_oo")(ia,la) * t2_1p.at("abab")(aa,bb,la,jb) )
    .allocate(tmps.at("0184_bb_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0184_bb_oo")(kb,jb)  = tmps.at("0107_bb_ooQ")(kb,jb,Q) * tmps.at("0183_Q")(Q) )
    
    // r2_1p.at("abab") += -1.00 <l,k||d,c>_abab t2_abab(a,b,i,k) t1_bb(c,j) t1_1p_aa(d,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,kb) * tmps.at("0184_bb_oo")(kb,jb) )
    
    // r2_2p.at("abab") += -2.00 <k,l||d,c>_abab t1_bb(c,j) t2_1p_abab(a,b,i,l) t1_1p_aa(d,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0184_bb_oo")(lb,jb) )
    .deallocate(tmps.at("0184_bb_oo"))
    .allocate(tmps.at("0185_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0185_Q")(Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t1_1p.at("bb")(bb,jb) )
    
    // r1_2p.at("aa") += +2.00 <a,j||c,b>_abab t1_1p_bb(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0113_aa_voQ")(aa,ia,Q) * tmps.at("0185_Q")(Q) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t1_bb(b,k) t2_1p_abab(a,d,i,j) t1_1p_bb(c,l) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_bb_vo")(db,kb)  = chol.at("bb_ovQ")(kb,db,Q) * tmps.at("0185_Q")(Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bb_vo")(db,kb) * t2_1p.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -2.00 <k,l||d,c>_abab t2_abab(a,b,k,j) t1_1p_bb(c,l) t1_1p_aa(d,i) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,ka)  = tmps.at("0104_aa_ooQ")(ka,ia,Q) * tmps.at("0185_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aa_oo")(ia,ka) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,j>_bbbb t2_1p_abab(a,b,i,l) t1_1p_bb(c,k) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0185_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t2_abab(a,b,i,k) t1_1p_bb(c,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = tmps.at("0115_bb_ooQ")(kb,jb,Q) * tmps.at("0185_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r1_1p.at("aa") += +1.00 <a,j||b,c>_abab t1_aa(b,i) t1_1p_bb(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0185_Q")(Q) )
    
    // r1_1p.at("aa") += -1.00 <j,k||b,c>_abab t2_aaaa(b,a,i,j) t1_1p_bb(c,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0185_Q")(Q) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_bbbb t2_1p_abab(a,c,i,k) t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0111_aa_voQ")(aa,ia,Q) * tmps.at("0185_Q")(Q) )
    
    // r1_1p.at("aa") += +1.00 <a,j||i,b>_abab t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0185_Q")(Q) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,j>_bbbb t2_abab(a,b,i,k) t1_1p_bb(c,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = chol.at("bb_ooQ")(kb,jb,Q) * tmps.at("0185_Q")(Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += -2.00 <k,l||d,c>_abab t1_aa(a,k) t2_1p_abab(d,b,i,j) t1_1p_bb(c,l) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_aa_vo")(da,ka)  = chol.at("aa_ovQ")(ka,da,Q) * tmps.at("0185_Q")(Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aa_vo")(da,ka) * t2_1p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r1_2p.at("aa") += -2.00 <j,k||c,b>_abab t1_aa(a,j) t1_1p_bb(b,k) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v0Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ja)  = tmps.at("0104_aa_ooQ")(ja,ia,Q) * tmps.at("0185_Q")(Q) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    .allocate(tmps.at("0186_bb_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0186_bb_oo")(kb,jb)  = tmps.at("0107_bb_ooQ")(kb,jb,Q) * tmps.at("0185_Q")(Q) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,b,i,k) t1_bb(c,j) t1_1p_bb(d,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,kb) * tmps.at("0186_bb_oo")(kb,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t1_bb(c,j) t2_1p_abab(a,b,i,l) t1_1p_bb(d,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0186_bb_oo")(lb,jb) )
    .deallocate(tmps.at("0186_bb_oo"))
    .allocate(tmps.at("0187_bb_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0187_bb_oo")(lb,jb)  = tmps.at("0115_bb_ooQ")(lb,jb,Q) * tmps.at("0181_Q")(Q) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,b,i,l) t1_bb(c,k) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,lb) * tmps.at("0187_bb_oo")(lb,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(c,k) t2_1p_abab(a,b,i,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0187_bb_oo")(lb,jb) )
    .deallocate(tmps.at("0187_bb_oo"))
    .allocate(tmps.at("0188_bb_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0188_bb_oo")(lb,jb)  = tmps.at("0115_bb_ooQ")(lb,jb,Q) * tmps.at("0176_Q")(Q) )
    
    // r2_1p.at("abab") += -1.00 <k,l||c,d>_abab t2_abab(a,b,i,l) t1_aa(c,k) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,lb) * tmps.at("0188_bb_oo")(lb,jb) )
    
    // r2_2p.at("abab") += -2.00 <k,l||c,d>_abab t1_aa(c,k) t2_1p_abab(a,b,i,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0188_bb_oo")(lb,jb) )
    .deallocate(tmps.at("0188_bb_oo"))
    .allocate(tmps.at("0189_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0189_aa_oo")(ka,ia)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,i>_aaaa t1_aa(b,j) t1_1p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1_1p.at("aa")(aa,ka) * tmps.at("0189_aa_oo")(ka,ia) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,i>_aaaa t1_aa(b,j) t1_2p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1_2p.at("aa")(aa,ka) * tmps.at("0189_aa_oo")(ka,ia) )
    
    // r1.at("aa") += +1.00 <k,j||b,i>_aaaa t1_aa(a,k) t1_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= t1.at("aa")(aa,ka) * tmps.at("0189_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,i>_aaaa t1_aa(c,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0189_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,i>_aaaa t1_aa(c,k) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0189_aa_oo")(la,ia) * t2_2p.at("abab")(aa,bb,la,jb) )
    
    // r2.at("abab") += +1.00 <l,k||c,i>_aaaa t2_abab(a,b,l,j) t1_aa(c,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0189_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0189_aa_oo"))
    .allocate(tmps.at("0190_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0190_aa_oo")(ka,ia)  = tmps.at("0098_aa_ooQ")(ka,ia,Q) * tmps.at("0181_Q")(Q) )
    
    // r1_1p.at("aa") += -1.00 <k,j||c,b>_abab t1_bb(b,j) t1_aa(c,i) t1_1p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1_1p.at("aa")(aa,ka) * tmps.at("0190_aa_oo")(ka,ia) )
    
    // r1_2p.at("aa") += -2.00 <k,j||c,b>_abab t1_bb(b,j) t1_aa(c,i) t1_2p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1_2p.at("aa")(aa,ka) * tmps.at("0190_aa_oo")(ka,ia) )
    
    // r1.at("aa") += -1.00 <k,j||c,b>_abab t1_aa(a,k) t1_bb(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= t1.at("aa")(aa,ka) * tmps.at("0190_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += -1.00 <l,k||d,c>_abab t1_bb(c,k) t1_aa(d,i) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0190_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||d,c>_abab t1_bb(c,k) t1_aa(d,i) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0190_aa_oo")(la,ia) * t2_2p.at("abab")(aa,bb,la,jb) )
    
    // r2.at("abab") += -1.00 <l,k||d,c>_abab t2_abab(a,b,l,j) t1_bb(c,k) t1_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0190_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0190_aa_oo"))
    .allocate(tmps.at("0191_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0191_aa_oo")(ka,ia)  = tmps.at("0098_aa_ooQ")(ka,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(b,j) t1_aa(c,i) t1_1p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1_1p.at("aa")(aa,ka) * tmps.at("0191_aa_oo")(ka,ia) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t1_aa(b,j) t1_aa(c,i) t1_2p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1_2p.at("aa")(aa,ka) * tmps.at("0191_aa_oo")(ka,ia) )
    
    // r1.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= t1.at("aa")(aa,ka) * tmps.at("0191_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0191_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )

    ;
  }
  // clang-format on
}

template void exachem::cc::cd_qed_ccsd_cs::resid_part3<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);