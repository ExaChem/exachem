/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_qed_ccsd_cs_resid_2.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_cs::resid_part2(
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
        
    // r2[abab] += -1.000 d-_bb(k,j) t2_1p_abab(a,b,i,k) 
    //            += -1.000 d-_bb(k,c) t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    //            += -1.000 d-_bb(k,c) t1_1p_bb(c,j) t2_abab(a,b,i,k) 
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0044_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_1p[abab] += -1.000 d-_bb(k,j) t0_1p t2_1p_abab(a,b,i,k) 
    //               += -1.000 d-_bb(k,c) t0_1p t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    //               += -1.000 d-_bb(k,c) t0_1p t1_1p_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0044_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_2p[abab] += -2.000 d+_bb(k,j) t2_1p_abab(a,b,i,k) 
    //               += -2.000 d+_bb(k,c) t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    //               += -2.000 d+_bb(k,c) t1_1p_bb(c,j) t2_abab(a,b,i,k) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0044_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_2p[abab] += -4.000 d-_bb(k,j) t0_2p t2_1p_abab(a,b,i,k) 
    //               += -4.000 d-_bb(k,c) t0_2p t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    //               += -4.000 d-_bb(k,c) t0_2p t1_1p_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.000 * t0_2p * tmps.at("0044_abba_vvoo")(aa,bb,jb,ia) )
    .deallocate(tmps.at("0044_abba_vvoo"))
    .allocate(tmps.at("0045_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0045_bb_oo")(kb,jb)  = dp.at("bb_ov")(kb,cb) * t1_2p.at("bb")(cb,jb) )
    
    // r2_2p[abab] += -2.000 d-_bb(k,c) t1_1p_aa(a,i) t1_bb(b,k) t1_2p_bb(c,j) 
    // flops: o2v2 += o2v1 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin1_bb_vo")(bb,jb)  = tmps.at("0045_bb_oo")(kb,jb) * t1.at("bb")(bb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_1p.at("aa")(aa,ia) * tmps.at("bin1_bb_vo")(bb,jb) )
    
    // r2_2p[abab] += -6.000 d-_bb(k,c) t1_2p_bb(c,j) t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * t2_1p.at("abab")(aa,bb,ia,kb) * tmps.at("0045_bb_oo")(kb,jb) )
    .allocate(tmps.at("0046_abba_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o2v1 o3v2 o2v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0046_abba_vvoo")(aa,bb,jb,ia)  = t2.at("abab")(aa,bb,ia,kb) * tmps.at("0045_bb_oo")(kb,jb) )
    ( tmps.at("0046_abba_vvoo")(aa,bb,jb,ia) += t2_1p.at("abab")(aa,bb,ia,kb) * tmps.at("0041_bb_oo")(kb,jb) )
    ( tmps.at("bin1_bb_oo")(jb,kb)  = t1.at("bb")(cb,jb) * dp.at("bb_ov")(kb,cb) )
    ( tmps.at("0046_abba_vvoo")(aa,bb,jb,ia) += t2_2p.at("abab")(aa,bb,ia,kb) * tmps.at("bin1_bb_oo")(jb,kb) )
    ( tmps.at("0046_abba_vvoo")(aa,bb,jb,ia) += t2_2p.at("abab")(aa,bb,ia,kb) * dp.at("bb_oo")(kb,jb) )
    .deallocate(tmps.at("0045_bb_oo"))
    .deallocate(tmps.at("0041_bb_oo"))
    
    // r2_1p[abab] += -2.000 d-_bb(k,j) t2_2p_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t1_bb(c,j) t2_2p_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t1_1p_bb(c,j) t2_1p_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t1_2p_bb(c,j) t2_abab(a,b,i,k) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0046_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_2p[abab] += -2.000 d-_bb(k,j) t0_1p t2_2p_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t0_1p t1_bb(c,j) t2_2p_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t0_1p t1_1p_bb(c,j) t2_1p_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t0_1p t1_2p_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t0_1p * tmps.at("0046_abba_vvoo")(aa,bb,jb,ia) )
    .deallocate(tmps.at("0046_abba_vvoo"))
    .allocate(tmps.at("0047_abab_vooo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v1
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1 o0v0Q1 o1v1 o3v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("0047_abab_vooo")(aa,lb,ia,jb)  = tmps.at("bin1_bb_vo")(db,lb) * t2.at("abab")(aa,db,ia,jb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("0047_abab_vooo")(aa,lb,ia,jb) += tmps.at("bin1_bb_vo")(db,lb) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2[abab] += -1.000 <k,l||c,d>_abab t1_bb(b,l) t1_aa(c,k) t2_abab(a,d,i,j) 
    //            += +1.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0047_abab_vooo")(aa,lb,ia,jb) * t1.at("bb")(bb,lb) )
    
    // r2_1p[abab] += -1.000 <k,l||c,d>_abab t1_1p_bb(b,l) t1_aa(c,k) t2_abab(a,d,i,j) 
    //               += +1.000 <l,k||c,d>_bbbb t1_1p_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0047_abab_vooo")(aa,lb,ia,jb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_abab t1_2p_bb(b,l) t1_aa(c,k) t2_abab(a,d,i,j) 
    //               += +2.000 <l,k||c,d>_bbbb t1_2p_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0047_abab_vooo")(aa,lb,ia,jb) * t1_2p.at("bb")(bb,lb) )
    .deallocate(tmps.at("0047_abab_vooo"))
    .allocate(tmps.at("0048_bbaa_vvov"))
    
    // flops: o1v3  = o1v3Q1
    //  mems: o1v3  = o1v3
    ( tmps.at("0048_bbaa_vvov")(bb,cb,ka,da)  = chol.at("bb_vvQ")(bb,cb,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    
    // r2_2p[abab] += -1.000 <k,b||d,c>_abab t1_aa(a,k) t2_2p_abab(d,c,i,j) 
    //               += -1.000 <k,b||c,d>_abab t1_aa(a,k) t2_2p_abab(c,d,i,j) 
    // flops: o2v2 += o3v3 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0048_bbaa_vvov")(bb,cb,ka,da) * t2_2p.at("abab")(da,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -2.000 <k,b||c,d>_abab t1_2p_aa(c,i) t2_abab(a,d,k,j) 
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_bbaa_vvoo")(bb,db,ia,ka)  = tmps.at("0048_bbaa_vvov")(bb,db,ka,ca) * t1_2p.at("aa")(ca,ia) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2.at("abab")(aa,db,ka,jb) * tmps.at("bin1_bbaa_vvoo")(bb,db,ia,ka) )
    .allocate(tmps.at("0049_aabb_ovoo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0049_aabb_ovoo")(ia,aa,jb,kb)  = chol.at("aa_ovQ")(ia,aa,Q) * chol.at("bb_ooQ")(jb,kb,Q) )
    
    // r2[abab] += -1.000 <a,k||i,j>_abab t1_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t1.at("bb")(bb,kb) * tmps.at("0049_aabb_ovoo")(ia,aa,jb,kb) )
    
    // r2_1p[abab] += -1.000 <a,k||i,j>_abab t1_1p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("bb")(bb,kb) * tmps.at("0049_aabb_ovoo")(ia,aa,jb,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||i,j>_abab t1_2p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_2p.at("bb")(bb,kb) * tmps.at("0049_aabb_ovoo")(ia,aa,jb,kb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,j>_abab t1_2p_aa(c,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = t1_2p.at("aa")(ca,ka) * tmps.at("0049_aabb_ovoo")(ka,ca,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2.at("abab")(aa,bb,ia,lb) * tmps.at("bin1_bb_oo")(jb,lb) )
    
    // r2_2p[abab] += +2.000 <k,l||c,j>_abab t1_aa(a,k) t2_2p_abab(c,b,i,l) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0049_aabb_ovoo")(ka,ca,lb,jb) * t2_2p.at("abab")(ca,bb,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    .allocate(tmps.at("0050_baab_vooo"))
    
    // flops: o3v1  = o4v2 o3v3 o3v1 o3v2 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("0050_baab_vooo")(bb,ka,ia,jb)  = -1.000 * t2.at("abab")(ca,bb,ia,lb) * tmps.at("0049_aabb_ovoo")(ka,ca,lb,jb) )
    ( tmps.at("0050_baab_vooo")(bb,ka,ia,jb) += tmps.at("0048_bbaa_vvov")(bb,cb,ka,da) * t2.at("abab")(da,cb,ia,jb) )
    ( tmps.at("0050_baab_vooo")(bb,ka,ia,jb) += f.at("aa_ov")(ka,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2[abab] += -1.000 f_aa(k,c) t1_aa(a,k) t2_abab(c,b,i,j) 
    //            += +1.000 <k,l||c,j>_abab t1_aa(a,k) t2_abab(c,b,i,l) 
    //            += -0.500 <k,b||d,c>_abab t1_aa(a,k) t2_abab(d,c,i,j) 
    //            += -0.500 <k,b||c,d>_abab t1_aa(a,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0050_baab_vooo")(bb,ka,ia,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += -1.000 f_aa(k,c) t1_1p_aa(a,k) t2_abab(c,b,i,j) 
    //               += +1.000 <k,l||c,j>_abab t1_1p_aa(a,k) t2_abab(c,b,i,l) 
    //               += -0.500 <k,b||d,c>_abab t1_1p_aa(a,k) t2_abab(d,c,i,j) 
    //               += -0.500 <k,b||c,d>_abab t1_1p_aa(a,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0050_baab_vooo")(bb,ka,ia,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 f_aa(k,c) t1_2p_aa(a,k) t2_abab(c,b,i,j) 
    //               += +2.000 <k,l||c,j>_abab t1_2p_aa(a,k) t2_abab(c,b,i,l) 
    //               += -1.000 <k,b||d,c>_abab t1_2p_aa(a,k) t2_abab(d,c,i,j) 
    //               += -1.000 <k,b||c,d>_abab t1_2p_aa(a,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0050_baab_vooo")(bb,ka,ia,jb) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0050_baab_vooo"))
    .allocate(tmps.at("0051_abab_vooo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v1
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1 o0v0Q1 o1v1 o3v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("0051_abab_vooo")(aa,lb,ia,jb)  = tmps.at("bin1_bb_vo")(db,lb) * t2_1p.at("abab")(aa,db,ia,jb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("0051_abab_vooo")(aa,lb,ia,jb) += tmps.at("bin1_bb_vo")(db,lb) * t2_1p.at("abab")(aa,db,ia,jb) )
    
    // r2_1p[abab] += -1.000 <k,l||c,d>_abab t1_bb(b,l) t1_aa(c,k) t2_1p_abab(a,d,i,j) 
    //               += +1.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0051_abab_vooo")(aa,lb,ia,jb) * t1.at("bb")(bb,lb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_abab t1_1p_bb(b,l) t1_aa(c,k) t2_1p_abab(a,d,i,j) 
    //               += +2.000 <l,k||c,d>_bbbb t1_1p_bb(b,l) t1_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0051_abab_vooo")(aa,lb,ia,jb) * t1_1p.at("bb")(bb,lb) )
    .deallocate(tmps.at("0051_abab_vooo"))
    .allocate(tmps.at("0052_bb_vv"))
    
    // flops: o0v2  = o0v2Q1 o0v2Q1 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0052_bb_vv")(bb,db)  = chol.at("bb_vvQ")(bb,db,Q) * tmps.at("0032_Q")(Q) )
    ( tmps.at("0052_bb_vv")(bb,db) += chol.at("bb_vvQ")(bb,db,Q) * tmps.at("0033_Q")(Q) )
    
    // r2_1p[abab] += +1.000 <b,k||d,c>_bbbb t1_1p_bb(c,k) t2_abab(a,d,i,j) 
    //               += +1.000 <k,b||c,d>_abab t1_1p_aa(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0052_bb_vv")(bb,db) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_2p[abab] += -2.000 <b,k||c,d>_bbbb t1_1p_bb(c,k) t2_1p_abab(a,d,i,j) 
    //               += +2.000 <k,b||c,d>_abab t1_1p_aa(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0052_bb_vv")(bb,db) * t2_1p.at("abab")(aa,db,ia,jb) )
    .deallocate(tmps.at("0052_bb_vv"))
    .allocate(tmps.at("0053_aabb_ovov"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0053_aabb_ovov")(ia,aa,jb,bb)  = chol.at("aa_ovQ")(ia,aa,Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    
    // r2[abab] += +1.000 <a,b||i,j>_abab 
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0053_aabb_ovov")(ia,aa,jb,bb) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_abab t2_abab(a,b,i,k) t2_2p_abab(d,c,l,j) 
    //               += -1.000 <l,k||c,d>_abab t2_abab(a,b,i,k) t2_2p_abab(c,d,l,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,kb)  = t2_2p.at("abab")(da,cb,la,jb) * tmps.at("0053_aabb_ovov")(la,da,kb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2.at("abab")(aa,bb,ia,kb) * tmps.at("bin1_bb_oo")(jb,kb) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t1_aa(a,k) t1_2p_aa(c,i) t2_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin2_baab_vooo")(db,ia,ka,lb)  = tmps.at("0053_aabb_ovov")(ka,ca,lb,db) * t1_2p.at("aa")(ca,ia) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin2_baab_vooo")(db,ia,ka,lb) * t2.at("bbbb")(db,bb,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_2p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin2_baab_vooo")(db,ia,ka,lb)  = t1.at("aa")(ca,ia) * tmps.at("0053_aabb_ovov")(ka,ca,lb,db) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin2_baab_vooo")(db,ia,ka,lb) * t2_2p.at("bbbb")(db,bb,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t1_aa(a,k) t1_1p_aa(c,i) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin2_baab_vooo")(db,ia,ka,lb)  = t1_1p.at("aa")(ca,ia) * tmps.at("0053_aabb_ovov")(ka,ca,lb,db) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin2_baab_vooo")(db,ia,ka,lb) * t2_1p.at("bbbb")(db,bb,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_abab(a,c,i,j) t2_2p_abab(d,b,k,l) 
    //               += -1.000 <l,k||d,c>_abab t2_abab(a,c,i,j) t2_2p_abab(d,b,l,k) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,cb)  = t2_2p.at("abab")(da,bb,ka,lb) * tmps.at("0053_aabb_ovov")(ka,da,lb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    .allocate(tmps.at("0054_abab_ovoo"))
    
    // flops: o3v1  = o3v2 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_baab_vooo")(db,ia,ka,lb)  = tmps.at("0053_aabb_ovov")(ka,ca,lb,db) * t1.at("aa")(ca,ia) )
    ( tmps.at("0054_abab_ovoo")(ka,bb,ia,jb)  = tmps.at("bin1_baab_vooo")(db,ia,ka,lb) * t2.at("bbbb")(db,bb,jb,lb) )
    
    // r2[abab] += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0054_abab_ovoo")(ka,bb,ia,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_abab t1_1p_aa(a,k) t1_aa(c,i) t2_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0054_abab_ovoo")(ka,bb,ia,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t1_2p_aa(a,k) t1_aa(c,i) t2_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0054_abab_ovoo")(ka,bb,ia,jb) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0054_abab_ovoo"))
    .allocate(tmps.at("0055_bbbb_ovov"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0055_bbbb_ovov")(lb,cb,kb,db)  = chol.at("bb_ovQ")(lb,cb,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    
    // r2_1p[abab] += +0.500 <l,k||d,c>_bbbb t2_abab(a,b,i,k) t2_1p_bbbb(d,c,j,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,kb)  = tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) * t2_1p.at("bbbb")(db,cb,jb,lb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.500 * t2.at("abab")(aa,bb,ia,kb) * tmps.at("bin1_bb_oo")(jb,kb) )
    
    // r2_1p[abab] += +0.500 <l,k||d,c>_bbbb t2_abab(a,b,i,k) t2_1p_bbbb(d,c,j,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,kb)  = tmps.at("0055_bbbb_ovov")(kb,db,lb,cb) * t2_1p.at("bbbb")(db,cb,jb,lb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.500 * t2.at("abab")(aa,bb,ia,kb) * tmps.at("bin1_bb_oo")(jb,kb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t2_1p_abab(a,d,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_abab_vvoo")(aa,cb,ia,kb)  = tmps.at("0055_bbbb_ovov")(lb,cb,kb,db) * t2_1p.at("abab")(aa,db,ia,lb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin1_abab_vvoo")(aa,cb,ia,kb) * t2.at("bbbb")(cb,bb,jb,kb) )
    
    // r2_2p[abab] += +1.000 <l,k||d,c>_bbbb t2_abab(a,b,i,k) t2_2p_bbbb(d,c,j,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,kb)  = tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) * t2_2p.at("bbbb")(db,cb,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ia,kb) * tmps.at("bin1_bb_oo")(jb,kb) )
    
    // r2_2p[abab] += +1.000 <l,k||d,c>_bbbb t2_abab(a,b,i,k) t2_2p_bbbb(d,c,j,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,kb)  = tmps.at("0055_bbbb_ovov")(kb,db,lb,cb) * t2_2p.at("bbbb")(db,cb,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,kb) * tmps.at("bin1_bb_oo")(jb,kb) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_bbbb t2_1p_abab(a,b,i,l) t2_1p_bbbb(d,c,j,k) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = t2_1p.at("bbbb")(db,cb,jb,kb) * tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("bin1_bb_oo")(jb,lb) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_bbbb t2_1p_abab(a,b,i,l) t2_1p_bbbb(d,c,j,k) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = t2_1p.at("bbbb")(db,cb,jb,kb) * tmps.at("0055_bbbb_ovov")(kb,db,lb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("bin1_bb_oo")(jb,lb) )
    
    // r2_2p[abab] += +1.000 <k,l||c,d>_bbbb t2_abab(a,c,i,j) t2_2p_bbbb(d,b,k,l) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,cb)  = t2_2p.at("bbbb")(db,bb,kb,lb) * tmps.at("0055_bbbb_ovov")(lb,db,kb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,cb,ia,jb) * tmps.at("bin1_bb_vv")(bb,cb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_2p_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb)  = tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) * t1_2p.at("bb")(cb,jb) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb) * t2.at("abab")(aa,db,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_2p_abab(a,d,i,l) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb)  = t1.at("bb")(cb,jb) * tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb) * t2_2p.at("abab")(aa,db,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_1p_bb(c,j) t2_1p_abab(a,d,i,l) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb)  = t1_1p.at("bb")(cb,jb) * tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb) * t2_1p.at("abab")(aa,db,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +1.000 <k,l||c,d>_bbbb t2_abab(a,c,i,j) t2_2p_bbbb(d,b,k,l) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,cb)  = tmps.at("0055_bbbb_ovov")(lb,cb,kb,db) * t2_2p.at("bbbb")(db,bb,kb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin1_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t2_2p_abab(a,d,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_abab_vvoo")(aa,cb,ia,kb)  = tmps.at("0055_bbbb_ovov")(lb,cb,kb,db) * t2_2p.at("abab")(aa,db,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_abab_vvoo")(aa,cb,ia,kb) * t2.at("bbbb")(cb,bb,jb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,lb)  = t1.at("bb")(cb,kb) * tmps.at("0055_bbbb_ovov")(kb,db,lb,cb) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("bin1_bb_vo")(db,lb) * t2_2p.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,lb) * t1.at("bb")(bb,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_2p_abab(a,d,i,l) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb)  = tmps.at("0055_bbbb_ovov")(kb,db,lb,cb) * t1.at("bb")(cb,jb) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb) * t2_2p.at("abab")(aa,db,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0056_baba_ovoo"))
    
    // flops: o3v1  = o3v2 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb)  = tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) * t1.at("bb")(cb,jb) )
    ( tmps.at("0056_baba_ovoo")(kb,aa,jb,ia)  = tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb) * t2.at("abab")(aa,db,ia,lb) )
    
    // r2[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0056_baba_ovoo")(kb,aa,jb,ia) * t1.at("bb")(bb,kb) )
    
    // r2_1p[abab] += -1.000 <k,l||c,d>_bbbb t1_1p_bb(b,k) t1_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0056_baba_ovoo")(kb,aa,jb,ia) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_bbbb t1_2p_bb(b,k) t1_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0056_baba_ovoo")(kb,aa,jb,ia) * t1_2p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0056_baba_ovoo"))
    .allocate(tmps.at("0057_abba_ovoo"))
    
    // flops: o3v1  = o3v2 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aabb_vooo")(da,ka,jb,lb)  = tmps.at("0053_aabb_ovov")(ka,da,lb,cb) * t1.at("bb")(cb,jb) )
    ( tmps.at("0057_abba_ovoo")(ka,bb,jb,ia)  = tmps.at("bin1_aabb_vooo")(da,ka,jb,lb) * t2.at("abab")(da,bb,ia,lb) )
    
    // r2[abab] += +1.000 <k,l||d,c>_abab t1_aa(a,k) t1_bb(c,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0057_abba_ovoo")(ka,bb,jb,ia) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += +1.000 <k,l||d,c>_abab t1_1p_aa(a,k) t1_bb(c,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0057_abba_ovoo")(ka,bb,jb,ia) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_abab t1_2p_aa(a,k) t1_bb(c,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0057_abba_ovoo")(ka,bb,jb,ia) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0057_abba_ovoo"))
    .allocate(tmps.at("0058_aaaa_ovov"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0058_aaaa_ovov")(la,ca,ka,da)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    
    // r1[aa] += +0.500 <k,j||c,b>_aaaa t1_aa(a,j) t2_aaaa(c,b,i,k) 
    // flops: o1v1 += o3v2 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_aa_oo")(ia,ja)  = tmps.at("0058_aaaa_ovov")(ja,ba,ka,ca) * t2.at("aaaa")(ca,ba,ia,ka) )
    ( r1.at("aa")(aa,ia) += 0.500 * tmps.at("bin1_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r1[aa] += +0.500 <k,j||c,b>_aaaa t1_aa(a,j) t2_aaaa(c,b,i,k) 
    // flops: o1v1 += o3v2 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_aa_oo")(ia,ja)  = tmps.at("0058_aaaa_ovov")(ja,ca,ka,ba) * t2.at("aaaa")(ca,ba,ia,ka) )
    ( r1.at("aa")(aa,ia) -= 0.500 * tmps.at("bin1_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_aa(a,j) t1_aa(b,i) t1_2p_aa(c,k) 
    // flops: o1v1 += o2v2 o2v1 o2v1
    //  mems: o1v1 += o1v1 o2v0 o1v1
    ( tmps.at("bin1_aa_vo")(ba,ja)  = tmps.at("0058_aaaa_ovov")(ka,ba,ja,ca) * t1_2p.at("aa")(ca,ka) )
    ( tmps.at("bin1_aa_oo")(ia,ja)  = t1.at("aa")(ba,ia) * tmps.at("bin1_aa_vo")(ba,ja) )
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("bin1_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t2_1p_aaaa(d,a,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_aaaa_vvoo")(aa,ca,ia,ka)  = tmps.at("0058_aaaa_ovov")(la,ca,ka,da) * t2_1p.at("aaaa")(da,aa,ia,la) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin1_aaaa_vvoo")(aa,ca,ia,ka) * t2.at("abab")(ca,bb,ka,jb) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_aaaa t2_1p_abab(a,b,l,j) t2_1p_aaaa(d,c,i,k) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_aa_oo")(ia,la)  = t2_1p.at("aaaa")(da,ca,ia,ka) * tmps.at("0058_aaaa_ovov")(ka,ca,la,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,bb,la,jb) * tmps.at("bin1_aa_oo")(ia,la) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_aaaa t2_1p_abab(a,b,l,j) t2_1p_aaaa(d,c,i,k) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_aa_oo")(ia,la)  = t2_1p.at("aaaa")(da,ca,ia,ka) * tmps.at("0058_aaaa_ovov")(ka,da,la,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,la,jb) * tmps.at("bin1_aa_oo")(ia,la) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_aaaa t1_1p_aa(a,l) t1_1p_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,la)  = t1_1p.at("aa")(ca,ka) * tmps.at("0058_aaaa_ovov")(ka,da,la,ca) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin1_aa_vo")(da,la) * t2.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_1p.at("aa")(aa,la) * tmps.at("bin1_baab_vooo")(bb,ia,la,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_1p_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_aaaa_vvoo")(aa,da,ia,la)  = tmps.at("0058_aaaa_ovov")(la,ca,ka,da) * t2_1p.at("aaaa")(ca,aa,ia,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_1p.at("abab")(da,bb,la,jb) * tmps.at("bin1_aaaa_vvoo")(aa,da,ia,la) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,ka)  = t1_2p.at("aa")(ca,la) * tmps.at("0058_aaaa_ovov")(ka,ca,la,da) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +1.000 <k,l||c,d>_aaaa t2_2p_aaaa(d,a,k,l) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_aa_vv")(aa,ca)  = tmps.at("0058_aaaa_ovov")(la,ca,ka,da) * t2_2p.at("aaaa")(da,aa,ka,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin1_aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_2p_aaaa(d,a,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_aaaa_vvoo")(aa,ca,ia,ka)  = tmps.at("0058_aaaa_ovov")(la,ca,ka,da) * t2_2p.at("aaaa")(da,aa,ia,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aaaa_vvoo")(aa,ca,ia,ka) * t2.at("abab")(ca,bb,ka,jb) )
    .allocate(tmps.at("0059_abab_ovoo"))
    
    // flops: o3v1  = o3v2 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aaaa_vooo")(da,ia,ka,la)  = t1.at("aa")(ca,ia) * tmps.at("0058_aaaa_ovov")(la,ca,ka,da) )
    ( tmps.at("0059_abab_ovoo")(ia,bb,ka,jb)  = tmps.at("bin1_aaaa_vooo")(da,ia,ka,la) * t2.at("abab")(da,bb,la,jb) )
    
    // r2_1p[abab] += -1.000 <k,l||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0059_abab_ovoo")(ia,bb,ka,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_aaaa t1_2p_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0059_abab_ovoo")(ia,bb,ka,jb) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0059_abab_ovoo"))
    .allocate(tmps.at("0060_baab_vooo"))
    
    // flops: o3v1  = o3v2 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aaaa_vooo")(da,ia,ka,la)  = t1.at("aa")(ca,ia) * tmps.at("0058_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("0060_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin1_aaaa_vooo")(da,ia,ka,la) * t2_1p.at("abab")(da,bb,la,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0060_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0060_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0060_baab_vooo"))
    .allocate(tmps.at("0061_aaaa_ovoo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0061_aaaa_ovoo")(la,da,ia,ka)  = t1.at("aa")(ca,ia) * tmps.at("0058_aaaa_ovov")(la,da,ka,ca) )
    
    // r1[aa] += +0.500 <j,k||b,c>_aaaa t1_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= 0.500 * tmps.at("0061_aaaa_ovoo")(ja,ca,ia,ka) * t2.at("aaaa")(ca,aa,ja,ka) )
    
    // r1_1p[aa] += +0.500 <j,k||b,c>_aaaa t1_aa(b,i) t2_1p_aaaa(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.500 * tmps.at("0061_aaaa_ovoo")(ja,ca,ia,ka) * t2_1p.at("aaaa")(ca,aa,ja,ka) )
    
    // r1_2p[aa] += +1.000 <j,k||b,c>_aaaa t1_aa(b,i) t2_2p_aaaa(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= tmps.at("0061_aaaa_ovoo")(ja,ca,ia,ka) * t2_2p.at("aaaa")(ca,aa,ja,ka) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0061_aaaa_ovoo")(la,da,ia,ka) * t2.at("abab")(da,bb,la,jb) )
    ( r2.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,la,jb) * tmps.at("0061_aaaa_ovoo")(ka,da,ia,la) )
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_aaaa t1_aa(c,i) t1_2p_aa(d,k) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_aa_oo")(ia,la)  = tmps.at("0061_aaaa_ovoo")(la,da,ia,ka) * t1_2p.at("aa")(da,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,bb,la,jb) * tmps.at("bin1_aa_oo")(ia,la) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0061_aaaa_ovoo")(la,da,ia,ka) * t2_2p.at("abab")(da,bb,la,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(da,bb,la,jb) * tmps.at("0061_aaaa_ovoo")(ka,da,ia,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0062_abab_ovoo"))
    
    // flops: o3v1  = o3v2 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aaaa_vooo")(da,ia,ka,la)  = tmps.at("0058_aaaa_ovov")(ka,ca,la,da) * t1.at("aa")(ca,ia) )
    ( tmps.at("0062_abab_ovoo")(ka,bb,ia,jb)  = tmps.at("bin1_aaaa_vooo")(da,ia,ka,la) * t2.at("abab")(da,bb,la,jb) )
    
    // r2_1p[abab] += -1.000 <k,l||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0062_abab_ovoo")(ka,bb,ia,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_aaaa t1_2p_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0062_abab_ovoo")(ka,bb,ia,jb) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0062_abab_ovoo"))
    .allocate(tmps.at("0063_aabb_vvoo"))
    
    // flops: o2v2  = o3v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0063_aabb_vvoo")(ca,aa,kb,jb)  = tmps.at("0053_aabb_ovov")(la,ca,kb,db) * t2_2p.at("abab")(aa,db,la,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_abab t2_2p_abab(a,d,l,j) t2_abab(c,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(ca,bb,ia,kb) * tmps.at("0063_aabb_vvoo")(ca,aa,kb,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t2_2p_abab(a,d,l,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t1.at("aa")(ca,ia) * tmps.at("0063_aabb_vvoo")(ca,aa,kb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .deallocate(tmps.at("0063_aabb_vvoo"))
    .allocate(tmps.at("0064_bbaa_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0064_bbaa_vvoo")(bb,db,ka,ia)  = tmps.at("0048_bbaa_vvov")(bb,db,ka,ca) * t1.at("aa")(ca,ia) )
    
    // r2[abab] += -1.000 <k,b||c,d>_abab t1_aa(c,i) t2_abab(a,d,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0064_bbaa_vvoo")(bb,db,ka,ia) * t2.at("abab")(aa,db,ka,jb) )
    
    // r2_1p[abab] += -1.000 <k,b||c,d>_abab t1_aa(c,i) t2_1p_abab(a,d,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0064_bbaa_vvoo")(bb,db,ka,ia) * t2_1p.at("abab")(aa,db,ka,jb) )
    
    // r2_2p[abab] += -2.000 <k,b||c,d>_abab t1_aa(c,i) t2_2p_abab(a,d,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0064_bbaa_vvoo")(bb,db,ka,ia) * t2_2p.at("abab")(aa,db,ka,jb) )
    .deallocate(tmps.at("0064_bbaa_vvoo"))
    .allocate(tmps.at("0065_aabb_vvov"))
    
    // flops: o1v3  = o1v3Q1
    //  mems: o1v3  = o1v3
    ( tmps.at("0065_aabb_vvov")(aa,da,kb,cb)  = chol.at("aa_vvQ")(aa,da,Q) * chol.at("bb_ovQ")(kb,cb,Q) )
    
    // r1[aa] += +0.500 <a,j||c,b>_abab t2_abab(c,b,i,j) 
    //          += +0.500 <a,j||b,c>_abab t2_abab(b,c,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0065_aabb_vvov")(aa,ca,jb,bb) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r1_1p[aa] += +0.500 <a,j||c,b>_abab t2_1p_abab(c,b,i,j) 
    //             += +0.500 <a,j||b,c>_abab t2_1p_abab(b,c,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0065_aabb_vvov")(aa,ca,jb,bb) * t2_1p.at("abab")(ca,bb,ia,jb) )
    
    // r1_2p[aa] += +1.000 <a,j||c,b>_abab t2_2p_abab(c,b,i,j) 
    //             += +1.000 <a,j||b,c>_abab t2_2p_abab(b,c,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0065_aabb_vvov")(aa,ca,jb,bb) * t2_2p.at("abab")(ca,bb,ia,jb) )
    
    // r2[abab] += +1.000 <a,b||c,j>_abab t1_aa(c,i) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(ca,ia) * tmps.at("0065_aabb_vvov")(ca,aa,jb,bb) )
    
    // r2_1p[abab] += +1.000 <a,b||c,j>_abab t1_1p_aa(c,i) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(ca,ia) * tmps.at("0065_aabb_vvov")(ca,aa,jb,bb) )
    
    // r2_2p[abab] += +2.000 <a,b||c,j>_abab t1_2p_aa(c,i) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_2p.at("aa")(ca,ia) * tmps.at("0065_aabb_vvov")(ca,aa,jb,bb) )
    
    // r2_2p[abab] += -2.000 <a,k||d,c>_abab t1_2p_bb(c,j) t2_abab(d,b,i,k) 
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_aabb_vvoo")(aa,da,jb,kb)  = tmps.at("0065_aabb_vvov")(aa,da,kb,cb) * t1_2p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2.at("abab")(da,bb,ia,kb) * tmps.at("bin1_aabb_vvoo")(aa,da,jb,kb) )
    
    // r2_2p[abab] += -1.000 <a,k||d,c>_abab t1_bb(b,k) t2_2p_abab(d,c,i,j) 
    //               += -1.000 <a,k||c,d>_abab t1_bb(b,k) t2_2p_abab(c,d,i,j) 
    // flops: o2v2 += o3v3 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0065_aabb_vvov")(aa,da,kb,cb) * t2_2p.at("abab")(da,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0066_aabb_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0066_aabb_vvoo")(aa,da,kb,jb)  = tmps.at("0065_aabb_vvov")(aa,da,kb,cb) * t1.at("bb")(cb,jb) )
    
    // r2[abab] += -1.000 <a,k||d,c>_abab t1_bb(c,j) t2_abab(d,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0066_aabb_vvoo")(aa,da,kb,jb) * t2.at("abab")(da,bb,ia,kb) )
    
    // r2_1p[abab] += -1.000 <a,k||d,c>_abab t1_bb(c,j) t2_1p_abab(d,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0066_aabb_vvoo")(aa,da,kb,jb) * t2_1p.at("abab")(da,bb,ia,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||d,c>_abab t1_bb(c,j) t2_2p_abab(d,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0066_aabb_vvoo")(aa,da,kb,jb) * t2_2p.at("abab")(da,bb,ia,kb) )
    .deallocate(tmps.at("0066_aabb_vvoo"))
    .allocate(tmps.at("0067_bb_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0067_bb_vv")(cb,bb)  = tmps.at("0055_bbbb_ovov")(lb,cb,kb,db) * t2_1p.at("bbbb")(db,bb,kb,lb) )
    
    // r2_1p[abab] += +0.500 <k,l||c,d>_bbbb t2_abab(a,c,i,j) t2_1p_bbbb(d,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.500 * t2.at("abab")(aa,cb,ia,jb) * tmps.at("0067_bb_vv")(cb,bb) )
    
    // r2_2p[abab] += +1.000 <k,l||d,c>_bbbb t2_1p_abab(a,d,i,j) t2_1p_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,db,ia,jb) * tmps.at("0067_bb_vv")(db,bb) )
    .deallocate(tmps.at("0067_bb_vv"))
    .allocate(tmps.at("0068_aa_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0068_aa_vv")(ca,aa)  = tmps.at("0058_aaaa_ovov")(la,ca,ka,da) * t2_1p.at("aaaa")(da,aa,ka,la) )
    
    // r2_1p[abab] += +0.500 <k,l||c,d>_aaaa t2_1p_aaaa(d,a,k,l) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.500 * t2.at("abab")(ca,bb,ia,jb) * tmps.at("0068_aa_vv")(ca,aa) )
    
    // r2_2p[abab] += +1.000 <k,l||d,c>_aaaa t2_1p_aaaa(c,a,k,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("0068_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0068_aa_vv"))
    .allocate(tmps.at("0069_bbaa_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0069_bbaa_vvoo")(bb,db,ka,ia)  = tmps.at("0048_bbaa_vvov")(bb,db,ka,ca) * t1_1p.at("aa")(ca,ia) )
    
    // r2_1p[abab] += -1.000 <k,b||c,d>_abab t1_1p_aa(c,i) t2_abab(a,d,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0069_bbaa_vvoo")(bb,db,ka,ia) * t2.at("abab")(aa,db,ka,jb) )
    
    // r2_2p[abab] += -2.000 <k,b||c,d>_abab t1_1p_aa(c,i) t2_1p_abab(a,d,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0069_bbaa_vvoo")(bb,db,ka,ia) * t2_1p.at("abab")(aa,db,ka,jb) )
    .deallocate(tmps.at("0069_bbaa_vvoo"))
    .allocate(tmps.at("0070_aabb_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0070_aabb_vvoo")(aa,da,kb,jb)  = tmps.at("0065_aabb_vvov")(aa,da,kb,cb) * t1_1p.at("bb")(cb,jb) )
    
    // r2_1p[abab] += -1.000 <a,k||d,c>_abab t1_1p_bb(c,j) t2_abab(d,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0070_aabb_vvoo")(aa,da,kb,jb) * t2.at("abab")(da,bb,ia,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||d,c>_abab t1_1p_bb(c,j) t2_1p_abab(d,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0070_aabb_vvoo")(aa,da,kb,jb) * t2_1p.at("abab")(da,bb,ia,kb) )
    .deallocate(tmps.at("0070_aabb_vvoo"))
    .allocate(tmps.at("0071_aa_oo"))
    
    // flops: o2v0  = o3v2
    //  mems: o2v0  = o2v0
    ( tmps.at("0071_aa_oo")(ja,ia)  = tmps.at("0053_aabb_ovov")(ja,ca,kb,bb) * t2.at("abab")(ca,bb,ia,kb) )
    
    // r1[aa] += -0.500 <j,k||c,b>_abab t1_aa(a,j) t2_abab(c,b,i,k) 
    //          += -0.500 <j,k||b,c>_abab t1_aa(a,j) t2_abab(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0071_aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    
    // r1_1p[aa] += -0.500 <j,k||c,b>_abab t1_1p_aa(a,j) t2_abab(c,b,i,k) 
    //             += -0.500 <j,k||b,c>_abab t1_1p_aa(a,j) t2_abab(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0071_aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    
    // r1_2p[aa] += -1.000 <j,k||c,b>_abab t1_2p_aa(a,j) t2_abab(c,b,i,k) 
    //             += -1.000 <j,k||b,c>_abab t1_2p_aa(a,j) t2_abab(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0071_aa_oo")(ja,ia) * t1_2p.at("aa")(aa,ja) )
    
    // r2[abab] += -0.500 <l,k||d,c>_abab t2_abab(a,b,l,j) t2_abab(d,c,i,k) 
    //            += -0.500 <l,k||c,d>_abab t2_abab(a,b,l,j) t2_abab(c,d,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0071_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_1p[abab] += -0.500 <l,k||d,c>_abab t2_1p_abab(a,b,l,j) t2_abab(d,c,i,k) 
    //               += -0.500 <l,k||c,d>_abab t2_1p_abab(a,b,l,j) t2_abab(c,d,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0071_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_abab t2_2p_abab(a,b,l,j) t2_abab(d,c,i,k) 
    //               += -1.000 <l,k||c,d>_abab t2_2p_abab(a,b,l,j) t2_abab(c,d,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0071_aa_oo")(la,ia) * t2_2p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0071_aa_oo"))
    .allocate(tmps.at("0072_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0072_aa_voQ")(ba,ia,Q)  = chol.at("bb_ovQ")(jb,ab,Q) * t2_1p.at("abab")(ba,ab,ia,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t2_1p_abab(a,d,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0072_aa_voQ")(aa,ia,Q) * tmps.at("0026_bb_voQ")(bb,jb,Q) )
    
    // r1_2p[aa] += -2.000 <k,j||b,c>_bbbb t1_1p_bb(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0072_aa_voQ")(aa,ia,Q) * tmps.at("0032_Q")(Q) )
    
    // r1_1p[aa] += +1.000 <j,k||b,c>_abab t1_aa(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0072_aa_voQ")(aa,ia,Q) * tmps.at("0030_Q")(Q) )
    
    // r1_1p[aa] += -1.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0072_aa_voQ")(aa,ia,Q) * tmps.at("0029_Q")(Q) )
    
    // r1_2p[aa] += +2.000 <j,k||b,c>_abab t1_1p_aa(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0072_aa_voQ")(aa,ia,Q) * tmps.at("0033_Q")(Q) )
    
    // r1_1p[aa] += -0.500 <j,k||i,b>_abab t2_1p_abab(a,b,j,k) 
    //             += -0.500 <k,j||i,b>_abab t2_1p_abab(a,b,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0072_aa_voQ")(aa,ja,Q) )
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_abab t2_1p_abab(a,d,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0072_aa_voQ")(aa,ia,Q) * tmps.at("0027_bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += +1.000 <b,k||j,c>_bbbb t2_1p_abab(a,c,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0072_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0073_bbaa_oovo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0073_bbaa_oovo")(kb,jb,aa,ia)  = chol.at("bb_ooQ")(kb,jb,Q) * tmps.at("0072_aa_voQ")(aa,ia,Q) )
    
    // r2_1p[abab] += +1.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_1p_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0073_bbaa_oovo")(kb,jb,aa,ia) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||j,c>_bbbb t1_1p_bb(b,k) t2_1p_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0073_bbaa_oovo")(kb,jb,aa,ia) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0073_bbaa_oovo"))
    .allocate(tmps.at("0074_aa_oo"))
    
    // flops: o2v0  = o3v2
    //  mems: o2v0  = o2v0
    ( tmps.at("0074_aa_oo")(ja,ia)  = tmps.at("0053_aabb_ovov")(ja,ca,kb,bb) * t2_1p.at("abab")(ca,bb,ia,kb) )
    
    // r1_1p[aa] += -0.500 <j,k||c,b>_abab t1_aa(a,j) t2_1p_abab(c,b,i,k) 
    //             += -0.500 <j,k||b,c>_abab t1_aa(a,j) t2_1p_abab(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0074_aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    
    // r1_2p[aa] += -1.000 <j,k||c,b>_abab t1_1p_aa(a,j) t2_1p_abab(c,b,i,k) 
    //             += -1.000 <j,k||b,c>_abab t1_1p_aa(a,j) t2_1p_abab(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0074_aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    
    // r2_1p[abab] += -0.500 <k,l||d,c>_abab t2_abab(a,b,k,j) t2_1p_abab(d,c,i,l) 
    //               += -0.500 <k,l||c,d>_abab t2_abab(a,b,k,j) t2_1p_abab(c,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0074_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_abab t2_1p_abab(a,b,l,j) t2_1p_abab(d,c,i,k) 
    //               += -1.000 <l,k||c,d>_abab t2_1p_abab(a,b,l,j) t2_1p_abab(c,d,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0074_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0074_aa_oo"))
    .allocate(tmps.at("0075_aabb_vooo"))
    
    // flops: o3v1  = o2v2 o3v2
    //  mems: o3v1  = o1v1 o3v1
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("0055_bbbb_ovov")(kb,db,lb,cb) * t1.at("bb")(cb,kb) )
    ( tmps.at("0075_aabb_vooo")(aa,ia,jb,lb)  = t2.at("abab")(aa,db,ia,jb) * tmps.at("bin1_bb_vo")(db,lb) )
    
    // r2[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,lb) * tmps.at("0075_aabb_vooo")(aa,ia,jb,lb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t1_1p_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0075_aabb_vooo")(aa,ia,jb,lb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_2p_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0075_aabb_vooo")(aa,ia,jb,lb) * t1_2p.at("bb")(bb,lb) )
    .deallocate(tmps.at("0075_aabb_vooo"))
    .allocate(tmps.at("0076_aaaa_ovoo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0076_aaaa_ovoo")(la,da,ia,ka)  = t1_2p.at("aa")(ca,ia) * tmps.at("0058_aaaa_ovov")(la,da,ka,ca) )
    
    // r1_2p[aa] += -1.000 <j,k||c,b>_aaaa t1_2p_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= tmps.at("0076_aaaa_ovoo")(ja,ca,ia,ka) * t2.at("aaaa")(ca,aa,ja,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_2p_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_aa_oo")(ia,la)  = tmps.at("0076_aaaa_ovoo")(la,ca,ia,ka) * t1.at("aa")(ca,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,bb,la,jb) * tmps.at("bin1_aa_oo")(ia,la) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0076_aaaa_ovoo")(la,da,ia,ka) * t2.at("abab")(da,bb,la,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,la,jb) * tmps.at("0076_aaaa_ovoo")(ka,da,ia,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0076_aaaa_ovoo"))
    .allocate(tmps.at("0077_aabb_vooo"))
    
    // flops: o3v1  = o2v2 o3v2
    //  mems: o3v1  = o1v1 o3v1
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("0055_bbbb_ovov")(kb,db,lb,cb) * t1.at("bb")(cb,kb) )
    ( tmps.at("0077_aabb_vooo")(aa,ia,jb,lb)  = t2_1p.at("abab")(aa,db,ia,jb) * tmps.at("bin1_bb_vo")(db,lb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,lb) * tmps.at("0077_aabb_vooo")(aa,ia,jb,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_1p_bb(b,l) t1_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0077_aabb_vooo")(aa,ia,jb,lb) * t1_1p.at("bb")(bb,lb) )
    .deallocate(tmps.at("0077_aabb_vooo"))
    .allocate(tmps.at("0078_bb_oo"))
    
    // flops: o2v0  = o3v2
    //  mems: o2v0  = o2v0
    ( tmps.at("0078_bb_oo")(jb,lb)  = t2_1p.at("abab")(da,cb,ka,jb) * tmps.at("0053_aabb_ovov")(ka,da,lb,cb) )
    
    // r2_1p[abab] += -0.500 <l,k||d,c>_abab t2_abab(a,b,i,k) t2_1p_abab(d,c,l,j) 
    //               += -0.500 <l,k||c,d>_abab t2_abab(a,b,i,k) t2_1p_abab(c,d,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0078_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_1p_abab(a,b,i,l) t2_1p_abab(d,c,k,j) 
    //               += -1.000 <k,l||c,d>_abab t2_1p_abab(a,b,i,l) t2_1p_abab(c,d,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0078_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    .deallocate(tmps.at("0078_bb_oo"))
    .allocate(tmps.at("0079_bb_voQ"))
    
    // flops: o1v1Q1  = o1v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0079_bb_voQ")(bb,jb,Q)  = chol.at("bb_vvQ")(bb,cb,Q) * t1_1p.at("bb")(cb,jb) )
    
    // r2_1p[abab] += +1.000 <a,b||i,c>_abab t1_1p_bb(c,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0079_bb_voQ")(bb,jb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[abab] += +2.000 <b,k||c,d>_bbbb t1_1p_bb(c,j) t2_1p_abab(a,d,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0072_aa_voQ")(aa,ia,Q) * tmps.at("0079_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0080_bb_vv"))
    
    // flops: o0v2  = o1v2Q1
    //  mems: o0v2  = o0v2
    ( tmps.at("0080_bb_vv")(db,bb)  = chol.at("bb_ovQ")(kb,db,Q) * tmps.at("0079_bb_voQ")(bb,kb,Q) )
    
    // r2_1p[abab] += +1.000 <b,k||d,c>_bbbb t1_1p_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,db,ia,jb) * tmps.at("0080_bb_vv")(db,bb) )
    
    // r2_2p[abab] += -2.000 <b,k||c,d>_bbbb t1_1p_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_1p.at("abab")(aa,db,ia,jb) * tmps.at("0080_bb_vv")(db,bb) )
    .deallocate(tmps.at("0080_bb_vv"))
    .allocate(tmps.at("0081_aa_voQ"))
    
    // flops: o1v1Q1  = o1v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0081_aa_voQ")(aa,ja,Q)  = chol.at("aa_vvQ")(aa,ba,Q) * t1_1p.at("aa")(ba,ja) )
    
    // r2_1p[abab] += -1.000 <a,k||c,d>_abab t1_1p_aa(c,i) t2_bbbb(d,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0081_aa_voQ")(aa,ia,Q) * tmps.at("0026_bb_voQ")(bb,jb,Q) )
    
    // r1_2p[aa] += +2.000 <a,j||c,b>_abab t1_1p_bb(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0081_aa_voQ")(aa,ia,Q) * tmps.at("0032_Q")(Q) )
    
    // r1_2p[aa] += -2.000 <a,j||b,c>_aaaa t1_1p_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0081_aa_voQ")(aa,ia,Q) * tmps.at("0033_Q")(Q) )
    
    // r2_1p[abab] += -1.000 <a,k||d,c>_aaaa t1_1p_aa(c,i) t2_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0081_aa_voQ")(aa,ia,Q) * tmps.at("0027_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <a,b||c,d>_abab t1_1p_aa(c,i) t1_1p_bb(d,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0081_aa_voQ")(aa,ia,Q) * tmps.at("0079_bb_voQ")(bb,jb,Q) )
    
    // r1_1p[aa] += -1.000 <a,j||b,c>_aaaa t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0081_aa_voQ")(aa,ia,Q) * tmps.at("0030_Q")(Q) )
    
    // r1_1p[aa] += +1.000 <a,j||c,b>_abab t1_bb(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0081_aa_voQ")(aa,ia,Q) * tmps.at("0029_Q")(Q) )
    .allocate(tmps.at("0082_aa_vv"))
    
    // flops: o0v2  = o1v2Q1
    //  mems: o0v2  = o0v2
    ( tmps.at("0082_aa_vv")(da,aa)  = chol.at("aa_ovQ")(ka,da,Q) * tmps.at("0081_aa_voQ")(aa,ka,Q) )
    
    // r2_1p[abab] += +1.000 <a,k||d,c>_aaaa t1_1p_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(da,bb,ia,jb) * tmps.at("0082_aa_vv")(da,aa) )
    
    // r2_2p[abab] += -2.000 <a,k||c,d>_aaaa t1_1p_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("0082_aa_vv")(da,aa) )
    .deallocate(tmps.at("0082_aa_vv"))
    .allocate(tmps.at("0083_aabb_ovoo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0083_aabb_ovoo")(la,da,jb,kb)  = t1_2p.at("bb")(cb,jb) * tmps.at("0053_aabb_ovov")(la,da,kb,cb) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_abab t1_bb(b,k) t1_2p_bb(c,j) t2_aaaa(d,a,i,l) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0083_aabb_ovoo")(la,da,jb,kb) * t2.at("aaaa")(da,aa,ia,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_abab t1_aa(a,k) t1_2p_bb(c,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,ia,lb) * tmps.at("0083_aabb_ovoo")(ka,da,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0083_aabb_ovoo"))
    .allocate(tmps.at("0084_aa_ov"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0084_aa_ov")(ka,ca)  = t1_1p.at("aa")(ba,ja) * tmps.at("0058_aaaa_ovov")(ka,ba,ja,ca) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_1p_aa(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t2_1p.at("aaaa")(ca,aa,ia,ka) * tmps.at("0084_aa_ov")(ka,ca) )
    
    // r2_1p[abab] += +1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0084_aa_ov")(ka,da) * t2.at("abab")(da,bb,ia,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0084_aa_ov")(ka,da) * t2_1p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    .deallocate(tmps.at("0084_aa_ov"))
    .allocate(tmps.at("0085_bb_ov"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0085_bb_ov")(kb,cb)  = t1_1p.at("bb")(bb,jb) * tmps.at("0055_bbbb_ovov")(kb,bb,jb,cb) )
    
    // r1_2p[aa] += -2.000 <k,j||b,c>_bbbb t1_1p_bb(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * t2_1p.at("abab")(aa,cb,ia,kb) * tmps.at("0085_bb_ov")(kb,cb) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_bbbb t1_bb(c,j) t1_1p_bb(d,k) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = tmps.at("0085_bb_ov")(lb,cb) * t1.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    .deallocate(tmps.at("0085_bb_ov"))
    .allocate(tmps.at("0086_bb_vo"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0086_bb_vo")(cb,lb)  = tmps.at("0055_bbbb_ovov")(kb,cb,lb,db) * t1_2p.at("bb")(db,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_2p_bb(c,l) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t2.at("abab")(aa,db,ia,jb) * tmps.at("0086_bb_vo")(db,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_bbbb t1_bb(c,j) t1_2p_bb(d,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = tmps.at("0086_bb_vo")(cb,lb) * t1.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    .deallocate(tmps.at("0086_bb_vo"))
    .allocate(tmps.at("0087_aa_ov"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0087_aa_ov")(ka,ca)  = t1.at("aa")(ba,ja) * tmps.at("0058_aaaa_ovov")(ka,ba,ja,ca) )
    
    // r1_1p[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t2_1p.at("aaaa")(ca,aa,ia,ka) * tmps.at("0087_aa_ov")(ka,ca) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_2p_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t2_2p.at("aaaa")(ca,aa,ia,ka) * tmps.at("0087_aa_ov")(ka,ca) )
    .deallocate(tmps.at("0087_aa_ov"))
    .allocate(tmps.at("0088_aaaa_ooov"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0088_aaaa_ooov")(ja,ia,ka,ba)  = chol.at("aa_ooQ")(ja,ia,Q) * chol.at("aa_ovQ")(ka,ba,Q) )
    
    // r1[aa] += +0.500 <j,k||i,b>_aaaa t2_aaaa(b,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= 0.500 * t2.at("aaaa")(ba,aa,ja,ka) * tmps.at("0088_aaaa_ooov")(ka,ia,ja,ba) )
    
    // r1_1p[aa] += +0.500 <j,k||i,b>_aaaa t2_1p_aaaa(b,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.500 * t2_1p.at("aaaa")(ba,aa,ja,ka) * tmps.at("0088_aaaa_ooov")(ka,ia,ja,ba) )
    
    // r1_2p[aa] += +1.000 <j,k||i,b>_aaaa t2_2p_aaaa(b,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= t2_2p.at("aaaa")(ba,aa,ja,ka) * tmps.at("0088_aaaa_ooov")(ka,ia,ja,ba) )
    
    // r2_2p[abab] += +2.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_abab(c,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0088_aaaa_ooov")(la,ia,ka,ca) * t2_2p.at("abab")(ca,bb,la,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    .allocate(tmps.at("0089_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0089_aa_oo")(ia,ka)  = tmps.at("0088_aaaa_ooov")(ja,ia,ka,ba) * t1.at("aa")(ba,ja) )
    
    // r1[aa] += -1.000 <k,j||i,b>_aaaa t1_aa(a,k) t1_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += t1.at("aa")(aa,ka) * tmps.at("0089_aa_oo")(ia,ka) )
    
    // r1_1p[aa] += -1.000 <k,j||i,b>_aaaa t1_1p_aa(a,k) t1_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1_1p.at("aa")(aa,ka) * tmps.at("0089_aa_oo")(ia,ka) )
    
    // r1_2p[aa] += -2.000 <k,j||i,b>_aaaa t1_2p_aa(a,k) t1_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1_2p.at("aa")(aa,ka) * tmps.at("0089_aa_oo")(ia,ka) )
    
    // r2[abab] += -1.000 <l,k||i,c>_aaaa t1_aa(c,k) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0089_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_1p[abab] += -1.000 <l,k||i,c>_aaaa t1_aa(c,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0089_aa_oo")(ia,la) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||i,c>_aaaa t1_aa(c,k) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0089_aa_oo")(ia,la) * t2_2p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0089_aa_oo"))
    .allocate(tmps.at("0090_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0090_aa_oo")(ia,ka)  = tmps.at("0088_aaaa_ooov")(ja,ia,ka,ba) * t1_1p.at("aa")(ba,ja) )
    
    // r1_1p[aa] += +1.000 <k,j||i,b>_aaaa t1_aa(a,j) t1_1p_aa(b,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1.at("aa")(aa,ja) * tmps.at("0090_aa_oo")(ia,ja) )
    
    // r1_2p[aa] += -2.000 <k,j||i,b>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1_1p.at("aa")(aa,ka) * tmps.at("0090_aa_oo")(ia,ka) )
    
    // r2_1p[abab] += +1.000 <k,l||i,c>_aaaa t1_1p_aa(c,k) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,jb) * tmps.at("0090_aa_oo")(ia,la) )
    
    // r2_2p[abab] += -2.000 <l,k||i,c>_aaaa t1_1p_aa(c,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0090_aa_oo")(ia,la) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0090_aa_oo"))
    .allocate(tmps.at("0091_bbbb_ooov"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0091_bbbb_ooov")(lb,jb,kb,cb)  = chol.at("bb_ooQ")(lb,jb,Q) * chol.at("bb_ovQ")(kb,cb,Q) )
    
    // r2_2p[abab] += +2.000 <k,l||j,c>_bbbb t1_2p_bb(c,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = tmps.at("0091_bbbb_ooov")(lb,jb,kb,cb) * t1_2p.at("bb")(cb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2.at("abab")(aa,bb,ia,lb) * tmps.at("bin1_bb_oo")(jb,lb) )
    
    // r2_2p[abab] += +2.000 <k,l||j,c>_bbbb t1_2p_bb(c,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = tmps.at("0091_bbbb_ooov")(kb,jb,lb,cb) * t1_2p.at("bb")(cb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,bb,ia,lb) * tmps.at("bin1_bb_oo")(jb,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_2p_abab(a,c,i,l) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0091_bbbb_ooov")(lb,jb,kb,cb) * t2_2p.at("abab")(aa,cb,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0092_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0092_bb_oo")(jb,lb)  = tmps.at("0091_bbbb_ooov")(kb,jb,lb,cb) * t1_1p.at("bb")(cb,kb) )
    
    // r2_1p[abab] += +1.000 <k,l||j,c>_bbbb t1_1p_bb(c,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0092_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[abab] += -2.000 <l,k||j,c>_bbbb t1_1p_bb(c,k) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0092_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    .deallocate(tmps.at("0092_bb_oo"))
    .allocate(tmps.at("0093_aa_vo"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0093_aa_vo")(ca,ka)  = tmps.at("0058_aaaa_ovov")(ja,ca,ka,ba) * t1.at("aa")(ba,ja) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_aa_oo")(ia,ka)  = t1_2p.at("aa")(ca,ia) * tmps.at("0093_aa_vo")(ca,ka) )
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_aa_oo")(ia,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,la,jb)  = t2_2p.at("abab")(da,bb,ia,jb) * tmps.at("0093_aa_vo")(da,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,la,jb) * t1.at("aa")(aa,la) )
    .allocate(tmps.at("0094_aa_oo"))
    
    // flops: o2v0  = o2v2 o2v1
    //  mems: o2v0  = o1v1 o2v0
    ( tmps.at("bin1_aa_vo")(ca,ka)  = tmps.at("0058_aaaa_ovov")(ja,ca,ka,ba) * t1.at("aa")(ba,ja) )
    ( tmps.at("0094_aa_oo")(ia,ka)  = t1.at("aa")(ca,ia) * tmps.at("bin1_aa_vo")(ca,ka) )
    
    // r1[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0094_aa_oo")(ia,ka) * t1.at("aa")(aa,ka) )
    
    // r1_1p[aa] += +1.000 <k,j||b,c>_aaaa t1_1p_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1_1p.at("aa")(aa,ka) * tmps.at("0094_aa_oo")(ia,ka) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_2p_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1_2p.at("aa")(aa,ka) * tmps.at("0094_aa_oo")(ia,ka) )
    .deallocate(tmps.at("0094_aa_oo"))
    .allocate(tmps.at("0095_aa_oo"))
    
    // flops: o2v0  = o2v2 o2v1
    //  mems: o2v0  = o1v1 o2v0
    ( tmps.at("bin1_aa_vo")(ba,ka)  = tmps.at("0058_aaaa_ovov")(ja,ba,ka,ca) * t1_1p.at("aa")(ca,ja) )
    ( tmps.at("0095_aa_oo")(ka,ia)  = tmps.at("bin1_aa_vo")(ba,ka) * t1.at("aa")(ba,ia) )
    
    // r1_1p[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(a,j) t1_aa(b,i) t1_1p_aa(c,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1.at("aa")(aa,ja) * tmps.at("0095_aa_oo")(ja,ia) )
    
    // r1_2p[aa] += -2.000 <k,j||b,c>_aaaa t1_1p_aa(a,k) t1_aa(b,i) t1_1p_aa(c,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1_1p.at("aa")(aa,ka) * tmps.at("0095_aa_oo")(ka,ia) )
    .deallocate(tmps.at("0095_aa_oo"))
    .allocate(tmps.at("0096_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0096_bb_voQ")(bb,jb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t2_1p.at("bbbb")(cb,bb,jb,kb) )
    
    // r2_2p[abab] += -2.000 <a,k||c,d>_abab t1_1p_aa(c,i) t2_1p_bbbb(d,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0081_aa_voQ")(aa,ia,Q) * tmps.at("0096_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t2_1p_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0072_aa_voQ")(aa,ia,Q) * tmps.at("0096_bb_voQ")(bb,jb,Q) )
    
    // r2_1p[abab] += -1.000 <a,k||i,c>_abab t2_1p_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0096_bb_voQ")(bb,jb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    .allocate(tmps.at("0097_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0097_bb_voQ")(bb,jb,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t2_1p.at("abab")(ca,bb,ka,jb) )
    
    // r2_1p[abab] += +1.000 <a,k||i,c>_aaaa t2_1p_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0097_bb_voQ")(bb,jb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_abab t2_1p_abab(a,c,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0072_aa_voQ")(aa,ia,Q) * tmps.at("0097_bb_voQ")(bb,jb,Q) )
    
    // r2_2p[abab] += +2.000 <a,k||c,d>_aaaa t1_1p_aa(c,i) t2_1p_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0081_aa_voQ")(aa,ia,Q) * tmps.at("0097_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0099_bb_ooQ"))
    ;
  }
  // clang-format on
}

template void exachem::cc::cd_qed_ccsd_cs::resid_part2<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);