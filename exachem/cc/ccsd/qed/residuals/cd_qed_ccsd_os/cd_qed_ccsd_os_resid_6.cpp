/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_qed_ccsd_os_resid_6.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_os::resid_part6(
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
        
    // r1[bb] += +0.500 <j,k||b,c>_bbbb t1_bb(b,i) t2_bbbb(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += 0.500 * tmps.at("0228_bbbb_ovoo")(kb,cb,ib,jb) * t2.at("bbbb")(cb,ab,jb,kb) )
    
    // r1_1p[bb] += +0.500 <j,k||b,c>_bbbb t1_bb(b,i) t2_1p_bbbb(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += 0.500 * tmps.at("0228_bbbb_ovoo")(kb,cb,ib,jb) * t2_1p.at("bbbb")(cb,ab,jb,kb) )
    
    // r1_2p[bb] += +1.000 <j,k||b,c>_bbbb t1_bb(b,i) t2_2p_bbbb(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += tmps.at("0228_bbbb_ovoo")(kb,cb,ib,jb) * t2_2p.at("bbbb")(cb,ab,jb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_2p_abab(a,d,i,l) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0228_bbbb_ovoo")(lb,db,jb,kb) * t2_2p.at("abab")(aa,db,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    .allocate(tmps.at("0229_bbbb_ovoo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0229_bbbb_ovoo")(lb,db,ib,kb)  = t1_1p.at("bb")(cb,ib) * tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) )
    
    // r1_1p[bb] += -0.500 <j,k||c,b>_bbbb t1_1p_bb(b,i) t2_bbbb(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += 0.500 * tmps.at("0229_bbbb_ovoo")(kb,cb,ib,jb) * t2.at("bbbb")(cb,ab,jb,kb) )
    
    // r1_2p[bb] += +1.000 <j,k||b,c>_bbbb t1_1p_bb(b,i) t2_1p_bbbb(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += tmps.at("0229_bbbb_ovoo")(kb,cb,ib,jb) * t2_1p.at("bbbb")(cb,ab,jb,kb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_bbbb t1_1p_bb(c,k) t1_1p_bb(d,j) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = t1_1p.at("bb")(cb,kb) * tmps.at("0229_bbbb_ovoo")(lb,cb,jb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,bb,ia,lb) * tmps.at("bin1_bb_oo")(jb,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_1p_bb(c,j) t2_1p_abab(a,d,i,l) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0229_bbbb_ovoo")(lb,db,jb,kb) * t2_1p.at("abab")(aa,db,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    .allocate(tmps.at("0230_aabb_vooo"))
    
    // flops: o3v1  = o4v2 o4v2 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1
    ( tmps.at("0230_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0228_bbbb_ovoo")(lb,db,jb,kb) * t2_1p.at("abab")(aa,db,ia,lb) )
    ( tmps.at("0230_aabb_vooo")(aa,ia,jb,kb) += tmps.at("0229_bbbb_ovoo")(lb,db,jb,kb) * t2.at("abab")(aa,db,ia,lb) )
    
    // r2_1p[abab] += -1.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_1p_bb(c,j) t2_abab(a,d,i,l) 
    //               += +1.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_1p_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0230_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_bbbb t1_1p_bb(b,k) t1_1p_bb(c,j) t2_abab(a,d,i,l) 
    //               += +2.000 <l,k||c,d>_bbbb t1_1p_bb(b,k) t1_bb(c,j) t2_1p_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0230_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0230_aabb_vooo"))
    .allocate(tmps.at("0231_bbbb_vvoo"))
    
    // flops: o2v2  = o3v3 o3v3 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0231_bbbb_vvoo")(bb,ab,ib,jb)  = tmps.at("0185_bbbb_voov")(ab,lb,jb,db) * t2.at("bbbb")(db,bb,ib,lb) )
    ( tmps.at("0231_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("0177_baba_voov")(ab,la,jb,da) * t2.at("abab")(da,bb,la,ib) )
    .deallocate(tmps.at("0185_bbbb_voov"))
    
    // r2[bbbb] += -1.000 P(i,j) <l,k||c,d>_aaaa t2_abab(c,a,k,i) t2_abab(d,b,l,j) 
    //            += -1.000 P(i,j) <l,k||c,d>_bbbb t2_bbbb(c,a,i,k) t2_bbbb(d,b,j,l) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0231_bbbb_vvoo")(bb,ab,jb,ib) )
    
    // r2[bbbb] += -1.000 P(i,j) <l,k||c,d>_aaaa t2_abab(c,a,k,i) t2_abab(d,b,l,j) 
    //            += -1.000 P(i,j) <l,k||c,d>_bbbb t2_bbbb(c,a,i,k) t2_bbbb(d,b,j,l) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0231_bbbb_vvoo")(bb,ab,ib,jb) )
    .deallocate(tmps.at("0231_bbbb_vvoo"))
    .allocate(tmps.at("0232_bbbb_vvoo"))
    
    // flops: o2v2  = o1v1Q1 o1v1Q1 o3v2 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o2v2 o2v3 o2v2 o2v2 o3v2 o3v2 o2v2 o1v2Q1 o2v3 o2v2
    //  mems: o2v2  = o0v0Q1 o1v1 o3v1 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o2v2 o2v2 o2v2 o1v1 o3v1 o2v2 o2v2 o0v2 o2v2 o2v2
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = tmps.at("bin1_bb_vo")(db,lb) * t2.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("0232_bbbb_vvoo")(bb,ab,ib,jb)  = tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) * t1.at("bb")(bb,lb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = tmps.at("bin1_bb_vo")(db,lb) * t2.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("0232_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) * t1.at("bb")(bb,lb) )
    ( tmps.at("0232_bbbb_vvoo")(bb,ab,ib,jb) += t2.at("bbbb")(db,bb,ib,jb) * tmps.at("0182_bb_vv")(ab,db) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,lb)  = t2.at("bbbb")(db,bb,ib,jb) * tmps.at("bin1_bb_vo")(db,lb) )
    ( tmps.at("0232_bbbb_vvoo")(bb,ab,ib,jb) += t1.at("bb")(ab,lb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,lb) )
    ( tmps.at("bin1_bb_vv")(bb,db)  = chol.at("bb_ovQ")(lb,db,Q) * tmps.at("0058_bb_voQ")(bb,lb,Q) )
    ( tmps.at("0232_bbbb_vvoo")(bb,ab,ib,jb) += tmps.at("bin1_bb_vv")(bb,db) * t2.at("bbbb")(db,ab,ib,jb) )
    
    // r2[bbbb] += -1.000 P(a,b) <a,k||c,d>_bbbb t1_bb(c,k) t2_bbbb(d,b,i,j) 
    //            += +1.000 P(a,b) <k,a||c,d>_abab t1_aa(c,k) t2_bbbb(d,b,i,j) 
    //            += -1.000 P(a,b) <k,l||c,d>_abab t1_bb(a,l) t1_aa(c,k) t2_bbbb(d,b,i,j) 
    //            += +1.000 P(a,b) <l,k||c,d>_bbbb t1_bb(a,l) t1_bb(c,k) t2_bbbb(d,b,i,j) 
    //            += +0.500 P(a,b) <k,l||c,d>_abab t2_bbbb(d,a,i,j) t2_abab(c,b,k,l) 
    //            += +0.500 P(a,b) <l,k||c,d>_abab t2_bbbb(d,a,i,j) t2_abab(c,b,l,k) 
    //            += +1.000 P(a,b) <l,k||c,d>_bbbb t1_bb(a,l) t1_bb(c,k) t2_bbbb(d,b,i,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0232_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2[bbbb] += -1.000 P(a,b) <a,k||c,d>_bbbb t1_bb(c,k) t2_bbbb(d,b,i,j) 
    //            += +1.000 P(a,b) <k,a||c,d>_abab t1_aa(c,k) t2_bbbb(d,b,i,j) 
    //            += -1.000 P(a,b) <k,l||c,d>_abab t1_bb(a,l) t1_aa(c,k) t2_bbbb(d,b,i,j) 
    //            += +1.000 P(a,b) <l,k||c,d>_bbbb t1_bb(a,l) t1_bb(c,k) t2_bbbb(d,b,i,j) 
    //            += +0.500 P(a,b) <k,l||c,d>_abab t2_bbbb(d,a,i,j) t2_abab(c,b,k,l) 
    //            += +0.500 P(a,b) <l,k||c,d>_abab t2_bbbb(d,a,i,j) t2_abab(c,b,l,k) 
    //            += +1.000 P(a,b) <l,k||c,d>_bbbb t1_bb(a,l) t1_bb(c,k) t2_bbbb(d,b,i,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0232_bbbb_vvoo")(bb,ab,ib,jb) )
    .deallocate(tmps.at("0232_bbbb_vvoo"))
    .allocate(tmps.at("0233_bbbb_voov"))
    
    // flops: o2v2  = o1v3 o2v3 o1v3 o2v3 o2v2 o1v3 o2v3 o2v2
    //  mems: o2v2  = o0v2 o2v2 o0v2 o2v2 o2v2 o0v2 o2v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,db)  = t1.at("bb")(cb,kb) * tmps.at("0086_bbbb_ovvv")(kb,db,bb,cb) )
    ( tmps.at("0233_bbbb_voov")(ab,ib,jb,bb)  = tmps.at("bin1_bb_vv")(bb,db) * t2_2p.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("bin1_bb_vv")(bb,db)  = t1_2p.at("bb")(cb,kb) * tmps.at("0086_bbbb_ovvv")(kb,db,bb,cb) )
    ( tmps.at("0233_bbbb_voov")(ab,ib,jb,bb) += tmps.at("bin1_bb_vv")(bb,db) * t2.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("bin1_bb_vv")(bb,db)  = t1_1p.at("bb")(cb,kb) * tmps.at("0086_bbbb_ovvv")(kb,db,bb,cb) )
    ( tmps.at("0233_bbbb_voov")(ab,ib,jb,bb) += tmps.at("bin1_bb_vv")(bb,db) * t2_1p.at("bbbb")(db,ab,ib,jb) )
    
    // r2_2p[bbbb] += +2.000 P(a,b) <a,k||d,c>_bbbb t1_2p_bb(c,k) t2_bbbb(d,b,i,j) 
    //               += -2.000 P(a,b) <a,k||c,d>_bbbb t1_1p_bb(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += -2.000 P(a,b) <a,k||c,d>_bbbb t1_bb(c,k) t2_2p_bbbb(d,b,i,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0233_bbbb_voov")(ab,ib,jb,bb) )
    
    // r2_2p[bbbb] += +2.000 P(a,b) <a,k||d,c>_bbbb t1_2p_bb(c,k) t2_bbbb(d,b,i,j) 
    //               += -2.000 P(a,b) <a,k||c,d>_bbbb t1_1p_bb(c,k) t2_1p_bbbb(d,b,i,j) 
    //               += -2.000 P(a,b) <a,k||c,d>_bbbb t1_bb(c,k) t2_2p_bbbb(d,b,i,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0233_bbbb_voov")(bb,ib,jb,ab) )
    .deallocate(tmps.at("0233_bbbb_voov"))
    .allocate(tmps.at("0234_bbbb_voov"))
    
    // flops: o2v2  = o1v3 o2v3 o1v3 o2v3 o2v2
    //  mems: o2v2  = o0v2 o2v2 o0v2 o2v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,db)  = t1.at("bb")(cb,kb) * tmps.at("0086_bbbb_ovvv")(kb,db,bb,cb) )
    ( tmps.at("0234_bbbb_voov")(ab,ib,jb,bb)  = tmps.at("bin1_bb_vv")(bb,db) * t2_1p.at("bbbb")(db,ab,ib,jb) )
    ( tmps.at("bin1_bb_vv")(bb,db)  = t1_1p.at("bb")(cb,kb) * tmps.at("0086_bbbb_ovvv")(kb,db,bb,cb) )
    ( tmps.at("0234_bbbb_voov")(ab,ib,jb,bb) += tmps.at("bin1_bb_vv")(bb,db) * t2.at("bbbb")(db,ab,ib,jb) )
    
    // r2_1p[bbbb] += +1.000 P(a,b) <a,k||d,c>_bbbb t1_1p_bb(c,k) t2_bbbb(d,b,i,j) 
    //               += -1.000 P(a,b) <a,k||c,d>_bbbb t1_bb(c,k) t2_1p_bbbb(d,b,i,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0234_bbbb_voov")(ab,ib,jb,bb) )
    
    // r2_1p[bbbb] += +1.000 P(a,b) <a,k||d,c>_bbbb t1_1p_bb(c,k) t2_bbbb(d,b,i,j) 
    //               += -1.000 P(a,b) <a,k||c,d>_bbbb t1_bb(c,k) t2_1p_bbbb(d,b,i,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0234_bbbb_voov")(bb,ib,jb,ab) )
    .deallocate(tmps.at("0234_bbbb_voov"))
    .allocate(tmps.at("0235_aaaa_voov"))
    
    // flops: o2v2  = o2v3 o2v3 o2v3 o2v3 o2v2
    //  mems: o2v2  = o0v2 o2v2 o0v2 o2v2 o2v2
    ( tmps.at("bin1_aa_vv")(aa,da)  = t2.at("aaaa")(ca,aa,ka,la) * tmps.at("0073_aaaa_ovov")(ka,da,la,ca) )
    ( tmps.at("0235_aaaa_voov")(ba,ia,ja,aa)  = tmps.at("bin1_aa_vv")(aa,da) * t2_2p.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = t2_1p.at("aaaa")(ca,aa,ka,la) * tmps.at("0073_aaaa_ovov")(ka,da,la,ca) )
    ( tmps.at("0235_aaaa_voov")(ba,ia,ja,aa) += tmps.at("bin1_aa_vv")(aa,da) * t2_1p.at("aaaa")(da,ba,ia,ja) )
    
    // r2_2p[aaaa] += -1.000 P(a,b) <k,l||d,c>_aaaa t2_1p_aaaa(d,a,i,j) t2_1p_aaaa(c,b,k,l) 
    //               += +1.000 P(a,b) <k,l||c,d>_aaaa t2_2p_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0235_aaaa_voov")(aa,ia,ja,ba) )
    
    // r2_2p[aaaa] += -1.000 P(a,b) <k,l||d,c>_aaaa t2_1p_aaaa(d,a,i,j) t2_1p_aaaa(c,b,k,l) 
    //               += +1.000 P(a,b) <k,l||c,d>_aaaa t2_2p_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0235_aaaa_voov")(ba,ia,ja,aa) )
    .deallocate(tmps.at("0235_aaaa_voov"))
    .allocate(tmps.at("0236_abab_vooo"))
    
    // flops: o3v1  = o3v2 o3v3 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1
    ( tmps.at("0236_abab_vooo")(aa,kb,ia,jb)  = f.at("bb_ov")(kb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    ( tmps.at("0236_abab_vooo")(aa,kb,ia,jb) += t2_1p.at("abab")(da,cb,ia,jb) * tmps.at("0099_aabb_vvov")(aa,da,kb,cb) )
    .deallocate(tmps.at("0099_aabb_vvov"))
    
    // r2_1p[abab] += -1.000 f_bb(k,c) t1_bb(b,k) t2_1p_abab(a,c,i,j) 
    //               += -0.500 <a,k||d,c>_abab t1_bb(b,k) t2_1p_abab(d,c,i,j) 
    //               += -0.500 <a,k||c,d>_abab t1_bb(b,k) t2_1p_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0236_abab_vooo")(aa,kb,ia,jb) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -2.000 f_bb(k,c) t1_1p_bb(b,k) t2_1p_abab(a,c,i,j) 
    //               += -1.000 <a,k||d,c>_abab t1_1p_bb(b,k) t2_1p_abab(d,c,i,j) 
    //               += -1.000 <a,k||c,d>_abab t1_1p_bb(b,k) t2_1p_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0236_abab_vooo")(aa,kb,ia,jb) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0236_abab_vooo"))
    .allocate(tmps.at("0237_bb_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0237_bb_vv")(cb,bb)  = tmps.at("0075_bbbb_ovov")(kb,cb,lb,db) * t2_1p.at("bbbb")(db,bb,kb,lb) )
    
    // r2_1p[abab] += +0.500 <k,l||c,d>_bbbb t2_abab(a,c,i,j) t2_1p_bbbb(d,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.500 * t2.at("abab")(aa,cb,ia,jb) * tmps.at("0237_bb_vv")(cb,bb) )
    
    // r2_2p[abab] += +1.000 <k,l||d,c>_bbbb t2_1p_abab(a,d,i,j) t2_1p_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,db,ia,jb) * tmps.at("0237_bb_vv")(db,bb) )
    .allocate(tmps.at("0238_bbbb_voov"))
    
    // flops: o2v2  = o2v3 o2v3 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0238_bbbb_voov")(bb,ib,jb,ab)  = tmps.at("0131_bb_vv")(db,ab) * t2_2p.at("bbbb")(db,bb,ib,jb) )
    ( tmps.at("0238_bbbb_voov")(bb,ib,jb,ab) += tmps.at("0237_bb_vv")(db,ab) * t2_1p.at("bbbb")(db,bb,ib,jb) )
    .deallocate(tmps.at("0237_bb_vv"))
    .deallocate(tmps.at("0131_bb_vv"))
    
    // r2_2p[bbbb] += -1.000 P(a,b) <k,l||d,c>_bbbb t2_1p_bbbb(d,a,i,j) t2_1p_bbbb(c,b,k,l) 
    //               += +1.000 P(a,b) <k,l||c,d>_bbbb t2_2p_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0238_bbbb_voov")(ab,ib,jb,bb) )
    
    // r2_2p[bbbb] += -1.000 P(a,b) <k,l||d,c>_bbbb t2_1p_bbbb(d,a,i,j) t2_1p_bbbb(c,b,k,l) 
    //               += +1.000 P(a,b) <k,l||c,d>_bbbb t2_2p_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0238_bbbb_voov")(bb,ib,jb,ab) )
    .deallocate(tmps.at("0238_bbbb_voov"))
    .allocate(tmps.at("0239_bbaa_vooo"))
    
    // flops: o3v1  = o3v1Q1 o3v1Q1 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1
    ( tmps.at("0239_bbaa_vooo")(bb,jb,ka,ia)  = -1.000 * tmps.at("0060_bb_voQ")(bb,jb,Q) * tmps.at("0053_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0239_bbaa_vooo")(bb,jb,ka,ia) += tmps.at("0059_bb_voQ")(bb,jb,Q) * tmps.at("0053_aa_ooQ")(ka,ia,Q) )
    
    // r2[abab] += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_bbbb(d,b,j,l) 
    //            += -1.000 <k,b||c,d>_abab t1_aa(a,k) t1_aa(c,i) t1_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0239_bbaa_vooo")(bb,jb,ka,ia) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_abab t1_1p_aa(a,k) t1_aa(c,i) t2_bbbb(d,b,j,l) 
    //               += -1.000 <k,b||c,d>_abab t1_1p_aa(a,k) t1_aa(c,i) t1_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0239_bbaa_vooo")(bb,jb,ka,ia) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t1_2p_aa(a,k) t1_aa(c,i) t2_bbbb(d,b,j,l) 
    //               += -2.000 <k,b||c,d>_abab t1_2p_aa(a,k) t1_aa(c,i) t1_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0239_bbaa_vooo")(bb,jb,ka,ia) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0239_bbaa_vooo"))
    .allocate(tmps.at("0240_bbaa_vooo"))
    
    // flops: o3v1  = o3v1Q1 o3v1Q1 o3v1 o3v1Q1 o3v1 o3v1Q1 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("0240_bbaa_vooo")(bb,jb,ka,ia)  = -1.000 * tmps.at("0060_bb_voQ")(bb,jb,Q) * tmps.at("0137_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0240_bbaa_vooo")(bb,jb,ka,ia) -= tmps.at("0191_bb_voQ")(bb,jb,Q) * tmps.at("0053_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0240_bbaa_vooo")(bb,jb,ka,ia) += tmps.at("0059_bb_voQ")(bb,jb,Q) * tmps.at("0137_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0240_bbaa_vooo")(bb,jb,ka,ia) += tmps.at("0136_bb_voQ")(bb,jb,Q) * tmps.at("0053_aa_ooQ")(ka,ia,Q) )
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_1p_aa(c,i) t2_bbbb(d,b,j,l) 
    //               += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += -1.000 <k,b||d,c>_abab t1_aa(a,k) t1_bb(c,j) t1_1p_aa(d,i) 
    //               += -1.000 <k,b||c,d>_abab t1_aa(a,k) t1_aa(c,i) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0240_bbaa_vooo")(bb,jb,ka,ia) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_abab t1_1p_aa(a,k) t1_1p_aa(c,i) t2_bbbb(d,b,j,l) 
    //               += +2.000 <k,l||c,d>_abab t1_1p_aa(a,k) t1_aa(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += -2.000 <k,b||d,c>_abab t1_1p_aa(a,k) t1_bb(c,j) t1_1p_aa(d,i) 
    //               += -2.000 <k,b||c,d>_abab t1_1p_aa(a,k) t1_aa(c,i) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0240_bbaa_vooo")(bb,jb,ka,ia) * t1_1p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0240_bbaa_vooo"))
    .allocate(tmps.at("0241_abab_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0241_abab_vooo")(aa,kb,ia,jb)  = t2.at("abab")(aa,cb,ia,jb) * dp.at("bb_ov")(kb,cb) )
    .allocate(tmps.at("0242_abab_vvoo"))
    
    // flops: o2v2  = o2v3 o3v2 o3v2 o2v2 o2v3 o2v2 o2v1 o3v2 o2v2 o2v1 o3v2 o2v2 o3v2 o2v2 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0242_abab_vvoo")(aa,bb,ia,jb)  = -1.000 * dp.at("aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = dp.at("aa_ov")(ka,ca) * t2.at("abab")(ca,bb,ia,jb) )
    ( tmps.at("0242_abab_vvoo")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    ( tmps.at("0242_abab_vvoo")(aa,bb,ia,jb) -= dp.at("bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    ( tmps.at("bin1_bb_oo")(jb,kb)  = dp.at("bb_ov")(kb,cb) * t1.at("bb")(cb,jb) )
    ( tmps.at("0242_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    ( tmps.at("bin1_aa_oo")(ia,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("0242_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_aa_oo")(ia,ka) * t2.at("abab")(aa,bb,ka,jb) )
    ( tmps.at("0242_abab_vvoo")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0241_abab_vooo")(aa,kb,ia,jb) )
    ( tmps.at("0242_abab_vvoo")(aa,bb,ia,jb) += dp.at("aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    ( tmps.at("0242_abab_vvoo")(aa,bb,ia,jb) += dp.at("bb_oo")(kb,jb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2[abab] += -1.000 d-_bb(k,c) t0_1p t1_bb(b,k) t2_abab(a,c,i,j) 
    //            += -1.000 d-_aa(k,i) t0_1p t2_abab(a,b,k,j) 
    //            += +1.000 d-_aa(a,c) t0_1p t2_abab(c,b,i,j) 
    //            += -1.000 d-_aa(k,c) t0_1p t1_aa(a,k) t2_abab(c,b,i,j) 
    //            += -1.000 d-_bb(k,c) t0_1p t1_bb(c,j) t2_abab(a,b,i,k) 
    //            += -1.000 d-_bb(k,j) t0_1p t2_abab(a,b,i,k) 
    //            += +1.000 d-_bb(b,c) t0_1p t2_abab(a,c,i,j) 
    //            += -1.000 d-_aa(k,c) t0_1p t1_aa(c,i) t2_abab(a,b,k,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0242_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_1p[abab] += -1.000 d+_bb(k,c) t1_bb(b,k) t2_abab(a,c,i,j) 
    //               += -1.000 d+_aa(k,i) t2_abab(a,b,k,j) 
    //               += +1.000 d+_aa(a,c) t2_abab(c,b,i,j) 
    //               += -1.000 d+_aa(k,c) t1_aa(a,k) t2_abab(c,b,i,j) 
    //               += -1.000 d+_bb(k,c) t1_bb(c,j) t2_abab(a,b,i,k) 
    //               += -1.000 d+_bb(k,j) t2_abab(a,b,i,k) 
    //               += +1.000 d+_bb(b,c) t2_abab(a,c,i,j) 
    //               += -1.000 d+_aa(k,c) t1_aa(c,i) t2_abab(a,b,k,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0242_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_1p[abab] += -2.000 d-_bb(k,c) t0_2p t1_bb(b,k) t2_abab(a,c,i,j) 
    //               += -2.000 d-_aa(k,i) t0_2p t2_abab(a,b,k,j) 
    //               += +2.000 d-_aa(a,c) t0_2p t2_abab(c,b,i,j) 
    //               += -2.000 d-_aa(k,c) t0_2p t1_aa(a,k) t2_abab(c,b,i,j) 
    //               += -2.000 d-_bb(k,c) t0_2p t1_bb(c,j) t2_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,j) t0_2p t2_abab(a,b,i,k) 
    //               += +2.000 d-_bb(b,c) t0_2p t2_abab(a,c,i,j) 
    //               += -2.000 d-_aa(k,c) t0_2p t1_aa(c,i) t2_abab(a,b,k,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.000 * t0_2p * tmps.at("0242_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0242_abab_vvoo"))
    .allocate(tmps.at("0243_abab_vvoo"))
    
    // flops: o2v2  = o2v3 o3v2 o3v2 o2v2 o3v2 o3v2 o2v2 o2v1 o3v2 o2v2 o2v3 o2v2 o2v1 o3v2 o2v2 o2v1 o3v2 o2v2 o3v2 o2v2 o3v2 o2v2 o3v2 o3v2 o2v2 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb)  = -1.000 * dp.at("aa_vv")(aa,ca) * t2_1p.at("abab")(ca,bb,ia,jb) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = dp.at("aa_ov")(ka,ca) * t2_1p.at("abab")(ca,bb,ia,jb) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = dp.at("aa_ov")(ka,ca) * t2.at("abab")(ca,bb,ia,jb) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    ( tmps.at("bin1_bb_oo")(jb,kb)  = dp.at("bb_ov")(kb,cb) * t1_1p.at("bb")(cb,jb) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) -= dp.at("bb_vv")(bb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    ( tmps.at("bin1_bb_oo")(jb,kb)  = dp.at("bb_ov")(kb,cb) * t1.at("bb")(cb,jb) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_bb_oo")(jb,kb) * t2_1p.at("abab")(aa,bb,ia,kb) )
    ( tmps.at("bin1_aa_oo")(ia,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_aa_oo")(ia,ka) * t2_1p.at("abab")(aa,bb,ka,jb) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ka,jb) * tmps.at("0031_aa_oo")(ka,ia) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) += t1_1p.at("bb")(bb,kb) * tmps.at("0241_abab_vooo")(aa,kb,ia,jb) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) += dp.at("aa_oo")(ka,ia) * t2_1p.at("abab")(aa,bb,ka,jb) )
    ( tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) += dp.at("bb_oo")(kb,jb) * t2_1p.at("abab")(aa,bb,ia,kb) )
    
    // r2[abab] += -1.000 d-_bb(k,c) t1_1p_bb(b,k) t2_abab(a,c,i,j) 
    //            += -1.000 d-_bb(k,c) t1_bb(b,k) t2_1p_abab(a,c,i,j) 
    //            += -1.000 d-_aa(k,i) t2_1p_abab(a,b,k,j) 
    //            += +1.000 d-_aa(a,c) t2_1p_abab(c,b,i,j) 
    //            += -1.000 d-_aa(k,c) t1_aa(a,k) t2_1p_abab(c,b,i,j) 
    //            += -1.000 d-_aa(k,c) t1_1p_aa(a,k) t2_abab(c,b,i,j) 
    //            += -1.000 d-_bb(k,c) t1_1p_bb(c,j) t2_abab(a,b,i,k) 
    //            += -1.000 d-_bb(k,c) t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    //            += -1.000 d-_bb(k,j) t2_1p_abab(a,b,i,k) 
    //            += +1.000 d-_bb(b,c) t2_1p_abab(a,c,i,j) 
    //            += -1.000 d-_aa(k,c) t1_aa(c,i) t2_1p_abab(a,b,k,j) 
    //            += -1.000 d-_aa(k,c) t1_1p_aa(c,i) t2_abab(a,b,k,j) 
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_1p[abab] += -1.000 d-_bb(k,c) t0_1p t1_1p_bb(b,k) t2_abab(a,c,i,j) 
    //               += -1.000 d-_bb(k,c) t0_1p t1_bb(b,k) t2_1p_abab(a,c,i,j) 
    //               += -1.000 d-_aa(k,i) t0_1p t2_1p_abab(a,b,k,j) 
    //               += +1.000 d-_aa(a,c) t0_1p t2_1p_abab(c,b,i,j) 
    //               += -1.000 d-_aa(k,c) t0_1p t1_aa(a,k) t2_1p_abab(c,b,i,j) 
    //               += -1.000 d-_aa(k,c) t0_1p t1_1p_aa(a,k) t2_abab(c,b,i,j) 
    //               += -1.000 d-_bb(k,c) t0_1p t1_1p_bb(c,j) t2_abab(a,b,i,k) 
    //               += -1.000 d-_bb(k,c) t0_1p t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    //               += -1.000 d-_bb(k,j) t0_1p t2_1p_abab(a,b,i,k) 
    //               += +1.000 d-_bb(b,c) t0_1p t2_1p_abab(a,c,i,j) 
    //               += -1.000 d-_aa(k,c) t0_1p t1_aa(c,i) t2_1p_abab(a,b,k,j) 
    //               += -1.000 d-_aa(k,c) t0_1p t1_1p_aa(c,i) t2_abab(a,b,k,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 d+_bb(k,c) t1_1p_bb(b,k) t2_abab(a,c,i,j) 
    //               += -2.000 d+_bb(k,c) t1_bb(b,k) t2_1p_abab(a,c,i,j) 
    //               += -2.000 d+_aa(k,i) t2_1p_abab(a,b,k,j) 
    //               += +2.000 d+_aa(a,c) t2_1p_abab(c,b,i,j) 
    //               += -2.000 d+_aa(k,c) t1_aa(a,k) t2_1p_abab(c,b,i,j) 
    //               += -2.000 d+_aa(k,c) t1_1p_aa(a,k) t2_abab(c,b,i,j) 
    //               += -2.000 d+_bb(k,c) t1_1p_bb(c,j) t2_abab(a,b,i,k) 
    //               += -2.000 d+_bb(k,c) t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    //               += -2.000 d+_bb(k,j) t2_1p_abab(a,b,i,k) 
    //               += +2.000 d+_bb(b,c) t2_1p_abab(a,c,i,j) 
    //               += -2.000 d+_aa(k,c) t1_aa(c,i) t2_1p_abab(a,b,k,j) 
    //               += -2.000 d+_aa(k,c) t1_1p_aa(c,i) t2_abab(a,b,k,j) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p[abab] += -4.000 d-_bb(k,c) t0_2p t1_1p_bb(b,k) t2_abab(a,c,i,j) 
    //               += -4.000 d-_bb(k,c) t0_2p t1_bb(b,k) t2_1p_abab(a,c,i,j) 
    //               += -4.000 d-_aa(k,i) t0_2p t2_1p_abab(a,b,k,j) 
    //               += +4.000 d-_aa(a,c) t0_2p t2_1p_abab(c,b,i,j) 
    //               += -4.000 d-_aa(k,c) t0_2p t1_aa(a,k) t2_1p_abab(c,b,i,j) 
    //               += -4.000 d-_aa(k,c) t0_2p t1_1p_aa(a,k) t2_abab(c,b,i,j) 
    //               += -4.000 d-_bb(k,c) t0_2p t1_1p_bb(c,j) t2_abab(a,b,i,k) 
    //               += -4.000 d-_bb(k,c) t0_2p t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    //               += -4.000 d-_bb(k,j) t0_2p t2_1p_abab(a,b,i,k) 
    //               += +4.000 d-_bb(b,c) t0_2p t2_1p_abab(a,c,i,j) 
    //               += -4.000 d-_aa(k,c) t0_2p t1_aa(c,i) t2_1p_abab(a,b,k,j) 
    //               += -4.000 d-_aa(k,c) t0_2p t1_1p_aa(c,i) t2_abab(a,b,k,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.000 * t0_2p * tmps.at("0243_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0243_abab_vvoo"))
    .allocate(tmps.at("0244_abab_vvoo"))
    
    // flops: o2v2  = o2v3 o3v2 o3v2 o2v2 o3v2 o3v2 o2v2 o3v2 o3v2 o2v2 o2v1 o3v2 o2v2 o2v1 o3v2 o2v2 o2v3 o2v2 o2v1 o3v2 o2v2 o2v1 o3v2 o2v2 o3v2 o2v2 o3v2 o2v2 o3v2 o2v2 o3v2 o3v2 o2v2 o3v2 o3v2 o2v2 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb)  = -1.000 * dp.at("aa_vv")(aa,ca) * t2_2p.at("abab")(ca,bb,ia,jb) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = dp.at("aa_ov")(ka,ca) * t2_2p.at("abab")(ca,bb,ia,jb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = dp.at("aa_ov")(ka,ca) * t2_1p.at("abab")(ca,bb,ia,jb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = dp.at("aa_ov")(ka,ca) * t2.at("abab")(ca,bb,ia,jb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += t1_2p.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    ( tmps.at("bin1_bb_oo")(jb,kb)  = dp.at("bb_ov")(kb,cb) * t1_2p.at("bb")(cb,jb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    ( tmps.at("bin1_bb_oo")(jb,kb)  = dp.at("bb_ov")(kb,cb) * t1_1p.at("bb")(cb,jb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_bb_oo")(jb,kb) * t2_1p.at("abab")(aa,bb,ia,kb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) -= dp.at("bb_vv")(bb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    ( tmps.at("bin1_bb_oo")(jb,kb)  = dp.at("bb_ov")(kb,cb) * t1.at("bb")(cb,jb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_bb_oo")(jb,kb) * t2_2p.at("abab")(aa,bb,ia,kb) )
    ( tmps.at("bin1_aa_oo")(ia,ka)  = dp.at("aa_ov")(ka,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_aa_oo")(ia,ka) * t2_2p.at("abab")(aa,bb,ka,jb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,ka,jb) * tmps.at("0031_aa_oo")(ka,ia) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ka,jb) * tmps.at("0033_aa_oo")(ka,ia) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += t1_2p.at("bb")(bb,kb) * tmps.at("0241_abab_vooo")(aa,kb,ia,jb) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += dp.at("aa_oo")(ka,ia) * t2_2p.at("abab")(aa,bb,ka,jb) )
    ( tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) += dp.at("bb_oo")(kb,jb) * t2_2p.at("abab")(aa,bb,ia,kb) )
    .deallocate(tmps.at("0241_abab_vooo"))
    
    // r2_1p[abab] += -2.000 d-_bb(k,c) t1_2p_bb(b,k) t2_abab(a,c,i,j) 
    //               += -2.000 d-_bb(k,c) t1_1p_bb(b,k) t2_1p_abab(a,c,i,j) 
    //               += -2.000 d-_bb(k,c) t1_bb(b,k) t2_2p_abab(a,c,i,j) 
    //               += -2.000 d-_aa(k,i) t2_2p_abab(a,b,k,j) 
    //               += +2.000 d-_aa(a,c) t2_2p_abab(c,b,i,j) 
    //               += -2.000 d-_aa(k,c) t1_aa(a,k) t2_2p_abab(c,b,i,j) 
    //               += -2.000 d-_aa(k,c) t1_1p_aa(a,k) t2_1p_abab(c,b,i,j) 
    //               += -2.000 d-_aa(k,c) t1_2p_aa(a,k) t2_abab(c,b,i,j) 
    //               += -2.000 d-_bb(k,c) t1_2p_bb(c,j) t2_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t1_1p_bb(c,j) t2_1p_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t1_bb(c,j) t2_2p_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,j) t2_2p_abab(a,b,i,k) 
    //               += +2.000 d-_bb(b,c) t2_2p_abab(a,c,i,j) 
    //               += -2.000 d-_aa(k,c) t1_aa(c,i) t2_2p_abab(a,b,k,j) 
    //               += -2.000 d-_aa(k,c) t1_1p_aa(c,i) t2_1p_abab(a,b,k,j) 
    //               += -2.000 d-_aa(k,c) t1_2p_aa(c,i) t2_abab(a,b,k,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 d-_bb(k,c) t0_1p t1_2p_bb(b,k) t2_abab(a,c,i,j) 
    //               += -2.000 d-_bb(k,c) t0_1p t1_1p_bb(b,k) t2_1p_abab(a,c,i,j) 
    //               += -2.000 d-_bb(k,c) t0_1p t1_bb(b,k) t2_2p_abab(a,c,i,j) 
    //               += -2.000 d-_aa(k,i) t0_1p t2_2p_abab(a,b,k,j) 
    //               += +2.000 d-_aa(a,c) t0_1p t2_2p_abab(c,b,i,j) 
    //               += -2.000 d-_aa(k,c) t0_1p t1_aa(a,k) t2_2p_abab(c,b,i,j) 
    //               += -2.000 d-_aa(k,c) t0_1p t1_1p_aa(a,k) t2_1p_abab(c,b,i,j) 
    //               += -2.000 d-_aa(k,c) t0_1p t1_2p_aa(a,k) t2_abab(c,b,i,j) 
    //               += -2.000 d-_bb(k,c) t0_1p t1_2p_bb(c,j) t2_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t0_1p t1_1p_bb(c,j) t2_1p_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t0_1p t1_bb(c,j) t2_2p_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,j) t0_1p t2_2p_abab(a,b,i,k) 
    //               += +2.000 d-_bb(b,c) t0_1p t2_2p_abab(a,c,i,j) 
    //               += -2.000 d-_aa(k,c) t0_1p t1_aa(c,i) t2_2p_abab(a,b,k,j) 
    //               += -2.000 d-_aa(k,c) t0_1p t1_1p_aa(c,i) t2_1p_abab(a,b,k,j) 
    //               += -2.000 d-_aa(k,c) t0_1p t1_2p_aa(c,i) t2_abab(a,b,k,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t0_1p * tmps.at("0244_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0244_abab_vvoo"))
    .allocate(tmps.at("0245_bbbb_ovoo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0245_bbbb_ovoo")(lb,db,ib,kb)  = t1_2p.at("bb")(cb,ib) * tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) )
    
    // r1_2p[bb] += +2.000 <k,j||b,c>_bbbb t1_bb(a,k) t1_bb(b,j) t1_2p_bb(c,i) 
    // flops: o1v1 += o3v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_bb_oo")(ib,kb)  = tmps.at("0245_bbbb_ovoo")(kb,bb,ib,jb) * t1.at("bb")(bb,jb) )
    ( r1_2p.at("bb")(ab,ib) += 2.000 * t1.at("bb")(ab,kb) * tmps.at("bin1_bb_oo")(ib,kb) )
    
    // r1_2p[bb] += -1.000 <j,k||c,b>_bbbb t1_2p_bb(b,i) t2_bbbb(c,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += tmps.at("0245_bbbb_ovoo")(kb,cb,ib,jb) * t2.at("bbbb")(cb,ab,jb,kb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_2p_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0245_bbbb_ovoo")(lb,db,jb,kb) * t2.at("abab")(aa,db,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    .allocate(tmps.at("0246_bbbb_vovo"))
    
    // flops: o2v2  = o4v2 o3v2 o4v2 o3v2 o2v2 o4v0Q1 o4v1 o3v2 o2v2 o2v1Q1 o4v0Q1 o4v1 o3v2 o2v2 o4v2 o3v2 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2
    //  mems: o2v2  = o3v1 o2v2 o3v1 o2v2 o2v2 o4v0 o3v1 o2v2 o2v2 o2v0Q1 o4v0 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = tmps.at("0228_bbbb_ovoo")(lb,db,jb,kb) * t2_2p.at("bbbb")(db,ab,ib,lb) )
    ( tmps.at("0246_bbbb_vovo")(ab,ib,bb,jb)  = t1.at("bb")(bb,kb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = tmps.at("0229_bbbb_ovoo")(lb,db,jb,kb) * t2_1p.at("bbbb")(db,ab,ib,lb) )
    ( tmps.at("0246_bbbb_vovo")(ab,ib,bb,jb) += t1.at("bb")(bb,kb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = tmps.at("0029_bb_ooQ")(kb,ib,Q) * tmps.at("0029_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t1.at("bb")(bb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("0246_bbbb_vovo")(ab,ib,bb,jb) += tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1.at("bb")(ab,kb) )
    ( tmps.at("bin1_bb_ooQ")(ib,kb,Q)  = t1.at("bb")(cb,ib) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = tmps.at("0026_bb_ooQ")(lb,jb,Q) * tmps.at("bin1_bb_ooQ")(ib,kb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t1_1p.at("bb")(bb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("0246_bbbb_vovo")(ab,ib,bb,jb) += tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1_1p.at("bb")(ab,kb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = tmps.at("0245_bbbb_ovoo")(lb,db,jb,kb) * t2.at("bbbb")(db,ab,ib,lb) )
    ( tmps.at("0246_bbbb_vovo")(ab,ib,bb,jb) += t1.at("bb")(bb,kb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) )
    ( tmps.at("0246_bbbb_vovo")(ab,ib,bb,jb) += tmps.at("0136_bb_voQ")(ab,ib,Q) * tmps.at("0136_bb_voQ")(bb,jb,Q) )
    ( tmps.at("0246_bbbb_vovo")(ab,ib,bb,jb) += tmps.at("0135_bb_voQ")(ab,ib,Q) * tmps.at("0135_bb_voQ")(bb,jb,Q) )
    ( tmps.at("0246_bbbb_vovo")(ab,ib,bb,jb) += tmps.at("0191_bb_voQ")(ab,ib,Q) * tmps.at("0191_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0245_bbbb_ovoo"))
    
    // r2_2p[bbbb] += +2.000 <l,k||d,c>_bbbb t1_bb(a,k) t1_bb(b,l) t1_1p_bb(c,i) t1_1p_bb(d,j) 
    //               += +2.000 <l,k||d,c>_bbbb t1_1p_bb(a,k) t1_1p_bb(b,l) t1_bb(c,i) t1_bb(d,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_1p_abab(c,a,k,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_1p_bbbb(c,a,i,k) t2_1p_bbbb(d,b,j,l) 
    //               += -2.000 <a,b||d,c>_bbbb t1_1p_bb(c,i) t1_1p_bb(d,j) 
    //               += +2.000 P(i,j) P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_2p_bb(c,i) t2_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_bb(c,i) t2_2p_bbbb(d,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0246_bbbb_vovo")(ab,ib,bb,jb) )
    
    // r2_2p[bbbb] += +2.000 <l,k||d,c>_bbbb t1_bb(a,k) t1_bb(b,l) t1_1p_bb(c,i) t1_1p_bb(d,j) 
    //               += +2.000 <l,k||d,c>_bbbb t1_1p_bb(a,k) t1_1p_bb(b,l) t1_bb(c,i) t1_bb(d,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_1p_abab(c,a,k,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_1p_bbbb(c,a,i,k) t2_1p_bbbb(d,b,j,l) 
    //               += -2.000 <a,b||d,c>_bbbb t1_1p_bb(c,i) t1_1p_bb(d,j) 
    //               += +2.000 P(i,j) P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_2p_bb(c,i) t2_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_bb(c,i) t2_2p_bbbb(d,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0246_bbbb_vovo")(ab,jb,bb,ib) )
    .deallocate(tmps.at("0246_bbbb_vovo"))
    .allocate(tmps.at("0247_aaaa_vovo"))
    
    // flops: o2v2  = o4v0Q1 o4v1 o3v2 o3v2 o4v2 o3v2 o2v2 o3v2 o4v2 o3v2 o2v2 o2v1Q1 o4v0Q1 o4v1 o3v2 o2v2 o3v2 o4v2 o3v2 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2
    //  mems: o2v2  = o4v0 o3v1 o2v2 o3v1 o3v1 o2v2 o2v2 o3v1 o3v1 o2v2 o2v2 o2v0Q1 o4v0 o3v1 o2v2 o2v2 o3v1 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0137_aa_ooQ")(ka,ia,Q) * tmps.at("0137_aa_ooQ")(la,ja,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0247_aaaa_vovo")(aa,ia,ba,ja)  = tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1.at("aa")(ba,la) )
    ( tmps.at("bin2_aaaa_vooo")(da,ja,ka,la)  = t1_1p.at("aa")(ca,ja) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = t2_1p.at("aaaa")(da,aa,ia,la) * tmps.at("bin2_aaaa_vooo")(da,ja,ka,la) )
    ( tmps.at("0247_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("bin2_aaaa_vooo")(da,ja,ka,la)  = t1.at("aa")(ca,ja) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = t2_2p.at("aaaa")(da,aa,ia,la) * tmps.at("bin2_aaaa_vooo")(da,ja,ka,la) )
    ( tmps.at("0247_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("bin1_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("bin1_aa_ooQ")(ia,ka,Q) * tmps.at("0053_aa_ooQ")(la,ja,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = t1_1p.at("aa")(aa,ka) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0247_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1_1p.at("aa")(ba,la) )
    ( tmps.at("bin2_aaaa_vooo")(da,ja,ka,la)  = t1_2p.at("aa")(ca,ja) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = t2.at("aaaa")(da,aa,ia,la) * tmps.at("bin2_aaaa_vooo")(da,ja,ka,la) )
    ( tmps.at("0247_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("0247_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0028_aa_voQ")(aa,ia,Q) * tmps.at("0028_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0247_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0140_aa_voQ")(aa,ia,Q) * tmps.at("0140_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0247_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0141_aa_voQ")(aa,ia,Q) * tmps.at("0141_aa_voQ")(ba,ja,Q) )
    
    // r2_2p[aaaa] += +2.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_aa(b,l) t1_1p_aa(c,i) t1_1p_aa(d,j) 
    //               += +2.000 <l,k||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_1p_abab(a,c,i,k) t2_1p_abab(b,d,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_1p_aaaa(c,a,i,k) t2_1p_aaaa(d,b,j,l) 
    //               += -2.000 <a,b||d,c>_aaaa t1_1p_aa(c,i) t1_1p_aa(d,j) 
    //               += +2.000 P(i,j) P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_aaaa(d,b,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0247_aaaa_vovo")(aa,ia,ba,ja) )
    
    // r2_2p[aaaa] += +2.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_aa(b,l) t1_1p_aa(c,i) t1_1p_aa(d,j) 
    //               += +2.000 <l,k||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_1p_abab(a,c,i,k) t2_1p_abab(b,d,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_1p_aaaa(c,a,i,k) t2_1p_aaaa(d,b,j,l) 
    //               += -2.000 <a,b||d,c>_aaaa t1_1p_aa(c,i) t1_1p_aa(d,j) 
    //               += +2.000 P(i,j) P(a,b) <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,i) t2_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_aaaa(d,b,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0247_aaaa_vovo")(aa,ja,ba,ia) )
    .deallocate(tmps.at("0247_aaaa_vovo"))
    .allocate(tmps.at("0248_abab_ovoo"))
    
    // flops: o3v1  = o3v3 o3v1Q1 o4v2 o3v2 o4v2 o3v1 o3v1 o3v2 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("0248_abab_ovoo")(ka,bb,ia,jb)  = t2_1p.at("abab")(da,cb,ia,jb) * tmps.at("0047_aabb_ovvv")(ka,da,bb,cb) )
    ( tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la)  = chol.at("aa_ooQ")(la,ia,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("0248_abab_ovoo")(ka,bb,ia,jb) -= t2_1p.at("abab")(ca,bb,la,jb) * tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la) )
    ( tmps.at("bin1_aaaa_vooo")(da,ia,ka,la)  = t1.at("aa")(ca,ia) * tmps.at("0073_aaaa_ovov")(ka,ca,la,da) )
    ( tmps.at("0248_abab_ovoo")(ka,bb,ia,jb) += t2_1p.at("abab")(da,bb,la,jb) * tmps.at("bin1_aaaa_vooo")(da,ia,ka,la) )
    ( tmps.at("0248_abab_ovoo")(ka,bb,ia,jb) += t2_1p.at("abab")(ca,bb,ia,jb) * f.at("aa_ov")(ka,ca) )
    .deallocate(tmps.at("0047_aabb_ovvv"))
    
    // r2_1p[abab] += -1.000 f_aa(k,c) t1_aa(a,k) t2_1p_abab(c,b,i,j) 
    //               += -0.500 <k,b||d,c>_abab t1_aa(a,k) t2_1p_abab(d,c,i,j) 
    //               += -0.500 <k,b||c,d>_abab t1_aa(a,k) t2_1p_abab(c,d,i,j) 
    //               += +1.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_abab(c,b,l,j) 
    //               += +1.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("0248_abab_ovoo")(ka,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 f_aa(k,c) t1_1p_aa(a,k) t2_1p_abab(c,b,i,j) 
    //               += -1.000 <k,b||d,c>_abab t1_1p_aa(a,k) t2_1p_abab(d,c,i,j) 
    //               += -1.000 <k,b||c,d>_abab t1_1p_aa(a,k) t2_1p_abab(c,d,i,j) 
    //               += +2.000 <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_abab(c,b,l,j) 
    //               += +2.000 <l,k||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_1p.at("aa")(aa,ka) * tmps.at("0248_abab_ovoo")(ka,bb,ia,jb) )
    .deallocate(tmps.at("0248_abab_ovoo"))
    .allocate(tmps.at("0249_bb_oo"))
    
    // flops: o2v0  = o2v0Q1 o2v0Q1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0249_bb_oo")(kb,ib)  = tmps.at("0029_bb_ooQ")(kb,ib,Q) * tmps.at("0148_Q")(Q) )
    ( tmps.at("0249_bb_oo")(kb,ib) += tmps.at("0029_bb_ooQ")(kb,ib,Q) * tmps.at("0049_Q")(Q) )
    
    // r1_1p[bb] += +1.000 <k,j||b,c>_bbbb t1_bb(a,k) t1_bb(b,j) t1_1p_bb(c,i) 
    //             += -1.000 <j,k||b,c>_abab t1_bb(a,k) t1_aa(b,j) t1_1p_bb(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0249_bb_oo")(kb,ib) * t1.at("bb")(ab,kb) )
    
    // r1_2p[bb] += +2.000 <k,j||b,c>_bbbb t1_1p_bb(a,k) t1_bb(b,j) t1_1p_bb(c,i) 
    //             += -2.000 <j,k||b,c>_abab t1_1p_bb(a,k) t1_aa(b,j) t1_1p_bb(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0249_bb_oo")(kb,ib) * t1_1p.at("bb")(ab,kb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(c,k) t1_1p_bb(d,j) t2_abab(a,b,i,l) 
    //               += -1.000 <k,l||c,d>_abab t1_aa(c,k) t1_1p_bb(d,j) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0249_bb_oo")(lb,jb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(c,k) t1_1p_bb(d,j) t2_1p_abab(a,b,i,l) 
    //               += -2.000 <k,l||c,d>_abab t1_aa(c,k) t1_1p_bb(d,j) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0249_bb_oo")(lb,jb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    .deallocate(tmps.at("0249_bb_oo"))
    .allocate(tmps.at("0250_aabb_oovo"))
    
    // flops: o3v1  = o4v1 o4v1 o4v0Q1 o4v1 o3v1
    //  mems: o3v1  = o4v0 o3v1 o4v0 o3v1 o3v1
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb)  = t1_1p.at("bb")(cb,jb) * tmps.at("0121_aabb_ooov")(ka,ia,lb,cb) )
    ( tmps.at("0250_aabb_oovo")(ka,ia,bb,jb)  = t1.at("bb")(bb,lb) * tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) )
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb)  = tmps.at("0026_bb_ooQ")(lb,jb,Q) * tmps.at("0137_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0250_aabb_oovo")(ka,ia,bb,jb) += t1.at("bb")(bb,lb) * tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) )
    
    // r2_1p[abab] += +1.000 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t1_bb(c,j) t1_1p_aa(d,i) 
    //               += +1.000 <k,l||i,c>_abab t1_aa(a,k) t1_bb(b,l) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0250_aabb_oovo")(ka,ia,bb,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_abab t1_1p_aa(a,l) t1_bb(b,k) t1_bb(c,j) t1_1p_aa(d,i) 
    //               += +2.000 <l,k||i,c>_abab t1_1p_aa(a,l) t1_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0250_aabb_oovo")(la,ia,bb,jb) * t1_1p.at("aa")(aa,la) )
    .deallocate(tmps.at("0250_aabb_oovo"))
    .allocate(tmps.at("0251_bb_oo"))
    
    // flops: o2v0  = o2v0Q1 o2v0Q1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0251_bb_oo")(kb,ib)  = tmps.at("0187_bb_ooQ")(kb,ib,Q) * tmps.at("0148_Q")(Q) )
    ( tmps.at("0251_bb_oo")(kb,ib) += tmps.at("0187_bb_ooQ")(kb,ib,Q) * tmps.at("0049_Q")(Q) )
    
    // r1_2p[bb] += +2.000 <k,j||b,c>_bbbb t1_bb(a,k) t1_bb(b,j) t1_2p_bb(c,i) 
    //             += -2.000 <j,k||b,c>_abab t1_bb(a,k) t1_aa(b,j) t1_2p_bb(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0251_bb_oo")(kb,ib) * t1.at("bb")(ab,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(c,k) t1_2p_bb(d,j) t2_abab(a,b,i,l) 
    //               += -2.000 <k,l||c,d>_abab t1_aa(c,k) t1_2p_bb(d,j) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0251_bb_oo")(lb,jb) * t2.at("abab")(aa,bb,ia,lb) )
    .deallocate(tmps.at("0251_bb_oo"))
    .allocate(tmps.at("0252_aa_oo"))
    
    // flops: o2v0  = o2v0Q1 o2v0Q1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0252_aa_oo")(ka,ia)  = tmps.at("0221_aa_ooQ")(ka,ia,Q) * tmps.at("0148_Q")(Q) )
    ( tmps.at("0252_aa_oo")(ka,ia) += tmps.at("0221_aa_ooQ")(ka,ia,Q) * tmps.at("0049_Q")(Q) )
    
    // r1_2p[aa] += -2.000 <k,j||c,b>_abab t1_aa(a,k) t1_bb(b,j) t1_2p_aa(c,i) 
    //             += +2.000 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0252_aa_oo")(ka,ia) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_abab t1_bb(c,k) t1_2p_aa(d,i) t2_abab(a,b,l,j) 
    //               += +2.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_2p_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0252_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0252_aa_oo"))
    .allocate(tmps.at("0253_aaaa_vvoo"))
    
    // flops: o2v2  = o1v1Q1 o1v1Q1 o3v2 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v2 o2v2 o3v2 o3v2 o2v2
    //  mems: o2v2  = o0v0Q1 o1v1 o3v1 o2v2 o0v0Q1 o1v1 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("bin1_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = tmps.at("bin1_aa_vo")(da,la) * t2.at("aaaa")(da,aa,ia,ja) )
    ( tmps.at("0253_aaaa_vvoo")(ba,aa,ia,ja)  = tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1.at("aa")(ba,la) )
    ( tmps.at("bin1_Q")(Q)  = t1.at("bb")(cb,kb) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("bin1_aa_vo")(da,la)  = chol.at("aa_ovQ")(la,da,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = tmps.at("bin1_aa_vo")(da,la) * t2.at("aaaa")(da,aa,ia,ja) )
    ( tmps.at("0253_aaaa_vvoo")(ba,aa,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1.at("aa")(ba,la) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la)  = t2.at("aaaa")(da,ba,ia,ja) * tmps.at("0105_aa_ov")(la,da) )
    ( tmps.at("0253_aaaa_vvoo")(ba,aa,ia,ja) += t1.at("aa")(aa,la) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,la) )
    .deallocate(tmps.at("0105_aa_ov"))
    
    // r2[aaaa] += +1.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_aaaa(d,b,i,j) 
    //            += -1.000 P(a,b) <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_aaaa(d,b,i,j) 
    //            += +1.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_aaaa(d,b,i,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0253_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2[aaaa] += +1.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_aaaa(d,b,i,j) 
    //            += -1.000 P(a,b) <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_aaaa(d,b,i,j) 
    //            += +1.000 P(a,b) <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_aaaa(d,b,i,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0253_aaaa_vvoo")(ba,aa,ia,ja) )
    .deallocate(tmps.at("0253_aaaa_vvoo"))
    .allocate(tmps.at("0254_bbbb_oovv"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0254_bbbb_oovv")(kb,ib,ab,db)  = t1.at("bb")(cb,ib) * tmps.at("0086_bbbb_ovvv")(kb,cb,ab,db) )
    
    // r1[bb] += -1.000 <a,j||b,c>_bbbb t1_bb(b,j) t1_bb(c,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= tmps.at("0254_bbbb_oovv")(jb,ib,ab,bb) * t1.at("bb")(bb,jb) )
    
    // r1_1p[bb] += +1.000 <a,j||b,c>_bbbb t1_bb(b,i) t1_1p_bb(c,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0254_bbbb_oovv")(jb,ib,ab,cb) * t1_1p.at("bb")(cb,jb) )
    
    // r1_2p[bb] += +2.000 <a,j||b,c>_bbbb t1_bb(b,i) t1_2p_bb(c,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0254_bbbb_oovv")(jb,ib,ab,cb) * t1_2p.at("bb")(cb,jb) )
    
    // r2[abab] += +1.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,db,ia,kb) * tmps.at("0254_bbbb_oovv")(kb,jb,bb,db) )
    
    // r2_1p[abab] += +1.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_1p_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,db,ia,kb) * tmps.at("0254_bbbb_oovv")(kb,jb,bb,db) )
    
    // r2_2p[abab] += +2.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_2p_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_2p.at("abab")(aa,db,ia,kb) * tmps.at("0254_bbbb_oovv")(kb,jb,bb,db) )
    .allocate(tmps.at("0255_bbbb_ovvo"))
    
    // flops: o2v2  = o3v2 o3v2 o3v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = t1.at("bb")(cb,jb) * tmps.at("0063_bbbb_oovv")(kb,ib,ab,cb) )
    ( tmps.at("0255_bbbb_ovvo")(ib,bb,ab,jb)  = t1.at("bb")(bb,kb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) )
    ( tmps.at("0255_bbbb_ovvo")(ib,bb,ab,jb) += t2.at("bbbb")(db,bb,jb,kb) * tmps.at("0254_bbbb_oovv")(kb,ib,ab,db) )
    
    // r2[bbbb] += -1.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_bbbb(d,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0255_bbbb_ovvo")(ib,bb,ab,jb) )
    
    // r2[bbbb] += -1.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_bbbb(d,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0255_bbbb_ovvo")(jb,bb,ab,ib) )
    
    // r2[bbbb] += -1.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_bbbb(d,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0255_bbbb_ovvo")(ib,ab,bb,jb) )
    
    // r2[bbbb] += -1.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_bbbb(d,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0255_bbbb_ovvo")(jb,ab,bb,ib) )
    .deallocate(tmps.at("0255_bbbb_ovvo"))
    .allocate(tmps.at("0256_bbbb_oovv"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0256_bbbb_oovv")(kb,ib,ab,db)  = t1_1p.at("bb")(cb,ib) * tmps.at("0086_bbbb_ovvv")(kb,cb,ab,db) )
    
    // r1_1p[bb] += -1.000 <a,j||b,c>_bbbb t1_bb(b,j) t1_1p_bb(c,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0256_bbbb_oovv")(jb,ib,ab,bb) * t1.at("bb")(bb,jb) )
    
    // r1_2p[bb] += -2.000 <a,j||b,c>_bbbb t1_1p_bb(b,j) t1_1p_bb(c,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0256_bbbb_oovv")(jb,ib,ab,bb) * t1_1p.at("bb")(bb,jb) )
    
    // r2_1p[abab] += -1.000 <b,k||d,c>_bbbb t1_1p_bb(c,j) t2_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,db,ia,kb) * tmps.at("0256_bbbb_oovv")(kb,jb,bb,db) )
    
    // r2_2p[abab] += +2.000 <b,k||c,d>_bbbb t1_1p_bb(c,j) t2_1p_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_1p.at("abab")(aa,db,ia,kb) * tmps.at("0256_bbbb_oovv")(kb,jb,bb,db) )
    .allocate(tmps.at("0257_bbbb_ovvo"))
    
    // flops: o2v2  = o3v2 o3v2 o3v2 o3v2 o2v2 o3v3 o2v2 o3v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = t1_1p.at("bb")(cb,jb) * tmps.at("0063_bbbb_oovv")(kb,ib,ab,cb) )
    ( tmps.at("0257_bbbb_ovvo")(ib,bb,ab,jb)  = t1.at("bb")(bb,kb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = t1.at("bb")(cb,jb) * tmps.at("0063_bbbb_oovv")(kb,ib,ab,cb) )
    ( tmps.at("0257_bbbb_ovvo")(ib,bb,ab,jb) += t1_1p.at("bb")(bb,kb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) )
    ( tmps.at("0257_bbbb_ovvo")(ib,bb,ab,jb) += t2_1p.at("bbbb")(db,bb,jb,kb) * tmps.at("0254_bbbb_oovv")(kb,ib,ab,db) )
    ( tmps.at("0257_bbbb_ovvo")(ib,bb,ab,jb) += t2.at("bbbb")(db,bb,jb,kb) * tmps.at("0256_bbbb_oovv")(kb,ib,ab,db) )
    
    // r2_1p[bbbb] += -1.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_1p_bbbb(d,b,j,k) 
    //               += +1.000 P(i,j) P(a,b) <a,k||d,c>_bbbb t1_1p_bb(c,i) t2_bbbb(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_1p_bb(b,k) t1_bb(c,j) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_1p_bb(c,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0257_bbbb_ovvo")(ib,bb,ab,jb) )
    
    // r2_1p[bbbb] += -1.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_1p_bbbb(d,b,j,k) 
    //               += +1.000 P(i,j) P(a,b) <a,k||d,c>_bbbb t1_1p_bb(c,i) t2_bbbb(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_1p_bb(b,k) t1_bb(c,j) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_1p_bb(c,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0257_bbbb_ovvo")(jb,bb,ab,ib) )
    
    // r2_1p[bbbb] += -1.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_1p_bbbb(d,b,j,k) 
    //               += +1.000 P(i,j) P(a,b) <a,k||d,c>_bbbb t1_1p_bb(c,i) t2_bbbb(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_1p_bb(b,k) t1_bb(c,j) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_1p_bb(c,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0257_bbbb_ovvo")(ib,ab,bb,jb) )
    
    // r2_1p[bbbb] += -1.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_1p_bbbb(d,b,j,k) 
    //               += +1.000 P(i,j) P(a,b) <a,k||d,c>_bbbb t1_1p_bb(c,i) t2_bbbb(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_1p_bb(b,k) t1_bb(c,j) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_1p_bb(c,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0257_bbbb_ovvo")(jb,ab,bb,ib) )
    .deallocate(tmps.at("0257_bbbb_ovvo"))
    .allocate(tmps.at("0258_bbbb_ovvo"))
    
    // flops: o2v2  = o3v2 o3v2 o3v2 o3v2 o3v2 o3v2 o2v2 o2v2 o3v3 o2v3 o3v3 o2v2 o2v2 o3v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o3v1 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = t1_2p.at("bb")(cb,jb) * tmps.at("0063_bbbb_oovv")(kb,ib,ab,cb) )
    ( tmps.at("0258_bbbb_ovvo")(ib,bb,ab,jb)  = t1.at("bb")(bb,kb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = t1_1p.at("bb")(cb,jb) * tmps.at("0063_bbbb_oovv")(kb,ib,ab,cb) )
    ( tmps.at("0258_bbbb_ovvo")(ib,bb,ab,jb) += t1_1p.at("bb")(bb,kb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = t1.at("bb")(cb,jb) * tmps.at("0063_bbbb_oovv")(kb,ib,ab,cb) )
    ( tmps.at("0258_bbbb_ovvo")(ib,bb,ab,jb) += t1_2p.at("bb")(bb,kb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) )
    ( tmps.at("0258_bbbb_ovvo")(ib,bb,ab,jb) += t2_2p.at("bbbb")(db,bb,jb,kb) * tmps.at("0254_bbbb_oovv")(kb,ib,ab,db) )
    ( tmps.at("bin1_bbbb_vvoo")(ab,db,ib,kb)  = tmps.at("0086_bbbb_ovvv")(kb,cb,ab,db) * t1_2p.at("bb")(cb,ib) )
    ( tmps.at("0258_bbbb_ovvo")(ib,bb,ab,jb) += t2.at("bbbb")(db,bb,jb,kb) * tmps.at("bin1_bbbb_vvoo")(ab,db,ib,kb) )
    ( tmps.at("0258_bbbb_ovvo")(ib,bb,ab,jb) += t2_1p.at("bbbb")(db,bb,jb,kb) * tmps.at("0256_bbbb_oovv")(kb,ib,ab,db) )
    .deallocate(tmps.at("0256_bbbb_oovv"))
    .deallocate(tmps.at("0254_bbbb_oovv"))
    .deallocate(tmps.at("0063_bbbb_oovv"))
    
    // r2_2p[bbbb] += -2.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_2p_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_2p_bb(b,k) t1_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_1p_bb(b,k) t1_1p_bb(c,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||d,c>_bbbb t1_2p_bb(c,i) t2_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_2p_bb(c,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0258_bbbb_ovvo")(ib,bb,ab,jb) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_2p_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_2p_bb(b,k) t1_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_1p_bb(b,k) t1_1p_bb(c,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||d,c>_bbbb t1_2p_bb(c,i) t2_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_2p_bb(c,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0258_bbbb_ovvo")(jb,bb,ab,ib) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_2p_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_2p_bb(b,k) t1_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_1p_bb(b,k) t1_1p_bb(c,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||d,c>_bbbb t1_2p_bb(c,i) t2_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_2p_bb(c,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0258_bbbb_ovvo")(ib,ab,bb,jb) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_bb(c,i) t2_2p_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||c,d>_bbbb t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_2p_bb(b,k) t1_bb(c,j) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_1p_bb(b,k) t1_1p_bb(c,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||d,c>_bbbb t1_2p_bb(c,i) t2_bbbb(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t1_bb(b,k) t1_2p_bb(c,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0258_bbbb_ovvo")(jb,ab,bb,ib) )
    .deallocate(tmps.at("0258_bbbb_ovvo"))
    .allocate(tmps.at("0259_aaaa_oovv"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0259_aaaa_oovv")(ka,ia,aa,da)  = t1.at("aa")(ca,ia) * tmps.at("0090_aaaa_ovvv")(ka,ca,aa,da) )
    
    // r1[aa] += -1.000 <a,j||b,c>_aaaa t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0259_aaaa_oovv")(ja,ia,aa,ba) * t1.at("aa")(ba,ja) )
    
    // r1_1p[aa] += +1.000 <a,j||b,c>_aaaa t1_aa(b,i) t1_1p_aa(c,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0259_aaaa_oovv")(ja,ia,aa,ca) * t1_1p.at("aa")(ca,ja) )
    
    // r1_2p[aa] += +2.000 <a,j||b,c>_aaaa t1_aa(b,i) t1_2p_aa(c,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0259_aaaa_oovv")(ja,ia,aa,ca) * t1_2p.at("aa")(ca,ja) )
    
    // r2[abab] += +1.000 <a,k||c,d>_aaaa t1_aa(c,i) t2_abab(d,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(da,bb,ka,jb) * tmps.at("0259_aaaa_oovv")(ka,ia,aa,da) )
    
    // r2_1p[abab] += +1.000 <a,k||c,d>_aaaa t1_aa(c,i) t2_1p_abab(d,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(da,bb,ka,jb) * tmps.at("0259_aaaa_oovv")(ka,ia,aa,da) )
    
    // r2_2p[abab] += +2.000 <a,k||c,d>_aaaa t1_aa(c,i) t2_2p_abab(d,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_2p.at("abab")(da,bb,ka,jb) * tmps.at("0259_aaaa_oovv")(ka,ia,aa,da) )
    .allocate(tmps.at("0260_aaaa_oovv"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0260_aaaa_oovv")(ka,ia,aa,da)  = t1_1p.at("aa")(ca,ia) * tmps.at("0090_aaaa_ovvv")(ka,ca,aa,da) )
    
    // r1_1p[aa] += -1.000 <a,j||b,c>_aaaa t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0260_aaaa_oovv")(ja,ia,aa,ba) * t1.at("aa")(ba,ja) )
    
    // r1_2p[aa] += -2.000 <a,j||b,c>_aaaa t1_1p_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0260_aaaa_oovv")(ja,ia,aa,ba) * t1_1p.at("aa")(ba,ja) )
    
    // r2_1p[abab] += -1.000 <a,k||d,c>_aaaa t1_1p_aa(c,i) t2_abab(d,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(da,bb,ka,jb) * tmps.at("0260_aaaa_oovv")(ka,ia,aa,da) )
    
    // r2_2p[abab] += +2.000 <a,k||c,d>_aaaa t1_1p_aa(c,i) t2_1p_abab(d,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_1p.at("abab")(da,bb,ka,jb) * tmps.at("0260_aaaa_oovv")(ka,ia,aa,da) )
    .allocate(tmps.at("0261_aaaa_ovvo"))
    
    // flops: o2v2  = o3v2 o3v2 o3v1Q1 o3v2 o3v1Q1 o3v2 o2v2 o2v2 o3v2 o3v2 o2v2 o3v3 o2v2 o3v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o3v1 o2v2 o3v1 o2v2 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = t1_1p.at("aa")(ca,ja) * tmps.at("0055_aaaa_oovv")(ka,ia,aa,ca) )
    ( tmps.at("0261_aaaa_ovvo")(ia,ba,aa,ja)  = t1.at("aa")(ba,ka) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("0028_aa_voQ")(aa,ja,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0261_aaaa_ovvo")(ia,ba,aa,ja) += t1.at("aa")(ba,ka) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("0025_aa_voQ")(aa,ja,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0261_aaaa_ovvo")(ia,ba,aa,ja) += t1_1p.at("aa")(ba,ka) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = t1.at("aa")(ca,ja) * tmps.at("0055_aaaa_oovv")(ka,ia,aa,ca) )
    ( tmps.at("0261_aaaa_ovvo")(ia,ba,aa,ja) += t1_1p.at("aa")(ba,ka) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) )
    ( tmps.at("0261_aaaa_ovvo")(ia,ba,aa,ja) += t2_1p.at("aaaa")(da,ba,ja,ka) * tmps.at("0259_aaaa_oovv")(ka,ia,aa,da) )
    ( tmps.at("0261_aaaa_ovvo")(ia,ba,aa,ja) += t2.at("aaaa")(da,ba,ja,ka) * tmps.at("0260_aaaa_oovv")(ka,ia,aa,da) )
    
    // r2_1p[aaaa] += -1.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_1p_aaaa(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_1p_aa(a,k) t2_abab(b,c,j,l) 
    //               += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_1p_abab(b,c,j,l) 
    //               += +1.000 P(i,j) P(a,b) <a,k||d,c>_aaaa t1_1p_aa(c,i) t2_aaaa(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0261_aaaa_ovvo")(ia,ba,aa,ja) )
    
    // r2_1p[aaaa] += -1.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_1p_aaaa(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_1p_aa(a,k) t2_abab(b,c,j,l) 
    //               += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_1p_abab(b,c,j,l) 
    //               += +1.000 P(i,j) P(a,b) <a,k||d,c>_aaaa t1_1p_aa(c,i) t2_aaaa(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0261_aaaa_ovvo")(ja,ba,aa,ia) )
    
    // r2_1p[aaaa] += -1.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_1p_aaaa(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_1p_aa(a,k) t2_abab(b,c,j,l) 
    //               += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_1p_abab(b,c,j,l) 
    //               += +1.000 P(i,j) P(a,b) <a,k||d,c>_aaaa t1_1p_aa(c,i) t2_aaaa(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0261_aaaa_ovvo")(ia,aa,ba,ja) )
    
    // r2_1p[aaaa] += -1.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_1p_aaaa(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_1p_aa(a,k) t2_abab(b,c,j,l) 
    //               += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_1p_abab(b,c,j,l) 
    //               += +1.000 P(i,j) P(a,b) <a,k||d,c>_aaaa t1_1p_aa(c,i) t2_aaaa(d,b,j,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_1p_aa(b,k) t1_aa(c,j) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0261_aaaa_ovvo")(ja,aa,ba,ia) )
    .deallocate(tmps.at("0261_aaaa_ovvo"))
    .allocate(tmps.at("0262_aaaa_ovvo"))
    
    // flops: o2v2  = o3v1Q1 o3v2 o3v1Q1 o3v2 o3v1Q1 o3v2 o3v2 o3v2 o2v2 o2v2 o2v3 o3v3 o2v2 o2v2 o3v2 o3v2 o2v2 o3v2 o3v2 o2v2 o3v3 o2v2 o3v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o3v1 o2v2 o3v1 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0025_aa_voQ")(aa,ja,Q) )
    ( tmps.at("0262_aaaa_ovvo")(ia,ba,aa,ja)  = tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1_2p.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0028_aa_voQ")(aa,ja,Q) )
    ( tmps.at("0262_aaaa_ovvo")(ia,ba,aa,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1_1p.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0144_aa_voQ")(aa,ja,Q) )
    ( tmps.at("0262_aaaa_ovvo")(ia,ba,aa,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("0055_aaaa_oovv")(ka,ia,aa,ca) * t1_2p.at("aa")(ca,ja) )
    ( tmps.at("0262_aaaa_ovvo")(ia,ba,aa,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_vvoo")(aa,da,ia,ka)  = t1_2p.at("aa")(ca,ia) * tmps.at("0090_aaaa_ovvv")(ka,ca,aa,da) )
    ( tmps.at("0262_aaaa_ovvo")(ia,ba,aa,ja) += tmps.at("bin1_aaaa_vvoo")(aa,da,ia,ka) * t2.at("aaaa")(da,ba,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("0055_aaaa_oovv")(ka,ia,aa,ca) * t1.at("aa")(ca,ja) )
    ( tmps.at("0262_aaaa_ovvo")(ia,ba,aa,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1_2p.at("aa")(ba,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("0055_aaaa_oovv")(ka,ia,aa,ca) * t1_1p.at("aa")(ca,ja) )
    ( tmps.at("0262_aaaa_ovvo")(ia,ba,aa,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1_1p.at("aa")(ba,ka) )
    ( tmps.at("0262_aaaa_ovvo")(ia,ba,aa,ja) += tmps.at("0259_aaaa_oovv")(ka,ia,aa,da) * t2_2p.at("aaaa")(da,ba,ja,ka) )
    ( tmps.at("0262_aaaa_ovvo")(ia,ba,aa,ja) += tmps.at("0260_aaaa_oovv")(ka,ia,aa,da) * t2_1p.at("aaaa")(da,ba,ja,ka) )
    .deallocate(tmps.at("0260_aaaa_oovv"))
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_2p_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_2p_aa(a,k) t2_abab(b,c,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_1p_aa(a,k) t2_1p_abab(b,c,j,l) 
    //               += -2.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_1p_aa(c,i) t2_1p_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_2p_aa(b,k) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_1p_aa(b,k) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||d,c>_aaaa t1_2p_aa(c,i) t2_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_2p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_2p_abab(b,c,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0262_aaaa_ovvo")(ia,ba,aa,ja) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_2p_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_2p_aa(a,k) t2_abab(b,c,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_1p_aa(a,k) t2_1p_abab(b,c,j,l) 
    //               += -2.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_1p_aa(c,i) t2_1p_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_2p_aa(b,k) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_1p_aa(b,k) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||d,c>_aaaa t1_2p_aa(c,i) t2_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_2p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_2p_abab(b,c,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0262_aaaa_ovvo")(ja,ba,aa,ia) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_2p_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_2p_aa(a,k) t2_abab(b,c,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_1p_aa(a,k) t2_1p_abab(b,c,j,l) 
    //               += -2.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_1p_aa(c,i) t2_1p_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_2p_aa(b,k) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_1p_aa(b,k) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||d,c>_aaaa t1_2p_aa(c,i) t2_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_2p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_2p_abab(b,c,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0262_aaaa_ovvo")(ia,aa,ba,ja) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_2p_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_2p_aa(a,k) t2_abab(b,c,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_1p_aa(a,k) t2_1p_abab(b,c,j,l) 
    //               += -2.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_1p_aa(c,i) t2_1p_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_2p_aa(b,k) t1_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_1p_aa(b,k) t1_1p_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) <a,k||d,c>_aaaa t1_2p_aa(c,i) t2_aaaa(d,b,j,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_2p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_2p_abab(b,c,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0262_aaaa_ovvo")(ja,aa,ba,ia) )
    .deallocate(tmps.at("0262_aaaa_ovvo"))
    .allocate(tmps.at("0263_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0263_bb_oo")(ib,kb)  = tmps.at("0064_bbbb_ooov")(jb,ib,kb,bb) * t1_1p.at("bb")(bb,jb) )
    
    // r1_1p[bb] += +1.000 <k,j||i,b>_bbbb t1_bb(a,j) t1_1p_bb(b,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0263_bb_oo")(ib,jb) * t1.at("bb")(ab,jb) )
    
    // r1_2p[bb] += -2.000 <k,j||i,b>_bbbb t1_1p_bb(a,k) t1_1p_bb(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0263_bb_oo")(ib,kb) * t1_1p.at("bb")(ab,kb) )
    
    // r2_1p[abab] += +1.000 <k,l||j,c>_bbbb t1_1p_bb(c,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ia,lb) * tmps.at("0263_bb_oo")(jb,lb) )
    
    // r2_2p[abab] += -2.000 <l,k||j,c>_bbbb t1_1p_bb(c,k) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0263_bb_oo")(jb,lb) )
    .allocate(tmps.at("0264_bbbb_ovvo"))
    
    // flops: o2v2  = o3v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0264_bbbb_ovvo")(jb,ab,bb,ib)  = t2_1p.at("bbbb")(ab,bb,ib,lb) * tmps.at("0263_bb_oo")(jb,lb) )
    ( tmps.at("0264_bbbb_ovvo")(jb,ab,bb,ib) += t2_2p.at("bbbb")(ab,bb,ib,lb) * tmps.at("0113_bb_oo")(jb,lb) )
    .deallocate(tmps.at("0113_bb_oo"))
    
    // r2_2p[bbbb] += +2.000 P(i,j) <l,k||i,c>_bbbb t1_bb(c,k) t2_2p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <l,k||i,c>_bbbb t1_1p_bb(c,k) t2_1p_bbbb(a,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0264_bbbb_ovvo")(ib,ab,bb,jb) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) <l,k||i,c>_bbbb t1_bb(c,k) t2_2p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <l,k||i,c>_bbbb t1_1p_bb(c,k) t2_1p_bbbb(a,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0264_bbbb_ovvo")(jb,ab,bb,ib) )
    .deallocate(tmps.at("0264_bbbb_ovvo"))
    .allocate(tmps.at("0265_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0265_aa_oo")(ia,ka)  = tmps.at("0056_aaaa_ooov")(ja,ia,ka,ba) * t1_1p.at("aa")(ba,ja) )
    
    // r1_1p[aa] += +1.000 <k,j||i,b>_aaaa t1_aa(a,j) t1_1p_aa(b,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0265_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r1_2p[aa] += -2.000 <k,j||i,b>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0265_aa_oo")(ia,ka) * t1_1p.at("aa")(aa,ka) )
    
    // r2_1p[abab] += +1.000 <k,l||i,c>_aaaa t1_1p_aa(c,k) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0265_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||i,c>_aaaa t1_1p_aa(c,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_1p.at("abab")(aa,bb,la,jb) * tmps.at("0265_aa_oo")(ia,la) )
    .allocate(tmps.at("0266_aaaa_ovvo"))
    
    // flops: o2v2  = o3v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0266_aaaa_ovvo")(ja,aa,ba,ia)  = t2_1p.at("aaaa")(aa,ba,ia,la) * tmps.at("0265_aa_oo")(ja,la) )
    ( tmps.at("0266_aaaa_ovvo")(ja,aa,ba,ia) += t2_2p.at("aaaa")(aa,ba,ia,la) * tmps.at("0115_aa_oo")(ja,la) )
    .deallocate(tmps.at("0115_aa_oo"))
    
    // r2_2p[aaaa] += +2.000 P(i,j) <l,k||i,c>_aaaa t1_aa(c,k) t2_2p_aaaa(a,b,j,l) 
    //               += +2.000 P(i,j) <l,k||i,c>_aaaa t1_1p_aa(c,k) t2_1p_aaaa(a,b,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0266_aaaa_ovvo")(ia,aa,ba,ja) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) <l,k||i,c>_aaaa t1_aa(c,k) t2_2p_aaaa(a,b,j,l) 
    //               += +2.000 P(i,j) <l,k||i,c>_aaaa t1_1p_aa(c,k) t2_1p_aaaa(a,b,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0266_aaaa_ovvo")(ja,aa,ba,ia) )
    .deallocate(tmps.at("0266_aaaa_ovvo"))
    .allocate(tmps.at("0267_aa_oo"))
    
    // flops: o2v0  = o2v1 o2v1Q1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0267_aa_oo")(ja,ia)  = f.at("aa_ov")(ja,ba) * t1_2p.at("aa")(ba,ia) )
    ( tmps.at("0267_aa_oo")(ja,ia) += tmps.at("0144_aa_voQ")(ca,ia,Q) * chol.at("aa_ovQ")(ja,ca,Q) )
    
    // r1_2p[aa] += -1.000 <j,k||c,b>_abab t1_aa(a,j) t2_2p_abab(c,b,i,k) 
    //             += -1.000 <j,k||b,c>_abab t1_aa(a,j) t2_2p_abab(b,c,i,k) 
    //             += -2.000 f_aa(j,b) t1_aa(a,j) t1_2p_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0267_aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    
    // r2_2p[abab] += -1.000 <k,l||d,c>_abab t2_abab(a,b,k,j) t2_2p_abab(d,c,i,l) 
    //               += -1.000 <k,l||c,d>_abab t2_abab(a,b,k,j) t2_2p_abab(c,d,i,l) 
    //               += -2.000 f_aa(k,c) t1_2p_aa(c,i) t2_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0267_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    .deallocate(tmps.at("0267_aa_oo"))
    .allocate(tmps.at("0268_abab_oooo"))
    
    // flops: o4v0  = o4v2
    //  mems: o4v0  = o4v0
    ( tmps.at("0268_abab_oooo")(ka,lb,ia,jb)  = tmps.at("0083_aabb_ovov")(ka,da,lb,cb) * t2.at("abab")(da,cb,ia,jb) )
    
    // r2[abab] += +0.250 <k,l||d,c>_abab t2_abab(a,b,k,l) t2_abab(d,c,i,j) 
    //            += +0.250 <k,l||c,d>_abab t2_abab(a,b,k,l) t2_abab(c,d,i,j) 
    //            += +0.250 <l,k||d,c>_abab t2_abab(a,b,l,k) t2_abab(d,c,i,j) 
    //            += +0.250 <l,k||c,d>_abab t2_abab(a,b,l,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ka,lb) * tmps.at("0268_abab_oooo")(ka,lb,ia,jb) )
    
    // r2_1p[abab] += +0.250 <k,l||d,c>_abab t2_1p_abab(a,b,k,l) t2_abab(d,c,i,j) 
    //               += +0.250 <k,l||c,d>_abab t2_1p_abab(a,b,k,l) t2_abab(c,d,i,j) 
    //               += +0.250 <l,k||d,c>_abab t2_1p_abab(a,b,l,k) t2_abab(d,c,i,j) 
    //               += +0.250 <l,k||c,d>_abab t2_1p_abab(a,b,l,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,ka,lb) * tmps.at("0268_abab_oooo")(ka,lb,ia,jb) )
    
    // r2_2p[abab] += +0.500 <k,l||d,c>_abab t2_2p_abab(a,b,k,l) t2_abab(d,c,i,j) 
    //               += +0.500 <k,l||c,d>_abab t2_2p_abab(a,b,k,l) t2_abab(c,d,i,j) 
    //               += +0.500 <l,k||d,c>_abab t2_2p_abab(a,b,l,k) t2_abab(d,c,i,j) 
    //               += +0.500 <l,k||c,d>_abab t2_2p_abab(a,b,l,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("abab")(aa,bb,ka,lb) * tmps.at("0268_abab_oooo")(ka,lb,ia,jb) )
    
    // r2_2p[abab] += +1.000 <k,l||d,c>_abab t1_aa(a,k) t1_2p_bb(b,l) t2_abab(d,c,i,j) 
    //               += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_2p_bb(b,l) t2_abab(c,d,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0268_abab_oooo")(ka,lb,ia,jb) * t1_2p.at("bb")(bb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0269_baab_vooo"))
    
    // flops: o3v1  = o4v0Q1 o4v0Q1 o4v0 o4v1 o4v1 o3v1 o4v1 o4v1 o3v1
    //  mems: o3v1  = o4v0 o4v0 o4v0 o3v1 o3v1 o3v1 o4v0 o3v1 o3v1
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) += tmps.at("0053_aa_ooQ")(ka,ia,Q) * tmps.at("0026_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0269_baab_vooo")(bb,ka,ia,jb)  = t1.at("bb")(bb,lb) * tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) )
    ( tmps.at("0269_baab_vooo")(bb,ka,ia,jb) += t1.at("bb")(bb,lb) * tmps.at("0268_abab_oooo")(ka,lb,ia,jb) )
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb)  = t1.at("aa")(ca,ia) * tmps.at("0211_aabb_ovoo")(ka,ca,lb,jb) )
    ( tmps.at("0269_baab_vooo")(bb,ka,ia,jb) += t1.at("bb")(bb,lb) * tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) )
    
    // r2[abab] += +1.000 <k,l||c,j>_abab t1_aa(a,k) t1_bb(b,l) t1_aa(c,i) 
    //            += +1.000 <k,l||i,j>_abab t1_aa(a,k) t1_bb(b,l) 
    //            += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t1_aa(c,i) t1_bb(d,j) 
    //            += +0.500 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t2_abab(d,c,i,j) 
    //            += +0.500 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t2_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0269_baab_vooo")(bb,ka,ia,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += +1.000 <l,k||c,j>_abab t1_1p_aa(a,l) t1_bb(b,k) t1_aa(c,i) 
    //               += +1.000 <l,k||i,j>_abab t1_1p_aa(a,l) t1_bb(b,k) 
    //               += +1.000 <l,k||c,d>_abab t1_1p_aa(a,l) t1_bb(b,k) t1_aa(c,i) t1_bb(d,j) 
    //               += +0.500 <l,k||d,c>_abab t1_1p_aa(a,l) t1_bb(b,k) t2_abab(d,c,i,j) 
    //               += +0.500 <l,k||c,d>_abab t1_1p_aa(a,l) t1_bb(b,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0269_baab_vooo")(bb,la,ia,jb) * t1_1p.at("aa")(aa,la) )
    
    // r2_2p[abab] += +2.000 <l,k||c,j>_abab t1_2p_aa(a,l) t1_bb(b,k) t1_aa(c,i) 
    //               += +2.000 <l,k||i,j>_abab t1_2p_aa(a,l) t1_bb(b,k) 
    //               += +2.000 <l,k||c,d>_abab t1_2p_aa(a,l) t1_bb(b,k) t1_aa(c,i) t1_bb(d,j) 
    //               += +1.000 <l,k||d,c>_abab t1_2p_aa(a,l) t1_bb(b,k) t2_abab(d,c,i,j) 
    //               += +1.000 <l,k||c,d>_abab t1_2p_aa(a,l) t1_bb(b,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0269_baab_vooo")(bb,la,ia,jb) * t1_2p.at("aa")(aa,la) )
    .deallocate(tmps.at("0269_baab_vooo"))
    .allocate(tmps.at("0270_baab_vooo"))
    
    // flops: o3v1  = o3v2 o4v0Q1 o4v0Q1 o4v0 o4v1 o3v1 o4v1 o4v1 o3v2 o4v2 o3v1 o3v1 o4v1 o3v1 o3v2 o4v2 o3v1
    //  mems: o3v1  = o3v1 o4v0 o4v0 o4v0 o3v1 o3v1 o4v0 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1 o3v1
    ( tmps.at("0270_baab_vooo")(bb,ka,ia,jb)  = t1_1p.at("aa")(ca,ia) * tmps.at("0177_baba_voov")(bb,ka,jb,ca) )
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) += tmps.at("0053_aa_ooQ")(ka,ia,Q) * tmps.at("0026_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0270_baab_vooo")(bb,ka,ia,jb) += t1_1p.at("bb")(bb,lb) * tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) )
    ( tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb)  = t1.at("aa")(ca,ia) * tmps.at("0211_aabb_ovoo")(ka,ca,lb,jb) )
    ( tmps.at("0270_baab_vooo")(bb,ka,ia,jb) += t1_1p.at("bb")(bb,lb) * tmps.at("bin1_aabb_oooo")(ia,ka,jb,lb) )
    ( tmps.at("bin1_aabb_vooo")(da,ka,jb,lb)  = t1.at("bb")(cb,jb) * tmps.at("0083_aabb_ovov")(ka,da,lb,cb) )
    ( tmps.at("0270_baab_vooo")(bb,ka,ia,jb) += tmps.at("bin1_aabb_vooo")(da,ka,jb,lb) * t2_1p.at("abab")(da,bb,ia,lb) )
    ( tmps.at("0270_baab_vooo")(bb,ka,ia,jb) += t1_1p.at("bb")(bb,lb) * tmps.at("0268_abab_oooo")(ka,lb,ia,jb) )
    ( tmps.at("bin1_aabb_vooo")(da,ka,jb,lb)  = tmps.at("0083_aabb_ovov")(ka,da,lb,cb) * t1_1p.at("bb")(cb,jb) )
    ( tmps.at("0270_baab_vooo")(bb,ka,ia,jb) += t2.at("abab")(da,bb,ia,lb) * tmps.at("bin1_aabb_vooo")(da,ka,jb,lb) )
    .deallocate(tmps.at("0268_abab_oooo"))
    .deallocate(tmps.at("0177_baba_voov"))
    .deallocate(tmps.at("0083_aabb_ovov"))
    
    // r2_1p[abab] += +1.000 <k,l||i,j>_abab t1_aa(a,k) t1_1p_bb(b,l) 
    //               += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_aa(c,i) t1_bb(d,j) 
    //               += +1.000 <k,l||c,j>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_aa(c,i) 
    //               += -1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_abab(d,b,l,j) 
    //               += +0.500 <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(b,l) t2_abab(d,c,i,j) 
    //               += +0.500 <k,l||c,d>_abab t1_aa(a,k) t1_1p_bb(b,l) t2_abab(c,d,i,j) 
    //               += +1.000 <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(c,j) t2_abab(d,b,i,l) 
    //               += +1.000 <k,l||d,c>_abab t1_aa(a,k) t1_bb(c,j) t2_1p_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0270_baab_vooo")(bb,ka,ia,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||i,j>_abab t1_1p_aa(a,k) t1_1p_bb(b,l) 
    //               += +2.000 <k,l||c,d>_abab t1_1p_aa(a,k) t1_1p_bb(b,l) t1_aa(c,i) t1_bb(d,j) 
    //               += +2.000 <k,l||c,j>_abab t1_1p_aa(a,k) t1_1p_bb(b,l) t1_aa(c,i) 
    //               += +2.000 <k,l||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(c,i) t2_abab(d,b,l,j) 
    //               += +1.000 <k,l||d,c>_abab t1_1p_aa(a,k) t1_1p_bb(b,l) t2_abab(d,c,i,j) 
    //               += +1.000 <k,l||c,d>_abab t1_1p_aa(a,k) t1_1p_bb(b,l) t2_abab(c,d,i,j) 
    //               += +2.000 <k,l||d,c>_abab t1_1p_aa(a,k) t1_1p_bb(c,j) t2_abab(d,b,i,l) 
    //               += +2.000 <k,l||d,c>_abab t1_1p_aa(a,k) t1_bb(c,j) t2_1p_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0270_baab_vooo")(bb,ka,ia,jb) * t1_1p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0270_baab_vooo"))
    .allocate(tmps.at("0271_aaaa_oooo"))
    
    // flops: o4v0  = o4v2 o4v2 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0
    ( tmps.at("0271_aaaa_oooo")(la,ia,ja,ka)  = -1.000 * t2.at("aaaa")(da,ca,ia,ja) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("0271_aaaa_oooo")(la,ia,ja,ka) += t2.at("aaaa")(da,ca,ia,ja) * tmps.at("0073_aaaa_ovov")(la,ca,ka,da) )
    
    // r2[aaaa] += +0.250 <k,l||d,c>_aaaa t2_aaaa(a,b,k,l) t2_aaaa(d,c,i,j) 
    //            += +0.250 <k,l||d,c>_aaaa t2_aaaa(a,b,k,l) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) -= 0.250 * tmps.at("0271_aaaa_oooo")(ka,ia,ja,la) * t2.at("aaaa")(aa,ba,ka,la) )
    
    // r2[aaaa] += -0.500 <l,k||d,c>_aaaa t1_aa(a,k) t1_aa(b,l) t2_aaaa(d,c,i,j) 
    //            += -0.500 <l,k||d,c>_aaaa t1_aa(a,k) t1_aa(b,l) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = t1.at("aa")(aa,ka) * tmps.at("0271_aaaa_oooo")(la,ia,ja,ka) )
    ( r2.at("aaaa")(aa,ba,ia,ja) += 0.500 * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1.at("aa")(ba,la) )
    
    // r2_1p[aaaa] += +0.250 <k,l||d,c>_aaaa t2_1p_aaaa(a,b,k,l) t2_aaaa(d,c,i,j) 
    //               += +0.250 <k,l||d,c>_aaaa t2_1p_aaaa(a,b,k,l) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 0.250 * tmps.at("0271_aaaa_oooo")(ka,ia,ja,la) * t2_1p.at("aaaa")(aa,ba,ka,la) )
    
    // r2_2p[aaaa] += -1.000 <l,k||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,l) t2_aaaa(d,c,i,j) 
    //               += -1.000 <l,k||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,l) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = tmps.at("0271_aaaa_oooo")(la,ia,ja,ka) * t1_1p.at("aa")(aa,ka) )
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += t1_1p.at("aa")(ba,la) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) )
    
    // r2_2p[aaaa] += +0.500 <k,l||d,c>_aaaa t2_2p_aaaa(a,b,k,l) t2_aaaa(d,c,i,j) 
    //               += +0.500 <k,l||d,c>_aaaa t2_2p_aaaa(a,b,k,l) t2_aaaa(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 0.500 * tmps.at("0271_aaaa_oooo")(ka,ia,ja,la) * t2_2p.at("aaaa")(aa,ba,ka,la) )
    .deallocate(tmps.at("0271_aaaa_oooo"))
    .allocate(tmps.at("0272_bbbb_oooo"))
    
    // flops: o4v0  = o4v2 o4v2 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0
    ( tmps.at("0272_bbbb_oooo")(lb,ib,jb,kb)  = -1.000 * t2.at("bbbb")(db,cb,ib,jb) * tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) )
    ( tmps.at("0272_bbbb_oooo")(lb,ib,jb,kb) += t2.at("bbbb")(db,cb,ib,jb) * tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) )
    
    // r2[bbbb] += +0.250 <k,l||d,c>_bbbb t2_bbbb(a,b,k,l) t2_bbbb(d,c,i,j) 
    //            += +0.250 <k,l||d,c>_bbbb t2_bbbb(a,b,k,l) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) -= 0.250 * tmps.at("0272_bbbb_oooo")(kb,ib,jb,lb) * t2.at("bbbb")(ab,bb,kb,lb) )
    
    // r2[bbbb] += -0.500 <l,k||d,c>_bbbb t1_bb(a,k) t1_bb(b,l) t2_bbbb(d,c,i,j) 
    //            += -0.500 <l,k||d,c>_bbbb t1_bb(a,k) t1_bb(b,l) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = t1.at("bb")(ab,kb) * tmps.at("0272_bbbb_oooo")(lb,ib,jb,kb) )
    ( r2.at("bbbb")(ab,bb,ib,jb) += 0.500 * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) * t1.at("bb")(bb,lb) )
    
    // r2_1p[bbbb] += +0.250 <k,l||d,c>_bbbb t2_1p_bbbb(a,b,k,l) t2_bbbb(d,c,i,j) 
    //               += +0.250 <k,l||d,c>_bbbb t2_1p_bbbb(a,b,k,l) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 0.250 * tmps.at("0272_bbbb_oooo")(kb,ib,jb,lb) * t2_1p.at("bbbb")(ab,bb,kb,lb) )
    
    // r2_2p[bbbb] += -1.000 <l,k||d,c>_bbbb t1_1p_bb(a,k) t1_1p_bb(b,l) t2_bbbb(d,c,i,j) 
    //               += -1.000 <l,k||d,c>_bbbb t1_1p_bb(a,k) t1_1p_bb(b,l) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = tmps.at("0272_bbbb_oooo")(lb,ib,jb,kb) * t1_1p.at("bb")(ab,kb) )
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += t1_1p.at("bb")(bb,lb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) )
    
    // r2_2p[bbbb] += +0.500 <k,l||d,c>_bbbb t2_2p_bbbb(a,b,k,l) t2_bbbb(d,c,i,j) 
    //               += +0.500 <k,l||d,c>_bbbb t2_2p_bbbb(a,b,k,l) t2_bbbb(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 0.500 * tmps.at("0272_bbbb_oooo")(kb,ib,jb,lb) * t2_2p.at("bbbb")(ab,bb,kb,lb) )
    .deallocate(tmps.at("0272_bbbb_oooo"))
    .allocate(tmps.at("0273_aa_oo"))
    
    // flops: o2v0  = o3v2 o3v2 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0273_aa_oo")(ia,ja)  = -1.000 * t2_1p.at("aaaa")(ca,ba,ia,ka) * tmps.at("0073_aaaa_ovov")(ka,ca,ja,ba) )
    ( tmps.at("0273_aa_oo")(ia,ja) += t2_1p.at("aaaa")(ca,ba,ia,ka) * tmps.at("0073_aaaa_ovov")(ka,ba,ja,ca) )
    
    // r1_1p[aa] += +0.500 <k,j||c,b>_aaaa t1_aa(a,j) t2_1p_aaaa(c,b,i,k) 
    //             += +0.500 <k,j||c,b>_aaaa t1_aa(a,j) t2_1p_aaaa(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.500 * tmps.at("0273_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r1_2p[aa] += +1.000 <k,j||c,b>_aaaa t1_1p_aa(a,j) t2_1p_aaaa(c,b,i,k) 
    //             += +1.000 <k,j||c,b>_aaaa t1_1p_aa(a,j) t2_1p_aaaa(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= tmps.at("0273_aa_oo")(ia,ja) * t1_1p.at("aa")(aa,ja) )
    
    // r2_1p[abab] += +0.500 <l,k||d,c>_aaaa t2_abab(a,b,k,j) t2_1p_aaaa(d,c,i,l) 
    //               += +0.500 <l,k||d,c>_aaaa t2_abab(a,b,k,j) t2_1p_aaaa(d,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.500 * tmps.at("0273_aa_oo")(ia,ka) * t2.at("abab")(aa,bb,ka,jb) )
    .deallocate(tmps.at("0273_aa_oo"))
    .allocate(tmps.at("0274_aaaa_oooo"))
    
    // flops: o4v0  = o4v2 o4v2 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0
    ( tmps.at("0274_aaaa_oooo")(la,ia,ja,ka)  = -1.000 * t2_1p.at("aaaa")(da,ca,ia,ja) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("0274_aaaa_oooo")(la,ia,ja,ka) += t2_1p.at("aaaa")(da,ca,ia,ja) * tmps.at("0073_aaaa_ovov")(la,ca,ka,da) )
    
    // r2_1p[aaaa] += +0.250 <k,l||d,c>_aaaa t2_aaaa(a,b,k,l) t2_1p_aaaa(d,c,i,j) 
    //               += +0.250 <k,l||d,c>_aaaa t2_aaaa(a,b,k,l) t2_1p_aaaa(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 0.250 * tmps.at("0274_aaaa_oooo")(ka,ia,ja,la) * t2.at("aaaa")(aa,ba,ka,la) )
    
    // r2_1p[aaaa] += -0.500 <l,k||d,c>_aaaa t1_aa(a,k) t1_aa(b,l) t2_1p_aaaa(d,c,i,j) 
    //               += -0.500 <l,k||d,c>_aaaa t1_aa(a,k) t1_aa(b,l) t2_1p_aaaa(d,c,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = t1.at("aa")(aa,ka) * tmps.at("0274_aaaa_oooo")(la,ia,ja,ka) )
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += 0.500 * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1.at("aa")(ba,la) )
    
    // r2_2p[aaaa] += +0.500 <k,l||d,c>_aaaa t2_1p_aaaa(a,b,k,l) t2_1p_aaaa(d,c,i,j) 
    //               += +0.500 <k,l||d,c>_aaaa t2_1p_aaaa(a,b,k,l) t2_1p_aaaa(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 0.500 * tmps.at("0274_aaaa_oooo")(ka,ia,ja,la) * t2_1p.at("aaaa")(aa,ba,ka,la) )
    .deallocate(tmps.at("0274_aaaa_oooo"))
    .allocate(tmps.at("0275_bb_oo"))
    
    // flops: o2v0  = o3v2 o3v2 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0275_bb_oo")(ib,jb)  = -1.000 * t2_1p.at("bbbb")(cb,bb,ib,kb) * tmps.at("0075_bbbb_ovov")(kb,cb,jb,bb) )
    ( tmps.at("0275_bb_oo")(ib,jb) += t2_1p.at("bbbb")(cb,bb,ib,kb) * tmps.at("0075_bbbb_ovov")(kb,bb,jb,cb) )
    
    // r1_1p[bb] += +0.500 <k,j||c,b>_bbbb t1_bb(a,j) t2_1p_bbbb(c,b,i,k) 
    //             += +0.500 <k,j||c,b>_bbbb t1_bb(a,j) t2_1p_bbbb(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= 0.500 * tmps.at("0275_bb_oo")(ib,jb) * t1.at("bb")(ab,jb) )
    
    // r1_2p[bb] += +1.000 <k,j||c,b>_bbbb t1_1p_bb(a,j) t2_1p_bbbb(c,b,i,k) 
    //             += +1.000 <k,j||c,b>_bbbb t1_1p_bb(a,j) t2_1p_bbbb(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= tmps.at("0275_bb_oo")(ib,jb) * t1_1p.at("bb")(ab,jb) )
    
    // r2_1p[abab] += +0.500 <l,k||d,c>_bbbb t2_abab(a,b,i,k) t2_1p_bbbb(d,c,j,l) 
    //               += +0.500 <l,k||d,c>_bbbb t2_abab(a,b,i,k) t2_1p_bbbb(d,c,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.500 * tmps.at("0275_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    .deallocate(tmps.at("0275_bb_oo"))
    .allocate(tmps.at("0276_bbbb_oooo"))
    
    // flops: o4v0  = o4v2 o4v2 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0
    ( tmps.at("0276_bbbb_oooo")(lb,ib,jb,kb)  = -1.000 * t2_1p.at("bbbb")(db,cb,ib,jb) * tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) )
    ( tmps.at("0276_bbbb_oooo")(lb,ib,jb,kb) += t2_1p.at("bbbb")(db,cb,ib,jb) * tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) )
    
    // r2_1p[bbbb] += +0.250 <k,l||d,c>_bbbb t2_bbbb(a,b,k,l) t2_1p_bbbb(d,c,i,j) 
    //               += +0.250 <k,l||d,c>_bbbb t2_bbbb(a,b,k,l) t2_1p_bbbb(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 0.250 * tmps.at("0276_bbbb_oooo")(kb,ib,jb,lb) * t2.at("bbbb")(ab,bb,kb,lb) )
    
    // r2_1p[bbbb] += -0.500 <l,k||d,c>_bbbb t1_bb(a,k) t1_bb(b,l) t2_1p_bbbb(d,c,i,j) 
    //               += -0.500 <l,k||d,c>_bbbb t1_bb(a,k) t1_bb(b,l) t2_1p_bbbb(d,c,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = t1.at("bb")(ab,kb) * tmps.at("0276_bbbb_oooo")(lb,ib,jb,kb) )
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 0.500 * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) * t1.at("bb")(bb,lb) )
    
    // r2_2p[bbbb] += +0.500 <k,l||d,c>_bbbb t2_1p_bbbb(a,b,k,l) t2_1p_bbbb(d,c,i,j) 
    //               += +0.500 <k,l||d,c>_bbbb t2_1p_bbbb(a,b,k,l) t2_1p_bbbb(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 0.500 * tmps.at("0276_bbbb_oooo")(kb,ib,jb,lb) * t2_1p.at("bbbb")(ab,bb,kb,lb) )
    .deallocate(tmps.at("0276_bbbb_oooo"))
    .allocate(tmps.at("0277_aa_oo"))
    
    // flops: o2v0  = o3v2 o3v2 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0277_aa_oo")(ia,ja)  = -1.000 * t2_2p.at("aaaa")(ca,ba,ia,ka) * tmps.at("0073_aaaa_ovov")(ka,ca,ja,ba) )
    ( tmps.at("0277_aa_oo")(ia,ja) += t2_2p.at("aaaa")(ca,ba,ia,ka) * tmps.at("0073_aaaa_ovov")(ka,ba,ja,ca) )
    
    // r1_2p[aa] += +1.000 <k,j||c,b>_aaaa t1_aa(a,j) t2_2p_aaaa(c,b,i,k) 
    //             += +1.000 <k,j||c,b>_aaaa t1_aa(a,j) t2_2p_aaaa(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= tmps.at("0277_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r2_2p[abab] += +1.000 <l,k||d,c>_aaaa t2_abab(a,b,k,j) t2_2p_aaaa(d,c,i,l) 
    //               += +1.000 <l,k||d,c>_aaaa t2_abab(a,b,k,j) t2_2p_aaaa(d,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("0277_aa_oo")(ia,ka) * t2.at("abab")(aa,bb,ka,jb) )
    .deallocate(tmps.at("0277_aa_oo"))
    .allocate(tmps.at("0278_bb_oo"))
    
    // flops: o2v0  = o3v2 o2v1Q1 o2v0 o3v2 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0 o2v0 o2v0
    ( tmps.at("0278_bb_oo")(ib,jb)  = -1.000 * tmps.at("0075_bbbb_ovov")(kb,cb,jb,bb) * t2_2p.at("bbbb")(cb,bb,ib,kb) )
    ( tmps.at("0278_bb_oo")(ib,jb) += 2.000 * chol.at("bb_ovQ")(jb,bb,Q) * tmps.at("0186_bb_voQ")(bb,ib,Q) )
    ( tmps.at("0278_bb_oo")(ib,jb) += tmps.at("0075_bbbb_ovov")(kb,bb,jb,cb) * t2_2p.at("bbbb")(cb,bb,ib,kb) )
    
    // r1_2p[bb] += +1.000 <k,j||c,b>_bbbb t1_bb(a,j) t2_2p_bbbb(c,b,i,k) 
    //             += +1.000 <k,j||c,b>_bbbb t1_bb(a,j) t2_2p_bbbb(c,b,i,k) 
    //             += -1.000 <k,j||c,b>_abab t1_bb(a,j) t2_2p_abab(c,b,k,i) 
    //             += -1.000 <k,j||b,c>_abab t1_bb(a,j) t2_2p_abab(b,c,k,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= tmps.at("0278_bb_oo")(ib,jb) * t1.at("bb")(ab,jb) )
    
    // r2_2p[abab] += +1.000 <l,k||d,c>_bbbb t2_abab(a,b,i,k) t2_2p_bbbb(d,c,j,l) 
    //               += +1.000 <l,k||d,c>_bbbb t2_abab(a,b,i,k) t2_2p_bbbb(d,c,j,l) 
    //               += -1.000 <l,k||d,c>_abab t2_abab(a,b,i,k) t2_2p_abab(d,c,l,j) 
    //               += -1.000 <l,k||c,d>_abab t2_abab(a,b,i,k) t2_2p_abab(c,d,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("0278_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    .deallocate(tmps.at("0278_bb_oo"))
    .allocate(tmps.at("0279_aa_oo"))
    
    // flops: o2v0  = o3v2 o3v2 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0279_aa_oo")(ja,ia)  = -1.000 * t2.at("aaaa")(ca,ba,ia,ka) * tmps.at("0073_aaaa_ovov")(ja,ca,ka,ba) )
    ( tmps.at("0279_aa_oo")(ja,ia) += t2.at("aaaa")(ca,ba,ia,ka) * tmps.at("0073_aaaa_ovov")(ja,ba,ka,ca) )
    
    // r1_1p[aa] += -0.500 <j,k||c,b>_aaaa t1_1p_aa(a,j) t2_aaaa(c,b,i,k) 
    //             += -0.500 <j,k||c,b>_aaaa t1_1p_aa(a,j) t2_aaaa(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 0.500 * tmps.at("0279_aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    
    // r1_2p[aa] += -1.000 <j,k||c,b>_aaaa t1_2p_aa(a,j) t2_aaaa(c,b,i,k) 
    //             += -1.000 <j,k||c,b>_aaaa t1_2p_aa(a,j) t2_aaaa(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += tmps.at("0279_aa_oo")(ja,ia) * t1_2p.at("aa")(aa,ja) )
    
    // r2[abab] += -0.500 <l,k||d,c>_aaaa t2_abab(a,b,l,j) t2_aaaa(d,c,i,k) 
    //            += -0.500 <l,k||d,c>_aaaa t2_abab(a,b,l,j) t2_aaaa(d,c,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += 0.500 * tmps.at("0279_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_1p[abab] += -0.500 <l,k||d,c>_aaaa t2_1p_abab(a,b,l,j) t2_aaaa(d,c,i,k) 
    //               += -0.500 <l,k||d,c>_aaaa t2_1p_abab(a,b,l,j) t2_aaaa(d,c,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.500 * tmps.at("0279_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_aaaa t2_2p_abab(a,b,l,j) t2_aaaa(d,c,i,k) 
    //               += -1.000 <l,k||d,c>_aaaa t2_2p_abab(a,b,l,j) t2_aaaa(d,c,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += tmps.at("0279_aa_oo")(la,ia) * t2_2p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0279_aa_oo"))
    .allocate(tmps.at("0280_bb_vv"))
    
    // flops: o0v2  = o1v2Q1 o0v2
    //  mems: o0v2  = o0v2 o0v2
    ( tmps.at("0280_bb_vv")(bb,db)  = -1.000 * tmps.at("0182_bb_vv")(bb,db) )
    ( tmps.at("0280_bb_vv")(bb,db) += chol.at("bb_ovQ")(lb,db,Q) * tmps.at("0058_bb_voQ")(bb,lb,Q) )
    .deallocate(tmps.at("0182_bb_vv"))
    
    // r2[abab] += -0.500 <k,l||c,d>_abab t2_abab(a,d,i,j) t2_abab(c,b,k,l) 
    //            += -0.500 <l,k||c,d>_abab t2_abab(a,d,i,j) t2_abab(c,b,l,k) 
    //            += -1.000 <b,k||c,d>_bbbb t1_bb(c,k) t2_abab(a,d,i,j) 
    //            += +1.000 <k,b||c,d>_abab t1_aa(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0280_bb_vv")(bb,db) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_1p[abab] += -0.500 <k,l||c,d>_abab t2_1p_abab(a,d,i,j) t2_abab(c,b,k,l) 
    //               += -0.500 <l,k||c,d>_abab t2_1p_abab(a,d,i,j) t2_abab(c,b,l,k) 
    //               += -1.000 <b,k||c,d>_bbbb t1_bb(c,k) t2_1p_abab(a,d,i,j) 
    //               += +1.000 <k,b||c,d>_abab t1_aa(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0280_bb_vv")(bb,db) * t2_1p.at("abab")(aa,db,ia,jb) )
    
    // r2_2p[abab] += -1.000 <k,l||c,d>_abab t2_2p_abab(a,d,i,j) t2_abab(c,b,k,l) 
    //               += -1.000 <l,k||c,d>_abab t2_2p_abab(a,d,i,j) t2_abab(c,b,l,k) 
    //               += -2.000 <b,k||c,d>_bbbb t1_bb(c,k) t2_2p_abab(a,d,i,j) 
    //               += +2.000 <k,b||c,d>_abab t1_aa(c,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0280_bb_vv")(bb,db) * t2_2p.at("abab")(aa,db,ia,jb) )
    .deallocate(tmps.at("0280_bb_vv"))
    .allocate(tmps.at("0281_bb_vo"))
    ;
  }
  // clang-format on
}

template void exachem::cc::cd_qed_ccsd_os::resid_part6<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);