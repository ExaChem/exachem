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
        
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0191_aa_oo")(la,ia) * t2_2p.at("abab")(aa,bb,la,jb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_aaaa t2_abab(a,b,l,j) t1_aa(c,k) t1_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0191_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0191_aa_oo"))
    .allocate(tmps.at("0192_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0192_aa_oo")(ja,ia)  = tmps.at("0098_aa_ooQ")(ja,ia,Q) * tmps.at("0185_Q")(Q) )
    
    // r1_1p.at("aa") += -1.00 <j,k||b,c>_abab t1_aa(a,j) t1_aa(b,i) t1_1p_bb(c,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1.at("aa")(aa,ja) * tmps.at("0192_aa_oo")(ja,ia) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_abab t1_aa(b,i) t1_1p_aa(a,k) t1_1p_bb(c,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0192_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += -1.00 <k,l||c,d>_abab t2_abab(a,b,k,j) t1_aa(c,i) t1_1p_bb(d,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0192_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_abab t1_aa(c,i) t2_1p_abab(a,b,l,j) t1_1p_bb(d,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0192_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0192_aa_oo"))
    .allocate(tmps.at("0193_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0193_aa_oo")(ka,ia)  = tmps.at("0104_aa_ooQ")(ka,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1.at("aa")(aa,ka) * tmps.at("0193_aa_oo")(ka,ia) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t1_aa(b,j) t1_1p_aa(a,k) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0193_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t2_abab(a,b,l,j) t1_aa(c,k) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0193_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(c,k) t2_1p_abab(a,b,l,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0193_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0193_aa_oo"))
    .allocate(tmps.at("0194_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0194_aa_oo")(ja,ia)  = chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0185_Q")(Q) )
    
    // r1_1p.at("aa") += -1.00 <j,k||i,b>_abab t1_aa(a,j) t1_1p_bb(b,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1.at("aa")(aa,ja) * tmps.at("0194_aa_oo")(ja,ia) )
    
    // r1_2p.at("aa") += -2.00 <k,j||i,b>_abab t1_1p_aa(a,k) t1_1p_bb(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0194_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += -1.00 <k,l||i,c>_abab t2_abab(a,b,k,j) t1_1p_bb(c,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0194_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||i,c>_abab t2_1p_abab(a,b,l,j) t1_1p_bb(c,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0194_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0194_aa_oo"))
    .allocate(tmps.at("0195_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0195_aa_oo")(ka,ia)  = tmps.at("0104_aa_ooQ")(ka,ia,Q) * tmps.at("0181_Q")(Q) )
    
    // r1_1p.at("aa") += -1.00 <k,j||c,b>_abab t1_aa(a,k) t1_bb(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1.at("aa")(aa,ka) * tmps.at("0195_aa_oo")(ka,ia) )
    
    // r1_2p.at("aa") += -2.00 <k,j||c,b>_abab t1_bb(b,j) t1_1p_aa(a,k) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0195_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += -1.00 <l,k||d,c>_abab t2_abab(a,b,l,j) t1_bb(c,k) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0195_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||d,c>_abab t1_bb(c,k) t2_1p_abab(a,b,l,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0195_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0195_aa_oo"))
    .allocate(tmps.at("0196_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0196_aa_oo")(ja,ia)  = chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0183_Q")(Q) )
    
    // r1_1p.at("aa") += -1.00 <k,j||b,i>_aaaa t1_aa(a,j) t1_1p_aa(b,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1.at("aa")(aa,ja) * tmps.at("0196_aa_oo")(ja,ia) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,i>_aaaa t1_1p_aa(a,k) t1_1p_aa(b,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0196_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,i>_aaaa t2_abab(a,b,k,j) t1_1p_aa(c,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0196_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,i>_aaaa t2_1p_abab(a,b,l,j) t1_1p_aa(c,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0196_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0196_aa_oo"))
    .allocate(tmps.at("0197_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0197_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1_2p.at("bb")(cb,kb) )
    
    // r2_2p.at("abab") += +2.00 <a,k||c,d>_abab t2_abab(c,b,i,j) t1_2p_bb(d,k) 
    // flops: o2v2 += o0v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_aa_vv")(aa,ca)  = chol.at("aa_vvQ")(aa,ca,Q) * tmps.at("0197_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 <k,b||c,d>_bbbb t2_abab(a,c,i,j) t1_2p_bb(d,k) 
    // flops: o2v2 += o0v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = chol.at("bb_vvQ")(bb,cb,Q) * tmps.at("0197_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 <k,l||c,d>_abab t1_aa(a,k) t2_abab(c,b,i,j) t1_2p_bb(d,l) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_aa_vo")(ca,ka)  = chol.at("aa_ovQ")(ka,ca,Q) * tmps.at("0197_Q")(Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aa_vo")(ca,ka) * t2.at("abab")(ca,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,c,i,j) t1_bb(b,k) t1_2p_bb(d,l) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_bb_vo")(cb,kb)  = chol.at("bb_ovQ")(kb,cb,Q) * tmps.at("0197_Q")(Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bb_vo")(cb,kb) * t2.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r1_2p.at("aa") += -2.00 <j,k||b,c>_abab t2_aaaa(b,a,i,j) t1_2p_bb(c,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0197_Q")(Q) )
    
    // r1_2p.at("aa") += +2.00 <a,j||i,b>_abab t1_2p_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0197_Q")(Q) )
    
    // r1_2p.at("aa") += +2.00 <a,j||b,c>_abab t1_aa(b,i) t1_2p_bb(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0197_Q")(Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,b,i,k) t1_bb(c,j) t1_2p_bb(d,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = tmps.at("0107_bb_ooQ")(kb,jb,Q) * tmps.at("0197_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,j>_bbbb t2_abab(a,b,i,k) t1_2p_bb(c,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = chol.at("bb_ooQ")(kb,jb,Q) * tmps.at("0197_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    .allocate(tmps.at("0198_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0198_aa_oo")(ja,ia)  = tmps.at("0098_aa_ooQ")(ja,ia,Q) * tmps.at("0197_Q")(Q) )
    
    // r1_2p.at("aa") += -2.00 <j,k||b,c>_abab t1_aa(a,j) t1_aa(b,i) t1_2p_bb(c,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1.at("aa")(aa,ja) * tmps.at("0198_aa_oo")(ja,ia) )
    
    // r2_2p.at("abab") += -2.00 <k,l||c,d>_abab t2_abab(a,b,k,j) t1_aa(c,i) t1_2p_bb(d,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0198_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    .deallocate(tmps.at("0198_aa_oo"))
    .allocate(tmps.at("0199_aa_ooQ"))
    
    // flops: o2v0Q1  = o2v1Q1
    //  mems: o2v0Q1  = o2v0Q1
    ( tmps.at("0199_aa_ooQ")(ja,ka,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t1_2p.at("aa")(ca,ka) )
    
    // r2_2p.at("abab") += +1.00 <l,k||d,c>_abab t2_abab(a,b,l,k) t1_bb(c,j) t1_2p_aa(d,i) 
    //            += +1.00 <k,l||d,c>_abab t2_abab(a,b,k,l) t1_bb(c,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o4v0Q1 o4v2
    //  mems: o2v2 += o4v0 o2v2
    ( tmps.at("bin_aabb_oooo")(ia,la,jb,kb)  = tmps.at("0199_aa_ooQ")(la,ia,Q) * tmps.at("0107_bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_oooo")(ia,la,jb,kb) * t2.at("abab")(aa,bb,la,kb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t1_bb(c,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o4v0Q1 o4v1 o3v2
    //  mems: o2v2 += o4v0 o3v1 o2v2
    ( tmps.at("bin_aabb_oooo")(ia,ka,jb,lb)  = tmps.at("0199_aa_ooQ")(ka,ia,Q) * tmps.at("0107_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aabb_oooo")(ia,ka,jb,lb) * t1.at("bb")(bb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += -2.00 <k,b||d,c>_abab t1_aa(a,k) t1_bb(c,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0097_bb_voQ")(bb,jb,Q) * tmps.at("0199_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t1_aa(a,k) t2_bbbb(c,b,j,l) t1_2p_aa(d,i) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0100_bb_voQ")(bb,jb,Q) * tmps.at("0199_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += -2.00 <k,b||c,j>_abab t1_aa(a,k) t1_2p_aa(c,i) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = chol.at("bb_voQ")(bb,jb,Q) * tmps.at("0199_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_aaaa t1_aa(a,k) t2_abab(c,b,l,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0102_bb_voQ")(bb,jb,Q) * tmps.at("0199_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r1_2p.at("aa") += -2.00 <k,j||c,b>_abab t1_aa(a,k) t1_bb(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v0Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ka)  = tmps.at("0199_aa_ooQ")(ka,ia,Q) * tmps.at("0181_Q")(Q) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_aa_oo")(ia,ka) * t1.at("aa")(aa,ka) )
    
    // r1_2p.at("aa") += +2.00 <j,a||b,c>_aaaa t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0109_aa_voQ")(aa,ja,Q) * tmps.at("0199_aa_ooQ")(ja,ia,Q) )
    
    // r1_2p.at("aa") += -1.00 <k,j||b,c>_aaaa t2_aaaa(b,a,k,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += tmps.at("0106_aa_voQ")(aa,ka,Q) * tmps.at("0199_aa_ooQ")(ka,ia,Q) )
    
    // r2_2p.at("abab") += +2.00 <k,a||c,d>_aaaa t2_abab(c,b,k,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o2v2Q1 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin_aaaa_vvoo")(aa,ca,ia,ka)  = chol.at("aa_vvQ")(aa,ca,Q) * tmps.at("0199_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aaaa_vvoo")(aa,ca,ia,ka) * t2.at("abab")(ca,bb,ka,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||d,c>_abab t2_abab(a,b,l,j) t1_bb(c,k) t1_2p_aa(d,i) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,la)  = tmps.at("0199_aa_ooQ")(la,ia,Q) * tmps.at("0181_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    .allocate(tmps.at("0200_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0200_aa_oo")(ka,ia)  = tmps.at("0199_aa_ooQ")(ka,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1.at("aa")(aa,ka) * tmps.at("0200_aa_oo")(ka,ia) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t2_abab(a,b,l,j) t1_aa(c,k) t1_2p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0200_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0200_aa_oo"))
    .allocate(tmps.at("0201_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0201_aa_oo")(ja,ia)  = chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0197_Q")(Q) )
    
    // r1_2p.at("aa") += -2.00 <j,k||i,b>_abab t1_aa(a,j) t1_2p_bb(b,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1.at("aa")(aa,ja) * tmps.at("0201_aa_oo")(ja,ia) )
    
    // r2_2p.at("abab") += -2.00 <k,l||i,c>_abab t2_abab(a,b,k,j) t1_2p_bb(c,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0201_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    .deallocate(tmps.at("0201_aa_oo"))
    .allocate(tmps.at("0202_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0202_Q")(Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t1_2p.at("aa")(ba,ja) )
    
    // r2_2p.at("abab") += -2.00 <l,k||d,c>_abab t2_abab(a,b,i,k) t1_bb(c,j) t1_2p_aa(d,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = tmps.at("0107_bb_ooQ")(kb,jb,Q) * tmps.at("0202_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,j>_abab t2_abab(a,b,i,k) t1_2p_aa(c,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = chol.at("bb_ooQ")(kb,jb,Q) * tmps.at("0202_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += -2.00 <k,a||c,d>_aaaa t2_abab(c,b,i,j) t1_2p_aa(d,k) 
    // flops: o2v2 += o0v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_aa_vv")(aa,ca)  = chol.at("aa_vvQ")(aa,ca,Q) * tmps.at("0202_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,b||d,c>_abab t2_abab(a,c,i,j) t1_2p_aa(d,k) 
    // flops: o2v2 += o0v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = chol.at("bb_vvQ")(bb,cb,Q) * tmps.at("0202_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(a,k) t2_abab(c,b,i,j) t1_2p_aa(d,l) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_aa_vo")(ca,ka)  = chol.at("aa_ovQ")(ka,ca,Q) * tmps.at("0202_Q")(Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aa_vo")(ca,ka) * t2.at("abab")(ca,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += -2.00 <l,k||d,c>_abab t2_abab(a,c,i,j) t1_bb(b,k) t1_2p_aa(d,l) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin_bb_vo")(cb,kb)  = chol.at("bb_ovQ")(kb,cb,Q) * tmps.at("0202_Q")(Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bb_vo")(cb,kb) * t2.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t1_aa(a,j) t1_aa(b,i) t1_2p_aa(c,k) 
    // flops: o1v1 += o2v0Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ja)  = tmps.at("0098_aa_ooQ")(ja,ia,Q) * tmps.at("0202_Q")(Q) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t2_abab(a,b,k,j) t1_aa(c,i) t1_2p_aa(d,l) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,ka)  = tmps.at("0098_aa_ooQ")(ka,ia,Q) * tmps.at("0202_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aa_oo")(ia,ka) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r1_2p.at("aa") += +2.00 <j,a||b,i>_aaaa t1_2p_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0202_Q")(Q) )
    
    // r1_2p.at("aa") += -2.00 <j,a||b,c>_aaaa t1_aa(b,i) t1_2p_aa(c,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0202_Q")(Q) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t2_aaaa(b,a,i,j) t1_2p_aa(c,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0202_Q")(Q) )
    .allocate(tmps.at("0203_aa_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0203_aa_oo")(ja,ia)  = chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0202_Q")(Q) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,i>_aaaa t1_aa(a,j) t1_2p_aa(b,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1.at("aa")(aa,ja) * tmps.at("0203_aa_oo")(ja,ia) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,i>_aaaa t2_abab(a,b,k,j) t1_2p_aa(c,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0203_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    .deallocate(tmps.at("0203_aa_oo"))
    .allocate(tmps.at("0204_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0204_bb_oo")(kb,jb)  = dp.at("bb_ov")(kb,cb) * t1_1p.at("bb")(cb,jb) )
    
    // r2_2p.at("abab") += -6.00 d-_bb(k,c) t1_1p_bb(c,j) t2_2p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.00 * t2_2p.at("abab")(aa,bb,ia,kb) * tmps.at("0204_bb_oo")(kb,jb) )
    
    // r2_2p.at("abab") += -2.00 d-_bb(k,c) t1_1p_aa(a,i) t1_1p_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v1 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = t1_1p.at("bb")(bb,kb) * tmps.at("0204_bb_oo")(kb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    .allocate(tmps.at("0205_bb_vo"))
    
    // flops: o1v1  = o2v1
    //  mems: o1v1  = o1v1
    ( tmps.at("0205_bb_vo")(bb,jb)  = t1.at("bb")(bb,kb) * tmps.at("0204_bb_oo")(kb,jb) )
    .deallocate(tmps.at("0204_bb_oo"))
    
    // r2_1p.at("abab") += -1.00 d-_bb(k,c) t1_bb(b,k) t1_1p_aa(a,i) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("aa")(aa,ia) * tmps.at("0205_bb_vo")(bb,jb) )
    
    // r2_2p.at("abab") += -4.00 d-_bb(k,c) t1_bb(b,k) t1_1p_bb(c,j) t1_2p_aa(a,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * t1_2p.at("aa")(aa,ia) * tmps.at("0205_bb_vo")(bb,jb) )
    .deallocate(tmps.at("0205_bb_vo"))
    .allocate(tmps.at("0206_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0206_bb_oo")(kb,jb)  = dp.at("bb_ov")(kb,cb) * t1.at("bb")(cb,jb) )
    
    // r2_1p.at("abab") += -2.00 d-_bb(k,c) t1_bb(b,k) t1_bb(c,j) t1_2p_aa(a,i) 
    // flops: o2v2 += o2v1 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = t1.at("bb")(bb,kb) * tmps.at("0206_bb_oo")(kb,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_2p.at("aa")(aa,ia) )
    
    // r2_2p.at("abab") += -2.00 d-_bb(k,c) t1_bb(c,j) t1_1p_aa(a,i) t1_2p_bb(b,k) 
    // flops: o2v2 += o2v1 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = t1_2p.at("bb")(bb,kb) * tmps.at("0206_bb_oo")(kb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2.at("abab") += -1.00 d-_bb(k,c) t1_bb(b,k) t1_bb(c,j) t1_1p_aa(a,i) 
    // flops: o2v2 += o2v1 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = t1.at("bb")(bb,kb) * tmps.at("0206_bb_oo")(kb,jb) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    .allocate(tmps.at("0207_bb_vo"))
    
    // flops: o1v1  = o2v1
    //  mems: o1v1  = o1v1
    ( tmps.at("0207_bb_vo")(bb,jb)  = t1_1p.at("bb")(bb,kb) * tmps.at("0206_bb_oo")(kb,jb) )
    .deallocate(tmps.at("0206_bb_oo"))
    
    // r2_1p.at("abab") += -1.00 d-_bb(k,c) t1_bb(c,j) t1_1p_aa(a,i) t1_1p_bb(b,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("aa")(aa,ia) * tmps.at("0207_bb_vo")(bb,jb) )
    
    // r2_2p.at("abab") += -4.00 d-_bb(k,c) t1_bb(c,j) t1_1p_bb(b,k) t1_2p_aa(a,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * t1_2p.at("aa")(aa,ia) * tmps.at("0207_bb_vo")(bb,jb) )
    .deallocate(tmps.at("0207_bb_vo"))
    .allocate(tmps.at("0208_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0208_aa_oo")(ja,ia)  = dp.at("aa_ov")(ja,ba) * t1_1p.at("aa")(ba,ia) )
    
    // r1_2p.at("aa") += -6.00 d-_aa(j,b) t1_1p_aa(b,i) t1_2p_aa(a,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 6.00 * t1_2p.at("aa")(aa,ja) * tmps.at("0208_aa_oo")(ja,ia) )
    
    // r2_2p.at("abab") += -6.00 d-_aa(k,c) t1_1p_aa(c,i) t2_2p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.00 * tmps.at("0208_aa_oo")(ka,ia) * t2_2p.at("abab")(aa,bb,ka,jb) )
    .allocate(tmps.at("0209_aa_vo"))
    
    // flops: o1v1  = o2v1
    //  mems: o1v1  = o1v1
    ( tmps.at("0209_aa_vo")(aa,ia)  = t1.at("aa")(aa,ja) * tmps.at("0208_aa_oo")(ja,ia) )
    
    // r1_1p.at("aa") += -1.00 d-_aa(j,b) t1_aa(a,j) t0_1p t1_1p_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t0_1p * tmps.at("0209_aa_vo")(aa,ia) )
    
    // r1_2p.at("aa") += -2.00 d+_aa(j,b) t1_aa(a,j) t1_1p_aa(b,i) 
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0209_aa_vo")(aa,ia) )
    
    // r1_2p.at("aa") += -4.00 d-_aa(j,b) t1_aa(a,j) t1_1p_aa(b,i) t0_2p 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 4.00 * t0_2p * tmps.at("0209_aa_vo")(aa,ia) )
    
    // r1.at("aa") += -1.00 d-_aa(j,b) t1_aa(a,j) t1_1p_aa(b,i) 
    ( r1.at("aa")(aa,ia) -= tmps.at("0209_aa_vo")(aa,ia) )
    
    // r2_1p.at("abab") += -1.00 d-_aa(k,c) t1_aa(a,k) t1_1p_bb(b,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0209_aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    
    // r2_2p.at("abab") += -4.00 d-_aa(k,c) t1_aa(a,k) t1_1p_aa(c,i) t1_2p_bb(b,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * tmps.at("0209_aa_vo")(aa,ia) * t1_2p.at("bb")(bb,jb) )
    .deallocate(tmps.at("0209_aa_vo"))
    .allocate(tmps.at("0210_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0210_aa_oo")(ja,ia)  = dp.at("aa_ov")(ja,ba) * t1.at("aa")(ba,ia) )
    .allocate(tmps.at("0211_aa_vo"))
    
    // flops: o1v1  = o2v1
    //  mems: o1v1  = o1v1
    ( tmps.at("0211_aa_vo")(aa,ia)  = t1_1p.at("aa")(aa,ja) * tmps.at("0210_aa_oo")(ja,ia) )
    
    // r1_1p.at("aa") += -1.00 d-_aa(j,b) t1_aa(b,i) t0_1p t1_1p_aa(a,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t0_1p * tmps.at("0211_aa_vo")(aa,ia) )
    
    // r1_2p.at("aa") += -2.00 d+_aa(j,b) t1_aa(b,i) t1_1p_aa(a,j) 
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0211_aa_vo")(aa,ia) )
    
    // r1_2p.at("aa") += -4.00 d-_aa(j,b) t1_aa(b,i) t1_1p_aa(a,j) t0_2p 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 4.00 * t0_2p * tmps.at("0211_aa_vo")(aa,ia) )
    
    // r1.at("aa") += -1.00 d-_aa(j,b) t1_aa(b,i) t1_1p_aa(a,j) 
    ( r1.at("aa")(aa,ia) -= tmps.at("0211_aa_vo")(aa,ia) )
    
    // r2_1p.at("abab") += -1.00 d-_aa(k,c) t1_aa(c,i) t1_1p_aa(a,k) t1_1p_bb(b,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0211_aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    
    // r2_2p.at("abab") += -4.00 d-_aa(k,c) t1_aa(c,i) t1_1p_aa(a,k) t1_2p_bb(b,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * tmps.at("0211_aa_vo")(aa,ia) * t1_2p.at("bb")(bb,jb) )
    .deallocate(tmps.at("0211_aa_vo"))
    .allocate(tmps.at("0212_aa_vo"))
    
    // flops: o1v1  = o2v1
    //  mems: o1v1  = o1v1
    ( tmps.at("0212_aa_vo")(aa,ia)  = t1.at("aa")(aa,ja) * tmps.at("0210_aa_oo")(ja,ia) )
    
    // r1_1p.at("aa") += -1.00 d+_aa(j,b) t1_aa(a,j) t1_aa(b,i) 
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0212_aa_vo")(aa,ia) )
    
    // r1_1p.at("aa") += -2.00 d-_aa(j,b) t1_aa(a,j) t1_aa(b,i) t0_2p 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 2.00 * t0_2p * tmps.at("0212_aa_vo")(aa,ia) )
    
    // r1.at("aa") += -1.00 d-_aa(j,b) t1_aa(a,j) t1_aa(b,i) t0_1p 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= t0_1p * tmps.at("0212_aa_vo")(aa,ia) )
    
    // r2_1p.at("abab") += -2.00 d-_aa(k,c) t1_aa(a,k) t1_aa(c,i) t1_2p_bb(b,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0212_aa_vo")(aa,ia) * t1_2p.at("bb")(bb,jb) )
    
    // r2.at("abab") += -1.00 d-_aa(k,c) t1_aa(a,k) t1_aa(c,i) t1_1p_bb(b,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0212_aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    .deallocate(tmps.at("0212_aa_vo"))
    .allocate(tmps.at("0213_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0213_aa_oo")(ja,ia)  = dp.at("aa_ov")(ja,ba) * t1_2p.at("aa")(ba,ia) )
    
    // r1_2p.at("aa") += -6.00 d-_aa(j,b) t1_1p_aa(a,j) t1_2p_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 6.00 * t1_1p.at("aa")(aa,ja) * tmps.at("0213_aa_oo")(ja,ia) )
    
    // r2_2p.at("abab") += -6.00 d-_aa(k,c) t2_1p_abab(a,b,k,j) t1_2p_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.00 * tmps.at("0213_aa_oo")(ka,ia) * t2_1p.at("abab")(aa,bb,ka,jb) )
    .allocate(tmps.at("0214_aa_vo"))
    
    // flops: o1v1  = o2v1
    //  mems: o1v1  = o1v1
    ( tmps.at("0214_aa_vo")(aa,ia)  = t1.at("aa")(aa,ja) * tmps.at("0213_aa_oo")(ja,ia) )
    .deallocate(tmps.at("0213_aa_oo"))
    
    // r1_1p.at("aa") += -2.00 d-_aa(j,b) t1_aa(a,j) t1_2p_aa(b,i) 
    ( r1_1p.at("aa")(aa,ia) -= 2.00 * tmps.at("0214_aa_vo")(aa,ia) )
    
    // r1_2p.at("aa") += -2.00 d-_aa(j,b) t1_aa(a,j) t0_1p t1_2p_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t0_1p * tmps.at("0214_aa_vo")(aa,ia) )
    
    // r2_2p.at("abab") += -2.00 d-_aa(k,c) t1_aa(a,k) t1_1p_bb(b,j) t1_2p_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0214_aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    .deallocate(tmps.at("0214_aa_vo"))
    .allocate(tmps.at("0215_aa_vo"))
    
    // flops: o1v1  = o2v1
    //  mems: o1v1  = o1v1
    ( tmps.at("0215_aa_vo")(aa,ia)  = t1_2p.at("aa")(aa,ja) * tmps.at("0210_aa_oo")(ja,ia) )
    .deallocate(tmps.at("0210_aa_oo"))
    
    // r1_1p.at("aa") += -2.00 d-_aa(j,b) t1_aa(b,i) t1_2p_aa(a,j) 
    ( r1_1p.at("aa")(aa,ia) -= 2.00 * tmps.at("0215_aa_vo")(aa,ia) )
    
    // r1_2p.at("aa") += -2.00 d-_aa(j,b) t1_aa(b,i) t0_1p t1_2p_aa(a,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t0_1p * tmps.at("0215_aa_vo")(aa,ia) )
    
    // r2_2p.at("abab") += -2.00 d-_aa(k,c) t1_aa(c,i) t1_1p_bb(b,j) t1_2p_aa(a,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0215_aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    .deallocate(tmps.at("0215_aa_vo"))
    .allocate(tmps.at("0216_aa_vo"))
    
    // flops: o1v1  = o2v1
    //  mems: o1v1  = o1v1
    ( tmps.at("0216_aa_vo")(aa,ia)  = t1_1p.at("aa")(aa,ja) * tmps.at("0208_aa_oo")(ja,ia) )
    .deallocate(tmps.at("0208_aa_oo"))
    
    // r1_1p.at("aa") += -2.00 d-_aa(j,b) t1_1p_aa(a,j) t1_1p_aa(b,i) 
    ( r1_1p.at("aa")(aa,ia) -= 2.00 * tmps.at("0216_aa_vo")(aa,ia) )
    
    // r1_2p.at("aa") += -2.00 d-_aa(j,b) t0_1p t1_1p_aa(a,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t0_1p * tmps.at("0216_aa_vo")(aa,ia) )
    
    // r2_2p.at("abab") += -2.00 d-_aa(k,c) t1_1p_aa(a,k) t1_1p_bb(b,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0216_aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    .deallocate(tmps.at("0216_aa_vo"))
    .allocate(tmps.at("0217_aabb_oooo"))
    
    // flops: o4v0  = o4v0Q1
    //  mems: o4v0  = o4v0
    ( tmps.at("0217_aabb_oooo")(la,ia,kb,jb)  = chol.at("aa_ooQ")(la,ia,Q) * tmps.at("0107_bb_ooQ")(kb,jb,Q) )
    
    // r2_1p.at("abab") += +0.50 <l,k||i,c>_abab t1_bb(c,j) t2_1p_abab(a,b,l,k) 
    //            += +0.50 <k,l||i,c>_abab t1_bb(c,j) t2_1p_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,la,kb) * tmps.at("0217_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_1p.at("abab") += +1.00 <k,l||i,c>_abab t1_aa(a,k) t1_bb(c,j) t1_1p_bb(b,l) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_1p.at("bb")(bb,lb) * tmps.at("0217_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +1.00 <l,k||i,c>_abab t1_bb(c,j) t2_2p_abab(a,b,l,k) 
    //            += +1.00 <k,l||i,c>_abab t1_bb(c,j) t2_2p_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_2p.at("abab")(aa,bb,la,kb) * tmps.at("0217_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||i,c>_abab t1_aa(a,k) t1_bb(c,j) t1_2p_bb(b,l) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_2p.at("bb")(bb,lb) * tmps.at("0217_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <k,l||i,c>_abab t1_bb(c,j) t1_1p_aa(a,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_1p.at("bb")(bb,lb) * tmps.at("0217_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2.at("abab") += +0.50 <l,k||i,c>_abab t2_abab(a,b,l,k) t1_bb(c,j) 
    //         += +0.50 <k,l||i,c>_abab t2_abab(a,b,k,l) t1_bb(c,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,kb) * tmps.at("0217_aabb_oooo")(la,ia,kb,jb) )
    .allocate(tmps.at("0218_baab_vooo"))
    
    // flops: o3v1  = o4v1
    //  mems: o3v1  = o3v1
    ( tmps.at("0218_baab_vooo")(bb,la,ia,jb)  = t1.at("bb")(bb,kb) * tmps.at("0217_aabb_oooo")(la,ia,kb,jb) )
    .deallocate(tmps.at("0217_aabb_oooo"))
    
    // r2_1p.at("abab") += +1.00 <l,k||i,c>_abab t1_bb(b,k) t1_bb(c,j) t1_1p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,la) * tmps.at("0218_baab_vooo")(bb,la,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||i,c>_abab t1_bb(b,k) t1_bb(c,j) t1_2p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("aa")(aa,la) * tmps.at("0218_baab_vooo")(bb,la,ia,jb) )
    
    // r2.at("abab") += +1.00 <k,l||i,c>_abab t1_aa(a,k) t1_bb(b,l) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0218_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0218_baab_vooo"))
    .allocate(tmps.at("0219_aabb_oooo"))
    
    // flops: o4v0  = o4v0Q1
    //  mems: o4v0  = o4v0
    ( tmps.at("0219_aabb_oooo")(la,ia,kb,jb)  = tmps.at("0098_aa_ooQ")(la,ia,Q) * tmps.at("0107_bb_ooQ")(kb,jb,Q) )
    
    // r2_1p.at("abab") += +0.50 <l,k||d,c>_abab t1_bb(c,j) t1_aa(d,i) t2_1p_abab(a,b,l,k) 
    //            += +0.50 <k,l||d,c>_abab t1_bb(c,j) t1_aa(d,i) t2_1p_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,la,kb) * tmps.at("0219_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||d,c>_abab t1_bb(c,j) t1_aa(d,i) t2_2p_abab(a,b,l,k) 
    //            += +1.00 <k,l||d,c>_abab t1_bb(c,j) t1_aa(d,i) t2_2p_abab(a,b,k,l) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_2p.at("abab")(aa,bb,la,kb) * tmps.at("0219_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t1_aa(a,k) t1_bb(c,j) t1_aa(d,i) t1_2p_bb(b,l) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_2p.at("bb")(bb,lb) * tmps.at("0219_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2.at("abab") += +0.50 <l,k||d,c>_abab t2_abab(a,b,l,k) t1_bb(c,j) t1_aa(d,i) 
    //         += +0.50 <k,l||d,c>_abab t2_abab(a,b,k,l) t1_bb(c,j) t1_aa(d,i) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,kb) * tmps.at("0219_aabb_oooo")(la,ia,kb,jb) )
    .allocate(tmps.at("0220_baab_vooo"))
    
    // flops: o3v1  = o4v1
    //  mems: o3v1  = o3v1
    ( tmps.at("0220_baab_vooo")(bb,la,ia,jb)  = t1.at("bb")(bb,kb) * tmps.at("0219_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t1_aa(d,i) t1_1p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,la) * tmps.at("0220_baab_vooo")(bb,la,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t1_aa(d,i) t1_2p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("aa")(aa,la) * tmps.at("0220_baab_vooo")(bb,la,ia,jb) )
    
    // r2.at("abab") += +1.00 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t1_bb(c,j) t1_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0220_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0220_baab_vooo"))
    .allocate(tmps.at("0221_aabb_oooo"))
    
    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    ( tmps.at("0221_aabb_oooo")(la,ia,jb,kb)  = t1_1p.at("bb")(cb,jb) * tmps.at("0163_aabb_ooov")(la,ia,kb,cb) )
    .deallocate(tmps.at("0163_aabb_ooov"))
    
    // r2_1p.at("abab") += +0.50 <l,k||i,c>_abab t2_abab(a,b,l,k) t1_1p_bb(c,j) 
    //            += +0.50 <k,l||i,c>_abab t2_abab(a,b,k,l) t1_1p_bb(c,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,kb) * tmps.at("0221_aabb_oooo")(la,ia,jb,kb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||i,c>_abab t2_1p_abab(a,b,l,k) t1_1p_bb(c,j) 
    //            += +1.00 <k,l||i,c>_abab t2_1p_abab(a,b,k,l) t1_1p_bb(c,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_1p.at("abab")(aa,bb,la,kb) * tmps.at("0221_aabb_oooo")(la,ia,jb,kb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||i,c>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_1p_bb(c,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_1p.at("bb")(bb,lb) * tmps.at("0221_aabb_oooo")(ka,ia,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0222_baab_vooo"))
    
    // flops: o3v1  = o4v1
    //  mems: o3v1  = o3v1
    ( tmps.at("0222_baab_vooo")(bb,ka,ia,jb)  = t1.at("bb")(bb,lb) * tmps.at("0221_aabb_oooo")(ka,ia,jb,lb) )
    .deallocate(tmps.at("0221_aabb_oooo"))
    
    // r2_1p.at("abab") += +1.00 <k,l||i,c>_abab t1_aa(a,k) t1_bb(b,l) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0222_baab_vooo")(bb,ka,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||i,c>_abab t1_bb(b,k) t1_1p_aa(a,l) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,la) * tmps.at("0222_baab_vooo")(bb,la,ia,jb) )
    .deallocate(tmps.at("0222_baab_vooo"))
    .allocate(tmps.at("0223_baab_vooo"))
    
    // flops: o3v1  = o4v1
    //  mems: o3v1  = o3v1
    ( tmps.at("0223_baab_vooo")(bb,ka,ia,jb)  = t1_1p.at("bb")(bb,lb) * tmps.at("0219_aabb_oooo")(ka,ia,lb,jb) )
    .deallocate(tmps.at("0219_aabb_oooo"))
    
    // r2_1p.at("abab") += +1.00 <k,l||d,c>_abab t1_aa(a,k) t1_bb(c,j) t1_aa(d,i) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0223_baab_vooo")(bb,ka,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t1_bb(c,j) t1_aa(d,i) t1_1p_aa(a,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0223_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0223_baab_vooo"))
    .allocate(tmps.at("0224_aabb_oooo"))
    
    // flops: o4v0  = o4v0Q1
    //  mems: o4v0  = o4v0
    ( tmps.at("0224_aabb_oooo")(la,ia,kb,jb)  = tmps.at("0104_aa_ooQ")(la,ia,Q) * tmps.at("0107_bb_ooQ")(kb,jb,Q) )
    
    // r2_1p.at("abab") += +0.50 <l,k||d,c>_abab t2_abab(a,b,l,k) t1_bb(c,j) t1_1p_aa(d,i) 
    //            += +0.50 <k,l||d,c>_abab t2_abab(a,b,k,l) t1_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,kb) * tmps.at("0224_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||d,c>_abab t1_bb(c,j) t2_1p_abab(a,b,l,k) t1_1p_aa(d,i) 
    //            += +1.00 <k,l||d,c>_abab t1_bb(c,j) t2_1p_abab(a,b,k,l) t1_1p_aa(d,i) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_1p.at("abab")(aa,bb,la,kb) * tmps.at("0224_aabb_oooo")(la,ia,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t1_aa(a,k) t1_bb(c,j) t1_1p_bb(b,l) t1_1p_aa(d,i) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1_1p.at("bb")(bb,lb) * tmps.at("0224_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0225_baab_vooo"))
    
    // flops: o3v1  = o4v1
    //  mems: o3v1  = o3v1
    ( tmps.at("0225_baab_vooo")(bb,ka,ia,jb)  = t1.at("bb")(bb,lb) * tmps.at("0224_aabb_oooo")(ka,ia,lb,jb) )
    .deallocate(tmps.at("0224_aabb_oooo"))
    
    // r2_1p.at("abab") += +1.00 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t1_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0225_baab_vooo")(bb,ka,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t1_1p_aa(a,l) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,la) * tmps.at("0225_baab_vooo")(bb,la,ia,jb) )
    .deallocate(tmps.at("0225_baab_vooo"))
    .allocate(tmps.at("0226_bb_oo"))
    
    // flops: o2v0  = o2v2 o2v1
    //  mems: o2v0  = o1v1 o2v0
    ( tmps.at("bin_bb_vo")(db,lb)  = t1.at("bb")(cb,kb) * tmps.at("0117_bbbb_ovov")(lb,cb,kb,db) )
    ( tmps.at("0226_bb_oo")(lb,jb)  = tmps.at("bin_bb_vo")(db,lb) * t1.at("bb")(db,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,j) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0226_bb_oo")(lb,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,j) t2_2p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_2p.at("abab")(aa,bb,ia,lb) * tmps.at("0226_bb_oo")(lb,jb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,b,i,l) t1_bb(c,k) t1_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ia,lb) * tmps.at("0226_bb_oo")(lb,jb) )
    .deallocate(tmps.at("0226_bb_oo"))
    .allocate(tmps.at("0227_bb_ov"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0227_bb_ov")(kb,cb)  = t1_1p.at("bb")(db,lb) * tmps.at("0117_bbbb_ovov")(kb,db,lb,cb) )
    .deallocate(tmps.at("0117_bbbb_ovov"))
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t2_abab(a,b,i,k) t1_1p_bb(c,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = t1_1p.at("bb")(db,jb) * tmps.at("0227_bb_ov")(kb,db) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t1_bb(b,k) t2_1p_abab(a,d,i,j) t1_1p_bb(c,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = t2_1p.at("abab")(aa,db,ia,jb) * tmps.at("0227_bb_ov")(kb,db) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0228_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0228_bb_oo")(kb,jb)  = t1.at("bb")(cb,jb) * tmps.at("0227_bb_ov")(kb,cb) )
    .deallocate(tmps.at("0227_bb_ov"))
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,b,i,k) t1_bb(c,j) t1_1p_bb(d,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ia,kb) * tmps.at("0228_bb_oo")(kb,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t1_bb(c,j) t2_1p_abab(a,b,i,l) t1_1p_bb(d,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_1p.at("abab")(aa,bb,ia,lb) * tmps.at("0228_bb_oo")(lb,jb) )
    .deallocate(tmps.at("0228_bb_oo"))
    .allocate(tmps.at("0229_aa_ov"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0229_aa_ov")(ka,ca)  = t1.at("aa")(ba,ja) * tmps.at("0094_aaaa_ovov")(ka,ba,ja,ca) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t1_aa(b,j) t2_2p_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * t2_2p.at("aaaa")(ca,aa,ia,ka) * tmps.at("0229_aa_ov")(ka,ca) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,la,jb)  = t2_2p.at("abab")(da,bb,ia,jb) * tmps.at("0229_aa_ov")(la,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,la,jb) * t1.at("aa")(aa,la) )
    .allocate(tmps.at("0230_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0230_aa_oo")(ka,ia)  = t1.at("aa")(ca,ia) * tmps.at("0229_aa_ov")(ka,ca) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(b,j) t1_aa(c,i) t1_1p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1_1p.at("aa")(aa,ka) * tmps.at("0230_aa_oo")(ka,ia) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t1_aa(b,j) t1_aa(c,i) t1_2p_aa(a,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * t1_2p.at("aa")(aa,ka) * tmps.at("0230_aa_oo")(ka,ia) )
    
    // r1.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += t1.at("aa")(aa,ka) * tmps.at("0230_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0230_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0230_aa_oo")(la,ia) * t2_2p.at("abab")(aa,bb,la,jb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_aaaa t2_abab(a,b,l,j) t1_aa(c,k) t1_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0230_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0230_aa_oo"))
    .allocate(tmps.at("0231_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0231_aa_oo")(ka,ia)  = t1_1p.at("aa")(ca,ia) * tmps.at("0229_aa_ov")(ka,ca) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1.at("aa")(aa,ka) * tmps.at("0231_aa_oo")(ka,ia) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t1_aa(b,j) t1_1p_aa(a,k) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0231_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t2_abab(a,b,l,j) t1_aa(c,k) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0231_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(c,k) t2_1p_abab(a,b,l,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0231_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0231_aa_oo"))
    .allocate(tmps.at("0232_aa_ov"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0232_aa_ov")(ja,ba)  = t1_1p.at("aa")(ca,ka) * tmps.at("0094_aaaa_ovov")(ja,ca,ka,ba) )
    .deallocate(tmps.at("0094_aaaa_ovov"))
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_aaaa t1_aa(a,k) t2_1p_abab(d,b,i,j) t1_1p_aa(c,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("0232_aa_ov")(ka,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0233_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0233_aa_oo")(ja,ia)  = t1.at("aa")(ba,ia) * tmps.at("0232_aa_ov")(ja,ba) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(a,j) t1_aa(b,i) t1_1p_aa(c,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1.at("aa")(aa,ja) * tmps.at("0233_aa_oo")(ja,ia) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_aaaa t1_aa(b,i) t1_1p_aa(a,k) t1_1p_aa(c,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0233_aa_oo")(ka,ia) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t2_abab(a,b,k,j) t1_aa(c,i) t1_1p_aa(d,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0233_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_aaaa t1_aa(c,i) t2_1p_abab(a,b,l,j) t1_1p_aa(d,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0233_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0233_aa_oo"))
    .allocate(tmps.at("0234_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0234_aa_oo")(ka,ia)  = t1_2p.at("aa")(ca,ia) * tmps.at("0229_aa_ov")(ka,ca) )
    .deallocate(tmps.at("0229_aa_ov"))
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * t1.at("aa")(aa,ka) * tmps.at("0234_aa_oo")(ka,ia) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t2_abab(a,b,l,j) t1_aa(c,k) t1_2p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0234_aa_oo")(la,ia) * t2.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0234_aa_oo"))
    .allocate(tmps.at("0235_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0235_aa_oo")(ja,ia)  = t1_1p.at("aa")(ca,ia) * tmps.at("0232_aa_ov")(ja,ca) )
    .deallocate(tmps.at("0232_aa_ov"))
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_aaaa t1_aa(a,j) t1_1p_aa(b,k) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * t1.at("aa")(aa,ja) * tmps.at("0235_aa_oo")(ja,ia) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_aaaa t2_abab(a,b,k,j) t1_1p_aa(c,l) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0235_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    .deallocate(tmps.at("0235_aa_oo"))
    .allocate(tmps.at("0236_aabb_oovo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0236_aabb_oovo")(ka,ia,bb,jb)  = tmps.at("0097_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    
    // r2_1p.at("abab") += -1.00 <k,b||i,c>_abab t1_bb(c,j) t1_1p_aa(a,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("aa")(aa,ka) * tmps.at("0236_aabb_oovo")(ka,ia,bb,jb) )
    
    // r2_2p.at("abab") += -2.00 <k,b||i,c>_abab t1_bb(c,j) t1_2p_aa(a,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_2p.at("aa")(aa,ka) * tmps.at("0236_aabb_oovo")(ka,ia,bb,jb) )
    
    // r2.at("abab") += -1.00 <k,b||i,c>_abab t1_aa(a,k) t1_bb(c,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("0236_aabb_oovo")(ka,ia,bb,jb) )
    .deallocate(tmps.at("0236_aabb_oovo"))
    .allocate(tmps.at("0237_aabb_oovo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0237_aabb_oovo")(la,ia,bb,jb)  = tmps.at("0102_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(la,ia,Q) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,i>_aaaa t2_abab(c,b,k,j) t1_1p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("aa")(aa,la) * tmps.at("0237_aabb_oovo")(la,ia,bb,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,i>_aaaa t2_abab(c,b,k,j) t1_2p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_2p.at("aa")(aa,la) * tmps.at("0237_aabb_oovo")(la,ia,bb,jb) )
    
    // r2.at("abab") += -1.00 <l,k||c,i>_aaaa t1_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("0237_aabb_oovo")(ka,ia,bb,jb) )
    .deallocate(tmps.at("0237_aabb_oovo"))
    .allocate(tmps.at("0238_aabb_oovo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0238_aabb_oovo")(la,ia,bb,jb)  = tmps.at("0100_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(la,ia,Q) )
    
    // r2_1p.at("abab") += +1.00 <l,k||i,c>_abab t2_bbbb(c,b,j,k) t1_1p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,la) * tmps.at("0238_aabb_oovo")(la,ia,bb,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||i,c>_abab t2_bbbb(c,b,j,k) t1_2p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("aa")(aa,la) * tmps.at("0238_aabb_oovo")(la,ia,bb,jb) )
    
    // r2.at("abab") += +1.00 <k,l||i,c>_abab t1_aa(a,k) t2_bbbb(c,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0238_aabb_oovo")(ka,ia,bb,jb) )
    .deallocate(tmps.at("0238_aabb_oovo"))
    .allocate(tmps.at("0239_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0239_bb_voQ")(bb,jb,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t2_1p.at("abab")(ca,bb,ka,jb) )
    
    // r2_2p.at("abab") += -1.00 <l,k||d,c>_abab t2_1p_abab(a,c,i,j) t2_1p_abab(d,b,l,k) 
    //            += -1.00 <k,l||d,c>_abab t2_1p_abab(a,c,i,j) t2_1p_abab(d,b,k,l) 
    // flops: o2v2 += o1v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = chol.at("bb_ovQ")(kb,cb,Q) * tmps.at("0239_bb_voQ")(bb,kb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_vv")(bb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(a,k) t2_1p_abab(d,b,l,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0239_bb_voQ")(bb,jb,Q) * tmps.at("0104_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += -0.50 <l,k||d,c>_abab t2_abab(a,c,i,j) t2_1p_abab(d,b,l,k) 
    //            += -0.50 <k,l||d,c>_abab t2_abab(a,c,i,j) t2_1p_abab(d,b,k,l) 
    // flops: o2v2 += o1v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = chol.at("bb_ovQ")(kb,cb,Q) * tmps.at("0239_bb_voQ")(bb,kb,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||d,c>_abab t2_1p_abab(a,c,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0111_aa_voQ")(aa,ia,Q) * tmps.at("0239_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += -2.00 <k,a||c,d>_aaaa t2_1p_abab(d,b,k,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0113_aa_voQ")(aa,ia,Q) * tmps.at("0239_bb_voQ")(bb,jb,Q) )
    
    // r2_1p.at("abab") += +1.00 <k,a||c,i>_aaaa t2_1p_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0239_bb_voQ")(bb,jb,Q) )
    
    // r2_1p.at("abab") += -0.50 <l,k||c,d>_abab t2_abab(a,b,i,k) t2_1p_abab(c,d,l,j) 
    //            += -0.50 <l,k||d,c>_abab t2_abab(a,b,i,k) t2_1p_abab(d,c,l,j) 
    // flops: o2v2 += o2v1Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = chol.at("bb_ovQ")(kb,db,Q) * tmps.at("0239_bb_voQ")(db,jb,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += -1.00 <k,l||c,d>_abab t2_1p_abab(a,b,i,l) t2_1p_abab(c,d,k,j) 
    //            += -1.00 <k,l||d,c>_abab t2_1p_abab(a,b,i,l) t2_1p_abab(d,c,k,j) 
    // flops: o2v2 += o2v1Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = chol.at("bb_ovQ")(lb,db,Q) * tmps.at("0239_bb_voQ")(db,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0239_bb_voQ")(bb,jb,Q) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0239_bb_voQ")(bb,jb,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += -1.00 <k,a||c,d>_aaaa t1_aa(c,i) t2_1p_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0239_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(c,i) t1_1p_aa(a,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0239_bb_voQ")(bb,jb,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    .allocate(tmps.at("0240_aabb_oovo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0240_aabb_oovo")(ka,ia,bb,jb)  = tmps.at("0239_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,i>_aaaa t1_aa(a,k) t2_1p_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("0240_aabb_oovo")(ka,ia,bb,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,i>_aaaa t1_1p_aa(a,k) t2_1p_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0240_aabb_oovo")(ka,ia,bb,jb) )
    .deallocate(tmps.at("0240_aabb_oovo"))
    .allocate(tmps.at("0241_bbaa_oovo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0241_bbaa_oovo")(kb,jb,aa,ia)  = tmps.at("0109_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    
    // r2_1p.at("abab") += -1.00 <a,k||c,j>_abab t1_aa(c,i) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("bb")(bb,kb) * tmps.at("0241_bbaa_oovo")(kb,jb,aa,ia) )
    
    // r2_2p.at("abab") += -2.00 <a,k||c,j>_abab t1_aa(c,i) t1_2p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_2p.at("bb")(bb,kb) * tmps.at("0241_bbaa_oovo")(kb,jb,aa,ia) )
    
    // r2.at("abab") += -1.00 <a,k||c,j>_abab t1_bb(b,k) t1_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t1.at("bb")(bb,kb) * tmps.at("0241_bbaa_oovo")(kb,jb,aa,ia) )
    .deallocate(tmps.at("0241_bbaa_oovo"))
    .allocate(tmps.at("0242_bbaa_oovo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0242_bbaa_oovo")(kb,jb,aa,ia)  = tmps.at("0113_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    
    // r2_1p.at("abab") += -1.00 <a,k||c,j>_abab t1_bb(b,k) t1_1p_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1.at("bb")(bb,kb) * tmps.at("0242_bbaa_oovo")(kb,jb,aa,ia) )
    
    // r2_2p.at("abab") += -2.00 <a,k||c,j>_abab t1_1p_bb(b,k) t1_1p_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_1p.at("bb")(bb,kb) * tmps.at("0242_bbaa_oovo")(kb,jb,aa,ia) )
    .deallocate(tmps.at("0242_bbaa_oovo"))
    .allocate(tmps.at("0243_bbaa_oovo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0243_bbaa_oovo")(kb,jb,aa,ia)  = tmps.at("0111_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,j>_bbbb t1_bb(b,k) t2_1p_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1.at("bb")(bb,kb) * tmps.at("0243_bbaa_oovo")(kb,jb,aa,ia) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,j>_bbbb t2_1p_abab(a,c,i,l) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_1p.at("bb")(bb,kb) * tmps.at("0243_bbaa_oovo")(kb,jb,aa,ia) )
    .deallocate(tmps.at("0243_bbaa_oovo"))
    .allocate(tmps.at("0244_aa_vv"))
    
    // flops: o0v2  = o1v2Q1
    //  mems: o0v2  = o0v2
    ( tmps.at("0244_aa_vv")(da,aa)  = tmps.at("0109_aa_voQ")(aa,ka,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    
    // r2_1p.at("abab") += +1.00 <k,a||c,d>_aaaa t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0244_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,a||c,d>_aaaa t1_aa(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0244_aa_vv")(da,aa) * t2_2p.at("abab")(da,bb,ia,jb) )
    
    // r2.at("abab") += +1.00 <k,a||c,d>_aaaa t2_abab(d,b,i,j) t1_aa(c,k) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0244_aa_vv")(da,aa) * t2.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0244_aa_vv"))
    .allocate(tmps.at("0245_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0245_bb_voQ")(bb,jb,Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t2_2p.at("bbbb")(cb,bb,jb,lb) )
    
    // r2_2p.at("abab") += -2.00 <a,k||c,d>_abab t1_aa(c,i) t2_2p_bbbb(d,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0245_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += -2.00 <a,k||i,c>_abab t2_2p_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0245_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += +2.00 <k,l||i,c>_abab t1_aa(a,k) t2_2p_bbbb(c,b,j,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0245_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_2p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0245_bb_voQ")(bb,jb,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,d>_abab t2_aaaa(c,a,i,k) t2_2p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0245_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0246_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0246_bb_voQ")(bb,jb,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t2_2p.at("abab")(ca,bb,ka,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,a||c,i>_aaaa t2_2p_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0246_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_abab t2_abab(a,b,i,k) t2_2p_abab(c,d,l,j) 
    //            += -1.00 <l,k||d,c>_abab t2_abab(a,b,i,k) t2_2p_abab(d,c,l,j) 
    // flops: o2v2 += o2v1Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = chol.at("bb_ovQ")(kb,db,Q) * tmps.at("0246_bb_voQ")(db,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0246_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,i>_aaaa t1_aa(a,k) t2_2p_abab(c,b,l,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0246_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0246_bb_voQ")(bb,jb,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += -2.00 <k,a||c,d>_aaaa t1_aa(c,i) t2_2p_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0246_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += -1.00 <l,k||d,c>_abab t2_abab(a,c,i,j) t2_2p_abab(d,b,l,k) 
    //            += -1.00 <k,l||d,c>_abab t2_abab(a,c,i,j) t2_2p_abab(d,b,k,l) 
    // flops: o2v2 += o1v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = chol.at("bb_ovQ")(kb,cb,Q) * tmps.at("0246_bb_voQ")(bb,kb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    .allocate(tmps.at("0247_bb_ooQ"))
    
    // flops: o2v0Q1  = o2v1Q1
    //  mems: o2v0Q1  = o2v0Q1
    ( tmps.at("0247_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(jb,cb,Q) * t1_2p.at("bb")(cb,kb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,j>_bbbb t2_abab(a,b,i,k) t1_2p_bb(c,l) 
    // flops: o2v2 += o3v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = tmps.at("0247_bb_ooQ")(kb,lb,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += -2.00 <a,k||i,c>_abab t1_bb(b,k) t1_2p_bb(c,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0247_bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_abab t2_aaaa(c,a,i,l) t1_bb(b,k) t1_2p_bb(d,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0247_bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -2.00 <k,l||c,d>_abab t2_abab(a,b,i,l) t1_aa(c,k) t1_2p_bb(d,j) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = tmps.at("0247_bb_ooQ")(lb,jb,Q) * tmps.at("0176_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p.at("abab") += +2.00 <k,b||c,d>_bbbb t2_abab(a,c,i,k) t1_2p_bb(d,j) 
    // flops: o2v2 += o2v2Q1 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin_bbbb_vvoo")(bb,cb,jb,kb)  = chol.at("bb_vvQ")(bb,cb,Q) * tmps.at("0247_bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bbbb_vvoo")(bb,cb,jb,kb) * t2.at("abab")(aa,cb,ia,kb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,b,i,l) t1_bb(c,k) t1_2p_bb(d,j) 
    // flops: o2v2 += o2v0Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,lb)  = tmps.at("0247_bb_ooQ")(lb,jb,Q) * tmps.at("0181_Q")(Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p.at("abab") += -2.00 <a,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t1_2p_bb(d,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0247_bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0248_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0248_aa_voQ")(ba,ia,Q)  = chol.at("bb_ovQ")(jb,cb,Q) * t2.at("abab")(ba,cb,ia,jb) )
    
    // r1_1p.at("aa") += -0.50 <k,j||c,b>_abab t2_abab(a,b,k,j) t1_1p_aa(c,i) 
    //          += -0.50 <j,k||c,b>_abab t2_abab(a,b,j,k) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0248_aa_voQ")(aa,ka,Q) * tmps.at("0104_aa_ooQ")(ka,ia,Q) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t2_abab(a,c,i,l) t1_bb(b,k) t1_2p_bb(d,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0247_bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .deallocate(tmps.at("0247_bb_ooQ"))
    
    // r1_2p.at("aa") += -1.00 <k,j||c,b>_abab t2_abab(a,b,k,j) t1_2p_aa(c,i) 
    //          += -1.00 <j,k||c,b>_abab t2_abab(a,b,j,k) t1_2p_aa(c,i) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0248_aa_voQ")(aa,ka,Q) * tmps.at("0199_aa_ooQ")(ka,ia,Q) )
    .deallocate(tmps.at("0199_aa_ooQ"))
    
    // r1_1p.at("aa") += -0.50 <k,j||b,c>_abab t2_abab(b,c,i,j) t1_1p_aa(a,k) 
    //          += -0.50 <k,j||c,b>_abab t2_abab(c,b,i,j) t1_1p_aa(a,k) 
    // flops: o1v1 += o2v1Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ka)  = chol.at("aa_ovQ")(ka,ba,Q) * tmps.at("0248_aa_voQ")(ba,ia,Q) )
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("bin_aa_oo")(ia,ka) * t1_1p.at("aa")(aa,ka) )
    
    // r1_2p.at("aa") += -1.00 <k,j||b,c>_abab t2_abab(b,c,i,j) t1_2p_aa(a,k) 
    //          += -1.00 <k,j||c,b>_abab t2_abab(c,b,i,j) t1_2p_aa(a,k) 
    // flops: o1v1 += o2v1Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ka)  = chol.at("aa_ovQ")(ka,ba,Q) * tmps.at("0248_aa_voQ")(ba,ia,Q) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_aa_oo")(ia,ka) * t1_2p.at("aa")(aa,ka) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0100_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t1_1p_bb(b,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0115_bb_ooQ")(lb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t2_abab(a,d,i,k) t1_bb(c,j) t1_2p_bb(b,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0107_bb_ooQ")(lb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1_2p.at("bb")(bb,lb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,d,i,l) t1_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0107_bb_ooQ")(kb,jb,Q) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r1.at("aa") += -1.00 <k,j||b,c>_bbbb t2_abab(a,c,i,k) t1_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0181_Q")(Q) )
    
    // r2_1p.at("abab") += -0.50 <l,k||c,d>_abab t2_abab(c,d,i,k) t2_1p_abab(a,b,l,j) 
    //            += -0.50 <l,k||d,c>_abab t2_abab(d,c,i,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o2v1Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,la)  = chol.at("aa_ovQ")(la,ca,Q) * tmps.at("0248_aa_voQ")(ca,ia,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aa_oo")(ia,la) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r1_2p.at("aa") += +2.00 <k,j||c,b>_abab t2_abab(a,b,i,j) t1_2p_aa(c,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0202_Q")(Q) )
    .deallocate(tmps.at("0202_Q"))
    
    // r1.at("aa") += +1.00 <j,k||b,c>_abab t2_abab(a,c,i,k) t1_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0176_Q")(Q) )
    
    // r2.at("abab") += -1.00 <k,b||c,d>_bbbb t2_abab(a,d,i,k) t1_bb(c,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0097_bb_voQ")(bb,jb,Q) )
    
    // r1.at("aa") += -0.50 <k,j||b,c>_abab t2_abab(a,c,k,j) t1_aa(b,i) 
    //       += -0.50 <j,k||b,c>_abab t2_abab(a,c,j,k) t1_aa(b,i) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0248_aa_voQ")(aa,ka,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    
    // r1.at("aa") += -0.50 <k,j||i,b>_abab t2_abab(a,b,k,j) 
    //       += -0.50 <j,k||i,b>_abab t2_abab(a,b,j,k) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0248_aa_voQ")(aa,ka,Q) )
    
    // r1.at("aa") += +0.50 <a,j||b,c>_abab t2_abab(b,c,i,j) 
    //       += +0.50 <a,j||c,b>_abab t2_abab(c,b,i,j) 
    // flops: o1v1 += o1v2Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += chol.at("aa_vvQ")(aa,ba,Q) * tmps.at("0248_aa_voQ")(ba,ia,Q) )
    
    // r1.at("aa") += -0.50 <j,k||b,c>_abab t1_aa(a,j) t2_abab(b,c,i,k) 
    //       += -0.50 <j,k||c,b>_abab t1_aa(a,j) t2_abab(c,b,i,k) 
    // flops: o1v1 += o2v1Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ja)  = chol.at("aa_ovQ")(ja,ba,Q) * tmps.at("0248_aa_voQ")(ba,ia,Q) )
    ( r1.at("aa")(aa,ia) -= tmps.at("bin_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_bbbb t2_abab(a,b,i,j) t1_2p_bb(c,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0197_Q")(Q) )
    .deallocate(tmps.at("0197_Q"))
    
    // r2.at("abab") += +1.00 <l,k||d,c>_abab t2_abab(a,c,i,k) t2_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0102_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_abab t2_abab(c,d,i,k) t2_2p_abab(a,b,l,j) 
    //            += -1.00 <l,k||d,c>_abab t2_abab(d,c,i,k) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o2v1Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,la)  = chol.at("aa_ovQ")(la,ca,Q) * tmps.at("0248_aa_voQ")(ca,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aa_oo")(ia,la) * t2_2p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,j>_bbbb t2_abab(a,c,i,k) t1_2p_bb(b,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("0248_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1_2p.at("bb")(bb,lb) )
    
    // r2_2p.at("abab") += +2.00 <k,b||c,d>_bbbb t2_abab(a,c,i,k) t1_2p_bb(d,j) 
    // flops: o2v2 += o1v2Q1 o2v2Q1
    //  mems: o2v2 += o1v1Q1 o2v2
    ( tmps.at("bin_bb_voQ")(bb,jb,Q)  = chol.at("bb_vvQ")(bb,db,Q) * t1_2p.at("bb")(db,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_voQ")(bb,jb,Q) * tmps.at("0248_aa_voQ")(aa,ia,Q) )
    
    // r2.at("abab") += -0.50 <l,k||c,d>_abab t2_abab(a,b,l,j) t2_abab(c,d,i,k) 
    //         += -0.50 <l,k||d,c>_abab t2_abab(a,b,l,j) t2_abab(d,c,i,k) 
    // flops: o2v2 += o2v1Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,la)  = chol.at("aa_ovQ")(la,ca,Q) * tmps.at("0248_aa_voQ")(ca,ia,Q) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    
    // r1_1p.at("aa") += -1.00 <k,j||b,c>_bbbb t2_abab(a,b,i,j) t1_1p_bb(c,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0185_Q")(Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||d,c>_abab t2_abab(a,c,i,k) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0246_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0246_bb_voQ"))
    
    // r2_1p.at("abab") += +1.00 <l,k||d,c>_abab t2_abab(a,c,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0239_bb_voQ")(bb,jb,Q) )
    
    // r1_1p.at("aa") += +1.00 <k,j||c,b>_abab t2_abab(a,b,i,j) t1_1p_aa(c,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0183_Q")(Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_2p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0245_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0245_bb_voQ"))
    
    // r2_1p.at("abab") += +1.00 <l,k||c,j>_bbbb t2_abab(a,c,i,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("0248_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1_1p.at("bb")(bb,lb) )
    
    // r2.at("abab") += -1.00 <l,k||c,j>_bbbb t2_abab(a,c,i,l) t1_bb(b,k) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0248_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,d>_bbbb t2_abab(a,d,i,k) t1_bb(c,j) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0107_bb_ooQ")(lb,jb,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,l) t1_bb(b,k) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0115_bb_ooQ")(kb,jb,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2.at("abab") += +1.00 <k,b||c,j>_bbbb t2_abab(a,c,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0248_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0249_aa_vv"))
    
    // flops: o0v2  = o1v2Q1
    //  mems: o0v2  = o0v2
    ( tmps.at("0249_aa_vv")(da,aa)  = tmps.at("0248_aa_voQ")(aa,la,Q) * chol.at("aa_ovQ")(la,da,Q) )
    
    // r2_1p.at("abab") += -0.50 <l,k||d,c>_abab t2_abab(a,c,l,k) t2_1p_abab(d,b,i,j) 
    //            += -0.50 <k,l||d,c>_abab t2_abab(a,c,k,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0249_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    
    // r2_2p.at("abab") += -1.00 <l,k||d,c>_abab t2_abab(a,c,l,k) t2_2p_abab(d,b,i,j) 
    //            += -1.00 <k,l||d,c>_abab t2_abab(a,c,k,l) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0249_aa_vv")(da,aa) * t2_2p.at("abab")(da,bb,ia,jb) )
    
    // r2.at("abab") += -0.50 <l,k||d,c>_abab t2_abab(a,c,l,k) t2_abab(d,b,i,j) 
    //         += -0.50 <k,l||d,c>_abab t2_abab(a,c,k,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0249_aa_vv")(da,aa) * t2.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0249_aa_vv"))
    .allocate(tmps.at("0250_bb_voQ"))
    
    // flops: o1v1Q1  = o1v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0250_bb_voQ")(bb,jb,Q)  = chol.at("bb_vvQ")(bb,cb,Q) * t1_1p.at("bb")(cb,jb) )
    
    // r2_1p.at("abab") += +1.00 <a,b||i,c>_abab t1_1p_bb(c,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0250_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += +2.00 <k,b||c,d>_bbbb t2_1p_abab(a,d,i,j) t1_1p_bb(c,k) 
    // flops: o2v2 += o1v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,db)  = chol.at("bb_ovQ")(kb,db,Q) * tmps.at("0250_bb_voQ")(bb,kb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_vv")(bb,db) * t2_1p.at("abab")(aa,db,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 <k,b||c,d>_bbbb t2_abab(a,c,i,k) t1_1p_bb(d,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0250_bb_voQ")(bb,jb,Q) )
    
    // r2_1p.at("abab") += -1.00 <k,b||c,d>_bbbb t2_abab(a,c,i,j) t1_1p_bb(d,k) 
    // flops: o2v2 += o1v2Q1 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin_bb_vv")(bb,cb)  = chol.at("bb_ovQ")(kb,cb,Q) * tmps.at("0250_bb_voQ")(bb,kb,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_1p.at("abab") += -1.00 <k,b||c,d>_abab t2_aaaa(c,a,i,k) t1_1p_bb(d,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0250_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += -2.00 <k,b||c,d>_bbbb t2_1p_abab(a,d,i,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0111_aa_voQ")(aa,ia,Q) * tmps.at("0250_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0251_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0251_bb_voQ")(bb,jb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t2_1p.at("bbbb")(cb,bb,jb,kb) )
    
    // r2_1p.at("abab") += +1.00 <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0251_bb_voQ")(bb,jb,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += -1.00 <a,k||i,c>_abab t2_1p_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0251_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += +2.00 <k,l||i,c>_abab t1_1p_aa(a,k) t2_1p_bbbb(c,b,j,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0251_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,d>_abab t1_aa(a,k) t2_1p_bbbb(d,b,j,l) t1_1p_aa(c,i) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0251_bb_voQ")(bb,jb,Q) * tmps.at("0104_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += +1.00 <k,l||i,c>_abab t1_aa(a,k) t2_1p_bbbb(c,b,j,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0251_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0248_aa_voQ")(aa,ia,Q) * tmps.at("0251_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0248_aa_voQ"))
    
    // r2_1p.at("abab") += -1.00 <a,k||c,d>_abab t1_aa(c,i) t2_1p_bbbb(d,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0109_aa_voQ")(aa,ia,Q) * tmps.at("0251_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0109_aa_voQ"))
    
    // r2_2p.at("abab") += -2.00 <a,k||c,d>_abab t2_1p_bbbb(d,b,j,k) t1_1p_aa(c,i) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0113_aa_voQ")(aa,ia,Q) * tmps.at("0251_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_1p_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0111_aa_voQ")(aa,ia,Q) * tmps.at("0251_bb_voQ")(bb,jb,Q) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,d>_abab t1_aa(c,i) t1_1p_aa(a,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0251_bb_voQ")(bb,jb,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += +1.00 <k,l||c,d>_abab t2_aaaa(c,a,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0106_aa_voQ")(aa,ia,Q) * tmps.at("0251_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0106_aa_voQ"))
    .allocate(tmps.at("0252_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0252_aa_voQ")(ca,ia,Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t2_1p.at("aaaa")(ba,ca,ia,ja) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,d>_abab t2_1p_aaaa(c,a,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0251_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0251_bb_voQ"))
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t2_1p_aaaa(c,a,i,k) t1_1p_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0183_Q")(Q) )
    .deallocate(tmps.at("0183_Q"))
    
    // r1_2p.at("aa") += -2.00 <k,j||c,b>_abab t2_1p_aaaa(c,a,i,k) t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0185_Q")(Q) )
    .deallocate(tmps.at("0185_Q"))
    
    // r2_2p.at("abab") += -2.00 <k,b||d,c>_abab t2_1p_aaaa(d,a,i,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0250_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0250_bb_voQ"))
    
    // r1_1p.at("aa") += -0.50 <k,j||b,i>_aaaa t2_1p_aaaa(b,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 0.50 * chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("0252_aa_voQ")(aa,ka,Q) )
    
    // r1_1p.at("aa") += -0.50 <j,a||b,c>_aaaa t2_1p_aaaa(b,c,i,j) 
    // flops: o1v1 += o1v2Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.50 * chol.at("aa_vvQ")(aa,ca,Q) * tmps.at("0252_aa_voQ")(ca,ia,Q) )
    
    // r1_2p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_1p_aa(a,j) t2_1p_aaaa(b,c,i,k) 
    // flops: o1v1 += o2v1Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ja)  = chol.at("aa_ovQ")(ja,ca,Q) * tmps.at("0252_aa_voQ")(ca,ia,Q) )
    ( r1_2p.at("aa")(aa,ia) += tmps.at("bin_aa_oo")(ia,ja) * t1_1p.at("aa")(aa,ja) )
    
    // r2_1p.at("abab") += -1.00 <k,b||c,j>_abab t2_1p_aaaa(c,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0252_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r2_1p.at("abab") += +0.50 <l,k||c,d>_aaaa t2_abab(a,b,k,j) t2_1p_aaaa(c,d,i,l) 
    // flops: o2v2 += o2v1Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,ka)  = chol.at("aa_ovQ")(ka,da,Q) * tmps.at("0252_aa_voQ")(da,ia,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.50 * tmps.at("bin_aa_oo")(ia,ka) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_aaaa t2_1p_abab(a,b,l,j) t2_1p_aaaa(c,d,i,k) 
    // flops: o2v2 += o2v1Q1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_aa_oo")(ia,la)  = chol.at("aa_ovQ")(la,da,Q) * tmps.at("0252_aa_voQ")(da,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_aa_oo")(ia,la) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t2_1p_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0239_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0239_bb_voQ"))
    
    // r1_2p.at("aa") += +1.00 <k,j||b,c>_aaaa t2_1p_aaaa(c,a,k,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += tmps.at("0252_aa_voQ")(aa,ka,Q) * tmps.at("0104_aa_ooQ")(ka,ia,Q) )
    .deallocate(tmps.at("0104_aa_ooQ"))
    
    // r1_1p.at("aa") += -1.00 <k,j||c,b>_abab t1_bb(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0181_Q")(Q) )
    .deallocate(tmps.at("0181_Q"))
    
    // r1_1p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0176_Q")(Q) )
    .deallocate(tmps.at("0176_Q"))
    
    // r1_1p.at("aa") += +0.50 <k,j||b,c>_aaaa t1_aa(b,i) t2_1p_aaaa(c,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 0.50 * tmps.at("0252_aa_voQ")(aa,ka,Q) * tmps.at("0098_aa_ooQ")(ka,ia,Q) )
    .deallocate(tmps.at("0098_aa_ooQ"))
    
    // r1_1p.at("aa") += +0.50 <k,j||b,c>_aaaa t1_aa(a,j) t2_1p_aaaa(b,c,i,k) 
    // flops: o1v1 += o2v1Q1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin_aa_oo")(ia,ja)  = chol.at("aa_ovQ")(ja,ca,Q) * tmps.at("0252_aa_voQ")(ca,ia,Q) )
    ( r1_1p.at("aa")(aa,ia) += 0.50 * tmps.at("bin_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,j>_abab t1_bb(b,k) t2_1p_aaaa(c,a,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0252_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,j>_abab t2_1p_aaaa(c,a,i,l) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0252_aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t2_1p_aaaa(d,a,i,l) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0107_bb_ooQ")(kb,jb,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||d,c>_abab t2_bbbb(c,b,j,k) t2_1p_aaaa(d,a,i,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0100_bb_voQ")(bb,jb,Q) * tmps.at("0252_aa_voQ")(aa,ia,Q) )
    .deallocate(tmps.at("0100_bb_voQ"))
    
    // r2_1p.at("abab") += -1.00 <k,b||d,c>_abab t1_bb(c,j) t2_1p_aaaa(d,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0097_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0097_bb_voQ"))
    
    // r2_2p.at("abab") += +2.00 <l,k||d,c>_abab t1_bb(c,j) t2_1p_aaaa(d,a,i,l) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0107_bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0107_bb_ooQ"))
    
    // r2_2p.at("abab") += +2.00 <l,k||d,c>_abab t1_bb(b,k) t2_1p_aaaa(d,a,i,l) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0115_bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .deallocate(tmps.at("0115_bb_ooQ"))
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t2_abab(c,b,k,j) t2_1p_aaaa(d,a,i,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0252_aa_voQ")(aa,ia,Q) * tmps.at("0102_bb_voQ")(bb,jb,Q) )
    .deallocate(tmps.at("0102_bb_voQ"))
    .allocate(tmps.at("0253_aa_vv"))
    
    // flops: o0v2  = o1v2Q1
    //  mems: o0v2  = o0v2
    ( tmps.at("0253_aa_vv")(ca,aa)  = tmps.at("0252_aa_voQ")(aa,la,Q) * chol.at("aa_ovQ")(la,ca,Q) )
    .deallocate(tmps.at("0252_aa_voQ"))
    
    // r2_1p.at("abab") += +0.50 <l,k||c,d>_aaaa t2_abab(c,b,i,j) t2_1p_aaaa(d,a,l,k) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.50 * tmps.at("0253_aa_vv")(ca,aa) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p.at("abab") += -1.00 <l,k||c,d>_aaaa t2_1p_aaaa(c,a,l,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += tmps.at("0253_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0253_aa_vv"))
    .allocate(tmps.at("0254_aa_vv"))
    
    // flops: o0v2  = o1v2Q1
    //  mems: o0v2  = o0v2
    ( tmps.at("0254_aa_vv")(ca,aa)  = tmps.at("0111_aa_voQ")(aa,la,Q) * chol.at("aa_ovQ")(la,ca,Q) )
    .deallocate(tmps.at("0111_aa_voQ"))
    
    // r2_1p.at("abab") += -0.50 <l,k||c,d>_abab t2_abab(c,b,i,j) t2_1p_abab(a,d,l,k) 
    //            += -0.50 <k,l||c,d>_abab t2_abab(c,b,i,j) t2_1p_abab(a,d,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0254_aa_vv")(ca,aa) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p.at("abab") += -1.00 <l,k||d,c>_abab t2_1p_abab(a,c,l,k) t2_1p_abab(d,b,i,j) 
    //            += -1.00 <k,l||d,c>_abab t2_1p_abab(a,c,k,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0254_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0254_aa_vv"))
    .allocate(tmps.at("0255_aa_vv"))
    
    // flops: o0v2  = o1v2Q1
    //  mems: o0v2  = o0v2
    ( tmps.at("0255_aa_vv")(ca,aa)  = tmps.at("0113_aa_voQ")(aa,ka,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    .deallocate(tmps.at("0113_aa_voQ"))
    
    // r2_1p.at("abab") += -1.00 <k,a||c,d>_aaaa t2_abab(c,b,i,j) t1_1p_aa(d,k) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0255_aa_vv")(ca,aa) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,a||c,d>_aaaa t2_1p_abab(d,b,i,j) t1_1p_aa(c,k) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0255_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0255_aa_vv"))

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