/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_qed_ccsd_os_resid_2.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_os::resid_part2(
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
        
    // flops: o2v2  = o3v2 o2v1 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v0 o2v2 o2v2
    ( tmps.at("0043_bbbb_vvoo")(ab,bb,ib,jb)  = dp.at("bb_oo")(kb,ib) * t2.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("bin1_bb_oo")(ib,kb)  = dp.at("bb_ov")(kb,cb) * t1.at("bb")(cb,ib) )
    ( tmps.at("0043_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_oo")(ib,kb) * t2.at("bbbb")(ab,bb,jb,kb) )
    
    // r2[bbbb] += +1.000 P(i,j) d-_bb(k,i) t0_1p t2_bbbb(a,b,j,k) 
    //            += +1.000 P(i,j) d-_bb(k,c) t0_1p t1_bb(c,i) t2_bbbb(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) -= t0_1p * tmps.at("0043_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2[bbbb] += +1.000 P(i,j) d-_bb(k,i) t0_1p t2_bbbb(a,b,j,k) 
    //            += +1.000 P(i,j) d-_bb(k,c) t0_1p t1_bb(c,i) t2_bbbb(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) += t0_1p * tmps.at("0043_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_1p[bbbb] += +1.000 P(i,j) d+_bb(k,i) t2_bbbb(a,b,j,k) 
    //               += +1.000 P(i,j) d+_bb(k,c) t1_bb(c,i) t2_bbbb(a,b,j,k) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0043_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_1p[bbbb] += +1.000 P(i,j) d+_bb(k,i) t2_bbbb(a,b,j,k) 
    //               += +1.000 P(i,j) d+_bb(k,c) t1_bb(c,i) t2_bbbb(a,b,j,k) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0043_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2_1p[bbbb] += +2.000 P(i,j) d-_bb(k,i) t0_2p t2_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t0_2p t1_bb(c,i) t2_bbbb(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * t0_2p * tmps.at("0043_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2_1p[bbbb] += +2.000 P(i,j) d-_bb(k,i) t0_2p t2_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) d-_bb(k,c) t0_2p t1_bb(c,i) t2_bbbb(a,b,j,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 2.000 * t0_2p * tmps.at("0043_bbbb_vvoo")(ab,bb,ib,jb) )
    .deallocate(tmps.at("0043_bbbb_vvoo"))
    .allocate(tmps.at("0044_bbbb_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o3v2 o3v2 o2v2 o2v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2.at("bbbb")(cb,ab,ib,jb) )
    ( tmps.at("0044_bbbb_vvoo")(ab,bb,ib,jb)  = tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1_1p.at("bb")(bb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2_1p.at("bbbb")(cb,ab,ib,jb) )
    ( tmps.at("0044_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("0044_bbbb_vvoo")(ab,bb,ib,jb) += dp.at("bb_vv")(ab,cb) * t2_1p.at("bbbb")(cb,bb,ib,jb) )
    
    // r2[bbbb] += +1.000 P(a,b) d-_bb(a,c) t2_1p_bbbb(c,b,i,j) 
    //            += -1.000 P(a,b) d-_bb(k,c) t1_1p_bb(a,k) t2_bbbb(c,b,i,j) 
    //            += -1.000 P(a,b) d-_bb(k,c) t1_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0044_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2[bbbb] += +1.000 P(a,b) d-_bb(a,c) t2_1p_bbbb(c,b,i,j) 
    //            += -1.000 P(a,b) d-_bb(k,c) t1_1p_bb(a,k) t2_bbbb(c,b,i,j) 
    //            += -1.000 P(a,b) d-_bb(k,c) t1_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0044_bbbb_vvoo")(bb,ab,ib,jb) )
    
    // r2_1p[bbbb] += +1.000 P(a,b) d-_bb(a,c) t0_1p t2_1p_bbbb(c,b,i,j) 
    //               += -1.000 P(a,b) d-_bb(k,c) t0_1p t1_1p_bb(a,k) t2_bbbb(c,b,i,j) 
    //               += -1.000 P(a,b) d-_bb(k,c) t0_1p t1_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= t0_1p * tmps.at("0044_bbbb_vvoo")(bb,ab,ib,jb) )
    
    // r2_1p[bbbb] += +1.000 P(a,b) d-_bb(a,c) t0_1p t2_1p_bbbb(c,b,i,j) 
    //               += -1.000 P(a,b) d-_bb(k,c) t0_1p t1_1p_bb(a,k) t2_bbbb(c,b,i,j) 
    //               += -1.000 P(a,b) d-_bb(k,c) t0_1p t1_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += t0_1p * tmps.at("0044_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_2p[bbbb] += +2.000 P(a,b) d+_bb(a,c) t2_1p_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d+_bb(k,c) t1_1p_bb(a,k) t2_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d+_bb(k,c) t1_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0044_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_2p[bbbb] += +2.000 P(a,b) d+_bb(a,c) t2_1p_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d+_bb(k,c) t1_1p_bb(a,k) t2_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d+_bb(k,c) t1_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0044_bbbb_vvoo")(bb,ab,ib,jb) )
    
    // r2_2p[bbbb] += +4.000 P(a,b) d-_bb(a,c) t0_2p t2_1p_bbbb(c,b,i,j) 
    //               += -4.000 P(a,b) d-_bb(k,c) t0_2p t1_1p_bb(a,k) t2_bbbb(c,b,i,j) 
    //               += -4.000 P(a,b) d-_bb(k,c) t0_2p t1_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 4.000 * t0_2p * tmps.at("0044_bbbb_vvoo")(bb,ab,ib,jb) )
    
    // r2_2p[bbbb] += +4.000 P(a,b) d-_bb(a,c) t0_2p t2_1p_bbbb(c,b,i,j) 
    //               += -4.000 P(a,b) d-_bb(k,c) t0_2p t1_1p_bb(a,k) t2_bbbb(c,b,i,j) 
    //               += -4.000 P(a,b) d-_bb(k,c) t0_2p t1_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 4.000 * t0_2p * tmps.at("0044_bbbb_vvoo")(ab,bb,ib,jb) )
    .deallocate(tmps.at("0044_bbbb_vvoo"))
    .allocate(tmps.at("0045_bbbb_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o3v2 o3v2 o2v2 o3v2 o3v2 o2v2 o2v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2.at("bbbb")(cb,ab,ib,jb) )
    ( tmps.at("0045_bbbb_vvoo")(ab,bb,ib,jb)  = tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1_2p.at("bb")(bb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2_1p.at("bbbb")(cb,ab,ib,jb) )
    ( tmps.at("0045_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1_1p.at("bb")(bb,kb) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2_2p.at("bbbb")(cb,ab,ib,jb) )
    ( tmps.at("0045_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("0045_bbbb_vvoo")(ab,bb,ib,jb) += dp.at("bb_vv")(ab,cb) * t2_2p.at("bbbb")(cb,bb,ib,jb) )
    
    // r2_1p[bbbb] += +2.000 P(a,b) d-_bb(a,c) t2_2p_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t1_2p_bb(a,k) t2_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t1_1p_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t1_bb(a,k) t2_2p_bbbb(c,b,i,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0045_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_1p[bbbb] += +2.000 P(a,b) d-_bb(a,c) t2_2p_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t1_2p_bb(a,k) t2_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t1_1p_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t1_bb(a,k) t2_2p_bbbb(c,b,i,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0045_bbbb_vvoo")(bb,ab,ib,jb) )
    
    // r2_2p[bbbb] += +2.000 P(a,b) d-_bb(a,c) t0_1p t2_2p_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t0_1p t1_2p_bb(a,k) t2_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t0_1p t1_1p_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t0_1p t1_bb(a,k) t2_2p_bbbb(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * t0_1p * tmps.at("0045_bbbb_vvoo")(bb,ab,ib,jb) )
    
    // r2_2p[bbbb] += +2.000 P(a,b) d-_bb(a,c) t0_1p t2_2p_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t0_1p t1_2p_bb(a,k) t2_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t0_1p t1_1p_bb(a,k) t2_1p_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t0_1p t1_bb(a,k) t2_2p_bbbb(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * t0_1p * tmps.at("0045_bbbb_vvoo")(ab,bb,ib,jb) )
    .deallocate(tmps.at("0045_bbbb_vvoo"))
    .allocate(tmps.at("0046_bbbb_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o2v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2.at("bbbb")(cb,ab,ib,jb) )
    ( tmps.at("0046_bbbb_vvoo")(ab,bb,ib,jb)  = tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("0046_bbbb_vvoo")(ab,bb,ib,jb) += dp.at("bb_vv")(ab,cb) * t2.at("bbbb")(cb,bb,ib,jb) )
    
    // r2[bbbb] += +1.000 P(a,b) d-_bb(a,c) t0_1p t2_bbbb(c,b,i,j) 
    //            += -1.000 P(a,b) d-_bb(k,c) t0_1p t1_bb(a,k) t2_bbbb(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) -= t0_1p * tmps.at("0046_bbbb_vvoo")(bb,ab,ib,jb) )
    
    // r2[bbbb] += +1.000 P(a,b) d-_bb(a,c) t0_1p t2_bbbb(c,b,i,j) 
    //            += -1.000 P(a,b) d-_bb(k,c) t0_1p t1_bb(a,k) t2_bbbb(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) += t0_1p * tmps.at("0046_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_1p[bbbb] += +1.000 P(a,b) d+_bb(a,c) t2_bbbb(c,b,i,j) 
    //               += -1.000 P(a,b) d+_bb(k,c) t1_bb(a,k) t2_bbbb(c,b,i,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0046_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_1p[bbbb] += +1.000 P(a,b) d+_bb(a,c) t2_bbbb(c,b,i,j) 
    //               += -1.000 P(a,b) d+_bb(k,c) t1_bb(a,k) t2_bbbb(c,b,i,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0046_bbbb_vvoo")(bb,ab,ib,jb) )
    
    // r2_1p[bbbb] += +2.000 P(a,b) d-_bb(a,c) t0_2p t2_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t0_2p t1_bb(a,k) t2_bbbb(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * t0_2p * tmps.at("0046_bbbb_vvoo")(bb,ab,ib,jb) )
    
    // r2_1p[bbbb] += +2.000 P(a,b) d-_bb(a,c) t0_2p t2_bbbb(c,b,i,j) 
    //               += -2.000 P(a,b) d-_bb(k,c) t0_2p t1_bb(a,k) t2_bbbb(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 2.000 * t0_2p * tmps.at("0046_bbbb_vvoo")(ab,bb,ib,jb) )
    .deallocate(tmps.at("0046_bbbb_vvoo"))
    .allocate(tmps.at("0047_aabb_ovvv"))
    
    // flops: o1v3  = o1v3Q1
    //  mems: o1v3  = o1v3
    ( tmps.at("0047_aabb_ovvv")(ka,da,bb,cb)  = chol.at("aa_ovQ")(ka,da,Q) * chol.at("bb_vvQ")(bb,cb,Q) )
    
    // r1[bb] += +0.500 <j,a||c,b>_abab t2_abab(c,b,j,i) 
    //          += +0.500 <j,a||b,c>_abab t2_abab(b,c,j,i) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += t2.at("abab")(ca,bb,ja,ib) * tmps.at("0047_aabb_ovvv")(ja,ca,ab,bb) )
    
    // r1_1p[bb] += +0.500 <j,a||c,b>_abab t2_1p_abab(c,b,j,i) 
    //             += +0.500 <j,a||b,c>_abab t2_1p_abab(b,c,j,i) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += t2_1p.at("abab")(ca,bb,ja,ib) * tmps.at("0047_aabb_ovvv")(ja,ca,ab,bb) )
    
    // r1_2p[bb] += +1.000 <j,a||c,b>_abab t2_2p_abab(c,b,j,i) 
    //             += +1.000 <j,a||b,c>_abab t2_2p_abab(b,c,j,i) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * t2_2p.at("abab")(ca,bb,ja,ib) * tmps.at("0047_aabb_ovvv")(ja,ca,ab,bb) )
    
    // r2_2p[abab] += -1.000 <k,b||d,c>_abab t1_aa(a,k) t2_2p_abab(d,c,i,j) 
    //               += -1.000 <k,b||c,d>_abab t1_aa(a,k) t2_2p_abab(c,d,i,j) 
    // flops: o2v2 += o3v3 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(da,cb,ia,jb) * tmps.at("0047_aabb_ovvv")(ka,da,bb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 <k,b||c,d>_abab t1_2p_aa(c,i) t2_abab(a,d,k,j) 
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_bbaa_vvoo")(bb,db,ia,ka)  = t1_2p.at("aa")(ca,ia) * tmps.at("0047_aabb_ovvv")(ka,ca,bb,db) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bbaa_vvoo")(bb,db,ia,ka) * t2.at("abab")(aa,db,ka,jb) )
    .allocate(tmps.at("0048_abab_ovoo"))
    
    // flops: o3v1  = o3v2 o3v3 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1
    ( tmps.at("0048_abab_ovoo")(ka,bb,ia,jb)  = t2.at("abab")(ca,bb,ia,jb) * f.at("aa_ov")(ka,ca) )
    ( tmps.at("0048_abab_ovoo")(ka,bb,ia,jb) += t2.at("abab")(da,cb,ia,jb) * tmps.at("0047_aabb_ovvv")(ka,da,bb,cb) )
    
    // r2[abab] += -1.000 f_aa(k,c) t1_aa(a,k) t2_abab(c,b,i,j) 
    //            += -0.500 <k,b||d,c>_abab t1_aa(a,k) t2_abab(d,c,i,j) 
    //            += -0.500 <k,b||c,d>_abab t1_aa(a,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0048_abab_ovoo")(ka,bb,ia,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += -1.000 f_aa(k,c) t1_1p_aa(a,k) t2_abab(c,b,i,j) 
    //               += -0.500 <k,b||d,c>_abab t1_1p_aa(a,k) t2_abab(d,c,i,j) 
    //               += -0.500 <k,b||c,d>_abab t1_1p_aa(a,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0048_abab_ovoo")(ka,bb,ia,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 f_aa(k,c) t1_2p_aa(a,k) t2_abab(c,b,i,j) 
    //               += -1.000 <k,b||d,c>_abab t1_2p_aa(a,k) t2_abab(d,c,i,j) 
    //               += -1.000 <k,b||c,d>_abab t1_2p_aa(a,k) t2_abab(c,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0048_abab_ovoo")(ka,bb,ia,jb) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0048_abab_ovoo"))
    .allocate(tmps.at("0049_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0049_Q")(Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t1.at("aa")(aa,ia) )
    
    // r1[aa] += +1.000 <a,j||i,b>_aaaa t1_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0049_Q")(Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r1[aa] += +1.000 <j,k||b,c>_abab t1_aa(b,j) t2_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0049_Q")(Q) )
    
    // r1_1p[aa] += +1.000 <j,k||b,c>_abab t1_aa(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0028_aa_voQ")(aa,ia,Q) * tmps.at("0049_Q")(Q) )
    
    // r1[bb] += +1.000 <j,a||b,i>_abab t1_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += tmps.at("0049_Q")(Q) * chol.at("bb_voQ")(ab,ib,Q) )
    .allocate(tmps.at("0050_aaaa_vvoo"))
    
    // flops: o2v2  = o0v2Q1 o2v3 o1v1Q1 o0v2Q1 o2v3 o2v2 o2v3 o2v2
    //  mems: o2v2  = o0v2 o2v2 o0v0Q1 o0v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aa_vv")(aa,da)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("0049_Q")(Q) )
    ( tmps.at("0050_aaaa_vvoo")(aa,ba,ia,ja)  = tmps.at("bin1_aa_vv")(aa,da) * t2.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vv")(aa,da)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("0050_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vv")(aa,da) * t2.at("aaaa")(da,ba,ia,ja) )
    ( tmps.at("0050_aaaa_vvoo")(aa,ba,ia,ja) += f.at("aa_vv")(aa,ca) * t2.at("aaaa")(ca,ba,ia,ja) )
    
    // r2[aaaa] += +1.000 P(a,b) f_aa(a,c) t2_aaaa(c,b,i,j) 
    //            += -1.000 P(a,b) <a,k||c,d>_aaaa t1_aa(c,k) t2_aaaa(d,b,i,j) 
    //            += +1.000 P(a,b) <a,k||d,c>_abab t1_bb(c,k) t2_aaaa(d,b,i,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0050_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2[aaaa] += +1.000 P(a,b) f_aa(a,c) t2_aaaa(c,b,i,j) 
    //            += -1.000 P(a,b) <a,k||c,d>_aaaa t1_aa(c,k) t2_aaaa(d,b,i,j) 
    //            += +1.000 P(a,b) <a,k||d,c>_abab t1_bb(c,k) t2_aaaa(d,b,i,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0050_aaaa_vvoo")(ba,aa,ia,ja) )
    .deallocate(tmps.at("0050_aaaa_vvoo"))
    .allocate(tmps.at("0051_aa_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0051_aa_voQ")(ba,ia,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t2.at("aaaa")(ca,ba,ia,ja) )
    
    // r1[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0049_Q")(Q) )
    
    // r1[aa] += +0.500 <j,k||i,b>_aaaa t2_aaaa(b,a,j,k) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += 0.500 * chol.at("aa_ooQ")(ja,ia,Q) * tmps.at("0051_aa_voQ")(aa,ja,Q) )
    
    // r2[abab] += -1.000 <k,b||c,j>_abab t2_aaaa(c,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0051_aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0052_aa_voQ"))
    
    // flops: o1v1Q1  = o1v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0052_aa_voQ")(aa,ja,Q)  = chol.at("aa_vvQ")(aa,ba,Q) * t1.at("aa")(ba,ja) )
    
    // r1[aa] += -1.000 <a,j||b,c>_aaaa t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0049_Q")(Q) )
    .allocate(tmps.at("0053_aa_ooQ"))
    
    // flops: o2v0Q1  = o2v1Q1
    //  mems: o2v0Q1  = o2v0Q1
    ( tmps.at("0053_aa_ooQ")(ja,ia,Q)  = chol.at("aa_ovQ")(ja,aa,Q) * t1.at("aa")(aa,ia) )
    
    // r1_1p[aa] += -0.500 <j,k||b,c>_abab t1_aa(b,i) t2_1p_abab(a,c,j,k) 
    //             += -0.500 <k,j||b,c>_abab t1_aa(b,i) t2_1p_abab(a,c,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0028_aa_voQ")(aa,ja,Q) * tmps.at("0053_aa_ooQ")(ja,ia,Q) )
    
    // r1[aa] += -0.500 <j,k||b,c>_abab t1_aa(b,i) t2_abab(a,c,j,k) 
    //          += -0.500 <k,j||b,c>_abab t1_aa(b,i) t2_abab(a,c,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0025_aa_voQ")(aa,ja,Q) * tmps.at("0053_aa_ooQ")(ja,ia,Q) )
    .allocate(tmps.at("0054_aaaa_vovo"))
    
    // flops: o2v2  = o2v1Q1 o4v0Q1 o4v1 o3v2 o2v2Q1 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2
    //  mems: o2v2  = o2v0Q1 o4v0 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0053_aa_ooQ")(la,ja,Q) * tmps.at("bin1_aa_ooQ")(ia,ka,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0054_aaaa_vovo")(aa,ia,ba,ja)  = tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) * t1.at("aa")(ba,la) )
    ( tmps.at("0054_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0025_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0054_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0052_aa_voQ")(ba,ja,Q) )
    ( tmps.at("0054_aaaa_vovo")(aa,ia,ba,ja) += tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0051_aa_voQ")(ba,ja,Q) )
    
    // r2[aaaa] += +1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //            += -1.000 P(i,j) <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_abab(b,d,j,l) 
    //            += -1.000 P(i,j) <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_aaaa(d,b,j,l) 
    //            += -1.000 <a,b||d,c>_aaaa t1_aa(c,i) t1_aa(d,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0054_aaaa_vovo")(aa,ia,ba,ja) )
    
    // r2[aaaa] += +1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_aa(b,l) t1_aa(c,i) t1_aa(d,j) 
    //            += -1.000 P(i,j) <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_abab(b,d,j,l) 
    //            += -1.000 P(i,j) <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_aaaa(d,b,j,l) 
    //            += -1.000 <a,b||d,c>_aaaa t1_aa(c,i) t1_aa(d,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0054_aaaa_vovo")(aa,ja,ba,ia) )
    .deallocate(tmps.at("0054_aaaa_vovo"))
    .allocate(tmps.at("0055_aaaa_oovv"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0055_aaaa_oovv")(ka,ia,aa,ca)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("aa_vvQ")(aa,ca,Q) )
    
    // r1[aa] += +1.000 <a,j||i,b>_aaaa t1_aa(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0055_aaaa_oovv")(ja,ia,aa,ba) * t1.at("aa")(ba,ja) )
    
    // r1_1p[aa] += +1.000 <a,j||i,b>_aaaa t1_1p_aa(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0055_aaaa_oovv")(ja,ia,aa,ba) * t1_1p.at("aa")(ba,ja) )
    
    // r1_2p[aa] += +2.000 <a,j||i,b>_aaaa t1_2p_aa(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0055_aaaa_oovv")(ja,ia,aa,ba) * t1_2p.at("aa")(ba,ja) )
    
    // r2[abab] += +1.000 <a,k||i,c>_aaaa t2_abab(c,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0055_aaaa_oovv")(ka,ia,aa,ca) * t2.at("abab")(ca,bb,ka,jb) )
    
    // r2_1p[abab] += +1.000 <a,k||i,c>_aaaa t2_1p_abab(c,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0055_aaaa_oovv")(ka,ia,aa,ca) * t2_1p.at("abab")(ca,bb,ka,jb) )
    
    // r2_2p[abab] += +2.000 <a,k||i,c>_aaaa t2_2p_abab(c,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0055_aaaa_oovv")(ka,ia,aa,ca) * t2_2p.at("abab")(ca,bb,ka,jb) )
    .allocate(tmps.at("0056_aaaa_ooov"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0056_aaaa_ooov")(ja,ka,ia,aa)  = chol.at("aa_ooQ")(ja,ka,Q) * chol.at("aa_ovQ")(ia,aa,Q) )
    
    // r1[aa] += +0.500 <j,k||i,b>_aaaa t2_aaaa(b,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= 0.500 * tmps.at("0056_aaaa_ooov")(ka,ia,ja,ba) * t2.at("aaaa")(ba,aa,ja,ka) )
    
    // r1_1p[aa] += +0.500 <j,k||i,b>_aaaa t2_1p_aaaa(b,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.500 * tmps.at("0056_aaaa_ooov")(ka,ia,ja,ba) * t2_1p.at("aaaa")(ba,aa,ja,ka) )
    
    // r1_2p[aa] += +1.000 <j,k||i,b>_aaaa t2_2p_aaaa(b,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= tmps.at("0056_aaaa_ooov")(ka,ia,ja,ba) * t2_2p.at("aaaa")(ba,aa,ja,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_abab(c,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(ca,bb,la,jb) * tmps.at("0056_aaaa_ooov")(la,ia,ka,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0057_aaaa_voov"))
    
    // flops: o2v2  = o3v3 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0057_aaaa_voov")(aa,ja,ia,ba)  = tmps.at("0055_aaaa_oovv")(ka,ja,ba,ca) * t2.at("aaaa")(ca,aa,ia,ka) )
    ( tmps.at("0057_aaaa_voov")(aa,ja,ia,ba) += tmps.at("0056_aaaa_ooov")(ja,ka,ia,ba) * t1.at("aa")(aa,ka) )
    
    // r2[aaaa] += -1.000 P(a,b) <a,k||i,j>_aaaa t1_aa(b,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_aaaa(c,b,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0057_aaaa_voov")(ba,ja,ia,aa) )
    
    // r2[aaaa] += -1.000 P(a,b) <a,k||i,j>_aaaa t1_aa(b,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_aaaa(c,b,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0057_aaaa_voov")(ba,ia,ja,aa) )
    
    // r2[aaaa] += -1.000 P(a,b) <a,k||i,j>_aaaa t1_aa(b,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_aaaa(c,b,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0057_aaaa_voov")(aa,ja,ia,ba) )
    
    // r2[aaaa] += -1.000 P(a,b) <a,k||i,j>_aaaa t1_aa(b,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_aaaa(c,b,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0057_aaaa_voov")(aa,ia,ja,ba) )
    .deallocate(tmps.at("0057_aaaa_voov"))
    .allocate(tmps.at("0058_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0058_bb_voQ")(bb,ib,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t2.at("abab")(ca,bb,ja,ib) )
    
    // r1[bb] += -1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_abab(c,a,k,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += tmps.at("0058_bb_voQ")(ab,ib,Q) * tmps.at("0049_Q")(Q) )
    
    // r1_1p[bb] += -0.500 <j,k||c,b>_abab t1_1p_bb(b,i) t2_abab(c,a,j,k) 
    //             += -0.500 <k,j||c,b>_abab t1_1p_bb(b,i) t2_abab(c,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0058_bb_voQ")(ab,kb,Q) * tmps.at("0029_bb_ooQ")(kb,ib,Q) )
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_abab t2_1p_abab(a,d,i,l) t2_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0028_aa_voQ")(aa,ia,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    
    // r1[bb] += -0.500 <j,k||b,i>_abab t2_abab(b,a,j,k) 
    //          += -0.500 <k,j||b,i>_abab t2_abab(b,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= chol.at("bb_ooQ")(kb,ib,Q) * tmps.at("0058_bb_voQ")(ab,kb,Q) )
    
    // r2[abab] += +1.000 <a,k||c,d>_aaaa t1_aa(c,i) t2_abab(d,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += +1.000 <a,k||i,c>_aaaa t2_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += +1.000 <l,k||d,c>_abab t2_abab(a,c,i,k) t2_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    
    // r1[bb] += -0.500 <j,k||c,b>_abab t1_bb(b,i) t2_abab(c,a,j,k) 
    //          += -0.500 <k,j||c,b>_abab t1_bb(b,i) t2_abab(c,a,k,j) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= tmps.at("0058_bb_voQ")(ab,kb,Q) * tmps.at("0026_bb_ooQ")(kb,ib,Q) )
    .allocate(tmps.at("0059_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0059_bb_voQ")(bb,ib,Q)  = chol.at("bb_ovQ")(jb,cb,Q) * t2.at("bbbb")(cb,bb,ib,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t2_1p_abab(a,d,i,l) t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0028_aa_voQ")(aa,ia,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += -1.000 <a,k||c,d>_abab t1_aa(c,i) t2_bbbb(d,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += +1.000 <k,l||c,d>_abab t2_aaaa(c,a,i,k) t2_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += +1.000 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_bbbb(d,b,j,l) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += -1.000 <a,k||i,c>_abab t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r1[bb] += -1.000 <j,k||b,c>_abab t1_aa(b,j) t2_bbbb(c,a,i,k) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= tmps.at("0059_bb_voQ")(ab,ib,Q) * tmps.at("0049_Q")(Q) )
    
    // r1[bb] += +0.500 <j,k||i,b>_bbbb t2_bbbb(b,a,j,k) 
    // flops: o1v1 += o2v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += 0.500 * chol.at("bb_ooQ")(jb,ib,Q) * tmps.at("0059_bb_voQ")(ab,jb,Q) )
    .allocate(tmps.at("0060_bb_voQ"))
    
    // flops: o1v1Q1  = o1v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0060_bb_voQ")(ab,jb,Q)  = chol.at("bb_vvQ")(ab,bb,Q) * t1.at("bb")(bb,jb) )
    
    // r2[abab] += +1.000 <a,b||i,c>_abab t1_bb(c,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += chol.at("aa_voQ")(aa,ia,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += +1.000 <a,b||c,d>_abab t1_aa(c,i) t1_bb(d,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0052_aa_voQ")(aa,ia,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += +1.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_abab(a,d,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0025_aa_voQ")(aa,ia,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    
    // r2[abab] += -1.000 <k,b||d,c>_abab t1_bb(c,j) t2_aaaa(d,a,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0051_aa_voQ")(aa,ia,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    
    // r1[bb] += +1.000 <j,a||b,c>_abab t1_aa(b,j) t1_bb(c,i) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += tmps.at("0060_bb_voQ")(ab,ib,Q) * tmps.at("0049_Q")(Q) )
    
    // r2_1p[abab] += +1.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_1p_abab(a,d,i,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0028_aa_voQ")(aa,ia,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    .allocate(tmps.at("0061_bbbb_vovo"))
    
    // flops: o2v2  = o2v1Q1 o4v0Q1 o4v1 o3v2 o2v2Q1 o2v2 o2v2Q1 o2v2 o2v2Q1 o2v2
    //  mems: o2v2  = o2v0Q1 o4v0 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bb_ooQ")(ib,kb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,ib) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = tmps.at("0026_bb_ooQ")(lb,jb,Q) * tmps.at("bin1_bb_ooQ")(ib,kb,Q) )
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = t1.at("bb")(ab,kb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("0061_bbbb_vovo")(ab,ib,bb,jb)  = tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) * t1.at("bb")(bb,lb) )
    ( tmps.at("0061_bbbb_vovo")(ab,ib,bb,jb) += tmps.at("0058_bb_voQ")(ab,ib,Q) * tmps.at("0058_bb_voQ")(bb,jb,Q) )
    ( tmps.at("0061_bbbb_vovo")(ab,ib,bb,jb) += tmps.at("0060_bb_voQ")(ab,ib,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    ( tmps.at("0061_bbbb_vovo")(ab,ib,bb,jb) += tmps.at("0059_bb_voQ")(ab,ib,Q) * tmps.at("0059_bb_voQ")(bb,jb,Q) )
    
    // r2[bbbb] += +1.000 <l,k||d,c>_bbbb t1_bb(a,k) t1_bb(b,l) t1_bb(c,i) t1_bb(d,j) 
    //            += -1.000 P(i,j) <l,k||c,d>_aaaa t2_abab(c,a,k,i) t2_abab(d,b,l,j) 
    //            += -1.000 P(i,j) <l,k||c,d>_bbbb t2_bbbb(c,a,i,k) t2_bbbb(d,b,j,l) 
    //            += -1.000 <a,b||d,c>_bbbb t1_bb(c,i) t1_bb(d,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0061_bbbb_vovo")(ab,ib,bb,jb) )
    
    // r2[bbbb] += +1.000 <l,k||d,c>_bbbb t1_bb(a,k) t1_bb(b,l) t1_bb(c,i) t1_bb(d,j) 
    //            += -1.000 P(i,j) <l,k||c,d>_aaaa t2_abab(c,a,k,i) t2_abab(d,b,l,j) 
    //            += -1.000 P(i,j) <l,k||c,d>_bbbb t2_bbbb(c,a,i,k) t2_bbbb(d,b,j,l) 
    //            += -1.000 <a,b||d,c>_bbbb t1_bb(c,i) t1_bb(d,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0061_bbbb_vovo")(ab,jb,bb,ib) )
    .deallocate(tmps.at("0061_bbbb_vovo"))
    .allocate(tmps.at("0062_bbbb_vovo"))
    
    // flops: o2v2  = o2v2Q1 o2v2Q1 o2v2 o3v1Q1 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2 o3v1 o2v2 o2v2
    ( tmps.at("0062_bbbb_vovo")(ab,ib,bb,jb)  = tmps.at("0058_bb_voQ")(ab,ib,Q) * tmps.at("0060_bb_voQ")(bb,jb,Q) )
    ( tmps.at("0062_bbbb_vovo")(ab,ib,bb,jb) += chol.at("bb_voQ")(bb,jb,Q) * tmps.at("0058_bb_voQ")(ab,ib,Q) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = tmps.at("0059_bb_voQ")(bb,jb,Q) * chol.at("bb_ooQ")(kb,ib,Q) )
    ( tmps.at("0062_bbbb_vovo")(ab,ib,bb,jb) += tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1.at("bb")(ab,kb) )
    
    // r2[bbbb] += -1.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_bb(a,k) t2_bbbb(c,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) <k,a||c,i>_abab t2_abab(c,b,k,j) 
    //            += +1.000 P(i,j) P(a,b) <k,a||d,c>_abab t1_bb(c,i) t2_abab(d,b,k,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0062_bbbb_vovo")(ab,ib,bb,jb) )
    
    // r2[bbbb] += -1.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_bb(a,k) t2_bbbb(c,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) <k,a||c,i>_abab t2_abab(c,b,k,j) 
    //            += +1.000 P(i,j) P(a,b) <k,a||d,c>_abab t1_bb(c,i) t2_abab(d,b,k,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0062_bbbb_vovo")(ab,jb,bb,ib) )
    
    // r2[bbbb] += -1.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_bb(a,k) t2_bbbb(c,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) <k,a||c,i>_abab t2_abab(c,b,k,j) 
    //            += +1.000 P(i,j) P(a,b) <k,a||d,c>_abab t1_bb(c,i) t2_abab(d,b,k,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0062_bbbb_vovo")(bb,jb,ab,ib) )
    
    // r2[bbbb] += -1.000 P(i,j) P(a,b) <l,k||i,c>_bbbb t1_bb(a,k) t2_bbbb(c,b,j,l) 
    //            += +1.000 P(i,j) P(a,b) <k,a||c,i>_abab t2_abab(c,b,k,j) 
    //            += +1.000 P(i,j) P(a,b) <k,a||d,c>_abab t1_bb(c,i) t2_abab(d,b,k,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0062_bbbb_vovo")(bb,ib,ab,jb) )
    .deallocate(tmps.at("0062_bbbb_vovo"))
    .allocate(tmps.at("0063_bbbb_oovv"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0063_bbbb_oovv")(kb,ib,ab,cb)  = chol.at("bb_ooQ")(kb,ib,Q) * chol.at("bb_vvQ")(ab,cb,Q) )
    
    // r1[bb] += +1.000 <a,j||i,b>_bbbb t1_bb(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= tmps.at("0063_bbbb_oovv")(jb,ib,ab,bb) * t1.at("bb")(bb,jb) )
    
    // r1_1p[bb] += +1.000 <a,j||i,b>_bbbb t1_1p_bb(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0063_bbbb_oovv")(jb,ib,ab,bb) * t1_1p.at("bb")(bb,jb) )
    
    // r1_2p[bb] += +2.000 <a,j||i,b>_bbbb t1_2p_bb(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0063_bbbb_oovv")(jb,ib,ab,bb) * t1_2p.at("bb")(bb,jb) )
    
    // r2[abab] += +1.000 <b,k||j,c>_bbbb t2_abab(a,c,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0063_bbbb_oovv")(kb,jb,bb,cb) * t2.at("abab")(aa,cb,ia,kb) )
    
    // r2_1p[abab] += +1.000 <b,k||j,c>_bbbb t2_1p_abab(a,c,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0063_bbbb_oovv")(kb,jb,bb,cb) * t2_1p.at("abab")(aa,cb,ia,kb) )
    
    // r2_2p[abab] += +2.000 <b,k||j,c>_bbbb t2_2p_abab(a,c,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0063_bbbb_oovv")(kb,jb,bb,cb) * t2_2p.at("abab")(aa,cb,ia,kb) )
    .allocate(tmps.at("0064_bbbb_ooov"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0064_bbbb_ooov")(jb,kb,ib,ab)  = chol.at("bb_ooQ")(jb,kb,Q) * chol.at("bb_ovQ")(ib,ab,Q) )
    
    // r1[bb] += +0.500 <j,k||i,b>_bbbb t2_bbbb(b,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= 0.500 * tmps.at("0064_bbbb_ooov")(kb,ib,jb,bb) * t2.at("bbbb")(bb,ab,jb,kb) )
    
    // r1_1p[bb] += +0.500 <j,k||i,b>_bbbb t2_1p_bbbb(b,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= 0.500 * tmps.at("0064_bbbb_ooov")(kb,ib,jb,bb) * t2_1p.at("bbbb")(bb,ab,jb,kb) )
    
    // r1_2p[bb] += +1.000 <j,k||i,b>_bbbb t2_2p_bbbb(b,a,j,k) 
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= tmps.at("0064_bbbb_ooov")(kb,ib,jb,bb) * t2_2p.at("bbbb")(bb,ab,jb,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_2p_abab(a,c,i,l) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t2_2p.at("abab")(aa,cb,ia,lb) * tmps.at("0064_bbbb_ooov")(lb,jb,kb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    .allocate(tmps.at("0065_bbbb_voov"))
    
    // flops: o2v2  = o3v3 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0065_bbbb_voov")(ab,jb,ib,bb)  = tmps.at("0063_bbbb_oovv")(kb,jb,bb,cb) * t2.at("bbbb")(cb,ab,ib,kb) )
    ( tmps.at("0065_bbbb_voov")(ab,jb,ib,bb) += tmps.at("0064_bbbb_ooov")(jb,kb,ib,bb) * t1.at("bb")(ab,kb) )
    
    // r2[bbbb] += -1.000 P(a,b) <a,k||i,j>_bbbb t1_bb(b,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_bbbb(c,b,j,k) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0065_bbbb_voov")(bb,jb,ib,ab) )
    
    // r2[bbbb] += -1.000 P(a,b) <a,k||i,j>_bbbb t1_bb(b,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_bbbb(c,b,j,k) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0065_bbbb_voov")(bb,ib,jb,ab) )
    
    // r2[bbbb] += -1.000 P(a,b) <a,k||i,j>_bbbb t1_bb(b,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_bbbb(c,b,j,k) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0065_bbbb_voov")(ab,jb,ib,bb) )
    
    // r2[bbbb] += -1.000 P(a,b) <a,k||i,j>_bbbb t1_bb(b,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_bbbb(c,b,j,k) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0065_bbbb_voov")(ab,ib,jb,bb) )
    .deallocate(tmps.at("0065_bbbb_voov"))
    .allocate(tmps.at("0066_aaaa_voov"))
    
    // flops: o2v2  = o3v3 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0066_aaaa_voov")(aa,ja,ia,ba)  = tmps.at("0055_aaaa_oovv")(ka,ja,ba,ca) * t2_1p.at("aaaa")(ca,aa,ia,ka) )
    ( tmps.at("0066_aaaa_voov")(aa,ja,ia,ba) += tmps.at("0056_aaaa_ooov")(ja,ka,ia,ba) * t1_1p.at("aa")(aa,ka) )
    
    // r2_1p[aaaa] += -1.000 P(a,b) <a,k||i,j>_aaaa t1_1p_aa(b,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_1p_aaaa(c,b,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0066_aaaa_voov")(ba,ja,ia,aa) )
    
    // r2_1p[aaaa] += -1.000 P(a,b) <a,k||i,j>_aaaa t1_1p_aa(b,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_1p_aaaa(c,b,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0066_aaaa_voov")(ba,ia,ja,aa) )
    
    // r2_1p[aaaa] += -1.000 P(a,b) <a,k||i,j>_aaaa t1_1p_aa(b,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_1p_aaaa(c,b,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0066_aaaa_voov")(aa,ja,ia,ba) )
    
    // r2_1p[aaaa] += -1.000 P(a,b) <a,k||i,j>_aaaa t1_1p_aa(b,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_1p_aaaa(c,b,j,k) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0066_aaaa_voov")(aa,ia,ja,ba) )
    .deallocate(tmps.at("0066_aaaa_voov"))
    .allocate(tmps.at("0067_bbbb_voov"))
    
    // flops: o2v2  = o3v3 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0067_bbbb_voov")(ab,jb,ib,bb)  = tmps.at("0063_bbbb_oovv")(kb,jb,bb,cb) * t2_1p.at("bbbb")(cb,ab,ib,kb) )
    ( tmps.at("0067_bbbb_voov")(ab,jb,ib,bb) += tmps.at("0064_bbbb_ooov")(jb,kb,ib,bb) * t1_1p.at("bb")(ab,kb) )
    
    // r2_1p[bbbb] += -1.000 P(a,b) <a,k||i,j>_bbbb t1_1p_bb(b,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_1p_bbbb(c,b,j,k) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0067_bbbb_voov")(bb,jb,ib,ab) )
    
    // r2_1p[bbbb] += -1.000 P(a,b) <a,k||i,j>_bbbb t1_1p_bb(b,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_1p_bbbb(c,b,j,k) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0067_bbbb_voov")(bb,ib,jb,ab) )
    
    // r2_1p[bbbb] += -1.000 P(a,b) <a,k||i,j>_bbbb t1_1p_bb(b,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_1p_bbbb(c,b,j,k) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0067_bbbb_voov")(ab,jb,ib,bb) )
    
    // r2_1p[bbbb] += -1.000 P(a,b) <a,k||i,j>_bbbb t1_1p_bb(b,k) 
    //               += -1.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_1p_bbbb(c,b,j,k) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0067_bbbb_voov")(ab,ib,jb,bb) )
    .deallocate(tmps.at("0067_bbbb_voov"))
    .allocate(tmps.at("0068_aaaa_voov"))
    
    // flops: o2v2  = o3v3 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0068_aaaa_voov")(aa,ja,ia,ba)  = tmps.at("0055_aaaa_oovv")(ka,ja,ba,ca) * t2_2p.at("aaaa")(ca,aa,ia,ka) )
    ( tmps.at("0068_aaaa_voov")(aa,ja,ia,ba) += tmps.at("0056_aaaa_ooov")(ja,ka,ia,ba) * t1_2p.at("aa")(aa,ka) )
    
    // r2_2p[aaaa] += -2.000 P(a,b) <a,k||i,j>_aaaa t1_2p_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_2p_aaaa(c,b,j,k) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0068_aaaa_voov")(ba,ja,ia,aa) )
    
    // r2_2p[aaaa] += -2.000 P(a,b) <a,k||i,j>_aaaa t1_2p_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_2p_aaaa(c,b,j,k) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0068_aaaa_voov")(ba,ia,ja,aa) )
    
    // r2_2p[aaaa] += -2.000 P(a,b) <a,k||i,j>_aaaa t1_2p_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_2p_aaaa(c,b,j,k) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0068_aaaa_voov")(aa,ja,ia,ba) )
    
    // r2_2p[aaaa] += -2.000 P(a,b) <a,k||i,j>_aaaa t1_2p_aa(b,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t2_2p_aaaa(c,b,j,k) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0068_aaaa_voov")(aa,ia,ja,ba) )
    .deallocate(tmps.at("0068_aaaa_voov"))
    .allocate(tmps.at("0069_bbbb_voov"))
    
    // flops: o2v2  = o3v3 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v2 o2v2
    ( tmps.at("0069_bbbb_voov")(ab,jb,ib,bb)  = tmps.at("0063_bbbb_oovv")(kb,jb,bb,cb) * t2_2p.at("bbbb")(cb,ab,ib,kb) )
    ( tmps.at("0069_bbbb_voov")(ab,jb,ib,bb) += tmps.at("0064_bbbb_ooov")(jb,kb,ib,bb) * t1_2p.at("bb")(ab,kb) )
    
    // r2_2p[bbbb] += -2.000 P(a,b) <a,k||i,j>_bbbb t1_2p_bb(b,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_2p_bbbb(c,b,j,k) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0069_bbbb_voov")(bb,jb,ib,ab) )
    
    // r2_2p[bbbb] += -2.000 P(a,b) <a,k||i,j>_bbbb t1_2p_bb(b,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_2p_bbbb(c,b,j,k) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0069_bbbb_voov")(bb,ib,jb,ab) )
    
    // r2_2p[bbbb] += -2.000 P(a,b) <a,k||i,j>_bbbb t1_2p_bb(b,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_2p_bbbb(c,b,j,k) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0069_bbbb_voov")(ab,jb,ib,bb) )
    
    // r2_2p[bbbb] += -2.000 P(a,b) <a,k||i,j>_bbbb t1_2p_bb(b,k) 
    //               += -2.000 P(i,j) P(a,b) <a,k||i,c>_bbbb t2_2p_bbbb(c,b,j,k) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0069_bbbb_voov")(ab,ib,jb,bb) )
    .deallocate(tmps.at("0069_bbbb_voov"))
    .allocate(tmps.at("0070_abab_vooo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v1
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1 o0v0Q1 o1v1 o3v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1_1p.at("aa")(ca,la) )
    ( tmps.at("bin1_bb_vo")(db,kb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0070_abab_vooo")(aa,kb,ia,jb)  = tmps.at("bin1_bb_vo")(db,kb) * t2.at("abab")(aa,db,ia,jb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t1_1p.at("bb")(cb,lb) )
    ( tmps.at("bin1_bb_vo")(db,kb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0070_abab_vooo")(aa,kb,ia,jb) += tmps.at("bin1_bb_vo")(db,kb) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_1p[abab] += -1.000 <l,k||c,d>_abab t1_bb(b,k) t1_1p_aa(c,l) t2_abab(a,d,i,j) 
    //               += +1.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_1p_bb(c,l) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0070_abab_vooo")(aa,kb,ia,jb) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_abab t1_1p_bb(b,l) t1_1p_aa(c,k) t2_abab(a,d,i,j) 
    //               += -2.000 <l,k||d,c>_bbbb t1_1p_bb(b,l) t1_1p_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0070_abab_vooo")(aa,lb,ia,jb) * t1_1p.at("bb")(bb,lb) )
    .deallocate(tmps.at("0070_abab_vooo"))
    .allocate(tmps.at("0071_abab_vooo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v1
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1 o0v0Q1 o1v1 o3v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("0071_abab_vooo")(aa,lb,ia,jb)  = tmps.at("bin1_bb_vo")(db,lb) * t2.at("abab")(aa,db,ia,jb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("0071_abab_vooo")(aa,lb,ia,jb) += tmps.at("bin1_bb_vo")(db,lb) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2[abab] += -1.000 <k,l||c,d>_abab t1_bb(b,l) t1_aa(c,k) t2_abab(a,d,i,j) 
    //            += +1.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0071_abab_vooo")(aa,lb,ia,jb) * t1.at("bb")(bb,lb) )
    
    // r2_1p[abab] += -1.000 <k,l||c,d>_abab t1_1p_bb(b,l) t1_aa(c,k) t2_abab(a,d,i,j) 
    //               += +1.000 <l,k||c,d>_bbbb t1_1p_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0071_abab_vooo")(aa,lb,ia,jb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_abab t1_2p_bb(b,l) t1_aa(c,k) t2_abab(a,d,i,j) 
    //               += +2.000 <l,k||c,d>_bbbb t1_2p_bb(b,l) t1_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0071_abab_vooo")(aa,lb,ia,jb) * t1_2p.at("bb")(bb,lb) )
    .deallocate(tmps.at("0071_abab_vooo"))
    .allocate(tmps.at("0072_abab_vooo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2 o1v1Q1 o1v1Q1 o3v2 o3v1
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1 o0v0Q1 o1v1 o3v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("0072_abab_vooo")(aa,lb,ia,jb)  = tmps.at("bin1_bb_vo")(db,lb) * t2_1p.at("abab")(aa,db,ia,jb) )
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("bin1_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("0072_abab_vooo")(aa,lb,ia,jb) += tmps.at("bin1_bb_vo")(db,lb) * t2_1p.at("abab")(aa,db,ia,jb) )
    
    // r2_1p[abab] += -1.000 <k,l||c,d>_abab t1_bb(b,l) t1_aa(c,k) t2_1p_abab(a,d,i,j) 
    //               += +1.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0072_abab_vooo")(aa,lb,ia,jb) * t1.at("bb")(bb,lb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_abab t1_1p_bb(b,l) t1_aa(c,k) t2_1p_abab(a,d,i,j) 
    //               += +2.000 <l,k||c,d>_bbbb t1_1p_bb(b,l) t1_bb(c,k) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0072_abab_vooo")(aa,lb,ia,jb) * t1_1p.at("bb")(bb,lb) )
    .deallocate(tmps.at("0072_abab_vooo"))
    .allocate(tmps.at("0073_aaaa_ovov"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0073_aaaa_ovov")(ia,aa,ja,ba)  = chol.at("aa_ovQ")(ia,aa,Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    
    // r1[aa] += +0.500 <k,j||c,b>_aaaa t1_aa(a,j) t2_aaaa(c,b,i,k) 
    // flops: o1v1 += o3v2 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_aa_oo")(ia,ja)  = t2.at("aaaa")(ca,ba,ia,ka) * tmps.at("0073_aaaa_ovov")(ka,ca,ja,ba) )
    ( r1.at("aa")(aa,ia) += 0.500 * t1.at("aa")(aa,ja) * tmps.at("bin1_aa_oo")(ia,ja) )
    
    // r1[aa] += +0.500 <k,j||c,b>_aaaa t1_aa(a,j) t2_aaaa(c,b,i,k) 
    // flops: o1v1 += o3v2 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_aa_oo")(ia,ja)  = t2.at("aaaa")(ca,ba,ia,ka) * tmps.at("0073_aaaa_ovov")(ka,ba,ja,ca) )
    ( r1.at("aa")(aa,ia) -= 0.500 * t1.at("aa")(aa,ja) * tmps.at("bin1_aa_oo")(ia,ja) )
    
    // r2[aaaa] += +1.000 <a,b||i,j>_aaaa 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0073_aaaa_ovov")(ia,aa,ja,ba) )
    
    // r2[aaaa] += +1.000 <a,b||i,j>_aaaa 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0073_aaaa_ovov")(ia,ba,ja,aa) )
    
    // r2_2p[aaaa] += -1.000 P(a,b) <k,l||d,c>_aaaa t2_1p_aaaa(d,a,i,j) t2_1p_aaaa(c,b,k,l) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_aa_vv")(ba,da)  = t2_1p.at("aaaa")(ca,ba,ka,la) * tmps.at("0073_aaaa_ovov")(ka,ca,la,da) )
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += t2_1p.at("aaaa")(da,aa,ia,ja) * tmps.at("bin1_aa_vv")(ba,da) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) <k,l||c,d>_aaaa t1_aa(c,i) t1_2p_aa(d,k) t2_aaaa(a,b,j,l) 
    // flops: o2v2 += o2v2 o2v1 o3v2
    //  mems: o2v2 += o1v1 o2v0 o2v2
    ( tmps.at("bin1_aa_vo")(ca,la)  = t1_2p.at("aa")(da,ka) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("bin1_aa_oo")(ja,la)  = tmps.at("bin1_aa_vo")(ca,la) * t1.at("aa")(ca,ja) )
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("bin1_aa_oo")(ja,la) * t2.at("aaaa")(aa,ba,ia,la) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,ka)  = t1_2p.at("aa")(ca,la) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin1_aa_vo")(da,ka) * t2.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_aaaa t2_1p_abab(a,b,l,j) t2_1p_aaaa(d,c,i,k) 
    //               += -1.000 <l,k||d,c>_aaaa t2_1p_abab(a,b,l,j) t2_1p_aaaa(d,c,i,k) 
    // flops: o2v2 += o3v2 o3v2 o2v0 o3v2
    //  mems: o2v2 += o2v0 o2v0 o2v0 o2v2
    ( tmps.at("bin1_aa_oo")(ia,la)  = -1.000 * t2_1p.at("aaaa")(da,ca,ia,ka) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("bin1_aa_oo")(ia,la) += t2_1p.at("aaaa")(da,ca,ia,ka) * tmps.at("0073_aaaa_ovov")(la,ca,ka,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += tmps.at("bin1_aa_oo")(ia,la) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_2p_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin1_aaaa_vooo")(da,ia,ka,la)  = t1_2p.at("aa")(ca,ia) * tmps.at("0073_aaaa_ovov")(ka,ca,la,da) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin1_aaaa_vooo")(da,ia,ka,la) * t2.at("abab")(da,bb,la,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_aaaa t1_1p_aa(c,k) t1_1p_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o2v2 o2v1 o3v2
    //  mems: o2v2 += o1v1 o2v0 o2v2
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("0073_aaaa_ovov")(la,ca,ka,da) * t1_1p.at("aa")(ca,ka) )
    ( tmps.at("bin1_aa_oo")(ia,la)  = t1_1p.at("aa")(da,ia) * tmps.at("bin1_aa_vo")(da,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_aaaa t1_1p_aa(a,l) t1_1p_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("0073_aaaa_ovov")(la,ca,ka,da) * t1_1p.at("aa")(ca,ka) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,la,jb)  = t2.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,la,jb) * t1_1p.at("aa")(aa,la) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_aaaa t1_1p_aa(a,k) t1_1p_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin1_aaaa_vooo")(da,ia,ka,la)  = tmps.at("0073_aaaa_ovov")(la,da,ka,ca) * t1_1p.at("aa")(ca,ia) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,la,jb) * tmps.at("bin1_aaaa_vooo")(da,ia,ka,la) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    .allocate(tmps.at("0074_aaaa_oooo"))
    
    // flops: o4v0  = o4v2 o4v2 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0
    ( tmps.at("0074_aaaa_oooo")(ka,ia,ja,la)  = -1.000 * t2_2p.at("aaaa")(da,ca,ia,ja) * tmps.at("0073_aaaa_ovov")(ka,da,la,ca) )
    ( tmps.at("0074_aaaa_oooo")(ka,ia,ja,la) += t2_2p.at("aaaa")(da,ca,ia,ja) * tmps.at("0073_aaaa_ovov")(ka,ca,la,da) )
    
    // r2_2p[aaaa] += -1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_aa(b,l) t2_2p_aaaa(d,c,i,j) 
    //               += -1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_aa(b,l) t2_2p_aaaa(d,c,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la)  = tmps.at("0074_aaaa_oooo")(la,ia,ja,ka) * t1.at("aa")(aa,ka) )
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += t1.at("aa")(ba,la) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,la) )
    
    // r2_2p[aaaa] += +0.500 <k,l||d,c>_aaaa t2_aaaa(a,b,k,l) t2_2p_aaaa(d,c,i,j) 
    //               += +0.500 <k,l||d,c>_aaaa t2_aaaa(a,b,k,l) t2_2p_aaaa(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 0.500 * tmps.at("0074_aaaa_oooo")(ka,ia,ja,la) * t2.at("aaaa")(aa,ba,ka,la) )
    .deallocate(tmps.at("0074_aaaa_oooo"))
    .allocate(tmps.at("0075_bbbb_ovov"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0075_bbbb_ovov")(ib,ab,jb,bb)  = chol.at("bb_ovQ")(ib,ab,Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    
    // r1[bb] += +0.500 <k,j||c,b>_bbbb t1_bb(a,j) t2_bbbb(c,b,i,k) 
    // flops: o1v1 += o3v2 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_bb_oo")(ib,jb)  = t2.at("bbbb")(cb,bb,ib,kb) * tmps.at("0075_bbbb_ovov")(kb,cb,jb,bb) )
    ( r1.at("bb")(ab,ib) += 0.500 * t1.at("bb")(ab,jb) * tmps.at("bin1_bb_oo")(ib,jb) )
    
    // r1[bb] += +0.500 <k,j||c,b>_bbbb t1_bb(a,j) t2_bbbb(c,b,i,k) 
    // flops: o1v1 += o3v2 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_bb_oo")(ib,jb)  = t2.at("bbbb")(cb,bb,ib,kb) * tmps.at("0075_bbbb_ovov")(kb,bb,jb,cb) )
    ( r1.at("bb")(ab,ib) -= 0.500 * t1.at("bb")(ab,jb) * tmps.at("bin1_bb_oo")(ib,jb) )
    
    // r2[bbbb] += +1.000 <a,b||i,j>_bbbb 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0075_bbbb_ovov")(ib,ab,jb,bb) )
    
    // r2[bbbb] += +1.000 <a,b||i,j>_bbbb 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0075_bbbb_ovov")(ib,bb,jb,ab) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_bbbb t2_1p_abab(a,b,i,l) t2_1p_bbbb(d,c,j,k) 
    //               += -1.000 <l,k||d,c>_bbbb t2_1p_abab(a,b,i,l) t2_1p_bbbb(d,c,j,k) 
    // flops: o2v2 += o3v2 o3v2 o2v0 o3v2
    //  mems: o2v2 += o2v0 o2v0 o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = -1.000 * t2_1p.at("bbbb")(db,cb,jb,kb) * tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) )
    ( tmps.at("bin1_bb_oo")(jb,lb) += t2_1p.at("bbbb")(db,cb,jb,kb) * tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += tmps.at("bin1_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[abab] += +1.000 <k,l||c,d>_bbbb t2_abab(a,c,i,j) t2_2p_bbbb(d,b,k,l) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,cb)  = t2_2p.at("bbbb")(db,bb,kb,lb) * tmps.at("0075_bbbb_ovov")(kb,db,lb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,cb,ia,jb) * tmps.at("bin1_bb_vv")(bb,cb) )
    
    // r2_2p[abab] += +1.000 <k,l||c,d>_bbbb t2_abab(a,c,i,j) t2_2p_bbbb(d,b,k,l) 
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,cb)  = tmps.at("0075_bbbb_ovov")(kb,cb,lb,db) * t2_2p.at("bbbb")(db,bb,kb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += tmps.at("bin1_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t2_1p_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_abab_vvoo")(aa,db,ia,lb)  = t2_1p.at("abab")(aa,cb,ia,kb) * tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_abab_vvoo")(aa,db,ia,lb) * t2_1p.at("bbbb")(db,bb,jb,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("bin1_bb_vo")(db,lb) * t2_2p.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,lb) * t1.at("bb")(bb,lb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_bbbb t1_1p_bb(b,l) t1_1p_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t1_1p.at("bb")(cb,kb) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("bin1_bb_vo")(db,lb) * t2.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,lb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_2p_bb(c,l) t2_abab(a,d,i,j) 
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,kb)  = t1_2p.at("bb")(cb,lb) * tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin1_bb_vo")(db,kb) * t2.at("abab")(aa,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0076_bbbb_oooo"))
    
    // flops: o4v0  = o4v2 o4v2 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0
    ( tmps.at("0076_bbbb_oooo")(kb,ib,jb,lb)  = -1.000 * t2_2p.at("bbbb")(db,cb,ib,jb) * tmps.at("0075_bbbb_ovov")(kb,db,lb,cb) )
    ( tmps.at("0076_bbbb_oooo")(kb,ib,jb,lb) += t2_2p.at("bbbb")(db,cb,ib,jb) * tmps.at("0075_bbbb_ovov")(kb,cb,lb,db) )
    
    // r2_2p[bbbb] += -1.000 <l,k||d,c>_bbbb t1_bb(a,k) t1_bb(b,l) t2_2p_bbbb(d,c,i,j) 
    //               += -1.000 <l,k||d,c>_bbbb t1_bb(a,k) t1_bb(b,l) t2_2p_bbbb(d,c,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb)  = tmps.at("0076_bbbb_oooo")(lb,ib,jb,kb) * t1.at("bb")(ab,kb) )
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += t1.at("bb")(bb,lb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,lb) )
    
    // r2_2p[bbbb] += +0.500 <k,l||d,c>_bbbb t2_bbbb(a,b,k,l) t2_2p_bbbb(d,c,i,j) 
    //               += +0.500 <k,l||d,c>_bbbb t2_bbbb(a,b,k,l) t2_2p_bbbb(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 0.500 * tmps.at("0076_bbbb_oooo")(kb,ib,jb,lb) * t2.at("bbbb")(ab,bb,kb,lb) )
    .deallocate(tmps.at("0076_bbbb_oooo"))
    .allocate(tmps.at("0077_aa_vo"))
    
    // flops: o1v1  = o2v2 o1v2 o1v1 o2v1 o2v1 o1v1 o2v1 o1v1 o2v1 o1v1 o2v1 o1v1 o2v2 o1v1
    //  mems: o1v1  = o1v1 o1v1 o1v1 o2v0 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1
    ( tmps.at("0077_aa_vo")(aa,ia)  = -1.000 * dp.at("bb_ov")(jb,bb) * t2_2p.at("abab")(aa,bb,ia,jb) )
    ( tmps.at("0077_aa_vo")(aa,ia) -= dp.at("aa_vv")(aa,ba) * t1_2p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aa_oo")(ia,ja)  = dp.at("aa_ov")(ja,ba) * t1.at("aa")(ba,ia) )
    ( tmps.at("0077_aa_vo")(aa,ia) += tmps.at("bin1_aa_oo")(ia,ja) * t1_2p.at("aa")(aa,ja) )
    ( tmps.at("0077_aa_vo")(aa,ia) += t1_1p.at("aa")(aa,ja) * tmps.at("0031_aa_oo")(ja,ia) )
    ( tmps.at("0077_aa_vo")(aa,ia) += t1.at("aa")(aa,ja) * tmps.at("0033_aa_oo")(ja,ia) )
    ( tmps.at("0077_aa_vo")(aa,ia) += dp.at("aa_oo")(ja,ia) * t1_2p.at("aa")(aa,ja) )
    ( tmps.at("0077_aa_vo")(aa,ia) += dp.at("aa_ov")(ja,ba) * t2_2p.at("aaaa")(ba,aa,ia,ja) )
    
    // r1_1p[aa] += -2.000 d-_aa(j,i) t1_2p_aa(a,j) 
    //             += -2.000 d-_aa(j,b) t2_2p_aaaa(b,a,i,j) 
    //             += +2.000 d-_aa(a,b) t1_2p_aa(b,i) 
    //             += +2.000 d-_bb(j,b) t2_2p_abab(a,b,i,j) 
    //             += -2.000 d-_aa(j,b) t1_2p_aa(a,j) t1_aa(b,i) 
    //             += -2.000 d-_aa(j,b) t1_1p_aa(a,j) t1_1p_aa(b,i) 
    //             += -2.000 d-_aa(j,b) t1_aa(a,j) t1_2p_aa(b,i) 
    ( r1_1p.at("aa")(aa,ia) -= 2.000 * tmps.at("0077_aa_vo")(aa,ia) )
    
    // r1_2p[aa] += -2.000 d-_aa(j,i) t0_1p t1_2p_aa(a,j) 
    //             += -2.000 d-_aa(j,b) t0_1p t2_2p_aaaa(b,a,i,j) 
    //             += +2.000 d-_aa(a,b) t0_1p t1_2p_aa(b,i) 
    //             += +2.000 d-_bb(j,b) t0_1p t2_2p_abab(a,b,i,j) 
    //             += -2.000 d-_aa(j,b) t0_1p t1_2p_aa(a,j) t1_aa(b,i) 
    //             += -2.000 d-_aa(j,b) t0_1p t1_1p_aa(a,j) t1_1p_aa(b,i) 
    //             += -2.000 d-_aa(j,b) t0_1p t1_aa(a,j) t1_2p_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * t0_1p * tmps.at("0077_aa_vo")(aa,ia) )
    
    // r2_2p[abab] += -2.000 d-_aa(k,i) t1_2p_aa(a,k) t1_1p_bb(b,j) 
    //               += -2.000 d-_aa(k,c) t1_1p_bb(b,j) t2_2p_aaaa(c,a,i,k) 
    //               += +2.000 d-_aa(a,c) t1_1p_bb(b,j) t1_2p_aa(c,i) 
    //               += +2.000 d-_bb(k,c) t1_1p_bb(b,j) t2_2p_abab(a,c,i,k) 
    //               += -2.000 d-_aa(k,c) t1_2p_aa(a,k) t1_1p_bb(b,j) t1_aa(c,i) 
    //               += -2.000 d-_aa(k,c) t1_1p_aa(a,k) t1_1p_bb(b,j) t1_1p_aa(c,i) 
    //               += -2.000 d-_aa(k,c) t1_aa(a,k) t1_1p_bb(b,j) t1_2p_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_1p.at("bb")(bb,jb) * tmps.at("0077_aa_vo")(aa,ia) )
    .deallocate(tmps.at("0077_aa_vo"))
    .allocate(tmps.at("0078_bb_vo"))
    
    // flops: o1v1  = o2v1 o2v1 o2v1 o1v1 o2v2 o1v1 o2v1 o1v1 o2v1 o1v1 o2v2 o1v1 o1v2 o1v1
    //  mems: o1v1  = o2v0 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1
    ( tmps.at("bin1_bb_oo")(ib,jb)  = dp.at("bb_ov")(jb,bb) * t1.at("bb")(bb,ib) )
    ( tmps.at("0078_bb_vo")(ab,ib)  = -1.000 * tmps.at("bin1_bb_oo")(ib,jb) * t1_2p.at("bb")(ab,jb) )
    ( tmps.at("0078_bb_vo")(ab,ib) -= t1_2p.at("bb")(ab,jb) * dp.at("bb_oo")(jb,ib) )
    ( tmps.at("0078_bb_vo")(ab,ib) -= t2_2p.at("bbbb")(bb,ab,ib,jb) * dp.at("bb_ov")(jb,bb) )
    ( tmps.at("0078_bb_vo")(ab,ib) -= tmps.at("0039_bb_oo")(jb,ib) * t1_1p.at("bb")(ab,jb) )
    ( tmps.at("0078_bb_vo")(ab,ib) -= t1.at("bb")(ab,jb) * tmps.at("0041_bb_oo")(jb,ib) )
    ( tmps.at("0078_bb_vo")(ab,ib) += dp.at("aa_ov")(ja,ba) * t2_2p.at("abab")(ba,ab,ja,ib) )
    ( tmps.at("0078_bb_vo")(ab,ib) += dp.at("bb_vv")(ab,bb) * t1_2p.at("bb")(bb,ib) )
    
    // r1_1p[bb] += +2.000 d-_aa(j,b) t2_2p_abab(b,a,j,i) 
    //             += -2.000 d-_bb(j,i) t1_2p_bb(a,j) 
    //             += -2.000 d-_bb(j,b) t2_2p_bbbb(b,a,i,j) 
    //             += +2.000 d-_bb(a,b) t1_2p_bb(b,i) 
    //             += -2.000 d-_bb(j,b) t1_2p_bb(a,j) t1_bb(b,i) 
    //             += -2.000 d-_bb(j,b) t1_1p_bb(a,j) t1_1p_bb(b,i) 
    //             += -2.000 d-_bb(j,b) t1_bb(a,j) t1_2p_bb(b,i) 
    ( r1_1p.at("bb")(ab,ib) += 2.000 * tmps.at("0078_bb_vo")(ab,ib) )
    
    // r1_2p[bb] += +2.000 d-_aa(j,b) t0_1p t2_2p_abab(b,a,j,i) 
    //             += -2.000 d-_bb(j,i) t0_1p t1_2p_bb(a,j) 
    //             += -2.000 d-_bb(j,b) t0_1p t2_2p_bbbb(b,a,i,j) 
    //             += +2.000 d-_bb(a,b) t0_1p t1_2p_bb(b,i) 
    //             += -2.000 d-_bb(j,b) t0_1p t1_2p_bb(a,j) t1_bb(b,i) 
    //             += -2.000 d-_bb(j,b) t0_1p t1_1p_bb(a,j) t1_1p_bb(b,i) 
    //             += -2.000 d-_bb(j,b) t0_1p t1_bb(a,j) t1_2p_bb(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * t0_1p * tmps.at("0078_bb_vo")(ab,ib) )
    
    // r2_2p[abab] += +2.000 d-_aa(k,c) t1_1p_aa(a,i) t2_2p_abab(c,b,k,j) 
    //               += -2.000 d-_bb(k,j) t1_1p_aa(a,i) t1_2p_bb(b,k) 
    //               += -2.000 d-_bb(k,c) t1_1p_aa(a,i) t2_2p_bbbb(c,b,j,k) 
    //               += +2.000 d-_bb(b,c) t1_1p_aa(a,i) t1_2p_bb(c,j) 
    //               += -2.000 d-_bb(k,c) t1_1p_aa(a,i) t1_2p_bb(b,k) t1_bb(c,j) 
    //               += -2.000 d-_bb(k,c) t1_1p_aa(a,i) t1_1p_bb(b,k) t1_1p_bb(c,j) 
    //               += -2.000 d-_bb(k,c) t1_1p_aa(a,i) t1_bb(b,k) t1_2p_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_1p.at("aa")(aa,ia) * tmps.at("0078_bb_vo")(bb,jb) )
    .deallocate(tmps.at("0078_bb_vo"))
    .allocate(tmps.at("0079_aabb_vooo"))
    
    // flops: o3v1  = o3v2 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb)  = t1.at("bb")(cb,jb) * tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) )
    ( tmps.at("0079_aabb_vooo")(aa,ia,jb,kb)  = t2.at("abab")(aa,db,ia,lb) * tmps.at("bin1_bbbb_vooo")(db,jb,kb,lb) )
    
    // r2[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0079_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_1p[abab] += -1.000 <k,l||c,d>_bbbb t1_1p_bb(b,k) t1_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0079_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_bbbb t1_2p_bb(b,k) t1_bb(c,j) t2_abab(a,d,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0079_aabb_vooo")(aa,ia,jb,kb) * t1_2p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0079_aabb_vooo"))
    .allocate(tmps.at("0080_aa_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0080_aa_vv")(da,aa)  = tmps.at("0073_aaaa_ovov")(la,da,ka,ca) * t2.at("aaaa")(ca,aa,ka,la) )
    
    // r1[aa] += +0.500 <j,k||b,c>_aaaa t1_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= 0.500 * t1.at("aa")(ba,ia) * tmps.at("0080_aa_vv")(ba,aa) )
    
    // r1_1p[aa] += -0.500 <j,k||c,b>_aaaa t1_1p_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.500 * t1_1p.at("aa")(ba,ia) * tmps.at("0080_aa_vv")(ba,aa) )
    
    // r1_2p[aa] += -1.000 <j,k||c,b>_aaaa t1_2p_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= t1_2p.at("aa")(ba,ia) * tmps.at("0080_aa_vv")(ba,aa) )
    
    // r2[aaaa] += -0.500 P(a,b) <k,l||d,c>_aaaa t2_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("aaaa")(aa,ba,ia,ja) -= 0.500 * tmps.at("0080_aa_vv")(da,aa) * t2.at("aaaa")(da,ba,ia,ja) )
    
    // r2[abab] += +0.500 <k,l||d,c>_aaaa t2_aaaa(c,a,k,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= 0.500 * tmps.at("0080_aa_vv")(da,aa) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2_1p[aaaa] += +0.500 P(a,b) <k,l||c,d>_aaaa t2_1p_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= 0.500 * tmps.at("0080_aa_vv")(da,aa) * t2_1p.at("aaaa")(da,ba,ia,ja) )
    
    // r2_1p[abab] += -0.500 <k,l||c,d>_aaaa t2_aaaa(c,a,k,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.500 * tmps.at("0080_aa_vv")(da,aa) * t2_1p.at("abab")(da,bb,ia,jb) )
    
    // r2_2p[aaaa] += +1.000 P(a,b) <k,l||c,d>_aaaa t2_2p_aaaa(d,a,i,j) t2_aaaa(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0080_aa_vv")(da,aa) * t2_2p.at("aaaa")(da,ba,ia,ja) )
    
    // r2_2p[abab] += -1.000 <k,l||c,d>_aaaa t2_aaaa(c,a,k,l) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("0080_aa_vv")(da,aa) * t2_2p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0080_aa_vv"))
    .allocate(tmps.at("0081_bb_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0081_bb_vv")(bb,db)  = t2.at("bbbb")(cb,bb,kb,lb) * tmps.at("0075_bbbb_ovov")(kb,cb,lb,db) )
    
    // r2[abab] += +0.500 <k,l||d,c>_bbbb t2_abab(a,d,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= 0.500 * tmps.at("0081_bb_vv")(bb,db) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2[bbbb] += -0.500 P(a,b) <k,l||d,c>_bbbb t2_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("bbbb")(ab,bb,ib,jb) += 0.500 * tmps.at("0081_bb_vv")(bb,db) * t2.at("bbbb")(db,ab,ib,jb) )
    
    // r2_1p[abab] += -0.500 <k,l||c,d>_bbbb t2_1p_abab(a,d,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 0.500 * tmps.at("0081_bb_vv")(bb,db) * t2_1p.at("abab")(aa,db,ia,jb) )
    
    // r2_1p[bbbb] += +0.500 P(a,b) <k,l||c,d>_bbbb t2_1p_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += 0.500 * tmps.at("0081_bb_vv")(bb,db) * t2_1p.at("bbbb")(db,ab,ib,jb) )
    
    // r2_2p[abab] += -1.000 <k,l||c,d>_bbbb t2_2p_abab(a,d,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= tmps.at("0081_bb_vv")(bb,db) * t2_2p.at("abab")(aa,db,ia,jb) )
    
    // r2_2p[bbbb] += +1.000 P(a,b) <k,l||c,d>_bbbb t2_2p_bbbb(d,a,i,j) t2_bbbb(c,b,k,l) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0081_bb_vv")(bb,db) * t2_2p.at("bbbb")(db,ab,ib,jb) )
    .deallocate(tmps.at("0081_bb_vv"))
    .allocate(tmps.at("0082_baab_vooo"))
    
    // flops: o3v1  = o3v2 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aaaa_vooo")(da,ia,ka,la)  = t1.at("aa")(ca,ia) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("0082_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin1_aaaa_vooo")(da,ia,ka,la) * t2.at("abab")(da,bb,la,jb) )
    
    // r2_1p[abab] += -1.000 <k,l||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("aa")(aa,ka) * tmps.at("0082_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_aaaa t1_2p_aa(a,k) t1_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_2p.at("aa")(aa,ka) * tmps.at("0082_baab_vooo")(bb,ia,ka,jb) )
    .deallocate(tmps.at("0082_baab_vooo"))
    .allocate(tmps.at("0083_aabb_ovov"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0083_aabb_ovov")(ia,aa,jb,bb)  = chol.at("aa_ovQ")(ia,aa,Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    
    // r2[abab] += +1.000 <a,b||i,j>_abab 
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0083_aabb_ovov")(ia,aa,jb,bb) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_abab t1_aa(a,k) t1_2p_bb(c,j) t2_abab(d,b,i,l) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(da,ka,jb,lb)  = tmps.at("0083_aabb_ovov")(ka,da,lb,cb) * t1_2p.at("bb")(cb,jb) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,ia,lb) * tmps.at("bin1_aabb_vooo")(da,ka,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_abab t1_aa(a,k) t1_bb(c,j) t2_2p_abab(d,b,i,l) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(da,ka,jb,lb)  = t1.at("bb")(cb,jb) * tmps.at("0083_aabb_ovov")(ka,da,lb,cb) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(da,bb,ia,lb) * tmps.at("bin1_aabb_vooo")(da,ka,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(c,j) t2_1p_abab(d,b,i,l) 
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(da,ka,jb,lb)  = t1_1p.at("bb")(cb,jb) * tmps.at("0083_aabb_ovov")(ka,da,lb,cb) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_1p.at("abab")(da,bb,ia,lb) * tmps.at("bin1_aabb_vooo")(da,ka,jb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0084_abab_oooo"))
    
    // flops: o4v0  = o4v2
    //  mems: o4v0  = o4v0
    ( tmps.at("0084_abab_oooo")(ka,lb,ia,jb)  = tmps.at("0083_aabb_ovov")(ka,da,lb,cb) * t2_2p.at("abab")(da,cb,ia,jb) )
    
    // r2_2p[abab] += +0.500 <k,l||d,c>_abab t2_abab(a,b,k,l) t2_2p_abab(d,c,i,j) 
    //               += +0.500 <k,l||c,d>_abab t2_abab(a,b,k,l) t2_2p_abab(c,d,i,j) 
    //               += +0.500 <l,k||d,c>_abab t2_abab(a,b,l,k) t2_2p_abab(d,c,i,j) 
    //               += +0.500 <l,k||c,d>_abab t2_abab(a,b,l,k) t2_2p_abab(c,d,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,bb,ka,lb) * tmps.at("0084_abab_oooo")(ka,lb,ia,jb) )
    
    // r2_2p[abab] += +1.000 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t2_2p_abab(d,c,i,j) 
    //               += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t2_2p_abab(c,d,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0084_abab_oooo")(ka,lb,ia,jb) * t1.at("bb")(bb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0084_abab_oooo"))
    .allocate(tmps.at("0085_aabb_vvoo"))
    
    // flops: o2v2  = o3v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0085_aabb_vvoo")(ca,aa,kb,jb)  = tmps.at("0083_aabb_ovov")(la,ca,kb,db) * t2_2p.at("abab")(aa,db,la,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_abab t2_2p_abab(a,d,l,j) t2_abab(c,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(ca,bb,ia,kb) * tmps.at("0085_aabb_vvoo")(ca,aa,kb,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t2_2p_abab(a,d,l,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t1.at("aa")(ca,ia) * tmps.at("0085_aabb_vvoo")(ca,aa,kb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .deallocate(tmps.at("0085_aabb_vvoo"))
    .allocate(tmps.at("0086_bbbb_ovvv"))
    
    // flops: o1v3  = o1v3Q1
    //  mems: o1v3  = o1v3
    ( tmps.at("0086_bbbb_ovvv")(kb,cb,bb,db)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("bb_vvQ")(bb,db,Q) )
    
    // r1[bb] += +0.500 <a,j||c,b>_bbbb t2_bbbb(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += 0.500 * t2.at("bbbb")(cb,bb,ib,jb) * tmps.at("0086_bbbb_ovvv")(jb,bb,ab,cb) )
    
    // r1[bb] += +0.500 <a,j||c,b>_bbbb t2_bbbb(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= 0.500 * t2.at("bbbb")(cb,bb,ib,jb) * tmps.at("0086_bbbb_ovvv")(jb,cb,ab,bb) )
    
    // r1_1p[bb] += +0.500 <a,j||c,b>_bbbb t2_1p_bbbb(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += 0.500 * t2_1p.at("bbbb")(cb,bb,ib,jb) * tmps.at("0086_bbbb_ovvv")(jb,bb,ab,cb) )
    
    // r1_1p[bb] += +0.500 <a,j||c,b>_bbbb t2_1p_bbbb(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= 0.500 * t2_1p.at("bbbb")(cb,bb,ib,jb) * tmps.at("0086_bbbb_ovvv")(jb,cb,ab,bb) )
    
    // r1_2p[bb] += +1.000 <a,j||c,b>_bbbb t2_2p_bbbb(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += t2_2p.at("bbbb")(cb,bb,ib,jb) * tmps.at("0086_bbbb_ovvv")(jb,bb,ab,cb) )
    
    // r1_2p[bb] += +1.000 <a,j||c,b>_bbbb t2_2p_bbbb(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= t2_2p.at("bbbb")(cb,bb,ib,jb) * tmps.at("0086_bbbb_ovvv")(jb,cb,ab,bb) )
    
    // r2_2p[abab] += +2.000 <b,k||d,c>_bbbb t1_2p_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o1v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_bb_vv")(bb,db)  = t1_2p.at("bb")(cb,kb) * tmps.at("0086_bbbb_ovvv")(kb,db,bb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_vv")(bb,db) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_2p[abab] += -2.000 <b,k||d,c>_bbbb t1_2p_bb(c,j) t2_abab(a,d,i,k) 
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_bbbb_vvoo")(bb,db,jb,kb)  = tmps.at("0086_bbbb_ovvv")(kb,cb,bb,db) * t1_2p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bbbb_vvoo")(bb,db,jb,kb) * t2.at("abab")(aa,db,ia,kb) )
    .allocate(tmps.at("0087_bbbb_ovvo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0087_bbbb_ovvo")(ib,ab,bb,jb)  = tmps.at("0086_bbbb_ovvv")(ib,ab,cb,bb) * t1.at("bb")(cb,jb) )
    
    // r2[bbbb] += +1.000 P(i,j) <a,b||i,c>_bbbb t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0087_bbbb_ovvo")(ib,bb,ab,jb) )
    
    // r2[bbbb] += +1.000 P(i,j) <a,b||i,c>_bbbb t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0087_bbbb_ovvo")(ib,ab,bb,jb) )
    
    // r2[bbbb] += +1.000 P(i,j) <a,b||i,c>_bbbb t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0087_bbbb_ovvo")(jb,bb,ab,ib) )
    
    // r2[bbbb] += +1.000 P(i,j) <a,b||i,c>_bbbb t1_bb(c,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0087_bbbb_ovvo")(jb,ab,bb,ib) )
    .deallocate(tmps.at("0087_bbbb_ovvo"))
    .allocate(tmps.at("0088_bbbb_ovvo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0088_bbbb_ovvo")(ib,ab,bb,jb)  = tmps.at("0086_bbbb_ovvv")(ib,ab,cb,bb) * t1_1p.at("bb")(cb,jb) )
    
    // r2_1p[bbbb] += +1.000 P(i,j) <a,b||i,c>_bbbb t1_1p_bb(c,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0088_bbbb_ovvo")(ib,bb,ab,jb) )
    
    // r2_1p[bbbb] += +1.000 P(i,j) <a,b||i,c>_bbbb t1_1p_bb(c,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0088_bbbb_ovvo")(ib,ab,bb,jb) )
    
    // r2_1p[bbbb] += +1.000 P(i,j) <a,b||i,c>_bbbb t1_1p_bb(c,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0088_bbbb_ovvo")(jb,bb,ab,ib) )
    
    // r2_1p[bbbb] += +1.000 P(i,j) <a,b||i,c>_bbbb t1_1p_bb(c,j) 
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0088_bbbb_ovvo")(jb,ab,bb,ib) )
    .deallocate(tmps.at("0088_bbbb_ovvo"))
    .allocate(tmps.at("0089_bbbb_ovvo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0089_bbbb_ovvo")(ib,ab,bb,jb)  = tmps.at("0086_bbbb_ovvv")(ib,ab,cb,bb) * t1_2p.at("bb")(cb,jb) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) <a,b||i,c>_bbbb t1_2p_bb(c,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0089_bbbb_ovvo")(ib,bb,ab,jb) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) <a,b||i,c>_bbbb t1_2p_bb(c,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0089_bbbb_ovvo")(ib,ab,bb,jb) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) <a,b||i,c>_bbbb t1_2p_bb(c,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0089_bbbb_ovvo")(jb,bb,ab,ib) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) <a,b||i,c>_bbbb t1_2p_bb(c,j) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0089_bbbb_ovvo")(jb,ab,bb,ib) )
    .deallocate(tmps.at("0089_bbbb_ovvo"))
    .allocate(tmps.at("0090_aaaa_ovvv"))
    
    // flops: o1v3  = o1v3Q1
    //  mems: o1v3  = o1v3
    ( tmps.at("0090_aaaa_ovvv")(ka,ca,ba,da)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("aa_vvQ")(ba,da,Q) )
    
    // r1[aa] += +0.500 <a,j||c,b>_aaaa t2_aaaa(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += 0.500 * t2.at("aaaa")(ca,ba,ia,ja) * tmps.at("0090_aaaa_ovvv")(ja,ba,aa,ca) )
    
    // r1[aa] += +0.500 <a,j||c,b>_aaaa t2_aaaa(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= 0.500 * t2.at("aaaa")(ca,ba,ia,ja) * tmps.at("0090_aaaa_ovvv")(ja,ca,aa,ba) )
    
    // r1_1p[aa] += +0.500 <a,j||c,b>_aaaa t2_1p_aaaa(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 0.500 * t2_1p.at("aaaa")(ca,ba,ia,ja) * tmps.at("0090_aaaa_ovvv")(ja,ba,aa,ca) )
    
    // r1_1p[aa] += +0.500 <a,j||c,b>_aaaa t2_1p_aaaa(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 0.500 * t2_1p.at("aaaa")(ca,ba,ia,ja) * tmps.at("0090_aaaa_ovvv")(ja,ca,aa,ba) )
    
    // r1_2p[aa] += +1.000 <a,j||c,b>_aaaa t2_2p_aaaa(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += t2_2p.at("aaaa")(ca,ba,ia,ja) * tmps.at("0090_aaaa_ovvv")(ja,ba,aa,ca) )
    
    // r1_2p[aa] += +1.000 <a,j||c,b>_aaaa t2_2p_aaaa(c,b,i,j) 
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= t2_2p.at("aaaa")(ca,ba,ia,ja) * tmps.at("0090_aaaa_ovvv")(ja,ca,aa,ba) )
    
    // r2_2p[abab] += +2.000 <a,k||d,c>_aaaa t1_2p_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o1v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    ( tmps.at("bin1_aa_vv")(aa,da)  = t1_2p.at("aa")(ca,ka) * tmps.at("0090_aaaa_ovvv")(ka,da,aa,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aa_vv")(aa,da) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 <a,k||d,c>_aaaa t1_2p_aa(c,i) t2_abab(d,b,k,j) 
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin1_aaaa_vvoo")(aa,da,ia,ka)  = tmps.at("0090_aaaa_ovvv")(ka,ca,aa,da) * t1_2p.at("aa")(ca,ia) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aaaa_vvoo")(aa,da,ia,ka) * t2.at("abab")(da,bb,ka,jb) )
    .allocate(tmps.at("0091_aaaa_ovvo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0091_aaaa_ovvo")(ia,aa,ba,ja)  = tmps.at("0090_aaaa_ovvv")(ia,aa,ca,ba) * t1.at("aa")(ca,ja) )
    
    // r2[aaaa] += +1.000 P(i,j) <a,b||i,c>_aaaa t1_aa(c,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0091_aaaa_ovvo")(ia,ba,aa,ja) )
    
    // r2[aaaa] += +1.000 P(i,j) <a,b||i,c>_aaaa t1_aa(c,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0091_aaaa_ovvo")(ia,aa,ba,ja) )
    
    // r2[aaaa] += +1.000 P(i,j) <a,b||i,c>_aaaa t1_aa(c,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0091_aaaa_ovvo")(ja,ba,aa,ia) )
    
    // r2[aaaa] += +1.000 P(i,j) <a,b||i,c>_aaaa t1_aa(c,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0091_aaaa_ovvo")(ja,aa,ba,ia) )
    .deallocate(tmps.at("0091_aaaa_ovvo"))
    .allocate(tmps.at("0092_aaaa_ovvo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0092_aaaa_ovvo")(ia,aa,ba,ja)  = tmps.at("0090_aaaa_ovvv")(ia,aa,ca,ba) * t1_1p.at("aa")(ca,ja) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) <a,b||i,c>_aaaa t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0092_aaaa_ovvo")(ia,ba,aa,ja) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) <a,b||i,c>_aaaa t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0092_aaaa_ovvo")(ia,aa,ba,ja) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) <a,b||i,c>_aaaa t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0092_aaaa_ovvo")(ja,ba,aa,ia) )
    
    // r2_1p[aaaa] += +1.000 P(i,j) <a,b||i,c>_aaaa t1_1p_aa(c,j) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0092_aaaa_ovvo")(ja,aa,ba,ia) )
    .deallocate(tmps.at("0092_aaaa_ovvo"))
    .allocate(tmps.at("0093_aaaa_ovvo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0093_aaaa_ovvo")(ia,aa,ba,ja)  = tmps.at("0090_aaaa_ovvv")(ia,aa,ca,ba) * t1_2p.at("aa")(ca,ja) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) <a,b||i,c>_aaaa t1_2p_aa(c,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0093_aaaa_ovvo")(ia,ba,aa,ja) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) <a,b||i,c>_aaaa t1_2p_aa(c,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0093_aaaa_ovvo")(ia,aa,ba,ja) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) <a,b||i,c>_aaaa t1_2p_aa(c,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0093_aaaa_ovvo")(ja,ba,aa,ia) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) <a,b||i,c>_aaaa t1_2p_aa(c,j) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0093_aaaa_ovvo")(ja,aa,ba,ia) )
    .deallocate(tmps.at("0093_aaaa_ovvo"))
    .allocate(tmps.at("0094_aa_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0094_aa_vv")(aa,da)  = t2.at("aaaa")(ca,aa,ka,la) * tmps.at("0073_aaaa_ovov")(la,ca,ka,da) )
    
    // r1[aa] += +0.500 <j,k||b,c>_aaaa t1_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += 0.500 * tmps.at("0094_aa_vv")(aa,ba) * t1.at("aa")(ba,ia) )
    
    // r1_1p[aa] += -0.500 <j,k||c,b>_aaaa t1_1p_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 0.500 * tmps.at("0094_aa_vv")(aa,ba) * t1_1p.at("aa")(ba,ia) )
    
    // r1_2p[aa] += -1.000 <j,k||c,b>_aaaa t1_2p_aa(b,i) t2_aaaa(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += tmps.at("0094_aa_vv")(aa,ba) * t1_2p.at("aa")(ba,ia) )
    
    // r2[abab] += +0.500 <k,l||d,c>_aaaa t2_aaaa(c,a,k,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += 0.500 * t2.at("abab")(da,bb,ia,jb) * tmps.at("0094_aa_vv")(aa,da) )
    
    // r2_1p[abab] += -0.500 <k,l||c,d>_aaaa t2_aaaa(c,a,k,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.500 * t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("0094_aa_vv")(aa,da) )
    
    // r2_2p[abab] += -1.000 <k,l||c,d>_aaaa t2_aaaa(c,a,k,l) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += t2_2p.at("abab")(da,bb,ia,jb) * tmps.at("0094_aa_vv")(aa,da) )
    .deallocate(tmps.at("0094_aa_vv"))
    .allocate(tmps.at("0095_bb_vv"))
    
    // flops: o0v2  = o2v3
    //  mems: o0v2  = o0v2
    ( tmps.at("0095_bb_vv")(db,ab)  = tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) * t2.at("bbbb")(cb,ab,kb,lb) )
    
    // r1[bb] += +0.500 <j,k||b,c>_bbbb t1_bb(b,i) t2_bbbb(c,a,j,k) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= 0.500 * t1.at("bb")(bb,ib) * tmps.at("0095_bb_vv")(bb,ab) )
    ;
  }
  // clang-format on
}

template void exachem::cc::cd_qed_ccsd_os::resid_part2<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);