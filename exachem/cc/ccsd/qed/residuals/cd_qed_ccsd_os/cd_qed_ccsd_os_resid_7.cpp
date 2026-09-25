/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_qed_ccsd_os_resid_7.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_os::resid_part7(
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
        
    // flops: o1v1  = o2v1 o2v2 o1v1 o2v1 o2v1 o2v1 o1v1 o1v1 o1v1
    //  mems: o1v1  = o1v1 o1v1 o1v1 o2v0 o1v1 o1v1 o1v1 o1v1 o1v1
    ( tmps.at("0281_bb_vo")(ab,ib)  = -1.000 * dp.at("bb_oo")(jb,ib) * t1_1p.at("bb")(ab,jb) )
    ( tmps.at("0281_bb_vo")(ab,ib) -= dp.at("bb_ov")(jb,bb) * t2_1p.at("bbbb")(bb,ab,ib,jb) )
    ( tmps.at("bin1_bb_oo")(ib,jb)  = dp.at("bb_ov")(jb,bb) * t1.at("bb")(bb,ib) )
    ( tmps.at("0281_bb_vo")(ab,ib) -= tmps.at("bin1_bb_oo")(ib,jb) * t1_1p.at("bb")(ab,jb) )
    ( tmps.at("0281_bb_vo")(ab,ib) -= t1.at("bb")(ab,jb) * tmps.at("0039_bb_oo")(jb,ib) )
    ( tmps.at("0281_bb_vo")(ab,ib) += tmps.at("0188_bb_vo")(ab,ib) )
    .deallocate(tmps.at("0188_bb_vo"))
    
    // r1[bb] += +1.000 d-_aa(j,b) t2_1p_abab(b,a,j,i) 
    //          += +1.000 d-_bb(a,b) t1_1p_bb(b,i) 
    //          += -1.000 d-_bb(j,i) t1_1p_bb(a,j) 
    //          += -1.000 d-_bb(j,b) t2_1p_bbbb(b,a,i,j) 
    //          += -1.000 d-_bb(j,b) t1_1p_bb(a,j) t1_bb(b,i) 
    //          += -1.000 d-_bb(j,b) t1_bb(a,j) t1_1p_bb(b,i) 
    ( r1.at("bb")(ab,ib) += tmps.at("0281_bb_vo")(ab,ib) )
    
    // r1_1p[bb] += +1.000 d-_aa(j,b) t0_1p t2_1p_abab(b,a,j,i) 
    //             += +1.000 d-_bb(a,b) t0_1p t1_1p_bb(b,i) 
    //             += -1.000 d-_bb(j,i) t0_1p t1_1p_bb(a,j) 
    //             += -1.000 d-_bb(j,b) t0_1p t2_1p_bbbb(b,a,i,j) 
    //             += -1.000 d-_bb(j,b) t0_1p t1_1p_bb(a,j) t1_bb(b,i) 
    //             += -1.000 d-_bb(j,b) t0_1p t1_bb(a,j) t1_1p_bb(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += t0_1p * tmps.at("0281_bb_vo")(ab,ib) )
    
    // r1_2p[bb] += +2.000 d+_aa(j,b) t2_1p_abab(b,a,j,i) 
    //             += +2.000 d+_bb(a,b) t1_1p_bb(b,i) 
    //             += -2.000 d+_bb(j,i) t1_1p_bb(a,j) 
    //             += -2.000 d+_bb(j,b) t2_1p_bbbb(b,a,i,j) 
    //             += -2.000 d+_bb(j,b) t1_1p_bb(a,j) t1_bb(b,i) 
    //             += -2.000 d+_bb(j,b) t1_bb(a,j) t1_1p_bb(b,i) 
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0281_bb_vo")(ab,ib) )
    
    // r1_2p[bb] += +4.000 d-_aa(j,b) t0_2p t2_1p_abab(b,a,j,i) 
    //             += +4.000 d-_bb(a,b) t0_2p t1_1p_bb(b,i) 
    //             += -4.000 d-_bb(j,i) t0_2p t1_1p_bb(a,j) 
    //             += -4.000 d-_bb(j,b) t0_2p t2_1p_bbbb(b,a,i,j) 
    //             += -4.000 d-_bb(j,b) t0_2p t1_1p_bb(a,j) t1_bb(b,i) 
    //             += -4.000 d-_bb(j,b) t0_2p t1_bb(a,j) t1_1p_bb(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 4.000 * t0_2p * tmps.at("0281_bb_vo")(ab,ib) )
    
    // r2_1p[abab] += +1.000 d-_aa(k,c) t1_1p_aa(a,i) t2_1p_abab(c,b,k,j) 
    //               += +1.000 d-_bb(b,c) t1_1p_aa(a,i) t1_1p_bb(c,j) 
    //               += -1.000 d-_bb(k,j) t1_1p_aa(a,i) t1_1p_bb(b,k) 
    //               += -1.000 d-_bb(k,c) t1_1p_aa(a,i) t2_1p_bbbb(c,b,j,k) 
    //               += -1.000 d-_bb(k,c) t1_1p_aa(a,i) t1_1p_bb(b,k) t1_bb(c,j) 
    //               += -1.000 d-_bb(k,c) t1_1p_aa(a,i) t1_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0281_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p[abab] += +4.000 d-_aa(k,c) t1_2p_aa(a,i) t2_1p_abab(c,b,k,j) 
    //               += +4.000 d-_bb(b,c) t1_2p_aa(a,i) t1_1p_bb(c,j) 
    //               += -4.000 d-_bb(k,j) t1_2p_aa(a,i) t1_1p_bb(b,k) 
    //               += -4.000 d-_bb(k,c) t1_2p_aa(a,i) t2_1p_bbbb(c,b,j,k) 
    //               += -4.000 d-_bb(k,c) t1_2p_aa(a,i) t1_1p_bb(b,k) t1_bb(c,j) 
    //               += -4.000 d-_bb(k,c) t1_2p_aa(a,i) t1_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.000 * tmps.at("0281_bb_vo")(bb,jb) * t1_2p.at("aa")(aa,ia) )
    .deallocate(tmps.at("0281_bb_vo"))
    .allocate(tmps.at("0282_bb_vo"))
    
    // flops: o1v1  = o1v1
    //  mems: o1v1  = o1v1
    ( tmps.at("0282_bb_vo")(ab,ib)  = -1.000 * tmps.at("0205_bb_vo")(ab,ib) )
    ( tmps.at("0282_bb_vo")(ab,ib) += tmps.at("0155_bb_vo")(ab,ib) )
    .deallocate(tmps.at("0205_bb_vo"))
    .deallocate(tmps.at("0155_bb_vo"))
    
    // r1[bb] += +1.000 d-_aa(j,b) t0_1p t2_abab(b,a,j,i) 
    //          += +1.000 d-_bb(a,b) t0_1p t1_bb(b,i) 
    //          += -1.000 d-_bb(j,i) t0_1p t1_bb(a,j) 
    //          += -1.000 d-_bb(j,b) t0_1p t2_bbbb(b,a,i,j) 
    //          += -1.000 d-_bb(j,b) t0_1p t1_bb(a,j) t1_bb(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += t0_1p * tmps.at("0282_bb_vo")(ab,ib) )
    
    // r1_1p[bb] += +1.000 d+_aa(j,b) t2_abab(b,a,j,i) 
    //             += +1.000 d+_bb(a,b) t1_bb(b,i) 
    //             += -1.000 d+_bb(j,i) t1_bb(a,j) 
    //             += -1.000 d+_bb(j,b) t2_bbbb(b,a,i,j) 
    //             += -1.000 d+_bb(j,b) t1_bb(a,j) t1_bb(b,i) 
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0282_bb_vo")(ab,ib) )
    
    // r1_1p[bb] += +2.000 d-_aa(j,b) t0_2p t2_abab(b,a,j,i) 
    //             += +2.000 d-_bb(a,b) t0_2p t1_bb(b,i) 
    //             += -2.000 d-_bb(j,i) t0_2p t1_bb(a,j) 
    //             += -2.000 d-_bb(j,b) t0_2p t2_bbbb(b,a,i,j) 
    //             += -2.000 d-_bb(j,b) t0_2p t1_bb(a,j) t1_bb(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += 2.000 * t0_2p * tmps.at("0282_bb_vo")(ab,ib) )
    
    // r2[abab] += +1.000 d-_aa(k,c) t1_1p_aa(a,i) t2_abab(c,b,k,j) 
    //            += +1.000 d-_bb(b,c) t1_1p_aa(a,i) t1_bb(c,j) 
    //            += -1.000 d-_bb(k,j) t1_1p_aa(a,i) t1_bb(b,k) 
    //            += -1.000 d-_bb(k,c) t1_1p_aa(a,i) t2_bbbb(c,b,j,k) 
    //            += -1.000 d-_bb(k,c) t1_1p_aa(a,i) t1_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0282_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_1p[abab] += +2.000 d-_aa(k,c) t1_2p_aa(a,i) t2_abab(c,b,k,j) 
    //               += +2.000 d-_bb(b,c) t1_2p_aa(a,i) t1_bb(c,j) 
    //               += -2.000 d-_bb(k,j) t1_2p_aa(a,i) t1_bb(b,k) 
    //               += -2.000 d-_bb(k,c) t1_2p_aa(a,i) t2_bbbb(c,b,j,k) 
    //               += -2.000 d-_bb(k,c) t1_2p_aa(a,i) t1_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0282_bb_vo")(bb,jb) * t1_2p.at("aa")(aa,ia) )
    .deallocate(tmps.at("0282_bb_vo"))
    .allocate(tmps.at("0283_aa_vo"))
    
    // flops: o1v1  = o1v2 o2v2 o1v1
    //  mems: o1v1  = o1v1 o1v1 o1v1
    ( tmps.at("0283_aa_vo")(aa,ia)  = dp.at("aa_vv")(aa,ba) * t1_1p.at("aa")(ba,ia) )
    ( tmps.at("0283_aa_vo")(aa,ia) += dp.at("bb_ov")(jb,bb) * t2_1p.at("abab")(aa,bb,ia,jb) )
    .allocate(tmps.at("0284_aa_vo"))
    
    // flops: o1v1  = o2v1 o2v2 o1v1 o2v1 o2v1 o2v1 o1v1 o1v1 o1v1
    //  mems: o1v1  = o1v1 o1v1 o1v1 o2v0 o1v1 o1v1 o1v1 o1v1 o1v1
    ( tmps.at("0284_aa_vo")(aa,ia)  = -1.000 * dp.at("aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    ( tmps.at("0284_aa_vo")(aa,ia) -= dp.at("aa_ov")(ja,ba) * t2_1p.at("aaaa")(ba,aa,ia,ja) )
    ( tmps.at("bin1_aa_oo")(ia,ja)  = dp.at("aa_ov")(ja,ba) * t1.at("aa")(ba,ia) )
    ( tmps.at("0284_aa_vo")(aa,ia) -= tmps.at("bin1_aa_oo")(ia,ja) * t1_1p.at("aa")(aa,ja) )
    ( tmps.at("0284_aa_vo")(aa,ia) -= t1.at("aa")(aa,ja) * tmps.at("0031_aa_oo")(ja,ia) )
    ( tmps.at("0284_aa_vo")(aa,ia) += tmps.at("0283_aa_vo")(aa,ia) )
    
    // r1[aa] += +1.000 d-_aa(a,b) t1_1p_aa(b,i) 
    //          += +1.000 d-_bb(j,b) t2_1p_abab(a,b,i,j) 
    //          += -1.000 d-_aa(j,i) t1_1p_aa(a,j) 
    //          += -1.000 d-_aa(j,b) t2_1p_aaaa(b,a,i,j) 
    //          += -1.000 d-_aa(j,b) t1_1p_aa(a,j) t1_aa(b,i) 
    //          += -1.000 d-_aa(j,b) t1_aa(a,j) t1_1p_aa(b,i) 
    ( r1.at("aa")(aa,ia) += tmps.at("0284_aa_vo")(aa,ia) )
    
    // r1_1p[aa] += +1.000 d-_aa(a,b) t0_1p t1_1p_aa(b,i) 
    //             += +1.000 d-_bb(j,b) t0_1p t2_1p_abab(a,b,i,j) 
    //             += -1.000 d-_aa(j,i) t0_1p t1_1p_aa(a,j) 
    //             += -1.000 d-_aa(j,b) t0_1p t2_1p_aaaa(b,a,i,j) 
    //             += -1.000 d-_aa(j,b) t0_1p t1_1p_aa(a,j) t1_aa(b,i) 
    //             += -1.000 d-_aa(j,b) t0_1p t1_aa(a,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t0_1p * tmps.at("0284_aa_vo")(aa,ia) )
    
    // r1_2p[aa] += +2.000 d+_aa(a,b) t1_1p_aa(b,i) 
    //             += +2.000 d+_bb(j,b) t2_1p_abab(a,b,i,j) 
    //             += -2.000 d+_aa(j,i) t1_1p_aa(a,j) 
    //             += -2.000 d+_aa(j,b) t2_1p_aaaa(b,a,i,j) 
    //             += -2.000 d+_aa(j,b) t1_1p_aa(a,j) t1_aa(b,i) 
    //             += -2.000 d+_aa(j,b) t1_aa(a,j) t1_1p_aa(b,i) 
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0284_aa_vo")(aa,ia) )
    
    // r1_2p[aa] += +4.000 d-_aa(a,b) t0_2p t1_1p_aa(b,i) 
    //             += +4.000 d-_bb(j,b) t0_2p t2_1p_abab(a,b,i,j) 
    //             += -4.000 d-_aa(j,i) t0_2p t1_1p_aa(a,j) 
    //             += -4.000 d-_aa(j,b) t0_2p t2_1p_aaaa(b,a,i,j) 
    //             += -4.000 d-_aa(j,b) t0_2p t1_1p_aa(a,j) t1_aa(b,i) 
    //             += -4.000 d-_aa(j,b) t0_2p t1_aa(a,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 4.000 * t0_2p * tmps.at("0284_aa_vo")(aa,ia) )
    
    // r2_1p[abab] += +1.000 d-_aa(a,c) t1_1p_bb(b,j) t1_1p_aa(c,i) 
    //               += +1.000 d-_bb(k,c) t1_1p_bb(b,j) t2_1p_abab(a,c,i,k) 
    //               += -1.000 d-_aa(k,i) t1_1p_aa(a,k) t1_1p_bb(b,j) 
    //               += -1.000 d-_aa(k,c) t1_1p_bb(b,j) t2_1p_aaaa(c,a,i,k) 
    //               += -1.000 d-_aa(k,c) t1_1p_aa(a,k) t1_1p_bb(b,j) t1_aa(c,i) 
    //               += -1.000 d-_aa(k,c) t1_aa(a,k) t1_1p_bb(b,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0284_aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    
    // r2_2p[abab] += +4.000 d-_aa(a,c) t1_2p_bb(b,j) t1_1p_aa(c,i) 
    //               += +4.000 d-_bb(k,c) t1_2p_bb(b,j) t2_1p_abab(a,c,i,k) 
    //               += -4.000 d-_aa(k,i) t1_1p_aa(a,k) t1_2p_bb(b,j) 
    //               += -4.000 d-_aa(k,c) t1_2p_bb(b,j) t2_1p_aaaa(c,a,i,k) 
    //               += -4.000 d-_aa(k,c) t1_1p_aa(a,k) t1_2p_bb(b,j) t1_aa(c,i) 
    //               += -4.000 d-_aa(k,c) t1_aa(a,k) t1_2p_bb(b,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.000 * tmps.at("0284_aa_vo")(aa,ia) * t1_2p.at("bb")(bb,jb) )
    .deallocate(tmps.at("0284_aa_vo"))
    .allocate(tmps.at("0285_aa_vo"))
    
    // flops: o1v1  = o2v1 o2v1 o2v1 o1v1 o2v2 o1v1 o1v1
    //  mems: o1v1  = o2v0 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1
    ( tmps.at("bin1_aa_oo")(ia,ja)  = dp.at("aa_ov")(ja,ba) * t1.at("aa")(ba,ia) )
    ( tmps.at("0285_aa_vo")(aa,ia)  = -1.000 * tmps.at("bin1_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    ( tmps.at("0285_aa_vo")(aa,ia) -= dp.at("aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    ( tmps.at("0285_aa_vo")(aa,ia) -= dp.at("aa_ov")(ja,ba) * t2.at("aaaa")(ba,aa,ia,ja) )
    ( tmps.at("0285_aa_vo")(aa,ia) += tmps.at("0153_aa_vo")(aa,ia) )
    
    // r1[aa] += +1.000 d-_aa(a,b) t0_1p t1_aa(b,i) 
    //          += +1.000 d-_bb(j,b) t0_1p t2_abab(a,b,i,j) 
    //          += -1.000 d-_aa(j,i) t0_1p t1_aa(a,j) 
    //          += -1.000 d-_aa(j,b) t0_1p t2_aaaa(b,a,i,j) 
    //          += -1.000 d-_aa(j,b) t0_1p t1_aa(a,j) t1_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += t0_1p * tmps.at("0285_aa_vo")(aa,ia) )
    
    // r1_1p[aa] += +1.000 d+_aa(a,b) t1_aa(b,i) 
    //             += +1.000 d+_bb(j,b) t2_abab(a,b,i,j) 
    //             += -1.000 d+_aa(j,i) t1_aa(a,j) 
    //             += -1.000 d+_aa(j,b) t2_aaaa(b,a,i,j) 
    //             += -1.000 d+_aa(j,b) t1_aa(a,j) t1_aa(b,i) 
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0285_aa_vo")(aa,ia) )
    
    // r1_1p[aa] += +2.000 d-_aa(a,b) t0_2p t1_aa(b,i) 
    //             += +2.000 d-_bb(j,b) t0_2p t2_abab(a,b,i,j) 
    //             += -2.000 d-_aa(j,i) t0_2p t1_aa(a,j) 
    //             += -2.000 d-_aa(j,b) t0_2p t2_aaaa(b,a,i,j) 
    //             += -2.000 d-_aa(j,b) t0_2p t1_aa(a,j) t1_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 2.000 * t0_2p * tmps.at("0285_aa_vo")(aa,ia) )
    
    // r2[abab] += +1.000 d-_aa(a,c) t1_1p_bb(b,j) t1_aa(c,i) 
    //            += +1.000 d-_bb(k,c) t1_1p_bb(b,j) t2_abab(a,c,i,k) 
    //            += -1.000 d-_aa(k,i) t1_aa(a,k) t1_1p_bb(b,j) 
    //            += -1.000 d-_aa(k,c) t1_1p_bb(b,j) t2_aaaa(c,a,i,k) 
    //            += -1.000 d-_aa(k,c) t1_aa(a,k) t1_1p_bb(b,j) t1_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0285_aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    
    // r2_1p[abab] += +2.000 d-_aa(a,c) t1_2p_bb(b,j) t1_aa(c,i) 
    //               += +2.000 d-_bb(k,c) t1_2p_bb(b,j) t2_abab(a,c,i,k) 
    //               += -2.000 d-_aa(k,i) t1_aa(a,k) t1_2p_bb(b,j) 
    //               += -2.000 d-_aa(k,c) t1_2p_bb(b,j) t2_aaaa(c,a,i,k) 
    //               += -2.000 d-_aa(k,c) t1_aa(a,k) t1_2p_bb(b,j) t1_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0285_aa_vo")(aa,ia) * t1_2p.at("bb")(bb,jb) )
    .deallocate(tmps.at("0285_aa_vo"))
    .allocate(tmps.at("0286_aaaa_ovvo"))
    
    // flops: o2v2  = o3v1Q1 o3v2 o3v2 o3v2 o2v2 o3v3 o2v2
    //  mems: o2v2  = o3v1 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("0025_aa_voQ")(aa,ja,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0286_aaaa_ovvo")(ia,ba,aa,ja)  = t1.at("aa")(ba,ka) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = t1.at("aa")(ca,ja) * tmps.at("0055_aaaa_oovv")(ka,ia,aa,ca) )
    ( tmps.at("0286_aaaa_ovvo")(ia,ba,aa,ja) += t1.at("aa")(ba,ka) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) )
    ( tmps.at("0286_aaaa_ovvo")(ia,ba,aa,ja) += t2.at("aaaa")(da,ba,ja,ka) * tmps.at("0259_aaaa_oovv")(ka,ia,aa,da) )
    .deallocate(tmps.at("0259_aaaa_oovv"))
    .deallocate(tmps.at("0055_aaaa_oovv"))
    
    // r2[aaaa] += -1.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_aaaa(d,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_aa(c,j) 
    //            += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_abab(b,c,j,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0286_aaaa_ovvo")(ia,ba,aa,ja) )
    
    // r2[aaaa] += -1.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_aaaa(d,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_aa(c,j) 
    //            += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_abab(b,c,j,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0286_aaaa_ovvo")(ja,ba,aa,ia) )
    
    // r2[aaaa] += -1.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_aaaa(d,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_aa(c,j) 
    //            += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_abab(b,c,j,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0286_aaaa_ovvo")(ia,aa,ba,ja) )
    
    // r2[aaaa] += -1.000 P(i,j) P(a,b) <a,k||c,d>_aaaa t1_aa(c,i) t2_aaaa(d,b,j,k) 
    //            += -1.000 P(i,j) P(a,b) <a,k||i,c>_aaaa t1_aa(b,k) t1_aa(c,j) 
    //            += -1.000 P(i,j) P(a,b) <k,l||i,c>_abab t1_aa(a,k) t2_abab(b,c,j,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0286_aaaa_ovvo")(ja,aa,ba,ia) )
    .deallocate(tmps.at("0286_aaaa_ovvo"))
    .allocate(tmps.at("0287_aabb_oooo"))
    
    // flops: o4v0  = o4v0Q1 o4v1 o4v0 o4v0Q1 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0 o4v0 o4v0
    ( tmps.at("0287_aabb_oooo")(ka,ia,lb,jb)  = tmps.at("0053_aa_ooQ")(ka,ia,Q) * tmps.at("0026_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0287_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0211_aabb_ovoo")(ka,ca,lb,jb) * t1.at("aa")(ca,ia) )
    ( tmps.at("0287_aabb_oooo")(ka,ia,lb,jb) += chol.at("aa_ooQ")(ka,ia,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    
    // r2[abab] += +0.500 <k,l||c,j>_abab t1_aa(c,i) t2_abab(a,b,k,l) 
    //            += +0.500 <l,k||c,j>_abab t1_aa(c,i) t2_abab(a,b,l,k) 
    //            += +0.500 <k,l||i,j>_abab t2_abab(a,b,k,l) 
    //            += +0.500 <l,k||i,j>_abab t2_abab(a,b,l,k) 
    //            += +0.500 <k,l||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_abab(a,b,k,l) 
    //            += +0.500 <l,k||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0287_aabb_oooo")(ka,ia,lb,jb) * t2.at("abab")(aa,bb,ka,lb) )
    
    // r2_1p[abab] += +0.500 <k,l||c,j>_abab t1_aa(c,i) t2_1p_abab(a,b,k,l) 
    //               += +0.500 <l,k||c,j>_abab t1_aa(c,i) t2_1p_abab(a,b,l,k) 
    //               += +0.500 <k,l||i,j>_abab t2_1p_abab(a,b,k,l) 
    //               += +0.500 <l,k||i,j>_abab t2_1p_abab(a,b,l,k) 
    //               += +0.500 <k,l||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_1p_abab(a,b,k,l) 
    //               += +0.500 <l,k||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_1p_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0287_aabb_oooo")(ka,ia,lb,jb) * t2_1p.at("abab")(aa,bb,ka,lb) )
    
    // r2_2p[abab] += +1.000 <k,l||c,j>_abab t1_aa(c,i) t2_2p_abab(a,b,k,l) 
    //               += +1.000 <l,k||c,j>_abab t1_aa(c,i) t2_2p_abab(a,b,l,k) 
    //               += +1.000 <k,l||i,j>_abab t2_2p_abab(a,b,k,l) 
    //               += +1.000 <l,k||i,j>_abab t2_2p_abab(a,b,l,k) 
    //               += +1.000 <k,l||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_2p_abab(a,b,k,l) 
    //               += +1.000 <l,k||c,d>_abab t1_aa(c,i) t1_bb(d,j) t2_2p_abab(a,b,l,k) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0287_aabb_oooo")(ka,ia,lb,jb) * t2_2p.at("abab")(aa,bb,ka,lb) )
    
    // r2_2p[abab] += +2.000 <k,l||c,j>_abab t1_aa(a,k) t1_2p_bb(b,l) t1_aa(c,i) 
    //               += +2.000 <k,l||i,j>_abab t1_aa(a,k) t1_2p_bb(b,l) 
    //               += +2.000 <k,l||c,d>_abab t1_aa(a,k) t1_2p_bb(b,l) t1_aa(c,i) t1_bb(d,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t1_2p.at("bb")(bb,lb) * tmps.at("0287_aabb_oooo")(ka,ia,lb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0287_aabb_oooo"))
    .allocate(tmps.at("0288_bb_oo"))
    
    // flops: o2v0  = o3v2 o3v2 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0
    ( tmps.at("0288_bb_oo")(jb,ib)  = -1.000 * t2.at("bbbb")(cb,bb,ib,kb) * tmps.at("0075_bbbb_ovov")(jb,cb,kb,bb) )
    ( tmps.at("0288_bb_oo")(jb,ib) += t2.at("bbbb")(cb,bb,ib,kb) * tmps.at("0075_bbbb_ovov")(jb,bb,kb,cb) )
    
    // r1_1p[bb] += -0.500 <j,k||c,b>_bbbb t1_1p_bb(a,j) t2_bbbb(c,b,i,k) 
    //             += -0.500 <j,k||c,b>_bbbb t1_1p_bb(a,j) t2_bbbb(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += 0.500 * tmps.at("0288_bb_oo")(jb,ib) * t1_1p.at("bb")(ab,jb) )
    
    // r1_2p[bb] += -1.000 <j,k||c,b>_bbbb t1_2p_bb(a,j) t2_bbbb(c,b,i,k) 
    //             += -1.000 <j,k||c,b>_bbbb t1_2p_bb(a,j) t2_bbbb(c,b,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += tmps.at("0288_bb_oo")(jb,ib) * t1_2p.at("bb")(ab,jb) )
    
    // r2[abab] += -0.500 <l,k||d,c>_bbbb t2_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    //            += -0.500 <l,k||d,c>_bbbb t2_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += 0.500 * tmps.at("0288_bb_oo")(lb,jb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_1p[abab] += -0.500 <l,k||d,c>_bbbb t2_1p_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    //               += -0.500 <l,k||d,c>_bbbb t2_1p_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 0.500 * tmps.at("0288_bb_oo")(lb,jb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_bbbb t2_2p_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    //               += -1.000 <l,k||d,c>_bbbb t2_2p_abab(a,b,i,l) t2_bbbb(d,c,j,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += tmps.at("0288_bb_oo")(lb,jb) * t2_2p.at("abab")(aa,bb,ia,lb) )
    .deallocate(tmps.at("0288_bb_oo"))
    .allocate(tmps.at("0289_aabb_oooo"))
    
    // flops: o4v0  = o4v1 o4v0Q1 o4v0 o4v0Q1 o4v1 o4v0 o4v0
    //  mems: o4v0  = o4v0 o4v0 o4v0 o4v0 o4v0 o4v0 o4v0
    ( tmps.at("0289_aabb_oooo")(ka,ia,lb,jb)  = t1_1p.at("aa")(ca,ia) * tmps.at("0211_aabb_ovoo")(ka,ca,lb,jb) )
    ( tmps.at("0289_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0137_aa_ooQ")(ka,ia,Q) * tmps.at("0026_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0289_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0053_aa_ooQ")(ka,ia,Q) * tmps.at("0029_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0289_aabb_oooo")(ka,ia,lb,jb) += tmps.at("0121_aabb_ooov")(ka,ia,lb,cb) * t1_1p.at("bb")(cb,jb) )
    
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
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,ka,lb) * tmps.at("0289_aabb_oooo")(ka,ia,lb,jb) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_1p_bb(c,j) 
    //               += +2.000 <k,l||c,d>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_aa(c,i) t1_1p_bb(d,j) 
    //               += +2.000 <k,l||c,j>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_1p_aa(c,i) 
    //               += +2.000 <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(b,l) t1_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("0289_aabb_oooo")(ka,ia,lb,jb) * t1_1p.at("bb")(bb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
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
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_1p.at("abab")(aa,bb,ka,lb) * tmps.at("0289_aabb_oooo")(ka,ia,lb,jb) )
    .deallocate(tmps.at("0289_aabb_oooo"))
    .allocate(tmps.at("0290_bbbb_ooov"))
    
    // flops: o3v1  = o3v2 o3v3 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1
    ( tmps.at("0290_bbbb_ooov")(kb,ib,jb,bb)  = t2.at("bbbb")(cb,bb,ib,jb) * f.at("bb_ov")(kb,cb) )
    ( tmps.at("0290_bbbb_ooov")(kb,ib,jb,bb) += 0.500 * t2.at("bbbb")(db,cb,ib,jb) * tmps.at("0086_bbbb_ovvv")(kb,db,bb,cb) )
    .allocate(tmps.at("0291_bbbb_ovov"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0291_bbbb_ovov")(ib,bb,jb,ab)  = tmps.at("0290_bbbb_ooov")(kb,ib,jb,ab) * t1.at("bb")(bb,kb) )
    
    // r2[bbbb] += -1.000 P(a,b) f_bb(k,c) t1_bb(a,k) t2_bbbb(c,b,i,j) 
    //            += -0.500 P(a,b) <a,k||d,c>_bbbb t1_bb(b,k) t2_bbbb(d,c,i,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0291_bbbb_ovov")(ib,ab,jb,bb) )
    
    // r2[bbbb] += -1.000 P(a,b) f_bb(k,c) t1_bb(a,k) t2_bbbb(c,b,i,j) 
    //            += -0.500 P(a,b) <a,k||d,c>_bbbb t1_bb(b,k) t2_bbbb(d,c,i,j) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0291_bbbb_ovov")(ib,bb,jb,ab) )
    .deallocate(tmps.at("0291_bbbb_ovov"))
    .allocate(tmps.at("0292_bbbb_vovo"))
    
    // flops: o2v2  = o4v2 o3v2
    //  mems: o2v2  = o3v1 o2v2
    ( tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb)  = t2.at("bbbb")(db,ab,jb,lb) * tmps.at("0228_bbbb_ovoo")(lb,db,ib,kb) )
    ( tmps.at("0292_bbbb_vovo")(ab,ib,bb,jb)  = t1.at("bb")(bb,kb) * tmps.at("bin1_bbbb_vooo")(ab,ib,jb,kb) )
    
    // r2[bbbb] += -1.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_bb(c,i) t2_bbbb(d,b,j,l) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0292_bbbb_vovo")(ab,ib,bb,jb) )
    
    // r2[bbbb] += -1.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_bb(c,i) t2_bbbb(d,b,j,l) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0292_bbbb_vovo")(ab,jb,bb,ib) )
    .deallocate(tmps.at("0292_bbbb_vovo"))
    .allocate(tmps.at("0293_bbbb_vooo"))
    
    // flops: o3v1  = o3v2 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_bbbb_vooo")(db,ib,kb,lb)  = t1.at("bb")(cb,ib) * tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) )
    ( tmps.at("0293_bbbb_vooo")(ab,ib,kb,jb)  = tmps.at("bin1_bbbb_vooo")(db,ib,kb,lb) * t2_1p.at("bbbb")(db,ab,jb,lb) )
    .allocate(tmps.at("0294_bbbb_vovo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0294_bbbb_vovo")(ab,ib,bb,jb)  = tmps.at("0293_bbbb_vooo")(ab,ib,kb,jb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,k) t1_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0294_bbbb_vovo")(ab,ib,bb,jb) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,k) t1_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0294_bbbb_vovo")(ab,jb,bb,ib) )
    .deallocate(tmps.at("0294_bbbb_vovo"))
    .allocate(tmps.at("0295_aaaa_ooov"))
    
    // flops: o3v1  = o3v3 o3v2 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1
    ( tmps.at("0295_aaaa_ooov")(ka,ia,ja,ba)  = 0.500 * tmps.at("0090_aaaa_ovvv")(ka,da,ba,ca) * t2.at("aaaa")(da,ca,ia,ja) )
    ( tmps.at("0295_aaaa_ooov")(ka,ia,ja,ba) += t2.at("aaaa")(ca,ba,ia,ja) * f.at("aa_ov")(ka,ca) )
    .allocate(tmps.at("0296_aaaa_ovov"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0296_aaaa_ovov")(ia,ba,ja,aa)  = t1.at("aa")(ba,ka) * tmps.at("0295_aaaa_ooov")(ka,ia,ja,aa) )
    
    // r2[aaaa] += -1.000 P(a,b) f_aa(k,c) t1_aa(a,k) t2_aaaa(c,b,i,j) 
    //            += -0.500 P(a,b) <a,k||d,c>_aaaa t1_aa(b,k) t2_aaaa(d,c,i,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0296_aaaa_ovov")(ia,aa,ja,ba) )
    
    // r2[aaaa] += -1.000 P(a,b) f_aa(k,c) t1_aa(a,k) t2_aaaa(c,b,i,j) 
    //            += -0.500 P(a,b) <a,k||d,c>_aaaa t1_aa(b,k) t2_aaaa(d,c,i,j) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0296_aaaa_ovov")(ia,ba,ja,aa) )
    .deallocate(tmps.at("0296_aaaa_ovov"))
    .allocate(tmps.at("0297_aaaa_vooo"))
    
    // flops: o3v1  = o3v2 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aaaa_vooo")(da,ia,ka,la)  = t1.at("aa")(ca,ia) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("0297_aaaa_vooo")(aa,ia,ka,ja)  = tmps.at("bin1_aaaa_vooo")(da,ia,ka,la) * t2.at("aaaa")(da,aa,ja,la) )
    .allocate(tmps.at("0298_aaaa_vovo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0298_aaaa_vovo")(aa,ia,ba,ja)  = t1.at("aa")(ba,ka) * tmps.at("0297_aaaa_vooo")(aa,ia,ka,ja) )
    
    // r2[aaaa] += -1.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0298_aaaa_vovo")(aa,ia,ba,ja) )
    
    // r2[aaaa] += -1.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0298_aaaa_vovo")(aa,ja,ba,ia) )
    .deallocate(tmps.at("0298_aaaa_vovo"))
    .allocate(tmps.at("0299_aaaa_vovo"))
    
    // flops: o2v2  = o3v2 o4v2 o3v2
    //  mems: o2v2  = o3v1 o3v1 o2v2
    ( tmps.at("bin2_aaaa_vooo")(da,ia,ka,la)  = tmps.at("0073_aaaa_ovov")(la,da,ka,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = t2_1p.at("aaaa")(da,aa,ja,la) * tmps.at("bin2_aaaa_vooo")(da,ia,ka,la) )
    ( tmps.at("0299_aaaa_vovo")(aa,ia,ba,ja)  = t1_1p.at("aa")(ba,ka) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0299_aaaa_vovo")(aa,ia,ba,ja) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0299_aaaa_vovo")(aa,ja,ba,ia) )
    .deallocate(tmps.at("0299_aaaa_vovo"))
    .allocate(tmps.at("0300_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0300_bb_oo")(kb,ib)  = tmps.at("0228_bbbb_ovoo")(kb,bb,ib,jb) * t1.at("bb")(bb,jb) )
    
    // r1[bb] += +1.000 <k,j||b,c>_bbbb t1_bb(a,k) t1_bb(b,j) t1_bb(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) += tmps.at("0300_bb_oo")(kb,ib) * t1.at("bb")(ab,kb) )
    
    // r1_1p[bb] += +1.000 <k,j||b,c>_bbbb t1_1p_bb(a,k) t1_bb(b,j) t1_bb(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0300_bb_oo")(kb,ib) * t1_1p.at("bb")(ab,kb) )
    
    // r1_2p[bb] += +2.000 <k,j||b,c>_bbbb t1_2p_bb(a,k) t1_bb(b,j) t1_bb(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0300_bb_oo")(kb,ib) * t1_2p.at("bb")(ab,kb) )
    .deallocate(tmps.at("0300_bb_oo"))
    .allocate(tmps.at("0301_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0301_aa_oo")(ia,ja)  = tmps.at("0164_aaaa_ooov")(ka,ia,ja,ca) * t1_2p.at("aa")(ca,ka) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_aa(a,j) t1_aa(b,i) t1_2p_aa(c,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0301_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) <k,l||c,d>_aaaa t1_aa(c,i) t1_2p_aa(d,k) t2_aaaa(a,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0301_aa_oo")(ia,la) * t2.at("aaaa")(aa,ba,ja,la) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_aaaa t1_aa(c,i) t1_2p_aa(d,k) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0301_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    .deallocate(tmps.at("0301_aa_oo"))
    .allocate(tmps.at("0302_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0302_bb_oo")(kb,ib)  = tmps.at("0228_bbbb_ovoo")(kb,cb,ib,jb) * t1_1p.at("bb")(cb,jb) )
    
    // r1_2p[bb] += -2.000 <k,j||b,c>_bbbb t1_1p_bb(a,k) t1_bb(b,i) t1_1p_bb(c,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * t1_1p.at("bb")(ab,kb) * tmps.at("0302_bb_oo")(kb,ib) )
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_bbbb t1_bb(c,j) t1_1p_bb(d,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0302_bb_oo")(lb,jb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_1p[bbbb] += -1.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_1p_bb(d,k) t2_bbbb(a,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) += tmps.at("0302_bb_oo")(lb,jb) * t2.at("bbbb")(ab,bb,ib,lb) )
    .deallocate(tmps.at("0302_bb_oo"))
    .allocate(tmps.at("0303_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0303_bb_oo")(lb,jb)  = tmps.at("0228_bbbb_ovoo")(lb,db,jb,kb) * t1_2p.at("bb")(db,kb) )
    
    // r2_2p[abab] += +2.000 <k,l||c,d>_bbbb t1_bb(c,j) t1_2p_bb(d,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0303_bb_oo")(lb,jb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_2p_bb(d,k) t2_bbbb(a,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0303_bb_oo")(lb,jb) * t2.at("bbbb")(ab,bb,ib,lb) )
    .deallocate(tmps.at("0303_bb_oo"))
    .allocate(tmps.at("0304_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0304_bb_oo")(kb,ib)  = tmps.at("0229_bbbb_ovoo")(kb,bb,ib,jb) * t1.at("bb")(bb,jb) )
    
    // r1_1p[bb] += +1.000 <k,j||b,c>_bbbb t1_bb(a,k) t1_bb(b,j) t1_1p_bb(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0304_bb_oo")(kb,ib) * t1.at("bb")(ab,kb) )
    
    // r1_2p[bb] += +2.000 <k,j||b,c>_bbbb t1_1p_bb(a,k) t1_bb(b,j) t1_1p_bb(c,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0304_bb_oo")(kb,ib) * t1_1p.at("bb")(ab,kb) )
    .deallocate(tmps.at("0304_bb_oo"))
    .allocate(tmps.at("0305_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0305_bb_oo")(ib,jb)  = tmps.at("0197_bbbb_ooov")(kb,ib,jb,cb) * t1_2p.at("bb")(cb,kb) )
    
    // r1_2p[bb] += +2.000 <k,j||b,c>_bbbb t1_bb(a,j) t1_bb(b,i) t1_2p_bb(c,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0305_bb_oo")(ib,jb) * t1.at("bb")(ab,jb) )
    
    // r2_2p[bbbb] += -2.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_2p_bb(d,k) t2_bbbb(a,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0305_bb_oo")(ib,lb) * t2.at("bbbb")(ab,bb,jb,lb) )
    .deallocate(tmps.at("0305_bb_oo"))
    .allocate(tmps.at("0306_aaaa_vvoo"))
    
    // flops: o2v2  = o3v2 o4v2 o3v2 o3v1Q1 o4v2 o3v2 o2v2 o2v2 o2v2 o2v2 o3v1Q1 o3v1Q1 o3v1 o3v2 o2v2 o2v2 o3v2 o3v2 o2v2 o2v2 o2v2
    //  mems: o2v2  = o3v1 o3v1 o2v2 o3v1 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o3v1 o3v1 o3v1 o2v2 o2v2 o2v2 o3v1 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin2_aaaa_vooo")(da,ja,ka,la)  = tmps.at("0073_aaaa_ovov")(la,ca,ka,da) * t1.at("aa")(ca,ja) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t2_1p.at("aaaa")(da,ba,ia,la) * tmps.at("bin2_aaaa_vooo")(da,ja,ka,la) )
    ( tmps.at("0306_aaaa_vvoo")(aa,ba,ia,ja)  = t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("aa_ooQ")(la,ia,Q) )
    ( tmps.at("bin2_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la) * t2_1p.at("aaaa")(ca,aa,ja,la) )
    ( tmps.at("0306_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin2_aaaa_vooo")(aa,ia,ja,ka) * t1.at("aa")(ba,ka) )
    ( tmps.at("0306_aaaa_vvoo")(aa,ba,ia,ja) += 2.000 * t1_2p.at("aa")(aa,ia) * tmps.at("0153_aa_vo")(ba,ja) )
    ( tmps.at("0306_aaaa_vvoo")(aa,ba,ia,ja) += 2.000 * dp.at("aa_vo")(aa,ia) * t1_2p.at("aa")(ba,ja) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = tmps.at("0025_aa_voQ")(ba,ia,Q) * tmps.at("0137_aa_ooQ")(ka,ja,Q) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) += tmps.at("0028_aa_voQ")(ba,ia,Q) * tmps.at("0053_aa_ooQ")(ka,ja,Q) )
    ( tmps.at("0306_aaaa_vvoo")(aa,ba,ia,ja) += t1.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = t1.at("aa")(ca,ia) * tmps.at("0162_aaaa_voov")(aa,ka,ja,ca) )
    ( tmps.at("0306_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) * t1_1p.at("aa")(ba,ka) )
    ( tmps.at("0306_aaaa_vvoo")(aa,ba,ia,ja) += t1_1p.at("aa")(aa,ia) * tmps.at("0283_aa_vo")(ba,ja) )
    .deallocate(tmps.at("0153_aa_vo"))
    
    // r2_1p[aaaa] += -1.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_aa(a,k) t1_1p_aa(c,i) t2_abab(b,d,j,l) 
    //               += -1.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_1p_abab(b,d,j,l) 
    //               += -1.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_1p_aa(c,j) 
    //               += +1.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_1p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(a,c) t1_2p_aa(b,i) t1_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) d-_bb(k,c) t1_2p_aa(a,i) t2_abab(b,c,j,k) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(a,i) t1_2p_aa(b,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += +1.000 P(i,j) P(a,b) <k,l||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0306_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_1p[aaaa] += -1.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_aa(a,k) t1_1p_aa(c,i) t2_abab(b,d,j,l) 
    //               += -1.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_1p_abab(b,d,j,l) 
    //               += -1.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_1p_aa(c,j) 
    //               += +1.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_1p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(a,c) t1_2p_aa(b,i) t1_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) d-_bb(k,c) t1_2p_aa(a,i) t2_abab(b,c,j,k) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(a,i) t1_2p_aa(b,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += +1.000 P(i,j) P(a,b) <k,l||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0306_aaaa_vvoo")(ba,aa,ja,ia) )
    
    // r2_1p[aaaa] += -1.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_aa(a,k) t1_1p_aa(c,i) t2_abab(b,d,j,l) 
    //               += -1.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_1p_abab(b,d,j,l) 
    //               += -1.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_1p_aa(c,j) 
    //               += +1.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_1p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(a,c) t1_2p_aa(b,i) t1_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) d-_bb(k,c) t1_2p_aa(a,i) t2_abab(b,c,j,k) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(a,i) t1_2p_aa(b,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += +1.000 P(i,j) P(a,b) <k,l||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0306_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_1p[aaaa] += -1.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_aa(a,k) t1_1p_aa(c,i) t2_abab(b,d,j,l) 
    //               += -1.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_aa(a,k) t1_aa(c,i) t2_1p_abab(b,d,j,l) 
    //               += -1.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -1.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_1p_aa(c,j) 
    //               += +1.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_1p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(a,c) t1_2p_aa(b,i) t1_aa(c,j) 
    //               += +2.000 P(i,j) P(a,b) d-_bb(k,c) t1_2p_aa(a,i) t2_abab(b,c,j,k) 
    //               += +2.000 P(i,j) P(a,b) d-_aa(a,i) t1_2p_aa(b,j) 
    //               += -1.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += +1.000 P(i,j) P(a,b) <k,l||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0306_aaaa_vvoo")(aa,ba,ia,ja) )
    .deallocate(tmps.at("0306_aaaa_vvoo"))
    .allocate(tmps.at("0307_aaaa_vvoo"))
    
    // flops: o2v2  = o3v1Q1 o4v2 o3v2 o1v2 o2v2 o3v1Q1 o3v1Q1 o3v1 o3v2 o2v2 o2v2 o2v2 o2v2 o3v1Q1 o4v2 o3v2 o2v2 o2v2 o3v2 o4v2 o3v2 o2v2 o3v2 o3v2 o2v2 o3v2 o4v2 o3v2 o2v2 o2v2 o2v2
    //  mems: o2v2  = o3v1 o3v1 o2v2 o1v1 o2v2 o3v1 o3v1 o3v1 o2v2 o2v2 o2v2 o1v1 o2v2 o3v1 o3v1 o2v2 o2v2 o2v2 o3v1 o3v1 o2v2 o2v2 o3v1 o2v2 o2v2 o3v1 o3v1 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("aa_ooQ")(la,ia,Q) )
    ( tmps.at("bin2_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la) * t2_1p.at("aaaa")(ca,aa,ja,la) )
    ( tmps.at("0307_aaaa_vvoo")(aa,ba,ia,ja)  = tmps.at("bin2_aaaa_vooo")(aa,ia,ja,ka) * t1_1p.at("aa")(ba,ka) )
    ( tmps.at("bin1_aa_vo")(aa,ia)  = dp.at("aa_vv")(aa,ca) * t1_2p.at("aa")(ca,ia) )
    ( tmps.at("0307_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vo")(aa,ia) * t1_1p.at("aa")(ba,ja) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = tmps.at("0025_aa_voQ")(ba,ia,Q) * tmps.at("0137_aa_ooQ")(ka,ja,Q) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) += tmps.at("0028_aa_voQ")(ba,ia,Q) * tmps.at("0053_aa_ooQ")(ka,ja,Q) )
    ( tmps.at("0307_aaaa_vvoo")(aa,ba,ia,ja) += t1_1p.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("bin1_aa_vo")(aa,ia)  = dp.at("bb_ov")(kb,cb) * t2_2p.at("abab")(aa,cb,ia,kb) )
    ( tmps.at("0307_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin1_aa_vo")(aa,ia) * t1_1p.at("aa")(ba,ja) )
    ( tmps.at("bin2_aaaa_vooo")(ca,ia,ka,la)  = chol.at("aa_ooQ")(la,ia,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = t2_2p.at("aaaa")(ca,aa,ja,la) * tmps.at("bin2_aaaa_vooo")(ca,ia,ka,la) )
    ( tmps.at("0307_aaaa_vvoo")(aa,ba,ia,ja) += t1.at("aa")(ba,ka) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) )
    ( tmps.at("bin2_aaaa_vooo")(da,ja,ka,la)  = tmps.at("0073_aaaa_ovov")(la,ca,ka,da) * t1.at("aa")(ca,ja) )
    ( tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka)  = t2_1p.at("aaaa")(da,ba,ia,la) * tmps.at("bin2_aaaa_vooo")(da,ja,ka,la) )
    ( tmps.at("0307_aaaa_vvoo")(aa,ba,ia,ja) += t1_1p.at("aa")(aa,ka) * tmps.at("bin1_aaaa_vooo")(ba,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka)  = tmps.at("0162_aaaa_voov")(aa,ka,ja,ca) * t1.at("aa")(ca,ia) )
    ( tmps.at("0307_aaaa_vvoo")(aa,ba,ia,ja) += t1_2p.at("aa")(ba,ka) * tmps.at("bin1_aaaa_vooo")(aa,ia,ja,ka) )
    ( tmps.at("bin1_aaaa_vooo")(da,ja,ka,la)  = t1_1p.at("aa")(ca,ja) * tmps.at("0073_aaaa_ovov")(la,ca,ka,da) )
    ( tmps.at("bin2_aaaa_vooo")(ba,ia,ja,ka)  = tmps.at("bin1_aaaa_vooo")(da,ja,ka,la) * t2_1p.at("aaaa")(da,ba,ia,la) )
    ( tmps.at("0307_aaaa_vvoo")(aa,ba,ia,ja) += tmps.at("bin2_aaaa_vooo")(ba,ia,ja,ka) * t1.at("aa")(aa,ka) )
    ( tmps.at("0307_aaaa_vvoo")(aa,ba,ia,ja) += 2.000 * tmps.at("0283_aa_vo")(ba,ja) * t1_2p.at("aa")(aa,ia) )
    .deallocate(tmps.at("0283_aa_vo"))
    .deallocate(tmps.at("0162_aaaa_voov"))
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_1p_aa(a,k) t1_1p_aa(c,i) t2_abab(b,d,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_1p_aa(a,k) t1_aa(c,i) t2_1p_abab(b,d,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(a,c) t1_2p_aa(b,i) t1_1p_aa(c,j) 
    //               += +4.000 P(i,j) P(a,b) d-_bb(k,c) t1_2p_aa(a,i) t2_1p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_2p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_2p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_aaaa(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <k,l||c,d>_aaaa t1_2p_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0307_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_1p_aa(a,k) t1_1p_aa(c,i) t2_abab(b,d,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_1p_aa(a,k) t1_aa(c,i) t2_1p_abab(b,d,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(a,c) t1_2p_aa(b,i) t1_1p_aa(c,j) 
    //               += +4.000 P(i,j) P(a,b) d-_bb(k,c) t1_2p_aa(a,i) t2_1p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_2p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_2p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_aaaa(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <k,l||c,d>_aaaa t1_2p_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0307_aaaa_vvoo")(ba,aa,ia,ja) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_1p_aa(a,k) t1_1p_aa(c,i) t2_abab(b,d,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_1p_aa(a,k) t1_aa(c,i) t2_1p_abab(b,d,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(a,c) t1_2p_aa(b,i) t1_1p_aa(c,j) 
    //               += +4.000 P(i,j) P(a,b) d-_bb(k,c) t1_2p_aa(a,i) t2_1p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_2p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_2p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_aaaa(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <k,l||c,d>_aaaa t1_2p_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0307_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_2p[aaaa] += -2.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_1p_aa(a,k) t1_1p_aa(c,i) t2_abab(b,d,j,l) 
    //               += -2.000 P(i,j) P(a,b) <k,l||c,d>_abab t1_1p_aa(a,k) t1_aa(c,i) t2_1p_abab(b,d,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_aaaa t1_1p_aa(a,k) t1_aa(c,i) t2_1p_aaaa(d,b,j,l) 
    //               += -4.000 P(i,j) P(a,b) d-_aa(a,c) t1_2p_aa(b,i) t1_1p_aa(c,j) 
    //               += +4.000 P(i,j) P(a,b) d-_bb(k,c) t1_2p_aa(a,i) t2_1p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_aaaa(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) d-_bb(k,c) t1_1p_aa(a,i) t2_2p_abab(b,c,j,k) 
    //               += -2.000 P(i,j) P(a,b) d-_aa(a,c) t1_1p_aa(b,i) t1_2p_aa(c,j) 
    //               += -2.000 P(i,j) P(a,b) <l,k||i,c>_aaaa t1_aa(a,k) t2_2p_aaaa(c,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <k,l||c,d>_aaaa t1_2p_aa(a,k) t1_aa(c,i) t2_aaaa(d,b,j,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0307_aaaa_vvoo")(ba,aa,ja,ia) )
    .deallocate(tmps.at("0307_aaaa_vvoo"))
    .allocate(tmps.at("0308_aa_oo"))
    
    // flops: o2v0  = o2v2Q1 o2v1Q1
    //  mems: o2v0  = o1v1Q1 o2v0
    ( tmps.at("bin1_aa_voQ")(da,ia,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t2.at("abab")(da,cb,ia,kb) )
    ( tmps.at("0308_aa_oo")(la,ia)  = chol.at("aa_ovQ")(la,da,Q) * tmps.at("bin1_aa_voQ")(da,ia,Q) )
    
    // r1[aa] += -0.500 <j,k||c,b>_abab t1_aa(a,j) t2_abab(c,b,i,k) 
    //          += -0.500 <j,k||b,c>_abab t1_aa(a,j) t2_abab(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= t1.at("aa")(aa,ja) * tmps.at("0308_aa_oo")(ja,ia) )
    
    // r1_1p[aa] += -0.500 <j,k||c,b>_abab t1_1p_aa(a,j) t2_abab(c,b,i,k) 
    //             += -0.500 <j,k||b,c>_abab t1_1p_aa(a,j) t2_abab(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1_1p.at("aa")(aa,ja) * tmps.at("0308_aa_oo")(ja,ia) )
    
    // r1_2p[aa] += -1.000 <j,k||c,b>_abab t1_2p_aa(a,j) t2_abab(c,b,i,k) 
    //             += -1.000 <j,k||b,c>_abab t1_2p_aa(a,j) t2_abab(b,c,i,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * t1_2p.at("aa")(aa,ja) * tmps.at("0308_aa_oo")(ja,ia) )
    
    // r2[abab] += -0.500 <l,k||d,c>_abab t2_abab(a,b,l,j) t2_abab(d,c,i,k) 
    //            += -0.500 <l,k||c,d>_abab t2_abab(a,b,l,j) t2_abab(c,d,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,la,jb) * tmps.at("0308_aa_oo")(la,ia) )
    
    // r2_1p[abab] += -0.500 <l,k||d,c>_abab t2_1p_abab(a,b,l,j) t2_abab(d,c,i,k) 
    //               += -0.500 <l,k||c,d>_abab t2_1p_abab(a,b,l,j) t2_abab(c,d,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0308_aa_oo")(la,ia) * t2_1p.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -1.000 <l,k||d,c>_abab t2_2p_abab(a,b,l,j) t2_abab(d,c,i,k) 
    //               += -1.000 <l,k||c,d>_abab t2_2p_abab(a,b,l,j) t2_abab(c,d,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0308_aa_oo")(la,ia) * t2_2p.at("abab")(aa,bb,la,jb) )
    .allocate(tmps.at("0309_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0309_aa_oo")(ia,la)  = tmps.at("0164_aaaa_ooov")(ka,ia,la,ca) * t1.at("aa")(ca,ka) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,jb) * tmps.at("0309_aa_oo")(ia,la) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,bb,la,jb) * tmps.at("0309_aa_oo")(ia,la) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_2p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("abab")(aa,bb,la,jb) * tmps.at("0309_aa_oo")(ia,la) )
    .allocate(tmps.at("0310_aaaa_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o2v1Q1 o4v0Q1 o4v2 o4v0Q1 o4v2 o2v2 o2v2 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v0 o2v2 o2v0Q1 o4v0 o2v2 o4v0 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aa_oo")(ja,la)  = t2.at("aaaa")(da,ca,ja,ka) * tmps.at("0073_aaaa_ovov")(la,ca,ka,da) )
    ( tmps.at("0310_aaaa_vvoo")(aa,ba,ja,ia)  = 0.500 * t2.at("aaaa")(aa,ba,ia,la) * tmps.at("bin1_aa_oo")(ja,la) )
    ( tmps.at("bin1_aa_ooQ")(ja,la,Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1.at("aa")(ca,ja) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("bin1_aa_ooQ")(ja,la,Q) )
    ( tmps.at("0310_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) * t2.at("aaaa")(aa,ba,ka,la) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0053_aa_ooQ")(ka,ia,Q) * chol.at("aa_ooQ")(la,ja,Q) )
    ( tmps.at("0310_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) * t2.at("aaaa")(aa,ba,ka,la) )
    ( tmps.at("0310_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("0309_aa_oo")(ja,la) * t2.at("aaaa")(aa,ba,ia,la) )
    ( tmps.at("0310_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("0308_aa_oo")(la,ia) * t2.at("aaaa")(aa,ba,ja,la) )
    
    // r2[aaaa] += -1.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_aaaa(a,b,j,l) 
    //            += -0.500 P(i,j) <l,k||d,c>_abab t2_aaaa(a,b,i,l) t2_abab(d,c,j,k) 
    //            += -0.500 P(i,j) <l,k||c,d>_abab t2_aaaa(a,b,i,l) t2_abab(c,d,j,k) 
    //            += +0.500 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_aaaa(a,b,k,l) 
    //            += +0.500 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_aaaa(a,b,k,l) 
    //            += -0.500 P(i,j) <l,k||d,c>_aaaa t2_aaaa(a,b,i,l) t2_aaaa(d,c,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0310_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2[aaaa] += -1.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_aaaa(a,b,j,l) 
    //            += -0.500 P(i,j) <l,k||d,c>_abab t2_aaaa(a,b,i,l) t2_abab(d,c,j,k) 
    //            += -0.500 P(i,j) <l,k||c,d>_abab t2_aaaa(a,b,i,l) t2_abab(c,d,j,k) 
    //            += +0.500 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_aaaa(a,b,k,l) 
    //            += +0.500 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_aaaa(a,b,k,l) 
    //            += -0.500 P(i,j) <l,k||d,c>_aaaa t2_aaaa(a,b,i,l) t2_aaaa(d,c,j,k) 
    ( r2.at("aaaa")(aa,ba,ia,ja) += tmps.at("0310_aaaa_vvoo")(aa,ba,ja,ia) )
    .deallocate(tmps.at("0310_aaaa_vvoo"))
    .allocate(tmps.at("0311_aaaa_ooov"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0311_aaaa_ooov")(ka,ia,ja,ca)  = t1_1p.at("aa")(ba,ia) * tmps.at("0073_aaaa_ovov")(ka,ba,ja,ca) )
    
    // r1_2p[aa] += -2.000 <k,j||b,c>_aaaa t1_aa(a,j) t1_1p_aa(b,k) t1_1p_aa(c,i) 
    // flops: o1v1 += o3v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    ( tmps.at("bin1_aa_oo")(ia,ja)  = tmps.at("0311_aaaa_ooov")(ka,ia,ja,ba) * t1_1p.at("aa")(ba,ka) )
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1.at("aa")(aa,ja) * tmps.at("bin1_aa_oo")(ia,ja) )
    
    // r2_1p[abab] += -1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(da,bb,la,jb) * tmps.at("0311_aaaa_ooov")(ka,ia,la,da) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_1p.at("abab")(da,bb,la,jb) * tmps.at("0311_aaaa_ooov")(ka,ia,la,da) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0312_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0312_aa_oo")(ia,la)  = tmps.at("0311_aaaa_ooov")(ka,ia,la,ca) * t1.at("aa")(ca,ka) )
    .deallocate(tmps.at("0311_aaaa_ooov"))
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_1p_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,bb,la,jb) * tmps.at("0312_aa_oo")(ia,la) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_1p_aa(d,i) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_1p.at("abab")(aa,bb,la,jb) * tmps.at("0312_aa_oo")(ia,la) )
    .allocate(tmps.at("0313_aaaa_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o3v2 o2v2 o3v2 o3v2 o4v0Q1 o4v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v2 o2v2 o4v0Q1 o4v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v2 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o4v0 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v2 o2v2 o4v0 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0313_aaaa_vvoo")(aa,ba,ja,ia)  = t2.at("aaaa")(aa,ba,ia,la) * tmps.at("0265_aa_oo")(ja,la) )
    ( tmps.at("bin1_aa_oo")(ia,la)  = tmps.at("0073_aaaa_ovov")(la,da,ka,ca) * t2.at("aaaa")(da,ca,ia,ka) )
    ( tmps.at("0313_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2_1p.at("aaaa")(aa,ba,ja,la) * tmps.at("bin1_aa_oo")(ia,la) )
    ( tmps.at("bin1_aa_oo")(ja,ka)  = t2_1p.at("aaaa")(da,ca,ja,la) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("0313_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2.at("aaaa")(aa,ba,ia,ka) * tmps.at("bin1_aa_oo")(ja,ka) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = chol.at("aa_ooQ")(la,ja,Q) * tmps.at("0137_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0313_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2.at("aaaa")(aa,ba,ka,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("bin1_aa_ooQ")(ja,la,Q)  = t1_1p.at("aa")(ca,ja) * chol.at("aa_ovQ")(la,ca,Q) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("bin1_aa_ooQ")(ja,la,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0313_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2.at("aaaa")(aa,ba,ka,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = chol.at("aa_ooQ")(la,ja,Q) * tmps.at("0053_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0313_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2_1p.at("aaaa")(aa,ba,ka,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("bin1_aa_ooQ")(ja,la,Q)  = t1.at("aa")(ca,ja) * chol.at("aa_ovQ")(la,ca,Q) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("bin1_aa_ooQ")(ja,la,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0313_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2_1p.at("aaaa")(aa,ba,ka,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("0313_aaaa_vvoo")(aa,ba,ja,ia) += t2.at("aaaa")(aa,ba,ia,la) * tmps.at("0312_aa_oo")(ja,la) )
    ( tmps.at("0313_aaaa_vvoo")(aa,ba,ja,ia) += t2_1p.at("aaaa")(aa,ba,ia,la) * tmps.at("0309_aa_oo")(ja,la) )
    .deallocate(tmps.at("0265_aa_oo"))
    
    // r2_1p[aaaa] += -1.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_1p_aa(d,i) t2_aaaa(a,b,j,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_1p_aaaa(a,b,j,l) 
    //               += +0.500 P(i,j) <l,k||d,c>_aaaa t2_aaaa(a,b,i,k) t2_1p_aaaa(d,c,j,l) 
    //               += -1.000 P(i,j) <k,l||i,c>_aaaa t1_1p_aa(c,k) t2_aaaa(a,b,j,l) 
    //               += -0.500 P(i,j) <l,k||d,c>_aaaa t2_1p_aaaa(a,b,i,l) t2_aaaa(d,c,j,k) 
    //               += +0.500 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_1p_aaaa(a,b,k,l) 
    //               += +0.500 P(i,j) <k,l||i,c>_aaaa t1_1p_aa(c,j) t2_aaaa(a,b,k,l) 
    //               += +0.500 P(i,j) <k,l||i,c>_aaaa t1_1p_aa(c,j) t2_aaaa(a,b,k,l) 
    //               += +0.500 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_1p_aaaa(a,b,k,l) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0313_aaaa_vvoo")(aa,ba,ia,ja) )
    
    // r2_1p[aaaa] += -1.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_1p_aa(d,i) t2_aaaa(a,b,j,l) 
    //               += -1.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_1p_aaaa(a,b,j,l) 
    //               += +0.500 P(i,j) <l,k||d,c>_aaaa t2_aaaa(a,b,i,k) t2_1p_aaaa(d,c,j,l) 
    //               += -1.000 P(i,j) <k,l||i,c>_aaaa t1_1p_aa(c,k) t2_aaaa(a,b,j,l) 
    //               += -0.500 P(i,j) <l,k||d,c>_aaaa t2_1p_aaaa(a,b,i,l) t2_aaaa(d,c,j,k) 
    //               += +0.500 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_1p_aaaa(a,b,k,l) 
    //               += +0.500 P(i,j) <k,l||i,c>_aaaa t1_1p_aa(c,j) t2_aaaa(a,b,k,l) 
    //               += +0.500 P(i,j) <k,l||i,c>_aaaa t1_1p_aa(c,j) t2_aaaa(a,b,k,l) 
    //               += +0.500 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_1p_aaaa(a,b,k,l) 
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) += tmps.at("0313_aaaa_vvoo")(aa,ba,ja,ia) )
    .deallocate(tmps.at("0313_aaaa_vvoo"))
    .allocate(tmps.at("0314_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0314_bb_oo")(jb,ib)  = f.at("bb_ov")(jb,bb) * t1_2p.at("bb")(bb,ib) )
    
    // r1_2p[bb] += -2.000 f_bb(j,b) t1_bb(a,j) t1_2p_bb(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0314_bb_oo")(jb,ib) * t1.at("bb")(ab,jb) )
    
    // r2_2p[abab] += -2.000 f_bb(k,c) t1_2p_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0314_bb_oo")(kb,jb) * t2.at("abab")(aa,bb,ia,kb) )
    .allocate(tmps.at("0315_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0315_bb_oo")(jb,ib)  = f.at("bb_ov")(jb,bb) * t1.at("bb")(bb,ib) )
    
    // r1[bb] += -1.000 f_bb(j,b) t1_bb(a,j) t1_bb(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("bb")(ab,ib) -= tmps.at("0315_bb_oo")(jb,ib) * t1.at("bb")(ab,jb) )
    
    // r1_1p[bb] += -1.000 f_bb(j,b) t1_1p_bb(a,j) t1_bb(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0315_bb_oo")(jb,ib) * t1_1p.at("bb")(ab,jb) )
    
    // r1_2p[bb] += -2.000 f_bb(j,b) t1_2p_bb(a,j) t1_bb(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0315_bb_oo")(jb,ib) * t1_2p.at("bb")(ab,jb) )
    
    // r2[abab] += -1.000 f_bb(k,c) t1_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0315_bb_oo")(kb,jb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_1p[abab] += -1.000 f_bb(k,c) t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0315_bb_oo")(kb,jb) * t2_1p.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p[abab] += -2.000 f_bb(k,c) t1_bb(c,j) t2_2p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0315_bb_oo")(kb,jb) * t2_2p.at("abab")(aa,bb,ia,kb) )
    .allocate(tmps.at("0316_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0316_bb_oo")(jb,ib)  = f.at("bb_ov")(jb,bb) * t1_1p.at("bb")(bb,ib) )
    
    // r1_1p[bb] += -1.000 f_bb(j,b) t1_bb(a,j) t1_1p_bb(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0316_bb_oo")(jb,ib) * t1.at("bb")(ab,jb) )
    
    // r1_2p[bb] += -2.000 f_bb(j,b) t1_1p_bb(a,j) t1_1p_bb(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0316_bb_oo")(jb,ib) * t1_1p.at("bb")(ab,jb) )
    
    // r2_1p[abab] += -1.000 f_bb(k,c) t1_1p_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0316_bb_oo")(kb,jb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p[abab] += -2.000 f_bb(k,c) t1_1p_bb(c,j) t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0316_bb_oo")(kb,jb) * t2_1p.at("abab")(aa,bb,ia,kb) )
    .allocate(tmps.at("0320_bb_oo"))
    
    // flops: o2v0  = o3v1 o2v0Q1 o2v0 o2v0Q1 o2v0
    //  mems: o2v0  = o2v0 o2v0 o2v0 o2v0 o2v0
    ( tmps.at("0320_bb_oo")(lb,ib)  = tmps.at("0064_bbbb_ooov")(lb,ib,kb,cb) * t1.at("bb")(cb,kb) )
    ( tmps.at("0320_bb_oo")(lb,ib) += tmps.at("0026_bb_ooQ")(lb,ib,Q) * tmps.at("0148_Q")(Q) )
    ( tmps.at("0320_bb_oo")(lb,ib) += tmps.at("0026_bb_ooQ")(lb,ib,Q) * tmps.at("0049_Q")(Q) )
    .allocate(tmps.at("0319_bb_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0319_bb_oo")(jb,ib)  = tmps.at("0026_bb_ooQ")(jb,ib,Q) * tmps.at("0199_Q")(Q) )
    .allocate(tmps.at("0318_bb_oo"))
    
    // flops: o2v0  = o2v0Q1
    //  mems: o2v0  = o2v0
    ( tmps.at("0318_bb_oo")(jb,ib)  = tmps.at("0026_bb_ooQ")(jb,ib,Q) * tmps.at("0150_Q")(Q) )
    .allocate(tmps.at("0317_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0317_bb_oo")(kb,ib)  = t1.at("aa")(ba,ja) * tmps.at("0211_aabb_ovoo")(ja,ba,kb,ib) )
    .allocate(tmps.at("0321_bbbb_vvoo"))
    
    // flops: o2v2  = o3v1 o2v0 o2v0Q1 o2v0 o3v2 o3v2 o3v2 o2v2 o3v2 o3v2 o3v2 o4v2 o3v2 o2v2 o3v2 o3v2 o2v2 o2v1 o3v2 o3v2 o4v2 o3v2 o2v2 o2v2 o2v2 o3v2 o2v2 o3v2 o3v2 o4v2 o3v2 o2v2 o3v2 o4v2 o2v1Q1 o3v2 o2v1Q1 o3v2 o2v2 o2v1Q1 o3v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v2 o2v1Q1 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v2 o2v1Q1 o2v1Q1 o4v0Q1 o4v2 o2v2 o3v2 o2v2 o3v2 o3v2 o2v2 o2v2 o1v1Q1 o2v1 o3v2 o2v2 o3v2 o4v2 o3v2 o2v2 o2v2 o2v2 o2v2 o2v1 o3v2 o2v2 o3v1 o2v0 o2v0Q1 o2v0 o3v2 o2v2 o3v1 o3v2 o2v2 o3v2 o2v2 o2v1 o3v2 o2v2 o3v1 o3v2 o2v2 o3v1 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v0 o2v0 o2v0 o2v0 o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o3v1 o3v1 o2v2 o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o3v1 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o3v1 o3v1 o2v2 o2v2 o2v2 o2v2 o2v0 o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v2 o2v0Q1 o2v0Q1 o4v0 o2v2 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v2 o2v0Q1 o2v0Q1 o4v0 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o1v1 o2v0 o2v2 o2v2 o3v1 o3v1 o2v2 o2v2 o2v2 o2v2 o2v2 o2v0 o2v2 o2v2 o2v0 o2v0 o2v0 o2v0 o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bb_oo")(ib,lb)  = t1_1p.at("bb")(cb,kb) * tmps.at("0064_bbbb_ooov")(lb,ib,kb,cb) )
    ( tmps.at("bin1_bb_oo")(ib,lb) += tmps.at("0318_bb_oo")(lb,ib) )
    ( tmps.at("bin1_bb_oo")(ib,lb) += tmps.at("0026_bb_ooQ")(lb,ib,Q) * tmps.at("0151_Q")(Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb)  = tmps.at("bin1_bb_oo")(ib,lb) * t2_1p.at("bbbb")(ab,bb,jb,lb) )
    ( tmps.at("bin1_bb_oo")(jb,lb)  = t2.at("bbbb")(db,cb,jb,kb) * tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += 0.500 * tmps.at("bin1_bb_oo")(jb,lb) * t2_2p.at("bbbb")(ab,bb,ib,lb) )
    ( tmps.at("bin1_bb_oo")(ib,kb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t2_2p.at("bbbb")(db,cb,ib,lb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += 0.500 * tmps.at("bin1_bb_oo")(ib,kb) * t2.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("bin2_bbbb_vooo")(db,ib,kb,lb)  = t1_1p.at("bb")(cb,ib) * tmps.at("0075_bbbb_ovov")(kb,cb,lb,db) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t2_1p.at("bbbb")(db,bb,jb,lb) * tmps.at("bin2_bbbb_vooo")(db,ib,kb,lb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1.at("bb")(ab,kb) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = t2_1p.at("bbbb")(db,cb,ib,kb) * tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += 0.500 * t2_1p.at("bbbb")(ab,bb,jb,lb) * tmps.at("bin1_bb_oo")(ib,lb) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = tmps.at("0208_bb_ov")(lb,db) * t1_1p.at("bb")(db,ib) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2.at("bbbb")(ab,bb,jb,lb) * tmps.at("bin1_bb_oo")(ib,lb) )
    ( tmps.at("bin2_bbbb_vooo")(db,ib,kb,lb)  = t1_2p.at("bb")(cb,ib) * tmps.at("0075_bbbb_ovov")(kb,cb,lb,db) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t2.at("bbbb")(db,bb,jb,lb) * tmps.at("bin2_bbbb_vooo")(db,ib,kb,lb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1.at("bb")(ab,kb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += f.at("bb_oo")(kb,ib) * t2_2p.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("0316_bb_oo")(kb,ib) * t2_1p.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("bin2_bbbb_vooo")(db,ib,kb,lb)  = t1.at("bb")(cb,ib) * tmps.at("0075_bbbb_ovov")(kb,cb,lb,db) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = t2_1p.at("bbbb")(db,bb,jb,lb) * tmps.at("bin2_bbbb_vooo")(db,ib,kb,lb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) * t1_1p.at("bb")(ab,kb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2.at("bbbb")(ab,bb,jb,kb) * tmps.at("0314_bb_oo")(kb,ib) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += 0.500 * t2_2p.at("bbbb")(ab,bb,kb,lb) * tmps.at("0192_bbbb_oooo")(kb,ib,lb,jb) )
    ( tmps.at("bin1_bb_oo")(ib,kb)  = chol.at("bb_ovQ")(kb,cb,Q) * tmps.at("0186_bb_voQ")(cb,ib,Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_oo")(ib,kb) * t2.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = chol.at("bb_ovQ")(lb,cb,Q) * tmps.at("0058_bb_voQ")(cb,ib,Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_oo")(ib,lb) * t2_2p.at("bbbb")(ab,bb,jb,lb) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = chol.at("bb_ovQ")(lb,cb,Q) * tmps.at("0135_bb_voQ")(cb,ib,Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_oo")(ib,lb) * t2_1p.at("bbbb")(ab,bb,jb,lb) )
    ( tmps.at("bin1_bb_ooQ")(ib,kb,Q)  = t1.at("bb")(cb,ib) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = 0.500 * tmps.at("bin1_bb_ooQ")(ib,kb,Q) * tmps.at("0187_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("bin1_bb_ooQ")(ib,kb,Q)  = t1_1p.at("bb")(cb,ib) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("bin2_bb_ooQ")(jb,lb,Q)  = t1_1p.at("bb")(db,jb) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = 0.500 * tmps.at("bin1_bb_ooQ")(ib,kb,Q) * tmps.at("bin2_bb_ooQ")(jb,lb,Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("bin1_bb_ooQ")(ib,kb,Q)  = t1_2p.at("bb")(db,ib) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = 0.500 * tmps.at("bin1_bb_ooQ")(ib,kb,Q) * tmps.at("0026_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("bin1_bb_ooQ")(ib,kb,Q)  = t1.at("bb")(cb,ib) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = 0.500 * tmps.at("bin1_bb_ooQ")(ib,kb,Q) * tmps.at("0029_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2_1p.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("bin1_bb_ooQ")(ib,kb,Q)  = t1_1p.at("bb")(db,ib) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = 0.500 * tmps.at("bin1_bb_ooQ")(ib,kb,Q) * tmps.at("0026_bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2_1p.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("bin1_bb_ooQ")(ib,kb,Q)  = t1.at("bb")(cb,ib) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("bin2_bb_ooQ")(jb,lb,Q)  = t1.at("bb")(db,jb) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = 0.500 * tmps.at("bin1_bb_ooQ")(ib,kb,Q) * tmps.at("bin2_bb_ooQ")(jb,lb,Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2_2p.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += 3.000 * tmps.at("0039_bb_oo")(kb,ib) * t2_2p.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("0315_bb_oo")(kb,ib) * t2_2p.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += 3.000 * tmps.at("0041_bb_oo")(kb,ib) * t2_1p.at("bbbb")(ab,bb,jb,kb) )
    ( tmps.at("bin1_bb_vo")(db,lb)  = chol.at("bb_ovQ")(lb,db,Q) * tmps.at("0150_Q")(Q) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = tmps.at("bin1_bb_vo")(db,lb) * t1_1p.at("bb")(db,ib) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_oo")(ib,lb) * t2.at("bbbb")(ab,bb,jb,lb) )
    ( tmps.at("bin2_bbbb_vooo")(db,ib,kb,lb)  = tmps.at("0075_bbbb_ovov")(kb,cb,lb,db) * t1.at("bb")(cb,ib) )
    ( tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb)  = tmps.at("bin2_bbbb_vooo")(db,ib,kb,lb) * t2_2p.at("bbbb")(db,bb,jb,lb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t1.at("bb")(ab,kb) * tmps.at("bin1_bbbb_vooo")(bb,ib,jb,kb) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = tmps.at("0202_bb_ov")(lb,db) * t1_1p.at("bb")(db,ib) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2_1p.at("bbbb")(ab,bb,jb,lb) * tmps.at("bin1_bb_oo")(ib,lb) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = t1_2p.at("bb")(cb,kb) * tmps.at("0064_bbbb_ooov")(lb,ib,kb,cb) )
    ( tmps.at("bin1_bb_oo")(ib,lb) += tmps.at("0319_bb_oo")(lb,ib) )
    ( tmps.at("bin1_bb_oo")(ib,lb) += tmps.at("0026_bb_ooQ")(lb,ib,Q) * tmps.at("0174_Q")(Q) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_oo")(ib,lb) * t2.at("bbbb")(ab,bb,jb,lb) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = t1_1p.at("aa")(ca,ka) * tmps.at("0211_aabb_ovoo")(ka,ca,lb,ib) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_oo")(ib,lb) * t2_1p.at("bbbb")(ab,bb,jb,lb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("0320_bb_oo")(lb,ib) * t2_2p.at("bbbb")(ab,bb,jb,lb) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = tmps.at("0202_bb_ov")(lb,db) * t1_2p.at("bb")(db,ib) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2.at("bbbb")(ab,bb,jb,lb) * tmps.at("bin1_bb_oo")(ib,lb) )
    ( tmps.at("bin1_bb_oo")(jb,lb)  = t1_1p.at("bb")(cb,kb) * tmps.at("0229_bbbb_ovoo")(lb,cb,jb,kb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += tmps.at("bin1_bb_oo")(jb,lb) * t2.at("bbbb")(ab,bb,ib,lb) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = tmps.at("0211_aabb_ovoo")(ka,ca,lb,ib) * t1_2p.at("aa")(ca,ka) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2.at("bbbb")(ab,bb,jb,lb) * tmps.at("bin1_bb_oo")(ib,lb) )
    ( tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) += t2_2p.at("bbbb")(ab,bb,jb,lb) * tmps.at("0317_bb_oo")(lb,ib) )
    .deallocate(tmps.at("0314_bb_oo"))
    .deallocate(tmps.at("0229_bbbb_ovoo"))
    .deallocate(tmps.at("0208_bb_ov"))
    .deallocate(tmps.at("0041_bb_oo"))
    .deallocate(tmps.at("0039_bb_oo"))
    
    // r2_2p[bbbb] += +2.000 P(i,j) <k,l||c,d>_bbbb t1_1p_bb(c,k) t1_1p_bb(d,i) t2_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <l,k||i,c>_bbbb t1_1p_bb(c,k) t2_1p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,i) t1_1p_bb(d,k) t2_1p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||d,c>_abab t1_bb(c,i) t1_1p_aa(d,k) t2_1p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <k,l||i,c>_bbbb t1_2p_bb(c,k) t2_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_2p_bb(d,k) t2_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||d,c>_abab t1_bb(c,i) t1_2p_aa(d,k) t2_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <l,k||i,c>_bbbb t1_bb(c,k) t2_2p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,i) t2_2p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,d>_abab t1_aa(c,k) t1_bb(d,i) t2_2p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_bb(c,i) t2_2p_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_2p_bb(c,i) t2_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,k) t1_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +1.000 P(i,j) <l,k||d,c>_bbbb t2_bbbb(a,b,i,k) t2_2p_bbbb(d,c,j,l) 
    //               += -1.000 P(i,j) <l,k||d,c>_bbbb t2_2p_bbbb(a,b,i,l) t2_bbbb(d,c,j,k) 
    //               += -1.000 P(i,j) <l,k||d,c>_abab t2_bbbb(a,b,i,k) t2_2p_abab(d,c,l,j) 
    //               += -1.000 P(i,j) <l,k||c,d>_abab t2_bbbb(a,b,i,k) t2_2p_abab(c,d,l,j) 
    //               += -1.000 P(i,j) <k,l||d,c>_abab t2_2p_bbbb(a,b,i,l) t2_abab(d,c,k,j) 
    //               += -1.000 P(i,j) <k,l||c,d>_abab t2_2p_bbbb(a,b,i,l) t2_abab(c,d,k,j) 
    //               += -1.000 P(i,j) <k,l||d,c>_abab t2_1p_bbbb(a,b,i,l) t2_1p_abab(d,c,k,j) 
    //               += -1.000 P(i,j) <k,l||c,d>_abab t2_1p_bbbb(a,b,i,l) t2_1p_abab(c,d,k,j) 
    //               += +2.000 P(i,j) f_bb(k,i) t2_2p_bbbb(a,b,j,k) 
    //               += +1.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_2p_bb(d,j) t2_bbbb(a,b,k,l) 
    //               += -1.000 <k,l||d,c>_bbbb t1_1p_bb(c,i) t1_1p_bb(d,j) t2_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_2p_bb(d,j) t2_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_1p_bb(d,j) t2_1p_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_1p_bb(d,j) t2_1p_bbbb(a,b,k,l) 
    //               += -1.000 <k,l||d,c>_bbbb t1_bb(c,i) t1_bb(d,j) t2_2p_bbbb(a,b,k,l) 
    //               += +1.000 <k,l||i,j>_bbbb t2_2p_bbbb(a,b,k,l) 
    //               += +6.000 P(i,j) d-_bb(k,c) t1_1p_bb(c,i) t2_2p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) f_bb(k,c) t1_bb(c,i) t2_2p_bbbb(a,b,j,k) 
    //               += +6.000 P(i,j) d-_bb(k,c) t1_2p_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) f_bb(k,c) t1_1p_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) f_bb(k,c) t1_2p_bb(c,i) t2_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) <k,l||c,d>_abab t1_1p_aa(c,k) t1_1p_bb(d,i) t2_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,d>_bbbb t1_1p_bb(c,k) t1_1p_bb(d,i) t2_bbbb(a,b,j,l) 
    //               += -1.000 P(i,j) <l,k||d,c>_bbbb t2_1p_bbbb(a,b,i,l) t2_1p_bbbb(d,c,j,k) 
    //               += +2.000 P(i,j) <k,l||c,i>_abab t1_aa(c,k) t2_2p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,i>_abab t1_1p_aa(c,k) t2_1p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,i>_abab t1_2p_aa(c,k) t2_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_1p_bb(d,i) t2_1p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,d>_abab t1_aa(c,k) t1_1p_bb(d,i) t2_1p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_2p_bb(d,i) t2_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,d>_abab t1_aa(c,k) t1_2p_bb(d,i) t2_bbbb(a,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0321_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) <k,l||c,d>_bbbb t1_1p_bb(c,k) t1_1p_bb(d,i) t2_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <l,k||i,c>_bbbb t1_1p_bb(c,k) t2_1p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,i) t1_1p_bb(d,k) t2_1p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||d,c>_abab t1_bb(c,i) t1_1p_aa(d,k) t2_1p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <k,l||i,c>_bbbb t1_2p_bb(c,k) t2_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_2p_bb(d,k) t2_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||d,c>_abab t1_bb(c,i) t1_2p_aa(d,k) t2_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <l,k||i,c>_bbbb t1_bb(c,k) t2_2p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,i) t2_2p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,d>_abab t1_aa(c,k) t1_bb(d,i) t2_2p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_bb(c,i) t2_2p_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_bb(a,k) t1_1p_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +2.000 P(i,j) P(a,b) <l,k||d,c>_bbbb t1_bb(a,k) t1_2p_bb(c,i) t2_bbbb(d,b,j,l) 
    //               += -2.000 P(i,j) P(a,b) <l,k||c,d>_bbbb t1_1p_bb(a,k) t1_bb(c,i) t2_1p_bbbb(d,b,j,l) 
    //               += +1.000 P(i,j) <l,k||d,c>_bbbb t2_bbbb(a,b,i,k) t2_2p_bbbb(d,c,j,l) 
    //               += -1.000 P(i,j) <l,k||d,c>_bbbb t2_2p_bbbb(a,b,i,l) t2_bbbb(d,c,j,k) 
    //               += -1.000 P(i,j) <l,k||d,c>_abab t2_bbbb(a,b,i,k) t2_2p_abab(d,c,l,j) 
    //               += -1.000 P(i,j) <l,k||c,d>_abab t2_bbbb(a,b,i,k) t2_2p_abab(c,d,l,j) 
    //               += -1.000 P(i,j) <k,l||d,c>_abab t2_2p_bbbb(a,b,i,l) t2_abab(d,c,k,j) 
    //               += -1.000 P(i,j) <k,l||c,d>_abab t2_2p_bbbb(a,b,i,l) t2_abab(c,d,k,j) 
    //               += -1.000 P(i,j) <k,l||d,c>_abab t2_1p_bbbb(a,b,i,l) t2_1p_abab(d,c,k,j) 
    //               += -1.000 P(i,j) <k,l||c,d>_abab t2_1p_bbbb(a,b,i,l) t2_1p_abab(c,d,k,j) 
    //               += +2.000 P(i,j) f_bb(k,i) t2_2p_bbbb(a,b,j,k) 
    //               += +1.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_2p_bb(d,j) t2_bbbb(a,b,k,l) 
    //               += -1.000 <k,l||d,c>_bbbb t1_1p_bb(c,i) t1_1p_bb(d,j) t2_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_2p_bb(d,j) t2_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_1p_bb(d,j) t2_1p_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_1p_bb(d,j) t2_1p_bbbb(a,b,k,l) 
    //               += -1.000 <k,l||d,c>_bbbb t1_bb(c,i) t1_bb(d,j) t2_2p_bbbb(a,b,k,l) 
    //               += +1.000 <k,l||i,j>_bbbb t2_2p_bbbb(a,b,k,l) 
    //               += +6.000 P(i,j) d-_bb(k,c) t1_1p_bb(c,i) t2_2p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) f_bb(k,c) t1_bb(c,i) t2_2p_bbbb(a,b,j,k) 
    //               += +6.000 P(i,j) d-_bb(k,c) t1_2p_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) f_bb(k,c) t1_1p_bb(c,i) t2_1p_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) f_bb(k,c) t1_2p_bb(c,i) t2_bbbb(a,b,j,k) 
    //               += +2.000 P(i,j) <k,l||c,d>_abab t1_1p_aa(c,k) t1_1p_bb(d,i) t2_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,d>_bbbb t1_1p_bb(c,k) t1_1p_bb(d,i) t2_bbbb(a,b,j,l) 
    //               += -1.000 P(i,j) <l,k||d,c>_bbbb t2_1p_bbbb(a,b,i,l) t2_1p_bbbb(d,c,j,k) 
    //               += +2.000 P(i,j) <k,l||c,i>_abab t1_aa(c,k) t2_2p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,i>_abab t1_1p_aa(c,k) t2_1p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,i>_abab t1_2p_aa(c,k) t2_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_1p_bb(d,i) t2_1p_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,d>_abab t1_aa(c,k) t1_1p_bb(d,i) t2_1p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_2p_bb(d,i) t2_bbbb(a,b,j,l) 
    //               += +2.000 P(i,j) <k,l||c,d>_abab t1_aa(c,k) t1_2p_bb(d,i) t2_bbbb(a,b,j,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0321_bbbb_vvoo")(ab,bb,jb,ib) )
    .deallocate(tmps.at("0321_bbbb_vvoo"))
    .allocate(tmps.at("0322_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0322_bb_oo")(ib,lb)  = tmps.at("0197_bbbb_ooov")(kb,ib,lb,cb) * t1.at("bb")(cb,kb) )
    
    // r2[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,j) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0322_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,j) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0322_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,j) t2_2p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0322_bb_oo")(jb,lb) * t2_2p.at("abab")(aa,bb,ia,lb) )
    .allocate(tmps.at("0323_bbbb_vvoo"))
    
    // flops: o2v2  = o3v2 o3v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o4v0Q1 o4v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v0 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o4v0 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = t2.at("bbbb")(db,cb,jb,kb) * tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) )
    ( tmps.at("0323_bbbb_vvoo")(ab,bb,jb,ib)  = 0.500 * t2.at("bbbb")(ab,bb,ib,lb) * tmps.at("bin1_bb_oo")(jb,lb) )
    ( tmps.at("bin1_bb_ooQ")(jb,lb,Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t1.at("bb")(cb,jb) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = chol.at("bb_ooQ")(kb,ib,Q) * tmps.at("bin1_bb_ooQ")(jb,lb,Q) )
    ( tmps.at("0323_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) * t2.at("bbbb")(ab,bb,kb,lb) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = tmps.at("0026_bb_ooQ")(kb,ib,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0323_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) * t2.at("bbbb")(ab,bb,kb,lb) )
    ( tmps.at("0323_bbbb_vvoo")(ab,bb,jb,ib) += t2.at("bbbb")(ab,bb,ib,lb) * tmps.at("0322_bb_oo")(jb,lb) )
    
    // r2[bbbb] += -1.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,i) t2_bbbb(a,b,j,l) 
    //            += +0.500 P(i,j) <k,l||i,c>_bbbb t1_bb(c,j) t2_bbbb(a,b,k,l) 
    //            += +0.500 P(i,j) <k,l||i,c>_bbbb t1_bb(c,j) t2_bbbb(a,b,k,l) 
    //            += -0.500 P(i,j) <l,k||d,c>_bbbb t2_bbbb(a,b,i,l) t2_bbbb(d,c,j,k) 
    ( r2.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0323_bbbb_vvoo")(ab,bb,ib,jb) )
    
    // r2[bbbb] += -1.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,i) t2_bbbb(a,b,j,l) 
    //            += +0.500 P(i,j) <k,l||i,c>_bbbb t1_bb(c,j) t2_bbbb(a,b,k,l) 
    //            += +0.500 P(i,j) <k,l||i,c>_bbbb t1_bb(c,j) t2_bbbb(a,b,k,l) 
    //            += -0.500 P(i,j) <l,k||d,c>_bbbb t2_bbbb(a,b,i,l) t2_bbbb(d,c,j,k) 
    ( r2.at("bbbb")(ab,bb,ib,jb) += tmps.at("0323_bbbb_vvoo")(ab,bb,jb,ib) )
    .deallocate(tmps.at("0323_bbbb_vvoo"))
    .allocate(tmps.at("0324_aa_vo"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0324_aa_vo")(ca,ka)  = tmps.at("0073_aaaa_ovov")(ja,ca,ka,ba) * t1.at("aa")(ba,ja) )
    
    // r1_1p[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0324_aa_vo")(ca,ka) * t2_1p.at("aaaa")(ca,aa,ia,ka) )
    
    // r1_1p[bb] += -1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_1p_abab(c,a,k,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) -= tmps.at("0324_aa_vo")(ca,ka) * t2_1p.at("abab")(ca,ab,ka,ib) )
    
    // r1_2p[aa] += +2.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_2p_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0324_aa_vo")(ca,ka) * t2_2p.at("aaaa")(ca,aa,ia,ka) )
    
    // r1_2p[bb] += -2.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_2p_abab(c,a,k,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) -= 2.000 * tmps.at("0324_aa_vo")(ca,ka) * t2_2p.at("abab")(ca,ab,ka,ib) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(c,k) t1_2p_aa(d,i) t2_abab(a,b,l,j) 
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_aa_oo")(ia,la)  = tmps.at("0324_aa_vo")(da,la) * t1_2p.at("aa")(da,ia) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    .allocate(tmps.at("0325_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0325_aa_oo")(ia,ja)  = tmps.at("0056_aaaa_ooov")(ka,ia,ja,ba) * t1_2p.at("aa")(ba,ka) )
    
    // r1_2p[aa] += +2.000 <k,j||i,b>_aaaa t1_aa(a,j) t1_2p_aa(b,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * tmps.at("0325_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_aaaa t1_2p_aa(c,k) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0325_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    .allocate(tmps.at("0326_aa_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0326_aa_oo")(ia,ja)  = tmps.at("0164_aaaa_ooov")(ka,ia,ja,ca) * t1_1p.at("aa")(ca,ka) )
    .deallocate(tmps.at("0164_aaaa_ooov"))
    
    // r1_1p[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(a,j) t1_aa(b,i) t1_1p_aa(c,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0326_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    
    // r2_1p[aaaa] += -1.000 P(i,j) <k,l||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,k) t2_aaaa(a,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("aaaa")(aa,ba,ia,ja) -= tmps.at("0326_aa_oo")(ia,la) * t2.at("aaaa")(aa,ba,ja,la) )
    
    // r2_1p[abab] += +1.000 <k,l||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,k) t2_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0326_aa_oo")(ia,la) * t2.at("abab")(aa,bb,la,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,k) t2_1p_abab(a,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0326_aa_oo")(ia,la) * t2_1p.at("abab")(aa,bb,la,jb) )
    .allocate(tmps.at("0327_aaaa_vvoo"))
    
    // flops: o2v2  = o2v2 o2v1 o3v2 o3v3 o3v3 o2v1 o3v2 o2v2 o3v2 o2v2 o2v2 o3v2 o3v2 o2v2 o3v3 o2v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o3v2 o3v2 o2v2 o4v0Q1 o4v2 o2v2 o4v0Q1 o4v2 o2v2 o3v2 o3v2 o2v2 o4v0Q1 o4v2 o2v2 o3v2 o2v2 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o1v1 o2v0 o2v2 o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v0 o2v2 o2v2 o4v0 o2v2 o2v2 o4v0 o2v2 o2v2 o2v0 o2v2 o2v2 o4v0 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("bin1_aa_vo")(da,la)  = t1_1p.at("aa")(ca,ka) * tmps.at("0073_aaaa_ovov")(la,ca,ka,da) )
    ( tmps.at("bin1_aa_oo")(ja,la)  = tmps.at("bin1_aa_vo")(da,la) * t1_1p.at("aa")(da,ja) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia)  = tmps.at("bin1_aa_oo")(ja,la) * t2.at("aaaa")(aa,ba,ia,la) )
    ( tmps.at("bin1_aaaa_vvoo")(aa,da,ja,la)  = tmps.at("0073_aaaa_ovov")(la,ca,ka,da) * t2_1p.at("aaaa")(ca,aa,ja,ka) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += tmps.at("bin1_aaaa_vvoo")(aa,da,ja,la) * t2_1p.at("aaaa")(da,ba,ia,la) )
    ( tmps.at("bin1_aa_oo")(ja,la)  = t1_2p.at("aa")(da,ja) * tmps.at("0324_aa_vo")(da,la) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += t2.at("aaaa")(aa,ba,ia,la) * tmps.at("bin1_aa_oo")(ja,la) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += t2.at("aaaa")(aa,ba,ia,la) * tmps.at("0325_aa_oo")(ja,la) )
    ( tmps.at("bin1_aa_oo")(ia,la)  = tmps.at("0073_aaaa_ovov")(la,da,ka,ca) * t2.at("aaaa")(da,ca,ia,ka) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2_2p.at("aaaa")(aa,ba,ja,la) * tmps.at("bin1_aa_oo")(ia,la) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += t2_1p.at("abab")(aa,cb,ja,kb) * tmps.at("0217_abab_vvoo")(ba,cb,ia,kb) )
    ( tmps.at("bin1_aa_ooQ")(ja,la,Q)  = t1_1p.at("aa")(ca,ja) * chol.at("aa_ovQ")(la,ca,Q) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("bin1_aa_ooQ")(ja,la,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2_1p.at("aaaa")(aa,ba,ka,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("bin1_aa_ooQ")(ja,la,Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1.at("aa")(ca,ja) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("bin1_aa_ooQ")(ja,la,Q) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) * t2_2p.at("aaaa")(aa,ba,ka,la) )
    ( tmps.at("bin1_aa_ooQ")(ja,la,Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1_2p.at("aa")(ca,ja) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = chol.at("aa_ooQ")(ka,ia,Q) * tmps.at("bin1_aa_ooQ")(ja,la,Q) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) * t2.at("aaaa")(aa,ba,ka,la) )
    ( tmps.at("bin1_aa_oo")(ja,ka)  = t2_2p.at("aaaa")(da,ca,ja,la) * tmps.at("0073_aaaa_ovov")(la,da,ka,ca) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2.at("aaaa")(aa,ba,ia,ka) * tmps.at("bin1_aa_oo")(ja,ka) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = chol.at("aa_ooQ")(la,ja,Q) * tmps.at("0137_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2_1p.at("aaaa")(aa,ba,ka,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = chol.at("aa_ooQ")(la,ja,Q) * tmps.at("0221_aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * t2.at("aaaa")(aa,ba,ka,la) * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) )
    ( tmps.at("bin1_aa_oo")(ja,la)  = tmps.at("0073_aaaa_ovov")(la,ca,ka,da) * t2_1p.at("aaaa")(da,ca,ja,ka) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * tmps.at("bin1_aa_oo")(ja,la) * t2_1p.at("aaaa")(aa,ba,ia,la) )
    ( tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la)  = tmps.at("0053_aa_ooQ")(ka,ia,Q) * chol.at("aa_ooQ")(la,ja,Q) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += 0.500 * tmps.at("bin1_aaaa_oooo")(ia,ja,ka,la) * t2_2p.at("aaaa")(aa,ba,ka,la) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += t2_1p.at("aaaa")(aa,ba,ia,la) * tmps.at("0326_aa_oo")(ja,la) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += t2_1p.at("aaaa")(aa,ba,ia,la) * tmps.at("0312_aa_oo")(ja,la) )
    ( tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) += t2_2p.at("aaaa")(aa,ba,ia,la) * tmps.at("0309_aa_oo")(ja,la) )
    .deallocate(tmps.at("0326_aa_oo"))
    .deallocate(tmps.at("0325_aa_oo"))
    .deallocate(tmps.at("0324_aa_vo"))
    .deallocate(tmps.at("0312_aa_oo"))
    .deallocate(tmps.at("0309_aa_oo"))
    .deallocate(tmps.at("0217_abab_vvoo"))
    
    // r2_2p[aaaa] += +2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,k) t2_1p_aaaa(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_1p_aa(d,i) t2_1p_aaaa(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_2p_aaaa(a,b,j,l) 
    //               += +1.000 P(i,j) <l,k||d,c>_aaaa t2_aaaa(a,b,i,k) t2_2p_aaaa(d,c,j,l) 
    //               += -2.000 P(i,j) <k,l||i,c>_aaaa t1_2p_aa(c,k) t2_aaaa(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_2p_aa(d,i) t2_aaaa(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_1p_aaaa(c,a,i,k) t2_1p_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_1p_abab(a,c,i,k) t2_1p_abab(b,d,j,l) 
    //               += -1.000 P(i,j) <l,k||d,c>_aaaa t2_2p_aaaa(a,b,i,l) t2_aaaa(d,c,j,k) 
    //               += -1.000 P(i,j) <l,k||d,c>_aaaa t2_1p_aaaa(a,b,i,l) t2_1p_aaaa(d,c,j,k) 
    //               += +2.000 P(i,j) <k,l||c,d>_aaaa t1_1p_aa(c,k) t1_1p_aa(d,i) t2_aaaa(a,b,j,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_2p_aa(c,j) t2_aaaa(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_2p_aa(c,j) t2_aaaa(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_1p_aa(c,j) t2_1p_aaaa(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_1p_aa(c,j) t2_1p_aaaa(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_2p_aaaa(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_2p_aaaa(a,b,k,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) += 2.000 * tmps.at("0327_aaaa_vvoo")(aa,ba,ja,ia) )
    
    // r2_2p[aaaa] += +2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,i) t1_1p_aa(d,k) t2_1p_aaaa(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_1p_aa(d,i) t2_1p_aaaa(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_aa(d,i) t2_2p_aaaa(a,b,j,l) 
    //               += +1.000 P(i,j) <l,k||d,c>_aaaa t2_aaaa(a,b,i,k) t2_2p_aaaa(d,c,j,l) 
    //               += -2.000 P(i,j) <k,l||i,c>_aaaa t1_2p_aa(c,k) t2_aaaa(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t1_aa(c,k) t1_2p_aa(d,i) t2_aaaa(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_1p_aaaa(c,a,i,k) t2_1p_aaaa(d,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_1p_abab(a,c,i,k) t2_1p_abab(b,d,j,l) 
    //               += -1.000 P(i,j) <l,k||d,c>_aaaa t2_2p_aaaa(a,b,i,l) t2_aaaa(d,c,j,k) 
    //               += -1.000 P(i,j) <l,k||d,c>_aaaa t2_1p_aaaa(a,b,i,l) t2_1p_aaaa(d,c,j,k) 
    //               += +2.000 P(i,j) <k,l||c,d>_aaaa t1_1p_aa(c,k) t1_1p_aa(d,i) t2_aaaa(a,b,j,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_2p_aa(c,j) t2_aaaa(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_2p_aa(c,j) t2_aaaa(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_1p_aa(c,j) t2_1p_aaaa(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_1p_aa(c,j) t2_1p_aaaa(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_2p_aaaa(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_aaaa t1_aa(c,j) t2_2p_aaaa(a,b,k,l) 
    ( r2_2p.at("aaaa")(aa,ba,ia,ja) -= 2.000 * tmps.at("0327_aaaa_vvoo")(aa,ba,ia,ja) )
    .deallocate(tmps.at("0327_aaaa_vvoo"))
    .allocate(tmps.at("0328_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0328_bb_oo")(ib,jb)  = tmps.at("0064_bbbb_ooov")(kb,ib,jb,bb) * t1_2p.at("bb")(bb,kb) )
    
    // r1_2p[bb] += +2.000 <k,j||i,b>_bbbb t1_bb(a,j) t1_2p_bb(b,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0328_bb_oo")(ib,jb) * t1.at("bb")(ab,jb) )
    
    // r2_2p[abab] += +2.000 <k,l||j,c>_bbbb t1_2p_bb(c,k) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,bb,ia,lb) * tmps.at("0328_bb_oo")(jb,lb) )
    .allocate(tmps.at("0329_bb_oo"))
    
    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0329_bb_oo")(ib,jb)  = tmps.at("0197_bbbb_ooov")(kb,ib,jb,cb) * t1_1p.at("bb")(cb,kb) )
    .deallocate(tmps.at("0197_bbbb_ooov"))
    
    // r1_1p[bb] += +1.000 <k,j||b,c>_bbbb t1_bb(a,j) t1_bb(b,i) t1_1p_bb(c,k) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0329_bb_oo")(ib,jb) * t1.at("bb")(ab,jb) )
    
    // r2_1p[bbbb] += -1.000 P(i,j) <k,l||c,d>_bbbb t1_bb(c,i) t1_1p_bb(d,k) t2_bbbb(a,b,j,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("bbbb")(ab,bb,ib,jb) -= tmps.at("0329_bb_oo")(ib,lb) * t2.at("bbbb")(ab,bb,jb,lb) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_bbbb t1_bb(c,j) t1_1p_bb(d,k) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0329_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    .allocate(tmps.at("0330_bb_vo"))
    
    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    ( tmps.at("0330_bb_vo")(cb,kb)  = tmps.at("0075_bbbb_ovov")(jb,cb,kb,bb) * t1.at("bb")(bb,jb) )
    
    // r1_1p[aa] += -1.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0330_bb_vo")(cb,kb) * t2_1p.at("abab")(aa,cb,ia,kb) )
    
    // r1_1p[bb] += +1.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_1p_bbbb(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("bb")(ab,ib) += tmps.at("0330_bb_vo")(cb,kb) * t2_1p.at("bbbb")(cb,ab,ib,kb) )
    
    // r1_2p[aa] += -2.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_2p_abab(a,c,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0330_bb_vo")(cb,kb) * t2_2p.at("abab")(aa,cb,ia,kb) )
    
    // r1_2p[bb] += +2.000 <k,j||b,c>_bbbb t1_bb(b,j) t2_2p_bbbb(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("bb")(ab,ib) += 2.000 * tmps.at("0330_bb_vo")(cb,kb) * t2_2p.at("bbbb")(cb,ab,ib,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(c,k) t1_2p_bb(d,j) t2_abab(a,b,i,l) 
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,lb)  = tmps.at("0330_bb_vo")(db,lb) * t1_2p.at("bb")(db,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,bb,ia,lb) * tmps.at("bin1_bb_oo")(jb,lb) )
    .allocate(tmps.at("0331_bb_oo"))
    
    // flops: o2v0  = o2v2 o2v1
    //  mems: o2v0  = o1v1 o2v0
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("0075_bbbb_ovov")(kb,db,lb,cb) * t1.at("bb")(cb,kb) )
    ( tmps.at("0331_bb_oo")(ib,lb)  = t1_1p.at("bb")(db,ib) * tmps.at("bin1_bb_vo")(db,lb) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_bbbb t1_bb(c,k) t1_1p_bb(d,j) t2_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0331_bb_oo")(jb,lb) * t2.at("abab")(aa,bb,ia,lb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(c,k) t1_1p_bb(d,j) t2_1p_abab(a,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0331_bb_oo")(jb,lb) * t2_1p.at("abab")(aa,bb,ia,lb) )
    .allocate(tmps.at("0332_bbbb_vvoo"))
    
    // flops: o2v2  = o3v2 o2v1 o3v2 o2v2 o3v3 o3v2 o2v2 o2v2 o3v3 o3v3 o2v2 o3v2 o3v2 o2v2 o3v2 o3v2 o4v0Q1 o4v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o4v0Q1 o4v2 o2v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v2 o2v2 o3v2 o3v2 o2v1Q1 o4v0Q1 o4v2 o2v2 o2v2 o4v0Q1 o4v2 o2v2 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v0 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2 o2v0 o2v2 o2v2 o2v0 o2v2 o4v0 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o4v0 o2v2 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v2 o2v2 o2v0 o2v2 o2v0Q1 o4v0 o2v2 o2v2 o2v2 o4v0 o2v2 o2v2 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib)  = t2.at("bbbb")(ab,bb,ib,lb) * tmps.at("0328_bb_oo")(jb,lb) )
    ( tmps.at("bin1_bb_oo")(jb,lb)  = 2.000 * t1_2p.at("bb")(db,jb) * tmps.at("0330_bb_vo")(db,lb) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * t2.at("bbbb")(ab,bb,ib,lb) * tmps.at("bin1_bb_oo")(jb,lb) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += t2_1p.at("abab")(ca,ab,ka,jb) * tmps.at("0109_abab_vvoo")(ca,bb,ka,ib) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += t2_1p.at("bbbb")(ab,bb,ib,lb) * tmps.at("0331_bb_oo")(jb,lb) )
    ( tmps.at("bin1_bbbb_vvoo")(ab,db,jb,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t2_1p.at("bbbb")(cb,ab,jb,kb) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += tmps.at("bin1_bbbb_vvoo")(ab,db,jb,lb) * t2_1p.at("bbbb")(db,bb,ib,lb) )
    ( tmps.at("bin1_bb_oo")(ib,lb)  = tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) * t2.at("bbbb")(db,cb,ib,kb) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * t2_2p.at("bbbb")(ab,bb,jb,lb) * tmps.at("bin1_bb_oo")(ib,lb) )
    ( tmps.at("bin1_bb_oo")(jb,kb)  = t2_2p.at("bbbb")(db,cb,jb,lb) * tmps.at("0075_bbbb_ovov")(lb,db,kb,cb) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * t2.at("bbbb")(ab,bb,ib,kb) * tmps.at("bin1_bb_oo")(jb,kb) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0187_bb_ooQ")(kb,ib,Q) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * t2.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("bin1_bb_ooQ")(jb,lb,Q)  = t1_2p.at("bb")(cb,jb) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = tmps.at("bin1_bb_ooQ")(jb,lb,Q) * chol.at("bb_ooQ")(kb,ib,Q) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * t2.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0029_bb_ooQ")(kb,ib,Q) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * t2_1p.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("bin1_bb_ooQ")(jb,lb,Q)  = t1_1p.at("bb")(cb,jb) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = tmps.at("bin1_bb_ooQ")(jb,lb,Q) * chol.at("bb_ooQ")(kb,ib,Q) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * t2_1p.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("bin1_bb_oo")(jb,lb)  = tmps.at("0075_bbbb_ovov")(lb,cb,kb,db) * t2_1p.at("bbbb")(db,cb,jb,kb) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * tmps.at("bin1_bb_oo")(jb,lb) * t2_1p.at("bbbb")(ab,bb,ib,lb) )
    ( tmps.at("bin1_bb_ooQ")(jb,lb,Q)  = t1.at("bb")(cb,jb) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = tmps.at("bin1_bb_ooQ")(jb,lb,Q) * chol.at("bb_ooQ")(kb,ib,Q) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * t2_2p.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * tmps.at("0026_bb_ooQ")(kb,ib,Q) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += 0.500 * t2_2p.at("bbbb")(ab,bb,kb,lb) * tmps.at("bin1_bbbb_oooo")(ib,jb,kb,lb) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += t2_1p.at("bbbb")(ab,bb,ib,lb) * tmps.at("0329_bb_oo")(jb,lb) )
    ( tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) += t2_2p.at("bbbb")(ab,bb,ib,lb) * tmps.at("0322_bb_oo")(jb,lb) )
    .deallocate(tmps.at("0330_bb_vo"))
    .deallocate(tmps.at("0329_bb_oo"))
    .deallocate(tmps.at("0328_bb_oo"))
    .deallocate(tmps.at("0109_abab_vvoo"))
    
    // r2_2p[bbbb] += +2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,i) t1_1p_bb(d,k) t2_1p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,i) t2_2p_bbbb(a,b,j,l) 
    //               += +1.000 P(i,j) <l,k||d,c>_bbbb t2_bbbb(a,b,i,k) t2_2p_bbbb(d,c,j,l) 
    //               += -2.000 P(i,j) <k,l||i,c>_bbbb t1_2p_bb(c,k) t2_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_2p_bb(d,i) t2_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_1p_abab(c,a,k,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_1p_bb(d,i) t2_1p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_1p_bbbb(c,a,i,k) t2_1p_bbbb(d,b,j,l) 
    //               += -1.000 P(i,j) <l,k||d,c>_bbbb t2_2p_bbbb(a,b,i,l) t2_bbbb(d,c,j,k) 
    //               += -1.000 P(i,j) <l,k||d,c>_bbbb t2_1p_bbbb(a,b,i,l) t2_1p_bbbb(d,c,j,k) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_2p_bb(c,j) t2_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_2p_bb(c,j) t2_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_1p_bb(c,j) t2_1p_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_1p_bb(c,j) t2_1p_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_bb(c,j) t2_2p_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_bb(c,j) t2_2p_bbbb(a,b,k,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) += 2.000 * tmps.at("0332_bbbb_vvoo")(ab,bb,jb,ib) )
    
    // r2_2p[bbbb] += +2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,i) t1_1p_bb(d,k) t2_1p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_bb(d,i) t2_2p_bbbb(a,b,j,l) 
    //               += +1.000 P(i,j) <l,k||d,c>_bbbb t2_bbbb(a,b,i,k) t2_2p_bbbb(d,c,j,l) 
    //               += -2.000 P(i,j) <k,l||i,c>_bbbb t1_2p_bb(c,k) t2_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_2p_bb(d,i) t2_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_aaaa t2_1p_abab(c,a,k,i) t2_1p_abab(d,b,l,j) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t1_bb(c,k) t1_1p_bb(d,i) t2_1p_bbbb(a,b,j,l) 
    //               += -2.000 P(i,j) <l,k||c,d>_bbbb t2_1p_bbbb(c,a,i,k) t2_1p_bbbb(d,b,j,l) 
    //               += -1.000 P(i,j) <l,k||d,c>_bbbb t2_2p_bbbb(a,b,i,l) t2_bbbb(d,c,j,k) 
    //               += -1.000 P(i,j) <l,k||d,c>_bbbb t2_1p_bbbb(a,b,i,l) t2_1p_bbbb(d,c,j,k) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_2p_bb(c,j) t2_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_2p_bb(c,j) t2_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_1p_bb(c,j) t2_1p_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_1p_bb(c,j) t2_1p_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_bb(c,j) t2_2p_bbbb(a,b,k,l) 
    //               += +1.000 P(i,j) <k,l||i,c>_bbbb t1_bb(c,j) t2_2p_bbbb(a,b,k,l) 
    ( r2_2p.at("bbbb")(ab,bb,ib,jb) -= 2.000 * tmps.at("0332_bbbb_vvoo")(ab,bb,ib,jb) )
    .deallocate(tmps.at("0332_bbbb_vvoo"))
    .allocate(tmps.at("0333_bbbb_vvoo"))
    ;
  }
  // clang-format on
}

template void exachem::cc::cd_qed_ccsd_os::resid_part7<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);