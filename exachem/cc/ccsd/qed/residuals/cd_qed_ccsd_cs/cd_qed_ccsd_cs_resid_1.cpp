/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, cholattelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_qed_ccsd_cs_resid_1.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_cs::resid_part1(
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
    ( scalars.at("0004")()  = dp.at("bb_ov")(ib,ab) * t1_1p.at("bb")(ab,ib) )
    ( scalars.at("0005")()  = dp.at("aa_ov")(ia,aa) * t1_1p.at("aa")(aa,ia) )
    ( scalars.at("0006")()  = dp.at("bb_ov")(ib,ab) * t1_2p.at("bb")(ab,ib) )
    ( scalars.at("0007")()  = dp.at("aa_ov")(ia,aa) * t1_2p.at("aa")(aa,ia) )
    ( scalars.at("0008")()  = scalars.at("0004")() )
    ( scalars.at("0008")() += scalars.at("0005")() )
    ( scalars.at("0009")()  = scalars.at("0006")() )
    ( scalars.at("0009")() += scalars.at("0007")() )
        
    // r1_1p[aa]  = +1.000 d+_aa(a,i) 
    ( r1_1p.at("aa")(aa,ia)  = dp.at("aa_vo")(aa,ia) )
    
    // r0_1p()  = +1.000 d+_aa(i,a) t1_aa(a,i) 
    //       += +1.000 d+_bb(i,a) t1_bb(a,i) 
    ( r0_1p()  = scalars.at("0003")() )
    
    // r0_2p()  = +2.000 d+_aa(i,a) t1_1p_aa(a,i) 
    //       += +2.000 d+_bb(i,a) t1_1p_bb(a,i) 
    ( r0_2p()  = 2.000 * scalars.at("0008")() )
    
    // r2_2p[abab]  = +4.000 t2_2p_abab(a,b,i,j) w0 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb)  = 4.000 * w0 * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r1[aa]  = +1.000 f_aa(a,i) 
    ( r1.at("aa")(aa,ia)  = f.at("aa_vo")(aa,ia) )
    
    // r2_1p[abab]  = +1.000 t2_1p_abab(a,b,i,j) w0 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb)  = w0 * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2[abab]  = +1.000 d-_aa(a,i) t1_1p_bb(b,j) 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2.at("abab")(aa,bb,ia,jb)  = dp.at("aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    
    // r1_2p[aa]  = +4.000 t1_2p_aa(a,i) w0 
    // flops: o1v1  = o1v1
    //  mems: o1v1  = o1v1
    ( r1_2p.at("aa")(aa,ia)  = 4.000 * w0 * t1_2p.at("aa")(aa,ia) )
    
    // r0_1p() += +2.000 d-_aa(i,a) t1_2p_aa(a,i) 
    //       += +2.000 d-_bb(i,a) t1_2p_bb(a,i) 
    ( r0_1p() += 2.000 * scalars.at("0009")() )
    
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
    
    // r1[aa] += +0.500 <a,j||c,b>_aaaa t2_aaaa(c,b,i,j) 
    // flops: o1v1 += o2v2Q1 o1v2Q1
    //  mems: o1v1 += o1v1Q1 o1v1
    ( tmps.at("bin1_aa_voQ")(ca,ia,Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t2.at("aaaa")(ca,ba,ia,ja) )
    ( r1.at("aa")(aa,ia) += 0.500 * chol.at("aa_vvQ")(aa,ca,Q) * tmps.at("bin1_aa_voQ")(ca,ia,Q) )
    
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
    ( r1_1p.at("aa")(aa,ia) += scalars.at("0008")() * t1_1p.at("aa")(aa,ia) )
    
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
    
    // r1_1p[aa] += +0.500 <a,j||c,b>_aaaa t2_1p_aaaa(c,b,i,j) 
    // flops: o1v1 += o2v2Q1 o1v2Q1
    //  mems: o1v1 += o1v1Q1 o1v1
    ( tmps.at("bin1_aa_voQ")(ca,ia,Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t2_1p.at("aaaa")(ca,ba,ia,ja) )
    ( r1_1p.at("aa")(aa,ia) += 0.500 * chol.at("aa_vvQ")(aa,ca,Q) * tmps.at("bin1_aa_voQ")(ca,ia,Q) )
    
    // r1_2p[aa] += +4.000 d-_aa(j,b) t1_2p_aa(a,i) t1_1p_aa(b,j) 
    //             += +4.000 d-_bb(j,b) t1_2p_aa(a,i) t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 4.000 * scalars.at("0008")() * t1_2p.at("aa")(aa,ia) )
    
    // r1_2p[aa] += +2.000 d-_aa(j,b) t1_1p_aa(a,i) t1_2p_aa(b,j) 
    //             += +2.000 d-_bb(j,b) t1_1p_aa(a,i) t1_2p_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * scalars.at("0009")() * t1_1p.at("aa")(aa,ia) )
    
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
    
    // r1_2p[aa] += +1.000 <a,j||c,b>_aaaa t2_2p_aaaa(c,b,i,j) 
    // flops: o1v1 += o2v2Q1 o1v2Q1
    //  mems: o1v1 += o1v1Q1 o1v1
    ( tmps.at("bin1_aa_voQ")(ca,ia,Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t2_2p.at("aaaa")(ca,ba,ia,ja) )
    ( r1_2p.at("aa")(aa,ia) += chol.at("aa_vvQ")(aa,ca,Q) * tmps.at("bin1_aa_voQ")(ca,ia,Q) )
    
    // r2[abab] += +1.000 d-_bb(b,j) t1_1p_aa(a,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += dp.at("bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
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
    ( r2.at("abab")(aa,bb,ia,jb) -= f.at("bb_oo")(kb,jb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2[abab] += +1.000 f_aa(a,c) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += f.at("aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2[abab] += +1.000 f_bb(b,c) t2_abab(a,c,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += f.at("bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_1p[abab] += +2.000 d-_aa(a,i) t1_2p_bb(b,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * dp.at("aa_vo")(aa,ia) * t1_2p.at("bb")(bb,jb) )
    
    // r2_1p[abab] += +2.000 d-_bb(b,j) t1_2p_aa(a,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * dp.at("bb_vo")(bb,jb) * t1_2p.at("aa")(aa,ia) )
    
    // r2_1p[abab] += +2.000 d-_aa(k,c) t1_aa(c,k) t2_2p_abab(a,b,i,j) 
    //               += +2.000 d-_bb(k,c) t1_bb(c,k) t2_2p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * scalars.at("0003")() * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r2_1p[abab] += +1.000 d-_aa(k,c) t1_1p_aa(c,k) t2_1p_abab(a,b,i,j) 
    //               += +1.000 d-_bb(k,c) t1_1p_bb(c,k) t2_1p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += scalars.at("0008")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2_1p[abab] += -1.000 f_aa(k,i) t2_1p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= f.at("aa_oo")(ka,ia) * t2_1p.at("abab")(aa,bb,ka,jb) )
    
    // r2_1p[abab] += -1.000 f_bb(k,j) t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= f.at("bb_oo")(kb,jb) * t2_1p.at("abab")(aa,bb,ia,kb) )
    
    // r2_1p[abab] += +1.000 f_aa(a,c) t2_1p_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += f.at("aa_vv")(aa,ca) * t2_1p.at("abab")(ca,bb,ia,jb) )
    
    // r2_1p[abab] += +1.000 f_bb(b,c) t2_1p_abab(a,c,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += f.at("bb_vv")(bb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p[abab] += +4.000 d-_aa(k,c) t1_1p_aa(c,k) t2_2p_abab(a,b,i,j) 
    //               += +4.000 d-_bb(k,c) t1_1p_bb(c,k) t2_2p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.000 * scalars.at("0008")() * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 d-_aa(k,c) t1_2p_aa(c,k) t2_1p_abab(a,b,i,j) 
    //               += +2.000 d-_bb(k,c) t1_2p_bb(c,k) t2_1p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * scalars.at("0009")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 f_aa(k,i) t2_2p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * f.at("aa_oo")(ka,ia) * t2_2p.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p[abab] += -2.000 f_bb(k,j) t2_2p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * f.at("bb_oo")(kb,jb) * t2_2p.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p[abab] += +2.000 f_aa(a,c) t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * f.at("aa_vv")(aa,ca) * t2_2p.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 f_bb(b,c) t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * f.at("bb_vv")(bb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p[abab] += -2.000 d-_bb(k,j) t1_1p_aa(a,i) t1_2p_bb(b,k) 
    // flops: o2v2 += o2v1 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin1_bb_vo")(bb,jb)  = t1_2p.at("bb")(bb,kb) * dp.at("bb_oo")(kb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p[abab] += -2.000 d-_bb(k,c) t1_1p_aa(a,i) t1_2p_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o2v1 o2v1 o2v2
    //  mems: o2v2 += o2v0 o1v1 o2v2
    ( tmps.at("bin1_bb_oo")(jb,kb)  = t1.at("bb")(cb,jb) * dp.at("bb_ov")(kb,cb) )
    ( tmps.at("bin1_bb_vo")(bb,jb)  = t1_2p.at("bb")(bb,kb) * tmps.at("bin1_bb_oo")(jb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p[abab] += +2.000 d-_bb(b,c) t1_1p_aa(a,i) t1_2p_bb(c,j) 
    // flops: o2v2 += o1v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin1_bb_vo")(bb,jb)  = t1_2p.at("bb")(cb,jb) * dp.at("bb_vv")(bb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p[abab] += +2.000 d-_aa(k,c) t1_1p_aa(a,i) t2_2p_abab(c,b,k,j) 
    // flops: o2v2 += o2v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin1_bb_vo")(bb,jb)  = t2_2p.at("abab")(ca,bb,ka,jb) * dp.at("aa_ov")(ka,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("bin1_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p[abab] += -2.000 d-_bb(k,c) t1_1p_aa(a,i) t2_2p_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin1_bb_vo")(bb,jb)  = t2_2p.at("bbbb")(cb,bb,jb,kb) * dp.at("bb_ov")(kb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p[abab] += -2.000 f_bb(k,c) t1_2p_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin1_bb_oo")(jb,kb)  = f.at("bb_ov")(kb,cb) * t1_2p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p[abab] += -2.000 f_aa(k,c) t1_aa(a,k) t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(ca,bb,ia,jb) * f.at("aa_ov")(ka,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -6.000 d-_aa(k,c) t1_1p_aa(a,k) t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_2p.at("abab")(ca,bb,ia,jb) * dp.at("aa_ov")(ka,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 f_bb(k,c) t1_bb(b,k) t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = f.at("bb_ov")(kb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    .allocate(tmps.at("0001_aabb_vvvv"))
    
    // flops: o0v4  = o0v4Q1
    //  mems: o0v4  = o0v4
    ( tmps.at("0001_aabb_vvvv")(aa,da,bb,cb)  = chol.at("aa_vvQ")(aa,da,Q) * chol.at("bb_vvQ")(bb,cb,Q) )
    
    // r2[abab] += +0.500 <a,b||d,c>_abab t2_abab(d,c,i,j) 
    //            += +0.500 <a,b||c,d>_abab t2_abab(c,d,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0001_aabb_vvvv")(aa,da,bb,cb) * t2.at("abab")(da,cb,ia,jb) )
    
    // r2_1p[abab] += +0.500 <a,b||d,c>_abab t2_1p_abab(d,c,i,j) 
    //               += +0.500 <a,b||c,d>_abab t2_1p_abab(c,d,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0001_aabb_vvvv")(aa,da,bb,cb) * t2_1p.at("abab")(da,cb,ia,jb) )
    
    // r2_2p[abab] += +1.000 <a,b||d,c>_abab t2_2p_abab(d,c,i,j) 
    //               += +1.000 <a,b||c,d>_abab t2_2p_abab(c,d,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0001_aabb_vvvv")(aa,da,bb,cb) * t2_2p.at("abab")(da,cb,ia,jb) )
    .deallocate(tmps.at("0001_aabb_vvvv"))
    .allocate(tmps.at("0002_aaaa_voov"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin1_aaaa_vvoo")(ca,da,ka,la)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("0002_aaaa_voov")(aa,la,ia,da)  = t2.at("aaaa")(ca,aa,ia,ka) * tmps.at("bin1_aaaa_vvoo")(ca,da,ka,la) )
    
    // r1[aa] += +1.000 <k,j||b,c>_aaaa t1_aa(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += t1.at("aa")(ba,ja) * tmps.at("0002_aaaa_voov")(aa,ja,ia,ba) )
    
    // r1_1p[aa] += +1.000 <j,k||c,b>_aaaa t1_1p_aa(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1_1p.at("aa")(ba,ja) * tmps.at("0002_aaaa_voov")(aa,ja,ia,ba) )
    
    // r1_2p[aa] += +2.000 <j,k||c,b>_aaaa t1_2p_aa(b,j) t2_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.000 * t1_2p.at("aa")(ba,ja) * tmps.at("0002_aaaa_voov")(aa,ja,ia,ba) )
    
    // r2[abab] += +1.000 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_abab(d,b,l,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(da,bb,la,jb) * tmps.at("0002_aaaa_voov")(aa,la,ia,da) )
    
    // r2_1p[abab] += +1.000 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(da,bb,la,jb) * tmps.at("0002_aaaa_voov")(aa,la,ia,da) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2_2p.at("abab")(da,bb,la,jb) * tmps.at("0002_aaaa_voov")(aa,la,ia,da) )
    .deallocate(tmps.at("0002_aaaa_voov"))
    .allocate(tmps.at("0003_abab_vooo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_bbbb_vooo")(cb,jb,kb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("0003_abab_vooo")(aa,jb,ia,kb)  = t2.at("abab")(aa,cb,ia,lb) * tmps.at("bin1_bbbb_vooo")(cb,jb,kb,lb) )
    
    // r2[abab] += +1.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0003_abab_vooo")(aa,jb,ia,kb) )
    
    // r2_1p[abab] += -1.000 <k,l||j,c>_bbbb t1_1p_bb(b,k) t2_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0003_abab_vooo")(aa,jb,ia,kb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -2.000 <k,l||j,c>_bbbb t1_2p_bb(b,k) t2_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0003_abab_vooo")(aa,jb,ia,kb) * t1_2p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0003_abab_vooo"))
    .allocate(tmps.at("0004_baba_vooo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la)  = chol.at("aa_ooQ")(la,ia,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("0004_baba_vooo")(bb,ia,jb,ka)  = t2.at("abab")(ca,bb,la,jb) * tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la) )
    
    // r2[abab] += +1.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0004_baba_vooo")(bb,ia,jb,ka) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += -1.000 <k,l||i,c>_aaaa t1_1p_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0004_baba_vooo")(bb,ia,jb,ka) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += -2.000 <k,l||i,c>_aaaa t1_2p_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0004_baba_vooo")(bb,ia,jb,ka) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0004_baba_vooo"))
    .allocate(tmps.at("0005_aaaa_vvoo"))
    
    // flops: o2v2  = o2v1Q1 o2v2Q1
    //  mems: o2v2  = o2v0Q1 o2v2
    ( tmps.at("bin1_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("0005_aaaa_vvoo")(aa,da,ka,ia)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("bin1_aa_ooQ")(ia,ka,Q) )
    
    // r1[aa] += -1.000 <a,j||b,c>_aaaa t1_aa(b,j) t1_aa(c,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0005_aaaa_vvoo")(aa,ba,ja,ia) * t1.at("aa")(ba,ja) )
    
    // r1_1p[aa] += +1.000 <a,j||b,c>_aaaa t1_aa(b,i) t1_1p_aa(c,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1_1p.at("aa")(ca,ja) * tmps.at("0005_aaaa_vvoo")(aa,ca,ja,ia) )
    
    // r1_2p[aa] += +2.000 <a,j||b,c>_aaaa t1_aa(b,i) t1_2p_aa(c,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * t1_2p.at("aa")(ca,ja) * tmps.at("0005_aaaa_vvoo")(aa,ca,ja,ia) )
    
    // r2[abab] += +1.000 <a,k||c,d>_aaaa t1_aa(c,i) t2_abab(d,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0005_aaaa_vvoo")(aa,da,ka,ia) * t2.at("abab")(da,bb,ka,jb) )
    
    // r2_1p[abab] += +1.000 <a,k||c,d>_aaaa t1_aa(c,i) t2_1p_abab(d,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0005_aaaa_vvoo")(aa,da,ka,ia) * t2_1p.at("abab")(da,bb,ka,jb) )
    
    // r2_2p[abab] += +2.000 <a,k||c,d>_aaaa t1_aa(c,i) t2_2p_abab(d,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0005_aaaa_vvoo")(aa,da,ka,ia) * t2_2p.at("abab")(da,bb,ka,jb) )
    .deallocate(tmps.at("0005_aaaa_vvoo"))
    .allocate(tmps.at("0006_aaaa_vvoo"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0006_aaaa_vvoo")(aa,ca,ka,ia)  = chol.at("aa_vvQ")(aa,ca,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    
    // r1[aa] += +1.000 <a,j||i,b>_aaaa t1_aa(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0006_aaaa_vvoo")(aa,ba,ja,ia) * t1.at("aa")(ba,ja) )
    
    // r1_1p[aa] += +1.000 <a,j||i,b>_aaaa t1_1p_aa(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0006_aaaa_vvoo")(aa,ba,ja,ia) * t1_1p.at("aa")(ba,ja) )
    
    // r1_2p[aa] += +2.000 <a,j||i,b>_aaaa t1_2p_aa(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0006_aaaa_vvoo")(aa,ba,ja,ia) * t1_2p.at("aa")(ba,ja) )
    
    // r2[abab] += +1.000 <a,k||i,c>_aaaa t2_abab(c,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(ca,bb,ka,jb) * tmps.at("0006_aaaa_vvoo")(aa,ca,ka,ia) )
    
    // r2_1p[abab] += +1.000 <a,k||i,c>_aaaa t2_1p_abab(c,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(ca,bb,ka,jb) * tmps.at("0006_aaaa_vvoo")(aa,ca,ka,ia) )
    
    // r2_2p[abab] += +2.000 <a,k||i,c>_aaaa t2_2p_abab(c,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_2p.at("abab")(ca,bb,ka,jb) * tmps.at("0006_aaaa_vvoo")(aa,ca,ka,ia) )
    .deallocate(tmps.at("0006_aaaa_vvoo"))
    .allocate(tmps.at("0007_bbbb_vvoo"))
    
    // flops: o2v2  = o2v1Q1 o2v2Q1
    //  mems: o2v2  = o2v0Q1 o2v2
    ( tmps.at("bin1_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,jb) )
    ( tmps.at("0007_bbbb_vvoo")(bb,db,kb,jb)  = chol.at("bb_vvQ")(bb,db,Q) * tmps.at("bin1_bb_ooQ")(jb,kb,Q) )
    
    // r2[abab] += +1.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0007_bbbb_vvoo")(bb,db,kb,jb) * t2.at("abab")(aa,db,ia,kb) )
    
    // r2_1p[abab] += +1.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_1p_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0007_bbbb_vvoo")(bb,db,kb,jb) * t2_1p.at("abab")(aa,db,ia,kb) )
    
    // r2_2p[abab] += +2.000 <b,k||c,d>_bbbb t1_bb(c,j) t2_2p_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0007_bbbb_vvoo")(bb,db,kb,jb) * t2_2p.at("abab")(aa,db,ia,kb) )
    .deallocate(tmps.at("0007_bbbb_vvoo"))
    .allocate(tmps.at("0008_bbbb_vvoo"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0008_bbbb_vvoo")(bb,cb,kb,jb)  = chol.at("bb_vvQ")(bb,cb,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    
    // r2[abab] += +1.000 <b,k||j,c>_bbbb t2_abab(a,c,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,cb,ia,kb) * tmps.at("0008_bbbb_vvoo")(bb,cb,kb,jb) )
    
    // r2_1p[abab] += +1.000 <b,k||j,c>_bbbb t2_1p_abab(a,c,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,cb,ia,kb) * tmps.at("0008_bbbb_vvoo")(bb,cb,kb,jb) )
    
    // r2_2p[abab] += +2.000 <b,k||j,c>_bbbb t2_2p_abab(a,c,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_2p.at("abab")(aa,cb,ia,kb) * tmps.at("0008_bbbb_vvoo")(bb,cb,kb,jb) )
    .deallocate(tmps.at("0008_bbbb_vvoo"))
    .allocate(tmps.at("0009_baba_vooo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la)  = chol.at("aa_ooQ")(la,ia,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("0009_baba_vooo")(bb,ia,jb,ka)  = t2_1p.at("abab")(ca,bb,la,jb) * tmps.at("bin1_aaaa_vooo")(ca,ia,ka,la) )
    
    // r2_1p[abab] += +1.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_1p_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0009_baba_vooo")(bb,ia,jb,ka) * t1.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <l,k||i,c>_aaaa t1_1p_aa(a,k) t2_1p_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0009_baba_vooo")(bb,ia,jb,ka) * t1_1p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0009_baba_vooo"))
    .allocate(tmps.at("0010_abab_vooo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin1_bbbb_vooo")(cb,jb,kb,lb)  = chol.at("bb_ooQ")(lb,jb,Q) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("0010_abab_vooo")(aa,jb,ia,kb)  = t2_1p.at("abab")(aa,cb,ia,lb) * tmps.at("bin1_bbbb_vooo")(cb,jb,kb,lb) )
    
    // r2_1p[abab] += +1.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_1p_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0010_abab_vooo")(aa,jb,ia,kb) )
    
    // r2_2p[abab] += +2.000 <l,k||j,c>_bbbb t1_1p_bb(b,k) t2_1p_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1_1p.at("bb")(bb,kb) * tmps.at("0010_abab_vooo")(aa,jb,ia,kb) )
    .deallocate(tmps.at("0010_abab_vooo"))
    .allocate(tmps.at("0011_aaaa_vvoo"))
    
    // flops: o2v2  = o2v1Q1 o2v2Q1
    //  mems: o2v2  = o2v0Q1 o2v2
    ( tmps.at("bin1_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1_1p.at("aa")(ca,ia) )
    ( tmps.at("0011_aaaa_vvoo")(aa,da,ka,ia)  = chol.at("aa_vvQ")(aa,da,Q) * tmps.at("bin1_aa_ooQ")(ia,ka,Q) )
    
    // r1_1p[aa] += -1.000 <a,j||b,c>_aaaa t1_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0011_aaaa_vvoo")(aa,ba,ja,ia) * t1.at("aa")(ba,ja) )
    
    // r1_2p[aa] += -2.000 <a,j||b,c>_aaaa t1_1p_aa(b,j) t1_1p_aa(c,i) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0011_aaaa_vvoo")(aa,ba,ja,ia) * t1_1p.at("aa")(ba,ja) )
    
    // r2_1p[abab] += -1.000 <a,k||d,c>_aaaa t1_1p_aa(c,i) t2_abab(d,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0011_aaaa_vvoo")(aa,da,ka,ia) * t2.at("abab")(da,bb,ka,jb) )
    
    // r2_2p[abab] += +2.000 <a,k||c,d>_aaaa t1_1p_aa(c,i) t2_1p_abab(d,b,k,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0011_aaaa_vvoo")(aa,da,ka,ia) * t2_1p.at("abab")(da,bb,ka,jb) )
    .deallocate(tmps.at("0011_aaaa_vvoo"))
    .allocate(tmps.at("0012_bbbb_vvoo"))
    
    // flops: o2v2  = o2v1Q1 o2v2Q1
    //  mems: o2v2  = o2v0Q1 o2v2
    ( tmps.at("bin1_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1_1p.at("bb")(cb,jb) )
    ( tmps.at("0012_bbbb_vvoo")(bb,db,kb,jb)  = chol.at("bb_vvQ")(bb,db,Q) * tmps.at("bin1_bb_ooQ")(jb,kb,Q) )
    
    // r2_1p[abab] += -1.000 <b,k||d,c>_bbbb t1_1p_bb(c,j) t2_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0012_bbbb_vvoo")(bb,db,kb,jb) * t2.at("abab")(aa,db,ia,kb) )
    
    // r2_2p[abab] += +2.000 <b,k||c,d>_bbbb t1_1p_bb(c,j) t2_1p_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0012_bbbb_vvoo")(bb,db,kb,jb) * t2_1p.at("abab")(aa,db,ia,kb) )
    .deallocate(tmps.at("0012_bbbb_vvoo"))
    .allocate(tmps.at("0013_baab_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0013_baab_vvoo")(bb,aa,ia,jb)  = dp.at("bb_vv")(bb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    
    // r2[abab] += +1.000 d-_bb(b,c) t2_1p_abab(a,c,i,j) 
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0013_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2_1p[abab] += +1.000 d-_bb(b,c) t0_1p t2_1p_abab(a,c,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t0_1p * tmps.at("0013_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2_2p[abab] += +2.000 d+_bb(b,c) t2_1p_abab(a,c,i,j) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0013_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2_2p[abab] += +4.000 d-_bb(b,c) t0_2p t2_1p_abab(a,c,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.000 * t0_2p * tmps.at("0013_baab_vvoo")(bb,aa,ia,jb) )
    .deallocate(tmps.at("0013_baab_vvoo"))
    .allocate(tmps.at("0014_baab_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0014_baab_vvoo")(bb,aa,ia,jb)  = dp.at("bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2[abab] += +1.000 d-_bb(b,c) t0_1p t2_abab(a,c,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t0_1p * tmps.at("0014_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2_1p[abab] += +1.000 d+_bb(b,c) t2_abab(a,c,i,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0014_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2_1p[abab] += +2.000 d-_bb(b,c) t0_2p t2_abab(a,c,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * t0_2p * tmps.at("0014_baab_vvoo")(bb,aa,ia,jb) )
    .deallocate(tmps.at("0014_baab_vvoo"))
    .allocate(tmps.at("0015_baab_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0015_baab_vvoo")(bb,aa,ia,jb)  = dp.at("bb_vv")(bb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    
    // r2_1p[abab] += +2.000 d-_bb(b,c) t2_2p_abab(a,c,i,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0015_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2_2p[abab] += +2.000 d-_bb(b,c) t0_1p t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t0_1p * tmps.at("0015_baab_vvoo")(bb,aa,ia,jb) )
    .deallocate(tmps.at("0015_baab_vvoo"))
    .allocate(tmps.at("0016_abab_ovoo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vo")(da,la)  = chol.at("aa_ovQ")(la,da,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("0016_abab_ovoo")(la,bb,ia,jb)  = tmps.at("bin1_aa_vo")(da,la) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2[abab] += -1.000 <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0016_abab_ovoo")(la,bb,ia,jb) * t1.at("aa")(aa,la) )
    
    // r2_1p[abab] += -1.000 <l,k||d,c>_abab t1_1p_aa(a,l) t1_bb(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("aa")(aa,la) * tmps.at("0016_abab_ovoo")(la,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_abab t1_2p_aa(a,l) t1_bb(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_2p.at("aa")(aa,la) * tmps.at("0016_abab_ovoo")(la,bb,ia,jb) )
    .deallocate(tmps.at("0016_abab_ovoo"))
    .allocate(tmps.at("0017_baab_ovoo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t1_1p.at("bb")(cb,lb) )
    ( tmps.at("bin1_bb_vo")(db,kb)  = chol.at("bb_ovQ")(kb,db,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("0017_baab_ovoo")(kb,aa,ia,jb)  = tmps.at("bin1_bb_vo")(db,kb) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||d,c>_bbbb t1_bb(b,k) t1_1p_bb(c,l) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0017_baab_ovoo")(kb,aa,ia,jb) * t1.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_bbbb t1_1p_bb(b,l) t1_1p_bb(c,k) t2_abab(a,d,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_1p.at("bb")(bb,lb) * tmps.at("0017_baab_ovoo")(lb,aa,ia,jb) )
    .deallocate(tmps.at("0017_baab_ovoo"))
    .allocate(tmps.at("0018_abab_ovoo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin1_aa_vo")(da,la)  = chol.at("aa_ovQ")(la,da,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("0018_abab_ovoo")(la,bb,ia,jb)  = tmps.at("bin1_aa_vo")(da,la) * t2_1p.at("abab")(da,bb,ia,jb) )
    
    // r2_1p[abab] += -1.000 <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0018_abab_ovoo")(la,bb,ia,jb) * t1.at("aa")(aa,la) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_abab t1_1p_aa(a,l) t1_bb(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_1p.at("aa")(aa,la) * tmps.at("0018_abab_ovoo")(la,bb,ia,jb) )
    .deallocate(tmps.at("0018_abab_ovoo"))
    .allocate(tmps.at("0019_abab_ovoo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1
    ( tmps.at("bin1_Q")(Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1_1p.at("aa")(ca,la) )
    ( tmps.at("bin1_aa_vo")(da,ka)  = chol.at("aa_ovQ")(ka,da,Q) * tmps.at("bin1_Q")(Q) )
    ( tmps.at("0019_abab_ovoo")(ka,bb,ia,jb)  = tmps.at("bin1_aa_vo")(da,ka) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2_1p[abab] += +1.000 <l,k||d,c>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("0019_abab_ovoo")(ka,bb,ia,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_aaaa t1_1p_aa(a,l) t1_1p_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0019_abab_ovoo")(la,bb,ia,jb) * t1_1p.at("aa")(aa,la) )
    .deallocate(tmps.at("0019_abab_ovoo"))
    .allocate(tmps.at("0020_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0020_aa_oo")(ja,ia)  = f.at("aa_ov")(ja,ba) * t1.at("aa")(ba,ia) )
    
    // r1[aa] += -1.000 f_aa(j,b) t1_aa(a,j) t1_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= tmps.at("0020_aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    
    // r1_1p[aa] += -1.000 f_aa(j,b) t1_1p_aa(a,j) t1_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0020_aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    
    // r1_2p[aa] += -2.000 f_aa(j,b) t1_2p_aa(a,j) t1_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0020_aa_oo")(ja,ia) * t1_2p.at("aa")(aa,ja) )
    
    // r2[abab] += -1.000 f_aa(k,c) t1_aa(c,i) t2_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0020_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_1p[abab] += -1.000 f_aa(k,c) t1_aa(c,i) t2_1p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0020_aa_oo")(ka,ia) * t2_1p.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p[abab] += -2.000 f_aa(k,c) t1_aa(c,i) t2_2p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0020_aa_oo")(ka,ia) * t2_2p.at("abab")(aa,bb,ka,jb) )
    .deallocate(tmps.at("0020_aa_oo"))
    .allocate(tmps.at("0021_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0021_aa_oo")(ja,ia)  = f.at("aa_ov")(ja,ba) * t1_1p.at("aa")(ba,ia) )
    
    // r1_1p[aa] += -1.000 f_aa(j,b) t1_aa(a,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0021_aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    
    // r1_2p[aa] += -2.000 f_aa(j,b) t1_1p_aa(a,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0021_aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    
    // r2_1p[abab] += -1.000 f_aa(k,c) t1_1p_aa(c,i) t2_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0021_aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p[abab] += -2.000 f_aa(k,c) t1_1p_aa(c,i) t2_1p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0021_aa_oo")(ka,ia) * t2_1p.at("abab")(aa,bb,ka,jb) )
    .deallocate(tmps.at("0021_aa_oo"))
    .allocate(tmps.at("0022_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0022_bb_oo")(kb,jb)  = f.at("bb_ov")(kb,cb) * t1.at("bb")(cb,jb) )
    
    // r2[abab] += -1.000 f_bb(k,c) t1_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,kb) * tmps.at("0022_bb_oo")(kb,jb) )
    
    // r2_1p[abab] += -1.000 f_bb(k,c) t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,bb,ia,kb) * tmps.at("0022_bb_oo")(kb,jb) )
    
    // r2_2p[abab] += -2.000 f_bb(k,c) t1_bb(c,j) t2_2p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_2p.at("abab")(aa,bb,ia,kb) * tmps.at("0022_bb_oo")(kb,jb) )
    .deallocate(tmps.at("0022_bb_oo"))
    .allocate(tmps.at("0023_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0023_bb_oo")(kb,jb)  = f.at("bb_ov")(kb,cb) * t1_1p.at("bb")(cb,jb) )
    
    // r2_1p[abab] += -1.000 f_bb(k,c) t1_1p_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,kb) * tmps.at("0023_bb_oo")(kb,jb) )
    
    // r2_2p[abab] += -2.000 f_bb(k,c) t1_1p_bb(c,j) t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t2_1p.at("abab")(aa,bb,ia,kb) * tmps.at("0023_bb_oo")(kb,jb) )
    .deallocate(tmps.at("0023_bb_oo"))
    .allocate(tmps.at("0024_abab_oooo"))
    
    // flops: o4v0  = o2v2Q1 o4v2
    //  mems: o4v0  = o2v2 o4v0
    ( tmps.at("bin1_abab_vvoo")(da,cb,ka,lb)  = chol.at("aa_ovQ")(ka,da,Q) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("0024_abab_oooo")(ka,lb,ia,jb)  = t2_2p.at("abab")(da,cb,ia,jb) * tmps.at("bin1_abab_vvoo")(da,cb,ka,lb) )
    
    // r2_2p[abab] += +1.000 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t2_2p_abab(d,c,i,j) 
    //               += +1.000 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t2_2p_abab(c,d,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t1.at("bb")(bb,lb) * tmps.at("0024_abab_oooo")(ka,lb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += +0.500 <k,l||d,c>_abab t2_abab(a,b,k,l) t2_2p_abab(d,c,i,j) 
    //               += +0.500 <k,l||c,d>_abab t2_abab(a,b,k,l) t2_2p_abab(c,d,i,j) 
    //               += +0.500 <l,k||d,c>_abab t2_abab(a,b,l,k) t2_2p_abab(d,c,i,j) 
    //               += +0.500 <l,k||c,d>_abab t2_abab(a,b,l,k) t2_2p_abab(c,d,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * t2.at("abab")(aa,bb,ka,lb) * tmps.at("0024_abab_oooo")(ka,lb,ia,jb) )
    .deallocate(tmps.at("0024_abab_oooo"))
    .allocate(tmps.at("0025_bbaa_oovo"))
    
    // flops: o3v1  = o2v2Q1 o3v1Q1
    //  mems: o3v1  = o1v1Q1 o3v1
    ( tmps.at("bin1_aa_voQ")(aa,ia,Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t2.at("abab")(aa,cb,ia,lb) )
    ( tmps.at("0025_bbaa_oovo")(kb,jb,aa,ia)  = chol.at("bb_ooQ")(kb,jb,Q) * tmps.at("bin1_aa_voQ")(aa,ia,Q) )
    
    // r2[abab] += +1.000 <l,k||j,c>_bbbb t1_bb(b,k) t2_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0025_bbaa_oovo")(kb,jb,aa,ia) * t1.at("bb")(bb,kb) )
    
    // r2_1p[abab] += -1.000 <k,l||j,c>_bbbb t1_1p_bb(b,k) t2_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0025_bbaa_oovo")(kb,jb,aa,ia) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p[abab] += -2.000 <k,l||j,c>_bbbb t1_2p_bb(b,k) t2_abab(a,c,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("0025_bbaa_oovo")(kb,jb,aa,ia) * t1_2p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0025_bbaa_oovo"))
    .allocate(tmps.at("0026_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0026_bb_voQ")(bb,jb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t2.at("bbbb")(cb,bb,jb,kb) )
    
    // r2[abab] += -1.000 <a,k||i,c>_abab t2_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0026_bb_voQ")(bb,jb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    .allocate(tmps.at("0027_bb_voQ"))
    
    // flops: o1v1Q1  = o2v2Q1
    //  mems: o1v1Q1  = o1v1Q1
    ( tmps.at("0027_bb_voQ")(bb,jb,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t2.at("abab")(ca,bb,ka,jb) )
    
    // r2[abab] += +1.000 <a,k||i,c>_aaaa t2_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0027_bb_voQ")(bb,jb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    .allocate(tmps.at("0028_aabb_oovo"))
    
    // flops: o3v1  = o3v1Q1 o3v1Q1 o3v1
    //  mems: o3v1  = o3v1 o3v1 o3v1
    ( tmps.at("0028_aabb_oovo")(ka,ia,bb,jb)  = -1.000 * tmps.at("0027_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    ( tmps.at("0028_aabb_oovo")(ka,ia,bb,jb) += tmps.at("0026_bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    
    // r2[abab] += +1.000 <k,l||i,c>_abab t1_aa(a,k) t2_bbbb(c,b,j,l) 
    //            += +1.000 <l,k||i,c>_aaaa t1_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0028_aabb_oovo")(ka,ia,bb,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p[abab] += +1.000 <k,l||i,c>_abab t1_1p_aa(a,k) t2_bbbb(c,b,j,l) 
    //               += -1.000 <k,l||i,c>_aaaa t1_1p_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0028_aabb_oovo")(ka,ia,bb,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p[abab] += +2.000 <k,l||i,c>_abab t1_2p_aa(a,k) t2_bbbb(c,b,j,l) 
    //               += -2.000 <k,l||i,c>_aaaa t1_2p_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0028_aabb_oovo")(ka,ia,bb,jb) * t1_2p.at("aa")(aa,ka) )
    .deallocate(tmps.at("0028_aabb_oovo"))
    .allocate(tmps.at("0029_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0029_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1.at("bb")(ab,ib) )
    
    // r1[aa] += +1.000 <a,j||i,b>_abab t1_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0029_Q")(Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[abab] += -2.000 <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,la)  = tmps.at("0029_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin1_aa_vo")(da,la) * t2_2p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,la) * tmps.at("bin1_baab_vooo")(bb,ia,la,jb) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_bbbb t1_bb(b,l) t1_bb(c,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("0029_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,lb)  = t2_2p.at("abab")(aa,db,ia,jb) * tmps.at("bin1_bb_vo")(db,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,lb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,lb) )
    .allocate(tmps.at("0030_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0030_Q")(Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t1.at("aa")(aa,ia) )
    
    // r1[aa] += +1.000 <a,j||i,b>_aaaa t1_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += tmps.at("0030_Q")(Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[abab] += +2.000 <l,k||c,d>_aaaa t1_aa(a,l) t1_aa(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,la)  = chol.at("aa_ovQ")(la,da,Q) * tmps.at("0030_Q")(Q) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin1_aa_vo")(da,la) * t2_2p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,la) * tmps.at("bin1_baab_vooo")(bb,ia,la,jb) )
    
    // r2_2p[abab] += -2.000 <k,l||c,d>_abab t1_bb(b,l) t1_aa(c,k) t2_2p_abab(a,d,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,lb)  = tmps.at("0030_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,lb)  = t2_2p.at("abab")(aa,db,ia,jb) * tmps.at("bin1_bb_vo")(db,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,lb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,lb) )
    .allocate(tmps.at("0031_aa_vv"))
    
    // flops: o0v2  = o0v2Q1 o0v2Q1 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0031_aa_vv")(aa,da)  = tmps.at("0029_Q")(Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( tmps.at("0031_aa_vv")(aa,da) += chol.at("aa_vvQ")(aa,da,Q) * tmps.at("0030_Q")(Q) )
    
    // r2[abab] += +1.000 <a,k||d,c>_abab t1_bb(c,k) t2_abab(d,b,i,j) 
    //            += -1.000 <a,k||c,d>_aaaa t1_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0031_aa_vv")(aa,da) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2_1p[abab] += +1.000 <a,k||d,c>_abab t1_bb(c,k) t2_1p_abab(d,b,i,j) 
    //               += -1.000 <a,k||c,d>_aaaa t1_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0031_aa_vv")(aa,da) * t2_1p.at("abab")(da,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <a,k||d,c>_abab t1_bb(c,k) t2_2p_abab(d,b,i,j) 
    //               += -2.000 <a,k||c,d>_aaaa t1_aa(c,k) t2_2p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0031_aa_vv")(aa,da) * t2_2p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0031_aa_vv"))
    .allocate(tmps.at("0032_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0032_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1_1p.at("bb")(ab,ib) )
    
    // r1_1p[aa] += +1.000 <a,j||i,b>_abab t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0032_Q")(Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[abab] += -2.000 <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(c,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,ka)  = chol.at("aa_ovQ")(ka,da,Q) * tmps.at("0032_Q")(Q) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin1_aa_vo")(da,ka) * t2_1p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("aa")(aa,ka) * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_bbbb t1_bb(b,k) t1_1p_bb(c,l) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,kb)  = tmps.at("0032_Q")(Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t2_1p.at("abab")(aa,db,ia,jb) * tmps.at("bin1_bb_vo")(db,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    .allocate(tmps.at("0033_Q"))
    
    // flops: o0v0Q1  = o1v1Q1
    //  mems: o0v0Q1  = o0v0Q1
    ( tmps.at("0033_Q")(Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t1_1p.at("aa")(aa,ia) )
    
    // r1_1p[aa] += +1.000 <a,j||i,b>_aaaa t1_1p_aa(b,j) 
    // flops: o1v1 += o1v1Q1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += tmps.at("0033_Q")(Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_abab t1_bb(b,k) t1_1p_aa(c,l) t2_1p_abab(a,d,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_bb_vo")(db,kb)  = tmps.at("0033_Q")(Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb)  = t2_1p.at("abab")(aa,db,ia,jb) * tmps.at("bin1_bb_vo")(db,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1.at("bb")(bb,kb) * tmps.at("bin1_aabb_vooo")(aa,ia,jb,kb) )
    
    // r2_2p[abab] += -2.000 <l,k||c,d>_aaaa t1_aa(a,k) t1_1p_aa(c,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    ( tmps.at("bin1_aa_vo")(da,ka)  = tmps.at("0033_Q")(Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("bin1_baab_vooo")(bb,ia,ka,jb)  = t2_1p.at("abab")(da,bb,ia,jb) * tmps.at("bin1_aa_vo")(da,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * tmps.at("bin1_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0034_aa_vv"))
    
    // flops: o0v2  = o0v2Q1 o0v2Q1 o0v2
    //  mems: o0v2  = o0v2 o0v2 o0v2
    ( tmps.at("0034_aa_vv")(aa,da)  = tmps.at("0032_Q")(Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( tmps.at("0034_aa_vv")(aa,da) += chol.at("aa_vvQ")(aa,da,Q) * tmps.at("0033_Q")(Q) )
    
    // r2_1p[abab] += +1.000 <a,k||d,c>_abab t1_1p_bb(c,k) t2_abab(d,b,i,j) 
    //               += +1.000 <a,k||d,c>_aaaa t1_1p_aa(c,k) t2_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0034_aa_vv")(aa,da) * t2.at("abab")(da,bb,ia,jb) )
    
    // r2_2p[abab] += +2.000 <a,k||d,c>_abab t1_1p_bb(c,k) t2_1p_abab(d,b,i,j) 
    //               += -2.000 <a,k||c,d>_aaaa t1_1p_aa(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0034_aa_vv")(aa,da) * t2_1p.at("abab")(da,bb,ia,jb) )
    .deallocate(tmps.at("0034_aa_vv"))
    .allocate(tmps.at("0035_aa_vo"))
    
    // flops: o1v1  = o2v2 o1v2 o1v1 o2v1 o2v1 o1v1 o2v2 o1v1 o2v1 o1v1
    //  mems: o1v1  = o1v1 o1v1 o1v1 o2v0 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1
    ( tmps.at("0035_aa_vo")(aa,ia)  = -1.000 * dp.at("bb_ov")(jb,bb) * t2.at("abab")(aa,bb,ia,jb) )
    ( tmps.at("0035_aa_vo")(aa,ia) -= dp.at("aa_vv")(aa,ba) * t1.at("aa")(ba,ia) )
    ( tmps.at("bin1_aa_oo")(ia,ja)  = dp.at("aa_ov")(ja,ba) * t1.at("aa")(ba,ia) )
    ( tmps.at("0035_aa_vo")(aa,ia) += tmps.at("bin1_aa_oo")(ia,ja) * t1.at("aa")(aa,ja) )
    ( tmps.at("0035_aa_vo")(aa,ia) += dp.at("aa_ov")(ja,ba) * t2.at("aaaa")(ba,aa,ia,ja) )
    ( tmps.at("0035_aa_vo")(aa,ia) += dp.at("aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    
    // r1[aa] += -1.000 d-_aa(j,i) t0_1p t1_aa(a,j) 
    //          += -1.000 d-_aa(j,b) t0_1p t2_aaaa(b,a,i,j) 
    //          += +1.000 d-_aa(a,b) t0_1p t1_aa(b,i) 
    //          += +1.000 d-_bb(j,b) t0_1p t2_abab(a,b,i,j) 
    //          += -1.000 d-_aa(j,b) t0_1p t1_aa(a,j) t1_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= t0_1p * tmps.at("0035_aa_vo")(aa,ia) )
    
    // r1_1p[aa] += -1.000 d+_aa(j,i) t1_aa(a,j) 
    //             += -1.000 d+_aa(j,b) t2_aaaa(b,a,i,j) 
    //             += +1.000 d+_aa(a,b) t1_aa(b,i) 
    //             += +1.000 d+_bb(j,b) t2_abab(a,b,i,j) 
    //             += -1.000 d+_aa(j,b) t1_aa(a,j) t1_aa(b,i) 
    ( r1_1p.at("aa")(aa,ia) -= tmps.at("0035_aa_vo")(aa,ia) )
    
    // r1_1p[aa] += -2.000 d-_aa(j,i) t0_2p t1_aa(a,j) 
    //             += -2.000 d-_aa(j,b) t0_2p t2_aaaa(b,a,i,j) 
    //             += +2.000 d-_aa(a,b) t0_2p t1_aa(b,i) 
    //             += +2.000 d-_bb(j,b) t0_2p t2_abab(a,b,i,j) 
    //             += -2.000 d-_aa(j,b) t0_2p t1_aa(a,j) t1_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= 2.000 * t0_2p * tmps.at("0035_aa_vo")(aa,ia) )
    
    // r2[abab] += -1.000 d-_aa(k,i) t1_aa(a,k) t1_1p_bb(b,j) 
    //            += -1.000 d-_aa(k,c) t1_1p_bb(b,j) t2_aaaa(c,a,i,k) 
    //            += +1.000 d-_aa(a,c) t1_1p_bb(b,j) t1_aa(c,i) 
    //            += +1.000 d-_bb(k,c) t1_1p_bb(b,j) t2_abab(a,c,i,k) 
    //            += -1.000 d-_aa(k,c) t1_aa(a,k) t1_1p_bb(b,j) t1_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t1_1p.at("bb")(bb,jb) * tmps.at("0035_aa_vo")(aa,ia) )
    
    // r2_1p[abab] += -2.000 d-_aa(k,i) t1_aa(a,k) t1_2p_bb(b,j) 
    //               += -2.000 d-_aa(k,c) t1_2p_bb(b,j) t2_aaaa(c,a,i,k) 
    //               += +2.000 d-_aa(a,c) t1_2p_bb(b,j) t1_aa(c,i) 
    //               += +2.000 d-_bb(k,c) t1_2p_bb(b,j) t2_abab(a,c,i,k) 
    //               += -2.000 d-_aa(k,c) t1_aa(a,k) t1_2p_bb(b,j) t1_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_2p.at("bb")(bb,jb) * tmps.at("0035_aa_vo")(aa,ia) )
    .deallocate(tmps.at("0035_aa_vo"))
    .allocate(tmps.at("0036_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0036_aa_oo")(ja,ia)  = dp.at("aa_ov")(ja,ba) * t1_1p.at("aa")(ba,ia) )
    
    // r1_2p[aa] += -6.000 d-_aa(j,b) t1_2p_aa(a,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 6.000 * tmps.at("0036_aa_oo")(ja,ia) * t1_2p.at("aa")(aa,ja) )
    
    // r2_2p[abab] += -6.000 d-_aa(k,c) t1_1p_aa(c,i) t2_2p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * tmps.at("0036_aa_oo")(ka,ia) * t2_2p.at("abab")(aa,bb,ka,jb) )
    .allocate(tmps.at("0037_aa_vo"))
    
    // flops: o1v1  = o2v2 o1v2 o1v1 o2v1 o2v1 o1v1 o2v1 o1v1 o2v1 o1v1 o2v2 o1v1
    //  mems: o1v1  = o1v1 o1v1 o1v1 o2v0 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1
    ( tmps.at("0037_aa_vo")(aa,ia)  = -1.000 * dp.at("bb_ov")(jb,bb) * t2_1p.at("abab")(aa,bb,ia,jb) )
    ( tmps.at("0037_aa_vo")(aa,ia) -= dp.at("aa_vv")(aa,ba) * t1_1p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aa_oo")(ia,ja)  = dp.at("aa_ov")(ja,ba) * t1.at("aa")(ba,ia) )
    ( tmps.at("0037_aa_vo")(aa,ia) += tmps.at("bin1_aa_oo")(ia,ja) * t1_1p.at("aa")(aa,ja) )
    ( tmps.at("0037_aa_vo")(aa,ia) += t1.at("aa")(aa,ja) * tmps.at("0036_aa_oo")(ja,ia) )
    ( tmps.at("0037_aa_vo")(aa,ia) += dp.at("aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    ( tmps.at("0037_aa_vo")(aa,ia) += dp.at("aa_ov")(ja,ba) * t2_1p.at("aaaa")(ba,aa,ia,ja) )
    
    // r1[aa] += -1.000 d-_aa(j,i) t1_1p_aa(a,j) 
    //          += -1.000 d-_aa(j,b) t2_1p_aaaa(b,a,i,j) 
    //          += +1.000 d-_aa(a,b) t1_1p_aa(b,i) 
    //          += +1.000 d-_bb(j,b) t2_1p_abab(a,b,i,j) 
    //          += -1.000 d-_aa(j,b) t1_1p_aa(a,j) t1_aa(b,i) 
    //          += -1.000 d-_aa(j,b) t1_aa(a,j) t1_1p_aa(b,i) 
    ( r1.at("aa")(aa,ia) -= tmps.at("0037_aa_vo")(aa,ia) )
    
    // r1_1p[aa] += -1.000 d-_aa(j,i) t0_1p t1_1p_aa(a,j) 
    //             += -1.000 d-_aa(j,b) t0_1p t2_1p_aaaa(b,a,i,j) 
    //             += +1.000 d-_aa(a,b) t0_1p t1_1p_aa(b,i) 
    //             += +1.000 d-_bb(j,b) t0_1p t2_1p_abab(a,b,i,j) 
    //             += -1.000 d-_aa(j,b) t0_1p t1_1p_aa(a,j) t1_aa(b,i) 
    //             += -1.000 d-_aa(j,b) t0_1p t1_aa(a,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t0_1p * tmps.at("0037_aa_vo")(aa,ia) )
    
    // r1_2p[aa] += -2.000 d+_aa(j,i) t1_1p_aa(a,j) 
    //             += -2.000 d+_aa(j,b) t2_1p_aaaa(b,a,i,j) 
    //             += +2.000 d+_aa(a,b) t1_1p_aa(b,i) 
    //             += +2.000 d+_bb(j,b) t2_1p_abab(a,b,i,j) 
    //             += -2.000 d+_aa(j,b) t1_1p_aa(a,j) t1_aa(b,i) 
    //             += -2.000 d+_aa(j,b) t1_aa(a,j) t1_1p_aa(b,i) 
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * tmps.at("0037_aa_vo")(aa,ia) )
    
    // r1_2p[aa] += -4.000 d-_aa(j,i) t0_2p t1_1p_aa(a,j) 
    //             += -4.000 d-_aa(j,b) t0_2p t2_1p_aaaa(b,a,i,j) 
    //             += +4.000 d-_aa(a,b) t0_2p t1_1p_aa(b,i) 
    //             += +4.000 d-_bb(j,b) t0_2p t2_1p_abab(a,b,i,j) 
    //             += -4.000 d-_aa(j,b) t0_2p t1_1p_aa(a,j) t1_aa(b,i) 
    //             += -4.000 d-_aa(j,b) t0_2p t1_aa(a,j) t1_1p_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 4.000 * t0_2p * tmps.at("0037_aa_vo")(aa,ia) )
    
    // r2_1p[abab] += -1.000 d-_aa(k,i) t1_1p_aa(a,k) t1_1p_bb(b,j) 
    //               += -1.000 d-_aa(k,c) t1_1p_bb(b,j) t2_1p_aaaa(c,a,i,k) 
    //               += +1.000 d-_aa(a,c) t1_1p_bb(b,j) t1_1p_aa(c,i) 
    //               += +1.000 d-_bb(k,c) t1_1p_bb(b,j) t2_1p_abab(a,c,i,k) 
    //               += -1.000 d-_aa(k,c) t1_1p_aa(a,k) t1_1p_bb(b,j) t1_aa(c,i) 
    //               += -1.000 d-_aa(k,c) t1_aa(a,k) t1_1p_bb(b,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("bb")(bb,jb) * tmps.at("0037_aa_vo")(aa,ia) )
    
    // r2_2p[abab] += -4.000 d-_aa(k,i) t1_1p_aa(a,k) t1_2p_bb(b,j) 
    //               += -4.000 d-_aa(k,c) t1_2p_bb(b,j) t2_1p_aaaa(c,a,i,k) 
    //               += +4.000 d-_aa(a,c) t1_2p_bb(b,j) t1_1p_aa(c,i) 
    //               += +4.000 d-_bb(k,c) t1_2p_bb(b,j) t2_1p_abab(a,c,i,k) 
    //               += -4.000 d-_aa(k,c) t1_1p_aa(a,k) t1_2p_bb(b,j) t1_aa(c,i) 
    //               += -4.000 d-_aa(k,c) t1_aa(a,k) t1_2p_bb(b,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.000 * t1_2p.at("bb")(bb,jb) * tmps.at("0037_aa_vo")(aa,ia) )
    .deallocate(tmps.at("0037_aa_vo"))
    .allocate(tmps.at("0038_aa_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0038_aa_oo")(ja,ia)  = dp.at("aa_ov")(ja,ba) * t1_2p.at("aa")(ba,ia) )
    
    // r1_2p[aa] += -6.000 d-_aa(j,b) t1_1p_aa(a,j) t1_2p_aa(b,i) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 6.000 * tmps.at("0038_aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    
    // r2_2p[abab] += -6.000 d-_aa(k,c) t1_2p_aa(c,i) t2_1p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * tmps.at("0038_aa_oo")(ka,ia) * t2_1p.at("abab")(aa,bb,ka,jb) )
    .allocate(tmps.at("0039_aa_vo"))
    
    // flops: o1v1  = o2v2 o1v2 o1v1 o2v1 o2v1 o1v1 o2v1 o1v1 o2v1 o1v1 o2v1 o1v1 o2v2 o1v1
    //  mems: o1v1  = o1v1 o1v1 o1v1 o2v0 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1
    ( tmps.at("0039_aa_vo")(aa,ia)  = -1.000 * dp.at("bb_ov")(jb,bb) * t2_2p.at("abab")(aa,bb,ia,jb) )
    ( tmps.at("0039_aa_vo")(aa,ia) -= dp.at("aa_vv")(aa,ba) * t1_2p.at("aa")(ba,ia) )
    ( tmps.at("bin1_aa_oo")(ia,ja)  = dp.at("aa_ov")(ja,ba) * t1.at("aa")(ba,ia) )
    ( tmps.at("0039_aa_vo")(aa,ia) += tmps.at("bin1_aa_oo")(ia,ja) * t1_2p.at("aa")(aa,ja) )
    ( tmps.at("0039_aa_vo")(aa,ia) += t1_1p.at("aa")(aa,ja) * tmps.at("0036_aa_oo")(ja,ia) )
    ( tmps.at("0039_aa_vo")(aa,ia) += t1.at("aa")(aa,ja) * tmps.at("0038_aa_oo")(ja,ia) )
    ( tmps.at("0039_aa_vo")(aa,ia) += dp.at("aa_oo")(ja,ia) * t1_2p.at("aa")(aa,ja) )
    ( tmps.at("0039_aa_vo")(aa,ia) += dp.at("aa_ov")(ja,ba) * t2_2p.at("aaaa")(ba,aa,ia,ja) )
    
    // r1_1p[aa] += -2.000 d-_aa(j,i) t1_2p_aa(a,j) 
    //             += -2.000 d-_aa(j,b) t2_2p_aaaa(b,a,i,j) 
    //             += +2.000 d-_aa(a,b) t1_2p_aa(b,i) 
    //             += +2.000 d-_bb(j,b) t2_2p_abab(a,b,i,j) 
    //             += -2.000 d-_aa(j,b) t1_2p_aa(a,j) t1_aa(b,i) 
    //             += -2.000 d-_aa(j,b) t1_1p_aa(a,j) t1_1p_aa(b,i) 
    //             += -2.000 d-_aa(j,b) t1_aa(a,j) t1_2p_aa(b,i) 
    ( r1_1p.at("aa")(aa,ia) -= 2.000 * tmps.at("0039_aa_vo")(aa,ia) )
    
    // r1_2p[aa] += -2.000 d-_aa(j,i) t0_1p t1_2p_aa(a,j) 
    //             += -2.000 d-_aa(j,b) t0_1p t2_2p_aaaa(b,a,i,j) 
    //             += +2.000 d-_aa(a,b) t0_1p t1_2p_aa(b,i) 
    //             += +2.000 d-_bb(j,b) t0_1p t2_2p_abab(a,b,i,j) 
    //             += -2.000 d-_aa(j,b) t0_1p t1_2p_aa(a,j) t1_aa(b,i) 
    //             += -2.000 d-_aa(j,b) t0_1p t1_1p_aa(a,j) t1_1p_aa(b,i) 
    //             += -2.000 d-_aa(j,b) t0_1p t1_aa(a,j) t1_2p_aa(b,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.000 * t0_1p * tmps.at("0039_aa_vo")(aa,ia) )
    
    // r2_2p[abab] += -2.000 d-_aa(k,i) t1_2p_aa(a,k) t1_1p_bb(b,j) 
    //               += -2.000 d-_aa(k,c) t1_1p_bb(b,j) t2_2p_aaaa(c,a,i,k) 
    //               += +2.000 d-_aa(a,c) t1_1p_bb(b,j) t1_2p_aa(c,i) 
    //               += +2.000 d-_bb(k,c) t1_1p_bb(b,j) t2_2p_abab(a,c,i,k) 
    //               += -2.000 d-_aa(k,c) t1_2p_aa(a,k) t1_1p_bb(b,j) t1_aa(c,i) 
    //               += -2.000 d-_aa(k,c) t1_1p_aa(a,k) t1_1p_bb(b,j) t1_1p_aa(c,i) 
    //               += -2.000 d-_aa(k,c) t1_aa(a,k) t1_1p_bb(b,j) t1_2p_aa(c,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_1p.at("bb")(bb,jb) * tmps.at("0039_aa_vo")(aa,ia) )
    .deallocate(tmps.at("0039_aa_vo"))
    .allocate(tmps.at("0040_bb_vo"))
    
    // flops: o1v1  = o2v1 o2v1 o2v1 o1v1 o2v2 o1v1 o2v2 o1v1 o1v2 o1v1
    //  mems: o1v1  = o2v0 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1
    ( tmps.at("bin1_bb_oo")(jb,kb)  = dp.at("bb_ov")(kb,cb) * t1.at("bb")(cb,jb) )
    ( tmps.at("0040_bb_vo")(bb,jb)  = -1.000 * tmps.at("bin1_bb_oo")(jb,kb) * t1.at("bb")(bb,kb) )
    ( tmps.at("0040_bb_vo")(bb,jb) -= dp.at("bb_oo")(kb,jb) * t1.at("bb")(bb,kb) )
    ( tmps.at("0040_bb_vo")(bb,jb) -= dp.at("bb_ov")(kb,cb) * t2.at("bbbb")(cb,bb,jb,kb) )
    ( tmps.at("0040_bb_vo")(bb,jb) += dp.at("aa_ov")(ka,ca) * t2.at("abab")(ca,bb,ka,jb) )
    ( tmps.at("0040_bb_vo")(bb,jb) += dp.at("bb_vv")(bb,cb) * t1.at("bb")(cb,jb) )
    
    // r2[abab] += +1.000 d-_aa(k,c) t1_1p_aa(a,i) t2_abab(c,b,k,j) 
    //            += -1.000 d-_bb(k,j) t1_1p_aa(a,i) t1_bb(b,k) 
    //            += -1.000 d-_bb(k,c) t1_1p_aa(a,i) t2_bbbb(c,b,j,k) 
    //            += +1.000 d-_bb(b,c) t1_1p_aa(a,i) t1_bb(c,j) 
    //            += -1.000 d-_bb(k,c) t1_1p_aa(a,i) t1_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0040_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_1p[abab] += +2.000 d-_aa(k,c) t1_2p_aa(a,i) t2_abab(c,b,k,j) 
    //               += -2.000 d-_bb(k,j) t1_2p_aa(a,i) t1_bb(b,k) 
    //               += -2.000 d-_bb(k,c) t1_2p_aa(a,i) t2_bbbb(c,b,j,k) 
    //               += +2.000 d-_bb(b,c) t1_2p_aa(a,i) t1_bb(c,j) 
    //               += -2.000 d-_bb(k,c) t1_2p_aa(a,i) t1_bb(b,k) t1_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.000 * tmps.at("0040_bb_vo")(bb,jb) * t1_2p.at("aa")(aa,ia) )
    .deallocate(tmps.at("0040_bb_vo"))
    .allocate(tmps.at("0041_bb_oo"))
    
    // flops: o2v0  = o2v1
    //  mems: o2v0  = o2v0
    ( tmps.at("0041_bb_oo")(kb,jb)  = dp.at("bb_ov")(kb,cb) * t1_1p.at("bb")(cb,jb) )
    
    // r2_2p[abab] += -2.000 d-_bb(k,c) t1_1p_aa(a,i) t1_1p_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v1 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin1_bb_vo")(bb,jb)  = tmps.at("0041_bb_oo")(kb,jb) * t1_1p.at("bb")(bb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.000 * t1_1p.at("aa")(aa,ia) * tmps.at("bin1_bb_vo")(bb,jb) )
    
    // r2_2p[abab] += -6.000 d-_bb(k,c) t1_1p_bb(c,j) t2_2p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.000 * t2_2p.at("abab")(aa,bb,ia,kb) * tmps.at("0041_bb_oo")(kb,jb) )
    .allocate(tmps.at("0042_bb_vo"))
    
    // flops: o1v1  = o2v1 o2v1 o2v1 o1v1 o2v1 o2v2 o1v1 o1v1 o2v2 o1v1 o1v2 o1v1
    //  mems: o1v1  = o2v0 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1 o1v1
    ( tmps.at("bin1_bb_oo")(jb,kb)  = t1.at("bb")(cb,jb) * dp.at("bb_ov")(kb,cb) )
    ( tmps.at("0042_bb_vo")(bb,jb)  = -1.000 * tmps.at("bin1_bb_oo")(jb,kb) * t1_1p.at("bb")(bb,kb) )
    ( tmps.at("0042_bb_vo")(bb,jb) -= dp.at("bb_oo")(kb,jb) * t1_1p.at("bb")(bb,kb) )
    ( tmps.at("0042_bb_vo")(bb,jb) -= t1.at("bb")(bb,kb) * tmps.at("0041_bb_oo")(kb,jb) )
    ( tmps.at("0042_bb_vo")(bb,jb) -= dp.at("bb_ov")(kb,cb) * t2_1p.at("bbbb")(cb,bb,jb,kb) )
    ( tmps.at("0042_bb_vo")(bb,jb) += dp.at("aa_ov")(ka,ca) * t2_1p.at("abab")(ca,bb,ka,jb) )
    ( tmps.at("0042_bb_vo")(bb,jb) += dp.at("bb_vv")(bb,cb) * t1_1p.at("bb")(cb,jb) )
    
    // r2_1p[abab] += +1.000 d-_aa(k,c) t1_1p_aa(a,i) t2_1p_abab(c,b,k,j) 
    //               += -1.000 d-_bb(k,j) t1_1p_aa(a,i) t1_1p_bb(b,k) 
    //               += -1.000 d-_bb(k,c) t1_1p_aa(a,i) t2_1p_bbbb(c,b,j,k) 
    //               += +1.000 d-_bb(b,c) t1_1p_aa(a,i) t1_1p_bb(c,j) 
    //               += -1.000 d-_bb(k,c) t1_1p_aa(a,i) t1_1p_bb(b,k) t1_bb(c,j) 
    //               += -1.000 d-_bb(k,c) t1_1p_aa(a,i) t1_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0042_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p[abab] += +4.000 d-_aa(k,c) t1_2p_aa(a,i) t2_1p_abab(c,b,k,j) 
    //               += -4.000 d-_bb(k,j) t1_2p_aa(a,i) t1_1p_bb(b,k) 
    //               += -4.000 d-_bb(k,c) t1_2p_aa(a,i) t2_1p_bbbb(c,b,j,k) 
    //               += +4.000 d-_bb(b,c) t1_2p_aa(a,i) t1_1p_bb(c,j) 
    //               += -4.000 d-_bb(k,c) t1_2p_aa(a,i) t1_1p_bb(b,k) t1_bb(c,j) 
    //               += -4.000 d-_bb(k,c) t1_2p_aa(a,i) t1_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.000 * tmps.at("0042_bb_vo")(bb,jb) * t1_2p.at("aa")(aa,ia) )
    .deallocate(tmps.at("0042_bb_vo"))
    .allocate(tmps.at("0043_abba_vvoo"))
    
    // flops: o2v2  = o2v1 o3v2 o3v2 o2v2
    //  mems: o2v2  = o2v0 o2v2 o2v2 o2v2
    ( tmps.at("bin1_bb_oo")(jb,kb)  = t1.at("bb")(cb,jb) * dp.at("bb_ov")(kb,cb) )
    ( tmps.at("0043_abba_vvoo")(aa,bb,jb,ia)  = t2.at("abab")(aa,bb,ia,kb) * tmps.at("bin1_bb_oo")(jb,kb) )
    ( tmps.at("0043_abba_vvoo")(aa,bb,jb,ia) += t2.at("abab")(aa,bb,ia,kb) * dp.at("bb_oo")(kb,jb) )
    
    // r2[abab] += -1.000 d-_bb(k,j) t0_1p t2_abab(a,b,i,k) 
    //            += -1.000 d-_bb(k,c) t0_1p t1_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0043_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_1p[abab] += -1.000 d+_bb(k,j) t2_abab(a,b,i,k) 
    //               += -1.000 d+_bb(k,c) t1_bb(c,j) t2_abab(a,b,i,k) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0043_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_1p[abab] += -2.000 d-_bb(k,j) t0_2p t2_abab(a,b,i,k) 
    //               += -2.000 d-_bb(k,c) t0_2p t1_bb(c,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.000 * t0_2p * tmps.at("0043_abba_vvoo")(aa,bb,jb,ia) )
    .deallocate(tmps.at("0043_abba_vvoo"))
    .allocate(tmps.at("0044_abba_vvoo"))
    
    // flops: o2v2  = o3v2 o2v1 o3v2 o2v2 o3v2 o2v2
    //  mems: o2v2  = o2v2 o2v0 o2v2 o2v2 o2v2 o2v2
    ( tmps.at("0044_abba_vvoo")(aa,bb,jb,ia)  = t2.at("abab")(aa,bb,ia,kb) * tmps.at("0041_bb_oo")(kb,jb) )
    ( tmps.at("bin1_bb_oo")(jb,kb)  = t1.at("bb")(cb,jb) * dp.at("bb_ov")(kb,cb) )
    ( tmps.at("0044_abba_vvoo")(aa,bb,jb,ia) += t2_1p.at("abab")(aa,bb,ia,kb) * tmps.at("bin1_bb_oo")(jb,kb) )
    ( tmps.at("0044_abba_vvoo")(aa,bb,jb,ia) += t2_1p.at("abab")(aa,bb,ia,kb) * dp.at("bb_oo")(kb,jb) )
    ;
  }
  // clang-format on
}

template void exachem::cc::cd_qed_ccsd_cs::resid_part1<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);