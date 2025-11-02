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
    ( tmps.at("bin_bb_voQ")(ab,jb,Q)  = chol.at("bb_ovQ")(ib,bb,Q) * t2.at("bbbb")(ab,bb,jb,ib) )
    ( scalars.at("0001")()  = tmps.at("bin_bb_voQ")(ab,jb,Q) * chol.at("bb_ovQ")(jb,ab,Q) )
    ( tmps.at("bin_bb_voQ")(ab,jb,Q)  = chol.at("bb_ovQ")(ib,bb,Q) * t2_2p.at("bbbb")(ab,bb,jb,ib) )
    ( scalars.at("0002")()  = tmps.at("bin_bb_voQ")(ab,jb,Q) * chol.at("bb_ovQ")(jb,ab,Q) )
    ( tmps.at("bin_bb_voQ")(bb,jb,Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t2_1p.at("bbbb")(ab,bb,jb,ib) )
    ( scalars.at("0003")()  = tmps.at("bin_bb_voQ")(bb,jb,Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    ( tmps.at("bin_bb_voQ")(bb,jb,Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t2.at("bbbb")(ab,bb,jb,ib) )
    ( scalars.at("0004")()  = tmps.at("bin_bb_voQ")(bb,jb,Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    ( tmps.at("bin_bb_voQ")(ab,jb,Q)  = chol.at("bb_ovQ")(ib,bb,Q) * t2_1p.at("bbbb")(ab,bb,jb,ib) )
    ( scalars.at("0005")()  = tmps.at("bin_bb_voQ")(ab,jb,Q) * chol.at("bb_ovQ")(jb,ab,Q) )
    ( tmps.at("bin_bb_voQ")(bb,jb,Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t2_2p.at("bbbb")(ab,bb,jb,ib) )
    ( scalars.at("0006")()  = tmps.at("bin_bb_voQ")(bb,jb,Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    ( tmps.at("bin_aa_voQ")(aa,ia,Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t2.at("abab")(aa,bb,ia,jb) )
    ( scalars.at("0007")()  = tmps.at("bin_aa_voQ")(aa,ia,Q) * chol.at("aa_ovQ")(ia,aa,Q) )
    ( tmps.at("bin_aa_voQ")(ba,ia,Q)  = chol.at("bb_ovQ")(jb,ab,Q) * t2_1p.at("abab")(ba,ab,ia,jb) )
    ( scalars.at("0008")()  = tmps.at("bin_aa_voQ")(ba,ia,Q) * chol.at("aa_ovQ")(ia,ba,Q) )
    ( tmps.at("bin_aa_voQ")(ba,ja,Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t2_2p.at("abab")(ba,ab,ja,ib) )
    ( scalars.at("0009")()  = tmps.at("bin_aa_voQ")(ba,ja,Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( tmps.at("bin_aa_voQ")(aa,ja,Q)  = chol.at("aa_ovQ")(ia,ba,Q) * t2.at("aaaa")(aa,ba,ja,ia) )
    ( scalars.at("0010")()  = tmps.at("bin_aa_voQ")(aa,ja,Q) * chol.at("aa_ovQ")(ja,aa,Q) )
    ( tmps.at("bin_aa_voQ")(ba,ja,Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t2_1p.at("aaaa")(aa,ba,ja,ia) )
    ( scalars.at("0011")()  = tmps.at("bin_aa_voQ")(ba,ja,Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( tmps.at("bin_aa_voQ")(ba,ja,Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t2_2p.at("aaaa")(aa,ba,ja,ia) )
    ( scalars.at("0012")()  = tmps.at("bin_aa_voQ")(ba,ja,Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( tmps.at("bin_aa_voQ")(aa,ja,Q)  = chol.at("aa_ovQ")(ia,ba,Q) * t2_1p.at("aaaa")(aa,ba,ja,ia) )
    ( scalars.at("0013")()  = tmps.at("bin_aa_voQ")(aa,ja,Q) * chol.at("aa_ovQ")(ja,aa,Q) )
    ( tmps.at("bin_aa_voQ")(aa,ja,Q)  = chol.at("aa_ovQ")(ia,ba,Q) * t2_2p.at("aaaa")(aa,ba,ja,ia) )
    ( scalars.at("0014")()  = tmps.at("bin_aa_voQ")(aa,ja,Q) * chol.at("aa_ovQ")(ja,aa,Q) )
    ( tmps.at("bin_aa_voQ")(ba,ja,Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t2.at("aaaa")(aa,ba,ja,ia) )
    ( scalars.at("0015")()  = tmps.at("bin_aa_voQ")(ba,ja,Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( tmps.at("bin_bb_ooQ")(ib,jb,Q)  = chol.at("bb_ovQ")(ib,bb,Q) * t1_1p.at("bb")(bb,jb) )
    ( tmps.at("bin_bb_vo")(ab,ib)  = tmps.at("bin_bb_ooQ")(ib,jb,Q) * chol.at("bb_ovQ")(jb,ab,Q) )
    ( scalars.at("0016")()  = tmps.at("bin_bb_vo")(ab,ib) * t1_1p.at("bb")(ab,ib) )
    ( tmps.at("bin_bb_ooQ")(ib,jb,Q)  = chol.at("bb_ovQ")(ib,bb,Q) * t1_1p.at("bb")(bb,jb) )
    ( tmps.at("bin_bb_vo")(ab,ib)  = tmps.at("bin_bb_ooQ")(ib,jb,Q) * chol.at("bb_ovQ")(jb,ab,Q) )
    ( scalars.at("0017")()  = tmps.at("bin_bb_vo")(ab,ib) * t1.at("bb")(ab,ib) )
    ( tmps.at("bin_bb_ooQ")(ib,jb,Q)  = chol.at("bb_ovQ")(ib,bb,Q) * t1_2p.at("bb")(bb,jb) )
    ( tmps.at("bin_bb_vo")(ab,ib)  = tmps.at("bin_bb_ooQ")(ib,jb,Q) * chol.at("bb_ovQ")(jb,ab,Q) )
    ( scalars.at("0018")()  = tmps.at("bin_bb_vo")(ab,ib) * t1.at("bb")(ab,ib) )
    ( tmps.at("bin_bb_ooQ")(ib,jb,Q)  = chol.at("bb_ovQ")(ib,bb,Q) * t1.at("bb")(bb,jb) )
    ( tmps.at("bin_bb_vo")(ab,ib)  = tmps.at("bin_bb_ooQ")(ib,jb,Q) * chol.at("bb_ovQ")(jb,ab,Q) )
    ( scalars.at("0019")()  = tmps.at("bin_bb_vo")(ab,ib) * t1.at("bb")(ab,ib) )
    ( tmps.at("bin_aa_ooQ")(ia,ja,Q)  = chol.at("aa_ovQ")(ia,ba,Q) * t1.at("aa")(ba,ja) )
    ( tmps.at("bin_aa_vo")(aa,ia)  = tmps.at("bin_aa_ooQ")(ia,ja,Q) * chol.at("aa_ovQ")(ja,aa,Q) )
    ( scalars.at("0020")()  = tmps.at("bin_aa_vo")(aa,ia) * t1.at("aa")(aa,ia) )
    ( tmps.at("bin_aa_ooQ")(ia,ja,Q)  = chol.at("aa_ovQ")(ia,ba,Q) * t1_1p.at("aa")(ba,ja) )
    ( tmps.at("bin_aa_vo")(aa,ia)  = tmps.at("bin_aa_ooQ")(ia,ja,Q) * chol.at("aa_ovQ")(ja,aa,Q) )
    ( scalars.at("0021")()  = tmps.at("bin_aa_vo")(aa,ia) * t1_1p.at("aa")(aa,ia) )
    ( tmps.at("bin_aa_ooQ")(ia,ja,Q)  = chol.at("aa_ovQ")(ia,ba,Q) * t1_1p.at("aa")(ba,ja) )
    ( tmps.at("bin_aa_vo")(aa,ia)  = tmps.at("bin_aa_ooQ")(ia,ja,Q) * chol.at("aa_ovQ")(ja,aa,Q) )
    ( scalars.at("0022")()  = tmps.at("bin_aa_vo")(aa,ia) * t1.at("aa")(aa,ia) )
    ( tmps.at("bin_aa_ooQ")(ia,ja,Q)  = chol.at("aa_ovQ")(ia,ba,Q) * t1_2p.at("aa")(ba,ja) )
    ( tmps.at("bin_aa_vo")(aa,ia)  = tmps.at("bin_aa_ooQ")(ia,ja,Q) * chol.at("aa_ovQ")(ja,aa,Q) )
    ( scalars.at("0023")()  = tmps.at("bin_aa_vo")(aa,ia) * t1.at("aa")(aa,ia) )
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1.at("bb")(ab,ib) )
    ( tmps.at("bin_bb_vo")(bb,jb)  = tmps.at("bin_Q")(Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    ( scalars.at("0024")()  = tmps.at("bin_bb_vo")(bb,jb) * t1.at("bb")(bb,jb) )
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1.at("bb")(ab,ib) )
    ( tmps.at("bin_bb_vo")(bb,jb)  = tmps.at("bin_Q")(Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    ( scalars.at("0025")()  = tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("bb")(bb,jb) )
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1.at("bb")(ab,ib) )
    ( tmps.at("bin_bb_vo")(bb,jb)  = tmps.at("bin_Q")(Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    ( scalars.at("0026")()  = tmps.at("bin_bb_vo")(bb,jb) * t1_2p.at("bb")(bb,jb) )
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1_1p.at("bb")(ab,ib) )
    ( tmps.at("bin_bb_vo")(bb,jb)  = tmps.at("bin_Q")(Q) * chol.at("bb_ovQ")(jb,bb,Q) )
    ( scalars.at("0027")()  = tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("bb")(bb,jb) )
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1.at("bb")(ab,ib) )
    ( tmps.at("bin_aa_vo")(ba,ja)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( scalars.at("0028")()  = tmps.at("bin_aa_vo")(ba,ja) * t1_1p.at("aa")(ba,ja) )
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1_1p.at("bb")(ab,ib) )
    ( tmps.at("bin_aa_vo")(ba,ja)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( scalars.at("0029")()  = tmps.at("bin_aa_vo")(ba,ja) * t1_1p.at("aa")(ba,ja) )
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1.at("bb")(ab,ib) )
    ( tmps.at("bin_aa_vo")(ba,ja)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( scalars.at("0030")()  = tmps.at("bin_aa_vo")(ba,ja) * t1_2p.at("aa")(ba,ja) )
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t1_2p.at("bb")(bb,jb) )
    ( tmps.at("bin_aa_vo")(aa,ia)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ia,aa,Q) )
    ( scalars.at("0031")()  = tmps.at("bin_aa_vo")(aa,ia) * t1.at("aa")(aa,ia) )
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(ib,ab,Q) * t1.at("bb")(ab,ib) )
    ( tmps.at("bin_aa_vo")(ba,ja)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( scalars.at("0032")()  = tmps.at("bin_aa_vo")(ba,ja) * t1.at("aa")(ba,ja) )
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(jb,bb,Q) * t1_1p.at("bb")(bb,jb) )
    ( tmps.at("bin_aa_vo")(aa,ia)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ia,aa,Q) )
    ( scalars.at("0033")()  = tmps.at("bin_aa_vo")(aa,ia) * t1.at("aa")(aa,ia) )
    ( tmps.at("bin_Q")(Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t1.at("aa")(aa,ia) )
    ( tmps.at("bin_aa_vo")(ba,ja)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( scalars.at("0034")()  = tmps.at("bin_aa_vo")(ba,ja) * t1_1p.at("aa")(ba,ja) )
    ( tmps.at("bin_Q")(Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t1.at("aa")(aa,ia) )
    ( tmps.at("bin_aa_vo")(ba,ja)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( scalars.at("0035")()  = tmps.at("bin_aa_vo")(ba,ja) * t1_2p.at("aa")(ba,ja) )
    ( tmps.at("bin_Q")(Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t1_1p.at("aa")(aa,ia) )
    ( tmps.at("bin_aa_vo")(ba,ja)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( scalars.at("0036")()  = tmps.at("bin_aa_vo")(ba,ja) * t1_1p.at("aa")(ba,ja) )
    ( tmps.at("bin_Q")(Q)  = chol.at("aa_ovQ")(ia,aa,Q) * t1.at("aa")(aa,ia) )
    ( tmps.at("bin_aa_vo")(ba,ja)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ja,ba,Q) )
    ( scalars.at("0037")()  = tmps.at("bin_aa_vo")(ba,ja) * t1.at("aa")(ba,ja) )
    ( scalars.at("0038")()  = f.at("bb_ov")(ib,ab) * t1.at("bb")(ab,ib) )
    ( scalars.at("0039")()  = f.at("bb_ov")(ib,ab) * t1_1p.at("bb")(ab,ib) )
    ( scalars.at("0040")()  = f.at("bb_ov")(ib,ab) * t1_2p.at("bb")(ab,ib) )
    ( scalars.at("0041")()  = f.at("aa_ov")(ia,aa) * t1.at("aa")(aa,ia) )
    ( scalars.at("0042")()  = f.at("aa_ov")(ia,aa) * t1_2p.at("aa")(aa,ia) )
    ( scalars.at("0043")()  = f.at("aa_ov")(ia,aa) * t1_1p.at("aa")(aa,ia) )
    ( scalars.at("0044")()  = dp.at("bb_ov")(ib,ab) * t1.at("bb")(ab,ib) )
    ( scalars.at("0045")()  = t0_1p * scalars.at("0044")() )
    ( scalars.at("0046")()  = dp.at("aa_ov")(ia,aa) * t1.at("aa")(aa,ia) )
    ( scalars.at("0047")()  = t0_1p * scalars.at("0046")() )
    ( scalars.at("0048")()  = t0_1p * w0 )
    ( scalars.at("0049")()  = t0_2p * scalars.at("0044")() )
    ( scalars.at("0050")()  = dp.at("aa_ov")(ia,aa) * t1_1p.at("aa")(aa,ia) )
    ( scalars.at("0051")()  = t0_1p * scalars.at("0050")() )
    ( scalars.at("0052")()  = dp.at("bb_ov")(ib,ab) * t1_1p.at("bb")(ab,ib) )
    ( scalars.at("0053")()  = t0_1p * scalars.at("0052")() )
    ( scalars.at("0054")()  = t0_2p * scalars.at("0046")() )
    ( scalars.at("0055")()  = t0_2p * w0 )
    ( scalars.at("0056")()  = dp.at("bb_ov")(ib,ab) * t1_2p.at("bb")(ab,ib) )
    ( scalars.at("0057")()  = t0_1p * scalars.at("0056")() )
    ( scalars.at("0058")()  = dp.at("aa_ov")(ia,aa) * t1_2p.at("aa")(aa,ia) )
    ( scalars.at("0059")()  = t0_1p * scalars.at("0058")() )
    ( scalars.at("0060")()  = t0_2p * scalars.at("0050")() )
    ( scalars.at("0061")()  = t0_2p * scalars.at("0052")() )
    
    // r2_1p.at("abab")  = +1.00 t2_1p_abab(a,b,i,j) w0 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb)  = w0 * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r1.at("aa")  = +1.00 f_aa(a,i) 
    ( r1.at("aa")(aa,ia)  = f.at("aa_vo")(aa,ia) )
    
    // energy()  = +0.250 <j,i||a,b>_bbbb t2_bbbb(a,b,j,i) 
    ( energy()  = 0.250 * scalars.at("0001")() )
    
    // r2_2p.at("abab")  = +4.00 t2_2p_abab(a,b,i,j) w0 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb)  = 4.00 * w0 * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r2.at("abab")  = +1.00 d-_aa(a,i) t1_1p_bb(b,j) 
    // flops: o2v2  = o2v2
    //  mems: o2v2  = o2v2
    ( r2.at("abab")(aa,bb,ia,jb)  = dp.at("aa_vo")(aa,ia) * t1_1p.at("bb")(bb,jb) )
    
    // r1_1p.at("aa")  = +1.00 d+_aa(a,i) 
    ( r1_1p.at("aa")(aa,ia)  = dp.at("aa_vo")(aa,ia) )
    
    // r1_2p.at("aa")  = +4.00 t1_2p_aa(a,i) w0 
    // flops: o1v1  = o1v1
    //  mems: o1v1  = o1v1
    ( r1_2p.at("aa")(aa,ia)  = 4.00 * w0 * t1_2p.at("aa")(aa,ia) )
    
    // r0_2p()  = +0.50 <j,i||a,b>_bbbb t2_2p_bbbb(a,b,j,i) 
    ( r0_2p()  = 0.50 * scalars.at("0002")() )
    
    // r0_1p()  = +0.250 <j,i||a,b>_bbbb t2_1p_bbbb(a,b,j,i) 
    ( r0_1p()  = -0.250 * scalars.at("0003")() )
    
    // energy() += +0.250 <j,i||a,b>_bbbb t2_bbbb(a,b,j,i) 
    ( energy() -= 0.250 * scalars.at("0004")() )
    
    // energy() += +0.250 <j,i||a,b>_abab t2_abab(a,b,j,i) 
    //        += +0.250 <i,j||a,b>_abab t2_abab(a,b,i,j) 
    //        += +0.250 <j,i||b,a>_abab t2_abab(b,a,j,i) 
    //        += +0.250 <i,j||b,a>_abab t2_abab(b,a,i,j) 
    ( energy() += scalars.at("0007")() )
    
    // energy() += +0.250 <j,i||a,b>_aaaa t2_aaaa(a,b,j,i) 
    ( energy() += 0.250 * scalars.at("0010")() )
    
    // energy() += +0.250 <j,i||a,b>_aaaa t2_aaaa(a,b,j,i) 
    ( energy() -= 0.250 * scalars.at("0015")() )
    
    // energy() += -0.50 <j,i||a,b>_bbbb t1_bb(a,i) t1_bb(b,j) 
    ( energy() -= 0.50 * scalars.at("0019")() )
    
    // energy() += -0.50 <j,i||a,b>_aaaa t1_aa(a,i) t1_aa(b,j) 
    ( energy() -= 0.50 * scalars.at("0020")() )
    
    // energy() += -0.50 <j,i||a,b>_bbbb t1_bb(a,i) t1_bb(b,j) 
    ( energy() += 0.50 * scalars.at("0024")() )
    
    // energy() += +0.50 <i,j||a,b>_abab t1_aa(a,i) t1_bb(b,j) 
    //        += +0.50 <j,i||b,a>_abab t1_bb(a,i) t1_aa(b,j) 
    ( energy() += scalars.at("0032")() )
    
    // energy() += -0.50 <j,i||a,b>_aaaa t1_aa(a,i) t1_aa(b,j) 
    ( energy() += 0.50 * scalars.at("0037")() )
    
    // energy() += +1.00 f_bb(i,a) t1_bb(a,i) 
    ( energy() += scalars.at("0038")() )
    
    // energy() += +1.00 f_aa(i,a) t1_aa(a,i) 
    ( energy() += scalars.at("0041")() )
    
    // energy() += +1.00 d-_bb(i,a) t1_bb(a,i) t0_1p 
    ( energy() += scalars.at("0045")() )
    
    // energy() += +1.00 d-_aa(i,a) t1_aa(a,i) t0_1p 
    ( energy() += scalars.at("0047")() )
    
    // energy() += +1.00 d-_aa(i,a) t1_1p_aa(a,i) 
    ( energy() += scalars.at("0050")() )
    
    // energy() += +1.00 d-_bb(i,a) t1_1p_bb(a,i) 
    ( energy() += scalars.at("0052")() )
    
    // r0_1p() += +0.250 <j,i||a,b>_bbbb t2_1p_bbbb(a,b,j,i) 
    ( r0_1p() += 0.250 * scalars.at("0005")() )
    
    // r0_1p() += +0.250 <j,i||a,b>_abab t2_1p_abab(a,b,j,i) 
    //       += +0.250 <i,j||a,b>_abab t2_1p_abab(a,b,i,j) 
    //       += +0.250 <j,i||b,a>_abab t2_1p_abab(b,a,j,i) 
    //       += +0.250 <i,j||b,a>_abab t2_1p_abab(b,a,i,j) 
    ( r0_1p() += scalars.at("0008")() )
    
    // r0_1p() += +0.250 <j,i||a,b>_aaaa t2_1p_aaaa(a,b,j,i) 
    ( r0_1p() -= 0.250 * scalars.at("0011")() )
    
    // r0_1p() += +0.250 <j,i||a,b>_aaaa t2_1p_aaaa(a,b,j,i) 
    ( r0_1p() += 0.250 * scalars.at("0013")() )
    
    // r0_1p() += -1.00 <j,i||a,b>_bbbb t1_bb(a,i) t1_1p_bb(b,j) 
    ( r0_1p() -= scalars.at("0017")() )
    
    // r0_1p() += -1.00 <j,i||a,b>_aaaa t1_aa(a,i) t1_1p_aa(b,j) 
    ( r0_1p() -= scalars.at("0022")() )
    
    // r0_1p() += -1.00 <j,i||a,b>_bbbb t1_bb(a,i) t1_1p_bb(b,j) 
    ( r0_1p() += scalars.at("0025")() )
    
    // r0_1p() += +1.00 <j,i||b,a>_abab t1_bb(a,i) t1_1p_aa(b,j) 
    ( r0_1p() += scalars.at("0028")() )
    
    // r0_1p() += +1.00 <i,j||a,b>_abab t1_aa(a,i) t1_1p_bb(b,j) 
    ( r0_1p() += scalars.at("0033")() )
    
    // r0_1p() += -1.00 <j,i||a,b>_aaaa t1_aa(a,i) t1_1p_aa(b,j) 
    ( r0_1p() += scalars.at("0034")() )
    
    // r0_1p() += +1.00 f_bb(i,a) t1_1p_bb(a,i) 
    ( r0_1p() += scalars.at("0039")() )
    
    // r0_1p() += +1.00 f_aa(i,a) t1_1p_aa(a,i) 
    ( r0_1p() += scalars.at("0043")() )
    
    // r0_1p() += +1.00 d+_bb(i,a) t1_bb(a,i) 
    ( r0_1p() += scalars.at("0044")() )
    
    // r0_1p() += +1.00 d+_aa(i,a) t1_aa(a,i) 
    ( r0_1p() += scalars.at("0046")() )
    
    // r0_1p() += +1.00 t0_1p w0 
    ( r0_1p() += scalars.at("0048")() )
    
    // r0_1p() += +2.00 d-_bb(i,a) t1_bb(a,i) t0_2p 
    ( r0_1p() += 2.00 * scalars.at("0049")() )
    
    // r0_1p() += +1.00 d-_aa(i,a) t0_1p t1_1p_aa(a,i) 
    ( r0_1p() += scalars.at("0051")() )
    
    // r0_1p() += +1.00 d-_bb(i,a) t0_1p t1_1p_bb(a,i) 
    ( r0_1p() += scalars.at("0053")() )
    
    // r0_1p() += +2.00 d-_aa(i,a) t1_aa(a,i) t0_2p 
    ( r0_1p() += 2.00 * scalars.at("0054")() )
    
    // r0_1p() += +2.00 d-_bb(i,a) t1_2p_bb(a,i) 
    ( r0_1p() += 2.00 * scalars.at("0056")() )
    
    // r0_1p() += +2.00 d-_aa(i,a) t1_2p_aa(a,i) 
    ( r0_1p() += 2.00 * scalars.at("0058")() )
    
    // r0_2p() += +0.50 <j,i||a,b>_bbbb t2_2p_bbbb(a,b,j,i) 
    ( r0_2p() -= 0.50 * scalars.at("0006")() )
    
    // r0_2p() += +0.50 <j,i||a,b>_abab t2_2p_abab(a,b,j,i) 
    //       += +0.50 <i,j||a,b>_abab t2_2p_abab(a,b,i,j) 
    //       += +0.50 <j,i||b,a>_abab t2_2p_abab(b,a,j,i) 
    //       += +0.50 <i,j||b,a>_abab t2_2p_abab(b,a,i,j) 
    ( r0_2p() += 2.00 * scalars.at("0009")() )
    
    // r0_2p() += +0.50 <j,i||a,b>_aaaa t2_2p_aaaa(a,b,j,i) 
    ( r0_2p() -= 0.50 * scalars.at("0012")() )
    
    // r0_2p() += +0.50 <j,i||a,b>_aaaa t2_2p_aaaa(a,b,j,i) 
    ( r0_2p() += 0.50 * scalars.at("0014")() )
    
    // r0_2p() += -1.00 <j,i||a,b>_bbbb t1_1p_bb(a,i) t1_1p_bb(b,j) 
    ( r0_2p() -= scalars.at("0016")() )
    
    // r0_2p() += -2.00 <j,i||a,b>_bbbb t1_bb(a,i) t1_2p_bb(b,j) 
    ( r0_2p() -= 2.00 * scalars.at("0018")() )
    
    // r0_2p() += -1.00 <j,i||a,b>_aaaa t1_1p_aa(a,i) t1_1p_aa(b,j) 
    ( r0_2p() -= scalars.at("0021")() )
    
    // r0_2p() += -2.00 <j,i||a,b>_aaaa t1_aa(a,i) t1_2p_aa(b,j) 
    ( r0_2p() -= 2.00 * scalars.at("0023")() )
    
    // r0_2p() += -2.00 <j,i||a,b>_bbbb t1_bb(a,i) t1_2p_bb(b,j) 
    ( r0_2p() += 2.00 * scalars.at("0026")() )
    
    // r0_2p() += -1.00 <j,i||a,b>_bbbb t1_1p_bb(a,i) t1_1p_bb(b,j) 
    ( r0_2p() += scalars.at("0027")() )
    
    // r0_2p() += +1.00 <i,j||a,b>_abab t1_1p_aa(a,i) t1_1p_bb(b,j) 
    //       += +1.00 <j,i||b,a>_abab t1_1p_bb(a,i) t1_1p_aa(b,j) 
    ( r0_2p() += 2.00 * scalars.at("0029")() )
    
    // r0_2p() += +2.00 <j,i||b,a>_abab t1_bb(a,i) t1_2p_aa(b,j) 
    ( r0_2p() += 2.00 * scalars.at("0030")() )
    
    // r0_2p() += +2.00 <i,j||a,b>_abab t1_aa(a,i) t1_2p_bb(b,j) 
    ( r0_2p() += 2.00 * scalars.at("0031")() )
    
    // r0_2p() += -2.00 <j,i||a,b>_aaaa t1_aa(a,i) t1_2p_aa(b,j) 
    ( r0_2p() += 2.00 * scalars.at("0035")() )
    
    // r0_2p() += -1.00 <j,i||a,b>_aaaa t1_1p_aa(a,i) t1_1p_aa(b,j) 
    ( r0_2p() += scalars.at("0036")() )
    
    // r0_2p() += +2.00 f_bb(i,a) t1_2p_bb(a,i) 
    ( r0_2p() += 2.00 * scalars.at("0040")() )
    
    // r0_2p() += +2.00 f_aa(i,a) t1_2p_aa(a,i) 
    ( r0_2p() += 2.00 * scalars.at("0042")() )
    
    // r0_2p() += +2.00 d+_aa(i,a) t1_1p_aa(a,i) 
    ( r0_2p() += 2.00 * scalars.at("0050")() )
    
    // r0_2p() += +2.00 d+_bb(i,a) t1_1p_bb(a,i) 
    ( r0_2p() += 2.00 * scalars.at("0052")() )
    
    // r0_2p() += +4.00 t0_2p w0 
    ( r0_2p() += 4.00 * scalars.at("0055")() )
    
    // r0_2p() += +2.00 d-_bb(i,a) t0_1p t1_2p_bb(a,i) 
    ( r0_2p() += 2.00 * scalars.at("0057")() )
    
    // r0_2p() += +2.00 d-_aa(i,a) t0_1p t1_2p_aa(a,i) 
    ( r0_2p() += 2.00 * scalars.at("0059")() )
    
    // r0_2p() += +4.00 d-_aa(i,a) t1_1p_aa(a,i) t0_2p 
    ( r0_2p() += 4.00 * scalars.at("0060")() )
    
    // r0_2p() += +4.00 d-_bb(i,a) t1_1p_bb(a,i) t0_2p 
    ( r0_2p() += 4.00 * scalars.at("0061")() )
    
    // r1_1p.at("aa") += +1.00 t1_1p_aa(a,i) w0 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += w0 * t1_1p.at("aa")(aa,ia) )
    
    // r1_1p.at("aa") += +2.00 d-_aa(a,i) t0_2p 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 2.00 * t0_2p * dp.at("aa_vo")(aa,ia) )
    
    // r1_1p.at("aa") += +2.00 d-_bb(j,b) t1_bb(b,j) t1_2p_aa(a,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 2.00 * scalars.at("0044")() * t1_2p.at("aa")(aa,ia) )
    
    // r1_1p.at("aa") += +2.00 d-_aa(j,b) t1_aa(b,j) t1_2p_aa(a,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += 2.00 * scalars.at("0046")() * t1_2p.at("aa")(aa,ia) )
    
    // r1_1p.at("aa") += +1.00 d-_aa(j,b) t1_1p_aa(a,i) t1_1p_aa(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += scalars.at("0050")() * t1_1p.at("aa")(aa,ia) )
    
    // r1_1p.at("aa") += +1.00 d-_bb(j,b) t1_1p_aa(a,i) t1_1p_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += scalars.at("0052")() * t1_1p.at("aa")(aa,ia) )
    
    // r1_1p.at("aa") += -1.00 f_aa(j,i) t1_1p_aa(a,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= f.at("aa_oo")(ja,ia) * t1_1p.at("aa")(aa,ja) )
    
    // r1_1p.at("aa") += +1.00 f_aa(a,b) t1_1p_aa(b,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += f.at("aa_vv")(aa,ba) * t1_1p.at("aa")(ba,ia) )
    
    // r1_1p.at("aa") += -1.00 f_aa(j,b) t2_1p_aaaa(b,a,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= f.at("aa_ov")(ja,ba) * t2_1p.at("aaaa")(ba,aa,ia,ja) )
    
    // r1_1p.at("aa") += +1.00 f_bb(j,b) t2_1p_abab(a,b,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += f.at("bb_ov")(jb,bb) * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r1_1p.at("aa") += -0.50 <j,a||b,c>_aaaa t2_1p_aaaa(b,c,i,j) 
    // flops: o1v1 += o2v2Q1 o1v2Q1
    //  mems: o1v1 += o1v1Q1 o1v1
    ( tmps.at("bin_aa_voQ")(ba,ia,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t2_1p.at("aaaa")(ba,ca,ia,ja) )
    ( r1_1p.at("aa")(aa,ia) += 0.50 * tmps.at("bin_aa_voQ")(ba,ia,Q) * chol.at("aa_vvQ")(aa,ba,Q) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,c>_aaaa t2_aaaa(b,a,i,j) t1_1p_aa(c,k) 
    // flops: o1v1 += o2v1Q1 o2v1Q1 o2v2
    //  mems: o1v1 += o2v0Q1 o1v1 o1v1
    ( tmps.at("bin_aa_ooQ")(ja,ka,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t1_1p.at("aa")(ca,ka) )
    ( tmps.at("bin_aa_vo")(ba,ja)  = tmps.at("bin_aa_ooQ")(ja,ka,Q) * chol.at("aa_ovQ")(ka,ba,Q) )
    ( r1_1p.at("aa")(aa,ia) += tmps.at("bin_aa_vo")(ba,ja) * t2.at("aaaa")(ba,aa,ia,ja) )
    
    // r1_2p.at("aa") += +4.00 d-_aa(j,b) t1_1p_aa(b,j) t1_2p_aa(a,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 4.00 * scalars.at("0050")() * t1_2p.at("aa")(aa,ia) )
    
    // r1_2p.at("aa") += +4.00 d-_bb(j,b) t1_1p_bb(b,j) t1_2p_aa(a,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 4.00 * scalars.at("0052")() * t1_2p.at("aa")(aa,ia) )
    
    // r1_2p.at("aa") += +2.00 d-_bb(j,b) t1_1p_aa(a,i) t1_2p_bb(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * scalars.at("0056")() * t1_1p.at("aa")(aa,ia) )
    
    // r1_2p.at("aa") += +2.00 d-_aa(j,b) t1_1p_aa(a,i) t1_2p_aa(b,j) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * scalars.at("0058")() * t1_1p.at("aa")(aa,ia) )
    
    // r1_2p.at("aa") += -2.00 f_aa(j,i) t1_2p_aa(a,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * f.at("aa_oo")(ja,ia) * t1_2p.at("aa")(aa,ja) )
    
    // r1_2p.at("aa") += +2.00 f_aa(a,b) t1_2p_aa(b,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * f.at("aa_vv")(aa,ba) * t1_2p.at("aa")(ba,ia) )
    
    // r1_2p.at("aa") += -2.00 f_aa(j,b) t2_2p_aaaa(b,a,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * f.at("aa_ov")(ja,ba) * t2_2p.at("aaaa")(ba,aa,ia,ja) )
    
    // r1_2p.at("aa") += +2.00 f_bb(j,b) t2_2p_abab(a,b,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * f.at("bb_ov")(jb,bb) * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r1_2p.at("aa") += +2.00 <j,a||b,i>_aaaa t1_2p_aa(b,j) 
    // flops: o1v1 += o2v1Q1 o1v2Q1
    //  mems: o1v1 += o1v1Q1 o1v1
    ( tmps.at("bin_aa_voQ")(ba,ia,Q)  = chol.at("aa_ooQ")(ja,ia,Q) * t1_2p.at("aa")(ba,ja) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_aa_voQ")(ba,ia,Q) * chol.at("aa_vvQ")(aa,ba,Q) )
    
    // r1_2p.at("aa") += -1.00 <j,a||b,c>_aaaa t2_2p_aaaa(b,c,i,j) 
    // flops: o1v1 += o2v2Q1 o1v2Q1
    //  mems: o1v1 += o1v1Q1 o1v1
    ( tmps.at("bin_aa_voQ")(ba,ia,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t2_2p.at("aaaa")(ba,ca,ia,ja) )
    ( r1_2p.at("aa")(aa,ia) += tmps.at("bin_aa_voQ")(ba,ia,Q) * chol.at("aa_vvQ")(aa,ba,Q) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t2_aaaa(b,a,i,j) t1_2p_aa(c,k) 
    // flops: o1v1 += o2v1Q1 o2v1Q1 o2v2
    //  mems: o1v1 += o2v0Q1 o1v1 o1v1
    ( tmps.at("bin_aa_ooQ")(ja,ka,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t1_2p.at("aa")(ca,ka) )
    ( tmps.at("bin_aa_vo")(ba,ja)  = tmps.at("bin_aa_ooQ")(ja,ka,Q) * chol.at("aa_ovQ")(ka,ba,Q) )
    ( r1_2p.at("aa")(aa,ia) += 2.00 * tmps.at("bin_aa_vo")(ba,ja) * t2.at("aaaa")(ba,aa,ia,ja) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_bbbb t1_bb(b,j) t2_2p_abab(a,c,i,k) 
    // flops: o1v1 += o2v1Q1 o2v1Q1 o2v2
    //  mems: o1v1 += o2v0Q1 o1v1 o1v1
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,bb,Q) * t1.at("bb")(bb,jb) )
    ( tmps.at("bin_bb_vo")(cb,kb)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("bb_ovQ")(jb,cb,Q) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_bb_vo")(cb,kb) * t2_2p.at("abab")(aa,cb,ia,kb) )
    
    // r1_2p.at("aa") += -2.00 <j,a||b,c>_aaaa t1_aa(b,i) t1_2p_aa(c,j) 
    // flops: o1v1 += o2v1Q1 o2v1Q1 o1v2Q1
    //  mems: o1v1 += o2v0Q1 o1v1Q1 o1v1
    ( tmps.at("bin_aa_ooQ")(ia,ja,Q)  = chol.at("aa_ovQ")(ja,ba,Q) * t1.at("aa")(ba,ia) )
    ( tmps.at("bin_aa_voQ")(ca,ia,Q)  = tmps.at("bin_aa_ooQ")(ia,ja,Q) * t1_2p.at("aa")(ca,ja) )
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * tmps.at("bin_aa_voQ")(ca,ia,Q) * chol.at("aa_vvQ")(aa,ca,Q) )
    
    // r1.at("aa") += +1.00 d-_aa(a,i) t0_1p 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += t0_1p * dp.at("aa_vo")(aa,ia) )
    
    // r1.at("aa") += +1.00 d-_bb(j,b) t1_bb(b,j) t1_1p_aa(a,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += scalars.at("0044")() * t1_1p.at("aa")(aa,ia) )
    
    // r1.at("aa") += +1.00 d-_aa(j,b) t1_aa(b,j) t1_1p_aa(a,i) 
    // flops: o1v1 += o1v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += scalars.at("0046")() * t1_1p.at("aa")(aa,ia) )
    
    // r1.at("aa") += -1.00 f_aa(j,i) t1_aa(a,j) 
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= f.at("aa_oo")(ja,ia) * t1.at("aa")(aa,ja) )
    
    // r1.at("aa") += +1.00 f_aa(a,b) t1_aa(b,i) 
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += f.at("aa_vv")(aa,ba) * t1.at("aa")(ba,ia) )
    
    // r1.at("aa") += -1.00 f_aa(j,b) t2_aaaa(b,a,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) -= f.at("aa_ov")(ja,ba) * t2.at("aaaa")(ba,aa,ia,ja) )
    
    // r1.at("aa") += +1.00 f_bb(j,b) t2_abab(a,b,i,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1.at("aa")(aa,ia) += f.at("bb_ov")(jb,bb) * t2.at("abab")(aa,bb,ia,jb) )
    
    // r1.at("aa") += -0.50 <j,a||b,c>_aaaa t2_aaaa(b,c,i,j) 
    // flops: o1v1 += o2v2Q1 o1v2Q1
    //  mems: o1v1 += o1v1Q1 o1v1
    ( tmps.at("bin_aa_voQ")(ba,ia,Q)  = chol.at("aa_ovQ")(ja,ca,Q) * t2.at("aaaa")(ba,ca,ia,ja) )
    ( r1.at("aa")(aa,ia) += 0.50 * tmps.at("bin_aa_voQ")(ba,ia,Q) * chol.at("aa_vvQ")(aa,ba,Q) )
    
    // r1.at("aa") += +1.00 <k,j||b,c>_aaaa t2_aaaa(c,a,i,k) t1_aa(b,j) 
    // flops: o1v1 += o2v1Q1 o2v1Q1 o2v2
    //  mems: o1v1 += o2v0Q1 o1v1 o1v1
    ( tmps.at("bin_aa_ooQ")(ja,ka,Q)  = chol.at("aa_ovQ")(ka,ba,Q) * t1.at("aa")(ba,ja) )
    ( tmps.at("bin_aa_vo")(ca,ka)  = tmps.at("bin_aa_ooQ")(ja,ka,Q) * chol.at("aa_ovQ")(ja,ca,Q) )
    ( r1.at("aa")(aa,ia) += tmps.at("bin_aa_vo")(ca,ka) * t2.at("aaaa")(ca,aa,ia,ka) )
    
    // r2_1p.at("abab") += +2.00 d-_bb(b,j) t1_2p_aa(a,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("aa")(aa,ia) * dp.at("bb_vo")(bb,jb) )
    
    // r2_1p.at("abab") += +2.00 d-_aa(a,i) t1_2p_bb(b,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.00 * dp.at("aa_vo")(aa,ia) * t1_2p.at("bb")(bb,jb) )
    
    // r2_1p.at("abab") += +2.00 d-_bb(k,c) t1_bb(c,k) t2_2p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.00 * scalars.at("0044")() * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r2_1p.at("abab") += +2.00 d-_aa(k,c) t1_aa(c,k) t2_2p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.00 * scalars.at("0046")() * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 d-_aa(k,c) t2_1p_abab(a,b,i,j) t1_1p_aa(c,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += scalars.at("0050")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 d-_bb(k,c) t2_1p_abab(a,b,i,j) t1_1p_bb(c,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += scalars.at("0052")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2_1p.at("abab") += -1.00 f_aa(k,i) t2_1p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= f.at("aa_oo")(ka,ia) * t2_1p.at("abab")(aa,bb,ka,jb) )
    
    // r2_1p.at("abab") += -1.00 f_bb(k,j) t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,bb,ia,kb) * f.at("bb_oo")(kb,jb) )
    
    // r2_1p.at("abab") += +1.00 f_aa(a,c) t2_1p_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += f.at("aa_vv")(aa,ca) * t2_1p.at("abab")(ca,bb,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 f_bb(b,c) t2_1p_abab(a,c,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2_1p.at("abab")(aa,cb,ia,jb) * f.at("bb_vv")(bb,cb) )
    
    // r2_1p.at("abab") += -1.00 d-_bb(k,j) t1_1p_aa(a,i) t1_1p_bb(b,k) 
    // flops: o2v2 += o2v1 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_oo")(kb,jb) * t1_1p.at("bb")(bb,kb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_1p.at("abab") += +1.00 d-_bb(b,c) t1_1p_aa(a,i) t1_1p_bb(c,j) 
    // flops: o2v2 += o1v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_vv")(bb,cb) * t1_1p.at("bb")(cb,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_1p.at("abab") += -2.00 d-_bb(k,c) t2_bbbb(c,b,j,k) t1_2p_aa(a,i) 
    // flops: o2v2 += o2v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_ov")(kb,cb) * t2.at("bbbb")(cb,bb,jb,kb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_2p.at("aa")(aa,ia) )
    
    // r2_1p.at("abab") += -1.00 d-_bb(k,c) t1_1p_aa(a,i) t2_1p_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_ov")(kb,cb) * t2_1p.at("bbbb")(cb,bb,jb,kb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_1p.at("abab") += -1.00 f_bb(k,c) t1_bb(b,k) t2_1p_abab(a,c,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = f.at("bb_ov")(kb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += -1.00 <a,k||i,j>_abab t1_1p_bb(b,k) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = chol.at("aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += +1.00 <k,a||c,i>_aaaa t2_1p_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin_aaaa_vvoo")(aa,ca,ia,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("aa_vvQ")(aa,ca,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aaaa_vvoo")(aa,ca,ia,ka) * t2_1p.at("abab")(ca,bb,ka,jb) )
    
    // r2_1p.at("abab") += -1.00 <a,k||i,c>_abab t1_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1_1p.at("bb")(cb,jb) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += -1.00 <k,b||c,j>_abab t1_aa(c,i) t1_1p_aa(a,k) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aa_ooQ")(ia,ka,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += +0.250 <l,k||c,d>_abab t2_abab(a,b,l,k) t2_1p_abab(c,d,i,j) 
    //            += +0.250 <l,k||d,c>_abab t2_abab(a,b,l,k) t2_1p_abab(d,c,i,j) 
    //            += +0.250 <k,l||c,d>_abab t2_abab(a,b,k,l) t2_1p_abab(c,d,i,j) 
    //            += +0.250 <k,l||d,c>_abab t2_abab(a,b,k,l) t2_1p_abab(d,c,i,j) 
    // flops: o2v2 += o2v2Q1 o4v2 o4v2
    //  mems: o2v2 += o2v2 o4v0 o2v2
    ( tmps.at("bin_abab_vvoo")(ca,db,la,kb)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin_aabb_oooo")(ia,la,jb,kb)  = tmps.at("bin_abab_vvoo")(ca,db,la,kb) * t2_1p.at("abab")(ca,db,ia,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_aabb_oooo")(ia,la,jb,kb) * t2.at("abab")(aa,bb,la,kb) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,j>_bbbb t1_bb(b,k) t2_1p_abab(a,c,i,l) 
    // flops: o2v2 += o3v1Q1 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin_bbbb_vooo")(cb,jb,kb,lb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bbbb_vooo")(cb,jb,kb,lb) * t2_1p.at("abab")(aa,cb,ia,lb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += +1.00 <k,a||c,d>_aaaa t2_abab(c,b,k,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o2v1Q1 o2v2Q1 o3v3
    //  mems: o2v2 += o2v0Q1 o2v2 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,da,Q) * t1_1p.at("aa")(da,ia) )
    ( tmps.at("bin_aaaa_vvoo")(aa,ca,ia,ka)  = tmps.at("bin_aa_ooQ")(ia,ka,Q) * chol.at("aa_vvQ")(aa,ca,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aaaa_vvoo")(aa,ca,ia,ka) * t2.at("abab")(ca,bb,ka,jb) )
    
    // r2_1p.at("abab") += -1.00 <k,a||c,d>_aaaa t1_aa(c,i) t2_1p_abab(d,b,k,j) 
    // flops: o2v2 += o2v1Q1 o2v2Q1 o3v3
    //  mems: o2v2 += o2v0Q1 o2v2 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin_aaaa_vvoo")(aa,da,ia,ka)  = tmps.at("bin_aa_ooQ")(ia,ka,Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aaaa_vvoo")(aa,da,ia,ka) * t2_1p.at("abab")(da,bb,ka,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1 o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2 o2v2
    ( tmps.at("bin2_aaaa_vvoo")(ca,da,ka,la)  = chol.at("aa_ovQ")(ka,da,Q) * chol.at("aa_ovQ")(la,ca,Q) )
    ( tmps.at("bin_aaaa_vvoo")(aa,da,ia,la)  = tmps.at("bin2_aaaa_vvoo")(ca,da,ka,la) * t2.at("aaaa")(ca,aa,ia,ka) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_aaaa_vvoo")(aa,da,ia,la) * t2_1p.at("abab")(da,bb,la,jb) )
    
    // r2_1p.at("abab") += -1.00 <a,k||c,d>_abab t2_abab(c,b,i,k) t1_1p_bb(d,j) 
    // flops: o2v2 += o2v1Q1 o2v2Q1 o3v3
    //  mems: o2v2 += o2v0Q1 o2v2 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,db,Q) * t1_1p.at("bb")(db,jb) )
    ( tmps.at("bin_aabb_vvoo")(aa,ca,jb,kb)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("aa_vvQ")(aa,ca,Q) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vvoo")(aa,ca,jb,kb) * t2.at("abab")(ca,bb,ia,kb) )
    
    // r2_1p.at("abab") += -0.50 <k,b||c,d>_abab t1_aa(a,k) t2_1p_abab(c,d,i,j) 
    //            += -0.50 <k,b||d,c>_abab t1_aa(a,k) t2_1p_abab(d,c,i,j) 
    // flops: o2v2 += o1v3Q1 o3v3 o3v2
    //  mems: o2v2 += o1v3 o3v1 o2v2
    ( tmps.at("bin_abba_vvvo")(ca,bb,db,ka)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("bb_vvQ")(bb,db,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_abba_vvvo")(ca,bb,db,ka) * t2_1p.at("abab")(ca,db,ia,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t1_aa(a,k) t2_abab(c,b,i,j) t1_1p_aa(d,l) 
    // flops: o2v2 += o1v1Q1 o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o0v0Q1 o1v1 o3v1 o2v2
    ( tmps.at("bin_Q")(Q)  = chol.at("aa_ovQ")(la,da,Q) * t1_1p.at("aa")(da,la) )
    ( tmps.at("bin_aa_vo")(ca,ka)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aa_vo")(ca,ka) * t2.at("abab")(ca,bb,ia,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += -1.00 <l,k||d,c>_abab t1_aa(a,l) t1_bb(c,k) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o0v0Q1 o1v1 o3v1 o2v2
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin_aa_vo")(da,la)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin_aa_vo")(da,la) * t2_1p.at("abab")(da,bb,ia,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_baab_vooo")(bb,ia,la,jb) * t1.at("aa")(aa,la) )
    
    // r2_1p.at("abab") += -1.00 <l,k||d,c>_abab t2_abab(a,c,i,j) t1_bb(b,k) t1_1p_aa(d,l) 
    // flops: o2v2 += o1v1Q1 o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o0v0Q1 o1v1 o3v1 o2v2
    ( tmps.at("bin_Q")(Q)  = chol.at("aa_ovQ")(la,da,Q) * t1_1p.at("aa")(da,la) )
    ( tmps.at("bin_bb_vo")(cb,kb)  = tmps.at("bin_Q")(Q) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bb_vo")(cb,kb) * t2.at("abab")(aa,cb,ia,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,j) t1_bb(b,k) t1_1p_bb(d,l) 
    // flops: o2v2 += o1v1Q1 o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o0v0Q1 o1v1 o3v1 o2v2
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(lb,db,Q) * t1_1p.at("bb")(db,lb) )
    ( tmps.at("bin_bb_vo")(cb,kb)  = tmps.at("bin_Q")(Q) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bb_vo")(cb,kb) * t2.at("abab")(aa,cb,ia,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t1_aa(a,k) t1_aa(c,i) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o4v2 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o3v1 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,la,Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin_aaaa_vooo")(da,ia,ka,la)  = tmps.at("bin_aa_ooQ")(ia,la,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aaaa_vooo")(da,ia,ka,la) * t2_1p.at("abab")(da,bb,la,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_1p.at("abab") += +1.00 <l,k||d,c>_abab t2_abab(d,b,i,k) t1_bb(c,j) t1_1p_aa(a,l) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o4v2 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o3v1 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,jb) )
    ( tmps.at("bin_aabb_vooo")(da,la,jb,kb)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin_aabb_vooo")(da,la,jb,kb) * t2.at("abab")(da,bb,ia,kb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_baab_vooo")(bb,ia,la,jb) * t1_1p.at("aa")(aa,la) )
    
    // r2_1p.at("abab") += +0.50 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t2_1p_abab(c,d,i,j) 
    //            += +0.50 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t2_1p_abab(d,c,i,j) 
    // flops: o2v2 += o2v2Q1 o4v2 o4v1 o3v2
    //  mems: o2v2 += o2v2 o4v0 o3v1 o2v2
    ( tmps.at("bin_abab_vvoo")(ca,db,ka,lb)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin_aabb_oooo")(ia,ka,jb,lb)  = tmps.at("bin_abab_vvoo")(ca,db,ka,lb) * t2_1p.at("abab")(ca,db,ia,jb) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aabb_oooo")(ia,ka,jb,lb) * t1.at("bb")(bb,lb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +4.00 d-_aa(k,c) t1_1p_aa(c,k) t2_2p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.00 * scalars.at("0050")() * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += +4.00 d-_bb(k,c) t1_1p_bb(c,k) t2_2p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.00 * scalars.at("0052")() * t2_2p.at("abab")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 d-_bb(k,c) t2_1p_abab(a,b,i,j) t1_2p_bb(c,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * scalars.at("0056")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 d-_aa(k,c) t2_1p_abab(a,b,i,j) t1_2p_aa(c,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * scalars.at("0058")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 f_aa(k,i) t2_2p_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * f.at("aa_oo")(ka,ia) * t2_2p.at("abab")(aa,bb,ka,jb) )
    
    // r2_2p.at("abab") += -2.00 f_bb(k,j) t2_2p_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_2p.at("abab")(aa,bb,ia,kb) * f.at("bb_oo")(kb,jb) )
    
    // r2_2p.at("abab") += +2.00 f_aa(a,c) t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * f.at("aa_vv")(aa,ca) * t2_2p.at("abab")(ca,bb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 f_bb(b,c) t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2_2p.at("abab")(aa,cb,ia,jb) * f.at("bb_vv")(bb,cb) )
    
    // r2_2p.at("abab") += -4.00 d-_bb(k,j) t1_1p_bb(b,k) t1_2p_aa(a,i) 
    // flops: o2v2 += o2v1 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_oo")(kb,jb) * t1_1p.at("bb")(bb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_2p.at("aa")(aa,ia) )
    
    // r2_2p.at("abab") += -2.00 d-_bb(k,j) t1_1p_aa(a,i) t1_2p_bb(b,k) 
    // flops: o2v2 += o2v1 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_oo")(kb,jb) * t1_2p.at("bb")(bb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p.at("abab") += +4.00 d-_bb(b,c) t1_1p_bb(c,j) t1_2p_aa(a,i) 
    // flops: o2v2 += o1v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_vv")(bb,cb) * t1_1p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_2p.at("aa")(aa,ia) )
    
    // r2_2p.at("abab") += +2.00 d-_bb(b,c) t1_1p_aa(a,i) t1_2p_bb(c,j) 
    // flops: o2v2 += o1v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_vv")(bb,cb) * t1_2p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p.at("abab") += +2.00 d-_aa(k,c) t1_1p_aa(a,i) t2_2p_abab(c,b,k,j) 
    // flops: o2v2 += o2v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("aa_ov")(ka,ca) * t2_2p.at("abab")(ca,bb,ka,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p.at("abab") += -4.00 d-_bb(k,c) t2_1p_bbbb(c,b,j,k) t1_2p_aa(a,i) 
    // flops: o2v2 += o2v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_ov")(kb,cb) * t2_1p.at("bbbb")(cb,bb,jb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_2p.at("aa")(aa,ia) )
    
    // r2_2p.at("abab") += -2.00 d-_bb(k,c) t1_1p_aa(a,i) t2_2p_bbbb(c,b,j,k) 
    // flops: o2v2 += o2v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_ov")(kb,cb) * t2_2p.at("bbbb")(cb,bb,jb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2_2p.at("abab") += -2.00 f_aa(k,c) t1_aa(a,k) t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = f.at("aa_ov")(ka,ca) * t2_2p.at("abab")(ca,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += -2.00 f_bb(k,c) t2_abab(a,b,i,k) t1_2p_bb(c,j) 
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = f.at("bb_ov")(kb,cb) * t1_2p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_2p.at("abab") += -2.00 f_bb(k,c) t1_bb(b,k) t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = f.at("bb_ov")(kb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -6.00 d-_bb(k,c) t1_1p_bb(b,k) t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = dp.at("bb_ov")(kb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 6.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -2.00 f_bb(k,c) t2_1p_abab(a,c,i,j) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = f.at("bb_ov")(kb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -2.00 <a,k||i,j>_abab t1_2p_bb(b,k) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = chol.at("aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1_2p.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += +2.00 <a,b||i,c>_abab t1_2p_bb(c,j) 
    // flops: o2v2 += o1v2Q1 o2v2Q1
    //  mems: o2v2 += o1v1Q1 o2v2
    ( tmps.at("bin_bb_voQ")(bb,jb,Q)  = chol.at("bb_vvQ")(bb,cb,Q) * t1_2p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_bb_voQ")(bb,jb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    
    // r2_2p.at("abab") += +2.00 <k,a||c,i>_aaaa t2_2p_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin_aaaa_vvoo")(aa,ca,ia,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("aa_vvQ")(aa,ca,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aaaa_vvoo")(aa,ca,ia,ka) * t2_2p.at("abab")(ca,bb,ka,jb) )
    
    // r2_2p.at("abab") += -2.00 <a,k||i,c>_abab t1_1p_bb(b,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1_1p.at("bb")(cb,jb) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("aa_voQ")(aa,ia,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -2.00 <k,b||c,j>_abab t1_aa(c,i) t1_2p_aa(a,k) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aa_ooQ")(ia,ka,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1_2p.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += -2.00 <k,b||c,d>_bbbb t2_abab(a,c,i,j) t1_2p_bb(d,k) 
    // flops: o2v2 += o1v2Q1 o1v2Q1 o2v3
    //  mems: o2v2 += o1v1Q1 o0v2 o2v2
    ( tmps.at("bin_bb_voQ")(bb,kb,Q)  = chol.at("bb_vvQ")(bb,db,Q) * t1_2p.at("bb")(db,kb) )
    ( tmps.at("bin_bb_vv")(bb,cb)  = tmps.at("bin_bb_voQ")(bb,kb,Q) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_2p.at("abab") += +0.50 <l,k||c,d>_abab t2_1p_abab(a,b,l,k) t2_1p_abab(c,d,i,j) 
    //            += +0.50 <l,k||d,c>_abab t2_1p_abab(a,b,l,k) t2_1p_abab(d,c,i,j) 
    //            += +0.50 <k,l||c,d>_abab t2_1p_abab(a,b,k,l) t2_1p_abab(c,d,i,j) 
    //            += +0.50 <k,l||d,c>_abab t2_1p_abab(a,b,k,l) t2_1p_abab(d,c,i,j) 
    // flops: o2v2 += o2v2Q1 o4v2 o4v2
    //  mems: o2v2 += o2v2 o4v0 o2v2
    ( tmps.at("bin_abab_vvoo")(ca,db,la,kb)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin_aabb_oooo")(ia,la,jb,kb)  = tmps.at("bin_abab_vvoo")(ca,db,la,kb) * t2_1p.at("abab")(ca,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_oooo")(ia,la,jb,kb) * t2_1p.at("abab")(aa,bb,la,kb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,j>_bbbb t2_1p_abab(a,c,i,l) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v1Q1 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin_bbbb_vooo")(cb,jb,kb,lb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bbbb_vooo")(cb,jb,kb,lb) * t2_1p.at("abab")(aa,cb,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,j>_bbbb t1_bb(b,k) t2_2p_abab(a,c,i,l) 
    // flops: o2v2 += o3v1Q1 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    ( tmps.at("bin_bbbb_vooo")(cb,jb,kb,lb)  = chol.at("bb_ovQ")(kb,cb,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bbbb_vooo")(cb,jb,kb,lb) * t2_2p.at("abab")(aa,cb,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -2.00 <k,a||c,d>_aaaa t2_1p_abab(d,b,k,j) t1_1p_aa(c,i) 
    // flops: o2v2 += o2v1Q1 o2v2Q1 o3v3
    //  mems: o2v2 += o2v0Q1 o2v2 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1_1p.at("aa")(ca,ia) )
    ( tmps.at("bin_aaaa_vvoo")(aa,da,ia,ka)  = tmps.at("bin_aa_ooQ")(ia,ka,Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aaaa_vvoo")(aa,da,ia,ka) * t2_1p.at("abab")(da,bb,ka,jb) )
    
    // r2_2p.at("abab") += -2.00 <k,a||c,d>_aaaa t1_aa(c,i) t2_2p_abab(d,b,k,j) 
    // flops: o2v2 += o2v1Q1 o2v2Q1 o3v3
    //  mems: o2v2 += o2v0Q1 o2v2 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin_aaaa_vvoo")(aa,da,ia,ka)  = tmps.at("bin_aa_ooQ")(ia,ka,Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aaaa_vvoo")(aa,da,ia,ka) * t2_2p.at("abab")(da,bb,ka,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_2p_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1 o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2 o2v2
    ( tmps.at("bin2_aaaa_vvoo")(ca,da,ka,la)  = chol.at("aa_ovQ")(ka,da,Q) * chol.at("aa_ovQ")(la,ca,Q) )
    ( tmps.at("bin_aaaa_vvoo")(aa,da,ia,la)  = tmps.at("bin2_aaaa_vvoo")(ca,da,ka,la) * t2.at("aaaa")(ca,aa,ia,ka) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aaaa_vvoo")(aa,da,ia,la) * t2_2p.at("abab")(da,bb,la,jb) )
    
    // r2_2p.at("abab") += -2.00 <a,k||d,c>_abab t2_1p_abab(d,b,i,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o2v1Q1 o2v2Q1 o3v3
    //  mems: o2v2 += o2v0Q1 o2v2 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1_1p.at("bb")(cb,jb) )
    ( tmps.at("bin_aabb_vvoo")(aa,da,jb,kb)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vvoo")(aa,da,jb,kb) * t2_1p.at("abab")(da,bb,ia,kb) )
    
    // r2_2p.at("abab") += -2.00 <a,k||c,d>_abab t2_abab(c,b,i,k) t1_2p_bb(d,j) 
    // flops: o2v2 += o2v1Q1 o2v2Q1 o3v3
    //  mems: o2v2 += o2v0Q1 o2v2 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,db,Q) * t1_2p.at("bb")(db,jb) )
    ( tmps.at("bin_aabb_vvoo")(aa,ca,jb,kb)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("aa_vvQ")(aa,ca,Q) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vvoo")(aa,ca,jb,kb) * t2.at("abab")(ca,bb,ia,kb) )
    
    // r2_2p.at("abab") += -1.00 <a,k||c,d>_abab t1_bb(b,k) t2_2p_abab(c,d,i,j) 
    //            += -1.00 <a,k||d,c>_abab t1_bb(b,k) t2_2p_abab(d,c,i,j) 
    // flops: o2v2 += o1v3Q1 o3v3 o3v2
    //  mems: o2v2 += o1v3 o3v1 o2v2
    ( tmps.at("bin_aabb_vvvo")(aa,ca,db,kb)  = chol.at("aa_vvQ")(aa,ca,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_aabb_vvvo")(aa,ca,db,kb) * t2_2p.at("abab")(ca,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_abab t2_abab(c,b,i,k) t2_2p_abab(a,d,l,j) 
    // flops: o2v2 += o2v2Q1 o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2 o2v2
    ( tmps.at("bin_abab_vvoo")(ca,db,la,kb)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin_aabb_vvoo")(aa,ca,jb,kb)  = tmps.at("bin_abab_vvoo")(ca,db,la,kb) * t2_2p.at("abab")(aa,db,la,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vvoo")(aa,ca,jb,kb) * t2.at("abab")(ca,bb,ia,kb) )
    
    // r2_2p.at("abab") += -1.00 <k,b||c,d>_abab t1_1p_aa(a,k) t2_1p_abab(c,d,i,j) 
    //            += -1.00 <k,b||d,c>_abab t1_1p_aa(a,k) t2_1p_abab(d,c,i,j) 
    // flops: o2v2 += o1v3Q1 o3v3 o3v2
    //  mems: o2v2 += o1v3 o3v1 o2v2
    ( tmps.at("bin_abba_vvvo")(ca,bb,db,ka)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("bb_vvQ")(bb,db,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_abba_vvvo")(ca,bb,db,ka) * t2_1p.at("abab")(ca,db,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_bbbb(c,b,j,k) t2_2p_abab(a,d,i,l) 
    // flops: o2v2 += o2v2Q1 o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2 o2v2
    ( tmps.at("bin_bbbb_vvoo")(cb,db,kb,lb)  = chol.at("bb_ovQ")(kb,db,Q) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("bin_abab_vvoo")(aa,cb,ia,kb)  = tmps.at("bin_bbbb_vvoo")(cb,db,kb,lb) * t2_2p.at("abab")(aa,db,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_abab_vvoo")(aa,cb,ia,kb) * t2.at("bbbb")(cb,bb,jb,kb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_aaaa t2_abab(c,b,i,j) t1_1p_aa(a,l) t1_1p_aa(d,k) 
    // flops: o2v2 += o1v1Q1 o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o0v0Q1 o1v1 o3v1 o2v2
    ( tmps.at("bin_Q")(Q)  = chol.at("aa_ovQ")(ka,da,Q) * t1_1p.at("aa")(da,ka) )
    ( tmps.at("bin_aa_vo")(ca,la)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(la,ca,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin_aa_vo")(ca,la) * t2.at("abab")(ca,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,la,jb) * t1_1p.at("aa")(aa,la) )
    
    // r2_2p.at("abab") += -2.00 <l,k||d,c>_abab t1_bb(c,k) t1_1p_aa(a,l) t2_1p_abab(d,b,i,j) 
    // flops: o2v2 += o1v1Q1 o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o0v0Q1 o1v1 o3v1 o2v2
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin_aa_vo")(da,la)  = tmps.at("bin_Q")(Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin_aa_vo")(da,la) * t2_1p.at("abab")(da,bb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_baab_vooo")(bb,ia,la,jb) * t1_1p.at("aa")(aa,la) )
    
    // r2_2p.at("abab") += -2.00 <k,l||d,c>_abab t2_abab(a,c,i,j) t1_1p_bb(b,l) t1_1p_aa(d,k) 
    // flops: o2v2 += o1v1Q1 o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o0v0Q1 o1v1 o3v1 o2v2
    ( tmps.at("bin_Q")(Q)  = chol.at("aa_ovQ")(ka,da,Q) * t1_1p.at("aa")(da,ka) )
    ( tmps.at("bin_bb_vo")(cb,lb)  = tmps.at("bin_Q")(Q) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("bin_bb_vo")(cb,lb) * t2.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_bbbb t2_abab(a,c,i,j) t1_1p_bb(b,l) t1_1p_bb(d,k) 
    // flops: o2v2 += o1v1Q1 o1v1Q1 o3v2 o3v2
    //  mems: o2v2 += o0v0Q1 o1v1 o3v1 o2v2
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(kb,db,Q) * t1_1p.at("bb")(db,kb) )
    ( tmps.at("bin_bb_vo")(cb,lb)  = tmps.at("bin_Q")(Q) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,lb)  = tmps.at("bin_bb_vo")(cb,lb) * t2.at("abab")(aa,cb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,lb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t1_aa(c,i) t1_1p_aa(a,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o4v2 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o3v1 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,la,Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin_aaaa_vooo")(da,ia,ka,la)  = tmps.at("bin_aa_ooQ")(ia,la,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aaaa_vooo")(da,ia,ka,la) * t2_1p.at("abab")(da,bb,la,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1_1p.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <l,k||d,c>_abab t2_abab(d,b,i,k) t1_bb(c,j) t1_2p_aa(a,l) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o4v2 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o3v1 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,jb) )
    ( tmps.at("bin_aabb_vooo")(da,la,jb,kb)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin_aabb_vooo")(da,la,jb,kb) * t2.at("abab")(da,bb,ia,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,la,jb) * t1_2p.at("aa")(aa,la) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_abab t1_bb(b,k) t1_aa(c,i) t2_2p_abab(a,d,l,j) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o4v2 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o3v1 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,la,Q)  = chol.at("aa_ovQ")(la,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin_baab_vooo")(db,ia,la,kb)  = tmps.at("bin_aa_ooQ")(ia,la,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_baab_vooo")(db,ia,la,kb) * t2_2p.at("abab")(aa,db,la,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_abab t1_bb(b,k) t1_1p_aa(a,l) t2_1p_abab(c,d,i,j) 
    //            += +1.00 <l,k||d,c>_abab t1_bb(b,k) t1_1p_aa(a,l) t2_1p_abab(d,c,i,j) 
    // flops: o2v2 += o2v2Q1 o4v2 o4v1 o3v2
    //  mems: o2v2 += o2v2 o4v0 o3v1 o2v2
    ( tmps.at("bin_abab_vvoo")(ca,db,la,kb)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin_aabb_oooo")(ia,la,jb,kb)  = tmps.at("bin_abab_vvoo")(ca,db,la,kb) * t2_1p.at("abab")(ca,db,ia,jb) )
    ( tmps.at("bin_baab_vooo")(bb,ia,la,jb)  = tmps.at("bin_aabb_oooo")(ia,la,jb,kb) * t1.at("bb")(bb,kb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,la,jb) * t1_1p.at("aa")(aa,la) )
    
    // r2_2p.at("abab") += +1.00 <k,l||c,d>_abab t1_aa(a,k) t1_1p_bb(b,l) t2_1p_abab(c,d,i,j) 
    //            += +1.00 <k,l||d,c>_abab t1_aa(a,k) t1_1p_bb(b,l) t2_1p_abab(d,c,i,j) 
    // flops: o2v2 += o2v2Q1 o4v2 o4v1 o3v2
    //  mems: o2v2 += o2v2 o4v0 o3v1 o2v2
    ( tmps.at("bin_abab_vvoo")(ca,db,ka,lb)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("bin_aabb_oooo")(ia,ka,jb,lb)  = tmps.at("bin_abab_vvoo")(ca,db,ka,lb) * t2_1p.at("abab")(ca,db,ia,jb) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aabb_oooo")(ia,ka,jb,lb) * t1_1p.at("bb")(bb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_2p_abab(a,d,i,l) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o4v2 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o3v1 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,lb,Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t1.at("bb")(cb,jb) )
    ( tmps.at("bin_bbbb_vooo")(db,jb,kb,lb)  = tmps.at("bin_bb_ooQ")(jb,lb,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("bin_bbbb_vooo")(db,jb,kb,lb) * t2_2p.at("abab")(aa,db,ia,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2.at("abab") += +1.00 d-_bb(b,j) t1_1p_aa(a,i) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,ia) * dp.at("bb_vo")(bb,jb) )
    
    // r2.at("abab") += +1.00 d-_bb(k,c) t1_bb(c,k) t2_1p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += scalars.at("0044")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2.at("abab") += +1.00 d-_aa(k,c) t1_aa(c,k) t2_1p_abab(a,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += scalars.at("0046")() * t2_1p.at("abab")(aa,bb,ia,jb) )
    
    // r2.at("abab") += -1.00 f_aa(k,i) t2_abab(a,b,k,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= f.at("aa_oo")(ka,ia) * t2.at("abab")(aa,bb,ka,jb) )
    
    // r2.at("abab") += -1.00 f_bb(k,j) t2_abab(a,b,i,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,bb,ia,kb) * f.at("bb_oo")(kb,jb) )
    
    // r2.at("abab") += +1.00 f_aa(a,c) t2_abab(c,b,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += f.at("aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2.at("abab") += +1.00 f_bb(b,c) t2_abab(a,c,i,j) 
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t2.at("abab")(aa,cb,ia,jb) * f.at("bb_vv")(bb,cb) )
    
    // r2.at("abab") += +1.00 <a,b||i,j>_abab 
    // flops: o2v2 += o2v2Q1
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += chol.at("aa_voQ")(aa,ia,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r2.at("abab") += -1.00 d-_bb(k,c) t2_bbbb(c,b,j,k) t1_1p_aa(a,i) 
    // flops: o2v2 += o2v2 o2v2
    //  mems: o2v2 += o1v1 o2v2
    ( tmps.at("bin_bb_vo")(bb,jb)  = dp.at("bb_ov")(kb,cb) * t2.at("bbbb")(cb,bb,jb,kb) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_bb_vo")(bb,jb) * t1_1p.at("aa")(aa,ia) )
    
    // r2.at("abab") += -1.00 <a,k||i,j>_abab t1_bb(b,k) 
    // flops: o2v2 += o3v1Q1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = chol.at("aa_voQ")(aa,ia,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2.at("abab") += +1.00 <k,a||c,i>_aaaa t2_abab(c,b,k,j) 
    // flops: o2v2 += o2v2Q1 o3v3
    //  mems: o2v2 += o2v2 o2v2
    ( tmps.at("bin_aaaa_vvoo")(aa,ca,ia,ka)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("aa_vvQ")(aa,ca,Q) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aaaa_vvoo")(aa,ca,ia,ka) * t2.at("abab")(ca,bb,ka,jb) )
    
    // r2.at("abab") += -1.00 <k,b||c,j>_abab t1_aa(a,k) t1_aa(c,i) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aa_ooQ")(ia,ka,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2.at("abab") += -1.00 <k,a||c,d>_aaaa t2_abab(d,b,k,j) t1_aa(c,i) 
    // flops: o2v2 += o2v1Q1 o2v2Q1 o3v3
    //  mems: o2v2 += o2v0Q1 o2v2 o2v2
    ( tmps.at("bin_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin_aaaa_vvoo")(aa,da,ia,ka)  = tmps.at("bin_aa_ooQ")(ia,ka,Q) * chol.at("aa_vvQ")(aa,da,Q) )
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("bin_aaaa_vvoo")(aa,da,ia,ka) * t2.at("abab")(da,bb,ka,jb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_abab(d,b,l,j) 
    // flops: o2v2 += o2v2Q1 o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2 o2v2
    ( tmps.at("bin2_aaaa_vvoo")(ca,da,ka,la)  = chol.at("aa_ovQ")(ka,da,Q) * chol.at("aa_ovQ")(la,ca,Q) )
    ( tmps.at("bin_aaaa_vvoo")(aa,da,ia,la)  = tmps.at("bin2_aaaa_vvoo")(ca,da,ka,la) * t2.at("aaaa")(ca,aa,ia,ka) )
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("bin_aaaa_vvoo")(aa,da,ia,la) * t2.at("abab")(da,bb,la,jb) )
    
    // r2.at("abab") += +1.00 <k,l||d,c>_abab t1_aa(a,k) t2_abab(d,b,i,l) t1_bb(c,j) 
    // flops: o2v2 += o2v1Q1 o3v1Q1 o4v2 o3v2
    //  mems: o2v2 += o2v0Q1 o3v1 o3v1 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,lb,Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t1.at("bb")(cb,jb) )
    ( tmps.at("bin_aabb_vooo")(da,ka,jb,lb)  = tmps.at("bin_bb_ooQ")(jb,lb,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aabb_vooo")(da,ka,jb,lb) * t2.at("abab")(da,bb,ia,lb) )
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .allocate(tmps.at("0001_baba_vvoo"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin_bbbb_vvoo")(cb,db,kb,lb)  = chol.at("bb_ovQ")(lb,cb,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0001_baba_vvoo")(cb,aa,kb,ia)  = tmps.at("bin_bbbb_vvoo")(cb,db,kb,lb) * t2_1p.at("abab")(aa,db,ia,lb) )
    
    // r1_1p.at("aa") += -1.00 <k,j||b,c>_bbbb t1_bb(b,j) t2_1p_abab(a,c,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) -= t1.at("bb")(bb,jb) * tmps.at("0001_baba_vvoo")(bb,aa,jb,ia) )
    
    // r1_2p.at("aa") += -2.00 <k,j||b,c>_bbbb t2_1p_abab(a,c,i,k) t1_1p_bb(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) -= 2.00 * t1_1p.at("bb")(bb,jb) * tmps.at("0001_baba_vvoo")(bb,aa,jb,ia) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_bbbb(c,b,j,k) t2_1p_abab(a,d,i,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("bbbb")(cb,bb,jb,kb) * tmps.at("0001_baba_vvoo")(cb,aa,kb,ia) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t1_bb(b,k) t1_bb(c,j) t2_1p_abab(a,d,i,l) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0001_baba_vvoo")(cb,aa,kb,ia) * t1.at("bb")(cb,jb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_1p_abab(a,c,i,k) t2_1p_bbbb(d,b,j,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0001_baba_vvoo")(db,aa,lb,ia) * t2_1p.at("bbbb")(db,bb,jb,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(b,k) t2_1p_abab(a,d,i,l) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0001_baba_vvoo")(cb,aa,kb,ia) * t1_1p.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t1_bb(c,j) t2_1p_abab(a,d,i,l) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0001_baba_vvoo")(cb,aa,kb,ia) * t1.at("bb")(cb,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0001_baba_vvoo"))
    .allocate(tmps.at("0002_baab_vooo"))
    
    // flops: o3v1  = o1v3Q1 o3v3
    //  mems: o3v1  = o1v3 o3v1
    ( tmps.at("bin_abba_vvvo")(ca,bb,db,ka)  = chol.at("bb_vvQ")(bb,db,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("0002_baab_vooo")(bb,ka,ia,jb)  = tmps.at("bin_abba_vvvo")(ca,bb,db,ka) * t2.at("abab")(ca,db,ia,jb) )
    
    // r2_1p.at("abab") += -0.50 <k,b||c,d>_abab t2_abab(c,d,i,j) t1_1p_aa(a,k) 
    //            += -0.50 <k,b||d,c>_abab t2_abab(d,c,i,j) t1_1p_aa(a,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("aa")(aa,ka) * tmps.at("0002_baab_vooo")(bb,ka,ia,jb) )
    
    // r2_2p.at("abab") += -1.00 <k,b||c,d>_abab t2_abab(c,d,i,j) t1_2p_aa(a,k) 
    //            += -1.00 <k,b||d,c>_abab t2_abab(d,c,i,j) t1_2p_aa(a,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_2p.at("aa")(aa,ka) * tmps.at("0002_baab_vooo")(bb,ka,ia,jb) )
    
    // r2.at("abab") += -0.50 <k,b||c,d>_abab t1_aa(a,k) t2_abab(c,d,i,j) 
    //         += -0.50 <k,b||d,c>_abab t1_aa(a,k) t2_abab(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("0002_baab_vooo")(bb,ka,ia,jb) )
    .deallocate(tmps.at("0002_baab_vooo"))
    .allocate(tmps.at("0003_abab_vooo"))
    
    // flops: o3v1  = o1v3Q1 o3v3
    //  mems: o3v1  = o1v3 o3v1
    ( tmps.at("bin_aabb_vvvo")(aa,ca,db,kb)  = chol.at("aa_vvQ")(aa,ca,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0003_abab_vooo")(aa,kb,ia,jb)  = tmps.at("bin_aabb_vvvo")(aa,ca,db,kb) * t2.at("abab")(ca,db,ia,jb) )
    
    // r2_1p.at("abab") += -0.50 <a,k||c,d>_abab t2_abab(c,d,i,j) t1_1p_bb(b,k) 
    //            += -0.50 <a,k||d,c>_abab t2_abab(d,c,i,j) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0003_abab_vooo")(aa,kb,ia,jb) * t1_1p.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -1.00 <a,k||c,d>_abab t2_abab(c,d,i,j) t1_2p_bb(b,k) 
    //            += -1.00 <a,k||d,c>_abab t2_abab(d,c,i,j) t1_2p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0003_abab_vooo")(aa,kb,ia,jb) * t1_2p.at("bb")(bb,kb) )
    
    // r2.at("abab") += -0.50 <a,k||c,d>_abab t1_bb(b,k) t2_abab(c,d,i,j) 
    //         += -0.50 <a,k||d,c>_abab t1_bb(b,k) t2_abab(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0003_abab_vooo")(aa,kb,ia,jb) * t1.at("bb")(bb,kb) )
    .deallocate(tmps.at("0003_abab_vooo"))
    .allocate(tmps.at("0004_abab_vooo"))
    
    // flops: o3v1  = o1v3Q1 o3v3
    //  mems: o3v1  = o1v3 o3v1
    ( tmps.at("bin_aabb_vvvo")(aa,ca,db,kb)  = chol.at("aa_vvQ")(aa,ca,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0004_abab_vooo")(aa,kb,ia,jb)  = tmps.at("bin_aabb_vvvo")(aa,ca,db,kb) * t2_1p.at("abab")(ca,db,ia,jb) )
    
    // r2_1p.at("abab") += -0.50 <a,k||c,d>_abab t1_bb(b,k) t2_1p_abab(c,d,i,j) 
    //            += -0.50 <a,k||d,c>_abab t1_bb(b,k) t2_1p_abab(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0004_abab_vooo")(aa,kb,ia,jb) * t1.at("bb")(bb,kb) )
    
    // r2_2p.at("abab") += -1.00 <a,k||c,d>_abab t1_1p_bb(b,k) t2_1p_abab(c,d,i,j) 
    //            += -1.00 <a,k||d,c>_abab t1_1p_bb(b,k) t2_1p_abab(d,c,i,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0004_abab_vooo")(aa,kb,ia,jb) * t1_1p.at("bb")(bb,kb) )
    .deallocate(tmps.at("0004_abab_vooo"))
    .allocate(tmps.at("0005_aaaa_vvoo"))
    
    // flops: o2v2  = o2v2Q1 o3v3
    //  mems: o2v2  = o2v2 o2v2
    ( tmps.at("bin_aaaa_vvoo")(ca,da,ka,la)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("0005_aaaa_vvoo")(ca,aa,ka,ia)  = tmps.at("bin_aaaa_vvoo")(ca,da,ka,la) * t2_1p.at("aaaa")(da,aa,ia,la) )
    
    // r1_1p.at("aa") += +1.00 <k,j||b,c>_aaaa t1_aa(b,j) t2_1p_aaaa(c,a,i,k) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_1p.at("aa")(aa,ia) += t1.at("aa")(ba,ja) * tmps.at("0005_aaaa_vvoo")(ba,aa,ja,ia) )
    
    // r1_2p.at("aa") += +2.00 <k,j||b,c>_aaaa t2_1p_aaaa(c,a,i,k) t1_1p_aa(b,j) 
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    ( r1_2p.at("aa")(aa,ia) += 2.00 * t1_1p.at("aa")(ba,ja) * tmps.at("0005_aaaa_vvoo")(ba,aa,ja,ia) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_aaaa t2_abab(c,b,k,j) t2_1p_aaaa(d,a,i,l) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t2.at("abab")(ca,bb,ka,jb) * tmps.at("0005_aaaa_vvoo")(ca,aa,ka,ia) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t2_1p_aaaa(c,a,i,k) t2_1p_abab(d,b,l,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0005_aaaa_vvoo")(da,aa,la,ia) * t2_1p.at("abab")(da,bb,la,jb) )
    .deallocate(tmps.at("0005_aaaa_vvoo"))
    .allocate(tmps.at("0006_baba_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin_bbbb_vooo")(cb,jb,kb,lb)  = chol.at("bb_ooQ")(kb,jb,Q) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("0006_baba_ovoo")(jb,aa,lb,ia)  = tmps.at("bin_bbbb_vooo")(cb,jb,kb,lb) * t2.at("abab")(aa,cb,ia,kb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,j>_bbbb t2_abab(a,c,i,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("bb")(bb,lb) * tmps.at("0006_baba_ovoo")(jb,aa,lb,ia) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,j>_bbbb t2_abab(a,c,i,k) t1_2p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("bb")(bb,lb) * tmps.at("0006_baba_ovoo")(jb,aa,lb,ia) )
    
    // r2.at("abab") += -1.00 <l,k||c,j>_bbbb t2_abab(a,c,i,l) t1_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0006_baba_ovoo")(jb,aa,kb,ia) )
    .deallocate(tmps.at("0006_baba_ovoo"))
    .allocate(tmps.at("0007_aabb_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin_baab_vooo")(cb,ia,ka,lb)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("bb_ovQ")(lb,cb,Q) )
    ( tmps.at("0007_aabb_ovoo")(ia,aa,lb,jb)  = tmps.at("bin_baab_vooo")(cb,ia,ka,lb) * t2.at("abab")(aa,cb,ka,jb) )
    
    // r2_1p.at("abab") += +1.00 <k,l||i,c>_abab t2_abab(a,c,k,j) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("bb")(bb,lb) * tmps.at("0007_aabb_ovoo")(ia,aa,lb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||i,c>_abab t2_abab(a,c,k,j) t1_2p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("bb")(bb,lb) * tmps.at("0007_aabb_ovoo")(ia,aa,lb,jb) )
    
    // r2.at("abab") += +1.00 <l,k||i,c>_abab t2_abab(a,c,l,j) t1_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0007_aabb_ovoo")(ia,aa,kb,jb) )
    .deallocate(tmps.at("0007_aabb_ovoo"))
    .allocate(tmps.at("0008_abba_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin_aabb_vooo")(ca,la,jb,kb)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    ( tmps.at("0008_abba_ovoo")(la,bb,jb,ia)  = tmps.at("bin_aabb_vooo")(ca,la,jb,kb) * t2.at("abab")(ca,bb,ia,kb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,j>_abab t2_abab(c,b,i,k) t1_1p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,la) * tmps.at("0008_abba_ovoo")(la,bb,jb,ia) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,j>_abab t2_abab(c,b,i,k) t1_2p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("aa")(aa,la) * tmps.at("0008_abba_ovoo")(la,bb,jb,ia) )
    
    // r2.at("abab") += +1.00 <k,l||c,j>_abab t1_aa(a,k) t2_abab(c,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0008_abba_ovoo")(ka,bb,jb,ia) )
    .deallocate(tmps.at("0008_abba_ovoo"))
    .allocate(tmps.at("0009_abab_oooo"))
    
    // flops: o4v0  = o2v2Q1 o4v2
    //  mems: o4v0  = o2v2 o4v0
    ( tmps.at("bin_abab_vvoo")(ca,db,la,kb)  = chol.at("aa_ovQ")(la,ca,Q) * chol.at("bb_ovQ")(kb,db,Q) )
    ( tmps.at("0009_abab_oooo")(la,kb,ia,jb)  = tmps.at("bin_abab_vvoo")(ca,db,la,kb) * t2_2p.at("abab")(ca,db,ia,jb) )
    
    // r2_2p.at("abab") += +0.50 <l,k||c,d>_abab t2_abab(a,b,l,k) t2_2p_abab(c,d,i,j) 
    //            += +0.50 <l,k||d,c>_abab t2_abab(a,b,l,k) t2_2p_abab(d,c,i,j) 
    //            += +0.50 <k,l||c,d>_abab t2_abab(a,b,k,l) t2_2p_abab(c,d,i,j) 
    //            += +0.50 <k,l||d,c>_abab t2_abab(a,b,k,l) t2_2p_abab(d,c,i,j) 
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t2.at("abab")(aa,bb,la,kb) * tmps.at("0009_abab_oooo")(la,kb,ia,jb) )
    
    // r2_2p.at("abab") += +1.00 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t2_2p_abab(c,d,i,j) 
    //            += +1.00 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t2_2p_abab(d,c,i,j) 
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t1.at("bb")(bb,lb) * tmps.at("0009_abab_oooo")(ka,lb,ia,jb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0009_abab_oooo"))
    .allocate(tmps.at("0010_aabb_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin_baab_vooo")(cb,ia,la,kb)  = chol.at("aa_ooQ")(la,ia,Q) * chol.at("bb_ovQ")(kb,cb,Q) )
    ( tmps.at("0010_aabb_ovoo")(ia,aa,kb,jb)  = tmps.at("bin_baab_vooo")(cb,ia,la,kb) * t2_1p.at("abab")(aa,cb,la,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||i,c>_abab t1_bb(b,k) t2_1p_abab(a,c,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("bb")(bb,kb) * tmps.at("0010_aabb_ovoo")(ia,aa,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||i,c>_abab t2_1p_abab(a,c,l,j) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("bb")(bb,kb) * tmps.at("0010_aabb_ovoo")(ia,aa,kb,jb) )
    .deallocate(tmps.at("0010_aabb_ovoo"))
    .allocate(tmps.at("0011_bbaa_ovoo"))
    
    // flops: o3v1  = o2v1Q1 o3v1Q1 o4v2
    //  mems: o3v1  = o2v0Q1 o3v1 o3v1
    ( tmps.at("bin_bb_ooQ")(jb,lb,Q)  = chol.at("bb_ovQ")(lb,cb,Q) * t1.at("bb")(cb,jb) )
    ( tmps.at("bin_aabb_vooo")(da,ka,jb,lb)  = tmps.at("bin_bb_ooQ")(jb,lb,Q) * chol.at("aa_ovQ")(ka,da,Q) )
    ( tmps.at("0011_bbaa_ovoo")(jb,bb,ka,ia)  = tmps.at("bin_aabb_vooo")(da,ka,jb,lb) * t2_1p.at("abab")(da,bb,ia,lb) )
    
    // r2_1p.at("abab") += +1.00 <k,l||d,c>_abab t1_aa(a,k) t1_bb(c,j) t2_1p_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0011_bbaa_ovoo")(jb,bb,ka,ia) )
    
    // r2_2p.at("abab") += +2.00 <k,l||d,c>_abab t1_bb(c,j) t1_1p_aa(a,k) t2_1p_abab(d,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0011_bbaa_ovoo")(jb,bb,ka,ia) )
    .deallocate(tmps.at("0011_bbaa_ovoo"))
    .allocate(tmps.at("0012_bbaa_ovoo"))
    
    // flops: o3v1  = o2v1Q1 o3v1Q1 o4v2
    //  mems: o3v1  = o2v0Q1 o3v1 o3v1
    ( tmps.at("bin_bb_ooQ")(jb,lb,Q)  = chol.at("bb_ovQ")(lb,db,Q) * t1_1p.at("bb")(db,jb) )
    ( tmps.at("bin_aabb_vooo")(ca,ka,jb,lb)  = tmps.at("bin_bb_ooQ")(jb,lb,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("0012_bbaa_ovoo")(jb,bb,ka,ia)  = tmps.at("bin_aabb_vooo")(ca,ka,jb,lb) * t2.at("abab")(ca,bb,ia,lb) )
    
    // r2_1p.at("abab") += +1.00 <k,l||c,d>_abab t1_aa(a,k) t2_abab(c,b,i,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0012_bbaa_ovoo")(jb,bb,ka,ia) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_abab t2_abab(c,b,i,k) t1_1p_aa(a,l) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,la) * tmps.at("0012_bbaa_ovoo")(jb,bb,la,ia) )
    .deallocate(tmps.at("0012_bbaa_ovoo"))
    .allocate(tmps.at("0013_abba_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin_aabb_vooo")(ca,ka,jb,lb)  = chol.at("aa_ovQ")(ka,ca,Q) * chol.at("bb_ooQ")(lb,jb,Q) )
    ( tmps.at("0013_abba_ovoo")(ka,bb,jb,ia)  = tmps.at("bin_aabb_vooo")(ca,ka,jb,lb) * t2_1p.at("abab")(ca,bb,ia,lb) )
    
    // r2_1p.at("abab") += +1.00 <k,l||c,j>_abab t1_aa(a,k) t2_1p_abab(c,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0013_abba_ovoo")(ka,bb,jb,ia) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,j>_abab t1_1p_aa(a,k) t2_1p_abab(c,b,i,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0013_abba_ovoo")(ka,bb,jb,ia) )
    .deallocate(tmps.at("0013_abba_ovoo"))
    .allocate(tmps.at("0014_abab_ovoo"))
    
    // flops: o3v1  = o2v1Q1 o3v1Q1 o4v2
    //  mems: o3v1  = o2v0Q1 o3v1 o3v1
    ( tmps.at("bin_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ia) )
    ( tmps.at("bin_aaaa_vooo")(da,ia,ka,la)  = tmps.at("bin_aa_ooQ")(ia,ka,Q) * chol.at("aa_ovQ")(la,da,Q) )
    ( tmps.at("0014_abab_ovoo")(ia,bb,la,jb)  = tmps.at("bin_aaaa_vooo")(da,ia,ka,la) * t2.at("abab")(da,bb,ka,jb) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,d>_aaaa t2_abab(d,b,k,j) t1_aa(c,i) t1_1p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,la) * tmps.at("0014_abab_ovoo")(ia,bb,la,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,d>_aaaa t2_abab(d,b,k,j) t1_aa(c,i) t1_2p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("aa")(aa,la) * tmps.at("0014_abab_ovoo")(ia,bb,la,jb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_aaaa t1_aa(a,k) t2_abab(d,b,l,j) t1_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0014_abab_ovoo")(ia,bb,ka,jb) )
    .deallocate(tmps.at("0014_abab_ovoo"))
    .allocate(tmps.at("0015_abab_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin_aaaa_vooo")(ca,ia,ka,la)  = chol.at("aa_ooQ")(ka,ia,Q) * chol.at("aa_ovQ")(la,ca,Q) )
    ( tmps.at("0015_abab_ovoo")(ia,bb,la,jb)  = tmps.at("bin_aaaa_vooo")(ca,ia,ka,la) * t2.at("abab")(ca,bb,ka,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,i>_aaaa t2_abab(c,b,k,j) t1_1p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1_1p.at("aa")(aa,la) * tmps.at("0015_abab_ovoo")(ia,bb,la,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,i>_aaaa t2_abab(c,b,k,j) t1_2p_aa(a,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_2p.at("aa")(aa,la) * tmps.at("0015_abab_ovoo")(ia,bb,la,jb) )
    
    // r2.at("abab") += -1.00 <l,k||c,i>_aaaa t1_aa(a,k) t2_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0015_abab_ovoo")(ia,bb,ka,jb) )
    .deallocate(tmps.at("0015_abab_ovoo"))
    .allocate(tmps.at("0016_abab_ovoo"))
    
    // flops: o3v1  = o2v1Q1 o3v1Q1 o4v2
    //  mems: o3v1  = o2v0Q1 o3v1 o3v1
    ( tmps.at("bin_aa_ooQ")(ia,la,Q)  = chol.at("aa_ovQ")(la,da,Q) * t1_1p.at("aa")(da,ia) )
    ( tmps.at("bin_aaaa_vooo")(ca,ia,ka,la)  = tmps.at("bin_aa_ooQ")(ia,la,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("0016_abab_ovoo")(ia,bb,ka,jb)  = tmps.at("bin_aaaa_vooo")(ca,ia,ka,la) * t2.at("abab")(ca,bb,la,jb) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,d>_aaaa t1_aa(a,k) t2_abab(c,b,l,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0016_abab_ovoo")(ia,bb,ka,jb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_aaaa t2_abab(c,b,k,j) t1_1p_aa(a,l) t1_1p_aa(d,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,la) * tmps.at("0016_abab_ovoo")(ia,bb,la,jb) )
    .deallocate(tmps.at("0016_abab_ovoo"))
    .allocate(tmps.at("0017_abab_ovoo"))
    
    // flops: o3v1  = o3v1Q1 o4v2
    //  mems: o3v1  = o3v1 o3v1
    ( tmps.at("bin_aaaa_vooo")(ca,ia,ka,la)  = chol.at("aa_ooQ")(la,ia,Q) * chol.at("aa_ovQ")(ka,ca,Q) )
    ( tmps.at("0017_abab_ovoo")(ia,bb,ka,jb)  = tmps.at("bin_aaaa_vooo")(ca,ia,ka,la) * t2_1p.at("abab")(ca,bb,la,jb) )
    
    // r2_1p.at("abab") += -1.00 <l,k||c,i>_aaaa t1_aa(a,k) t2_1p_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t1.at("aa")(aa,ka) * tmps.at("0017_abab_ovoo")(ia,bb,ka,jb) )
    
    // r2_2p.at("abab") += -2.00 <l,k||c,i>_aaaa t1_1p_aa(a,k) t2_1p_abab(c,b,l,j) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0017_abab_ovoo")(ia,bb,ka,jb) )
    .deallocate(tmps.at("0017_abab_ovoo"))
    .allocate(tmps.at("0018_aabb_vvvv"))
    
    // flops: o0v4  = o0v4Q1
    //  mems: o0v4  = o0v4
    ( tmps.at("0018_aabb_vvvv")(aa,ca,bb,db)  = chol.at("aa_vvQ")(aa,ca,Q) * chol.at("bb_vvQ")(bb,db,Q) )
    
    // r2_1p.at("abab") += +0.50 <a,b||c,d>_abab t2_1p_abab(c,d,i,j) 
    //            += +0.50 <a,b||d,c>_abab t2_1p_abab(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0018_aabb_vvvv")(aa,ca,bb,db) * t2_1p.at("abab")(ca,db,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 <a,b||d,c>_abab t1_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o1v4 o2v3
    //  mems: o2v2 += o1v3 o2v2
    ( tmps.at("bin_abba_vvvo")(aa,bb,cb,ia)  = t1_1p.at("aa")(da,ia) * tmps.at("0018_aabb_vvvv")(aa,da,bb,cb) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_abba_vvvo")(aa,bb,cb,ia) * t1.at("bb")(cb,jb) )
    
    // r2_1p.at("abab") += +1.00 <a,b||c,d>_abab t1_aa(c,i) t1_1p_bb(d,j) 
    // flops: o2v2 += o1v4 o2v3
    //  mems: o2v2 += o1v3 o2v2
    ( tmps.at("bin_abba_vvvo")(aa,bb,db,ia)  = t1.at("aa")(ca,ia) * tmps.at("0018_aabb_vvvv")(aa,ca,bb,db) )
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("bin_abba_vvvo")(aa,bb,db,ia) * t1_1p.at("bb")(db,jb) )
    
    // r2_2p.at("abab") += +1.00 <a,b||c,d>_abab t2_2p_abab(c,d,i,j) 
    //            += +1.00 <a,b||d,c>_abab t2_2p_abab(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0018_aabb_vvvv")(aa,ca,bb,db) * t2_2p.at("abab")(ca,db,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 <a,b||c,d>_abab t1_aa(c,i) t1_2p_bb(d,j) 
    // flops: o2v2 += o1v4 o2v3
    //  mems: o2v2 += o1v3 o2v2
    ( tmps.at("bin_abba_vvvo")(aa,bb,db,ia)  = t1.at("aa")(ca,ia) * tmps.at("0018_aabb_vvvv")(aa,ca,bb,db) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_abba_vvvo")(aa,bb,db,ia) * t1_2p.at("bb")(db,jb) )
    
    // r2_2p.at("abab") += +2.00 <a,b||d,c>_abab t1_1p_bb(c,j) t1_1p_aa(d,i) 
    // flops: o2v2 += o1v4 o2v3
    //  mems: o2v2 += o1v3 o2v2
    ( tmps.at("bin_abba_vvvo")(aa,bb,cb,ia)  = t1_1p.at("aa")(da,ia) * tmps.at("0018_aabb_vvvv")(aa,da,bb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_abba_vvvo")(aa,bb,cb,ia) * t1_1p.at("bb")(cb,jb) )
    
    // r2_2p.at("abab") += +2.00 <a,b||d,c>_abab t1_bb(c,j) t1_2p_aa(d,i) 
    // flops: o2v2 += o1v4 o2v3
    //  mems: o2v2 += o1v3 o2v2
    ( tmps.at("bin_abba_vvvo")(aa,bb,cb,ia)  = t1_2p.at("aa")(da,ia) * tmps.at("0018_aabb_vvvv")(aa,da,bb,cb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_abba_vvvo")(aa,bb,cb,ia) * t1.at("bb")(cb,jb) )
    
    // r2.at("abab") += +0.50 <a,b||c,d>_abab t2_abab(c,d,i,j) 
    //         += +0.50 <a,b||d,c>_abab t2_abab(d,c,i,j) 
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0018_aabb_vvvv")(aa,ca,bb,db) * t2.at("abab")(ca,db,ia,jb) )
    
    // r2.at("abab") += +1.00 <a,b||d,c>_abab t1_bb(c,j) t1_aa(d,i) 
    // flops: o2v2 += o1v4 o2v3
    //  mems: o2v2 += o1v3 o2v2
    ( tmps.at("bin_abba_vvvo")(aa,bb,cb,ia)  = t1.at("aa")(da,ia) * tmps.at("0018_aabb_vvvv")(aa,da,bb,cb) )
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("bin_abba_vvvo")(aa,bb,cb,ia) * t1.at("bb")(cb,jb) )
    .deallocate(tmps.at("0018_aabb_vvvv"))
    .allocate(tmps.at("0019_bbbb_oovv"))
    
    // flops: o2v2  = o2v1Q1 o2v2Q1
    //  mems: o2v2  = o2v0Q1 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,jb) )
    ( tmps.at("0019_bbbb_oovv")(kb,jb,bb,db)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("bb_vvQ")(bb,db,Q) )
    
    // r2_1p.at("abab") += -1.00 <k,b||c,d>_bbbb t1_bb(c,j) t2_1p_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,db,ia,kb) * tmps.at("0019_bbbb_oovv")(kb,jb,bb,db) )
    
    // r2_2p.at("abab") += -2.00 <k,b||c,d>_bbbb t1_bb(c,j) t2_2p_abab(a,d,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_2p.at("abab")(aa,db,ia,kb) * tmps.at("0019_bbbb_oovv")(kb,jb,bb,db) )
    
    // r2.at("abab") += -1.00 <k,b||c,d>_bbbb t2_abab(a,d,i,k) t1_bb(c,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,db,ia,kb) * tmps.at("0019_bbbb_oovv")(kb,jb,bb,db) )
    .deallocate(tmps.at("0019_bbbb_oovv"))
    .allocate(tmps.at("0020_bbbb_vvoo"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0020_bbbb_vvoo")(bb,cb,kb,jb)  = chol.at("bb_vvQ")(bb,cb,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    
    // r2_1p.at("abab") += +1.00 <k,b||c,j>_bbbb t2_1p_abab(a,c,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2_1p.at("abab")(aa,cb,ia,kb) * tmps.at("0020_bbbb_vvoo")(bb,cb,kb,jb) )
    
    // r2_2p.at("abab") += +2.00 <k,b||c,j>_bbbb t2_2p_abab(a,c,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_2p.at("abab")(aa,cb,ia,kb) * tmps.at("0020_bbbb_vvoo")(bb,cb,kb,jb) )
    
    // r2.at("abab") += +1.00 <k,b||c,j>_bbbb t2_abab(a,c,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,cb,ia,kb) * tmps.at("0020_bbbb_vvoo")(bb,cb,kb,jb) )
    .deallocate(tmps.at("0020_bbbb_vvoo"))
    .allocate(tmps.at("0021_bbbb_oovv"))
    
    // flops: o2v2  = o2v1Q1 o2v2Q1
    //  mems: o2v2  = o2v0Q1 o2v2
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,db,Q) * t1_1p.at("bb")(db,jb) )
    ( tmps.at("0021_bbbb_oovv")(kb,jb,bb,cb)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("bb_vvQ")(bb,cb,Q) )
    
    // r2_1p.at("abab") += +1.00 <k,b||c,d>_bbbb t2_abab(a,c,i,k) t1_1p_bb(d,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t2.at("abab")(aa,cb,ia,kb) * tmps.at("0021_bbbb_oovv")(kb,jb,bb,cb) )
    
    // r2_2p.at("abab") += -2.00 <k,b||c,d>_bbbb t2_1p_abab(a,d,i,k) t1_1p_bb(c,j) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t2_1p.at("abab")(aa,db,ia,kb) * tmps.at("0021_bbbb_oovv")(kb,jb,bb,db) )
    .deallocate(tmps.at("0021_bbbb_oovv"))
    .allocate(tmps.at("0022_aabb_vvoo"))
    
    // flops: o2v2  = o2v2Q1
    //  mems: o2v2  = o2v2
    ( tmps.at("0022_aabb_vvoo")(aa,ca,kb,jb)  = chol.at("aa_vvQ")(aa,ca,Q) * chol.at("bb_ooQ")(kb,jb,Q) )
    
    // r2_1p.at("abab") += -1.00 <a,k||c,j>_abab t2_1p_abab(c,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0022_aabb_vvoo")(aa,ca,kb,jb) * t2_1p.at("abab")(ca,bb,ia,kb) )
    
    // r2_2p.at("abab") += -2.00 <a,k||c,j>_abab t2_2p_abab(c,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0022_aabb_vvoo")(aa,ca,kb,jb) * t2_2p.at("abab")(ca,bb,ia,kb) )
    
    // r2_2p.at("abab") += -2.00 <a,k||c,j>_abab t1_bb(b,k) t1_2p_aa(c,i) 
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_aabb_vooo")(aa,ia,jb,kb)  = tmps.at("0022_aabb_vvoo")(aa,ca,kb,jb) * t1_2p.at("aa")(ca,ia) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("bin_aabb_vooo")(aa,ia,jb,kb) * t1.at("bb")(bb,kb) )
    
    // r2.at("abab") += -1.00 <a,k||c,j>_abab t2_abab(c,b,i,k) 
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0022_aabb_vvoo")(aa,ca,kb,jb) * t2.at("abab")(ca,bb,ia,kb) )
    .deallocate(tmps.at("0022_aabb_vvoo"))
    .allocate(tmps.at("0023_baab_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0023_baab_vvoo")(bb,aa,ia,jb)  = dp.at("bb_vv")(bb,cb) * t2_1p.at("abab")(aa,cb,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 d-_bb(b,c) t0_1p t2_1p_abab(a,c,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t0_1p * tmps.at("0023_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 d+_bb(b,c) t2_1p_abab(a,c,i,j) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0023_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2_2p.at("abab") += +4.00 d-_bb(b,c) t2_1p_abab(a,c,i,j) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.00 * t0_2p * tmps.at("0023_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2.at("abab") += +1.00 d-_bb(b,c) t2_1p_abab(a,c,i,j) 
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0023_baab_vvoo")(bb,aa,ia,jb) )
    .deallocate(tmps.at("0023_baab_vvoo"))
    .allocate(tmps.at("0024_baab_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0024_baab_vvoo")(bb,aa,ia,jb)  = dp.at("bb_vv")(bb,cb) * t2.at("abab")(aa,cb,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 d+_bb(b,c) t2_abab(a,c,i,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0024_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2_1p.at("abab") += +2.00 d-_bb(b,c) t2_abab(a,c,i,j) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.00 * t0_2p * tmps.at("0024_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2.at("abab") += +1.00 d-_bb(b,c) t2_abab(a,c,i,j) t0_1p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t0_1p * tmps.at("0024_baab_vvoo")(bb,aa,ia,jb) )
    .deallocate(tmps.at("0024_baab_vvoo"))
    .allocate(tmps.at("0025_baab_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0025_baab_vvoo")(bb,aa,ia,jb)  = dp.at("bb_vv")(bb,cb) * t2_2p.at("abab")(aa,cb,ia,jb) )
    
    // r2_1p.at("abab") += +2.00 d-_bb(b,c) t2_2p_abab(a,c,i,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0025_baab_vvoo")(bb,aa,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 d-_bb(b,c) t0_1p t2_2p_abab(a,c,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t0_1p * tmps.at("0025_baab_vvoo")(bb,aa,ia,jb) )
    .deallocate(tmps.at("0025_baab_vvoo"))
    .allocate(tmps.at("0026_abab_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0026_abab_vvoo")(aa,bb,ia,jb)  = dp.at("aa_vv")(aa,ca) * t2_1p.at("abab")(ca,bb,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 d-_aa(a,c) t0_1p t2_1p_abab(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += t0_1p * tmps.at("0026_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 d+_aa(a,c) t2_1p_abab(c,b,i,j) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0026_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += +4.00 d-_aa(a,c) t2_1p_abab(c,b,i,j) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 4.00 * t0_2p * tmps.at("0026_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2.at("abab") += +1.00 d-_aa(a,c) t2_1p_abab(c,b,i,j) 
    ( r2.at("abab")(aa,bb,ia,jb) += tmps.at("0026_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0026_abab_vvoo"))
    .allocate(tmps.at("0027_abab_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0027_abab_vvoo")(aa,bb,ia,jb)  = dp.at("aa_vv")(aa,ca) * t2.at("abab")(ca,bb,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 d+_aa(a,c) t2_abab(c,b,i,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) += tmps.at("0027_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_1p.at("abab") += +2.00 d-_aa(a,c) t2_abab(c,b,i,j) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.00 * t0_2p * tmps.at("0027_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2.at("abab") += +1.00 d-_aa(a,c) t2_abab(c,b,i,j) t0_1p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) += t0_1p * tmps.at("0027_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0027_abab_vvoo"))
    .allocate(tmps.at("0028_abab_vvoo"))
    
    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    ( tmps.at("0028_abab_vvoo")(aa,bb,ia,jb)  = dp.at("aa_vv")(aa,ca) * t2_2p.at("abab")(ca,bb,ia,jb) )
    
    // r2_1p.at("abab") += +2.00 d-_aa(a,c) t2_2p_abab(c,b,i,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("0028_abab_vvoo")(aa,bb,ia,jb) )
    
    // r2_2p.at("abab") += +2.00 d-_aa(a,c) t0_1p t2_2p_abab(c,b,i,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * t0_1p * tmps.at("0028_abab_vvoo")(aa,bb,ia,jb) )
    .deallocate(tmps.at("0028_abab_vvoo"))
    .allocate(tmps.at("0029_bbaa_vooo"))
    
    // flops: o3v1  = o3v1Q1
    //  mems: o3v1  = o3v1
    ( tmps.at("0029_bbaa_vooo")(bb,jb,ka,ia)  = chol.at("bb_voQ")(bb,jb,Q) * chol.at("aa_ooQ")(ka,ia,Q) )
    
    // r2_1p.at("abab") += -1.00 <k,b||i,j>_abab t1_1p_aa(a,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("aa")(aa,ka) * tmps.at("0029_bbaa_vooo")(bb,jb,ka,ia) )
    
    // r2_2p.at("abab") += -2.00 <k,b||i,j>_abab t1_2p_aa(a,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_2p.at("aa")(aa,ka) * tmps.at("0029_bbaa_vooo")(bb,jb,ka,ia) )
    
    // r2.at("abab") += -1.00 <k,b||i,j>_abab t1_aa(a,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("0029_bbaa_vooo")(bb,jb,ka,ia) )
    .deallocate(tmps.at("0029_bbaa_vooo"))
    .allocate(tmps.at("0030_aabb_oovo"))
    
    // flops: o3v1  = o2v1Q1 o3v1Q1
    //  mems: o3v1  = o2v0Q1 o3v1
    ( tmps.at("bin_aa_ooQ")(ia,ka,Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1_1p.at("aa")(ca,ia) )
    ( tmps.at("0030_aabb_oovo")(ka,ia,bb,jb)  = tmps.at("bin_aa_ooQ")(ia,ka,Q) * chol.at("bb_voQ")(bb,jb,Q) )
    
    // r2_1p.at("abab") += -1.00 <k,b||c,j>_abab t1_aa(a,k) t1_1p_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1.at("aa")(aa,ka) * tmps.at("0030_aabb_oovo")(ka,ia,bb,jb) )
    
    // r2_2p.at("abab") += -2.00 <k,b||c,j>_abab t1_1p_aa(a,k) t1_1p_aa(c,i) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_1p.at("aa")(aa,ka) * tmps.at("0030_aabb_oovo")(ka,ia,bb,jb) )
    .deallocate(tmps.at("0030_aabb_oovo"))
    .allocate(tmps.at("0031_bbaa_ooov"))
    
    // flops: o3v1  = o2v1Q1 o3v1Q1
    //  mems: o3v1  = o2v0Q1 o3v1
    ( tmps.at("bin_bb_ooQ")(jb,kb,Q)  = chol.at("bb_ovQ")(kb,db,Q) * t1_2p.at("bb")(db,jb) )
    ( tmps.at("0031_bbaa_ooov")(kb,jb,la,ca)  = tmps.at("bin_bb_ooQ")(jb,kb,Q) * chol.at("aa_ovQ")(la,ca,Q) )
    
    // r2_2p.at("abab") += +1.00 <l,k||c,d>_abab t2_abab(a,b,l,k) t1_aa(c,i) t1_2p_bb(d,j) 
    //            += +1.00 <k,l||c,d>_abab t2_abab(a,b,k,l) t1_aa(c,i) t1_2p_bb(d,j) 
    // flops: o2v2 += o4v1 o4v2
    //  mems: o2v2 += o4v0 o2v2
    ( tmps.at("bin_aabb_oooo")(ia,la,jb,kb)  = t1.at("aa")(ca,ia) * tmps.at("0031_bbaa_ooov")(kb,jb,la,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_aabb_oooo")(ia,la,jb,kb) * t2.at("abab")(aa,bb,la,kb) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,d>_abab t1_aa(a,k) t2_abab(c,b,i,l) t1_2p_bb(d,j) 
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = t2.at("abab")(ca,bb,ia,lb) * tmps.at("0031_bbaa_ooov")(lb,jb,ka,ca) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    
    // r2_2p.at("abab") += +2.00 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t1_aa(c,i) t1_2p_bb(d,j) 
    // flops: o2v2 += o4v1 o4v1 o3v2
    //  mems: o2v2 += o4v0 o3v1 o2v2
    ( tmps.at("bin_aabb_oooo")(ia,ka,jb,lb)  = t1.at("aa")(ca,ia) * tmps.at("0031_bbaa_ooov")(lb,jb,ka,ca) )
    ( tmps.at("bin_baab_vooo")(bb,ia,ka,jb)  = tmps.at("bin_aabb_oooo")(ia,ka,jb,lb) * t1.at("bb")(bb,lb) )
    ( r2_2p.at("abab")(aa,bb,ia,jb) += 2.00 * tmps.at("bin_baab_vooo")(bb,ia,ka,jb) * t1.at("aa")(aa,ka) )
    .deallocate(tmps.at("0031_bbaa_ooov"))
    .allocate(tmps.at("0032_abba_vvoo"))
    
    // flops: o2v2  = o2v1 o3v2
    //  mems: o2v2  = o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = dp.at("bb_ov")(kb,cb) * t1.at("bb")(cb,jb) )
    ( tmps.at("0032_abba_vvoo")(aa,bb,jb,ia)  = tmps.at("bin_bb_oo")(jb,kb) * t2_1p.at("abab")(aa,bb,ia,kb) )
    
    // r2_1p.at("abab") += -1.00 d-_bb(k,c) t1_bb(c,j) t0_1p t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0032_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_2p.at("abab") += -2.00 d+_bb(k,c) t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0032_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_2p.at("abab") += -4.00 d-_bb(k,c) t1_bb(c,j) t2_1p_abab(a,b,i,k) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * t0_2p * tmps.at("0032_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2.at("abab") += -1.00 d-_bb(k,c) t1_bb(c,j) t2_1p_abab(a,b,i,k) 
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0032_abba_vvoo")(aa,bb,jb,ia) )
    .deallocate(tmps.at("0032_abba_vvoo"))
    .allocate(tmps.at("0033_abba_vvoo"))
    
    // flops: o2v2  = o2v1 o3v2
    //  mems: o2v2  = o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = dp.at("bb_ov")(kb,cb) * t1_1p.at("bb")(cb,jb) )
    ( tmps.at("0033_abba_vvoo")(aa,bb,jb,ia)  = tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_1p.at("abab") += -1.00 d-_bb(k,c) t2_abab(a,b,i,k) t0_1p t1_1p_bb(c,j) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0033_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_2p.at("abab") += -2.00 d+_bb(k,c) t2_abab(a,b,i,k) t1_1p_bb(c,j) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0033_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_2p.at("abab") += -4.00 d-_bb(k,c) t2_abab(a,b,i,k) t1_1p_bb(c,j) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * t0_2p * tmps.at("0033_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2.at("abab") += -1.00 d-_bb(k,c) t2_abab(a,b,i,k) t1_1p_bb(c,j) 
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0033_abba_vvoo")(aa,bb,jb,ia) )
    .deallocate(tmps.at("0033_abba_vvoo"))
    .allocate(tmps.at("0034_abba_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0034_abba_vvoo")(aa,bb,jb,ia)  = dp.at("bb_oo")(kb,jb) * t2_1p.at("abab")(aa,bb,ia,kb) )
    
    // r2_1p.at("abab") += -1.00 d-_bb(k,j) t0_1p t2_1p_abab(a,b,i,k) 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0034_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_2p.at("abab") += -2.00 d+_bb(k,j) t2_1p_abab(a,b,i,k) 
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0034_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_2p.at("abab") += -4.00 d-_bb(k,j) t2_1p_abab(a,b,i,k) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 4.00 * t0_2p * tmps.at("0034_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2.at("abab") += -1.00 d-_bb(k,j) t2_1p_abab(a,b,i,k) 
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0034_abba_vvoo")(aa,bb,jb,ia) )
    .deallocate(tmps.at("0034_abba_vvoo"))
    .allocate(tmps.at("0035_baab_ovoo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1
    ( tmps.at("bin_Q")(Q)  = chol.at("bb_ovQ")(kb,cb,Q) * t1.at("bb")(cb,kb) )
    ( tmps.at("bin_bb_vo")(db,lb)  = tmps.at("bin_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("0035_baab_ovoo")(lb,aa,ia,jb)  = tmps.at("bin_bb_vo")(db,lb) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_1p.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,d,i,j) t1_bb(c,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0035_baab_ovoo")(lb,aa,ia,jb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p.at("abab") += +2.00 <l,k||c,d>_bbbb t2_abab(a,d,i,j) t1_bb(c,k) t1_2p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0035_baab_ovoo")(lb,aa,ia,jb) * t1_2p.at("bb")(bb,lb) )
    
    // r2.at("abab") += +1.00 <l,k||c,d>_bbbb t2_abab(a,d,i,j) t1_bb(b,l) t1_bb(c,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0035_baab_ovoo")(lb,aa,ia,jb) * t1.at("bb")(bb,lb) )
    .deallocate(tmps.at("0035_baab_ovoo"))
    .allocate(tmps.at("0036_baab_ovoo"))
    
    // flops: o3v1  = o1v1Q1 o1v1Q1 o3v2
    //  mems: o3v1  = o0v0Q1 o1v1 o3v1
    ( tmps.at("bin_Q")(Q)  = chol.at("aa_ovQ")(ka,ca,Q) * t1.at("aa")(ca,ka) )
    ( tmps.at("bin_bb_vo")(db,lb)  = tmps.at("bin_Q")(Q) * chol.at("bb_ovQ")(lb,db,Q) )
    ( tmps.at("0036_baab_ovoo")(lb,aa,ia,jb)  = tmps.at("bin_bb_vo")(db,lb) * t2.at("abab")(aa,db,ia,jb) )
    
    // r2_1p.at("abab") += -1.00 <k,l||c,d>_abab t2_abab(a,d,i,j) t1_aa(c,k) t1_1p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0036_baab_ovoo")(lb,aa,ia,jb) * t1_1p.at("bb")(bb,lb) )
    
    // r2_2p.at("abab") += -2.00 <k,l||c,d>_abab t2_abab(a,d,i,j) t1_aa(c,k) t1_2p_bb(b,l) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * tmps.at("0036_baab_ovoo")(lb,aa,ia,jb) * t1_2p.at("bb")(bb,lb) )
    
    // r2.at("abab") += -1.00 <k,l||c,d>_abab t2_abab(a,d,i,j) t1_bb(b,l) t1_aa(c,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= tmps.at("0036_baab_ovoo")(lb,aa,ia,jb) * t1.at("bb")(bb,lb) )
    .deallocate(tmps.at("0036_baab_ovoo"))
    .allocate(tmps.at("0037_abab_vooo"))
    
    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    ( tmps.at("0037_abab_vooo")(aa,kb,ia,jb)  = t2.at("abab")(aa,cb,ia,jb) * f.at("bb_ov")(kb,cb) )
    
    // r2_1p.at("abab") += -1.00 f_bb(k,c) t2_abab(a,c,i,j) t1_1p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= t1_1p.at("bb")(bb,kb) * tmps.at("0037_abab_vooo")(aa,kb,ia,jb) )
    
    // r2_2p.at("abab") += -2.00 f_bb(k,c) t2_abab(a,c,i,j) t1_2p_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2_2p.at("abab")(aa,bb,ia,jb) -= 2.00 * t1_2p.at("bb")(bb,kb) * tmps.at("0037_abab_vooo")(aa,kb,ia,jb) )
    
    // r2.at("abab") += -1.00 f_bb(k,c) t2_abab(a,c,i,j) t1_bb(b,k) 
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t1.at("bb")(bb,kb) * tmps.at("0037_abab_vooo")(aa,kb,ia,jb) )
    .deallocate(tmps.at("0037_abab_vooo"))
    .allocate(tmps.at("0038_abba_vvoo"))
    
    // flops: o2v2  = o2v1 o3v2
    //  mems: o2v2  = o2v0 o2v2
    ( tmps.at("bin_bb_oo")(jb,kb)  = dp.at("bb_ov")(kb,cb) * t1.at("bb")(cb,jb) )
    ( tmps.at("0038_abba_vvoo")(aa,bb,jb,ia)  = tmps.at("bin_bb_oo")(jb,kb) * t2.at("abab")(aa,bb,ia,kb) )
    
    // r2_1p.at("abab") += -1.00 d+_bb(k,c) t2_abab(a,b,i,k) t1_bb(c,j) 
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= tmps.at("0038_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2_1p.at("abab") += -2.00 d-_bb(k,c) t2_abab(a,b,i,k) t1_bb(c,j) t0_2p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2_1p.at("abab")(aa,bb,ia,jb) -= 2.00 * t0_2p * tmps.at("0038_abba_vvoo")(aa,bb,jb,ia) )
    
    // r2.at("abab") += -1.00 d-_bb(k,c) t2_abab(a,b,i,k) t1_bb(c,j) t0_1p 
    // flops: o2v2 += o2v2
    //  mems: o2v2 += o2v2
    ( r2.at("abab")(aa,bb,ia,jb) -= t0_1p * tmps.at("0038_abba_vvoo")(aa,bb,jb,ia) )
    .deallocate(tmps.at("0038_abba_vvoo"))
    .allocate(tmps.at("0039_abba_vvoo"))
    
    // flops: o2v2  = o3v2
    //  mems: o2v2  = o2v2
    ( tmps.at("0039_abba_vvoo")(aa,bb,jb,ia)  = dp.at("bb_oo")(kb,jb) * t2.at("abab")(aa,bb,ia,kb) )
    
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