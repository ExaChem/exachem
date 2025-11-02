/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "ccsd_cs.hpp"

namespace exachem::cc::ccsd_cs {

template<typename T>
void residuals(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO, const TensorMap<T>& f,
               const TensorMap<T>& eri, const TensorMap<T>& t1, const TensorMap<T>& t2,
               Tensor<T>& energy, TensorMap<T>& r1, TensorMap<T>& r2) {
  const TiledIndexSpace& O       = MO("occ");
  const TiledIndexSpace& V       = MO("virt");
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

  std::tie(aa, ba, ca, da) = Va.labels<4>("all");
  std::tie(ab, bb, cb, db) = Vb.labels<4>("all");
  std::tie(ia, ja, ka, la) = Oa.labels<4>("all");
  std::tie(ib, jb, kb, lb) = Ob.labels<4>("all");

  TensorMap<T> tmps, scalars;

  const auto timer_start = std::chrono::high_resolution_clock::now();
  const auto profile     = chem_env.ioptions.ccsd_options.profile_ccsd;

  {
    tmps["bin1_aa_oo"]     = declare<T>(chem_env, "bin1_aa_oo");
    tmps["bin1_aa_vo"]     = declare<T>(chem_env, "bin1_aa_vo");
    tmps["bin1_aa_vv"]     = declare<T>(chem_env, "bin1_aa_vv");
    tmps["bin1_aaaa_vvoo"] = declare<T>(chem_env, "bin1_aaaa_vvoo");
    tmps["bin1_aabb_oooo"] = declare<T>(chem_env, "bin1_aabb_oooo");
    tmps["bin1_aabb_vooo"] = declare<T>(chem_env, "bin1_aabb_vooo");
    tmps["bin1_aabb_vvoo"] = declare<T>(chem_env, "bin1_aabb_vvoo");
    tmps["bin1_abab_vvoo"] = declare<T>(chem_env, "bin1_abab_vvoo");
    tmps["bin1_baab_vooo"] = declare<T>(chem_env, "bin1_baab_vooo");
    tmps["bin1_bb_oo"]     = declare<T>(chem_env, "bin1_bb_oo");
    tmps["bin1_bb_vo"]     = declare<T>(chem_env, "bin1_bb_vo");
    tmps["bin1_bb_vv"]     = declare<T>(chem_env, "bin1_bb_vv");
    tmps["bin1_bbaa_vvoo"] = declare<T>(chem_env, "bin1_bbaa_vvoo");
    tmps["bin1_bbbb_vooo"] = declare<T>(chem_env, "bin1_bbbb_vooo");
    tmps["bin1_bbbb_vvoo"] = declare<T>(chem_env, "bin1_bbbb_vvoo");
    tmps["bin2_aabb_vooo"] = declare<T>(chem_env, "bin2_aabb_vooo");
    tmps["bin2_baab_vooo"] = declare<T>(chem_env, "bin2_baab_vooo");

    scalars["0001"]() = Tensor<T>{};
    scalars["0002"]() = Tensor<T>{};
    scalars["0003"]() = Tensor<T>{};
    scalars["0004"]() = Tensor<T>{};
    scalars["0005"]() = Tensor<T>{};
    scalars["0006"]() = Tensor<T>{};
    scalars["0007"]() = Tensor<T>{};
    scalars["0008"]() = Tensor<T>{};
  }

  for(auto& [name, tmp]: tmps) sch.allocate(tmp);
  for(auto& [name, scalar]: scalars) sch.allocate(scalar);

  {
    tmps["0001_bb_ov"]     = declare<T>(chem_env, "0001_bb_ov");
    tmps["0002_abab_oooo"] = declare<T>(chem_env, "0002_abab_oooo");
    tmps["0003_abab_voov"] = declare<T>(chem_env, "0003_abab_voov");
    tmps["0004_baab_vovo"] = declare<T>(chem_env, "0004_baab_vovo");
    tmps["0005_aa_oo"]     = declare<T>(chem_env, "0005_aa_oo");
    tmps["0006_aaaa_oovo"] = declare<T>(chem_env, "0006_aaaa_oovo");
    tmps["0007_abab_oooo"] = declare<T>(chem_env, "0007_abab_oooo");
    tmps["0008_aa_vv"]     = declare<T>(chem_env, "0008_aa_vv");
    tmps["0009_bb_ov"]     = declare<T>(chem_env, "0009_bb_ov");
    tmps["0010_aa_ov"]     = declare<T>(chem_env, "0010_aa_ov");
    tmps["0011_abba_oovo"] = declare<T>(chem_env, "0011_abba_oovo");
    tmps["0012_abba_oooo"] = declare<T>(chem_env, "0012_abba_oooo");
    tmps["0013_aa_oo"]     = declare<T>(chem_env, "0013_aa_oo");
  }

  sch(tmps.at("bin1_bb_vo")(bb, jb) = eri.at("bbbb_oovv")(ib, jb, ab, bb) * t1.at("bb")(ab, ib))(
    scalars.at("0001")() = tmps.at("bin1_bb_vo")(bb, jb) * t1.at("bb")(bb, jb))(
    scalars.at("0002")() = eri.at("bbbb_oovv")(ib, jb, ab, bb) * t2.at("bbbb")(ab, bb, jb, ib))(
    tmps.at("bin1_aa_vo")(ba, ja) = eri.at("abab_oovv")(ja, ib, ba, ab) * t1.at("bb")(ab, ib))(
    scalars.at("0003")() = tmps.at("bin1_aa_vo")(ba, ja) * t1.at("aa")(ba, ja))(
    scalars.at("0004")() = eri.at("abab_oovv")(ia, jb, aa, bb) * t2.at("abab")(aa, bb, ia, jb))(
    tmps.at("bin1_aa_vo")(ba, ja) = eri.at("aaaa_oovv")(ia, ja, aa, ba) * t1.at("aa")(aa, ia))(
    scalars.at("0005")() = tmps.at("bin1_aa_vo")(ba, ja) * t1.at("aa")(ba, ja))(
    scalars.at("0006")() = eri.at("aaaa_oovv")(ia, ja, aa, ba) * t2.at("aaaa")(aa, ba, ja, ia))(
    scalars.at("0007")() = f.at("bb_ov")(ib, ab) * t1.at("bb")(ab, ib))(
    scalars.at("0008")() = f.at("aa_ov")(ia, aa) * t1.at("aa")(aa, ia))

    // r2[abab]  = +1.00 <a,b||i,j>_abab
    (r2.at("abab")(aa, bb, ia, jb) = eri.at("abab_vvoo")(aa, bb, ia, jb))

    // energy()  = -0.50 <j,i||a,b>_bbbb t1_bb(a,i) t1_bb(b,j)
    (energy() = 0.50 * scalars.at("0001")())

    // r1[aa]  = +1.00 f_aa(a,i)
    (r1.at("aa")(aa, ia) = f.at("aa_vo")(aa, ia))

    // energy() += +0.250 <j,i||a,b>_bbbb t2_bbbb(a,b,j,i)
    (energy() -= 0.250 * scalars.at("0002")())

    // energy() += +0.50 <i,j||a,b>_abab t1_aa(a,i) t1_bb(b,j)
    //        += +0.50 <j,i||b,a>_abab t1_bb(a,i) t1_aa(b,j)
    (energy() += scalars.at("0003")())

    // energy() += +0.250 <j,i||a,b>_abab t2_abab(a,b,j,i)
    //        += +0.250 <i,j||a,b>_abab t2_abab(a,b,i,j)
    //        += +0.250 <j,i||b,a>_abab t2_abab(b,a,j,i)
    //        += +0.250 <i,j||b,a>_abab t2_abab(b,a,i,j)
    (energy() += scalars.at("0004")())

    // energy() += -0.50 <j,i||a,b>_aaaa t1_aa(a,i) t1_aa(b,j)
    (energy() += 0.50 * scalars.at("0005")())

    // energy() += +0.250 <j,i||a,b>_aaaa t2_aaaa(a,b,j,i)
    (energy() -= 0.250 * scalars.at("0006")())

    // energy() += +1.00 f_bb(i,a) t1_bb(a,i)
    (energy() += scalars.at("0007")())

    // energy() += +1.00 f_aa(i,a) t1_aa(a,i)
    (energy() += scalars.at("0008")())

    // r1[aa] += -1.00 f_aa(j,i) t1_aa(a,j)
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -= f.at("aa_oo")(ja, ia) * t1.at("aa")(aa, ja))

    // r1[aa] += +1.00 f_aa(a,b) t1_aa(b,i)
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) += f.at("aa_vv")(aa, ba) * t1.at("aa")(ba, ia))

    // r1[aa] += -1.00 f_aa(j,b) t2_aaaa(b,a,i,j)
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -= f.at("aa_ov")(ja, ba) * t2.at("aaaa")(ba, aa, ia, ja))

    // r1[aa] += +1.00 <j,a||b,i>_aaaa t1_aa(b,j)
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -= eri.at("aaaa_vovo")(aa, ja, ba, ia) * t1.at("aa")(ba, ja))

    // r1[aa] += +1.00 f_bb(j,b) t2_abab(a,b,i,j)
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) += f.at("bb_ov")(jb, bb) * t2.at("abab")(aa, bb, ia, jb))

    // r1[aa] += +1.00 <a,j||i,b>_abab t1_bb(b,j)
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -= eri.at("abba_vovo")(aa, jb, bb, ia) * t1.at("bb")(bb, jb))

    // r1[aa] += -0.50 <k,j||b,i>_aaaa t2_aaaa(b,a,k,j)
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) +=
     0.50 * eri.at("aaaa_oovo")(ja, ka, ba, ia) * t2.at("aaaa")(ba, aa, ka, ja))

    // r1[aa] += -0.50 <k,j||i,b>_abab t2_abab(a,b,k,j)
    //          += -0.50 <j,k||i,b>_abab t2_abab(a,b,j,k)
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) += eri.at("abba_oovo")(ka, jb, bb, ia) * t2.at("abab")(aa, bb, ka, jb))

    // r1[aa] += -0.50 <j,a||b,c>_aaaa t2_aaaa(b,c,i,j)
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) +=
     0.50 * eri.at("aaaa_vovv")(aa, ja, ba, ca) * t2.at("aaaa")(ba, ca, ia, ja))

    // r1[aa] += +0.50 <a,j||b,c>_abab t2_abab(b,c,i,j)
    //          += +0.50 <a,j||c,b>_abab t2_abab(c,b,i,j)
    // flops: o1v1 += o2v3
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) += eri.at("abab_vovv")(aa, jb, ba, cb) * t2.at("abab")(ba, cb, ia, jb))

    // r1[aa] += -1.00 f_aa(j,b) t1_aa(a,j) t1_aa(b,i)
    // flops: o1v1 += o2v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    (tmps.at("bin1_aa_oo")(ia, ja) = f.at("aa_ov")(ja, ba) * t1.at("aa")(ba, ia))(
      r1.at("aa")(aa, ia) -= tmps.at("bin1_aa_oo")(ia, ja) * t1.at("aa")(aa, ja))

    // r1[aa] += +1.00 <k,j||b,i>_aaaa t1_aa(a,k) t1_aa(b,j)
    // flops: o1v1 += o3v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    (tmps.at("bin1_aa_oo")(ia, ka) = eri.at("aaaa_oovo")(ja, ka, ba, ia) * t1.at("aa")(ba, ja))(
      r1.at("aa")(aa, ia) -= tmps.at("bin1_aa_oo")(ia, ka) * t1.at("aa")(aa, ka))

    // r1[aa] += -1.00 <k,j||i,b>_abab t1_aa(a,k) t1_bb(b,j)
    // flops: o1v1 += o3v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    (tmps.at("bin1_aa_oo")(ia, ka) = eri.at("abba_oovo")(ka, jb, bb, ia) * t1.at("bb")(bb, jb))(
      r1.at("aa")(aa, ia) += tmps.at("bin1_aa_oo")(ia, ka) * t1.at("aa")(aa, ka))

    // r1[aa] += +1.00 <a,j||c,b>_abab t1_bb(b,j) t1_aa(c,i)
    // flops: o1v1 += o1v3 o1v2
    //  mems: o1v1 += o0v2 o1v1
    (tmps.at("bin1_aa_vv")(aa, ca) = eri.at("abab_vovv")(aa, jb, ca, bb) * t1.at("bb")(bb, jb))(
      r1.at("aa")(aa, ia) += tmps.at("bin1_aa_vv")(aa, ca) * t1.at("aa")(ca, ia))

    // r1[aa] += +0.50 <k,j||b,c>_aaaa t1_aa(a,j) t2_aaaa(b,c,i,k)
    // flops: o1v1 += o3v2 o2v1
    //  mems: o1v1 += o2v0 o1v1
    (tmps.at("bin1_aa_oo")(ia, ja) =
       eri.at("aaaa_oovv")(ja, ka, ba, ca) * t2.at("aaaa")(ba, ca, ia, ka))(
      r1.at("aa")(aa, ia) -= 0.50 * tmps.at("bin1_aa_oo")(ia, ja) * t1.at("aa")(aa, ja))

    // r2[abab] += -1.00 <k,b||i,j>_abab t1_aa(a,k)
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) += eri.at("baab_vooo")(bb, ka, ia, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += -1.00 f_aa(k,i) t2_abab(a,b,k,j)
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -= f.at("aa_oo")(ka, ia) * t2.at("abab")(aa, bb, ka, jb))

    // r2[abab] += -1.00 f_bb(k,j) t2_abab(a,b,i,k)
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -= f.at("bb_oo")(kb, jb) * t2.at("abab")(aa, bb, ia, kb))

    // r2[abab] += -1.00 <a,k||i,j>_abab t1_bb(b,k)
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -= eri.at("abab_vooo")(aa, kb, ia, jb) * t1.at("bb")(bb, kb))

    // r2[abab] += +1.00 <a,b||c,j>_abab t1_aa(c,i)
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) += eri.at("abab_vvvo")(aa, bb, ca, jb) * t1.at("aa")(ca, ia))

    // r2[abab] += +1.00 f_aa(a,c) t2_abab(c,b,i,j)
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) += f.at("aa_vv")(aa, ca) * t2.at("abab")(ca, bb, ia, jb))

    // r2[abab] += +1.00 <a,b||i,c>_abab t1_bb(c,j)
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -= eri.at("abba_vvvo")(aa, bb, cb, ia) * t1.at("bb")(cb, jb))

    // r2[abab] += +1.00 f_bb(b,c) t2_abab(a,c,i,j)
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) += f.at("bb_vv")(bb, cb) * t2.at("abab")(aa, cb, ia, jb))

    // r2[abab] += +0.50 <l,k||i,j>_abab t2_abab(a,b,l,k)
    //            += +0.50 <k,l||i,j>_abab t2_abab(a,b,k,l)
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) +=
     eri.at("abab_oooo")(la, kb, ia, jb) * t2.at("abab")(aa, bb, la, kb))

    // r2[abab] += -1.00 <k,b||c,j>_abab t2_aaaa(c,a,i,k)
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) +=
     eri.at("baab_vovo")(bb, ka, ca, jb) * t2.at("aaaa")(ca, aa, ia, ka))

    // r2[abab] += +1.00 <k,a||c,i>_aaaa t2_abab(c,b,k,j)
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -=
     eri.at("aaaa_vovo")(aa, ka, ca, ia) * t2.at("abab")(ca, bb, ka, jb))

    // r2[abab] += -1.00 <a,k||c,j>_abab t2_abab(c,b,i,k)
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -=
     eri.at("abab_vovo")(aa, kb, ca, jb) * t2.at("abab")(ca, bb, ia, kb))

    // r2[abab] += -1.00 <k,b||i,c>_abab t2_abab(a,c,k,j)
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -=
     eri.at("baba_vovo")(bb, ka, cb, ia) * t2.at("abab")(aa, cb, ka, jb))

    // r2[abab] += +1.00 <k,b||c,j>_bbbb t2_abab(a,c,i,k)
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -=
     eri.at("bbbb_vovo")(bb, kb, cb, jb) * t2.at("abab")(aa, cb, ia, kb))

    // r2[abab] += -1.00 <a,k||i,c>_abab t2_bbbb(c,b,j,k)
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) +=
     eri.at("abba_vovo")(aa, kb, cb, ia) * t2.at("bbbb")(cb, bb, jb, kb))

    // r2[abab] += +0.50 <a,b||c,d>_abab t2_abab(c,d,i,j)
    //            += +0.50 <a,b||d,c>_abab t2_abab(d,c,i,j)
    // flops: o2v2 += o2v4
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) +=
     eri.at("abab_vvvv")(aa, bb, ca, db) * t2.at("abab")(ca, db, ia, jb))

    // r2[abab] += -1.00 f_aa(k,c) t2_abab(a,b,k,j) t1_aa(c,i)
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_aa_oo")(ia, ka) = f.at("aa_ov")(ka, ca) * t1.at("aa")(ca, ia))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_aa_oo")(ia, ka) * t2.at("abab")(aa, bb, ka, jb))

    // r2[abab] += +1.00 <l,k||c,i>_aaaa t2_abab(a,b,l,j) t1_aa(c,k)
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_aa_oo")(ia, la) = eri.at("aaaa_oovo")(ka, la, ca, ia) * t1.at("aa")(ca, ka))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_aa_oo")(ia, la) * t2.at("abab")(aa, bb, la, jb))

    // r2[abab] += -1.00 <l,k||i,c>_abab t2_abab(a,b,l,j) t1_bb(c,k)
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_aa_oo")(ia, la) = eri.at("abba_oovo")(la, kb, cb, ia) * t1.at("bb")(cb, kb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_aa_oo")(ia, la) * t2.at("abab")(aa, bb, la, jb))

    // r2[abab] += +1.00 <k,l||i,j>_abab t1_aa(a,k) t1_bb(b,l)
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       eri.at("abab_oooo")(ka, lb, ia, jb) *
       t1.at("bb")(bb, lb))(r2.at("abab")(aa, bb, ia, jb) +=
                            tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += -0.50 <l,k||c,d>_aaaa t2_abab(a,b,l,j) t2_aaaa(c,d,i,k)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_aa_oo")(ia, la) =
       eri.at("aaaa_oovv")(ka, la, ca, da) * t2.at("aaaa")(ca, da, ia, ka))(
      r2.at("abab")(aa, bb, ia, jb) +=
      0.50 * tmps.at("bin1_aa_oo")(ia, la) * t2.at("abab")(aa, bb, la, jb))

    // r2[abab] += -1.00 <k,b||c,j>_abab t1_aa(a,k) t1_aa(c,i)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       eri.at("baab_vovo")(bb, ka, ca, jb) *
       t1.at("aa")(ca, ia))(r2.at("abab")(aa, bb, ia, jb) +=
                            tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += -1.00 f_aa(k,c) t1_aa(a,k) t2_abab(c,b,i,j)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       f.at("aa_ov")(ka, ca) * t2.at("abab")(ca, bb, ia, jb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += -1.00 f_bb(k,c) t2_abab(a,b,i,k) t1_bb(c,j)
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_bb_oo")(jb, kb) = f.at("bb_ov")(kb, cb) * t1.at("bb")(cb, jb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_bb_oo")(jb, kb) * t2.at("abab")(aa, bb, ia, kb))

    // r2[abab] += -1.00 <k,l||c,j>_abab t2_abab(a,b,i,l) t1_aa(c,k)
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_bb_oo")(jb, lb) = eri.at("abab_oovo")(ka, lb, ca, jb) * t1.at("aa")(ca, ka))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_bb_oo")(jb, lb) * t2.at("abab")(aa, bb, ia, lb))

    // r2[abab] += +1.00 <l,k||c,j>_bbbb t2_abab(a,b,i,l) t1_bb(c,k)
    // flops: o2v2 += o3v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_bb_oo")(jb, lb) = eri.at("bbbb_oovo")(kb, lb, cb, jb) * t1.at("bb")(cb, kb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_bb_oo")(jb, lb) * t2.at("abab")(aa, bb, ia, lb))

    // r2[abab] += -1.00 <a,k||c,j>_abab t1_bb(b,k) t1_aa(c,i)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) =
       eri.at("abab_vovo")(aa, kb, ca, jb) *
       t1.at("aa")(ca, ia))(r2.at("abab")(aa, bb, ia, jb) -=
                            tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))

    // r2[abab] += -1.00 <a,k||i,c>_abab t1_bb(b,k) t1_bb(c,j)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) =
       eri.at("abba_vovo")(aa, kb, cb, ia) *
       t1.at("bb")(cb, jb))(r2.at("abab")(aa, bb, ia, jb) +=
                            tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))

    // r2[abab] += -0.50 <k,l||c,d>_abab t2_abab(a,b,i,l) t2_abab(c,d,k,j)
    //            += -0.50 <k,l||d,c>_abab t2_abab(a,b,i,l) t2_abab(d,c,k,j)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_bb_oo")(jb, lb) =
       eri.at("abab_oovv")(ka, lb, ca, db) *
       t2.at("abab")(ca, db, ka, jb))(r2.at("abab")(aa, bb, ia, jb) -=
                                      tmps.at("bin1_bb_oo")(jb, lb) * t2.at("abab")(aa, bb, ia, lb))

    // r2[abab] += -1.00 f_bb(k,c) t2_abab(a,c,i,j) t1_bb(b,k)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) =
       f.at("bb_ov")(kb, cb) * t2.at("abab")(aa, cb, ia, jb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))

    // r2[abab] += -1.00 <k,b||i,c>_abab t1_aa(a,k) t1_bb(c,j)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       eri.at("baba_vovo")(bb, ka, cb, ia) *
       t1.at("bb")(cb, jb))(r2.at("abab")(aa, bb, ia, jb) -=
                            tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += -0.50 <l,k||c,d>_bbbb t2_abab(a,b,i,l) t2_bbbb(c,d,j,k)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_bb_oo")(jb, lb) =
       eri.at("bbbb_oovv")(kb, lb, cb, db) * t2.at("bbbb")(cb, db, jb, kb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      0.50 * tmps.at("bin1_bb_oo")(jb, lb) * t2.at("abab")(aa, bb, ia, lb))

    // r2[abab] += +1.00 <a,k||d,c>_abab t2_abab(d,b,i,j) t1_bb(c,k)
    // flops: o2v2 += o1v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    (tmps.at("bin1_aa_vv")(aa, da) = eri.at("abab_vovv")(aa, kb, da, cb) * t1.at("bb")(cb, kb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_aa_vv")(aa, da) * t2.at("abab")(da, bb, ia, jb))

    // r2[abab] += -0.50 <l,k||c,d>_aaaa t2_aaaa(c,a,l,k) t2_abab(d,b,i,j)
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    (tmps.at("bin1_aa_vv")(aa, da) =
       eri.at("aaaa_oovv")(ka, la, ca, da) * t2.at("aaaa")(ca, aa, la, ka))(
      r2.at("abab")(aa, bb, ia, jb) +=
      0.50 * tmps.at("bin1_aa_vv")(aa, da) * t2.at("abab")(da, bb, ia, jb))

    // r2[abab] += -0.50 <l,k||d,c>_abab t2_abab(a,c,l,k) t2_abab(d,b,i,j)
    //            += -0.50 <k,l||d,c>_abab t2_abab(a,c,k,l) t2_abab(d,b,i,j)
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    (tmps.at("bin1_aa_vv")(aa, da) =
       eri.at("abab_oovv")(la, kb, da, cb) *
       t2.at("abab")(aa, cb, la, kb))(r2.at("abab")(aa, bb, ia, jb) -=
                                      tmps.at("bin1_aa_vv")(aa, da) * t2.at("abab")(da, bb, ia, jb))

    // r2[abab] += +1.00 <k,b||c,d>_abab t2_abab(a,d,i,j) t1_aa(c,k)
    // flops: o2v2 += o1v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    (tmps.at("bin1_bb_vv")(bb, db) = eri.at("baab_vovv")(bb, ka, ca, db) * t1.at("aa")(ca, ka))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_bb_vv")(bb, db) * t2.at("abab")(aa, db, ia, jb))

    // r2[abab] += +1.00 <k,b||c,d>_bbbb t2_abab(a,d,i,j) t1_bb(c,k)
    // flops: o2v2 += o1v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    (tmps.at("bin1_bb_vv")(bb, db) = eri.at("bbbb_vovv")(bb, kb, cb, db) * t1.at("bb")(cb, kb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_bb_vv")(bb, db) * t2.at("abab")(aa, db, ia, jb))

    // r2[abab] += -0.50 <l,k||d,c>_abab t2_abab(a,c,i,j) t2_abab(d,b,l,k)
    //            += -0.50 <k,l||d,c>_abab t2_abab(a,c,i,j) t2_abab(d,b,k,l)
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    (tmps.at("bin1_bb_vv")(bb, cb) =
       eri.at("abab_oovv")(la, kb, da, cb) *
       t2.at("abab")(da, bb, la, kb))(r2.at("abab")(aa, bb, ia, jb) -=
                                      tmps.at("bin1_bb_vv")(bb, cb) * t2.at("abab")(aa, cb, ia, jb))

    // r2[abab] += +0.50 <l,k||c,d>_bbbb t2_abab(a,c,i,j) t2_bbbb(d,b,l,k)
    // flops: o2v2 += o2v3 o2v3
    //  mems: o2v2 += o0v2 o2v2
    (tmps.at("bin1_bb_vv")(bb, cb) =
       eri.at("bbbb_oovv")(kb, lb, cb, db) * t2.at("bbbb")(db, bb, lb, kb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      0.50 * tmps.at("bin1_bb_vv")(bb, cb) * t2.at("abab")(aa, cb, ia, jb))

    // r2[abab] += +1.00 <a,b||d,c>_abab t1_bb(c,j) t1_aa(d,i)
    // flops: o2v2 += o1v4 o2v3
    //  mems: o2v2 += o1v3 o2v2
    (tmps.at("bin1_abab_vvoo")(da, cb, ia, jb) = t1.at("aa")(da, ia) * t1.at("bb")(cb, jb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      eri.at("abab_vvvv")(aa, bb, da, cb) * tmps.at("bin1_abab_vvoo")(da, cb, ia, jb))

    // r2[abab] += +1.00 <l,k||c,j>_abab t2_aaaa(c,a,i,l) t1_bb(b,k)
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) =
       eri.at("abab_oovo")(la, kb, ca, jb) * t2.at("aaaa")(ca, aa, ia, la))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))

    // r2[abab] += -1.00 <l,k||c,i>_aaaa t1_aa(a,k) t2_abab(c,b,l,j)
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       eri.at("aaaa_oovo")(ka, la, ca, ia) * t2.at("abab")(ca, bb, la, jb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += +0.50 <l,k||c,j>_abab t2_abab(a,b,l,k) t1_aa(c,i)
    //            += +0.50 <k,l||c,j>_abab t2_abab(a,b,k,l) t1_aa(c,i)
    // flops: o2v2 += o4v1 o4v2
    //  mems: o2v2 += o4v0 o2v2
    (tmps.at("bin1_aabb_oooo")(ia, la, jb, kb) =
       eri.at("abab_oovo")(la, kb, ca, jb) * t1.at("aa")(ca, ia))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_aabb_oooo")(ia, la, jb, kb) * t2.at("abab")(aa, bb, la, kb))

    // r2[abab] += +1.00 <k,l||c,j>_abab t1_aa(a,k) t2_abab(c,b,i,l)
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       eri.at("abab_oovo")(ka, lb, ca, jb) * t2.at("abab")(ca, bb, ia, lb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += +1.00 <l,k||i,c>_abab t2_abab(a,c,l,j) t1_bb(b,k)
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) =
       eri.at("abba_oovo")(la, kb, cb, ia) * t2.at("abab")(aa, cb, la, jb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))

    // r2[abab] += -1.00 <l,k||c,j>_bbbb t2_abab(a,c,i,l) t1_bb(b,k)
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) =
       eri.at("bbbb_oovo")(kb, lb, cb, jb) * t2.at("abab")(aa, cb, ia, lb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))

    // r2[abab] += +1.00 <k,l||i,c>_abab t1_aa(a,k) t2_bbbb(c,b,j,l)
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       eri.at("abba_oovo")(ka, lb, cb, ia) * t2.at("bbbb")(cb, bb, jb, lb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += -1.00 <k,a||c,d>_aaaa t2_abab(d,b,k,j) t1_aa(c,i)
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    (tmps.at("bin1_aaaa_vvoo")(aa, da, ia, ka) =
       eri.at("aaaa_vovv")(aa, ka, ca, da) * t1.at("aa")(ca, ia))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_aaaa_vvoo")(aa, da, ia, ka) * t2.at("abab")(da, bb, ka, jb))

    // r2[abab] += +1.00 <l,k||c,d>_aaaa t2_aaaa(c,a,i,k) t2_abab(d,b,l,j)
    // flops: o2v2 += o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    (tmps.at("bin1_aaaa_vvoo")(aa, da, ia, la) =
       eri.at("aaaa_oovv")(ka, la, ca, da) * t2.at("aaaa")(ca, aa, ia, ka))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_aaaa_vvoo")(aa, da, ia, la) * t2.at("abab")(da, bb, la, jb))

    // r2[abab] += +1.00 <l,k||d,c>_abab t2_abab(a,c,i,k) t2_abab(d,b,l,j)
    // flops: o2v2 += o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    (tmps.at("bin1_aaaa_vvoo")(aa, da, ia, la) =
       eri.at("abab_oovv")(la, kb, da, cb) * t2.at("abab")(aa, cb, ia, kb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_aaaa_vvoo")(aa, da, ia, la) * t2.at("abab")(da, bb, la, jb))

    // r2[abab] += -0.50 <a,k||c,d>_abab t1_bb(b,k) t2_abab(c,d,i,j)
    //            += -0.50 <a,k||d,c>_abab t1_bb(b,k) t2_abab(d,c,i,j)
    // flops: o2v2 += o3v3 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) =
       eri.at("abab_vovv")(aa, kb, ca, db) * t2.at("abab")(ca, db, ia, jb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))

    // r2[abab] += -1.00 <a,k||d,c>_abab t2_abab(d,b,i,k) t1_bb(c,j)
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    (tmps.at("bin1_aabb_vvoo")(aa, da, jb, kb) =
       eri.at("abab_vovv")(aa, kb, da, cb) * t1.at("bb")(cb, jb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_aabb_vvoo")(aa, da, jb, kb) * t2.at("abab")(da, bb, ia, kb))

    // r2[abab] += +1.00 <k,l||d,c>_abab t2_abab(a,c,k,j) t2_abab(d,b,i,l)
    // flops: o2v2 += o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    (tmps.at("bin1_aabb_vvoo")(aa, da, jb, lb) =
       eri.at("abab_oovv")(ka, lb, da, cb) * t2.at("abab")(aa, cb, ka, jb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_aabb_vvoo")(aa, da, jb, lb) * t2.at("abab")(da, bb, ia, lb))

    // r2[abab] += -0.50 <k,b||c,d>_abab t1_aa(a,k) t2_abab(c,d,i,j)
    //            += -0.50 <k,b||d,c>_abab t1_aa(a,k) t2_abab(d,c,i,j)
    // flops: o2v2 += o3v3 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       eri.at("baab_vovv")(bb, ka, ca, db) * t2.at("abab")(ca, db, ia, jb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += -1.00 <k,b||c,d>_abab t2_abab(a,d,k,j) t1_aa(c,i)
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    (tmps.at("bin1_bbaa_vvoo")(bb, db, ia, ka) =
       eri.at("baab_vovv")(bb, ka, ca, db) * t1.at("aa")(ca, ia))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_bbaa_vvoo")(bb, db, ia, ka) * t2.at("abab")(aa, db, ka, jb))

    // r2[abab] += -1.00 <a,k||c,d>_abab t2_bbbb(d,b,j,k) t1_aa(c,i)
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    (tmps.at("bin1_abab_vvoo")(aa, db, ia, kb) =
       eri.at("abab_vovv")(aa, kb, ca, db) * t1.at("aa")(ca, ia))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_abab_vvoo")(aa, db, ia, kb) * t2.at("bbbb")(db, bb, jb, kb))

    // r2[abab] += -1.00 <k,b||c,d>_bbbb t2_abab(a,d,i,k) t1_bb(c,j)
    // flops: o2v2 += o2v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    (tmps.at("bin1_bbbb_vvoo")(bb, db, jb, kb) =
       eri.at("bbbb_vovv")(bb, kb, cb, db) * t1.at("bb")(cb, jb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin1_bbbb_vvoo")(bb, db, jb, kb) * t2.at("abab")(aa, db, ia, kb))

    // r2[abab] += +1.00 <l,k||c,d>_bbbb t2_abab(a,c,i,k) t2_bbbb(d,b,j,l)
    // flops: o2v2 += o3v3 o3v3
    //  mems: o2v2 += o2v2 o2v2
    (tmps.at("bin1_abab_vvoo")(aa, db, ia, lb) =
       eri.at("bbbb_oovv")(kb, lb, cb, db) * t2.at("abab")(aa, cb, ia, kb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin1_abab_vvoo")(aa, db, ia, lb) * t2.at("bbbb")(db, bb, jb, lb))

    // r2[abab] += +1.00 <k,l||c,j>_abab t1_aa(a,k) t1_bb(b,l) t1_aa(c,i)
    // flops: o2v2 += o4v1 o4v1 o3v2
    //  mems: o2v2 += o4v0 o3v1 o2v2
    (tmps.at("bin1_aabb_oooo")(ia, ka, jb, lb) =
       eri.at("abab_oovo")(ka, lb, ca, jb) *
       t1.at("aa")(ca, ia))(tmps.at("bin2_baab_vooo")(bb, ia, ka, jb) =
                              tmps.at("bin1_aabb_oooo")(ia, ka, jb, lb) * t1.at("bb")(bb, lb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin2_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += -1.00 <l,k||d,c>_abab t1_aa(a,l) t2_abab(d,b,i,j) t1_bb(c,k)
    // flops: o2v2 += o2v2 o3v2 o3v2
    //  mems: o2v2 += o1v1 o3v1 o2v2
    (tmps.at("bin1_aa_vo")(da, la) = eri.at("abab_oovv")(la, kb, da, cb) * t1.at("bb")(cb, kb))(
      tmps.at("bin2_baab_vooo")(bb, ia, la, jb) =
        tmps.at("bin1_aa_vo")(da, la) * t2.at("abab")(da, bb, ia, jb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin2_baab_vooo")(bb, ia, la, jb) * t1.at("aa")(aa, la))

    // r2[abab] += -1.00 <a,k||d,c>_abab t1_bb(b,k) t1_bb(c,j) t1_aa(d,i)
    // flops: o2v2 += o2v3 o3v2 o3v2
    //  mems: o2v2 += o2v2 o3v1 o2v2
    (tmps.at("bin1_abab_vvoo")(aa, cb, ia, kb) =
       eri.at("abab_vovv")(aa, kb, da, cb) *
       t1.at("aa")(da, ia))(tmps.at("bin2_aabb_vooo")(aa, ia, jb, kb) =
                              tmps.at("bin1_abab_vvoo")(aa, cb, ia, kb) * t1.at("bb")(cb, jb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin2_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))

    // r2[abab] += +1.00 <k,l||d,c>_abab t1_aa(a,k) t2_abab(d,b,i,l) t1_bb(c,j)
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(da, ka, jb, lb) =
       eri.at("abab_oovv")(ka, lb, da, cb) * t1.at("bb")(cb, jb))(
      tmps.at("bin2_baab_vooo")(bb, ia, ka, jb) =
        tmps.at("bin1_aabb_vooo")(da, ka, jb, lb) * t2.at("abab")(da, bb, ia, lb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin2_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))

    // r2[abab] += +1.00 <l,k||c,d>_abab t2_abab(a,d,l,j) t1_bb(b,k) t1_aa(c,i)
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(db, ia, la, kb) =
       eri.at("abab_oovv")(la, kb, ca, db) * t1.at("aa")(ca, ia))(
      tmps.at("bin2_aabb_vooo")(aa, ia, jb, kb) =
        tmps.at("bin1_baab_vooo")(db, ia, la, kb) * t2.at("abab")(aa, db, la, jb))(
      r2.at("abab")(aa, bb, ia, jb) +=
      tmps.at("bin2_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))

    // r2[abab] += +1.00 <l,k||c,d>_bbbb t2_abab(a,d,i,l) t1_bb(b,k) t1_bb(c,j)
    // flops: o2v2 += o3v2 o4v2 o3v2
    //  mems: o2v2 += o3v1 o3v1 o2v2
    (tmps.at("bin1_bbbb_vooo")(db, jb, kb, lb) =
       eri.at("bbbb_oovv")(kb, lb, cb, db) * t1.at("bb")(cb, jb))(
      tmps.at("bin2_aabb_vooo")(aa, ia, jb, kb) =
        tmps.at("bin1_bbbb_vooo")(db, jb, kb, lb) * t2.at("abab")(aa, db, ia, lb))(
      r2.at("abab")(aa, bb, ia, jb) -=
      tmps.at("bin2_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))
      .allocate(tmps.at("0001_bb_ov"))

    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    (tmps.at("0001_bb_ov")(kb, cb) = eri.at("bbbb_oovv")(jb, kb, bb, cb) * t1.at("bb")(bb, jb))

    // r1[aa] += -1.00 <k,j||b,c>_bbbb t2_abab(a,c,i,k) t1_bb(b,j)
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) += t2.at("abab")(aa, cb, ia, kb) * tmps.at("0001_bb_ov")(kb, cb))

    // r2[abab] += +1.00 <l,k||c,d>_bbbb t2_abab(a,b,i,l) t1_bb(c,k) t1_bb(d,j)
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_bb_oo")(jb, lb) = t1.at("bb")(db, jb) * tmps.at("0001_bb_ov")(lb, db))(
        r2.at("abab")(aa, bb, ia, jb) -=
        tmps.at("bin1_bb_oo")(jb, lb) * t2.at("abab")(aa, bb, ia, lb))

    // r2[abab] += +1.00 <l,k||c,d>_bbbb t2_abab(a,d,i,j) t1_bb(b,l) t1_bb(c,k)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_bb_vv")(bb, db) = tmps.at("0001_bb_ov")(lb, db) * t1.at("bb")(bb, lb))(
        r2.at("abab")(aa, bb, ia, jb) -=
        t2.at("abab")(aa, db, ia, jb) * tmps.at("bin1_bb_vv")(bb, db))
      .deallocate(tmps.at("0001_bb_ov"))
      .allocate(tmps.at("0002_abab_oooo"))

    // flops: o4v0  = o4v2
    //  mems: o4v0  = o4v0
    (tmps.at("0002_abab_oooo")(la, kb, ia, jb) =
       eri.at("abab_oovv")(la, kb, ca, db) * t2.at("abab")(ca, db, ia, jb))

    // r2[abab] += +0.250 <l,k||c,d>_abab t2_abab(a,b,l,k) t2_abab(c,d,i,j)
    //            += +0.250 <l,k||d,c>_abab t2_abab(a,b,l,k) t2_abab(d,c,i,j)
    //            += +0.250 <k,l||c,d>_abab t2_abab(a,b,k,l) t2_abab(c,d,i,j)
    //            += +0.250 <k,l||d,c>_abab t2_abab(a,b,k,l) t2_abab(d,c,i,j)
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) +=
     t2.at("abab")(aa, bb, la, kb) * tmps.at("0002_abab_oooo")(la, kb, ia, jb))

    // r2[abab] += +0.50 <k,l||c,d>_abab t1_aa(a,k) t1_bb(b,l) t2_abab(c,d,i,j)
    //            += +0.50 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t2_abab(d,c,i,j)
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, lb) =
       tmps.at("0002_abab_oooo")(ka, lb, ia, jb) *
       t1.at("aa")(aa, ka))(r2.at("abab")(aa, bb, ia, jb) +=
                            t1.at("bb")(bb, lb) * tmps.at("bin1_aabb_vooo")(aa, ia, jb, lb))
      .deallocate(tmps.at("0002_abab_oooo"))
      .allocate(tmps.at("0003_abab_voov"))

    // flops: o2v2  = o3v3
    //  mems: o2v2  = o2v2
    (tmps.at("0003_abab_voov")(aa, lb, ia, db) =
       eri.at("abab_oovv")(ka, lb, ca, db) * t2.at("aaaa")(ca, aa, ia, ka))

    // r1[aa] += -1.00 <k,j||c,b>_abab t2_aaaa(c,a,i,k) t1_bb(b,j)
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -= t1.at("bb")(bb, jb) * tmps.at("0003_abab_voov")(aa, jb, ia, bb))

    // r2[abab] += +1.00 <k,l||c,d>_abab t2_aaaa(c,a,i,k) t2_bbbb(d,b,j,l)
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) +=
     tmps.at("0003_abab_voov")(aa, lb, ia, db) * t2.at("bbbb")(db, bb, jb, lb))

    // r2[abab] += +1.00 <l,k||d,c>_abab t2_aaaa(d,a,i,l) t1_bb(b,k) t1_bb(c,j)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) =
       tmps.at("0003_abab_voov")(aa, kb, ia, cb) *
       t1.at("bb")(cb, jb))(r2.at("abab")(aa, bb, ia, jb) +=
                            tmps.at("bin1_aabb_vooo")(aa, ia, jb, kb) * t1.at("bb")(bb, kb))
      .deallocate(tmps.at("0003_abab_voov"))
      .allocate(tmps.at("0004_baab_vovo"))

    // flops: o2v2  = o2v3
    //  mems: o2v2  = o2v2
    (tmps.at("0004_baab_vovo")(bb, ka, da, jb) =
       eri.at("baab_vovv")(bb, ka, da, cb) * t1.at("bb")(cb, jb))

    // r2[abab] += -1.00 <k,b||d,c>_abab t2_aaaa(d,a,i,k) t1_bb(c,j)
    // flops: o2v2 += o3v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) +=
     t2.at("aaaa")(da, aa, ia, ka) * tmps.at("0004_baab_vovo")(bb, ka, da, jb))

    // r2[abab] += -1.00 <k,b||d,c>_abab t1_aa(a,k) t1_bb(c,j) t1_aa(d,i)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       t1.at("aa")(da, ia) * tmps.at("0004_baab_vovo")(bb, ka, da, jb))(
        r2.at("abab")(aa, bb, ia, jb) +=
        tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))
      .deallocate(tmps.at("0004_baab_vovo"))
      .allocate(tmps.at("0005_aa_oo"))

    // flops: o2v0  = o3v2
    //  mems: o2v0  = o2v0
    (tmps.at("0005_aa_oo")(ja, ia) =
       eri.at("abab_oovv")(ja, kb, ba, cb) * t2.at("abab")(ba, cb, ia, kb))

    // r1[aa] += -0.50 <j,k||b,c>_abab t1_aa(a,j) t2_abab(b,c,i,k)
    //          += -0.50 <j,k||c,b>_abab t1_aa(a,j) t2_abab(c,b,i,k)
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -= t1.at("aa")(aa, ja) * tmps.at("0005_aa_oo")(ja, ia))

    // r2[abab] += -0.50 <l,k||c,d>_abab t2_abab(a,b,l,j) t2_abab(c,d,i,k)
    //            += -0.50 <l,k||d,c>_abab t2_abab(a,b,l,j) t2_abab(d,c,i,k)
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -= tmps.at("0005_aa_oo")(la, ia) * t2.at("abab")(aa, bb, la, jb))
      .deallocate(tmps.at("0005_aa_oo"))
      .allocate(tmps.at("0006_aaaa_oovo"))

    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    (tmps.at("0006_aaaa_oovo")(ja, ka, ca, ia) =
       eri.at("aaaa_oovv")(ja, ka, ba, ca) * t1.at("aa")(ba, ia))

    // r1[aa] += +0.50 <k,j||b,c>_aaaa t2_aaaa(c,a,k,j) t1_aa(b,i)
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -=
     0.50 * t2.at("aaaa")(ca, aa, ka, ja) * tmps.at("0006_aaaa_oovo")(ja, ka, ca, ia))

    // r2[abab] += +1.00 <l,k||c,d>_aaaa t1_aa(a,k) t2_abab(d,b,l,j) t1_aa(c,i)
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       t2.at("abab")(da, bb, la, jb) * tmps.at("0006_aaaa_oovo")(ka, la, da, ia))(
        r2.at("abab")(aa, bb, ia, jb) -=
        tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))
      .deallocate(tmps.at("0006_aaaa_oovo"))
      .allocate(tmps.at("0007_abab_oooo"))

    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    (tmps.at("0007_abab_oooo")(la, kb, ia, jb) =
       eri.at("abba_oovo")(la, kb, cb, ia) * t1.at("bb")(cb, jb))

    // r2[abab] += +0.50 <l,k||i,c>_abab t2_abab(a,b,l,k) t1_bb(c,j)
    //            += +0.50 <k,l||i,c>_abab t2_abab(a,b,k,l) t1_bb(c,j)
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -=
     t2.at("abab")(aa, bb, la, kb) * tmps.at("0007_abab_oooo")(la, kb, ia, jb))

    // r2[abab] += +1.00 <k,l||i,c>_abab t1_aa(a,k) t1_bb(b,l) t1_bb(c,j)
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, lb) =
       tmps.at("0007_abab_oooo")(ka, lb, ia, jb) *
       t1.at("aa")(aa, ka))(r2.at("abab")(aa, bb, ia, jb) -=
                            t1.at("bb")(bb, lb) * tmps.at("bin1_aabb_vooo")(aa, ia, jb, lb))
      .deallocate(tmps.at("0007_abab_oooo"))
      .allocate(tmps.at("0008_aa_vv"))

    // flops: o0v2  = o1v3
    //  mems: o0v2  = o0v2
    (tmps.at("0008_aa_vv")(aa, ca) = eri.at("aaaa_vovv")(aa, ja, ba, ca) * t1.at("aa")(ba, ja))

    // r1[aa] += +1.00 <j,a||b,c>_aaaa t1_aa(b,j) t1_aa(c,i)
    // flops: o1v1 += o1v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -= t1.at("aa")(ca, ia) * tmps.at("0008_aa_vv")(aa, ca))

    // r2[abab] += +1.00 <k,a||c,d>_aaaa t2_abab(d,b,i,j) t1_aa(c,k)
    // flops: o2v2 += o2v3
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -= tmps.at("0008_aa_vv")(aa, da) * t2.at("abab")(da, bb, ia, jb))
      .deallocate(tmps.at("0008_aa_vv"))
      .allocate(tmps.at("0009_bb_ov"))

    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    (tmps.at("0009_bb_ov")(kb, cb) = eri.at("abab_oovv")(ja, kb, ba, cb) * t1.at("aa")(ba, ja))

    // r1[aa] += +1.00 <j,k||b,c>_abab t2_abab(a,c,i,k) t1_aa(b,j)
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) += t2.at("abab")(aa, cb, ia, kb) * tmps.at("0009_bb_ov")(kb, cb))

    // r2[abab] += -1.00 <k,l||c,d>_abab t2_abab(a,b,i,l) t1_aa(c,k) t1_bb(d,j)
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_bb_oo")(jb, lb) = t1.at("bb")(db, jb) * tmps.at("0009_bb_ov")(lb, db))(
        r2.at("abab")(aa, bb, ia, jb) -=
        tmps.at("bin1_bb_oo")(jb, lb) * t2.at("abab")(aa, bb, ia, lb))

    // r2[abab] += -1.00 <k,l||c,d>_abab t2_abab(a,d,i,j) t1_bb(b,l) t1_aa(c,k)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_bb_vv")(bb, db) = tmps.at("0009_bb_ov")(lb, db) * t1.at("bb")(bb, lb))(
        r2.at("abab")(aa, bb, ia, jb) -=
        t2.at("abab")(aa, db, ia, jb) * tmps.at("bin1_bb_vv")(bb, db))
      .deallocate(tmps.at("0009_bb_ov"))
      .allocate(tmps.at("0010_aa_ov"))

    // flops: o1v1  = o2v2
    //  mems: o1v1  = o1v1
    (tmps.at("0010_aa_ov")(ka, ca) = eri.at("aaaa_oovv")(ja, ka, ba, ca) * t1.at("aa")(ba, ja))

    // r1[aa] += +1.00 <k,j||b,c>_aaaa t2_aaaa(c,a,i,k) t1_aa(b,j)
    // flops: o1v1 += o2v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -= t2.at("aaaa")(ca, aa, ia, ka) * tmps.at("0010_aa_ov")(ka, ca))

    // r1[aa] += +1.00 <k,j||b,c>_aaaa t1_aa(a,k) t1_aa(b,j) t1_aa(c,i)
    // flops: o1v1 += o2v1 o2v1
    //  mems: o1v1 += o2v0 o1v1
    (tmps.at("bin1_aa_oo")(ia, ka) = t1.at("aa")(ca, ia) * tmps.at("0010_aa_ov")(ka, ca))(
        r1.at("aa")(aa, ia) -= tmps.at("bin1_aa_oo")(ia, ka) * t1.at("aa")(aa, ka))

    // r2[abab] += +1.00 <l,k||c,d>_aaaa t2_abab(a,b,l,j) t1_aa(c,k) t1_aa(d,i)
    // flops: o2v2 += o2v1 o3v2
    //  mems: o2v2 += o2v0 o2v2
    (tmps.at("bin1_aa_oo")(ia, la) = t1.at("aa")(da, ia) * tmps.at("0010_aa_ov")(la, da))(
        r2.at("abab")(aa, bb, ia, jb) -=
        tmps.at("bin1_aa_oo")(ia, la) * t2.at("abab")(aa, bb, la, jb))

    // r2[abab] += +1.00 <l,k||c,d>_aaaa t1_aa(a,l) t2_abab(d,b,i,j) t1_aa(c,k)
    // flops: o2v2 += o3v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aa_vv")(aa, da) = tmps.at("0010_aa_ov")(la, da) * t1.at("aa")(aa, la))(
        r2.at("abab")(aa, bb, ia, jb) -=
        t2.at("abab")(da, bb, ia, jb) * tmps.at("bin1_aa_vv")(aa, da))
      .deallocate(tmps.at("0010_aa_ov"))
      .allocate(tmps.at("0011_abba_oovo"))

    // flops: o3v1  = o3v2
    //  mems: o3v1  = o3v1
    (tmps.at("0011_abba_oovo")(ka, jb, cb, ia) =
       eri.at("abab_oovv")(ka, jb, ba, cb) * t1.at("aa")(ba, ia))

    // r1[aa] += -0.50 <k,j||b,c>_abab t2_abab(a,c,k,j) t1_aa(b,i)
    //          += -0.50 <j,k||b,c>_abab t2_abab(a,c,j,k) t1_aa(b,i)
    // flops: o1v1 += o3v2
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -=
     t2.at("abab")(aa, cb, ka, jb) * tmps.at("0011_abba_oovo")(ka, jb, cb, ia))

    // r2[abab] += +1.00 <k,l||c,d>_abab t1_aa(a,k) t2_bbbb(d,b,j,l) t1_aa(c,i)
    // flops: o2v2 += o4v2 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) =
       t2.at("bbbb")(db, bb, jb, lb) * tmps.at("0011_abba_oovo")(ka, lb, db, ia))(
        r2.at("abab")(aa, bb, ia, jb) +=
        tmps.at("bin1_baab_vooo")(bb, ia, ka, jb) * t1.at("aa")(aa, ka))
      .allocate(tmps.at("0012_abba_oooo"))

    // flops: o4v0  = o4v1
    //  mems: o4v0  = o4v0
    (tmps.at("0012_abba_oooo")(la, kb, jb, ia) =
       t1.at("bb")(cb, jb) * tmps.at("0011_abba_oovo")(la, kb, cb, ia))

    // r2[abab] += +0.50 <l,k||d,c>_abab t2_abab(a,b,l,k) t1_bb(c,j) t1_aa(d,i)
    //            += +0.50 <k,l||d,c>_abab t2_abab(a,b,k,l) t1_bb(c,j) t1_aa(d,i)
    // flops: o2v2 += o4v2
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) +=
     t2.at("abab")(aa, bb, la, kb) * tmps.at("0012_abba_oooo")(la, kb, jb, ia))

    // r2[abab] += +1.00 <k,l||d,c>_abab t1_aa(a,k) t1_bb(b,l) t1_bb(c,j) t1_aa(d,i)
    // flops: o2v2 += o4v1 o3v2
    //  mems: o2v2 += o3v1 o2v2
    (tmps.at("bin1_aabb_vooo")(aa, ia, jb, lb) =
       tmps.at("0012_abba_oooo")(ka, lb, jb, ia) *
       t1.at("aa")(aa, ka))(r2.at("abab")(aa, bb, ia, jb) +=
                            t1.at("bb")(bb, lb) * tmps.at("bin1_aabb_vooo")(aa, ia, jb, lb))
      .deallocate(tmps.at("0012_abba_oooo"))
      .allocate(tmps.at("0013_aa_oo"))

    // flops: o2v0  = o3v1
    //  mems: o2v0  = o2v0
    (tmps.at("0013_aa_oo")(ka, ia) =
       t1.at("bb")(bb, jb) * tmps.at("0011_abba_oovo")(ka, jb, bb, ia))
      .deallocate(tmps.at("0011_abba_oovo"))

    // r1[aa] += -1.00 <k,j||c,b>_abab t1_aa(a,k) t1_bb(b,j) t1_aa(c,i)
    // flops: o1v1 += o2v1
    //  mems: o1v1 += o1v1
    (r1.at("aa")(aa, ia) -= t1.at("aa")(aa, ka) * tmps.at("0013_aa_oo")(ka, ia))

    // r2[abab] += -1.00 <l,k||d,c>_abab t2_abab(a,b,l,j) t1_bb(c,k) t1_aa(d,i)
    // flops: o2v2 += o3v2
    //  mems: o2v2 += o2v2
    (r2.at("abab")(aa, bb, ia, jb) -= tmps.at("0013_aa_oo")(la, ia) * t2.at("abab")(aa, bb, la, jb))
      .deallocate(tmps.at("0013_aa_oo"));

  sch.execute(sch.ec().exhw(), profile);

  // Deallocate temporary tensors
  for(auto& [name, tmp]: tmps) {
    if(tmp.is_allocated()) sch.deallocate(tmp);
  }
  for(auto& [name, scalar]: scalars) {
    if(scalar.is_allocated()) sch.deallocate(scalar);
  }

  sch.execute();

  const auto timer_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();
}

template<typename T>
std::tuple<TensorMap<T>, // fock
           TensorMap<T>  // eri
           >
extract_spin_blocks(Scheduler& sch, ChemEnv& chem_env, const Tensor<T>& d_f1,
                    const Tensor<T>& cholVpr) {
  TiledIndexSpace&       MO      = chem_env.is_context.MSO;
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
  TiledIndexLabel cind;

  std::tie(aa, ba, ca, da) = Va.labels<4>("all");
  std::tie(ab, bb, cb, db) = Vb.labels<4>("all");
  std::tie(ia, ja, ka, la) = Oa.labels<4>("all");
  std::tie(ib, jb, kb, lb) = Ob.labels<4>("all");
  std::tie(cind)           = CI.labels<1>("all");

  // one body integrals
  TensorMap<T> f = oei_spin_blocks<T>(sch, chem_env, d_f1, false);

  auto build_eri_block = [&](Tensor<T>& v2_block, const TiledIndexLabel& p,
                             const TiledIndexLabel& q, const TiledIndexLabel& r,
                             const TiledIndexLabel& s) {
    // clang-format off
    sch
      (v2_block(p, q, r, s)          = cholVpr(p, r, cind) * cholVpr(q, s, cind))
      (v2_block(p, q, r, s)         -= cholVpr(p, s, cind) * cholVpr(q, r, cind))
      ;
    // clang-format on
  };

  TensorMap<T>             eri; // electron repulsion integrals
  std::vector<std::string> eri_blocks = {
    "aaaa_oovo", "aaaa_oovv", "aaaa_vovo", "aaaa_vovv", "abab_oooo", "abab_oovo",
    "abab_oovv", "abab_vooo", "abab_vovo", "abab_vovv", "abab_vvoo", "abab_vvvo",
    "abab_vvvv", "abba_oovo", "abba_vovo", "abba_vvvo", "baab_vooo", "baab_vovo",
    "baab_vovv", "baba_vovo", "bbbb_oovo", "bbbb_oovv", "bbbb_vovo", "bbbb_vovv"};

  for(const auto& block: eri_blocks) {
    eri[block] = declare<T>(chem_env, block);
    sch.allocate(eri.at(block));

    TiledIndexLabel p, q, r, s;
    if(block[0] == 'a' && block[0 + 5] == 'o') p = ia;
    if(block[0] == 'b' && block[0 + 5] == 'o') p = ib;
    if(block[0] == 'a' && block[0 + 5] == 'v') p = aa;
    if(block[0] == 'b' && block[0 + 5] == 'v') p = ab;

    if(block[1] == 'a' && block[1 + 5] == 'o') q = ja;
    if(block[1] == 'b' && block[1 + 5] == 'o') q = jb;
    if(block[1] == 'a' && block[1 + 5] == 'v') q = ba;
    if(block[1] == 'b' && block[1 + 5] == 'v') q = bb;

    if(block[2] == 'a' && block[2 + 5] == 'o') r = ka;
    if(block[2] == 'b' && block[2 + 5] == 'o') r = kb;
    if(block[2] == 'a' && block[2 + 5] == 'v') r = ca;
    if(block[2] == 'b' && block[2 + 5] == 'v') r = cb;

    if(block[3] == 'a' && block[3 + 5] == 'o') s = la;
    if(block[3] == 'b' && block[3 + 5] == 'o') s = lb;
    if(block[3] == 'a' && block[3 + 5] == 'v') s = da;
    if(block[3] == 'b' && block[3 + 5] == 'v') s = db;

    build_eri_block(eri.at(block), p, q, r, s);
  }
  sch.execute();

  return {f, eri};
}

template<typename T>
std::tuple<double, double>
ccsd_v2_driver(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_t1,
               Tensor<T>& d_t2, Tensor<T>& d_r1, Tensor<T>& d_r2, std::vector<Tensor<T>>& d_r1s,
               std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
               std::vector<Tensor<T>>& d_t2s, TensorMap<T>& f, TensorMap<T>& eri,
               std::vector<T>& p_evl_sorted, bool ccsd_restart = false, std::string ccsd_fp = "") {
  auto cc_t1 = std::chrono::high_resolution_clock::now();

  SystemData& sys_data    = chem_env.sys_data;
  int         maxiter     = chem_env.ioptions.ccsd_options.ccsd_maxiter;
  int         ndiis       = chem_env.ioptions.ccsd_options.ndiis;
  double      thresh      = chem_env.ioptions.ccsd_options.threshold;
  bool        writet      = chem_env.ioptions.ccsd_options.writet;
  int         writet_iter = chem_env.ioptions.ccsd_options.writet_iter;
  double      zshiftl     = chem_env.ioptions.ccsd_options.lshift;
  bool        profile     = chem_env.ioptions.ccsd_options.profile_ccsd;
  double      residual    = 0.0;
  double      energy      = 0.0;
  int         niter       = 0;

  const TAMM_SIZE n_occ_alpha = static_cast<TAMM_SIZE>(sys_data.n_occ_alpha);
  const TAMM_SIZE n_occ_beta  = static_cast<TAMM_SIZE>(sys_data.n_occ_beta);

  std::string t1file = ccsd_fp + ".t1amp";
  std::string t2file = ccsd_fp + ".t2amp";

  std::cout.precision(15);

  Tensor<T> d_e{};
  Tensor<T>::allocate(&ec, d_e);
  Scheduler sch{ec};

  const TiledIndexSpace& O       = MO("occ");
  const TiledIndexSpace& V       = MO("virt");
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

  std::tie(aa, ba, ca, da) = Va.labels<4>("all");
  std::tie(ab, bb, cb, db) = Vb.labels<4>("all");
  std::tie(ia, ja, ka, la) = Oa.labels<4>("all");
  std::tie(ib, jb, kb, lb) = Ob.labels<4>("all");

  print_ccsd_header(ec.print());

  TensorMap<T> t1, t2;
  t1["aa"]   = declare<T>(chem_env, "aa_vo");
  t1["bb"]   = declare<T>(chem_env, "bb_vo");
  t2["aaaa"] = declare<T>(chem_env, "aaaa_vvoo");
  t2["abab"] = declare<T>(chem_env, "abab_vvoo");
  t2["bbbb"] = declare<T>(chem_env, "bbbb_vvoo");

  TensorMap<T> r1, r2;
  r1["aa"]   = declare<T>(chem_env, "aa_vo");
  r2["abab"] = declare<T>(chem_env, "abab_vvoo");

  // allocate tensors
  for(auto& [name, t]: t1) sch.allocate(t);
  for(auto& [name, t]: t2) sch.allocate(t);
  for(auto& [name, t]: r1) sch.allocate(t);
  for(auto& [name, t]: r2) sch.allocate(t);

  Tensor<T> tmp_aaaa = declare<T>(chem_env, "tmp_aaaa_vvoo");
  // clang-format off
    sch

    // aa and bb <= aa
    (   t1.at("aa")(aa,ia)            = d_t1(aa,ia))
    .exact_copy(t1.at("bb")(ab, ib), t1.at("aa")(ab, ib))
    // abab
    (   t2.at("abab")(aa,bb,ia,jb)    = d_t2(aa,bb,ia,jb))
    // aaaa <= abab - baab - abba + baba
    .allocate(tmp_aaaa)
    (t2.at("aaaa")() = 0.0) (tmp_aaaa() = 0.0)
    .exact_copy(tmp_aaaa(aa, ba, ia, ja), t2.at("abab")(aa, ba, ia, ja))
    (t2.at("aaaa")() = tmp_aaaa())
    (t2.at("aaaa")(aa, ba, ia, ja) -= tmp_aaaa(ba, aa, ia, ja))
    .deallocate(tmp_aaaa)
    // bbbb <= aaaa
    .exact_copy(t2.at("bbbb")(ab, bb, ib, jb), t2.at("aaaa")(ab, bb, ib, jb))
    (   d_r1(aa,ia)          = r1.at("aa")(aa,ia))
    (   d_r2(aa,bb,ia,jb)    = r2.at("abab")(aa,bb,ia,jb))
    .execute(ec.exhw(), profile);
  // clang-format on

  sch.execute(ec.exhw(), profile);

  if(!ccsd_restart) {
    Tensor<T> d_r1_residual{}, d_r2_residual{};
    Tensor<T>::allocate(&ec, d_r1_residual, d_r2_residual);

    for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
        const auto timer_start = std::chrono::high_resolution_clock::now();

        niter   = iter;
        int off = iter - titer;

        sch((d_t1s[off])() = d_t1())((d_t2s[off])() = d_t2()).execute();

        Tensor<T> tmp_aaaa = declare<T>(chem_env, "tmp_aaaa_vvoo");
        // clang-format off
          sch

          // aa and bb <= aa
          (   t1.at("aa")(aa,ia)            = d_t1(aa,ia))
          .exact_copy(t1.at("bb")(ab, ib), t1.at("aa")(ab, ib))
          // abab
          (   t2.at("abab")(aa,bb,ia,jb)    = d_t2(aa,bb,ia,jb))
          // aaaa <= abab - baab - abba + baba
          .allocate(tmp_aaaa)
          (t2.at("aaaa")() = 0.0) (tmp_aaaa() = 0.0)
          .exact_copy(tmp_aaaa(aa, ba, ia, ja), t2.at("abab")(aa, ba, ia, ja))
          (t2.at("aaaa")() = tmp_aaaa())
          (t2.at("aaaa")(aa, ba, ia, ja) -= tmp_aaaa(ba, aa, ia, ja))
          .deallocate(tmp_aaaa)
          // bbbb <= aaaa
          .exact_copy(t2.at("bbbb")(ab, bb, ib, jb), t2.at("aaaa")(ab, bb, ib, jb))
          .execute();
        // clang-format on

        ccsd_cs::residuals<T>(sch, chem_env, MO, f, eri, t1, t2, d_e, r1, r2);

        sch.execute(ec.exhw(), profile);

        // clang-format off
          sch
          (   d_r1(aa,ia)          = r1.at("aa")(aa,ia))
          (   d_r2(aa,bb,ia,jb)    = r2.at("abab")(aa,bb,ia,jb))
          .execute(ec.exhw(), profile);
        // clang-format on

        std::tie(residual, energy) = rest(ec, MO, d_r1, d_r2, d_t1, d_t2, d_e, d_r1_residual,
                                          d_r2_residual, p_evl_sorted, zshiftl, n_occ_alpha,
                                          n_occ_beta);

        update_r2(ec, d_r2());

        sch((d_r1s[off])() = d_r1())((d_r2s[off])() = d_r2()).execute();

        const auto timer_end = std::chrono::high_resolution_clock::now();
        auto       iter_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start))
            .count();

        iteration_print(chem_env, ec.pg(), iter, residual, energy, iter_time);

        if(writet && ((iter + 1) % writet_iter == 0)) {
          write_to_disk(d_t1, t1file);
          write_to_disk(d_t2, t2file);
        }

        if(residual < thresh) { break; }
      }

      if(residual < thresh || titer + ndiis >= maxiter) { break; }
      if(ec.pg().rank() == 0) {
        std::cout << " MICROCYCLE DIIS UPDATE:";
        std::cout.width(21);
        std::cout << std::right << std::min(titer + ndiis, maxiter) + 1 << std::endl;
      }

      std::vector<std::vector<Tensor<T>>> rs{d_r1s, d_r2s};
      std::vector<std::vector<Tensor<T>>> ts{d_t1s, d_t2s};
      std::vector<Tensor<T>>              next_t{d_t1, d_t2};
      diis<T>(ec, rs, ts, next_t);
    }

    if(writet) {
      write_to_disk(d_t1, t1file);
      write_to_disk(d_t2, t2file);
    }

    Tensor<T>::deallocate(d_r1_residual, d_r2_residual);

  } // no restart
  else {
    ccsd_cs::residuals<T>(sch, chem_env, MO, f, eri, t1, t2, d_e, r1, r2);

    sch.execute(ec.exhw(), profile);

    energy   = get_scalar(d_e);
    residual = 0.0;
  }

  for(auto& [name, t]: t1) sch.deallocate(t);
  for(auto& [name, t]: t2) sch.deallocate(t);
  for(auto& [name, t]: r1) sch.deallocate(t);
  for(auto& [name, t]: r2) sch.deallocate(t);

  chem_env.cc_context.ccsd_correlation_energy = energy;
  chem_env.cc_context.ccsd_total_energy       = chem_env.scf_context.hf_energy + energy;

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  if(ec.pg().rank() == 0) {
    sys_data.results["output"]["CCSD"]["n_iterations"]                = niter + 1;
    sys_data.results["output"]["CCSD"]["final_energy"]["correlation"] = energy;
    sys_data.results["output"]["CCSD"]["final_energy"]["total"] =
      chem_env.cc_context.ccsd_total_energy;
    sys_data.results["output"]["CCSD"]["performance"]["total_time"] = ccsd_time;
    chem_env.write_json_data();
  }

  return std::make_tuple(residual, energy);
}

void ccsd_cs_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T   = double;
  auto rank = ec.pg().rank();

  cholesky_2e::cholesky_2e_driver(ec, chem_env);

  std::string files_prefix = chem_env.get_files_prefix();

  CDContext& cd_context = chem_env.cd_context;
  CCContext& cc_context = chem_env.cc_context;
  cc_context.init_filenames(files_prefix);
  CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;

  auto        debug      = ccsd_options.debug;
  bool        scf_conv   = chem_env.scf_context.no_scf;
  std::string t1file     = cc_context.t1file;
  std::string t2file     = cc_context.t2file;
  const bool  ccsdstatus = cc_context.is_converged(chem_env.run_context, "ccsd");

  bool ccsd_restart = ccsd_options.readt ||
                      ((fs::exists(t1file) && fs::exists(t2file) && fs::exists(cd_context.f1file) &&
                        fs::exists(cd_context.v2file)));

  TiledIndexSpace& MO      = chem_env.is_context.MSO;
  TiledIndexSpace  N       = MO("all");
  Tensor<T>        d_f1    = chem_env.cd_context.d_f1;
  Tensor<T>        cholVpr = chem_env.cd_context.cholV2;

  std::vector<T>         p_evl_sorted;
  Tensor<T>              d_r1, d_r2, d_t1, d_t2;
  std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;

  std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) =
    setupTensors(ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && ccsdstatus && scf_conv);

  if(ccsd_restart) {
    if(fs::exists(t1file) && fs::exists(t2file)) {
      read_from_disk(d_t1, t1file);
      read_from_disk(d_t2, t2file);
    }
    p_evl_sorted = tamm::diagonal(d_f1);
  }

  if(rank == 0 && debug) {
    print_vector(p_evl_sorted, files_prefix + ".eigen_values.txt");
    cout << "Eigen values written to file: " << files_prefix + ".eigen_values.txt" << endl << endl;
  }

  ec.pg().barrier();

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  ccsd_restart = ccsd_restart && ccsdstatus && scf_conv;

  Scheduler sch{ec};

  TensorMap<T> f, eri;
  std::tie(f, eri) = extract_spin_blocks<T>(sch, chem_env, d_f1, cholVpr);

  free_tensors(cholVpr);

  auto [residual, corr_energy] =
    ccsd_cs::ccsd_v2_driver<T>(chem_env, ec, MO, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
                               f, eri, p_evl_sorted, ccsd_restart, files_prefix);

  for(auto& [block, tensor]: f) sch.deallocate(tensor);
  for(auto& [block, tensor]: eri) sch.deallocate(tensor);

  ccsd_stats(ec, chem_env.scf_context.hf_energy, residual, corr_energy, ccsd_options.threshold);

  if(ccsd_options.writet && !ccsdstatus) {
    // write_to_disk(d_t1,t1file);
    // write_to_disk(d_t2,t2file);
    chem_env.run_context["ccsd"]["converged"] = true;
  }
  else if(!ccsdstatus) chem_env.run_context["ccsd"]["converged"] = false;
  if(rank == 0) chem_env.write_run_context();

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << std::endl
              << "Time taken for Closed Shell CCSD: " << std::fixed << std::setprecision(2)
              << ccsd_time << " secs" << std::endl;

  cc_print(chem_env, d_t1, d_t2, files_prefix);

  if(!ccsd_restart) {
    free_tensors(d_r1, d_r2);
    free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
  }
  free_tensors(d_t1, d_t2, d_f1);

  ec.flush_and_sync();
  // delete ec;
}

template std::tuple<double, double>
ccsd_v2_driver<double>(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO,
                       Tensor<double>& d_t1, Tensor<double>& d_t2, Tensor<double>& d_r1,
                       Tensor<double>& d_r2, std::vector<Tensor<double>>& d_r1s,
                       std::vector<Tensor<double>>& d_r2s, std::vector<Tensor<double>>& d_t1s,
                       std::vector<Tensor<double>>& d_t2s, TensorMap<double>& f,
                       TensorMap<double>& eri, std::vector<double>& p_evl_sorted,
                       bool ccsd_restart = false, std::string ccsd_fp = "");

template void residuals<double>(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
                                const TensorMap<double>& f, const TensorMap<double>& eri,
                                const TensorMap<double>& t1, const TensorMap<double>& t2,
                                Tensor<double>& energy, TensorMap<double>& r1,
                                TensorMap<double>& r2);

template std::tuple<TensorMap<double>, // fock
                    TensorMap<double>  // eri
                    >
extract_spin_blocks(Scheduler& sch, ChemEnv& chem_env, const Tensor<double>& d_f1,
                    const Tensor<double>& cholVpr);

}; // namespace exachem::cc::ccsd_cs