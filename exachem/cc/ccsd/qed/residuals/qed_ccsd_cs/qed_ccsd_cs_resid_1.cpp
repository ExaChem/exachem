/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_cs_resid_1.hpp"

template<typename T>
void exachem::cc::qed_ccsd_cs::resid_part1(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<T>& tmps, TensorMap<T>& scalars,
  const TensorMap<T>& f, const TensorMap<T>& eri, const TensorMap<T>& dp, const double w0,
  const TensorMap<T>& t1, const TensorMap<T>& t2, const double t0_1p, const TensorMap<T>& t1_1p,
  const TensorMap<T>& t2_1p, const double t0_2p, const TensorMap<T>& t1_2p,
  const TensorMap<T>& t2_2p, Tensor<T>& energy, TensorMap<T>& r1, TensorMap<T>& r2,
  Tensor<T>& r0_1p, TensorMap<T>& r1_1p, TensorMap<T>& r2_1p, Tensor<T>& r0_2p, TensorMap<T>& r1_2p,
  TensorMap<T>& r2_2p) {
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

  {
    sch(r2_2p.at("abab")(aa, bb, ia, jb) = 4.00 * w0 * t2_2p.at("abab")(aa, bb, ia, jb))(
      r1_2p.at("aa")(aa, ia) = 4.00 * w0 * t1_2p.at("aa")(aa, ia))(
      r1.at("aa")(aa, ia) = f.at("aa_vo")(aa, ia))(r2_1p.at("abab")(aa, bb, ia, jb) =
                                                     w0 * t2_1p.at("abab")(aa, bb, ia, jb))(
      r2.at("abab")(aa, bb, ia, jb) = eri.at("abab_vvoo")(aa, bb, ia, jb))(
      energy() = scalars.at("0003")())(r1_1p.at("aa")(aa, ia) = dp.at("aa_vo")(aa, ia))(
      r0_2p() = 2.00 * scalars.at("0003")())(r0_1p() = scalars.at("0023")())(
      energy() -= 0.250 * scalars.at("0035")())(r0_2p() += 4.00 * scalars.at("0054")())(
      r1_1p.at("aa")(aa, ia) += w0 * t1_1p.at("aa")(aa, ia))(r1_1p.at("aa")(aa, ia) +=
                                                             2.00 * t0_2p * dp.at("aa_vo")(aa, ia))(
      r1.at("aa")(aa, ia) += t0_1p * dp.at("aa_vo")(aa, ia))
      .allocate(tmps.at("0016_baab_vooo"))(tmps.at("0016_baab_vooo")(ab, ia, ja, kb) =
                                             dp.at("aa_ov")(ia, ba) *
                                             t2_1p.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0017_abab_vvoo"))(tmps.at("0017_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0016_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0014_baab_vooo"))(tmps.at("0014_baab_vooo")(ab, ia, ja, kb) =
                                             dp.at("aa_ov")(ia, ba) * t2.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0015_abab_vvoo"))(tmps.at("0015_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0014_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0012_abab_vooo"))(tmps.at("0012_abab_vooo")(aa, ib, ja, kb) =
                                             dp.at("bb_ov")(ib, bb) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0013_baab_vvoo"))(tmps.at("0013_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0012_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0010_abab_vooo"))(tmps.at("0010_abab_vooo")(aa, ib, ja, kb) =
                                             dp.at("bb_ov")(ib, bb) *
                                             t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0011_baab_vvoo"))(tmps.at("0011_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0010_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0008_abab_vvoo"))(tmps.at("0008_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_oo")(ka, ia) *
                                             t2_1p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0007_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1_1p.at("aa")(ca, ia))(
        tmps.at("0007_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0006_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1.at("aa")(ca, ia))(
        tmps.at("0006_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2_1p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0005_abab_vvoo"))(tmps.at("0005_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("bb_oo")(kb, jb) *
                                             t2_1p.at("abab")(aa, bb, ia, kb))
      .allocate(tmps.at("0004_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1_1p.at("bb")(cb, ib))(
        tmps.at("0004_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0003_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1.at("bb")(cb, ib))(
        tmps.at("0003_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0002_abab_vvoo"))(tmps.at("0002_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_vv")(aa, ca) *
                                             t2_1p.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0001_abab_vvoo"))(tmps.at("0001_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("bb_vv")(bb, cb) *
                                             t2_1p.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0009_abab_vvoo"))(tmps.at("0009_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0001_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0009_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0005_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0009_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0003_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0009_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0004_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0009_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0008_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0009_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0002_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0009_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0007_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0009_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0006_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0008_abab_vvoo"))
      .deallocate(tmps.at("0007_abab_vvoo"))
      .deallocate(tmps.at("0006_abab_vvoo"))
      .deallocate(tmps.at("0005_abab_vvoo"))
      .deallocate(tmps.at("0004_abba_vvoo"))
      .deallocate(tmps.at("0003_abba_vvoo"))
      .deallocate(tmps.at("0002_abab_vvoo"))
      .deallocate(tmps.at("0001_abab_vvoo"))
      .allocate(tmps.at("0018_abab_vvoo"))(tmps.at("0018_abab_vvoo")(aa, bb, ia, jb) =
                                             tmps.at("0009_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0018_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0015_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0018_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0013_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0018_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0011_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0018_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0017_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0017_abab_vvoo"))
      .deallocate(tmps.at("0015_abab_vvoo"))
      .deallocate(tmps.at("0013_baab_vvoo"))
      .deallocate(tmps.at("0011_baab_vvoo"))
      .deallocate(tmps.at("0009_abab_vvoo"))(r2_1p.at("abab")(aa, bb, ia, jb) -=
                                             t0_1p * tmps.at("0018_abab_vvoo")(aa, bb, ia, jb))(
        r2_2p.at("abab")(aa, bb, ia, jb) -= 2.00 * tmps.at("0018_abab_vvoo")(aa, bb, ia, jb))(
        r2_2p.at("abab")(aa, bb, ia, jb) -=
        4.00 * t0_2p * tmps.at("0018_abab_vvoo")(aa, bb, ia, jb))(
        r2.at("abab")(aa, bb, ia, jb) -= tmps.at("0018_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0018_abab_vvoo"))
      .allocate(tmps.at("0027_abab_vvoo"))(tmps.at("0027_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0014_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0026_baab_vvoo"))(tmps.at("0026_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0012_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0024_abab_vvoo"))(tmps.at("0024_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_oo")(ka, ia) * t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0023_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1.at("aa")(ca, ia))(
        tmps.at("0023_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0022_abab_vvoo"))(tmps.at("0022_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("bb_oo")(kb, jb) * t2.at("abab")(aa, bb, ia, kb))
      .allocate(tmps.at("0021_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1.at("bb")(cb, ib))(
        tmps.at("0021_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0020_abab_vvoo"))(tmps.at("0020_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_vv")(aa, ca) * t2.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0019_abab_vvoo"))(tmps.at("0019_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("bb_vv")(bb, cb) * t2.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0025_abab_vvoo"))(tmps.at("0025_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0019_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0025_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0020_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0025_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0022_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0025_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0021_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0025_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0023_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0025_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0024_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0024_abab_vvoo"))
      .deallocate(tmps.at("0023_abab_vvoo"))
      .deallocate(tmps.at("0022_abab_vvoo"))
      .deallocate(tmps.at("0021_abba_vvoo"))
      .deallocate(tmps.at("0020_abab_vvoo"))
      .deallocate(tmps.at("0019_abab_vvoo"))
      .allocate(tmps.at("0028_baab_vvoo"))(tmps.at("0028_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0025_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0028_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0026_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0028_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0027_abab_vvoo")(ba, ab, ia, jb))
      .deallocate(tmps.at("0027_abab_vvoo"))
      .deallocate(tmps.at("0026_baab_vvoo"))
      .deallocate(tmps.at("0025_abab_vvoo"))(r2_1p.at("abab")(aa, bb, ia, jb) -=
                                             tmps.at("0028_baab_vvoo")(bb, aa, ia, jb))(
        r2_1p.at("abab")(aa, bb, ia, jb) -=
        2.00 * t0_2p * tmps.at("0028_baab_vvoo")(bb, aa, ia, jb))(
        r2.at("abab")(aa, bb, ia, jb) -= t0_1p * tmps.at("0028_baab_vvoo")(bb, aa, ia, jb))
      .deallocate(tmps.at("0028_baab_vvoo"))
      .allocate(tmps.at("0053_abab_vvoo"))(tmps.at("0053_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0014_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0014_baab_vooo"))
      .allocate(tmps.at("0051_baab_vooo"))(tmps.at("0051_baab_vooo")(ab, ia, ja, kb) =
                                             dp.at("aa_ov")(ia, ba) *
                                             t2_2p.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0052_abab_vvoo"))(tmps.at("0052_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0051_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0050_abab_vvoo"))(tmps.at("0050_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0016_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0048_abab_vooo"))(tmps.at("0048_abab_vooo")(aa, ib, ja, kb) =
                                             dp.at("bb_ov")(ib, bb) *
                                             t2_2p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0049_baab_vvoo"))(tmps.at("0049_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0048_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0047_baab_vvoo"))(tmps.at("0047_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0010_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0046_baab_vvoo"))(tmps.at("0046_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0012_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0012_abab_vooo"))
      .allocate(tmps.at("0038_abab_vvoo"))(tmps.at("0038_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_oo")(ka, ia) *
                                             t2_2p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0037_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1_2p.at("aa")(ca, ia))(
        tmps.at("0037_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0036_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1_1p.at("aa")(ca, ia))(
        tmps.at("0036_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2_1p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0035_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1.at("aa")(ca, ia))(
        tmps.at("0035_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2_2p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0034_abab_vvoo"))(tmps.at("0034_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("bb_oo")(kb, jb) *
                                             t2_2p.at("abab")(aa, bb, ia, kb))
      .allocate(tmps.at("0033_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1_1p.at("bb")(cb, ib))(
        tmps.at("0033_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0032_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1_2p.at("bb")(cb, ib))(
        tmps.at("0032_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0031_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1.at("bb")(cb, ib))(
        tmps.at("0031_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2_2p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0030_abab_vvoo"))(tmps.at("0030_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_vv")(aa, ca) *
                                             t2_2p.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0029_abab_vvoo"))(tmps.at("0029_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("bb_vv")(bb, cb) *
                                             t2_2p.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0045_abab_vvoo"))(tmps.at("0045_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0034_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0045_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0035_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0045_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0029_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0045_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0033_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0045_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0030_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0045_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0032_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0045_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0031_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0045_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0037_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0045_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0036_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0045_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0038_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0038_abab_vvoo"))
      .deallocate(tmps.at("0037_abab_vvoo"))
      .deallocate(tmps.at("0036_abab_vvoo"))
      .deallocate(tmps.at("0035_abab_vvoo"))
      .deallocate(tmps.at("0034_abab_vvoo"))
      .deallocate(tmps.at("0033_abba_vvoo"))
      .deallocate(tmps.at("0032_abba_vvoo"))
      .deallocate(tmps.at("0031_abba_vvoo"))
      .deallocate(tmps.at("0030_abab_vvoo"))
      .deallocate(tmps.at("0029_abab_vvoo"))
      .allocate(tmps.at("0054_abab_vvoo"))(tmps.at("0054_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0045_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0054_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0046_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0054_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0047_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0054_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0049_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0054_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0053_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0054_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0052_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0054_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0050_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0053_abab_vvoo"))
      .deallocate(tmps.at("0052_abab_vvoo"))
      .deallocate(tmps.at("0050_abab_vvoo"))
      .deallocate(tmps.at("0049_baab_vvoo"))
      .deallocate(tmps.at("0047_baab_vvoo"))
      .deallocate(tmps.at("0046_baab_vvoo"))
      .deallocate(tmps.at("0045_abab_vvoo"))(r2_1p.at("abab")(aa, bb, ia, jb) -=
                                             2.00 * tmps.at("0054_abab_vvoo")(aa, bb, ia, jb))(
        r2_2p.at("abab")(aa, bb, ia, jb) -=
        2.00 * t0_1p * tmps.at("0054_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0054_abab_vvoo"))
      .allocate(tmps.at("0175_aa_vo"))(tmps.at("0175_aa_vo")(aa, ia) =
                                         dp.at("aa_oo")(ja, ia) * t1.at("aa")(aa, ja))
      .allocate(tmps.at("0174_aa_vo"))(tmps.at("0174_aa_vo")(aa, ia) =
                                         dp.at("aa_vv")(aa, ba) * t1.at("aa")(ba, ia))
      .allocate(tmps.at("0173_aa_vo"))(tmps.at("0173_aa_vo")(aa, ia) =
                                         dp.at("aa_ov")(ja, ba) * t2.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0172_aa_vo"))(tmps.at("0172_aa_vo")(aa, ia) =
                                         dp.at("bb_ov")(jb, bb) * t2.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0176_aa_vo"))(tmps.at("0176_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0173_aa_vo")(aa, ia))(
        tmps.at("0176_aa_vo")(aa, ia) += tmps.at("0174_aa_vo")(aa, ia))(
        tmps.at("0176_aa_vo")(aa, ia) -= tmps.at("0175_aa_vo")(aa, ia))(
        tmps.at("0176_aa_vo")(aa, ia) += tmps.at("0172_aa_vo")(aa, ia))
      .deallocate(tmps.at("0175_aa_vo"))
      .deallocate(tmps.at("0174_aa_vo"))
      .deallocate(tmps.at("0173_aa_vo"))
      .deallocate(tmps.at("0172_aa_vo"))(r1_1p.at("aa")(aa, ia) += tmps.at("0176_aa_vo")(aa, ia))(
        r1_1p.at("aa")(aa, ia) += 2.00 * t0_2p * tmps.at("0176_aa_vo")(aa, ia))(
        r1.at("aa")(aa, ia) += t0_1p * tmps.at("0176_aa_vo")(aa, ia))
      .allocate(tmps.at("0042_aa_oo"))(tmps.at("0042_aa_oo")(ia, ja) =
                                         dp.at("aa_ov")(ia, aa) * t1.at("aa")(aa, ja))
      .allocate(tmps.at("0224_aa_vo"))(tmps.at("0224_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0042_aa_oo")(ja, ia))(
        r1_1p.at("aa")(aa, ia) -= tmps.at("0224_aa_vo")(aa, ia))(
        r1_1p.at("aa")(aa, ia) -= 2.00 * t0_2p * tmps.at("0224_aa_vo")(aa, ia))(
        r1.at("aa")(aa, ia) -= t0_1p * tmps.at("0224_aa_vo")(aa, ia))
      .allocate(tmps.at("0225_baba_vvoo"))(tmps.at("0225_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0224_aa_vo")(ba, ja) * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("0129_abab_oooo"))(tmps.at("0129_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abba_oovo")(ia, jb, ab, ka) *
                                             t1.at("bb")(ab, lb))
      .allocate(tmps.at("0128_abab_oooo"))(tmps.at("0128_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t2.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0130_abab_oooo"))(tmps.at("0130_abab_oooo")(ia, jb, ka, lb) =
                                             -1.00 * tmps.at("0128_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("0130_abab_oooo")(ia, jb, ka, lb) += tmps.at("0129_abab_oooo")(ia, jb, ka, lb))
      .deallocate(tmps.at("0129_abab_oooo"))
      .deallocate(tmps.at("0128_abab_oooo"))
      .allocate(tmps.at("0221_baab_vooo"))(tmps.at("0221_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0130_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0195_abab_oovo"))(tmps.at("0195_abab_oovo")(ia, jb, aa, kb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0220_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0195_abab_oovo")(ia, lb, ba, kb))(
        tmps.at("0220_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1.at("bb")(ab, lb))
      .allocate(tmps.at("0089_abab_ovvo"))(tmps.at("0089_abab_ovvo")(ia, ab, ba, jb) =
                                             t2.at("bbbb")(cb, ab, jb, kb) *
                                             eri.at("abab_oovv")(ia, kb, ba, cb))
      .allocate(tmps.at("0219_abab_ovoo"))(tmps.at("0219_abab_ovoo")(ia, ab, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0089_abab_ovvo")(ia, ab, ba, kb))
      .allocate(tmps.at("0222_abab_ovoo"))(tmps.at("0222_abab_ovoo")(ia, ab, ja, kb) =
                                             -1.00 * tmps.at("0221_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0222_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0219_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0222_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0220_abab_ovoo")(ia, ab, ja, kb))
      .deallocate(tmps.at("0221_baab_vooo"))
      .deallocate(tmps.at("0220_abab_ovoo"))
      .deallocate(tmps.at("0219_abab_ovoo"))
      .allocate(tmps.at("0223_abab_vvoo"))(tmps.at("0223_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0222_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0214_aa_ov"))(tmps.at("0214_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ja, ia, ba, aa) * t1.at("aa")(ba, ja))
      .allocate(tmps.at("0216_aa_oo"))(tmps.at("0216_aa_oo")(ia, ja) =
                                         t1.at("aa")(aa, ja) * tmps.at("0214_aa_ov")(ia, aa))
      .allocate(tmps.at("0213_abba_oovo"))(tmps.at("0213_abba_oovo")(ia, jb, ab, ka) =
                                             eri.at("abab_oovv")(ia, jb, ba, ab) *
                                             t1.at("aa")(ba, ka))
      .allocate(tmps.at("0215_aa_oo"))(tmps.at("0215_aa_oo")(ia, ja) =
                                         t1.at("bb")(ab, kb) *
                                         tmps.at("0213_abba_oovo")(ia, kb, ab, ja))
      .allocate(tmps.at("0217_aa_oo"))(tmps.at("0217_aa_oo")(ia, ja) =
                                         tmps.at("0215_aa_oo")(ia, ja))(
        tmps.at("0217_aa_oo")(ia, ja) += tmps.at("0216_aa_oo")(ia, ja))
      .deallocate(tmps.at("0216_aa_oo"))
      .deallocate(tmps.at("0215_aa_oo"))
      .allocate(tmps.at("0218_abba_vvoo"))(tmps.at("0218_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0217_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0085_baab_vovo"))(tmps.at("0085_baab_vovo")(ab, ia, ba, jb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t1.at("bb")(cb, jb))
      .allocate(tmps.at("0211_baab_vooo"))(tmps.at("0211_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0085_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("0212_abab_vvoo"))(tmps.at("0212_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0211_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0096_abab_vovo"))(tmps.at("0096_abab_vovo")(aa, ib, ba, jb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t1.at("bb")(cb, jb))
      .allocate(tmps.at("0209_abab_vooo"))(tmps.at("0209_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0096_abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("0210_baab_vvoo"))(tmps.at("0210_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0209_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("0091_abba_voov"))(tmps.at("0091_abba_voov")(aa, ib, jb, ba) =
                                             eri.at("abab_oovv")(ka, ib, ba, cb) *
                                             t2.at("abab")(aa, cb, ka, jb))
      .allocate(tmps.at("0206_abab_vooo"))(tmps.at("0206_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0091_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("0204_abab_voov"))(tmps.at("0204_abab_voov")(aa, ib, ja, bb) =
                                             eri.at("abab_oovv")(ka, ib, ca, bb) *
                                             t2.at("aaaa")(ca, aa, ja, ka))
      .allocate(tmps.at("0205_abba_vooo"))(tmps.at("0205_abba_vooo")(aa, ib, jb, ka) =
                                             t1.at("bb")(bb, jb) *
                                             tmps.at("0204_abab_voov")(aa, ib, ka, bb))
      .allocate(tmps.at("0207_abab_vooo"))(tmps.at("0207_abab_vooo")(aa, ib, ja, kb) =
                                             tmps.at("0205_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0207_abab_vooo")(aa, ib, ja, kb) += tmps.at("0206_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("0206_abab_vooo"))
      .deallocate(tmps.at("0205_abba_vooo"))
      .allocate(tmps.at("0208_baab_vvoo"))(tmps.at("0208_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0207_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("0081_aaaa_oovo"))(tmps.at("0081_aaaa_oovo")(ia, ja, aa, ka) =
                                             eri.at("aaaa_oovv")(ia, ja, ba, aa) *
                                             t1.at("aa")(ba, ka))
      .allocate(tmps.at("0082_baba_vooo"))(tmps.at("0082_baba_vooo")(ab, ia, jb, ka) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0081_aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("0203_abba_vvoo"))(tmps.at("0203_abba_vvoo")(aa, bb, ib, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0082_baba_vooo")(bb, ka, ib, ja))
      .deallocate(tmps.at("0082_baba_vooo"))
      .allocate(tmps.at("0198_bb_ov"))(tmps.at("0198_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(jb, ib, bb, ab) * t1.at("bb")(bb, jb))
      .allocate(tmps.at("0200_bb_oo"))(tmps.at("0200_bb_oo")(ib, jb) =
                                         t1.at("bb")(ab, jb) * tmps.at("0198_bb_ov")(ib, ab))
      .allocate(tmps.at("0199_bb_oo"))(tmps.at("0199_bb_oo")(ib, jb) =
                                         t1.at("aa")(aa, ka) *
                                         tmps.at("0195_abab_oovo")(ka, ib, aa, jb))
      .allocate(tmps.at("0201_bb_oo"))(tmps.at("0201_bb_oo")(ib, jb) =
                                         tmps.at("0199_bb_oo")(ib, jb))(
        tmps.at("0201_bb_oo")(ib, jb) += tmps.at("0200_bb_oo")(ib, jb))
      .deallocate(tmps.at("0200_bb_oo"))
      .deallocate(tmps.at("0199_bb_oo"))
      .allocate(tmps.at("0202_abab_vvoo"))(tmps.at("0202_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0201_bb_oo")(kb, jb))
      .allocate(tmps.at("0196_abab_oooo"))(tmps.at("0196_abab_oooo")(ia, jb, ka, lb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0195_abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("0197_abab_vvoo"))(tmps.at("0197_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0196_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0079_bbbb_oovo"))(tmps.at("0079_bbbb_oovo")(ib, jb, ab, kb) =
                                             eri.at("bbbb_oovv")(ib, jb, bb, ab) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0080_abab_vooo"))(tmps.at("0080_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, lb) *
                                             tmps.at("0079_bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("0194_baab_vvoo"))(tmps.at("0194_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0080_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .deallocate(tmps.at("0080_abab_vooo"))
      .allocate(tmps.at("0039_bb_oo"))(tmps.at("0039_bb_oo")(ib, jb) =
                                         dp.at("bb_ov")(ib, ab) * t1.at("bb")(ab, jb))
      .allocate(tmps.at("0192_bb_vo"))(tmps.at("0192_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0039_bb_oo")(jb, ib))
      .allocate(tmps.at("0193_abab_vvoo"))(tmps.at("0193_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0192_bb_vo")(bb, jb))
      .allocate(tmps.at("0056_abab_vooo"))(tmps.at("0056_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("bbbb_oovo")(ib, lb, bb, kb) *
                                             t2.at("abab")(aa, bb, ja, lb))
      .allocate(tmps.at("0190_baab_vvoo"))(tmps.at("0190_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0056_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0056_abab_vooo"))
      .allocate(tmps.at("0187_abab_vooo"))(tmps.at("0187_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_oovo")(la, ib, ba, kb) *
                                             t2.at("aaaa")(ba, aa, ja, la))
      .allocate(tmps.at("0186_abba_vooo"))(tmps.at("0186_abba_vooo")(aa, ib, jb, ka) =
                                             eri.at("abba_oovo")(la, ib, bb, ka) *
                                             t2.at("abab")(aa, bb, la, jb))
      .allocate(tmps.at("0188_abab_vooo"))(tmps.at("0188_abab_vooo")(aa, ib, ja, kb) =
                                             tmps.at("0187_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0188_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0186_abba_vooo")(aa, ib, kb, ja))
      .deallocate(tmps.at("0187_abab_vooo"))
      .deallocate(tmps.at("0186_abba_vooo"))
      .allocate(tmps.at("0189_baab_vvoo"))(tmps.at("0189_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0188_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0184_abba_vvvo"))(tmps.at("0184_abba_vvvo")(aa, bb, cb, ia) =
                                             eri.at("abab_vvvv")(aa, bb, da, cb) *
                                             t1.at("aa")(da, ia))
      .allocate(tmps.at("0185_abba_vvoo"))(tmps.at("0185_abba_vvoo")(aa, bb, ib, ja) =
                                             t1.at("bb")(cb, ib) *
                                             tmps.at("0184_abba_vvvo")(aa, bb, cb, ja))
      .allocate(tmps.at("0181_bb_oo"))(tmps.at("0181_bb_oo")(ib, jb) =
                                         eri.at("abab_oovo")(ka, ib, aa, jb) * t1.at("aa")(aa, ka))
      .allocate(tmps.at("0180_bb_oo"))(tmps.at("0180_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovo")(kb, ib, ab, jb) * t1.at("bb")(ab, kb))
      .allocate(tmps.at("0179_bb_oo"))(tmps.at("0179_bb_oo")(ib, jb) =
                                         eri.at("abab_oovv")(ka, ib, aa, bb) *
                                         t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0178_bb_oo"))(tmps.at("0178_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovv")(kb, ib, ab, bb) *
                                         t2.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0182_bb_oo"))(tmps.at("0182_bb_oo")(ib, jb) =
                                         -0.50 * tmps.at("0178_bb_oo")(ib, jb))(
        tmps.at("0182_bb_oo")(ib, jb) += tmps.at("0179_bb_oo")(ib, jb))(
        tmps.at("0182_bb_oo")(ib, jb) += tmps.at("0180_bb_oo")(ib, jb))(
        tmps.at("0182_bb_oo")(ib, jb) += tmps.at("0181_bb_oo")(ib, jb))
      .deallocate(tmps.at("0181_bb_oo"))
      .deallocate(tmps.at("0180_bb_oo"))
      .deallocate(tmps.at("0179_bb_oo"))
      .deallocate(tmps.at("0178_bb_oo"))
      .allocate(tmps.at("0183_abab_vvoo"))(tmps.at("0183_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0182_bb_oo")(kb, jb))
      .allocate(tmps.at("0177_baba_vvoo"))(tmps.at("0177_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("0176_aa_vo")(ba, ja))
      .allocate(tmps.at("0055_bb_vv"))(tmps.at("0055_bb_vv")(ab, bb) =
                                         eri.at("bbbb_oovv")(ib, jb, bb, cb) *
                                         t2.at("bbbb")(cb, ab, jb, ib))
      .allocate(tmps.at("0171_abab_vvoo"))(tmps.at("0171_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0055_bb_vv")(bb, cb))
      .deallocate(tmps.at("0055_bb_vv"))
      .allocate(tmps.at("0168_baab_vooo"))(tmps.at("0168_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baab_vovo")(ab, ia, ba, kb) *
                                             t1.at("aa")(ba, ja))
      .allocate(tmps.at("0167_baab_vooo"))(tmps.at("0167_baab_vooo")(ab, ia, ja, kb) =
                                             f.at("aa_ov")(ia, ba) * t2.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0166_baab_vooo"))(tmps.at("0166_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baba_vovo")(ab, ia, bb, ja) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0165_baab_vooo"))(tmps.at("0165_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t2.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0169_baab_vooo"))(tmps.at("0169_baab_vooo")(ab, ia, ja, kb) =
                                             -1.00 * tmps.at("0165_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0169_baab_vooo")(ab, ia, ja, kb) += tmps.at("0167_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0169_baab_vooo")(ab, ia, ja, kb) += tmps.at("0166_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0169_baab_vooo")(ab, ia, ja, kb) -= tmps.at("0168_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("0168_baab_vooo"))
      .deallocate(tmps.at("0167_baab_vooo"))
      .deallocate(tmps.at("0166_baab_vooo"))
      .deallocate(tmps.at("0165_baab_vooo"))
      .allocate(tmps.at("0170_abab_vvoo"))(tmps.at("0170_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0169_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0163_aaaa_voov"))(tmps.at("0163_aaaa_voov")(aa, ia, ja, ba) =
                                             eri.at("aaaa_oovv")(ka, ia, ca, ba) *
                                             t2.at("aaaa")(ca, aa, ja, ka))
      .allocate(tmps.at("0164_baba_vvoo"))(tmps.at("0164_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0163_aaaa_voov")(ba, ka, ja, ca))
      .allocate(tmps.at("0161_aa_vv"))(tmps.at("0161_aa_vv")(aa, ba) =
                                         eri.at("abab_vovv")(aa, ib, ba, cb) * t1.at("bb")(cb, ib))
      .allocate(tmps.at("0162_baab_vvoo"))(tmps.at("0162_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0161_aa_vv")(ba, ca))
      .allocate(tmps.at("0158_baab_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("abab_oovv")(la, ib, ca, bb) * t1.at("aa")(ca, la))(
        tmps.at("0158_baab_ovoo")(ib, aa, ja, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0157_baab_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("bbbb_oovv")(lb, ib, cb, bb) * t1.at("bb")(cb, lb))(
        tmps.at("0157_baab_ovoo")(ib, aa, ja, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0159_baab_ovoo"))(tmps.at("0159_baab_ovoo")(ib, aa, ja, kb) =
                                             tmps.at("0157_baab_ovoo")(ib, aa, ja, kb))(
        tmps.at("0159_baab_ovoo")(ib, aa, ja, kb) += tmps.at("0158_baab_ovoo")(ib, aa, ja, kb))
      .deallocate(tmps.at("0158_baab_ovoo"))
      .deallocate(tmps.at("0157_baab_ovoo"))
      .allocate(tmps.at("0160_baab_vvoo"))(tmps.at("0160_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0159_baab_ovoo")(kb, ba, ia, jb))
      .allocate(tmps.at("0154_baab_vooo"))(tmps.at("0154_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("abab_oooo")(ia, lb, ja, kb) *
                                             t1.at("bb")(ab, lb))
      .allocate(tmps.at("0153_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             eri.at("abab_oovo")(ia, lb, ba, kb) *
                                             t1.at("aa")(ba, ja))(
        tmps.at("0153_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1.at("bb")(ab, lb))
      .allocate(tmps.at("0152_baab_vooo"))(tmps.at("0152_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("abab_oovo")(ia, lb, ba, kb) *
                                             t2.at("abab")(ba, ab, ja, lb))
      .allocate(tmps.at("0151_abba_ovoo"))(tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) =
                                             eri.at("abab_oovv")(ia, lb, ba, cb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("0151_abba_ovoo")(ia, ab, jb, ka) =
          tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) * t2.at("abab")(ba, ab, ka, lb))
      .allocate(tmps.at("0150_baba_vooo"))(tmps.at("0150_baba_vooo")(ab, ia, jb, ka) =
                                             eri.at("abba_oovo")(ia, lb, bb, ka) *
                                             t2.at("bbbb")(bb, ab, jb, lb))
      .allocate(tmps.at("0155_baab_vooo"))(tmps.at("0155_baab_vooo")(ab, ia, ja, kb) =
                                             tmps.at("0152_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0155_baab_vooo")(ab, ia, ja, kb) += tmps.at("0151_abba_ovoo")(ia, ab, kb, ja))(
        tmps.at("0155_baab_vooo")(ab, ia, ja, kb) += tmps.at("0154_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0155_baab_vooo")(ab, ia, ja, kb) += tmps.at("0153_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0155_baab_vooo")(ab, ia, ja, kb) -= tmps.at("0150_baba_vooo")(ab, ia, kb, ja))
      .deallocate(tmps.at("0154_baab_vooo"))
      .deallocate(tmps.at("0153_abab_ovoo"))
      .deallocate(tmps.at("0152_baab_vooo"))
      .deallocate(tmps.at("0151_abba_ovoo"))
      .deallocate(tmps.at("0150_baba_vooo"))
      .allocate(tmps.at("0156_abab_vvoo"))(tmps.at("0156_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0155_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0147_abab_vooo"))(tmps.at("0147_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_vovo")(aa, ib, ba, kb) *
                                             t1.at("aa")(ba, ja))
      .allocate(tmps.at("0146_abab_vooo"))(tmps.at("0146_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abba_vovo")(aa, ib, bb, ja) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0145_abab_vooo"))(tmps.at("0145_abab_vooo")(aa, ib, ja, kb) =
                                             f.at("bb_ov")(ib, bb) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0144_abab_vooo"))(tmps.at("0144_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t2.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0148_abab_vooo"))(tmps.at("0148_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("0146_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0148_abab_vooo")(aa, ib, ja, kb) += tmps.at("0144_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0148_abab_vooo")(aa, ib, ja, kb) += tmps.at("0147_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0148_abab_vooo")(aa, ib, ja, kb) += tmps.at("0145_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("0147_abab_vooo"))
      .deallocate(tmps.at("0146_abab_vooo"))
      .deallocate(tmps.at("0145_abab_vooo"))
      .deallocate(tmps.at("0144_abab_vooo"))
      .allocate(tmps.at("0149_baab_vvoo"))(tmps.at("0149_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0148_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0141_bb_vo"))(tmps.at("0141_bb_vo")(ab, ib) =
                                         dp.at("bb_oo")(jb, ib) * t1.at("bb")(ab, jb))
      .allocate(tmps.at("0140_bb_vo"))(tmps.at("0140_bb_vo")(ab, ib) =
                                         dp.at("bb_vv")(ab, bb) * t1.at("bb")(bb, ib))
      .allocate(tmps.at("0139_bb_vo"))(tmps.at("0139_bb_vo")(ab, ib) =
                                         dp.at("aa_ov")(ja, ba) * t2.at("abab")(ba, ab, ja, ib))
      .allocate(tmps.at("0138_bb_vo"))(tmps.at("0138_bb_vo")(ab, ib) =
                                         dp.at("bb_ov")(jb, bb) * t2.at("bbbb")(bb, ab, ib, jb))
      .allocate(tmps.at("0142_bb_vo"))(tmps.at("0142_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("0138_bb_vo")(ab, ib))(
        tmps.at("0142_bb_vo")(ab, ib) -= tmps.at("0141_bb_vo")(ab, ib))(
        tmps.at("0142_bb_vo")(ab, ib) += tmps.at("0139_bb_vo")(ab, ib))(
        tmps.at("0142_bb_vo")(ab, ib) += tmps.at("0140_bb_vo")(ab, ib))
      .deallocate(tmps.at("0141_bb_vo"))
      .deallocate(tmps.at("0140_bb_vo"))
      .deallocate(tmps.at("0139_bb_vo"))
      .deallocate(tmps.at("0138_bb_vo"))
      .allocate(tmps.at("0143_abab_vvoo"))(tmps.at("0143_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0142_bb_vo")(bb, jb))
      .allocate(tmps.at("0135_abab_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("aaaa_oovv")(la, ia, ca, ba) * t1.at("aa")(ca, la))(
        tmps.at("0135_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aa_vo")(ba, ia) * t2.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0134_abab_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("abab_oovv")(ia, lb, ba, cb) * t1.at("bb")(cb, lb))(
        tmps.at("0134_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aa_vo")(ba, ia) * t2.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0136_abab_ovoo"))(tmps.at("0136_abab_ovoo")(ia, ab, ja, kb) =
                                             tmps.at("0134_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0136_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0135_abab_ovoo")(ia, ab, ja, kb))
      .deallocate(tmps.at("0135_abab_ovoo"))
      .deallocate(tmps.at("0134_abab_ovoo"))
      .allocate(tmps.at("0137_abab_vvoo"))(tmps.at("0137_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0136_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0132_aaaa_voov"))(tmps.at("0132_aaaa_voov")(aa, ia, ja, ba) =
                                             eri.at("abab_oovv")(ia, kb, ba, cb) *
                                             t2.at("abab")(aa, cb, ja, kb))
      .allocate(tmps.at("0133_baba_vvoo"))(tmps.at("0133_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0132_aaaa_voov")(ba, ka, ja, ca))
      .allocate(tmps.at("0131_abab_vvoo"))(tmps.at("0131_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0130_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0126_baba_vovo"))(tmps.at("0126_baba_vovo")(ab, ia, bb, ja) =
                                             eri.at("baab_vovv")(ab, ia, ca, bb) *
                                             t1.at("aa")(ca, ja))
      .allocate(tmps.at("0127_abba_vvoo"))(tmps.at("0127_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, cb, ka, ib) *
                                             tmps.at("0126_baba_vovo")(bb, ka, cb, ja))
      .allocate(tmps.at("0123_aa_oo"))(tmps.at("0123_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovo")(ka, ia, aa, ja) * t1.at("aa")(aa, ka))
      .allocate(tmps.at("0122_aa_oo"))(tmps.at("0122_aa_oo")(ia, ja) =
                                         eri.at("abba_oovo")(ia, kb, ab, ja) * t1.at("bb")(ab, kb))
      .allocate(tmps.at("0124_aa_oo"))(tmps.at("0124_aa_oo")(ia, ja) =
                                         -1.00 * tmps.at("0123_aa_oo")(ia, ja))(
        tmps.at("0124_aa_oo")(ia, ja) += tmps.at("0122_aa_oo")(ia, ja))
      .deallocate(tmps.at("0123_aa_oo"))
      .deallocate(tmps.at("0122_aa_oo"))
      .allocate(tmps.at("0125_abba_vvoo"))(tmps.at("0125_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0124_aa_oo")(ka, ja))
      .allocate(tmps.at("0119_bb_vv"))(tmps.at("0119_bb_vv")(ab, bb) =
                                         eri.at("baab_vovv")(ab, ia, ca, bb) * t1.at("aa")(ca, ia))
      .allocate(tmps.at("0118_bb_vv"))(tmps.at("0118_bb_vv")(ab, bb) =
                                         eri.at("bbbb_vovv")(ab, ib, cb, bb) * t1.at("bb")(cb, ib))
      .allocate(tmps.at("0120_bb_vv"))(tmps.at("0120_bb_vv")(ab, bb) =
                                         tmps.at("0118_bb_vv")(ab, bb))(
        tmps.at("0120_bb_vv")(ab, bb) += tmps.at("0119_bb_vv")(ab, bb))
      .deallocate(tmps.at("0119_bb_vv"))
      .deallocate(tmps.at("0118_bb_vv"))
      .allocate(tmps.at("0121_abab_vvoo"))(tmps.at("0121_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0120_bb_vv")(bb, cb))
      .allocate(tmps.at("0116_aa_oo"))(tmps.at("0116_aa_oo")(ia, ja) =
                                         eri.at("abab_oovv")(ia, kb, aa, bb) *
                                         t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0117_abba_vvoo"))(tmps.at("0117_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0116_aa_oo")(ka, ja))
      .allocate(tmps.at("0114_bbbb_vovo"))(tmps.at("0114_bbbb_vovo")(ab, ib, bb, jb) =
                                             eri.at("bbbb_vovv")(ab, ib, cb, bb) *
                                             t1.at("bb")(cb, jb))
      .allocate(tmps.at("0115_abab_vvoo"))(tmps.at("0115_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0114_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("0112_abab_voov"))(tmps.at("0112_abab_voov")(aa, ib, ja, bb) =
                                             eri.at("bbbb_oovv")(kb, ib, cb, bb) *
                                             t2.at("abab")(aa, cb, ja, kb))
      .allocate(tmps.at("0113_baba_vvoo"))(tmps.at("0113_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0112_abab_voov")(ba, kb, ja, cb))
      .allocate(tmps.at("0110_aa_vv"))(tmps.at("0110_aa_vv")(aa, ba) =
                                         eri.at("abab_oovv")(ia, jb, ba, cb) *
                                         t2.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0111_baab_vvoo"))(tmps.at("0111_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0110_aa_vv")(ba, ca))
      .allocate(tmps.at("0108_abab_oooo"))(tmps.at("0108_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abab_oovo")(ia, jb, aa, lb) *
                                             t1.at("aa")(aa, ka))
      .allocate(tmps.at("0109_abab_vvoo"))(tmps.at("0109_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0108_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0106_abba_vovo"))(tmps.at("0106_abba_vovo")(aa, ib, bb, ja) =
                                             eri.at("abab_vovv")(aa, ib, ca, bb) *
                                             t1.at("aa")(ca, ja))
      .allocate(tmps.at("0107_baba_vvoo"))(tmps.at("0107_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0106_abba_vovo")(ba, kb, cb, ja))
      .allocate(tmps.at("0104_aa_vv"))(tmps.at("0104_aa_vv")(aa, ba) =
                                         eri.at("aaaa_vovv")(aa, ia, ca, ba) * t1.at("aa")(ca, ia))
      .allocate(tmps.at("0105_baab_vvoo"))(tmps.at("0105_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0104_aa_vv")(ba, ca))
      .allocate(tmps.at("0102_aa_oo"))(tmps.at("0102_aa_oo")(ia, ja) =
                                         f.at("aa_ov")(ia, aa) * t1.at("aa")(aa, ja))
      .allocate(tmps.at("0103_abba_vvoo"))(tmps.at("0103_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0102_aa_oo")(ka, ja))
      .allocate(tmps.at("0100_aa_vv"))(tmps.at("0100_aa_vv")(aa, ba) =
                                         eri.at("aaaa_oovv")(ia, ja, ca, ba) *
                                         t2.at("aaaa")(ca, aa, ja, ia))
      .allocate(tmps.at("0101_baab_vvoo"))(tmps.at("0101_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0100_aa_vv")(ba, ca))
      .allocate(tmps.at("0098_aaaa_vovo"))(tmps.at("0098_aaaa_vovo")(aa, ia, ba, ja) =
                                             eri.at("aaaa_vovv")(aa, ia, ca, ba) *
                                             t1.at("aa")(ca, ja))
      .allocate(tmps.at("0099_baba_vvoo"))(tmps.at("0099_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0098_aaaa_vovo")(ba, ka, ca, ja))
      .allocate(tmps.at("0097_baab_vvoo"))(tmps.at("0097_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0096_abab_vovo")(ba, kb, ca, jb))
      .allocate(tmps.at("0094_bb_vv"))(tmps.at("0094_bb_vv")(ab, bb) =
                                         eri.at("abab_oovv")(ia, jb, ca, bb) *
                                         t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0095_abab_vvoo"))(tmps.at("0095_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0094_bb_vv")(bb, cb))
      .allocate(tmps.at("0057_baba_vooo"))(tmps.at("0057_baba_vooo")(ab, ia, jb, ka) =
                                             eri.at("aaaa_oovo")(ia, la, ba, ka) *
                                             t2.at("abab")(ba, ab, la, jb))
      .allocate(tmps.at("0093_abba_vvoo"))(tmps.at("0093_abba_vvoo")(aa, bb, ib, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0057_baba_vooo")(bb, ka, ib, ja))
      .deallocate(tmps.at("0057_baba_vooo"))
      .allocate(tmps.at("0092_baab_vvoo"))(tmps.at("0092_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0091_abba_voov")(ba, kb, jb, ca))
      .allocate(tmps.at("0090_abab_vvoo"))(tmps.at("0090_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0089_abab_ovvo")(ka, bb, ca, jb))
      .allocate(tmps.at("0087_aa_oo"))(tmps.at("0087_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovv")(ka, ia, aa, ba) *
                                         t2.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0088_abba_vvoo"))(tmps.at("0088_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0087_aa_oo")(ka, ja))
      .allocate(tmps.at("0086_abab_vvoo"))(tmps.at("0086_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0085_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("0083_bb_oo"))(tmps.at("0083_bb_oo")(ib, jb) =
                                         f.at("bb_ov")(ib, ab) * t1.at("bb")(ab, jb))
      .allocate(tmps.at("0084_abab_vvoo"))(tmps.at("0084_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0083_bb_oo")(kb, jb))
      .allocate(tmps.at("0077_abab_vvoo"))(tmps.at("0077_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_oooo")(ka, lb, ia, jb) *
                                             t2.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0076_abab_vvoo"))(tmps.at("0076_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vooo")(aa, kb, ia, jb) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0075_abab_vvoo"))(tmps.at("0075_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vovo")(aa, kb, cb, ia) *
                                             t2.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("0074_abab_vvoo"))(tmps.at("0074_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_vo")(aa, ia) * t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0073_abab_vvoo"))(tmps.at("0073_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("bb_vv")(bb, cb) * t2.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0072_abab_vvoo"))(tmps.at("0072_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vvvo")(aa, bb, cb, ia) *
                                             t1.at("bb")(cb, jb))
      .allocate(tmps.at("0071_abab_vvoo"))(tmps.at("0071_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("bb_vo")(bb, jb) * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0070_abab_vvoo"))(tmps.at("0070_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0015")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0069_abab_vvoo"))(tmps.at("0069_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0013")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0068_abab_vvoo"))(tmps.at("0068_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("aaaa_vovo")(aa, ka, ca, ia) *
                                             t2.at("abab")(ca, bb, ka, jb))
      .allocate(tmps.at("0067_abab_vvoo"))(tmps.at("0067_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_oo")(ka, ia) * t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0066_abab_vvoo"))(tmps.at("0066_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("bbbb_vovo")(bb, kb, cb, jb) *
                                             t2.at("abab")(aa, cb, ia, kb))
      .allocate(tmps.at("0065_abab_vvoo"))(tmps.at("0065_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vvvv")(aa, bb, ca, db) *
                                             t2.at("abab")(ca, db, ia, jb))
      .allocate(tmps.at("0064_abba_vvoo"))(tmps.at("0064_abba_vvoo")(aa, bb, ib, ja) =
                                             eri.at("abab_vovo")(aa, kb, ca, ib) *
                                             t2.at("abab")(ca, bb, ja, kb))
      .allocate(tmps.at("0063_abab_vvoo"))(tmps.at("0063_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("baab_vovo")(bb, ka, ca, jb) *
                                             t2.at("aaaa")(ca, aa, ia, ka))
      .allocate(tmps.at("0062_abba_vvoo"))(tmps.at("0062_abba_vvoo")(aa, bb, ib, ja) =
                                             eri.at("baba_vovo")(bb, ka, cb, ja) *
                                             t2.at("abab")(aa, cb, ka, ib))
      .allocate(tmps.at("0061_abab_vvoo"))(tmps.at("0061_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("baab_vooo")(bb, ka, ia, jb) *
                                             t1.at("aa")(aa, ka))
      .allocate(tmps.at("0060_abab_vvoo"))(tmps.at("0060_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_vv")(aa, ca) * t2.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0059_abab_vvoo"))(tmps.at("0059_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("bb_oo")(kb, jb) * t2.at("abab")(aa, bb, ia, kb))
      .allocate(tmps.at("0058_abab_vvoo"))(tmps.at("0058_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vvvo")(aa, bb, ca, jb) *
                                             t1.at("aa")(ca, ia))
      .allocate(tmps.at("0078_abab_vvoo"))(tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0059_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0068_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0072_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0069_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0074_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0076_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0070_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0073_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0071_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0062_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0060_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0067_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0066_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0077_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0061_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0064_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0075_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0065_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0058_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0078_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0063_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0077_abab_vvoo"))
      .deallocate(tmps.at("0076_abab_vvoo"))
      .deallocate(tmps.at("0075_abab_vvoo"))
      .deallocate(tmps.at("0074_abab_vvoo"))
      .deallocate(tmps.at("0073_abab_vvoo"))
      .deallocate(tmps.at("0072_abab_vvoo"))
      .deallocate(tmps.at("0071_abab_vvoo"))
      .deallocate(tmps.at("0070_abab_vvoo"))
      .deallocate(tmps.at("0069_abab_vvoo"))
      .deallocate(tmps.at("0068_abab_vvoo"))
      .deallocate(tmps.at("0067_abab_vvoo"))
      .deallocate(tmps.at("0066_abab_vvoo"))
      .deallocate(tmps.at("0065_abab_vvoo"))
      .deallocate(tmps.at("0064_abba_vvoo"))
      .deallocate(tmps.at("0063_abab_vvoo"))
      .deallocate(tmps.at("0062_abba_vvoo"))
      .deallocate(tmps.at("0061_abab_vvoo"))
      .deallocate(tmps.at("0060_abab_vvoo"))
      .deallocate(tmps.at("0059_abab_vvoo"))
      .deallocate(tmps.at("0058_abab_vvoo"))
      .allocate(tmps.at("0191_baab_vvoo"))(tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) =
                                             -1.00 * tmps.at("0171_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0084_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0095_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0101_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0097_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0103_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0113_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0107_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0078_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0143_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0160_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0090_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0121_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0177_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0131_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0088_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0105_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0093_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0183_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0086_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0189_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0092_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0133_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0170_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0149_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0125_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0162_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0190_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0117_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0099_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0109_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0164_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0115_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0127_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0137_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0111_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 *
        tmps.at("0185_abba_vvoo")(ba, ab, jb, ia))(tmps.at("0191_baab_vvoo")(ab, ba, ia, jb) +=
                                                   2.00 * tmps.at("0156_abab_vvoo")(ba, ab, ia, jb))
      .deallocate(tmps.at("0190_baab_vvoo"))
      .deallocate(tmps.at("0189_baab_vvoo"))
      .deallocate(tmps.at("0185_abba_vvoo"))
      .deallocate(tmps.at("0183_abab_vvoo"))
      .deallocate(tmps.at("0177_baba_vvoo"))
      .deallocate(tmps.at("0171_abab_vvoo"))
      .deallocate(tmps.at("0170_abab_vvoo"))
      .deallocate(tmps.at("0164_baba_vvoo"))
      .deallocate(tmps.at("0162_baab_vvoo"))
      .deallocate(tmps.at("0160_baab_vvoo"))
      .deallocate(tmps.at("0156_abab_vvoo"))
      .deallocate(tmps.at("0149_baab_vvoo"))
      .deallocate(tmps.at("0143_abab_vvoo"))
      .deallocate(tmps.at("0137_abab_vvoo"))
      .deallocate(tmps.at("0133_baba_vvoo"))
      .deallocate(tmps.at("0131_abab_vvoo"))
      .deallocate(tmps.at("0127_abba_vvoo"))
      .deallocate(tmps.at("0125_abba_vvoo"))
      .deallocate(tmps.at("0121_abab_vvoo"))
      .deallocate(tmps.at("0117_abba_vvoo"))
      .deallocate(tmps.at("0115_abab_vvoo"))
      .deallocate(tmps.at("0113_baba_vvoo"))
      .deallocate(tmps.at("0111_baab_vvoo"))
      .deallocate(tmps.at("0109_abab_vvoo"))
      .deallocate(tmps.at("0107_baba_vvoo"))
      .deallocate(tmps.at("0105_baab_vvoo"))
      .deallocate(tmps.at("0103_abba_vvoo"))
      .deallocate(tmps.at("0101_baab_vvoo"))
      .deallocate(tmps.at("0099_baba_vvoo"))
      .deallocate(tmps.at("0097_baab_vvoo"))
      .deallocate(tmps.at("0095_abab_vvoo"))
      .deallocate(tmps.at("0093_abba_vvoo"))
      .deallocate(tmps.at("0092_baab_vvoo"))
      .deallocate(tmps.at("0090_abab_vvoo"))
      .deallocate(tmps.at("0088_abba_vvoo"))
      .deallocate(tmps.at("0086_abab_vvoo"))
      .deallocate(tmps.at("0084_abab_vvoo"))
      .deallocate(tmps.at("0078_abab_vvoo"))
      .allocate(tmps.at("0226_baab_vvoo"))(tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) =
                                             -2.00 * tmps.at("0193_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0202_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0194_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0197_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0212_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0218_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0223_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0191_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0208_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0203_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 *
        tmps.at("0210_baab_vvoo")(ab, ba, ia, jb))(tmps.at("0226_baab_vvoo")(ab, ba, ia, jb) -=
                                                   2.00 * tmps.at("0225_baba_vvoo")(ab, ba, jb, ia))
      .deallocate(tmps.at("0225_baba_vvoo"))
      .deallocate(tmps.at("0223_abab_vvoo"))
      .deallocate(tmps.at("0218_abba_vvoo"))
      .deallocate(tmps.at("0212_abab_vvoo"))
      .deallocate(tmps.at("0210_baab_vvoo"))
      .deallocate(tmps.at("0208_baab_vvoo"))
      .deallocate(tmps.at("0203_abba_vvoo"))
      .deallocate(tmps.at("0202_abab_vvoo"))
      .deallocate(tmps.at("0197_abab_vvoo"))
      .deallocate(tmps.at("0194_baab_vvoo"))
      .deallocate(tmps.at("0193_abab_vvoo"))
      .deallocate(tmps.at("0191_baab_vvoo"))(r2.at("abab")(aa, bb, ia, jb) +=
                                             0.50 * tmps.at("0226_baab_vvoo")(bb, aa, ia, jb))
      .deallocate(tmps.at("0226_baab_vvoo"))
      .allocate(tmps.at("0315_aa_ov"))(tmps.at("0315_aa_ov")(ia, aa) =
                                         eri.at("abab_oovv")(ia, jb, aa, bb) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0319_aa_oo"))(tmps.at("0319_aa_oo")(ia, ja) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0315_aa_ov")(ia, aa))
      .allocate(tmps.at("0318_aa_oo"))(tmps.at("0318_aa_oo")(ia, ja) =
                                         t1_2p.at("aa")(aa, ka) *
                                         tmps.at("0081_aaaa_oovo")(ia, ka, aa, ja))
      .allocate(tmps.at("0310_aaaa_oovo"))(tmps.at("0310_aaaa_oovo")(ia, ja, aa, ka) =
                                             eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                             t1_1p.at("aa")(ba, ka))
      .allocate(tmps.at("0317_aa_oo"))(tmps.at("0317_aa_oo")(ia, ja) =
                                         t1_1p.at("aa")(aa, ka) *
                                         tmps.at("0310_aaaa_oovo")(ia, ka, aa, ja))
      .allocate(tmps.at("0316_aa_oo"))(tmps.at("0316_aa_oo")(ia, ja) =
                                         t1_2p.at("bb")(ab, kb) *
                                         tmps.at("0213_abba_oovo")(ia, kb, ab, ja))
      .allocate(tmps.at("0320_aa_oo"))(tmps.at("0320_aa_oo")(ia, ja) =
                                         -1.00 * tmps.at("0317_aa_oo")(ia, ja))(
        tmps.at("0320_aa_oo")(ia, ja) += tmps.at("0318_aa_oo")(ia, ja))(
        tmps.at("0320_aa_oo")(ia, ja) += tmps.at("0319_aa_oo")(ia, ja))(
        tmps.at("0320_aa_oo")(ia, ja) += tmps.at("0316_aa_oo")(ia, ja))
      .deallocate(tmps.at("0319_aa_oo"))
      .deallocate(tmps.at("0318_aa_oo"))
      .deallocate(tmps.at("0317_aa_oo"))
      .deallocate(tmps.at("0316_aa_oo"))
      .allocate(tmps.at("0321_aa_vo"))(tmps.at("0321_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0320_aa_oo")(ja, ia))
      .allocate(tmps.at("0284_aa_ov"))(tmps.at("0284_aa_ov")(ia, aa) =
                                         eri.at("abab_oovv")(ia, jb, aa, bb) * t1.at("bb")(bb, jb))
      .allocate(tmps.at("0312_aa_oo"))(tmps.at("0312_aa_oo")(ia, ja) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0284_aa_ov")(ia, aa))
      .allocate(tmps.at("0311_aa_oo"))(tmps.at("0311_aa_oo")(ia, ja) =
                                         t1.at("aa")(aa, ka) *
                                         tmps.at("0310_aaaa_oovo")(ka, ia, aa, ja))
      .allocate(tmps.at("0313_aa_oo"))(tmps.at("0313_aa_oo")(ia, ja) =
                                         tmps.at("0311_aa_oo")(ia, ja))(
        tmps.at("0313_aa_oo")(ia, ja) += tmps.at("0312_aa_oo")(ia, ja))
      .deallocate(tmps.at("0312_aa_oo"))
      .deallocate(tmps.at("0311_aa_oo"))
      .allocate(tmps.at("0314_aa_vo"))(tmps.at("0314_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0313_aa_oo")(ja, ia))
      .allocate(tmps.at("0307_aa_oo"))(tmps.at("0307_aa_oo")(ia, ja) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0284_aa_ov")(ia, aa))
      .allocate(tmps.at("0306_aa_oo"))(tmps.at("0306_aa_oo")(ia, ja) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0214_aa_ov")(ia, aa))
      .allocate(tmps.at("0308_aa_oo"))(tmps.at("0308_aa_oo")(ia, ja) =
                                         tmps.at("0306_aa_oo")(ia, ja))(
        tmps.at("0308_aa_oo")(ia, ja) += tmps.at("0307_aa_oo")(ia, ja))
      .deallocate(tmps.at("0307_aa_oo"))
      .deallocate(tmps.at("0306_aa_oo"))
      .allocate(tmps.at("0309_aa_vo"))(tmps.at("0309_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0308_aa_oo")(ja, ia))
      .allocate(tmps.at("0305_aa_vo"))(tmps.at("0305_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0217_aa_oo")(ja, ia))
      .allocate(tmps.at("0303_aa_oo"))(tmps.at("0303_aa_oo")(ia, ja) =
                                         t1_1p.at("aa")(aa, ka) *
                                         tmps.at("0081_aaaa_oovo")(ka, ia, aa, ja))
      .allocate(tmps.at("0304_aa_vo"))(tmps.at("0304_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0303_aa_oo")(ja, ia))
      .allocate(tmps.at("0301_aa_oo"))(tmps.at("0301_aa_oo")(ia, ja) =
                                         t1_1p.at("bb")(ab, kb) *
                                         tmps.at("0213_abba_oovo")(ia, kb, ab, ja))
      .allocate(tmps.at("0302_aa_vo"))(tmps.at("0302_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0301_aa_oo")(ja, ia))
      .allocate(tmps.at("0299_aa_vo"))(tmps.at("0299_aa_vo")(aa, ia) =
                                         t1_2p.at("bb")(bb, jb) *
                                         tmps.at("0204_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0298_aa_vo"))(tmps.at("0298_aa_vo")(aa, ia) =
                                         t1_2p.at("bb")(bb, jb) *
                                         tmps.at("0106_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0295_aa_oo"))(tmps.at("0295_aa_oo")(ia, ja) =
                                         f.at("aa_ov")(ia, aa) * t1_2p.at("aa")(aa, ja))
      .allocate(tmps.at("0294_aa_oo"))(tmps.at("0294_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovo")(ia, ka, aa, ja) *
                                         t1_2p.at("aa")(aa, ka))
      .allocate(tmps.at("0293_aa_oo"))(tmps.at("0293_aa_oo")(ia, ja) =
                                         eri.at("abba_oovo")(ia, kb, ab, ja) *
                                         t1_2p.at("bb")(ab, kb))
      .allocate(tmps.at("0292_aa_oo"))(tmps.at("0292_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovv")(ia, ka, aa, ba) *
                                         t2_2p.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0291_aa_oo"))(tmps.at("0291_aa_oo")(ia, ja) =
                                         eri.at("abab_oovv")(ia, kb, aa, bb) *
                                         t2_2p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0296_aa_oo"))(tmps.at("0296_aa_oo")(ia, ja) =
                                         -1.00 * tmps.at("0293_aa_oo")(ia, ja))(
        tmps.at("0296_aa_oo")(ia, ja) -= tmps.at("0294_aa_oo")(ia, ja))(
        tmps.at("0296_aa_oo")(ia, ja) += 0.50 * tmps.at("0292_aa_oo")(ia, ja))(
        tmps.at("0296_aa_oo")(ia, ja) += tmps.at("0291_aa_oo")(ia, ja))(
        tmps.at("0296_aa_oo")(ia, ja) += tmps.at("0295_aa_oo")(ia, ja))
      .deallocate(tmps.at("0295_aa_oo"))
      .deallocate(tmps.at("0294_aa_oo"))
      .deallocate(tmps.at("0293_aa_oo"))
      .deallocate(tmps.at("0292_aa_oo"))
      .deallocate(tmps.at("0291_aa_oo"))
      .allocate(tmps.at("0297_aa_vo"))(tmps.at("0297_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0296_aa_oo")(ja, ia))
      .allocate(tmps.at("0290_aa_vo"))(tmps.at("0290_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ja) *
                                         tmps.at("0163_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0289_aa_vo"))(tmps.at("0289_aa_vo")(aa, ia) =
                                         t2_2p.at("aaaa")(ba, aa, ja, ka) *
                                         tmps.at("0081_aaaa_oovo")(ka, ja, ba, ia))
      .allocate(tmps.at("0287_abab_voov"))(tmps.at("0287_abab_voov")(aa, ib, ja, bb) =
                                             eri.at("bbbb_oovv")(ib, kb, bb, cb) *
                                             t2_2p.at("abab")(aa, cb, ja, kb))
      .allocate(tmps.at("0288_aa_vo"))(tmps.at("0288_aa_vo")(aa, ia) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0287_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0286_aa_vo"))(tmps.at("0286_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0124_aa_oo")(ja, ia))
      .allocate(tmps.at("0285_aa_vo"))(tmps.at("0285_aa_vo")(aa, ia) =
                                         t2_2p.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0284_aa_ov")(ja, ba))
      .allocate(tmps.at("0282_aa_oo"))(tmps.at("0282_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovo")(ka, ia, aa, ja) *
                                         t1_1p.at("aa")(aa, ka))
      .allocate(tmps.at("0283_aa_vo"))(tmps.at("0283_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0282_aa_oo")(ja, ia))
      .allocate(tmps.at("0280_abab_voov"))(tmps.at("0280_abab_voov")(aa, ib, ja, bb) =
                                             eri.at("bbbb_oovv")(ib, kb, bb, cb) *
                                             t2_1p.at("abab")(aa, cb, ja, kb))
      .allocate(tmps.at("0281_aa_vo"))(tmps.at("0281_aa_vo")(aa, ia) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0280_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0279_aa_vo"))(tmps.at("0279_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ja) *
                                         tmps.at("0098_aaaa_vovo")(aa, ja, ba, ia))
      .allocate(tmps.at("0277_aa_oo"))(tmps.at("0277_aa_oo")(ia, ja) =
                                         eri.at("abba_oovo")(ia, kb, ab, ja) *
                                         t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0278_aa_vo"))(tmps.at("0278_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0277_aa_oo")(ja, ia))
      .allocate(tmps.at("0275_aa_vv"))(tmps.at("0275_aa_vv")(aa, ba) =
                                         eri.at("aaaa_vovv")(aa, ia, ca, ba) *
                                         t1_1p.at("aa")(ca, ia));
  }
}

template void exachem::cc::qed_ccsd_cs::resid_part1<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);