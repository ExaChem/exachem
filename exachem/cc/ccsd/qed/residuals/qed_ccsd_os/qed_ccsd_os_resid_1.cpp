/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_os_resid_1.hpp"

template<typename T>
void exachem::cc::qed_ccsd_os::resid_1(
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
    sch(energy() = 0.00)(r0_1p() = 0.00)(r0_2p() = 0.00)(r2_1p.at("bbbb")(ab, bb, ib, jb) =
                                                           w0 * t2_1p.at("bbbb")(ab, bb, ib, jb))(
      r2.at("aaaa")(aa, ba, ia, ja) = eri.at("aaaa_vvoo")(aa, ba, ia, ja))(
      r2_2p.at("bbbb")(ab, bb, ib, jb) = 4.00 * w0 * t2_2p.at("bbbb")(ab, bb, ib, jb))(
      r1_1p.at("bb")(ab, ib) = dp.at("bb_vo")(ab, ib))(
      r2_2p.at("abab")(aa, bb, ia, jb) = 4.00 * w0 * t2_2p.at("abab")(aa, bb, ia, jb))(
      r2.at("abab")(aa, bb, ia, jb) = eri.at("abab_vvoo")(aa, bb, ia, jb))(
      r1_2p.at("aa")(aa, ia) = 4.00 * w0 * t1_2p.at("aa")(aa, ia))(
      r2.at("bbbb")(ab, bb, ib, jb) = eri.at("bbbb_vvoo")(ab, bb, ib, jb))(
      r2_2p.at("aaaa")(aa, ba, ia, ja) = 4.00 * w0 * t2_2p.at("aaaa")(aa, ba, ia, ja))(
      r1_2p.at("bb")(ab, ib) = 4.00 * w0 * t1_2p.at("bb")(ab, ib))(
      r2_1p.at("abab")(aa, bb, ia, jb) = w0 * t2_1p.at("abab")(aa, bb, ia, jb))(
      r2_1p.at("aaaa")(aa, ba, ia, ja) = w0 * t2_1p.at("aaaa")(aa, ba, ia, ja))(
      r1.at("aa")(aa, ia) = f.at("aa_vo")(aa, ia))(r1.at("bb")(ab, ib) = f.at("bb_vo")(ab, ib))(
      energy() = scalars.at("0003")())(r1_1p.at("aa")(aa, ia) = dp.at("aa_vo")(aa, ia))(
      r0_2p() = 2.00 * scalars.at("0003")())(r0_1p() = scalars.at("0023")())(
      energy() -= 0.250 * scalars.at("0035")())(r0_2p() += 4.00 * scalars.at("0054")())(
      r1_1p.at("aa")(aa, ia) += w0 * t1_1p.at("aa")(aa, ia))(r1_1p.at("aa")(aa, ia) +=
                                                             2.00 * t0_2p * dp.at("aa_vo")(aa, ia))(
      r1_1p.at("bb")(ab, ib) += 2.00 * t0_2p * dp.at("bb_vo")(ab, ib))(r1_1p.at("bb")(ab, ib) +=
                                                                       w0 * t1_1p.at("bb")(ab, ib))(
      r1.at("aa")(aa, ia) += t0_1p * dp.at("aa_vo")(aa, ia))(r1.at("bb")(ab, ib) +=
                                                             t0_1p * dp.at("bb_vo")(ab, ib))
      .allocate(tmps.at("0001_bbbb_vvoo"))(tmps.at("bin_bb_vv")(ab, cb) =
                                             eri.at("abab_oovv")(ka, lb, da, cb) *
                                             t2.at("abab")(da, ab, ka, lb))(
        tmps.at("0001_bbbb_vvoo")(ab, bb, ib, jb) =
          tmps.at("bin_bb_vv")(ab, cb) * t2.at("bbbb")(cb, bb, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) += tmps.at("0001_bbbb_vvoo")(bb, ab, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) -= tmps.at("0001_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0001_bbbb_vvoo"))
      .allocate(tmps.at("0002_bbbb_vvoo"))(tmps.at("bin_bb_vv")(ab, cb) =
                                             eri.at("abab_oovv")(ka, lb, da, cb) *
                                             t2_1p.at("abab")(da, ab, ka, lb))(
        tmps.at("0002_bbbb_vvoo")(ab, bb, ib, jb) =
          tmps.at("bin_bb_vv")(ab, cb) * t2_1p.at("bbbb")(cb, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= 2.00 * tmps.at("0002_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("0002_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("0002_bbbb_vvoo"))
      .allocate(tmps.at("0005_bbbb_vvoo"))(tmps.at("0005_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             dp.at("bb_oo")(kb, jb))
      .allocate(tmps.at("0004_bbbb_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1_1p.at("bb")(cb, ib))(
        tmps.at("0004_bbbb_vvoo")(ab, bb, ib, jb) =
          tmps.at("bin_bb_oo")(ib, kb) * t2.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0003_bbbb_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1.at("bb")(cb, ib))(
        tmps.at("0003_bbbb_vvoo")(ab, bb, ib, jb) =
          tmps.at("bin_bb_oo")(ib, kb) * t2_1p.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0006_bbbb_vvoo"))(tmps.at("0006_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0004_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0006_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0005_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0006_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0003_bbbb_vvoo")(ab, bb, jb, ib))
      .deallocate(tmps.at("0005_bbbb_vvoo"))
      .deallocate(tmps.at("0004_bbbb_vvoo"))
      .deallocate(tmps.at("0003_bbbb_vvoo"))(tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb) =
                                               t0_1p * tmps.at("0006_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) -= tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += tmps.at("perm_bbbb_vvoo")(ab, bb, jb, ib))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= 2.00 * tmps.at("0006_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("0006_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb) =
          4.00 * t0_2p * tmps.at("0006_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += tmps.at("perm_bbbb_vvoo")(ab, bb, jb, ib))(
        r2.at("bbbb")(ab, bb, ib, jb) -= tmps.at("0006_bbbb_vvoo")(ab, bb, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) += tmps.at("0006_bbbb_vvoo")(ab, bb, jb, ib))
      .deallocate(tmps.at("0006_bbbb_vvoo"))
      .allocate(tmps.at("0008_bbbb_vvoo"))(tmps.at("0008_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) * dp.at("bb_oo")(kb, jb))
      .allocate(tmps.at("0007_bbbb_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1.at("bb")(cb, ib))(
        tmps.at("0007_bbbb_vvoo")(ab, bb, ib, jb) =
          tmps.at("bin_bb_oo")(ib, kb) * t2.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0010_bbbb_vvoo"))(tmps.at("0010_bbbb_vvoo")(ab, bb, ib, jb) =
                                             tmps.at("0007_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0010_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0008_bbbb_vvoo")(ab, bb, jb, ib))
      .deallocate(tmps.at("0008_bbbb_vvoo"))
      .deallocate(tmps.at("0007_bbbb_vvoo"))(r2_1p.at("bbbb")(ab, bb, ib, jb) -=
                                             tmps.at("0010_bbbb_vvoo")(ab, bb, jb, ib))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += tmps.at("0010_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb) =
          2.00 * t0_2p * tmps.at("0010_bbbb_vvoo")(ab, bb, jb, ib))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) -= tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += tmps.at("perm_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb) =
          t0_1p * tmps.at("0010_bbbb_vvoo")(ab, bb, jb, ib))(
        r2.at("bbbb")(ab, bb, ib, jb) -= tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) += tmps.at("perm_bbbb_vvoo")(ab, bb, jb, ib))
      .deallocate(tmps.at("0010_bbbb_vvoo"))
      .allocate(tmps.at("0014_bbbb_vvoo"))(tmps.at("0014_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, ib, kb) *
                                             dp.at("bb_oo")(kb, jb))
      .allocate(tmps.at("0013_bbbb_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1_1p.at("bb")(cb, ib))(
        tmps.at("0013_bbbb_vvoo")(ab, bb, ib, jb) =
          tmps.at("bin_bb_oo")(ib, kb) * t2_1p.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0012_bbbb_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1.at("bb")(cb, ib))(
        tmps.at("0012_bbbb_vvoo")(ab, bb, ib, jb) =
          tmps.at("bin_bb_oo")(ib, kb) * t2_2p.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0011_bbbb_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1_2p.at("bb")(cb, ib))(
        tmps.at("0011_bbbb_vvoo")(ab, bb, ib, jb) =
          tmps.at("bin_bb_oo")(ib, kb) * t2.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0017_bbbb_vvoo"))(tmps.at("0017_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0011_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("0017_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0012_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0017_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0013_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0017_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0014_bbbb_vvoo")(ab, bb, jb, ib))
      .deallocate(tmps.at("0014_bbbb_vvoo"))
      .deallocate(tmps.at("0013_bbbb_vvoo"))
      .deallocate(tmps.at("0012_bbbb_vvoo"))
      .deallocate(tmps.at("0011_bbbb_vvoo"))(r2_1p.at("bbbb")(ab, bb, ib, jb) -=
                                             2.00 * tmps.at("0017_bbbb_vvoo")(ab, bb, jb, ib))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("0017_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb) =
          2.00 * t0_1p * tmps.at("0017_bbbb_vvoo")(ab, bb, jb, ib))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += tmps.at("perm_bbbb_vvoo")(ab, bb, jb, ib))
      .deallocate(tmps.at("0017_bbbb_vvoo"))
      .allocate(tmps.at("0020_aaaa_vvoo"))(tmps.at("0020_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             dp.at("aa_oo")(ka, ja))
      .allocate(tmps.at("0019_aaaa_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1_1p.at("aa")(ca, ia))(
        tmps.at("0019_aaaa_vvoo")(aa, ba, ia, ja) =
          tmps.at("bin_aa_oo")(ia, ka) * t2.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0018_aaaa_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1.at("aa")(ca, ia))(
        tmps.at("0018_aaaa_vvoo")(aa, ba, ia, ja) =
          tmps.at("bin_aa_oo")(ia, ka) * t2_1p.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0021_aaaa_vvoo"))(tmps.at("0021_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0019_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("0021_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0018_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0021_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0020_aaaa_vvoo")(aa, ba, ja, ia))
      .deallocate(tmps.at("0020_aaaa_vvoo"))
      .deallocate(tmps.at("0019_aaaa_vvoo"))
      .deallocate(tmps.at("0018_aaaa_vvoo"))(tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja) =
                                               t0_1p * tmps.at("0021_aaaa_vvoo")(aa, ba, ja, ia))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += tmps.at("perm_aaaa_vvoo")(aa, ba, ja, ia))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= 2.00 * tmps.at("0021_aaaa_vvoo")(aa, ba, ja, ia))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += 2.00 * tmps.at("0021_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja) =
          4.00 * t0_2p * tmps.at("0021_aaaa_vvoo")(aa, ba, ja, ia))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += tmps.at("perm_aaaa_vvoo")(aa, ba, ja, ia))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0021_aaaa_vvoo")(aa, ba, ja, ia))(
        r2.at("aaaa")(aa, ba, ia, ja) += tmps.at("0021_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0021_aaaa_vvoo"))
      .allocate(tmps.at("0025_aaaa_vvoo"))(tmps.at("0025_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) * dp.at("aa_oo")(ka, ja))
      .allocate(tmps.at("0024_aaaa_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1.at("aa")(ca, ia))(
        tmps.at("0024_aaaa_vvoo")(aa, ba, ia, ja) =
          tmps.at("bin_aa_oo")(ia, ka) * t2.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0026_aaaa_vvoo"))(tmps.at("0026_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("0024_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0026_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0025_aaaa_vvoo")(aa, ba, ja, ia))
      .deallocate(tmps.at("0025_aaaa_vvoo"))
      .deallocate(tmps.at("0024_aaaa_vvoo"))(r2_1p.at("aaaa")(aa, ba, ia, ja) -=
                                             tmps.at("0026_aaaa_vvoo")(aa, ba, ja, ia))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += tmps.at("0026_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja) =
          2.00 * t0_2p * tmps.at("0026_aaaa_vvoo")(aa, ba, ja, ia))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += tmps.at("perm_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja) =
          t0_1p * tmps.at("0026_aaaa_vvoo")(aa, ba, ja, ia))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) += tmps.at("perm_aaaa_vvoo")(aa, ba, ja, ia))
      .deallocate(tmps.at("0026_aaaa_vvoo"))
      .allocate(tmps.at("0030_aaaa_vvoo"))(tmps.at("0030_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ia, ka) *
                                             dp.at("aa_oo")(ka, ja))
      .allocate(tmps.at("0029_aaaa_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1.at("aa")(ca, ia))(
        tmps.at("0029_aaaa_vvoo")(aa, ba, ia, ja) =
          tmps.at("bin_aa_oo")(ia, ka) * t2_2p.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0028_aaaa_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1_1p.at("aa")(ca, ia))(
        tmps.at("0028_aaaa_vvoo")(aa, ba, ia, ja) =
          tmps.at("bin_aa_oo")(ia, ka) * t2_1p.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0027_aaaa_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1_2p.at("aa")(ca, ia))(
        tmps.at("0027_aaaa_vvoo")(aa, ba, ia, ja) =
          tmps.at("bin_aa_oo")(ia, ka) * t2.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0032_aaaa_vvoo"))(tmps.at("0032_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("0028_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0032_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0027_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("0032_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0029_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0032_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0030_aaaa_vvoo")(aa, ba, ja, ia))
      .deallocate(tmps.at("0030_aaaa_vvoo"))
      .deallocate(tmps.at("0029_aaaa_vvoo"))
      .deallocate(tmps.at("0028_aaaa_vvoo"))
      .deallocate(tmps.at("0027_aaaa_vvoo"))(r2_1p.at("aaaa")(aa, ba, ia, ja) -=
                                             2.00 * tmps.at("0032_aaaa_vvoo")(aa, ba, ja, ia))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += 2.00 * tmps.at("0032_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja) =
          2.00 * t0_1p * tmps.at("0032_aaaa_vvoo")(aa, ba, ja, ia))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += tmps.at("perm_aaaa_vvoo")(aa, ba, ja, ia))
      .deallocate(tmps.at("0032_aaaa_vvoo"))
      .allocate(tmps.at("0033_baba_voov"))(tmps.at("0033_baba_voov")(ab, ia, jb, ba) =
                                             t2.at("bbbb")(cb, ab, jb, kb) *
                                             eri.at("abab_oovv")(ia, kb, ba, cb))
      .allocate(tmps.at("0034_bbbb_vvoo"))(tmps.at("0034_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0033_baba_voov")(bb, ka, jb, ca))(
        r2.at("bbbb")(ab, bb, ib, jb) += tmps.at("0034_bbbb_vvoo")(ab, bb, jb, ib))(
        r2.at("bbbb")(ab, bb, ib, jb) -= tmps.at("0034_bbbb_vvoo")(ab, bb, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) += tmps.at("0034_bbbb_vvoo")(bb, ab, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) -= tmps.at("0034_bbbb_vvoo")(bb, ab, jb, ib))
      .deallocate(tmps.at("0034_bbbb_vvoo"))
      .allocate(tmps.at("0035_baba_voov"))(tmps.at("0035_baba_voov")(ab, ia, jb, ba) =
                                             t2_1p.at("bbbb")(cb, ab, jb, kb) *
                                             eri.at("abab_oovv")(ia, kb, ba, cb))
      .allocate(tmps.at("0036_bbbb_vvoo"))(tmps.at("0036_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0035_baba_voov")(bb, ka, jb, ca))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("0036_bbbb_vvoo")(ab, bb, jb, ib))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= 2.00 * tmps.at("0036_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("0036_bbbb_vvoo")(bb, ab, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= 2.00 * tmps.at("0036_bbbb_vvoo")(bb, ab, jb, ib))
      .deallocate(tmps.at("0036_bbbb_vvoo"))
      .allocate(tmps.at("0037_aaaa_voov"))(tmps.at("0037_aaaa_voov")(aa, ia, ja, ba) =
                                             t2.at("abab")(aa, cb, ja, kb) *
                                             eri.at("abab_oovv")(ia, kb, ba, cb))
      .allocate(tmps.at("0038_aaaa_vvoo"))(tmps.at("0038_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0037_aaaa_voov")(ba, ka, ja, ca))(
        r2.at("aaaa")(aa, ba, ia, ja) += tmps.at("0038_aaaa_vvoo")(aa, ba, ja, ia))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0038_aaaa_vvoo")(aa, ba, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) += tmps.at("0038_aaaa_vvoo")(ba, aa, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0038_aaaa_vvoo")(ba, aa, ja, ia))
      .deallocate(tmps.at("0038_aaaa_vvoo"))
      .allocate(tmps.at("0039_aaaa_voov"))(tmps.at("0039_aaaa_voov")(aa, ia, ja, ba) =
                                             t2_1p.at("abab")(aa, cb, ja, kb) *
                                             eri.at("abab_oovv")(ia, kb, ba, cb))
      .allocate(tmps.at("0040_aaaa_vvoo"))(tmps.at("0040_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0039_aaaa_voov")(ba, ka, ja, ca))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += 2.00 * tmps.at("0040_aaaa_vvoo")(ba, aa, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= 2.00 * tmps.at("0040_aaaa_vvoo")(ba, aa, ja, ia))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += 2.00 * tmps.at("0040_aaaa_vvoo")(aa, ba, ja, ia))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= 2.00 * tmps.at("0040_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0040_aaaa_vvoo"))
      .allocate(tmps.at("0041_aa_vv"))(tmps.at("0041_aa_vv")(aa, ba) =
                                         eri.at("abab_oovv")(ia, jb, ba, cb) *
                                         t2.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0042_aaaa_vvoo"))(tmps.at("0042_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0041_aa_vv")(ba, ca))(
        r2.at("aaaa")(aa, ba, ia, ja) += tmps.at("0042_aaaa_vvoo")(aa, ba, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0042_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("0042_aaaa_vvoo"))
      .allocate(tmps.at("0043_aa_vv"))(tmps.at("0043_aa_vv")(aa, ba) =
                                         eri.at("abab_oovv")(ia, jb, ba, cb) *
                                         t2_1p.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0044_aaaa_vvoo"))(tmps.at("0044_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0043_aa_vv")(ba, ca))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= 2.00 * tmps.at("0044_aaaa_vvoo")(ba, aa, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += 2.00 * tmps.at("0044_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0044_aaaa_vvoo"))
      .allocate(tmps.at("0048_bbbb_vooo"))(tmps.at("0048_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_1p.at("bbbb")(bb, ab, jb, kb) *
                                             dp.at("bb_ov")(ib, bb))
      .allocate(tmps.at("0049_bbbb_vvoo"))(tmps.at("0049_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0048_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0046_bbbb_vooo"))(tmps.at("0046_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("bbbb")(bb, ab, jb, kb) * dp.at("bb_ov")(ib, bb))
      .allocate(tmps.at("0047_bbbb_vvoo"))(tmps.at("0047_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0046_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0045_bbbb_vvoo"))(tmps.at("0045_bbbb_vvoo")(ab, bb, ib, jb) =
                                             dp.at("bb_vv")(ab, cb) *
                                             t2_1p.at("bbbb")(cb, bb, ib, jb))
      .allocate(tmps.at("0050_bbbb_vvoo"))(tmps.at("0050_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0049_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0050_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0045_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0050_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0047_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("0049_bbbb_vvoo"))
      .deallocate(tmps.at("0047_bbbb_vvoo"))
      .deallocate(tmps.at("0045_bbbb_vvoo"))(tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb) =
                                               t0_1p * tmps.at("0050_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) -= tmps.at("perm_bbbb_vvoo")(bb, ab, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("0050_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= 2.00 * tmps.at("0050_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb) =
          4.00 * t0_2p * tmps.at("0050_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= tmps.at("perm_bbbb_vvoo")(bb, ab, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) += tmps.at("0050_bbbb_vvoo")(ab, bb, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) -= tmps.at("0050_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("0050_bbbb_vvoo"))
      .allocate(tmps.at("0052_aaaa_vooo"))(tmps.at("0052_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, ka) * dp.at("aa_ov")(ia, ba))
      .allocate(tmps.at("0053_aaaa_vvoo"))(tmps.at("0053_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0052_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0051_aaaa_vvoo"))(tmps.at("0051_aaaa_vvoo")(aa, ba, ia, ja) =
                                             dp.at("aa_vv")(aa, ca) * t2.at("aaaa")(ca, ba, ia, ja))
      .allocate(tmps.at("0054_aaaa_vvoo"))(tmps.at("0054_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0053_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0054_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0051_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0053_aaaa_vvoo"))
      .deallocate(tmps.at("0051_aaaa_vvoo"))(r2_1p.at("aaaa")(aa, ba, ia, ja) +=
                                             tmps.at("0054_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0054_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja) =
          2.00 * t0_2p * tmps.at("0054_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("perm_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja) =
          t0_1p * tmps.at("0054_aaaa_vvoo")(aa, ba, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) += tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("perm_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("0054_aaaa_vvoo"))
      .allocate(tmps.at("0064_baab_vooo"))(tmps.at("0064_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, kb) * dp.at("aa_ov")(ia, ba))
      .allocate(tmps.at("0065_abab_vvoo"))(tmps.at("0065_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0064_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0062_abab_vooo"))(tmps.at("0062_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, kb) * dp.at("bb_ov")(ib, bb))
      .allocate(tmps.at("0063_baab_vvoo"))(tmps.at("0063_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0062_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("0060_abab_vvoo"))(tmps.at("0060_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_oo")(ka, ia) * t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0059_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1.at("aa")(ca, ia))(
        tmps.at("0059_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0058_abab_vvoo"))(tmps.at("0058_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) * dp.at("bb_oo")(kb, jb))
      .allocate(tmps.at("0057_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1.at("bb")(cb, ib))(
        tmps.at("0057_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0056_abab_vvoo"))(tmps.at("0056_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_vv")(aa, ca) * t2.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0055_abab_vvoo"))(tmps.at("0055_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) * dp.at("bb_vv")(bb, cb))
      .allocate(tmps.at("0061_abab_vvoo"))(tmps.at("0061_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0058_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0061_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0059_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0061_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0056_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0061_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0057_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0061_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0055_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0061_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0060_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0060_abab_vvoo"))
      .deallocate(tmps.at("0059_abab_vvoo"))
      .deallocate(tmps.at("0058_abab_vvoo"))
      .deallocate(tmps.at("0057_abba_vvoo"))
      .deallocate(tmps.at("0056_abab_vvoo"))
      .deallocate(tmps.at("0055_abab_vvoo"))
      .allocate(tmps.at("0066_abab_vvoo"))(tmps.at("0066_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0065_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0066_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0063_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0066_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0061_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0065_abab_vvoo"))
      .deallocate(tmps.at("0063_baab_vvoo"))
      .deallocate(tmps.at("0061_abab_vvoo"))(r2_1p.at("abab")(aa, bb, ia, jb) +=
                                             tmps.at("0066_abab_vvoo")(aa, bb, ia, jb))(
        r2_1p.at("abab")(aa, bb, ia, jb) +=
        2.00 * t0_2p * tmps.at("0066_abab_vvoo")(aa, bb, ia, jb))(
        r2.at("abab")(aa, bb, ia, jb) += t0_1p * tmps.at("0066_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0066_abab_vvoo"))
      .allocate(tmps.at("0068_bbbb_vvoo"))(tmps.at("0068_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0046_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0067_bbbb_vvoo"))(tmps.at("0067_bbbb_vvoo")(ab, bb, ib, jb) =
                                             dp.at("bb_vv")(ab, cb) * t2.at("bbbb")(cb, bb, ib, jb))
      .allocate(tmps.at("0069_bbbb_vvoo"))(tmps.at("0069_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0067_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0069_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0068_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0068_bbbb_vvoo"))
      .deallocate(tmps.at("0067_bbbb_vvoo"))(r2_1p.at("bbbb")(ab, bb, ib, jb) -=
                                             tmps.at("0069_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += tmps.at("0069_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb) =
          2.00 * t0_2p * tmps.at("0069_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) -= tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += tmps.at("perm_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb) =
          t0_1p * tmps.at("0069_bbbb_vvoo")(ab, bb, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) -= tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) += tmps.at("perm_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("0069_bbbb_vvoo"))
      .allocate(tmps.at("0074_bbbb_vvoo"))(tmps.at("0074_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0048_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0073_bbbb_vvoo"))(tmps.at("0073_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0046_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0046_bbbb_vooo"))
      .allocate(tmps.at("0071_bbbb_vooo"))(tmps.at("0071_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_2p.at("bbbb")(bb, ab, jb, kb) *
                                             dp.at("bb_ov")(ib, bb))
      .allocate(tmps.at("0072_bbbb_vvoo"))(tmps.at("0072_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0071_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0070_bbbb_vvoo"))(tmps.at("0070_bbbb_vvoo")(ab, bb, ib, jb) =
                                             dp.at("bb_vv")(ab, cb) *
                                             t2_2p.at("bbbb")(cb, bb, ib, jb))
      .allocate(tmps.at("0075_bbbb_vvoo"))(tmps.at("0075_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0070_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0075_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0072_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0075_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0074_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0075_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("0073_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("0074_bbbb_vvoo"))
      .deallocate(tmps.at("0073_bbbb_vvoo"))
      .deallocate(tmps.at("0072_bbbb_vvoo"))
      .deallocate(tmps.at("0070_bbbb_vvoo"))(r2_1p.at("bbbb")(ab, bb, ib, jb) -=
                                             2.00 * tmps.at("0075_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("0075_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb) =
          2.00 * t0_1p * tmps.at("0075_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= tmps.at("perm_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += tmps.at("perm_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("0075_bbbb_vvoo"))
      .allocate(tmps.at("0089_baab_vooo"))(tmps.at("0089_baab_vooo")(ab, ia, ja, kb) =
                                             t2_1p.at("abab")(ba, ab, ja, kb) *
                                             dp.at("aa_ov")(ia, ba))
      .allocate(tmps.at("0090_abab_vvoo"))(tmps.at("0090_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0089_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0088_abab_vvoo"))(tmps.at("0088_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0064_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0086_abab_vooo"))(tmps.at("0086_abab_vooo")(aa, ib, ja, kb) =
                                             t2_1p.at("abab")(aa, bb, ja, kb) *
                                             dp.at("bb_ov")(ib, bb))
      .allocate(tmps.at("0087_baab_vvoo"))(tmps.at("0087_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0086_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("0085_baab_vvoo"))(tmps.at("0085_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0062_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0083_abab_vvoo"))(tmps.at("0083_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_oo")(ka, ia) *
                                             t2_1p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0082_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1_1p.at("aa")(ca, ia))(
        tmps.at("0082_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0081_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1.at("aa")(ca, ia))(
        tmps.at("0081_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2_1p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0080_abab_vvoo"))(tmps.at("0080_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             dp.at("bb_oo")(kb, jb))
      .allocate(tmps.at("0079_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1_1p.at("bb")(cb, ib))(
        tmps.at("0079_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0078_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1.at("bb")(cb, ib))(
        tmps.at("0078_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0077_abab_vvoo"))(tmps.at("0077_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_vv")(aa, ca) *
                                             t2_1p.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0076_abab_vvoo"))(tmps.at("0076_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             dp.at("bb_vv")(bb, cb))
      .allocate(tmps.at("0084_abab_vvoo"))(tmps.at("0084_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0076_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0084_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0079_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0084_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0080_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0084_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0078_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0084_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0081_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0084_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0082_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0084_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0077_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0084_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0083_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0083_abab_vvoo"))
      .deallocate(tmps.at("0082_abab_vvoo"))
      .deallocate(tmps.at("0081_abab_vvoo"))
      .deallocate(tmps.at("0080_abab_vvoo"))
      .deallocate(tmps.at("0079_abba_vvoo"))
      .deallocate(tmps.at("0078_abba_vvoo"))
      .deallocate(tmps.at("0077_abab_vvoo"))
      .deallocate(tmps.at("0076_abab_vvoo"))
      .allocate(tmps.at("0091_baab_vvoo"))(tmps.at("0091_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0084_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0091_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0085_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0091_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0087_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0091_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0090_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0091_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0088_abab_vvoo")(ba, ab, ia, jb))
      .deallocate(tmps.at("0090_abab_vvoo"))
      .deallocate(tmps.at("0088_abab_vvoo"))
      .deallocate(tmps.at("0087_baab_vvoo"))
      .deallocate(tmps.at("0085_baab_vvoo"))
      .deallocate(tmps.at("0084_abab_vvoo"))(r2_1p.at("abab")(aa, bb, ia, jb) -=
                                             t0_1p * tmps.at("0091_baab_vvoo")(bb, aa, ia, jb))(
        r2_2p.at("abab")(aa, bb, ia, jb) -= 2.00 * tmps.at("0091_baab_vvoo")(bb, aa, ia, jb))(
        r2_2p.at("abab")(aa, bb, ia, jb) -=
        4.00 * t0_2p * tmps.at("0091_baab_vvoo")(bb, aa, ia, jb))(
        r2.at("abab")(aa, bb, ia, jb) -= tmps.at("0091_baab_vvoo")(bb, aa, ia, jb))
      .deallocate(tmps.at("0091_baab_vvoo"))
      .allocate(tmps.at("0110_abab_vvoo"))(tmps.at("0110_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0064_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0064_baab_vooo"))
      .allocate(tmps.at("0108_baab_vooo"))(tmps.at("0108_baab_vooo")(ab, ia, ja, kb) =
                                             t2_2p.at("abab")(ba, ab, ja, kb) *
                                             dp.at("aa_ov")(ia, ba))
      .allocate(tmps.at("0109_abab_vvoo"))(tmps.at("0109_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0108_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0107_abab_vvoo"))(tmps.at("0107_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0089_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0105_abab_vooo"))(tmps.at("0105_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("abab")(aa, bb, ja, kb) *
                                             dp.at("bb_ov")(ib, bb))
      .allocate(tmps.at("0106_baab_vvoo"))(tmps.at("0106_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0105_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("0104_baab_vvoo"))(tmps.at("0104_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0062_abab_vooo")(ba, kb, ia, jb) *
                                             t1_2p.at("bb")(ab, kb))
      .deallocate(tmps.at("0062_abab_vooo"))
      .allocate(tmps.at("0103_baab_vvoo"))(tmps.at("0103_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0086_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0101_abab_vvoo"))(tmps.at("0101_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_oo")(ka, ia) *
                                             t2_2p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0100_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1_2p.at("aa")(ca, ia))(
        tmps.at("0100_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0099_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1_1p.at("aa")(ca, ia))(
        tmps.at("0099_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2_1p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0098_abab_vvoo"))(tmps.at("bin_aa_oo")(ia, ka) =
                                             dp.at("aa_ov")(ka, ca) * t1.at("aa")(ca, ia))(
        tmps.at("0098_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_aa_oo")(ia, ka) * t2_2p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0097_abab_vvoo"))(tmps.at("0097_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             dp.at("bb_oo")(kb, jb))
      .allocate(tmps.at("0096_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1_2p.at("bb")(cb, ib))(
        tmps.at("0096_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0095_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1_1p.at("bb")(cb, ib))(
        tmps.at("0095_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0094_abba_vvoo"))(tmps.at("bin_bb_oo")(ib, kb) =
                                             dp.at("bb_ov")(kb, cb) * t1.at("bb")(cb, ib))(
        tmps.at("0094_abba_vvoo")(aa, bb, ib, ja) =
          tmps.at("bin_bb_oo")(ib, kb) * t2_2p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0093_abab_vvoo"))(tmps.at("0093_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_vv")(aa, ca) *
                                             t2_2p.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0092_abab_vvoo"))(tmps.at("0092_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, jb) *
                                             dp.at("bb_vv")(bb, cb))
      .allocate(tmps.at("0102_abab_vvoo"))(tmps.at("0102_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0094_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0102_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0095_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0102_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0097_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0102_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0100_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0102_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0101_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0102_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0092_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0102_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0093_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0102_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0099_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0102_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0098_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0102_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0096_abba_vvoo")(aa, bb, jb, ia))
      .deallocate(tmps.at("0101_abab_vvoo"))
      .deallocate(tmps.at("0100_abab_vvoo"))
      .deallocate(tmps.at("0099_abab_vvoo"))
      .deallocate(tmps.at("0098_abab_vvoo"))
      .deallocate(tmps.at("0097_abab_vvoo"))
      .deallocate(tmps.at("0096_abba_vvoo"))
      .deallocate(tmps.at("0095_abba_vvoo"))
      .deallocate(tmps.at("0094_abba_vvoo"))
      .deallocate(tmps.at("0093_abab_vvoo"))
      .deallocate(tmps.at("0092_abab_vvoo"))
      .allocate(tmps.at("0111_abab_vvoo"))(tmps.at("0111_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0102_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0111_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0103_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0111_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0109_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0111_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0106_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0111_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0110_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0111_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0104_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0111_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0107_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0110_abab_vvoo"))
      .deallocate(tmps.at("0109_abab_vvoo"))
      .deallocate(tmps.at("0107_abab_vvoo"))
      .deallocate(tmps.at("0106_baab_vvoo"))
      .deallocate(tmps.at("0104_baab_vvoo"))
      .deallocate(tmps.at("0103_baab_vvoo"))
      .deallocate(tmps.at("0102_abab_vvoo"))(r2_1p.at("abab")(aa, bb, ia, jb) -=
                                             2.00 * tmps.at("0111_abab_vvoo")(aa, bb, ia, jb))(
        r2_2p.at("abab")(aa, bb, ia, jb) -=
        2.00 * t0_1p * tmps.at("0111_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0111_abab_vvoo"))
      .allocate(tmps.at("0115_aaaa_vvoo"))(tmps.at("0115_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0052_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0113_aaaa_vooo"))(tmps.at("0113_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_1p.at("aaaa")(ba, aa, ja, ka) *
                                             dp.at("aa_ov")(ia, ba))
      .allocate(tmps.at("0114_aaaa_vvoo"))(tmps.at("0114_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0113_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0112_aaaa_vvoo"))(tmps.at("0112_aaaa_vvoo")(aa, ba, ia, ja) =
                                             dp.at("aa_vv")(aa, ca) *
                                             t2_1p.at("aaaa")(ca, ba, ia, ja))
      .allocate(tmps.at("0116_aaaa_vvoo"))(tmps.at("0116_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0112_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0116_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0115_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0116_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0114_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0115_aaaa_vvoo"))
      .deallocate(tmps.at("0114_aaaa_vvoo"))
      .deallocate(tmps.at("0112_aaaa_vvoo"))(tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja) =
                                               t0_1p * tmps.at("0116_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += tmps.at("perm_aaaa_vvoo")(ba, aa, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= 2.00 * tmps.at("0116_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += 2.00 * tmps.at("0116_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja) =
          4.00 * t0_2p * tmps.at("0116_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += tmps.at("perm_aaaa_vvoo")(ba, aa, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0116_aaaa_vvoo")(aa, ba, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) += tmps.at("0116_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("0116_aaaa_vvoo"))
      .allocate(tmps.at("0120_aaaa_vooo"))(tmps.at("0120_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_2p.at("aaaa")(ba, aa, ja, ka) *
                                             dp.at("aa_ov")(ia, ba))
      .allocate(tmps.at("0121_aaaa_vvoo"))(tmps.at("0121_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0120_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0119_aaaa_vvoo"))(tmps.at("0119_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0113_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0118_aaaa_vvoo"))(tmps.at("0118_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0052_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0052_aaaa_vooo"))
      .allocate(tmps.at("0117_aaaa_vvoo"))(tmps.at("0117_aaaa_vvoo")(aa, ba, ia, ja) =
                                             dp.at("aa_vv")(aa, ca) *
                                             t2_2p.at("aaaa")(ca, ba, ia, ja))
      .allocate(tmps.at("0122_aaaa_vvoo"))(tmps.at("0122_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0117_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0122_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0119_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0122_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0121_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0122_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0118_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("0121_aaaa_vvoo"))
      .deallocate(tmps.at("0119_aaaa_vvoo"))
      .deallocate(tmps.at("0118_aaaa_vvoo"))
      .deallocate(tmps.at("0117_aaaa_vvoo"))(r2_1p.at("aaaa")(aa, ba, ia, ja) -=
                                             2.00 * tmps.at("0122_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += 2.00 * tmps.at("0122_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja) =
          2.00 * t0_1p * tmps.at("0122_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("perm_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += tmps.at("perm_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("0122_aaaa_vvoo"))
      .allocate(tmps.at("0153_bb_vo"))(tmps.at("0153_bb_vo")(ab, ib) =
                                         dp.at("aa_ov")(ja, ba) * t2_1p.at("abab")(ba, ab, ja, ib))
      .allocate(tmps.at("0152_bb_vo"))(tmps.at("0152_bb_vo")(ab, ib) =
                                         dp.at("bb_ov")(jb, bb) * t2_1p.at("bbbb")(bb, ab, ib, jb))
      .allocate(tmps.at("0154_bb_vo"))(tmps.at("0154_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("0153_bb_vo")(ab, ib))(
        tmps.at("0154_bb_vo")(ab, ib) += tmps.at("0152_bb_vo")(ab, ib))
      .deallocate(tmps.at("0153_bb_vo"))
      .deallocate(tmps.at("0152_bb_vo"))(r1_1p.at("bb")(ab, ib) -=
                                         t0_1p * tmps.at("0154_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) -= 2.00 * tmps.at("0154_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) -= 4.00 * t0_2p * tmps.at("0154_bb_vo")(ab, ib))(
        r1.at("bb")(ab, ib) -= tmps.at("0154_bb_vo")(ab, ib))
      .allocate(tmps.at("0157_bb_vo"))(tmps.at("0157_bb_vo")(ab, ib) =
                                         dp.at("bb_oo")(jb, ib) * t1_1p.at("bb")(ab, jb))
      .allocate(tmps.at("0156_bb_vo"))(tmps.at("0156_bb_vo")(ab, ib) =
                                         dp.at("bb_vv")(ab, bb) * t1_1p.at("bb")(bb, ib))
      .allocate(tmps.at("0158_bb_vo"))(tmps.at("0158_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("0157_bb_vo")(ab, ib))(
        tmps.at("0158_bb_vo")(ab, ib) += tmps.at("0156_bb_vo")(ab, ib))
      .deallocate(tmps.at("0157_bb_vo"))
      .deallocate(tmps.at("0156_bb_vo"))(r1_1p.at("bb")(ab, ib) +=
                                         t0_1p * tmps.at("0158_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) += 2.00 * tmps.at("0158_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) += 4.00 * t0_2p * tmps.at("0158_bb_vo")(ab, ib))(
        r1.at("bb")(ab, ib) += tmps.at("0158_bb_vo")(ab, ib))
      .allocate(tmps.at("0163_bb_vo"))(tmps.at("0163_bb_vo")(ab, ib) =
                                         dp.at("bb_oo")(jb, ib) * t1.at("bb")(ab, jb))
      .allocate(tmps.at("0162_bb_vo"))(tmps.at("0162_bb_vo")(ab, ib) =
                                         dp.at("bb_vv")(ab, bb) * t1.at("bb")(bb, ib))
      .allocate(tmps.at("0161_bb_vo"))(tmps.at("0161_bb_vo")(ab, ib) =
                                         dp.at("aa_ov")(ja, ba) * t2.at("abab")(ba, ab, ja, ib))
      .allocate(tmps.at("0160_bb_vo"))(tmps.at("0160_bb_vo")(ab, ib) =
                                         dp.at("bb_ov")(jb, bb) * t2.at("bbbb")(bb, ab, ib, jb))
      .allocate(tmps.at("0164_bb_vo"))(tmps.at("0164_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("0161_bb_vo")(ab, ib))(
        tmps.at("0164_bb_vo")(ab, ib) -= tmps.at("0162_bb_vo")(ab, ib))(
        tmps.at("0164_bb_vo")(ab, ib) += tmps.at("0160_bb_vo")(ab, ib))(
        tmps.at("0164_bb_vo")(ab, ib) += tmps.at("0163_bb_vo")(ab, ib))
      .deallocate(tmps.at("0163_bb_vo"))
      .deallocate(tmps.at("0162_bb_vo"))
      .deallocate(tmps.at("0161_bb_vo"))
      .deallocate(tmps.at("0160_bb_vo"))(r1_1p.at("bb")(ab, ib) -= tmps.at("0164_bb_vo")(ab, ib))(
        r1_1p.at("bb")(ab, ib) -= 2.00 * t0_2p * tmps.at("0164_bb_vo")(ab, ib))(
        r1.at("bb")(ab, ib) -= t0_1p * tmps.at("0164_bb_vo")(ab, ib))
      .allocate(tmps.at("0009_bb_oo"))(tmps.at("0009_bb_oo")(ib, jb) =
                                         dp.at("bb_ov")(ib, ab) * t1.at("bb")(ab, jb))
      .allocate(tmps.at("0183_bb_vo"))(tmps.at("0183_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0009_bb_oo")(jb, ib))(
        r1_1p.at("bb")(ab, ib) -= tmps.at("0183_bb_vo")(ab, ib))(
        r1_1p.at("bb")(ab, ib) -= 2.00 * t0_2p * tmps.at("0183_bb_vo")(ab, ib))(
        r1.at("bb")(ab, ib) -= t0_1p * tmps.at("0183_bb_vo")(ab, ib))
      .allocate(tmps.at("0187_bb_vo"))(tmps.at("0187_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0009_bb_oo")(jb, ib))
      .allocate(tmps.at("0015_bb_oo"))(tmps.at("0015_bb_oo")(ib, jb) =
                                         dp.at("bb_ov")(ib, ab) * t1_1p.at("bb")(ab, jb))
      .allocate(tmps.at("0186_bb_vo"))(tmps.at("0186_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0015_bb_oo")(jb, ib))
      .allocate(tmps.at("0188_bb_vo"))(tmps.at("0188_bb_vo")(ab, ib) =
                                         tmps.at("0186_bb_vo")(ab, ib))(
        tmps.at("0188_bb_vo")(ab, ib) += tmps.at("0187_bb_vo")(ab, ib))
      .deallocate(tmps.at("0187_bb_vo"))
      .deallocate(tmps.at("0186_bb_vo"))(r1_1p.at("bb")(ab, ib) -=
                                         t0_1p * tmps.at("0188_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) -= 2.00 * tmps.at("0188_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) -= 4.00 * t0_2p * tmps.at("0188_bb_vo")(ab, ib))(
        r1.at("bb")(ab, ib) -= tmps.at("0188_bb_vo")(ab, ib))
      .allocate(tmps.at("0189_bbbb_vvoo"))(tmps.at("0189_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("0188_bb_vo")(bb, jb))
      .allocate(tmps.at("0127_bbbb_oovo"))(tmps.at("0127_bbbb_oovo")(ib, jb, ab, kb) =
                                             eri.at("bbbb_oovv")(ib, jb, ab, bb) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("0128_bbbb_vooo"))(tmps.at("0128_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("bbbb")(bb, ab, jb, lb) *
                                             tmps.at("0127_bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("0185_bbbb_vvoo"))(tmps.at("0185_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0128_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0128_bbbb_vooo"))
      .allocate(tmps.at("0184_bbbb_vvoo"))(tmps.at("0184_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, ib) * tmps.at("0183_bb_vo")(bb, jb))
      .allocate(tmps.at("0141_bbbb_vovo"))(tmps.at("0141_bbbb_vovo")(ab, ib, bb, jb) =
                                             eri.at("bbbb_vovv")(ab, ib, cb, bb) *
                                             t1.at("bb")(cb, jb))
      .allocate(tmps.at("0181_bbbb_vooo"))(tmps.at("0181_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1_1p.at("bb")(bb, jb) *
                                             tmps.at("0141_bbbb_vovo")(ab, ib, bb, kb))
      .allocate(tmps.at("0182_bbbb_vvoo"))(tmps.at("0182_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0181_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0178_bbbb_oooo"))(tmps.at("0178_bbbb_oooo")(ib, jb, kb, lb) =
                                             t1.at("bb")(ab, kb) *
                                             eri.at("bbbb_oovo")(ib, jb, ab, lb))
      .allocate(tmps.at("0179_bbbb_vooo"))(tmps.at("0179_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0178_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("0180_bbbb_vvoo"))(tmps.at("0180_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0179_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0174_bbbb_vooo"))(tmps.at("0174_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("bbbb")(bb, ab, jb, lb) *
                                             eri.at("bbbb_oovo")(lb, ib, bb, kb))
      .allocate(tmps.at("0173_bbbb_ovoo"))(tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) =
                                             eri.at("bbbb_oovv")(lb, ib, cb, bb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("0173_bbbb_ovoo")(ib, ab, jb, kb) =
          tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) * t2.at("bbbb")(bb, ab, kb, lb))
      .allocate(tmps.at("0175_bbbb_vooo"))(tmps.at("0175_bbbb_vooo")(ab, ib, jb, kb) =
                                             -1.00 * tmps.at("0173_bbbb_ovoo")(ib, ab, kb, jb))(
        tmps.at("0175_bbbb_vooo")(ab, ib, jb, kb) += tmps.at("0174_bbbb_vooo")(ab, ib, jb, kb))
      .deallocate(tmps.at("0174_bbbb_vooo"))
      .deallocate(tmps.at("0173_bbbb_ovoo"))
      .allocate(tmps.at("0176_bbbb_vvoo"))(tmps.at("0176_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0175_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0172_bbbb_vvoo"))(tmps.at("0172_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0033_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("0169_bbbb_vooo"))(tmps.at("0169_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_1p.at("abab")(ba, ab, la, jb) *
                                             eri.at("abab_oovo")(la, ib, ba, kb))
      .allocate(tmps.at("0168_bbbb_ovoo"))(tmps.at("bin_aabb_vooo")(ba, la, ib, jb) =
                                             eri.at("abab_oovv")(la, ib, ba, cb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("0168_bbbb_ovoo")(ib, ab, jb, kb) =
          tmps.at("bin_aabb_vooo")(ba, la, ib, jb) * t2_1p.at("abab")(ba, ab, la, kb))
      .allocate(tmps.at("0167_bbbb_vooo"))(tmps.at("0167_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_1p.at("bbbb")(bb, ab, jb, lb) *
                                             eri.at("bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("0166_bbbb_ovoo"))(tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) =
                                             eri.at("bbbb_oovv")(ib, lb, cb, bb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("0166_bbbb_ovoo")(ib, ab, jb, kb) =
          tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) * t2_1p.at("bbbb")(bb, ab, kb, lb))
      .allocate(tmps.at("0170_bbbb_ovoo"))(tmps.at("0170_bbbb_ovoo")(ib, ab, jb, kb) =
                                             -1.00 * tmps.at("0166_bbbb_ovoo")(ib, ab, jb, kb))(
        tmps.at("0170_bbbb_ovoo")(ib, ab, jb, kb) += tmps.at("0167_bbbb_vooo")(ab, ib, kb, jb))(
        tmps.at("0170_bbbb_ovoo")(ib, ab, jb, kb) += tmps.at("0168_bbbb_ovoo")(ib, ab, jb, kb))(
        tmps.at("0170_bbbb_ovoo")(ib, ab, jb, kb) += tmps.at("0169_bbbb_vooo")(ab, ib, kb, jb))
      .deallocate(tmps.at("0169_bbbb_vooo"))
      .deallocate(tmps.at("0168_bbbb_ovoo"))
      .deallocate(tmps.at("0167_bbbb_vooo"))
      .deallocate(tmps.at("0166_bbbb_ovoo"))
      .allocate(tmps.at("0171_bbbb_vvoo"))(tmps.at("0171_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0170_bbbb_ovoo")(kb, bb, ib, jb))
      .allocate(tmps.at("0165_bbbb_vvoo"))(tmps.at("0165_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, ib) * tmps.at("0164_bb_vo")(bb, jb))
      .allocate(tmps.at("0159_bbbb_vvoo"))(tmps.at("0159_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("0158_bb_vo")(bb, jb))
      .allocate(tmps.at("0155_bbbb_vvoo"))(tmps.at("0155_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("0154_bb_vo")(bb, jb))
      .allocate(tmps.at("0149_bbbb_vooo"))(tmps.at("0149_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             eri.at("abab_oovo")(la, ib, ba, kb))
      .allocate(tmps.at("0148_bbbb_ovoo"))(tmps.at("bin_aabb_vooo")(ba, la, ib, jb) =
                                             eri.at("abab_oovv")(la, ib, ba, cb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("0148_bbbb_ovoo")(ib, ab, jb, kb) =
          tmps.at("bin_aabb_vooo")(ba, la, ib, jb) * t2.at("abab")(ba, ab, la, kb))
      .allocate(tmps.at("0150_bbbb_ovoo"))(tmps.at("0150_bbbb_ovoo")(ib, ab, jb, kb) =
                                             tmps.at("0149_bbbb_vooo")(ab, ib, kb, jb))(
        tmps.at("0150_bbbb_ovoo")(ib, ab, jb, kb) += tmps.at("0148_bbbb_ovoo")(ib, ab, jb, kb))
      .deallocate(tmps.at("0149_bbbb_vooo"))
      .deallocate(tmps.at("0148_bbbb_ovoo"))
      .allocate(tmps.at("0151_bbbb_vvoo"))(tmps.at("0151_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0150_bbbb_ovoo")(kb, bb, ib, jb))
      .allocate(tmps.at("0146_bbbb_ovoo"))(tmps.at("bin_aabb_vooo")(ba, la, ib, jb) =
                                             eri.at("abab_oovv")(la, ib, ba, cb) *
                                             t1_1p.at("bb")(cb, jb))(
        tmps.at("0146_bbbb_ovoo")(ib, ab, jb, kb) =
          tmps.at("bin_aabb_vooo")(ba, la, ib, jb) * t2.at("abab")(ba, ab, la, kb))
      .allocate(tmps.at("0147_bbbb_vvoo"))(tmps.at("0147_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0146_bbbb_ovoo")(kb, bb, ib, jb))
      .allocate(tmps.at("0144_bbbb_voov"))(tmps.at("0144_bbbb_voov")(ab, ib, jb, bb) =
                                             t2.at("bbbb")(cb, ab, jb, kb) *
                                             eri.at("bbbb_oovv")(kb, ib, cb, bb))
      .allocate(tmps.at("0145_bbbb_vvoo"))(tmps.at("0145_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0144_bbbb_voov")(bb, kb, jb, cb))
      .allocate(tmps.at("0143_bbbb_vvoo"))(tmps.at("0143_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0035_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("0142_bbbb_vvoo"))(tmps.at("0142_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0141_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("0139_bbbb_vooo"))(tmps.at("0139_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(bb, jb) *
                                             eri.at("bbbb_vovo")(ab, ib, bb, kb))
      .allocate(tmps.at("0140_bbbb_vvoo"))(tmps.at("0140_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0139_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0137_baba_voov"))(tmps.at("0137_baba_voov")(ab, ia, jb, ba) =
                                             t2.at("abab")(ca, ab, ka, jb) *
                                             eri.at("aaaa_oovv")(ka, ia, ca, ba))
      .allocate(tmps.at("0138_bbbb_vvoo"))(tmps.at("0138_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0137_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("0135_baab_vovo"))(tmps.at("0135_baab_vovo")(ab, ia, ba, jb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t1.at("bb")(cb, jb))
      .allocate(tmps.at("0136_bbbb_vvoo"))(tmps.at("0136_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0135_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("0133_baab_vovo"))(tmps.at("0133_baab_vovo")(ab, ia, ba, jb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t1_1p.at("bb")(cb, jb))
      .allocate(tmps.at("0134_bbbb_vvoo"))(tmps.at("0134_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0133_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("0131_bbbb_vovo"))(tmps.at("0131_bbbb_vovo")(ab, ib, bb, jb) =
                                             eri.at("bbbb_vovv")(ab, ib, bb, cb) *
                                             t1_1p.at("bb")(cb, jb))
      .allocate(tmps.at("0132_bbbb_vvoo"))(tmps.at("0132_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0131_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("0129_bbbb_vooo"))(tmps.at("0129_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1_1p.at("bb")(bb, jb) *
                                             eri.at("bbbb_vovo")(ab, ib, bb, kb))
      .allocate(tmps.at("0130_bbbb_vvoo"))(tmps.at("0130_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0129_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0125_bbbb_vvoo"))(tmps.at("0125_bbbb_vvoo")(ab, bb, ib, jb) =
                                             dp.at("bb_vo")(ab, ib) * t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("0124_bbbb_vvoo"))(tmps.at("0124_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("baab_vovo")(ab, ka, ca, ib) *
                                             t2_1p.at("abab")(ca, bb, ka, jb))
      .allocate(tmps.at("0123_bbbb_vvoo"))(tmps.at("0123_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_vovo")(ab, kb, cb, ib) *
                                             t2_1p.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("0126_bbbb_vvoo"))(tmps.at("0126_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0124_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0126_bbbb_vvoo")(ab, bb, ib, jb) +=
        tmps.at("0123_bbbb_vvoo")(ab, bb, ib, jb))(tmps.at("0126_bbbb_vvoo")(ab, bb, ib, jb) +=
                                                   2.00 * tmps.at("0125_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0125_bbbb_vvoo"))
      .deallocate(tmps.at("0124_bbbb_vvoo"))
      .deallocate(tmps.at("0123_bbbb_vvoo"))
      .allocate(tmps.at("0177_bbbb_vvoo"))(tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -0.50 * tmps.at("0130_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("0138_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("0176_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("0132_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0165_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("0126_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("0143_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("0140_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("0145_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("0159_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("0171_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("0134_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("0142_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("0151_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("0147_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("0155_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 *
        tmps.at("0136_bbbb_vvoo")(ab, bb, ib, jb))(tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb) +=
                                                   0.50 * tmps.at("0172_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0176_bbbb_vvoo"))
      .deallocate(tmps.at("0172_bbbb_vvoo"))
      .deallocate(tmps.at("0171_bbbb_vvoo"))
      .deallocate(tmps.at("0165_bbbb_vvoo"))
      .deallocate(tmps.at("0159_bbbb_vvoo"))
      .deallocate(tmps.at("0155_bbbb_vvoo"))
      .deallocate(tmps.at("0151_bbbb_vvoo"))
      .deallocate(tmps.at("0147_bbbb_vvoo"))
      .deallocate(tmps.at("0145_bbbb_vvoo"))
      .deallocate(tmps.at("0143_bbbb_vvoo"))
      .deallocate(tmps.at("0142_bbbb_vvoo"))
      .deallocate(tmps.at("0140_bbbb_vvoo"))
      .deallocate(tmps.at("0138_bbbb_vvoo"))
      .deallocate(tmps.at("0136_bbbb_vvoo"))
      .deallocate(tmps.at("0134_bbbb_vvoo"))
      .deallocate(tmps.at("0132_bbbb_vvoo"))
      .deallocate(tmps.at("0130_bbbb_vvoo"))
      .deallocate(tmps.at("0126_bbbb_vvoo"))
      .allocate(tmps.at("0190_bbbb_vvoo"))(tmps.at("0190_bbbb_vvoo")(ab, bb, ib, jb) =
                                             0.50 * tmps.at("0180_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0190_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("0182_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0190_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("0185_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("0190_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0177_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0190_bbbb_vvoo")(ab, bb, ib, jb) +=
        tmps.at("0184_bbbb_vvoo")(ab, bb, ib, jb))(tmps.at("0190_bbbb_vvoo")(ab, bb, ib, jb) +=
                                                   0.50 * tmps.at("0189_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0189_bbbb_vvoo"))
      .deallocate(tmps.at("0185_bbbb_vvoo"))
      .deallocate(tmps.at("0184_bbbb_vvoo"))
      .deallocate(tmps.at("0182_bbbb_vvoo"))
      .deallocate(tmps.at("0180_bbbb_vvoo"))
      .deallocate(tmps.at("0177_bbbb_vvoo"))(r2_1p.at("bbbb")(ab, bb, ib, jb) +=
                                             2.00 * tmps.at("0190_bbbb_vvoo")(bb, ab, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) -= 2.00 * tmps.at("0190_bbbb_vvoo")(bb, ab, jb, ib))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) -= 2.00 * tmps.at("0190_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("0190_bbbb_vvoo")(ab, bb, jb, ib))
      .deallocate(tmps.at("0190_bbbb_vvoo"))
      .allocate(tmps.at("0198_aaaa_oooo"))(tmps.at("0198_aaaa_oooo")(ia, ja, ka, la) =
                                             eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                             t2_1p.at("aaaa")(aa, ba, ka, la))
      .allocate(tmps.at("0206_aaaa_vooo"))(tmps.at("0206_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             tmps.at("0198_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("0207_aaaa_vvoo"))(tmps.at("0207_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0206_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0203_aaaa_oovo"))(tmps.at("0203_aaaa_oovo")(ia, ja, aa, ka) =
                                             eri.at("aaaa_oovv")(ia, ja, ba, aa) *
                                             t1.at("aa")(ba, ka))
      .allocate(tmps.at("0204_aaaa_oooo"))(tmps.at("0204_aaaa_oooo")(ia, ja, ka, la) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0203_aaaa_oovo")(ia, ja, aa, la))
      .allocate(tmps.at("0205_aaaa_vvoo"))(tmps.at("0205_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0204_aaaa_oooo")(la, ka, ia, ja))
      .allocate(tmps.at("0200_aaaa_oooo"))(tmps.at("0200_aaaa_oooo")(ia, ja, ka, la) =
                                             eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                             t2.at("aaaa")(aa, ba, ka, la))
      .allocate(tmps.at("0201_aaaa_vvoo"))(tmps.at("0201_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0200_aaaa_oooo")(la, ka, ia, ja))
      .allocate(tmps.at("0199_aaaa_vvoo"))(tmps.at("0199_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0198_aaaa_oooo")(la, ka, ia, ja))
      .allocate(tmps.at("0196_aaaa_vvoo"))(tmps.at("0196_aaaa_vvoo")(aa, ba, ia, ja) =
                                             scalars.at("0002")() *
                                             t2_1p.at("aaaa")(aa, ba, ia, ja))
      .allocate(tmps.at("0195_aaaa_vvoo"))(tmps.at("0195_aaaa_vvoo")(aa, ba, ia, ja) =
                                             scalars.at("0001")() *
                                             t2_1p.at("aaaa")(aa, ba, ia, ja))
      .allocate(tmps.at("0194_aaaa_vvoo"))(tmps.at("0194_aaaa_vvoo")(aa, ba, ia, ja) =
                                             scalars.at("0013")() *
                                             t2_2p.at("aaaa")(aa, ba, ia, ja))
      .allocate(tmps.at("0193_aaaa_vvoo"))(tmps.at("0193_aaaa_vvoo")(aa, ba, ia, ja) =
                                             scalars.at("0015")() *
                                             t2_2p.at("aaaa")(aa, ba, ia, ja))
      .allocate(tmps.at("0192_aaaa_vvoo"))(tmps.at("0192_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_vvvv")(aa, ba, ca, da) *
                                             t2_1p.at("aaaa")(ca, da, ia, ja))
      .allocate(tmps.at("0191_aaaa_vvoo"))(tmps.at("0191_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_oooo")(ka, la, ia, ja) *
                                             t2_1p.at("aaaa")(aa, ba, la, ka))
      .allocate(tmps.at("0197_aaaa_vvoo"))(tmps.at("0197_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0192_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0197_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 * tmps.at("0195_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0197_aaaa_vvoo")(aa, ba, ia, ja) -=
        4.00 * tmps.at("0194_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0197_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0191_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0197_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 *
        tmps.at("0196_aaaa_vvoo")(aa, ba, ia, ja))(tmps.at("0197_aaaa_vvoo")(aa, ba, ia, ja) -=
                                                   4.00 * tmps.at("0193_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0196_aaaa_vvoo"))
      .deallocate(tmps.at("0195_aaaa_vvoo"))
      .deallocate(tmps.at("0194_aaaa_vvoo"))
      .deallocate(tmps.at("0193_aaaa_vvoo"))
      .deallocate(tmps.at("0192_aaaa_vvoo"))
      .deallocate(tmps.at("0191_aaaa_vvoo"))
      .allocate(tmps.at("0202_aaaa_vvoo"))(tmps.at("0202_aaaa_vvoo")(aa, ba, ia, ja) =
                                             0.50 * tmps.at("0199_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0202_aaaa_vvoo")(aa, ba, ia, ja) +=
        tmps.at("0197_aaaa_vvoo")(aa, ba, ia, ja))(tmps.at("0202_aaaa_vvoo")(aa, ba, ia, ja) +=
                                                   0.50 * tmps.at("0201_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0201_aaaa_vvoo"))
      .deallocate(tmps.at("0199_aaaa_vvoo"))
      .deallocate(tmps.at("0197_aaaa_vvoo"))
      .allocate(tmps.at("0208_aaaa_vvoo"))(tmps.at("0208_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0205_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0208_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0207_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0208_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0202_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0207_aaaa_vvoo"))
      .deallocate(tmps.at("0205_aaaa_vvoo"))
      .deallocate(tmps.at("0202_aaaa_vvoo"))(r2_1p.at("aaaa")(aa, ba, ia, ja) -=
                                             0.50 * tmps.at("0208_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0208_aaaa_vvoo"))
      .allocate(tmps.at("0224_aa_ov"))(tmps.at("0224_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ja, ia, ba, aa) * t1.at("aa")(ba, ja))
      .allocate(tmps.at("0247_aa_oo"))(tmps.at("0247_aa_oo")(ia, ja) =
                                         t1.at("aa")(aa, ja) * tmps.at("0224_aa_ov")(ia, aa))
      .allocate(tmps.at("0232_aa_ov"))(tmps.at("0232_aa_ov")(ia, aa) =
                                         eri.at("abab_oovv")(ia, jb, aa, bb) * t1.at("bb")(bb, jb))
      .allocate(tmps.at("0246_aa_oo"))(tmps.at("0246_aa_oo")(ia, ja) =
                                         t1.at("aa")(aa, ja) * tmps.at("0232_aa_ov")(ia, aa))
      .allocate(tmps.at("0248_aa_oo"))(tmps.at("0248_aa_oo")(ia, ja) =
                                         tmps.at("0246_aa_oo")(ia, ja))(
        tmps.at("0248_aa_oo")(ia, ja) += tmps.at("0247_aa_oo")(ia, ja))
      .deallocate(tmps.at("0247_aa_oo"))
      .deallocate(tmps.at("0246_aa_oo"))
      .allocate(tmps.at("0249_aa_vo"))(tmps.at("0249_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0248_aa_oo")(ja, ia))
      .allocate(tmps.at("0243_aa_oo"))(tmps.at("0243_aa_oo")(ia, ja) =
                                         f.at("aa_ov")(ia, aa) * t1.at("aa")(aa, ja))
      .allocate(tmps.at("0244_aa_vo"))(tmps.at("0244_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0243_aa_oo")(ja, ia))
      .allocate(tmps.at("0240_aa_oo"))(tmps.at("0240_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovo")(ka, ia, aa, ja) * t1.at("aa")(aa, ka))
      .allocate(tmps.at("0239_aa_oo"))(tmps.at("0239_aa_oo")(ia, ja) =
                                         eri.at("abba_oovo")(ia, kb, ab, ja) * t1.at("bb")(ab, kb))
      .allocate(tmps.at("0241_aa_oo"))(tmps.at("0241_aa_oo")(ia, ja) =
                                         -1.00 * tmps.at("0239_aa_oo")(ia, ja))(
        tmps.at("0241_aa_oo")(ia, ja) += tmps.at("0240_aa_oo")(ia, ja))
      .deallocate(tmps.at("0240_aa_oo"))
      .deallocate(tmps.at("0239_aa_oo"))
      .allocate(tmps.at("0242_aa_vo"))(tmps.at("0242_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0241_aa_oo")(ja, ia))
      .allocate(tmps.at("0237_bb_ov"))(tmps.at("0237_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(jb, ib, bb, ab) * t1.at("bb")(bb, jb))
      .allocate(tmps.at("0238_aa_vo"))(tmps.at("0238_aa_vo")(aa, ia) =
                                         t2.at("abab")(aa, bb, ia, jb) *
                                         tmps.at("0237_bb_ov")(jb, bb))
      .allocate(tmps.at("0236_aa_vo"))(tmps.at("0236_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0037_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0234_aa_oo"))(tmps.at("0234_aa_oo")(ia, ja) =
                                         eri.at("abab_oovv")(ia, kb, aa, bb) *
                                         t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0235_aa_vo"))(tmps.at("0235_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0234_aa_oo")(ja, ia))
      .allocate(tmps.at("0233_aa_vo"))(tmps.at("0233_aa_vo")(aa, ia) =
                                         t2.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0232_aa_ov")(ja, ba))
      .allocate(tmps.at("0230_abba_vovo"))(tmps.at("0230_abba_vovo")(aa, ib, bb, ja) =
                                             eri.at("abab_vovv")(aa, ib, ca, bb) *
                                             t1.at("aa")(ca, ja))
      .allocate(tmps.at("0231_aa_vo"))(tmps.at("0231_aa_vo")(aa, ia) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0230_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0229_aa_vo"))(tmps.at("0229_aa_vo")(aa, ia) =
                                         t2.at("aaaa")(ba, aa, ja, ka) *
                                         tmps.at("0203_aaaa_oovo")(ka, ja, ba, ia))
      .allocate(tmps.at("0209_aa_oo"))(tmps.at("0209_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovv")(ia, ka, aa, ba) *
                                         t2.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0228_aa_vo"))(tmps.at("0228_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0209_aa_oo")(ja, ia))
      .deallocate(tmps.at("0209_aa_oo"))
      .allocate(tmps.at("0226_aa_vv"))(tmps.at("0226_aa_vv")(aa, ba) =
                                         eri.at("aaaa_vovv")(aa, ia, ca, ba) * t1.at("aa")(ca, ia))
      .allocate(tmps.at("0227_aa_vo"))(tmps.at("0227_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ia) * tmps.at("0226_aa_vv")(aa, ba))
      .allocate(tmps.at("0225_aa_vo"))(tmps.at("0225_aa_vo")(aa, ia) =
                                         t2.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0224_aa_ov")(ja, ba))
      .allocate(tmps.at("0223_aa_vo"))(tmps.at("0223_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ia) * tmps.at("0041_aa_vv")(aa, ba))
      .allocate(tmps.at("0221_aa_vo"))(tmps.at("0221_aa_vo")(aa, ia) =
                                         f.at("aa_vv")(aa, ba) * t1.at("aa")(ba, ia))
      .allocate(tmps.at("0220_aa_vo"))(tmps.at("0220_aa_vo")(aa, ia) =
                                         eri.at("abba_oovo")(ja, kb, bb, ia) *
                                         t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0219_aa_vo"))(tmps.at("0219_aa_vo")(aa, ia) =
                                         scalars.at("0015")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0218_aa_vo"))(tmps.at("0218_aa_vo")(aa, ia) =
                                         scalars.at("0013")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0217_aa_vo"))(tmps.at("0217_aa_vo")(aa, ia) =
                                         eri.at("abba_vovo")(aa, jb, bb, ia) * t1.at("bb")(bb, jb))
      .allocate(tmps.at("0216_aa_vo"))(tmps.at("0216_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovv")(aa, ja, ba, ca) *
                                         t2.at("aaaa")(ba, ca, ia, ja))
      .allocate(tmps.at("0215_aa_vo"))(tmps.at("0215_aa_vo")(aa, ia) =
                                         f.at("aa_oo")(ja, ia) * t1.at("aa")(aa, ja))
      .allocate(tmps.at("0214_aa_vo"))(tmps.at("0214_aa_vo")(aa, ia) =
                                         eri.at("aaaa_oovo")(ja, ka, ba, ia) *
                                         t2.at("aaaa")(ba, aa, ka, ja))
      .allocate(tmps.at("0213_aa_vo"))(tmps.at("0213_aa_vo")(aa, ia) =
                                         f.at("bb_ov")(jb, bb) * t2.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0212_aa_vo"))(tmps.at("0212_aa_vo")(aa, ia) =
                                         eri.at("abab_vovv")(aa, jb, ba, cb) *
                                         t2.at("abab")(ba, cb, ia, jb))
      .allocate(tmps.at("0211_aa_vo"))(tmps.at("0211_aa_vo")(aa, ia) =
                                         f.at("aa_ov")(ja, ba) * t2.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0210_aa_vo"))(tmps.at("0210_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovo")(aa, ja, ba, ia) * t1.at("aa")(ba, ja))
      .allocate(tmps.at("0222_aa_vo"))(tmps.at("0222_aa_vo")(aa, ia) =
                                         -0.50 * tmps.at("0214_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) -= 0.50 * tmps.at("0216_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) -= tmps.at("0218_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) -= tmps.at("0221_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) += tmps.at("0210_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) += tmps.at("0211_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) += tmps.at("0215_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) -= tmps.at("0220_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) -= tmps.at("0212_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) -= tmps.at("0213_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) += tmps.at("0217_aa_vo")(aa, ia))(
        tmps.at("0222_aa_vo")(aa, ia) -= tmps.at("0219_aa_vo")(aa, ia))
      .deallocate(tmps.at("0221_aa_vo"))
      .deallocate(tmps.at("0220_aa_vo"))
      .deallocate(tmps.at("0219_aa_vo"))
      .deallocate(tmps.at("0218_aa_vo"))
      .deallocate(tmps.at("0217_aa_vo"))
      .deallocate(tmps.at("0216_aa_vo"))
      .deallocate(tmps.at("0215_aa_vo"))
      .deallocate(tmps.at("0214_aa_vo"))
      .deallocate(tmps.at("0213_aa_vo"))
      .deallocate(tmps.at("0212_aa_vo"))
      .deallocate(tmps.at("0211_aa_vo"))
      .deallocate(tmps.at("0210_aa_vo"));
  }
}

template void exachem::cc::qed_ccsd_os::resid_1<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);