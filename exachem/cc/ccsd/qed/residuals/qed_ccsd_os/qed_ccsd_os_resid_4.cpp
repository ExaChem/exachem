/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_os_resid_4.hpp"

template<typename T>
void exachem::cc::qed_ccsd_os::resid_4(
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
    sch
      .allocate(tmps.at("0875_abab_oooo"))(tmps.at("0875_abab_oooo")(ia, jb, ka, lb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0286_abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("0876_abab_vvoo"))(tmps.at("0876_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0875_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0816_abba_voov"))(tmps.at("0816_abba_voov")(aa, ib, jb, ba) =
                                             t2.at("abab")(aa, cb, ka, jb) *
                                             eri.at("abab_oovv")(ka, ib, ba, cb))
      .allocate(tmps.at("0873_abab_vooo"))(tmps.at("0873_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0816_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("0874_baab_vvoo"))(tmps.at("0874_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0873_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0872_abab_vvoo"))(tmps.at("0872_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0183_bb_vo")(bb, jb))
      .allocate(tmps.at("0794_baba_vooo"))(tmps.at("0794_baba_vooo")(ab, ia, jb, ka) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0203_aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("0871_abba_vvoo"))(tmps.at("0871_abba_vvoo")(aa, bb, ib, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0794_baba_vooo")(bb, ka, ib, ja))
      .deallocate(tmps.at("0794_baba_vooo"))
      .allocate(tmps.at("0870_baba_vvoo"))(tmps.at("0870_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("0713_aa_vo")(ba, ja))
      .allocate(tmps.at("0771_baba_vooo"))(tmps.at("0771_baba_vooo")(ab, ia, jb, ka) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             eri.at("aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("0868_abba_vvoo"))(tmps.at("0868_abba_vvoo")(aa, bb, ib, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0771_baba_vooo")(bb, ka, ib, ja))
      .deallocate(tmps.at("0771_baba_vooo"))
      .allocate(tmps.at("0867_abba_vvoo"))(tmps.at("0867_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0241_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0866_baab_vvoo"))(tmps.at("0866_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0374_aa_vv")(ba, ca) *
                                             t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0865_abab_vvoo"))(tmps.at("0865_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0500_bb_vv")(bb, cb))
      .allocate(tmps.at("0864_abab_vvoo"))(tmps.at("0864_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0271_bb_oo")(kb, jb))
      .allocate(tmps.at("0861_baab_vooo"))(tmps.at("0861_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("bb")(ab, lb) *
                                             eri.at("abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0860_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             t1.at("aa")(ba, ja) *
                                             eri.at("abab_oovo")(ia, lb, ba, kb))(
        tmps.at("0860_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1.at("bb")(ab, lb))
      .allocate(tmps.at("0859_baab_vooo"))(tmps.at("0859_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, lb) *
                                             eri.at("abab_oovo")(ia, lb, ba, kb))
      .allocate(tmps.at("0858_abba_ovoo"))(tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) =
                                             eri.at("abab_oovv")(ia, lb, ba, cb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("0858_abba_ovoo")(ia, ab, jb, ka) =
          tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) * t2.at("abab")(ba, ab, ka, lb))
      .allocate(tmps.at("0857_baba_vooo"))(tmps.at("0857_baba_vooo")(ab, ia, jb, ka) =
                                             t2.at("bbbb")(bb, ab, jb, lb) *
                                             eri.at("abba_oovo")(ia, lb, bb, ka))
      .allocate(tmps.at("0862_baba_vooo"))(tmps.at("0862_baba_vooo")(ab, ia, jb, ka) =
                                             -1.00 * tmps.at("0858_abba_ovoo")(ia, ab, jb, ka))(
        tmps.at("0862_baba_vooo")(ab, ia, jb, ka) -= tmps.at("0859_baab_vooo")(ab, ia, ka, jb))(
        tmps.at("0862_baba_vooo")(ab, ia, jb, ka) -= tmps.at("0860_abab_ovoo")(ia, ab, ka, jb))(
        tmps.at("0862_baba_vooo")(ab, ia, jb, ka) -= tmps.at("0861_baab_vooo")(ab, ia, ka, jb))(
        tmps.at("0862_baba_vooo")(ab, ia, jb, ka) += tmps.at("0857_baba_vooo")(ab, ia, jb, ka))
      .deallocate(tmps.at("0861_baab_vooo"))
      .deallocate(tmps.at("0860_abab_ovoo"))
      .deallocate(tmps.at("0859_baab_vooo"))
      .deallocate(tmps.at("0858_abba_ovoo"))
      .deallocate(tmps.at("0857_baba_vooo"))
      .allocate(tmps.at("0863_abba_vvoo"))(tmps.at("0863_abba_vvoo")(aa, bb, ib, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0862_baba_vooo")(bb, ka, ib, ja))
      .allocate(tmps.at("0854_abab_vooo"))(tmps.at("0854_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             eri.at("abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("0853_abab_vooo"))(tmps.at("0853_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abba_vovo")(aa, ib, bb, ja) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0852_abab_vooo"))(tmps.at("0852_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, kb) * f.at("bb_ov")(ib, bb))
      .allocate(tmps.at("0851_abab_vooo"))(tmps.at("0851_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t2.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0855_abab_vooo"))(tmps.at("0855_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("0851_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0855_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0852_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0855_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0854_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0855_abab_vooo")(aa, ib, ja, kb) += tmps.at("0853_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("0854_abab_vooo"))
      .deallocate(tmps.at("0853_abab_vooo"))
      .deallocate(tmps.at("0852_abab_vooo"))
      .deallocate(tmps.at("0851_abab_vooo"))
      .allocate(tmps.at("0856_baab_vvoo"))(tmps.at("0856_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0855_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("0850_abab_vvoo"))(tmps.at("0850_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0164_bb_vo")(bb, jb))
      .allocate(tmps.at("0849_baba_vvoo"))(tmps.at("0849_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0710_aa_vo")(ba, ja) * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("0846_abab_vooo"))(tmps.at("0846_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("aaaa")(ba, aa, ja, la) *
                                             eri.at("abab_oovo")(la, ib, ba, kb))
      .allocate(tmps.at("0845_baba_ovoo"))(tmps.at("bin_aabb_vooo")(ba, la, ib, jb) =
                                             eri.at("abab_oovv")(la, ib, ba, cb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("0845_baba_ovoo")(ib, aa, jb, ka) =
          tmps.at("bin_aabb_vooo")(ba, la, ib, jb) * t2.at("aaaa")(ba, aa, ka, la))
      .allocate(tmps.at("0844_abba_vooo"))(tmps.at("0844_abba_vooo")(aa, ib, jb, ka) =
                                             t2.at("abab")(aa, bb, la, jb) *
                                             eri.at("abba_oovo")(la, ib, bb, ka))
      .allocate(tmps.at("0847_baba_ovoo"))(tmps.at("0847_baba_ovoo")(ib, aa, jb, ka) =
                                             tmps.at("0846_abab_vooo")(aa, ib, ka, jb))(
        tmps.at("0847_baba_ovoo")(ib, aa, jb, ka) -= tmps.at("0844_abba_vooo")(aa, ib, jb, ka))(
        tmps.at("0847_baba_ovoo")(ib, aa, jb, ka) += tmps.at("0845_baba_ovoo")(ib, aa, jb, ka))
      .deallocate(tmps.at("0846_abab_vooo"))
      .deallocate(tmps.at("0845_baba_ovoo"))
      .deallocate(tmps.at("0844_abba_vooo"))
      .allocate(tmps.at("0848_baba_vvoo"))(tmps.at("0848_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0847_baba_ovoo")(kb, ba, ib, ja) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("0841_baab_vooo"))(tmps.at("0841_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             eri.at("baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("0840_baab_vooo"))(tmps.at("0840_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, kb) * f.at("aa_ov")(ia, ba))
      .allocate(tmps.at("0839_baab_vooo"))(tmps.at("0839_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baba_vovo")(ab, ia, bb, ja) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0838_baab_vooo"))(tmps.at("0838_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t2.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0842_baab_vooo"))(tmps.at("0842_baab_vooo")(ab, ia, ja, kb) =
                                             -1.00 * tmps.at("0838_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0842_baab_vooo")(ab, ia, ja, kb) += tmps.at("0839_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0842_baab_vooo")(ab, ia, ja, kb) -= tmps.at("0841_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0842_baab_vooo")(ab, ia, ja, kb) += tmps.at("0840_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("0841_baab_vooo"))
      .deallocate(tmps.at("0840_baab_vooo"))
      .deallocate(tmps.at("0839_baab_vooo"))
      .deallocate(tmps.at("0838_baab_vooo"))
      .allocate(tmps.at("0843_abab_vvoo"))(tmps.at("0843_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0842_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0835_abab_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("aaaa_oovv")(la, ia, ca, ba) * t1.at("aa")(ca, la))(
        tmps.at("0835_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aa_vo")(ba, ia) * t2.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0834_abab_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("abab_oovv")(ia, lb, ba, cb) * t1.at("bb")(cb, lb))(
        tmps.at("0834_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aa_vo")(ba, ia) * t2.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0836_abab_ovoo"))(tmps.at("0836_abab_ovoo")(ia, ab, ja, kb) =
                                             tmps.at("0834_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0836_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0835_abab_ovoo")(ia, ab, ja, kb))
      .deallocate(tmps.at("0835_abab_ovoo"))
      .deallocate(tmps.at("0834_abab_ovoo"))
      .allocate(tmps.at("0837_abab_vvoo"))(tmps.at("0837_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0836_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0833_abba_vvoo"))(tmps.at("0833_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0234_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0832_abab_vvoo"))(tmps.at("0832_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0831_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0828_baba_vvoo"))(tmps.at("0828_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0230_abba_vovo")(ba, kb, cb, ja) *
                                             t2.at("bbbb")(cb, ab, ib, kb))
      .allocate(tmps.at("0825_baab_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("abab_oovv")(la, ib, ca, bb) * t1.at("aa")(ca, la))(
        tmps.at("0825_baab_ovoo")(ib, aa, ja, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0824_baab_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("bbbb_oovv")(lb, ib, cb, bb) * t1.at("bb")(cb, lb))(
        tmps.at("0824_baab_ovoo")(ib, aa, ja, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0826_baab_ovoo"))(tmps.at("0826_baab_ovoo")(ib, aa, ja, kb) =
                                             tmps.at("0824_baab_ovoo")(ib, aa, ja, kb))(
        tmps.at("0826_baab_ovoo")(ib, aa, ja, kb) += tmps.at("0825_baab_ovoo")(ib, aa, ja, kb))
      .deallocate(tmps.at("0825_baab_ovoo"))
      .deallocate(tmps.at("0824_baab_ovoo"))
      .allocate(tmps.at("0827_baab_vvoo"))(tmps.at("0827_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0826_baab_ovoo")(kb, ba, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("0823_abab_vvoo"))(tmps.at("0823_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0281_bb_oo")(kb, jb))
      .allocate(tmps.at("0821_abba_vvvo"))(tmps.at("0821_abba_vvvo")(aa, bb, cb, ia) =
                                             eri.at("abab_vvvv")(aa, bb, da, cb) *
                                             t1.at("aa")(da, ia))
      .allocate(tmps.at("0822_abba_vvoo"))(tmps.at("0822_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0821_abba_vvvo")(aa, bb, cb, ja) *
                                             t1.at("bb")(cb, ib))
      .allocate(tmps.at("0819_abab_oooo"))(tmps.at("0819_abab_oooo")(ia, jb, ka, lb) =
                                             t1.at("aa")(aa, ka) *
                                             eri.at("abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("0820_abab_vvoo"))(tmps.at("0820_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0819_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0818_abab_vvoo"))(tmps.at("0818_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0135_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("0817_baab_vvoo"))(tmps.at("0817_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0816_abba_voov")(ba, kb, jb, ca) *
                                             t2.at("abab")(ca, ab, ia, kb))
      .allocate(tmps.at("0770_abab_vooo"))(tmps.at("0770_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, lb) *
                                             eri.at("bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("0815_baab_vvoo"))(tmps.at("0815_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0770_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .deallocate(tmps.at("0770_abab_vooo"))
      .allocate(tmps.at("0814_abba_vvoo"))(tmps.at("0814_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0243_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0813_baab_vvoo"))(tmps.at("0813_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0226_aa_vv")(ba, ca) *
                                             t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0812_baab_vvoo"))(tmps.at("0812_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0811_abab_vovo")(ba, kb, ca, jb) *
                                             t2.at("abab")(ca, ab, ia, kb))
      .allocate(tmps.at("0810_abab_vvoo"))(tmps.at("0810_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0033_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("0809_abab_vvoo"))(tmps.at("0809_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0299_bb_vv")(bb, cb))
      .allocate(tmps.at("0808_baba_vvoo"))(tmps.at("0808_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0376_abab_voov")(ba, kb, ja, cb) *
                                             t2.at("bbbb")(cb, ab, ib, kb))
      .allocate(tmps.at("0807_baba_vvoo"))(tmps.at("0807_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0037_aaaa_voov")(ba, ka, ja, ca) *
                                             t2.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("0806_abab_vvoo"))(tmps.at("0806_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0276_bb_oo")(kb, jb))
      .allocate(tmps.at("0805_abab_vvoo"))(tmps.at("0805_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0141_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("0804_baba_vvoo"))(tmps.at("0804_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0363_aaaa_vovo")(ba, ka, ca, ja) *
                                             t2.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("0803_abab_vvoo"))(tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0297_bb_vv")(bb, cb))
      .allocate(tmps.at("0802_baba_vvoo"))(tmps.at("0802_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0346_aaaa_voov")(ba, ka, ja, ca) *
                                             t2.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("0801_abab_vvoo"))(tmps.at("0801_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0505_bb_vv")(bb, cb))
      .allocate(tmps.at("0799_baba_vovo"))(tmps.at("0799_baba_vovo")(ab, ia, bb, ja) =
                                             eri.at("baab_vovv")(ab, ia, ca, bb) *
                                             t1.at("aa")(ca, ja))
      .allocate(tmps.at("0800_abba_vvoo"))(tmps.at("0800_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, cb, ka, ib) *
                                             tmps.at("0799_baba_vovo")(bb, ka, cb, ja))
      .allocate(tmps.at("0798_baab_vvoo"))(tmps.at("0798_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0664_aa_vv")(ba, ca) *
                                             t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0797_baab_vvoo"))(tmps.at("0797_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0041_aa_vv")(ba, ca) *
                                             t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0796_abba_vvoo"))(tmps.at("0796_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0390_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0795_abab_vvoo"))(tmps.at("0795_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0273_bb_oo")(kb, jb))
      .allocate(tmps.at("0791_abab_vvoo"))(tmps.at("0791_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) * f.at("bb_vv")(bb, cb))
      .allocate(tmps.at("0790_abab_vvoo"))(tmps.at("0790_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_vo")(aa, ia) * t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0789_abab_vvoo"))(tmps.at("0789_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vvvv")(aa, bb, ca, db) *
                                             t2.at("abab")(ca, db, ia, jb))
      .allocate(tmps.at("0788_abab_vvoo"))(tmps.at("0788_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_oooo")(ka, lb, ia, jb) *
                                             t2.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0787_abba_vvoo"))(tmps.at("0787_abba_vvoo")(aa, bb, ib, ja) =
                                             eri.at("abab_vovo")(aa, kb, ca, ib) *
                                             t2.at("abab")(ca, bb, ja, kb))
      .allocate(tmps.at("0786_abab_vvoo"))(tmps.at("0786_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0015")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0785_abab_vvoo"))(tmps.at("0785_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vooo")(aa, kb, ia, jb) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0784_abab_vvoo"))(tmps.at("0784_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0013")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0783_abab_vvoo"))(tmps.at("0783_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vvvo")(aa, bb, cb, ia) *
                                             t1.at("bb")(cb, jb))
      .allocate(tmps.at("0782_abab_vvoo"))(tmps.at("0782_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, kb) *
                                             eri.at("bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("0781_abab_vvoo"))(tmps.at("0781_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_oo")(ka, ia) * t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0780_abab_vvoo"))(tmps.at("0780_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             eri.at("baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0779_abab_vvoo"))(tmps.at("0779_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("aaaa_vovo")(aa, ka, ca, ia) *
                                             t2.at("abab")(ca, bb, ka, jb))
      .allocate(tmps.at("0778_abab_vvoo"))(tmps.at("0778_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) * f.at("bb_oo")(kb, jb))
      .allocate(tmps.at("0777_abab_vvoo"))(tmps.at("0777_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vovo")(aa, kb, cb, ia) *
                                             t2.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("0776_abab_vvoo"))(tmps.at("0776_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_vv")(aa, ca) * t2.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0775_abba_vvoo"))(tmps.at("0775_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, cb, ka, ib) *
                                             eri.at("baba_vovo")(bb, ka, cb, ja))
      .allocate(tmps.at("0774_abab_vvoo"))(tmps.at("0774_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(ca, ia) *
                                             eri.at("abab_vvvo")(aa, bb, ca, jb))
      .allocate(tmps.at("0773_abab_vvoo"))(tmps.at("0773_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             eri.at("baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("0772_abab_vvoo"))(tmps.at("0772_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * dp.at("bb_vo")(bb, jb))
      .allocate(tmps.at("0792_abab_vvoo"))(tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0772_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0777_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0783_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0787_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0786_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0791_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0788_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0775_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0779_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0780_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0781_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0782_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0778_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0773_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0789_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0790_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0776_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0774_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0785_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0792_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0784_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0791_abab_vvoo"))
      .deallocate(tmps.at("0790_abab_vvoo"))
      .deallocate(tmps.at("0789_abab_vvoo"))
      .deallocate(tmps.at("0788_abab_vvoo"))
      .deallocate(tmps.at("0787_abba_vvoo"))
      .deallocate(tmps.at("0786_abab_vvoo"))
      .deallocate(tmps.at("0785_abab_vvoo"))
      .deallocate(tmps.at("0784_abab_vvoo"))
      .deallocate(tmps.at("0783_abab_vvoo"))
      .deallocate(tmps.at("0782_abab_vvoo"))
      .deallocate(tmps.at("0781_abab_vvoo"))
      .deallocate(tmps.at("0780_abab_vvoo"))
      .deallocate(tmps.at("0779_abab_vvoo"))
      .deallocate(tmps.at("0778_abab_vvoo"))
      .deallocate(tmps.at("0777_abab_vvoo"))
      .deallocate(tmps.at("0776_abab_vvoo"))
      .deallocate(tmps.at("0775_abba_vvoo"))
      .deallocate(tmps.at("0774_abab_vvoo"))
      .deallocate(tmps.at("0773_abab_vvoo"))
      .deallocate(tmps.at("0772_abab_vvoo"))
      .allocate(tmps.at("0869_abba_vvoo"))(tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) =
                                             -0.50 * tmps.at("0806_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0868_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0814_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0827_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -=
        0.50 * tmps.at("0866_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0795_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0804_baba_vvoo")(bb, aa, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0864_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0801_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0802_baba_vvoo")(bb, aa, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0863_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0808_baba_vvoo")(bb, aa, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0856_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0828_baba_vvoo")(bb, aa, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0849_baba_vvoo")(bb, aa, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0812_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0815_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0809_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0818_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0850_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0817_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0832_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0848_baba_vvoo")(bb, aa, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0797_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0837_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0803_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0798_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0807_baba_vvoo")(bb, aa, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0792_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0820_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0823_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) +=
        0.50 * tmps.at("0865_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0843_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0822_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -=
        0.50 * tmps.at("0796_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0800_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0833_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0810_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0813_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0867_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("0869_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("0805_abab_vvoo")(aa, bb, ja, ib))
      .deallocate(tmps.at("0868_abba_vvoo"))
      .deallocate(tmps.at("0867_abba_vvoo"))
      .deallocate(tmps.at("0866_baab_vvoo"))
      .deallocate(tmps.at("0865_abab_vvoo"))
      .deallocate(tmps.at("0864_abab_vvoo"))
      .deallocate(tmps.at("0863_abba_vvoo"))
      .deallocate(tmps.at("0856_baab_vvoo"))
      .deallocate(tmps.at("0850_abab_vvoo"))
      .deallocate(tmps.at("0849_baba_vvoo"))
      .deallocate(tmps.at("0848_baba_vvoo"))
      .deallocate(tmps.at("0843_abab_vvoo"))
      .deallocate(tmps.at("0837_abab_vvoo"))
      .deallocate(tmps.at("0833_abba_vvoo"))
      .deallocate(tmps.at("0832_abab_vvoo"))
      .deallocate(tmps.at("0828_baba_vvoo"))
      .deallocate(tmps.at("0827_baab_vvoo"))
      .deallocate(tmps.at("0823_abab_vvoo"))
      .deallocate(tmps.at("0822_abba_vvoo"))
      .deallocate(tmps.at("0820_abab_vvoo"))
      .deallocate(tmps.at("0818_abab_vvoo"))
      .deallocate(tmps.at("0817_baab_vvoo"))
      .deallocate(tmps.at("0815_baab_vvoo"))
      .deallocate(tmps.at("0814_abba_vvoo"))
      .deallocate(tmps.at("0813_baab_vvoo"))
      .deallocate(tmps.at("0812_baab_vvoo"))
      .deallocate(tmps.at("0810_abab_vvoo"))
      .deallocate(tmps.at("0809_abab_vvoo"))
      .deallocate(tmps.at("0808_baba_vvoo"))
      .deallocate(tmps.at("0807_baba_vvoo"))
      .deallocate(tmps.at("0806_abab_vvoo"))
      .deallocate(tmps.at("0805_abab_vvoo"))
      .deallocate(tmps.at("0804_baba_vvoo"))
      .deallocate(tmps.at("0803_abab_vvoo"))
      .deallocate(tmps.at("0802_baba_vvoo"))
      .deallocate(tmps.at("0801_abab_vvoo"))
      .deallocate(tmps.at("0800_abba_vvoo"))
      .deallocate(tmps.at("0798_baab_vvoo"))
      .deallocate(tmps.at("0797_baab_vvoo"))
      .deallocate(tmps.at("0796_abba_vvoo"))
      .deallocate(tmps.at("0795_abab_vvoo"))
      .deallocate(tmps.at("0792_abab_vvoo"))
      .allocate(tmps.at("0889_baba_vvoo"))(tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) =
                                             -1.00 * tmps.at("0874_baab_vvoo")(ab, ba, ja, ib))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0876_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0872_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0880_baab_vvoo")(ab, ba, ja, ib))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0870_baba_vvoo")(ab, ba, ib, ja))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0878_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0886_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0888_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0885_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0869_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0871_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("0889_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0877_baab_vvoo")(ab, ba, ja, ib))
      .deallocate(tmps.at("0888_abab_vvoo"))
      .deallocate(tmps.at("0886_abba_vvoo"))
      .deallocate(tmps.at("0885_abab_vvoo"))
      .deallocate(tmps.at("0880_baab_vvoo"))
      .deallocate(tmps.at("0878_abab_vvoo"))
      .deallocate(tmps.at("0877_baab_vvoo"))
      .deallocate(tmps.at("0876_abab_vvoo"))
      .deallocate(tmps.at("0874_baab_vvoo"))
      .deallocate(tmps.at("0872_abab_vvoo"))
      .deallocate(tmps.at("0871_abba_vvoo"))
      .deallocate(tmps.at("0870_baba_vvoo"))
      .deallocate(tmps.at("0869_abba_vvoo"))(r2.at("abab")(aa, bb, ia, jb) -=
                                             tmps.at("0889_baba_vvoo")(bb, aa, jb, ia))
      .deallocate(tmps.at("0889_baba_vvoo"))
      .allocate(tmps.at("0975_abab_oooo"))(tmps.at("0975_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abba_oovo")(ia, jb, ab, ka) *
                                             t1_1p.at("bb")(ab, lb))
      .allocate(tmps.at("0974_abab_oooo"))(tmps.at("0974_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t2_1p.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0976_abab_oooo"))(tmps.at("0976_abab_oooo")(ia, jb, ka, lb) =
                                             -1.00 * tmps.at("0974_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("0976_abab_oooo")(ia, jb, ka, lb) += tmps.at("0975_abab_oooo")(ia, jb, ka, lb))
      .deallocate(tmps.at("0975_abab_oooo"))
      .deallocate(tmps.at("0974_abab_oooo"))
      .allocate(tmps.at("1097_baab_vooo"))(tmps.at("1097_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0976_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1096_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0593_abab_oovo")(ia, lb, ba, kb))(
        tmps.at("1096_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1.at("bb")(ab, lb))
      .allocate(tmps.at("1095_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0286_abab_oovo")(ia, lb, ba, kb))(
        tmps.at("1095_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1.at("bb")(ab, lb))
      .allocate(tmps.at("1094_baab_vooo"))(tmps.at("1094_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0033_baba_voov")(ab, ia, kb, ba))
      .allocate(tmps.at("1098_baab_vooo"))(tmps.at("1098_baab_vooo")(ab, ia, ja, kb) =
                                             -1.00 * tmps.at("1097_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1098_baab_vooo")(ab, ia, ja, kb) += tmps.at("1094_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1098_baab_vooo")(ab, ia, ja, kb) += tmps.at("1095_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("1098_baab_vooo")(ab, ia, ja, kb) += tmps.at("1096_abab_ovoo")(ia, ab, ja, kb))
      .deallocate(tmps.at("1097_baab_vooo"))
      .deallocate(tmps.at("1096_abab_ovoo"))
      .deallocate(tmps.at("1095_abab_ovoo"))
      .deallocate(tmps.at("1094_baab_vooo"))
      .allocate(tmps.at("1099_abab_vvoo"))(tmps.at("1099_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1098_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("1093_abab_vvoo"))(tmps.at("1093_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0884_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("1090_baab_vooo"))(tmps.at("1090_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0831_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1089_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0286_abab_oovo")(ia, lb, ba, kb))(
        tmps.at("1089_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1_1p.at("bb")(ab, lb))
      .allocate(tmps.at("1088_baab_vooo"))(tmps.at("1088_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0035_baba_voov")(ab, ia, kb, ba))
      .allocate(tmps.at("1087_baab_vooo"))(tmps.at("1087_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0133_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("1086_baab_vooo"))(tmps.at("1086_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0135_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("1091_baab_vooo"))(tmps.at("1091_baab_vooo")(ab, ia, ja, kb) =
                                             tmps.at("1086_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1091_baab_vooo")(ab, ia, ja, kb) += tmps.at("1088_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1091_baab_vooo")(ab, ia, ja, kb) += tmps.at("1087_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1091_baab_vooo")(ab, ia, ja, kb) += tmps.at("1089_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("1091_baab_vooo")(ab, ia, ja, kb) += tmps.at("1090_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("1090_baab_vooo"))
      .deallocate(tmps.at("1089_abab_ovoo"))
      .deallocate(tmps.at("1088_baab_vooo"))
      .deallocate(tmps.at("1087_baab_vooo"))
      .deallocate(tmps.at("1086_baab_vooo"))
      .allocate(tmps.at("1092_abab_vvoo"))(tmps.at("1092_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1091_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("1083_abab_vooo"))(tmps.at("1083_abab_vooo")(aa, ib, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0811_abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("0940_abba_voov"))(tmps.at("0940_abba_voov")(aa, ib, jb, ba) =
                                             t2_1p.at("abab")(aa, cb, ka, jb) *
                                             eri.at("abab_oovv")(ka, ib, ba, cb))
      .allocate(tmps.at("1082_abab_vooo"))(tmps.at("1082_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0940_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("0999_abab_vovo"))(tmps.at("0999_abab_vovo")(aa, ib, ba, jb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t1_1p.at("bb")(cb, jb))
      .allocate(tmps.at("1081_abab_vooo"))(tmps.at("1081_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0999_abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("1084_abab_vooo"))(tmps.at("1084_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("1082_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1084_abab_vooo")(aa, ib, ja, kb) += tmps.at("1081_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1084_abab_vooo")(aa, ib, ja, kb) += tmps.at("1083_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("1083_abab_vooo"))
      .deallocate(tmps.at("1082_abab_vooo"))
      .deallocate(tmps.at("1081_abab_vooo"))
      .allocate(tmps.at("1085_baab_vvoo"))(tmps.at("1085_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1084_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0914_bb_ov"))(tmps.at("0914_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(ib, jb, ab, bb) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0915_abab_vooo"))(tmps.at("0915_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("0914_bb_ov")(ib, bb))
      .allocate(tmps.at("0913_abab_vooo"))(tmps.at("0913_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, lb) *
                                             tmps.at("0127_bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("1054_abab_vooo"))(tmps.at("1054_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("0915_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1054_abab_vooo")(aa, ib, ja, kb) += tmps.at("0913_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("0915_abab_vooo"))
      .deallocate(tmps.at("0913_abab_vooo"))
      .allocate(tmps.at("1080_baab_vvoo"))(tmps.at("1080_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1054_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("1054_abab_vooo"))
      .allocate(tmps.at("1077_abab_oooo"))(tmps.at("1077_abab_oooo")(ia, jb, ka, lb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0593_abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("1076_abab_oooo"))(tmps.at("1076_abab_oooo")(ia, jb, ka, lb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0286_abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("1078_abab_oooo"))(tmps.at("1078_abab_oooo")(ia, jb, ka, lb) =
                                             tmps.at("1076_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("1078_abab_oooo")(ia, jb, ka, lb) += tmps.at("1077_abab_oooo")(ia, jb, ka, lb))
      .deallocate(tmps.at("1077_abab_oooo"))
      .deallocate(tmps.at("1076_abab_oooo"))
      .allocate(tmps.at("1079_abab_vvoo"))(tmps.at("1079_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("1078_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0918_baba_vooo"))(tmps.at("0918_baba_vooo")(ab, ia, jb, ka) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0410_aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("0916_aa_ov"))(tmps.at("0916_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0917_baab_vooo"))(tmps.at("0917_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0916_aa_ov")(ia, ba))
      .allocate(tmps.at("1055_baab_vooo"))(tmps.at("1055_baab_vooo")(ab, ia, ja, kb) =
                                             tmps.at("0917_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1055_baab_vooo")(ab, ia, ja, kb) -= tmps.at("0918_baba_vooo")(ab, ia, kb, ja))
      .deallocate(tmps.at("0918_baba_vooo"))
      .deallocate(tmps.at("0917_baab_vooo"))
      .allocate(tmps.at("1075_abab_vvoo"))(tmps.at("1075_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1055_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("1055_baab_vooo"))
      .allocate(tmps.at("1074_abba_vvoo"))(tmps.at("1074_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0248_aa_oo")(ka, ja))
      .allocate(tmps.at("1073_abab_vvoo"))(tmps.at("1073_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0289_bb_oo")(kb, jb))
      .allocate(tmps.at("1072_abab_vvoo"))(tmps.at("1072_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0188_bb_vo")(bb, jb))
      .allocate(tmps.at("1071_baab_vvoo"))(tmps.at("1071_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0873_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("1070_abba_vvoo"))(tmps.at("1070_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0413_aa_oo")(ka, ja))
      .allocate(tmps.at("1069_abab_vvoo"))(tmps.at("1069_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0183_bb_vo")(bb, jb))
      .deallocate(tmps.at("0183_bb_vo"))
      .allocate(tmps.at("1068_abba_vvoo"))(tmps.at("1068_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0476_aa_oo")(ka, ja))
      .allocate(tmps.at("1067_abab_vvoo"))(tmps.at("1067_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0596_bb_oo")(kb, jb))
      .allocate(tmps.at("1066_abba_vvoo"))(tmps.at("1066_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0422_aa_oo")(ka, ja))
      .allocate(tmps.at("1065_abab_vvoo"))(tmps.at("1065_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0598_bb_oo")(kb, jb))
      .allocate(tmps.at("1064_baba_vvoo"))(tmps.at("1064_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("0765_aa_vo")(ba, ja))
      .allocate(tmps.at("1063_baba_vvoo"))(tmps.at("1063_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_2p.at("bb")(ab, ib) * tmps.at("0713_aa_vo")(ba, ja))
      .deallocate(tmps.at("0713_aa_vo"))
      .allocate(tmps.at("1062_baab_vvoo"))(tmps.at("1062_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0879_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("1061_abab_vvoo"))(tmps.at("1061_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0654_bb_oo")(kb, jb))
      .allocate(tmps.at("1060_abab_vvoo"))(tmps.at("1060_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0887_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("1058_abab_vooo"))(tmps.at("1058_abab_vooo")(aa, ib, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0816_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("1059_baab_vvoo"))(tmps.at("1059_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1058_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("1057_abab_vvoo"))(tmps.at("1057_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0875_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("1053_baab_vvoo"))(tmps.at("1053_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0043_aa_vv")(ba, ca) *
                                             t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1050_abab_vooo"))(tmps.at("1050_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, lb) *
                                             eri.at("bbbb_oovo")(lb, ib, bb, kb))
      .allocate(tmps.at("1049_baba_ovoo"))(tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) =
                                             eri.at("bbbb_oovv")(lb, ib, cb, bb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("1049_baba_ovoo")(ib, aa, jb, ka) =
          tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) * t2.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("1051_abab_vooo"))(tmps.at("1051_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("1049_baba_ovoo")(ib, aa, kb, ja))(
        tmps.at("1051_abab_vooo")(aa, ib, ja, kb) += tmps.at("1050_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("1050_abab_vooo"))
      .deallocate(tmps.at("1049_baba_ovoo"))
      .allocate(tmps.at("1052_baab_vvoo"))(tmps.at("1052_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1051_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("1048_abab_vvoo"))(tmps.at("1048_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0581_bb_vv")(bb, cb))
      .allocate(tmps.at("1047_baab_vvoo"))(tmps.at("1047_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0816_abba_voov")(ba, kb, jb, ca) *
                                             t2_1p.at("abab")(ca, ab, ia, kb))
      .allocate(tmps.at("1046_abab_vvoo"))(tmps.at("1046_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0568_bb_oo")(kb, jb))
      .allocate(tmps.at("1045_abab_vvoo"))(tmps.at("1045_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0276_bb_oo")(kb, jb))
      .allocate(tmps.at("1044_abab_vvoo"))(tmps.at("1044_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0133_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("1041_abab_vooo"))(tmps.at("1041_abab_vooo")(aa, ib, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             eri.at("abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("1040_abab_vooo"))(tmps.at("1040_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abba_vovo")(aa, ib, bb, ja) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("1039_abab_vooo"))(tmps.at("1039_abab_vooo")(aa, ib, ja, kb) =
                                             t2_1p.at("abab")(aa, bb, ja, kb) *
                                             f.at("bb_ov")(ib, bb))
      .allocate(tmps.at("1038_abab_vooo"))(tmps.at("1038_abab_vooo")(aa, ib, ja, kb) =
                                             t2_1p.at("aaaa")(ba, aa, ja, la) *
                                             eri.at("abab_oovo")(la, ib, ba, kb))
      .allocate(tmps.at("1037_baba_ovoo"))(tmps.at("bin_aabb_vooo")(ba, la, ib, jb) =
                                             eri.at("abab_oovv")(la, ib, ba, cb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("1037_baba_ovoo")(ib, aa, jb, ka) =
          tmps.at("bin_aabb_vooo")(ba, la, ib, jb) * t2_1p.at("aaaa")(ba, aa, ka, la))
      .allocate(tmps.at("1036_abba_vooo"))(tmps.at("1036_abba_vooo")(aa, ib, jb, ka) =
                                             t2_1p.at("abab")(aa, bb, la, jb) *
                                             eri.at("abba_oovo")(la, ib, bb, ka))
      .allocate(tmps.at("1035_abab_vooo"))(tmps.at("1035_abab_vooo")(aa, ib, ja, kb) =
                                             t2_1p.at("abab")(aa, bb, ja, lb) *
                                             eri.at("bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("1034_baba_ovoo"))(tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) =
                                             eri.at("bbbb_oovv")(ib, lb, cb, bb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("1034_baba_ovoo")(ib, aa, jb, ka) =
          tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) * t2_1p.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("1033_abab_vooo"))(tmps.at("1033_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t2_1p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("1042_abab_vooo"))(tmps.at("1042_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("1035_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1042_abab_vooo")(aa, ib, ja, kb) += tmps.at("1033_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1042_abab_vooo")(aa, ib, ja, kb) += tmps.at("1034_baba_ovoo")(ib, aa, kb, ja))(
        tmps.at("1042_abab_vooo")(aa, ib, ja, kb) += tmps.at("1036_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("1042_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1038_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1042_abab_vooo")(aa, ib, ja, kb) += tmps.at("1039_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1042_abab_vooo")(aa, ib, ja, kb) += tmps.at("1041_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1042_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1040_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1042_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1037_baba_ovoo")(ib, aa, kb, ja))
      .deallocate(tmps.at("1041_abab_vooo"))
      .deallocate(tmps.at("1040_abab_vooo"))
      .deallocate(tmps.at("1039_abab_vooo"))
      .deallocate(tmps.at("1038_abab_vooo"))
      .deallocate(tmps.at("1037_baba_ovoo"))
      .deallocate(tmps.at("1036_abba_vooo"))
      .deallocate(tmps.at("1035_abab_vooo"))
      .deallocate(tmps.at("1034_baba_ovoo"))
      .deallocate(tmps.at("1033_abab_vooo"))
      .allocate(tmps.at("1043_baab_vvoo"))(tmps.at("1043_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1042_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("1030_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             eri.at("abab_oovo")(ia, lb, ba, kb))(
        tmps.at("1030_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1.at("bb")(ab, lb))
      .allocate(tmps.at("1029_abab_ovoo"))(tmps.at("bin_aa_vo")(ba, ia) =
                                             eri.at("abab_oovv")(ia, lb, ba, cb) *
                                             t1_1p.at("bb")(cb, lb))(
        tmps.at("1029_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aa_vo")(ba, ia) * t2.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("1028_abba_ovoo"))(tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) =
                                             eri.at("abab_oovv")(ia, lb, ba, cb) *
                                             t1_1p.at("bb")(cb, jb))(
        tmps.at("1028_abba_ovoo")(ia, ab, jb, ka) =
          tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) * t2.at("abab")(ba, ab, ka, lb))
      .allocate(tmps.at("1031_abab_ovoo"))(tmps.at("1031_abab_ovoo")(ia, ab, ja, kb) =
                                             -1.00 * tmps.at("1029_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("1031_abab_ovoo")(ia, ab, ja, kb) += tmps.at("1030_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("1031_abab_ovoo")(ia, ab, ja, kb) += tmps.at("1028_abba_ovoo")(ia, ab, kb, ja))
      .deallocate(tmps.at("1030_abab_ovoo"))
      .deallocate(tmps.at("1029_abab_ovoo"))
      .deallocate(tmps.at("1028_abba_ovoo"))
      .allocate(tmps.at("1032_abab_vvoo"))(tmps.at("1032_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1031_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("1027_abba_vvoo"))(tmps.at("1027_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0344_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1024_baab_vooo"))(tmps.at("1024_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             eri.at("abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1023_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             t1.at("aa")(ba, ja) *
                                             eri.at("abab_oovo")(ia, lb, ba, kb))(
        tmps.at("1023_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1_1p.at("bb")(ab, lb))
      .allocate(tmps.at("1022_baab_vooo"))(tmps.at("1022_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             eri.at("baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("1021_baab_vooo"))(tmps.at("1021_baab_vooo")(ab, ia, ja, kb) =
                                             t2_1p.at("abab")(ba, ab, ja, kb) *
                                             f.at("aa_ov")(ia, ba))
      .allocate(tmps.at("1020_baab_vooo"))(tmps.at("1020_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baba_vovo")(ab, ia, bb, ja) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("1019_baba_vooo"))(tmps.at("1019_baba_vooo")(ab, ia, jb, ka) =
                                             t2_1p.at("abab")(ba, ab, la, jb) *
                                             eri.at("aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("1018_abab_ovoo"))(tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) =
                                             eri.at("aaaa_oovv")(ia, la, ca, ba) *
                                             t1.at("aa")(ca, ja))(
        tmps.at("1018_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) * t2_1p.at("abab")(ba, ab, la, kb))
      .allocate(tmps.at("1017_baab_vooo"))(tmps.at("1017_baab_vooo")(ab, ia, ja, kb) =
                                             t2_1p.at("abab")(ba, ab, ja, lb) *
                                             eri.at("abab_oovo")(ia, lb, ba, kb))
      .allocate(tmps.at("1016_abba_ovoo"))(tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) =
                                             eri.at("abab_oovv")(ia, lb, ba, cb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("1016_abba_ovoo")(ia, ab, jb, ka) =
          tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) * t2_1p.at("abab")(ba, ab, ka, lb))
      .allocate(tmps.at("1015_baba_vooo"))(tmps.at("1015_baba_vooo")(ab, ia, jb, ka) =
                                             t2_1p.at("bbbb")(bb, ab, jb, lb) *
                                             eri.at("abba_oovo")(ia, lb, bb, ka))
      .allocate(tmps.at("1014_baab_vooo"))(tmps.at("1014_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t2_1p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("1025_baab_vooo"))(tmps.at("1025_baab_vooo")(ab, ia, ja, kb) =
                                             -1.00 * tmps.at("1020_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1025_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1015_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("1025_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1018_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("1025_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1021_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1025_baab_vooo")(ab, ia, ja, kb) += tmps.at("1014_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1025_baab_vooo")(ab, ia, ja, kb) += tmps.at("1017_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1025_baab_vooo")(ab, ia, ja, kb) += tmps.at("1024_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1025_baab_vooo")(ab, ia, ja, kb) += tmps.at("1023_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("1025_baab_vooo")(ab, ia, ja, kb) += tmps.at("1019_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("1025_baab_vooo")(ab, ia, ja, kb) += tmps.at("1016_abba_ovoo")(ia, ab, kb, ja))(
        tmps.at("1025_baab_vooo")(ab, ia, ja, kb) += tmps.at("1022_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("1024_baab_vooo"))
      .deallocate(tmps.at("1023_abab_ovoo"))
      .deallocate(tmps.at("1022_baab_vooo"))
      .deallocate(tmps.at("1021_baab_vooo"))
      .deallocate(tmps.at("1020_baab_vooo"))
      .deallocate(tmps.at("1019_baba_vooo"))
      .deallocate(tmps.at("1018_abab_ovoo"))
      .deallocate(tmps.at("1017_baab_vooo"))
      .deallocate(tmps.at("1016_abba_ovoo"))
      .deallocate(tmps.at("1015_baba_vooo"))
      .deallocate(tmps.at("1014_baab_vooo"))
      .allocate(tmps.at("1026_abab_vvoo"))(tmps.at("1026_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1025_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("1012_aa_vv"))(tmps.at("1012_aa_vv")(aa, ba) =
                                         eri.at("abab_vovv")(aa, ib, ba, cb) *
                                         t1_1p.at("bb")(cb, ib))
      .allocate(tmps.at("1013_baab_vvoo"))(tmps.at("1013_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1012_aa_vv")(ba, ca) *
                                             t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1011_baba_vvoo"))(tmps.at("1011_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0737_aa_vo")(ba, ja) * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("1010_abba_vvoo"))(tmps.at("1010_abba_vvoo")(aa, bb, ib, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0862_baba_vooo")(bb, ka, ib, ja))
      .allocate(tmps.at("1009_abab_vvoo"))(tmps.at("1009_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0154_bb_vo")(bb, jb))
      .allocate(tmps.at("1006_baab_ovoo"))(tmps.at("bin_bb_vo")(bb, ib) =
                                             eri.at("abab_oovv")(la, ib, ca, bb) *
                                             t1_1p.at("aa")(ca, la))(
        tmps.at("1006_baab_ovoo")(ib, aa, ja, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("1005_baba_ovoo"))(tmps.at("bin_aabb_vooo")(ba, la, ib, jb) =
                                             eri.at("abab_oovv")(la, ib, ba, cb) *
                                             t1_1p.at("bb")(cb, jb))(
        tmps.at("1005_baba_ovoo")(ib, aa, jb, ka) =
          tmps.at("bin_aabb_vooo")(ba, la, ib, jb) * t2.at("aaaa")(ba, aa, ka, la))
      .allocate(tmps.at("1007_baab_ovoo"))(tmps.at("1007_baab_ovoo")(ib, aa, ja, kb) =
                                             -1.00 * tmps.at("1005_baba_ovoo")(ib, aa, kb, ja))(
        tmps.at("1007_baab_ovoo")(ib, aa, ja, kb) += tmps.at("1006_baab_ovoo")(ib, aa, ja, kb))
      .deallocate(tmps.at("1006_baab_ovoo"))
      .deallocate(tmps.at("1005_baba_ovoo"));
  }
}

template void exachem::cc::qed_ccsd_os::resid_4<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);