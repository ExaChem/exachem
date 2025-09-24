/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_os_resid_5.hpp"

template<typename T>
void exachem::cc::qed_ccsd_os::resid_5(
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

      .allocate(tmps.at("1008_baab_vvoo"))(tmps.at("1008_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1007_baab_ovoo")(kb, ba, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("1004_abab_vvoo"))(tmps.at("1004_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0842_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("1003_abab_vvoo"))(tmps.at("1003_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0528_bb_oo")(kb, jb))
      .allocate(tmps.at("1002_baab_vvoo"))(tmps.at("1002_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0855_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("1001_abba_vvoo"))(tmps.at("1001_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0456_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1000_baab_vvoo"))(tmps.at("1000_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0999_abab_vovo")(ba, kb, ca, jb) *
                                             t2.at("abab")(ca, ab, ia, kb))
      .allocate(tmps.at("0998_baba_vvoo"))(tmps.at("0998_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0710_aa_vo")(ba, ja) * t1_2p.at("bb")(ab, ib))
      .deallocate(tmps.at("0710_aa_vo"))
      .allocate(tmps.at("0996_bb_vv"))(tmps.at("0996_bb_vv")(ab, bb) =
                                         eri.at("bbbb_vovv")(ab, ib, bb, cb) *
                                         t1_1p.at("bb")(cb, ib))
      .allocate(tmps.at("0997_abab_vvoo"))(tmps.at("0997_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0996_bb_vv")(bb, cb))
      .allocate(tmps.at("0994_abab_oooo"))(tmps.at("0994_abab_oooo")(ia, jb, ka, lb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             eri.at("abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("0995_abab_vvoo"))(tmps.at("0995_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0994_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0993_abab_vvoo"))(tmps.at("0993_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0281_bb_oo")(kb, jb))
      .allocate(tmps.at("0992_abab_vvoo"))(tmps.at("0992_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0566_bb_oo")(kb, jb))
      .allocate(tmps.at("0990_abba_vvvo"))(tmps.at("0990_abba_vvvo")(aa, bb, cb, ia) =
                                             eri.at("abab_vvvv")(aa, bb, da, cb) *
                                             t1_1p.at("aa")(da, ia))
      .allocate(tmps.at("0991_abba_vvoo"))(tmps.at("0991_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0990_abba_vvvo")(aa, bb, cb, ja) *
                                             t1.at("bb")(cb, ib))
      .allocate(tmps.at("0989_baab_vvoo"))(tmps.at("0989_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0826_baab_ovoo")(kb, ba, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0987_aa_vv"))(tmps.at("0987_aa_vv")(aa, ba) =
                                         eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                         t1_1p.at("aa")(ca, ia))
      .allocate(tmps.at("0988_baab_vvoo"))(tmps.at("0988_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0987_aa_vv")(ba, ca) *
                                             t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0984_baab_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("abab_oovv")(la, ib, ca, bb) * t1.at("aa")(ca, la))(
        tmps.at("0984_baab_ovoo")(ib, aa, ja, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0983_baab_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("bbbb_oovv")(lb, ib, cb, bb) * t1.at("bb")(cb, lb))(
        tmps.at("0983_baab_ovoo")(ib, aa, ja, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0985_baab_ovoo"))(tmps.at("0985_baab_ovoo")(ib, aa, ja, kb) =
                                             tmps.at("0983_baab_ovoo")(ib, aa, ja, kb))(
        tmps.at("0985_baab_ovoo")(ib, aa, ja, kb) += tmps.at("0984_baab_ovoo")(ib, aa, ja, kb))
      .deallocate(tmps.at("0984_baab_ovoo"))
      .deallocate(tmps.at("0983_baab_ovoo"))
      .allocate(tmps.at("0986_baab_vvoo"))(tmps.at("0986_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0985_baab_ovoo")(kb, ba, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .allocate(tmps.at("0982_baab_vvoo"))(tmps.at("0982_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0041_aa_vv")(ba, ca) *
                                             t2_1p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0981_baba_vvoo"))(tmps.at("0981_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0741_aa_vo")(ba, ja) * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("0980_baba_vvoo"))(tmps.at("0980_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0039_aaaa_voov")(ba, ka, ja, ca) *
                                             t2.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("0979_abab_vvoo"))(tmps.at("0979_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0299_bb_vv")(bb, cb))
      .allocate(tmps.at("0978_baba_vvoo"))(tmps.at("0978_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0847_baba_ovoo")(kb, ba, ib, ja) *
                                             t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0977_abab_vvoo"))(tmps.at("0977_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0976_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0973_abab_vvoo"))(tmps.at("0973_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0831_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0970_baba_vooo"))(tmps.at("0970_baba_vooo")(ab, ia, jb, ka) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             eri.at("aaaa_oovo")(la, ia, ba, ka))
      .allocate(tmps.at("0969_abab_ovoo"))(tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) =
                                             eri.at("aaaa_oovv")(la, ia, ca, ba) *
                                             t1.at("aa")(ca, ja))(
        tmps.at("0969_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) * t2.at("abab")(ba, ab, la, kb))
      .allocate(tmps.at("0971_baba_vooo"))(tmps.at("0971_baba_vooo")(ab, ia, jb, ka) =
                                             -1.00 * tmps.at("0969_abab_ovoo")(ia, ab, ka, jb))(
        tmps.at("0971_baba_vooo")(ab, ia, jb, ka) += tmps.at("0970_baba_vooo")(ab, ia, jb, ka))
      .deallocate(tmps.at("0970_baba_vooo"))
      .deallocate(tmps.at("0969_abab_ovoo"))
      .allocate(tmps.at("0972_abba_vvoo"))(tmps.at("0972_abba_vvoo")(aa, bb, ib, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0971_baba_vooo")(bb, ka, ib, ja))
      .allocate(tmps.at("0968_abba_vvoo"))(tmps.at("0968_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0241_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0967_baba_vvoo"))(tmps.at("0967_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0392_abba_vovo")(ba, kb, cb, ja) *
                                             t2.at("bbbb")(cb, ab, ib, kb))
      .allocate(tmps.at("0966_abab_vvoo"))(tmps.at("0966_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0033_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("0965_abab_vvoo"))(tmps.at("0965_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0271_bb_oo")(kb, jb))
      .allocate(tmps.at("0964_baba_vvoo"))(tmps.at("0964_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0348_aaaa_vovo")(ba, ka, ca, ja) *
                                             t2.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("0963_abab_vvoo"))(tmps.at("0963_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0164_bb_vo")(bb, jb))
      .deallocate(tmps.at("0164_bb_vo"))
      .allocate(tmps.at("0962_baba_vvoo"))(tmps.at("0962_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0230_abba_vovo")(ba, kb, cb, ja) *
                                             t2_1p.at("bbbb")(cb, ab, ib, kb))
      .allocate(tmps.at("0961_abab_vvoo"))(tmps.at("0961_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0583_bb_vv")(bb, cb))
      .allocate(tmps.at("0958_abab_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("aaaa_oovv")(la, ia, ca, ba) * t1.at("aa")(ca, la))(
        tmps.at("0958_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aa_vo")(ba, ia) * t2_1p.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0957_abab_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("abab_oovv")(ia, lb, ba, cb) * t1.at("bb")(cb, lb))(
        tmps.at("0957_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aa_vo")(ba, ia) * t2_1p.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0959_abab_ovoo"))(tmps.at("0959_abab_ovoo")(ia, ab, ja, kb) =
                                             tmps.at("0957_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0959_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0958_abab_ovoo")(ia, ab, ja, kb))
      .deallocate(tmps.at("0958_abab_ovoo"))
      .deallocate(tmps.at("0957_abab_ovoo"))
      .allocate(tmps.at("0960_abab_vvoo"))(tmps.at("0960_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0959_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0956_baab_vvoo"))(tmps.at("0956_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0664_aa_vv")(ba, ca) *
                                             t2_1p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0955_abab_vvoo"))(tmps.at("0955_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0158_bb_vo")(bb, jb))
      .allocate(tmps.at("0954_abab_vvoo"))(tmps.at("0954_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0534_bb_oo")(kb, jb))
      .allocate(tmps.at("0953_baba_vvoo"))(tmps.at("0953_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0363_aaaa_vovo")(ba, ka, ca, ja) *
                                             t2_1p.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("0951_baba_vovo"))(tmps.at("0951_baba_vovo")(ab, ia, bb, ja) =
                                             eri.at("baab_vovv")(ab, ia, ca, bb) *
                                             t1_1p.at("aa")(ca, ja))
      .allocate(tmps.at("0952_abba_vvoo"))(tmps.at("0952_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, cb, ka, ib) *
                                             tmps.at("0951_baba_vovo")(bb, ka, cb, ja))
      .allocate(tmps.at("0950_abab_vvoo"))(tmps.at("0950_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0642_bb_oo")(kb, jb))
      .allocate(tmps.at("0949_abab_vvoo"))(tmps.at("0949_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0819_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0948_abab_vvoo"))(tmps.at("0948_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0836_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0947_baab_vvoo"))(tmps.at("0947_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0374_aa_vv")(ba, ca) *
                                             t2_1p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0946_baab_vvoo"))(tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0811_abab_vovo")(ba, kb, ca, jb) *
                                             t2_1p.at("abab")(ca, ab, ia, kb))
      .allocate(tmps.at("0945_abab_vvoo"))(tmps.at("0945_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0544_bb_vv")(bb, cb))
      .allocate(tmps.at("0944_abab_vvoo"))(tmps.at("0944_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0135_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("0943_abab_vvoo"))(tmps.at("0943_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0137_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("0942_abba_vvoo"))(tmps.at("0942_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0369_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0941_baab_vvoo"))(tmps.at("0941_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0940_abba_voov")(ba, kb, jb, ca) *
                                             t2.at("abab")(ca, ab, ia, kb))
      .allocate(tmps.at("0939_abba_vvoo"))(tmps.at("0939_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0394_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0938_baba_vvoo"))(tmps.at("0938_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0376_abab_voov")(ba, kb, ja, cb) *
                                             t2_1p.at("bbbb")(cb, ab, ib, kb))
      .allocate(tmps.at("0937_abab_vvoo"))(tmps.at("0937_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0144_bbbb_voov")(bb, kb, jb, cb))
      .allocate(tmps.at("0936_baba_vvoo"))(tmps.at("0936_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0037_aaaa_voov")(ba, ka, ja, ca) *
                                             t2_1p.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("0935_abba_vvoo"))(tmps.at("0935_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0360_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0934_abab_vvoo"))(tmps.at("0934_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0035_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("0933_abba_vvoo"))(tmps.at("0933_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0821_abba_vvvo")(aa, bb, cb, ja) *
                                             t1_1p.at("bb")(cb, ib))
      .allocate(tmps.at("0932_abba_vvoo"))(tmps.at("0932_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, cb, ka, ib) *
                                             tmps.at("0799_baba_vovo")(bb, ka, cb, ja))
      .allocate(tmps.at("0931_abab_vvoo"))(tmps.at("0931_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0141_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("0930_abba_vvoo"))(tmps.at("0930_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0234_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0929_baba_vvoo"))(tmps.at("0929_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0346_aaaa_voov")(ba, ka, ja, ca) *
                                             t2_1p.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("0928_abba_vvoo"))(tmps.at("0928_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0390_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0927_abab_vvoo"))(tmps.at("0927_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0505_bb_vv")(bb, cb))
      .allocate(tmps.at("0926_baab_vvoo"))(tmps.at("0926_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0407_aa_vv")(ba, ca) *
                                             t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0925_baab_vvoo"))(tmps.at("0925_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0226_aa_vv")(ba, ca) *
                                             t2_1p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0924_abba_vvoo"))(tmps.at("0924_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0243_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0923_abab_vvoo"))(tmps.at("0923_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0131_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("0921_bb_vv"))(tmps.at("0921_bb_vv")(ab, bb) =
                                         eri.at("baab_vovv")(ab, ia, ca, bb) *
                                         t1_1p.at("aa")(ca, ia))
      .allocate(tmps.at("0922_abab_vvoo"))(tmps.at("0922_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0921_bb_vv")(bb, cb))
      .allocate(tmps.at("0920_abab_vvoo"))(tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0297_bb_vv")(bb, cb))
      .allocate(tmps.at("0919_abab_vvoo"))(tmps.at("0919_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0273_bb_oo")(kb, jb))
      .allocate(tmps.at("0911_abab_vvoo"))(tmps.at("0911_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             eri.at("baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("0910_abab_vvoo"))(tmps.at("0910_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_oooo")(ka, lb, ia, jb) *
                                             t2_1p.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0909_abab_vvoo"))(tmps.at("0909_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(ca, ia) *
                                             eri.at("abab_vvvo")(aa, bb, ca, jb))
      .allocate(tmps.at("0908_abab_vvoo"))(tmps.at("0908_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0002")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0907_abab_vvoo"))(tmps.at("0907_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0001")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0906_abab_vvoo"))(tmps.at("0906_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0015")() *
                                             t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0905_abab_vvoo"))(tmps.at("0905_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0013")() *
                                             t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0904_abab_vvoo"))(tmps.at("0904_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             eri.at("baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0903_abba_vvoo"))(tmps.at("0903_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, cb, ka, ib) *
                                             eri.at("baba_vovo")(bb, ka, cb, ja))
      .allocate(tmps.at("0902_abba_vvoo"))(tmps.at("0902_abba_vvoo")(aa, bb, ib, ja) =
                                             eri.at("abab_vovo")(aa, kb, ca, ib) *
                                             t2_1p.at("abab")(ca, bb, ja, kb))
      .allocate(tmps.at("0901_abab_vvoo"))(tmps.at("0901_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("aaaa_vovo")(aa, ka, ca, ia) *
                                             t2_1p.at("abab")(ca, bb, ka, jb))
      .allocate(tmps.at("0900_abab_vvoo"))(tmps.at("0900_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vovo")(aa, kb, cb, ia) *
                                             t2_1p.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("0899_abab_vvoo"))(tmps.at("0899_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vvvo")(aa, bb, cb, ia) *
                                             t1_1p.at("bb")(cb, jb))
      .allocate(tmps.at("0898_abab_vvoo"))(tmps.at("0898_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             f.at("bb_oo")(kb, jb))
      .allocate(tmps.at("0897_abab_vvoo"))(tmps.at("0897_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             f.at("bb_vv")(bb, cb))
      .allocate(tmps.at("0896_abab_vvoo"))(tmps.at("0896_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_oo")(ka, ia) *
                                             t2_1p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0895_abab_vvoo"))(tmps.at("0895_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vvvv")(aa, bb, ca, db) *
                                             t2_1p.at("abab")(ca, db, ia, jb))
      .allocate(tmps.at("0894_abab_vvoo"))(tmps.at("0894_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, kb) *
                                             eri.at("bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("0893_abab_vvoo"))(tmps.at("0893_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ia) * dp.at("bb_vo")(bb, jb))
      .allocate(tmps.at("0892_abab_vvoo"))(tmps.at("0892_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_vv")(aa, ca) *
                                             t2_1p.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0891_abab_vvoo"))(tmps.at("0891_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vooo")(aa, kb, ia, jb) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("0890_abab_vvoo"))(tmps.at("0890_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_vo")(aa, ia) * t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("0912_abab_vvoo"))(tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0891_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0901_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0909_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0892_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) +=
        2.00 * tmps.at("0893_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0911_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0908_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) +=
        2.00 * tmps.at("0905_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0898_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0897_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0896_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) +=
        2.00 * tmps.at("0906_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) +=
        2.00 * tmps.at("0890_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0895_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0910_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0899_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0904_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0902_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0903_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0907_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0900_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0912_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0894_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0911_abab_vvoo"))
      .deallocate(tmps.at("0910_abab_vvoo"))
      .deallocate(tmps.at("0909_abab_vvoo"))
      .deallocate(tmps.at("0908_abab_vvoo"))
      .deallocate(tmps.at("0907_abab_vvoo"))
      .deallocate(tmps.at("0906_abab_vvoo"))
      .deallocate(tmps.at("0905_abab_vvoo"))
      .deallocate(tmps.at("0904_abab_vvoo"))
      .deallocate(tmps.at("0903_abba_vvoo"))
      .deallocate(tmps.at("0902_abba_vvoo"))
      .deallocate(tmps.at("0901_abab_vvoo"))
      .deallocate(tmps.at("0900_abab_vvoo"))
      .deallocate(tmps.at("0899_abab_vvoo"))
      .deallocate(tmps.at("0898_abab_vvoo"))
      .deallocate(tmps.at("0897_abab_vvoo"))
      .deallocate(tmps.at("0896_abab_vvoo"))
      .deallocate(tmps.at("0895_abab_vvoo"))
      .deallocate(tmps.at("0894_abab_vvoo"))
      .deallocate(tmps.at("0893_abab_vvoo"))
      .deallocate(tmps.at("0892_abab_vvoo"))
      .deallocate(tmps.at("0891_abab_vvoo"))
      .deallocate(tmps.at("0890_abab_vvoo"))
      .allocate(tmps.at("1056_baab_vvoo"))(tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) =
                                             -0.50 * tmps.at("1045_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -=
        0.50 * tmps.at("1048_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0912_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0950_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0955_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0980_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0997_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0960_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0965_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0946_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0979_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0967_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0964_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0948_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0939_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("1002_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0930_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) +=
        0.50 * tmps.at("0945_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1000_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0988_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0962_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0963_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0929_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0931_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0956_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0995_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) +=
        0.50 * tmps.at("0926_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0923_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0977_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0925_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("1001_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1027_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("1047_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0961_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1008_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0998_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -=
        0.50 * tmps.at("0928_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0953_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) +=
        0.50 * tmps.at("0954_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0941_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0920_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0924_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("1013_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1046_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1003_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0932_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0933_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0922_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0936_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0981_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0938_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0991_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) +=
        0.50 * tmps.at("0935_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0992_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0934_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0949_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -=
        0.50 * tmps.at("0947_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0982_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0966_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0973_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0944_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0943_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("1011_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0972_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0968_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("1026_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1004_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0952_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1010_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1009_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("1032_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0989_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0993_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0942_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1052_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0986_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1053_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("1043_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0937_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0927_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("1044_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0919_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("1056_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0978_baba_vvoo")(ab, ba, jb, ia))
      .deallocate(tmps.at("1053_baab_vvoo"))
      .deallocate(tmps.at("1052_baab_vvoo"))
      .deallocate(tmps.at("1048_abab_vvoo"))
      .deallocate(tmps.at("1047_baab_vvoo"))
      .deallocate(tmps.at("1046_abab_vvoo"))
      .deallocate(tmps.at("1045_abab_vvoo"))
      .deallocate(tmps.at("1044_abab_vvoo"))
      .deallocate(tmps.at("1043_baab_vvoo"))
      .deallocate(tmps.at("1032_abab_vvoo"))
      .deallocate(tmps.at("1027_abba_vvoo"))
      .deallocate(tmps.at("1026_abab_vvoo"))
      .deallocate(tmps.at("1013_baab_vvoo"))
      .deallocate(tmps.at("1011_baba_vvoo"))
      .deallocate(tmps.at("1010_abba_vvoo"))
      .deallocate(tmps.at("1009_abab_vvoo"))
      .deallocate(tmps.at("1008_baab_vvoo"))
      .deallocate(tmps.at("1004_abab_vvoo"))
      .deallocate(tmps.at("1003_abab_vvoo"))
      .deallocate(tmps.at("1002_baab_vvoo"))
      .deallocate(tmps.at("1001_abba_vvoo"))
      .deallocate(tmps.at("1000_baab_vvoo"))
      .deallocate(tmps.at("0998_baba_vvoo"))
      .deallocate(tmps.at("0997_abab_vvoo"))
      .deallocate(tmps.at("0995_abab_vvoo"))
      .deallocate(tmps.at("0993_abab_vvoo"))
      .deallocate(tmps.at("0992_abab_vvoo"))
      .deallocate(tmps.at("0991_abba_vvoo"))
      .deallocate(tmps.at("0989_baab_vvoo"))
      .deallocate(tmps.at("0988_baab_vvoo"))
      .deallocate(tmps.at("0986_baab_vvoo"))
      .deallocate(tmps.at("0982_baab_vvoo"))
      .deallocate(tmps.at("0981_baba_vvoo"))
      .deallocate(tmps.at("0980_baba_vvoo"))
      .deallocate(tmps.at("0979_abab_vvoo"))
      .deallocate(tmps.at("0978_baba_vvoo"))
      .deallocate(tmps.at("0977_abab_vvoo"))
      .deallocate(tmps.at("0973_abab_vvoo"))
      .deallocate(tmps.at("0972_abba_vvoo"))
      .deallocate(tmps.at("0968_abba_vvoo"))
      .deallocate(tmps.at("0967_baba_vvoo"))
      .deallocate(tmps.at("0966_abab_vvoo"))
      .deallocate(tmps.at("0965_abab_vvoo"))
      .deallocate(tmps.at("0964_baba_vvoo"))
      .deallocate(tmps.at("0963_abab_vvoo"))
      .deallocate(tmps.at("0962_baba_vvoo"))
      .deallocate(tmps.at("0961_abab_vvoo"))
      .deallocate(tmps.at("0960_abab_vvoo"))
      .deallocate(tmps.at("0956_baab_vvoo"))
      .deallocate(tmps.at("0955_abab_vvoo"))
      .deallocate(tmps.at("0954_abab_vvoo"))
      .deallocate(tmps.at("0953_baba_vvoo"))
      .deallocate(tmps.at("0952_abba_vvoo"))
      .deallocate(tmps.at("0950_abab_vvoo"))
      .deallocate(tmps.at("0949_abab_vvoo"))
      .deallocate(tmps.at("0948_abab_vvoo"))
      .deallocate(tmps.at("0947_baab_vvoo"))
      .deallocate(tmps.at("0946_baab_vvoo"))
      .deallocate(tmps.at("0945_abab_vvoo"))
      .deallocate(tmps.at("0944_abab_vvoo"))
      .deallocate(tmps.at("0943_abab_vvoo"))
      .deallocate(tmps.at("0942_abba_vvoo"))
      .deallocate(tmps.at("0941_baab_vvoo"))
      .deallocate(tmps.at("0939_abba_vvoo"))
      .deallocate(tmps.at("0938_baba_vvoo"))
      .deallocate(tmps.at("0937_abab_vvoo"))
      .deallocate(tmps.at("0936_baba_vvoo"))
      .deallocate(tmps.at("0935_abba_vvoo"))
      .deallocate(tmps.at("0934_abab_vvoo"))
      .deallocate(tmps.at("0933_abba_vvoo"))
      .deallocate(tmps.at("0932_abba_vvoo"))
      .deallocate(tmps.at("0931_abab_vvoo"))
      .deallocate(tmps.at("0930_abba_vvoo"))
      .deallocate(tmps.at("0929_baba_vvoo"))
      .deallocate(tmps.at("0928_abba_vvoo"))
      .deallocate(tmps.at("0927_abab_vvoo"))
      .deallocate(tmps.at("0926_baab_vvoo"))
      .deallocate(tmps.at("0925_baab_vvoo"))
      .deallocate(tmps.at("0924_abba_vvoo"))
      .deallocate(tmps.at("0923_abab_vvoo"))
      .deallocate(tmps.at("0922_abab_vvoo"))
      .deallocate(tmps.at("0920_abab_vvoo"))
      .deallocate(tmps.at("0919_abab_vvoo"))
      .deallocate(tmps.at("0912_abab_vvoo"))
      .allocate(tmps.at("1100_abba_vvoo"))(tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) =
                                             -1.00 * tmps.at("1057_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) +=
        2.00 * tmps.at("1069_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("1071_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1065_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1068_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("1059_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1066_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1056_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) +=
        2.00 * tmps.at("1063_baba_vvoo")(bb, aa, ib, ja))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1072_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1061_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("1060_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1062_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("1092_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("1080_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1067_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1085_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("1099_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("1093_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1075_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1074_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1064_baba_vvoo")(bb, aa, ib, ja))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1070_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) -= tmps.at("1079_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("1100_abba_vvoo")(aa, bb, ib, ja) += tmps.at("1073_abab_vvoo")(aa, bb, ja, ib))
      .deallocate(tmps.at("1099_abab_vvoo"))
      .deallocate(tmps.at("1093_abab_vvoo"))
      .deallocate(tmps.at("1092_abab_vvoo"))
      .deallocate(tmps.at("1085_baab_vvoo"))
      .deallocate(tmps.at("1080_baab_vvoo"))
      .deallocate(tmps.at("1079_abab_vvoo"))
      .deallocate(tmps.at("1075_abab_vvoo"))
      .deallocate(tmps.at("1074_abba_vvoo"))
      .deallocate(tmps.at("1073_abab_vvoo"))
      .deallocate(tmps.at("1072_abab_vvoo"))
      .deallocate(tmps.at("1071_baab_vvoo"))
      .deallocate(tmps.at("1070_abba_vvoo"))
      .deallocate(tmps.at("1069_abab_vvoo"))
      .deallocate(tmps.at("1068_abba_vvoo"))
      .deallocate(tmps.at("1067_abab_vvoo"))
      .deallocate(tmps.at("1066_abba_vvoo"))
      .deallocate(tmps.at("1065_abab_vvoo"))
      .deallocate(tmps.at("1064_baba_vvoo"))
      .deallocate(tmps.at("1063_baba_vvoo"))
      .deallocate(tmps.at("1062_baab_vvoo"))
      .deallocate(tmps.at("1061_abab_vvoo"))
      .deallocate(tmps.at("1060_abab_vvoo"))
      .deallocate(tmps.at("1059_baab_vvoo"))
      .deallocate(tmps.at("1057_abab_vvoo"))
      .deallocate(tmps.at("1056_baab_vvoo"))(r2_1p.at("abab")(aa, bb, ia, jb) -=
                                             tmps.at("1100_abba_vvoo")(aa, bb, jb, ia))
      .deallocate(tmps.at("1100_abba_vvoo"))
      .allocate(tmps.at("1134_aaaa_vooo"))(tmps.at("1134_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             tmps.at("0204_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1135_aaaa_vvoo"))(tmps.at("1135_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1134_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1131_aaaa_vooo"))(tmps.at("1131_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             tmps.at("0200_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1132_aaaa_vvoo"))(tmps.at("1132_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1131_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1130_aaaa_vvoo"))(tmps.at("1130_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0674_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1104_aaaa_vooo"))(tmps.at("1104_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, ka) *
                                             tmps.at("0916_aa_ov")(ia, ba))
      .deallocate(tmps.at("0916_aa_ov"))
      .allocate(tmps.at("1129_aaaa_vvoo"))(tmps.at("1129_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1104_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1104_aaaa_vooo"))
      .allocate(tmps.at("1127_aaaa_vvoo"))(tmps.at("1127_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0669_aaaa_ovoo")(ka, ba, ia, ja))
      .allocate(tmps.at("1124_aaaa_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("aaaa_oovv")(la, ia, ca, ba) * t1.at("aa")(ca, la))(
        tmps.at("1124_aaaa_ovoo")(ia, aa, ja, ka) =
          tmps.at("bin_aa_vo")(ba, ia) * t2_1p.at("aaaa")(ba, aa, ja, ka))
      .allocate(tmps.at("1123_aaaa_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("abab_oovv")(ia, lb, ba, cb) * t1.at("bb")(cb, lb))(
        tmps.at("1123_aaaa_ovoo")(ia, aa, ja, ka) =
          tmps.at("bin_aa_vo")(ba, ia) * t2_1p.at("aaaa")(ba, aa, ja, ka))
      .allocate(tmps.at("1125_aaaa_ovoo"))(tmps.at("1125_aaaa_ovoo")(ia, aa, ja, ka) =
                                             tmps.at("1123_aaaa_ovoo")(ia, aa, ja, ka))(
        tmps.at("1125_aaaa_ovoo")(ia, aa, ja, ka) += tmps.at("1124_aaaa_ovoo")(ia, aa, ja, ka))
      .deallocate(tmps.at("1124_aaaa_ovoo"))
      .deallocate(tmps.at("1123_aaaa_ovoo"))
      .allocate(tmps.at("1126_aaaa_vvoo"))(tmps.at("1126_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1125_aaaa_ovoo")(ka, ba, ia, ja))
      .allocate(tmps.at("1122_aaaa_vvoo"))(tmps.at("1122_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0664_aa_vv")(ba, ca))
      .allocate(tmps.at("1121_aaaa_vvoo"))(tmps.at("1121_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0407_aa_vv")(ba, ca))
      .allocate(tmps.at("1119_aaaa_vooo"))(tmps.at("1119_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_1p.at("aaaa")(ba, aa, ja, ka) *
                                             f.at("aa_ov")(ia, ba))
      .allocate(tmps.at("1120_aaaa_vvoo"))(tmps.at("1120_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1119_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1117_aaaa_ovoo"))(tmps.at("bin_aa_vo")(ba, ia) =
                                             eri.at("abab_oovv")(ia, lb, ba, cb) *
                                             t1_1p.at("bb")(cb, lb))(
        tmps.at("1117_aaaa_ovoo")(ia, aa, ja, ka) =
          tmps.at("bin_aa_vo")(ba, ia) * t2.at("aaaa")(ba, aa, ja, ka))
      .allocate(tmps.at("1118_aaaa_vvoo"))(tmps.at("1118_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1117_aaaa_ovoo")(ka, ba, ia, ja))
      .allocate(tmps.at("1116_aaaa_vvoo"))(tmps.at("1116_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0662_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1115_aaaa_vvoo"))(tmps.at("1115_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0226_aa_vv")(ba, ca))
      .allocate(tmps.at("1114_aaaa_vvoo"))(tmps.at("1114_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0671_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1112_aaaa_vooo"))(tmps.at("1112_aaaa_vooo")(aa, ia, ja, ka) =
                                             eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                             t2_1p.at("aaaa")(ba, ca, ja, ka))
      .allocate(tmps.at("1113_aaaa_vvoo"))(tmps.at("1113_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1112_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1111_aaaa_vvoo"))(tmps.at("1111_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("1012_aa_vv")(ba, ca))
      .allocate(tmps.at("1110_aaaa_vvoo"))(tmps.at("1110_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0374_aa_vv")(ba, ca))
      .allocate(tmps.at("1108_aaaa_vooo"))(tmps.at("1108_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             eri.at("aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1109_aaaa_vvoo"))(tmps.at("1109_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1108_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1107_aaaa_vvoo"))(tmps.at("1107_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0041_aa_vv")(ba, ca))
      .allocate(tmps.at("1106_aaaa_vvoo"))(tmps.at("1106_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0043_aa_vv")(ba, ca))
      .allocate(tmps.at("1105_aaaa_vvoo"))(tmps.at("1105_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0987_aa_vv")(ba, ca))
      .deallocate(tmps.at("0987_aa_vv"))
      .allocate(tmps.at("1102_aaaa_vvoo"))(tmps.at("1102_aaaa_vvoo")(aa, ba, ia, ja) =
                                             f.at("aa_vv")(aa, ca) *
                                             t2_1p.at("aaaa")(ca, ba, ia, ja))
      .allocate(tmps.at("1101_aaaa_vvoo"))(tmps.at("1101_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_vooo")(aa, ka, ia, ja) *
                                             t1_1p.at("aa")(ba, ka))
      .allocate(tmps.at("1103_aaaa_vvoo"))(tmps.at("1103_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("1101_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1103_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1102_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1102_aaaa_vvoo"))
      .deallocate(tmps.at("1101_aaaa_vvoo"))
      .allocate(tmps.at("1128_aaaa_vvoo"))(tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.50 * tmps.at("1121_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1103_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1106_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1126_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1107_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.50 * tmps.at("1113_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1111_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1122_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1109_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1105_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1110_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1127_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1114_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1115_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1120_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) +=
        tmps.at("1118_aaaa_vvoo")(aa, ba, ia, ja))(tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja) +=
                                                   0.50 * tmps.at("1116_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1127_aaaa_vvoo"))
      .deallocate(tmps.at("1126_aaaa_vvoo"))
      .deallocate(tmps.at("1122_aaaa_vvoo"))
      .deallocate(tmps.at("1121_aaaa_vvoo"))
      .deallocate(tmps.at("1120_aaaa_vvoo"))
      .deallocate(tmps.at("1118_aaaa_vvoo"))
      .deallocate(tmps.at("1116_aaaa_vvoo"))
      .deallocate(tmps.at("1115_aaaa_vvoo"))
      .deallocate(tmps.at("1114_aaaa_vvoo"))
      .deallocate(tmps.at("1113_aaaa_vvoo"))
      .deallocate(tmps.at("1111_aaaa_vvoo"))
      .deallocate(tmps.at("1110_aaaa_vvoo"))
      .deallocate(tmps.at("1109_aaaa_vvoo"))
      .deallocate(tmps.at("1107_aaaa_vvoo"))
      .deallocate(tmps.at("1106_aaaa_vvoo"))
      .deallocate(tmps.at("1105_aaaa_vvoo"))
      .deallocate(tmps.at("1103_aaaa_vvoo"))
      .allocate(tmps.at("1133_aaaa_vvoo"))(tmps.at("1133_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1129_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1133_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1132_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1133_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1130_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1133_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1128_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1132_aaaa_vvoo"))
      .deallocate(tmps.at("1130_aaaa_vvoo"))
      .deallocate(tmps.at("1129_aaaa_vvoo"))
      .deallocate(tmps.at("1128_aaaa_vvoo"))
      .allocate(tmps.at("1136_aaaa_vvoo"))(tmps.at("1136_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1133_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1136_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1135_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1135_aaaa_vvoo"))
      .deallocate(tmps.at("1133_aaaa_vvoo"))(r2_1p.at("aaaa")(aa, ba, ia, ja) -=
                                             tmps.at("1136_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += tmps.at("1136_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1136_aaaa_vvoo"))
      .allocate(tmps.at("1260_bb_vo"))(tmps.at("1260_bb_vo")(ab, ib) =
                                         dp.at("bb_oo")(jb, ib) * t1_2p.at("bb")(ab, jb))(
        r1_1p.at("bb")(ab, ib) -= 2.00 * tmps.at("1260_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) -= 2.00 * t0_1p * tmps.at("1260_bb_vo")(ab, ib))
      .allocate(tmps.at("1320_aa_vo"))(tmps.at("1320_aa_vo")(aa, ia) =
                                         dp.at("aa_vv")(aa, ba) * t1_2p.at("aa")(ba, ia))(
        r1_1p.at("aa")(aa, ia) += 2.00 * tmps.at("1320_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) += 2.00 * t0_1p * tmps.at("1320_aa_vo")(aa, ia))
      .allocate(tmps.at("1326_bb_vo"))(tmps.at("1326_bb_vo")(ab, ib) =
                                         dp.at("bb_vv")(ab, bb) * t1_2p.at("bb")(bb, ib))(
        r1_1p.at("bb")(ab, ib) += 2.00 * tmps.at("1326_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) += 2.00 * t0_1p * tmps.at("1326_bb_vo")(ab, ib))
      .allocate(tmps.at("1334_bb_vo"))(tmps.at("1334_bb_vo")(ab, ib) =
                                         dp.at("aa_ov")(ja, ba) * t2_2p.at("abab")(ba, ab, ja, ib))
      .allocate(tmps.at("1333_bb_vo"))(tmps.at("1333_bb_vo")(ab, ib) =
                                         dp.at("bb_ov")(jb, bb) * t2_2p.at("bbbb")(bb, ab, ib, jb))
      .allocate(tmps.at("1335_bb_vo"))(tmps.at("1335_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("1333_bb_vo")(ab, ib))(
        tmps.at("1335_bb_vo")(ab, ib) += tmps.at("1334_bb_vo")(ab, ib))
      .deallocate(tmps.at("1334_bb_vo"))
      .deallocate(tmps.at("1333_bb_vo"))(r1_1p.at("bb")(ab, ib) +=
                                         2.00 * tmps.at("1335_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) += 2.00 * t0_1p * tmps.at("1335_bb_vo")(ab, ib))
      .allocate(tmps.at("1354_aa_vo"))(tmps.at("1354_aa_vo")(aa, ia) =
                                         dp.at("aa_oo")(ja, ia) * t1_2p.at("aa")(aa, ja))(
        r1_1p.at("aa")(aa, ia) -= 2.00 * tmps.at("1354_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 2.00 * t0_1p * tmps.at("1354_aa_vo")(aa, ia))
      .allocate(tmps.at("1364_aa_vo"))(tmps.at("1364_aa_vo")(aa, ia) =
                                         dp.at("aa_ov")(ja, ba) * t2_2p.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("1363_aa_vo"))(tmps.at("1363_aa_vo")(aa, ia) =
                                         dp.at("bb_ov")(jb, bb) * t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("1365_aa_vo"))(tmps.at("1365_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("1363_aa_vo")(aa, ia))(
        tmps.at("1365_aa_vo")(aa, ia) += tmps.at("1364_aa_vo")(aa, ia))
      .deallocate(tmps.at("1364_aa_vo"))
      .deallocate(tmps.at("1363_aa_vo"))(r1_1p.at("aa")(aa, ia) -=
                                         2.00 * tmps.at("1365_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 2.00 * t0_1p * tmps.at("1365_aa_vo")(aa, ia))
      .allocate(tmps.at("1393_aa_vo"))(tmps.at("1393_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0022_aa_oo")(ja, ia))
      .deallocate(tmps.at("0022_aa_oo"))(r1_1p.at("aa")(aa, ia) -=
                                         2.00 * tmps.at("1393_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 2.00 * t0_1p * tmps.at("1393_aa_vo")(aa, ia))
      .allocate(tmps.at("1396_bb_vo"))(tmps.at("1396_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0016_bb_oo")(jb, ib))(
        r1_1p.at("bb")(ab, ib) -= 2.00 * tmps.at("1396_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) -= 2.00 * t0_1p * tmps.at("1396_bb_vo")(ab, ib))
      .allocate(tmps.at("1399_bb_vo"))(tmps.at("1399_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(ab, jb) * tmps.at("0009_bb_oo")(jb, ib))
      .deallocate(tmps.at("0009_bb_oo"))(r1_1p.at("bb")(ab, ib) -=
                                         2.00 * tmps.at("1399_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) -= 2.00 * t0_1p * tmps.at("1399_bb_vo")(ab, ib))
      .allocate(tmps.at("1406_bb_vo"))(tmps.at("1406_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0015_bb_oo")(jb, ib))(
        r1_1p.at("bb")(ab, ib) -= 2.00 * tmps.at("1406_bb_vo")(ab, ib))(
        r1_2p.at("bb")(ab, ib) -= 2.00 * t0_1p * tmps.at("1406_bb_vo")(ab, ib))
      .allocate(tmps.at("1408_aa_vo"))(tmps.at("1408_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0023_aa_oo")(ja, ia))(
        r1_1p.at("aa")(aa, ia) -= 2.00 * tmps.at("1408_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 2.00 * t0_1p * tmps.at("1408_aa_vo")(aa, ia))
      .allocate(tmps.at("1411_aa_vo"))(tmps.at("1411_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0031_aa_oo")(ja, ia))(
        r1_1p.at("aa")(aa, ia) -= 2.00 * tmps.at("1411_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 2.00 * t0_1p * tmps.at("1411_aa_vo")(aa, ia))
      .allocate(tmps.at("1185_abab_oooo"))(tmps.at("1185_abab_oooo")(ia, jb, ka, lb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0601_abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("1184_abab_oooo"))(tmps.at("1184_abab_oooo")(ia, jb, ka, lb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0593_abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("1183_abab_oooo"))(tmps.at("1183_abab_oooo")(ia, jb, ka, lb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0286_abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("1382_abab_oooo"))(tmps.at("1382_abab_oooo")(ia, jb, ka, lb) =
                                             tmps.at("1183_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("1382_abab_oooo")(ia, jb, ka, lb) += tmps.at("1184_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("1382_abab_oooo")(ia, jb, ka, lb) += tmps.at("1185_abab_oooo")(ia, jb, ka, lb))
      .deallocate(tmps.at("1185_abab_oooo"))
      .deallocate(tmps.at("1184_abab_oooo"))
      .deallocate(tmps.at("1183_abab_oooo"))
      .allocate(tmps.at("1392_baab_vooo"))(tmps.at("1392_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("1382_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1391_baab_vooo"))(tmps.at("1391_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("1078_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1390_baab_vooo"))(tmps.at("1390_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("bb")(ab, lb) *
                                             tmps.at("0875_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1438_baab_vooo"))(tmps.at("1438_baab_vooo")(ab, ia, ja, kb) =
                                             tmps.at("1390_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1438_baab_vooo")(ab, ia, ja, kb) += tmps.at("1391_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1438_baab_vooo")(ab, ia, ja, kb) += tmps.at("1392_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("1392_baab_vooo"))
      .deallocate(tmps.at("1391_baab_vooo"))
      .deallocate(tmps.at("1390_baab_vooo"))
      .allocate(tmps.at("1440_abab_vvoo"))(tmps.at("1440_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1438_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("1438_baab_vooo"))
      .allocate(tmps.at("1240_aa_ov"))(tmps.at("1240_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                         t1_2p.at("aa")(ba, ja))
      .allocate(tmps.at("1241_baab_vooo"))(tmps.at("1241_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("1240_aa_ov")(ia, ba))
      .allocate(tmps.at("1238_aaaa_oovo"))(tmps.at("1238_aaaa_oovo")(ia, ja, aa, ka) =
                                             eri.at("aaaa_oovv")(ia, ja, ba, aa) *
                                             t1_1p.at("aa")(ba, ka))
      .allocate(tmps.at("1239_baba_vooo"))(tmps.at("1239_baba_vooo")(ab, ia, jb, ka) =
                                             t2_1p.at("abab")(ba, ab, la, jb) *
                                             tmps.at("1238_aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("1237_baab_vooo"))(tmps.at("1237_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0354_aa_ov")(ia, ba))
      .allocate(tmps.at("1236_baab_vooo"))(tmps.at("1236_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0976_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1235_baba_vooo"))(tmps.at("1235_baba_vooo")(ab, ia, jb, ka) =
                                             t2_2p.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0203_aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("1140_abab_oooo"))(tmps.at("1140_abab_oooo")(ia, jb, ka, lb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             eri.at("abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("1139_abab_oooo"))(tmps.at("1139_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abba_oovo")(ia, jb, ab, ka) *
                                             t1_2p.at("bb")(ab, lb))
      .allocate(tmps.at("1138_abab_oooo"))(tmps.at("1138_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t2_2p.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("1179_abab_oooo"))(tmps.at("1179_abab_oooo")(ia, jb, ka, lb) =
                                             -1.00 * tmps.at("1138_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("1179_abab_oooo")(ia, jb, ka, lb) -= tmps.at("1140_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("1179_abab_oooo")(ia, jb, ka, lb) += tmps.at("1139_abab_oooo")(ia, jb, ka, lb))
      .deallocate(tmps.at("1140_abab_oooo"))
      .deallocate(tmps.at("1139_abab_oooo"))
      .deallocate(tmps.at("1138_abab_oooo"))
      .allocate(tmps.at("1234_baab_vooo"))(tmps.at("1234_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("1179_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1233_baab_vooo"))(tmps.at("1233_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("aa")(ba, ja) *
                                             tmps.at("0135_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("1230_aa_ov"))(tmps.at("1230_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ia, ja, ba, aa) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("1231_baab_vooo"))(tmps.at("1231_baab_vooo")(ab, ia, ja, kb) =
                                             t2_1p.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("1230_aa_ov")(ia, ba))
      .allocate(tmps.at("1229_baab_vooo"))(tmps.at("1229_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0536_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("1228_baab_vooo"))(tmps.at("1228_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0557_baba_voov")(ab, ia, kb, ba))
      .allocate(tmps.at("1227_baab_vooo"))(tmps.at("1227_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("bb")(ab, lb) *
                                             tmps.at("0831_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1224_baba_vooo"))(tmps.at("1224_baba_vooo")(ab, ia, jb, ka) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0428_aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("1223_baab_vooo"))(tmps.at("1223_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, lb) *
                                             tmps.at("0601_abab_oovo")(ia, lb, ba, kb))
      .allocate(tmps.at("1222_baab_vooo"))(tmps.at("1222_baab_vooo")(ab, ia, ja, kb) =
                                             t2_1p.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0365_aa_ov")(ia, ba))
      .allocate(tmps.at("1221_baab_vooo"))(tmps.at("1221_baab_vooo")(ab, ia, ja, kb) =
                                             t2_2p.at("abab")(ba, ab, ja, lb) *
                                             tmps.at("0286_abab_oovo")(ia, lb, ba, kb))
      .allocate(tmps.at("1220_baab_vooo"))(tmps.at("1220_baab_vooo")(ab, ia, ja, kb) =
                                             t2_1p.at("abab")(ba, ab, ja, lb) *
                                             tmps.at("0593_abab_oovo")(ia, lb, ba, kb))
      .allocate(tmps.at("1219_baab_vooo"))(tmps.at("1219_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("bb")(ab, lb) *
                                             tmps.at("0819_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1218_baab_vooo"))(tmps.at("1218_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0133_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("1217_baab_vooo"))(tmps.at("1217_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("aa")(ba, ja) *
                                             tmps.at("0033_baba_voov")(ab, ia, kb, ba))
      .allocate(tmps.at("1216_baab_vooo"))(tmps.at("1216_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0035_baba_voov")(ab, ia, kb, ba))
      .allocate(tmps.at("1213_baab_vooo"))(tmps.at("1213_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0994_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1387_baab_vooo"))(tmps.at("1387_baab_vooo")(ab, ia, ja, kb) =
                                             -1.00 * tmps.at("1213_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1219_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) += tmps.at("1222_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) += tmps.at("1237_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1223_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1224_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) += tmps.at("1234_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) += tmps.at("1235_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1228_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1229_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1227_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1221_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1220_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) += tmps.at("1236_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) += tmps.at("1241_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) += tmps.at("1239_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1218_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1217_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1216_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1231_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1387_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1233_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("1241_baab_vooo"))
      .deallocate(tmps.at("1239_baba_vooo"))
      .deallocate(tmps.at("1237_baab_vooo"))
      .deallocate(tmps.at("1236_baab_vooo"))
      .deallocate(tmps.at("1235_baba_vooo"))
      .deallocate(tmps.at("1234_baab_vooo"))
      .deallocate(tmps.at("1233_baab_vooo"))
      .deallocate(tmps.at("1231_baab_vooo"))
      .deallocate(tmps.at("1229_baab_vooo"))
      .deallocate(tmps.at("1228_baab_vooo"))
      .deallocate(tmps.at("1227_baab_vooo"))
      .deallocate(tmps.at("1224_baba_vooo"))
      .deallocate(tmps.at("1223_baab_vooo"))
      .deallocate(tmps.at("1222_baab_vooo"))
      .deallocate(tmps.at("1221_baab_vooo"))
      .deallocate(tmps.at("1220_baab_vooo"))
      .deallocate(tmps.at("1219_baab_vooo"))
      .deallocate(tmps.at("1218_baab_vooo"))
      .deallocate(tmps.at("1217_baab_vooo"))
      .deallocate(tmps.at("1216_baab_vooo"))
      .deallocate(tmps.at("1213_baab_vooo"))
      .allocate(tmps.at("1437_abab_vvoo"))(tmps.at("1437_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1387_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("1387_baab_vooo"))
      .allocate(tmps.at("1212_abab_vooo"))(tmps.at("1212_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("aaaa")(ba, aa, ja, la) *
                                             tmps.at("0601_abab_oovo")(la, ib, ba, kb))
      .allocate(tmps.at("1209_bb_ov"))(tmps.at("1209_bb_ov")(ib, ab) =
                                         eri.at("abab_oovv")(ja, ib, ba, ab) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("1208_bb_ov"))(tmps.at("1208_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(ib, jb, bb, ab) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("1210_bb_ov"))(tmps.at("1210_bb_ov")(ib, ab) =
                                         -1.00 * tmps.at("1209_bb_ov")(ib, ab))(
        tmps.at("1210_bb_ov")(ib, ab) += tmps.at("1208_bb_ov")(ib, ab))
      .deallocate(tmps.at("1209_bb_ov"))
      .deallocate(tmps.at("1208_bb_ov"))
      .allocate(tmps.at("1211_abab_vooo"))(tmps.at("1211_abab_vooo")(aa, ib, ja, kb) =
                                             t2_1p.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("1210_bb_ov")(ib, bb))
      .allocate(tmps.at("1205_abab_vooo"))(tmps.at("1205_abab_vooo")(aa, ib, ja, kb) =
                                             t2_1p.at("aaaa")(ba, aa, ja, la) *
                                             tmps.at("0593_abab_oovo")(la, ib, ba, kb))
      .allocate(tmps.at("1204_abab_vooo"))(tmps.at("1204_abab_vooo")(aa, ib, ja, kb) =
                                             t1_2p.at("aa")(ba, ja) *
                                             tmps.at("0816_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("1202_abab_vooo"))(tmps.at("1202_abab_vooo")(aa, ib, ja, kb) =
                                             t1_2p.at("aa")(ba, ja) *
                                             tmps.at("0811_abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("1199_bb_ov"))(tmps.at("1199_bb_ov")(ib, ab) =
                                         eri.at("abab_oovv")(ja, ib, ba, ab) *
                                         t1_2p.at("aa")(ba, ja))
      .allocate(tmps.at("1198_bb_ov"))(tmps.at("1198_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(ib, jb, ab, bb) *
                                         t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("1200_bb_ov"))(tmps.at("1200_bb_ov")(ib, ab) =
                                         tmps.at("1198_bb_ov")(ib, ab))(
        tmps.at("1200_bb_ov")(ib, ab) += tmps.at("1199_bb_ov")(ib, ab))
      .deallocate(tmps.at("1199_bb_ov"))
      .deallocate(tmps.at("1198_bb_ov"))
      .allocate(tmps.at("1201_abab_vooo"))(tmps.at("1201_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("1200_bb_ov")(ib, bb))
      .allocate(tmps.at("1197_abab_vooo"))(tmps.at("1197_abab_vooo")(aa, ib, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0940_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("1196_abab_vooo"))(tmps.at("1196_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, lb) *
                                             tmps.at("0600_bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("1195_abab_vooo"))(tmps.at("1195_abab_vooo")(aa, ib, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0999_abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("1193_bbbb_oovo"))(tmps.at("1193_bbbb_oovo")(ib, jb, ab, kb) =
                                             eri.at("bbbb_oovv")(ib, jb, bb, ab) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("1194_abab_vooo"))(tmps.at("1194_abab_vooo")(aa, ib, ja, kb) =
                                             t2_1p.at("abab")(aa, bb, ja, lb) *
                                             tmps.at("1193_bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("1192_abab_vooo"))(tmps.at("1192_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("aaaa")(ba, aa, ja, la) *
                                             tmps.at("0286_abab_oovo")(la, ib, ba, kb))
      .allocate(tmps.at("1137_abba_voov"))(tmps.at("1137_abba_voov")(aa, ib, jb, ba) =
                                             t2_2p.at("abab")(aa, cb, ka, jb) *
                                             eri.at("abab_oovv")(ka, ib, ba, cb))
      .allocate(tmps.at("1191_abab_vooo"))(tmps.at("1191_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("1137_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("1188_abba_vooo"))(tmps.at("1188_abba_vooo")(aa, ib, jb, ka) =
                                             tmps.at("0230_abba_vovo")(aa, ib, bb, ka) *
                                             t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("1186_abab_vooo"))(tmps.at("1186_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("abab")(aa, bb, ja, lb) *
                                             tmps.at("0256_bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("1388_abab_vooo"))(tmps.at("1388_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("1186_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1194_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1202_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) += tmps.at("1196_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) += tmps.at("1197_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) += tmps.at("1205_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) += tmps.at("1204_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1188_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) += tmps.at("1211_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) += tmps.at("1212_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1195_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) += tmps.at("1192_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1201_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1388_abab_vooo")(aa, ib, ja, kb) += tmps.at("1191_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("1212_abab_vooo"))
      .deallocate(tmps.at("1211_abab_vooo"))
      .deallocate(tmps.at("1205_abab_vooo"))
      .deallocate(tmps.at("1204_abab_vooo"))
      .deallocate(tmps.at("1202_abab_vooo"))
      .deallocate(tmps.at("1201_abab_vooo"))
      .deallocate(tmps.at("1197_abab_vooo"))
      .deallocate(tmps.at("1196_abab_vooo"))
      .deallocate(tmps.at("1195_abab_vooo"))
      .deallocate(tmps.at("1194_abab_vooo"))
      .deallocate(tmps.at("1192_abab_vooo"))
      .deallocate(tmps.at("1191_abab_vooo"))
      .deallocate(tmps.at("1188_abba_vooo"))
      .deallocate(tmps.at("1186_abab_vooo"))
      .allocate(tmps.at("1436_baab_vvoo"))(tmps.at("1436_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1388_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("1388_abab_vooo"))
      .allocate(tmps.at("1435_abab_vvoo"))(tmps.at("1435_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0590_bb_oo")(kb, jb))
      .allocate(tmps.at("1434_abab_vvoo"))(tmps.at("1434_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1091_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("1091_baab_vooo"))
      .allocate(tmps.at("1433_abab_vvoo"))(tmps.at("1433_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1098_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("1098_baab_vooo"))
      .allocate(tmps.at("1432_abab_vvoo"))(tmps.at("1432_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0884_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0884_abab_ovoo"))
      .allocate(tmps.at("1431_baab_vvoo"))(tmps.at("1431_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1084_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("1084_abab_vooo"))
      .allocate(tmps.at("1226_baab_vooo"))(tmps.at("1226_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0137_baba_voov")(ab, ia, kb, ba))
      .allocate(tmps.at("1214_aa_ov"))(tmps.at("1214_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ja, ia, aa, ba) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("1215_baab_vooo"))(tmps.at("1215_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("1214_aa_ov")(ia, ba))
      .allocate(tmps.at("1385_baab_vooo"))(tmps.at("1385_baab_vooo")(ab, ia, ja, kb) =
                                             -1.00 * tmps.at("1226_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1385_baab_vooo")(ab, ia, ja, kb) += tmps.at("1215_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("1226_baab_vooo"))
      .deallocate(tmps.at("1215_baab_vooo"))
      .allocate(tmps.at("1430_abab_vvoo"))(tmps.at("1430_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1385_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("1385_baab_vooo"))
      .allocate(tmps.at("1429_abab_vvoo"))(tmps.at("1429_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0604_bb_oo")(kb, jb))
      .allocate(tmps.at("1428_abba_vvoo"))(tmps.at("1428_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0413_aa_oo")(ka, ja))
      .allocate(tmps.at("1427_abab_vvoo"))(tmps.at("1427_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0608_bb_oo")(kb, jb))
      .allocate(tmps.at("1426_abab_vvoo"))(tmps.at("1426_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0289_bb_oo")(kb, jb))
      .allocate(tmps.at("1232_baab_vooo"))(tmps.at("1232_baab_vooo")(ab, ia, ja, kb) =
                                             t2_2p.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0232_aa_ov")(ia, ba))
      .allocate(tmps.at("1225_baab_vooo"))(tmps.at("1225_baab_vooo")(ab, ia, ja, kb) =
                                             t2_2p.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0224_aa_ov")(ia, ba))
      .allocate(tmps.at("1386_baab_vooo"))(tmps.at("1386_baab_vooo")(ab, ia, ja, kb) =
                                             tmps.at("1225_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1386_baab_vooo")(ab, ia, ja, kb) += tmps.at("1232_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("1232_baab_vooo"))
      .deallocate(tmps.at("1225_baab_vooo"))
      .allocate(tmps.at("1425_abab_vvoo"))(tmps.at("1425_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1386_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("1386_baab_vooo"))
      .allocate(tmps.at("1424_abab_vvoo"))(tmps.at("1424_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("1078_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("1078_abab_oooo"));
  }
}

template void exachem::cc::qed_ccsd_os::resid_5<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);