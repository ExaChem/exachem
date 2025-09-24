/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_cs_resid_3.hpp"

template<typename T>
void exachem::cc::qed_ccsd_cs::resid_part3(
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
      .allocate(tmps.at("0787_abab_vooo"))(tmps.at("0787_abab_vooo")(aa, ib, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0096_abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("0786_abab_vooo"))(tmps.at("0786_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0519_abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("0785_abab_vooo"))(tmps.at("0785_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0516_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("0784_abba_vooo"))(tmps.at("0784_abba_vooo")(aa, ib, jb, ka) =
                                             t1.at("bb")(bb, jb) *
                                             tmps.at("0280_abab_voov")(aa, ib, ka, bb))
      .allocate(tmps.at("0783_abba_vooo"))(tmps.at("0783_abba_vooo")(aa, ib, jb, ka) =
                                             t1.at("bb")(bb, jb) *
                                             tmps.at("0257_abab_voov")(aa, ib, ka, bb))
      .allocate(tmps.at("0788_abab_vooo"))(tmps.at("0788_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("0783_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0788_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0785_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0788_abab_vooo")(aa, ib, ja, kb) += tmps.at("0786_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0788_abab_vooo")(aa, ib, ja, kb) += tmps.at("0787_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0788_abab_vooo")(aa, ib, ja, kb) += tmps.at("0784_abba_vooo")(aa, ib, kb, ja))
      .deallocate(tmps.at("0787_abab_vooo"))
      .deallocate(tmps.at("0786_abab_vooo"))
      .deallocate(tmps.at("0785_abab_vooo"))
      .deallocate(tmps.at("0784_abba_vooo"))
      .deallocate(tmps.at("0783_abba_vooo"))
      .allocate(tmps.at("0789_baab_vvoo"))(tmps.at("0789_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0788_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0782_abba_vvoo"))(tmps.at("0782_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0303_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .deallocate(tmps.at("0303_aa_oo"))
      .allocate(tmps.at("0781_baba_vvoo"))(tmps.at("0781_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0723_aa_vo")(ba, ja) * t1_1p.at("bb")(ab, ib))
      .deallocate(tmps.at("0723_aa_vo"))
      .allocate(tmps.at("0514_abba_vooo"))(tmps.at("0514_abba_vooo")(aa, ib, jb, ka) =
                                             t1_1p.at("bb")(bb, jb) *
                                             tmps.at("0112_abab_voov")(aa, ib, ka, bb))
      .allocate(tmps.at("0509_bb_ov"))(tmps.at("0509_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(jb, ib, ab, bb) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0510_abab_vooo"))(tmps.at("0510_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("0509_bb_ov")(ib, bb))
      .deallocate(tmps.at("0509_bb_ov"))
      .allocate(tmps.at("0727_abab_vooo"))(tmps.at("0727_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("0514_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0727_abab_vooo")(aa, ib, ja, kb) += tmps.at("0510_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("0514_abba_vooo"))
      .deallocate(tmps.at("0510_abab_vooo"))
      .allocate(tmps.at("0780_baab_vvoo"))(tmps.at("0780_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0727_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .deallocate(tmps.at("0727_abab_vooo"))
      .allocate(tmps.at("0777_baab_vooo"))(tmps.at("0777_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0549_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0776_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0195_abab_oovo")(ia, lb, ba, kb))(
        tmps.at("0776_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1.at("bb")(ab, lb))
      .allocate(tmps.at("0775_abab_ovoo"))(tmps.at("0775_abab_ovoo")(ia, ab, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0089_abab_ovvo")(ia, ab, ba, kb))
      .allocate(tmps.at("0778_abab_ovoo"))(tmps.at("0778_abab_ovoo")(ia, ab, ja, kb) =
                                             -1.00 * tmps.at("0777_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0778_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0776_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0778_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0775_abab_ovoo")(ia, ab, ja, kb))
      .deallocate(tmps.at("0777_baab_vooo"))
      .deallocate(tmps.at("0776_abab_ovoo"))
      .deallocate(tmps.at("0775_abab_ovoo"))
      .allocate(tmps.at("0779_abab_vvoo"))(tmps.at("0779_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0778_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0529_abab_vooo"))(tmps.at("0529_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("0255_bb_ov")(ib, bb))
      .allocate(tmps.at("0521_abab_vooo"))(tmps.at("0521_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("0198_bb_ov")(ib, bb))
      .allocate(tmps.at("0729_abab_vooo"))(tmps.at("0729_abab_vooo")(aa, ib, ja, kb) =
                                             tmps.at("0521_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0729_abab_vooo")(aa, ib, ja, kb) += tmps.at("0529_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("0529_abab_vooo"))
      .deallocate(tmps.at("0521_abab_vooo"))
      .allocate(tmps.at("0774_baab_vvoo"))(tmps.at("0774_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0729_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .deallocate(tmps.at("0729_abab_vooo"))
      .allocate(tmps.at("0040_bb_oo"))(tmps.at("0040_bb_oo")(ib, jb) =
                                         dp.at("bb_ov")(ib, ab) * t1_1p.at("bb")(ab, jb))
      .allocate(tmps.at("0507_bb_vo"))(tmps.at("0507_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0040_bb_oo")(jb, ib))
      .allocate(tmps.at("0041_bb_oo"))(tmps.at("0041_bb_oo")(ib, jb) =
                                         dp.at("bb_ov")(ib, ab) * t1_2p.at("bb")(ab, jb))
      .allocate(tmps.at("0506_bb_vo"))(tmps.at("0506_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0041_bb_oo")(jb, ib))
      .allocate(tmps.at("0505_bb_vo"))(tmps.at("0505_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(ab, jb) * tmps.at("0039_bb_oo")(jb, ib))
      .allocate(tmps.at("0726_bb_vo"))(tmps.at("0726_bb_vo")(ab, ib) =
                                         tmps.at("0505_bb_vo")(ab, ib))(
        tmps.at("0726_bb_vo")(ab, ib) += tmps.at("0507_bb_vo")(ab, ib))(
        tmps.at("0726_bb_vo")(ab, ib) += tmps.at("0506_bb_vo")(ab, ib))
      .deallocate(tmps.at("0507_bb_vo"))
      .deallocate(tmps.at("0506_bb_vo"))
      .deallocate(tmps.at("0505_bb_vo"))
      .allocate(tmps.at("0773_abab_vvoo"))(tmps.at("0773_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0726_bb_vo")(bb, jb))
      .deallocate(tmps.at("0726_bb_vo"))
      .allocate(tmps.at("0772_abba_vvoo"))(tmps.at("0772_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0313_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0558_baab_vooo"))(tmps.at("0558_baab_vooo")(ab, ia, ja, kb) =
                                             t2_2p.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0214_aa_ov")(ia, ba))
      .deallocate(tmps.at("0214_aa_ov"))
      .allocate(tmps.at("0555_baab_vooo"))(tmps.at("0555_baab_vooo")(ab, ia, ja, kb) =
                                             t2_2p.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0284_aa_ov")(ia, ba))
      .deallocate(tmps.at("0284_aa_ov"))
      .allocate(tmps.at("0732_baab_vooo"))(tmps.at("0732_baab_vooo")(ab, ia, ja, kb) =
                                             tmps.at("0555_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0732_baab_vooo")(ab, ia, ja, kb) += tmps.at("0558_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("0558_baab_vooo"))
      .deallocate(tmps.at("0555_baab_vooo"))
      .allocate(tmps.at("0771_abab_vvoo"))(tmps.at("0771_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0732_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0732_baab_vooo"))
      .allocate(tmps.at("0501_bb_oo"))(tmps.at("0501_bb_oo")(ib, jb) =
                                         t1_1p.at("bb")(ab, kb) *
                                         tmps.at("0079_bbbb_oovo")(kb, ib, ab, jb))
      .allocate(tmps.at("0770_abab_vvoo"))(tmps.at("0770_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0501_bb_oo")(kb, jb))
      .deallocate(tmps.at("0501_bb_oo"))
      .allocate(tmps.at("0769_abba_vvoo"))(tmps.at("0769_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0308_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .deallocate(tmps.at("0308_aa_oo"))
      .allocate(tmps.at("0768_abba_vvoo"))(tmps.at("0768_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0217_aa_oo")(ka, ja) *
                                             t2_2p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0765_bb_oo"))(tmps.at("0765_bb_oo")(ib, jb) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0198_bb_ov")(ib, ab))
      .allocate(tmps.at("0764_bb_oo"))(tmps.at("0764_bb_oo")(ib, jb) =
                                         t1.at("aa")(aa, ka) *
                                         tmps.at("0491_abab_oovo")(ka, ib, aa, jb))
      .deallocate(tmps.at("0491_abab_oovo"))
      .allocate(tmps.at("0766_bb_oo"))(tmps.at("0766_bb_oo")(ib, jb) =
                                         tmps.at("0764_bb_oo")(ib, jb))(
        tmps.at("0766_bb_oo")(ib, jb) += tmps.at("0765_bb_oo")(ib, jb))
      .deallocate(tmps.at("0765_bb_oo"))
      .deallocate(tmps.at("0764_bb_oo"))
      .allocate(tmps.at("0767_abab_vvoo"))(tmps.at("0767_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0766_bb_oo")(kb, jb))
      .allocate(tmps.at("0763_baba_vvoo"))(tmps.at("0763_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0762_aa_vo")(ba, ja) * t1_2p.at("bb")(ab, ib))
      .allocate(tmps.at("0759_abba_vvoo"))(tmps.at("0759_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0736_abba_oooo")(ka, lb, ib, ja))
      .allocate(tmps.at("0758_abab_vvoo"))(tmps.at("0758_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0201_bb_oo")(kb, jb))
      .allocate(tmps.at("0502_bb_oo"))(tmps.at("0502_bb_oo")(ib, jb) =
                                         t1_2p.at("bb")(ab, jb) * tmps.at("0198_bb_ov")(ib, ab))
      .deallocate(tmps.at("0198_bb_ov"))
      .allocate(tmps.at("0499_bb_oo"))(tmps.at("0499_bb_oo")(ib, jb) =
                                         t1_2p.at("bb")(ab, jb) * tmps.at("0255_bb_ov")(ib, ab))
      .deallocate(tmps.at("0255_bb_ov"))
      .allocate(tmps.at("0725_bb_oo"))(tmps.at("0725_bb_oo")(ib, jb) =
                                         tmps.at("0499_bb_oo")(ib, jb))(
        tmps.at("0725_bb_oo")(ib, jb) += tmps.at("0502_bb_oo")(ib, jb))
      .deallocate(tmps.at("0502_bb_oo"))
      .deallocate(tmps.at("0499_bb_oo"))
      .allocate(tmps.at("0757_abab_vvoo"))(tmps.at("0757_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0725_bb_oo")(kb, jb))
      .deallocate(tmps.at("0725_bb_oo"))
      .allocate(tmps.at("0756_baab_vvoo"))(tmps.at("0756_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0207_abab_vooo")(ba, kb, ia, jb) *
                                             t1_2p.at("bb")(ab, kb))
      .allocate(tmps.at("0755_abab_vvoo"))(tmps.at("0755_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0722_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0722_abab_oooo"))
      .allocate(tmps.at("0752_bb_vo"))(tmps.at("0752_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0040_bb_oo")(jb, ib))
      .allocate(tmps.at("0751_bb_vo"))(tmps.at("0751_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0039_bb_oo")(jb, ib))
      .deallocate(tmps.at("0039_bb_oo"))
      .allocate(tmps.at("0753_bb_vo"))(tmps.at("0753_bb_vo")(ab, ib) =
                                         tmps.at("0751_bb_vo")(ab, ib))(
        tmps.at("0753_bb_vo")(ab, ib) += tmps.at("0752_bb_vo")(ab, ib))
      .deallocate(tmps.at("0752_bb_vo"))
      .deallocate(tmps.at("0751_bb_vo"))
      .allocate(tmps.at("0754_abab_vvoo"))(tmps.at("0754_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0753_bb_vo")(bb, jb))
      .allocate(tmps.at("0748_abab_vooo"))(tmps.at("0748_abab_vooo")(aa, ib, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0091_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("0747_abba_vooo"))(tmps.at("0747_abba_vooo")(aa, ib, jb, ka) =
                                             t1_1p.at("bb")(bb, jb) *
                                             tmps.at("0204_abab_voov")(aa, ib, ka, bb))
      .allocate(tmps.at("0749_abab_vooo"))(tmps.at("0749_abab_vooo")(aa, ib, ja, kb) =
                                             tmps.at("0747_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0749_abab_vooo")(aa, ib, ja, kb) += tmps.at("0748_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("0748_abab_vooo"))
      .deallocate(tmps.at("0747_abba_vooo"))
      .allocate(tmps.at("0750_baab_vvoo"))(tmps.at("0750_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0749_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0745_bb_oo"))(tmps.at("0745_bb_oo")(ib, jb) =
                                         t1_1p.at("aa")(aa, ka) *
                                         tmps.at("0195_abab_oovo")(ka, ib, aa, jb))
      .deallocate(tmps.at("0195_abab_oovo"))
      .allocate(tmps.at("0746_abab_vvoo"))(tmps.at("0746_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0745_bb_oo")(kb, jb))
      .allocate(tmps.at("0744_baab_vvoo"))(tmps.at("0744_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0209_abab_vooo")(ba, kb, ia, jb) *
                                             t1_2p.at("bb")(ab, kb))
      .allocate(tmps.at("0743_abab_vvoo"))(tmps.at("0743_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0211_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0742_abab_vvoo"))(tmps.at("0742_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0196_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0741_abba_vvoo"))(tmps.at("0741_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0301_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("0740_abab_vvoo"))(tmps.at("0740_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0734_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0721_abab_vvoo"))(tmps.at("0721_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0041_bb_oo")(kb, jb))
      .deallocate(tmps.at("0041_bb_oo"))
      .allocate(tmps.at("0720_baab_vvoo"))(tmps.at("0720_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0010_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0010_abab_vooo"))
      .allocate(tmps.at("0719_baab_vvoo"))(tmps.at("0719_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0516_abba_voov")(ba, kb, jb, ca))
      .allocate(tmps.at("0718_abba_vvoo"))(tmps.at("0718_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0277_aa_oo")(ka, ja))
      .allocate(tmps.at("0717_baba_vvoo"))(tmps.at("0717_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_2p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0132_aaaa_voov")(ba, ka, ja, ca))
      .allocate(tmps.at("0716_abba_vvoo"))(tmps.at("0716_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0282_aa_oo")(ka, ja))
      .deallocate(tmps.at("0282_aa_oo"))
      .allocate(tmps.at("0713_baab_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("abab_oovv")(la, ib, ca, bb) * t1.at("aa")(ca, la))(
        tmps.at("0713_baab_ovoo")(ib, aa, ja, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0712_baab_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("bbbb_oovv")(lb, ib, cb, bb) * t1.at("bb")(cb, lb))(
        tmps.at("0712_baab_ovoo")(ib, aa, ja, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0714_baab_ovoo"))(tmps.at("0714_baab_ovoo")(ib, aa, ja, kb) =
                                             tmps.at("0712_baab_ovoo")(ib, aa, ja, kb))(
        tmps.at("0714_baab_ovoo")(ib, aa, ja, kb) += tmps.at("0713_baab_ovoo")(ib, aa, ja, kb))
      .deallocate(tmps.at("0713_baab_ovoo"))
      .deallocate(tmps.at("0712_baab_ovoo"))
      .allocate(tmps.at("0715_baab_vvoo"))(tmps.at("0715_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0714_baab_ovoo")(kb, ba, ia, jb))
      .allocate(tmps.at("0711_baba_vvoo"))(tmps.at("0711_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_2p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0204_abab_voov")(ba, kb, ja, cb))
      .allocate(tmps.at("0452_bb_vo"))(tmps.at("0452_bb_vo")(ab, ib) =
                                         dp.at("bb_oo")(jb, ib) * t1_2p.at("bb")(ab, jb))
      .allocate(tmps.at("0451_bb_vo"))(tmps.at("0451_bb_vo")(ab, ib) =
                                         dp.at("aa_ov")(ja, ba) * t2_2p.at("abab")(ba, ab, ja, ib))
      .allocate(tmps.at("0450_bb_vo"))(tmps.at("0450_bb_vo")(ab, ib) =
                                         dp.at("bb_ov")(jb, bb) * t2_2p.at("bbbb")(bb, ab, ib, jb))
      .allocate(tmps.at("0449_bb_vo"))(tmps.at("0449_bb_vo")(ab, ib) =
                                         dp.at("bb_vv")(ab, bb) * t1_2p.at("bb")(bb, ib))
      .allocate(tmps.at("0453_bb_vo"))(tmps.at("0453_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("0449_bb_vo")(ab, ib))(
        tmps.at("0453_bb_vo")(ab, ib) += tmps.at("0452_bb_vo")(ab, ib))(
        tmps.at("0453_bb_vo")(ab, ib) -= tmps.at("0451_bb_vo")(ab, ib))(
        tmps.at("0453_bb_vo")(ab, ib) += tmps.at("0450_bb_vo")(ab, ib))
      .deallocate(tmps.at("0452_bb_vo"))
      .deallocate(tmps.at("0451_bb_vo"))
      .deallocate(tmps.at("0450_bb_vo"))
      .deallocate(tmps.at("0449_bb_vo"))
      .allocate(tmps.at("0710_abab_vvoo"))(tmps.at("0710_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0453_bb_vo")(bb, jb))
      .deallocate(tmps.at("0453_bb_vo"))
      .allocate(tmps.at("0707_baab_vooo"))(tmps.at("0707_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("abab_oooo")(ia, lb, ja, kb) *
                                             t1_1p.at("bb")(ab, lb))
      .allocate(tmps.at("0706_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             eri.at("abab_oovo")(ia, lb, ba, kb) *
                                             t1.at("aa")(ba, ja))(
        tmps.at("0706_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1_1p.at("bb")(ab, lb))
      .allocate(tmps.at("0705_baab_vooo"))(tmps.at("0705_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baab_vovo")(ab, ia, ba, kb) *
                                             t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0704_baab_vooo"))(tmps.at("0704_baab_vooo")(ab, ia, ja, kb) =
                                             f.at("aa_ov")(ia, ba) *
                                             t2_1p.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0703_baab_vooo"))(tmps.at("0703_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baba_vovo")(ab, ia, bb, ja) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("0702_baba_vooo"))(tmps.at("0702_baba_vooo")(ab, ia, jb, ka) =
                                             eri.at("aaaa_oovo")(ia, la, ba, ka) *
                                             t2_1p.at("abab")(ba, ab, la, jb))
      .allocate(tmps.at("0701_abab_ovoo"))(tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) =
                                             eri.at("aaaa_oovv")(ia, la, ca, ba) *
                                             t1.at("aa")(ca, ja))(
        tmps.at("0701_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) * t2_1p.at("abab")(ba, ab, la, kb))
      .allocate(tmps.at("0700_baab_vooo"))(tmps.at("0700_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("abab_oovo")(ia, lb, ba, kb) *
                                             t2_1p.at("abab")(ba, ab, ja, lb))
      .allocate(tmps.at("0699_abba_ovoo"))(tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) =
                                             eri.at("abab_oovv")(ia, lb, ba, cb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("0699_abba_ovoo")(ia, ab, jb, ka) =
          tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) * t2_1p.at("abab")(ba, ab, ka, lb))
      .allocate(tmps.at("0698_baba_vooo"))(tmps.at("0698_baba_vooo")(ab, ia, jb, ka) =
                                             eri.at("abba_oovo")(ia, lb, bb, ka) *
                                             t2_1p.at("bbbb")(bb, ab, jb, lb))
      .allocate(tmps.at("0697_abab_ovoo"))(tmps.at("bin_baab_vooo")(bb, ia, ja, lb) =
                                             eri.at("abab_oovv")(ia, lb, ca, bb) *
                                             t1.at("aa")(ca, ja))(
        tmps.at("0697_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_baab_vooo")(bb, ia, ja, lb) * t2_1p.at("bbbb")(bb, ab, kb, lb))
      .allocate(tmps.at("0696_baab_vooo"))(tmps.at("0696_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t2_1p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0708_baab_vooo"))(tmps.at("0708_baab_vooo")(ab, ia, ja, kb) =
                                             -1.00 * tmps.at("0703_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) -= tmps.at("0698_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) -= tmps.at("0701_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) += tmps.at("0696_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) += tmps.at("0699_abba_ovoo")(ia, ab, kb, ja))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) += tmps.at("0700_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) += tmps.at("0707_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) += tmps.at("0697_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) += tmps.at("0702_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) -= tmps.at("0704_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) += tmps.at("0706_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0708_baab_vooo")(ab, ia, ja, kb) += tmps.at("0705_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("0707_baab_vooo"))
      .deallocate(tmps.at("0706_abab_ovoo"))
      .deallocate(tmps.at("0705_baab_vooo"))
      .deallocate(tmps.at("0704_baab_vooo"))
      .deallocate(tmps.at("0703_baab_vooo"))
      .deallocate(tmps.at("0702_baba_vooo"))
      .deallocate(tmps.at("0701_abab_ovoo"))
      .deallocate(tmps.at("0700_baab_vooo"))
      .deallocate(tmps.at("0699_abba_ovoo"))
      .deallocate(tmps.at("0698_baba_vooo"))
      .deallocate(tmps.at("0697_abab_ovoo"))
      .deallocate(tmps.at("0696_baab_vooo"))
      .allocate(tmps.at("0709_abab_vvoo"))(tmps.at("0709_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0708_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0695_abab_vvoo"))(tmps.at("0695_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0085_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("0484_baab_vooo"))(tmps.at("0484_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baba_vovo")(ab, ia, bb, ja) *
                                             t1_2p.at("bb")(bb, kb))
      .allocate(tmps.at("0483_baba_vooo"))(tmps.at("0483_baba_vooo")(ab, ia, jb, ka) =
                                             eri.at("aaaa_oovo")(ia, la, ba, ka) *
                                             t2_2p.at("abab")(ba, ab, la, jb))
      .allocate(tmps.at("0482_abab_ovoo"))(tmps.at("bin_baab_vooo")(bb, ia, ja, lb) =
                                             eri.at("abab_oovv")(ia, lb, ca, bb) *
                                             t1_1p.at("aa")(ca, ja))(
        tmps.at("0482_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_baab_vooo")(bb, ia, ja, lb) * t2_1p.at("bbbb")(bb, ab, kb, lb))
      .allocate(tmps.at("0481_baab_vooo"))(tmps.at("0481_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baab_vovo")(ab, ia, ba, kb) *
                                             t1_2p.at("aa")(ba, ja))
      .allocate(tmps.at("0480_baab_vooo"))(tmps.at("0480_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("abab_oovo")(ia, lb, ba, kb) *
                                             t2_2p.at("abab")(ba, ab, ja, lb))
      .allocate(tmps.at("0479_baab_vooo"))(tmps.at("0479_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("abab_oooo")(ia, lb, ja, kb) *
                                             t1_2p.at("bb")(ab, lb))
      .allocate(tmps.at("0478_baab_vooo"))(tmps.at("0478_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t2_2p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0477_baba_vooo"))(tmps.at("0477_baba_vooo")(ab, ia, jb, ka) =
                                             eri.at("abba_oovo")(ia, lb, bb, ka) *
                                             t2_2p.at("bbbb")(bb, ab, jb, lb))
      .allocate(tmps.at("0476_baab_vooo"))(tmps.at("0476_baab_vooo")(ab, ia, ja, kb) =
                                             f.at("aa_ov")(ia, ba) *
                                             t2_2p.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0485_abab_ovoo"))(tmps.at("0485_abab_ovoo")(ia, ab, ja, kb) =
                                             -1.00 * tmps.at("0476_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0485_abab_ovoo")(ia, ab, ja, kb) -= tmps.at("0484_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0485_abab_ovoo")(ia, ab, ja, kb) -= tmps.at("0477_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("0485_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0480_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0485_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0482_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0485_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0481_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0485_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0478_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0485_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0479_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0485_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0483_baba_vooo")(ab, ia, kb, ja))
      .deallocate(tmps.at("0484_baab_vooo"))
      .deallocate(tmps.at("0483_baba_vooo"))
      .deallocate(tmps.at("0482_abab_ovoo"))
      .deallocate(tmps.at("0481_baab_vooo"))
      .deallocate(tmps.at("0480_baab_vooo"))
      .deallocate(tmps.at("0479_baab_vooo"))
      .deallocate(tmps.at("0478_baab_vooo"))
      .deallocate(tmps.at("0477_baba_vooo"))
      .deallocate(tmps.at("0476_baab_vooo"))
      .allocate(tmps.at("0694_abab_vvoo"))(tmps.at("0694_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0485_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0485_abab_ovoo"))
      .allocate(tmps.at("0691_abab_vooo"))(tmps.at("0691_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_vovo")(aa, ib, ba, kb) *
                                             t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0690_abab_vooo"))(tmps.at("0690_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abba_vovo")(aa, ib, bb, ja) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("0689_abab_vooo"))(tmps.at("0689_abab_vooo")(aa, ib, ja, kb) =
                                             f.at("bb_ov")(ib, bb) *
                                             t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0688_abab_vooo"))(tmps.at("0688_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_oovo")(la, ib, ba, kb) *
                                             t2_1p.at("aaaa")(ba, aa, ja, la))
      .allocate(tmps.at("0687_abba_vooo"))(tmps.at("0687_abba_vooo")(aa, ib, jb, ka) =
                                             eri.at("abba_oovo")(la, ib, bb, ka) *
                                             t2_1p.at("abab")(aa, bb, la, jb))
      .allocate(tmps.at("0686_abab_vooo"))(tmps.at("0686_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("bbbb_oovo")(ib, lb, bb, kb) *
                                             t2_1p.at("abab")(aa, bb, ja, lb))
      .allocate(tmps.at("0685_abab_vooo"))(tmps.at("0685_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t2_1p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0692_abab_vooo"))(tmps.at("0692_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("0685_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0692_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0689_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0692_abab_vooo")(aa, ib, ja, kb) += tmps.at("0688_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0692_abab_vooo")(aa, ib, ja, kb) += tmps.at("0690_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0692_abab_vooo")(aa, ib, ja, kb) += tmps.at("0686_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0692_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0687_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0692_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0691_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("0691_abab_vooo"))
      .deallocate(tmps.at("0690_abab_vooo"))
      .deallocate(tmps.at("0689_abab_vooo"))
      .deallocate(tmps.at("0688_abab_vooo"))
      .deallocate(tmps.at("0687_abba_vooo"))
      .deallocate(tmps.at("0686_abab_vooo"))
      .deallocate(tmps.at("0685_abab_vooo"))
      .allocate(tmps.at("0693_baab_vvoo"))(tmps.at("0693_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0692_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0684_abba_vvoo"))(tmps.at("0684_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0296_aa_oo")(ka, ja))
      .deallocate(tmps.at("0296_aa_oo"))
      .allocate(tmps.at("0682_aaaa_voov"))(tmps.at("0682_aaaa_voov")(aa, ia, ja, ba) =
                                             eri.at("aaaa_oovv")(ka, ia, ca, ba) *
                                             t2_1p.at("aaaa")(ca, aa, ja, ka))
      .allocate(tmps.at("0683_baba_vvoo"))(tmps.at("0683_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0682_aaaa_voov")(ba, ka, ja, ca))
      .deallocate(tmps.at("0682_aaaa_voov"))
      .allocate(tmps.at("0474_baba_ovoo"))(tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) =
                                             eri.at("bbbb_oovv")(ib, lb, bb, cb) *
                                             t1_2p.at("bb")(cb, jb))(
        tmps.at("0474_baba_ovoo")(ib, aa, jb, ka) =
          tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) * t2.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0473_abab_vooo"))(tmps.at("0473_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("bbbb_oovo")(ib, lb, bb, kb) *
                                             t2_2p.at("abab")(aa, bb, ja, lb))
      .allocate(tmps.at("0472_abba_vooo"))(tmps.at("0472_abba_vooo")(aa, ib, jb, ka) =
                                             eri.at("abba_oovo")(la, ib, bb, ka) *
                                             t2_2p.at("abab")(aa, bb, la, jb))
      .allocate(tmps.at("0471_abab_vooo"))(tmps.at("0471_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abba_vovo")(aa, ib, bb, ja) *
                                             t1_2p.at("bb")(bb, kb))
      .allocate(tmps.at("0470_abab_vooo"))(tmps.at("0470_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t2_2p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0469_abab_vooo"))(tmps.at("0469_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_oovo")(la, ib, ba, kb) *
                                             t2_2p.at("aaaa")(ba, aa, ja, la))
      .allocate(tmps.at("0468_abab_vooo"))(tmps.at("0468_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_vovo")(aa, ib, ba, kb) *
                                             t1_2p.at("aa")(ba, ja))
      .allocate(tmps.at("0467_abab_vooo"))(tmps.at("0467_abab_vooo")(aa, ib, ja, kb) =
                                             f.at("bb_ov")(ib, bb) *
                                             t2_2p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0475_abab_vooo"))(tmps.at("0475_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("0467_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0475_abab_vooo")(aa, ib, ja, kb) += tmps.at("0469_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0475_abab_vooo")(aa, ib, ja, kb) += tmps.at("0473_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0475_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0470_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0475_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0472_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0475_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0468_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0475_abab_vooo")(aa, ib, ja, kb) += tmps.at("0471_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0475_abab_vooo")(aa, ib, ja, kb) += tmps.at("0474_baba_ovoo")(ib, aa, kb, ja))
      .deallocate(tmps.at("0474_baba_ovoo"))
      .deallocate(tmps.at("0473_abab_vooo"))
      .deallocate(tmps.at("0472_abba_vooo"))
      .deallocate(tmps.at("0471_abab_vooo"))
      .deallocate(tmps.at("0470_abab_vooo"))
      .deallocate(tmps.at("0469_abab_vooo"))
      .deallocate(tmps.at("0468_abab_vooo"))
      .deallocate(tmps.at("0467_abab_vooo"))
      .allocate(tmps.at("0681_baab_vvoo"))(tmps.at("0681_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0475_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0475_abab_vooo"))
      .allocate(tmps.at("0680_abab_vvoo"))(tmps.at("0680_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0120_bb_vv")(bb, cb))
      .allocate(tmps.at("0679_abab_vvoo"))(tmps.at("0679_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0431_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0431_abab_oooo"))
      .allocate(tmps.at("0677_aa_oo"))(tmps.at("0677_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovv")(ka, ia, aa, ba) *
                                         t2_1p.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0678_abba_vvoo"))(tmps.at("0678_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0677_aa_oo")(ka, ja))
      .deallocate(tmps.at("0677_aa_oo"))
      .allocate(tmps.at("0676_abba_vvoo"))(tmps.at("0676_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0244_aa_oo")(ka, ja))
      .allocate(tmps.at("0675_abab_vvoo"))(tmps.at("0675_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0182_bb_oo")(kb, jb))
      .allocate(tmps.at("0674_abab_vvoo"))(tmps.at("0674_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0531_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0673_abba_vvoo"))(tmps.at("0673_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_2p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0087_aa_oo")(ka, ja))
      .allocate(tmps.at("0671_baba_vovo"))(tmps.at("0671_baba_vovo")(ab, ia, bb, ja) =
                                             eri.at("baab_vovv")(ab, ia, ca, bb) *
                                             t1_1p.at("aa")(ca, ja))
      .allocate(tmps.at("0672_abba_vvoo"))(tmps.at("0672_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, cb, ka, ib) *
                                             tmps.at("0671_baba_vovo")(bb, ka, cb, ja))
      .allocate(tmps.at("0670_abab_vvoo"))(tmps.at("0670_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0114_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("0669_abab_vvoo"))(tmps.at("0669_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0541_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("0668_baab_vvoo"))(tmps.at("0668_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0159_baab_ovoo")(kb, ba, ia, jb))
      .allocate(tmps.at("0465_bb_vv"))(tmps.at("0465_bb_vv")(ab, bb) =
                                         eri.at("baab_vovv")(ab, ia, ca, bb) *
                                         t1_2p.at("aa")(ca, ia))
      .allocate(tmps.at("0464_bb_vv"))(tmps.at("0464_bb_vv")(ab, bb) =
                                         eri.at("abab_oovv")(ia, jb, ca, bb) *
                                         t2_2p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0463_bb_vv"))(tmps.at("0463_bb_vv")(ab, bb) =
                                         eri.at("bbbb_oovv")(ib, jb, bb, cb) *
                                         t2_2p.at("bbbb")(cb, ab, jb, ib))
      .allocate(tmps.at("0462_bb_vv"))(tmps.at("0462_bb_vv")(ab, bb) =
                                         eri.at("bbbb_vovv")(ab, ib, bb, cb) *
                                         t1_2p.at("bb")(cb, ib))
      .allocate(tmps.at("0466_bb_vv"))(tmps.at("0466_bb_vv")(ab, bb) =
                                         -0.50 * tmps.at("0463_bb_vv")(ab, bb))(
        tmps.at("0466_bb_vv")(ab, bb) -= tmps.at("0464_bb_vv")(ab, bb))(
        tmps.at("0466_bb_vv")(ab, bb) += tmps.at("0462_bb_vv")(ab, bb))(
        tmps.at("0466_bb_vv")(ab, bb) -= tmps.at("0465_bb_vv")(ab, bb))
      .deallocate(tmps.at("0465_bb_vv"))
      .deallocate(tmps.at("0464_bb_vv"))
      .deallocate(tmps.at("0463_bb_vv"))
      .deallocate(tmps.at("0462_bb_vv"))
      .allocate(tmps.at("0667_abab_vvoo"))(tmps.at("0667_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0466_bb_vv")(bb, cb))
      .deallocate(tmps.at("0466_bb_vv"))
      .allocate(tmps.at("0666_baab_vvoo"))(tmps.at("0666_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_2p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0110_aa_vv")(ba, ca))
      .allocate(tmps.at("0665_abba_vvoo"))(tmps.at("0665_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_2p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0124_aa_oo")(ka, ja))
      .allocate(tmps.at("0664_baab_vvoo"))(tmps.at("0664_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0148_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0663_abba_vvoo"))(tmps.at("0663_abba_vvoo")(aa, bb, ib, ja) =
                                             t1_2p.at("bb")(cb, ib) *
                                             tmps.at("0184_abba_vvvo")(aa, bb, cb, ja))
      .allocate(tmps.at("0662_baba_vvoo"))(tmps.at("0662_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_2p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0098_aaaa_vovo")(ba, ka, ca, ja))
      .allocate(tmps.at("0460_aa_vv"))(tmps.at("0460_aa_vv")(aa, ba) =
                                         eri.at("abab_vovv")(aa, ib, ba, cb) *
                                         t1_2p.at("bb")(cb, ib))
      .allocate(tmps.at("0459_aa_vv"))(tmps.at("0459_aa_vv")(aa, ba) =
                                         eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                         t1_2p.at("aa")(ca, ia))
      .allocate(tmps.at("0458_aa_vv"))(tmps.at("0458_aa_vv")(aa, ba) =
                                         eri.at("abab_oovv")(ia, jb, ba, cb) *
                                         t2_2p.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0457_aa_vv"))(tmps.at("0457_aa_vv")(aa, ba) =
                                         eri.at("aaaa_oovv")(ia, ja, ba, ca) *
                                         t2_2p.at("aaaa")(ca, aa, ja, ia))
      .allocate(tmps.at("0461_aa_vv"))(tmps.at("0461_aa_vv")(aa, ba) =
                                         -0.50 * tmps.at("0457_aa_vv")(aa, ba))(
        tmps.at("0461_aa_vv")(aa, ba) += tmps.at("0460_aa_vv")(aa, ba))(
        tmps.at("0461_aa_vv")(aa, ba) += tmps.at("0459_aa_vv")(aa, ba))(
        tmps.at("0461_aa_vv")(aa, ba) -= tmps.at("0458_aa_vv")(aa, ba))
      .deallocate(tmps.at("0460_aa_vv"))
      .deallocate(tmps.at("0459_aa_vv"))
      .deallocate(tmps.at("0458_aa_vv"))
      .deallocate(tmps.at("0457_aa_vv"))
      .allocate(tmps.at("0661_baab_vvoo"))(tmps.at("0661_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0461_aa_vv")(ba, ca))
      .deallocate(tmps.at("0461_aa_vv"))
      .allocate(tmps.at("0660_baab_vvoo"))(tmps.at("0660_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0527_abba_voov")(ba, kb, jb, ca))
      .deallocate(tmps.at("0527_abba_voov"))
      .allocate(tmps.at("0658_aa_vv"))(tmps.at("0658_aa_vv")(aa, ba) =
                                         eri.at("abab_vovv")(aa, ib, ba, cb) *
                                         t1_1p.at("bb")(cb, ib))
      .allocate(tmps.at("0659_baab_vvoo"))(tmps.at("0659_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0658_aa_vv")(ba, ca))
      .allocate(tmps.at("0657_abab_vvoo"))(tmps.at("0657_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0136_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0656_abab_vvoo"))(tmps.at("0656_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0130_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0655_abab_vvoo"))(tmps.at("0655_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0155_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0654_abab_vvoo"))(tmps.at("0654_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0169_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0652_aa_vv"))(tmps.at("0652_aa_vv")(aa, ba) =
                                         eri.at("aaaa_oovv")(ia, ja, ca, ba) *
                                         t2_1p.at("aaaa")(ca, aa, ja, ia))
      .allocate(tmps.at("0653_baab_vvoo"))(tmps.at("0653_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0652_aa_vv")(ba, ca))
      .deallocate(tmps.at("0652_aa_vv"))
      .allocate(tmps.at("0649_abab_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("aaaa_oovv")(la, ia, ca, ba) * t1.at("aa")(ca, la))(
        tmps.at("0649_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aa_vo")(ba, ia) * t2_1p.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0648_abab_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("abab_oovv")(ia, lb, ba, cb) * t1.at("bb")(cb, lb))(
        tmps.at("0648_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aa_vo")(ba, ia) * t2_1p.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0650_abab_ovoo"))(tmps.at("0650_abab_ovoo")(ia, ab, ja, kb) =
                                             tmps.at("0648_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0650_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0649_abab_ovoo")(ia, ab, ja, kb))
      .deallocate(tmps.at("0649_abab_ovoo"))
      .deallocate(tmps.at("0648_abab_ovoo"))
      .allocate(tmps.at("0651_abab_vvoo"))(tmps.at("0651_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0650_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0647_baab_vvoo"))(tmps.at("0647_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0275_aa_vv")(ba, ca))
      .deallocate(tmps.at("0275_aa_vv"))
      .allocate(tmps.at("0644_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             eri.at("abab_oovo")(ia, lb, ba, kb) *
                                             t1_1p.at("aa")(ba, ja))(
        tmps.at("0644_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1.at("bb")(ab, lb))
      .allocate(tmps.at("0643_abab_ovoo"))(tmps.at("bin_aa_vo")(ba, ia) =
                                             eri.at("abab_oovv")(ia, lb, ba, cb) *
                                             t1_1p.at("bb")(cb, lb))(
        tmps.at("0643_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aa_vo")(ba, ia) * t2.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0642_abba_ovoo"))(tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) =
                                             eri.at("abab_oovv")(ia, lb, ba, cb) *
                                             t1_1p.at("bb")(cb, jb))(
        tmps.at("0642_abba_ovoo")(ia, ab, jb, ka) =
          tmps.at("bin_aabb_vooo")(ba, ia, jb, lb) * t2.at("abab")(ba, ab, ka, lb))
      .allocate(tmps.at("0645_abab_ovoo"))(tmps.at("0645_abab_ovoo")(ia, ab, ja, kb) =
                                             -1.00 * tmps.at("0644_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0645_abab_ovoo")(ia, ab, ja, kb) -= tmps.at("0642_abba_ovoo")(ia, ab, kb, ja))(
        tmps.at("0645_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0643_abab_ovoo")(ia, ab, ja, kb))
      .deallocate(tmps.at("0644_abab_ovoo"))
      .deallocate(tmps.at("0643_abab_ovoo"))
      .deallocate(tmps.at("0642_abba_ovoo"))
      .allocate(tmps.at("0646_abab_vvoo"))(tmps.at("0646_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0645_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0640_bb_oo"))(tmps.at("0640_bb_oo")(ib, jb) =
                                         f.at("bb_ov")(ib, ab) * t1_1p.at("bb")(ab, jb))
      .allocate(tmps.at("0641_abab_vvoo"))(tmps.at("0641_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0640_bb_oo")(kb, jb))
      .allocate(tmps.at("0639_baab_vvoo"))(tmps.at("0639_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_2p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0161_aa_vv")(ba, ca))
      .allocate(tmps.at("0636_baba_vooo"))(tmps.at("0636_baba_vooo")(ab, ia, jb, ka) =
                                             eri.at("aaaa_oovo")(la, ia, ba, ka) *
                                             t2.at("abab")(ba, ab, la, jb))
      .allocate(tmps.at("0635_abab_ovoo"))(tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) =
                                             eri.at("aaaa_oovv")(la, ia, ca, ba) *
                                             t1.at("aa")(ca, ja))(
        tmps.at("0635_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) * t2.at("abab")(ba, ab, la, kb))
      .allocate(tmps.at("0637_abab_ovoo"))(tmps.at("0637_abab_ovoo")(ia, ab, ja, kb) =
                                             -1.00 * tmps.at("0636_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("0637_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0635_abab_ovoo")(ia, ab, ja, kb))
      .deallocate(tmps.at("0636_baba_vooo"))
      .deallocate(tmps.at("0635_abab_ovoo"))
      .allocate(tmps.at("0638_abab_vvoo"))(tmps.at("0638_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0637_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0634_abab_vvoo"))(tmps.at("0634_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0549_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0633_baab_vvoo"))(tmps.at("0633_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0519_abab_vovo")(ba, kb, ca, jb))
      .allocate(tmps.at("0632_baba_vvoo"))(tmps.at("0632_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_2p.at("bb")(ab, ib) * tmps.at("0631_aa_vo")(ba, ja))
      .allocate(tmps.at("0624_abab_vooo"))(tmps.at("0624_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("bbbb_oovo")(lb, ib, bb, kb) *
                                             t2.at("abab")(aa, bb, ja, lb))
      .allocate(tmps.at("0623_baba_ovoo"))(tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) =
                                             eri.at("bbbb_oovv")(lb, ib, cb, bb) *
                                             t1.at("bb")(cb, jb))(
        tmps.at("0623_baba_ovoo")(ib, aa, jb, ka) =
          tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) * t2.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0625_baba_ovoo"))(tmps.at("0625_baba_ovoo")(ib, aa, jb, ka) =
                                             -1.00 * tmps.at("0624_abab_vooo")(aa, ib, ka, jb))(
        tmps.at("0625_baba_ovoo")(ib, aa, jb, ka) += tmps.at("0623_baba_ovoo")(ib, aa, jb, ka))
      .deallocate(tmps.at("0624_abab_vooo"))
      .deallocate(tmps.at("0623_baba_ovoo"))
      .allocate(tmps.at("0626_baba_vvoo"))(tmps.at("0626_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0625_baba_ovoo")(kb, ba, ib, ja))
      .allocate(tmps.at("0622_abab_vvoo"))(tmps.at("0622_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0108_abab_oooo")(ka, lb, ia, jb))
      .allocate(tmps.at("0620_bb_vv"))(tmps.at("0620_bb_vv")(ab, bb) =
                                         eri.at("bbbb_oovv")(ib, jb, cb, bb) *
                                         t2.at("bbbb")(cb, ab, jb, ib))
      .allocate(tmps.at("0621_abab_vvoo"))(tmps.at("0621_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0620_bb_vv")(bb, cb))
      .allocate(tmps.at("0444_bb_oo"))(tmps.at("0444_bb_oo")(ib, jb) =
                                         f.at("bb_ov")(ib, ab) * t1_2p.at("bb")(ab, jb))
      .allocate(tmps.at("0443_bb_oo"))(tmps.at("0443_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovv")(ib, kb, ab, bb) *
                                         t2_2p.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0442_bb_oo"))(tmps.at("0442_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovo")(ib, kb, ab, jb) *
                                         t1_2p.at("bb")(ab, kb))
      .allocate(tmps.at("0441_bb_oo"))(tmps.at("0441_bb_oo")(ib, jb) =
                                         eri.at("abab_oovv")(ka, ib, aa, bb) *
                                         t2_2p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0440_bb_oo"))(tmps.at("0440_bb_oo")(ib, jb) =
                                         eri.at("abab_oovo")(ka, ib, aa, jb) *
                                         t1_2p.at("aa")(aa, ka))
      .allocate(tmps.at("0445_bb_oo"))(tmps.at("0445_bb_oo")(ib, jb) =
                                         -1.00 * tmps.at("0442_bb_oo")(ib, jb))(
        tmps.at("0445_bb_oo")(ib, jb) += tmps.at("0440_bb_oo")(ib, jb))(
        tmps.at("0445_bb_oo")(ib, jb) += 0.50 * tmps.at("0443_bb_oo")(ib, jb))(
        tmps.at("0445_bb_oo")(ib, jb) += tmps.at("0441_bb_oo")(ib, jb))(
        tmps.at("0445_bb_oo")(ib, jb) += tmps.at("0444_bb_oo")(ib, jb))
      .deallocate(tmps.at("0444_bb_oo"))
      .deallocate(tmps.at("0443_bb_oo"))
      .deallocate(tmps.at("0442_bb_oo"))
      .deallocate(tmps.at("0441_bb_oo"))
      .deallocate(tmps.at("0440_bb_oo"))
      .allocate(tmps.at("0619_abab_vvoo"))(tmps.at("0619_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0445_bb_oo")(kb, jb))
      .deallocate(tmps.at("0445_bb_oo"))
      .allocate(tmps.at("0438_bb_oo"))(tmps.at("0438_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovv")(kb, ib, ab, bb) *
                                         t2_1p.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0437_bb_oo"))(tmps.at("0437_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovo")(kb, ib, ab, jb) *
                                         t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0439_bb_oo"))(tmps.at("0439_bb_oo")(ib, jb) =
                                         -0.50 * tmps.at("0438_bb_oo")(ib, jb))(
        tmps.at("0439_bb_oo")(ib, jb) += tmps.at("0437_bb_oo")(ib, jb))
      .deallocate(tmps.at("0438_bb_oo"))
      .deallocate(tmps.at("0437_bb_oo"))
      .allocate(tmps.at("0618_abab_vvoo"))(tmps.at("0618_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0439_bb_oo")(kb, jb))
      .deallocate(tmps.at("0439_bb_oo"))
      .allocate(tmps.at("0488_aaaa_voov"))(tmps.at("0488_aaaa_voov")(aa, ia, ja, ba) =
                                             eri.at("aaaa_oovv")(ia, ka, ba, ca) *
                                             t2_2p.at("aaaa")(ca, aa, ja, ka))
      .allocate(tmps.at("0487_aaaa_vovo"))(tmps.at("0487_aaaa_vovo")(aa, ia, ba, ja) =
                                             eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                             t1_2p.at("aa")(ca, ja))
      .allocate(tmps.at("0486_aaaa_voov"))(tmps.at("0486_aaaa_voov")(aa, ia, ja, ba) =
                                             eri.at("abab_oovv")(ia, kb, ba, cb) *
                                             t2_2p.at("abab")(aa, cb, ja, kb))
      .allocate(tmps.at("0489_aaaa_voov"))(tmps.at("0489_aaaa_voov")(aa, ia, ja, ba) =
                                             tmps.at("0487_aaaa_vovo")(aa, ia, ba, ja))(
        tmps.at("0489_aaaa_voov")(aa, ia, ja, ba) += tmps.at("0488_aaaa_voov")(aa, ia, ja, ba))(
        tmps.at("0489_aaaa_voov")(aa, ia, ja, ba) -= tmps.at("0486_aaaa_voov")(aa, ia, ja, ba))
      .deallocate(tmps.at("0488_aaaa_voov"))
      .deallocate(tmps.at("0487_aaaa_vovo"))
      .deallocate(tmps.at("0486_aaaa_voov"))
      .allocate(tmps.at("0617_baba_vvoo"))(tmps.at("0617_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0489_aaaa_voov")(ba, ka, ja, ca))
      .deallocate(tmps.at("0489_aaaa_voov"))
      .allocate(tmps.at("0616_abab_vvoo"))(tmps.at("0616_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0094_bb_vv")(bb, cb))
      .allocate(tmps.at("0615_abba_vvoo"))(tmps.at("0615_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_2p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0043_aa_oo")(ka, ja))
      .deallocate(tmps.at("0043_aa_oo"))
      .allocate(tmps.at("0612_bb_vo"))(tmps.at("0612_bb_vo")(ab, ib) =
                                         dp.at("bb_oo")(jb, ib) * t1_1p.at("bb")(ab, jb))
      .allocate(tmps.at("0611_bb_vo"))(tmps.at("0611_bb_vo")(ab, ib) =
                                         dp.at("bb_vv")(ab, bb) * t1_1p.at("bb")(bb, ib))
      .allocate(tmps.at("0610_bb_vo"))(tmps.at("0610_bb_vo")(ab, ib) =
                                         dp.at("aa_ov")(ja, ba) * t2_1p.at("abab")(ba, ab, ja, ib))
      .allocate(tmps.at("0609_bb_vo"))(tmps.at("0609_bb_vo")(ab, ib) =
                                         dp.at("bb_ov")(jb, bb) * t2_1p.at("bbbb")(bb, ab, ib, jb))
      .allocate(tmps.at("0613_bb_vo"))(tmps.at("0613_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("0610_bb_vo")(ab, ib))(
        tmps.at("0613_bb_vo")(ab, ib) -= tmps.at("0611_bb_vo")(ab, ib))(
        tmps.at("0613_bb_vo")(ab, ib) += tmps.at("0612_bb_vo")(ab, ib))(
        tmps.at("0613_bb_vo")(ab, ib) += tmps.at("0609_bb_vo")(ab, ib))
      .deallocate(tmps.at("0612_bb_vo"))
      .deallocate(tmps.at("0611_bb_vo"))
      .deallocate(tmps.at("0610_bb_vo"))
      .deallocate(tmps.at("0609_bb_vo"))
      .allocate(tmps.at("0614_abab_vvoo"))(tmps.at("0614_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0613_bb_vo")(bb, jb))
      .allocate(tmps.at("0608_baba_vvoo"))(tmps.at("0608_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("0436_aa_vo")(ba, ja))
      .deallocate(tmps.at("0436_aa_vo"))
      .allocate(tmps.at("0607_baab_vvoo"))(tmps.at("0607_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0271_aa_vv")(ba, ca))
      .allocate(tmps.at("0606_baba_vvoo"))(tmps.at("0606_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_2p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0106_abba_vovo")(ba, kb, cb, ja))
      .allocate(tmps.at("0605_baab_vvoo"))(tmps.at("0605_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_2p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0104_aa_vv")(ba, ca))
      .allocate(tmps.at("0604_abba_vvoo"))(tmps.at("0604_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0269_aa_oo")(ka, ja))
      .allocate(tmps.at("0602_aaaa_vovo"))(tmps.at("0602_aaaa_vovo")(aa, ia, ba, ja) =
                                             eri.at("aaaa_vovv")(aa, ia, ca, ba) *
                                             t1_1p.at("aa")(ca, ja))
      .allocate(tmps.at("0603_baba_vvoo"))(tmps.at("0603_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0602_aaaa_vovo")(ba, ka, ca, ja))
      .deallocate(tmps.at("0602_aaaa_vovo"))
      .allocate(tmps.at("0600_baab_ovoo"))(tmps.at("bin_bb_vo")(bb, ib) =
                                             eri.at("abab_oovv")(la, ib, ca, bb) *
                                             t1_1p.at("aa")(ca, la))(
        tmps.at("0600_baab_ovoo")(ib, aa, ja, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0601_baab_vvoo"))(tmps.at("0601_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0600_baab_ovoo")(kb, ba, ia, jb))
      .allocate(tmps.at("0599_baba_vvoo"))(tmps.at("0599_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0287_abab_voov")(ba, kb, ja, cb))
      .deallocate(tmps.at("0287_abab_voov"))
      .allocate(tmps.at("0596_bb_vv"))(tmps.at("0596_bb_vv")(ab, bb) =
                                         eri.at("abab_oovv")(ia, jb, ca, bb) *
                                         t2_1p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0595_bb_vv"))(tmps.at("0595_bb_vv")(ab, bb) =
                                         eri.at("bbbb_oovv")(ib, jb, bb, cb) *
                                         t2_1p.at("bbbb")(cb, ab, jb, ib))
      .allocate(tmps.at("0597_bb_vv"))(tmps.at("0597_bb_vv")(ab, bb) =
                                         0.50 * tmps.at("0595_bb_vv")(ab, bb))(
        tmps.at("0597_bb_vv")(ab, bb) += tmps.at("0596_bb_vv")(ab, bb))
      .deallocate(tmps.at("0596_bb_vv"))
      .deallocate(tmps.at("0595_bb_vv"))
      .allocate(tmps.at("0598_abab_vvoo"))(tmps.at("0598_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0597_bb_vv")(bb, cb))
      .allocate(tmps.at("0594_baba_vvoo"))(tmps.at("0594_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0257_abab_voov")(ba, kb, ja, cb))
      .deallocate(tmps.at("0257_abab_voov"))
      .allocate(tmps.at("0593_abba_vvoo"))(tmps.at("0593_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_2p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0116_aa_oo")(ka, ja))
      .allocate(tmps.at("0592_baab_vvoo"))(tmps.at("0592_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_2p.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0096_abab_vovo")(ba, kb, ca, jb))
      .allocate(tmps.at("0591_abab_vvoo"))(tmps.at("0591_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0051_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0051_baab_vooo"))
      .allocate(tmps.at("0590_abab_vvoo"))(tmps.at("0590_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0089_abab_ovvo")(ka, bb, ca, jb))
      .allocate(tmps.at("0589_baba_vvoo"))(tmps.at("0589_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_2p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0163_aaaa_voov")(ba, ka, ja, ca))
      .allocate(tmps.at("0587_abba_vvvo"))(tmps.at("0587_abba_vvvo")(aa, bb, cb, ia) =
                                             eri.at("abab_vvvv")(aa, bb, da, cb) *
                                             t1_1p.at("aa")(da, ia))
      .allocate(tmps.at("0588_abba_vvoo"))(tmps.at("0588_abba_vvoo")(aa, bb, ib, ja) =
                                             t1_1p.at("bb")(cb, ib) *
                                             tmps.at("0587_abba_vvvo")(aa, bb, cb, ja))
      .allocate(tmps.at("0586_abba_vvoo"))(tmps.at("0586_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_2p.at("abab")(aa, cb, ka, ib) *
                                             tmps.at("0126_baba_vovo")(bb, ka, cb, ja))
      .allocate(tmps.at("0585_abba_vvoo"))(tmps.at("0585_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0044_aa_oo")(ka, ja))
      .deallocate(tmps.at("0044_aa_oo"))
      .allocate(tmps.at("0584_baab_vvoo"))(tmps.at("0584_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_2p.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0091_abba_voov")(ba, kb, jb, ca))
      .allocate(tmps.at("0583_abab_vvoo"))(tmps.at("0583_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0040_bb_oo")(kb, jb))
      .deallocate(tmps.at("0040_bb_oo"))
      .allocate(tmps.at("0582_baab_vvoo"))(tmps.at("0582_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0048_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0048_abab_vooo"))
      .allocate(tmps.at("0581_baba_vvoo"))(tmps.at("0581_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0262_abba_vovo")(ba, kb, cb, ja))
      .allocate(tmps.at("0578_bb_oo"))(tmps.at("0578_bb_oo")(ib, jb) =
                                         eri.at("abab_oovo")(ka, ib, aa, jb) *
                                         t1_1p.at("aa")(aa, ka))
      .allocate(tmps.at("0577_bb_oo"))(tmps.at("0577_bb_oo")(ib, jb) =
                                         eri.at("abab_oovv")(ka, ib, aa, bb) *
                                         t2_1p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0579_bb_oo"))(tmps.at("0579_bb_oo")(ib, jb) =
                                         tmps.at("0577_bb_oo")(ib, jb))(
        tmps.at("0579_bb_oo")(ib, jb) += tmps.at("0578_bb_oo")(ib, jb))
      .deallocate(tmps.at("0578_bb_oo"))
      .deallocate(tmps.at("0577_bb_oo"))
      .allocate(tmps.at("0580_abab_vvoo"))(tmps.at("0580_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0579_bb_oo")(kb, jb))
      .allocate(tmps.at("0576_baab_vvoo"))(tmps.at("0576_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_2p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0100_aa_vv")(ba, ca))
      .allocate(tmps.at("0575_baba_vvoo"))(tmps.at("0575_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_2p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0112_abab_voov")(ba, kb, ja, cb))
      .allocate(tmps.at("0573_abab_vovo"))(tmps.at("0573_abab_vovo")(aa, ib, ba, jb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t1_2p.at("bb")(cb, jb))
      .allocate(tmps.at("0574_baab_vvoo"))(tmps.at("0574_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0573_abab_vovo")(ba, kb, ca, jb))
      .deallocate(tmps.at("0573_abab_vovo"))
      .allocate(tmps.at("0572_baab_vvoo"))(tmps.at("0572_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0188_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0571_abab_vvoo"))(tmps.at("0571_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0083_bb_oo")(kb, jb))
      .allocate(tmps.at("0569_bb_vv"))(tmps.at("0569_bb_vv")(ab, bb) =
                                         eri.at("baab_vovv")(ab, ia, ca, bb) *
                                         t1_1p.at("aa")(ca, ia))
      .allocate(tmps.at("0570_abab_vvoo"))(tmps.at("0570_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0569_bb_vv")(bb, cb))
      .allocate(tmps.at("0568_baba_vvoo"))(tmps.at("0568_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0273_aaaa_voov")(ba, ka, ja, ca))
      .allocate(tmps.at("0567_abab_vvoo"))(tmps.at("0567_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0016_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0016_baab_vooo"))
      .allocate(tmps.at("0566_abba_vvoo"))(tmps.at("0566_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_2p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0102_aa_oo")(ka, ja))
      .allocate(tmps.at("0564_bb_vv"))(tmps.at("0564_bb_vv")(ab, bb) =
                                         eri.at("bbbb_vovv")(ab, ib, cb, bb) *
                                         t1_1p.at("bb")(cb, ib))
      .allocate(tmps.at("0565_abab_vvoo"))(tmps.at("0565_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0564_bb_vv")(bb, cb))
      .deallocate(tmps.at("0564_bb_vv"))
      .allocate(tmps.at("0426_abab_vvoo"))(tmps.at("0426_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("aaaa_vovo")(aa, ka, ca, ia) *
                                             t2_2p.at("abab")(ca, bb, ka, jb))
      .allocate(tmps.at("0425_baab_vvoo"))(tmps.at("bin_bbaa_vvoo")(ab, cb, ia, ka) =
                                             eri.at("baab_vovv")(ab, ka, da, cb) *
                                             t1_2p.at("aa")(da, ia))(
        tmps.at("0425_baab_vvoo")(ab, ba, ia, jb) =
          tmps.at("bin_bbaa_vvoo")(ab, cb, ia, ka) * t2.at("abab")(ba, cb, ka, jb))
      .allocate(tmps.at("0424_baba_vvoo"))(tmps.at("bin_bbbb_vvoo")(ab, cb, ib, kb) =
                                             eri.at("bbbb_vovv")(ab, kb, db, cb) *
                                             t1_1p.at("bb")(db, ib))(
        tmps.at("0424_baba_vvoo")(ab, ba, ib, ja) =
          tmps.at("bin_bbbb_vvoo")(ab, cb, ib, kb) * t2_1p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0423_abab_vvoo"))(tmps.at("bin_abba_vvvo")(aa, bb, cb, ia) =
                                             eri.at("abab_vvvv")(aa, bb, da, cb) *
                                             t1_2p.at("aa")(da, ia))(
        tmps.at("0423_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_abba_vvvo")(aa, bb, cb, ia) * t1.at("bb")(cb, jb))
      .allocate(tmps.at("0422_abab_vvoo"))(tmps.at("bin_abab_vvoo")(aa, cb, ia, kb) =
                                             eri.at("bbbb_oovv")(lb, kb, db, cb) *
                                             t2_1p.at("abab")(aa, db, ia, lb))(
        tmps.at("0422_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_abab_vvoo")(aa, cb, ia, kb) * t2_1p.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("0421_abab_vvoo"))(tmps.at("0421_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vvvv")(aa, bb, ca, db) *
                                             t2_2p.at("abab")(ca, db, ia, jb))
      .allocate(tmps.at("0420_abba_vvoo"))(tmps.at("0420_abba_vvoo")(aa, bb, ib, ja) =
                                             eri.at("baba_vovo")(bb, ka, cb, ja) *
                                             t2_2p.at("abab")(aa, cb, ka, ib))
      .allocate(tmps.at("0419_abab_vvoo"))(tmps.at("0419_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_oooo")(ka, lb, ia, jb) *
                                             t2_2p.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0418_abab_vvoo"))(tmps.at("0418_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_oo")(ka, ia) *
                                             t2_2p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0417_abab_vvoo"))(tmps.at("0417_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_vv")(aa, ca) *
                                             t2_2p.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0416_abab_vvoo"))(tmps.at("0416_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vvvo")(aa, bb, cb, ia) *
                                             t1_2p.at("bb")(cb, jb))
      .allocate(tmps.at("0415_abab_vvoo"))(tmps.at("0415_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("bbbb_vovo")(bb, kb, cb, jb) *
                                             t2_2p.at("abab")(aa, cb, ia, kb))
      .allocate(tmps.at("0414_abab_vvoo"))(tmps.at("0414_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vvvo")(aa, bb, ca, jb) *
                                             t1_2p.at("aa")(ca, ia))
      .allocate(tmps.at("0413_abab_vvoo"))(tmps.at("0413_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("baab_vooo")(bb, ka, ia, jb) *
                                             t1_2p.at("aa")(aa, ka))
      .allocate(tmps.at("0412_abba_vvoo"))(tmps.at("0412_abba_vvoo")(aa, bb, ib, ja) =
                                             eri.at("abab_vovo")(aa, kb, ca, ib) *
                                             t2_2p.at("abab")(ca, bb, ja, kb))
      .allocate(tmps.at("0411_abab_vvoo"))(tmps.at("0411_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vovo")(aa, kb, cb, ia) *
                                             t2_2p.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("0410_abab_vvoo"))(tmps.at("0410_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0002")() *
                                             t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0409_abab_vvoo"))(tmps.at("0409_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vooo")(aa, kb, ia, jb) *
                                             t1_2p.at("bb")(bb, kb))
      .allocate(tmps.at("0408_abab_vvoo"))(tmps.at("0408_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("bb_oo")(kb, jb) *
                                             t2_2p.at("abab")(aa, bb, ia, kb))
      .allocate(tmps.at("0407_abab_vvoo"))(tmps.at("0407_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0014")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0406_abab_vvoo"))(tmps.at("0406_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0001")() *
                                             t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0405_abab_vvoo"))(tmps.at("0405_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0016")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0404_abab_vvoo"))(tmps.at("0404_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("baab_vovo")(bb, ka, ca, jb) *
                                             t2_2p.at("aaaa")(ca, aa, ia, ka))
      .allocate(tmps.at("0403_abab_vvoo"))(tmps.at("bin_abab_vvoo")(aa, cb, ia, kb) =
                                             eri.at("abab_vovv")(aa, kb, da, cb) *
                                             t1_2p.at("aa")(da, ia))(
        tmps.at("0403_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_abab_vvoo")(aa, cb, ia, kb) * t2.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("0402_baba_vvoo"))(tmps.at("bin_bbbb_vvoo")(ab, cb, ib, kb) =
                                             eri.at("bbbb_vovv")(ab, kb, cb, db) *
                                             t1_2p.at("bb")(db, ib))(
        tmps.at("0402_baba_vvoo")(ab, ba, ib, ja) =
          tmps.at("bin_bbbb_vvoo")(ab, cb, ib, kb) * t2.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0401_abab_vvoo"))(tmps.at("0401_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("bb_vv")(bb, cb) *
                                             t2_2p.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0400_baba_vvoo"))(tmps.at("bin_abab_vvoo")(ca, ab, ka, ib) =
                                             eri.at("baab_vovv")(ab, ka, ca, db) *
                                             t1_2p.at("bb")(db, ib))(
        tmps.at("0400_baba_vvoo")(ab, ba, ib, ja) =
          tmps.at("bin_abab_vvoo")(ca, ab, ka, ib) * t2.at("aaaa")(ca, ba, ja, ka))
      .allocate(tmps.at("0427_baba_vvoo"))(tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) =
                                             -1.00 * tmps.at("0415_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0418_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0412_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0407_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0400_baba_vvoo")(ab, ba, ib, ja))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0413_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0411_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0425_baab_vvoo")(ab, ba, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0420_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0422_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0419_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0401_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0405_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0414_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0416_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0408_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0424_baba_vvoo")(ab, ba, ib, ja))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) +=
        2.00 * tmps.at("0410_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0426_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0404_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) +=
        2.00 * tmps.at("0406_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0423_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0409_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0402_baba_vvoo")(ab, ba, ib, ja))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0417_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("0403_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("0427_baba_vvoo")(ab, ba, ib, ja) += tmps.at("0421_abab_vvoo")(ba, ab, ja, ib))
      .deallocate(tmps.at("0426_abab_vvoo"))
      .deallocate(tmps.at("0425_baab_vvoo"))
      .deallocate(tmps.at("0424_baba_vvoo"))
      .deallocate(tmps.at("0423_abab_vvoo"))
      .deallocate(tmps.at("0422_abab_vvoo"))
      .deallocate(tmps.at("0421_abab_vvoo"))
      .deallocate(tmps.at("0420_abba_vvoo"))
      .deallocate(tmps.at("0419_abab_vvoo"))
      .deallocate(tmps.at("0418_abab_vvoo"))
      .deallocate(tmps.at("0417_abab_vvoo"))
      .deallocate(tmps.at("0416_abab_vvoo"))
      .deallocate(tmps.at("0415_abab_vvoo"))
      .deallocate(tmps.at("0414_abab_vvoo"))
      .deallocate(tmps.at("0413_abab_vvoo"))
      .deallocate(tmps.at("0412_abba_vvoo"))
      .deallocate(tmps.at("0411_abab_vvoo"))
      .deallocate(tmps.at("0410_abab_vvoo"))
      .deallocate(tmps.at("0409_abab_vvoo"))
      .deallocate(tmps.at("0408_abab_vvoo"))
      .deallocate(tmps.at("0407_abab_vvoo"))
      .deallocate(tmps.at("0406_abab_vvoo"))
      .deallocate(tmps.at("0405_abab_vvoo"))
      .deallocate(tmps.at("0404_abab_vvoo"))
      .deallocate(tmps.at("0403_abab_vvoo"))
      .deallocate(tmps.at("0402_baba_vvoo"))
      .deallocate(tmps.at("0401_abab_vvoo"))
      .deallocate(tmps.at("0400_baba_vvoo"));
  }
}

template void exachem::cc::qed_ccsd_cs::resid_part3<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);