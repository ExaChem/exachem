/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_os_resid_3.hpp"

template<typename T>
void exachem::cc::qed_ccsd_os::resid_3(
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
      .allocate(tmps.at("0510_bb_vo"))(tmps.at("0510_bb_vo")(ab, ib) =
                                         -0.50 * tmps.at("0495_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) += tmps.at("0497_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) += tmps.at("0498_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) += tmps.at("0503_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) -= tmps.at("0499_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) += tmps.at("0502_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) += 0.50 * tmps.at("0501_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) += tmps.at("0506_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) += tmps.at("0509_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) -= tmps.at("0508_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) += tmps.at("0507_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) += 0.50 * tmps.at("0504_bb_vo")(ab, ib))(
        tmps.at("0510_bb_vo")(ab, ib) += tmps.at("0496_bb_vo")(ab, ib))
      .deallocate(tmps.at("0509_bb_vo"))
      .deallocate(tmps.at("0508_bb_vo"))
      .deallocate(tmps.at("0507_bb_vo"))
      .deallocate(tmps.at("0506_bb_vo"))
      .deallocate(tmps.at("0504_bb_vo"))
      .deallocate(tmps.at("0503_bb_vo"))
      .deallocate(tmps.at("0502_bb_vo"))
      .deallocate(tmps.at("0501_bb_vo"))
      .deallocate(tmps.at("0499_bb_vo"))
      .deallocate(tmps.at("0498_bb_vo"))
      .deallocate(tmps.at("0497_bb_vo"))
      .deallocate(tmps.at("0496_bb_vo"))
      .deallocate(tmps.at("0495_bb_vo"))
      .allocate(tmps.at("0512_bb_vo"))(tmps.at("0512_bb_vo")(ab, ib) =
                                         tmps.at("0510_bb_vo")(ab, ib))(
        tmps.at("0512_bb_vo")(ab, ib) += tmps.at("0511_bb_vo")(ab, ib))
      .deallocate(tmps.at("0511_bb_vo"))
      .deallocate(tmps.at("0510_bb_vo"))(r1.at("bb")(ab, ib) -= tmps.at("0512_bb_vo")(ab, ib))
      .deallocate(tmps.at("0512_bb_vo"))
      .allocate(tmps.at("0593_abab_oovo"))(tmps.at("0593_abab_oovo")(ia, jb, aa, kb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("0607_bb_oo"))(tmps.at("0607_bb_oo")(ib, jb) =
                                         t1_1p.at("aa")(aa, ka) *
                                         tmps.at("0593_abab_oovo")(ka, ib, aa, jb))
      .allocate(tmps.at("0606_bb_oo"))(tmps.at("0606_bb_oo")(ib, jb) =
                                         t1_1p.at("bb")(ab, kb) *
                                         tmps.at("0127_bbbb_oovo")(ib, kb, ab, jb))
      .allocate(tmps.at("0608_bb_oo"))(tmps.at("0608_bb_oo")(ib, jb) =
                                         -1.00 * tmps.at("0607_bb_oo")(ib, jb))(
        tmps.at("0608_bb_oo")(ib, jb) += tmps.at("0606_bb_oo")(ib, jb))
      .deallocate(tmps.at("0607_bb_oo"))
      .deallocate(tmps.at("0606_bb_oo"))
      .allocate(tmps.at("0609_bb_vo"))(tmps.at("0609_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0608_bb_oo")(jb, ib))
      .allocate(tmps.at("0601_abab_oovo"))(tmps.at("0601_abab_oovo")(ia, jb, aa, kb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t1_2p.at("bb")(bb, kb))
      .allocate(tmps.at("0603_bb_oo"))(tmps.at("0603_bb_oo")(ib, jb) =
                                         t1.at("aa")(aa, ka) *
                                         tmps.at("0601_abab_oovo")(ka, ib, aa, jb))
      .allocate(tmps.at("0600_bbbb_oovo"))(tmps.at("0600_bbbb_oovo")(ib, jb, ab, kb) =
                                             eri.at("bbbb_oovv")(ib, jb, ab, bb) *
                                             t1_2p.at("bb")(bb, kb))
      .allocate(tmps.at("0602_bb_oo"))(tmps.at("0602_bb_oo")(ib, jb) =
                                         t1.at("bb")(ab, kb) *
                                         tmps.at("0600_bbbb_oovo")(kb, ib, ab, jb))
      .allocate(tmps.at("0604_bb_oo"))(tmps.at("0604_bb_oo")(ib, jb) =
                                         tmps.at("0602_bb_oo")(ib, jb))(
        tmps.at("0604_bb_oo")(ib, jb) += tmps.at("0603_bb_oo")(ib, jb))
      .deallocate(tmps.at("0603_bb_oo"))
      .deallocate(tmps.at("0602_bb_oo"))
      .allocate(tmps.at("0605_bb_vo"))(tmps.at("0605_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0604_bb_oo")(jb, ib))
      .allocate(tmps.at("0598_bb_oo"))(tmps.at("0598_bb_oo")(ib, jb) =
                                         t1_1p.at("aa")(aa, ka) *
                                         tmps.at("0286_abab_oovo")(ka, ib, aa, jb))
      .allocate(tmps.at("0599_bb_vo"))(tmps.at("0599_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0598_bb_oo")(jb, ib))
      .allocate(tmps.at("0595_bb_oo"))(tmps.at("0595_bb_oo")(ib, jb) =
                                         t1.at("aa")(aa, ka) *
                                         tmps.at("0593_abab_oovo")(ka, ib, aa, jb))
      .allocate(tmps.at("0594_bb_oo"))(tmps.at("0594_bb_oo")(ib, jb) =
                                         t1.at("bb")(ab, kb) *
                                         tmps.at("0127_bbbb_oovo")(kb, ib, ab, jb))
      .allocate(tmps.at("0596_bb_oo"))(tmps.at("0596_bb_oo")(ib, jb) =
                                         tmps.at("0594_bb_oo")(ib, jb))(
        tmps.at("0596_bb_oo")(ib, jb) += tmps.at("0595_bb_oo")(ib, jb))
      .deallocate(tmps.at("0595_bb_oo"))
      .deallocate(tmps.at("0594_bb_oo"))
      .allocate(tmps.at("0597_bb_vo"))(tmps.at("0597_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0596_bb_oo")(jb, ib))
      .allocate(tmps.at("0592_bb_vo"))(tmps.at("0592_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(ab, jb) * tmps.at("0289_bb_oo")(jb, ib))
      .allocate(tmps.at("0589_bb_oo"))(tmps.at("0589_bb_oo")(ib, jb) =
                                         t1_2p.at("aa")(aa, ka) *
                                         tmps.at("0286_abab_oovo")(ka, ib, aa, jb))
      .allocate(tmps.at("0588_bb_oo"))(tmps.at("0588_bb_oo")(ib, jb) =
                                         t1_2p.at("bb")(ab, kb) *
                                         tmps.at("0256_bbbb_oovo")(ib, kb, ab, jb))
      .allocate(tmps.at("0590_bb_oo"))(tmps.at("0590_bb_oo")(ib, jb) =
                                         tmps.at("0588_bb_oo")(ib, jb))(
        tmps.at("0590_bb_oo")(ib, jb) += tmps.at("0589_bb_oo")(ib, jb))
      .deallocate(tmps.at("0589_bb_oo"))
      .deallocate(tmps.at("0588_bb_oo"))
      .allocate(tmps.at("0591_bb_vo"))(tmps.at("0591_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0590_bb_oo")(jb, ib))
      .allocate(tmps.at("0586_bb_oo"))(tmps.at("0586_bb_oo")(ib, jb) =
                                         t1_1p.at("bb")(ab, kb) *
                                         tmps.at("0256_bbbb_oovo")(kb, ib, ab, jb))
      .allocate(tmps.at("0587_bb_vo"))(tmps.at("0587_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0586_bb_oo")(jb, ib))
      .allocate(tmps.at("0583_bb_vv"))(tmps.at("0583_bb_vv")(ab, bb) =
                                         eri.at("abab_oovv")(ia, jb, ca, bb) *
                                         t2_1p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0584_bb_vo"))(tmps.at("0584_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(bb, ib) * tmps.at("0583_bb_vv")(ab, bb))
      .allocate(tmps.at("0581_bb_vv"))(tmps.at("0581_bb_vv")(ab, bb) =
                                         eri.at("bbbb_oovv")(ib, jb, cb, bb) *
                                         t2.at("bbbb")(cb, ab, jb, ib))
      .allocate(tmps.at("0582_bb_vo"))(tmps.at("0582_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(bb, ib) * tmps.at("0581_bb_vv")(ab, bb))
      .allocate(tmps.at("0580_bb_vo"))(tmps.at("0580_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(ab, jb) * tmps.at("0281_bb_oo")(jb, ib))
      .allocate(tmps.at("0579_bb_vo"))(tmps.at("0579_bb_vo")(ab, ib) =
                                         t2_1p.at("bbbb")(bb, ab, ib, jb) *
                                         tmps.at("0381_bb_ov")(jb, bb))
      .deallocate(tmps.at("0381_bb_ov"))
      .allocate(tmps.at("0576_bb_oo"))(tmps.at("0576_bb_oo")(ib, jb) =
                                         f.at("bb_ov")(ib, ab) * t1_2p.at("bb")(ab, jb))
      .allocate(tmps.at("0575_bb_oo"))(tmps.at("0575_bb_oo")(ib, jb) =
                                         eri.at("abab_oovv")(ka, ib, aa, bb) *
                                         t2_2p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0574_bb_oo"))(tmps.at("0574_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovv")(ib, kb, ab, bb) *
                                         t2_2p.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0577_bb_oo"))(tmps.at("0577_bb_oo")(ib, jb) =
                                         2.00 * tmps.at("0575_bb_oo")(ib, jb))(
        tmps.at("0577_bb_oo")(ib, jb) += tmps.at("0574_bb_oo")(ib, jb))(
        tmps.at("0577_bb_oo")(ib, jb) += 2.00 * tmps.at("0576_bb_oo")(ib, jb))
      .deallocate(tmps.at("0576_bb_oo"))
      .deallocate(tmps.at("0575_bb_oo"))
      .deallocate(tmps.at("0574_bb_oo"))
      .allocate(tmps.at("0578_bb_vo"))(tmps.at("0578_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0577_bb_oo")(jb, ib))
      .allocate(tmps.at("0571_bb_oo"))(tmps.at("0571_bb_oo")(ib, jb) =
                                         eri.at("abab_oovo")(ka, ib, aa, jb) *
                                         t1_2p.at("aa")(aa, ka))
      .allocate(tmps.at("0570_bb_oo"))(tmps.at("0570_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovo")(ib, kb, ab, jb) *
                                         t1_2p.at("bb")(ab, kb))
      .allocate(tmps.at("0572_bb_oo"))(tmps.at("0572_bb_oo")(ib, jb) =
                                         -1.00 * tmps.at("0570_bb_oo")(ib, jb))(
        tmps.at("0572_bb_oo")(ib, jb) += tmps.at("0571_bb_oo")(ib, jb))
      .deallocate(tmps.at("0571_bb_oo"))
      .deallocate(tmps.at("0570_bb_oo"))
      .allocate(tmps.at("0573_bb_vo"))(tmps.at("0573_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0572_bb_oo")(jb, ib))
      .allocate(tmps.at("0568_bb_oo"))(tmps.at("0568_bb_oo")(ib, jb) =
                                         eri.at("abab_oovo")(ka, ib, aa, jb) *
                                         t1_1p.at("aa")(aa, ka))
      .allocate(tmps.at("0569_bb_vo"))(tmps.at("0569_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0568_bb_oo")(jb, ib))
      .allocate(tmps.at("0566_bb_oo"))(tmps.at("0566_bb_oo")(ib, jb) =
                                         eri.at("abab_oovv")(ka, ib, aa, bb) *
                                         t2_1p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0567_bb_vo"))(tmps.at("0567_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0566_bb_oo")(jb, ib))
      .allocate(tmps.at("0563_bb_vv"))(tmps.at("0563_bb_vv")(ab, bb) =
                                         eri.at("abab_oovv")(ia, jb, ca, bb) *
                                         t2_2p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0562_bb_vv"))(tmps.at("0562_bb_vv")(ab, bb) =
                                         eri.at("bbbb_oovv")(ib, jb, bb, cb) *
                                         t2_2p.at("bbbb")(cb, ab, jb, ib))
      .allocate(tmps.at("0564_bb_vv"))(tmps.at("0564_bb_vv")(ab, bb) =
                                         0.50 * tmps.at("0562_bb_vv")(ab, bb))(
        tmps.at("0564_bb_vv")(ab, bb) += tmps.at("0563_bb_vv")(ab, bb))
      .deallocate(tmps.at("0563_bb_vv"))
      .deallocate(tmps.at("0562_bb_vv"))
      .allocate(tmps.at("0565_bb_vo"))(tmps.at("0565_bb_vo")(ab, ib) =
                                         t1.at("bb")(bb, ib) * tmps.at("0564_bb_vv")(ab, bb))
      .allocate(tmps.at("0560_bbbb_vovo"))(tmps.at("0560_bbbb_vovo")(ab, ib, bb, jb) =
                                             eri.at("bbbb_vovv")(ab, ib, bb, cb) *
                                             t1_2p.at("bb")(cb, jb))
      .allocate(tmps.at("0561_bb_vo"))(tmps.at("0561_bb_vo")(ab, ib) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0560_bbbb_vovo")(ab, jb, bb, ib))
      .allocate(tmps.at("0559_bb_vo"))(tmps.at("0559_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(bb, ib) * tmps.at("0505_bb_vv")(ab, bb))
      .allocate(tmps.at("0557_baba_voov"))(tmps.at("0557_baba_voov")(ab, ia, jb, ba) =
                                             t2_2p.at("bbbb")(cb, ab, jb, kb) *
                                             eri.at("abab_oovv")(ia, kb, ba, cb))
      .allocate(tmps.at("0558_bb_vo"))(tmps.at("0558_bb_vo")(ab, ib) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0557_baba_voov")(ab, ja, ib, ba))
      .allocate(tmps.at("0556_bb_vo"))(tmps.at("0556_bb_vo")(ab, ib) =
                                         t2_1p.at("abab")(ba, ab, ja, ib) *
                                         tmps.at("0365_aa_ov")(ja, ba))
      .allocate(tmps.at("0555_bb_vo"))(tmps.at("0555_bb_vo")(ab, ib) =
                                         t2.at("abab")(ba, ab, ja, ib) *
                                         tmps.at("0354_aa_ov")(ja, ba))
      .allocate(tmps.at("0554_bb_vo"))(tmps.at("0554_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(ab, jb) * tmps.at("0015_bb_oo")(jb, ib))
      .allocate(tmps.at("0553_bb_vo"))(tmps.at("0553_bb_vo")(ab, ib) =
                                         t2_1p.at("abab")(ba, ab, ja, ib) *
                                         tmps.at("0357_aa_ov")(ja, ba))
      .deallocate(tmps.at("0357_aa_ov"))
      .allocate(tmps.at("0552_bb_vo"))(tmps.at("0552_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(ab, jb) * tmps.at("0271_bb_oo")(jb, ib))
      .allocate(tmps.at("0551_bb_vo"))(tmps.at("0551_bb_vo")(ab, ib) =
                                         t2_2p.at("abab")(ba, ab, ja, ib) *
                                         tmps.at("0232_aa_ov")(ja, ba))
      .allocate(tmps.at("0550_bb_vo"))(tmps.at("0550_bb_vo")(ab, ib) =
                                         t2_2p.at("bbbb")(bb, ab, ib, jb) *
                                         tmps.at("0237_bb_ov")(jb, bb))
      .allocate(tmps.at("0549_bb_vo"))(tmps.at("0549_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(bb, jb) *
                                         tmps.at("0141_bbbb_vovo")(ab, jb, bb, ib))
      .allocate(tmps.at("0548_bb_vo"))(tmps.at("0548_bb_vo")(ab, ib) =
                                         t1_2p.at("aa")(ba, ja) *
                                         tmps.at("0033_baba_voov")(ab, ja, ib, ba))
      .allocate(tmps.at("0547_bb_vo"))(tmps.at("0547_bb_vo")(ab, ib) =
                                         t1_2p.at("aa")(ba, ja) *
                                         tmps.at("0137_baba_voov")(ab, ja, ib, ba))
      .allocate(tmps.at("0546_bb_vo"))(tmps.at("0546_bb_vo")(ab, ib) =
                                         t1_2p.at("aa")(ba, ja) *
                                         tmps.at("0135_baab_vovo")(ab, ja, ba, ib))
      .allocate(tmps.at("0544_bb_vv"))(tmps.at("0544_bb_vv")(ab, bb) =
                                         eri.at("bbbb_oovv")(ib, jb, bb, cb) *
                                         t2_1p.at("bbbb")(cb, ab, jb, ib))
      .allocate(tmps.at("0545_bb_vo"))(tmps.at("0545_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(bb, ib) * tmps.at("0544_bb_vv")(ab, bb))
      .allocate(tmps.at("0543_bb_vo"))(tmps.at("0543_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(bb, jb) *
                                         tmps.at("0144_bbbb_voov")(ab, jb, ib, bb))
      .allocate(tmps.at("0542_bb_vo"))(tmps.at("0542_bb_vo")(ab, ib) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0133_baab_vovo")(ab, ja, ba, ib))
      .allocate(tmps.at("0541_bb_vo"))(tmps.at("0541_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(ab, jb) * tmps.at("0276_bb_oo")(jb, ib))
      .allocate(tmps.at("0540_bb_vo"))(tmps.at("0540_bb_vo")(ab, ib) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0035_baba_voov")(ab, ja, ib, ba))
      .allocate(tmps.at("0539_bb_vo"))(tmps.at("0539_bb_vo")(ab, ib) =
                                         t2_2p.at("abab")(ba, ab, ja, ib) *
                                         tmps.at("0224_aa_ov")(ja, ba))
      .allocate(tmps.at("0538_bb_vo"))(tmps.at("0538_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0131_bbbb_vovo")(ab, jb, bb, ib))
      .allocate(tmps.at("0536_baab_vovo"))(tmps.at("0536_baab_vovo")(ab, ia, ba, jb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t1_2p.at("bb")(cb, jb))
      .allocate(tmps.at("0537_bb_vo"))(tmps.at("0537_bb_vo")(ab, ib) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0536_baab_vovo")(ab, ja, ba, ib))
      .allocate(tmps.at("0534_bb_oo"))(tmps.at("0534_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovv")(ib, kb, ab, bb) *
                                         t2_1p.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0535_bb_vo"))(tmps.at("0535_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0534_bb_oo")(jb, ib))
      .allocate(tmps.at("0532_bb_oo"))(tmps.at("0532_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovo")(kb, ib, ab, jb) *
                                         t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0533_bb_vo"))(tmps.at("0533_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0532_bb_oo")(jb, ib))
      .allocate(tmps.at("0016_bb_oo"))(tmps.at("0016_bb_oo")(ib, jb) =
                                         dp.at("bb_ov")(ib, ab) * t1_2p.at("bb")(ab, jb))
      .allocate(tmps.at("0531_bb_vo"))(tmps.at("0531_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0016_bb_oo")(jb, ib))
      .allocate(tmps.at("0530_bb_vo"))(tmps.at("0530_bb_vo")(ab, ib) =
                                         t1_2p.at("bb")(ab, jb) * tmps.at("0273_bb_oo")(jb, ib))
      .allocate(tmps.at("0528_bb_oo"))(tmps.at("0528_bb_oo")(ib, jb) =
                                         f.at("bb_ov")(ib, ab) * t1_1p.at("bb")(ab, jb))
      .allocate(tmps.at("0529_bb_vo"))(tmps.at("0529_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0528_bb_oo")(jb, ib))
      .allocate(tmps.at("0526_bb_vo"))(tmps.at("0526_bb_vo")(ab, ib) =
                                         scalars.at("0014")() * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("0525_bb_vo"))(tmps.at("0525_bb_vo")(ab, ib) =
                                         scalars.at("0001")() * t1_2p.at("bb")(ab, ib))
      .allocate(tmps.at("0524_bb_vo"))(tmps.at("0524_bb_vo")(ab, ib) =
                                         scalars.at("0016")() * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("0523_bb_vo"))(tmps.at("0523_bb_vo")(ab, ib) =
                                         f.at("aa_ov")(ja, ba) * t2_2p.at("abab")(ba, ab, ja, ib))
      .allocate(tmps.at("0522_bb_vo"))(tmps.at("0522_bb_vo")(ab, ib) =
                                         eri.at("bbbb_vovo")(ab, jb, bb, ib) *
                                         t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("0521_bb_vo"))(tmps.at("0521_bb_vo")(ab, ib) =
                                         scalars.at("0002")() * t1_2p.at("bb")(ab, ib))
      .allocate(tmps.at("0520_bb_vo"))(tmps.at("0520_bb_vo")(ab, ib) =
                                         eri.at("bbbb_oovo")(jb, kb, bb, ib) *
                                         t2_2p.at("bbbb")(bb, ab, kb, jb))
      .allocate(tmps.at("0519_bb_vo"))(tmps.at("0519_bb_vo")(ab, ib) =
                                         eri.at("abab_oovo")(ja, kb, ba, ib) *
                                         t2_2p.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0518_bb_vo"))(tmps.at("0518_bb_vo")(ab, ib) =
                                         eri.at("bbbb_vovv")(ab, jb, bb, cb) *
                                         t2_2p.at("bbbb")(bb, cb, ib, jb))
      .allocate(tmps.at("0517_bb_vo"))(tmps.at("0517_bb_vo")(ab, ib) =
                                         f.at("bb_oo")(jb, ib) * t1_2p.at("bb")(ab, jb))
      .allocate(tmps.at("0516_bb_vo"))(tmps.at("0516_bb_vo")(ab, ib) =
                                         eri.at("baab_vovv")(ab, ja, ba, cb) *
                                         t2_2p.at("abab")(ba, cb, ja, ib))
      .allocate(tmps.at("0515_bb_vo"))(tmps.at("0515_bb_vo")(ab, ib) =
                                         f.at("bb_ov")(jb, bb) * t2_2p.at("bbbb")(bb, ab, ib, jb))
      .allocate(tmps.at("0514_bb_vo"))(tmps.at("0514_bb_vo")(ab, ib) =
                                         eri.at("baab_vovo")(ab, ja, ba, ib) *
                                         t1_2p.at("aa")(ba, ja))
      .allocate(tmps.at("0513_bb_vo"))(tmps.at("0513_bb_vo")(ab, ib) =
                                         f.at("bb_vv")(ab, bb) * t1_2p.at("bb")(bb, ib))
      .allocate(tmps.at("0527_bb_vo"))(tmps.at("0527_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("0514_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) -= tmps.at("0515_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) += tmps.at("0524_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) -= tmps.at("0519_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) += 2.00 * tmps.at("0521_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) -= tmps.at("0516_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) -= tmps.at("0517_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) += 2.00 * tmps.at("0525_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) += 0.50 * tmps.at("0518_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) += tmps.at("0523_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) -= tmps.at("0522_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) += tmps.at("0513_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) += 0.50 * tmps.at("0520_bb_vo")(ab, ib))(
        tmps.at("0527_bb_vo")(ab, ib) += tmps.at("0526_bb_vo")(ab, ib))
      .deallocate(tmps.at("0526_bb_vo"))
      .deallocate(tmps.at("0525_bb_vo"))
      .deallocate(tmps.at("0524_bb_vo"))
      .deallocate(tmps.at("0523_bb_vo"))
      .deallocate(tmps.at("0522_bb_vo"))
      .deallocate(tmps.at("0521_bb_vo"))
      .deallocate(tmps.at("0520_bb_vo"))
      .deallocate(tmps.at("0519_bb_vo"))
      .deallocate(tmps.at("0518_bb_vo"))
      .deallocate(tmps.at("0517_bb_vo"))
      .deallocate(tmps.at("0516_bb_vo"))
      .deallocate(tmps.at("0515_bb_vo"))
      .deallocate(tmps.at("0514_bb_vo"))
      .deallocate(tmps.at("0513_bb_vo"))
      .allocate(tmps.at("0585_bb_vo"))(tmps.at("0585_bb_vo")(ab, ib) =
                                         -0.50 * tmps.at("0535_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0529_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0530_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0533_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0573_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) += tmps.at("0555_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0584_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= 3.00 * tmps.at("0531_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0580_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= 3.00 * tmps.at("0554_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0558_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) += tmps.at("0553_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0538_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) += tmps.at("0556_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0537_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0543_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0565_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0540_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) += 0.50 * tmps.at("0582_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) += tmps.at("0539_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0550_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) += tmps.at("0549_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) += tmps.at("0527_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0548_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0579_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0567_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= 0.50 * tmps.at("0578_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0561_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= 0.50 * tmps.at("0545_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) += tmps.at("0551_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) += 0.50 * tmps.at("0541_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) += tmps.at("0547_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0542_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0569_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0546_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0559_bb_vo")(ab, ib))(
        tmps.at("0585_bb_vo")(ab, ib) -= tmps.at("0552_bb_vo")(ab, ib))
      .deallocate(tmps.at("0584_bb_vo"))
      .deallocate(tmps.at("0582_bb_vo"))
      .deallocate(tmps.at("0580_bb_vo"))
      .deallocate(tmps.at("0579_bb_vo"))
      .deallocate(tmps.at("0578_bb_vo"))
      .deallocate(tmps.at("0573_bb_vo"))
      .deallocate(tmps.at("0569_bb_vo"))
      .deallocate(tmps.at("0567_bb_vo"))
      .deallocate(tmps.at("0565_bb_vo"))
      .deallocate(tmps.at("0561_bb_vo"))
      .deallocate(tmps.at("0559_bb_vo"))
      .deallocate(tmps.at("0558_bb_vo"))
      .deallocate(tmps.at("0556_bb_vo"))
      .deallocate(tmps.at("0555_bb_vo"))
      .deallocate(tmps.at("0554_bb_vo"))
      .deallocate(tmps.at("0553_bb_vo"))
      .deallocate(tmps.at("0552_bb_vo"))
      .deallocate(tmps.at("0551_bb_vo"))
      .deallocate(tmps.at("0550_bb_vo"))
      .deallocate(tmps.at("0549_bb_vo"))
      .deallocate(tmps.at("0548_bb_vo"))
      .deallocate(tmps.at("0547_bb_vo"))
      .deallocate(tmps.at("0546_bb_vo"))
      .deallocate(tmps.at("0545_bb_vo"))
      .deallocate(tmps.at("0543_bb_vo"))
      .deallocate(tmps.at("0542_bb_vo"))
      .deallocate(tmps.at("0541_bb_vo"))
      .deallocate(tmps.at("0540_bb_vo"))
      .deallocate(tmps.at("0539_bb_vo"))
      .deallocate(tmps.at("0538_bb_vo"))
      .deallocate(tmps.at("0537_bb_vo"))
      .deallocate(tmps.at("0535_bb_vo"))
      .deallocate(tmps.at("0533_bb_vo"))
      .deallocate(tmps.at("0531_bb_vo"))
      .deallocate(tmps.at("0530_bb_vo"))
      .deallocate(tmps.at("0529_bb_vo"))
      .deallocate(tmps.at("0527_bb_vo"))
      .allocate(tmps.at("0610_bb_vo"))(tmps.at("0610_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("0591_bb_vo")(ab, ib))(
        tmps.at("0610_bb_vo")(ab, ib) += tmps.at("0587_bb_vo")(ab, ib))(
        tmps.at("0610_bb_vo")(ab, ib) -= tmps.at("0605_bb_vo")(ab, ib))(
        tmps.at("0610_bb_vo")(ab, ib) += tmps.at("0585_bb_vo")(ab, ib))(
        tmps.at("0610_bb_vo")(ab, ib) -= tmps.at("0592_bb_vo")(ab, ib))(
        tmps.at("0610_bb_vo")(ab, ib) += tmps.at("0609_bb_vo")(ab, ib))(
        tmps.at("0610_bb_vo")(ab, ib) -= tmps.at("0599_bb_vo")(ab, ib))(
        tmps.at("0610_bb_vo")(ab, ib) -= tmps.at("0597_bb_vo")(ab, ib))
      .deallocate(tmps.at("0609_bb_vo"))
      .deallocate(tmps.at("0605_bb_vo"))
      .deallocate(tmps.at("0599_bb_vo"))
      .deallocate(tmps.at("0597_bb_vo"))
      .deallocate(tmps.at("0592_bb_vo"))
      .deallocate(tmps.at("0591_bb_vo"))
      .deallocate(tmps.at("0587_bb_vo"))
      .deallocate(tmps.at("0585_bb_vo"))(r1_2p.at("bb")(ab, ib) +=
                                         2.00 * tmps.at("0610_bb_vo")(ab, ib))
      .deallocate(tmps.at("0610_bb_vo"))
      .allocate(tmps.at("0657_bb_vo"))(tmps.at("0657_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0596_bb_oo")(jb, ib))
      .allocate(tmps.at("0656_bb_vo"))(tmps.at("0656_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0289_bb_oo")(jb, ib))
      .allocate(tmps.at("0654_bb_oo"))(tmps.at("0654_bb_oo")(ib, jb) =
                                         t1_1p.at("bb")(ab, kb) *
                                         tmps.at("0256_bbbb_oovo")(ib, kb, ab, jb))
      .allocate(tmps.at("0655_bb_vo"))(tmps.at("0655_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0654_bb_oo")(jb, ib))
      .allocate(tmps.at("0653_bb_vo"))(tmps.at("0653_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0598_bb_oo")(jb, ib))
      .allocate(tmps.at("0651_bb_vo"))(tmps.at("0651_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0276_bb_oo")(jb, ib))
      .allocate(tmps.at("0650_bb_vo"))(tmps.at("0650_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0141_bbbb_vovo")(ab, jb, bb, ib))
      .allocate(tmps.at("0649_bb_vo"))(tmps.at("0649_bb_vo")(ab, ib) =
                                         t1.at("bb")(bb, ib) * tmps.at("0544_bb_vv")(ab, bb))
      .allocate(tmps.at("0648_bb_vo"))(tmps.at("0648_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0281_bb_oo")(jb, ib))
      .allocate(tmps.at("0647_bb_vo"))(tmps.at("0647_bb_vo")(ab, ib) =
                                         t2.at("abab")(ba, ab, ja, ib) *
                                         tmps.at("0365_aa_ov")(ja, ba))
      .allocate(tmps.at("0646_bb_vo"))(tmps.at("0646_bb_vo")(ab, ib) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0135_baab_vovo")(ab, ja, ba, ib))
      .allocate(tmps.at("0645_bb_vo"))(tmps.at("0645_bb_vo")(ab, ib) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0033_baba_voov")(ab, ja, ib, ba))
      .allocate(tmps.at("0644_bb_vo"))(tmps.at("0644_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0534_bb_oo")(jb, ib))
      .allocate(tmps.at("0642_bb_oo"))(tmps.at("0642_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovo")(ib, kb, ab, jb) *
                                         t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0643_bb_vo"))(tmps.at("0643_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0642_bb_oo")(jb, ib))
      .allocate(tmps.at("0641_bb_vo"))(tmps.at("0641_bb_vo")(ab, ib) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0137_baba_voov")(ab, ja, ib, ba))
      .allocate(tmps.at("0640_bb_vo"))(tmps.at("0640_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0566_bb_oo")(jb, ib))
      .allocate(tmps.at("0639_bb_vo"))(tmps.at("0639_bb_vo")(ab, ib) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0133_baab_vovo")(ab, ja, ba, ib))
      .allocate(tmps.at("0638_bb_vo"))(tmps.at("0638_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0273_bb_oo")(jb, ib))
      .allocate(tmps.at("0637_bb_vo"))(tmps.at("0637_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(bb, ib) * tmps.at("0581_bb_vv")(ab, bb))
      .allocate(tmps.at("0636_bb_vo"))(tmps.at("0636_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0568_bb_oo")(jb, ib))
      .allocate(tmps.at("0635_bb_vo"))(tmps.at("0635_bb_vo")(ab, ib) =
                                         t2_1p.at("abab")(ba, ab, ja, ib) *
                                         tmps.at("0232_aa_ov")(ja, ba))
      .allocate(tmps.at("0634_bb_vo"))(tmps.at("0634_bb_vo")(ab, ib) =
                                         t2_1p.at("bbbb")(bb, ab, ib, jb) *
                                         tmps.at("0237_bb_ov")(jb, bb))
      .allocate(tmps.at("0633_bb_vo"))(tmps.at("0633_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0144_bbbb_voov")(ab, jb, ib, bb))
      .allocate(tmps.at("0632_bb_vo"))(tmps.at("0632_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0528_bb_oo")(jb, ib))
      .allocate(tmps.at("0631_bb_vo"))(tmps.at("0631_bb_vo")(ab, ib) =
                                         t1.at("bb")(bb, ib) * tmps.at("0583_bb_vv")(ab, bb))
      .allocate(tmps.at("0630_bb_vo"))(tmps.at("0630_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(bb, ib) * tmps.at("0505_bb_vv")(ab, bb))
      .allocate(tmps.at("0629_bb_vo"))(tmps.at("0629_bb_vo")(ab, ib) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0271_bb_oo")(jb, ib))
      .allocate(tmps.at("0628_bb_vo"))(tmps.at("0628_bb_vo")(ab, ib) =
                                         t2_1p.at("abab")(ba, ab, ja, ib) *
                                         tmps.at("0224_aa_ov")(ja, ba))
      .allocate(tmps.at("0627_bb_vo"))(tmps.at("0627_bb_vo")(ab, ib) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0035_baba_voov")(ab, ja, ib, ba))
      .allocate(tmps.at("0626_bb_vo"))(tmps.at("0626_bb_vo")(ab, ib) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0131_bbbb_vovo")(ab, jb, bb, ib))
      .allocate(tmps.at("0624_bb_vo"))(tmps.at("0624_bb_vo")(ab, ib) =
                                         f.at("bb_oo")(jb, ib) * t1_1p.at("bb")(ab, jb))
      .allocate(tmps.at("0623_bb_vo"))(tmps.at("0623_bb_vo")(ab, ib) =
                                         scalars.at("0013")() * t1_2p.at("bb")(ab, ib))
      .allocate(tmps.at("0622_bb_vo"))(tmps.at("0622_bb_vo")(ab, ib) =
                                         scalars.at("0015")() * t1_2p.at("bb")(ab, ib))
      .allocate(tmps.at("0621_bb_vo"))(tmps.at("0621_bb_vo")(ab, ib) =
                                         scalars.at("0002")() * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("0620_bb_vo"))(tmps.at("0620_bb_vo")(ab, ib) =
                                         eri.at("bbbb_vovo")(ab, jb, bb, ib) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0619_bb_vo"))(tmps.at("0619_bb_vo")(ab, ib) =
                                         scalars.at("0001")() * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("0618_bb_vo"))(tmps.at("0618_bb_vo")(ab, ib) =
                                         eri.at("bbbb_oovo")(jb, kb, bb, ib) *
                                         t2_1p.at("bbbb")(bb, ab, kb, jb))
      .allocate(tmps.at("0617_bb_vo"))(tmps.at("0617_bb_vo")(ab, ib) =
                                         f.at("aa_ov")(ja, ba) * t2_1p.at("abab")(ba, ab, ja, ib))
      .allocate(tmps.at("0616_bb_vo"))(tmps.at("0616_bb_vo")(ab, ib) =
                                         eri.at("baab_vovo")(ab, ja, ba, ib) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0615_bb_vo"))(tmps.at("0615_bb_vo")(ab, ib) =
                                         eri.at("bbbb_vovv")(ab, jb, bb, cb) *
                                         t2_1p.at("bbbb")(bb, cb, ib, jb))
      .allocate(tmps.at("0614_bb_vo"))(tmps.at("0614_bb_vo")(ab, ib) =
                                         eri.at("baab_vovv")(ab, ja, ba, cb) *
                                         t2_1p.at("abab")(ba, cb, ja, ib))
      .allocate(tmps.at("0613_bb_vo"))(tmps.at("0613_bb_vo")(ab, ib) =
                                         eri.at("abab_oovo")(ja, kb, ba, ib) *
                                         t2_1p.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0612_bb_vo"))(tmps.at("0612_bb_vo")(ab, ib) =
                                         f.at("bb_ov")(jb, bb) * t2_1p.at("bbbb")(bb, ab, ib, jb))
      .allocate(tmps.at("0611_bb_vo"))(tmps.at("0611_bb_vo")(ab, ib) =
                                         f.at("bb_vv")(ab, bb) * t1_1p.at("bb")(bb, ib))
      .allocate(tmps.at("0625_bb_vo"))(tmps.at("0625_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("0612_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) += 2.00 * tmps.at("0622_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) += tmps.at("0621_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) += 2.00 * tmps.at("0623_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) += 0.50 * tmps.at("0618_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) -= tmps.at("0614_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) += tmps.at("0611_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) += tmps.at("0619_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) -= tmps.at("0624_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) -= tmps.at("0613_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) += 0.50 * tmps.at("0615_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) -= tmps.at("0616_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) -= tmps.at("0620_bb_vo")(ab, ib))(
        tmps.at("0625_bb_vo")(ab, ib) += tmps.at("0617_bb_vo")(ab, ib))
      .deallocate(tmps.at("0624_bb_vo"))
      .deallocate(tmps.at("0623_bb_vo"))
      .deallocate(tmps.at("0622_bb_vo"))
      .deallocate(tmps.at("0621_bb_vo"))
      .deallocate(tmps.at("0620_bb_vo"))
      .deallocate(tmps.at("0619_bb_vo"))
      .deallocate(tmps.at("0618_bb_vo"))
      .deallocate(tmps.at("0617_bb_vo"))
      .deallocate(tmps.at("0616_bb_vo"))
      .deallocate(tmps.at("0615_bb_vo"))
      .deallocate(tmps.at("0614_bb_vo"))
      .deallocate(tmps.at("0613_bb_vo"))
      .deallocate(tmps.at("0612_bb_vo"))
      .deallocate(tmps.at("0611_bb_vo"))
      .allocate(tmps.at("0652_bb_vo"))(tmps.at("0652_bb_vo")(ab, ib) =
                                         -0.50 * tmps.at("0644_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= 0.50 * tmps.at("0649_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0630_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0646_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) += tmps.at("0625_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) += 0.50 * tmps.at("0651_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0648_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) += tmps.at("0650_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0634_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0645_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0633_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) += tmps.at("0635_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0638_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) += 0.50 * tmps.at("0637_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) += tmps.at("0647_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0631_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) += tmps.at("0641_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0640_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0636_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0639_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) += tmps.at("0628_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) += tmps.at("0643_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0632_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0627_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0629_bb_vo")(ab, ib))(
        tmps.at("0652_bb_vo")(ab, ib) -= tmps.at("0626_bb_vo")(ab, ib))
      .deallocate(tmps.at("0651_bb_vo"))
      .deallocate(tmps.at("0650_bb_vo"))
      .deallocate(tmps.at("0649_bb_vo"))
      .deallocate(tmps.at("0648_bb_vo"))
      .deallocate(tmps.at("0647_bb_vo"))
      .deallocate(tmps.at("0646_bb_vo"))
      .deallocate(tmps.at("0645_bb_vo"))
      .deallocate(tmps.at("0644_bb_vo"))
      .deallocate(tmps.at("0643_bb_vo"))
      .deallocate(tmps.at("0641_bb_vo"))
      .deallocate(tmps.at("0640_bb_vo"))
      .deallocate(tmps.at("0639_bb_vo"))
      .deallocate(tmps.at("0638_bb_vo"))
      .deallocate(tmps.at("0637_bb_vo"))
      .deallocate(tmps.at("0636_bb_vo"))
      .deallocate(tmps.at("0635_bb_vo"))
      .deallocate(tmps.at("0634_bb_vo"))
      .deallocate(tmps.at("0633_bb_vo"))
      .deallocate(tmps.at("0632_bb_vo"))
      .deallocate(tmps.at("0631_bb_vo"))
      .deallocate(tmps.at("0630_bb_vo"))
      .deallocate(tmps.at("0629_bb_vo"))
      .deallocate(tmps.at("0628_bb_vo"))
      .deallocate(tmps.at("0627_bb_vo"))
      .deallocate(tmps.at("0626_bb_vo"))
      .deallocate(tmps.at("0625_bb_vo"))
      .allocate(tmps.at("0658_bb_vo"))(tmps.at("0658_bb_vo")(ab, ib) =
                                         -1.00 * tmps.at("0652_bb_vo")(ab, ib))(
        tmps.at("0658_bb_vo")(ab, ib) += tmps.at("0655_bb_vo")(ab, ib))(
        tmps.at("0658_bb_vo")(ab, ib) += tmps.at("0656_bb_vo")(ab, ib))(
        tmps.at("0658_bb_vo")(ab, ib) += tmps.at("0653_bb_vo")(ab, ib))(
        tmps.at("0658_bb_vo")(ab, ib) += tmps.at("0657_bb_vo")(ab, ib))
      .deallocate(tmps.at("0657_bb_vo"))
      .deallocate(tmps.at("0656_bb_vo"))
      .deallocate(tmps.at("0655_bb_vo"))
      .deallocate(tmps.at("0653_bb_vo"))
      .deallocate(tmps.at("0652_bb_vo"))(r1_1p.at("bb")(ab, ib) -= tmps.at("0658_bb_vo")(ab, ib))
      .deallocate(tmps.at("0658_bb_vo"))
      .allocate(tmps.at("0674_aaaa_vooo"))(tmps.at("0674_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0363_aaaa_vovo")(aa, ia, ba, ka))
      .allocate(tmps.at("0675_aaaa_vvoo"))(tmps.at("0675_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0674_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0671_aaaa_vooo"))(tmps.at("0671_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, ka) * f.at("aa_ov")(ia, ba))
      .allocate(tmps.at("0672_aaaa_vvoo"))(tmps.at("0672_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0671_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0668_aaaa_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("aaaa_oovv")(la, ia, ca, ba) * t1.at("aa")(ca, la))(
        tmps.at("0668_aaaa_ovoo")(ia, aa, ja, ka) =
          tmps.at("bin_aa_vo")(ba, ia) * t2.at("aaaa")(ba, aa, ja, ka))
      .allocate(tmps.at("0667_aaaa_ovoo"))(
        tmps.at("bin_aa_vo")(ba, ia) = eri.at("abab_oovv")(ia, lb, ba, cb) * t1.at("bb")(cb, lb))(
        tmps.at("0667_aaaa_ovoo")(ia, aa, ja, ka) =
          tmps.at("bin_aa_vo")(ba, ia) * t2.at("aaaa")(ba, aa, ja, ka))
      .allocate(tmps.at("0669_aaaa_ovoo"))(tmps.at("0669_aaaa_ovoo")(ia, aa, ja, ka) =
                                             tmps.at("0667_aaaa_ovoo")(ia, aa, ja, ka))(
        tmps.at("0669_aaaa_ovoo")(ia, aa, ja, ka) += tmps.at("0668_aaaa_ovoo")(ia, aa, ja, ka))
      .deallocate(tmps.at("0668_aaaa_ovoo"))
      .deallocate(tmps.at("0667_aaaa_ovoo"))
      .allocate(tmps.at("0670_aaaa_vvoo"))(tmps.at("0670_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0669_aaaa_ovoo")(ka, ba, ia, ja))
      .allocate(tmps.at("0666_aaaa_vvoo"))(tmps.at("0666_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0226_aa_vv")(ba, ca))
      .allocate(tmps.at("0664_aa_vv"))(tmps.at("0664_aa_vv")(aa, ba) =
                                         eri.at("abab_vovv")(aa, ib, ba, cb) * t1.at("bb")(cb, ib))
      .allocate(tmps.at("0665_aaaa_vvoo"))(tmps.at("0665_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0664_aa_vv")(ba, ca))
      .allocate(tmps.at("0662_aaaa_vooo"))(tmps.at("0662_aaaa_vooo")(aa, ia, ja, ka) =
                                             eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                             t2.at("aaaa")(ba, ca, ja, ka))
      .allocate(tmps.at("0663_aaaa_vvoo"))(tmps.at("0663_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0662_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0660_aaaa_vvoo"))(tmps.at("0660_aaaa_vvoo")(aa, ba, ia, ja) =
                                             f.at("aa_vv")(aa, ca) * t2.at("aaaa")(ca, ba, ia, ja))
      .allocate(tmps.at("0659_aaaa_vvoo"))(tmps.at("0659_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_vooo")(aa, ka, ia, ja) *
                                             t1.at("aa")(ba, ka))
      .allocate(tmps.at("0661_aaaa_vvoo"))(tmps.at("0661_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0659_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0661_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0660_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0660_aaaa_vvoo"))
      .deallocate(tmps.at("0659_aaaa_vvoo"))
      .allocate(tmps.at("0673_aaaa_vvoo"))(tmps.at("0673_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.50 * tmps.at("0663_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0673_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0665_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0673_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0670_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0673_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0666_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0673_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0672_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0673_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0661_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("0672_aaaa_vvoo"))
      .deallocate(tmps.at("0670_aaaa_vvoo"))
      .deallocate(tmps.at("0666_aaaa_vvoo"))
      .deallocate(tmps.at("0665_aaaa_vvoo"))
      .deallocate(tmps.at("0663_aaaa_vvoo"))
      .deallocate(tmps.at("0661_aaaa_vvoo"))
      .allocate(tmps.at("0676_aaaa_vvoo"))(tmps.at("0676_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("0673_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0676_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0675_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0675_aaaa_vvoo"))
      .deallocate(tmps.at("0673_aaaa_vvoo"))(r2.at("aaaa")(aa, ba, ia, ja) +=
                                             tmps.at("0676_aaaa_vvoo")(ba, aa, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0676_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0676_aaaa_vvoo"))
      .allocate(tmps.at("0691_aaaa_vvoo"))(tmps.at("0691_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0248_aa_oo")(ka, ja))
      .allocate(tmps.at("0681_aaaa_oooo"))(tmps.at("0681_aaaa_oooo")(ia, ja, ka, la) =
                                             t1.at("aa")(aa, ka) *
                                             eri.at("aaaa_oovo")(ia, ja, aa, la))
      .allocate(tmps.at("0689_aaaa_vooo"))(tmps.at("0689_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             tmps.at("0681_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("0690_aaaa_vvoo"))(tmps.at("0690_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0689_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0687_aaaa_vvoo"))(tmps.at("0687_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0241_aa_oo")(ka, ja))
      .allocate(tmps.at("0686_aaaa_vvoo"))(tmps.at("0686_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0390_aa_oo")(ka, ja))
      .allocate(tmps.at("0685_aaaa_vvoo"))(tmps.at("0685_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0234_aa_oo")(ka, ja))
      .allocate(tmps.at("0684_aaaa_vvoo"))(tmps.at("0684_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0346_aaaa_voov")(ba, ka, ja, ca))
      .allocate(tmps.at("0683_aaaa_vvoo"))(tmps.at("0683_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0376_abab_voov")(ba, kb, ja, cb))
      .allocate(tmps.at("0682_aaaa_vvoo"))(tmps.at("0682_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0681_aaaa_oooo")(la, ka, ia, ja))
      .allocate(tmps.at("0680_aaaa_vvoo"))(tmps.at("0680_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0243_aa_oo")(ka, ja))
      .allocate(tmps.at("0678_aaaa_vvoo"))(tmps.at("0678_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(ca, ia) *
                                             eri.at("aaaa_vvvo")(aa, ba, ca, ja))
      .allocate(tmps.at("0677_aaaa_vvoo"))(tmps.at("0677_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) * f.at("aa_oo")(ka, ja))
      .allocate(tmps.at("0679_aaaa_vvoo"))(tmps.at("0679_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0678_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0679_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0677_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0678_aaaa_vvoo"))
      .deallocate(tmps.at("0677_aaaa_vvoo"))
      .allocate(tmps.at("0688_aaaa_vvoo"))(tmps.at("0688_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.50 * tmps.at("0686_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0688_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.50 * tmps.at("0682_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0688_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0680_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0688_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0684_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0688_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0685_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0688_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0679_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0688_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0687_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0688_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0683_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0687_aaaa_vvoo"))
      .deallocate(tmps.at("0686_aaaa_vvoo"))
      .deallocate(tmps.at("0685_aaaa_vvoo"))
      .deallocate(tmps.at("0684_aaaa_vvoo"))
      .deallocate(tmps.at("0683_aaaa_vvoo"))
      .deallocate(tmps.at("0682_aaaa_vvoo"))
      .deallocate(tmps.at("0680_aaaa_vvoo"))
      .deallocate(tmps.at("0679_aaaa_vvoo"))
      .allocate(tmps.at("0692_aaaa_vvoo"))(tmps.at("0692_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0688_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0692_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0690_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0692_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0691_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("0691_aaaa_vvoo"))
      .deallocate(tmps.at("0690_aaaa_vvoo"))
      .deallocate(tmps.at("0688_aaaa_vvoo"))(r2.at("aaaa")(aa, ba, ia, ja) +=
                                             tmps.at("0692_aaaa_vvoo")(ba, aa, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0692_aaaa_vvoo")(ba, aa, ja, ia))
      .deallocate(tmps.at("0692_aaaa_vvoo"))
      .allocate(tmps.at("0709_aa_vo"))(tmps.at("0709_aa_vo")(aa, ia) =
                                         dp.at("aa_oo")(ja, ia) * t1.at("aa")(aa, ja))
      .allocate(tmps.at("0708_aa_vo"))(tmps.at("0708_aa_vo")(aa, ia) =
                                         dp.at("aa_vv")(aa, ba) * t1.at("aa")(ba, ia))
      .allocate(tmps.at("0707_aa_vo"))(tmps.at("0707_aa_vo")(aa, ia) =
                                         dp.at("aa_ov")(ja, ba) * t2.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0706_aa_vo"))(tmps.at("0706_aa_vo")(aa, ia) =
                                         dp.at("bb_ov")(jb, bb) * t2.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0710_aa_vo"))(tmps.at("0710_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0706_aa_vo")(aa, ia))(
        tmps.at("0710_aa_vo")(aa, ia) += tmps.at("0709_aa_vo")(aa, ia))(
        tmps.at("0710_aa_vo")(aa, ia) -= tmps.at("0708_aa_vo")(aa, ia))(
        tmps.at("0710_aa_vo")(aa, ia) += tmps.at("0707_aa_vo")(aa, ia))
      .deallocate(tmps.at("0709_aa_vo"))
      .deallocate(tmps.at("0708_aa_vo"))
      .deallocate(tmps.at("0707_aa_vo"))
      .deallocate(tmps.at("0706_aa_vo"))(r1_1p.at("aa")(aa, ia) -= tmps.at("0710_aa_vo")(aa, ia))(
        r1_1p.at("aa")(aa, ia) -= 2.00 * t0_2p * tmps.at("0710_aa_vo")(aa, ia))(
        r1.at("aa")(aa, ia) -= t0_1p * tmps.at("0710_aa_vo")(aa, ia))
      .allocate(tmps.at("0022_aa_oo"))(tmps.at("0022_aa_oo")(ia, ja) =
                                         dp.at("aa_ov")(ia, aa) * t1.at("aa")(aa, ja))
      .allocate(tmps.at("0713_aa_vo"))(tmps.at("0713_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0022_aa_oo")(ja, ia))(
        r1_1p.at("aa")(aa, ia) -= tmps.at("0713_aa_vo")(aa, ia))(
        r1_1p.at("aa")(aa, ia) -= 2.00 * t0_2p * tmps.at("0713_aa_vo")(aa, ia))(
        r1.at("aa")(aa, ia) -= t0_1p * tmps.at("0713_aa_vo")(aa, ia))
      .allocate(tmps.at("0716_aaaa_vooo"))(tmps.at("0716_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0037_aaaa_voov")(aa, ia, ka, ba))
      .allocate(tmps.at("0717_aaaa_vvoo"))(tmps.at("0717_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0716_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0698_aaaa_vooo"))(tmps.at("0698_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, la) *
                                             tmps.at("0203_aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("0715_aaaa_vvoo"))(tmps.at("0715_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0698_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0698_aaaa_vooo"))
      .allocate(tmps.at("0714_aaaa_vvoo"))(tmps.at("0714_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0713_aa_vo")(ba, ja))
      .allocate(tmps.at("0711_aaaa_vvoo"))(tmps.at("0711_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0710_aa_vo")(ba, ja))
      .allocate(tmps.at("0704_aaaa_vooo"))(tmps.at("0704_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("abab")(aa, bb, ja, lb) *
                                             eri.at("abba_oovo")(ia, lb, bb, ka))
      .allocate(tmps.at("0705_aaaa_vvoo"))(tmps.at("0705_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0704_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0693_aaaa_vooo"))(tmps.at("0693_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, la) *
                                             eri.at("aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("0703_aaaa_vvoo"))(tmps.at("0703_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0693_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0693_aaaa_vooo"))
      .allocate(tmps.at("0701_aaaa_vooo"))(tmps.at("0701_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(ba, ja) *
                                             eri.at("aaaa_vovo")(aa, ia, ba, ka))
      .allocate(tmps.at("0702_aaaa_vvoo"))(tmps.at("0702_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0701_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0700_aaaa_vvoo"))(tmps.at("0700_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0363_aaaa_vovo")(ba, ka, ca, ja))
      .allocate(tmps.at("0699_aaaa_vvoo"))(tmps.at("0699_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0230_abba_vovo")(ba, kb, cb, ja))
      .allocate(tmps.at("0696_aaaa_vvoo"))(tmps.at("0696_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_vovo")(aa, ka, ca, ia) *
                                             t2.at("aaaa")(ca, ba, ja, ka))
      .allocate(tmps.at("0695_aaaa_vvoo"))(tmps.at("0695_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("abba_vovo")(aa, kb, cb, ia) *
                                             t2.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0694_aaaa_vvoo"))(tmps.at("0694_aaaa_vvoo")(aa, ba, ia, ja) =
                                             dp.at("aa_vo")(aa, ia) * t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0697_aaaa_vvoo"))(tmps.at("0697_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0695_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0697_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0696_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0697_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0694_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0696_aaaa_vvoo"))
      .deallocate(tmps.at("0695_aaaa_vvoo"))
      .deallocate(tmps.at("0694_aaaa_vvoo"))
      .allocate(tmps.at("0712_aaaa_vvoo"))(tmps.at("0712_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0703_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0712_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0711_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0712_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0705_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0712_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0700_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0712_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0697_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("0712_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0699_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0712_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0702_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("0711_aaaa_vvoo"))
      .deallocate(tmps.at("0705_aaaa_vvoo"))
      .deallocate(tmps.at("0703_aaaa_vvoo"))
      .deallocate(tmps.at("0702_aaaa_vvoo"))
      .deallocate(tmps.at("0700_aaaa_vvoo"))
      .deallocate(tmps.at("0699_aaaa_vvoo"))
      .deallocate(tmps.at("0697_aaaa_vvoo"))
      .allocate(tmps.at("0718_aaaa_vvoo"))(tmps.at("0718_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0712_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("0718_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0715_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("0718_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0717_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0718_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0714_aaaa_vvoo")(ba, aa, ja, ia))
      .deallocate(tmps.at("0717_aaaa_vvoo"))
      .deallocate(tmps.at("0715_aaaa_vvoo"))
      .deallocate(tmps.at("0714_aaaa_vvoo"))
      .deallocate(tmps.at("0712_aaaa_vvoo"))(r2.at("aaaa")(aa, ba, ia, ja) +=
                                             tmps.at("0718_aaaa_vvoo")(aa, ba, ja, ia))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0718_aaaa_vvoo")(aa, ba, ia, ja))(
        r2.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0718_aaaa_vvoo")(ba, aa, ja, ia))(
        r2.at("aaaa")(aa, ba, ia, ja) += tmps.at("0718_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("0718_aaaa_vvoo"))
      .allocate(tmps.at("0736_aa_vo"))(tmps.at("0736_aa_vo")(aa, ia) =
                                         dp.at("aa_ov")(ja, ba) * t2_1p.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0735_aa_vo"))(tmps.at("0735_aa_vo")(aa, ia) =
                                         dp.at("bb_ov")(jb, bb) * t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0737_aa_vo"))(tmps.at("0737_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0736_aa_vo")(aa, ia))(
        tmps.at("0737_aa_vo")(aa, ia) += tmps.at("0735_aa_vo")(aa, ia))
      .deallocate(tmps.at("0736_aa_vo"))
      .deallocate(tmps.at("0735_aa_vo"))(r1_1p.at("aa")(aa, ia) +=
                                         t0_1p * tmps.at("0737_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) += 2.00 * tmps.at("0737_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) += 4.00 * t0_2p * tmps.at("0737_aa_vo")(aa, ia))(
        r1.at("aa")(aa, ia) += tmps.at("0737_aa_vo")(aa, ia))
      .allocate(tmps.at("0740_aa_vo"))(tmps.at("0740_aa_vo")(aa, ia) =
                                         dp.at("aa_oo")(ja, ia) * t1_1p.at("aa")(aa, ja))
      .allocate(tmps.at("0739_aa_vo"))(tmps.at("0739_aa_vo")(aa, ia) =
                                         dp.at("aa_vv")(aa, ba) * t1_1p.at("aa")(ba, ia))
      .allocate(tmps.at("0741_aa_vo"))(tmps.at("0741_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0739_aa_vo")(aa, ia))(
        tmps.at("0741_aa_vo")(aa, ia) += tmps.at("0740_aa_vo")(aa, ia))
      .deallocate(tmps.at("0740_aa_vo"))
      .deallocate(tmps.at("0739_aa_vo"))(r1_1p.at("aa")(aa, ia) -=
                                         t0_1p * tmps.at("0741_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 2.00 * tmps.at("0741_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 4.00 * t0_2p * tmps.at("0741_aa_vo")(aa, ia))(
        r1.at("aa")(aa, ia) -= tmps.at("0741_aa_vo")(aa, ia))
      .allocate(tmps.at("0764_aa_vo"))(tmps.at("0764_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0023_aa_oo")(ja, ia))
      .allocate(tmps.at("0763_aa_vo"))(tmps.at("0763_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0022_aa_oo")(ja, ia))
      .allocate(tmps.at("0765_aa_vo"))(tmps.at("0765_aa_vo")(aa, ia) =
                                         tmps.at("0763_aa_vo")(aa, ia))(
        tmps.at("0765_aa_vo")(aa, ia) += tmps.at("0764_aa_vo")(aa, ia))
      .deallocate(tmps.at("0764_aa_vo"))
      .deallocate(tmps.at("0763_aa_vo"))(r1_1p.at("aa")(aa, ia) -=
                                         t0_1p * tmps.at("0765_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 2.00 * tmps.at("0765_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 4.00 * t0_2p * tmps.at("0765_aa_vo")(aa, ia))(
        r1.at("aa")(aa, ia) -= tmps.at("0765_aa_vo")(aa, ia))
      .allocate(tmps.at("0767_aaaa_vooo"))(tmps.at("0767_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0037_aaaa_voov")(aa, ia, ka, ba))
      .allocate(tmps.at("0768_aaaa_vvoo"))(tmps.at("0768_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0767_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0766_aaaa_vvoo"))(tmps.at("0766_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0765_aa_vo")(ba, ja))
      .allocate(tmps.at("0762_aaaa_vvoo"))(tmps.at("0762_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0689_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0760_aaaa_vooo"))(tmps.at("0760_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0363_aaaa_vovo")(aa, ia, ba, ka))
      .allocate(tmps.at("0761_aaaa_vvoo"))(tmps.at("0761_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0760_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0759_aaaa_vvoo"))(tmps.at("0759_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0716_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0723_aaaa_vooo"))(tmps.at("0723_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, la) *
                                             tmps.at("0410_aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("0758_aaaa_vvoo"))(tmps.at("0758_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0723_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0723_aaaa_vooo"))
      .allocate(tmps.at("0756_aaaa_vooo"))(tmps.at("0756_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0039_aaaa_voov")(aa, ia, ka, ba))
      .allocate(tmps.at("0757_aaaa_vvoo"))(tmps.at("0757_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0756_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0755_aaaa_vvoo"))(tmps.at("0755_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0713_aa_vo")(ba, ja))
      .allocate(tmps.at("0753_aaaa_vvoo"))(tmps.at("0753_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0710_aa_vo")(ba, ja))
      .allocate(tmps.at("0752_aaaa_vvoo"))(tmps.at("0752_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0230_abba_vovo")(ba, kb, cb, ja))
      .allocate(tmps.at("0749_aaaa_vooo"))(tmps.at("0749_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, la) *
                                             eri.at("aaaa_oovo")(la, ia, ba, ka))
      .allocate(tmps.at("0748_aaaa_ovoo"))(tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) =
                                             eri.at("aaaa_oovv")(la, ia, ca, ba) *
                                             t1.at("aa")(ca, ja))(
        tmps.at("0748_aaaa_ovoo")(ia, aa, ja, ka) =
          tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) * t2.at("aaaa")(ba, aa, ka, la))
      .allocate(tmps.at("0750_aaaa_vooo"))(tmps.at("0750_aaaa_vooo")(aa, ia, ja, ka) =
                                             -1.00 * tmps.at("0748_aaaa_ovoo")(ia, aa, ka, ja))(
        tmps.at("0750_aaaa_vooo")(aa, ia, ja, ka) += tmps.at("0749_aaaa_vooo")(aa, ia, ja, ka))
      .deallocate(tmps.at("0749_aaaa_vooo"))
      .deallocate(tmps.at("0748_aaaa_ovoo"))
      .allocate(tmps.at("0751_aaaa_vvoo"))(tmps.at("0751_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0750_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0745_aaaa_vooo"))(tmps.at("0745_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_1p.at("aaaa")(ba, aa, ja, la) *
                                             eri.at("aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("0744_aaaa_ovoo"))(tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) =
                                             eri.at("aaaa_oovv")(ia, la, ca, ba) *
                                             t1.at("aa")(ca, ja))(
        tmps.at("0744_aaaa_ovoo")(ia, aa, ja, ka) =
          tmps.at("bin_aaaa_vooo")(ba, ia, ja, la) * t2_1p.at("aaaa")(ba, aa, ka, la))
      .allocate(tmps.at("0743_aaaa_vooo"))(tmps.at("0743_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_1p.at("abab")(aa, bb, ja, lb) *
                                             eri.at("abba_oovo")(ia, lb, bb, ka))
      .allocate(tmps.at("0746_aaaa_vooo"))(tmps.at("0746_aaaa_vooo")(aa, ia, ja, ka) =
                                             -1.00 * tmps.at("0743_aaaa_vooo")(aa, ia, ja, ka))(
        tmps.at("0746_aaaa_vooo")(aa, ia, ja, ka) -= tmps.at("0744_aaaa_ovoo")(ia, aa, ka, ja))(
        tmps.at("0746_aaaa_vooo")(aa, ia, ja, ka) += tmps.at("0745_aaaa_vooo")(aa, ia, ja, ka))
      .deallocate(tmps.at("0745_aaaa_vooo"))
      .deallocate(tmps.at("0744_aaaa_ovoo"))
      .deallocate(tmps.at("0743_aaaa_vooo"))
      .allocate(tmps.at("0747_aaaa_vvoo"))(tmps.at("0747_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0746_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0742_aaaa_vvoo"))(tmps.at("0742_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0741_aa_vo")(ba, ja))
      .allocate(tmps.at("0738_aaaa_vvoo"))(tmps.at("0738_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0737_aa_vo")(ba, ja))
      .allocate(tmps.at("0734_aaaa_vvoo"))(tmps.at("0734_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0704_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0733_aaaa_vvoo"))(tmps.at("0733_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0348_aaaa_vovo")(ba, ka, ca, ja))
      .allocate(tmps.at("0732_aaaa_vvoo"))(tmps.at("0732_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0346_aaaa_voov")(ba, ka, ja, ca))
      .allocate(tmps.at("0731_aaaa_vvoo"))(tmps.at("0731_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0392_abba_vovo")(ba, kb, cb, ja))
      .allocate(tmps.at("0730_aaaa_vvoo"))(tmps.at("0730_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0039_aaaa_voov")(ba, ka, ja, ca))
      .allocate(tmps.at("0729_aaaa_vvoo"))(tmps.at("0729_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0037_aaaa_voov")(ba, ka, ja, ca))
      .allocate(tmps.at("0728_aaaa_vvoo"))(tmps.at("0728_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0701_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0726_aaaa_vooo"))(tmps.at("0726_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_1p.at("aa")(ba, ja) *
                                             eri.at("aaaa_vovo")(aa, ia, ba, ka))
      .allocate(tmps.at("0727_aaaa_vvoo"))(tmps.at("0727_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0726_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("0725_aaaa_vvoo"))(tmps.at("0725_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0376_abab_voov")(ba, kb, ja, cb))
      .allocate(tmps.at("0724_aaaa_vvoo"))(tmps.at("0724_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0363_aaaa_vovo")(ba, ka, ca, ja))
      .allocate(tmps.at("0721_aaaa_vvoo"))(tmps.at("0721_aaaa_vvoo")(aa, ba, ia, ja) =
                                             dp.at("aa_vo")(aa, ia) * t1_2p.at("aa")(ba, ja))
      .allocate(tmps.at("0720_aaaa_vvoo"))(tmps.at("0720_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_vovo")(aa, ka, ca, ia) *
                                             t2_1p.at("aaaa")(ca, ba, ja, ka))
      .allocate(tmps.at("0719_aaaa_vvoo"))(tmps.at("0719_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("abba_vovo")(aa, kb, cb, ia) *
                                             t2_1p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0722_aaaa_vvoo"))(tmps.at("0722_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.50 * tmps.at("0719_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0722_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.50 * tmps.at("0720_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0722_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0721_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("0721_aaaa_vvoo"))
      .deallocate(tmps.at("0720_aaaa_vvoo"))
      .deallocate(tmps.at("0719_aaaa_vvoo"))
      .allocate(tmps.at("0754_aaaa_vvoo"))(tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0724_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0729_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0731_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0733_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0728_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0738_aaaa_vvoo")(ba, aa, ja, ia))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0732_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) +=
        2.00 * tmps.at("0722_aaaa_vvoo")(ba, aa, ja, ia))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0725_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 * tmps.at("0753_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0734_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0727_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0742_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0752_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0751_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0747_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0754_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0730_aaaa_vvoo")(ba, aa, ja, ia))
      .deallocate(tmps.at("0753_aaaa_vvoo"))
      .deallocate(tmps.at("0752_aaaa_vvoo"))
      .deallocate(tmps.at("0751_aaaa_vvoo"))
      .deallocate(tmps.at("0747_aaaa_vvoo"))
      .deallocate(tmps.at("0742_aaaa_vvoo"))
      .deallocate(tmps.at("0738_aaaa_vvoo"))
      .deallocate(tmps.at("0734_aaaa_vvoo"))
      .deallocate(tmps.at("0733_aaaa_vvoo"))
      .deallocate(tmps.at("0732_aaaa_vvoo"))
      .deallocate(tmps.at("0731_aaaa_vvoo"))
      .deallocate(tmps.at("0730_aaaa_vvoo"))
      .deallocate(tmps.at("0729_aaaa_vvoo"))
      .deallocate(tmps.at("0728_aaaa_vvoo"))
      .deallocate(tmps.at("0727_aaaa_vvoo"))
      .deallocate(tmps.at("0725_aaaa_vvoo"))
      .deallocate(tmps.at("0724_aaaa_vvoo"))
      .deallocate(tmps.at("0722_aaaa_vvoo"))
      .allocate(tmps.at("0769_aaaa_vvoo"))(tmps.at("0769_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("0757_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("0769_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0766_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0769_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0768_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("0769_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 * tmps.at("0755_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0769_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0759_aaaa_vvoo")(ba, aa, ja, ia))(
        tmps.at("0769_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0754_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0769_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0761_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0769_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("0762_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("0769_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("0758_aaaa_vvoo")(aa, ba, ja, ia))
      .deallocate(tmps.at("0768_aaaa_vvoo"))
      .deallocate(tmps.at("0766_aaaa_vvoo"))
      .deallocate(tmps.at("0762_aaaa_vvoo"))
      .deallocate(tmps.at("0761_aaaa_vvoo"))
      .deallocate(tmps.at("0759_aaaa_vvoo"))
      .deallocate(tmps.at("0758_aaaa_vvoo"))
      .deallocate(tmps.at("0757_aaaa_vvoo"))
      .deallocate(tmps.at("0755_aaaa_vvoo"))
      .deallocate(tmps.at("0754_aaaa_vvoo"))(r2_1p.at("aaaa")(aa, ba, ia, ja) -=
                                             tmps.at("0769_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += tmps.at("0769_aaaa_vvoo")(aa, ba, ja, ia))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += tmps.at("0769_aaaa_vvoo")(ba, aa, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("0769_aaaa_vvoo")(ba, aa, ja, ia))
      .deallocate(tmps.at("0769_aaaa_vvoo"))
      .allocate(tmps.at("0887_baab_vooo"))(tmps.at("0887_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0135_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("0888_abab_vvoo"))(tmps.at("0888_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0887_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0886_abba_vvoo"))(tmps.at("0886_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0248_aa_oo")(ka, ja))
      .allocate(tmps.at("0830_abab_oooo"))(tmps.at("0830_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abba_oovo")(ia, jb, ab, ka) *
                                             t1.at("bb")(ab, lb))
      .allocate(tmps.at("0829_abab_oooo"))(tmps.at("0829_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t2.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0831_abab_oooo"))(tmps.at("0831_abab_oooo")(ia, jb, ka, lb) =
                                             -1.00 * tmps.at("0830_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("0831_abab_oooo")(ia, jb, ka, lb) += tmps.at("0829_abab_oooo")(ia, jb, ka, lb))
      .deallocate(tmps.at("0830_abab_oooo"))
      .deallocate(tmps.at("0829_abab_oooo"))
      .allocate(tmps.at("0883_baab_vooo"))(tmps.at("0883_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0831_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0882_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0286_abab_oovo")(ia, lb, ba, kb))(
        tmps.at("0882_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1.at("bb")(ab, lb))
      .allocate(tmps.at("0881_baab_vooo"))(tmps.at("0881_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0033_baba_voov")(ab, ia, kb, ba))
      .allocate(tmps.at("0884_abab_ovoo"))(tmps.at("0884_abab_ovoo")(ia, ab, ja, kb) =
                                             tmps.at("0881_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0884_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0882_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0884_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0883_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("0883_baab_vooo"))
      .deallocate(tmps.at("0882_abab_ovoo"))
      .deallocate(tmps.at("0881_baab_vooo"))
      .allocate(tmps.at("0885_abab_vvoo"))(tmps.at("0885_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0884_abab_ovoo")(ka, bb, ia, jb))
      .allocate(tmps.at("0811_abab_vovo"))(tmps.at("0811_abab_vovo")(aa, ib, ba, jb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t1.at("bb")(cb, jb))
      .allocate(tmps.at("0879_abab_vooo"))(tmps.at("0879_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0811_abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("0880_baab_vvoo"))(tmps.at("0880_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0879_abab_vooo")(ba, kb, ia, jb))
      .allocate(tmps.at("0878_abab_vvoo"))(tmps.at("0878_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0289_bb_oo")(kb, jb))
      .allocate(tmps.at("0793_abab_vooo"))(tmps.at("0793_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, lb) *
                                             tmps.at("0256_bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("0877_baab_vvoo"))(tmps.at("0877_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0793_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0793_abab_vooo"));
  }
}

template void exachem::cc::qed_ccsd_os::resid_3<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);