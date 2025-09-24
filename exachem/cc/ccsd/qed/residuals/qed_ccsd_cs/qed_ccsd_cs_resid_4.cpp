/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_cs_resid_4.hpp"

template<typename T>
void exachem::cc::qed_ccsd_cs::resid_part4(
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
      .allocate(tmps.at("0733_baab_vvoo"))(tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) =
                                             -0.50 * tmps.at("0621_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0568_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0679_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0654_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0647_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) +=
        3.00 * tmps.at("0721_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0594_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0674_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -=
        0.50 * tmps.at("0678_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0709_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0584_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0670_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0580_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0665_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0659_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0565_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0634_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0656_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0588_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0427_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0694_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0681_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) +=
        3.00 * tmps.at("0720_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0601_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0675_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0616_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -=
        2.00 * tmps.at("0632_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0619_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0608_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0606_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0593_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0672_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0715_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0589_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0680_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0718_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0586_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0661_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0572_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0669_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0581_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0657_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0651_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0683_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0668_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0693_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0641_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0719_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) +=
        3.00 * tmps.at("0585_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0662_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0710_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0605_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0655_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0663_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0684_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -=
        0.50 * tmps.at("0576_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0603_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) +=
        3.00 * tmps.at("0583_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0599_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0664_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0711_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) +=
        3.00 * tmps.at("0615_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) +=
        3.00 * tmps.at("0582_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0633_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0666_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0646_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0574_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -=
        0.50 * tmps.at("0673_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0716_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0571_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0622_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0667_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -=
        0.50 * tmps.at("0653_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0607_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0695_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0618_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0598_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0566_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0638_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0570_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0660_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0639_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0617_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0626_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0604_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0590_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0614_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0676_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) +=
        3.00 * tmps.at("0567_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0717_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) +=
        3.00 * tmps.at("0591_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0575_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0733_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0592_baab_vvoo")(ab, ba, ia, jb))
      .deallocate(tmps.at("0721_abab_vvoo"))
      .deallocate(tmps.at("0720_baab_vvoo"))
      .deallocate(tmps.at("0719_baab_vvoo"))
      .deallocate(tmps.at("0718_abba_vvoo"))
      .deallocate(tmps.at("0717_baba_vvoo"))
      .deallocate(tmps.at("0716_abba_vvoo"))
      .deallocate(tmps.at("0715_baab_vvoo"))
      .deallocate(tmps.at("0711_baba_vvoo"))
      .deallocate(tmps.at("0710_abab_vvoo"))
      .deallocate(tmps.at("0709_abab_vvoo"))
      .deallocate(tmps.at("0695_abab_vvoo"))
      .deallocate(tmps.at("0694_abab_vvoo"))
      .deallocate(tmps.at("0693_baab_vvoo"))
      .deallocate(tmps.at("0684_abba_vvoo"))
      .deallocate(tmps.at("0683_baba_vvoo"))
      .deallocate(tmps.at("0681_baab_vvoo"))
      .deallocate(tmps.at("0680_abab_vvoo"))
      .deallocate(tmps.at("0679_abab_vvoo"))
      .deallocate(tmps.at("0678_abba_vvoo"))
      .deallocate(tmps.at("0676_abba_vvoo"))
      .deallocate(tmps.at("0675_abab_vvoo"))
      .deallocate(tmps.at("0674_abab_vvoo"))
      .deallocate(tmps.at("0673_abba_vvoo"))
      .deallocate(tmps.at("0672_abba_vvoo"))
      .deallocate(tmps.at("0670_abab_vvoo"))
      .deallocate(tmps.at("0669_abab_vvoo"))
      .deallocate(tmps.at("0668_baab_vvoo"))
      .deallocate(tmps.at("0667_abab_vvoo"))
      .deallocate(tmps.at("0666_baab_vvoo"))
      .deallocate(tmps.at("0665_abba_vvoo"))
      .deallocate(tmps.at("0664_baab_vvoo"))
      .deallocate(tmps.at("0663_abba_vvoo"))
      .deallocate(tmps.at("0662_baba_vvoo"))
      .deallocate(tmps.at("0661_baab_vvoo"))
      .deallocate(tmps.at("0660_baab_vvoo"))
      .deallocate(tmps.at("0659_baab_vvoo"))
      .deallocate(tmps.at("0657_abab_vvoo"))
      .deallocate(tmps.at("0656_abab_vvoo"))
      .deallocate(tmps.at("0655_abab_vvoo"))
      .deallocate(tmps.at("0654_abab_vvoo"))
      .deallocate(tmps.at("0653_baab_vvoo"))
      .deallocate(tmps.at("0651_abab_vvoo"))
      .deallocate(tmps.at("0647_baab_vvoo"))
      .deallocate(tmps.at("0646_abab_vvoo"))
      .deallocate(tmps.at("0641_abab_vvoo"))
      .deallocate(tmps.at("0639_baab_vvoo"))
      .deallocate(tmps.at("0638_abab_vvoo"))
      .deallocate(tmps.at("0634_abab_vvoo"))
      .deallocate(tmps.at("0633_baab_vvoo"))
      .deallocate(tmps.at("0632_baba_vvoo"))
      .deallocate(tmps.at("0626_baba_vvoo"))
      .deallocate(tmps.at("0622_abab_vvoo"))
      .deallocate(tmps.at("0621_abab_vvoo"))
      .deallocate(tmps.at("0619_abab_vvoo"))
      .deallocate(tmps.at("0618_abab_vvoo"))
      .deallocate(tmps.at("0617_baba_vvoo"))
      .deallocate(tmps.at("0616_abab_vvoo"))
      .deallocate(tmps.at("0615_abba_vvoo"))
      .deallocate(tmps.at("0614_abab_vvoo"))
      .deallocate(tmps.at("0608_baba_vvoo"))
      .deallocate(tmps.at("0607_baab_vvoo"))
      .deallocate(tmps.at("0606_baba_vvoo"))
      .deallocate(tmps.at("0605_baab_vvoo"))
      .deallocate(tmps.at("0604_abba_vvoo"))
      .deallocate(tmps.at("0603_baba_vvoo"))
      .deallocate(tmps.at("0601_baab_vvoo"))
      .deallocate(tmps.at("0599_baba_vvoo"))
      .deallocate(tmps.at("0598_abab_vvoo"))
      .deallocate(tmps.at("0594_baba_vvoo"))
      .deallocate(tmps.at("0593_abba_vvoo"))
      .deallocate(tmps.at("0592_baab_vvoo"))
      .deallocate(tmps.at("0591_abab_vvoo"))
      .deallocate(tmps.at("0590_abab_vvoo"))
      .deallocate(tmps.at("0589_baba_vvoo"))
      .deallocate(tmps.at("0588_abba_vvoo"))
      .deallocate(tmps.at("0586_abba_vvoo"))
      .deallocate(tmps.at("0585_abba_vvoo"))
      .deallocate(tmps.at("0584_baab_vvoo"))
      .deallocate(tmps.at("0583_abab_vvoo"))
      .deallocate(tmps.at("0582_baab_vvoo"))
      .deallocate(tmps.at("0581_baba_vvoo"))
      .deallocate(tmps.at("0580_abab_vvoo"))
      .deallocate(tmps.at("0576_baab_vvoo"))
      .deallocate(tmps.at("0575_baba_vvoo"))
      .deallocate(tmps.at("0574_baab_vvoo"))
      .deallocate(tmps.at("0572_baab_vvoo"))
      .deallocate(tmps.at("0571_abab_vvoo"))
      .deallocate(tmps.at("0570_abab_vvoo"))
      .deallocate(tmps.at("0568_baba_vvoo"))
      .deallocate(tmps.at("0567_abab_vvoo"))
      .deallocate(tmps.at("0566_abba_vvoo"))
      .deallocate(tmps.at("0565_abab_vvoo"))
      .deallocate(tmps.at("0427_baba_vvoo"))
      .allocate(tmps.at("0803_abab_vvoo"))(tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0746_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0758_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0771_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0769_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -=
        2.00 * tmps.at("0754_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0755_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0770_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0800_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0791_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0750_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0801_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0781_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0773_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0743_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0782_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0768_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0767_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -=
        2.00 * tmps.at("0763_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0772_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0757_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0740_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0733_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0774_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0779_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0790_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0756_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0742_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0744_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0780_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0759_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0741_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0798_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0789_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0797_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0803_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0799_abba_vvoo")(aa, bb, jb, ia))
      .deallocate(tmps.at("0801_abab_vvoo"))
      .deallocate(tmps.at("0800_baab_vvoo"))
      .deallocate(tmps.at("0799_abba_vvoo"))
      .deallocate(tmps.at("0798_abab_vvoo"))
      .deallocate(tmps.at("0797_abab_vvoo"))
      .deallocate(tmps.at("0791_abab_vvoo"))
      .deallocate(tmps.at("0790_abab_vvoo"))
      .deallocate(tmps.at("0789_baab_vvoo"))
      .deallocate(tmps.at("0782_abba_vvoo"))
      .deallocate(tmps.at("0781_baba_vvoo"))
      .deallocate(tmps.at("0780_baab_vvoo"))
      .deallocate(tmps.at("0779_abab_vvoo"))
      .deallocate(tmps.at("0774_baab_vvoo"))
      .deallocate(tmps.at("0773_abab_vvoo"))
      .deallocate(tmps.at("0772_abba_vvoo"))
      .deallocate(tmps.at("0771_abab_vvoo"))
      .deallocate(tmps.at("0770_abab_vvoo"))
      .deallocate(tmps.at("0769_abba_vvoo"))
      .deallocate(tmps.at("0768_abba_vvoo"))
      .deallocate(tmps.at("0767_abab_vvoo"))
      .deallocate(tmps.at("0763_baba_vvoo"))
      .deallocate(tmps.at("0759_abba_vvoo"))
      .deallocate(tmps.at("0758_abab_vvoo"))
      .deallocate(tmps.at("0757_abab_vvoo"))
      .deallocate(tmps.at("0756_baab_vvoo"))
      .deallocate(tmps.at("0755_abab_vvoo"))
      .deallocate(tmps.at("0754_abab_vvoo"))
      .deallocate(tmps.at("0750_baab_vvoo"))
      .deallocate(tmps.at("0746_abab_vvoo"))
      .deallocate(tmps.at("0744_baab_vvoo"))
      .deallocate(tmps.at("0743_abab_vvoo"))
      .deallocate(tmps.at("0742_abab_vvoo"))
      .deallocate(tmps.at("0741_abba_vvoo"))
      .deallocate(tmps.at("0740_abab_vvoo"))
      .deallocate(tmps.at("0733_baab_vvoo"))
      .allocate(tmps.at("0807_abba_vvoo"))(tmps.at("0807_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0803_abab_vvoo")(aa, bb, ja, ib))(
        tmps.at("0807_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0805_abba_vvoo")(aa, bb, ib, ja))(
        tmps.at("0807_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0806_abba_vvoo")(aa, bb, ib, ja))
      .deallocate(tmps.at("0806_abba_vvoo"))
      .deallocate(tmps.at("0805_abba_vvoo"))
      .deallocate(tmps.at("0803_abab_vvoo"))(r2_2p.at("abab")(aa, bb, ia, jb) +=
                                             2.00 * tmps.at("0807_abba_vvoo")(aa, bb, jb, ia))
      .deallocate(tmps.at("0807_abba_vvoo"))
      .allocate(tmps.at("0947_abba_vvoo"))(tmps.at("0947_abba_vvoo")(aa, bb, ib, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0804_baba_vooo")(bb, ka, ib, ja))
      .deallocate(tmps.at("0804_baba_vooo"))
      .allocate(tmps.at("0945_abab_vvoo"))(tmps.at("0945_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0222_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0222_abab_ovoo"))
      .allocate(tmps.at("0811_bb_ov"))(tmps.at("0811_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(ib, jb, ab, bb) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0842_abab_vooo"))(tmps.at("0842_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("0811_bb_ov")(ib, bb))
      .deallocate(tmps.at("0811_bb_ov"))
      .allocate(tmps.at("0944_baab_vvoo"))(tmps.at("0944_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0842_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .deallocate(tmps.at("0842_abab_vooo"))
      .allocate(tmps.at("0943_abab_vvoo"))(tmps.at("0943_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0778_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0778_abab_ovoo"))
      .allocate(tmps.at("0942_baab_vvoo"))(tmps.at("0942_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0788_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .deallocate(tmps.at("0788_abab_vooo"))
      .allocate(tmps.at("0941_abab_vvoo"))(tmps.at("0941_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0796_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0796_baab_vooo"))
      .allocate(tmps.at("0940_baab_vvoo"))(tmps.at("0940_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0207_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .deallocate(tmps.at("0207_abab_vooo"))
      .allocate(tmps.at("0939_abba_vvoo"))(tmps.at("0939_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0217_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .deallocate(tmps.at("0217_aa_oo"))
      .allocate(tmps.at("0938_baba_vvoo"))(tmps.at("0938_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0762_aa_vo")(ba, ja) * t1_1p.at("bb")(ab, ib))
      .deallocate(tmps.at("0762_aa_vo"))
      .allocate(tmps.at("0937_abab_vvoo"))(tmps.at("0937_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0201_bb_oo")(kb, jb))
      .deallocate(tmps.at("0201_bb_oo"))
      .allocate(tmps.at("0936_abab_vvoo"))(tmps.at("0936_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0196_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0196_abab_oooo"))
      .allocate(tmps.at("0935_baab_vvoo"))(tmps.at("0935_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0749_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .deallocate(tmps.at("0749_abab_vooo"))
      .allocate(tmps.at("0934_abba_vvoo"))(tmps.at("0934_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0313_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .deallocate(tmps.at("0313_aa_oo"))
      .allocate(tmps.at("0933_abba_vvoo"))(tmps.at("0933_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0736_abba_oooo")(ka, lb, ib, ja))
      .deallocate(tmps.at("0736_abba_oooo"))
      .allocate(tmps.at("0932_baba_vvoo"))(tmps.at("0932_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0224_aa_vo")(ba, ja) * t1_2p.at("bb")(ab, ib))
      .deallocate(tmps.at("0224_aa_vo"))
      .allocate(tmps.at("0931_abab_vvoo"))(tmps.at("0931_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0211_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0211_baab_vooo"))
      .allocate(tmps.at("0810_aa_ov"))(tmps.at("0810_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0844_baab_vooo"))(tmps.at("0844_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0810_aa_ov")(ia, ba))
      .deallocate(tmps.at("0810_aa_ov"))
      .allocate(tmps.at("0843_baba_vooo"))(tmps.at("0843_baba_vooo")(ab, ia, jb, ka) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0310_aaaa_oovo")(ia, la, ba, ka))
      .deallocate(tmps.at("0310_aaaa_oovo"))
      .allocate(tmps.at("0919_baba_vooo"))(tmps.at("0919_baba_vooo")(ab, ia, jb, ka) =
                                             -1.00 * tmps.at("0844_baab_vooo")(ab, ia, ka, jb))(
        tmps.at("0919_baba_vooo")(ab, ia, jb, ka) += tmps.at("0843_baba_vooo")(ab, ia, jb, ka))
      .deallocate(tmps.at("0844_baab_vooo"))
      .deallocate(tmps.at("0843_baba_vooo"))
      .allocate(tmps.at("0930_abba_vvoo"))(tmps.at("0930_abba_vvoo")(aa, bb, ib, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0919_baba_vooo")(bb, ka, ib, ja))
      .deallocate(tmps.at("0919_baba_vooo"))
      .allocate(tmps.at("0929_abba_vvoo"))(tmps.at("0929_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0301_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .deallocate(tmps.at("0301_aa_oo"))
      .allocate(tmps.at("0928_abab_vvoo"))(tmps.at("0928_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0734_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0734_abab_oooo"))
      .allocate(tmps.at("0927_abab_vvoo"))(tmps.at("0927_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0192_bb_vo")(bb, jb))
      .deallocate(tmps.at("0192_bb_vo"))
      .allocate(tmps.at("0926_abab_vvoo"))(tmps.at("0926_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0753_bb_vo")(bb, jb))
      .deallocate(tmps.at("0753_bb_vo"))
      .allocate(tmps.at("0925_abba_vvoo"))(tmps.at("0925_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0366_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .deallocate(tmps.at("0366_aa_oo"))
      .allocate(tmps.at("0841_bb_oo"))(tmps.at("0841_bb_oo")(ib, jb) =
                                         t1_1p.at("bb")(ab, kb) *
                                         tmps.at("0079_bbbb_oovo")(ib, kb, ab, jb))
      .deallocate(tmps.at("0079_bbbb_oovo"))
      .allocate(tmps.at("0924_abab_vvoo"))(tmps.at("0924_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0841_bb_oo")(kb, jb))
      .deallocate(tmps.at("0841_bb_oo"))
      .allocate(tmps.at("0923_abab_vvoo"))(tmps.at("0923_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0766_bb_oo")(kb, jb))
      .deallocate(tmps.at("0766_bb_oo"))
      .allocate(tmps.at("0922_abab_vvoo"))(tmps.at("0922_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0745_bb_oo")(kb, jb))
      .deallocate(tmps.at("0745_bb_oo"))
      .allocate(tmps.at("0921_baab_vvoo"))(tmps.at("0921_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0209_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .deallocate(tmps.at("0209_abab_vooo"))
      .allocate(tmps.at("0918_abab_vvoo"))(tmps.at("0918_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0708_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0708_baab_vooo"))
      .allocate(tmps.at("0917_baba_vvoo"))(tmps.at("0917_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0625_baba_ovoo")(kb, ba, ib, ja))
      .deallocate(tmps.at("0625_baba_ovoo"))
      .allocate(tmps.at("0916_baab_vvoo"))(tmps.at("0916_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0692_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0692_abab_vooo"))
      .allocate(tmps.at("0915_abab_vvoo"))(tmps.at("0915_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("0613_bb_vo")(bb, jb))
      .deallocate(tmps.at("0613_bb_vo"))
      .allocate(tmps.at("0914_baab_vvoo"))(tmps.at("0914_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0714_baab_ovoo")(kb, ba, ia, jb))
      .deallocate(tmps.at("0714_baab_ovoo"))
      .allocate(tmps.at("0913_abab_vvoo"))(tmps.at("0913_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0645_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0645_abab_ovoo"))
      .allocate(tmps.at("0912_baba_vvoo"))(tmps.at("0912_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_2p.at("bb")(ab, ib) * tmps.at("0176_aa_vo")(ba, ja))
      .deallocate(tmps.at("0176_aa_vo"))
      .allocate(tmps.at("0911_abab_vvoo"))(tmps.at("0911_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0182_bb_oo")(kb, jb))
      .deallocate(tmps.at("0182_bb_oo"))
      .allocate(tmps.at("0910_abab_vvoo"))(tmps.at("0910_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0637_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0637_abab_ovoo"))
      .allocate(tmps.at("0909_abab_vvoo"))(tmps.at("0909_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0155_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0155_baab_vooo"))
      .allocate(tmps.at("0908_baba_vvoo"))(tmps.at("0908_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0273_aaaa_voov")(ba, ka, ja, ca))
      .deallocate(tmps.at("0273_aaaa_voov"))
      .allocate(tmps.at("0907_abab_vvoo"))(tmps.at("0907_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0169_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0169_baab_vooo"))
      .allocate(tmps.at("0906_abab_vvoo"))(tmps.at("0906_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0136_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0136_abab_ovoo"))
      .allocate(tmps.at("0905_baab_vvoo"))(tmps.at("0905_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0600_baab_ovoo")(kb, ba, ia, jb))
      .deallocate(tmps.at("0600_baab_ovoo"))
      .allocate(tmps.at("0904_baab_vvoo"))(tmps.at("0904_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0148_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0148_abab_vooo"))
      .allocate(tmps.at("0903_baba_vvoo"))(tmps.at("0903_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0280_abab_voov")(ba, kb, ja, cb))
      .deallocate(tmps.at("0280_abab_voov"))
      .allocate(tmps.at("0902_abab_vvoo"))(tmps.at("0902_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0650_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0650_abab_ovoo"))
      .allocate(tmps.at("0901_baab_vvoo"))(tmps.at("0901_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0251_aa_vv")(ba, ca))
      .deallocate(tmps.at("0251_aa_vv"))
      .allocate(tmps.at("0900_baab_vvoo"))(tmps.at("0900_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0159_baab_ovoo")(kb, ba, ia, jb))
      .deallocate(tmps.at("0159_baab_ovoo"))
      .allocate(tmps.at("0899_abab_vvoo"))(tmps.at("0899_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0142_bb_vo")(bb, jb))
      .deallocate(tmps.at("0142_bb_vo"))
      .allocate(tmps.at("0898_abab_vvoo"))(tmps.at("0898_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0089_abab_ovvo")(ka, bb, ca, jb))
      .deallocate(tmps.at("0089_abab_ovvo"))
      .allocate(tmps.at("0814_baba_ovoo"))(tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) =
                                             eri.at("bbbb_oovv")(ib, lb, bb, cb) *
                                             t1_1p.at("bb")(cb, jb))(
        tmps.at("0814_baba_ovoo")(ib, aa, jb, ka) =
          tmps.at("bin_bbbb_vooo")(bb, ib, jb, lb) * t2.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0897_baba_vvoo"))(tmps.at("0897_baba_vvoo")(ab, ba, ib, ja) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0814_baba_ovoo")(kb, ba, ib, ja))
      .deallocate(tmps.at("0814_baba_ovoo"))
      .allocate(tmps.at("0896_abba_vvoo"))(tmps.at("0896_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0277_aa_oo")(ka, ja))
      .deallocate(tmps.at("0277_aa_oo"))
      .allocate(tmps.at("0895_abab_vvoo"))(tmps.at("0895_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0085_baab_vovo")(bb, ka, ca, jb))
      .deallocate(tmps.at("0085_baab_vovo"))
      .allocate(tmps.at("0894_baab_vvoo"))(tmps.at("0894_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0096_abab_vovo")(ba, kb, ca, jb))
      .deallocate(tmps.at("0096_abab_vovo"))
      .allocate(tmps.at("0893_abab_vvoo"))(tmps.at("0893_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0597_bb_vv")(bb, cb))
      .deallocate(tmps.at("0597_bb_vv"))
      .allocate(tmps.at("0892_baab_vvoo"))(tmps.at("0892_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0110_aa_vv")(ba, ca))
      .deallocate(tmps.at("0110_aa_vv"))
      .allocate(tmps.at("0891_baab_vvoo"))(tmps.at("0891_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0100_aa_vv")(ba, ca))
      .deallocate(tmps.at("0100_aa_vv"))
      .allocate(tmps.at("0890_baab_vvoo"))(tmps.at("0890_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0188_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0188_abab_vooo"))
      .allocate(tmps.at("0889_baab_vvoo"))(tmps.at("0889_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0519_abab_vovo")(ba, kb, ca, jb))
      .deallocate(tmps.at("0519_abab_vovo"))
      .allocate(tmps.at("0888_baab_vvoo"))(tmps.at("0888_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0271_aa_vv")(ba, ca))
      .deallocate(tmps.at("0271_aa_vv"))
      .allocate(tmps.at("0887_baba_vvoo"))(tmps.at("0887_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0264_aaaa_voov")(ba, ka, ja, ca))
      .deallocate(tmps.at("0264_aaaa_voov"))
      .allocate(tmps.at("0886_baba_vvoo"))(tmps.at("0886_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0132_aaaa_voov")(ba, ka, ja, ca))
      .deallocate(tmps.at("0132_aaaa_voov"))
      .allocate(tmps.at("0885_abba_vvoo"))(tmps.at("0885_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, cb, ka, ib) *
                                             tmps.at("0126_baba_vovo")(bb, ka, cb, ja))
      .deallocate(tmps.at("0126_baba_vovo"))
      .allocate(tmps.at("0884_abba_vvoo"))(tmps.at("0884_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0116_aa_oo")(ka, ja))
      .deallocate(tmps.at("0116_aa_oo"))
      .allocate(tmps.at("0883_abba_vvoo"))(tmps.at("0883_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0249_aa_oo")(ka, ja))
      .deallocate(tmps.at("0249_aa_oo"))
      .allocate(tmps.at("0882_baab_vvoo"))(tmps.at("0882_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0104_aa_vv")(ba, ca))
      .deallocate(tmps.at("0104_aa_vv"))
      .allocate(tmps.at("0881_baba_vvoo"))(tmps.at("0881_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0098_aaaa_vovo")(ba, ka, ca, ja))
      .deallocate(tmps.at("0098_aaaa_vovo"))
      .allocate(tmps.at("0813_bb_vv"))(tmps.at("0813_bb_vv")(ab, bb) =
                                         eri.at("bbbb_vovv")(ab, ib, bb, cb) *
                                         t1_1p.at("bb")(cb, ib))
      .allocate(tmps.at("0880_abab_vvoo"))(tmps.at("0880_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0813_bb_vv")(bb, cb))
      .deallocate(tmps.at("0813_bb_vv"))
      .allocate(tmps.at("0879_baba_vvoo"))(tmps.at("0879_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0163_aaaa_voov")(ba, ka, ja, ca))
      .deallocate(tmps.at("0163_aaaa_voov"))
      .allocate(tmps.at("0878_baba_vvoo"))(tmps.at("0878_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0262_abba_vovo")(ba, kb, cb, ja))
      .deallocate(tmps.at("0262_abba_vovo"))
      .allocate(tmps.at("0877_abab_vvoo"))(tmps.at("0877_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0549_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0549_abab_oooo"))
      .allocate(tmps.at("0876_abab_vvoo"))(tmps.at("0876_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0579_bb_oo")(kb, jb))
      .deallocate(tmps.at("0579_bb_oo"))
      .allocate(tmps.at("0875_abba_vvoo"))(tmps.at("0875_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0269_aa_oo")(ka, ja))
      .deallocate(tmps.at("0269_aa_oo"))
      .allocate(tmps.at("0874_baba_vvoo"))(tmps.at("0874_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0106_abba_vovo")(ba, kb, cb, ja))
      .deallocate(tmps.at("0106_abba_vovo"))
      .allocate(tmps.at("0873_baba_vvoo"))(tmps.at("0873_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0204_abab_voov")(ba, kb, ja, cb))
      .deallocate(tmps.at("0204_abab_voov"))
      .allocate(tmps.at("0872_abab_vvoo"))(tmps.at("0872_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0130_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0130_abab_oooo"))
      .allocate(tmps.at("0809_bb_oo"))(tmps.at("0809_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovo")(ib, kb, ab, jb) *
                                         t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0808_bb_oo"))(tmps.at("0808_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovv")(ib, kb, ab, bb) *
                                         t2_1p.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0839_bb_oo"))(tmps.at("0839_bb_oo")(ib, jb) =
                                         -2.00 * tmps.at("0809_bb_oo")(ib, jb))(
        tmps.at("0839_bb_oo")(ib, jb) += tmps.at("0808_bb_oo")(ib, jb))
      .deallocate(tmps.at("0809_bb_oo"))
      .deallocate(tmps.at("0808_bb_oo"))
      .allocate(tmps.at("0871_abab_vvoo"))(tmps.at("0871_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0839_bb_oo")(kb, jb))
      .deallocate(tmps.at("0839_bb_oo"))
      .allocate(tmps.at("0870_abba_vvoo"))(tmps.at("0870_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, cb, ka, ib) *
                                             tmps.at("0671_baba_vovo")(bb, ka, cb, ja))
      .deallocate(tmps.at("0671_baba_vovo"))
      .allocate(tmps.at("0869_abab_vvoo"))(tmps.at("0869_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0640_bb_oo")(kb, jb))
      .deallocate(tmps.at("0640_bb_oo"))
      .allocate(tmps.at("0868_abba_vvoo"))(tmps.at("0868_abba_vvoo")(aa, bb, ib, ja) =
                                             t1.at("bb")(cb, ib) *
                                             tmps.at("0587_abba_vvvo")(aa, bb, cb, ja))
      .deallocate(tmps.at("0587_abba_vvvo"))
      .allocate(tmps.at("0867_abab_vvoo"))(tmps.at("0867_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0569_bb_vv")(bb, cb))
      .deallocate(tmps.at("0569_bb_vv"))
      .allocate(tmps.at("0866_abab_vvoo"))(tmps.at("0866_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0541_baab_vovo")(bb, ka, ca, jb))
      .deallocate(tmps.at("0541_baab_vovo"))
      .allocate(tmps.at("0865_baab_vvoo"))(tmps.at("0865_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0161_aa_vv")(ba, ca))
      .deallocate(tmps.at("0161_aa_vv"))
      .allocate(tmps.at("0864_abba_vvoo"))(tmps.at("0864_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0124_aa_oo")(ka, ja))
      .deallocate(tmps.at("0124_aa_oo"))
      .allocate(tmps.at("0863_baab_vvoo"))(tmps.at("0863_baab_vvoo")(ab, ba, ia, jb) =
                                             t2_1p.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0091_abba_voov")(ba, kb, jb, ca))
      .deallocate(tmps.at("0091_abba_voov"))
      .allocate(tmps.at("0862_baba_vvoo"))(tmps.at("0862_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("0631_aa_vo")(ba, ja))
      .deallocate(tmps.at("0631_aa_vo"))
      .allocate(tmps.at("0861_abba_vvoo"))(tmps.at("0861_abba_vvoo")(aa, bb, ib, ja) =
                                             t1_1p.at("bb")(cb, ib) *
                                             tmps.at("0184_abba_vvvo")(aa, bb, cb, ja))
      .deallocate(tmps.at("0184_abba_vvvo"))
      .allocate(tmps.at("0860_baba_vvoo"))(tmps.at("0860_baba_vvoo")(ab, ba, ib, ja) =
                                             t2_1p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0112_abab_voov")(ba, kb, ja, cb))
      .deallocate(tmps.at("0112_abab_voov"))
      .allocate(tmps.at("0859_abab_vvoo"))(tmps.at("0859_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0120_bb_vv")(bb, cb))
      .deallocate(tmps.at("0120_bb_vv"))
      .allocate(tmps.at("0815_aaaa_vovo"))(tmps.at("0815_aaaa_vovo")(aa, ia, ba, ja) =
                                             eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                             t1_1p.at("aa")(ca, ja))
      .allocate(tmps.at("0858_baba_vvoo"))(tmps.at("0858_baba_vvoo")(ab, ba, ib, ja) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0815_aaaa_vovo")(ba, ka, ca, ja))
      .deallocate(tmps.at("0815_aaaa_vovo"))
      .allocate(tmps.at("0857_abab_vvoo"))(tmps.at("0857_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0114_bbbb_vovo")(bb, kb, cb, jb))
      .deallocate(tmps.at("0114_bbbb_vovo"))
      .allocate(tmps.at("0856_baab_vvoo"))(tmps.at("0856_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, kb) *
                                             tmps.at("0516_abba_voov")(ba, kb, jb, ca))
      .deallocate(tmps.at("0516_abba_voov"))
      .allocate(tmps.at("0855_abab_vvoo"))(tmps.at("0855_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0620_bb_vv")(bb, cb))
      .deallocate(tmps.at("0620_bb_vv"))
      .allocate(tmps.at("0812_aa_vv"))(tmps.at("0812_aa_vv")(aa, ba) =
                                         eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                         t1_1p.at("aa")(ca, ia))
      .allocate(tmps.at("0854_baab_vvoo"))(tmps.at("0854_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0812_aa_vv")(ba, ca))
      .deallocate(tmps.at("0812_aa_vv"))
      .allocate(tmps.at("0853_abba_vvoo"))(tmps.at("0853_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0087_aa_oo")(ka, ja))
      .deallocate(tmps.at("0087_aa_oo"))
      .allocate(tmps.at("0852_abab_vvoo"))(tmps.at("0852_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0094_bb_vv")(bb, cb))
      .deallocate(tmps.at("0094_bb_vv"))
      .allocate(tmps.at("0851_abab_vvoo"))(tmps.at("0851_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0108_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0108_abab_oooo"))
      .allocate(tmps.at("0850_abab_vvoo"))(tmps.at("0850_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0531_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0531_abab_oooo"))
      .allocate(tmps.at("0849_abba_vvoo"))(tmps.at("0849_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0102_aa_oo")(ka, ja))
      .deallocate(tmps.at("0102_aa_oo"))
      .allocate(tmps.at("0848_abba_vvoo"))(tmps.at("0848_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0244_aa_oo")(ka, ja))
      .deallocate(tmps.at("0244_aa_oo"))
      .allocate(tmps.at("0847_abba_vvoo"))(tmps.at("0847_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0345_aa_oo")(ka, ja))
      .deallocate(tmps.at("0345_aa_oo"))
      .allocate(tmps.at("0846_abab_vvoo"))(tmps.at("0846_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0083_bb_oo")(kb, jb))
      .deallocate(tmps.at("0083_bb_oo"))
      .allocate(tmps.at("0845_baab_vvoo"))(tmps.at("0845_baab_vvoo")(ab, ba, ia, jb) =
                                             t2.at("abab")(ca, ab, ia, jb) *
                                             tmps.at("0658_aa_vv")(ba, ca))
      .deallocate(tmps.at("0658_aa_vv"))
      .allocate(tmps.at("0838_abab_vvoo"))(tmps.at("0838_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0015")() *
                                             t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0837_abba_vvoo"))(tmps.at("0837_abba_vvoo")(aa, bb, ib, ja) =
                                             eri.at("abab_vovo")(aa, kb, ca, ib) *
                                             t2_1p.at("abab")(ca, bb, ja, kb))
      .allocate(tmps.at("0836_abab_vvoo"))(tmps.at("0836_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0013")() *
                                             t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0835_abab_vvoo"))(tmps.at("0835_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("bb_vo")(bb, jb) * t1_2p.at("aa")(aa, ia))
      .allocate(tmps.at("0834_abab_vvoo"))(tmps.at("0834_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0001")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0833_abab_vvoo"))(tmps.at("0833_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0002")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0832_baba_vvoo"))(tmps.at("bin_bbbb_vvoo")(ab, cb, ib, kb) =
                                             eri.at("bbbb_vovv")(ab, kb, cb, db) *
                                             t1_1p.at("bb")(db, ib))(
        tmps.at("0832_baba_vvoo")(ab, ba, ib, ja) =
          tmps.at("bin_bbbb_vvoo")(ab, cb, ib, kb) * t2.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("0831_abab_vvoo"))(tmps.at("0831_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("baab_vooo")(bb, ka, ia, jb) *
                                             t1_1p.at("aa")(aa, ka))
      .allocate(tmps.at("0830_abba_vvoo"))(tmps.at("0830_abba_vvoo")(aa, bb, ib, ja) =
                                             eri.at("baba_vovo")(bb, ka, cb, ja) *
                                             t2_1p.at("abab")(aa, cb, ka, ib))
      .allocate(tmps.at("0829_abab_vvoo"))(tmps.at("0829_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_oooo")(ka, lb, ia, jb) *
                                             t2_1p.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0828_abab_vvoo"))(tmps.at("0828_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vovo")(aa, kb, cb, ia) *
                                             t2_1p.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("0827_abab_vvoo"))(tmps.at("0827_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_oo")(ka, ia) *
                                             t2_1p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0826_abab_vvoo"))(tmps.at("0826_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vvvv")(aa, bb, ca, db) *
                                             t2_1p.at("abab")(ca, db, ia, jb))
      .allocate(tmps.at("0825_abab_vvoo"))(tmps.at("0825_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("aaaa_vovo")(aa, ka, ca, ia) *
                                             t2_1p.at("abab")(ca, bb, ka, jb))
      .allocate(tmps.at("0824_abab_vvoo"))(tmps.at("0824_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vvvo")(aa, bb, cb, ia) *
                                             t1_1p.at("bb")(cb, jb))
      .allocate(tmps.at("0823_abab_vvoo"))(tmps.at("0823_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("bb_oo")(kb, jb) *
                                             t2_1p.at("abab")(aa, bb, ia, kb))
      .allocate(tmps.at("0822_abab_vvoo"))(tmps.at("0822_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("bb_vv")(bb, cb) *
                                             t2_1p.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0821_abab_vvoo"))(tmps.at("0821_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("bbbb_vovo")(bb, kb, cb, jb) *
                                             t2_1p.at("abab")(aa, cb, ia, kb))
      .allocate(tmps.at("0820_abab_vvoo"))(tmps.at("0820_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_vv")(aa, ca) *
                                             t2_1p.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("0819_abab_vvoo"))(tmps.at("0819_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("baab_vovo")(bb, ka, ca, jb) *
                                             t2_1p.at("aaaa")(ca, aa, ia, ka))
      .allocate(tmps.at("0818_abab_vvoo"))(tmps.at("0818_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vvvo")(aa, bb, ca, jb) *
                                             t1_1p.at("aa")(ca, ia))
      .allocate(tmps.at("0817_abab_vvoo"))(tmps.at("0817_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vooo")(aa, kb, ia, jb) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("0816_abab_vvoo"))(tmps.at("0816_abab_vvoo")(aa, bb, ia, jb) =
                                             dp.at("aa_vo")(aa, ia) * t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("0840_abab_vvoo"))(tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("0818_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0819_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0820_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0826_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0833_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0821_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0830_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0832_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -=
        2.00 * tmps.at("0816_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0823_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0834_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0817_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -=
        2.00 * tmps.at("0836_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0837_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -=
        2.00 * tmps.at("0835_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -=
        2.00 * tmps.at("0838_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0825_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0822_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0827_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0828_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0824_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0829_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0840_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0831_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0838_abab_vvoo"))
      .deallocate(tmps.at("0837_abba_vvoo"))
      .deallocate(tmps.at("0836_abab_vvoo"))
      .deallocate(tmps.at("0835_abab_vvoo"))
      .deallocate(tmps.at("0834_abab_vvoo"))
      .deallocate(tmps.at("0833_abab_vvoo"))
      .deallocate(tmps.at("0832_baba_vvoo"))
      .deallocate(tmps.at("0831_abab_vvoo"))
      .deallocate(tmps.at("0830_abba_vvoo"))
      .deallocate(tmps.at("0829_abab_vvoo"))
      .deallocate(tmps.at("0828_abab_vvoo"))
      .deallocate(tmps.at("0827_abab_vvoo"))
      .deallocate(tmps.at("0826_abab_vvoo"))
      .deallocate(tmps.at("0825_abab_vvoo"))
      .deallocate(tmps.at("0824_abab_vvoo"))
      .deallocate(tmps.at("0823_abab_vvoo"))
      .deallocate(tmps.at("0822_abab_vvoo"))
      .deallocate(tmps.at("0821_abab_vvoo"))
      .deallocate(tmps.at("0820_abab_vvoo"))
      .deallocate(tmps.at("0819_abab_vvoo"))
      .deallocate(tmps.at("0818_abab_vvoo"))
      .deallocate(tmps.at("0817_abab_vvoo"))
      .deallocate(tmps.at("0816_abab_vvoo"))
      .allocate(tmps.at("0920_abab_vvoo"))(tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) =
                                             -0.50 * tmps.at("0855_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0867_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -=
        0.50 * tmps.at("0853_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0869_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0890_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0902_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0859_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0850_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0863_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0846_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0895_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -=
        2.00 * tmps.at("0912_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0856_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0911_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0857_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0880_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0896_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0915_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) +=
        0.50 * tmps.at("0883_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0881_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0893_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0894_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0898_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0884_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -=
        2.00 * tmps.at("0899_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) +=
        0.50 * tmps.at("0901_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0887_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0864_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0906_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0858_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0907_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0861_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0886_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0905_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0903_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0848_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0874_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) +=
        0.50 * tmps.at("0871_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0870_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0910_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0909_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0888_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0862_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0876_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0851_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0904_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0847_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0873_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0918_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0889_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0885_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0872_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0868_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0917_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0865_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -=
        0.50 * tmps.at("0891_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0882_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0900_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0840_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0849_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0878_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0866_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0908_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0897_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0860_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0892_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0916_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0852_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0913_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0875_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0914_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0879_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0845_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("0854_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("0920_abab_vvoo")(aa, bb, ia, jb) += tmps.at("0877_abab_vvoo")(aa, bb, ia, jb))
      .deallocate(tmps.at("0918_abab_vvoo"))
      .deallocate(tmps.at("0917_baba_vvoo"))
      .deallocate(tmps.at("0916_baab_vvoo"))
      .deallocate(tmps.at("0915_abab_vvoo"))
      .deallocate(tmps.at("0914_baab_vvoo"))
      .deallocate(tmps.at("0913_abab_vvoo"))
      .deallocate(tmps.at("0912_baba_vvoo"))
      .deallocate(tmps.at("0911_abab_vvoo"))
      .deallocate(tmps.at("0910_abab_vvoo"))
      .deallocate(tmps.at("0909_abab_vvoo"))
      .deallocate(tmps.at("0908_baba_vvoo"))
      .deallocate(tmps.at("0907_abab_vvoo"))
      .deallocate(tmps.at("0906_abab_vvoo"))
      .deallocate(tmps.at("0905_baab_vvoo"))
      .deallocate(tmps.at("0904_baab_vvoo"))
      .deallocate(tmps.at("0903_baba_vvoo"))
      .deallocate(tmps.at("0902_abab_vvoo"))
      .deallocate(tmps.at("0901_baab_vvoo"))
      .deallocate(tmps.at("0900_baab_vvoo"))
      .deallocate(tmps.at("0899_abab_vvoo"))
      .deallocate(tmps.at("0898_abab_vvoo"))
      .deallocate(tmps.at("0897_baba_vvoo"))
      .deallocate(tmps.at("0896_abba_vvoo"))
      .deallocate(tmps.at("0895_abab_vvoo"))
      .deallocate(tmps.at("0894_baab_vvoo"))
      .deallocate(tmps.at("0893_abab_vvoo"))
      .deallocate(tmps.at("0892_baab_vvoo"))
      .deallocate(tmps.at("0891_baab_vvoo"))
      .deallocate(tmps.at("0890_baab_vvoo"))
      .deallocate(tmps.at("0889_baab_vvoo"))
      .deallocate(tmps.at("0888_baab_vvoo"))
      .deallocate(tmps.at("0887_baba_vvoo"))
      .deallocate(tmps.at("0886_baba_vvoo"))
      .deallocate(tmps.at("0885_abba_vvoo"))
      .deallocate(tmps.at("0884_abba_vvoo"))
      .deallocate(tmps.at("0883_abba_vvoo"))
      .deallocate(tmps.at("0882_baab_vvoo"))
      .deallocate(tmps.at("0881_baba_vvoo"))
      .deallocate(tmps.at("0880_abab_vvoo"))
      .deallocate(tmps.at("0879_baba_vvoo"))
      .deallocate(tmps.at("0878_baba_vvoo"))
      .deallocate(tmps.at("0877_abab_vvoo"))
      .deallocate(tmps.at("0876_abab_vvoo"))
      .deallocate(tmps.at("0875_abba_vvoo"))
      .deallocate(tmps.at("0874_baba_vvoo"))
      .deallocate(tmps.at("0873_baba_vvoo"))
      .deallocate(tmps.at("0872_abab_vvoo"))
      .deallocate(tmps.at("0871_abab_vvoo"))
      .deallocate(tmps.at("0870_abba_vvoo"))
      .deallocate(tmps.at("0869_abab_vvoo"))
      .deallocate(tmps.at("0868_abba_vvoo"))
      .deallocate(tmps.at("0867_abab_vvoo"))
      .deallocate(tmps.at("0866_abab_vvoo"))
      .deallocate(tmps.at("0865_baab_vvoo"))
      .deallocate(tmps.at("0864_abba_vvoo"))
      .deallocate(tmps.at("0863_baab_vvoo"))
      .deallocate(tmps.at("0862_baba_vvoo"))
      .deallocate(tmps.at("0861_abba_vvoo"))
      .deallocate(tmps.at("0860_baba_vvoo"))
      .deallocate(tmps.at("0859_abab_vvoo"))
      .deallocate(tmps.at("0858_baba_vvoo"))
      .deallocate(tmps.at("0857_abab_vvoo"))
      .deallocate(tmps.at("0856_baab_vvoo"))
      .deallocate(tmps.at("0855_abab_vvoo"))
      .deallocate(tmps.at("0854_baab_vvoo"))
      .deallocate(tmps.at("0853_abba_vvoo"))
      .deallocate(tmps.at("0852_abab_vvoo"))
      .deallocate(tmps.at("0851_abab_vvoo"))
      .deallocate(tmps.at("0850_abab_vvoo"))
      .deallocate(tmps.at("0849_abba_vvoo"))
      .deallocate(tmps.at("0848_abba_vvoo"))
      .deallocate(tmps.at("0847_abba_vvoo"))
      .deallocate(tmps.at("0846_abab_vvoo"))
      .deallocate(tmps.at("0845_baab_vvoo"))
      .deallocate(tmps.at("0840_abab_vvoo"))
      .allocate(tmps.at("0946_baab_vvoo"))(tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) =
                                             -1.00 * tmps.at("0928_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0931_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0941_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0920_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0940_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0942_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0923_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0939_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0938_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0930_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0927_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0924_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0935_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0922_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) +=
        2.00 * tmps.at("0932_baba_vvoo")(ab, ba, jb, ia))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0933_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0929_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0936_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0934_abba_vvoo")(ba, ab, jb, ia))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0926_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0921_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0945_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) -= tmps.at("0943_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0944_baab_vvoo")(ab, ba, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0937_abab_vvoo")(ba, ab, ia, jb))(
        tmps.at("0946_baab_vvoo")(ab, ba, ia, jb) += tmps.at("0925_abba_vvoo")(ba, ab, jb, ia))
      .deallocate(tmps.at("0945_abab_vvoo"))
      .deallocate(tmps.at("0944_baab_vvoo"))
      .deallocate(tmps.at("0943_abab_vvoo"))
      .deallocate(tmps.at("0942_baab_vvoo"))
      .deallocate(tmps.at("0941_abab_vvoo"))
      .deallocate(tmps.at("0940_baab_vvoo"))
      .deallocate(tmps.at("0939_abba_vvoo"))
      .deallocate(tmps.at("0938_baba_vvoo"))
      .deallocate(tmps.at("0937_abab_vvoo"))
      .deallocate(tmps.at("0936_abab_vvoo"))
      .deallocate(tmps.at("0935_baab_vvoo"))
      .deallocate(tmps.at("0934_abba_vvoo"))
      .deallocate(tmps.at("0933_abba_vvoo"))
      .deallocate(tmps.at("0932_baba_vvoo"))
      .deallocate(tmps.at("0931_abab_vvoo"))
      .deallocate(tmps.at("0930_abba_vvoo"))
      .deallocate(tmps.at("0929_abba_vvoo"))
      .deallocate(tmps.at("0928_abab_vvoo"))
      .deallocate(tmps.at("0927_abab_vvoo"))
      .deallocate(tmps.at("0926_abab_vvoo"))
      .deallocate(tmps.at("0925_abba_vvoo"))
      .deallocate(tmps.at("0924_abab_vvoo"))
      .deallocate(tmps.at("0923_abab_vvoo"))
      .deallocate(tmps.at("0922_abab_vvoo"))
      .deallocate(tmps.at("0921_baab_vvoo"))
      .deallocate(tmps.at("0920_abab_vvoo"))
      .allocate(tmps.at("0948_abba_vvoo"))(tmps.at("0948_abba_vvoo")(aa, bb, ib, ja) =
                                             -1.00 * tmps.at("0946_baab_vvoo")(bb, aa, ja, ib))(
        tmps.at("0948_abba_vvoo")(aa, bb, ib, ja) += tmps.at("0947_abba_vvoo")(aa, bb, ib, ja))
      .deallocate(tmps.at("0947_abba_vvoo"))
      .deallocate(tmps.at("0946_baab_vvoo"))(r2_1p.at("abab")(aa, bb, ia, jb) +=
                                             tmps.at("0948_abba_vvoo")(aa, bb, jb, ia))
      .deallocate(tmps.at("0948_abba_vvoo"));
  }
}

template void exachem::cc::qed_ccsd_cs::resid_part4<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);