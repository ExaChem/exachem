/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_os_resid_2.hpp"

template<typename T>
void exachem::cc::qed_ccsd_os::resid_2(
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

      .allocate(tmps.at("0245_aa_vo"))(tmps.at("0245_aa_vo")(aa, ia) =
                                         -0.50 * tmps.at("0228_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) -= 0.50 * tmps.at("0229_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) += tmps.at("0231_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) -= tmps.at("0223_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) -= tmps.at("0233_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) -= tmps.at("0225_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) -= tmps.at("0244_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) += tmps.at("0238_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) -= tmps.at("0222_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) -= tmps.at("0227_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) += tmps.at("0236_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) -= tmps.at("0235_aa_vo")(aa, ia))(
        tmps.at("0245_aa_vo")(aa, ia) -= tmps.at("0242_aa_vo")(aa, ia))
      .deallocate(tmps.at("0244_aa_vo"))
      .deallocate(tmps.at("0242_aa_vo"))
      .deallocate(tmps.at("0238_aa_vo"))
      .deallocate(tmps.at("0236_aa_vo"))
      .deallocate(tmps.at("0235_aa_vo"))
      .deallocate(tmps.at("0233_aa_vo"))
      .deallocate(tmps.at("0231_aa_vo"))
      .deallocate(tmps.at("0229_aa_vo"))
      .deallocate(tmps.at("0228_aa_vo"))
      .deallocate(tmps.at("0227_aa_vo"))
      .deallocate(tmps.at("0225_aa_vo"))
      .deallocate(tmps.at("0223_aa_vo"))
      .deallocate(tmps.at("0222_aa_vo"))
      .allocate(tmps.at("0250_aa_vo"))(tmps.at("0250_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0249_aa_vo")(aa, ia))(
        tmps.at("0250_aa_vo")(aa, ia) += tmps.at("0245_aa_vo")(aa, ia))
      .deallocate(tmps.at("0249_aa_vo"))
      .deallocate(tmps.at("0245_aa_vo"))(r1.at("aa")(aa, ia) += tmps.at("0250_aa_vo")(aa, ia))
      .deallocate(tmps.at("0250_aa_vo"))
      .allocate(tmps.at("0256_bbbb_oovo"))(tmps.at("0256_bbbb_oovo")(ib, jb, ab, kb) =
                                             eri.at("bbbb_oovv")(ib, jb, bb, ab) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0257_bbbb_vooo"))(tmps.at("0257_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("bbbb")(bb, ab, jb, lb) *
                                             tmps.at("0256_bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("0266_bbbb_vvoo"))(tmps.at("0266_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0257_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0257_bbbb_vooo"))
      .allocate(tmps.at("0265_bbbb_vvoo"))(tmps.at("0265_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("0183_bb_vo")(bb, jb))
      .allocate(tmps.at("0263_bbbb_vvoo"))(tmps.at("0263_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0141_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("0262_bbbb_vvoo"))(tmps.at("0262_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("0164_bb_vo")(bb, jb))
      .allocate(tmps.at("0261_bbbb_vvoo"))(tmps.at("0261_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0150_bbbb_ovoo")(kb, bb, ib, jb))
      .allocate(tmps.at("0251_bbbb_vooo"))(tmps.at("0251_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("bbbb")(bb, ab, jb, lb) *
                                             eri.at("bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("0260_bbbb_vvoo"))(tmps.at("0260_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0251_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0251_bbbb_vooo"))
      .allocate(tmps.at("0259_bbbb_vvoo"))(tmps.at("0259_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0139_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0258_bbbb_vvoo"))(tmps.at("0258_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0135_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("0254_bbbb_vvoo"))(tmps.at("0254_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("baab_vovo")(ab, ka, ca, ib) *
                                             t2.at("abab")(ca, bb, ka, jb))
      .allocate(tmps.at("0253_bbbb_vvoo"))(tmps.at("0253_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_vovo")(ab, kb, cb, ib) *
                                             t2.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("0252_bbbb_vvoo"))(tmps.at("0252_bbbb_vvoo")(ab, bb, ib, jb) =
                                             dp.at("bb_vo")(ab, ib) * t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0255_bbbb_vvoo"))(tmps.at("0255_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0254_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0255_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0252_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0255_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0253_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0254_bbbb_vvoo"))
      .deallocate(tmps.at("0253_bbbb_vvoo"))
      .deallocate(tmps.at("0252_bbbb_vvoo"))
      .allocate(tmps.at("0264_bbbb_vvoo"))(tmps.at("0264_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0261_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0264_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0255_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0264_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0259_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("0264_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("0260_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("0264_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("0263_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("0264_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("0258_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("0264_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("0262_bbbb_vvoo")(bb, ab, jb, ib))
      .deallocate(tmps.at("0263_bbbb_vvoo"))
      .deallocate(tmps.at("0262_bbbb_vvoo"))
      .deallocate(tmps.at("0261_bbbb_vvoo"))
      .deallocate(tmps.at("0260_bbbb_vvoo"))
      .deallocate(tmps.at("0259_bbbb_vvoo"))
      .deallocate(tmps.at("0258_bbbb_vvoo"))
      .deallocate(tmps.at("0255_bbbb_vvoo"))
      .allocate(tmps.at("0267_bbbb_vvoo"))(tmps.at("0267_bbbb_vvoo")(ab, bb, ib, jb) =
                                             tmps.at("0266_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("0267_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("0265_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("0267_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0264_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0266_bbbb_vvoo"))
      .deallocate(tmps.at("0265_bbbb_vvoo"))
      .deallocate(tmps.at("0264_bbbb_vvoo"))(r2.at("bbbb")(ab, bb, ib, jb) -=
                                             tmps.at("0267_bbbb_vvoo")(ab, bb, jb, ib))(
        r2.at("bbbb")(ab, bb, ib, jb) += tmps.at("0267_bbbb_vvoo")(ab, bb, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) += tmps.at("0267_bbbb_vvoo")(bb, ab, jb, ib))(
        r2.at("bbbb")(ab, bb, ib, jb) -= tmps.at("0267_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("0267_bbbb_vvoo"))
      .allocate(tmps.at("0288_bb_oo"))(tmps.at("0288_bb_oo")(ib, jb) =
                                         t1.at("bb")(ab, jb) * tmps.at("0237_bb_ov")(ib, ab))
      .allocate(tmps.at("0286_abab_oovo"))(tmps.at("0286_abab_oovo")(ia, jb, aa, kb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0287_bb_oo"))(tmps.at("0287_bb_oo")(ib, jb) =
                                         t1.at("aa")(aa, ka) *
                                         tmps.at("0286_abab_oovo")(ka, ib, aa, jb))
      .allocate(tmps.at("0289_bb_oo"))(tmps.at("0289_bb_oo")(ib, jb) =
                                         tmps.at("0287_bb_oo")(ib, jb))(
        tmps.at("0289_bb_oo")(ib, jb) += tmps.at("0288_bb_oo")(ib, jb))
      .deallocate(tmps.at("0288_bb_oo"))
      .deallocate(tmps.at("0287_bb_oo"))
      .allocate(tmps.at("0290_bbbb_vvoo"))(tmps.at("0290_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0289_bb_oo")(kb, jb))
      .allocate(tmps.at("0285_bbbb_vvoo"))(tmps.at("0285_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0179_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0283_bbbb_vvoo"))(tmps.at("0283_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0144_bbbb_voov")(bb, kb, jb, cb))
      .allocate(tmps.at("0280_bb_oo"))(tmps.at("0280_bb_oo")(ib, jb) =
                                         eri.at("abab_oovo")(ka, ib, aa, jb) * t1.at("aa")(aa, ka))
      .allocate(tmps.at("0279_bb_oo"))(tmps.at("0279_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovo")(kb, ib, ab, jb) * t1.at("bb")(ab, kb))
      .allocate(tmps.at("0281_bb_oo"))(tmps.at("0281_bb_oo")(ib, jb) =
                                         tmps.at("0279_bb_oo")(ib, jb))(
        tmps.at("0281_bb_oo")(ib, jb) += tmps.at("0280_bb_oo")(ib, jb))
      .deallocate(tmps.at("0280_bb_oo"))
      .deallocate(tmps.at("0279_bb_oo"))
      .allocate(tmps.at("0282_bbbb_vvoo"))(tmps.at("0282_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0281_bb_oo")(kb, jb))
      .allocate(tmps.at("0278_bbbb_vvoo"))(tmps.at("0278_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0178_bbbb_oooo")(lb, kb, ib, jb))
      .allocate(tmps.at("0276_bb_oo"))(tmps.at("0276_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovv")(kb, ib, ab, bb) *
                                         t2.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0277_bbbb_vvoo"))(tmps.at("0277_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0276_bb_oo")(kb, jb))
      .allocate(tmps.at("0275_bbbb_vvoo"))(tmps.at("0275_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0137_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("0273_bb_oo"))(tmps.at("0273_bb_oo")(ib, jb) =
                                         f.at("bb_ov")(ib, ab) * t1.at("bb")(ab, jb))
      .allocate(tmps.at("0274_bbbb_vvoo"))(tmps.at("0274_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0273_bb_oo")(kb, jb))
      .allocate(tmps.at("0271_bb_oo"))(tmps.at("0271_bb_oo")(ib, jb) =
                                         eri.at("abab_oovv")(ka, ib, aa, bb) *
                                         t2.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("0272_bbbb_vvoo"))(tmps.at("0272_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0271_bb_oo")(kb, jb))
      .allocate(tmps.at("0269_bbbb_vvoo"))(tmps.at("0269_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(cb, ib) *
                                             eri.at("bbbb_vvvo")(ab, bb, cb, jb))
      .allocate(tmps.at("0268_bbbb_vvoo"))(tmps.at("0268_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) * f.at("bb_oo")(kb, jb))
      .allocate(tmps.at("0270_bbbb_vvoo"))(tmps.at("0270_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0268_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0270_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0269_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0269_bbbb_vvoo"))
      .deallocate(tmps.at("0268_bbbb_vvoo"))
      .allocate(tmps.at("0284_bbbb_vvoo"))(tmps.at("0284_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0277_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0284_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 * tmps.at("0270_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0284_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0278_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0284_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("0282_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0284_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("0283_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("0284_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("0272_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0284_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 *
        tmps.at("0275_bbbb_vvoo")(bb, ab, ib, jb))(tmps.at("0284_bbbb_vvoo")(ab, bb, ib, jb) +=
                                                   2.00 * tmps.at("0274_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0283_bbbb_vvoo"))
      .deallocate(tmps.at("0282_bbbb_vvoo"))
      .deallocate(tmps.at("0278_bbbb_vvoo"))
      .deallocate(tmps.at("0277_bbbb_vvoo"))
      .deallocate(tmps.at("0275_bbbb_vvoo"))
      .deallocate(tmps.at("0274_bbbb_vvoo"))
      .deallocate(tmps.at("0272_bbbb_vvoo"))
      .deallocate(tmps.at("0270_bbbb_vvoo"))
      .allocate(tmps.at("0291_bbbb_vvoo"))(tmps.at("0291_bbbb_vvoo")(ab, bb, ib, jb) =
                                             2.00 * tmps.at("0290_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0291_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 * tmps.at("0285_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("0291_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0284_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0290_bbbb_vvoo"))
      .deallocate(tmps.at("0285_bbbb_vvoo"))
      .deallocate(tmps.at("0284_bbbb_vvoo"))(r2.at("bbbb")(ab, bb, ib, jb) -=
                                             0.50 * tmps.at("0291_bbbb_vvoo")(ab, bb, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) += 0.50 * tmps.at("0291_bbbb_vvoo")(ab, bb, jb, ib))
      .deallocate(tmps.at("0291_bbbb_vvoo"))
      .allocate(tmps.at("0308_bbbb_vooo"))(tmps.at("0308_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(bb, jb) *
                                             tmps.at("0141_bbbb_vovo")(ab, ib, bb, kb))
      .allocate(tmps.at("0309_bbbb_vvoo"))(tmps.at("0309_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0308_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0304_bbbb_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("abab_oovv")(la, ib, ca, bb) * t1.at("aa")(ca, la))(
        tmps.at("0304_bbbb_ovoo")(ib, ab, jb, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2.at("bbbb")(bb, ab, jb, kb))
      .allocate(tmps.at("0303_bbbb_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("bbbb_oovv")(lb, ib, cb, bb) * t1.at("bb")(cb, lb))(
        tmps.at("0303_bbbb_ovoo")(ib, ab, jb, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2.at("bbbb")(bb, ab, jb, kb))
      .allocate(tmps.at("0305_bbbb_ovoo"))(tmps.at("0305_bbbb_ovoo")(ib, ab, jb, kb) =
                                             tmps.at("0303_bbbb_ovoo")(ib, ab, jb, kb))(
        tmps.at("0305_bbbb_ovoo")(ib, ab, jb, kb) += tmps.at("0304_bbbb_ovoo")(ib, ab, jb, kb))
      .deallocate(tmps.at("0304_bbbb_ovoo"))
      .deallocate(tmps.at("0303_bbbb_ovoo"))
      .allocate(tmps.at("0306_bbbb_vvoo"))(tmps.at("0306_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0305_bbbb_ovoo")(kb, bb, ib, jb))
      .allocate(tmps.at("0301_bbbb_vooo"))(tmps.at("0301_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("bbbb")(bb, ab, jb, kb) * f.at("bb_ov")(ib, bb))
      .allocate(tmps.at("0302_bbbb_vvoo"))(tmps.at("0302_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0301_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0299_bb_vv"))(tmps.at("0299_bb_vv")(ab, bb) =
                                         eri.at("baab_vovv")(ab, ia, ca, bb) * t1.at("aa")(ca, ia))
      .allocate(tmps.at("0300_bbbb_vvoo"))(tmps.at("0300_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0299_bb_vv")(bb, cb))
      .allocate(tmps.at("0297_bb_vv"))(tmps.at("0297_bb_vv")(ab, bb) =
                                         eri.at("bbbb_vovv")(ab, ib, cb, bb) * t1.at("bb")(cb, ib))
      .allocate(tmps.at("0298_bbbb_vvoo"))(tmps.at("0298_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0297_bb_vv")(bb, cb))
      .allocate(tmps.at("0295_bbbb_vooo"))(tmps.at("0295_bbbb_vooo")(ab, ib, jb, kb) =
                                             eri.at("bbbb_vovv")(ab, ib, bb, cb) *
                                             t2.at("bbbb")(bb, cb, jb, kb))
      .allocate(tmps.at("0296_bbbb_vvoo"))(tmps.at("0296_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0295_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0293_bbbb_vvoo"))(tmps.at("0293_bbbb_vvoo")(ab, bb, ib, jb) =
                                             f.at("bb_vv")(ab, cb) * t2.at("bbbb")(cb, bb, ib, jb))
      .allocate(tmps.at("0292_bbbb_vvoo"))(tmps.at("0292_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_vooo")(ab, kb, ib, jb) *
                                             t1.at("bb")(bb, kb))
      .allocate(tmps.at("0294_bbbb_vvoo"))(tmps.at("0294_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0293_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0294_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0292_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0293_bbbb_vvoo"))
      .deallocate(tmps.at("0292_bbbb_vvoo"))
      .allocate(tmps.at("0307_bbbb_vvoo"))(tmps.at("0307_bbbb_vvoo")(ab, bb, ib, jb) =
                                             tmps.at("0294_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0307_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0302_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0307_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0306_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0307_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0298_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("0307_bbbb_vvoo")(ab, bb, ib, jb) +=
        tmps.at("0300_bbbb_vvoo")(bb, ab, ib, jb))(tmps.at("0307_bbbb_vvoo")(ab, bb, ib, jb) +=
                                                   0.50 * tmps.at("0296_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("0306_bbbb_vvoo"))
      .deallocate(tmps.at("0302_bbbb_vvoo"))
      .deallocate(tmps.at("0300_bbbb_vvoo"))
      .deallocate(tmps.at("0298_bbbb_vvoo"))
      .deallocate(tmps.at("0296_bbbb_vvoo"))
      .deallocate(tmps.at("0294_bbbb_vvoo"))
      .allocate(tmps.at("0310_bbbb_vvoo"))(tmps.at("0310_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("0309_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("0310_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0307_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0309_bbbb_vvoo"))
      .deallocate(tmps.at("0307_bbbb_vvoo"))(r2.at("bbbb")(ab, bb, ib, jb) -=
                                             tmps.at("0310_bbbb_vvoo")(ab, bb, ib, jb))(
        r2.at("bbbb")(ab, bb, ib, jb) += tmps.at("0310_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("0310_bbbb_vvoo"))
      .allocate(tmps.at("0320_bbbb_oooo"))(tmps.at("0320_bbbb_oooo")(ib, jb, kb, lb) =
                                             eri.at("bbbb_oovv")(ib, jb, ab, bb) *
                                             t2_1p.at("bbbb")(ab, bb, kb, lb))
      .allocate(tmps.at("0325_bbbb_vooo"))(tmps.at("0325_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0320_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("0326_bbbb_vvoo"))(tmps.at("0326_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0325_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("0323_bbbb_oooo"))(tmps.at("0323_bbbb_oooo")(ib, jb, kb, lb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("0256_bbbb_oovo")(ib, jb, ab, lb))
      .allocate(tmps.at("0324_bbbb_vvoo"))(tmps.at("0324_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0323_bbbb_oooo")(lb, kb, ib, jb))
      .allocate(tmps.at("0321_bbbb_vvoo"))(tmps.at("0321_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0320_bbbb_oooo")(lb, kb, ib, jb))
      .allocate(tmps.at("0318_bbbb_oooo"))(tmps.at("0318_bbbb_oooo")(ib, jb, kb, lb) =
                                             eri.at("bbbb_oovv")(ib, jb, ab, bb) *
                                             t2.at("bbbb")(ab, bb, kb, lb))
      .allocate(tmps.at("0319_bbbb_vvoo"))(tmps.at("0319_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0318_bbbb_oooo")(lb, kb, ib, jb))
      .allocate(tmps.at("0316_bbbb_vvoo"))(tmps.at("0316_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_vvvv")(ab, bb, cb, db) *
                                             t2_1p.at("bbbb")(cb, db, ib, jb))
      .allocate(tmps.at("0315_bbbb_vvoo"))(tmps.at("0315_bbbb_vvoo")(ab, bb, ib, jb) =
                                             scalars.at("0013")() *
                                             t2_2p.at("bbbb")(ab, bb, ib, jb))
      .allocate(tmps.at("0314_bbbb_vvoo"))(tmps.at("0314_bbbb_vvoo")(ab, bb, ib, jb) =
                                             scalars.at("0002")() *
                                             t2_1p.at("bbbb")(ab, bb, ib, jb))
      .allocate(tmps.at("0313_bbbb_vvoo"))(tmps.at("0313_bbbb_vvoo")(ab, bb, ib, jb) =
                                             scalars.at("0001")() *
                                             t2_1p.at("bbbb")(ab, bb, ib, jb))
      .allocate(tmps.at("0312_bbbb_vvoo"))(tmps.at("0312_bbbb_vvoo")(ab, bb, ib, jb) =
                                             scalars.at("0015")() *
                                             t2_2p.at("bbbb")(ab, bb, ib, jb))
      .allocate(tmps.at("0311_bbbb_vvoo"))(tmps.at("0311_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_oooo")(kb, lb, ib, jb) *
                                             t2_1p.at("bbbb")(ab, bb, lb, kb))
      .allocate(tmps.at("0317_bbbb_vvoo"))(tmps.at("0317_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -0.250 * tmps.at("0311_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0317_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0315_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0317_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0312_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0317_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("0313_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0317_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.250 *
        tmps.at("0316_bbbb_vvoo")(ab, bb, ib, jb))(tmps.at("0317_bbbb_vvoo")(ab, bb, ib, jb) +=
                                                   0.50 * tmps.at("0314_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0316_bbbb_vvoo"))
      .deallocate(tmps.at("0315_bbbb_vvoo"))
      .deallocate(tmps.at("0314_bbbb_vvoo"))
      .deallocate(tmps.at("0313_bbbb_vvoo"))
      .deallocate(tmps.at("0312_bbbb_vvoo"))
      .deallocate(tmps.at("0311_bbbb_vvoo"))
      .allocate(tmps.at("0322_bbbb_vvoo"))(tmps.at("0322_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -8.00 * tmps.at("0317_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0322_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0321_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0322_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0319_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0321_bbbb_vvoo"))
      .deallocate(tmps.at("0319_bbbb_vvoo"))
      .deallocate(tmps.at("0317_bbbb_vvoo"))
      .allocate(tmps.at("0327_bbbb_vvoo"))(tmps.at("0327_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -2.00 * tmps.at("0324_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("0327_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 * tmps.at("0326_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("0327_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("0322_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0326_bbbb_vvoo"))
      .deallocate(tmps.at("0324_bbbb_vvoo"))
      .deallocate(tmps.at("0322_bbbb_vvoo"))(r2_1p.at("bbbb")(ab, bb, ib, jb) -=
                                             0.250 * tmps.at("0327_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("0327_bbbb_vvoo"))
      .allocate(tmps.at("0430_aa_oo"))(tmps.at("0430_aa_oo")(ia, ja) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0232_aa_ov")(ia, aa))
      .allocate(tmps.at("0428_aaaa_oovo"))(tmps.at("0428_aaaa_oovo")(ia, ja, aa, ka) =
                                             eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                             t1_2p.at("aa")(ba, ka))
      .allocate(tmps.at("0429_aa_oo"))(tmps.at("0429_aa_oo")(ia, ja) =
                                         t1.at("aa")(aa, ka) *
                                         tmps.at("0428_aaaa_oovo")(ka, ia, aa, ja))
      .allocate(tmps.at("0431_aa_oo"))(tmps.at("0431_aa_oo")(ia, ja) =
                                         tmps.at("0429_aa_oo")(ia, ja))(
        tmps.at("0431_aa_oo")(ia, ja) += tmps.at("0430_aa_oo")(ia, ja))
      .deallocate(tmps.at("0430_aa_oo"))
      .deallocate(tmps.at("0429_aa_oo"))
      .allocate(tmps.at("0432_aa_vo"))(tmps.at("0432_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0431_aa_oo")(ja, ia))
      .allocate(tmps.at("0354_aa_ov"))(tmps.at("0354_aa_ov")(ia, aa) =
                                         eri.at("abab_oovv")(ia, jb, aa, bb) *
                                         t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("0425_aa_oo"))(tmps.at("0425_aa_oo")(ia, ja) =
                                         t1.at("aa")(aa, ja) * tmps.at("0354_aa_ov")(ia, aa))
      .allocate(tmps.at("0424_aa_oo"))(tmps.at("0424_aa_oo")(ia, ja) =
                                         t1_2p.at("aa")(aa, ka) *
                                         tmps.at("0203_aaaa_oovo")(ia, ka, aa, ja))
      .allocate(tmps.at("0426_aa_oo"))(tmps.at("0426_aa_oo")(ia, ja) =
                                         tmps.at("0424_aa_oo")(ia, ja))(
        tmps.at("0426_aa_oo")(ia, ja) += tmps.at("0425_aa_oo")(ia, ja))
      .deallocate(tmps.at("0425_aa_oo"))
      .deallocate(tmps.at("0424_aa_oo"))
      .allocate(tmps.at("0427_aa_vo"))(tmps.at("0427_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0426_aa_oo")(ja, ia))
      .allocate(tmps.at("0365_aa_ov"))(tmps.at("0365_aa_ov")(ia, aa) =
                                         eri.at("abab_oovv")(ia, jb, aa, bb) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0422_aa_oo"))(tmps.at("0422_aa_oo")(ia, ja) =
                                         t1.at("aa")(aa, ja) * tmps.at("0365_aa_ov")(ia, aa))
      .allocate(tmps.at("0423_aa_vo"))(tmps.at("0423_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0422_aa_oo")(ja, ia))
      .allocate(tmps.at("0421_aa_vo"))(tmps.at("0421_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0248_aa_oo")(ja, ia))
      .allocate(tmps.at("0419_aa_oo"))(tmps.at("0419_aa_oo")(ia, ja) =
                                         t1_1p.at("aa")(aa, ka) *
                                         tmps.at("0203_aaaa_oovo")(ka, ia, aa, ja))
      .allocate(tmps.at("0420_aa_vo"))(tmps.at("0420_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0419_aa_oo")(ja, ia))
      .allocate(tmps.at("0416_aa_oo"))(tmps.at("0416_aa_oo")(ia, ja) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0365_aa_ov")(ia, aa))
      .allocate(tmps.at("0410_aaaa_oovo"))(tmps.at("0410_aaaa_oovo")(ia, ja, aa, ka) =
                                             eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                             t1_1p.at("aa")(ba, ka))
      .allocate(tmps.at("0415_aa_oo"))(tmps.at("0415_aa_oo")(ia, ja) =
                                         t1_1p.at("aa")(aa, ka) *
                                         tmps.at("0410_aaaa_oovo")(ia, ka, aa, ja))
      .allocate(tmps.at("0417_aa_oo"))(tmps.at("0417_aa_oo")(ia, ja) =
                                         -1.00 * tmps.at("0416_aa_oo")(ia, ja))(
        tmps.at("0417_aa_oo")(ia, ja) += tmps.at("0415_aa_oo")(ia, ja))
      .deallocate(tmps.at("0416_aa_oo"))
      .deallocate(tmps.at("0415_aa_oo"))
      .allocate(tmps.at("0418_aa_vo"))(tmps.at("0418_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0417_aa_oo")(ja, ia))
      .allocate(tmps.at("0412_aa_oo"))(tmps.at("0412_aa_oo")(ia, ja) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0232_aa_ov")(ia, aa))
      .allocate(tmps.at("0411_aa_oo"))(tmps.at("0411_aa_oo")(ia, ja) =
                                         t1.at("aa")(aa, ka) *
                                         tmps.at("0410_aaaa_oovo")(ka, ia, aa, ja))
      .allocate(tmps.at("0413_aa_oo"))(tmps.at("0413_aa_oo")(ia, ja) =
                                         tmps.at("0411_aa_oo")(ia, ja))(
        tmps.at("0413_aa_oo")(ia, ja) += tmps.at("0412_aa_oo")(ia, ja))
      .deallocate(tmps.at("0412_aa_oo"))
      .deallocate(tmps.at("0411_aa_oo"))
      .allocate(tmps.at("0414_aa_vo"))(tmps.at("0414_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0413_aa_oo")(ja, ia))
      .allocate(tmps.at("0407_aa_vv"))(tmps.at("0407_aa_vv")(aa, ba) =
                                         eri.at("aaaa_oovv")(ia, ja, ba, ca) *
                                         t2_1p.at("aaaa")(ca, aa, ja, ia))
      .allocate(tmps.at("0408_aa_vo"))(tmps.at("0408_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ia) * tmps.at("0407_aa_vv")(aa, ba))
      .allocate(tmps.at("0406_aa_vo"))(tmps.at("0406_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ia) * tmps.at("0043_aa_vv")(aa, ba))
      .allocate(tmps.at("0405_aa_vo"))(tmps.at("0405_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0241_aa_oo")(ja, ia))
      .allocate(tmps.at("0402_aa_vv"))(tmps.at("0402_aa_vv")(aa, ba) =
                                         eri.at("aaaa_oovv")(ia, ja, ba, ca) *
                                         t2_2p.at("aaaa")(ca, aa, ja, ia))
      .allocate(tmps.at("0401_aa_vv"))(tmps.at("0401_aa_vv")(aa, ba) =
                                         eri.at("abab_oovv")(ia, jb, ba, cb) *
                                         t2_2p.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0403_aa_vv"))(tmps.at("0403_aa_vv")(aa, ba) =
                                         2.00 * tmps.at("0401_aa_vv")(aa, ba))(
        tmps.at("0403_aa_vv")(aa, ba) += tmps.at("0402_aa_vv")(aa, ba))
      .deallocate(tmps.at("0402_aa_vv"))
      .deallocate(tmps.at("0401_aa_vv"))
      .allocate(tmps.at("0404_aa_vo"))(tmps.at("0404_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ia) * tmps.at("0403_aa_vv")(aa, ba))
      .allocate(tmps.at("0398_aa_oo"))(tmps.at("0398_aa_oo")(ia, ja) =
                                         f.at("aa_ov")(ia, aa) * t1_2p.at("aa")(aa, ja))
      .allocate(tmps.at("0397_aa_oo"))(tmps.at("0397_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovv")(ia, ka, aa, ba) *
                                         t2_2p.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0396_aa_oo"))(tmps.at("0396_aa_oo")(ia, ja) =
                                         eri.at("abab_oovv")(ia, kb, aa, bb) *
                                         t2_2p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0399_aa_oo"))(tmps.at("0399_aa_oo")(ia, ja) =
                                         2.00 * tmps.at("0396_aa_oo")(ia, ja))(
        tmps.at("0399_aa_oo")(ia, ja) += 2.00 * tmps.at("0398_aa_oo")(ia, ja))(
        tmps.at("0399_aa_oo")(ia, ja) += tmps.at("0397_aa_oo")(ia, ja))
      .deallocate(tmps.at("0398_aa_oo"))
      .deallocate(tmps.at("0397_aa_oo"))
      .deallocate(tmps.at("0396_aa_oo"))
      .allocate(tmps.at("0400_aa_vo"))(tmps.at("0400_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0399_aa_oo")(ja, ia))
      .allocate(tmps.at("0394_aa_oo"))(tmps.at("0394_aa_oo")(ia, ja) =
                                         eri.at("abba_oovo")(ia, kb, ab, ja) *
                                         t1_1p.at("bb")(ab, kb))
      .allocate(tmps.at("0395_aa_vo"))(tmps.at("0395_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0394_aa_oo")(ja, ia))
      .allocate(tmps.at("0392_abba_vovo"))(tmps.at("0392_abba_vovo")(aa, ib, bb, ja) =
                                             eri.at("abab_vovv")(aa, ib, ca, bb) *
                                             t1_1p.at("aa")(ca, ja))
      .allocate(tmps.at("0393_aa_vo"))(tmps.at("0393_aa_vo")(aa, ia) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0392_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0390_aa_oo"))(tmps.at("0390_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovv")(ka, ia, aa, ba) *
                                         t2.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0391_aa_vo"))(tmps.at("0391_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0390_aa_oo")(ja, ia))
      .allocate(tmps.at("0389_aa_vo"))(tmps.at("0389_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0039_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0387_aa_oo"))(tmps.at("0387_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovo")(ka, ia, aa, ja) *
                                         t1_1p.at("aa")(aa, ka))
      .allocate(tmps.at("0388_aa_vo"))(tmps.at("0388_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0387_aa_oo")(ja, ia))
      .allocate(tmps.at("0384_aa_oo"))(tmps.at("0384_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovo")(ia, ka, aa, ja) *
                                         t1_2p.at("aa")(aa, ka))
      .allocate(tmps.at("0383_aa_oo"))(tmps.at("0383_aa_oo")(ia, ja) =
                                         eri.at("abba_oovo")(ia, kb, ab, ja) *
                                         t1_2p.at("bb")(ab, kb))
      .allocate(tmps.at("0385_aa_oo"))(tmps.at("0385_aa_oo")(ia, ja) =
                                         tmps.at("0383_aa_oo")(ia, ja))(
        tmps.at("0385_aa_oo")(ia, ja) += tmps.at("0384_aa_oo")(ia, ja))
      .deallocate(tmps.at("0384_aa_oo"))
      .deallocate(tmps.at("0383_aa_oo"))
      .allocate(tmps.at("0386_aa_vo"))(tmps.at("0386_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0385_aa_oo")(ja, ia))
      .allocate(tmps.at("0381_bb_ov"))(tmps.at("0381_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(jb, ib, bb, ab) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0382_aa_vo"))(tmps.at("0382_aa_vo")(aa, ia) =
                                         t2_1p.at("abab")(aa, bb, ia, jb) *
                                         tmps.at("0381_bb_ov")(jb, bb))
      .allocate(tmps.at("0379_aaaa_voov"))(tmps.at("0379_aaaa_voov")(aa, ia, ja, ba) =
                                             t2_2p.at("abab")(aa, cb, ja, kb) *
                                             eri.at("abab_oovv")(ia, kb, ba, cb))
      .allocate(tmps.at("0380_aa_vo"))(tmps.at("0380_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0379_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0378_aa_vo"))(tmps.at("0378_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ja) *
                                         tmps.at("0037_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0376_abab_voov"))(tmps.at("0376_abab_voov")(aa, ib, ja, bb) =
                                             t2.at("abab")(aa, cb, ja, kb) *
                                             eri.at("bbbb_oovv")(kb, ib, cb, bb))
      .allocate(tmps.at("0377_aa_vo"))(tmps.at("0377_aa_vo")(aa, ia) =
                                         t1_2p.at("bb")(bb, jb) *
                                         tmps.at("0376_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0374_aa_vv"))(tmps.at("0374_aa_vv")(aa, ba) =
                                         eri.at("aaaa_oovv")(ia, ja, ca, ba) *
                                         t2.at("aaaa")(ca, aa, ja, ia))
      .allocate(tmps.at("0375_aa_vo"))(tmps.at("0375_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ia) * tmps.at("0374_aa_vv")(aa, ba))
      .allocate(tmps.at("0372_abba_vovo"))(tmps.at("0372_abba_vovo")(aa, ib, bb, ja) =
                                             eri.at("abab_vovv")(aa, ib, ca, bb) *
                                             t1_2p.at("aa")(ca, ja))
      .allocate(tmps.at("0373_aa_vo"))(tmps.at("0373_aa_vo")(aa, ia) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0372_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0371_aa_vo"))(tmps.at("0371_aa_vo")(aa, ia) =
                                         t2_2p.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0232_aa_ov")(ja, ba))
      .allocate(tmps.at("0369_aa_oo"))(tmps.at("0369_aa_oo")(ia, ja) =
                                         eri.at("abab_oovv")(ia, kb, aa, bb) *
                                         t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0370_aa_vo"))(tmps.at("0370_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0369_aa_oo")(ja, ia))
      .allocate(tmps.at("0368_aa_vo"))(tmps.at("0368_aa_vo")(aa, ia) =
                                         t1_2p.at("bb")(bb, jb) *
                                         tmps.at("0230_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0367_aa_vo"))(tmps.at("0367_aa_vo")(aa, ia) =
                                         t2_2p.at("abab")(aa, bb, ia, jb) *
                                         tmps.at("0237_bb_ov")(jb, bb))
      .allocate(tmps.at("0366_aa_vo"))(tmps.at("0366_aa_vo")(aa, ia) =
                                         t2_1p.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0365_aa_ov")(ja, ba))
      .allocate(tmps.at("0363_aaaa_vovo"))(tmps.at("0363_aaaa_vovo")(aa, ia, ba, ja) =
                                             eri.at("aaaa_vovv")(aa, ia, ca, ba) *
                                             t1.at("aa")(ca, ja))
      .allocate(tmps.at("0364_aa_vo"))(tmps.at("0364_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ja) *
                                         tmps.at("0363_aaaa_vovo")(aa, ja, ba, ia))
      .allocate(tmps.at("0362_aa_vo"))(tmps.at("0362_aa_vo")(aa, ia) =
                                         t2_2p.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0224_aa_ov")(ja, ba))
      .allocate(tmps.at("0360_aa_oo"))(tmps.at("0360_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovv")(ia, ka, aa, ba) *
                                         t2_1p.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0361_aa_vo"))(tmps.at("0361_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0360_aa_oo")(ja, ia))
      .allocate(tmps.at("0359_aa_vo"))(tmps.at("0359_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0234_aa_oo")(ja, ia))
      .allocate(tmps.at("0357_aa_ov"))(tmps.at("0357_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ja, ia, ba, aa) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0358_aa_vo"))(tmps.at("0358_aa_vo")(aa, ia) =
                                         t2_1p.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0357_aa_ov")(ja, ba))
      .allocate(tmps.at("0356_aa_vo"))(tmps.at("0356_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ia) * tmps.at("0041_aa_vv")(aa, ba))
      .allocate(tmps.at("0355_aa_vo"))(tmps.at("0355_aa_vo")(aa, ia) =
                                         t2.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0354_aa_ov")(ja, ba))
      .allocate(tmps.at("0023_aa_oo"))(tmps.at("0023_aa_oo")(ia, ja) =
                                         dp.at("aa_ov")(ia, aa) * t1_1p.at("aa")(aa, ja))
      .allocate(tmps.at("0353_aa_vo"))(tmps.at("0353_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0023_aa_oo")(ja, ia))
      .allocate(tmps.at("0352_aa_vo"))(tmps.at("0352_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0243_aa_oo")(ja, ia))
      .allocate(tmps.at("0350_aaaa_vovo"))(tmps.at("0350_aaaa_vovo")(aa, ia, ba, ja) =
                                             eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                             t1_2p.at("aa")(ca, ja))
      .allocate(tmps.at("0351_aa_vo"))(tmps.at("0351_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0350_aaaa_vovo")(aa, ja, ba, ia))
      .allocate(tmps.at("0348_aaaa_vovo"))(tmps.at("0348_aaaa_vovo")(aa, ia, ba, ja) =
                                             eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                             t1_1p.at("aa")(ca, ja))
      .allocate(tmps.at("0349_aa_vo"))(tmps.at("0349_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0348_aaaa_vovo")(aa, ja, ba, ia))
      .allocate(tmps.at("0346_aaaa_voov"))(tmps.at("0346_aaaa_voov")(aa, ia, ja, ba) =
                                             t2.at("aaaa")(ca, aa, ja, ka) *
                                             eri.at("aaaa_oovv")(ka, ia, ca, ba))
      .allocate(tmps.at("0347_aa_vo"))(tmps.at("0347_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ja) *
                                         tmps.at("0346_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0344_aa_oo"))(tmps.at("0344_aa_oo")(ia, ja) =
                                         f.at("aa_ov")(ia, aa) * t1_1p.at("aa")(aa, ja))
      .allocate(tmps.at("0345_aa_vo"))(tmps.at("0345_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0344_aa_oo")(ja, ia))
      .allocate(tmps.at("0031_aa_oo"))(tmps.at("0031_aa_oo")(ia, ja) =
                                         dp.at("aa_ov")(ia, aa) * t1_2p.at("aa")(aa, ja))
      .allocate(tmps.at("0343_aa_vo"))(tmps.at("0343_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0031_aa_oo")(ja, ia))
      .allocate(tmps.at("0341_aa_vo"))(tmps.at("0341_aa_vo")(aa, ia) =
                                         scalars.at("0002")() * t1_2p.at("aa")(aa, ia))
      .allocate(tmps.at("0340_aa_vo"))(tmps.at("0340_aa_vo")(aa, ia) =
                                         f.at("aa_ov")(ja, ba) * t2_2p.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0339_aa_vo"))(tmps.at("0339_aa_vo")(aa, ia) =
                                         eri.at("abba_oovo")(ja, kb, bb, ia) *
                                         t2_2p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0338_aa_vo"))(tmps.at("0338_aa_vo")(aa, ia) =
                                         f.at("aa_vv")(aa, ba) * t1_2p.at("aa")(ba, ia))
      .allocate(tmps.at("0337_aa_vo"))(tmps.at("0337_aa_vo")(aa, ia) =
                                         scalars.at("0001")() * t1_2p.at("aa")(aa, ia))
      .allocate(tmps.at("0336_aa_vo"))(tmps.at("0336_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovv")(aa, ja, ba, ca) *
                                         t2_2p.at("aaaa")(ba, ca, ia, ja))
      .allocate(tmps.at("0335_aa_vo"))(tmps.at("0335_aa_vo")(aa, ia) =
                                         scalars.at("0016")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0334_aa_vo"))(tmps.at("0334_aa_vo")(aa, ia) =
                                         scalars.at("0014")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0333_aa_vo"))(tmps.at("0333_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovo")(aa, ja, ba, ia) *
                                         t1_2p.at("aa")(ba, ja))
      .allocate(tmps.at("0332_aa_vo"))(tmps.at("0332_aa_vo")(aa, ia) =
                                         eri.at("aaaa_oovo")(ja, ka, ba, ia) *
                                         t2_2p.at("aaaa")(ba, aa, ka, ja))
      .allocate(tmps.at("0331_aa_vo"))(tmps.at("0331_aa_vo")(aa, ia) =
                                         f.at("aa_oo")(ja, ia) * t1_2p.at("aa")(aa, ja))
      .allocate(tmps.at("0330_aa_vo"))(tmps.at("0330_aa_vo")(aa, ia) =
                                         eri.at("abab_vovv")(aa, jb, ba, cb) *
                                         t2_2p.at("abab")(ba, cb, ia, jb))
      .allocate(tmps.at("0329_aa_vo"))(tmps.at("0329_aa_vo")(aa, ia) =
                                         eri.at("abba_vovo")(aa, jb, bb, ia) *
                                         t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("0328_aa_vo"))(tmps.at("0328_aa_vo")(aa, ia) =
                                         f.at("bb_ov")(jb, bb) * t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0342_aa_vo"))(tmps.at("0342_aa_vo")(aa, ia) =
                                         -0.50 * tmps.at("0332_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) += tmps.at("0340_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) += tmps.at("0329_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) += tmps.at("0333_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) -= tmps.at("0338_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) -= tmps.at("0330_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) -= tmps.at("0339_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) -= 2.00 * tmps.at("0341_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) += tmps.at("0331_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) -= tmps.at("0328_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) -= 0.50 * tmps.at("0336_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) -= tmps.at("0334_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) -= tmps.at("0335_aa_vo")(aa, ia))(
        tmps.at("0342_aa_vo")(aa, ia) -= 2.00 * tmps.at("0337_aa_vo")(aa, ia))
      .deallocate(tmps.at("0341_aa_vo"))
      .deallocate(tmps.at("0340_aa_vo"))
      .deallocate(tmps.at("0339_aa_vo"))
      .deallocate(tmps.at("0338_aa_vo"))
      .deallocate(tmps.at("0337_aa_vo"))
      .deallocate(tmps.at("0336_aa_vo"))
      .deallocate(tmps.at("0335_aa_vo"))
      .deallocate(tmps.at("0334_aa_vo"))
      .deallocate(tmps.at("0333_aa_vo"))
      .deallocate(tmps.at("0332_aa_vo"))
      .deallocate(tmps.at("0331_aa_vo"))
      .deallocate(tmps.at("0330_aa_vo"))
      .deallocate(tmps.at("0329_aa_vo"))
      .deallocate(tmps.at("0328_aa_vo"))
      .allocate(tmps.at("0409_aa_vo"))(tmps.at("0409_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0361_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= tmps.at("0408_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += tmps.at("0391_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0352_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0386_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0389_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0368_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0367_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0351_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0378_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0366_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0388_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0364_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0380_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0362_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 6.00 * tmps.at("0343_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0358_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0395_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0405_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0349_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 6.00 * tmps.at("0353_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0371_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0342_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= tmps.at("0400_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0393_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0347_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= tmps.at("0404_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0355_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0359_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0345_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += tmps.at("0375_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0373_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0377_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) += 2.00 * tmps.at("0382_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0406_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0370_aa_vo")(aa, ia))(
        tmps.at("0409_aa_vo")(aa, ia) -= 2.00 * tmps.at("0356_aa_vo")(aa, ia))
      .deallocate(tmps.at("0408_aa_vo"))
      .deallocate(tmps.at("0406_aa_vo"))
      .deallocate(tmps.at("0405_aa_vo"))
      .deallocate(tmps.at("0404_aa_vo"))
      .deallocate(tmps.at("0400_aa_vo"))
      .deallocate(tmps.at("0395_aa_vo"))
      .deallocate(tmps.at("0393_aa_vo"))
      .deallocate(tmps.at("0391_aa_vo"))
      .deallocate(tmps.at("0389_aa_vo"))
      .deallocate(tmps.at("0388_aa_vo"))
      .deallocate(tmps.at("0386_aa_vo"))
      .deallocate(tmps.at("0382_aa_vo"))
      .deallocate(tmps.at("0380_aa_vo"))
      .deallocate(tmps.at("0378_aa_vo"))
      .deallocate(tmps.at("0377_aa_vo"))
      .deallocate(tmps.at("0375_aa_vo"))
      .deallocate(tmps.at("0373_aa_vo"))
      .deallocate(tmps.at("0371_aa_vo"))
      .deallocate(tmps.at("0370_aa_vo"))
      .deallocate(tmps.at("0368_aa_vo"))
      .deallocate(tmps.at("0367_aa_vo"))
      .deallocate(tmps.at("0366_aa_vo"))
      .deallocate(tmps.at("0364_aa_vo"))
      .deallocate(tmps.at("0362_aa_vo"))
      .deallocate(tmps.at("0361_aa_vo"))
      .deallocate(tmps.at("0359_aa_vo"))
      .deallocate(tmps.at("0358_aa_vo"))
      .deallocate(tmps.at("0356_aa_vo"))
      .deallocate(tmps.at("0355_aa_vo"))
      .deallocate(tmps.at("0353_aa_vo"))
      .deallocate(tmps.at("0352_aa_vo"))
      .deallocate(tmps.at("0351_aa_vo"))
      .deallocate(tmps.at("0349_aa_vo"))
      .deallocate(tmps.at("0347_aa_vo"))
      .deallocate(tmps.at("0345_aa_vo"))
      .deallocate(tmps.at("0343_aa_vo"))
      .deallocate(tmps.at("0342_aa_vo"))
      .allocate(tmps.at("0433_aa_vo"))(tmps.at("0433_aa_vo")(aa, ia) =
                                         -0.50 * tmps.at("0409_aa_vo")(aa, ia))(
        tmps.at("0433_aa_vo")(aa, ia) -= tmps.at("0418_aa_vo")(aa, ia))(
        tmps.at("0433_aa_vo")(aa, ia) -= tmps.at("0420_aa_vo")(aa, ia))(
        tmps.at("0433_aa_vo")(aa, ia) += tmps.at("0414_aa_vo")(aa, ia))(
        tmps.at("0433_aa_vo")(aa, ia) += tmps.at("0427_aa_vo")(aa, ia))(
        tmps.at("0433_aa_vo")(aa, ia) += tmps.at("0432_aa_vo")(aa, ia))(
        tmps.at("0433_aa_vo")(aa, ia) += tmps.at("0423_aa_vo")(aa, ia))(
        tmps.at("0433_aa_vo")(aa, ia) += tmps.at("0421_aa_vo")(aa, ia))
      .deallocate(tmps.at("0432_aa_vo"))
      .deallocate(tmps.at("0427_aa_vo"))
      .deallocate(tmps.at("0423_aa_vo"))
      .deallocate(tmps.at("0421_aa_vo"))
      .deallocate(tmps.at("0420_aa_vo"))
      .deallocate(tmps.at("0418_aa_vo"))
      .deallocate(tmps.at("0414_aa_vo"))
      .deallocate(tmps.at("0409_aa_vo"))(r1_2p.at("aa")(aa, ia) -=
                                         2.00 * tmps.at("0433_aa_vo")(aa, ia))
      .deallocate(tmps.at("0433_aa_vo"))
      .allocate(tmps.at("0480_aa_vo"))(tmps.at("0480_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0413_aa_oo")(ja, ia))
      .allocate(tmps.at("0479_aa_vo"))(tmps.at("0479_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0248_aa_oo")(ja, ia))
      .allocate(tmps.at("0478_aa_vo"))(tmps.at("0478_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0422_aa_oo")(ja, ia))
      .allocate(tmps.at("0476_aa_oo"))(tmps.at("0476_aa_oo")(ia, ja) =
                                         t1_1p.at("aa")(aa, ka) *
                                         tmps.at("0203_aaaa_oovo")(ia, ka, aa, ja))
      .allocate(tmps.at("0477_aa_vo"))(tmps.at("0477_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0476_aa_oo")(ja, ia))
      .allocate(tmps.at("0474_aa_vo"))(tmps.at("0474_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0241_aa_oo")(ja, ia))
      .allocate(tmps.at("0473_aa_vo"))(tmps.at("0473_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0234_aa_oo")(ja, ia))
      .allocate(tmps.at("0472_aa_vo"))(tmps.at("0472_aa_vo")(aa, ia) =
                                         t2_1p.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0224_aa_ov")(ja, ba))
      .allocate(tmps.at("0471_aa_vo"))(tmps.at("0471_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0344_aa_oo")(ja, ia))
      .allocate(tmps.at("0470_aa_vo"))(tmps.at("0470_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ia) * tmps.at("0041_aa_vv")(aa, ba))
      .allocate(tmps.at("0469_aa_vo"))(tmps.at("0469_aa_vo")(aa, ia) =
                                         t2_1p.at("abab")(aa, bb, ia, jb) *
                                         tmps.at("0237_bb_ov")(jb, bb))
      .allocate(tmps.at("0468_aa_vo"))(tmps.at("0468_aa_vo")(aa, ia) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0230_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0467_aa_vo"))(tmps.at("0467_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0390_aa_oo")(ja, ia))
      .allocate(tmps.at("0466_aa_vo"))(tmps.at("0466_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0348_aaaa_vovo")(aa, ja, ba, ia))
      .allocate(tmps.at("0465_aa_vo"))(tmps.at("0465_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ia) * tmps.at("0407_aa_vv")(aa, ba))
      .allocate(tmps.at("0464_aa_vo"))(tmps.at("0464_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ia) * tmps.at("0043_aa_vv")(aa, ba))
      .allocate(tmps.at("0463_aa_vo"))(tmps.at("0463_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0039_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0462_aa_vo"))(tmps.at("0462_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0037_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0461_aa_vo"))(tmps.at("0461_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0369_aa_oo")(ja, ia))
      .allocate(tmps.at("0460_aa_vo"))(tmps.at("0460_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ia) * tmps.at("0374_aa_vv")(aa, ba))
      .allocate(tmps.at("0459_aa_vo"))(tmps.at("0459_aa_vo")(aa, ia) =
                                         t2_1p.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0232_aa_ov")(ja, ba))
      .allocate(tmps.at("0458_aa_vo"))(tmps.at("0458_aa_vo")(aa, ia) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0376_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0456_aa_oo"))(tmps.at("0456_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovo")(ia, ka, aa, ja) *
                                         t1_1p.at("aa")(aa, ka))
      .allocate(tmps.at("0457_aa_vo"))(tmps.at("0457_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0456_aa_oo")(ja, ia))
      .allocate(tmps.at("0455_aa_vo"))(tmps.at("0455_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0360_aa_oo")(ja, ia))
      .allocate(tmps.at("0454_aa_vo"))(tmps.at("0454_aa_vo")(aa, ia) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0392_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0453_aa_vo"))(tmps.at("0453_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0394_aa_oo")(ja, ia))
      .allocate(tmps.at("0452_aa_vo"))(tmps.at("0452_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0243_aa_oo")(ja, ia))
      .allocate(tmps.at("0451_aa_vo"))(tmps.at("0451_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0363_aaaa_vovo")(aa, ja, ba, ia))
      .allocate(tmps.at("0450_aa_vo"))(tmps.at("0450_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0346_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0449_aa_vo"))(tmps.at("0449_aa_vo")(aa, ia) =
                                         t2.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0365_aa_ov")(ja, ba))
      .allocate(tmps.at("0447_aa_vo"))(tmps.at("0447_aa_vo")(aa, ia) =
                                         eri.at("abba_vovo")(aa, jb, bb, ia) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0446_aa_vo"))(tmps.at("0446_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovo")(aa, ja, ba, ia) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0445_aa_vo"))(tmps.at("0445_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovv")(aa, ja, ba, ca) *
                                         t2_1p.at("aaaa")(ba, ca, ia, ja))
      .allocate(tmps.at("0444_aa_vo"))(tmps.at("0444_aa_vo")(aa, ia) =
                                         f.at("aa_oo")(ja, ia) * t1_1p.at("aa")(aa, ja))
      .allocate(tmps.at("0443_aa_vo"))(tmps.at("0443_aa_vo")(aa, ia) =
                                         scalars.at("0013")() * t1_2p.at("aa")(aa, ia))
      .allocate(tmps.at("0442_aa_vo"))(tmps.at("0442_aa_vo")(aa, ia) =
                                         eri.at("aaaa_oovo")(ja, ka, ba, ia) *
                                         t2_1p.at("aaaa")(ba, aa, ka, ja))
      .allocate(tmps.at("0441_aa_vo"))(tmps.at("0441_aa_vo")(aa, ia) =
                                         eri.at("abba_oovo")(ja, kb, bb, ia) *
                                         t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0440_aa_vo"))(tmps.at("0440_aa_vo")(aa, ia) =
                                         scalars.at("0015")() * t1_2p.at("aa")(aa, ia))
      .allocate(tmps.at("0439_aa_vo"))(tmps.at("0439_aa_vo")(aa, ia) =
                                         scalars.at("0001")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0438_aa_vo"))(tmps.at("0438_aa_vo")(aa, ia) =
                                         eri.at("abab_vovv")(aa, jb, ba, cb) *
                                         t2_1p.at("abab")(ba, cb, ia, jb))
      .allocate(tmps.at("0437_aa_vo"))(tmps.at("0437_aa_vo")(aa, ia) =
                                         f.at("aa_vv")(aa, ba) * t1_1p.at("aa")(ba, ia))
      .allocate(tmps.at("0436_aa_vo"))(tmps.at("0436_aa_vo")(aa, ia) =
                                         scalars.at("0002")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0435_aa_vo"))(tmps.at("0435_aa_vo")(aa, ia) =
                                         f.at("aa_ov")(ja, ba) * t2_1p.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0434_aa_vo"))(tmps.at("0434_aa_vo")(aa, ia) =
                                         f.at("bb_ov")(jb, bb) * t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0448_aa_vo"))(tmps.at("0448_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0435_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) -= tmps.at("0444_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) -= tmps.at("0447_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) += tmps.at("0439_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) += 0.50 * tmps.at("0445_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) += tmps.at("0441_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) -= tmps.at("0446_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) += tmps.at("0436_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) += tmps.at("0434_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) += 2.00 * tmps.at("0440_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) += tmps.at("0437_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) += tmps.at("0438_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) += 0.50 * tmps.at("0442_aa_vo")(aa, ia))(
        tmps.at("0448_aa_vo")(aa, ia) += 2.00 * tmps.at("0443_aa_vo")(aa, ia))
      .deallocate(tmps.at("0447_aa_vo"))
      .deallocate(tmps.at("0446_aa_vo"))
      .deallocate(tmps.at("0445_aa_vo"))
      .deallocate(tmps.at("0444_aa_vo"))
      .deallocate(tmps.at("0443_aa_vo"))
      .deallocate(tmps.at("0442_aa_vo"))
      .deallocate(tmps.at("0441_aa_vo"))
      .deallocate(tmps.at("0440_aa_vo"))
      .deallocate(tmps.at("0439_aa_vo"))
      .deallocate(tmps.at("0438_aa_vo"))
      .deallocate(tmps.at("0437_aa_vo"))
      .deallocate(tmps.at("0436_aa_vo"))
      .deallocate(tmps.at("0435_aa_vo"))
      .deallocate(tmps.at("0434_aa_vo"))
      .allocate(tmps.at("0475_aa_vo"))(tmps.at("0475_aa_vo")(aa, ia) =
                                         -0.50 * tmps.at("0460_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= tmps.at("0457_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= tmps.at("0458_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0461_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0466_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= tmps.at("0463_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0472_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= tmps.at("0451_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= tmps.at("0462_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= 0.50 * tmps.at("0467_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0459_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0452_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += 0.50 * tmps.at("0465_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0473_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= tmps.at("0469_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0464_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= tmps.at("0454_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= tmps.at("0453_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0449_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += 0.50 * tmps.at("0455_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0474_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0450_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= tmps.at("0448_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) -= tmps.at("0468_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0470_aa_vo")(aa, ia))(
        tmps.at("0475_aa_vo")(aa, ia) += tmps.at("0471_aa_vo")(aa, ia))
      .deallocate(tmps.at("0474_aa_vo"))
      .deallocate(tmps.at("0473_aa_vo"))
      .deallocate(tmps.at("0472_aa_vo"))
      .deallocate(tmps.at("0471_aa_vo"))
      .deallocate(tmps.at("0470_aa_vo"))
      .deallocate(tmps.at("0469_aa_vo"))
      .deallocate(tmps.at("0468_aa_vo"))
      .deallocate(tmps.at("0467_aa_vo"))
      .deallocate(tmps.at("0466_aa_vo"))
      .deallocate(tmps.at("0465_aa_vo"))
      .deallocate(tmps.at("0464_aa_vo"))
      .deallocate(tmps.at("0463_aa_vo"))
      .deallocate(tmps.at("0462_aa_vo"))
      .deallocate(tmps.at("0461_aa_vo"))
      .deallocate(tmps.at("0460_aa_vo"))
      .deallocate(tmps.at("0459_aa_vo"))
      .deallocate(tmps.at("0458_aa_vo"))
      .deallocate(tmps.at("0457_aa_vo"))
      .deallocate(tmps.at("0455_aa_vo"))
      .deallocate(tmps.at("0454_aa_vo"))
      .deallocate(tmps.at("0453_aa_vo"))
      .deallocate(tmps.at("0452_aa_vo"))
      .deallocate(tmps.at("0451_aa_vo"))
      .deallocate(tmps.at("0450_aa_vo"))
      .deallocate(tmps.at("0449_aa_vo"))
      .deallocate(tmps.at("0448_aa_vo"))
      .allocate(tmps.at("0481_aa_vo"))(tmps.at("0481_aa_vo")(aa, ia) =
                                         tmps.at("0475_aa_vo")(aa, ia))(
        tmps.at("0481_aa_vo")(aa, ia) += tmps.at("0477_aa_vo")(aa, ia))(
        tmps.at("0481_aa_vo")(aa, ia) += tmps.at("0478_aa_vo")(aa, ia))(
        tmps.at("0481_aa_vo")(aa, ia) += tmps.at("0480_aa_vo")(aa, ia))(
        tmps.at("0481_aa_vo")(aa, ia) += tmps.at("0479_aa_vo")(aa, ia))
      .deallocate(tmps.at("0480_aa_vo"))
      .deallocate(tmps.at("0479_aa_vo"))
      .deallocate(tmps.at("0478_aa_vo"))
      .deallocate(tmps.at("0477_aa_vo"))
      .deallocate(tmps.at("0475_aa_vo"))(r1_1p.at("aa")(aa, ia) -= tmps.at("0481_aa_vo")(aa, ia))
      .deallocate(tmps.at("0481_aa_vo"))
      .allocate(tmps.at("0511_bb_vo"))(tmps.at("0511_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0289_bb_oo")(jb, ib))
      .allocate(tmps.at("0509_bb_vo"))(tmps.at("0509_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0281_bb_oo")(jb, ib))
      .allocate(tmps.at("0508_bb_vo"))(tmps.at("0508_bb_vo")(ab, ib) =
                                         t2.at("abab")(ba, ab, ja, ib) *
                                         tmps.at("0232_aa_ov")(ja, ba))
      .allocate(tmps.at("0507_bb_vo"))(tmps.at("0507_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0273_bb_oo")(jb, ib))
      .allocate(tmps.at("0505_bb_vv"))(tmps.at("0505_bb_vv")(ab, bb) =
                                         eri.at("abab_oovv")(ia, jb, ca, bb) *
                                         t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("0506_bb_vo"))(tmps.at("0506_bb_vo")(ab, ib) =
                                         t1.at("bb")(bb, ib) * tmps.at("0505_bb_vv")(ab, bb))
      .allocate(tmps.at("0482_bb_oo"))(tmps.at("0482_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovv")(ib, kb, ab, bb) *
                                         t2.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("0504_bb_vo"))(tmps.at("0504_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0482_bb_oo")(jb, ib))
      .deallocate(tmps.at("0482_bb_oo"))
      .allocate(tmps.at("0503_bb_vo"))(tmps.at("0503_bb_vo")(ab, ib) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0033_baba_voov")(ab, ja, ib, ba))
      .allocate(tmps.at("0502_bb_vo"))(tmps.at("0502_bb_vo")(ab, ib) =
                                         t2.at("bbbb")(bb, ab, ib, jb) *
                                         tmps.at("0237_bb_ov")(jb, bb))
      .allocate(tmps.at("0500_bb_vv"))(tmps.at("0500_bb_vv")(ab, bb) =
                                         eri.at("bbbb_oovv")(ib, jb, bb, cb) *
                                         t2.at("bbbb")(cb, ab, jb, ib))
      .allocate(tmps.at("0501_bb_vo"))(tmps.at("0501_bb_vo")(ab, ib) =
                                         t1.at("bb")(bb, ib) * tmps.at("0500_bb_vv")(ab, bb))
      .allocate(tmps.at("0499_bb_vo"))(tmps.at("0499_bb_vo")(ab, ib) =
                                         t2.at("abab")(ba, ab, ja, ib) *
                                         tmps.at("0224_aa_ov")(ja, ba))
      .allocate(tmps.at("0498_bb_vo"))(tmps.at("0498_bb_vo")(ab, ib) =
                                         t1.at("bb")(bb, ib) * tmps.at("0297_bb_vv")(ab, bb))
      .allocate(tmps.at("0497_bb_vo"))(tmps.at("0497_bb_vo")(ab, ib) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0135_baab_vovo")(ab, ja, ba, ib))
      .allocate(tmps.at("0496_bb_vo"))(tmps.at("0496_bb_vo")(ab, ib) =
                                         t1.at("bb")(ab, jb) * tmps.at("0271_bb_oo")(jb, ib))
      .allocate(tmps.at("0494_bb_vo"))(tmps.at("0494_bb_vo")(ab, ib) =
                                         eri.at("bbbb_vovo")(ab, jb, bb, ib) * t1.at("bb")(bb, jb))
      .allocate(tmps.at("0493_bb_vo"))(tmps.at("0493_bb_vo")(ab, ib) =
                                         eri.at("bbbb_oovo")(jb, kb, bb, ib) *
                                         t2.at("bbbb")(bb, ab, kb, jb))
      .allocate(tmps.at("0492_bb_vo"))(tmps.at("0492_bb_vo")(ab, ib) =
                                         scalars.at("0013")() * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("0491_bb_vo"))(tmps.at("0491_bb_vo")(ab, ib) =
                                         scalars.at("0015")() * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("0490_bb_vo"))(tmps.at("0490_bb_vo")(ab, ib) =
                                         eri.at("bbbb_vovv")(ab, jb, bb, cb) *
                                         t2.at("bbbb")(bb, cb, ib, jb))
      .allocate(tmps.at("0489_bb_vo"))(tmps.at("0489_bb_vo")(ab, ib) =
                                         eri.at("baab_vovo")(ab, ja, ba, ib) * t1.at("aa")(ba, ja))
      .allocate(tmps.at("0488_bb_vo"))(tmps.at("0488_bb_vo")(ab, ib) =
                                         eri.at("baab_vovv")(ab, ja, ba, cb) *
                                         t2.at("abab")(ba, cb, ja, ib))
      .allocate(tmps.at("0487_bb_vo"))(tmps.at("0487_bb_vo")(ab, ib) =
                                         f.at("bb_vv")(ab, bb) * t1.at("bb")(bb, ib))
      .allocate(tmps.at("0486_bb_vo"))(tmps.at("0486_bb_vo")(ab, ib) =
                                         eri.at("abab_oovo")(ja, kb, ba, ib) *
                                         t2.at("abab")(ba, ab, ja, kb))
      .allocate(tmps.at("0485_bb_vo"))(tmps.at("0485_bb_vo")(ab, ib) =
                                         f.at("bb_oo")(jb, ib) * t1.at("bb")(ab, jb))
      .allocate(tmps.at("0484_bb_vo"))(tmps.at("0484_bb_vo")(ab, ib) =
                                         f.at("aa_ov")(ja, ba) * t2.at("abab")(ba, ab, ja, ib))
      .allocate(tmps.at("0483_bb_vo"))(tmps.at("0483_bb_vo")(ab, ib) =
                                         f.at("bb_ov")(jb, bb) * t2.at("bbbb")(bb, ab, ib, jb))
      .allocate(tmps.at("0495_bb_vo"))(tmps.at("0495_bb_vo")(ab, ib) =
                                         -2.00 * tmps.at("0483_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) += 2.00 * tmps.at("0491_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) += 2.00 * tmps.at("0492_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) += tmps.at("0493_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) -= 2.00 * tmps.at("0489_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) -= 2.00 * tmps.at("0494_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) -= 2.00 * tmps.at("0488_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) -= 2.00 * tmps.at("0485_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) += tmps.at("0490_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) += 2.00 * tmps.at("0487_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) -= 2.00 * tmps.at("0486_bb_vo")(ab, ib))(
        tmps.at("0495_bb_vo")(ab, ib) += 2.00 * tmps.at("0484_bb_vo")(ab, ib))
      .deallocate(tmps.at("0494_bb_vo"))
      .deallocate(tmps.at("0493_bb_vo"))
      .deallocate(tmps.at("0492_bb_vo"))
      .deallocate(tmps.at("0491_bb_vo"))
      .deallocate(tmps.at("0490_bb_vo"))
      .deallocate(tmps.at("0489_bb_vo"))
      .deallocate(tmps.at("0488_bb_vo"))
      .deallocate(tmps.at("0487_bb_vo"))
      .deallocate(tmps.at("0486_bb_vo"))
      .deallocate(tmps.at("0485_bb_vo"))
      .deallocate(tmps.at("0484_bb_vo"))
      .deallocate(tmps.at("0483_bb_vo"));
  }
}

template void exachem::cc::qed_ccsd_os::resid_2<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);