/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_cs_resid_2.hpp"

template<typename T>
void exachem::cc::qed_ccsd_cs::resid_part2(
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
      .allocate(tmps.at("0276_aa_vo"))(tmps.at("0276_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ia) * tmps.at("0275_aa_vv")(aa, ba))
      .allocate(tmps.at("0273_aaaa_voov"))(tmps.at("0273_aaaa_voov")(aa, ia, ja, ba) =
                                             eri.at("abab_oovv")(ia, kb, ba, cb) *
                                             t2_1p.at("abab")(aa, cb, ja, kb))
      .allocate(tmps.at("0274_aa_vo"))(tmps.at("0274_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0273_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0271_aa_vv"))(tmps.at("0271_aa_vv")(aa, ba) =
                                         eri.at("abab_oovv")(ia, jb, ba, cb) *
                                         t2_1p.at("abab")(aa, cb, ia, jb))
      .allocate(tmps.at("0272_aa_vo"))(tmps.at("0272_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ia) * tmps.at("0271_aa_vv")(aa, ba))
      .allocate(tmps.at("0269_aa_oo"))(tmps.at("0269_aa_oo")(ia, ja) =
                                         eri.at("abab_oovv")(ia, kb, aa, bb) *
                                         t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0270_aa_vo"))(tmps.at("0270_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0269_aa_oo")(ja, ia))
      .allocate(tmps.at("0044_aa_oo"))(tmps.at("0044_aa_oo")(ia, ja) =
                                         dp.at("aa_ov")(ia, aa) * t1_2p.at("aa")(aa, ja))
      .allocate(tmps.at("0268_aa_vo"))(tmps.at("0268_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0044_aa_oo")(ja, ia))
      .allocate(tmps.at("0267_aa_vo"))(tmps.at("0267_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ia) * tmps.at("0104_aa_vv")(aa, ba))
      .allocate(tmps.at("0266_aa_vo"))(tmps.at("0266_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ja) *
                                         tmps.at("0132_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0264_aaaa_voov"))(tmps.at("0264_aaaa_voov")(aa, ia, ja, ba) =
                                             eri.at("aaaa_oovv")(ia, ka, ba, ca) *
                                             t2_1p.at("aaaa")(ca, aa, ja, ka))
      .allocate(tmps.at("0265_aa_vo"))(tmps.at("0265_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0264_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0262_abba_vovo"))(tmps.at("0262_abba_vovo")(aa, ib, bb, ja) =
                                             eri.at("abab_vovv")(aa, ib, ca, bb) *
                                             t1_1p.at("aa")(ca, ja))
      .allocate(tmps.at("0263_aa_vo"))(tmps.at("0263_aa_vo")(aa, ia) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0262_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0261_aa_vo"))(tmps.at("0261_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ia) * tmps.at("0110_aa_vv")(aa, ba))
      .allocate(tmps.at("0260_aa_vo"))(tmps.at("0260_aa_vo")(aa, ia) =
                                         t2_2p.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0214_aa_ov")(ja, ba))
      .allocate(tmps.at("0259_aa_vo"))(tmps.at("0259_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ia) * tmps.at("0161_aa_vv")(aa, ba))
      .allocate(tmps.at("0257_abab_voov"))(tmps.at("0257_abab_voov")(aa, ib, ja, bb) =
                                             eri.at("abab_oovv")(ka, ib, ca, bb) *
                                             t2_1p.at("aaaa")(ca, aa, ja, ka))
      .allocate(tmps.at("0258_aa_vo"))(tmps.at("0258_aa_vo")(aa, ia) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0257_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0255_bb_ov"))(tmps.at("0255_bb_ov")(ib, ab) =
                                         eri.at("abab_oovv")(ja, ib, ba, ab) * t1.at("aa")(ba, ja))
      .allocate(tmps.at("0256_aa_vo"))(tmps.at("0256_aa_vo")(aa, ia) =
                                         t2_2p.at("abab")(aa, bb, ia, jb) *
                                         tmps.at("0255_bb_ov")(jb, bb))
      .allocate(tmps.at("0254_aa_vo"))(tmps.at("0254_aa_vo")(aa, ia) =
                                         t2_2p.at("abab")(aa, bb, ja, kb) *
                                         tmps.at("0213_abba_oovo")(ja, kb, bb, ia))
      .allocate(tmps.at("0043_aa_oo"))(tmps.at("0043_aa_oo")(ia, ja) =
                                         dp.at("aa_ov")(ia, aa) * t1_1p.at("aa")(aa, ja))
      .allocate(tmps.at("0253_aa_vo"))(tmps.at("0253_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0043_aa_oo")(ja, ia))
      .allocate(tmps.at("0251_aa_vv"))(tmps.at("0251_aa_vv")(aa, ba) =
                                         eri.at("aaaa_oovv")(ia, ja, ba, ca) *
                                         t2_1p.at("aaaa")(ca, aa, ja, ia))
      .allocate(tmps.at("0252_aa_vo"))(tmps.at("0252_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ia) * tmps.at("0251_aa_vv")(aa, ba))
      .allocate(tmps.at("0249_aa_oo"))(tmps.at("0249_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovv")(ia, ka, aa, ba) *
                                         t2_1p.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0250_aa_vo"))(tmps.at("0250_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0249_aa_oo")(ja, ia))
      .allocate(tmps.at("0248_aa_vo"))(tmps.at("0248_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0116_aa_oo")(ja, ia))
      .allocate(tmps.at("0247_aa_vo"))(tmps.at("0247_aa_vo")(aa, ia) =
                                         t1_2p.at("bb")(bb, jb) *
                                         tmps.at("0112_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0246_aa_vo"))(tmps.at("0246_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0102_aa_oo")(ja, ia))
      .allocate(tmps.at("0244_aa_oo"))(tmps.at("0244_aa_oo")(ia, ja) =
                                         f.at("aa_ov")(ia, aa) * t1_1p.at("aa")(aa, ja))
      .allocate(tmps.at("0245_aa_vo"))(tmps.at("0245_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0244_aa_oo")(ja, ia))
      .allocate(tmps.at("0243_aa_vo"))(tmps.at("0243_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(ba, ia) * tmps.at("0100_aa_vv")(aa, ba))
      .allocate(tmps.at("0242_aa_vo"))(tmps.at("0242_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0087_aa_oo")(ja, ia))
      .allocate(tmps.at("0240_aa_vo"))(tmps.at("0240_aa_vo")(aa, ia) =
                                         eri.at("aaaa_oovo")(ja, ka, ba, ia) *
                                         t2_2p.at("aaaa")(ba, aa, ka, ja))
      .allocate(tmps.at("0239_aa_vo"))(tmps.at("0239_aa_vo")(aa, ia) =
                                         f.at("bb_ov")(jb, bb) * t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0238_aa_vo"))(tmps.at("0238_aa_vo")(aa, ia) =
                                         scalars.at("0002")() * t1_2p.at("aa")(aa, ia))
      .allocate(tmps.at("0237_aa_vo"))(tmps.at("0237_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovo")(aa, ja, ba, ia) *
                                         t1_2p.at("aa")(ba, ja))
      .allocate(tmps.at("0236_aa_vo"))(tmps.at("0236_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovv")(aa, ja, ba, ca) *
                                         t2_2p.at("aaaa")(ba, ca, ia, ja))
      .allocate(tmps.at("0235_aa_vo"))(tmps.at("0235_aa_vo")(aa, ia) =
                                         scalars.at("0001")() * t1_2p.at("aa")(aa, ia))
      .allocate(tmps.at("0234_aa_vo"))(tmps.at("0234_aa_vo")(aa, ia) =
                                         scalars.at("0014")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0233_aa_vo"))(tmps.at("0233_aa_vo")(aa, ia) =
                                         f.at("aa_oo")(ja, ia) * t1_2p.at("aa")(aa, ja))
      .allocate(tmps.at("0232_aa_vo"))(tmps.at("0232_aa_vo")(aa, ia) =
                                         scalars.at("0016")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0231_aa_vo"))(tmps.at("0231_aa_vo")(aa, ia) =
                                         eri.at("abba_oovo")(ja, kb, bb, ia) *
                                         t2_2p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0230_aa_vo"))(tmps.at("0230_aa_vo")(aa, ia) =
                                         f.at("aa_ov")(ja, ba) * t2_2p.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0229_aa_vo"))(tmps.at("0229_aa_vo")(aa, ia) =
                                         f.at("aa_vv")(aa, ba) * t1_2p.at("aa")(ba, ia))
      .allocate(tmps.at("0228_aa_vo"))(tmps.at("0228_aa_vo")(aa, ia) =
                                         eri.at("abab_vovv")(aa, jb, ba, cb) *
                                         t2_2p.at("abab")(ba, cb, ia, jb))
      .allocate(tmps.at("0227_aa_vo"))(tmps.at("0227_aa_vo")(aa, ia) =
                                         eri.at("abba_vovo")(aa, jb, bb, ia) *
                                         t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("0241_aa_vo"))(tmps.at("0241_aa_vo")(aa, ia) =
                                         -0.50 * tmps.at("0236_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) -= tmps.at("0234_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) -= tmps.at("0239_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) += tmps.at("0230_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) -= 2.00 * tmps.at("0238_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) -= 2.00 * tmps.at("0235_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) -= tmps.at("0229_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) -= tmps.at("0228_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) += tmps.at("0233_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) -= tmps.at("0232_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) -= tmps.at("0231_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) += tmps.at("0227_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) -= 0.50 * tmps.at("0240_aa_vo")(aa, ia))(
        tmps.at("0241_aa_vo")(aa, ia) += tmps.at("0237_aa_vo")(aa, ia))
      .deallocate(tmps.at("0240_aa_vo"))
      .deallocate(tmps.at("0239_aa_vo"))
      .deallocate(tmps.at("0238_aa_vo"))
      .deallocate(tmps.at("0237_aa_vo"))
      .deallocate(tmps.at("0236_aa_vo"))
      .deallocate(tmps.at("0235_aa_vo"))
      .deallocate(tmps.at("0234_aa_vo"))
      .deallocate(tmps.at("0233_aa_vo"))
      .deallocate(tmps.at("0232_aa_vo"))
      .deallocate(tmps.at("0231_aa_vo"))
      .deallocate(tmps.at("0230_aa_vo"))
      .deallocate(tmps.at("0229_aa_vo"))
      .deallocate(tmps.at("0228_aa_vo"))
      .deallocate(tmps.at("0227_aa_vo"))
      .allocate(tmps.at("0300_aa_vo"))(tmps.at("0300_aa_vo")(aa, ia) =
                                         -0.50 * tmps.at("0242_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= 0.50 * tmps.at("0243_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0278_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += 3.00 * tmps.at("0268_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0258_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0254_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0299_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += 3.00 * tmps.at("0253_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0247_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0276_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0248_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += 0.50 * tmps.at("0250_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += 0.50 * tmps.at("0252_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0274_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0286_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0259_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0256_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0270_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0261_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0267_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0265_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0245_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0298_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0246_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += 0.50 * tmps.at("0289_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0283_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0260_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0297_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0241_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0281_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0285_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0279_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0263_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0272_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0266_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) += tmps.at("0290_aa_vo")(aa, ia))(
        tmps.at("0300_aa_vo")(aa, ia) -= tmps.at("0288_aa_vo")(aa, ia))
      .deallocate(tmps.at("0299_aa_vo"))
      .deallocate(tmps.at("0298_aa_vo"))
      .deallocate(tmps.at("0297_aa_vo"))
      .deallocate(tmps.at("0290_aa_vo"))
      .deallocate(tmps.at("0289_aa_vo"))
      .deallocate(tmps.at("0288_aa_vo"))
      .deallocate(tmps.at("0286_aa_vo"))
      .deallocate(tmps.at("0285_aa_vo"))
      .deallocate(tmps.at("0283_aa_vo"))
      .deallocate(tmps.at("0281_aa_vo"))
      .deallocate(tmps.at("0279_aa_vo"))
      .deallocate(tmps.at("0278_aa_vo"))
      .deallocate(tmps.at("0276_aa_vo"))
      .deallocate(tmps.at("0274_aa_vo"))
      .deallocate(tmps.at("0272_aa_vo"))
      .deallocate(tmps.at("0270_aa_vo"))
      .deallocate(tmps.at("0268_aa_vo"))
      .deallocate(tmps.at("0267_aa_vo"))
      .deallocate(tmps.at("0266_aa_vo"))
      .deallocate(tmps.at("0265_aa_vo"))
      .deallocate(tmps.at("0263_aa_vo"))
      .deallocate(tmps.at("0261_aa_vo"))
      .deallocate(tmps.at("0260_aa_vo"))
      .deallocate(tmps.at("0259_aa_vo"))
      .deallocate(tmps.at("0258_aa_vo"))
      .deallocate(tmps.at("0256_aa_vo"))
      .deallocate(tmps.at("0254_aa_vo"))
      .deallocate(tmps.at("0253_aa_vo"))
      .deallocate(tmps.at("0252_aa_vo"))
      .deallocate(tmps.at("0250_aa_vo"))
      .deallocate(tmps.at("0248_aa_vo"))
      .deallocate(tmps.at("0247_aa_vo"))
      .deallocate(tmps.at("0246_aa_vo"))
      .deallocate(tmps.at("0245_aa_vo"))
      .deallocate(tmps.at("0243_aa_vo"))
      .deallocate(tmps.at("0242_aa_vo"))
      .deallocate(tmps.at("0241_aa_vo"))
      .allocate(tmps.at("0322_aa_vo"))(tmps.at("0322_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0304_aa_vo")(aa, ia))(
        tmps.at("0322_aa_vo")(aa, ia) += tmps.at("0300_aa_vo")(aa, ia))(
        tmps.at("0322_aa_vo")(aa, ia) += tmps.at("0302_aa_vo")(aa, ia))(
        tmps.at("0322_aa_vo")(aa, ia) += tmps.at("0309_aa_vo")(aa, ia))(
        tmps.at("0322_aa_vo")(aa, ia) += tmps.at("0321_aa_vo")(aa, ia))(
        tmps.at("0322_aa_vo")(aa, ia) += tmps.at("0314_aa_vo")(aa, ia))(
        tmps.at("0322_aa_vo")(aa, ia) += tmps.at("0305_aa_vo")(aa, ia))
      .deallocate(tmps.at("0321_aa_vo"))
      .deallocate(tmps.at("0314_aa_vo"))
      .deallocate(tmps.at("0309_aa_vo"))
      .deallocate(tmps.at("0305_aa_vo"))
      .deallocate(tmps.at("0304_aa_vo"))
      .deallocate(tmps.at("0302_aa_vo"))
      .deallocate(tmps.at("0300_aa_vo"))(r1_2p.at("aa")(aa, ia) -=
                                         2.00 * tmps.at("0322_aa_vo")(aa, ia))
      .deallocate(tmps.at("0322_aa_vo"))
      .allocate(tmps.at("0369_aa_vo"))(tmps.at("0369_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0313_aa_oo")(ja, ia))
      .allocate(tmps.at("0368_aa_vo"))(tmps.at("0368_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0217_aa_oo")(ja, ia))
      .allocate(tmps.at("0366_aa_oo"))(tmps.at("0366_aa_oo")(ia, ja) =
                                         t1_1p.at("aa")(aa, ka) *
                                         tmps.at("0081_aaaa_oovo")(ia, ka, aa, ja))
      .allocate(tmps.at("0367_aa_vo"))(tmps.at("0367_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0366_aa_oo")(ja, ia))
      .allocate(tmps.at("0365_aa_vo"))(tmps.at("0365_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0301_aa_oo")(ja, ia))
      .allocate(tmps.at("0363_aa_vo"))(tmps.at("0363_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0124_aa_oo")(ja, ia))
      .allocate(tmps.at("0362_aa_vo"))(tmps.at("0362_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ia) * tmps.at("0271_aa_vv")(aa, ba))
      .allocate(tmps.at("0361_aa_vo"))(tmps.at("0361_aa_vo")(aa, ia) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0204_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0360_aa_vo"))(tmps.at("0360_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ia) * tmps.at("0104_aa_vv")(aa, ba))
      .allocate(tmps.at("0359_aa_vo"))(tmps.at("0359_aa_vo")(aa, ia) =
                                         t2_1p.at("aaaa")(ba, aa, ja, ka) *
                                         tmps.at("0081_aaaa_oovo")(ka, ja, ba, ia))
      .allocate(tmps.at("0358_aa_vo"))(tmps.at("0358_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0132_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0357_aa_vo"))(tmps.at("0357_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0277_aa_oo")(ja, ia))
      .allocate(tmps.at("0356_aa_vo"))(tmps.at("0356_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0098_aaaa_vovo")(aa, ja, ba, ia))
      .allocate(tmps.at("0355_aa_vo"))(tmps.at("0355_aa_vo")(aa, ia) =
                                         t2_1p.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0214_aa_ov")(ja, ba))
      .allocate(tmps.at("0354_aa_vo"))(tmps.at("0354_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ja) *
                                         tmps.at("0163_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0353_aa_vo"))(tmps.at("0353_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0244_aa_oo")(ja, ia))
      .allocate(tmps.at("0352_aa_vo"))(tmps.at("0352_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0269_aa_oo")(ja, ia))
      .allocate(tmps.at("0351_aa_vo"))(tmps.at("0351_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0249_aa_oo")(ja, ia))
      .allocate(tmps.at("0350_aa_vo"))(tmps.at("0350_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0116_aa_oo")(ja, ia))
      .allocate(tmps.at("0349_aa_vo"))(tmps.at("0349_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0273_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0348_aa_vo"))(tmps.at("0348_aa_vo")(aa, ia) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0106_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0347_aa_vo"))(tmps.at("0347_aa_vo")(aa, ia) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0257_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0345_aa_oo"))(tmps.at("0345_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovo")(ia, ka, aa, ja) *
                                         t1_1p.at("aa")(aa, ka))
      .allocate(tmps.at("0346_aa_vo"))(tmps.at("0346_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0345_aa_oo")(ja, ia))
      .allocate(tmps.at("0344_aa_vo"))(tmps.at("0344_aa_vo")(aa, ia) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0280_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0343_aa_vo"))(tmps.at("0343_aa_vo")(aa, ia) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0262_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0342_aa_vo"))(tmps.at("0342_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0102_aa_oo")(ja, ia))
      .allocate(tmps.at("0341_aa_vo"))(tmps.at("0341_aa_vo")(aa, ia) =
                                         t1_1p.at("bb")(bb, jb) *
                                         tmps.at("0112_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0340_aa_vo"))(tmps.at("0340_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ia) * tmps.at("0110_aa_vv")(aa, ba))
      .allocate(tmps.at("0339_aa_vo"))(tmps.at("0339_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0087_aa_oo")(ja, ia))
      .allocate(tmps.at("0338_aa_vo"))(tmps.at("0338_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(ba, ia) * tmps.at("0100_aa_vv")(aa, ba))
      .allocate(tmps.at("0336_aa_vo"))(tmps.at("0336_aa_vo")(aa, ia) =
                                         eri.at("abba_vovo")(aa, jb, bb, ia) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0335_aa_vo"))(tmps.at("0335_aa_vo")(aa, ia) =
                                         scalars.at("0013")() * t1_2p.at("aa")(aa, ia))
      .allocate(tmps.at("0334_aa_vo"))(tmps.at("0334_aa_vo")(aa, ia) =
                                         eri.at("abab_vovv")(aa, jb, ba, cb) *
                                         t2_1p.at("abab")(ba, cb, ia, jb))
      .allocate(tmps.at("0333_aa_vo"))(tmps.at("0333_aa_vo")(aa, ia) =
                                         scalars.at("0001")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0332_aa_vo"))(tmps.at("0332_aa_vo")(aa, ia) =
                                         scalars.at("0015")() * t1_2p.at("aa")(aa, ia))
      .allocate(tmps.at("0331_aa_vo"))(tmps.at("0331_aa_vo")(aa, ia) =
                                         scalars.at("0002")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0330_aa_vo"))(tmps.at("0330_aa_vo")(aa, ia) =
                                         eri.at("aaaa_oovo")(ja, ka, ba, ia) *
                                         t2_1p.at("aaaa")(ba, aa, ka, ja))
      .allocate(tmps.at("0329_aa_vo"))(tmps.at("0329_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovv")(aa, ja, ba, ca) *
                                         t2_1p.at("aaaa")(ba, ca, ia, ja))
      .allocate(tmps.at("0328_aa_vo"))(tmps.at("0328_aa_vo")(aa, ia) =
                                         eri.at("abba_oovo")(ja, kb, bb, ia) *
                                         t2_1p.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0327_aa_vo"))(tmps.at("0327_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovo")(aa, ja, ba, ia) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0326_aa_vo"))(tmps.at("0326_aa_vo")(aa, ia) =
                                         f.at("aa_vv")(aa, ba) * t1_1p.at("aa")(ba, ia))
      .allocate(tmps.at("0325_aa_vo"))(tmps.at("0325_aa_vo")(aa, ia) =
                                         f.at("aa_oo")(ja, ia) * t1_1p.at("aa")(aa, ja))
      .allocate(tmps.at("0324_aa_vo"))(tmps.at("0324_aa_vo")(aa, ia) =
                                         f.at("aa_ov")(ja, ba) * t2_1p.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0323_aa_vo"))(tmps.at("0323_aa_vo")(aa, ia) =
                                         f.at("bb_ov")(jb, bb) * t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0337_aa_vo"))(tmps.at("0337_aa_vo")(aa, ia) =
                                         -2.00 * tmps.at("0324_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) -= 2.00 * tmps.at("0325_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) += tmps.at("0329_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) += 2.00 * tmps.at("0331_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) += tmps.at("0330_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) += 4.00 * tmps.at("0335_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) += 2.00 * tmps.at("0334_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) += 2.00 * tmps.at("0326_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) -= 2.00 * tmps.at("0327_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) += 2.00 * tmps.at("0328_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) -= 2.00 * tmps.at("0336_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) += 4.00 * tmps.at("0332_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) += 2.00 * tmps.at("0333_aa_vo")(aa, ia))(
        tmps.at("0337_aa_vo")(aa, ia) += 2.00 * tmps.at("0323_aa_vo")(aa, ia))
      .deallocate(tmps.at("0336_aa_vo"))
      .deallocate(tmps.at("0335_aa_vo"))
      .deallocate(tmps.at("0334_aa_vo"))
      .deallocate(tmps.at("0333_aa_vo"))
      .deallocate(tmps.at("0332_aa_vo"))
      .deallocate(tmps.at("0331_aa_vo"))
      .deallocate(tmps.at("0330_aa_vo"))
      .deallocate(tmps.at("0329_aa_vo"))
      .deallocate(tmps.at("0328_aa_vo"))
      .deallocate(tmps.at("0327_aa_vo"))
      .deallocate(tmps.at("0326_aa_vo"))
      .deallocate(tmps.at("0325_aa_vo"))
      .deallocate(tmps.at("0324_aa_vo"))
      .deallocate(tmps.at("0323_aa_vo"))
      .allocate(tmps.at("0364_aa_vo"))(tmps.at("0364_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0351_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= tmps.at("0359_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0350_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += tmps.at("0339_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += 2.00 * tmps.at("0348_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0342_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0340_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += 2.00 * tmps.at("0346_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0353_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0347_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += 2.00 * tmps.at("0357_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0360_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0361_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0355_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += 2.00 * tmps.at("0349_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += tmps.at("0338_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0354_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += 2.00 * tmps.at("0363_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += 2.00 * tmps.at("0341_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += tmps.at("0337_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += 2.00 * tmps.at("0343_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0352_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += 2.00 * tmps.at("0356_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += 2.00 * tmps.at("0358_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) += 2.00 * tmps.at("0344_aa_vo")(aa, ia))(
        tmps.at("0364_aa_vo")(aa, ia) -= 2.00 * tmps.at("0362_aa_vo")(aa, ia))
      .deallocate(tmps.at("0363_aa_vo"))
      .deallocate(tmps.at("0362_aa_vo"))
      .deallocate(tmps.at("0361_aa_vo"))
      .deallocate(tmps.at("0360_aa_vo"))
      .deallocate(tmps.at("0359_aa_vo"))
      .deallocate(tmps.at("0358_aa_vo"))
      .deallocate(tmps.at("0357_aa_vo"))
      .deallocate(tmps.at("0356_aa_vo"))
      .deallocate(tmps.at("0355_aa_vo"))
      .deallocate(tmps.at("0354_aa_vo"))
      .deallocate(tmps.at("0353_aa_vo"))
      .deallocate(tmps.at("0352_aa_vo"))
      .deallocate(tmps.at("0351_aa_vo"))
      .deallocate(tmps.at("0350_aa_vo"))
      .deallocate(tmps.at("0349_aa_vo"))
      .deallocate(tmps.at("0348_aa_vo"))
      .deallocate(tmps.at("0347_aa_vo"))
      .deallocate(tmps.at("0346_aa_vo"))
      .deallocate(tmps.at("0344_aa_vo"))
      .deallocate(tmps.at("0343_aa_vo"))
      .deallocate(tmps.at("0342_aa_vo"))
      .deallocate(tmps.at("0341_aa_vo"))
      .deallocate(tmps.at("0340_aa_vo"))
      .deallocate(tmps.at("0339_aa_vo"))
      .deallocate(tmps.at("0338_aa_vo"))
      .deallocate(tmps.at("0337_aa_vo"))
      .allocate(tmps.at("0370_aa_vo"))(tmps.at("0370_aa_vo")(aa, ia) =
                                         -0.50 * tmps.at("0364_aa_vo")(aa, ia))(
        tmps.at("0370_aa_vo")(aa, ia) += tmps.at("0365_aa_vo")(aa, ia))(
        tmps.at("0370_aa_vo")(aa, ia) += tmps.at("0367_aa_vo")(aa, ia))(
        tmps.at("0370_aa_vo")(aa, ia) += tmps.at("0369_aa_vo")(aa, ia))(
        tmps.at("0370_aa_vo")(aa, ia) += tmps.at("0368_aa_vo")(aa, ia))
      .deallocate(tmps.at("0369_aa_vo"))
      .deallocate(tmps.at("0368_aa_vo"))
      .deallocate(tmps.at("0367_aa_vo"))
      .deallocate(tmps.at("0365_aa_vo"))
      .deallocate(tmps.at("0364_aa_vo"))(r1_1p.at("aa")(aa, ia) -= tmps.at("0370_aa_vo")(aa, ia))
      .deallocate(tmps.at("0370_aa_vo"))
      .allocate(tmps.at("0398_aa_vo"))(tmps.at("0398_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0217_aa_oo")(ja, ia))
      .allocate(tmps.at("0396_aa_vo"))(tmps.at("0396_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0124_aa_oo")(ja, ia))
      .allocate(tmps.at("0395_aa_vo"))(tmps.at("0395_aa_vo")(aa, ia) =
                                         t2.at("aaaa")(ba, aa, ia, ja) *
                                         tmps.at("0214_aa_ov")(ja, ba))
      .allocate(tmps.at("0394_aa_vo"))(tmps.at("0394_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ia) * tmps.at("0110_aa_vv")(aa, ba))
      .allocate(tmps.at("0393_aa_vo"))(tmps.at("0393_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ja) *
                                         tmps.at("0132_aaaa_voov")(aa, ja, ia, ba))
      .allocate(tmps.at("0371_aa_oo"))(tmps.at("0371_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovv")(ia, ka, aa, ba) *
                                         t2.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("0392_aa_vo"))(tmps.at("0392_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0371_aa_oo")(ja, ia))
      .deallocate(tmps.at("0371_aa_oo"))
      .allocate(tmps.at("0391_aa_vo"))(tmps.at("0391_aa_vo")(aa, ia) =
                                         t1.at("aa")(ba, ia) * tmps.at("0104_aa_vv")(aa, ba))
      .allocate(tmps.at("0390_aa_vo"))(tmps.at("0390_aa_vo")(aa, ia) =
                                         t2.at("abab")(aa, bb, ia, jb) *
                                         tmps.at("0198_bb_ov")(jb, bb))
      .allocate(tmps.at("0389_aa_vo"))(tmps.at("0389_aa_vo")(aa, ia) =
                                         t2.at("aaaa")(ba, aa, ja, ka) *
                                         tmps.at("0081_aaaa_oovo")(ka, ja, ba, ia))
      .allocate(tmps.at("0388_aa_vo"))(tmps.at("0388_aa_vo")(aa, ia) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0204_abab_voov")(aa, jb, ia, bb))
      .allocate(tmps.at("0387_aa_vo"))(tmps.at("0387_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0116_aa_oo")(ja, ia))
      .allocate(tmps.at("0386_aa_vo"))(tmps.at("0386_aa_vo")(aa, ia) =
                                         t1.at("bb")(bb, jb) *
                                         tmps.at("0106_abba_vovo")(aa, jb, bb, ia))
      .allocate(tmps.at("0385_aa_vo"))(tmps.at("0385_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0102_aa_oo")(ja, ia))
      .allocate(tmps.at("0383_aa_vo"))(tmps.at("0383_aa_vo")(aa, ia) =
                                         f.at("aa_ov")(ja, ba) * t2.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0382_aa_vo"))(tmps.at("0382_aa_vo")(aa, ia) =
                                         scalars.at("0013")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0381_aa_vo"))(tmps.at("0381_aa_vo")(aa, ia) =
                                         eri.at("aaaa_oovo")(ja, ka, ba, ia) *
                                         t2.at("aaaa")(ba, aa, ka, ja))
      .allocate(tmps.at("0380_aa_vo"))(tmps.at("0380_aa_vo")(aa, ia) =
                                         scalars.at("0015")() * t1_1p.at("aa")(aa, ia))
      .allocate(tmps.at("0379_aa_vo"))(tmps.at("0379_aa_vo")(aa, ia) =
                                         eri.at("abab_vovv")(aa, jb, ba, cb) *
                                         t2.at("abab")(ba, cb, ia, jb))
      .allocate(tmps.at("0378_aa_vo"))(tmps.at("0378_aa_vo")(aa, ia) =
                                         eri.at("abba_oovo")(ja, kb, bb, ia) *
                                         t2.at("abab")(aa, bb, ja, kb))
      .allocate(tmps.at("0377_aa_vo"))(tmps.at("0377_aa_vo")(aa, ia) =
                                         f.at("bb_ov")(jb, bb) * t2.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0376_aa_vo"))(tmps.at("0376_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovv")(aa, ja, ba, ca) *
                                         t2.at("aaaa")(ba, ca, ia, ja))
      .allocate(tmps.at("0375_aa_vo"))(tmps.at("0375_aa_vo")(aa, ia) =
                                         eri.at("aaaa_vovo")(aa, ja, ba, ia) * t1.at("aa")(ba, ja))
      .allocate(tmps.at("0374_aa_vo"))(tmps.at("0374_aa_vo")(aa, ia) =
                                         eri.at("abba_vovo")(aa, jb, bb, ia) * t1.at("bb")(bb, jb))
      .allocate(tmps.at("0373_aa_vo"))(tmps.at("0373_aa_vo")(aa, ia) =
                                         f.at("aa_oo")(ja, ia) * t1.at("aa")(aa, ja))
      .allocate(tmps.at("0372_aa_vo"))(tmps.at("0372_aa_vo")(aa, ia) =
                                         f.at("aa_vv")(aa, ba) * t1.at("aa")(ba, ia))
      .allocate(tmps.at("0384_aa_vo"))(tmps.at("0384_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0373_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) -= tmps.at("0374_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) += tmps.at("0372_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) += 0.50 * tmps.at("0381_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) += tmps.at("0379_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) += tmps.at("0380_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) -= tmps.at("0383_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) += tmps.at("0378_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) -= tmps.at("0375_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) += tmps.at("0382_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) += tmps.at("0377_aa_vo")(aa, ia))(
        tmps.at("0384_aa_vo")(aa, ia) += 0.50 * tmps.at("0376_aa_vo")(aa, ia))
      .deallocate(tmps.at("0383_aa_vo"))
      .deallocate(tmps.at("0382_aa_vo"))
      .deallocate(tmps.at("0381_aa_vo"))
      .deallocate(tmps.at("0380_aa_vo"))
      .deallocate(tmps.at("0379_aa_vo"))
      .deallocate(tmps.at("0378_aa_vo"))
      .deallocate(tmps.at("0377_aa_vo"))
      .deallocate(tmps.at("0376_aa_vo"))
      .deallocate(tmps.at("0375_aa_vo"))
      .deallocate(tmps.at("0374_aa_vo"))
      .deallocate(tmps.at("0373_aa_vo"))
      .deallocate(tmps.at("0372_aa_vo"))
      .allocate(tmps.at("0397_aa_vo"))(tmps.at("0397_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0384_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) -= tmps.at("0386_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) -= tmps.at("0390_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) += 0.50 * tmps.at("0389_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) += tmps.at("0385_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) += tmps.at("0394_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) += tmps.at("0387_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) += tmps.at("0395_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) += 0.50 * tmps.at("0392_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) -= tmps.at("0396_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) += tmps.at("0388_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) += tmps.at("0391_aa_vo")(aa, ia))(
        tmps.at("0397_aa_vo")(aa, ia) -= tmps.at("0393_aa_vo")(aa, ia))
      .deallocate(tmps.at("0396_aa_vo"))
      .deallocate(tmps.at("0395_aa_vo"))
      .deallocate(tmps.at("0394_aa_vo"))
      .deallocate(tmps.at("0393_aa_vo"))
      .deallocate(tmps.at("0392_aa_vo"))
      .deallocate(tmps.at("0391_aa_vo"))
      .deallocate(tmps.at("0390_aa_vo"))
      .deallocate(tmps.at("0389_aa_vo"))
      .deallocate(tmps.at("0388_aa_vo"))
      .deallocate(tmps.at("0387_aa_vo"))
      .deallocate(tmps.at("0386_aa_vo"))
      .deallocate(tmps.at("0385_aa_vo"))
      .deallocate(tmps.at("0384_aa_vo"))
      .allocate(tmps.at("0399_aa_vo"))(tmps.at("0399_aa_vo")(aa, ia) =
                                         tmps.at("0397_aa_vo")(aa, ia))(
        tmps.at("0399_aa_vo")(aa, ia) += tmps.at("0398_aa_vo")(aa, ia))
      .deallocate(tmps.at("0398_aa_vo"))
      .deallocate(tmps.at("0397_aa_vo"))(r1.at("aa")(aa, ia) -= tmps.at("0399_aa_vo")(aa, ia))
      .deallocate(tmps.at("0399_aa_vo"))
      .allocate(tmps.at("0435_aa_vo"))(tmps.at("0435_aa_vo")(aa, ia) =
                                         dp.at("aa_oo")(ja, ia) * t1_2p.at("aa")(aa, ja))
      .allocate(tmps.at("0434_aa_vo"))(tmps.at("0434_aa_vo")(aa, ia) =
                                         dp.at("aa_vv")(aa, ba) * t1_2p.at("aa")(ba, ia))
      .allocate(tmps.at("0433_aa_vo"))(tmps.at("0433_aa_vo")(aa, ia) =
                                         dp.at("aa_ov")(ja, ba) * t2_2p.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0432_aa_vo"))(tmps.at("0432_aa_vo")(aa, ia) =
                                         dp.at("bb_ov")(jb, bb) * t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0436_aa_vo"))(tmps.at("0436_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0432_aa_vo")(aa, ia))(
        tmps.at("0436_aa_vo")(aa, ia) += tmps.at("0435_aa_vo")(aa, ia))(
        tmps.at("0436_aa_vo")(aa, ia) -= tmps.at("0434_aa_vo")(aa, ia))(
        tmps.at("0436_aa_vo")(aa, ia) += tmps.at("0433_aa_vo")(aa, ia))
      .deallocate(tmps.at("0435_aa_vo"))
      .deallocate(tmps.at("0434_aa_vo"))
      .deallocate(tmps.at("0433_aa_vo"))
      .deallocate(tmps.at("0432_aa_vo"))(r1_1p.at("aa")(aa, ia) -=
                                         2.00 * tmps.at("0436_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 2.00 * t0_1p * tmps.at("0436_aa_vo")(aa, ia))
      .allocate(tmps.at("0630_aa_vo"))(tmps.at("0630_aa_vo")(aa, ia) =
                                         dp.at("aa_oo")(ja, ia) * t1_1p.at("aa")(aa, ja))
      .allocate(tmps.at("0629_aa_vo"))(tmps.at("0629_aa_vo")(aa, ia) =
                                         dp.at("aa_vv")(aa, ba) * t1_1p.at("aa")(ba, ia))
      .allocate(tmps.at("0628_aa_vo"))(tmps.at("0628_aa_vo")(aa, ia) =
                                         dp.at("aa_ov")(ja, ba) * t2_1p.at("aaaa")(ba, aa, ia, ja))
      .allocate(tmps.at("0627_aa_vo"))(tmps.at("0627_aa_vo")(aa, ia) =
                                         dp.at("bb_ov")(jb, bb) * t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("0631_aa_vo"))(tmps.at("0631_aa_vo")(aa, ia) =
                                         -1.00 * tmps.at("0628_aa_vo")(aa, ia))(
        tmps.at("0631_aa_vo")(aa, ia) -= tmps.at("0630_aa_vo")(aa, ia))(
        tmps.at("0631_aa_vo")(aa, ia) += tmps.at("0629_aa_vo")(aa, ia))(
        tmps.at("0631_aa_vo")(aa, ia) += tmps.at("0627_aa_vo")(aa, ia))
      .deallocate(tmps.at("0630_aa_vo"))
      .deallocate(tmps.at("0629_aa_vo"))
      .deallocate(tmps.at("0628_aa_vo"))
      .deallocate(tmps.at("0627_aa_vo"))(r1_1p.at("aa")(aa, ia) +=
                                         t0_1p * tmps.at("0631_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) += 2.00 * tmps.at("0631_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) += 4.00 * t0_2p * tmps.at("0631_aa_vo")(aa, ia))(
        r1.at("aa")(aa, ia) += tmps.at("0631_aa_vo")(aa, ia))
      .allocate(tmps.at("0496_aa_vo"))(tmps.at("0496_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0043_aa_oo")(ja, ia))
      .allocate(tmps.at("0495_aa_vo"))(tmps.at("0495_aa_vo")(aa, ia) =
                                         t1_2p.at("aa")(aa, ja) * tmps.at("0042_aa_oo")(ja, ia))
      .allocate(tmps.at("0494_aa_vo"))(tmps.at("0494_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0044_aa_oo")(ja, ia))
      .allocate(tmps.at("0723_aa_vo"))(tmps.at("0723_aa_vo")(aa, ia) =
                                         tmps.at("0494_aa_vo")(aa, ia))(
        tmps.at("0723_aa_vo")(aa, ia) += tmps.at("0495_aa_vo")(aa, ia))(
        tmps.at("0723_aa_vo")(aa, ia) += tmps.at("0496_aa_vo")(aa, ia))
      .deallocate(tmps.at("0496_aa_vo"))
      .deallocate(tmps.at("0495_aa_vo"))
      .deallocate(tmps.at("0494_aa_vo"))(r1_1p.at("aa")(aa, ia) -=
                                         2.00 * tmps.at("0723_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 2.00 * t0_1p * tmps.at("0723_aa_vo")(aa, ia))
      .allocate(tmps.at("0761_aa_vo"))(tmps.at("0761_aa_vo")(aa, ia) =
                                         t1.at("aa")(aa, ja) * tmps.at("0043_aa_oo")(ja, ia))
      .allocate(tmps.at("0760_aa_vo"))(tmps.at("0760_aa_vo")(aa, ia) =
                                         t1_1p.at("aa")(aa, ja) * tmps.at("0042_aa_oo")(ja, ia))
      .deallocate(tmps.at("0042_aa_oo"))
      .allocate(tmps.at("0762_aa_vo"))(tmps.at("0762_aa_vo")(aa, ia) =
                                         tmps.at("0760_aa_vo")(aa, ia))(
        tmps.at("0762_aa_vo")(aa, ia) += tmps.at("0761_aa_vo")(aa, ia))
      .deallocate(tmps.at("0761_aa_vo"))
      .deallocate(tmps.at("0760_aa_vo"))(r1_1p.at("aa")(aa, ia) -=
                                         t0_1p * tmps.at("0762_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 2.00 * tmps.at("0762_aa_vo")(aa, ia))(
        r1_2p.at("aa")(aa, ia) -= 4.00 * t0_2p * tmps.at("0762_aa_vo")(aa, ia))(
        r1.at("aa")(aa, ia) -= tmps.at("0762_aa_vo")(aa, ia))
      .allocate(tmps.at("0493_abab_oooo"))(tmps.at("0493_abab_oooo")(ia, jb, ka, lb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0195_abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("0491_abab_oovo"))(tmps.at("0491_abab_oovo")(ia, jb, aa, kb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("0492_abab_oooo"))(tmps.at("0492_abab_oooo")(ia, jb, ka, lb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0491_abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("0490_abba_oooo"))(tmps.at("0490_abba_oooo")(ia, jb, kb, la) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0213_abba_oovo")(ia, jb, ab, la))
      .allocate(tmps.at("0722_abab_oooo"))(tmps.at("0722_abab_oooo")(ia, jb, ka, lb) =
                                             tmps.at("0492_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("0722_abab_oooo")(ia, jb, ka, lb) += tmps.at("0490_abba_oooo")(ia, jb, lb, ka))(
        tmps.at("0722_abab_oooo")(ia, jb, ka, lb) += tmps.at("0493_abab_oooo")(ia, jb, ka, lb))
      .deallocate(tmps.at("0493_abab_oooo"))
      .deallocate(tmps.at("0492_abab_oooo"))
      .deallocate(tmps.at("0490_abba_oooo"))
      .allocate(tmps.at("0739_baab_vooo"))(tmps.at("0739_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0722_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0738_baab_vooo"))(tmps.at("0738_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("bb")(ab, lb) *
                                             tmps.at("0196_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0736_abba_oooo"))(tmps.at("0736_abba_oooo")(ia, jb, kb, la) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0213_abba_oovo")(ia, jb, ab, la))
      .allocate(tmps.at("0737_baba_vooo"))(tmps.at("0737_baba_vooo")(ab, ia, jb, ka) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0736_abba_oooo")(ia, lb, jb, ka))
      .allocate(tmps.at("0734_abab_oooo"))(tmps.at("0734_abab_oooo")(ia, jb, ka, lb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0195_abab_oovo")(ia, jb, aa, lb))
      .allocate(tmps.at("0735_baab_vooo"))(tmps.at("0735_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0734_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0802_baba_vooo"))(tmps.at("0802_baba_vooo")(ab, ia, jb, ka) =
                                             tmps.at("0735_baab_vooo")(ab, ia, ka, jb))(
        tmps.at("0802_baba_vooo")(ab, ia, jb, ka) += tmps.at("0737_baba_vooo")(ab, ia, jb, ka))(
        tmps.at("0802_baba_vooo")(ab, ia, jb, ka) += tmps.at("0738_baab_vooo")(ab, ia, ka, jb))(
        tmps.at("0802_baba_vooo")(ab, ia, jb, ka) += tmps.at("0739_baab_vooo")(ab, ia, ka, jb))
      .deallocate(tmps.at("0739_baab_vooo"))
      .deallocate(tmps.at("0738_baab_vooo"))
      .deallocate(tmps.at("0737_baba_vooo"))
      .deallocate(tmps.at("0735_baab_vooo"))
      .allocate(tmps.at("0806_abba_vvoo"))(tmps.at("0806_abba_vvoo")(aa, bb, ib, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0802_baba_vooo")(bb, ka, ib, ja))
      .deallocate(tmps.at("0802_baba_vooo"))
      .allocate(tmps.at("0804_baba_vooo"))(tmps.at("0804_baba_vooo")(ab, ia, jb, ka) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0736_abba_oooo")(ia, lb, jb, ka))
      .allocate(tmps.at("0805_abba_vvoo"))(tmps.at("0805_abba_vvoo")(aa, bb, ib, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0804_baba_vooo")(bb, ka, ib, ja))
      .allocate(tmps.at("0563_baba_vooo"))(tmps.at("0563_baba_vooo")(ab, ia, jb, ka) =
                                             t2_2p.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0081_aaaa_oovo")(ia, la, ba, ka))
      .deallocate(tmps.at("0081_aaaa_oovo"))
      .allocate(tmps.at("0561_abab_oovo"))(tmps.at("0561_abab_oovo")(ia, jb, aa, kb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t1_2p.at("bb")(bb, kb))
      .allocate(tmps.at("0562_baab_vooo"))(tmps.at("0562_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, lb) *
                                             tmps.at("0561_abab_oovo")(ia, lb, ba, kb))
      .deallocate(tmps.at("0561_abab_oovo"))
      .allocate(tmps.at("0560_baab_vooo"))(tmps.at("0560_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("bb")(ab, lb) *
                                             tmps.at("0130_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0430_abab_oooo"))(tmps.at("0430_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abab_oovo")(ia, jb, aa, lb) *
                                             t1_2p.at("aa")(aa, ka))
      .allocate(tmps.at("0429_abab_oooo"))(tmps.at("0429_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abba_oovo")(ia, jb, ab, ka) *
                                             t1_2p.at("bb")(ab, lb))
      .allocate(tmps.at("0428_abab_oooo"))(tmps.at("0428_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t2_2p.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0431_abab_oooo"))(tmps.at("0431_abab_oooo")(ia, jb, ka, lb) =
                                             -1.00 * tmps.at("0429_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("0431_abab_oooo")(ia, jb, ka, lb) += tmps.at("0428_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("0431_abab_oooo")(ia, jb, ka, lb) += tmps.at("0430_abab_oooo")(ia, jb, ka, lb))
      .deallocate(tmps.at("0430_abab_oooo"))
      .deallocate(tmps.at("0429_abab_oooo"))
      .deallocate(tmps.at("0428_abab_oooo"))
      .allocate(tmps.at("0559_baab_vooo"))(tmps.at("0559_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0431_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0557_baba_vooo"))(tmps.at("0557_baba_vooo")(ab, ia, jb, ka) =
                                             t1_2p.at("bb")(bb, jb) *
                                             tmps.at("0126_baba_vovo")(ab, ia, bb, ka))
      .allocate(tmps.at("0556_baab_vooo"))(tmps.at("0556_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("bb")(ab, lb) *
                                             tmps.at("0108_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0554_baba_vooo"))(tmps.at("0554_baba_vooo")(ab, ia, jb, ka) =
                                             t2_2p.at("bbbb")(bb, ab, jb, lb) *
                                             tmps.at("0213_abba_oovo")(ia, lb, bb, ka))
      .deallocate(tmps.at("0213_abba_oovo"))
      .allocate(tmps.at("0553_baab_vooo"))(tmps.at("0553_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("aa")(ba, ja) *
                                             tmps.at("0085_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("0551_aa_ov"))(tmps.at("0551_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ia, ja, ba, aa) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0552_baab_vooo"))(tmps.at("0552_baab_vooo")(ab, ia, ja, kb) =
                                             t2_1p.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0551_aa_ov")(ia, ba))
      .deallocate(tmps.at("0551_aa_ov"))
      .allocate(tmps.at("0548_abab_oooo"))(tmps.at("0548_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abba_oovo")(ia, jb, ab, ka) *
                                             t1_1p.at("bb")(ab, lb))
      .allocate(tmps.at("0547_abab_oooo"))(tmps.at("0547_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abab_oovv")(ia, jb, aa, bb) *
                                             t2_1p.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("0549_abab_oooo"))(tmps.at("0549_abab_oooo")(ia, jb, ka, lb) =
                                             -1.00 * tmps.at("0547_abab_oooo")(ia, jb, ka, lb))(
        tmps.at("0549_abab_oooo")(ia, jb, ka, lb) += tmps.at("0548_abab_oooo")(ia, jb, ka, lb))
      .deallocate(tmps.at("0548_abab_oooo"))
      .deallocate(tmps.at("0547_abab_oooo"))
      .allocate(tmps.at("0550_baab_vooo"))(tmps.at("0550_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0549_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0544_abab_ovoo"))(tmps.at("0544_abab_ovoo")(ia, ab, ja, kb) =
                                             t1_2p.at("aa")(ba, ja) *
                                             tmps.at("0089_abab_ovvo")(ia, ab, ba, kb))
      .allocate(tmps.at("0447_aa_ov"))(tmps.at("0447_aa_ov")(ia, aa) =
                                         eri.at("abab_oovv")(ia, jb, aa, bb) *
                                         t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("0446_aa_ov"))(tmps.at("0446_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                         t1_2p.at("aa")(ba, ja))
      .allocate(tmps.at("0448_aa_ov"))(tmps.at("0448_aa_ov")(ia, aa) =
                                         tmps.at("0446_aa_ov")(ia, aa))(
        tmps.at("0448_aa_ov")(ia, aa) += tmps.at("0447_aa_ov")(ia, aa))
      .deallocate(tmps.at("0447_aa_ov"))
      .deallocate(tmps.at("0446_aa_ov"))
      .allocate(tmps.at("0543_baab_vooo"))(tmps.at("0543_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0448_aa_ov")(ia, ba))
      .deallocate(tmps.at("0448_aa_ov"))
      .allocate(tmps.at("0541_baab_vovo"))(tmps.at("0541_baab_vovo")(ab, ia, ba, jb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t1_1p.at("bb")(cb, jb))
      .allocate(tmps.at("0542_baab_vooo"))(tmps.at("0542_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0541_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("0540_baab_vooo"))(tmps.at("0540_baab_vooo")(ab, ia, ja, kb) =
                                             t2_1p.at("abab")(ba, ab, ja, lb) *
                                             tmps.at("0491_abab_oovo")(ia, lb, ba, kb))
      .allocate(tmps.at("0539_baab_vooo"))(tmps.at("0539_baab_vooo")(ab, ia, ja, kb) =
                                             t2_1p.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0315_aa_ov")(ia, ba))
      .deallocate(tmps.at("0315_aa_ov"))
      .allocate(tmps.at("0536_aaaa_oovo"))(tmps.at("0536_aaaa_oovo")(ia, ja, aa, ka) =
                                             eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                             t1_2p.at("aa")(ba, ka))
      .allocate(tmps.at("0537_baba_vooo"))(tmps.at("0537_baba_vooo")(ab, ia, jb, ka) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0536_aaaa_oovo")(ia, la, ba, ka))
      .deallocate(tmps.at("0536_aaaa_oovo"))
      .allocate(tmps.at("0535_baab_vooo"))(tmps.at("0535_baab_vooo")(ab, ia, ja, kb) =
                                             t2_2p.at("abab")(ba, ab, ja, lb) *
                                             tmps.at("0195_abab_oovo")(ia, lb, ba, kb))
      .allocate(tmps.at("0533_aaaa_oovo"))(tmps.at("0533_aaaa_oovo")(ia, ja, aa, ka) =
                                             eri.at("aaaa_oovv")(ia, ja, ba, aa) *
                                             t1_1p.at("aa")(ba, ka))
      .allocate(tmps.at("0534_baba_vooo"))(tmps.at("0534_baba_vooo")(ab, ia, jb, ka) =
                                             t2_1p.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0533_aaaa_oovo")(ia, la, ba, ka))
      .deallocate(tmps.at("0533_aaaa_oovo"))
      .allocate(tmps.at("0531_abab_oooo"))(tmps.at("0531_abab_oooo")(ia, jb, ka, lb) =
                                             eri.at("abab_oovo")(ia, jb, aa, lb) *
                                             t1_1p.at("aa")(aa, ka))
      .allocate(tmps.at("0532_baab_vooo"))(tmps.at("0532_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0531_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0730_abab_ovoo"))(tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) =
                                             -1.00 * tmps.at("0534_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) -= tmps.at("0539_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) -= tmps.at("0560_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0540_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0552_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0544_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0554_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0553_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0556_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0542_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0535_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0559_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0537_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0562_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) -= tmps.at("0550_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) -= tmps.at("0563_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0532_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) -= tmps.at("0543_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0730_abab_ovoo")(ia, ab, ja, kb) += tmps.at("0557_baba_vooo")(ab, ia, kb, ja))
      .deallocate(tmps.at("0563_baba_vooo"))
      .deallocate(tmps.at("0562_baab_vooo"))
      .deallocate(tmps.at("0560_baab_vooo"))
      .deallocate(tmps.at("0559_baab_vooo"))
      .deallocate(tmps.at("0557_baba_vooo"))
      .deallocate(tmps.at("0556_baab_vooo"))
      .deallocate(tmps.at("0554_baba_vooo"))
      .deallocate(tmps.at("0553_baab_vooo"))
      .deallocate(tmps.at("0552_baab_vooo"))
      .deallocate(tmps.at("0550_baab_vooo"))
      .deallocate(tmps.at("0544_abab_ovoo"))
      .deallocate(tmps.at("0543_baab_vooo"))
      .deallocate(tmps.at("0542_baab_vooo"))
      .deallocate(tmps.at("0540_baab_vooo"))
      .deallocate(tmps.at("0539_baab_vooo"))
      .deallocate(tmps.at("0537_baba_vooo"))
      .deallocate(tmps.at("0535_baab_vooo"))
      .deallocate(tmps.at("0534_baba_vooo"))
      .deallocate(tmps.at("0532_baab_vooo"))
      .allocate(tmps.at("0801_abab_vvoo"))(tmps.at("0801_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("0730_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0730_abab_ovoo"))
      .allocate(tmps.at("0530_abab_vooo"))(tmps.at("0530_abab_vooo")(aa, ib, ja, kb) =
                                             t1_2p.at("aa")(ba, ja) *
                                             tmps.at("0091_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("0527_abba_voov"))(tmps.at("0527_abba_voov")(aa, ib, jb, ba) =
                                             eri.at("abab_oovv")(ka, ib, ba, cb) *
                                             t2_2p.at("abab")(aa, cb, ka, jb))
      .allocate(tmps.at("0528_abab_vooo"))(tmps.at("0528_abab_vooo")(aa, ib, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0527_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("0526_abab_vooo"))(tmps.at("0526_abab_vooo")(aa, ib, ja, kb) =
                                             t1_2p.at("aa")(ba, ja) *
                                             tmps.at("0096_abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("0455_bb_ov"))(tmps.at("0455_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(ib, jb, ab, bb) *
                                         t1_2p.at("bb")(bb, jb))
      .allocate(tmps.at("0454_bb_ov"))(tmps.at("0454_bb_ov")(ib, ab) =
                                         eri.at("abab_oovv")(ja, ib, ba, ab) *
                                         t1_2p.at("aa")(ba, ja))
      .allocate(tmps.at("0456_bb_ov"))(tmps.at("0456_bb_ov")(ib, ab) =
                                         tmps.at("0454_bb_ov")(ib, ab))(
        tmps.at("0456_bb_ov")(ib, ab) += tmps.at("0455_bb_ov")(ib, ab))
      .deallocate(tmps.at("0455_bb_ov"))
      .deallocate(tmps.at("0454_bb_ov"))
      .allocate(tmps.at("0525_abab_vooo"))(tmps.at("0525_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("0456_bb_ov")(ib, bb))
      .deallocate(tmps.at("0456_bb_ov"))
      .allocate(tmps.at("0524_abba_vooo"))(tmps.at("0524_abba_vooo")(aa, ib, jb, ka) =
                                             t1_2p.at("bb")(bb, jb) *
                                             tmps.at("0204_abab_voov")(aa, ib, ka, bb))
      .allocate(tmps.at("0503_bb_ov"))(tmps.at("0503_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(ib, jb, bb, ab) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("0523_abab_vooo"))(tmps.at("0523_abab_vooo")(aa, ib, ja, kb) =
                                             t2_1p.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("0503_bb_ov")(ib, bb))
      .allocate(tmps.at("0522_abab_vooo"))(tmps.at("0522_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("aaaa")(ba, aa, ja, la) *
                                             tmps.at("0195_abab_oovo")(la, ib, ba, kb))
      .allocate(tmps.at("0519_abab_vovo"))(tmps.at("0519_abab_vovo")(aa, ib, ba, jb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t1_1p.at("bb")(cb, jb))
      .allocate(tmps.at("0520_abab_vooo"))(tmps.at("0520_abab_vooo")(aa, ib, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0519_abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("0518_abba_vooo"))(tmps.at("0518_abba_vooo")(aa, ib, jb, ka) =
                                             t1.at("bb")(bb, jb) *
                                             tmps.at("0287_abab_voov")(aa, ib, ka, bb))
      .allocate(tmps.at("0516_abba_voov"))(tmps.at("0516_abba_voov")(aa, ib, jb, ba) =
                                             eri.at("abab_oovv")(ka, ib, ba, cb) *
                                             t2_1p.at("abab")(aa, cb, ka, jb))
      .allocate(tmps.at("0517_abab_vooo"))(tmps.at("0517_abab_vooo")(aa, ib, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0516_abba_voov")(aa, ib, kb, ba))
      .allocate(tmps.at("0515_abba_vooo"))(tmps.at("0515_abba_vooo")(aa, ib, jb, ka) =
                                             t1_1p.at("bb")(bb, jb) *
                                             tmps.at("0257_abab_voov")(aa, ib, ka, bb))
      .allocate(tmps.at("0512_bb_ov"))(tmps.at("0512_bb_ov")(ib, ab) =
                                         eri.at("abab_oovv")(ja, ib, ba, ab) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0513_abab_vooo"))(tmps.at("0513_abab_vooo")(aa, ib, ja, kb) =
                                             t2_1p.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("0512_bb_ov")(ib, bb))
      .deallocate(tmps.at("0512_bb_ov"))
      .allocate(tmps.at("0511_abba_vooo"))(tmps.at("0511_abba_vooo")(aa, ib, jb, ka) =
                                             t1_1p.at("bb")(bb, jb) *
                                             tmps.at("0280_abab_voov")(aa, ib, ka, bb))
      .allocate(tmps.at("0508_abba_vooo"))(tmps.at("0508_abba_vooo")(aa, ib, jb, ka) =
                                             t1_2p.at("bb")(bb, jb) *
                                             tmps.at("0106_abba_vovo")(aa, ib, bb, ka))
      .allocate(tmps.at("0728_abab_vooo"))(tmps.at("0728_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("0513_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) += tmps.at("0515_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) += tmps.at("0522_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) += tmps.at("0528_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0518_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0508_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0525_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) += tmps.at("0530_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0520_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) += tmps.at("0517_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0511_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) -= tmps.at("0526_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) += tmps.at("0524_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("0728_abab_vooo")(aa, ib, ja, kb) += tmps.at("0523_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("0530_abab_vooo"))
      .deallocate(tmps.at("0528_abab_vooo"))
      .deallocate(tmps.at("0526_abab_vooo"))
      .deallocate(tmps.at("0525_abab_vooo"))
      .deallocate(tmps.at("0524_abba_vooo"))
      .deallocate(tmps.at("0523_abab_vooo"))
      .deallocate(tmps.at("0522_abab_vooo"))
      .deallocate(tmps.at("0520_abab_vooo"))
      .deallocate(tmps.at("0518_abba_vooo"))
      .deallocate(tmps.at("0517_abab_vooo"))
      .deallocate(tmps.at("0515_abba_vooo"))
      .deallocate(tmps.at("0513_abab_vooo"))
      .deallocate(tmps.at("0511_abba_vooo"))
      .deallocate(tmps.at("0508_abba_vooo"))
      .allocate(tmps.at("0800_baab_vvoo"))(tmps.at("0800_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0728_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .deallocate(tmps.at("0728_abab_vooo"))
      .allocate(tmps.at("0799_abba_vvoo"))(tmps.at("0799_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0320_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .deallocate(tmps.at("0320_aa_oo"))
      .allocate(tmps.at("0504_bb_oo"))(tmps.at("0504_bb_oo")(ib, jb) =
                                         t1_1p.at("bb")(ab, jb) * tmps.at("0503_bb_ov")(ib, ab))
      .deallocate(tmps.at("0503_bb_ov"))
      .allocate(tmps.at("0500_bb_oo"))(tmps.at("0500_bb_oo")(ib, jb) =
                                         t1_2p.at("aa")(aa, ka) *
                                         tmps.at("0195_abab_oovo")(ka, ib, aa, jb))
      .allocate(tmps.at("0498_bb_oo"))(tmps.at("0498_bb_oo")(ib, jb) =
                                         t1_2p.at("bb")(ab, kb) *
                                         tmps.at("0079_bbbb_oovo")(ib, kb, ab, jb))
      .allocate(tmps.at("0497_bb_oo"))(tmps.at("0497_bb_oo")(ib, jb) =
                                         t1_1p.at("aa")(aa, ka) *
                                         tmps.at("0491_abab_oovo")(ka, ib, aa, jb))
      .allocate(tmps.at("0724_bb_oo"))(tmps.at("0724_bb_oo")(ib, jb) =
                                         -1.00 * tmps.at("0504_bb_oo")(ib, jb))(
        tmps.at("0724_bb_oo")(ib, jb) += tmps.at("0497_bb_oo")(ib, jb))(
        tmps.at("0724_bb_oo")(ib, jb) += tmps.at("0500_bb_oo")(ib, jb))(
        tmps.at("0724_bb_oo")(ib, jb) += tmps.at("0498_bb_oo")(ib, jb))
      .deallocate(tmps.at("0504_bb_oo"))
      .deallocate(tmps.at("0500_bb_oo"))
      .deallocate(tmps.at("0498_bb_oo"))
      .deallocate(tmps.at("0497_bb_oo"))
      .allocate(tmps.at("0798_abab_vvoo"))(tmps.at("0798_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0724_bb_oo")(kb, jb))
      .deallocate(tmps.at("0724_bb_oo"))
      .allocate(tmps.at("0795_baab_vooo"))(tmps.at("0795_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0130_abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("0794_abab_ovoo"))(tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0195_abab_oovo")(ia, lb, ba, kb))(
        tmps.at("0794_abab_ovoo")(ia, ab, ja, kb) =
          tmps.at("bin_aabb_oooo")(ia, ja, kb, lb) * t1_1p.at("bb")(ab, lb))
      .allocate(tmps.at("0793_baab_vooo"))(tmps.at("0793_baab_vooo")(ab, ia, ja, kb) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0541_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("0792_baab_vooo"))(tmps.at("0792_baab_vooo")(ab, ia, ja, kb) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0085_baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("0796_baab_vooo"))(tmps.at("0796_baab_vooo")(ab, ia, ja, kb) =
                                             -1.00 * tmps.at("0795_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0796_baab_vooo")(ab, ia, ja, kb) += tmps.at("0792_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("0796_baab_vooo")(ab, ia, ja, kb) += tmps.at("0794_abab_ovoo")(ia, ab, ja, kb))(
        tmps.at("0796_baab_vooo")(ab, ia, ja, kb) += tmps.at("0793_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("0795_baab_vooo"))
      .deallocate(tmps.at("0794_abab_ovoo"))
      .deallocate(tmps.at("0793_baab_vooo"))
      .deallocate(tmps.at("0792_baab_vooo"))
      .allocate(tmps.at("0797_abab_vvoo"))(tmps.at("0797_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0796_baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("0545_aa_ov"))(tmps.at("0545_aa_ov")(ia, aa) =
                                         eri.at("aaaa_oovv")(ja, ia, aa, ba) *
                                         t1_1p.at("aa")(ba, ja))
      .allocate(tmps.at("0546_baab_vooo"))(tmps.at("0546_baab_vooo")(ab, ia, ja, kb) =
                                             t2.at("abab")(ba, ab, ja, kb) *
                                             tmps.at("0545_aa_ov")(ia, ba))
      .deallocate(tmps.at("0545_aa_ov"))
      .allocate(tmps.at("0538_baba_vooo"))(tmps.at("0538_baba_vooo")(ab, ia, jb, ka) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0310_aaaa_oovo")(la, ia, ba, ka))
      .allocate(tmps.at("0731_baab_vooo"))(tmps.at("0731_baab_vooo")(ab, ia, ja, kb) =
                                             -1.00 * tmps.at("0538_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("0731_baab_vooo")(ab, ia, ja, kb) += tmps.at("0546_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("0546_baab_vooo"))
      .deallocate(tmps.at("0538_baba_vooo"))
      .allocate(tmps.at("0791_abab_vvoo"))(tmps.at("0791_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0731_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0731_baab_vooo"))
      .allocate(tmps.at("0790_abab_vvoo"))(tmps.at("0790_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0222_abab_ovoo")(ka, bb, ia, jb));
  }
}

template void exachem::cc::qed_ccsd_cs::resid_part2<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);