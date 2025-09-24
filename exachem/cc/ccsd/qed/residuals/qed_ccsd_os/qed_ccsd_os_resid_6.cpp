/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_os_resid_6.hpp"

template<typename T>
void exachem::cc::qed_ccsd_os::resid_6(
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
      .allocate(tmps.at("1423_abab_vvoo"))(tmps.at("1423_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0188_bb_vo")(bb, jb))
      .allocate(tmps.at("1422_abba_vvoo"))(tmps.at("1422_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0417_aa_oo")(ka, ja))
      .allocate(tmps.at("1421_abab_vvoo"))(tmps.at("1421_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("1382_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("1382_abab_oooo"))
      .allocate(tmps.at("1420_abab_vvoo"))(tmps.at("1420_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0596_bb_oo")(kb, jb))
      .allocate(tmps.at("1206_bb_ov"))(tmps.at("1206_bb_ov")(ib, ab) =
                                         eri.at("bbbb_oovv")(jb, ib, ab, bb) *
                                         t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("1207_abab_vooo"))(tmps.at("1207_abab_vooo")(aa, ib, ja, kb) =
                                             t2.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("1206_bb_ov")(ib, bb))
      .allocate(tmps.at("1203_abba_vooo"))(tmps.at("1203_abba_vooo")(aa, ib, jb, ka) =
                                             tmps.at("0376_abab_voov")(aa, ib, ka, bb) *
                                             t1_1p.at("bb")(bb, jb))
      .allocate(tmps.at("1384_abab_vooo"))(tmps.at("1384_abab_vooo")(aa, ib, ja, kb) =
                                             tmps.at("1207_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1384_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1203_abba_vooo")(aa, ib, kb, ja))
      .deallocate(tmps.at("1207_abab_vooo"))
      .deallocate(tmps.at("1203_abba_vooo"))
      .allocate(tmps.at("1419_baab_vvoo"))(tmps.at("1419_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1384_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("1384_abab_vooo"))
      .allocate(tmps.at("1418_abba_vvoo"))(tmps.at("1418_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0431_aa_oo")(ka, ja))
      .allocate(tmps.at("1189_bb_ov"))(tmps.at("1189_bb_ov")(ib, ab) =
                                         eri.at("abab_oovv")(ja, ib, ba, ab) * t1.at("aa")(ba, ja))
      .allocate(tmps.at("1190_abab_vooo"))(tmps.at("1190_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("1189_bb_ov")(ib, bb))
      .allocate(tmps.at("1187_abab_vooo"))(tmps.at("1187_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("abab")(aa, bb, ja, kb) *
                                             tmps.at("0237_bb_ov")(ib, bb))
      .allocate(tmps.at("1383_abab_vooo"))(tmps.at("1383_abab_vooo")(aa, ib, ja, kb) =
                                             tmps.at("1187_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1383_abab_vooo")(aa, ib, ja, kb) += tmps.at("1190_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("1190_abab_vooo"))
      .deallocate(tmps.at("1187_abab_vooo"))
      .allocate(tmps.at("1417_baab_vvoo"))(tmps.at("1417_baab_vvoo")(ab, ba, ia, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1383_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("1383_abab_vooo"))
      .allocate(tmps.at("1416_baab_vvoo"))(tmps.at("1416_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1058_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("1058_abab_vooo"))
      .allocate(tmps.at("1415_abba_vvoo"))(tmps.at("1415_abba_vvoo")(aa, bb, ib, ja) =
                                             t2.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0426_aa_oo")(ka, ja))
      .allocate(tmps.at("1414_baba_vvoo"))(tmps.at("1414_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_2p.at("bb")(ab, ib) * tmps.at("0765_aa_vo")(ba, ja))
      .allocate(tmps.at("1413_abba_vvoo"))(tmps.at("1413_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_2p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0248_aa_oo")(ka, ja))
      .allocate(tmps.at("1412_baba_vvoo"))(tmps.at("1412_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("1411_aa_vo")(ba, ja))
      .allocate(tmps.at("1410_baab_vvoo"))(tmps.at("1410_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0879_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0879_abab_vooo"))
      .allocate(tmps.at("1409_baba_vvoo"))(tmps.at("1409_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("1408_aa_vo")(ba, ja))
      .allocate(tmps.at("1407_abab_vvoo"))(tmps.at("1407_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1406_bb_vo")(bb, jb))
      .allocate(tmps.at("1405_abab_vvoo"))(tmps.at("1405_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0586_bb_oo")(kb, jb))
      .allocate(tmps.at("1404_abba_vvoo"))(tmps.at("1404_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0422_aa_oo")(ka, ja))
      .allocate(tmps.at("1403_abba_vvoo"))(tmps.at("1403_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, bb, ka, ib) *
                                             tmps.at("0419_aa_oo")(ka, ja))
      .allocate(tmps.at("1402_baab_vvoo"))(tmps.at("1402_baab_vvoo")(ab, ba, ia, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0873_abab_vooo")(ba, kb, ia, jb))
      .deallocate(tmps.at("0873_abab_vooo"))
      .allocate(tmps.at("1401_abab_vvoo"))(tmps.at("1401_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0887_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0887_baab_vooo"))
      .allocate(tmps.at("1400_abab_vvoo"))(tmps.at("1400_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1399_bb_vo")(bb, jb))
      .allocate(tmps.at("1398_abab_vvoo"))(tmps.at("1398_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0598_bb_oo")(kb, jb))
      .allocate(tmps.at("1397_abab_vvoo"))(tmps.at("1397_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1396_bb_vo")(bb, jb))
      .allocate(tmps.at("1395_abab_vvoo"))(tmps.at("1395_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0875_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0875_abab_oooo"))
      .allocate(tmps.at("1394_baba_vvoo"))(tmps.at("1394_baba_vvoo")(ab, ba, ib, ja) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("1393_aa_vo")(ba, ja))
      .allocate(tmps.at("1381_abab_vvoo"))(tmps.at("1381_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0505_bb_vv")(bb, cb))
      .allocate(tmps.at("1380_abab_vvoo"))(tmps.at("1380_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0536_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("1378_bb_vv"))(tmps.at("1378_bb_vv")(ab, bb) =
                                         eri.at("bbbb_vovv")(ab, ib, cb, bb) *
                                         t1_1p.at("bb")(cb, ib))
      .allocate(tmps.at("1379_abab_vvoo"))(tmps.at("1379_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("1378_bb_vv")(bb, cb))
      .allocate(tmps.at("1377_abab_vvoo"))(tmps.at("1377_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0836_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0836_abab_ovoo"))
      .allocate(tmps.at("1376_abab_vvoo"))(tmps.at("1376_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0831_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0831_abab_oooo"))
      .allocate(tmps.at("1374_aaaa_voov"))(tmps.at("1374_aaaa_voov")(aa, ia, ja, ba) =
                                             t2_1p.at("aaaa")(ca, aa, ja, ka) *
                                             eri.at("aaaa_oovv")(ka, ia, ca, ba))
      .allocate(tmps.at("1375_baba_vvoo"))(tmps.at("1375_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("1374_aaaa_voov")(ba, ka, ja, ca) *
                                             t2_1p.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("1373_abba_vvoo"))(tmps.at("1373_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0990_abba_vvvo")(aa, bb, cb, ja) *
                                             t1_1p.at("bb")(cb, ib))
      .deallocate(tmps.at("0990_abba_vvvo"))
      .allocate(tmps.at("1372_baab_vvoo"))(tmps.at("1372_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0999_abab_vovo")(ba, kb, ca, jb) *
                                             t2_1p.at("abab")(ca, ab, ia, kb))
      .deallocate(tmps.at("0999_abab_vovo"))
      .allocate(tmps.at("1371_baab_vvoo"))(tmps.at("1371_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0816_abba_voov")(ba, kb, jb, ca) *
                                             t2_2p.at("abab")(ca, ab, ia, kb))
      .deallocate(tmps.at("0816_abba_voov"))
      .allocate(tmps.at("1370_baab_vvoo"))(tmps.at("1370_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0403_aa_vv")(ba, ca) *
                                             t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1369_abba_vvoo"))(tmps.at("1369_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0031_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1368_baab_vvoo"))(tmps.at("1368_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0664_aa_vv")(ba, ca) *
                                             t2_2p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1367_abab_vvoo"))(tmps.at("1367_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0528_bb_oo")(kb, jb))
      .allocate(tmps.at("1366_baba_vvoo"))(tmps.at("1366_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("1365_aa_vo")(ba, ja) * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("1362_abab_vvoo"))(tmps.at("1362_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0557_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("1361_abab_vvoo"))(tmps.at("1361_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0921_bb_vv")(bb, cb))
      .allocate(tmps.at("1360_abab_vvoo"))(tmps.at("1360_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0532_bb_oo")(kb, jb))
      .allocate(tmps.at("1359_abab_vvoo"))(tmps.at("1359_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0133_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("1358_abab_vvoo"))(tmps.at("1358_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0568_bb_oo")(kb, jb))
      .allocate(tmps.at("1357_abba_vvoo"))(tmps.at("1357_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0387_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1356_abab_vvoo"))(tmps.at("1356_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0158_bb_vo")(bb, jb))
      .allocate(tmps.at("1355_baba_vvoo"))(tmps.at("1355_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("1354_aa_vo")(ba, ja) * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("1353_abab_vvoo"))(tmps.at("1353_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0276_bb_oo")(kb, jb))
      .allocate(tmps.at("1352_abab_vvoo"))(tmps.at("1352_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0035_baba_voov")(bb, ka, jb, ca))
      .deallocate(tmps.at("0035_baba_voov"))
      .allocate(tmps.at("1350_aa_oo"))(tmps.at("1350_aa_oo")(ia, ja) =
                                         eri.at("aaaa_oovv")(ka, ia, aa, ba) *
                                         t2_1p.at("aaaa")(aa, ba, ja, ka))
      .allocate(tmps.at("1351_abba_vvoo"))(tmps.at("1351_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("1350_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1349_abab_vvoo"))(tmps.at("1349_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1025_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("1025_baab_vooo"))
      .allocate(tmps.at("1155_baab_vooo"))(tmps.at("1155_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baba_vovo")(ab, ia, bb, ja) *
                                             t1_2p.at("bb")(bb, kb))
      .allocate(tmps.at("1154_baba_vooo"))(tmps.at("1154_baba_vooo")(ab, ia, jb, ka) =
                                             t2_2p.at("abab")(ba, ab, la, jb) *
                                             eri.at("aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("1153_baab_vooo"))(tmps.at("1153_baab_vooo")(ab, ia, ja, kb) =
                                             eri.at("baab_vovv")(ab, ia, ba, cb) *
                                             t2_2p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("1152_baab_vooo"))(tmps.at("1152_baab_vooo")(ab, ia, ja, kb) =
                                             t2_2p.at("abab")(ba, ab, ja, lb) *
                                             eri.at("abab_oovo")(ia, lb, ba, kb))
      .allocate(tmps.at("1151_baab_vooo"))(tmps.at("1151_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("aa")(ba, ja) *
                                             eri.at("baab_vovo")(ab, ia, ba, kb))
      .allocate(tmps.at("1150_baab_vooo"))(tmps.at("1150_baab_vooo")(ab, ia, ja, kb) =
                                             t2_2p.at("abab")(ba, ab, ja, kb) *
                                             f.at("aa_ov")(ia, ba))
      .allocate(tmps.at("1149_baba_vooo"))(tmps.at("1149_baba_vooo")(ab, ia, jb, ka) =
                                             t2_2p.at("bbbb")(bb, ab, jb, lb) *
                                             eri.at("abba_oovo")(ia, lb, bb, ka))
      .allocate(tmps.at("1148_baab_vooo"))(tmps.at("1148_baab_vooo")(ab, ia, ja, kb) =
                                             t1_2p.at("bb")(ab, lb) *
                                             eri.at("abab_oooo")(ia, lb, ja, kb))
      .allocate(tmps.at("1181_baab_vooo"))(tmps.at("1181_baab_vooo")(ab, ia, ja, kb) =
                                             -1.00 * tmps.at("1148_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1181_baab_vooo")(ab, ia, ja, kb) += tmps.at("1155_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1181_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1154_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("1181_baab_vooo")(ab, ia, ja, kb) += tmps.at("1149_baba_vooo")(ab, ia, kb, ja))(
        tmps.at("1181_baab_vooo")(ab, ia, ja, kb) += tmps.at("1150_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1181_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1153_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1181_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1152_baab_vooo")(ab, ia, ja, kb))(
        tmps.at("1181_baab_vooo")(ab, ia, ja, kb) -= tmps.at("1151_baab_vooo")(ab, ia, ja, kb))
      .deallocate(tmps.at("1155_baab_vooo"))
      .deallocate(tmps.at("1154_baba_vooo"))
      .deallocate(tmps.at("1153_baab_vooo"))
      .deallocate(tmps.at("1152_baab_vooo"))
      .deallocate(tmps.at("1151_baab_vooo"))
      .deallocate(tmps.at("1150_baab_vooo"))
      .deallocate(tmps.at("1149_baba_vooo"))
      .deallocate(tmps.at("1148_baab_vooo"))
      .allocate(tmps.at("1348_abab_vvoo"))(tmps.at("1348_abab_vvoo")(aa, bb, ia, jb) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1181_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("1181_baab_vooo"))
      .allocate(tmps.at("1347_abab_vvoo"))(tmps.at("1347_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("1179_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("1179_abab_oooo"))
      .allocate(tmps.at("1346_abba_vvoo"))(tmps.at("1346_abba_vvoo")(aa, bb, ib, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0971_baba_vooo")(bb, ka, ib, ja))
      .deallocate(tmps.at("0971_baba_vooo"))
      .allocate(tmps.at("1345_abab_vvoo"))(tmps.at("1345_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1031_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("1031_abab_ovoo"))
      .allocate(tmps.at("1344_abab_vvoo"))(tmps.at("1344_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0154_bb_vo")(bb, jb))
      .allocate(tmps.at("1343_baab_vvoo"))(tmps.at("1343_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1042_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .deallocate(tmps.at("1042_abab_vooo"))
      .allocate(tmps.at("1342_baba_vvoo"))(tmps.at("1342_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0376_abab_voov")(ba, kb, ja, cb) *
                                             t2_2p.at("bbbb")(cb, ab, ib, kb))
      .allocate(tmps.at("1147_abab_vooo"))(tmps.at("1147_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t2_2p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("1146_abab_vooo"))(tmps.at("1146_abab_vooo")(aa, ib, ja, kb) =
                                             eri.at("abba_vovo")(aa, ib, bb, ja) *
                                             t1_2p.at("bb")(bb, kb))
      .allocate(tmps.at("1145_abab_vooo"))(tmps.at("1145_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("abab")(aa, bb, ja, kb) *
                                             f.at("bb_ov")(ib, bb))
      .allocate(tmps.at("1144_abab_vooo"))(tmps.at("1144_abab_vooo")(aa, ib, ja, kb) =
                                             t1_2p.at("aa")(ba, ja) *
                                             eri.at("abab_vovo")(aa, ib, ba, kb))
      .allocate(tmps.at("1143_abab_vooo"))(tmps.at("1143_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("abab")(aa, bb, ja, lb) *
                                             eri.at("bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("1142_abba_vooo"))(tmps.at("1142_abba_vooo")(aa, ib, jb, ka) =
                                             t2_2p.at("abab")(aa, bb, la, jb) *
                                             eri.at("abba_oovo")(la, ib, bb, ka))
      .allocate(tmps.at("1141_abab_vooo"))(tmps.at("1141_abab_vooo")(aa, ib, ja, kb) =
                                             t2_2p.at("aaaa")(ba, aa, ja, la) *
                                             eri.at("abab_oovo")(la, ib, ba, kb))
      .allocate(tmps.at("1180_abab_vooo"))(tmps.at("1180_abab_vooo")(aa, ib, ja, kb) =
                                             -1.00 * tmps.at("1141_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1180_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1146_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1180_abab_vooo")(aa, ib, ja, kb) += tmps.at("1142_abba_vooo")(aa, ib, kb, ja))(
        tmps.at("1180_abab_vooo")(aa, ib, ja, kb) += tmps.at("1144_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1180_abab_vooo")(aa, ib, ja, kb) += tmps.at("1147_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1180_abab_vooo")(aa, ib, ja, kb) += tmps.at("1145_abab_vooo")(aa, ib, ja, kb))(
        tmps.at("1180_abab_vooo")(aa, ib, ja, kb) -= tmps.at("1143_abab_vooo")(aa, ib, ja, kb))
      .deallocate(tmps.at("1147_abab_vooo"))
      .deallocate(tmps.at("1146_abab_vooo"))
      .deallocate(tmps.at("1145_abab_vooo"))
      .deallocate(tmps.at("1144_abab_vooo"))
      .deallocate(tmps.at("1143_abab_vooo"))
      .deallocate(tmps.at("1142_abba_vooo"))
      .deallocate(tmps.at("1141_abab_vooo"))
      .allocate(tmps.at("1341_baab_vvoo"))(tmps.at("1341_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1180_abab_vooo")(ba, kb, ia, jb) *
                                             t1.at("bb")(ab, kb))
      .deallocate(tmps.at("1180_abab_vooo"))
      .allocate(tmps.at("1340_abab_vvoo"))(tmps.at("1340_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0033_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("1339_abab_vvoo"))(tmps.at("1339_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0141_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("1338_baba_vvoo"))(tmps.at("1338_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0230_abba_vovo")(ba, kb, cb, ja) *
                                             t2_2p.at("bbbb")(cb, ab, ib, kb))
      .allocate(tmps.at("1337_abab_vvoo"))(tmps.at("1337_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0297_bb_vv")(bb, cb))
      .allocate(tmps.at("1336_abab_vvoo"))(tmps.at("1336_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1335_bb_vo")(bb, jb))
      .allocate(tmps.at("1332_abba_vvoo"))(tmps.at("1332_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_2p.at("abab")(aa, cb, ka, ib) *
                                             tmps.at("0799_baba_vovo")(bb, ka, cb, ja))
      .deallocate(tmps.at("0799_baba_vovo"))
      .allocate(tmps.at("1329_aa_vv"))(tmps.at("1329_aa_vv")(aa, ba) =
                                         eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                         t1_2p.at("aa")(ca, ia))
      .allocate(tmps.at("1328_aa_vv"))(tmps.at("1328_aa_vv")(aa, ba) =
                                         eri.at("abab_vovv")(aa, ib, ba, cb) *
                                         t1_2p.at("bb")(cb, ib))
      .allocate(tmps.at("1330_aa_vv"))(tmps.at("1330_aa_vv")(aa, ba) =
                                         tmps.at("1328_aa_vv")(aa, ba))(
        tmps.at("1330_aa_vv")(aa, ba) += tmps.at("1329_aa_vv")(aa, ba))
      .deallocate(tmps.at("1329_aa_vv"))
      .deallocate(tmps.at("1328_aa_vv"))
      .allocate(tmps.at("1331_baab_vvoo"))(tmps.at("1331_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1330_aa_vv")(ba, ca) *
                                             t2.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1327_abab_vvoo"))(tmps.at("1327_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1326_bb_vo")(bb, jb))
      .allocate(tmps.at("1325_abab_vvoo"))(tmps.at("1325_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0299_bb_vv")(bb, cb))
      .allocate(tmps.at("1323_bb_oo"))(tmps.at("1323_bb_oo")(ib, jb) =
                                         eri.at("bbbb_oovv")(kb, ib, ab, bb) *
                                         t2_1p.at("bbbb")(ab, bb, jb, kb))
      .allocate(tmps.at("1324_abab_vvoo"))(tmps.at("1324_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("1323_bb_oo")(kb, jb))
      .allocate(tmps.at("1322_baab_vvoo"))(tmps.at("1322_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0985_baab_ovoo")(kb, ba, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .deallocate(tmps.at("0985_baab_ovoo"))
      .allocate(tmps.at("1321_baba_vvoo"))(tmps.at("1321_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("1320_aa_vo")(ba, ja) * t1_1p.at("bb")(ab, ib))
      .allocate(tmps.at("1319_abba_vvoo"))(tmps.at("1319_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0390_aa_oo")(ka, ja) *
                                             t2_2p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1318_baab_vvoo"))(tmps.at("1318_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1012_aa_vv")(ba, ca) *
                                             t2_1p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1317_abab_vvoo"))(tmps.at("1317_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0016_bb_oo")(kb, jb))
      .allocate(tmps.at("1316_baab_vvoo"))(tmps.at("1316_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1051_abab_vooo")(ba, kb, ia, jb) *
                                             t1_2p.at("bb")(ab, kb))
      .deallocate(tmps.at("1051_abab_vooo"))
      .allocate(tmps.at("1315_abab_vvoo"))(tmps.at("1315_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0281_bb_oo")(kb, jb))
      .allocate(tmps.at("1314_abab_vvoo"))(tmps.at("1314_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0842_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0842_baab_vooo"))
      .allocate(tmps.at("1313_baba_vvoo"))(tmps.at("1313_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0847_baba_ovoo")(kb, ba, ib, ja) *
                                             t1_2p.at("bb")(ab, kb))
      .deallocate(tmps.at("0847_baba_ovoo"))
      .allocate(tmps.at("1312_abab_vvoo"))(tmps.at("1312_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0976_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0976_abab_oooo"))
      .allocate(tmps.at("1311_abba_vvoo"))(tmps.at("1311_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0234_aa_oo")(ka, ja) *
                                             t2_2p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1310_baab_vvoo"))(tmps.at("1310_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0043_aa_vv")(ba, ca) *
                                             t2_1p.at("abab")(ca, ab, ia, jb))
      .deallocate(tmps.at("0043_aa_vv"))
      .allocate(tmps.at("1309_abba_vvoo"))(tmps.at("1309_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0241_aa_oo")(ka, ja) *
                                             t2_2p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1308_baab_vvoo"))(tmps.at("1308_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0855_abab_vooo")(ba, kb, ia, jb) *
                                             t1_2p.at("bb")(ab, kb))
      .deallocate(tmps.at("0855_abab_vooo"))
      .allocate(tmps.at("1307_abab_vvoo"))(tmps.at("1307_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0819_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0819_abab_oooo"))
      .allocate(tmps.at("1306_baab_vvoo"))(tmps.at("1306_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0940_abba_voov")(ba, kb, jb, ca) *
                                             t2_1p.at("abab")(ca, ab, ia, kb))
      .deallocate(tmps.at("0940_abba_voov"))
      .allocate(tmps.at("1305_abba_vvoo"))(tmps.at("1305_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0023_aa_oo")(ka, ja) *
                                             t2_2p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1304_baab_vvoo"))(tmps.at("1304_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1007_baab_ovoo")(kb, ba, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .deallocate(tmps.at("1007_baab_ovoo"))
      .allocate(tmps.at("1303_abab_vvoo"))(tmps.at("1303_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ka, lb) *
                                             tmps.at("0994_abab_oooo")(ka, lb, ia, jb))
      .deallocate(tmps.at("0994_abab_oooo"))
      .allocate(tmps.at("1302_abab_vvoo"))(tmps.at("1302_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0144_bbbb_voov")(bb, kb, jb, cb))
      .allocate(tmps.at("1301_baba_vvoo"))(tmps.at("1301_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0363_aaaa_vovo")(ba, ka, ca, ja) *
                                             t2_2p.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("1300_baab_vvoo"))(tmps.at("1300_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0826_baab_ovoo")(kb, ba, ia, jb) *
                                             t1_2p.at("bb")(ab, kb))
      .deallocate(tmps.at("0826_baab_ovoo"))
      .allocate(tmps.at("1299_abab_vvoo"))(tmps.at("1299_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0135_baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("1298_abab_vvoo"))(tmps.at("1298_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0577_bb_oo")(kb, jb))
      .allocate(tmps.at("1297_baba_vvoo"))(tmps.at("1297_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0037_aaaa_voov")(ba, ka, ja, ca) *
                                             t2_2p.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("1296_abab_vvoo"))(tmps.at("1296_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0572_bb_oo")(kb, jb))
      .allocate(tmps.at("1295_abab_vvoo"))(tmps.at("1295_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0560_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("1294_abba_vvoo"))(tmps.at("1294_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0399_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1291_bb_vv"))(tmps.at("1291_bb_vv")(ab, bb) =
                                         eri.at("baab_vovv")(ab, ia, ca, bb) *
                                         t1_2p.at("aa")(ca, ia))
      .allocate(tmps.at("1290_bb_vv"))(tmps.at("1290_bb_vv")(ab, bb) =
                                         eri.at("bbbb_vovv")(ab, ib, bb, cb) *
                                         t1_2p.at("bb")(cb, ib))
      .allocate(tmps.at("1292_bb_vv"))(tmps.at("1292_bb_vv")(ab, bb) =
                                         -1.00 * tmps.at("1290_bb_vv")(ab, bb))(
        tmps.at("1292_bb_vv")(ab, bb) += tmps.at("1291_bb_vv")(ab, bb))
      .deallocate(tmps.at("1291_bb_vv"))
      .deallocate(tmps.at("1290_bb_vv"))
      .allocate(tmps.at("1293_abab_vvoo"))(tmps.at("1293_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("1292_bb_vv")(bb, cb))
      .allocate(tmps.at("1289_baba_vvoo"))(tmps.at("1289_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0372_abba_vovo")(ba, kb, cb, ja) *
                                             t2.at("bbbb")(cb, ab, ib, kb))
      .allocate(tmps.at("1288_abab_vvoo"))(tmps.at("1288_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0273_bb_oo")(kb, jb))
      .allocate(tmps.at("1287_abab_vvoo"))(tmps.at("1287_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0271_bb_oo")(kb, jb))
      .allocate(tmps.at("1286_abab_vvoo"))(tmps.at("1286_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0089_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0089_baab_vooo"))
      .allocate(tmps.at("1285_baba_vvoo"))(tmps.at("1285_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0737_aa_vo")(ba, ja) * t1_2p.at("bb")(ab, ib))
      .allocate(tmps.at("1284_abba_vvoo"))(tmps.at("1284_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0385_aa_oo")(ka, ja) *
                                             t2.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1283_baab_vvoo"))(tmps.at("1283_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0105_abab_vooo")(ba, kb, ia, jb) *
                                             t1_1p.at("bb")(ab, kb))
      .deallocate(tmps.at("0105_abab_vooo"))
      .allocate(tmps.at("1282_abab_vvoo"))(tmps.at("1282_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0581_bb_vv")(bb, cb))
      .allocate(tmps.at("1281_abab_vvoo"))(tmps.at("1281_abab_vvoo")(aa, bb, ia, jb) =
                                             t2.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0564_bb_vv")(bb, cb))
      .allocate(tmps.at("1280_abab_vvoo"))(tmps.at("1280_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0959_abab_ovoo")(ka, bb, ia, jb))
      .deallocate(tmps.at("0959_abab_ovoo"))
      .allocate(tmps.at("1279_baab_vvoo"))(tmps.at("1279_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1137_abba_voov")(ba, kb, jb, ca) *
                                             t2.at("abab")(ca, ab, ia, kb))
      .deallocate(tmps.at("1137_abba_voov"))
      .allocate(tmps.at("1278_baba_vvoo"))(tmps.at("1278_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0039_aaaa_voov")(ba, ka, ja, ca) *
                                             t2_1p.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("1277_baba_vvoo"))(tmps.at("1277_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0379_aaaa_voov")(ba, ka, ja, ca) *
                                             t2.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("1276_abba_vvoo"))(tmps.at("1276_abba_vvoo")(aa, bb, ib, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0862_baba_vooo")(bb, ka, ib, ja))
      .deallocate(tmps.at("0862_baba_vooo"))
      .allocate(tmps.at("1275_baba_vvoo"))(tmps.at("1275_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0350_aaaa_vovo")(ba, ka, ca, ja) *
                                             t2.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("1274_baba_vvoo"))(tmps.at("1274_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0741_aa_vo")(ba, ja) * t1_2p.at("bb")(ab, ib))
      .allocate(tmps.at("1156_abab_vovo"))(tmps.at("1156_abab_vovo")(aa, ib, ba, jb) =
                                             eri.at("abab_vovv")(aa, ib, ba, cb) *
                                             t1_2p.at("bb")(cb, jb))
      .allocate(tmps.at("1273_baab_vvoo"))(tmps.at("1273_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1156_abab_vovo")(ba, kb, ca, jb) *
                                             t2.at("abab")(ca, ab, ia, kb))
      .deallocate(tmps.at("1156_abab_vovo"))
      .allocate(tmps.at("1271_aa_vv"))(tmps.at("1271_aa_vv")(aa, ba) =
                                         eri.at("aaaa_vovv")(aa, ia, ca, ba) *
                                         t1_1p.at("aa")(ca, ia))
      .allocate(tmps.at("1272_baab_vvoo"))(tmps.at("1272_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1271_aa_vv")(ba, ca) *
                                             t2_1p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1270_baab_vvoo"))(tmps.at("1270_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0374_aa_vv")(ba, ca) *
                                             t2_2p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1269_baab_vvoo"))(tmps.at("1269_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0086_abab_vooo")(ba, kb, ia, jb) *
                                             t1_2p.at("bb")(ab, kb))
      .deallocate(tmps.at("0086_abab_vooo"))
      .allocate(tmps.at("1268_baba_vvoo"))(tmps.at("1268_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0392_abba_vovo")(ba, kb, cb, ja) *
                                             t2_1p.at("bbbb")(cb, ab, ib, kb))
      .allocate(tmps.at("1267_abab_vvoo"))(tmps.at("1267_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0544_bb_vv")(bb, cb))
      .allocate(tmps.at("1266_baab_vvoo"))(tmps.at("1266_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0811_abab_vovo")(ba, kb, ca, jb) *
                                             t2_2p.at("abab")(ca, ab, ia, kb))
      .deallocate(tmps.at("0811_abab_vovo"))
      .allocate(tmps.at("1265_abba_vvoo"))(tmps.at("1265_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0344_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1264_baba_vvoo"))(tmps.at("1264_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("0346_aaaa_voov")(ba, ka, ja, ca) *
                                             t2_2p.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("1263_abba_vvoo"))(tmps.at("1263_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_1p.at("abab")(aa, cb, ka, ib) *
                                             tmps.at("0951_baba_vovo")(bb, ka, cb, ja))
      .deallocate(tmps.at("0951_baba_vovo"))
      .allocate(tmps.at("1262_abba_vvoo"))(tmps.at("1262_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0369_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1261_abab_vvoo"))(tmps.at("1261_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1260_bb_vo")(bb, jb))
      .allocate(tmps.at("1259_abab_vvoo"))(tmps.at("1259_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0137_baba_voov")(bb, ka, jb, ca))
      .allocate(tmps.at("1257_aaaa_vovo"))(tmps.at("1257_aaaa_vovo")(aa, ia, ba, ja) =
                                             eri.at("aaaa_vovv")(aa, ia, ca, ba) *
                                             t1_1p.at("aa")(ca, ja))
      .allocate(tmps.at("1258_baba_vvoo"))(tmps.at("1258_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("1257_aaaa_vovo")(ba, ka, ca, ja) *
                                             t2_1p.at("abab")(ca, ab, ka, ib))
      .allocate(tmps.at("1256_abab_vvoo"))(tmps.at("1256_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0108_baab_vooo")(bb, ka, ia, jb))
      .deallocate(tmps.at("0108_baab_vooo"))
      .allocate(tmps.at("1255_abba_vvoo"))(tmps.at("1255_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0394_aa_oo")(ka, ja) *
                                             t2_1p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1253_aa_vv"))(tmps.at("1253_aa_vv")(aa, ba) =
                                         eri.at("aaaa_oovv")(ia, ja, ca, ba) *
                                         t2_1p.at("aaaa")(ca, aa, ja, ia))
      .allocate(tmps.at("1254_baab_vvoo"))(tmps.at("1254_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("1253_aa_vv")(ba, ca) *
                                             t2_1p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1252_baab_vvoo"))(tmps.at("1252_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0226_aa_vv")(ba, ca) *
                                             t2_2p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1250_abab_voov"))(tmps.at("1250_abab_voov")(aa, ib, ja, bb) =
                                             t2_1p.at("abab")(aa, cb, ja, kb) *
                                             eri.at("bbbb_oovv")(kb, ib, cb, bb))
      .allocate(tmps.at("1251_baba_vvoo"))(tmps.at("1251_baba_vvoo")(ab, ba, ib, ja) =
                                             tmps.at("1250_abab_voov")(ba, kb, ja, cb) *
                                             t2_1p.at("bbbb")(cb, ab, ib, kb))
      .allocate(tmps.at("1249_baab_vvoo"))(tmps.at("1249_baab_vvoo")(ab, ba, ia, jb) =
                                             tmps.at("0041_aa_vv")(ba, ca) *
                                             t2_2p.at("abab")(ca, ab, ia, jb))
      .allocate(tmps.at("1248_abab_vvoo"))(tmps.at("1248_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, jb) *
                                             tmps.at("0583_bb_vv")(bb, cb))
      .allocate(tmps.at("1247_abab_vvoo"))(tmps.at("1247_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0015_bb_oo")(kb, jb))
      .allocate(tmps.at("1245_bbbb_vovo"))(tmps.at("1245_bbbb_vovo")(ab, ib, bb, jb) =
                                             eri.at("bbbb_vovv")(ab, ib, cb, bb) *
                                             t1_1p.at("bb")(cb, jb))
      .allocate(tmps.at("1246_abab_vvoo"))(tmps.at("1246_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("1245_bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("1244_abba_vvoo"))(tmps.at("1244_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0243_aa_oo")(ka, ja) *
                                             t2_2p.at("abab")(aa, bb, ka, ib))
      .allocate(tmps.at("1243_abba_vvoo"))(tmps.at("1243_abba_vvoo")(aa, bb, ib, ja) =
                                             tmps.at("0821_abba_vvvo")(aa, bb, cb, ja) *
                                             t1_2p.at("bb")(cb, ib))
      .deallocate(tmps.at("0821_abba_vvvo"))
      .allocate(tmps.at("1242_abab_vvoo"))(tmps.at("1242_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_1p.at("abab")(aa, bb, ia, kb) *
                                             tmps.at("0566_bb_oo")(kb, jb))
      .allocate(tmps.at("1178_abab_vvoo"))(tmps.at("1178_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0016")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("1177_abab_vvoo"))(tmps.at("1177_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, bb, ia, kb) *
                                             f.at("bb_oo")(kb, jb))
      .allocate(tmps.at("1176_abab_vvoo"))(tmps.at("1176_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vvvo")(aa, bb, cb, ia) *
                                             t1_2p.at("bb")(cb, jb))
      .allocate(tmps.at("1175_abab_vvoo"))(tmps.at("1175_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0002")() *
                                             t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("1174_abab_vvoo"))(tmps.at("1174_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0001")() *
                                             t2_2p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("1173_abab_vvoo"))(tmps.at("bin_abba_vvvo")(aa, bb, cb, ia) =
                                             eri.at("abab_vvvv")(aa, bb, da, cb) *
                                             t1_2p.at("aa")(da, ia))(
        tmps.at("1173_abab_vvoo")(aa, bb, ia, jb) =
          tmps.at("bin_abba_vvvo")(aa, bb, cb, ia) * t1.at("bb")(cb, jb))
      .allocate(tmps.at("1172_abab_vvoo"))(tmps.at("1172_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("aaaa_vovo")(aa, ka, ca, ia) *
                                             t2_2p.at("abab")(ca, bb, ka, jb))
      .allocate(tmps.at("1171_abab_vvoo"))(tmps.at("1171_abab_vvoo")(aa, bb, ia, jb) =
                                             scalars.at("0014")() *
                                             t2_1p.at("abab")(aa, bb, ia, jb))
      .allocate(tmps.at("1170_abab_vvoo"))(tmps.at("1170_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, jb) *
                                             f.at("bb_vv")(bb, cb))
      .allocate(tmps.at("1169_baab_vvoo"))(tmps.at("bin_bbaa_vvoo")(ab, cb, ia, ka) =
                                             eri.at("baab_vovv")(ab, ka, da, cb) *
                                             t1_2p.at("aa")(da, ia))(
        tmps.at("1169_baab_vvoo")(ab, ba, ia, jb) =
          tmps.at("bin_bbaa_vvoo")(ab, cb, ia, ka) * t2.at("abab")(ba, cb, ka, jb))
      .allocate(tmps.at("1168_abab_vvoo"))(tmps.at("1168_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_oooo")(ka, lb, ia, jb) *
                                             t2_2p.at("abab")(aa, bb, ka, lb))
      .allocate(tmps.at("1167_abab_vvoo"))(tmps.at("1167_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vooo")(aa, kb, ia, jb) *
                                             t1_2p.at("bb")(bb, kb))
      .allocate(tmps.at("1166_abba_vvoo"))(tmps.at("1166_abba_vvoo")(aa, bb, ib, ja) =
                                             t2_2p.at("abab")(aa, cb, ka, ib) *
                                             eri.at("baba_vovo")(bb, ka, cb, ja))
      .allocate(tmps.at("1165_abab_vvoo"))(tmps.at("1165_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_oo")(ka, ia) *
                                             t2_2p.at("abab")(aa, bb, ka, jb))
      .allocate(tmps.at("1164_abab_vvoo"))(tmps.at("1164_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abab_vvvv")(aa, bb, ca, db) *
                                             t2_2p.at("abab")(ca, db, ia, jb))
      .allocate(tmps.at("1163_abba_vvoo"))(tmps.at("1163_abba_vvoo")(aa, bb, ib, ja) =
                                             eri.at("abab_vovo")(aa, kb, ca, ib) *
                                             t2_2p.at("abab")(ca, bb, ja, kb))
      .allocate(tmps.at("1162_abab_vvoo"))(tmps.at("1162_abab_vvoo")(aa, bb, ia, jb) =
                                             eri.at("abba_vovo")(aa, kb, cb, ia) *
                                             t2_2p.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("1161_abab_vvoo"))(tmps.at("1161_abab_vvoo")(aa, bb, ia, jb) =
                                             f.at("aa_vv")(aa, ca) *
                                             t2_2p.at("abab")(ca, bb, ia, jb))
      .allocate(tmps.at("1160_abab_vvoo"))(tmps.at("1160_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(ca, ia) *
                                             eri.at("abab_vvvo")(aa, bb, ca, jb))
      .allocate(tmps.at("1159_abab_vvoo"))(tmps.at("1159_abab_vvoo")(aa, bb, ia, jb) =
                                             t1_2p.at("aa")(aa, ka) *
                                             eri.at("baab_vooo")(bb, ka, ia, jb))
      .allocate(tmps.at("1158_abab_vvoo"))(tmps.at("1158_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ka) *
                                             eri.at("baab_vovo")(bb, ka, ca, jb))
      .allocate(tmps.at("1157_abab_vvoo"))(tmps.at("1157_abab_vvoo")(aa, bb, ia, jb) =
                                             t2_2p.at("abab")(aa, cb, ia, kb) *
                                             eri.at("bbbb_vovo")(bb, kb, cb, jb))
      .allocate(tmps.at("1182_abab_vvoo"))(tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) =
                                             -1.00 * tmps.at("1157_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1167_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1161_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1159_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1178_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1173_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1160_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1158_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1168_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) +=
        2.00 * tmps.at("1174_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) +=
        2.00 * tmps.at("1175_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1177_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1172_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1162_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1165_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1176_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1166_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1170_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1171_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1169_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1164_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1182_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1163_abba_vvoo")(aa, bb, jb, ia))
      .deallocate(tmps.at("1178_abab_vvoo"))
      .deallocate(tmps.at("1177_abab_vvoo"))
      .deallocate(tmps.at("1176_abab_vvoo"))
      .deallocate(tmps.at("1175_abab_vvoo"))
      .deallocate(tmps.at("1174_abab_vvoo"))
      .deallocate(tmps.at("1173_abab_vvoo"))
      .deallocate(tmps.at("1172_abab_vvoo"))
      .deallocate(tmps.at("1171_abab_vvoo"))
      .deallocate(tmps.at("1170_abab_vvoo"))
      .deallocate(tmps.at("1169_baab_vvoo"))
      .deallocate(tmps.at("1168_abab_vvoo"))
      .deallocate(tmps.at("1167_abab_vvoo"))
      .deallocate(tmps.at("1166_abba_vvoo"))
      .deallocate(tmps.at("1165_abab_vvoo"))
      .deallocate(tmps.at("1164_abab_vvoo"))
      .deallocate(tmps.at("1163_abba_vvoo"))
      .deallocate(tmps.at("1162_abab_vvoo"))
      .deallocate(tmps.at("1161_abab_vvoo"))
      .deallocate(tmps.at("1160_abab_vvoo"))
      .deallocate(tmps.at("1159_abab_vvoo"))
      .deallocate(tmps.at("1158_abab_vvoo"))
      .deallocate(tmps.at("1157_abab_vvoo"))
      .allocate(tmps.at("1389_abab_vvoo"))(tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) =
                                             -0.50 * tmps.at("1267_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        0.50 * tmps.at("1298_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1248_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1264_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1367_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        0.50 * tmps.at("1370_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1327_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1372_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1251_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1308_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1243_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1295_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1279_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1297_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        3.00 * tmps.at("1369_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1373_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1300_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1262_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1288_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1309_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) +=
        0.50 * tmps.at("1351_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1281_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1338_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1380_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1342_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1348_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1340_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1313_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1312_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) +=
        2.00 * tmps.at("1285_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1375_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1318_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        3.00 * tmps.at("1247_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1301_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1182_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1361_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1296_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1289_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1280_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1303_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1331_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1249_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1339_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1347_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1268_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1368_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1357_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1299_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) +=
        0.50 * tmps.at("1254_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) +=
        0.50 * tmps.at("1324_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1293_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1359_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1284_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1332_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1276_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1355_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1314_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1265_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1272_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1246_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1322_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1349_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1273_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) +=
        0.50 * tmps.at("1319_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1346_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1366_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1362_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1244_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1336_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) +=
        0.50 * tmps.at("1282_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1315_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1242_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1255_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1345_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1261_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1263_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1287_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1266_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) +=
        0.50 * tmps.at("1353_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        3.00 * tmps.at("1283_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1343_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1259_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        3.00 * tmps.at("1305_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1376_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1379_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1306_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1252_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1371_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        3.00 * tmps.at("1286_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1304_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1352_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1277_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1321_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) +=
        0.50 * tmps.at("1270_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        3.00 * tmps.at("1317_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1302_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        0.50 * tmps.at("1294_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        3.00 * tmps.at("1269_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1381_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1341_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1307_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1316_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1325_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1377_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1360_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) +=
        2.00 * tmps.at("1356_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        2.00 * tmps.at("1344_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        3.00 * tmps.at("1256_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1337_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -=
        2.00 * tmps.at("1274_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1278_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1358_abab_vvoo")(aa, bb, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) += tmps.at("1258_baba_vvoo")(bb, aa, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1311_abba_vvoo")(aa, bb, jb, ia))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1310_baab_vvoo")(bb, aa, ia, jb))(
        tmps.at("1389_abab_vvoo")(aa, bb, ia, jb) -= tmps.at("1275_baba_vvoo")(bb, aa, jb, ia))
      .deallocate(tmps.at("1381_abab_vvoo"))
      .deallocate(tmps.at("1380_abab_vvoo"))
      .deallocate(tmps.at("1379_abab_vvoo"))
      .deallocate(tmps.at("1377_abab_vvoo"))
      .deallocate(tmps.at("1376_abab_vvoo"))
      .deallocate(tmps.at("1375_baba_vvoo"))
      .deallocate(tmps.at("1373_abba_vvoo"))
      .deallocate(tmps.at("1372_baab_vvoo"))
      .deallocate(tmps.at("1371_baab_vvoo"))
      .deallocate(tmps.at("1370_baab_vvoo"))
      .deallocate(tmps.at("1369_abba_vvoo"))
      .deallocate(tmps.at("1368_baab_vvoo"))
      .deallocate(tmps.at("1367_abab_vvoo"))
      .deallocate(tmps.at("1366_baba_vvoo"))
      .deallocate(tmps.at("1362_abab_vvoo"))
      .deallocate(tmps.at("1361_abab_vvoo"))
      .deallocate(tmps.at("1360_abab_vvoo"))
      .deallocate(tmps.at("1359_abab_vvoo"))
      .deallocate(tmps.at("1358_abab_vvoo"))
      .deallocate(tmps.at("1357_abba_vvoo"))
      .deallocate(tmps.at("1356_abab_vvoo"))
      .deallocate(tmps.at("1355_baba_vvoo"))
      .deallocate(tmps.at("1353_abab_vvoo"))
      .deallocate(tmps.at("1352_abab_vvoo"))
      .deallocate(tmps.at("1351_abba_vvoo"))
      .deallocate(tmps.at("1349_abab_vvoo"))
      .deallocate(tmps.at("1348_abab_vvoo"))
      .deallocate(tmps.at("1347_abab_vvoo"))
      .deallocate(tmps.at("1346_abba_vvoo"))
      .deallocate(tmps.at("1345_abab_vvoo"))
      .deallocate(tmps.at("1344_abab_vvoo"))
      .deallocate(tmps.at("1343_baab_vvoo"))
      .deallocate(tmps.at("1342_baba_vvoo"))
      .deallocate(tmps.at("1341_baab_vvoo"))
      .deallocate(tmps.at("1340_abab_vvoo"))
      .deallocate(tmps.at("1339_abab_vvoo"))
      .deallocate(tmps.at("1338_baba_vvoo"))
      .deallocate(tmps.at("1337_abab_vvoo"))
      .deallocate(tmps.at("1336_abab_vvoo"))
      .deallocate(tmps.at("1332_abba_vvoo"))
      .deallocate(tmps.at("1331_baab_vvoo"))
      .deallocate(tmps.at("1327_abab_vvoo"))
      .deallocate(tmps.at("1325_abab_vvoo"))
      .deallocate(tmps.at("1324_abab_vvoo"))
      .deallocate(tmps.at("1322_baab_vvoo"))
      .deallocate(tmps.at("1321_baba_vvoo"))
      .deallocate(tmps.at("1319_abba_vvoo"))
      .deallocate(tmps.at("1318_baab_vvoo"))
      .deallocate(tmps.at("1317_abab_vvoo"))
      .deallocate(tmps.at("1316_baab_vvoo"))
      .deallocate(tmps.at("1315_abab_vvoo"))
      .deallocate(tmps.at("1314_abab_vvoo"))
      .deallocate(tmps.at("1313_baba_vvoo"))
      .deallocate(tmps.at("1312_abab_vvoo"))
      .deallocate(tmps.at("1311_abba_vvoo"))
      .deallocate(tmps.at("1310_baab_vvoo"))
      .deallocate(tmps.at("1309_abba_vvoo"))
      .deallocate(tmps.at("1308_baab_vvoo"))
      .deallocate(tmps.at("1307_abab_vvoo"))
      .deallocate(tmps.at("1306_baab_vvoo"))
      .deallocate(tmps.at("1305_abba_vvoo"))
      .deallocate(tmps.at("1304_baab_vvoo"))
      .deallocate(tmps.at("1303_abab_vvoo"))
      .deallocate(tmps.at("1302_abab_vvoo"))
      .deallocate(tmps.at("1301_baba_vvoo"))
      .deallocate(tmps.at("1300_baab_vvoo"))
      .deallocate(tmps.at("1299_abab_vvoo"))
      .deallocate(tmps.at("1298_abab_vvoo"))
      .deallocate(tmps.at("1297_baba_vvoo"))
      .deallocate(tmps.at("1296_abab_vvoo"))
      .deallocate(tmps.at("1295_abab_vvoo"))
      .deallocate(tmps.at("1294_abba_vvoo"))
      .deallocate(tmps.at("1293_abab_vvoo"))
      .deallocate(tmps.at("1289_baba_vvoo"))
      .deallocate(tmps.at("1288_abab_vvoo"))
      .deallocate(tmps.at("1287_abab_vvoo"))
      .deallocate(tmps.at("1286_abab_vvoo"))
      .deallocate(tmps.at("1285_baba_vvoo"))
      .deallocate(tmps.at("1284_abba_vvoo"))
      .deallocate(tmps.at("1283_baab_vvoo"))
      .deallocate(tmps.at("1282_abab_vvoo"))
      .deallocate(tmps.at("1281_abab_vvoo"))
      .deallocate(tmps.at("1280_abab_vvoo"))
      .deallocate(tmps.at("1279_baab_vvoo"))
      .deallocate(tmps.at("1278_baba_vvoo"))
      .deallocate(tmps.at("1277_baba_vvoo"))
      .deallocate(tmps.at("1276_abba_vvoo"))
      .deallocate(tmps.at("1275_baba_vvoo"))
      .deallocate(tmps.at("1274_baba_vvoo"))
      .deallocate(tmps.at("1273_baab_vvoo"))
      .deallocate(tmps.at("1272_baab_vvoo"))
      .deallocate(tmps.at("1270_baab_vvoo"))
      .deallocate(tmps.at("1269_baab_vvoo"))
      .deallocate(tmps.at("1268_baba_vvoo"))
      .deallocate(tmps.at("1267_abab_vvoo"))
      .deallocate(tmps.at("1266_baab_vvoo"))
      .deallocate(tmps.at("1265_abba_vvoo"))
      .deallocate(tmps.at("1264_baba_vvoo"))
      .deallocate(tmps.at("1263_abba_vvoo"))
      .deallocate(tmps.at("1262_abba_vvoo"))
      .deallocate(tmps.at("1261_abab_vvoo"))
      .deallocate(tmps.at("1259_abab_vvoo"))
      .deallocate(tmps.at("1258_baba_vvoo"))
      .deallocate(tmps.at("1256_abab_vvoo"))
      .deallocate(tmps.at("1255_abba_vvoo"))
      .deallocate(tmps.at("1254_baab_vvoo"))
      .deallocate(tmps.at("1252_baab_vvoo"))
      .deallocate(tmps.at("1251_baba_vvoo"))
      .deallocate(tmps.at("1249_baab_vvoo"))
      .deallocate(tmps.at("1248_abab_vvoo"))
      .deallocate(tmps.at("1247_abab_vvoo"))
      .deallocate(tmps.at("1246_abab_vvoo"))
      .deallocate(tmps.at("1244_abba_vvoo"))
      .deallocate(tmps.at("1243_abba_vvoo"))
      .deallocate(tmps.at("1242_abab_vvoo"))
      .deallocate(tmps.at("1182_abab_vvoo"))
      .allocate(tmps.at("1439_baba_vvoo"))(tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) =
                                             -1.00 * tmps.at("1389_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1400_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1407_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1412_baba_vvoo")(ab, ba, ib, ja))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1435_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1432_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1417_baab_vvoo")(ab, ba, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1395_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1428_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1427_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1403_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1410_baab_vvoo")(ab, ba, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1420_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1425_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1415_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1409_baba_vvoo")(ab, ba, ib, ja))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1402_baab_vvoo")(ab, ba, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1426_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1436_baab_vvoo")(ab, ba, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1424_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) +=
        2.00 * tmps.at("1414_baba_vvoo")(ab, ba, ib, ja))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1404_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1433_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1419_baab_vvoo")(ab, ba, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1430_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1413_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1431_baab_vvoo")(ab, ba, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1398_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1437_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1421_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1418_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1401_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1405_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1397_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) +=
        2.00 * tmps.at("1423_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1422_abba_vvoo")(ba, ab, ib, ja))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1434_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1429_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) -= tmps.at("1416_baab_vvoo")(ab, ba, ja, ib))(
        tmps.at("1439_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1394_baba_vvoo")(ab, ba, ib, ja))
      .deallocate(tmps.at("1437_abab_vvoo"))
      .deallocate(tmps.at("1436_baab_vvoo"))
      .deallocate(tmps.at("1435_abab_vvoo"))
      .deallocate(tmps.at("1434_abab_vvoo"))
      .deallocate(tmps.at("1433_abab_vvoo"))
      .deallocate(tmps.at("1432_abab_vvoo"))
      .deallocate(tmps.at("1431_baab_vvoo"))
      .deallocate(tmps.at("1430_abab_vvoo"))
      .deallocate(tmps.at("1429_abab_vvoo"))
      .deallocate(tmps.at("1428_abba_vvoo"))
      .deallocate(tmps.at("1427_abab_vvoo"))
      .deallocate(tmps.at("1426_abab_vvoo"))
      .deallocate(tmps.at("1425_abab_vvoo"))
      .deallocate(tmps.at("1424_abab_vvoo"))
      .deallocate(tmps.at("1423_abab_vvoo"))
      .deallocate(tmps.at("1422_abba_vvoo"))
      .deallocate(tmps.at("1421_abab_vvoo"))
      .deallocate(tmps.at("1420_abab_vvoo"))
      .deallocate(tmps.at("1419_baab_vvoo"))
      .deallocate(tmps.at("1418_abba_vvoo"))
      .deallocate(tmps.at("1417_baab_vvoo"))
      .deallocate(tmps.at("1416_baab_vvoo"))
      .deallocate(tmps.at("1415_abba_vvoo"))
      .deallocate(tmps.at("1414_baba_vvoo"))
      .deallocate(tmps.at("1413_abba_vvoo"))
      .deallocate(tmps.at("1412_baba_vvoo"))
      .deallocate(tmps.at("1410_baab_vvoo"))
      .deallocate(tmps.at("1409_baba_vvoo"))
      .deallocate(tmps.at("1407_abab_vvoo"))
      .deallocate(tmps.at("1405_abab_vvoo"))
      .deallocate(tmps.at("1404_abba_vvoo"))
      .deallocate(tmps.at("1403_abba_vvoo"))
      .deallocate(tmps.at("1402_baab_vvoo"))
      .deallocate(tmps.at("1401_abab_vvoo"))
      .deallocate(tmps.at("1400_abab_vvoo"))
      .deallocate(tmps.at("1398_abab_vvoo"))
      .deallocate(tmps.at("1397_abab_vvoo"))
      .deallocate(tmps.at("1395_abab_vvoo"))
      .deallocate(tmps.at("1394_baba_vvoo"))
      .deallocate(tmps.at("1389_abab_vvoo"))
      .allocate(tmps.at("1441_baba_vvoo"))(tmps.at("1441_baba_vvoo")(ab, ba, ib, ja) =
                                             -1.00 * tmps.at("1440_abab_vvoo")(ba, ab, ja, ib))(
        tmps.at("1441_baba_vvoo")(ab, ba, ib, ja) += tmps.at("1439_baba_vvoo")(ab, ba, ib, ja))
      .deallocate(tmps.at("1440_abab_vvoo"))
      .deallocate(tmps.at("1439_baba_vvoo"))(r2_2p.at("abab")(aa, bb, ia, jb) -=
                                             2.00 * tmps.at("1441_baba_vvoo")(bb, aa, jb, ia))
      .deallocate(tmps.at("1441_baba_vvoo"))
      .allocate(tmps.at("1453_bbbb_oooo"))(tmps.at("1453_bbbb_oooo")(ib, jb, kb, lb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0127_bbbb_oovo")(ib, jb, ab, lb))
      .deallocate(tmps.at("0127_bbbb_oovo"))
      .allocate(tmps.at("1463_bbbb_vooo"))(tmps.at("1463_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("1453_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1471_bbbb_vvoo"))(tmps.at("1471_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1463_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1463_bbbb_vooo"))
      .allocate(tmps.at("1464_bbbb_vooo"))(tmps.at("1464_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0323_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1470_bbbb_vvoo"))(tmps.at("1470_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1464_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1464_bbbb_vooo"))
      .allocate(tmps.at("1468_bbbb_vvoo"))(tmps.at("1468_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0323_bbbb_oooo")(lb, kb, ib, jb))
      .allocate(tmps.at("1455_bbbb_vooo"))(tmps.at("1455_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0318_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1467_bbbb_vvoo"))(tmps.at("1467_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1455_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1455_bbbb_vooo"))
      .allocate(tmps.at("1442_bbbb_oooo"))(tmps.at("1442_bbbb_oooo")(ib, jb, kb, lb) =
                                             eri.at("bbbb_oovv")(ib, jb, ab, bb) *
                                             t2_2p.at("bbbb")(ab, bb, kb, lb))
      .allocate(tmps.at("1454_bbbb_vooo"))(tmps.at("1454_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("1442_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1466_bbbb_vvoo"))(tmps.at("1466_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1454_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1454_bbbb_vooo"))
      .allocate(tmps.at("1465_bbbb_vvoo"))(tmps.at("1465_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("1453_bbbb_oooo")(lb, kb, ib, jb))
      .deallocate(tmps.at("1453_bbbb_oooo"))
      .allocate(tmps.at("1461_bbbb_vvoo"))(tmps.at("1461_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0320_bbbb_oooo")(lb, kb, ib, jb))
      .deallocate(tmps.at("0320_bbbb_oooo"))
      .allocate(tmps.at("1444_bbbb_vooo"))(tmps.at("1444_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             eri.at("bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1460_bbbb_vvoo"))(tmps.at("1460_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1444_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1444_bbbb_vooo"))
      .allocate(tmps.at("1459_bbbb_vvoo"))(tmps.at("1459_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("1442_bbbb_oooo")(lb, kb, ib, jb))
      .deallocate(tmps.at("1442_bbbb_oooo"))
      .allocate(tmps.at("1458_bbbb_vvoo"))(tmps.at("1458_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0318_bbbb_oooo")(lb, kb, ib, jb))
      .allocate(tmps.at("1457_bbbb_vvoo"))(tmps.at("1457_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0544_bb_vv")(bb, cb))
      .allocate(tmps.at("1443_bb_vv"))(tmps.at("1443_bb_vv")(ab, bb) =
                                         eri.at("bbbb_oovv")(ib, jb, cb, bb) *
                                         t2_1p.at("bbbb")(cb, ab, jb, ib))
      .allocate(tmps.at("1456_bbbb_vvoo"))(tmps.at("1456_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("1443_bb_vv")(bb, cb))
      .deallocate(tmps.at("1443_bb_vv"))
      .allocate(tmps.at("1451_bbbb_vvoo"))(tmps.at("1451_bbbb_vvoo")(ab, bb, ib, jb) =
                                             scalars.at("0002")() *
                                             t2_2p.at("bbbb")(ab, bb, ib, jb))
      .allocate(tmps.at("1450_bbbb_vvoo"))(tmps.at("1450_bbbb_vvoo")(ab, bb, ib, jb) =
                                             scalars.at("0001")() *
                                             t2_2p.at("bbbb")(ab, bb, ib, jb))
      .allocate(tmps.at("1449_bbbb_vvoo"))(tmps.at("1449_bbbb_vvoo")(ab, bb, ib, jb) =
                                             scalars.at("0016")() *
                                             t2_1p.at("bbbb")(ab, bb, ib, jb))
      .allocate(tmps.at("1448_bbbb_vvoo"))(tmps.at("bin_bbbb_vvvo")(ab, bb, cb, ib) =
                                             eri.at("bbbb_vvvv")(ab, bb, cb, db) *
                                             t1_1p.at("bb")(db, ib))(
        tmps.at("1448_bbbb_vvoo")(ab, bb, ib, jb) =
          tmps.at("bin_bbbb_vvvo")(ab, bb, cb, ib) * t1_1p.at("bb")(cb, jb))
      .allocate(tmps.at("1447_bbbb_vvoo"))(tmps.at("1447_bbbb_vvoo")(ab, bb, ib, jb) =
                                             scalars.at("0014")() *
                                             t2_1p.at("bbbb")(ab, bb, ib, jb))
      .allocate(tmps.at("1446_bbbb_vvoo"))(tmps.at("1446_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_oooo")(kb, lb, ib, jb) *
                                             t2_2p.at("bbbb")(ab, bb, lb, kb))
      .allocate(tmps.at("1445_bbbb_vvoo"))(tmps.at("1445_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_vvvv")(ab, bb, cb, db) *
                                             t2_2p.at("bbbb")(cb, db, ib, jb))
      .allocate(tmps.at("1452_bbbb_vvoo"))(tmps.at("1452_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -0.250 * tmps.at("1446_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1452_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("1448_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1452_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.250 * tmps.at("1445_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1452_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1450_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1452_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1451_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1452_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 *
        tmps.at("1447_bbbb_vvoo")(ab, bb, ib, jb))(tmps.at("1452_bbbb_vvoo")(ab, bb, ib, jb) +=
                                                   0.50 * tmps.at("1449_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1451_bbbb_vvoo"))
      .deallocate(tmps.at("1450_bbbb_vvoo"))
      .deallocate(tmps.at("1449_bbbb_vvoo"))
      .deallocate(tmps.at("1448_bbbb_vvoo"))
      .deallocate(tmps.at("1447_bbbb_vvoo"))
      .deallocate(tmps.at("1446_bbbb_vvoo"))
      .deallocate(tmps.at("1445_bbbb_vvoo"));
  }
}

template void exachem::cc::qed_ccsd_os::resid_6<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);