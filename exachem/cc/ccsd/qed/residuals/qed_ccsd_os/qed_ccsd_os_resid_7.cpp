/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_os_resid_7.hpp"

template<typename T>
void exachem::cc::qed_ccsd_os::resid_7(
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
      .allocate(tmps.at("1462_bbbb_vvoo"))(tmps.at("1462_bbbb_vvoo")(ab, bb, ib, jb) =
                                             2.00 * tmps.at("1460_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1462_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("1461_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1462_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1456_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1462_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("1459_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1462_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1457_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1462_bbbb_vvoo")(ab, bb, ib, jb) +=
        4.00 *
        tmps.at("1452_bbbb_vvoo")(bb, ab, ib, jb))(tmps.at("1462_bbbb_vvoo")(ab, bb, ib, jb) -=
                                                   0.50 * tmps.at("1458_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1461_bbbb_vvoo"))
      .deallocate(tmps.at("1460_bbbb_vvoo"))
      .deallocate(tmps.at("1459_bbbb_vvoo"))
      .deallocate(tmps.at("1458_bbbb_vvoo"))
      .deallocate(tmps.at("1457_bbbb_vvoo"))
      .deallocate(tmps.at("1456_bbbb_vvoo"))
      .deallocate(tmps.at("1452_bbbb_vvoo"))
      .allocate(tmps.at("1469_bbbb_vvoo"))(tmps.at("1469_bbbb_vvoo")(ab, bb, ib, jb) =
                                             tmps.at("1465_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1469_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1462_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1469_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1468_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1469_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1467_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1469_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1466_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1468_bbbb_vvoo"))
      .deallocate(tmps.at("1467_bbbb_vvoo"))
      .deallocate(tmps.at("1466_bbbb_vvoo"))
      .deallocate(tmps.at("1465_bbbb_vvoo"))
      .deallocate(tmps.at("1462_bbbb_vvoo"))
      .allocate(tmps.at("1472_bbbb_vvoo"))(tmps.at("1472_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -2.00 * tmps.at("1471_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("1472_bbbb_vvoo")(ab, bb, ib, jb) +=
        tmps.at("1469_bbbb_vvoo")(ab, bb, ib, jb))(tmps.at("1472_bbbb_vvoo")(ab, bb, ib, jb) -=
                                                   2.00 * tmps.at("1470_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1471_bbbb_vvoo"))
      .deallocate(tmps.at("1470_bbbb_vvoo"))
      .deallocate(tmps.at("1469_bbbb_vvoo"))(r2_2p.at("bbbb")(ab, bb, ib, jb) +=
                                             tmps.at("1472_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1472_bbbb_vvoo"))
      .allocate(tmps.at("1492_aaaa_oooo"))(tmps.at("1492_aaaa_oooo")(ia, ja, ka, la) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0203_aaaa_oovo")(ia, ja, aa, la))
      .allocate(tmps.at("1500_aaaa_vooo"))(tmps.at("1500_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             tmps.at("1492_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1501_aaaa_vvoo"))(tmps.at("1501_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1500_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1498_aaaa_vvoo"))(tmps.at("1498_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0413_aa_oo")(ka, ja))
      .allocate(tmps.at("1497_aaaa_vvoo"))(tmps.at("1497_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0248_aa_oo")(ka, ja))
      .allocate(tmps.at("1496_aaaa_vvoo"))(tmps.at("1496_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0422_aa_oo")(ka, ja))
      .allocate(tmps.at("1486_aaaa_oooo"))(tmps.at("1486_aaaa_oooo")(ia, ja, ka, la) =
                                             t1_1p.at("aa")(aa, ka) *
                                             eri.at("aaaa_oovo")(ia, ja, aa, la))
      .allocate(tmps.at("1494_aaaa_vooo"))(tmps.at("1494_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             tmps.at("1486_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1495_aaaa_vvoo"))(tmps.at("1495_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1494_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1493_aaaa_vvoo"))(tmps.at("1493_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("1492_aaaa_oooo")(la, ka, ia, ja))
      .allocate(tmps.at("1491_aaaa_vvoo"))(tmps.at("1491_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0476_aa_oo")(ka, ja))
      .deallocate(tmps.at("0476_aa_oo"))
      .allocate(tmps.at("1489_aaaa_vvoo"))(tmps.at("1489_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0681_aaaa_oooo")(la, ka, ia, ja))
      .allocate(tmps.at("1488_aaaa_vvoo"))(tmps.at("1488_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0241_aa_oo")(ka, ja))
      .allocate(tmps.at("1487_aaaa_vvoo"))(tmps.at("1487_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("1486_aaaa_oooo")(la, ka, ia, ja))
      .allocate(tmps.at("1485_aaaa_vvoo"))(tmps.at("1485_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0394_aa_oo")(ka, ja))
      .allocate(tmps.at("1484_aaaa_vvoo"))(tmps.at("1484_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0344_aa_oo")(ka, ja))
      .allocate(tmps.at("1482_aaaa_vvvo"))(tmps.at("1482_aaaa_vvvo")(aa, ba, ca, ia) =
                                             eri.at("aaaa_vvvv")(aa, ba, da, ca) *
                                             t1.at("aa")(da, ia))
      .allocate(tmps.at("1483_aaaa_vvoo"))(tmps.at("1483_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(ca, ia) *
                                             tmps.at("1482_aaaa_vvvo")(aa, ba, ca, ja))
      .allocate(tmps.at("1481_aaaa_vvoo"))(tmps.at("1481_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0390_aa_oo")(ka, ja))
      .allocate(tmps.at("1480_aaaa_vvoo"))(tmps.at("1480_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0360_aa_oo")(ka, ja))
      .deallocate(tmps.at("0360_aa_oo"))
      .allocate(tmps.at("1479_aaaa_vvoo"))(tmps.at("1479_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0369_aa_oo")(ka, ja))
      .allocate(tmps.at("1478_aaaa_vvoo"))(tmps.at("1478_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0234_aa_oo")(ka, ja))
      .allocate(tmps.at("1477_aaaa_vvoo"))(tmps.at("1477_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0243_aa_oo")(ka, ja))
      .allocate(tmps.at("1476_aaaa_vvoo"))(tmps.at("1476_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0456_aa_oo")(ka, ja))
      .deallocate(tmps.at("0456_aa_oo"))
      .allocate(tmps.at("1474_aaaa_vvoo"))(tmps.at("1474_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(ca, ia) *
                                             eri.at("aaaa_vvvo")(aa, ba, ca, ja))
      .allocate(tmps.at("1473_aaaa_vvoo"))(tmps.at("1473_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             f.at("aa_oo")(ka, ja))
      .allocate(tmps.at("1475_aaaa_vvoo"))(tmps.at("1475_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("1474_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1475_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1473_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1474_aaaa_vvoo"))
      .deallocate(tmps.at("1473_aaaa_vvoo"))
      .allocate(tmps.at("1490_aaaa_vvoo"))(tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.50 * tmps.at("1481_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1485_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.50 * tmps.at("1487_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1477_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1488_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1478_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1479_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1475_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1484_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.50 * tmps.at("1489_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1483_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1480_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1476_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1489_aaaa_vvoo"))
      .deallocate(tmps.at("1488_aaaa_vvoo"))
      .deallocate(tmps.at("1487_aaaa_vvoo"))
      .deallocate(tmps.at("1485_aaaa_vvoo"))
      .deallocate(tmps.at("1484_aaaa_vvoo"))
      .deallocate(tmps.at("1483_aaaa_vvoo"))
      .deallocate(tmps.at("1481_aaaa_vvoo"))
      .deallocate(tmps.at("1480_aaaa_vvoo"))
      .deallocate(tmps.at("1479_aaaa_vvoo"))
      .deallocate(tmps.at("1478_aaaa_vvoo"))
      .deallocate(tmps.at("1477_aaaa_vvoo"))
      .deallocate(tmps.at("1476_aaaa_vvoo"))
      .deallocate(tmps.at("1475_aaaa_vvoo"))
      .allocate(tmps.at("1499_aaaa_vvoo"))(tmps.at("1499_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.50 * tmps.at("1493_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1499_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1490_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1499_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1498_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1499_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1495_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1499_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1491_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1499_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1496_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1499_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1497_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1498_aaaa_vvoo"))
      .deallocate(tmps.at("1497_aaaa_vvoo"))
      .deallocate(tmps.at("1496_aaaa_vvoo"))
      .deallocate(tmps.at("1495_aaaa_vvoo"))
      .deallocate(tmps.at("1493_aaaa_vvoo"))
      .deallocate(tmps.at("1491_aaaa_vvoo"))
      .deallocate(tmps.at("1490_aaaa_vvoo"))
      .allocate(tmps.at("1502_aaaa_vvoo"))(tmps.at("1502_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1499_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1502_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1501_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1501_aaaa_vvoo"))
      .deallocate(tmps.at("1499_aaaa_vvoo"))(r2_1p.at("aaaa")(aa, ba, ia, ja) -=
                                             tmps.at("1502_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_1p.at("aaaa")(aa, ba, ia, ja) += tmps.at("1502_aaaa_vvoo")(aa, ba, ja, ia))
      .deallocate(tmps.at("1502_aaaa_vvoo"))
      .allocate(tmps.at("1509_bbbb_oooo"))(tmps.at("1509_bbbb_oooo")(ib, jb, kb, lb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0256_bbbb_oovo")(ib, jb, ab, lb))
      .allocate(tmps.at("1534_bbbb_vooo"))(tmps.at("1534_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("1509_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1548_bbbb_vvoo"))(tmps.at("1548_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1534_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1534_bbbb_vooo"))
      .allocate(tmps.at("1546_bbbb_vvoo"))(tmps.at("1546_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0586_bb_oo")(kb, jb))
      .deallocate(tmps.at("0586_bb_oo"))
      .allocate(tmps.at("1545_bbbb_vvoo"))(tmps.at("1545_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0590_bb_oo")(kb, jb))
      .deallocate(tmps.at("0590_bb_oo"))
      .allocate(tmps.at("1544_bbbb_vvoo"))(tmps.at("1544_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("1509_bbbb_oooo")(lb, kb, ib, jb))
      .deallocate(tmps.at("1509_bbbb_oooo"))
      .allocate(tmps.at("1543_bbbb_vvoo"))(tmps.at("1543_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0596_bb_oo")(kb, jb))
      .allocate(tmps.at("1542_bbbb_vvoo"))(tmps.at("1542_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0608_bb_oo")(kb, jb))
      .deallocate(tmps.at("0608_bb_oo"))
      .allocate(tmps.at("1541_bbbb_vvoo"))(tmps.at("1541_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0604_bb_oo")(kb, jb))
      .deallocate(tmps.at("0604_bb_oo"))
      .allocate(tmps.at("1540_bbbb_vvoo"))(tmps.at("1540_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0289_bb_oo")(kb, jb))
      .allocate(tmps.at("1503_bbbb_oooo"))(tmps.at("1503_bbbb_oooo")(ib, jb, kb, lb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             eri.at("bbbb_oovo")(ib, jb, ab, lb))
      .allocate(tmps.at("1510_bbbb_vooo"))(tmps.at("1510_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("1503_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1539_bbbb_vvoo"))(tmps.at("1539_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1510_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1510_bbbb_vooo"))
      .allocate(tmps.at("1511_bbbb_vooo"))(tmps.at("1511_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1_1p.at("bb")(ab, lb) *
                                             tmps.at("0178_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1538_bbbb_vvoo"))(tmps.at("1538_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1511_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1511_bbbb_vooo"))
      .allocate(tmps.at("1536_bbbb_oooo"))(tmps.at("1536_bbbb_oooo")(ib, jb, kb, lb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0256_bbbb_oovo")(ib, jb, ab, lb))
      .allocate(tmps.at("1537_bbbb_vvoo"))(tmps.at("1537_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("1536_bbbb_oooo")(lb, kb, ib, jb))
      .allocate(tmps.at("1535_bbbb_vvoo"))(tmps.at("1535_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0598_bb_oo")(kb, jb))
      .allocate(tmps.at("1532_bbbb_vvoo"))(tmps.at("1532_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0273_bb_oo")(kb, jb))
      .allocate(tmps.at("1531_bbbb_vvoo"))(tmps.at("1531_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("1323_bb_oo")(kb, jb))
      .deallocate(tmps.at("1323_bb_oo"))
      .allocate(tmps.at("1530_bbbb_vvoo"))(tmps.at("1530_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0281_bb_oo")(kb, jb))
      .allocate(tmps.at("1529_bbbb_vvoo"))(tmps.at("1529_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0572_bb_oo")(kb, jb))
      .deallocate(tmps.at("0572_bb_oo"))
      .allocate(tmps.at("1528_bbbb_vvoo"))(tmps.at("1528_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0532_bb_oo")(kb, jb))
      .deallocate(tmps.at("0532_bb_oo"))
      .allocate(tmps.at("1527_bbbb_vvoo"))(tmps.at("1527_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0016_bb_oo")(kb, jb))
      .deallocate(tmps.at("0016_bb_oo"))
      .allocate(tmps.at("1525_bbbb_vvvo"))(tmps.at("1525_bbbb_vvvo")(ab, bb, cb, ib) =
                                             eri.at("bbbb_vvvv")(ab, bb, db, cb) *
                                             t1.at("bb")(db, ib))
      .allocate(tmps.at("1526_bbbb_vvoo"))(tmps.at("1526_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(cb, ib) *
                                             tmps.at("1525_bbbb_vvvo")(ab, bb, cb, jb))
      .allocate(tmps.at("1524_bbbb_vvoo"))(tmps.at("1524_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0566_bb_oo")(kb, jb))
      .allocate(tmps.at("1523_bbbb_vvoo"))(tmps.at("1523_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0015_bb_oo")(kb, jb))
      .deallocate(tmps.at("0015_bb_oo"))
      .allocate(tmps.at("1507_bbbb_voov"))(tmps.at("1507_bbbb_voov")(ab, ib, jb, bb) =
                                             t2_1p.at("bbbb")(cb, ab, jb, kb) *
                                             eri.at("bbbb_oovv")(kb, ib, cb, bb))
      .allocate(tmps.at("1522_bbbb_vvoo"))(tmps.at("1522_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("1507_bbbb_voov")(bb, kb, jb, cb))
      .deallocate(tmps.at("1507_bbbb_voov"))
      .allocate(tmps.at("1521_bbbb_vvoo"))(tmps.at("1521_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0577_bb_oo")(kb, jb))
      .deallocate(tmps.at("0577_bb_oo"))
      .allocate(tmps.at("1520_bbbb_vvoo"))(tmps.at("1520_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0271_bb_oo")(kb, jb))
      .allocate(tmps.at("1519_bbbb_vvoo"))(tmps.at("1519_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0528_bb_oo")(kb, jb))
      .allocate(tmps.at("1504_baba_voov"))(tmps.at("1504_baba_voov")(ab, ia, jb, ba) =
                                             t2_1p.at("abab")(ca, ab, ka, jb) *
                                             eri.at("aaaa_oovv")(ka, ia, ca, ba))
      .allocate(tmps.at("1518_bbbb_vvoo"))(tmps.at("1518_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("1504_baba_voov")(bb, ka, jb, ca))
      .deallocate(tmps.at("1504_baba_voov"))
      .allocate(tmps.at("1517_bbbb_vvoo"))(tmps.at("1517_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("1503_bbbb_oooo")(lb, kb, ib, jb))
      .deallocate(tmps.at("1503_bbbb_oooo"))
      .allocate(tmps.at("1515_bbbb_oooo"))(tmps.at("1515_bbbb_oooo")(ib, jb, kb, lb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             eri.at("bbbb_oovo")(ib, jb, ab, lb))
      .allocate(tmps.at("1516_bbbb_vvoo"))(tmps.at("1516_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("1515_bbbb_oooo")(lb, kb, ib, jb))
      .allocate(tmps.at("1514_bbbb_vvoo"))(tmps.at("1514_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0178_bbbb_oooo")(lb, kb, ib, jb))
      .allocate(tmps.at("1513_bbbb_vvoo"))(tmps.at("1513_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0276_bb_oo")(kb, jb))
      .allocate(tmps.at("1512_bbbb_vvoo"))(tmps.at("1512_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0568_bb_oo")(kb, jb))
      .allocate(tmps.at("1506_bbbb_vvoo"))(tmps.at("1506_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(ab, bb, ib, kb) *
                                             f.at("bb_oo")(kb, jb))
      .allocate(tmps.at("1505_bbbb_vvoo"))(tmps.at("1505_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(cb, ib) *
                                             eri.at("bbbb_vvvo")(ab, bb, cb, jb))
      .allocate(tmps.at("1508_bbbb_vvoo"))(tmps.at("1508_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1505_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1508_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1506_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1506_bbbb_vvoo"))
      .deallocate(tmps.at("1505_bbbb_vvoo"))
      .allocate(tmps.at("1533_bbbb_vvoo"))(tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) =
                                             tmps.at("1518_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) +=
        3.00 * tmps.at("1523_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1529_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1526_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("1513_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) -=
        3.00 * tmps.at("1527_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1522_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1519_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1512_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1508_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("1531_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1524_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1520_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("1516_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("1514_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1528_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("1521_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1530_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) +=
        tmps.at("1532_bbbb_vvoo")(bb, ab, ib, jb))(tmps.at("1533_bbbb_vvoo")(ab, bb, ib, jb) +=
                                                   0.50 * tmps.at("1517_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1532_bbbb_vvoo"))
      .deallocate(tmps.at("1531_bbbb_vvoo"))
      .deallocate(tmps.at("1530_bbbb_vvoo"))
      .deallocate(tmps.at("1529_bbbb_vvoo"))
      .deallocate(tmps.at("1528_bbbb_vvoo"))
      .deallocate(tmps.at("1527_bbbb_vvoo"))
      .deallocate(tmps.at("1526_bbbb_vvoo"))
      .deallocate(tmps.at("1524_bbbb_vvoo"))
      .deallocate(tmps.at("1523_bbbb_vvoo"))
      .deallocate(tmps.at("1522_bbbb_vvoo"))
      .deallocate(tmps.at("1521_bbbb_vvoo"))
      .deallocate(tmps.at("1520_bbbb_vvoo"))
      .deallocate(tmps.at("1519_bbbb_vvoo"))
      .deallocate(tmps.at("1518_bbbb_vvoo"))
      .deallocate(tmps.at("1517_bbbb_vvoo"))
      .deallocate(tmps.at("1516_bbbb_vvoo"))
      .deallocate(tmps.at("1514_bbbb_vvoo"))
      .deallocate(tmps.at("1513_bbbb_vvoo"))
      .deallocate(tmps.at("1512_bbbb_vvoo"))
      .deallocate(tmps.at("1508_bbbb_vvoo"))
      .allocate(tmps.at("1547_bbbb_vvoo"))(tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1535_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1540_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1546_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("1544_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1545_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1539_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1541_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1533_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1542_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1538_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("1537_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1543_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1546_bbbb_vvoo"))
      .deallocate(tmps.at("1545_bbbb_vvoo"))
      .deallocate(tmps.at("1544_bbbb_vvoo"))
      .deallocate(tmps.at("1543_bbbb_vvoo"))
      .deallocate(tmps.at("1542_bbbb_vvoo"))
      .deallocate(tmps.at("1541_bbbb_vvoo"))
      .deallocate(tmps.at("1540_bbbb_vvoo"))
      .deallocate(tmps.at("1539_bbbb_vvoo"))
      .deallocate(tmps.at("1538_bbbb_vvoo"))
      .deallocate(tmps.at("1537_bbbb_vvoo"))
      .deallocate(tmps.at("1535_bbbb_vvoo"))
      .deallocate(tmps.at("1533_bbbb_vvoo"))
      .allocate(tmps.at("1549_bbbb_vvoo"))(tmps.at("1549_bbbb_vvoo")(ab, bb, ib, jb) =
                                             tmps.at("1547_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1549_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1548_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1548_bbbb_vvoo"))
      .deallocate(tmps.at("1547_bbbb_vvoo"))(r2_2p.at("bbbb")(ab, bb, ib, jb) +=
                                             2.00 * tmps.at("1549_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= 2.00 * tmps.at("1549_bbbb_vvoo")(ab, bb, jb, ib))
      .deallocate(tmps.at("1549_bbbb_vvoo"))
      .allocate(tmps.at("1554_aaaa_oooo"))(tmps.at("1554_aaaa_oooo")(ia, ja, ka, la) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0203_aaaa_oovo")(ia, ja, aa, la))
      .allocate(tmps.at("1577_aaaa_vooo"))(tmps.at("1577_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             tmps.at("1554_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1590_aaaa_vvoo"))(tmps.at("1590_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1577_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1577_aaaa_vooo"))
      .allocate(tmps.at("1588_aaaa_vvoo"))(tmps.at("1588_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0422_aa_oo")(ka, ja))
      .deallocate(tmps.at("0422_aa_oo"))
      .allocate(tmps.at("1587_aaaa_vvoo"))(tmps.at("1587_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0413_aa_oo")(ka, ja))
      .deallocate(tmps.at("0413_aa_oo"))
      .allocate(tmps.at("1586_aaaa_vvoo"))(tmps.at("1586_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0431_aa_oo")(ka, ja))
      .deallocate(tmps.at("0431_aa_oo"))
      .allocate(tmps.at("1585_aaaa_vvoo"))(tmps.at("1585_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0426_aa_oo")(ka, ja))
      .deallocate(tmps.at("0426_aa_oo"))
      .allocate(tmps.at("1550_aaaa_oooo"))(tmps.at("1550_aaaa_oooo")(ia, ja, ka, la) =
                                             t1_2p.at("aa")(aa, ka) *
                                             eri.at("aaaa_oovo")(ia, ja, aa, la))
      .allocate(tmps.at("1556_aaaa_vooo"))(tmps.at("1556_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             tmps.at("1550_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1584_aaaa_vvoo"))(tmps.at("1584_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1556_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1556_aaaa_vooo"))
      .allocate(tmps.at("1583_aaaa_vvoo"))(tmps.at("1583_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0248_aa_oo")(ka, ja))
      .deallocate(tmps.at("0248_aa_oo"))
      .allocate(tmps.at("1582_aaaa_vvoo"))(tmps.at("1582_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0419_aa_oo")(ka, ja))
      .deallocate(tmps.at("0419_aa_oo"))
      .allocate(tmps.at("1581_aaaa_vvoo"))(tmps.at("1581_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0417_aa_oo")(ka, ja))
      .deallocate(tmps.at("0417_aa_oo"))
      .allocate(tmps.at("1555_aaaa_vooo"))(tmps.at("1555_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_1p.at("aa")(aa, la) *
                                             tmps.at("0681_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1580_aaaa_vvoo"))(tmps.at("1580_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1555_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1555_aaaa_vooo"))
      .allocate(tmps.at("1579_aaaa_vvoo"))(tmps.at("1579_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("1554_aaaa_oooo")(la, ka, ia, ja))
      .deallocate(tmps.at("1554_aaaa_oooo"))
      .allocate(tmps.at("1578_aaaa_vvoo"))(tmps.at("1578_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("1492_aaaa_oooo")(la, ka, ia, ja))
      .deallocate(tmps.at("1492_aaaa_oooo"))
      .allocate(tmps.at("1575_aaaa_vvoo"))(tmps.at("1575_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("1350_aa_oo")(ka, ja))
      .deallocate(tmps.at("1350_aa_oo"))
      .allocate(tmps.at("1574_aaaa_vvoo"))(tmps.at("1574_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0681_aaaa_oooo")(la, ka, ia, ja))
      .deallocate(tmps.at("0681_aaaa_oooo"))
      .allocate(tmps.at("1573_aaaa_vvoo"))(tmps.at("1573_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0344_aa_oo")(ka, ja))
      .deallocate(tmps.at("0344_aa_oo"))
      .allocate(tmps.at("1572_aaaa_vvoo"))(tmps.at("1572_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0241_aa_oo")(ka, ja))
      .deallocate(tmps.at("0241_aa_oo"))
      .allocate(tmps.at("1571_aaaa_vvoo"))(tmps.at("1571_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0385_aa_oo")(ka, ja))
      .deallocate(tmps.at("0385_aa_oo"))
      .allocate(tmps.at("1570_aaaa_vvoo"))(tmps.at("1570_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0399_aa_oo")(ka, ja))
      .deallocate(tmps.at("0399_aa_oo"))
      .allocate(tmps.at("1569_aaaa_vvoo"))(tmps.at("1569_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0023_aa_oo")(ka, ja))
      .deallocate(tmps.at("0023_aa_oo"))
      .allocate(tmps.at("1568_aaaa_vvoo"))(tmps.at("1568_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0394_aa_oo")(ka, ja))
      .deallocate(tmps.at("0394_aa_oo"))
      .allocate(tmps.at("1567_aaaa_vvoo"))(tmps.at("1567_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0369_aa_oo")(ka, ja))
      .deallocate(tmps.at("0369_aa_oo"))
      .allocate(tmps.at("1566_aaaa_vvoo"))(tmps.at("1566_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0234_aa_oo")(ka, ja))
      .deallocate(tmps.at("0234_aa_oo"))
      .allocate(tmps.at("1565_aaaa_vvoo"))(tmps.at("1565_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0387_aa_oo")(ka, ja))
      .deallocate(tmps.at("0387_aa_oo"))
      .allocate(tmps.at("1564_aaaa_vvoo"))(tmps.at("1564_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0390_aa_oo")(ka, ja))
      .deallocate(tmps.at("0390_aa_oo"))
      .allocate(tmps.at("1563_aaaa_vvoo"))(tmps.at("1563_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("1250_abab_voov")(ba, kb, ja, cb))
      .deallocate(tmps.at("1250_abab_voov"))
      .allocate(tmps.at("1562_aaaa_vvoo"))(tmps.at("1562_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("1486_aaaa_oooo")(la, ka, ia, ja))
      .deallocate(tmps.at("1486_aaaa_oooo"))
      .allocate(tmps.at("1561_aaaa_vvoo"))(tmps.at("1561_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("1374_aaaa_voov")(ba, ka, ja, ca))
      .deallocate(tmps.at("1374_aaaa_voov"))
      .allocate(tmps.at("1560_aaaa_vvoo"))(tmps.at("1560_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0031_aa_oo")(ka, ja))
      .deallocate(tmps.at("0031_aa_oo"))
      .allocate(tmps.at("1559_aaaa_vvoo"))(tmps.at("1559_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(ca, ia) *
                                             tmps.at("1482_aaaa_vvvo")(aa, ba, ca, ja))
      .allocate(tmps.at("1558_aaaa_vvoo"))(tmps.at("1558_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ia, ka) *
                                             tmps.at("0243_aa_oo")(ka, ja))
      .deallocate(tmps.at("0243_aa_oo"))
      .allocate(tmps.at("1557_aaaa_vvoo"))(tmps.at("1557_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("1550_aaaa_oooo")(la, ka, ia, ja))
      .deallocate(tmps.at("1550_aaaa_oooo"))
      .allocate(tmps.at("1552_aaaa_vvoo"))(tmps.at("1552_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ia, ka) *
                                             f.at("aa_oo")(ka, ja))
      .allocate(tmps.at("1551_aaaa_vvoo"))(tmps.at("1551_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(ca, ia) *
                                             eri.at("aaaa_vvvo")(aa, ba, ca, ja))
      .allocate(tmps.at("1553_aaaa_vvoo"))(tmps.at("1553_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("1552_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1553_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1551_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1552_aaaa_vvoo"))
      .deallocate(tmps.at("1551_aaaa_vvoo"))
      .allocate(tmps.at("1576_aaaa_vvoo"))(tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.50 * tmps.at("1557_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1559_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1565_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1553_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) +=
        3.00 * tmps.at("1560_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1568_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1571_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1572_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1561_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.50 * tmps.at("1570_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1573_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1566_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1567_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1574_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -=
        3.00 * tmps.at("1569_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.50 * tmps.at("1575_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1563_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1562_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) -=
        tmps.at("1558_aaaa_vvoo")(aa, ba, ia, ja))(tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja) +=
                                                   0.50 * tmps.at("1564_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1575_aaaa_vvoo"))
      .deallocate(tmps.at("1574_aaaa_vvoo"))
      .deallocate(tmps.at("1573_aaaa_vvoo"))
      .deallocate(tmps.at("1572_aaaa_vvoo"))
      .deallocate(tmps.at("1571_aaaa_vvoo"))
      .deallocate(tmps.at("1570_aaaa_vvoo"))
      .deallocate(tmps.at("1569_aaaa_vvoo"))
      .deallocate(tmps.at("1568_aaaa_vvoo"))
      .deallocate(tmps.at("1567_aaaa_vvoo"))
      .deallocate(tmps.at("1566_aaaa_vvoo"))
      .deallocate(tmps.at("1565_aaaa_vvoo"))
      .deallocate(tmps.at("1564_aaaa_vvoo"))
      .deallocate(tmps.at("1563_aaaa_vvoo"))
      .deallocate(tmps.at("1562_aaaa_vvoo"))
      .deallocate(tmps.at("1561_aaaa_vvoo"))
      .deallocate(tmps.at("1560_aaaa_vvoo"))
      .deallocate(tmps.at("1559_aaaa_vvoo"))
      .deallocate(tmps.at("1558_aaaa_vvoo"))
      .deallocate(tmps.at("1557_aaaa_vvoo"))
      .deallocate(tmps.at("1553_aaaa_vvoo"))
      .allocate(tmps.at("1589_aaaa_vvoo"))(tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -2.00 * tmps.at("1583_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 * tmps.at("1588_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 * tmps.at("1581_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) +=
        2.00 * tmps.at("1582_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) +=
        2.00 * tmps.at("1586_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1578_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1579_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) +=
        2.00 * tmps.at("1584_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 * tmps.at("1587_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 * tmps.at("1585_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) +=
        2.00 *
        tmps.at("1580_aaaa_vvoo")(ba, aa, ia, ja))(tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja) +=
                                                   2.00 * tmps.at("1576_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1588_aaaa_vvoo"))
      .deallocate(tmps.at("1587_aaaa_vvoo"))
      .deallocate(tmps.at("1586_aaaa_vvoo"))
      .deallocate(tmps.at("1585_aaaa_vvoo"))
      .deallocate(tmps.at("1584_aaaa_vvoo"))
      .deallocate(tmps.at("1583_aaaa_vvoo"))
      .deallocate(tmps.at("1582_aaaa_vvoo"))
      .deallocate(tmps.at("1581_aaaa_vvoo"))
      .deallocate(tmps.at("1580_aaaa_vvoo"))
      .deallocate(tmps.at("1579_aaaa_vvoo"))
      .deallocate(tmps.at("1578_aaaa_vvoo"))
      .deallocate(tmps.at("1576_aaaa_vvoo"))
      .allocate(tmps.at("1591_aaaa_vvoo"))(tmps.at("1591_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1589_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1591_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 * tmps.at("1590_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1590_aaaa_vvoo"))
      .deallocate(tmps.at("1589_aaaa_vvoo"))(r2_2p.at("aaaa")(aa, ba, ia, ja) +=
                                             tmps.at("1591_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= tmps.at("1591_aaaa_vvoo")(aa, ba, ja, ia))
      .deallocate(tmps.at("1591_aaaa_vvoo"))
      .allocate(tmps.at("1607_aaaa_vvoo"))(tmps.at("1607_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1134_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1605_aaaa_vvoo"))(tmps.at("1605_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0204_aaaa_oooo")(la, ka, ia, ja))
      .allocate(tmps.at("1604_aaaa_vvoo"))(tmps.at("1604_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1131_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1592_aa_vv"))(tmps.at("1592_aa_vv")(aa, ba) =
                                         eri.at("aaaa_oovv")(ia, ja, ba, ca) *
                                         t2.at("aaaa")(ca, aa, ja, ia))
      .allocate(tmps.at("1602_aaaa_vvoo"))(tmps.at("1602_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("1592_aa_vv")(ba, ca))
      .deallocate(tmps.at("1592_aa_vv"))
      .allocate(tmps.at("1601_aaaa_vvoo"))(tmps.at("1601_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(ca, ia) *
                                             tmps.at("1482_aaaa_vvvo")(aa, ba, ca, ja))
      .deallocate(tmps.at("1482_aaaa_vvvo"))
      .allocate(tmps.at("1600_aaaa_vvoo"))(tmps.at("1600_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0200_aaaa_oooo")(la, ka, ia, ja))
      .allocate(tmps.at("1599_aaaa_vvoo"))(tmps.at("1599_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0374_aa_vv")(ba, ca))
      .allocate(tmps.at("1598_aaaa_vvoo"))(tmps.at("1598_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1108_aaaa_vooo")(ba, ka, ia, ja))
      .allocate(tmps.at("1596_aaaa_vvoo"))(tmps.at("1596_aaaa_vvoo")(aa, ba, ia, ja) =
                                             scalars.at("0015")() *
                                             t2_1p.at("aaaa")(aa, ba, ia, ja))
      .allocate(tmps.at("1595_aaaa_vvoo"))(tmps.at("1595_aaaa_vvoo")(aa, ba, ia, ja) =
                                             scalars.at("0013")() *
                                             t2_1p.at("aaaa")(aa, ba, ia, ja))
      .allocate(tmps.at("1594_aaaa_vvoo"))(tmps.at("1594_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_oooo")(ka, la, ia, ja) *
                                             t2.at("aaaa")(aa, ba, la, ka))
      .allocate(tmps.at("1593_aaaa_vvoo"))(tmps.at("1593_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_vvvv")(aa, ba, ca, da) *
                                             t2.at("aaaa")(ca, da, ia, ja))
      .allocate(tmps.at("1597_aaaa_vvoo"))(tmps.at("1597_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.50 * tmps.at("1594_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1597_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1595_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1597_aaaa_vvoo")(aa, ba, ia, ja) +=
        tmps.at("1596_aaaa_vvoo")(aa, ba, ia, ja))(tmps.at("1597_aaaa_vvoo")(aa, ba, ia, ja) +=
                                                   0.50 * tmps.at("1593_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1596_aaaa_vvoo"))
      .deallocate(tmps.at("1595_aaaa_vvoo"))
      .deallocate(tmps.at("1594_aaaa_vvoo"))
      .deallocate(tmps.at("1593_aaaa_vvoo"))
      .allocate(tmps.at("1603_aaaa_vvoo"))(tmps.at("1603_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.50 * tmps.at("1600_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1603_aaaa_vvoo")(aa, ba, ia, ja) +=
        2.00 * tmps.at("1598_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1603_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1602_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1603_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1599_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1603_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 *
        tmps.at("1601_aaaa_vvoo")(aa, ba, ia, ja))(tmps.at("1603_aaaa_vvoo")(aa, ba, ia, ja) +=
                                                   2.00 * tmps.at("1597_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1602_aaaa_vvoo"))
      .deallocate(tmps.at("1601_aaaa_vvoo"))
      .deallocate(tmps.at("1600_aaaa_vvoo"))
      .deallocate(tmps.at("1599_aaaa_vvoo"))
      .deallocate(tmps.at("1598_aaaa_vvoo"))
      .deallocate(tmps.at("1597_aaaa_vvoo"))
      .allocate(tmps.at("1606_aaaa_vvoo"))(tmps.at("1606_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1603_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1606_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1605_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1606_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1604_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1605_aaaa_vvoo"))
      .deallocate(tmps.at("1604_aaaa_vvoo"))
      .deallocate(tmps.at("1603_aaaa_vvoo"))
      .allocate(tmps.at("1608_aaaa_vvoo"))(tmps.at("1608_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1607_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1608_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1606_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1607_aaaa_vvoo"))
      .deallocate(tmps.at("1606_aaaa_vvoo"))(r2.at("aaaa")(aa, ba, ia, ja) -=
                                             tmps.at("1608_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1608_aaaa_vvoo"))
      .allocate(tmps.at("1619_aaaa_oooo"))(tmps.at("1619_aaaa_oooo")(ia, ja, ka, la) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0410_aaaa_oovo")(ia, ja, aa, la))
      .deallocate(tmps.at("0410_aaaa_oovo"))
      .allocate(tmps.at("1630_aaaa_vooo"))(tmps.at("1630_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             tmps.at("1619_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1637_aaaa_vvoo"))(tmps.at("1637_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1630_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1630_aaaa_vooo"))
      .allocate(tmps.at("1629_aaaa_vooo"))(tmps.at("1629_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_1p.at("aa")(aa, la) *
                                             tmps.at("0204_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1636_aaaa_vvoo"))(tmps.at("1636_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1629_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1629_aaaa_vooo"))
      .allocate(tmps.at("1620_aaaa_vooo"))(tmps.at("1620_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_1p.at("aa")(aa, la) *
                                             tmps.at("0200_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1634_aaaa_vvoo"))(tmps.at("1634_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1620_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1620_aaaa_vooo"))
      .allocate(tmps.at("1633_aaaa_vvoo"))(tmps.at("1633_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0204_aaaa_oooo")(la, ka, ia, ja))
      .deallocate(tmps.at("0204_aaaa_oooo"))
      .allocate(tmps.at("1632_aaaa_vvoo"))(tmps.at("1632_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("1619_aaaa_oooo")(la, ka, ia, ja))
      .deallocate(tmps.at("1619_aaaa_oooo"))
      .allocate(tmps.at("1609_aaaa_oooo"))(tmps.at("1609_aaaa_oooo")(ia, ja, ka, la) =
                                             eri.at("aaaa_oovv")(ia, ja, aa, ba) *
                                             t2_2p.at("aaaa")(aa, ba, ka, la))
      .allocate(tmps.at("1621_aaaa_vooo"))(tmps.at("1621_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(aa, la) *
                                             tmps.at("1609_aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1631_aaaa_vvoo"))(tmps.at("1631_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1621_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1621_aaaa_vooo"))
      .allocate(tmps.at("1627_aaaa_vvoo"))(tmps.at("1627_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0407_aa_vv")(ba, ca))
      .deallocate(tmps.at("0407_aa_vv"))
      .allocate(tmps.at("1626_aaaa_vvoo"))(tmps.at("1626_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0198_aaaa_oooo")(la, ka, ia, ja))
      .deallocate(tmps.at("0198_aaaa_oooo"))
      .allocate(tmps.at("1625_aaaa_vvoo"))(tmps.at("1625_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("0200_aaaa_oooo")(la, ka, ia, ja))
      .deallocate(tmps.at("0200_aaaa_oooo"))
      .allocate(tmps.at("1610_aaaa_vooo"))(tmps.at("1610_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_1p.at("aa")(aa, la) *
                                             eri.at("aaaa_oooo")(la, ia, ja, ka))
      .allocate(tmps.at("1624_aaaa_vvoo"))(tmps.at("1624_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1610_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1610_aaaa_vooo"))
      .allocate(tmps.at("1623_aaaa_vvoo"))(tmps.at("1623_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(aa, ba, ka, la) *
                                             tmps.at("1609_aaaa_oooo")(la, ka, ia, ja))
      .deallocate(tmps.at("1609_aaaa_oooo"))
      .allocate(tmps.at("1622_aaaa_vvoo"))(tmps.at("1622_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("1253_aa_vv")(ba, ca))
      .deallocate(tmps.at("1253_aa_vv"))
      .allocate(tmps.at("1617_aaaa_vvoo"))(tmps.at("1617_aaaa_vvoo")(aa, ba, ia, ja) =
                                             scalars.at("0016")() *
                                             t2_1p.at("aaaa")(aa, ba, ia, ja))
      .allocate(tmps.at("1616_aaaa_vvoo"))(tmps.at("1616_aaaa_vvoo")(aa, ba, ia, ja) =
                                             scalars.at("0014")() *
                                             t2_1p.at("aaaa")(aa, ba, ia, ja))
      .allocate(tmps.at("1615_aaaa_vvoo"))(tmps.at("1615_aaaa_vvoo")(aa, ba, ia, ja) =
                                             scalars.at("0001")() *
                                             t2_2p.at("aaaa")(aa, ba, ia, ja))
      .allocate(tmps.at("1614_aaaa_vvoo"))(tmps.at("bin_aaaa_vvvo")(aa, ba, ca, ia) =
                                             eri.at("aaaa_vvvv")(aa, ba, ca, da) *
                                             t1_1p.at("aa")(da, ia))(
        tmps.at("1614_aaaa_vvoo")(aa, ba, ia, ja) =
          tmps.at("bin_aaaa_vvvo")(aa, ba, ca, ia) * t1_1p.at("aa")(ca, ja))
      .allocate(tmps.at("1613_aaaa_vvoo"))(tmps.at("1613_aaaa_vvoo")(aa, ba, ia, ja) =
                                             scalars.at("0002")() *
                                             t2_2p.at("aaaa")(aa, ba, ia, ja))
      .allocate(tmps.at("1612_aaaa_vvoo"))(tmps.at("1612_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_oooo")(ka, la, ia, ja) *
                                             t2_2p.at("aaaa")(aa, ba, la, ka))
      .allocate(tmps.at("1611_aaaa_vvoo"))(tmps.at("1611_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_vvvv")(aa, ba, ca, da) *
                                             t2_2p.at("aaaa")(ca, da, ia, ja))
      .allocate(tmps.at("1618_aaaa_vvoo"))(tmps.at("1618_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.250 * tmps.at("1612_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1618_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.50 * tmps.at("1616_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1618_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1613_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1618_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.250 * tmps.at("1611_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1618_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1615_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1618_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 *
        tmps.at("1614_aaaa_vvoo")(aa, ba, ia, ja))(tmps.at("1618_aaaa_vvoo")(aa, ba, ia, ja) +=
                                                   0.50 * tmps.at("1617_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1617_aaaa_vvoo"))
      .deallocate(tmps.at("1616_aaaa_vvoo"))
      .deallocate(tmps.at("1615_aaaa_vvoo"))
      .deallocate(tmps.at("1614_aaaa_vvoo"))
      .deallocate(tmps.at("1613_aaaa_vvoo"))
      .deallocate(tmps.at("1612_aaaa_vvoo"))
      .deallocate(tmps.at("1611_aaaa_vvoo"))
      .allocate(tmps.at("1628_aaaa_vvoo"))(tmps.at("1628_aaaa_vvoo")(aa, ba, ia, ja) =
                                             2.00 * tmps.at("1624_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1628_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1623_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1628_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1622_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1628_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1627_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1628_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1626_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1628_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 *
        tmps.at("1625_aaaa_vvoo")(ba, aa, ia, ja))(tmps.at("1628_aaaa_vvoo")(aa, ba, ia, ja) +=
                                                   4.00 * tmps.at("1618_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1627_aaaa_vvoo"))
      .deallocate(tmps.at("1626_aaaa_vvoo"))
      .deallocate(tmps.at("1625_aaaa_vvoo"))
      .deallocate(tmps.at("1624_aaaa_vvoo"))
      .deallocate(tmps.at("1623_aaaa_vvoo"))
      .deallocate(tmps.at("1622_aaaa_vvoo"))
      .deallocate(tmps.at("1618_aaaa_vvoo"))
      .allocate(tmps.at("1635_aaaa_vvoo"))(tmps.at("1635_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1628_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1635_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1631_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1635_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1632_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1635_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1633_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1635_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1634_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1634_aaaa_vvoo"))
      .deallocate(tmps.at("1633_aaaa_vvoo"))
      .deallocate(tmps.at("1632_aaaa_vvoo"))
      .deallocate(tmps.at("1631_aaaa_vvoo"))
      .deallocate(tmps.at("1628_aaaa_vvoo"))
      .allocate(tmps.at("1638_aaaa_vvoo"))(tmps.at("1638_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1636_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1638_aaaa_vvoo")(aa, ba, ia, ja) +=
        tmps.at("1637_aaaa_vvoo")(aa, ba, ja, ia))(tmps.at("1638_aaaa_vvoo")(aa, ba, ia, ja) -=
                                                   0.50 * tmps.at("1635_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1637_aaaa_vvoo"))
      .deallocate(tmps.at("1636_aaaa_vvoo"))
      .deallocate(tmps.at("1635_aaaa_vvoo"))(r2_2p.at("aaaa")(aa, ba, ia, ja) -=
                                             2.00 * tmps.at("1638_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1638_aaaa_vvoo"))
      .allocate(tmps.at("1692_aaaa_vvoo"))(tmps.at("1692_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1500_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1500_aaaa_vooo"))
      .allocate(tmps.at("1690_aaaa_vvoo"))(tmps.at("1690_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0716_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0716_aaaa_vooo"))
      .allocate(tmps.at("1653_aaaa_vooo"))(tmps.at("1653_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, la) *
                                             tmps.at("0428_aaaa_oovo")(ia, la, ba, ka))
      .deallocate(tmps.at("0428_aaaa_oovo"))
      .allocate(tmps.at("1652_aaaa_vooo"))(tmps.at("1652_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_2p.at("aaaa")(ba, aa, ja, la) *
                                             tmps.at("0203_aaaa_oovo")(ia, la, ba, ka))
      .deallocate(tmps.at("0203_aaaa_oovo"))
      .allocate(tmps.at("1649_aaaa_vooo"))(tmps.at("1649_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_1p.at("aaaa")(ba, aa, ja, la) *
                                             tmps.at("1238_aaaa_oovo")(ia, la, ba, ka))
      .deallocate(tmps.at("1238_aaaa_oovo"))
      .allocate(tmps.at("1648_aaaa_vooo"))(tmps.at("1648_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_2p.at("aa")(ba, ja) *
                                             tmps.at("0037_aaaa_voov")(aa, ia, ka, ba))
      .allocate(tmps.at("1647_aaaa_vooo"))(tmps.at("1647_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1.at("aa")(ba, ja) *
                                             tmps.at("0379_aaaa_voov")(aa, ia, ka, ba))
      .allocate(tmps.at("1646_aaaa_vooo"))(tmps.at("1646_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0039_aaaa_voov")(aa, ia, ka, ba))
      .deallocate(tmps.at("0039_aaaa_voov"))
      .allocate(tmps.at("1676_aaaa_vooo"))(tmps.at("1676_aaaa_vooo")(aa, ia, ja, ka) =
                                             -1.00 * tmps.at("1652_aaaa_vooo")(aa, ia, ka, ja))(
        tmps.at("1676_aaaa_vooo")(aa, ia, ja, ka) -= tmps.at("1653_aaaa_vooo")(aa, ia, ja, ka))(
        tmps.at("1676_aaaa_vooo")(aa, ia, ja, ka) -= tmps.at("1648_aaaa_vooo")(aa, ia, ka, ja))(
        tmps.at("1676_aaaa_vooo")(aa, ia, ja, ka) += tmps.at("1646_aaaa_vooo")(aa, ia, ja, ka))(
        tmps.at("1676_aaaa_vooo")(aa, ia, ja, ka) += tmps.at("1647_aaaa_vooo")(aa, ia, ja, ka))(
        tmps.at("1676_aaaa_vooo")(aa, ia, ja, ka) -= tmps.at("1649_aaaa_vooo")(aa, ia, ka, ja))
      .deallocate(tmps.at("1653_aaaa_vooo"))
      .deallocate(tmps.at("1652_aaaa_vooo"))
      .deallocate(tmps.at("1649_aaaa_vooo"))
      .deallocate(tmps.at("1648_aaaa_vooo"))
      .deallocate(tmps.at("1647_aaaa_vooo"))
      .deallocate(tmps.at("1646_aaaa_vooo"))
      .allocate(tmps.at("1689_aaaa_vvoo"))(tmps.at("1689_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1676_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1676_aaaa_vooo"))
      .allocate(tmps.at("1688_aaaa_vvoo"))(tmps.at("1688_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0760_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0760_aaaa_vooo"))
      .allocate(tmps.at("1687_aaaa_vvoo"))(tmps.at("1687_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0765_aa_vo")(ba, ja))
      .deallocate(tmps.at("0765_aa_vo"))
      .allocate(tmps.at("1686_aaaa_vvoo"))(tmps.at("1686_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0767_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0767_aaaa_vooo"))
      .allocate(tmps.at("1685_aaaa_vvoo"))(tmps.at("1685_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1494_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1494_aaaa_vooo"))
      .allocate(tmps.at("1651_aaaa_vooo"))(tmps.at("1651_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_2p.at("aa")(ba, ja) *
                                             tmps.at("0363_aaaa_vovo")(aa, ia, ba, ka))
      .allocate(tmps.at("1684_aaaa_vvoo"))(tmps.at("1684_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1651_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1651_aaaa_vooo"))
      .allocate(tmps.at("1683_aaaa_vvoo"))(tmps.at("1683_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1408_aa_vo")(ba, ja))
      .deallocate(tmps.at("1408_aa_vo"))
      .allocate(tmps.at("1682_aaaa_vvoo"))(tmps.at("1682_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0689_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0689_aaaa_vooo"))
      .allocate(tmps.at("1650_aaaa_vooo"))(tmps.at("1650_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0346_aaaa_voov")(aa, ia, ka, ba))
      .allocate(tmps.at("1681_aaaa_vvoo"))(tmps.at("1681_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1650_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1650_aaaa_vooo"))
      .allocate(tmps.at("1680_aaaa_vvoo"))(tmps.at("1680_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0756_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0756_aaaa_vooo"))
      .allocate(tmps.at("1679_aaaa_vvoo"))(tmps.at("1679_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1393_aa_vo")(ba, ja))
      .deallocate(tmps.at("1393_aa_vo"))
      .allocate(tmps.at("1678_aaaa_vvoo"))(tmps.at("1678_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1411_aa_vo")(ba, ja))
      .deallocate(tmps.at("1411_aa_vo"))
      .allocate(tmps.at("1675_aaaa_vvoo"))(tmps.at("1675_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1320_aa_vo")(ba, ja))
      .deallocate(tmps.at("1320_aa_vo"))
      .allocate(tmps.at("1674_aaaa_vvoo"))(tmps.at("1674_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0392_abba_vovo")(ba, kb, cb, ja))
      .deallocate(tmps.at("0392_abba_vovo"))
      .allocate(tmps.at("1673_aaaa_vvoo"))(tmps.at("1673_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0346_aaaa_voov")(ba, ka, ja, ca))
      .deallocate(tmps.at("0346_aaaa_voov"))
      .allocate(tmps.at("1640_aaaa_vooo"))(tmps.at("1640_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_2p.at("abab")(aa, bb, ja, lb) *
                                             eri.at("abba_oovo")(ia, lb, bb, ka))
      .allocate(tmps.at("1639_aaaa_vooo"))(tmps.at("1639_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_2p.at("aaaa")(ba, aa, ja, la) *
                                             eri.at("aaaa_oovo")(ia, la, ba, ka))
      .allocate(tmps.at("1644_aaaa_vooo"))(tmps.at("1644_aaaa_vooo")(aa, ia, ja, ka) =
                                             -1.00 * tmps.at("1639_aaaa_vooo")(aa, ia, ja, ka))(
        tmps.at("1644_aaaa_vooo")(aa, ia, ja, ka) += tmps.at("1640_aaaa_vooo")(aa, ia, ja, ka))
      .deallocate(tmps.at("1640_aaaa_vooo"))
      .deallocate(tmps.at("1639_aaaa_vooo"))
      .allocate(tmps.at("1672_aaaa_vvoo"))(tmps.at("1672_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1644_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1644_aaaa_vooo"))
      .allocate(tmps.at("1671_aaaa_vvoo"))(tmps.at("1671_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1354_aa_vo")(ba, ja))
      .deallocate(tmps.at("1354_aa_vo"))
      .allocate(tmps.at("1670_aaaa_vvoo"))(tmps.at("1670_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0746_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0746_aaaa_vooo"))
      .allocate(tmps.at("1669_aaaa_vvoo"))(tmps.at("1669_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0704_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0704_aaaa_vooo"))
      .allocate(tmps.at("1668_aaaa_vvoo"))(tmps.at("1668_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0737_aa_vo")(ba, ja))
      .deallocate(tmps.at("0737_aa_vo"))
      .allocate(tmps.at("1667_aaaa_vvoo"))(tmps.at("1667_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0750_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0750_aaaa_vooo"))
      .allocate(tmps.at("1666_aaaa_vvoo"))(tmps.at("1666_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ia) * tmps.at("1365_aa_vo")(ba, ja))
      .deallocate(tmps.at("1365_aa_vo"))
      .allocate(tmps.at("1665_aaaa_vvoo"))(tmps.at("1665_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0701_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0701_aaaa_vooo"))
      .allocate(tmps.at("1664_aaaa_vvoo"))(tmps.at("1664_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ia) * tmps.at("0741_aa_vo")(ba, ja))
      .deallocate(tmps.at("0741_aa_vo"))
      .allocate(tmps.at("1663_aaaa_vvoo"))(tmps.at("1663_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0230_abba_vovo")(ba, kb, cb, ja))
      .deallocate(tmps.at("0230_abba_vovo"))
      .allocate(tmps.at("1662_aaaa_vvoo"))(tmps.at("1662_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0379_aaaa_voov")(ba, ka, ja, ca))
      .deallocate(tmps.at("0379_aaaa_voov"))
      .allocate(tmps.at("1661_aaaa_vvoo"))(tmps.at("1661_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0376_abab_voov")(ba, kb, ja, cb))
      .deallocate(tmps.at("0376_abab_voov"))
      .allocate(tmps.at("1660_aaaa_vvoo"))(tmps.at("1660_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0037_aaaa_voov")(ba, ka, ja, ca))
      .deallocate(tmps.at("0037_aaaa_voov"))
      .allocate(tmps.at("1659_aaaa_vvoo"))(tmps.at("1659_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("abab")(aa, cb, ia, kb) *
                                             tmps.at("0372_abba_vovo")(ba, kb, cb, ja))
      .deallocate(tmps.at("0372_abba_vovo"))
      .allocate(tmps.at("1641_aaaa_vooo"))(tmps.at("1641_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_2p.at("aa")(ba, ja) *
                                             eri.at("aaaa_vovo")(aa, ia, ba, ka))
      .allocate(tmps.at("1658_aaaa_vvoo"))(tmps.at("1658_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1641_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1641_aaaa_vooo"))
      .allocate(tmps.at("1657_aaaa_vvoo"))(tmps.at("1657_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0726_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0726_aaaa_vooo"))
      .allocate(tmps.at("1656_aaaa_vvoo"))(tmps.at("1656_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0363_aaaa_vovo")(ba, ka, ca, ja))
      .deallocate(tmps.at("0363_aaaa_vovo"))
      .allocate(tmps.at("1655_aaaa_vvoo"))(tmps.at("1655_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("1257_aaaa_vovo")(ba, ka, ca, ja))
      .deallocate(tmps.at("1257_aaaa_vovo"))
      .allocate(tmps.at("1654_aaaa_vvoo"))(tmps.at("1654_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ka) *
                                             tmps.at("0350_aaaa_vovo")(ba, ka, ca, ja))
      .deallocate(tmps.at("0350_aaaa_vovo"))
      .allocate(tmps.at("1643_aaaa_vvoo"))(tmps.at("1643_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("abba_vovo")(aa, kb, cb, ia) *
                                             t2_2p.at("abab")(ba, cb, ja, kb))
      .allocate(tmps.at("1642_aaaa_vvoo"))(tmps.at("1642_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_vovo")(aa, ka, ca, ia) *
                                             t2_2p.at("aaaa")(ca, ba, ja, ka))
      .allocate(tmps.at("1645_aaaa_vvoo"))(tmps.at("1645_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("1642_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1645_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1643_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1643_aaaa_vvoo"))
      .deallocate(tmps.at("1642_aaaa_vvoo"))
      .allocate(tmps.at("1677_aaaa_vvoo"))(tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("1657_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1658_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1661_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1656_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1669_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1675_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) -=
        2.00 * tmps.at("1668_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1655_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1673_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1659_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1663_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1654_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1667_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1674_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1666_aaaa_vvoo")(ba, aa, ja, ia))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) +=
        2.00 * tmps.at("1664_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1662_aaaa_vvoo")(ba, aa, ja, ia))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1672_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1645_aaaa_vvoo")(ba, aa, ja, ia))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1665_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1670_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1660_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1671_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1675_aaaa_vvoo"))
      .deallocate(tmps.at("1674_aaaa_vvoo"))
      .deallocate(tmps.at("1673_aaaa_vvoo"))
      .deallocate(tmps.at("1672_aaaa_vvoo"))
      .deallocate(tmps.at("1671_aaaa_vvoo"))
      .deallocate(tmps.at("1670_aaaa_vvoo"))
      .deallocate(tmps.at("1669_aaaa_vvoo"))
      .deallocate(tmps.at("1668_aaaa_vvoo"))
      .deallocate(tmps.at("1667_aaaa_vvoo"))
      .deallocate(tmps.at("1666_aaaa_vvoo"))
      .deallocate(tmps.at("1665_aaaa_vvoo"))
      .deallocate(tmps.at("1664_aaaa_vvoo"))
      .deallocate(tmps.at("1663_aaaa_vvoo"))
      .deallocate(tmps.at("1662_aaaa_vvoo"))
      .deallocate(tmps.at("1661_aaaa_vvoo"))
      .deallocate(tmps.at("1660_aaaa_vvoo"))
      .deallocate(tmps.at("1659_aaaa_vvoo"))
      .deallocate(tmps.at("1658_aaaa_vvoo"))
      .deallocate(tmps.at("1657_aaaa_vvoo"))
      .deallocate(tmps.at("1656_aaaa_vvoo"))
      .deallocate(tmps.at("1655_aaaa_vvoo"))
      .deallocate(tmps.at("1654_aaaa_vvoo"))
      .deallocate(tmps.at("1645_aaaa_vvoo"))
      .allocate(tmps.at("1691_aaaa_vvoo"))(tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1677_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1690_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1682_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1680_aaaa_vvoo")(ba, aa, ja, ia))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1685_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1683_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1681_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) +=
        2.00 * tmps.at("1687_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1686_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1678_aaaa_vvoo")(aa, ba, ja, ia))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1679_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1688_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1689_aaaa_vvoo")(ba, aa, ja, ia))(
        tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1684_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1690_aaaa_vvoo"))
      .deallocate(tmps.at("1689_aaaa_vvoo"))
      .deallocate(tmps.at("1688_aaaa_vvoo"))
      .deallocate(tmps.at("1687_aaaa_vvoo"))
      .deallocate(tmps.at("1686_aaaa_vvoo"))
      .deallocate(tmps.at("1685_aaaa_vvoo"))
      .deallocate(tmps.at("1684_aaaa_vvoo"))
      .deallocate(tmps.at("1683_aaaa_vvoo"))
      .deallocate(tmps.at("1682_aaaa_vvoo"))
      .deallocate(tmps.at("1681_aaaa_vvoo"))
      .deallocate(tmps.at("1680_aaaa_vvoo"))
      .deallocate(tmps.at("1679_aaaa_vvoo"))
      .deallocate(tmps.at("1678_aaaa_vvoo"))
      .deallocate(tmps.at("1677_aaaa_vvoo"))
      .allocate(tmps.at("1693_aaaa_vvoo"))(tmps.at("1693_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("1692_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1693_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1691_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1692_aaaa_vvoo"))
      .deallocate(tmps.at("1691_aaaa_vvoo"))(r2_2p.at("aaaa")(aa, ba, ia, ja) +=
                                             2.00 * tmps.at("1693_aaaa_vvoo")(ba, aa, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= 2.00 * tmps.at("1693_aaaa_vvoo")(ba, aa, ja, ia))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) -= 2.00 * tmps.at("1693_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += 2.00 * tmps.at("1693_aaaa_vvoo")(aa, ba, ja, ia))
      .deallocate(tmps.at("1693_aaaa_vvoo"));
  }
}

template void exachem::cc::qed_ccsd_os::resid_7<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);