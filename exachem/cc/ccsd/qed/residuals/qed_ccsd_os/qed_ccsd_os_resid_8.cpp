/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_os_resid_8.hpp"

template<typename T>
void exachem::cc::qed_ccsd_os::resid_8(
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
      .allocate(tmps.at("1710_bbbb_vooo"))(tmps.at("1710_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0323_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1711_bbbb_vvoo"))(tmps.at("1711_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1710_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1707_bbbb_vooo"))(tmps.at("1707_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("0318_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1708_bbbb_vvoo"))(tmps.at("1708_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1707_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1706_bbbb_vvoo"))(tmps.at("1706_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0323_bbbb_oooo")(lb, kb, ib, jb))
      .deallocate(tmps.at("0323_bbbb_oooo"))
      .allocate(tmps.at("1704_bbbb_vvoo"))(tmps.at("1704_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(cb, ib) *
                                             tmps.at("1525_bbbb_vvvo")(ab, bb, cb, jb))
      .allocate(tmps.at("1702_bbbb_vooo"))(tmps.at("1702_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             eri.at("bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1703_bbbb_vvoo"))(tmps.at("1703_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1702_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1701_bbbb_vvoo"))(tmps.at("1701_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0318_bbbb_oooo")(lb, kb, ib, jb))
      .deallocate(tmps.at("0318_bbbb_oooo"))
      .allocate(tmps.at("1700_bbbb_vvoo"))(tmps.at("1700_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0500_bb_vv")(bb, cb))
      .deallocate(tmps.at("0500_bb_vv"))
      .allocate(tmps.at("1699_bbbb_vvoo"))(tmps.at("1699_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0581_bb_vv")(bb, cb))
      .allocate(tmps.at("1697_bbbb_vvoo"))(tmps.at("1697_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_oooo")(kb, lb, ib, jb) *
                                             t2.at("bbbb")(ab, bb, lb, kb))
      .allocate(tmps.at("1696_bbbb_vvoo"))(tmps.at("1696_bbbb_vvoo")(ab, bb, ib, jb) =
                                             scalars.at("0015")() *
                                             t2_1p.at("bbbb")(ab, bb, ib, jb))
      .allocate(tmps.at("1695_bbbb_vvoo"))(tmps.at("1695_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_vvvv")(ab, bb, cb, db) *
                                             t2.at("bbbb")(cb, db, ib, jb))
      .allocate(tmps.at("1694_bbbb_vvoo"))(tmps.at("1694_bbbb_vvoo")(ab, bb, ib, jb) =
                                             scalars.at("0013")() *
                                             t2_1p.at("bbbb")(ab, bb, ib, jb))
      .allocate(tmps.at("1698_bbbb_vvoo"))(tmps.at("1698_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1695_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1698_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 * tmps.at("1694_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1698_bbbb_vvoo")(ab, bb, ib, jb) +=
        tmps.at("1697_bbbb_vvoo")(ab, bb, ib, jb))(tmps.at("1698_bbbb_vvoo")(ab, bb, ib, jb) -=
                                                   2.00 * tmps.at("1696_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1697_bbbb_vvoo"))
      .deallocate(tmps.at("1696_bbbb_vvoo"))
      .deallocate(tmps.at("1695_bbbb_vvoo"))
      .deallocate(tmps.at("1694_bbbb_vvoo"))
      .allocate(tmps.at("1705_bbbb_vvoo"))(tmps.at("1705_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1700_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1705_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("1701_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1705_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 * tmps.at("1703_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1705_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1698_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1705_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1704_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1705_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1699_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1704_bbbb_vvoo"))
      .deallocate(tmps.at("1703_bbbb_vvoo"))
      .deallocate(tmps.at("1701_bbbb_vvoo"))
      .deallocate(tmps.at("1700_bbbb_vvoo"))
      .deallocate(tmps.at("1699_bbbb_vvoo"))
      .deallocate(tmps.at("1698_bbbb_vvoo"))
      .allocate(tmps.at("1709_bbbb_vvoo"))(tmps.at("1709_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1705_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1709_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1706_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1709_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1708_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1708_bbbb_vvoo"))
      .deallocate(tmps.at("1706_bbbb_vvoo"))
      .deallocate(tmps.at("1705_bbbb_vvoo"))
      .allocate(tmps.at("1712_bbbb_vvoo"))(tmps.at("1712_bbbb_vvoo")(ab, bb, ib, jb) =
                                             tmps.at("1711_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1712_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("1709_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1711_bbbb_vvoo"))
      .deallocate(tmps.at("1709_bbbb_vvoo"))(r2.at("bbbb")(ab, bb, ib, jb) -=
                                             tmps.at("1712_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1712_bbbb_vvoo"))
      .allocate(tmps.at("1761_bbbb_vvoo"))(tmps.at("1761_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("1710_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1718_bbbb_vooo"))(tmps.at("1718_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1_1p.at("bb")(bb, jb) *
                                             tmps.at("0131_bbbb_vovo")(ab, ib, bb, kb))
      .deallocate(tmps.at("0131_bbbb_vovo"))
      .allocate(tmps.at("1759_bbbb_vvoo"))(tmps.at("1759_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1718_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1718_bbbb_vooo"))
      .allocate(tmps.at("1721_bbbb_vooo"))(tmps.at("1721_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_1p.at("bbbb")(bb, ab, jb, kb) *
                                             tmps.at("1210_bb_ov")(ib, bb))
      .deallocate(tmps.at("1210_bb_ov"))
      .allocate(tmps.at("1720_bbbb_vooo"))(tmps.at("1720_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("bbbb")(bb, ab, jb, kb) *
                                             tmps.at("1200_bb_ov")(ib, bb))
      .deallocate(tmps.at("1200_bb_ov"))
      .allocate(tmps.at("1752_bbbb_vooo"))(tmps.at("1752_bbbb_vooo")(ab, ib, jb, kb) =
                                             -1.00 * tmps.at("1720_bbbb_vooo")(ab, ib, jb, kb))(
        tmps.at("1752_bbbb_vooo")(ab, ib, jb, kb) += tmps.at("1721_bbbb_vooo")(ab, ib, jb, kb))
      .deallocate(tmps.at("1721_bbbb_vooo"))
      .deallocate(tmps.at("1720_bbbb_vooo"))
      .allocate(tmps.at("1758_bbbb_vvoo"))(tmps.at("1758_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1752_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1752_bbbb_vooo"))
      .allocate(tmps.at("1722_bbbb_vooo"))(tmps.at("1722_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_2p.at("bbbb")(bb, ab, jb, kb) *
                                             tmps.at("1189_bb_ov")(ib, bb))
      .deallocate(tmps.at("1189_bb_ov"))
      .allocate(tmps.at("1719_bbbb_vooo"))(tmps.at("1719_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_2p.at("bbbb")(bb, ab, jb, kb) *
                                             tmps.at("0237_bb_ov")(ib, bb))
      .deallocate(tmps.at("0237_bb_ov"))
      .allocate(tmps.at("1751_bbbb_vooo"))(tmps.at("1751_bbbb_vooo")(ab, ib, jb, kb) =
                                             tmps.at("1719_bbbb_vooo")(ab, ib, jb, kb))(
        tmps.at("1751_bbbb_vooo")(ab, ib, jb, kb) += tmps.at("1722_bbbb_vooo")(ab, ib, jb, kb))
      .deallocate(tmps.at("1722_bbbb_vooo"))
      .deallocate(tmps.at("1719_bbbb_vooo"))
      .allocate(tmps.at("1757_bbbb_vvoo"))(tmps.at("1757_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1751_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1751_bbbb_vooo"))
      .allocate(tmps.at("1756_bbbb_vvoo"))(tmps.at("1756_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0308_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1755_bbbb_vvoo"))(tmps.at("1755_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0325_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0325_bbbb_vooo"))
      .allocate(tmps.at("1723_bbbb_vooo"))(tmps.at("1723_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("bbbb")(bb, ab, jb, kb) *
                                             tmps.at("1206_bb_ov")(ib, bb))
      .deallocate(tmps.at("1206_bb_ov"))
      .allocate(tmps.at("1754_bbbb_vvoo"))(tmps.at("1754_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1723_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1723_bbbb_vooo"))
      .allocate(tmps.at("1753_bbbb_vvoo"))(tmps.at("1753_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("1707_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1749_bbbb_vvoo"))(tmps.at("1749_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0295_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1748_bbbb_vvoo"))(tmps.at("1748_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0921_bb_vv")(bb, cb))
      .allocate(tmps.at("1747_bbbb_vvoo"))(tmps.at("1747_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("1292_bb_vv")(bb, cb))
      .deallocate(tmps.at("1292_bb_vv"))
      .allocate(tmps.at("1746_bbbb_vvoo"))(tmps.at("1746_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0305_bbbb_ovoo")(kb, bb, ib, jb))
      .allocate(tmps.at("1745_bbbb_vvoo"))(tmps.at("1745_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0048_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0048_bbbb_vooo"))
      .allocate(tmps.at("1743_bbbb_ovoo"))(tmps.at("bin_bb_vo")(bb, ib) =
                                             eri.at("abab_oovv")(la, ib, ca, bb) *
                                             t1_1p.at("aa")(ca, la))(
        tmps.at("1743_bbbb_ovoo")(ib, ab, jb, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2.at("bbbb")(bb, ab, jb, kb))
      .allocate(tmps.at("1744_bbbb_vvoo"))(tmps.at("1744_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1743_bbbb_ovoo")(kb, bb, ib, jb))
      .allocate(tmps.at("1713_bbbb_vooo"))(tmps.at("1713_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_2p.at("bbbb")(bb, ab, jb, kb) *
                                             f.at("bb_ov")(ib, bb))
      .allocate(tmps.at("1742_bbbb_vvoo"))(tmps.at("1742_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1713_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1713_bbbb_vooo"))
      .allocate(tmps.at("1741_bbbb_vvoo"))(tmps.at("1741_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0564_bb_vv")(bb, cb))
      .deallocate(tmps.at("0564_bb_vv"))
      .allocate(tmps.at("1740_bbbb_vvoo"))(tmps.at("1740_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0581_bb_vv")(bb, cb))
      .allocate(tmps.at("1737_bbbb_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("abab_oovv")(la, ib, ca, bb) * t1.at("aa")(ca, la))(
        tmps.at("1737_bbbb_ovoo")(ib, ab, jb, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2_1p.at("bbbb")(bb, ab, jb, kb))
      .allocate(tmps.at("1736_bbbb_ovoo"))(
        tmps.at("bin_bb_vo")(bb, ib) = eri.at("bbbb_oovv")(lb, ib, cb, bb) * t1.at("bb")(cb, lb))(
        tmps.at("1736_bbbb_ovoo")(ib, ab, jb, kb) =
          tmps.at("bin_bb_vo")(bb, ib) * t2_1p.at("bbbb")(bb, ab, jb, kb))
      .allocate(tmps.at("1738_bbbb_ovoo"))(tmps.at("1738_bbbb_ovoo")(ib, ab, jb, kb) =
                                             tmps.at("1736_bbbb_ovoo")(ib, ab, jb, kb))(
        tmps.at("1738_bbbb_ovoo")(ib, ab, jb, kb) += tmps.at("1737_bbbb_ovoo")(ib, ab, jb, kb))
      .deallocate(tmps.at("1737_bbbb_ovoo"))
      .deallocate(tmps.at("1736_bbbb_ovoo"))
      .allocate(tmps.at("1739_bbbb_vvoo"))(tmps.at("1739_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1738_bbbb_ovoo")(kb, bb, ib, jb))
      .allocate(tmps.at("1735_bbbb_vvoo"))(tmps.at("1735_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0301_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1733_bbbb_vooo"))(tmps.at("1733_bbbb_vooo")(ab, ib, jb, kb) =
                                             eri.at("bbbb_vovv")(ab, ib, bb, cb) *
                                             t2_1p.at("bbbb")(bb, cb, jb, kb))
      .allocate(tmps.at("1734_bbbb_vvoo"))(tmps.at("1734_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1733_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1732_bbbb_vvoo"))(tmps.at("1732_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0505_bb_vv")(bb, cb))
      .allocate(tmps.at("1731_bbbb_vvoo"))(tmps.at("1731_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0299_bb_vv")(bb, cb))
      .allocate(tmps.at("1730_bbbb_vvoo"))(tmps.at("1730_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("1702_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1729_bbbb_vvoo"))(tmps.at("1729_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0071_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0071_bbbb_vooo"))
      .allocate(tmps.at("1728_bbbb_vvoo"))(tmps.at("1728_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("1378_bb_vv")(bb, cb))
      .deallocate(tmps.at("1378_bb_vv"))
      .allocate(tmps.at("1726_bbbb_vooo"))(tmps.at("1726_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_1p.at("bbbb")(bb, ab, jb, kb) *
                                             f.at("bb_ov")(ib, bb))
      .allocate(tmps.at("1727_bbbb_vvoo"))(tmps.at("1727_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1726_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1725_bbbb_vvoo"))(tmps.at("1725_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0297_bb_vv")(bb, cb))
      .allocate(tmps.at("1714_bbbb_vooo"))(tmps.at("1714_bbbb_vooo")(ab, ib, jb, kb) =
                                             eri.at("bbbb_vovv")(ab, ib, bb, cb) *
                                             t2_2p.at("bbbb")(bb, cb, jb, kb))
      .allocate(tmps.at("1724_bbbb_vvoo"))(tmps.at("1724_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1714_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1714_bbbb_vooo"))
      .allocate(tmps.at("1716_bbbb_vvoo"))(tmps.at("1716_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_vooo")(ab, kb, ib, jb) *
                                             t1_2p.at("bb")(bb, kb))
      .allocate(tmps.at("1715_bbbb_vvoo"))(tmps.at("1715_bbbb_vvoo")(ab, bb, ib, jb) =
                                             f.at("bb_vv")(ab, cb) *
                                             t2_2p.at("bbbb")(cb, bb, ib, jb))
      .allocate(tmps.at("1717_bbbb_vvoo"))(tmps.at("1717_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1716_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1717_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1715_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1716_bbbb_vvoo"))
      .deallocate(tmps.at("1715_bbbb_vvoo"))
      .allocate(tmps.at("1750_bbbb_vvoo"))(tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1740_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 * tmps.at("1730_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) -=
        6.00 * tmps.at("1745_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1725_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 * tmps.at("1717_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1724_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1734_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1732_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1727_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1742_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1749_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1728_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1731_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 * tmps.at("1744_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1747_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1748_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1739_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 * tmps.at("1746_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) +=
        6.00 * tmps.at("1729_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 *
        tmps.at("1735_bbbb_vvoo")(ab, bb, ib, jb))(tmps.at("1750_bbbb_vvoo")(ab, bb, ib, jb) -=
                                                   2.00 * tmps.at("1741_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1749_bbbb_vvoo"))
      .deallocate(tmps.at("1748_bbbb_vvoo"))
      .deallocate(tmps.at("1747_bbbb_vvoo"))
      .deallocate(tmps.at("1746_bbbb_vvoo"))
      .deallocate(tmps.at("1745_bbbb_vvoo"))
      .deallocate(tmps.at("1744_bbbb_vvoo"))
      .deallocate(tmps.at("1742_bbbb_vvoo"))
      .deallocate(tmps.at("1741_bbbb_vvoo"))
      .deallocate(tmps.at("1740_bbbb_vvoo"))
      .deallocate(tmps.at("1739_bbbb_vvoo"))
      .deallocate(tmps.at("1735_bbbb_vvoo"))
      .deallocate(tmps.at("1734_bbbb_vvoo"))
      .deallocate(tmps.at("1732_bbbb_vvoo"))
      .deallocate(tmps.at("1731_bbbb_vvoo"))
      .deallocate(tmps.at("1730_bbbb_vvoo"))
      .deallocate(tmps.at("1729_bbbb_vvoo"))
      .deallocate(tmps.at("1728_bbbb_vvoo"))
      .deallocate(tmps.at("1727_bbbb_vvoo"))
      .deallocate(tmps.at("1725_bbbb_vvoo"))
      .deallocate(tmps.at("1724_bbbb_vvoo"))
      .deallocate(tmps.at("1717_bbbb_vvoo"))
      .allocate(tmps.at("1760_bbbb_vvoo"))(tmps.at("1760_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -0.50 * tmps.at("1753_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1760_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("1755_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1760_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1758_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1760_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("1750_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1760_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1757_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1760_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1756_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1760_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1754_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1760_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1759_bbbb_vvoo")(bb, ab, jb, ib))
      .deallocate(tmps.at("1759_bbbb_vvoo"))
      .deallocate(tmps.at("1758_bbbb_vvoo"))
      .deallocate(tmps.at("1757_bbbb_vvoo"))
      .deallocate(tmps.at("1756_bbbb_vvoo"))
      .deallocate(tmps.at("1755_bbbb_vvoo"))
      .deallocate(tmps.at("1754_bbbb_vvoo"))
      .deallocate(tmps.at("1753_bbbb_vvoo"))
      .deallocate(tmps.at("1750_bbbb_vvoo"))
      .allocate(tmps.at("1762_bbbb_vvoo"))(tmps.at("1762_bbbb_vvoo")(ab, bb, ib, jb) =
                                             tmps.at("1761_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1762_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1760_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1761_bbbb_vvoo"))
      .deallocate(tmps.at("1760_bbbb_vvoo"))(r2_2p.at("bbbb")(ab, bb, ib, jb) -=
                                             2.00 * tmps.at("1762_bbbb_vvoo")(bb, ab, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("1762_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1762_bbbb_vvoo"))
      .allocate(tmps.at("1815_bbbb_vooo"))(tmps.at("1815_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("1536_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1816_bbbb_vvoo"))(tmps.at("1816_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1815_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1813_bbbb_vvoo"))(tmps.at("1813_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("1396_bb_vo")(bb, jb))
      .deallocate(tmps.at("1396_bb_vo"))
      .allocate(tmps.at("1777_bbbb_vooo"))(tmps.at("1777_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_1p.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0593_abab_oovo")(la, ib, ba, kb))
      .deallocate(tmps.at("0593_abab_oovo"))
      .allocate(tmps.at("1775_bbbb_vooo"))(tmps.at("1775_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_2p.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0286_abab_oovo")(la, ib, ba, kb))
      .deallocate(tmps.at("0286_abab_oovo"))
      .allocate(tmps.at("1773_bbbb_vooo"))(tmps.at("1773_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_2p.at("bbbb")(bb, ab, jb, lb) *
                                             tmps.at("0256_bbbb_oovo")(ib, lb, bb, kb))
      .deallocate(tmps.at("0256_bbbb_oovo"))
      .allocate(tmps.at("1772_bbbb_vooo"))(tmps.at("1772_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_1p.at("bbbb")(bb, ab, jb, lb) *
                                             tmps.at("1193_bbbb_oovo")(ib, lb, bb, kb))
      .deallocate(tmps.at("1193_bbbb_oovo"))
      .allocate(tmps.at("1771_bbbb_vooo"))(tmps.at("1771_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("bbbb")(bb, ab, jb, lb) *
                                             tmps.at("0600_bbbb_oovo")(ib, lb, bb, kb))
      .deallocate(tmps.at("0600_bbbb_oovo"))
      .allocate(tmps.at("1770_bbbb_vooo"))(tmps.at("1770_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("abab")(ba, ab, la, jb) *
                                             tmps.at("0601_abab_oovo")(la, ib, ba, kb))
      .deallocate(tmps.at("0601_abab_oovo"))
      .allocate(tmps.at("1802_bbbb_vooo"))(tmps.at("1802_bbbb_vooo")(ab, ib, jb, kb) =
                                             tmps.at("1770_bbbb_vooo")(ab, ib, jb, kb))(
        tmps.at("1802_bbbb_vooo")(ab, ib, jb, kb) += tmps.at("1773_bbbb_vooo")(ab, ib, kb, jb))(
        tmps.at("1802_bbbb_vooo")(ab, ib, jb, kb) += tmps.at("1772_bbbb_vooo")(ab, ib, kb, jb))(
        tmps.at("1802_bbbb_vooo")(ab, ib, jb, kb) -= tmps.at("1775_bbbb_vooo")(ab, ib, kb, jb))(
        tmps.at("1802_bbbb_vooo")(ab, ib, jb, kb) += tmps.at("1771_bbbb_vooo")(ab, ib, jb, kb))(
        tmps.at("1802_bbbb_vooo")(ab, ib, jb, kb) -= tmps.at("1777_bbbb_vooo")(ab, ib, kb, jb))
      .deallocate(tmps.at("1777_bbbb_vooo"))
      .deallocate(tmps.at("1775_bbbb_vooo"))
      .deallocate(tmps.at("1773_bbbb_vooo"))
      .deallocate(tmps.at("1772_bbbb_vooo"))
      .deallocate(tmps.at("1771_bbbb_vooo"))
      .deallocate(tmps.at("1770_bbbb_vooo"))
      .allocate(tmps.at("1812_bbbb_vvoo"))(tmps.at("1812_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1802_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1802_bbbb_vooo"))
      .allocate(tmps.at("1774_bbbb_vooo"))(tmps.at("1774_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1_1p.at("bb")(bb, jb) *
                                             tmps.at("0144_bbbb_voov")(ab, ib, kb, bb))
      .allocate(tmps.at("1811_bbbb_vvoo"))(tmps.at("1811_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1774_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1774_bbbb_vooo"))
      .allocate(tmps.at("1810_bbbb_vvoo"))(tmps.at("1810_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, ib) * tmps.at("0188_bb_vo")(bb, jb))
      .deallocate(tmps.at("0188_bb_vo"))
      .allocate(tmps.at("1809_bbbb_vvoo"))(tmps.at("1809_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("1406_bb_vo")(bb, jb))
      .deallocate(tmps.at("1406_bb_vo"))
      .allocate(tmps.at("1808_bbbb_vvoo"))(tmps.at("1808_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0179_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0179_bbbb_vooo"))
      .allocate(tmps.at("1776_bbbb_vooo"))(tmps.at("1776_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1_2p.at("bb")(bb, jb) *
                                             tmps.at("0141_bbbb_vovo")(ab, ib, bb, kb))
      .allocate(tmps.at("1807_bbbb_vvoo"))(tmps.at("1807_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1776_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1776_bbbb_vooo"))
      .allocate(tmps.at("1805_bbbb_vooo"))(tmps.at("1805_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1.at("bb")(ab, lb) *
                                             tmps.at("1515_bbbb_oooo")(lb, ib, jb, kb))
      .allocate(tmps.at("1806_bbbb_vvoo"))(tmps.at("1806_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1805_bbbb_vooo")(bb, kb, ib, jb))
      .allocate(tmps.at("1804_bbbb_vvoo"))(tmps.at("1804_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0181_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0181_bbbb_vooo"))
      .allocate(tmps.at("1803_bbbb_vvoo"))(tmps.at("1803_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("1399_bb_vo")(bb, jb))
      .deallocate(tmps.at("1399_bb_vo"))
      .allocate(tmps.at("1800_bbbb_vvoo"))(tmps.at("1800_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("1260_bb_vo")(bb, jb))
      .deallocate(tmps.at("1260_bb_vo"))
      .allocate(tmps.at("1799_bbbb_vvoo"))(tmps.at("1799_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0141_bbbb_vovo")(bb, kb, cb, jb))
      .deallocate(tmps.at("0141_bbbb_vovo"))
      .allocate(tmps.at("1798_bbbb_vvoo"))(tmps.at("1798_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0557_baba_voov")(bb, ka, jb, ca))
      .deallocate(tmps.at("0557_baba_voov"))
      .allocate(tmps.at("1797_bbbb_vvoo"))(tmps.at("1797_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0135_baab_vovo")(bb, ka, ca, jb))
      .deallocate(tmps.at("0135_baab_vovo"))
      .allocate(tmps.at("1796_bbbb_vvoo"))(tmps.at("1796_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0146_bbbb_ovoo")(kb, bb, ib, jb))
      .deallocate(tmps.at("0146_bbbb_ovoo"))
      .allocate(tmps.at("1795_bbbb_vvoo"))(tmps.at("1795_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("1326_bb_vo")(bb, jb))
      .deallocate(tmps.at("1326_bb_vo"))
      .allocate(tmps.at("1794_bbbb_vvoo"))(tmps.at("1794_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0144_bbbb_voov")(bb, kb, jb, cb))
      .deallocate(tmps.at("0144_bbbb_voov"))
      .allocate(tmps.at("1793_bbbb_vvoo"))(tmps.at("1793_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, ib) * tmps.at("1335_bb_vo")(bb, jb))
      .deallocate(tmps.at("1335_bb_vo"))
      .allocate(tmps.at("1792_bbbb_vvoo"))(tmps.at("1792_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0170_bbbb_ovoo")(kb, bb, ib, jb))
      .deallocate(tmps.at("0170_bbbb_ovoo"))
      .allocate(tmps.at("1791_bbbb_vvoo"))(tmps.at("1791_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, ib) * tmps.at("0154_bb_vo")(bb, jb))
      .deallocate(tmps.at("0154_bb_vo"))
      .allocate(tmps.at("1764_bbbb_vooo"))(tmps.at("1764_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_2p.at("abab")(ba, ab, la, jb) *
                                             eri.at("abab_oovo")(la, ib, ba, kb))
      .allocate(tmps.at("1763_bbbb_vooo"))(tmps.at("1763_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2_2p.at("bbbb")(bb, ab, jb, lb) *
                                             eri.at("bbbb_oovo")(ib, lb, bb, kb))
      .allocate(tmps.at("1768_bbbb_vooo"))(tmps.at("1768_bbbb_vooo")(ab, ib, jb, kb) =
                                             tmps.at("1763_bbbb_vooo")(ab, ib, jb, kb))(
        tmps.at("1768_bbbb_vooo")(ab, ib, jb, kb) += tmps.at("1764_bbbb_vooo")(ab, ib, jb, kb))
      .deallocate(tmps.at("1764_bbbb_vooo"))
      .deallocate(tmps.at("1763_bbbb_vooo"))
      .allocate(tmps.at("1790_bbbb_vvoo"))(tmps.at("1790_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1768_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1768_bbbb_vooo"))
      .allocate(tmps.at("1789_bbbb_vvoo"))(tmps.at("1789_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("1245_bbbb_vovo")(bb, kb, cb, jb))
      .deallocate(tmps.at("1245_bbbb_vovo"))
      .allocate(tmps.at("1788_bbbb_vvoo"))(tmps.at("1788_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0137_baba_voov")(bb, ka, jb, ca))
      .deallocate(tmps.at("0137_baba_voov"))
      .allocate(tmps.at("1787_bbbb_vvoo"))(tmps.at("1787_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, ib) * tmps.at("0158_bb_vo")(bb, jb))
      .deallocate(tmps.at("0158_bb_vo"))
      .allocate(tmps.at("1786_bbbb_vvoo"))(tmps.at("1786_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, kb) *
                                             tmps.at("0560_bbbb_vovo")(bb, kb, cb, jb))
      .deallocate(tmps.at("0560_bbbb_vovo"))
      .allocate(tmps.at("1785_bbbb_vvoo"))(tmps.at("1785_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0129_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0129_bbbb_vooo"))
      .allocate(tmps.at("1784_bbbb_vvoo"))(tmps.at("1784_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_2p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0033_baba_voov")(bb, ka, jb, ca))
      .deallocate(tmps.at("0033_baba_voov"))
      .allocate(tmps.at("1783_bbbb_vvoo"))(tmps.at("1783_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0536_baab_vovo")(bb, ka, ca, jb))
      .deallocate(tmps.at("0536_baab_vovo"))
      .allocate(tmps.at("1782_bbbb_vvoo"))(tmps.at("1782_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0175_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0175_bbbb_vooo"))
      .allocate(tmps.at("1781_bbbb_vvoo"))(tmps.at("1781_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("abab")(ca, ab, ka, ib) *
                                             tmps.at("0133_baab_vovo")(bb, ka, ca, jb))
      .deallocate(tmps.at("0133_baab_vovo"))
      .allocate(tmps.at("1780_bbbb_vvoo"))(tmps.at("1780_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0150_bbbb_ovoo")(kb, bb, ib, jb))
      .deallocate(tmps.at("0150_bbbb_ovoo"))
      .allocate(tmps.at("1765_bbbb_vooo"))(tmps.at("1765_bbbb_vooo")(ab, ib, jb, kb) =
                                             t1_2p.at("bb")(bb, jb) *
                                             eri.at("bbbb_vovo")(ab, ib, bb, kb))
      .allocate(tmps.at("1779_bbbb_vvoo"))(tmps.at("1779_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1765_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1765_bbbb_vooo"))
      .allocate(tmps.at("1778_bbbb_vvoo"))(tmps.at("1778_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_2p.at("bb")(ab, kb) *
                                             tmps.at("0139_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0139_bbbb_vooo"))
      .allocate(tmps.at("1767_bbbb_vvoo"))(tmps.at("1767_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("baab_vovo")(ab, ka, ca, ib) *
                                             t2_2p.at("abab")(ca, bb, ka, jb))
      .allocate(tmps.at("1766_bbbb_vvoo"))(tmps.at("1766_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_vovo")(ab, kb, cb, ib) *
                                             t2_2p.at("bbbb")(cb, bb, jb, kb))
      .allocate(tmps.at("1769_bbbb_vvoo"))(tmps.at("1769_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1767_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1769_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1766_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1767_bbbb_vvoo"))
      .deallocate(tmps.at("1766_bbbb_vvoo"))
      .allocate(tmps.at("1801_bbbb_vvoo"))(tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1781_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -=
        2.00 * tmps.at("1791_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1794_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1780_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1795_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1790_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1787_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1778_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1796_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1788_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1786_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1785_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1779_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1792_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1783_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1800_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1799_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1798_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1784_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1789_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1782_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1793_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1797_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1769_bbbb_vvoo")(bb, ab, jb, ib))
      .deallocate(tmps.at("1800_bbbb_vvoo"))
      .deallocate(tmps.at("1799_bbbb_vvoo"))
      .deallocate(tmps.at("1798_bbbb_vvoo"))
      .deallocate(tmps.at("1797_bbbb_vvoo"))
      .deallocate(tmps.at("1796_bbbb_vvoo"))
      .deallocate(tmps.at("1795_bbbb_vvoo"))
      .deallocate(tmps.at("1794_bbbb_vvoo"))
      .deallocate(tmps.at("1793_bbbb_vvoo"))
      .deallocate(tmps.at("1792_bbbb_vvoo"))
      .deallocate(tmps.at("1791_bbbb_vvoo"))
      .deallocate(tmps.at("1790_bbbb_vvoo"))
      .deallocate(tmps.at("1789_bbbb_vvoo"))
      .deallocate(tmps.at("1788_bbbb_vvoo"))
      .deallocate(tmps.at("1787_bbbb_vvoo"))
      .deallocate(tmps.at("1786_bbbb_vvoo"))
      .deallocate(tmps.at("1785_bbbb_vvoo"))
      .deallocate(tmps.at("1784_bbbb_vvoo"))
      .deallocate(tmps.at("1783_bbbb_vvoo"))
      .deallocate(tmps.at("1782_bbbb_vvoo"))
      .deallocate(tmps.at("1781_bbbb_vvoo"))
      .deallocate(tmps.at("1780_bbbb_vvoo"))
      .deallocate(tmps.at("1779_bbbb_vvoo"))
      .deallocate(tmps.at("1778_bbbb_vvoo"))
      .deallocate(tmps.at("1769_bbbb_vvoo"))
      .allocate(tmps.at("1814_bbbb_vvoo"))(tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1801_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1806_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1803_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1809_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1813_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1812_bbbb_vvoo")(bb, ab, jb, ib))(
        tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1808_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1804_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1807_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) +=
        2.00 * tmps.at("1810_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1811_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1813_bbbb_vvoo"))
      .deallocate(tmps.at("1812_bbbb_vvoo"))
      .deallocate(tmps.at("1811_bbbb_vvoo"))
      .deallocate(tmps.at("1810_bbbb_vvoo"))
      .deallocate(tmps.at("1809_bbbb_vvoo"))
      .deallocate(tmps.at("1808_bbbb_vvoo"))
      .deallocate(tmps.at("1807_bbbb_vvoo"))
      .deallocate(tmps.at("1806_bbbb_vvoo"))
      .deallocate(tmps.at("1804_bbbb_vvoo"))
      .deallocate(tmps.at("1803_bbbb_vvoo"))
      .deallocate(tmps.at("1801_bbbb_vvoo"))
      .allocate(tmps.at("1817_bbbb_vvoo"))(tmps.at("1817_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1814_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1817_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1816_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1816_bbbb_vvoo"))
      .deallocate(tmps.at("1814_bbbb_vvoo"))(r2_2p.at("bbbb")(ab, bb, ib, jb) -=
                                             2.00 * tmps.at("1817_bbbb_vvoo")(bb, ab, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("1817_bbbb_vvoo")(bb, ab, jb, ib))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) += 2.00 * tmps.at("1817_bbbb_vvoo")(ab, bb, ib, jb))(
        r2_2p.at("bbbb")(ab, bb, ib, jb) -= 2.00 * tmps.at("1817_bbbb_vvoo")(ab, bb, jb, ib))
      .deallocate(tmps.at("1817_bbbb_vvoo"))
      .allocate(tmps.at("1843_bbbb_vvoo"))(tmps.at("1843_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1710_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1710_bbbb_vooo"))
      .allocate(tmps.at("1841_bbbb_vvoo"))(tmps.at("1841_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1707_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1707_bbbb_vooo"))
      .allocate(tmps.at("1840_bbbb_vvoo"))(tmps.at("1840_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0308_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0308_bbbb_vooo"))
      .allocate(tmps.at("1821_bbbb_vooo"))(tmps.at("1821_bbbb_vooo")(ab, ib, jb, kb) =
                                             t2.at("bbbb")(bb, ab, jb, kb) *
                                             tmps.at("0914_bb_ov")(ib, bb))
      .deallocate(tmps.at("0914_bb_ov"))
      .allocate(tmps.at("1839_bbbb_vvoo"))(tmps.at("1839_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1821_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1821_bbbb_vooo"))
      .allocate(tmps.at("1837_bbbb_vvoo"))(tmps.at("1837_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0544_bb_vv")(bb, cb))
      .deallocate(tmps.at("0544_bb_vv"))
      .allocate(tmps.at("1836_bbbb_vvoo"))(tmps.at("1836_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0305_bbbb_ovoo")(kb, bb, ib, jb))
      .deallocate(tmps.at("0305_bbbb_ovoo"))
      .allocate(tmps.at("1835_bbbb_vvoo"))(tmps.at("1835_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1738_bbbb_ovoo")(kb, bb, ib, jb))
      .deallocate(tmps.at("1738_bbbb_ovoo"))
      .allocate(tmps.at("1834_bbbb_vvoo"))(tmps.at("1834_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1726_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1726_bbbb_vooo"))
      .allocate(tmps.at("1833_bbbb_vvoo"))(tmps.at("1833_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0505_bb_vv")(bb, cb))
      .deallocate(tmps.at("0505_bb_vv"))
      .allocate(tmps.at("1832_bbbb_vvoo"))(tmps.at("1832_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0581_bb_vv")(bb, cb))
      .deallocate(tmps.at("0581_bb_vv"))
      .allocate(tmps.at("1831_bbbb_vvoo"))(tmps.at("1831_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1743_bbbb_ovoo")(kb, bb, ib, jb))
      .deallocate(tmps.at("1743_bbbb_ovoo"))
      .allocate(tmps.at("1830_bbbb_vvoo"))(tmps.at("1830_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0295_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0295_bbbb_vooo"))
      .allocate(tmps.at("1829_bbbb_vvoo"))(tmps.at("1829_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0297_bb_vv")(bb, cb))
      .deallocate(tmps.at("0297_bb_vv"))
      .allocate(tmps.at("1828_bbbb_vvoo"))(tmps.at("1828_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0583_bb_vv")(bb, cb))
      .deallocate(tmps.at("0583_bb_vv"))
      .allocate(tmps.at("1827_bbbb_vvoo"))(tmps.at("1827_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0996_bb_vv")(bb, cb))
      .deallocate(tmps.at("0996_bb_vv"))
      .allocate(tmps.at("1826_bbbb_vvoo"))(tmps.at("1826_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("0301_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("0301_bbbb_vooo"))
      .allocate(tmps.at("1825_bbbb_vvoo"))(tmps.at("1825_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0921_bb_vv")(bb, cb))
      .deallocate(tmps.at("0921_bb_vv"))
      .allocate(tmps.at("1824_bbbb_vvoo"))(tmps.at("1824_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(cb, ab, ib, jb) *
                                             tmps.at("0299_bb_vv")(bb, cb))
      .deallocate(tmps.at("0299_bb_vv"))
      .allocate(tmps.at("1823_bbbb_vvoo"))(tmps.at("1823_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1733_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1733_bbbb_vooo"))
      .allocate(tmps.at("1822_bbbb_vvoo"))(tmps.at("1822_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(ab, kb) *
                                             tmps.at("1702_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1702_bbbb_vooo"))
      .allocate(tmps.at("1819_bbbb_vvoo"))(tmps.at("1819_bbbb_vvoo")(ab, bb, ib, jb) =
                                             f.at("bb_vv")(ab, cb) *
                                             t2_1p.at("bbbb")(cb, bb, ib, jb))
      .allocate(tmps.at("1818_bbbb_vvoo"))(tmps.at("1818_bbbb_vvoo")(ab, bb, ib, jb) =
                                             eri.at("bbbb_vooo")(ab, kb, ib, jb) *
                                             t1_1p.at("bb")(bb, kb))
      .allocate(tmps.at("1820_bbbb_vvoo"))(tmps.at("1820_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1819_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1820_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1818_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1819_bbbb_vvoo"))
      .deallocate(tmps.at("1818_bbbb_vvoo"))
      .allocate(tmps.at("1838_bbbb_vvoo"))(tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -0.50 * tmps.at("1832_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1822_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1827_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1836_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1828_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1833_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1834_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1820_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("1823_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1835_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1831_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1826_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("1837_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1824_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1829_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) +=
        tmps.at("1825_bbbb_vvoo")(ab, bb, ib, jb))(tmps.at("1838_bbbb_vvoo")(ab, bb, ib, jb) +=
                                                   0.50 * tmps.at("1830_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1837_bbbb_vvoo"))
      .deallocate(tmps.at("1836_bbbb_vvoo"))
      .deallocate(tmps.at("1835_bbbb_vvoo"))
      .deallocate(tmps.at("1834_bbbb_vvoo"))
      .deallocate(tmps.at("1833_bbbb_vvoo"))
      .deallocate(tmps.at("1832_bbbb_vvoo"))
      .deallocate(tmps.at("1831_bbbb_vvoo"))
      .deallocate(tmps.at("1830_bbbb_vvoo"))
      .deallocate(tmps.at("1829_bbbb_vvoo"))
      .deallocate(tmps.at("1828_bbbb_vvoo"))
      .deallocate(tmps.at("1827_bbbb_vvoo"))
      .deallocate(tmps.at("1826_bbbb_vvoo"))
      .deallocate(tmps.at("1825_bbbb_vvoo"))
      .deallocate(tmps.at("1824_bbbb_vvoo"))
      .deallocate(tmps.at("1823_bbbb_vvoo"))
      .deallocate(tmps.at("1822_bbbb_vvoo"))
      .deallocate(tmps.at("1820_bbbb_vvoo"))
      .allocate(tmps.at("1842_bbbb_vvoo"))(tmps.at("1842_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -0.50 * tmps.at("1841_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1842_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1840_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1842_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1839_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1842_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1838_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1841_bbbb_vvoo"))
      .deallocate(tmps.at("1840_bbbb_vvoo"))
      .deallocate(tmps.at("1839_bbbb_vvoo"))
      .deallocate(tmps.at("1838_bbbb_vvoo"))
      .allocate(tmps.at("1844_bbbb_vvoo"))(tmps.at("1844_bbbb_vvoo")(ab, bb, ib, jb) =
                                             tmps.at("1843_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1844_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1842_bbbb_vvoo")(bb, ab, ib, jb))
      .deallocate(tmps.at("1843_bbbb_vvoo"))
      .deallocate(tmps.at("1842_bbbb_vvoo"))(r2_1p.at("bbbb")(ab, bb, ib, jb) -=
                                             tmps.at("1844_bbbb_vvoo")(bb, ab, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += tmps.at("1844_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1844_bbbb_vvoo"))
      .allocate(tmps.at("1868_bbbb_vvoo"))(tmps.at("1868_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1815_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1815_bbbb_vooo"))
      .allocate(tmps.at("1866_bbbb_vvoo"))(tmps.at("1866_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0596_bb_oo")(kb, jb))
      .deallocate(tmps.at("0596_bb_oo"))
      .allocate(tmps.at("1865_bbbb_vvoo"))(tmps.at("1865_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0289_bb_oo")(kb, jb))
      .deallocate(tmps.at("0289_bb_oo"))
      .allocate(tmps.at("1864_bbbb_vvoo"))(tmps.at("1864_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("1536_bbbb_oooo")(lb, kb, ib, jb))
      .deallocate(tmps.at("1536_bbbb_oooo"))
      .allocate(tmps.at("1863_bbbb_vvoo"))(tmps.at("1863_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0654_bb_oo")(kb, jb))
      .deallocate(tmps.at("0654_bb_oo"))
      .allocate(tmps.at("1862_bbbb_vvoo"))(tmps.at("1862_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1.at("bb")(ab, kb) *
                                             tmps.at("1805_bbbb_vooo")(bb, kb, ib, jb))
      .deallocate(tmps.at("1805_bbbb_vooo"))
      .allocate(tmps.at("1861_bbbb_vvoo"))(tmps.at("1861_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0598_bb_oo")(kb, jb))
      .deallocate(tmps.at("0598_bb_oo"))
      .allocate(tmps.at("1859_bbbb_vvoo"))(tmps.at("1859_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0566_bb_oo")(kb, jb))
      .deallocate(tmps.at("0566_bb_oo"))
      .allocate(tmps.at("1858_bbbb_vvoo"))(tmps.at("1858_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("0178_bbbb_oooo")(lb, kb, ib, jb))
      .deallocate(tmps.at("0178_bbbb_oooo"))
      .allocate(tmps.at("1857_bbbb_vvoo"))(tmps.at("1857_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0281_bb_oo")(kb, jb))
      .deallocate(tmps.at("0281_bb_oo"))
      .allocate(tmps.at("1856_bbbb_vvoo"))(tmps.at("1856_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0271_bb_oo")(kb, jb))
      .deallocate(tmps.at("0271_bb_oo"))
      .allocate(tmps.at("1855_bbbb_vvoo"))(tmps.at("1855_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0276_bb_oo")(kb, jb))
      .deallocate(tmps.at("0276_bb_oo"))
      .allocate(tmps.at("1854_bbbb_vvoo"))(tmps.at("1854_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(cb, ib) *
                                             tmps.at("1525_bbbb_vvvo")(ab, bb, cb, jb))
      .deallocate(tmps.at("1525_bbbb_vvvo"))
      .allocate(tmps.at("1853_bbbb_vvoo"))(tmps.at("1853_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0273_bb_oo")(kb, jb))
      .deallocate(tmps.at("0273_bb_oo"))
      .allocate(tmps.at("1852_bbbb_vvoo"))(tmps.at("1852_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, kb, lb) *
                                             tmps.at("1515_bbbb_oooo")(lb, kb, ib, jb))
      .deallocate(tmps.at("1515_bbbb_oooo"))
      .allocate(tmps.at("1851_bbbb_vvoo"))(tmps.at("1851_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0534_bb_oo")(kb, jb))
      .deallocate(tmps.at("0534_bb_oo"))
      .allocate(tmps.at("1850_bbbb_vvoo"))(tmps.at("1850_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0568_bb_oo")(kb, jb))
      .deallocate(tmps.at("0568_bb_oo"))
      .allocate(tmps.at("1849_bbbb_vvoo"))(tmps.at("1849_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0642_bb_oo")(kb, jb))
      .deallocate(tmps.at("0642_bb_oo"))
      .allocate(tmps.at("1848_bbbb_vvoo"))(tmps.at("1848_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2.at("bbbb")(ab, bb, ib, kb) *
                                             tmps.at("0528_bb_oo")(kb, jb))
      .deallocate(tmps.at("0528_bb_oo"))
      .allocate(tmps.at("1846_bbbb_vvoo"))(tmps.at("1846_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t1_1p.at("bb")(cb, ib) *
                                             eri.at("bbbb_vvvo")(ab, bb, cb, jb))
      .allocate(tmps.at("1845_bbbb_vvoo"))(tmps.at("1845_bbbb_vvoo")(ab, bb, ib, jb) =
                                             t2_1p.at("bbbb")(ab, bb, ib, kb) *
                                             f.at("bb_oo")(kb, jb))
      .allocate(tmps.at("1847_bbbb_vvoo"))(tmps.at("1847_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1846_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1847_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1845_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1846_bbbb_vvoo"))
      .deallocate(tmps.at("1845_bbbb_vvoo"))
      .allocate(tmps.at("1860_bbbb_vvoo"))(tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -0.50 * tmps.at("1852_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1847_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1856_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("1855_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("1851_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1849_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1853_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1859_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1848_bbbb_vvoo")(ab, bb, jb, ib))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1857_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1850_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) -=
        0.50 * tmps.at("1858_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1860_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1854_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1859_bbbb_vvoo"))
      .deallocate(tmps.at("1858_bbbb_vvoo"))
      .deallocate(tmps.at("1857_bbbb_vvoo"))
      .deallocate(tmps.at("1856_bbbb_vvoo"))
      .deallocate(tmps.at("1855_bbbb_vvoo"))
      .deallocate(tmps.at("1854_bbbb_vvoo"))
      .deallocate(tmps.at("1853_bbbb_vvoo"))
      .deallocate(tmps.at("1852_bbbb_vvoo"))
      .deallocate(tmps.at("1851_bbbb_vvoo"))
      .deallocate(tmps.at("1850_bbbb_vvoo"))
      .deallocate(tmps.at("1849_bbbb_vvoo"))
      .deallocate(tmps.at("1848_bbbb_vvoo"))
      .deallocate(tmps.at("1847_bbbb_vvoo"))
      .allocate(tmps.at("1867_bbbb_vvoo"))(tmps.at("1867_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1861_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1867_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1863_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1867_bbbb_vvoo")(ab, bb, ib, jb) -= tmps.at("1865_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1867_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1860_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1867_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1862_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1867_bbbb_vvoo")(ab, bb, ib, jb) +=
        0.50 * tmps.at("1864_bbbb_vvoo")(bb, ab, ib, jb))(
        tmps.at("1867_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1866_bbbb_vvoo")(bb, ab, jb, ib))
      .deallocate(tmps.at("1866_bbbb_vvoo"))
      .deallocate(tmps.at("1865_bbbb_vvoo"))
      .deallocate(tmps.at("1864_bbbb_vvoo"))
      .deallocate(tmps.at("1863_bbbb_vvoo"))
      .deallocate(tmps.at("1862_bbbb_vvoo"))
      .deallocate(tmps.at("1861_bbbb_vvoo"))
      .deallocate(tmps.at("1860_bbbb_vvoo"))
      .allocate(tmps.at("1869_bbbb_vvoo"))(tmps.at("1869_bbbb_vvoo")(ab, bb, ib, jb) =
                                             -1.00 * tmps.at("1867_bbbb_vvoo")(ab, bb, ib, jb))(
        tmps.at("1869_bbbb_vvoo")(ab, bb, ib, jb) += tmps.at("1868_bbbb_vvoo")(ab, bb, ib, jb))
      .deallocate(tmps.at("1868_bbbb_vvoo"))
      .deallocate(tmps.at("1867_bbbb_vvoo"))(r2_1p.at("bbbb")(ab, bb, ib, jb) -=
                                             tmps.at("1869_bbbb_vvoo")(bb, ab, ib, jb))(
        r2_1p.at("bbbb")(ab, bb, ib, jb) += tmps.at("1869_bbbb_vvoo")(bb, ab, jb, ib))
      .deallocate(tmps.at("1869_bbbb_vvoo"))
      .allocate(tmps.at("1914_aaaa_vvoo"))(tmps.at("1914_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("1134_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1134_aaaa_vooo"))
      .allocate(tmps.at("1882_aaaa_vooo"))(tmps.at("1882_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_1p.at("aaaa")(ba, aa, ja, ka) *
                                             tmps.at("1230_aa_ov")(ia, ba))
      .deallocate(tmps.at("1230_aa_ov"))
      .allocate(tmps.at("1879_aaaa_vooo"))(tmps.at("1879_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, ka) *
                                             tmps.at("0354_aa_ov")(ia, ba))
      .deallocate(tmps.at("0354_aa_ov"))
      .allocate(tmps.at("1877_aaaa_vooo"))(tmps.at("1877_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, ka) *
                                             tmps.at("1240_aa_ov")(ia, ba))
      .deallocate(tmps.at("1240_aa_ov"))
      .allocate(tmps.at("1876_aaaa_vooo"))(tmps.at("1876_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_1p.at("aaaa")(ba, aa, ja, ka) *
                                             tmps.at("0365_aa_ov")(ia, ba))
      .deallocate(tmps.at("0365_aa_ov"))
      .allocate(tmps.at("1903_aaaa_vooo"))(tmps.at("1903_aaaa_vooo")(aa, ia, ja, ka) =
                                             -1.00 * tmps.at("1882_aaaa_vooo")(aa, ia, ja, ka))(
        tmps.at("1903_aaaa_vooo")(aa, ia, ja, ka) += tmps.at("1876_aaaa_vooo")(aa, ia, ja, ka))(
        tmps.at("1903_aaaa_vooo")(aa, ia, ja, ka) += tmps.at("1879_aaaa_vooo")(aa, ia, ja, ka))(
        tmps.at("1903_aaaa_vooo")(aa, ia, ja, ka) += tmps.at("1877_aaaa_vooo")(aa, ia, ja, ka))
      .deallocate(tmps.at("1882_aaaa_vooo"))
      .deallocate(tmps.at("1879_aaaa_vooo"))
      .deallocate(tmps.at("1877_aaaa_vooo"))
      .deallocate(tmps.at("1876_aaaa_vooo"))
      .allocate(tmps.at("1912_aaaa_vvoo"))(tmps.at("1912_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1903_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1903_aaaa_vooo"))
      .allocate(tmps.at("1881_aaaa_vooo"))(tmps.at("1881_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_2p.at("aaaa")(ba, aa, ja, ka) *
                                             tmps.at("0224_aa_ov")(ia, ba))
      .deallocate(tmps.at("0224_aa_ov"))
      .allocate(tmps.at("1878_aaaa_vooo"))(tmps.at("1878_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_2p.at("aaaa")(ba, aa, ja, ka) *
                                             tmps.at("0232_aa_ov")(ia, ba))
      .deallocate(tmps.at("0232_aa_ov"))
      .allocate(tmps.at("1904_aaaa_vooo"))(tmps.at("1904_aaaa_vooo")(aa, ia, ja, ka) =
                                             tmps.at("1878_aaaa_vooo")(aa, ia, ja, ka))(
        tmps.at("1904_aaaa_vooo")(aa, ia, ja, ka) += tmps.at("1881_aaaa_vooo")(aa, ia, ja, ka))
      .deallocate(tmps.at("1881_aaaa_vooo"))
      .deallocate(tmps.at("1878_aaaa_vooo"))
      .allocate(tmps.at("1911_aaaa_vvoo"))(tmps.at("1911_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1904_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1904_aaaa_vooo"))
      .allocate(tmps.at("1910_aaaa_vvoo"))(tmps.at("1910_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0206_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0206_aaaa_vooo"))
      .allocate(tmps.at("1909_aaaa_vvoo"))(tmps.at("1909_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0674_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0674_aaaa_vooo"))
      .allocate(tmps.at("1908_aaaa_vvoo"))(tmps.at("1908_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("1131_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1131_aaaa_vooo"))
      .allocate(tmps.at("1880_aaaa_vooo"))(tmps.at("1880_aaaa_vooo")(aa, ia, ja, ka) =
                                             t1_1p.at("aa")(ba, ja) *
                                             tmps.at("0348_aaaa_vovo")(aa, ia, ba, ka))
      .deallocate(tmps.at("0348_aaaa_vovo"))
      .allocate(tmps.at("1907_aaaa_vvoo"))(tmps.at("1907_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1880_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1880_aaaa_vooo"))
      .allocate(tmps.at("1875_aaaa_vooo"))(tmps.at("1875_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2.at("aaaa")(ba, aa, ja, ka) *
                                             tmps.at("1214_aa_ov")(ia, ba))
      .deallocate(tmps.at("1214_aa_ov"))
      .allocate(tmps.at("1906_aaaa_vvoo"))(tmps.at("1906_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1875_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1875_aaaa_vooo"))
      .allocate(tmps.at("1870_aaaa_vooo"))(tmps.at("1870_aaaa_vooo")(aa, ia, ja, ka) =
                                             eri.at("aaaa_vovv")(aa, ia, ba, ca) *
                                             t2_2p.at("aaaa")(ba, ca, ja, ka))
      .allocate(tmps.at("1902_aaaa_vvoo"))(tmps.at("1902_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1870_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1870_aaaa_vooo"))
      .allocate(tmps.at("1901_aaaa_vvoo"))(tmps.at("1901_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("1330_aa_vv")(ba, ca))
      .deallocate(tmps.at("1330_aa_vv"))
      .allocate(tmps.at("1900_aaaa_vvoo"))(tmps.at("1900_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1117_aaaa_ovoo")(ka, ba, ia, ja))
      .deallocate(tmps.at("1117_aaaa_ovoo"))
      .allocate(tmps.at("1899_aaaa_vvoo"))(tmps.at("1899_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0671_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0671_aaaa_vooo"))
      .allocate(tmps.at("1898_aaaa_vvoo"))(tmps.at("1898_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1125_aaaa_ovoo")(ka, ba, ia, ja))
      .deallocate(tmps.at("1125_aaaa_ovoo"))
      .allocate(tmps.at("1897_aaaa_vvoo"))(tmps.at("1897_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0669_aaaa_ovoo")(ka, ba, ia, ja))
      .deallocate(tmps.at("0669_aaaa_ovoo"))
      .allocate(tmps.at("1896_aaaa_vvoo"))(tmps.at("1896_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0664_aa_vv")(ba, ca))
      .deallocate(tmps.at("0664_aa_vv"))
      .allocate(tmps.at("1895_aaaa_vvoo"))(tmps.at("1895_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0403_aa_vv")(ba, ca))
      .deallocate(tmps.at("0403_aa_vv"))
      .allocate(tmps.at("1894_aaaa_vvoo"))(tmps.at("1894_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("1012_aa_vv")(ba, ca))
      .deallocate(tmps.at("1012_aa_vv"))
      .allocate(tmps.at("1893_aaaa_vvoo"))(tmps.at("1893_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1119_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1119_aaaa_vooo"))
      .allocate(tmps.at("1892_aaaa_vvoo"))(tmps.at("1892_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_1p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("1271_aa_vv")(ba, ca))
      .deallocate(tmps.at("1271_aa_vv"))
      .allocate(tmps.at("1891_aaaa_vvoo"))(tmps.at("1891_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0662_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0662_aaaa_vooo"))
      .allocate(tmps.at("1890_aaaa_vvoo"))(tmps.at("1890_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("0113_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0113_aaaa_vooo"))
      .allocate(tmps.at("1889_aaaa_vvoo"))(tmps.at("1889_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("0120_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("0120_aaaa_vooo"))
      .allocate(tmps.at("1871_aaaa_vooo"))(tmps.at("1871_aaaa_vooo")(aa, ia, ja, ka) =
                                             t2_2p.at("aaaa")(ba, aa, ja, ka) *
                                             f.at("aa_ov")(ia, ba))
      .allocate(tmps.at("1888_aaaa_vvoo"))(tmps.at("1888_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1.at("aa")(aa, ka) *
                                             tmps.at("1871_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1871_aaaa_vooo"))
      .allocate(tmps.at("1887_aaaa_vvoo"))(tmps.at("1887_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0226_aa_vv")(ba, ca))
      .deallocate(tmps.at("0226_aa_vv"))
      .allocate(tmps.at("1886_aaaa_vvoo"))(tmps.at("1886_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_1p.at("aa")(aa, ka) *
                                             tmps.at("1112_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1112_aaaa_vooo"))
      .allocate(tmps.at("1885_aaaa_vvoo"))(tmps.at("1885_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t1_2p.at("aa")(aa, ka) *
                                             tmps.at("1108_aaaa_vooo")(ba, ka, ia, ja))
      .deallocate(tmps.at("1108_aaaa_vooo"))
      .allocate(tmps.at("1884_aaaa_vvoo"))(tmps.at("1884_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0041_aa_vv")(ba, ca))
      .deallocate(tmps.at("0041_aa_vv"))
      .allocate(tmps.at("1883_aaaa_vvoo"))(tmps.at("1883_aaaa_vvoo")(aa, ba, ia, ja) =
                                             t2_2p.at("aaaa")(ca, aa, ia, ja) *
                                             tmps.at("0374_aa_vv")(ba, ca))
      .deallocate(tmps.at("0374_aa_vv"))
      .allocate(tmps.at("1873_aaaa_vvoo"))(tmps.at("1873_aaaa_vvoo")(aa, ba, ia, ja) =
                                             eri.at("aaaa_vooo")(aa, ka, ia, ja) *
                                             t1_2p.at("aa")(ba, ka))
      .allocate(tmps.at("1872_aaaa_vvoo"))(tmps.at("1872_aaaa_vvoo")(aa, ba, ia, ja) =
                                             f.at("aa_vv")(aa, ca) *
                                             t2_2p.at("aaaa")(ca, ba, ia, ja))
      .allocate(tmps.at("1874_aaaa_vvoo"))(tmps.at("1874_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -1.00 * tmps.at("1873_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1874_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1872_aaaa_vvoo")(aa, ba, ia, ja))
      .deallocate(tmps.at("1873_aaaa_vvoo"))
      .deallocate(tmps.at("1872_aaaa_vvoo"))
      .allocate(tmps.at("1905_aaaa_vvoo"))(tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) =
                                             -0.50 * tmps.at("1886_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1891_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1887_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) -=
        3.00 * tmps.at("1889_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1885_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1896_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1900_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1897_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.50 * tmps.at("1883_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) +=
        3.00 * tmps.at("1890_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1894_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1902_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1899_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1892_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) +=
        0.50 * tmps.at("1895_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1901_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1898_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1874_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1893_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1884_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1905_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1888_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1902_aaaa_vvoo"))
      .deallocate(tmps.at("1901_aaaa_vvoo"))
      .deallocate(tmps.at("1900_aaaa_vvoo"))
      .deallocate(tmps.at("1899_aaaa_vvoo"))
      .deallocate(tmps.at("1898_aaaa_vvoo"))
      .deallocate(tmps.at("1897_aaaa_vvoo"))
      .deallocate(tmps.at("1896_aaaa_vvoo"))
      .deallocate(tmps.at("1895_aaaa_vvoo"))
      .deallocate(tmps.at("1894_aaaa_vvoo"))
      .deallocate(tmps.at("1893_aaaa_vvoo"))
      .deallocate(tmps.at("1892_aaaa_vvoo"))
      .deallocate(tmps.at("1891_aaaa_vvoo"))
      .deallocate(tmps.at("1890_aaaa_vvoo"))
      .deallocate(tmps.at("1889_aaaa_vvoo"))
      .deallocate(tmps.at("1888_aaaa_vvoo"))
      .deallocate(tmps.at("1887_aaaa_vvoo"))
      .deallocate(tmps.at("1886_aaaa_vvoo"))
      .deallocate(tmps.at("1885_aaaa_vvoo"))
      .deallocate(tmps.at("1884_aaaa_vvoo"))
      .deallocate(tmps.at("1883_aaaa_vvoo"))
      .deallocate(tmps.at("1874_aaaa_vvoo"))
      .allocate(tmps.at("1913_aaaa_vvoo"))(tmps.at("1913_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1911_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1913_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1908_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1913_aaaa_vvoo")(aa, ba, ia, ja) -=
        0.50 * tmps.at("1910_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1913_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1909_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1913_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1912_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1913_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1905_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1913_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1906_aaaa_vvoo")(ba, aa, ia, ja))(
        tmps.at("1913_aaaa_vvoo")(aa, ba, ia, ja) -= tmps.at("1907_aaaa_vvoo")(ba, aa, ja, ia))
      .deallocate(tmps.at("1912_aaaa_vvoo"))
      .deallocate(tmps.at("1911_aaaa_vvoo"))
      .deallocate(tmps.at("1910_aaaa_vvoo"))
      .deallocate(tmps.at("1909_aaaa_vvoo"))
      .deallocate(tmps.at("1908_aaaa_vvoo"))
      .deallocate(tmps.at("1907_aaaa_vvoo"))
      .deallocate(tmps.at("1906_aaaa_vvoo"))
      .deallocate(tmps.at("1905_aaaa_vvoo"))
      .allocate(tmps.at("1915_aaaa_vvoo"))(tmps.at("1915_aaaa_vvoo")(aa, ba, ia, ja) =
                                             tmps.at("1913_aaaa_vvoo")(aa, ba, ia, ja))(
        tmps.at("1915_aaaa_vvoo")(aa, ba, ia, ja) += tmps.at("1914_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1914_aaaa_vvoo"))
      .deallocate(tmps.at("1913_aaaa_vvoo"))(r2_2p.at("aaaa")(aa, ba, ia, ja) -=
                                             2.00 * tmps.at("1915_aaaa_vvoo")(aa, ba, ia, ja))(
        r2_2p.at("aaaa")(aa, ba, ia, ja) += 2.00 * tmps.at("1915_aaaa_vvoo")(ba, aa, ia, ja))
      .deallocate(tmps.at("1915_aaaa_vvoo"));
  }
}

template void exachem::cc::qed_ccsd_os::resid_8<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);