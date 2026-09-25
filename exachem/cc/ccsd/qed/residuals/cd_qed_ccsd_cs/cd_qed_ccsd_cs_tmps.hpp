/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "../../cd_qed_ccsd_cs.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_cs::build_tmps(Scheduler& sch, ChemEnv& chem_env, TensorMap<T>& tmps,
                                             TensorMap<T>& scalars, const TensorMap<T>& f,
                                             const TensorMap<T>& chol, const TensorMap<T>& dp,
                                             const double w0, const TensorMap<T>& t1,
                                             const TensorMap<T>& t2, const double t0_1p,
                                             const TensorMap<T>& t1_1p, const TensorMap<T>& t2_1p,
                                             const double t0_2p, const TensorMap<T>& t1_2p,
                                             const TensorMap<T>& t2_2p) {
  TiledIndexSpace& MO      = chem_env.is_context.MSO;
  TiledIndexSpace& Q       = chem_env.is_context.CI;
  const int        otiles  = MO("occ").num_tiles();
  const int        vtiles  = MO("virt").num_tiles();
  const int        oatiles = MO("occ_alpha").num_tiles();
  const int        vatiles = MO("virt_alpha").num_tiles();

  const TiledIndexSpace Oa{MO("occ"), range(oatiles)};
  const TiledIndexSpace Va{MO("virt"), range(vatiles)};
  const TiledIndexSpace Ob{MO("occ"), range(oatiles, otiles)};
  const TiledIndexSpace Vb{MO("virt"), range(vatiles, vtiles)};

  {
    scalars["0001"]()      = Tensor<T>{};
    scalars["0002"]()      = Tensor<T>{};
    scalars["0003"]()      = Tensor<T>{};
    scalars["0004"]()      = Tensor<T>{};
    scalars["0005"]()      = Tensor<T>{};
    scalars["0006"]()      = Tensor<T>{};
    scalars["0007"]()      = Tensor<T>{};
    scalars["0008"]()      = Tensor<T>{};
    scalars["0009"]()      = Tensor<T>{};
    tmps["bin1_Q"]         = Tensor<T>{Q};
    tmps["bin1_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["bin1_aa_ooQ"]    = Tensor<T>{Oa, Oa, Q};
    tmps["bin1_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["bin1_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["bin1_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["bin1_aaaa_vooo"] = Tensor<T>{Va, Oa, Oa, Oa};
    tmps["bin1_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["bin1_aabb_oooo"] = Tensor<T>{Oa, Oa, Ob, Ob};
    tmps["bin1_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["bin1_aabb_vvoo"] = Tensor<T>{Va, Va, Ob, Ob};
    tmps["bin1_abab_vvoo"] = Tensor<T>{Va, Vb, Oa, Ob};
    tmps["bin1_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["bin1_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["bin1_bb_ooQ"]    = Tensor<T>{Ob, Ob, Q};
    tmps["bin1_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["bin1_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["bin1_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["bin1_bbaa_vvoo"] = Tensor<T>{Vb, Vb, Oa, Oa};
    tmps["bin1_bbbb_vooo"] = Tensor<T>{Vb, Ob, Ob, Ob};
    tmps["bin1_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["bin2_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
  }

  for(auto& [name, tmp]: tmps) sch.allocate(tmp);
  for(auto& [name, scalar]: scalars) sch.allocate(scalar);

  {
    tmps["0001_aabb_vvvv"] = Tensor<T>{Va, Va, Vb, Vb};
    tmps["0002_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0003_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0004_baba_vooo"] = Tensor<T>{Vb, Oa, Ob, Oa};
    tmps["0005_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0006_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0007_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0008_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0009_baba_vooo"] = Tensor<T>{Vb, Oa, Ob, Oa};
    tmps["0010_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0011_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0012_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0013_baab_vvoo"] = Tensor<T>{Vb, Va, Oa, Ob};
    tmps["0014_baab_vvoo"] = Tensor<T>{Vb, Va, Oa, Ob};
    tmps["0015_baab_vvoo"] = Tensor<T>{Vb, Va, Oa, Ob};
    tmps["0016_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0017_baab_ovoo"] = Tensor<T>{Ob, Va, Oa, Ob};
    tmps["0018_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0019_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0020_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0021_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0022_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0023_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0024_abab_oooo"] = Tensor<T>{Oa, Ob, Oa, Ob};
    tmps["0025_bbaa_oovo"] = Tensor<T>{Ob, Ob, Va, Oa};
    tmps["0026_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0027_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0028_aabb_oovo"] = Tensor<T>{Oa, Oa, Vb, Ob};
    tmps["0029_Q"]         = Tensor<T>{Q};
    tmps["0030_Q"]         = Tensor<T>{Q};
    tmps["0031_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0032_Q"]         = Tensor<T>{Q};
    tmps["0033_Q"]         = Tensor<T>{Q};
    tmps["0034_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0035_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0036_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0037_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0038_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0039_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0040_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0041_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0042_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0043_abba_vvoo"] = Tensor<T>{Va, Vb, Ob, Oa};
    tmps["0044_abba_vvoo"] = Tensor<T>{Va, Vb, Ob, Oa};
    tmps["0045_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0046_abba_vvoo"] = Tensor<T>{Va, Vb, Ob, Oa};
    tmps["0047_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0048_bbaa_vvov"] = Tensor<T>{Vb, Vb, Oa, Va};
    tmps["0049_aabb_ovoo"] = Tensor<T>{Oa, Va, Ob, Ob};
    tmps["0050_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0051_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0052_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0053_aabb_ovov"] = Tensor<T>{Oa, Va, Ob, Vb};
    tmps["0054_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0055_bbbb_ovov"] = Tensor<T>{Ob, Vb, Ob, Vb};
    tmps["0056_baba_ovoo"] = Tensor<T>{Ob, Va, Ob, Oa};
    tmps["0057_abba_ovoo"] = Tensor<T>{Oa, Vb, Ob, Oa};
    tmps["0058_aaaa_ovov"] = Tensor<T>{Oa, Va, Oa, Va};
    tmps["0059_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0060_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0061_aaaa_ovoo"] = Tensor<T>{Oa, Va, Oa, Oa};
    tmps["0062_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0063_aabb_vvoo"] = Tensor<T>{Va, Va, Ob, Ob};
    tmps["0064_bbaa_vvoo"] = Tensor<T>{Vb, Vb, Oa, Oa};
    tmps["0065_aabb_vvov"] = Tensor<T>{Va, Va, Ob, Vb};
    tmps["0066_aabb_vvoo"] = Tensor<T>{Va, Va, Ob, Ob};
    tmps["0067_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0068_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0069_bbaa_vvoo"] = Tensor<T>{Vb, Vb, Oa, Oa};
    tmps["0070_aabb_vvoo"] = Tensor<T>{Va, Va, Ob, Ob};
    tmps["0071_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0072_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0073_bbaa_oovo"] = Tensor<T>{Ob, Ob, Va, Oa};
    tmps["0074_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0075_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["0076_aaaa_ovoo"] = Tensor<T>{Oa, Va, Oa, Oa};
    tmps["0077_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["0078_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0079_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0080_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0081_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0082_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0083_aabb_ovoo"] = Tensor<T>{Oa, Va, Ob, Ob};
    tmps["0084_aa_ov"]     = Tensor<T>{Oa, Va};
    tmps["0085_bb_ov"]     = Tensor<T>{Ob, Vb};
    tmps["0086_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0087_aa_ov"]     = Tensor<T>{Oa, Va};
    tmps["0088_aaaa_ooov"] = Tensor<T>{Oa, Oa, Oa, Va};
    tmps["0089_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0090_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0091_bbbb_ooov"] = Tensor<T>{Ob, Ob, Ob, Vb};
    tmps["0092_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0093_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0094_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0095_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0096_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0097_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0098_aa_ooQ"]    = Tensor<T>{Oa, Oa, Q};
    tmps["0099_bb_ooQ"]    = Tensor<T>{Ob, Ob, Q};
    tmps["0100_aabb_oooo"] = Tensor<T>{Oa, Oa, Ob, Ob};
    tmps["0101_aabb_oovo"] = Tensor<T>{Oa, Oa, Vb, Ob};
    tmps["0102_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0103_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0104_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0105_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0106_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0107_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0108_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0109_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0110_abab_vvoo"] = Tensor<T>{Va, Vb, Oa, Ob};
    tmps["0111_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0112_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0113_abab_vvoo"] = Tensor<T>{Va, Vb, Oa, Ob};
    tmps["0114_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0115_abab_vvoo"] = Tensor<T>{Va, Vb, Oa, Ob};
    tmps["0116_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0117_aabb_vvoo"] = Tensor<T>{Va, Va, Ob, Ob};
    tmps["0118_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["0119_bb_ov"]     = Tensor<T>{Ob, Vb};
    tmps["0120_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0121_abab_oooo"] = Tensor<T>{Oa, Ob, Oa, Ob};
    tmps["0122_aabb_ooov"] = Tensor<T>{Oa, Oa, Ob, Vb};
    tmps["0123_aa_ooQ"]    = Tensor<T>{Oa, Oa, Q};
    tmps["0124_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0125_abab_oooo"] = Tensor<T>{Oa, Ob, Oa, Ob};
    tmps["0126_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0127_abab_voov"] = Tensor<T>{Va, Ob, Oa, Vb};
    tmps["0128_abba_voov"] = Tensor<T>{Va, Ob, Ob, Va};
    tmps["0129_abba_vooo"] = Tensor<T>{Va, Ob, Ob, Oa};
    tmps["0130_bbaa_vvoo"] = Tensor<T>{Vb, Vb, Oa, Oa};
    tmps["0131_bbaa_vooo"] = Tensor<T>{Vb, Ob, Oa, Oa};
    tmps["0132_bb_ooQ"]    = Tensor<T>{Ob, Ob, Q};
    tmps["0133_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0134_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0135_abba_vooo"] = Tensor<T>{Va, Ob, Ob, Oa};
    tmps["0136_bbaa_vooo"] = Tensor<T>{Vb, Ob, Oa, Oa};
    tmps["0137_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0138_aabb_ovoo"] = Tensor<T>{Oa, Va, Ob, Ob};
    tmps["0139_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["0140_aabb_ovoo"] = Tensor<T>{Oa, Va, Ob, Ob};
    tmps["0141_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["0142_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0143_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0144_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0145_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0146_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0147_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0148_abab_voov"] = Tensor<T>{Va, Ob, Oa, Vb};
    tmps["0149_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0150_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0151_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0152_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0153_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0154_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0155_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0156_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0157_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0158_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0159_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0160_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0161_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0162_Q"]         = Tensor<T>{Q};
    tmps["0163_Q"]         = Tensor<T>{Q};
    tmps["0164_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0165_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0166_aabb_oovo"] = Tensor<T>{Oa, Oa, Vb, Ob};
    tmps["0167_aabb_oovo"] = Tensor<T>{Oa, Oa, Vb, Ob};
    tmps["0168_aa_ooQ"]    = Tensor<T>{Oa, Oa, Q};
    tmps["0169_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0170_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0171_baba_ovoo"] = Tensor<T>{Ob, Va, Ob, Oa};
    tmps["0172_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0173_abba_ovoo"] = Tensor<T>{Oa, Vb, Ob, Oa};
    tmps["0174_aabb_oooo"] = Tensor<T>{Oa, Oa, Ob, Ob};
    tmps["0175_bb_ooQ"]    = Tensor<T>{Ob, Ob, Q};
    tmps["0176_aabb_oooo"] = Tensor<T>{Oa, Oa, Ob, Ob};
    tmps["0177_aabb_oooo"] = Tensor<T>{Oa, Oa, Ob, Ob};
    tmps["0178_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0179_aaaa_ooov"] = Tensor<T>{Oa, Oa, Oa, Va};
    tmps["0180_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0181_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0182_aabb_vvoo"] = Tensor<T>{Va, Va, Ob, Ob};
    tmps["0183_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["0184_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0185_aaaa_ovoo"] = Tensor<T>{Oa, Va, Oa, Oa};
    tmps["0186_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0187_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0188_aa_oo"]     = Tensor<T>{Oa, Oa};
  }

  {
    // clang-format off
    sch
      (scalars.at("0001")() = 0.0)(scalars.at("0002")() = 0.0)(scalars.at("0003")() = 0.0)
      (scalars.at("0004")() = 0.0)(scalars.at("0005")() = 0.0)(scalars.at("0006")() = 0.0)
      (scalars.at("0007")() = 0.0)(scalars.at("0008")() = 0.0)(scalars.at("0009")() = 0.0)
    ;
    // clang-format on
  }
}

template void exachem::cc::cd_qed_ccsd_cs::build_tmps<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p);