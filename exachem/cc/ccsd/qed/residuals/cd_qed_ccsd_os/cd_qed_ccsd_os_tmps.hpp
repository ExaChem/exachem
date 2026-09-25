/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "../../cd_qed_ccsd_os.hpp"

template<typename T>
void exachem::cc::cd_qed_ccsd_os::build_tmps(Scheduler& sch, ChemEnv& chem_env, TensorMap<T>& tmps,
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
    // clang-format off
    scalars["0001"]() = Tensor<T>{}; scalars["0002"]() = Tensor<T>{}; scalars["0003"]() = Tensor<T>{};
    scalars["0004"]() = Tensor<T>{}; scalars["0005"]() = Tensor<T>{}; scalars["0006"]() = Tensor<T>{};
    scalars["0007"]() = Tensor<T>{}; scalars["0008"]() = Tensor<T>{}; scalars["0009"]() = Tensor<T>{};
    // clang-format on

    tmps["bin1_Q"]         = Tensor<T>{Q};
    tmps["bin1_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["bin1_aa_ooQ"]    = Tensor<T>{Oa, Oa, Q};
    tmps["bin1_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["bin1_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["bin1_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["bin1_aaaa_oooo"] = Tensor<T>{Oa, Oa, Oa, Oa};
    tmps["bin1_aaaa_vooo"] = Tensor<T>{Va, Oa, Oa, Oa};
    tmps["bin1_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["bin1_aaaa_vvvo"] = Tensor<T>{Va, Va, Va, Oa};
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
    tmps["bin1_bbbb_oooo"] = Tensor<T>{Ob, Ob, Ob, Ob};
    tmps["bin1_bbbb_vooo"] = Tensor<T>{Vb, Ob, Ob, Ob};
    tmps["bin1_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["bin1_bbbb_vvvo"] = Tensor<T>{Vb, Vb, Vb, Ob};
    tmps["bin2_aa_ooQ"]    = Tensor<T>{Oa, Oa, Q};
    tmps["bin2_aaaa_vooo"] = Tensor<T>{Va, Oa, Oa, Oa};
    tmps["bin2_bb_ooQ"]    = Tensor<T>{Ob, Ob, Q};
    tmps["bin2_bbbb_vooo"] = Tensor<T>{Vb, Ob, Ob, Ob};
  }

  for(auto& [name, tmp]: tmps) sch.allocate(tmp);
  for(auto& [name, scalar]: scalars) sch.allocate(scalar);

  {
    tmps["0001_bbbb_vvvv"] = Tensor<T>{Vb, Vb, Vb, Vb};
    tmps["0002_aaaa_vvvv"] = Tensor<T>{Va, Va, Va, Va};
    tmps["0003_aabb_vvvv"] = Tensor<T>{Va, Va, Vb, Vb};
    tmps["0004_bbbb_ovoo"] = Tensor<T>{Ob, Vb, Ob, Ob};
    tmps["0005_aaaa_ovoo"] = Tensor<T>{Oa, Va, Oa, Oa};
    tmps["0006_bbbb_ovoo"] = Tensor<T>{Ob, Vb, Ob, Ob};
    tmps["0007_aaaa_ovoo"] = Tensor<T>{Oa, Va, Oa, Oa};
    tmps["0008_baba_ovoo"] = Tensor<T>{Ob, Va, Ob, Oa};
    tmps["0009_baab_ovoo"] = Tensor<T>{Ob, Va, Oa, Ob};
    tmps["0010_bbaa_ovoo"] = Tensor<T>{Ob, Vb, Oa, Oa};
    tmps["0011_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0012_baba_ovoo"] = Tensor<T>{Ob, Va, Ob, Oa};
    tmps["0013_baab_ovoo"] = Tensor<T>{Ob, Va, Oa, Ob};
    tmps["0014_bbaa_ovoo"] = Tensor<T>{Ob, Vb, Oa, Oa};
    tmps["0015_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0016_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0017_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0018_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0019_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0020_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0021_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0022_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0023_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0024_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0025_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0026_bb_ooQ"]    = Tensor<T>{Ob, Ob, Q};
    tmps["0027_bbaa_oovo"] = Tensor<T>{Ob, Ob, Va, Oa};
    tmps["0028_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0029_bb_ooQ"]    = Tensor<T>{Ob, Ob, Q};
    tmps["0030_bbaa_oovo"] = Tensor<T>{Ob, Ob, Va, Oa};
    tmps["0031_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0032_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0033_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0034_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0035_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0036_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0037_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0038_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0039_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0040_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0041_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0042_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0043_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0044_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0045_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0046_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0047_aabb_ovvv"] = Tensor<T>{Oa, Va, Vb, Vb};
    tmps["0048_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0049_Q"]         = Tensor<T>{Q};
    tmps["0050_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0051_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0052_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0053_aa_ooQ"]    = Tensor<T>{Oa, Oa, Q};
    tmps["0054_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0055_aaaa_oovv"] = Tensor<T>{Oa, Oa, Va, Va};
    tmps["0056_aaaa_ooov"] = Tensor<T>{Oa, Oa, Oa, Va};
    tmps["0057_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0058_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0059_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0060_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0061_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0062_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0063_bbbb_oovv"] = Tensor<T>{Ob, Ob, Vb, Vb};
    tmps["0064_bbbb_ooov"] = Tensor<T>{Ob, Ob, Ob, Vb};
    tmps["0065_bbbb_voov"] = Tensor<T>{Vb, Ob, Ob, Vb};
    tmps["0066_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0067_bbbb_voov"] = Tensor<T>{Vb, Ob, Ob, Vb};
    tmps["0068_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0069_bbbb_voov"] = Tensor<T>{Vb, Ob, Ob, Vb};
    tmps["0070_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0071_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0072_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0073_aaaa_ovov"] = Tensor<T>{Oa, Va, Oa, Va};
    tmps["0074_aaaa_oooo"] = Tensor<T>{Oa, Oa, Oa, Oa};
    tmps["0075_bbbb_ovov"] = Tensor<T>{Ob, Vb, Ob, Vb};
    tmps["0076_bbbb_oooo"] = Tensor<T>{Ob, Ob, Ob, Ob};
    tmps["0077_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0078_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0079_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["0080_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0081_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0082_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0083_aabb_ovov"] = Tensor<T>{Oa, Va, Ob, Vb};
    tmps["0084_abab_oooo"] = Tensor<T>{Oa, Ob, Oa, Ob};
    tmps["0085_aabb_vvoo"] = Tensor<T>{Va, Va, Ob, Ob};
    tmps["0086_bbbb_ovvv"] = Tensor<T>{Ob, Vb, Vb, Vb};
    tmps["0087_bbbb_ovvo"] = Tensor<T>{Ob, Vb, Vb, Ob};
    tmps["0088_bbbb_ovvo"] = Tensor<T>{Ob, Vb, Vb, Ob};
    tmps["0089_bbbb_ovvo"] = Tensor<T>{Ob, Vb, Vb, Ob};
    tmps["0090_aaaa_ovvv"] = Tensor<T>{Oa, Va, Va, Va};
    tmps["0091_aaaa_ovvo"] = Tensor<T>{Oa, Va, Va, Oa};
    tmps["0092_aaaa_ovvo"] = Tensor<T>{Oa, Va, Va, Oa};
    tmps["0093_aaaa_ovvo"] = Tensor<T>{Oa, Va, Va, Oa};
    tmps["0094_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0095_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0096_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0097_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0098_aabb_oovv"] = Tensor<T>{Oa, Oa, Vb, Vb};
    tmps["0099_aabb_vvov"] = Tensor<T>{Va, Va, Ob, Vb};
    tmps["0100_aabb_vvoo"] = Tensor<T>{Va, Va, Ob, Ob};
    tmps["0101_bbbb_voov"] = Tensor<T>{Vb, Ob, Ob, Vb};
    tmps["0102_aabb_oovv"] = Tensor<T>{Oa, Oa, Vb, Vb};
    tmps["0103_aabb_vvoo"] = Tensor<T>{Va, Va, Ob, Ob};
    tmps["0104_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0105_aa_ov"]     = Tensor<T>{Oa, Va};
    tmps["0106_baab_ovoo"] = Tensor<T>{Ob, Va, Oa, Ob};
    tmps["0107_baab_ovoo"] = Tensor<T>{Ob, Va, Oa, Ob};
    tmps["0108_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0109_abab_vvoo"] = Tensor<T>{Va, Vb, Oa, Ob};
    tmps["0110_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0111_bbbb_ovvo"] = Tensor<T>{Ob, Vb, Vb, Ob};
    tmps["0112_bbbb_ovvo"] = Tensor<T>{Ob, Vb, Vb, Ob};
    tmps["0113_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0114_aaaa_ovvo"] = Tensor<T>{Oa, Va, Va, Oa};
    tmps["0115_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0116_aaaa_ovvo"] = Tensor<T>{Oa, Va, Va, Oa};
    tmps["0117_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0118_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0119_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0120_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0121_aabb_ooov"] = Tensor<T>{Oa, Oa, Ob, Vb};
    tmps["0122_aabb_oooo"] = Tensor<T>{Oa, Oa, Ob, Ob};
    tmps["0123_aabb_oooo"] = Tensor<T>{Oa, Oa, Ob, Ob};
    tmps["0124_aabb_oovo"] = Tensor<T>{Oa, Oa, Vb, Ob};
    tmps["0125_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0126_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0127_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0128_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0129_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0130_bbbb_voov"] = Tensor<T>{Vb, Ob, Ob, Vb};
    tmps["0131_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0132_bbbb_voov"] = Tensor<T>{Vb, Ob, Ob, Vb};
    tmps["0133_aabb_oovv"] = Tensor<T>{Oa, Oa, Vb, Vb};
    tmps["0134_aabb_oovo"] = Tensor<T>{Oa, Oa, Vb, Ob};
    tmps["0135_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0136_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0137_aa_ooQ"]    = Tensor<T>{Oa, Oa, Q};
    tmps["0138_aabb_oovo"] = Tensor<T>{Oa, Oa, Vb, Ob};
    tmps["0139_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0140_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0141_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0142_aaaa_oooo"] = Tensor<T>{Oa, Oa, Oa, Oa};
    tmps["0143_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0144_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0145_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0146_aa_voQ"]    = Tensor<T>{Va, Oa, Q};
    tmps["0147_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0148_Q"]         = Tensor<T>{Q};
    tmps["0149_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0150_Q"]         = Tensor<T>{Q};
    tmps["0151_Q"]         = Tensor<T>{Q};
    tmps["0152_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0153_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0154_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0155_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0156_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0157_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0158_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0159_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0160_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0161_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0162_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0163_abab_voov"] = Tensor<T>{Va, Ob, Oa, Vb};
    tmps["0164_aaaa_ooov"] = Tensor<T>{Oa, Oa, Oa, Va};
    tmps["0165_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0166_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0167_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0168_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0169_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0170_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0171_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0172_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0173_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0174_Q"]         = Tensor<T>{Q};
    tmps["0175_aa_ov"]     = Tensor<T>{Oa, Va};
    tmps["0176_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0177_baba_voov"] = Tensor<T>{Vb, Oa, Ob, Va};
    tmps["0178_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0179_aabb_vvoo"] = Tensor<T>{Va, Va, Ob, Ob};
    tmps["0180_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["0181_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0182_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0183_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0184_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0185_bbbb_voov"] = Tensor<T>{Vb, Ob, Ob, Vb};
    tmps["0186_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0187_bb_ooQ"]    = Tensor<T>{Ob, Ob, Q};
    tmps["0188_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0189_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0190_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0191_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0192_bbbb_oooo"] = Tensor<T>{Ob, Ob, Ob, Ob};
    tmps["0193_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0194_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0195_bb_voQ"]    = Tensor<T>{Vb, Ob, Q};
    tmps["0196_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0197_bbbb_ooov"] = Tensor<T>{Ob, Ob, Ob, Vb};
    tmps["0198_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0199_Q"]         = Tensor<T>{Q};
    tmps["0200_bb_ov"]     = Tensor<T>{Ob, Vb};
    tmps["0201_bb_ov"]     = Tensor<T>{Ob, Vb};
    tmps["0202_bb_ov"]     = Tensor<T>{Ob, Vb};
    tmps["0203_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0204_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0205_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0206_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0207_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0208_bb_ov"]     = Tensor<T>{Ob, Vb};
    tmps["0209_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0210_abab_oooo"] = Tensor<T>{Oa, Ob, Oa, Ob};
    tmps["0211_aabb_ovoo"] = Tensor<T>{Oa, Va, Ob, Ob};
    tmps["0212_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0213_abba_voov"] = Tensor<T>{Va, Ob, Ob, Va};
    tmps["0214_abba_vooo"] = Tensor<T>{Va, Ob, Ob, Oa};
    tmps["0215_aa_vv"]     = Tensor<T>{Va, Va};
    tmps["0216_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0217_abab_vvoo"] = Tensor<T>{Va, Vb, Oa, Ob};
    tmps["0218_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["0219_abba_voov"] = Tensor<T>{Va, Ob, Ob, Va};
    tmps["0220_abba_vooo"] = Tensor<T>{Va, Ob, Ob, Oa};
    tmps["0221_aa_ooQ"]    = Tensor<T>{Oa, Oa, Q};
    tmps["0222_aabb_oooo"] = Tensor<T>{Oa, Oa, Ob, Ob};
    tmps["0223_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0224_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0225_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0226_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0227_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0228_bbbb_ovoo"] = Tensor<T>{Ob, Vb, Ob, Ob};
    tmps["0229_bbbb_ovoo"] = Tensor<T>{Ob, Vb, Ob, Ob};
    tmps["0230_aabb_vooo"] = Tensor<T>{Va, Oa, Ob, Ob};
    tmps["0231_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0232_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0233_bbbb_voov"] = Tensor<T>{Vb, Ob, Ob, Vb};
    tmps["0234_bbbb_voov"] = Tensor<T>{Vb, Ob, Ob, Vb};
    tmps["0235_aaaa_voov"] = Tensor<T>{Va, Oa, Oa, Va};
    tmps["0236_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0237_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0238_bbbb_voov"] = Tensor<T>{Vb, Ob, Ob, Vb};
    tmps["0239_bbaa_vooo"] = Tensor<T>{Vb, Ob, Oa, Oa};
    tmps["0240_bbaa_vooo"] = Tensor<T>{Vb, Ob, Oa, Oa};
    tmps["0241_abab_vooo"] = Tensor<T>{Va, Ob, Oa, Ob};
    tmps["0242_abab_vvoo"] = Tensor<T>{Va, Vb, Oa, Ob};
    tmps["0243_abab_vvoo"] = Tensor<T>{Va, Vb, Oa, Ob};
    tmps["0244_abab_vvoo"] = Tensor<T>{Va, Vb, Oa, Ob};
    tmps["0245_bbbb_ovoo"] = Tensor<T>{Ob, Vb, Ob, Ob};
    tmps["0246_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0247_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0248_abab_ovoo"] = Tensor<T>{Oa, Vb, Oa, Ob};
    tmps["0249_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0250_aabb_oovo"] = Tensor<T>{Oa, Oa, Vb, Ob};
    tmps["0251_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0252_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0253_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0254_bbbb_oovv"] = Tensor<T>{Ob, Ob, Vb, Vb};
    tmps["0255_bbbb_ovvo"] = Tensor<T>{Ob, Vb, Vb, Ob};
    tmps["0256_bbbb_oovv"] = Tensor<T>{Ob, Ob, Vb, Vb};
    tmps["0257_bbbb_ovvo"] = Tensor<T>{Ob, Vb, Vb, Ob};
    tmps["0258_bbbb_ovvo"] = Tensor<T>{Ob, Vb, Vb, Ob};
    tmps["0259_aaaa_oovv"] = Tensor<T>{Oa, Oa, Va, Va};
    tmps["0260_aaaa_oovv"] = Tensor<T>{Oa, Oa, Va, Va};
    tmps["0261_aaaa_ovvo"] = Tensor<T>{Oa, Va, Va, Oa};
    tmps["0262_aaaa_ovvo"] = Tensor<T>{Oa, Va, Va, Oa};
    tmps["0263_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0264_bbbb_ovvo"] = Tensor<T>{Ob, Vb, Vb, Ob};
    tmps["0265_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0266_aaaa_ovvo"] = Tensor<T>{Oa, Va, Va, Oa};
    tmps["0267_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0268_abab_oooo"] = Tensor<T>{Oa, Ob, Oa, Ob};
    tmps["0269_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0270_baab_vooo"] = Tensor<T>{Vb, Oa, Oa, Ob};
    tmps["0271_aaaa_oooo"] = Tensor<T>{Oa, Oa, Oa, Oa};
    tmps["0272_bbbb_oooo"] = Tensor<T>{Ob, Ob, Ob, Ob};
    tmps["0273_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0274_aaaa_oooo"] = Tensor<T>{Oa, Oa, Oa, Oa};
    tmps["0275_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0276_bbbb_oooo"] = Tensor<T>{Ob, Ob, Ob, Ob};
    tmps["0277_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0278_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0279_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0280_bb_vv"]     = Tensor<T>{Vb, Vb};
    tmps["0281_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0282_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0283_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0284_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0285_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0286_aaaa_ovvo"] = Tensor<T>{Oa, Va, Va, Oa};
    tmps["0287_aabb_oooo"] = Tensor<T>{Oa, Oa, Ob, Ob};
    tmps["0288_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0289_aabb_oooo"] = Tensor<T>{Oa, Oa, Ob, Ob};
    tmps["0290_bbbb_ooov"] = Tensor<T>{Ob, Ob, Ob, Vb};
    tmps["0291_bbbb_ovov"] = Tensor<T>{Ob, Vb, Ob, Vb};
    tmps["0292_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0293_bbbb_vooo"] = Tensor<T>{Vb, Ob, Ob, Ob};
    tmps["0294_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0295_aaaa_ooov"] = Tensor<T>{Oa, Oa, Oa, Va};
    tmps["0296_aaaa_ovov"] = Tensor<T>{Oa, Va, Oa, Va};
    tmps["0297_aaaa_vooo"] = Tensor<T>{Va, Oa, Oa, Oa};
    tmps["0298_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0299_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0300_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0301_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0302_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0303_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0304_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0305_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0306_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0307_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0308_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0309_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0310_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0311_aaaa_ooov"] = Tensor<T>{Oa, Oa, Oa, Va};
    tmps["0312_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0313_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0314_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0315_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0316_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0317_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0318_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0319_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0320_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0321_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0322_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0323_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0324_aa_vo"]     = Tensor<T>{Va, Oa};
    tmps["0325_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0326_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0327_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0328_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0329_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0330_bb_vo"]     = Tensor<T>{Vb, Ob};
    tmps["0331_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0332_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0333_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0334_bbbb_ovov"] = Tensor<T>{Ob, Vb, Ob, Vb};
    tmps["0335_bbbb_ovov"] = Tensor<T>{Ob, Vb, Ob, Vb};
    tmps["0336_aaaa_ovov"] = Tensor<T>{Oa, Va, Oa, Va};
    tmps["0337_aaaa_ovov"] = Tensor<T>{Oa, Va, Oa, Va};
    tmps["0338_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0339_bbbb_vooo"] = Tensor<T>{Vb, Ob, Ob, Ob};
    tmps["0340_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0341_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0342_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0343_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0344_bbbb_vovo"] = Tensor<T>{Vb, Ob, Vb, Ob};
    tmps["0345_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0346_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0347_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0348_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0349_aaaa_vovo"] = Tensor<T>{Va, Oa, Va, Oa};
    tmps["0350_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0351_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0352_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0353_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0354_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0355_bbbb_vvoo"] = Tensor<T>{Vb, Vb, Ob, Ob};
    tmps["0356_aa_oo"]     = Tensor<T>{Oa, Oa};
    tmps["0357_aaaa_vvoo"] = Tensor<T>{Va, Va, Oa, Oa};
    tmps["0358_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0359_bb_oo"]     = Tensor<T>{Ob, Ob};
    tmps["0360_bb_oo"]     = Tensor<T>{Ob, Ob};
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

template void exachem::cc::cd_qed_ccsd_os::build_tmps<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& chol, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p);