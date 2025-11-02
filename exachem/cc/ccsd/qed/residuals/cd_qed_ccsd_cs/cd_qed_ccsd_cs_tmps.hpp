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
  {
    scalars["0001"]() = Tensor<T>{};
    scalars["0002"]() = Tensor<T>{};
    scalars["0003"]() = Tensor<T>{};
    scalars["0004"]() = Tensor<T>{};
    scalars["0005"]() = Tensor<T>{};
    scalars["0006"]() = Tensor<T>{};
    scalars["0007"]() = Tensor<T>{};
    scalars["0008"]() = Tensor<T>{};
    scalars["0009"]() = Tensor<T>{};
    scalars["0010"]() = Tensor<T>{};
    scalars["0011"]() = Tensor<T>{};
    scalars["0012"]() = Tensor<T>{};
    scalars["0013"]() = Tensor<T>{};
    scalars["0014"]() = Tensor<T>{};
    scalars["0015"]() = Tensor<T>{};
    scalars["0016"]() = Tensor<T>{};
    scalars["0017"]() = Tensor<T>{};
    scalars["0018"]() = Tensor<T>{};
    scalars["0019"]() = Tensor<T>{};
    scalars["0020"]() = Tensor<T>{};
    scalars["0021"]() = Tensor<T>{};
    scalars["0022"]() = Tensor<T>{};
    scalars["0023"]() = Tensor<T>{};
    scalars["0024"]() = Tensor<T>{};
    scalars["0025"]() = Tensor<T>{};
    scalars["0026"]() = Tensor<T>{};
    scalars["0027"]() = Tensor<T>{};
    scalars["0028"]() = Tensor<T>{};
    scalars["0029"]() = Tensor<T>{};
    scalars["0030"]() = Tensor<T>{};
    scalars["0031"]() = Tensor<T>{};
    scalars["0032"]() = Tensor<T>{};
    scalars["0033"]() = Tensor<T>{};
    scalars["0034"]() = Tensor<T>{};
    scalars["0035"]() = Tensor<T>{};
    scalars["0036"]() = Tensor<T>{};
    scalars["0037"]() = Tensor<T>{};
    scalars["0038"]() = Tensor<T>{};
    scalars["0039"]() = Tensor<T>{};
    scalars["0040"]() = Tensor<T>{};
    scalars["0041"]() = Tensor<T>{};
    scalars["0042"]() = Tensor<T>{};
    scalars["0043"]() = Tensor<T>{};
    scalars["0044"]() = Tensor<T>{};
    scalars["0045"]() = Tensor<T>{};
    scalars["0046"]() = Tensor<T>{};
    scalars["0047"]() = Tensor<T>{};
    scalars["0048"]() = Tensor<T>{};
    scalars["0049"]() = Tensor<T>{};
    scalars["0050"]() = Tensor<T>{};
    scalars["0051"]() = Tensor<T>{};
    scalars["0052"]() = Tensor<T>{};
    scalars["0053"]() = Tensor<T>{};
    scalars["0054"]() = Tensor<T>{};
    scalars["0055"]() = Tensor<T>{};
    scalars["0056"]() = Tensor<T>{};
    scalars["0057"]() = Tensor<T>{};
    scalars["0058"]() = Tensor<T>{};
    scalars["0059"]() = Tensor<T>{};
    scalars["0060"]() = Tensor<T>{};
    scalars["0061"]() = Tensor<T>{};
    scalars["0062"]() = Tensor<T>{};
    scalars["0063"]() = Tensor<T>{};
    scalars["0064"]() = Tensor<T>{};
    scalars["0065"]() = Tensor<T>{};
    scalars["0066"]() = Tensor<T>{};
    scalars["0067"]() = Tensor<T>{};
    scalars["0068"]() = Tensor<T>{};

    tmps["bin2_aaaa_vvoo"] = declare<T>(chem_env, "bin2_aaaa_vvoo");
    tmps["bin_Q"]          = declare<T>(chem_env, "bin_Q");
    tmps["bin_aa_oo"]      = declare<T>(chem_env, "bin_aa_oo");
    tmps["bin_aa_ooQ"]     = declare<T>(chem_env, "bin_aa_ooQ");
    tmps["bin_aa_vo"]      = declare<T>(chem_env, "bin_aa_vo");
    tmps["bin_aa_voQ"]     = declare<T>(chem_env, "bin_aa_voQ");
    tmps["bin_aa_vv"]      = declare<T>(chem_env, "bin_aa_vv");
    tmps["bin_aaaa_vooo"]  = declare<T>(chem_env, "bin_aaaa_vooo");
    tmps["bin_aaaa_vvoo"]  = declare<T>(chem_env, "bin_aaaa_vvoo");
    tmps["bin_aabb_oooo"]  = declare<T>(chem_env, "bin_aabb_oooo");
    tmps["bin_aabb_vooo"]  = declare<T>(chem_env, "bin_aabb_vooo");
    tmps["bin_aabb_vvoo"]  = declare<T>(chem_env, "bin_aabb_vvoo");
    tmps["bin_aabb_vvvo"]  = declare<T>(chem_env, "bin_aabb_vvvo");
    tmps["bin_abab_vvoo"]  = declare<T>(chem_env, "bin_abab_vvoo");
    tmps["bin_abba_vvvo"]  = declare<T>(chem_env, "bin_abba_vvvo");
    tmps["bin_baab_vooo"]  = declare<T>(chem_env, "bin_baab_vooo");
    tmps["bin_bb_oo"]      = declare<T>(chem_env, "bin_bb_oo");
    tmps["bin_bb_ooQ"]     = declare<T>(chem_env, "bin_bb_ooQ");
    tmps["bin_bb_vo"]      = declare<T>(chem_env, "bin_bb_vo");
    tmps["bin_bb_voQ"]     = declare<T>(chem_env, "bin_bb_voQ");
    tmps["bin_bb_vv"]      = declare<T>(chem_env, "bin_bb_vv");
    tmps["bin_bbaa_vvoo"]  = declare<T>(chem_env, "bin_bbaa_vvoo");
    tmps["bin_bbbb_vooo"]  = declare<T>(chem_env, "bin_bbbb_vooo");
    tmps["bin_bbbb_vvoo"]  = declare<T>(chem_env, "bin_bbbb_vvoo");
  }

  for(auto& [name, tmp]: tmps) sch.allocate(tmp);
  for(auto& [name, scalar]: scalars) sch.allocate(scalar);

  {
    tmps["0001_baba_vvoo"] = declare<T>(chem_env, "0001_baba_vvoo");
    tmps["0002_baab_vooo"] = declare<T>(chem_env, "0002_baab_vooo");
    tmps["0003_abab_vooo"] = declare<T>(chem_env, "0003_abab_vooo");
    tmps["0004_abab_vooo"] = declare<T>(chem_env, "0004_abab_vooo");
    tmps["0005_aaaa_vvoo"] = declare<T>(chem_env, "0005_aaaa_vvoo");
    tmps["0006_baba_ovoo"] = declare<T>(chem_env, "0006_baba_ovoo");
    tmps["0007_aabb_ovoo"] = declare<T>(chem_env, "0007_aabb_ovoo");
    tmps["0008_abba_ovoo"] = declare<T>(chem_env, "0008_abba_ovoo");
    tmps["0009_abab_oooo"] = declare<T>(chem_env, "0009_abab_oooo");
    tmps["0010_aabb_ovoo"] = declare<T>(chem_env, "0010_aabb_ovoo");
    tmps["0011_bbaa_ovoo"] = declare<T>(chem_env, "0011_bbaa_ovoo");
    tmps["0012_bbaa_ovoo"] = declare<T>(chem_env, "0012_bbaa_ovoo");
    tmps["0013_abba_ovoo"] = declare<T>(chem_env, "0013_abba_ovoo");
    tmps["0014_abab_ovoo"] = declare<T>(chem_env, "0014_abab_ovoo");
    tmps["0015_abab_ovoo"] = declare<T>(chem_env, "0015_abab_ovoo");
    tmps["0016_abab_ovoo"] = declare<T>(chem_env, "0016_abab_ovoo");
    tmps["0017_abab_ovoo"] = declare<T>(chem_env, "0017_abab_ovoo");
    tmps["0018_aabb_vvvv"] = declare<T>(chem_env, "0018_aabb_vvvv");
    tmps["0019_bbbb_oovv"] = declare<T>(chem_env, "0019_bbbb_oovv");
    tmps["0020_bbbb_vvoo"] = declare<T>(chem_env, "0020_bbbb_vvoo");
    tmps["0021_bbbb_oovv"] = declare<T>(chem_env, "0021_bbbb_oovv");
    tmps["0022_aabb_vvoo"] = declare<T>(chem_env, "0022_aabb_vvoo");
    tmps["0023_baab_vvoo"] = declare<T>(chem_env, "0023_baab_vvoo");
    tmps["0024_baab_vvoo"] = declare<T>(chem_env, "0024_baab_vvoo");
    tmps["0025_baab_vvoo"] = declare<T>(chem_env, "0025_baab_vvoo");
    tmps["0026_abab_vvoo"] = declare<T>(chem_env, "0026_abab_vvoo");
    tmps["0027_abab_vvoo"] = declare<T>(chem_env, "0027_abab_vvoo");
    tmps["0028_abab_vvoo"] = declare<T>(chem_env, "0028_abab_vvoo");
    tmps["0029_bbaa_vooo"] = declare<T>(chem_env, "0029_bbaa_vooo");
    tmps["0030_aabb_oovo"] = declare<T>(chem_env, "0030_aabb_oovo");
    tmps["0031_bbaa_ooov"] = declare<T>(chem_env, "0031_bbaa_ooov");
    tmps["0032_abba_vvoo"] = declare<T>(chem_env, "0032_abba_vvoo");
    tmps["0033_abba_vvoo"] = declare<T>(chem_env, "0033_abba_vvoo");
    tmps["0034_abba_vvoo"] = declare<T>(chem_env, "0034_abba_vvoo");
    tmps["0035_baab_ovoo"] = declare<T>(chem_env, "0035_baab_ovoo");
    tmps["0036_baab_ovoo"] = declare<T>(chem_env, "0036_baab_ovoo");
    tmps["0037_abab_vooo"] = declare<T>(chem_env, "0037_abab_vooo");
    tmps["0038_abba_vvoo"] = declare<T>(chem_env, "0038_abba_vvoo");
    tmps["0039_abba_vvoo"] = declare<T>(chem_env, "0039_abba_vvoo");
    tmps["0040_baab_ovoo"] = declare<T>(chem_env, "0040_baab_ovoo");
    tmps["0041_baab_ovoo"] = declare<T>(chem_env, "0041_baab_ovoo");
    tmps["0042_abba_vvoo"] = declare<T>(chem_env, "0042_abba_vvoo");
    tmps["0043_abba_vvoo"] = declare<T>(chem_env, "0043_abba_vvoo");
    tmps["0044_abba_vvoo"] = declare<T>(chem_env, "0044_abba_vvoo");
    tmps["0045_abba_vvoo"] = declare<T>(chem_env, "0045_abba_vvoo");
    tmps["0046_abab_vvoo"] = declare<T>(chem_env, "0046_abab_vvoo");
    tmps["0047_abab_vvoo"] = declare<T>(chem_env, "0047_abab_vvoo");
    tmps["0048_abab_vvoo"] = declare<T>(chem_env, "0048_abab_vvoo");
    tmps["0049_abab_ovoo"] = declare<T>(chem_env, "0049_abab_ovoo");
    tmps["0050_abab_ovoo"] = declare<T>(chem_env, "0050_abab_ovoo");
    tmps["0051_baab_vooo"] = declare<T>(chem_env, "0051_baab_vooo");
    tmps["0052_abab_vvoo"] = declare<T>(chem_env, "0052_abab_vvoo");
    tmps["0053_abab_vvoo"] = declare<T>(chem_env, "0053_abab_vvoo");
    tmps["0054_abab_ovoo"] = declare<T>(chem_env, "0054_abab_ovoo");
    tmps["0055_abab_ovoo"] = declare<T>(chem_env, "0055_abab_ovoo");
    tmps["0056_baab_vooo"] = declare<T>(chem_env, "0056_baab_vooo");
    tmps["0057_abab_vvoo"] = declare<T>(chem_env, "0057_abab_vvoo");
    tmps["0058_abab_vvoo"] = declare<T>(chem_env, "0058_abab_vvoo");
    tmps["0059_abab_vvoo"] = declare<T>(chem_env, "0059_abab_vvoo");
    tmps["0060_abab_vvoo"] = declare<T>(chem_env, "0060_abab_vvoo");
    tmps["0061_aa_vo"]     = declare<T>(chem_env, "0061_aa_vo");
    tmps["0062_aa_vo"]     = declare<T>(chem_env, "0062_aa_vo");
    tmps["0063_aa_vo"]     = declare<T>(chem_env, "0063_aa_vo");
    tmps["0064_bb_vo"]     = declare<T>(chem_env, "0064_bb_vo");
    tmps["0065_bb_vo"]     = declare<T>(chem_env, "0065_bb_vo");
    tmps["0066_aa_vo"]     = declare<T>(chem_env, "0066_aa_vo");
    tmps["0067_aa_vo"]     = declare<T>(chem_env, "0067_aa_vo");
    tmps["0068_aa_vo"]     = declare<T>(chem_env, "0068_aa_vo");
    tmps["0069_bb_vv"]     = declare<T>(chem_env, "0069_bb_vv");
    tmps["0070_bb_vv"]     = declare<T>(chem_env, "0070_bb_vv");
    tmps["0071_bb_vv"]     = declare<T>(chem_env, "0071_bb_vv");
    tmps["0072_bb_vv"]     = declare<T>(chem_env, "0072_bb_vv");
    tmps["0073_aa_vv"]     = declare<T>(chem_env, "0073_aa_vv");
    tmps["0074_aa_vv"]     = declare<T>(chem_env, "0074_aa_vv");
    tmps["0075_aa_vv"]     = declare<T>(chem_env, "0075_aa_vv");
    tmps["0076_aa_vv"]     = declare<T>(chem_env, "0076_aa_vv");
    tmps["0077_bb_vo"]     = declare<T>(chem_env, "0077_bb_vo");
    tmps["0078_aa_vo"]     = declare<T>(chem_env, "0078_aa_vo");
    tmps["0079_aa_vo"]     = declare<T>(chem_env, "0079_aa_vo");
    tmps["0080_aa_vo"]     = declare<T>(chem_env, "0080_aa_vo");
    tmps["0081_bb_oo"]     = declare<T>(chem_env, "0081_bb_oo");
    tmps["0082_bb_oo"]     = declare<T>(chem_env, "0082_bb_oo");
    tmps["0083_bb_oo"]     = declare<T>(chem_env, "0083_bb_oo");
    tmps["0084_bb_vo"]     = declare<T>(chem_env, "0084_bb_vo");
    tmps["0085_aa_oo"]     = declare<T>(chem_env, "0085_aa_oo");
    tmps["0086_aa_vo"]     = declare<T>(chem_env, "0086_aa_vo");
    tmps["0087_aa_vo"]     = declare<T>(chem_env, "0087_aa_vo");
    tmps["0088_aa_oo"]     = declare<T>(chem_env, "0088_aa_oo");
    tmps["0089_aa_vo"]     = declare<T>(chem_env, "0089_aa_vo");
    tmps["0090_aa_oo"]     = declare<T>(chem_env, "0090_aa_oo");
    tmps["0091_bbaa_vvov"] = declare<T>(chem_env, "0091_bbaa_vvov");
    tmps["0092_bbaa_vvoo"] = declare<T>(chem_env, "0092_bbaa_vvoo");
    tmps["0093_bbaa_vvoo"] = declare<T>(chem_env, "0093_bbaa_vvoo");
    tmps["0094_aaaa_ovov"] = declare<T>(chem_env, "0094_aaaa_ovov");
    tmps["0095_aa_vv"]     = declare<T>(chem_env, "0095_aa_vv");
    tmps["0096_aa_vv"]     = declare<T>(chem_env, "0096_aa_vv");
    tmps["0097_bb_voQ"]    = declare<T>(chem_env, "0097_bb_voQ");
    tmps["0098_aa_ooQ"]    = declare<T>(chem_env, "0098_aa_ooQ");
    tmps["0099_bbaa_vooo"] = declare<T>(chem_env, "0099_bbaa_vooo");
    tmps["0100_bb_voQ"]    = declare<T>(chem_env, "0100_bb_voQ");
    tmps["0101_bbaa_vooo"] = declare<T>(chem_env, "0101_bbaa_vooo");
    tmps["0102_bb_voQ"]    = declare<T>(chem_env, "0102_bb_voQ");
    tmps["0103_bbaa_vooo"] = declare<T>(chem_env, "0103_bbaa_vooo");
    tmps["0104_aa_ooQ"]    = declare<T>(chem_env, "0104_aa_ooQ");
    tmps["0105_bbaa_vooo"] = declare<T>(chem_env, "0105_bbaa_vooo");
    tmps["0106_aa_voQ"]    = declare<T>(chem_env, "0106_aa_voQ");
    tmps["0107_bb_ooQ"]    = declare<T>(chem_env, "0107_bb_ooQ");
    tmps["0108_aabb_vooo"] = declare<T>(chem_env, "0108_aabb_vooo");
    tmps["0109_aa_voQ"]    = declare<T>(chem_env, "0109_aa_voQ");
    tmps["0110_aabb_vooo"] = declare<T>(chem_env, "0110_aabb_vooo");
    tmps["0111_aa_voQ"]    = declare<T>(chem_env, "0111_aa_voQ");
    tmps["0112_aabb_vooo"] = declare<T>(chem_env, "0112_aabb_vooo");
    tmps["0113_aa_voQ"]    = declare<T>(chem_env, "0113_aa_voQ");
    tmps["0114_aabb_vooo"] = declare<T>(chem_env, "0114_aabb_vooo");
    tmps["0115_bb_ooQ"]    = declare<T>(chem_env, "0115_bb_ooQ");
    tmps["0116_aabb_vooo"] = declare<T>(chem_env, "0116_aabb_vooo");
    tmps["0117_bbbb_ovov"] = declare<T>(chem_env, "0117_bbbb_ovov");
    tmps["0118_bb_oo"]     = declare<T>(chem_env, "0118_bb_oo");
    tmps["0119_bb_oo"]     = declare<T>(chem_env, "0119_bb_oo");
    tmps["0120_bb_oo"]     = declare<T>(chem_env, "0120_bb_oo");
    tmps["0121_bbaa_vvoo"] = declare<T>(chem_env, "0121_bbaa_vvoo");
    tmps["0122_baab_vooo"] = declare<T>(chem_env, "0122_baab_vooo");
    tmps["0123_abab_vooo"] = declare<T>(chem_env, "0123_abab_vooo");
    tmps["0124_abab_vvoo"] = declare<T>(chem_env, "0124_abab_vvoo");
    tmps["0125_abab_vooo"] = declare<T>(chem_env, "0125_abab_vooo");
    tmps["0126_abab_vvoo"] = declare<T>(chem_env, "0126_abab_vvoo");
    tmps["0127_baab_ovoo"] = declare<T>(chem_env, "0127_baab_ovoo");
    tmps["0128_baba_vvoo"] = declare<T>(chem_env, "0128_baba_vvoo");
    tmps["0129_abba_vooo"] = declare<T>(chem_env, "0129_abba_vooo");
    tmps["0130_abab_vvoo"] = declare<T>(chem_env, "0130_abab_vvoo");
    tmps["0131_baab_ovoo"] = declare<T>(chem_env, "0131_baab_ovoo");
    tmps["0132_abab_vvoo"] = declare<T>(chem_env, "0132_abab_vvoo");
    tmps["0133_abab_vvoo"] = declare<T>(chem_env, "0133_abab_vvoo");
    tmps["0134_abab_vvoo"] = declare<T>(chem_env, "0134_abab_vvoo");
    tmps["0135_baab_vooo"] = declare<T>(chem_env, "0135_baab_vooo");
    tmps["0136_abab_vvoo"] = declare<T>(chem_env, "0136_abab_vvoo");
    tmps["0137_baab_vooo"] = declare<T>(chem_env, "0137_baab_vooo");
    tmps["0138_abab_vvoo"] = declare<T>(chem_env, "0138_abab_vvoo");
    tmps["0139_abab_vvoo"] = declare<T>(chem_env, "0139_abab_vvoo");
    tmps["0140_baab_vooo"] = declare<T>(chem_env, "0140_baab_vooo");
    tmps["0141_abab_vvoo"] = declare<T>(chem_env, "0141_abab_vvoo");
    tmps["0142_abab_vvoo"] = declare<T>(chem_env, "0142_abab_vvoo");
    tmps["0143_abab_vvoo"] = declare<T>(chem_env, "0143_abab_vvoo");
    tmps["0144_aabb_vvoo"] = declare<T>(chem_env, "0144_aabb_vvoo");
    tmps["0145_aabb_vooo"] = declare<T>(chem_env, "0145_aabb_vooo");
    tmps["0146_aabb_vooo"] = declare<T>(chem_env, "0146_aabb_vooo");
    tmps["0147_aabb_vvoo"] = declare<T>(chem_env, "0147_aabb_vvoo");
    tmps["0148_aabb_vooo"] = declare<T>(chem_env, "0148_aabb_vooo");
    tmps["0149_aa_oo"]     = declare<T>(chem_env, "0149_aa_oo");
    tmps["0150_aaaa_ooov"] = declare<T>(chem_env, "0150_aaaa_ooov");
    tmps["0151_aa_oo"]     = declare<T>(chem_env, "0151_aa_oo");
    tmps["0152_aaaa_ooov"] = declare<T>(chem_env, "0152_aaaa_ooov");
    tmps["0153_aa_oo"]     = declare<T>(chem_env, "0153_aa_oo");
    tmps["0154_aabb_ovoo"] = declare<T>(chem_env, "0154_aabb_ovoo");
    tmps["0155_abba_ovoo"] = declare<T>(chem_env, "0155_abba_ovoo");
    tmps["0156_abab_oooo"] = declare<T>(chem_env, "0156_abab_oooo");
    tmps["0157_baab_vooo"] = declare<T>(chem_env, "0157_baab_vooo");
    tmps["0158_aabb_oooo"] = declare<T>(chem_env, "0158_aabb_oooo");
    tmps["0159_baab_vooo"] = declare<T>(chem_env, "0159_baab_vooo");
    tmps["0160_bbaa_ooov"] = declare<T>(chem_env, "0160_bbaa_ooov");
    tmps["0161_bbaa_ovoo"] = declare<T>(chem_env, "0161_bbaa_ovoo");
    tmps["0162_abba_ovoo"] = declare<T>(chem_env, "0162_abba_ovoo");
    tmps["0163_aabb_ooov"] = declare<T>(chem_env, "0163_aabb_ooov");
    tmps["0164_aabb_oooo"] = declare<T>(chem_env, "0164_aabb_oooo");
    tmps["0165_baab_vooo"] = declare<T>(chem_env, "0165_baab_vooo");
    tmps["0166_baab_vooo"] = declare<T>(chem_env, "0166_baab_vooo");
    tmps["0167_aabb_oooo"] = declare<T>(chem_env, "0167_aabb_oooo");
    tmps["0168_bbaa_oooo"] = declare<T>(chem_env, "0168_bbaa_oooo");
    tmps["0169_aabb_oooo"] = declare<T>(chem_env, "0169_aabb_oooo");
    tmps["0170_bbaa_oooo"] = declare<T>(chem_env, "0170_bbaa_oooo");
    tmps["0171_aabb_oooo"] = declare<T>(chem_env, "0171_aabb_oooo");
    tmps["0172_bb_oo"]     = declare<T>(chem_env, "0172_bb_oo");
    tmps["0173_aaaa_ooov"] = declare<T>(chem_env, "0173_aaaa_ooov");
    tmps["0174_aa_oo"]     = declare<T>(chem_env, "0174_aa_oo");
    tmps["0175_aa_oo"]     = declare<T>(chem_env, "0175_aa_oo");
    tmps["0176_Q"]         = declare<T>(chem_env, "0176_Q");
    tmps["0177_bb_oo"]     = declare<T>(chem_env, "0177_bb_oo");
    tmps["0178_aa_voQ"]    = declare<T>(chem_env, "0178_aa_voQ");
    tmps["0179_aa_voQ"]    = declare<T>(chem_env, "0179_aa_voQ");
    tmps["0180_aa_voQ"]    = declare<T>(chem_env, "0180_aa_voQ");
    tmps["0181_Q"]         = declare<T>(chem_env, "0181_Q");
    tmps["0182_bb_oo"]     = declare<T>(chem_env, "0182_bb_oo");
    tmps["0183_Q"]         = declare<T>(chem_env, "0183_Q");
    tmps["0184_bb_oo"]     = declare<T>(chem_env, "0184_bb_oo");
    tmps["0185_Q"]         = declare<T>(chem_env, "0185_Q");
    tmps["0186_bb_oo"]     = declare<T>(chem_env, "0186_bb_oo");
    tmps["0187_bb_oo"]     = declare<T>(chem_env, "0187_bb_oo");
    tmps["0188_bb_oo"]     = declare<T>(chem_env, "0188_bb_oo");
    tmps["0189_aa_oo"]     = declare<T>(chem_env, "0189_aa_oo");
    tmps["0190_aa_oo"]     = declare<T>(chem_env, "0190_aa_oo");
    tmps["0191_aa_oo"]     = declare<T>(chem_env, "0191_aa_oo");
    tmps["0192_aa_oo"]     = declare<T>(chem_env, "0192_aa_oo");
    tmps["0193_aa_oo"]     = declare<T>(chem_env, "0193_aa_oo");
    tmps["0194_aa_oo"]     = declare<T>(chem_env, "0194_aa_oo");
    tmps["0195_aa_oo"]     = declare<T>(chem_env, "0195_aa_oo");
    tmps["0196_aa_oo"]     = declare<T>(chem_env, "0196_aa_oo");
    tmps["0197_Q"]         = declare<T>(chem_env, "0197_Q");
    tmps["0198_aa_oo"]     = declare<T>(chem_env, "0198_aa_oo");
    tmps["0199_aa_ooQ"]    = declare<T>(chem_env, "0199_aa_ooQ");
    tmps["0200_aa_oo"]     = declare<T>(chem_env, "0200_aa_oo");
    tmps["0201_aa_oo"]     = declare<T>(chem_env, "0201_aa_oo");
    tmps["0202_Q"]         = declare<T>(chem_env, "0202_Q");
    tmps["0203_aa_oo"]     = declare<T>(chem_env, "0203_aa_oo");
    tmps["0204_bb_oo"]     = declare<T>(chem_env, "0204_bb_oo");
    tmps["0205_bb_vo"]     = declare<T>(chem_env, "0205_bb_vo");
    tmps["0206_bb_oo"]     = declare<T>(chem_env, "0206_bb_oo");
    tmps["0207_bb_vo"]     = declare<T>(chem_env, "0207_bb_vo");
    tmps["0208_aa_oo"]     = declare<T>(chem_env, "0208_aa_oo");
    tmps["0209_aa_vo"]     = declare<T>(chem_env, "0209_aa_vo");
    tmps["0210_aa_oo"]     = declare<T>(chem_env, "0210_aa_oo");
    tmps["0211_aa_vo"]     = declare<T>(chem_env, "0211_aa_vo");
    tmps["0212_aa_vo"]     = declare<T>(chem_env, "0212_aa_vo");
    tmps["0213_aa_oo"]     = declare<T>(chem_env, "0213_aa_oo");
    tmps["0214_aa_vo"]     = declare<T>(chem_env, "0214_aa_vo");
    tmps["0215_aa_vo"]     = declare<T>(chem_env, "0215_aa_vo");
    tmps["0216_aa_vo"]     = declare<T>(chem_env, "0216_aa_vo");
    tmps["0217_aabb_oooo"] = declare<T>(chem_env, "0217_aabb_oooo");
    tmps["0218_baab_vooo"] = declare<T>(chem_env, "0218_baab_vooo");
    tmps["0219_aabb_oooo"] = declare<T>(chem_env, "0219_aabb_oooo");
    tmps["0220_baab_vooo"] = declare<T>(chem_env, "0220_baab_vooo");
    tmps["0221_aabb_oooo"] = declare<T>(chem_env, "0221_aabb_oooo");
    tmps["0222_baab_vooo"] = declare<T>(chem_env, "0222_baab_vooo");
    tmps["0223_baab_vooo"] = declare<T>(chem_env, "0223_baab_vooo");
    tmps["0224_aabb_oooo"] = declare<T>(chem_env, "0224_aabb_oooo");
    tmps["0225_baab_vooo"] = declare<T>(chem_env, "0225_baab_vooo");
    tmps["0226_bb_oo"]     = declare<T>(chem_env, "0226_bb_oo");
    tmps["0227_bb_ov"]     = declare<T>(chem_env, "0227_bb_ov");
    tmps["0228_bb_oo"]     = declare<T>(chem_env, "0228_bb_oo");
    tmps["0229_aa_ov"]     = declare<T>(chem_env, "0229_aa_ov");
    tmps["0230_aa_oo"]     = declare<T>(chem_env, "0230_aa_oo");
    tmps["0231_aa_oo"]     = declare<T>(chem_env, "0231_aa_oo");
    tmps["0232_aa_ov"]     = declare<T>(chem_env, "0232_aa_ov");
    tmps["0233_aa_oo"]     = declare<T>(chem_env, "0233_aa_oo");
    tmps["0234_aa_oo"]     = declare<T>(chem_env, "0234_aa_oo");
    tmps["0235_aa_oo"]     = declare<T>(chem_env, "0235_aa_oo");
    tmps["0236_aabb_oovo"] = declare<T>(chem_env, "0236_aabb_oovo");
    tmps["0237_aabb_oovo"] = declare<T>(chem_env, "0237_aabb_oovo");
    tmps["0238_aabb_oovo"] = declare<T>(chem_env, "0238_aabb_oovo");
    tmps["0239_bb_voQ"]    = declare<T>(chem_env, "0239_bb_voQ");
    tmps["0240_aabb_oovo"] = declare<T>(chem_env, "0240_aabb_oovo");
    tmps["0241_bbaa_oovo"] = declare<T>(chem_env, "0241_bbaa_oovo");
    tmps["0242_bbaa_oovo"] = declare<T>(chem_env, "0242_bbaa_oovo");
    tmps["0243_bbaa_oovo"] = declare<T>(chem_env, "0243_bbaa_oovo");
    tmps["0244_aa_vv"]     = declare<T>(chem_env, "0244_aa_vv");
    tmps["0245_bb_voQ"]    = declare<T>(chem_env, "0245_bb_voQ");
    tmps["0246_bb_voQ"]    = declare<T>(chem_env, "0246_bb_voQ");
    tmps["0247_bb_ooQ"]    = declare<T>(chem_env, "0247_bb_ooQ");
    tmps["0248_aa_voQ"]    = declare<T>(chem_env, "0248_aa_voQ");
    tmps["0249_aa_vv"]     = declare<T>(chem_env, "0249_aa_vv");
    tmps["0250_bb_voQ"]    = declare<T>(chem_env, "0250_bb_voQ");
    tmps["0251_bb_voQ"]    = declare<T>(chem_env, "0251_bb_voQ");
    tmps["0252_aa_voQ"]    = declare<T>(chem_env, "0252_aa_voQ");
    tmps["0253_aa_vv"]     = declare<T>(chem_env, "0253_aa_vv");
    tmps["0254_aa_vv"]     = declare<T>(chem_env, "0254_aa_vv");
    tmps["0255_aa_vv"]     = declare<T>(chem_env, "0255_aa_vv");
  }

  {
    // clang-format off
    sch
      (scalars.at("0001")() = 0.0)(scalars.at("0002")() = 0.0)(scalars.at("0003")() = 0.0)
      (scalars.at("0004")() = 0.0)(scalars.at("0005")() = 0.0)(scalars.at("0006")() = 0.0)
      (scalars.at("0007")() = 0.0)(scalars.at("0008")() = 0.0)(scalars.at("0009")() = 0.0)
      (scalars.at("0010")() = 0.0)(scalars.at("0011")() = 0.0)(scalars.at("0012")() = 0.0)
      (scalars.at("0013")() = 0.0)(scalars.at("0014")() = 0.0)(scalars.at("0015")() = 0.0)
      (scalars.at("0016")() = 0.0)(scalars.at("0017")() = 0.0)(scalars.at("0018")() = 0.0)
      (scalars.at("0019")() = 0.0)(scalars.at("0020")() = 0.0)(scalars.at("0021")() = 0.0)
      (scalars.at("0022")() = 0.0)(scalars.at("0023")() = 0.0)(scalars.at("0024")() = 0.0)
      (scalars.at("0025")() = 0.0)(scalars.at("0026")() = 0.0)(scalars.at("0027")() = 0.0)
      (scalars.at("0028")() = 0.0)(scalars.at("0029")() = 0.0)(scalars.at("0030")() = 0.0)
      (scalars.at("0031")() = 0.0)(scalars.at("0032")() = 0.0)(scalars.at("0033")() = 0.0)
      (scalars.at("0034")() = 0.0)(scalars.at("0035")() = 0.0)(scalars.at("0036")() = 0.0)
      (scalars.at("0037")() = 0.0)(scalars.at("0038")() = 0.0)(scalars.at("0039")() = 0.0)
      (scalars.at("0040")() = 0.0)(scalars.at("0041")() = 0.0)(scalars.at("0042")() = 0.0)
      (scalars.at("0043")() = 0.0)(scalars.at("0044")() = 0.0)(scalars.at("0045")() = 0.0)
      (scalars.at("0046")() = 0.0)(scalars.at("0047")() = 0.0)(scalars.at("0048")() = 0.0)
      (scalars.at("0049")() = 0.0)(scalars.at("0050")() = 0.0)(scalars.at("0051")() = 0.0)
      (scalars.at("0052")() = 0.0)(scalars.at("0053")() = 0.0)(scalars.at("0054")() = 0.0)
      (scalars.at("0055")() = 0.0)(scalars.at("0056")() = 0.0)(scalars.at("0057")() = 0.0)
      (scalars.at("0058")() = 0.0)(scalars.at("0059")() = 0.0)(scalars.at("0060")() = 0.0)
      (scalars.at("0061")() = 0.0)(scalars.at("0062")() = 0.0)(scalars.at("0063")() = 0.0)
      (scalars.at("0064")() = 0.0)(scalars.at("0065")() = 0.0)(scalars.at("0066")() = 0.0)
      (scalars.at("0067")() = 0.0)(scalars.at("0068")() = 0.0)
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