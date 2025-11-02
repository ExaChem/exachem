/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/ccsd_util.hpp"

template<typename T>
void exachem::cc::qed_ccsd_cs::build_tmps(Scheduler& sch, ChemEnv& chem_env, TensorMap<T>& tmps,
                                          TensorMap<T>& scalars, const TensorMap<T>& f,
                                          const TensorMap<T>& eri, const TensorMap<T>& dp,
                                          const double w0, const TensorMap<T>& t1,
                                          const TensorMap<T>& t2, const double t0_1p,
                                          const TensorMap<T>& t1_1p, const TensorMap<T>& t2_1p,
                                          const double t0_2p, const TensorMap<T>& t1_2p,
                                          const TensorMap<T>& t2_2p) {
  TiledIndexSpace&       MO      = chem_env.is_context.MSO;
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
    tmps["bin_aa_oo"]     = declare<T>(chem_env, "bin_aa_oo");
    tmps["bin_aa_vo"]     = declare<T>(chem_env, "bin_aa_vo");
    tmps["bin_aaaa_vooo"] = declare<T>(chem_env, "bin_aaaa_vooo");
    tmps["bin_aabb_oooo"] = declare<T>(chem_env, "bin_aabb_oooo");
    tmps["bin_aabb_vooo"] = declare<T>(chem_env, "bin_aabb_vooo");
    tmps["bin_abab_vvoo"] = declare<T>(chem_env, "bin_abab_vvoo");
    tmps["bin_abba_vvvo"] = declare<T>(chem_env, "bin_abba_vvvo");
    tmps["bin_baab_vooo"] = declare<T>(chem_env, "bin_baab_vooo");
    tmps["bin_bb_oo"]     = declare<T>(chem_env, "bin_bb_oo");
    tmps["bin_bb_vo"]     = declare<T>(chem_env, "bin_bb_vo");
    tmps["bin_bbaa_vvoo"] = declare<T>(chem_env, "bin_bbaa_vvoo");
    tmps["bin_bbbb_vooo"] = declare<T>(chem_env, "bin_bbbb_vooo");
    tmps["bin_bbbb_vvoo"] = declare<T>(chem_env, "bin_bbbb_vvoo");

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
  }

  for(auto& [name, tmp]: tmps) sch.allocate(tmp);
  for(auto& [name, scalar]: scalars) sch.allocate(scalar);

  {
    tmps["0001_abab_vvoo"] = declare<T>(chem_env, "0001_abab_vvoo");
    tmps["0002_abab_vvoo"] = declare<T>(chem_env, "0002_abab_vvoo");
    tmps["0003_abba_vvoo"] = declare<T>(chem_env, "0003_abba_vvoo");
    tmps["0004_abba_vvoo"] = declare<T>(chem_env, "0004_abba_vvoo");
    tmps["0005_abab_vvoo"] = declare<T>(chem_env, "0005_abab_vvoo");
    tmps["0006_abab_vvoo"] = declare<T>(chem_env, "0006_abab_vvoo");
    tmps["0007_abab_vvoo"] = declare<T>(chem_env, "0007_abab_vvoo");
    tmps["0008_abab_vvoo"] = declare<T>(chem_env, "0008_abab_vvoo");
    tmps["0009_abab_vvoo"] = declare<T>(chem_env, "0009_abab_vvoo");
    tmps["0010_abab_vooo"] = declare<T>(chem_env, "0010_abab_vooo");
    tmps["0011_baab_vvoo"] = declare<T>(chem_env, "0011_baab_vvoo");
    tmps["0012_abab_vooo"] = declare<T>(chem_env, "0012_abab_vooo");
    tmps["0013_baab_vvoo"] = declare<T>(chem_env, "0013_baab_vvoo");
    tmps["0014_baab_vooo"] = declare<T>(chem_env, "0014_baab_vooo");
    tmps["0015_abab_vvoo"] = declare<T>(chem_env, "0015_abab_vvoo");
    tmps["0016_baab_vooo"] = declare<T>(chem_env, "0016_baab_vooo");
    tmps["0017_abab_vvoo"] = declare<T>(chem_env, "0017_abab_vvoo");
    tmps["0018_abab_vvoo"] = declare<T>(chem_env, "0018_abab_vvoo");
    tmps["0019_abab_vvoo"] = declare<T>(chem_env, "0019_abab_vvoo");
    tmps["0020_abab_vvoo"] = declare<T>(chem_env, "0020_abab_vvoo");
    tmps["0021_abba_vvoo"] = declare<T>(chem_env, "0021_abba_vvoo");
    tmps["0022_abab_vvoo"] = declare<T>(chem_env, "0022_abab_vvoo");
    tmps["0023_abab_vvoo"] = declare<T>(chem_env, "0023_abab_vvoo");
    tmps["0024_abab_vvoo"] = declare<T>(chem_env, "0024_abab_vvoo");
    tmps["0025_abab_vvoo"] = declare<T>(chem_env, "0025_abab_vvoo");
    tmps["0026_baab_vvoo"] = declare<T>(chem_env, "0026_baab_vvoo");
    tmps["0027_abab_vvoo"] = declare<T>(chem_env, "0027_abab_vvoo");
    tmps["0028_baab_vvoo"] = declare<T>(chem_env, "0028_baab_vvoo");
    tmps["0029_abab_vvoo"] = declare<T>(chem_env, "0029_abab_vvoo");
    tmps["0030_abab_vvoo"] = declare<T>(chem_env, "0030_abab_vvoo");
    tmps["0031_abba_vvoo"] = declare<T>(chem_env, "0031_abba_vvoo");
    tmps["0032_abba_vvoo"] = declare<T>(chem_env, "0032_abba_vvoo");
    tmps["0033_abba_vvoo"] = declare<T>(chem_env, "0033_abba_vvoo");
    tmps["0034_abab_vvoo"] = declare<T>(chem_env, "0034_abab_vvoo");
    tmps["0035_abab_vvoo"] = declare<T>(chem_env, "0035_abab_vvoo");
    tmps["0036_abab_vvoo"] = declare<T>(chem_env, "0036_abab_vvoo");
    tmps["0037_abab_vvoo"] = declare<T>(chem_env, "0037_abab_vvoo");
    tmps["0038_abab_vvoo"] = declare<T>(chem_env, "0038_abab_vvoo");
    tmps["0039_bb_oo"]     = declare<T>(chem_env, "0039_bb_oo");
    tmps["0040_bb_oo"]     = declare<T>(chem_env, "0040_bb_oo");
    tmps["0041_bb_oo"]     = declare<T>(chem_env, "0041_bb_oo");
    tmps["0042_aa_oo"]     = declare<T>(chem_env, "0042_aa_oo");
    tmps["0043_aa_oo"]     = declare<T>(chem_env, "0043_aa_oo");
    tmps["0044_aa_oo"]     = declare<T>(chem_env, "0044_aa_oo");
    tmps["0045_abab_vvoo"] = declare<T>(chem_env, "0045_abab_vvoo");
    tmps["0046_baab_vvoo"] = declare<T>(chem_env, "0046_baab_vvoo");
    tmps["0047_baab_vvoo"] = declare<T>(chem_env, "0047_baab_vvoo");
    tmps["0048_abab_vooo"] = declare<T>(chem_env, "0048_abab_vooo");
    tmps["0049_baab_vvoo"] = declare<T>(chem_env, "0049_baab_vvoo");
    tmps["0050_abab_vvoo"] = declare<T>(chem_env, "0050_abab_vvoo");
    tmps["0051_baab_vooo"] = declare<T>(chem_env, "0051_baab_vooo");
    tmps["0052_abab_vvoo"] = declare<T>(chem_env, "0052_abab_vvoo");
    tmps["0053_abab_vvoo"] = declare<T>(chem_env, "0053_abab_vvoo");
    tmps["0054_abab_vvoo"] = declare<T>(chem_env, "0054_abab_vvoo");
    tmps["0055_bb_vv"]     = declare<T>(chem_env, "0055_bb_vv");
    tmps["0056_abab_vooo"] = declare<T>(chem_env, "0056_abab_vooo");
    tmps["0057_baba_vooo"] = declare<T>(chem_env, "0057_baba_vooo");
    tmps["0058_abab_vvoo"] = declare<T>(chem_env, "0058_abab_vvoo");
    tmps["0059_abab_vvoo"] = declare<T>(chem_env, "0059_abab_vvoo");
    tmps["0060_abab_vvoo"] = declare<T>(chem_env, "0060_abab_vvoo");
    tmps["0061_abab_vvoo"] = declare<T>(chem_env, "0061_abab_vvoo");
    tmps["0062_abba_vvoo"] = declare<T>(chem_env, "0062_abba_vvoo");
    tmps["0063_abab_vvoo"] = declare<T>(chem_env, "0063_abab_vvoo");
    tmps["0064_abba_vvoo"] = declare<T>(chem_env, "0064_abba_vvoo");
    tmps["0065_abab_vvoo"] = declare<T>(chem_env, "0065_abab_vvoo");
    tmps["0066_abab_vvoo"] = declare<T>(chem_env, "0066_abab_vvoo");
    tmps["0067_abab_vvoo"] = declare<T>(chem_env, "0067_abab_vvoo");
    tmps["0068_abab_vvoo"] = declare<T>(chem_env, "0068_abab_vvoo");
    tmps["0069_abab_vvoo"] = declare<T>(chem_env, "0069_abab_vvoo");
    tmps["0070_abab_vvoo"] = declare<T>(chem_env, "0070_abab_vvoo");
    tmps["0071_abab_vvoo"] = declare<T>(chem_env, "0071_abab_vvoo");
    tmps["0072_abab_vvoo"] = declare<T>(chem_env, "0072_abab_vvoo");
    tmps["0073_abab_vvoo"] = declare<T>(chem_env, "0073_abab_vvoo");
    tmps["0074_abab_vvoo"] = declare<T>(chem_env, "0074_abab_vvoo");
    tmps["0075_abab_vvoo"] = declare<T>(chem_env, "0075_abab_vvoo");
    tmps["0076_abab_vvoo"] = declare<T>(chem_env, "0076_abab_vvoo");
    tmps["0077_abab_vvoo"] = declare<T>(chem_env, "0077_abab_vvoo");
    tmps["0078_abab_vvoo"] = declare<T>(chem_env, "0078_abab_vvoo");
    tmps["0079_bbbb_oovo"] = declare<T>(chem_env, "0079_bbbb_oovo");
    tmps["0080_abab_vooo"] = declare<T>(chem_env, "0080_abab_vooo");
    tmps["0081_aaaa_oovo"] = declare<T>(chem_env, "0081_aaaa_oovo");
    tmps["0082_baba_vooo"] = declare<T>(chem_env, "0082_baba_vooo");
    tmps["0083_bb_oo"]     = declare<T>(chem_env, "0083_bb_oo");
    tmps["0084_abab_vvoo"] = declare<T>(chem_env, "0084_abab_vvoo");
    tmps["0085_baab_vovo"] = declare<T>(chem_env, "0085_baab_vovo");
    tmps["0086_abab_vvoo"] = declare<T>(chem_env, "0086_abab_vvoo");
    tmps["0087_aa_oo"]     = declare<T>(chem_env, "0087_aa_oo");
    tmps["0088_abba_vvoo"] = declare<T>(chem_env, "0088_abba_vvoo");
    tmps["0089_abab_ovvo"] = declare<T>(chem_env, "0089_abab_ovvo");
    tmps["0090_abab_vvoo"] = declare<T>(chem_env, "0090_abab_vvoo");
    tmps["0091_abba_voov"] = declare<T>(chem_env, "0091_abba_voov");
    tmps["0092_baab_vvoo"] = declare<T>(chem_env, "0092_baab_vvoo");
    tmps["0093_abba_vvoo"] = declare<T>(chem_env, "0093_abba_vvoo");
    tmps["0094_bb_vv"]     = declare<T>(chem_env, "0094_bb_vv");
    tmps["0095_abab_vvoo"] = declare<T>(chem_env, "0095_abab_vvoo");
    tmps["0096_abab_vovo"] = declare<T>(chem_env, "0096_abab_vovo");
    tmps["0097_baab_vvoo"] = declare<T>(chem_env, "0097_baab_vvoo");
    tmps["0098_aaaa_vovo"] = declare<T>(chem_env, "0098_aaaa_vovo");
    tmps["0099_baba_vvoo"] = declare<T>(chem_env, "0099_baba_vvoo");
    tmps["0100_aa_vv"]     = declare<T>(chem_env, "0100_aa_vv");
    tmps["0101_baab_vvoo"] = declare<T>(chem_env, "0101_baab_vvoo");
    tmps["0102_aa_oo"]     = declare<T>(chem_env, "0102_aa_oo");
    tmps["0103_abba_vvoo"] = declare<T>(chem_env, "0103_abba_vvoo");
    tmps["0104_aa_vv"]     = declare<T>(chem_env, "0104_aa_vv");
    tmps["0105_baab_vvoo"] = declare<T>(chem_env, "0105_baab_vvoo");
    tmps["0106_abba_vovo"] = declare<T>(chem_env, "0106_abba_vovo");
    tmps["0107_baba_vvoo"] = declare<T>(chem_env, "0107_baba_vvoo");
    tmps["0108_abab_oooo"] = declare<T>(chem_env, "0108_abab_oooo");
    tmps["0109_abab_vvoo"] = declare<T>(chem_env, "0109_abab_vvoo");
    tmps["0110_aa_vv"]     = declare<T>(chem_env, "0110_aa_vv");
    tmps["0111_baab_vvoo"] = declare<T>(chem_env, "0111_baab_vvoo");
    tmps["0112_abab_voov"] = declare<T>(chem_env, "0112_abab_voov");
    tmps["0113_baba_vvoo"] = declare<T>(chem_env, "0113_baba_vvoo");
    tmps["0114_bbbb_vovo"] = declare<T>(chem_env, "0114_bbbb_vovo");
    tmps["0115_abab_vvoo"] = declare<T>(chem_env, "0115_abab_vvoo");
    tmps["0116_aa_oo"]     = declare<T>(chem_env, "0116_aa_oo");
    tmps["0117_abba_vvoo"] = declare<T>(chem_env, "0117_abba_vvoo");
    tmps["0118_bb_vv"]     = declare<T>(chem_env, "0118_bb_vv");
    tmps["0119_bb_vv"]     = declare<T>(chem_env, "0119_bb_vv");
    tmps["0120_bb_vv"]     = declare<T>(chem_env, "0120_bb_vv");
    tmps["0121_abab_vvoo"] = declare<T>(chem_env, "0121_abab_vvoo");
    tmps["0122_aa_oo"]     = declare<T>(chem_env, "0122_aa_oo");
    tmps["0123_aa_oo"]     = declare<T>(chem_env, "0123_aa_oo");
    tmps["0124_aa_oo"]     = declare<T>(chem_env, "0124_aa_oo");
    tmps["0125_abba_vvoo"] = declare<T>(chem_env, "0125_abba_vvoo");
    tmps["0126_baba_vovo"] = declare<T>(chem_env, "0126_baba_vovo");
    tmps["0127_abba_vvoo"] = declare<T>(chem_env, "0127_abba_vvoo");
    tmps["0128_abab_oooo"] = declare<T>(chem_env, "0128_abab_oooo");
    tmps["0129_abab_oooo"] = declare<T>(chem_env, "0129_abab_oooo");
    tmps["0130_abab_oooo"] = declare<T>(chem_env, "0130_abab_oooo");
    tmps["0131_abab_vvoo"] = declare<T>(chem_env, "0131_abab_vvoo");
    tmps["0132_aaaa_voov"] = declare<T>(chem_env, "0132_aaaa_voov");
    tmps["0133_baba_vvoo"] = declare<T>(chem_env, "0133_baba_vvoo");
    tmps["0134_abab_ovoo"] = declare<T>(chem_env, "0134_abab_ovoo");
    tmps["0135_abab_ovoo"] = declare<T>(chem_env, "0135_abab_ovoo");
    tmps["0136_abab_ovoo"] = declare<T>(chem_env, "0136_abab_ovoo");
    tmps["0137_abab_vvoo"] = declare<T>(chem_env, "0137_abab_vvoo");
    tmps["0138_bb_vo"]     = declare<T>(chem_env, "0138_bb_vo");
    tmps["0139_bb_vo"]     = declare<T>(chem_env, "0139_bb_vo");
    tmps["0140_bb_vo"]     = declare<T>(chem_env, "0140_bb_vo");
    tmps["0141_bb_vo"]     = declare<T>(chem_env, "0141_bb_vo");
    tmps["0142_bb_vo"]     = declare<T>(chem_env, "0142_bb_vo");
    tmps["0143_abab_vvoo"] = declare<T>(chem_env, "0143_abab_vvoo");
    tmps["0144_abab_vooo"] = declare<T>(chem_env, "0144_abab_vooo");
    tmps["0145_abab_vooo"] = declare<T>(chem_env, "0145_abab_vooo");
    tmps["0146_abab_vooo"] = declare<T>(chem_env, "0146_abab_vooo");
    tmps["0147_abab_vooo"] = declare<T>(chem_env, "0147_abab_vooo");
    tmps["0148_abab_vooo"] = declare<T>(chem_env, "0148_abab_vooo");
    tmps["0149_baab_vvoo"] = declare<T>(chem_env, "0149_baab_vvoo");
    tmps["0150_baba_vooo"] = declare<T>(chem_env, "0150_baba_vooo");
    tmps["0151_abba_ovoo"] = declare<T>(chem_env, "0151_abba_ovoo");
    tmps["0152_baab_vooo"] = declare<T>(chem_env, "0152_baab_vooo");
    tmps["0153_abab_ovoo"] = declare<T>(chem_env, "0153_abab_ovoo");
    tmps["0154_baab_vooo"] = declare<T>(chem_env, "0154_baab_vooo");
    tmps["0155_baab_vooo"] = declare<T>(chem_env, "0155_baab_vooo");
    tmps["0156_abab_vvoo"] = declare<T>(chem_env, "0156_abab_vvoo");
    tmps["0157_baab_ovoo"] = declare<T>(chem_env, "0157_baab_ovoo");
    tmps["0158_baab_ovoo"] = declare<T>(chem_env, "0158_baab_ovoo");
    tmps["0159_baab_ovoo"] = declare<T>(chem_env, "0159_baab_ovoo");
    tmps["0160_baab_vvoo"] = declare<T>(chem_env, "0160_baab_vvoo");
    tmps["0161_aa_vv"]     = declare<T>(chem_env, "0161_aa_vv");
    tmps["0162_baab_vvoo"] = declare<T>(chem_env, "0162_baab_vvoo");
    tmps["0163_aaaa_voov"] = declare<T>(chem_env, "0163_aaaa_voov");
    tmps["0164_baba_vvoo"] = declare<T>(chem_env, "0164_baba_vvoo");
    tmps["0165_baab_vooo"] = declare<T>(chem_env, "0165_baab_vooo");
    tmps["0166_baab_vooo"] = declare<T>(chem_env, "0166_baab_vooo");
    tmps["0167_baab_vooo"] = declare<T>(chem_env, "0167_baab_vooo");
    tmps["0168_baab_vooo"] = declare<T>(chem_env, "0168_baab_vooo");
    tmps["0169_baab_vooo"] = declare<T>(chem_env, "0169_baab_vooo");
    tmps["0170_abab_vvoo"] = declare<T>(chem_env, "0170_abab_vvoo");
    tmps["0171_abab_vvoo"] = declare<T>(chem_env, "0171_abab_vvoo");
    tmps["0172_aa_vo"]     = declare<T>(chem_env, "0172_aa_vo");
    tmps["0173_aa_vo"]     = declare<T>(chem_env, "0173_aa_vo");
    tmps["0174_aa_vo"]     = declare<T>(chem_env, "0174_aa_vo");
    tmps["0175_aa_vo"]     = declare<T>(chem_env, "0175_aa_vo");
    tmps["0176_aa_vo"]     = declare<T>(chem_env, "0176_aa_vo");
    tmps["0177_baba_vvoo"] = declare<T>(chem_env, "0177_baba_vvoo");
    tmps["0178_bb_oo"]     = declare<T>(chem_env, "0178_bb_oo");
    tmps["0179_bb_oo"]     = declare<T>(chem_env, "0179_bb_oo");
    tmps["0180_bb_oo"]     = declare<T>(chem_env, "0180_bb_oo");
    tmps["0181_bb_oo"]     = declare<T>(chem_env, "0181_bb_oo");
    tmps["0182_bb_oo"]     = declare<T>(chem_env, "0182_bb_oo");
    tmps["0183_abab_vvoo"] = declare<T>(chem_env, "0183_abab_vvoo");
    tmps["0184_abba_vvvo"] = declare<T>(chem_env, "0184_abba_vvvo");
    tmps["0185_abba_vvoo"] = declare<T>(chem_env, "0185_abba_vvoo");
    tmps["0186_abba_vooo"] = declare<T>(chem_env, "0186_abba_vooo");
    tmps["0187_abab_vooo"] = declare<T>(chem_env, "0187_abab_vooo");
    tmps["0188_abab_vooo"] = declare<T>(chem_env, "0188_abab_vooo");
    tmps["0189_baab_vvoo"] = declare<T>(chem_env, "0189_baab_vvoo");
    tmps["0190_baab_vvoo"] = declare<T>(chem_env, "0190_baab_vvoo");
    tmps["0191_baab_vvoo"] = declare<T>(chem_env, "0191_baab_vvoo");
    tmps["0192_bb_vo"]     = declare<T>(chem_env, "0192_bb_vo");
    tmps["0193_abab_vvoo"] = declare<T>(chem_env, "0193_abab_vvoo");
    tmps["0194_baab_vvoo"] = declare<T>(chem_env, "0194_baab_vvoo");
    tmps["0195_abab_oovo"] = declare<T>(chem_env, "0195_abab_oovo");
    tmps["0196_abab_oooo"] = declare<T>(chem_env, "0196_abab_oooo");
    tmps["0197_abab_vvoo"] = declare<T>(chem_env, "0197_abab_vvoo");
    tmps["0198_bb_ov"]     = declare<T>(chem_env, "0198_bb_ov");
    tmps["0199_bb_oo"]     = declare<T>(chem_env, "0199_bb_oo");
    tmps["0200_bb_oo"]     = declare<T>(chem_env, "0200_bb_oo");
    tmps["0201_bb_oo"]     = declare<T>(chem_env, "0201_bb_oo");
    tmps["0202_abab_vvoo"] = declare<T>(chem_env, "0202_abab_vvoo");
    tmps["0203_abba_vvoo"] = declare<T>(chem_env, "0203_abba_vvoo");
    tmps["0204_abab_voov"] = declare<T>(chem_env, "0204_abab_voov");
    tmps["0205_abba_vooo"] = declare<T>(chem_env, "0205_abba_vooo");
    tmps["0206_abab_vooo"] = declare<T>(chem_env, "0206_abab_vooo");
    tmps["0207_abab_vooo"] = declare<T>(chem_env, "0207_abab_vooo");
    tmps["0208_baab_vvoo"] = declare<T>(chem_env, "0208_baab_vvoo");
    tmps["0209_abab_vooo"] = declare<T>(chem_env, "0209_abab_vooo");
    tmps["0210_baab_vvoo"] = declare<T>(chem_env, "0210_baab_vvoo");
    tmps["0211_baab_vooo"] = declare<T>(chem_env, "0211_baab_vooo");
    tmps["0212_abab_vvoo"] = declare<T>(chem_env, "0212_abab_vvoo");
    tmps["0213_abba_oovo"] = declare<T>(chem_env, "0213_abba_oovo");
    tmps["0214_aa_ov"]     = declare<T>(chem_env, "0214_aa_ov");
    tmps["0215_aa_oo"]     = declare<T>(chem_env, "0215_aa_oo");
    tmps["0216_aa_oo"]     = declare<T>(chem_env, "0216_aa_oo");
    tmps["0217_aa_oo"]     = declare<T>(chem_env, "0217_aa_oo");
    tmps["0218_abba_vvoo"] = declare<T>(chem_env, "0218_abba_vvoo");
    tmps["0219_abab_ovoo"] = declare<T>(chem_env, "0219_abab_ovoo");
    tmps["0220_abab_ovoo"] = declare<T>(chem_env, "0220_abab_ovoo");
    tmps["0221_baab_vooo"] = declare<T>(chem_env, "0221_baab_vooo");
    tmps["0222_abab_ovoo"] = declare<T>(chem_env, "0222_abab_ovoo");
    tmps["0223_abab_vvoo"] = declare<T>(chem_env, "0223_abab_vvoo");
    tmps["0224_aa_vo"]     = declare<T>(chem_env, "0224_aa_vo");
    tmps["0225_baba_vvoo"] = declare<T>(chem_env, "0225_baba_vvoo");
    tmps["0226_baab_vvoo"] = declare<T>(chem_env, "0226_baab_vvoo");
    tmps["0227_aa_vo"]     = declare<T>(chem_env, "0227_aa_vo");
    tmps["0228_aa_vo"]     = declare<T>(chem_env, "0228_aa_vo");
    tmps["0229_aa_vo"]     = declare<T>(chem_env, "0229_aa_vo");
    tmps["0230_aa_vo"]     = declare<T>(chem_env, "0230_aa_vo");
    tmps["0231_aa_vo"]     = declare<T>(chem_env, "0231_aa_vo");
    tmps["0232_aa_vo"]     = declare<T>(chem_env, "0232_aa_vo");
    tmps["0233_aa_vo"]     = declare<T>(chem_env, "0233_aa_vo");
    tmps["0234_aa_vo"]     = declare<T>(chem_env, "0234_aa_vo");
    tmps["0235_aa_vo"]     = declare<T>(chem_env, "0235_aa_vo");
    tmps["0236_aa_vo"]     = declare<T>(chem_env, "0236_aa_vo");
    tmps["0237_aa_vo"]     = declare<T>(chem_env, "0237_aa_vo");
    tmps["0238_aa_vo"]     = declare<T>(chem_env, "0238_aa_vo");
    tmps["0239_aa_vo"]     = declare<T>(chem_env, "0239_aa_vo");
    tmps["0240_aa_vo"]     = declare<T>(chem_env, "0240_aa_vo");
    tmps["0241_aa_vo"]     = declare<T>(chem_env, "0241_aa_vo");
    tmps["0242_aa_vo"]     = declare<T>(chem_env, "0242_aa_vo");
    tmps["0243_aa_vo"]     = declare<T>(chem_env, "0243_aa_vo");
    tmps["0244_aa_oo"]     = declare<T>(chem_env, "0244_aa_oo");
    tmps["0245_aa_vo"]     = declare<T>(chem_env, "0245_aa_vo");
    tmps["0246_aa_vo"]     = declare<T>(chem_env, "0246_aa_vo");
    tmps["0247_aa_vo"]     = declare<T>(chem_env, "0247_aa_vo");
    tmps["0248_aa_vo"]     = declare<T>(chem_env, "0248_aa_vo");
    tmps["0249_aa_oo"]     = declare<T>(chem_env, "0249_aa_oo");
    tmps["0250_aa_vo"]     = declare<T>(chem_env, "0250_aa_vo");
    tmps["0251_aa_vv"]     = declare<T>(chem_env, "0251_aa_vv");
    tmps["0252_aa_vo"]     = declare<T>(chem_env, "0252_aa_vo");
    tmps["0253_aa_vo"]     = declare<T>(chem_env, "0253_aa_vo");
    tmps["0254_aa_vo"]     = declare<T>(chem_env, "0254_aa_vo");
    tmps["0255_bb_ov"]     = declare<T>(chem_env, "0255_bb_ov");
    tmps["0256_aa_vo"]     = declare<T>(chem_env, "0256_aa_vo");
    tmps["0257_abab_voov"] = declare<T>(chem_env, "0257_abab_voov");
    tmps["0258_aa_vo"]     = declare<T>(chem_env, "0258_aa_vo");
    tmps["0259_aa_vo"]     = declare<T>(chem_env, "0259_aa_vo");
    tmps["0260_aa_vo"]     = declare<T>(chem_env, "0260_aa_vo");
    tmps["0261_aa_vo"]     = declare<T>(chem_env, "0261_aa_vo");
    tmps["0262_abba_vovo"] = declare<T>(chem_env, "0262_abba_vovo");
    tmps["0263_aa_vo"]     = declare<T>(chem_env, "0263_aa_vo");
    tmps["0264_aaaa_voov"] = declare<T>(chem_env, "0264_aaaa_voov");
    tmps["0265_aa_vo"]     = declare<T>(chem_env, "0265_aa_vo");
    tmps["0266_aa_vo"]     = declare<T>(chem_env, "0266_aa_vo");
    tmps["0267_aa_vo"]     = declare<T>(chem_env, "0267_aa_vo");
    tmps["0268_aa_vo"]     = declare<T>(chem_env, "0268_aa_vo");
    tmps["0269_aa_oo"]     = declare<T>(chem_env, "0269_aa_oo");
    tmps["0270_aa_vo"]     = declare<T>(chem_env, "0270_aa_vo");
    tmps["0271_aa_vv"]     = declare<T>(chem_env, "0271_aa_vv");
    tmps["0272_aa_vo"]     = declare<T>(chem_env, "0272_aa_vo");
    tmps["0273_aaaa_voov"] = declare<T>(chem_env, "0273_aaaa_voov");
    tmps["0274_aa_vo"]     = declare<T>(chem_env, "0274_aa_vo");
    tmps["0275_aa_vv"]     = declare<T>(chem_env, "0275_aa_vv");
    tmps["0276_aa_vo"]     = declare<T>(chem_env, "0276_aa_vo");
    tmps["0277_aa_oo"]     = declare<T>(chem_env, "0277_aa_oo");
    tmps["0278_aa_vo"]     = declare<T>(chem_env, "0278_aa_vo");
    tmps["0279_aa_vo"]     = declare<T>(chem_env, "0279_aa_vo");
    tmps["0280_abab_voov"] = declare<T>(chem_env, "0280_abab_voov");
    tmps["0281_aa_vo"]     = declare<T>(chem_env, "0281_aa_vo");
    tmps["0282_aa_oo"]     = declare<T>(chem_env, "0282_aa_oo");
    tmps["0283_aa_vo"]     = declare<T>(chem_env, "0283_aa_vo");
    tmps["0284_aa_ov"]     = declare<T>(chem_env, "0284_aa_ov");
    tmps["0285_aa_vo"]     = declare<T>(chem_env, "0285_aa_vo");
    tmps["0286_aa_vo"]     = declare<T>(chem_env, "0286_aa_vo");
    tmps["0287_abab_voov"] = declare<T>(chem_env, "0287_abab_voov");
    tmps["0288_aa_vo"]     = declare<T>(chem_env, "0288_aa_vo");
    tmps["0289_aa_vo"]     = declare<T>(chem_env, "0289_aa_vo");
    tmps["0290_aa_vo"]     = declare<T>(chem_env, "0290_aa_vo");
    tmps["0291_aa_oo"]     = declare<T>(chem_env, "0291_aa_oo");
    tmps["0292_aa_oo"]     = declare<T>(chem_env, "0292_aa_oo");
    tmps["0293_aa_oo"]     = declare<T>(chem_env, "0293_aa_oo");
    tmps["0294_aa_oo"]     = declare<T>(chem_env, "0294_aa_oo");
    tmps["0295_aa_oo"]     = declare<T>(chem_env, "0295_aa_oo");
    tmps["0296_aa_oo"]     = declare<T>(chem_env, "0296_aa_oo");
    tmps["0297_aa_vo"]     = declare<T>(chem_env, "0297_aa_vo");
    tmps["0298_aa_vo"]     = declare<T>(chem_env, "0298_aa_vo");
    tmps["0299_aa_vo"]     = declare<T>(chem_env, "0299_aa_vo");
    tmps["0300_aa_vo"]     = declare<T>(chem_env, "0300_aa_vo");
    tmps["0301_aa_oo"]     = declare<T>(chem_env, "0301_aa_oo");
    tmps["0302_aa_vo"]     = declare<T>(chem_env, "0302_aa_vo");
    tmps["0303_aa_oo"]     = declare<T>(chem_env, "0303_aa_oo");
    tmps["0304_aa_vo"]     = declare<T>(chem_env, "0304_aa_vo");
    tmps["0305_aa_vo"]     = declare<T>(chem_env, "0305_aa_vo");
    tmps["0306_aa_oo"]     = declare<T>(chem_env, "0306_aa_oo");
    tmps["0307_aa_oo"]     = declare<T>(chem_env, "0307_aa_oo");
    tmps["0308_aa_oo"]     = declare<T>(chem_env, "0308_aa_oo");
    tmps["0309_aa_vo"]     = declare<T>(chem_env, "0309_aa_vo");
    tmps["0310_aaaa_oovo"] = declare<T>(chem_env, "0310_aaaa_oovo");
    tmps["0311_aa_oo"]     = declare<T>(chem_env, "0311_aa_oo");
    tmps["0312_aa_oo"]     = declare<T>(chem_env, "0312_aa_oo");
    tmps["0313_aa_oo"]     = declare<T>(chem_env, "0313_aa_oo");
    tmps["0314_aa_vo"]     = declare<T>(chem_env, "0314_aa_vo");
    tmps["0315_aa_ov"]     = declare<T>(chem_env, "0315_aa_ov");
    tmps["0316_aa_oo"]     = declare<T>(chem_env, "0316_aa_oo");
    tmps["0317_aa_oo"]     = declare<T>(chem_env, "0317_aa_oo");
    tmps["0318_aa_oo"]     = declare<T>(chem_env, "0318_aa_oo");
    tmps["0319_aa_oo"]     = declare<T>(chem_env, "0319_aa_oo");
    tmps["0320_aa_oo"]     = declare<T>(chem_env, "0320_aa_oo");
    tmps["0321_aa_vo"]     = declare<T>(chem_env, "0321_aa_vo");
    tmps["0322_aa_vo"]     = declare<T>(chem_env, "0322_aa_vo");
    tmps["0323_aa_vo"]     = declare<T>(chem_env, "0323_aa_vo");
    tmps["0324_aa_vo"]     = declare<T>(chem_env, "0324_aa_vo");
    tmps["0325_aa_vo"]     = declare<T>(chem_env, "0325_aa_vo");
    tmps["0326_aa_vo"]     = declare<T>(chem_env, "0326_aa_vo");
    tmps["0327_aa_vo"]     = declare<T>(chem_env, "0327_aa_vo");
    tmps["0328_aa_vo"]     = declare<T>(chem_env, "0328_aa_vo");
    tmps["0329_aa_vo"]     = declare<T>(chem_env, "0329_aa_vo");
    tmps["0330_aa_vo"]     = declare<T>(chem_env, "0330_aa_vo");
    tmps["0331_aa_vo"]     = declare<T>(chem_env, "0331_aa_vo");
    tmps["0332_aa_vo"]     = declare<T>(chem_env, "0332_aa_vo");
    tmps["0333_aa_vo"]     = declare<T>(chem_env, "0333_aa_vo");
    tmps["0334_aa_vo"]     = declare<T>(chem_env, "0334_aa_vo");
    tmps["0335_aa_vo"]     = declare<T>(chem_env, "0335_aa_vo");
    tmps["0336_aa_vo"]     = declare<T>(chem_env, "0336_aa_vo");
    tmps["0337_aa_vo"]     = declare<T>(chem_env, "0337_aa_vo");
    tmps["0338_aa_vo"]     = declare<T>(chem_env, "0338_aa_vo");
    tmps["0339_aa_vo"]     = declare<T>(chem_env, "0339_aa_vo");
    tmps["0340_aa_vo"]     = declare<T>(chem_env, "0340_aa_vo");
    tmps["0341_aa_vo"]     = declare<T>(chem_env, "0341_aa_vo");
    tmps["0342_aa_vo"]     = declare<T>(chem_env, "0342_aa_vo");
    tmps["0343_aa_vo"]     = declare<T>(chem_env, "0343_aa_vo");
    tmps["0344_aa_vo"]     = declare<T>(chem_env, "0344_aa_vo");
    tmps["0345_aa_oo"]     = declare<T>(chem_env, "0345_aa_oo");
    tmps["0346_aa_vo"]     = declare<T>(chem_env, "0346_aa_vo");
    tmps["0347_aa_vo"]     = declare<T>(chem_env, "0347_aa_vo");
    tmps["0348_aa_vo"]     = declare<T>(chem_env, "0348_aa_vo");
    tmps["0349_aa_vo"]     = declare<T>(chem_env, "0349_aa_vo");
    tmps["0350_aa_vo"]     = declare<T>(chem_env, "0350_aa_vo");
    tmps["0351_aa_vo"]     = declare<T>(chem_env, "0351_aa_vo");
    tmps["0352_aa_vo"]     = declare<T>(chem_env, "0352_aa_vo");
    tmps["0353_aa_vo"]     = declare<T>(chem_env, "0353_aa_vo");
    tmps["0354_aa_vo"]     = declare<T>(chem_env, "0354_aa_vo");
    tmps["0355_aa_vo"]     = declare<T>(chem_env, "0355_aa_vo");
    tmps["0356_aa_vo"]     = declare<T>(chem_env, "0356_aa_vo");
    tmps["0357_aa_vo"]     = declare<T>(chem_env, "0357_aa_vo");
    tmps["0358_aa_vo"]     = declare<T>(chem_env, "0358_aa_vo");
    tmps["0359_aa_vo"]     = declare<T>(chem_env, "0359_aa_vo");
    tmps["0360_aa_vo"]     = declare<T>(chem_env, "0360_aa_vo");
    tmps["0361_aa_vo"]     = declare<T>(chem_env, "0361_aa_vo");
    tmps["0362_aa_vo"]     = declare<T>(chem_env, "0362_aa_vo");
    tmps["0363_aa_vo"]     = declare<T>(chem_env, "0363_aa_vo");
    tmps["0364_aa_vo"]     = declare<T>(chem_env, "0364_aa_vo");
    tmps["0365_aa_vo"]     = declare<T>(chem_env, "0365_aa_vo");
    tmps["0366_aa_oo"]     = declare<T>(chem_env, "0366_aa_oo");
    tmps["0367_aa_vo"]     = declare<T>(chem_env, "0367_aa_vo");
    tmps["0368_aa_vo"]     = declare<T>(chem_env, "0368_aa_vo");
    tmps["0369_aa_vo"]     = declare<T>(chem_env, "0369_aa_vo");
    tmps["0370_aa_vo"]     = declare<T>(chem_env, "0370_aa_vo");
    tmps["0371_aa_oo"]     = declare<T>(chem_env, "0371_aa_oo");
    tmps["0372_aa_vo"]     = declare<T>(chem_env, "0372_aa_vo");
    tmps["0373_aa_vo"]     = declare<T>(chem_env, "0373_aa_vo");
    tmps["0374_aa_vo"]     = declare<T>(chem_env, "0374_aa_vo");
    tmps["0375_aa_vo"]     = declare<T>(chem_env, "0375_aa_vo");
    tmps["0376_aa_vo"]     = declare<T>(chem_env, "0376_aa_vo");
    tmps["0377_aa_vo"]     = declare<T>(chem_env, "0377_aa_vo");
    tmps["0378_aa_vo"]     = declare<T>(chem_env, "0378_aa_vo");
    tmps["0379_aa_vo"]     = declare<T>(chem_env, "0379_aa_vo");
    tmps["0380_aa_vo"]     = declare<T>(chem_env, "0380_aa_vo");
    tmps["0381_aa_vo"]     = declare<T>(chem_env, "0381_aa_vo");
    tmps["0382_aa_vo"]     = declare<T>(chem_env, "0382_aa_vo");
    tmps["0383_aa_vo"]     = declare<T>(chem_env, "0383_aa_vo");
    tmps["0384_aa_vo"]     = declare<T>(chem_env, "0384_aa_vo");
    tmps["0385_aa_vo"]     = declare<T>(chem_env, "0385_aa_vo");
    tmps["0386_aa_vo"]     = declare<T>(chem_env, "0386_aa_vo");
    tmps["0387_aa_vo"]     = declare<T>(chem_env, "0387_aa_vo");
    tmps["0388_aa_vo"]     = declare<T>(chem_env, "0388_aa_vo");
    tmps["0389_aa_vo"]     = declare<T>(chem_env, "0389_aa_vo");
    tmps["0390_aa_vo"]     = declare<T>(chem_env, "0390_aa_vo");
    tmps["0391_aa_vo"]     = declare<T>(chem_env, "0391_aa_vo");
    tmps["0392_aa_vo"]     = declare<T>(chem_env, "0392_aa_vo");
    tmps["0393_aa_vo"]     = declare<T>(chem_env, "0393_aa_vo");
    tmps["0394_aa_vo"]     = declare<T>(chem_env, "0394_aa_vo");
    tmps["0395_aa_vo"]     = declare<T>(chem_env, "0395_aa_vo");
    tmps["0396_aa_vo"]     = declare<T>(chem_env, "0396_aa_vo");
    tmps["0397_aa_vo"]     = declare<T>(chem_env, "0397_aa_vo");
    tmps["0398_aa_vo"]     = declare<T>(chem_env, "0398_aa_vo");
    tmps["0399_aa_vo"]     = declare<T>(chem_env, "0399_aa_vo");
    tmps["0400_baba_vvoo"] = declare<T>(chem_env, "0400_baba_vvoo");
    tmps["0401_abab_vvoo"] = declare<T>(chem_env, "0401_abab_vvoo");
    tmps["0402_baba_vvoo"] = declare<T>(chem_env, "0402_baba_vvoo");
    tmps["0403_abab_vvoo"] = declare<T>(chem_env, "0403_abab_vvoo");
    tmps["0404_abab_vvoo"] = declare<T>(chem_env, "0404_abab_vvoo");
    tmps["0405_abab_vvoo"] = declare<T>(chem_env, "0405_abab_vvoo");
    tmps["0406_abab_vvoo"] = declare<T>(chem_env, "0406_abab_vvoo");
    tmps["0407_abab_vvoo"] = declare<T>(chem_env, "0407_abab_vvoo");
    tmps["0408_abab_vvoo"] = declare<T>(chem_env, "0408_abab_vvoo");
    tmps["0409_abab_vvoo"] = declare<T>(chem_env, "0409_abab_vvoo");
    tmps["0410_abab_vvoo"] = declare<T>(chem_env, "0410_abab_vvoo");
    tmps["0411_abab_vvoo"] = declare<T>(chem_env, "0411_abab_vvoo");
    tmps["0412_abba_vvoo"] = declare<T>(chem_env, "0412_abba_vvoo");
    tmps["0413_abab_vvoo"] = declare<T>(chem_env, "0413_abab_vvoo");
    tmps["0414_abab_vvoo"] = declare<T>(chem_env, "0414_abab_vvoo");
    tmps["0415_abab_vvoo"] = declare<T>(chem_env, "0415_abab_vvoo");
    tmps["0416_abab_vvoo"] = declare<T>(chem_env, "0416_abab_vvoo");
    tmps["0417_abab_vvoo"] = declare<T>(chem_env, "0417_abab_vvoo");
    tmps["0418_abab_vvoo"] = declare<T>(chem_env, "0418_abab_vvoo");
    tmps["0419_abab_vvoo"] = declare<T>(chem_env, "0419_abab_vvoo");
    tmps["0420_abba_vvoo"] = declare<T>(chem_env, "0420_abba_vvoo");
    tmps["0421_abab_vvoo"] = declare<T>(chem_env, "0421_abab_vvoo");
    tmps["0422_abab_vvoo"] = declare<T>(chem_env, "0422_abab_vvoo");
    tmps["0423_abab_vvoo"] = declare<T>(chem_env, "0423_abab_vvoo");
    tmps["0424_baba_vvoo"] = declare<T>(chem_env, "0424_baba_vvoo");
    tmps["0425_baab_vvoo"] = declare<T>(chem_env, "0425_baab_vvoo");
    tmps["0426_abab_vvoo"] = declare<T>(chem_env, "0426_abab_vvoo");
    tmps["0427_baba_vvoo"] = declare<T>(chem_env, "0427_baba_vvoo");
    tmps["0428_abab_oooo"] = declare<T>(chem_env, "0428_abab_oooo");
    tmps["0429_abab_oooo"] = declare<T>(chem_env, "0429_abab_oooo");
    tmps["0430_abab_oooo"] = declare<T>(chem_env, "0430_abab_oooo");
    tmps["0431_abab_oooo"] = declare<T>(chem_env, "0431_abab_oooo");
    tmps["0432_aa_vo"]     = declare<T>(chem_env, "0432_aa_vo");
    tmps["0433_aa_vo"]     = declare<T>(chem_env, "0433_aa_vo");
    tmps["0434_aa_vo"]     = declare<T>(chem_env, "0434_aa_vo");
    tmps["0435_aa_vo"]     = declare<T>(chem_env, "0435_aa_vo");
    tmps["0436_aa_vo"]     = declare<T>(chem_env, "0436_aa_vo");
    tmps["0437_bb_oo"]     = declare<T>(chem_env, "0437_bb_oo");
    tmps["0438_bb_oo"]     = declare<T>(chem_env, "0438_bb_oo");
    tmps["0439_bb_oo"]     = declare<T>(chem_env, "0439_bb_oo");
    tmps["0440_bb_oo"]     = declare<T>(chem_env, "0440_bb_oo");
    tmps["0441_bb_oo"]     = declare<T>(chem_env, "0441_bb_oo");
    tmps["0442_bb_oo"]     = declare<T>(chem_env, "0442_bb_oo");
    tmps["0443_bb_oo"]     = declare<T>(chem_env, "0443_bb_oo");
    tmps["0444_bb_oo"]     = declare<T>(chem_env, "0444_bb_oo");
    tmps["0445_bb_oo"]     = declare<T>(chem_env, "0445_bb_oo");
    tmps["0446_aa_ov"]     = declare<T>(chem_env, "0446_aa_ov");
    tmps["0447_aa_ov"]     = declare<T>(chem_env, "0447_aa_ov");
    tmps["0448_aa_ov"]     = declare<T>(chem_env, "0448_aa_ov");
    tmps["0449_bb_vo"]     = declare<T>(chem_env, "0449_bb_vo");
    tmps["0450_bb_vo"]     = declare<T>(chem_env, "0450_bb_vo");
    tmps["0451_bb_vo"]     = declare<T>(chem_env, "0451_bb_vo");
    tmps["0452_bb_vo"]     = declare<T>(chem_env, "0452_bb_vo");
    tmps["0453_bb_vo"]     = declare<T>(chem_env, "0453_bb_vo");
    tmps["0454_bb_ov"]     = declare<T>(chem_env, "0454_bb_ov");
    tmps["0455_bb_ov"]     = declare<T>(chem_env, "0455_bb_ov");
    tmps["0456_bb_ov"]     = declare<T>(chem_env, "0456_bb_ov");
    tmps["0457_aa_vv"]     = declare<T>(chem_env, "0457_aa_vv");
    tmps["0458_aa_vv"]     = declare<T>(chem_env, "0458_aa_vv");
    tmps["0459_aa_vv"]     = declare<T>(chem_env, "0459_aa_vv");
    tmps["0460_aa_vv"]     = declare<T>(chem_env, "0460_aa_vv");
    tmps["0461_aa_vv"]     = declare<T>(chem_env, "0461_aa_vv");
    tmps["0462_bb_vv"]     = declare<T>(chem_env, "0462_bb_vv");
    tmps["0463_bb_vv"]     = declare<T>(chem_env, "0463_bb_vv");
    tmps["0464_bb_vv"]     = declare<T>(chem_env, "0464_bb_vv");
    tmps["0465_bb_vv"]     = declare<T>(chem_env, "0465_bb_vv");
    tmps["0466_bb_vv"]     = declare<T>(chem_env, "0466_bb_vv");
    tmps["0467_abab_vooo"] = declare<T>(chem_env, "0467_abab_vooo");
    tmps["0468_abab_vooo"] = declare<T>(chem_env, "0468_abab_vooo");
    tmps["0469_abab_vooo"] = declare<T>(chem_env, "0469_abab_vooo");
    tmps["0470_abab_vooo"] = declare<T>(chem_env, "0470_abab_vooo");
    tmps["0471_abab_vooo"] = declare<T>(chem_env, "0471_abab_vooo");
    tmps["0472_abba_vooo"] = declare<T>(chem_env, "0472_abba_vooo");
    tmps["0473_abab_vooo"] = declare<T>(chem_env, "0473_abab_vooo");
    tmps["0474_baba_ovoo"] = declare<T>(chem_env, "0474_baba_ovoo");
    tmps["0475_abab_vooo"] = declare<T>(chem_env, "0475_abab_vooo");
    tmps["0476_baab_vooo"] = declare<T>(chem_env, "0476_baab_vooo");
    tmps["0477_baba_vooo"] = declare<T>(chem_env, "0477_baba_vooo");
    tmps["0478_baab_vooo"] = declare<T>(chem_env, "0478_baab_vooo");
    tmps["0479_baab_vooo"] = declare<T>(chem_env, "0479_baab_vooo");
    tmps["0480_baab_vooo"] = declare<T>(chem_env, "0480_baab_vooo");
    tmps["0481_baab_vooo"] = declare<T>(chem_env, "0481_baab_vooo");
    tmps["0482_abab_ovoo"] = declare<T>(chem_env, "0482_abab_ovoo");
    tmps["0483_baba_vooo"] = declare<T>(chem_env, "0483_baba_vooo");
    tmps["0484_baab_vooo"] = declare<T>(chem_env, "0484_baab_vooo");
    tmps["0485_abab_ovoo"] = declare<T>(chem_env, "0485_abab_ovoo");
    tmps["0486_aaaa_voov"] = declare<T>(chem_env, "0486_aaaa_voov");
    tmps["0487_aaaa_vovo"] = declare<T>(chem_env, "0487_aaaa_vovo");
    tmps["0488_aaaa_voov"] = declare<T>(chem_env, "0488_aaaa_voov");
    tmps["0489_aaaa_voov"] = declare<T>(chem_env, "0489_aaaa_voov");
    tmps["0490_abba_oooo"] = declare<T>(chem_env, "0490_abba_oooo");
    tmps["0491_abab_oovo"] = declare<T>(chem_env, "0491_abab_oovo");
    tmps["0492_abab_oooo"] = declare<T>(chem_env, "0492_abab_oooo");
    tmps["0493_abab_oooo"] = declare<T>(chem_env, "0493_abab_oooo");
    tmps["0494_aa_vo"]     = declare<T>(chem_env, "0494_aa_vo");
    tmps["0495_aa_vo"]     = declare<T>(chem_env, "0495_aa_vo");
    tmps["0496_aa_vo"]     = declare<T>(chem_env, "0496_aa_vo");
    tmps["0497_bb_oo"]     = declare<T>(chem_env, "0497_bb_oo");
    tmps["0498_bb_oo"]     = declare<T>(chem_env, "0498_bb_oo");
    tmps["0499_bb_oo"]     = declare<T>(chem_env, "0499_bb_oo");
    tmps["0500_bb_oo"]     = declare<T>(chem_env, "0500_bb_oo");
    tmps["0501_bb_oo"]     = declare<T>(chem_env, "0501_bb_oo");
    tmps["0502_bb_oo"]     = declare<T>(chem_env, "0502_bb_oo");
    tmps["0503_bb_ov"]     = declare<T>(chem_env, "0503_bb_ov");
    tmps["0504_bb_oo"]     = declare<T>(chem_env, "0504_bb_oo");
    tmps["0505_bb_vo"]     = declare<T>(chem_env, "0505_bb_vo");
    tmps["0506_bb_vo"]     = declare<T>(chem_env, "0506_bb_vo");
    tmps["0507_bb_vo"]     = declare<T>(chem_env, "0507_bb_vo");
    tmps["0508_abba_vooo"] = declare<T>(chem_env, "0508_abba_vooo");
    tmps["0509_bb_ov"]     = declare<T>(chem_env, "0509_bb_ov");
    tmps["0510_abab_vooo"] = declare<T>(chem_env, "0510_abab_vooo");
    tmps["0511_abba_vooo"] = declare<T>(chem_env, "0511_abba_vooo");
    tmps["0512_bb_ov"]     = declare<T>(chem_env, "0512_bb_ov");
    tmps["0513_abab_vooo"] = declare<T>(chem_env, "0513_abab_vooo");
    tmps["0514_abba_vooo"] = declare<T>(chem_env, "0514_abba_vooo");
    tmps["0515_abba_vooo"] = declare<T>(chem_env, "0515_abba_vooo");
    tmps["0516_abba_voov"] = declare<T>(chem_env, "0516_abba_voov");
    tmps["0517_abab_vooo"] = declare<T>(chem_env, "0517_abab_vooo");
    tmps["0518_abba_vooo"] = declare<T>(chem_env, "0518_abba_vooo");
    tmps["0519_abab_vovo"] = declare<T>(chem_env, "0519_abab_vovo");
    tmps["0520_abab_vooo"] = declare<T>(chem_env, "0520_abab_vooo");
    tmps["0521_abab_vooo"] = declare<T>(chem_env, "0521_abab_vooo");
    tmps["0522_abab_vooo"] = declare<T>(chem_env, "0522_abab_vooo");
    tmps["0523_abab_vooo"] = declare<T>(chem_env, "0523_abab_vooo");
    tmps["0524_abba_vooo"] = declare<T>(chem_env, "0524_abba_vooo");
    tmps["0525_abab_vooo"] = declare<T>(chem_env, "0525_abab_vooo");
    tmps["0526_abab_vooo"] = declare<T>(chem_env, "0526_abab_vooo");
    tmps["0527_abba_voov"] = declare<T>(chem_env, "0527_abba_voov");
    tmps["0528_abab_vooo"] = declare<T>(chem_env, "0528_abab_vooo");
    tmps["0529_abab_vooo"] = declare<T>(chem_env, "0529_abab_vooo");
    tmps["0530_abab_vooo"] = declare<T>(chem_env, "0530_abab_vooo");
    tmps["0531_abab_oooo"] = declare<T>(chem_env, "0531_abab_oooo");
    tmps["0532_baab_vooo"] = declare<T>(chem_env, "0532_baab_vooo");
    tmps["0533_aaaa_oovo"] = declare<T>(chem_env, "0533_aaaa_oovo");
    tmps["0534_baba_vooo"] = declare<T>(chem_env, "0534_baba_vooo");
    tmps["0535_baab_vooo"] = declare<T>(chem_env, "0535_baab_vooo");
    tmps["0536_aaaa_oovo"] = declare<T>(chem_env, "0536_aaaa_oovo");
    tmps["0537_baba_vooo"] = declare<T>(chem_env, "0537_baba_vooo");
    tmps["0538_baba_vooo"] = declare<T>(chem_env, "0538_baba_vooo");
    tmps["0539_baab_vooo"] = declare<T>(chem_env, "0539_baab_vooo");
    tmps["0540_baab_vooo"] = declare<T>(chem_env, "0540_baab_vooo");
    tmps["0541_baab_vovo"] = declare<T>(chem_env, "0541_baab_vovo");
    tmps["0542_baab_vooo"] = declare<T>(chem_env, "0542_baab_vooo");
    tmps["0543_baab_vooo"] = declare<T>(chem_env, "0543_baab_vooo");
    tmps["0544_abab_ovoo"] = declare<T>(chem_env, "0544_abab_ovoo");
    tmps["0545_aa_ov"]     = declare<T>(chem_env, "0545_aa_ov");
    tmps["0546_baab_vooo"] = declare<T>(chem_env, "0546_baab_vooo");
    tmps["0547_abab_oooo"] = declare<T>(chem_env, "0547_abab_oooo");
    tmps["0548_abab_oooo"] = declare<T>(chem_env, "0548_abab_oooo");
    tmps["0549_abab_oooo"] = declare<T>(chem_env, "0549_abab_oooo");
    tmps["0550_baab_vooo"] = declare<T>(chem_env, "0550_baab_vooo");
    tmps["0551_aa_ov"]     = declare<T>(chem_env, "0551_aa_ov");
    tmps["0552_baab_vooo"] = declare<T>(chem_env, "0552_baab_vooo");
    tmps["0553_baab_vooo"] = declare<T>(chem_env, "0553_baab_vooo");
    tmps["0554_baba_vooo"] = declare<T>(chem_env, "0554_baba_vooo");
    tmps["0555_baab_vooo"] = declare<T>(chem_env, "0555_baab_vooo");
    tmps["0556_baab_vooo"] = declare<T>(chem_env, "0556_baab_vooo");
    tmps["0557_baba_vooo"] = declare<T>(chem_env, "0557_baba_vooo");
    tmps["0558_baab_vooo"] = declare<T>(chem_env, "0558_baab_vooo");
    tmps["0559_baab_vooo"] = declare<T>(chem_env, "0559_baab_vooo");
    tmps["0560_baab_vooo"] = declare<T>(chem_env, "0560_baab_vooo");
    tmps["0561_abab_oovo"] = declare<T>(chem_env, "0561_abab_oovo");
    tmps["0562_baab_vooo"] = declare<T>(chem_env, "0562_baab_vooo");
    tmps["0563_baba_vooo"] = declare<T>(chem_env, "0563_baba_vooo");
    tmps["0564_bb_vv"]     = declare<T>(chem_env, "0564_bb_vv");
    tmps["0565_abab_vvoo"] = declare<T>(chem_env, "0565_abab_vvoo");
    tmps["0566_abba_vvoo"] = declare<T>(chem_env, "0566_abba_vvoo");
    tmps["0567_abab_vvoo"] = declare<T>(chem_env, "0567_abab_vvoo");
    tmps["0568_baba_vvoo"] = declare<T>(chem_env, "0568_baba_vvoo");
    tmps["0569_bb_vv"]     = declare<T>(chem_env, "0569_bb_vv");
    tmps["0570_abab_vvoo"] = declare<T>(chem_env, "0570_abab_vvoo");
    tmps["0571_abab_vvoo"] = declare<T>(chem_env, "0571_abab_vvoo");
    tmps["0572_baab_vvoo"] = declare<T>(chem_env, "0572_baab_vvoo");
    tmps["0573_abab_vovo"] = declare<T>(chem_env, "0573_abab_vovo");
    tmps["0574_baab_vvoo"] = declare<T>(chem_env, "0574_baab_vvoo");
    tmps["0575_baba_vvoo"] = declare<T>(chem_env, "0575_baba_vvoo");
    tmps["0576_baab_vvoo"] = declare<T>(chem_env, "0576_baab_vvoo");
    tmps["0577_bb_oo"]     = declare<T>(chem_env, "0577_bb_oo");
    tmps["0578_bb_oo"]     = declare<T>(chem_env, "0578_bb_oo");
    tmps["0579_bb_oo"]     = declare<T>(chem_env, "0579_bb_oo");
    tmps["0580_abab_vvoo"] = declare<T>(chem_env, "0580_abab_vvoo");
    tmps["0581_baba_vvoo"] = declare<T>(chem_env, "0581_baba_vvoo");
    tmps["0582_baab_vvoo"] = declare<T>(chem_env, "0582_baab_vvoo");
    tmps["0583_abab_vvoo"] = declare<T>(chem_env, "0583_abab_vvoo");
    tmps["0584_baab_vvoo"] = declare<T>(chem_env, "0584_baab_vvoo");
    tmps["0585_abba_vvoo"] = declare<T>(chem_env, "0585_abba_vvoo");
    tmps["0586_abba_vvoo"] = declare<T>(chem_env, "0586_abba_vvoo");
    tmps["0587_abba_vvvo"] = declare<T>(chem_env, "0587_abba_vvvo");
    tmps["0588_abba_vvoo"] = declare<T>(chem_env, "0588_abba_vvoo");
    tmps["0589_baba_vvoo"] = declare<T>(chem_env, "0589_baba_vvoo");
    tmps["0590_abab_vvoo"] = declare<T>(chem_env, "0590_abab_vvoo");
    tmps["0591_abab_vvoo"] = declare<T>(chem_env, "0591_abab_vvoo");
    tmps["0592_baab_vvoo"] = declare<T>(chem_env, "0592_baab_vvoo");
    tmps["0593_abba_vvoo"] = declare<T>(chem_env, "0593_abba_vvoo");
    tmps["0594_baba_vvoo"] = declare<T>(chem_env, "0594_baba_vvoo");
    tmps["0595_bb_vv"]     = declare<T>(chem_env, "0595_bb_vv");
    tmps["0596_bb_vv"]     = declare<T>(chem_env, "0596_bb_vv");
    tmps["0597_bb_vv"]     = declare<T>(chem_env, "0597_bb_vv");
    tmps["0598_abab_vvoo"] = declare<T>(chem_env, "0598_abab_vvoo");
    tmps["0599_baba_vvoo"] = declare<T>(chem_env, "0599_baba_vvoo");
    tmps["0600_baab_ovoo"] = declare<T>(chem_env, "0600_baab_ovoo");
    tmps["0601_baab_vvoo"] = declare<T>(chem_env, "0601_baab_vvoo");
    tmps["0602_aaaa_vovo"] = declare<T>(chem_env, "0602_aaaa_vovo");
    tmps["0603_baba_vvoo"] = declare<T>(chem_env, "0603_baba_vvoo");
    tmps["0604_abba_vvoo"] = declare<T>(chem_env, "0604_abba_vvoo");
    tmps["0605_baab_vvoo"] = declare<T>(chem_env, "0605_baab_vvoo");
    tmps["0606_baba_vvoo"] = declare<T>(chem_env, "0606_baba_vvoo");
    tmps["0607_baab_vvoo"] = declare<T>(chem_env, "0607_baab_vvoo");
    tmps["0608_baba_vvoo"] = declare<T>(chem_env, "0608_baba_vvoo");
    tmps["0609_bb_vo"]     = declare<T>(chem_env, "0609_bb_vo");
    tmps["0610_bb_vo"]     = declare<T>(chem_env, "0610_bb_vo");
    tmps["0611_bb_vo"]     = declare<T>(chem_env, "0611_bb_vo");
    tmps["0612_bb_vo"]     = declare<T>(chem_env, "0612_bb_vo");
    tmps["0613_bb_vo"]     = declare<T>(chem_env, "0613_bb_vo");
    tmps["0614_abab_vvoo"] = declare<T>(chem_env, "0614_abab_vvoo");
    tmps["0615_abba_vvoo"] = declare<T>(chem_env, "0615_abba_vvoo");
    tmps["0616_abab_vvoo"] = declare<T>(chem_env, "0616_abab_vvoo");
    tmps["0617_baba_vvoo"] = declare<T>(chem_env, "0617_baba_vvoo");
    tmps["0618_abab_vvoo"] = declare<T>(chem_env, "0618_abab_vvoo");
    tmps["0619_abab_vvoo"] = declare<T>(chem_env, "0619_abab_vvoo");
    tmps["0620_bb_vv"]     = declare<T>(chem_env, "0620_bb_vv");
    tmps["0621_abab_vvoo"] = declare<T>(chem_env, "0621_abab_vvoo");
    tmps["0622_abab_vvoo"] = declare<T>(chem_env, "0622_abab_vvoo");
    tmps["0623_baba_ovoo"] = declare<T>(chem_env, "0623_baba_ovoo");
    tmps["0624_abab_vooo"] = declare<T>(chem_env, "0624_abab_vooo");
    tmps["0625_baba_ovoo"] = declare<T>(chem_env, "0625_baba_ovoo");
    tmps["0626_baba_vvoo"] = declare<T>(chem_env, "0626_baba_vvoo");
    tmps["0627_aa_vo"]     = declare<T>(chem_env, "0627_aa_vo");
    tmps["0628_aa_vo"]     = declare<T>(chem_env, "0628_aa_vo");
    tmps["0629_aa_vo"]     = declare<T>(chem_env, "0629_aa_vo");
    tmps["0630_aa_vo"]     = declare<T>(chem_env, "0630_aa_vo");
    tmps["0631_aa_vo"]     = declare<T>(chem_env, "0631_aa_vo");
    tmps["0632_baba_vvoo"] = declare<T>(chem_env, "0632_baba_vvoo");
    tmps["0633_baab_vvoo"] = declare<T>(chem_env, "0633_baab_vvoo");
    tmps["0634_abab_vvoo"] = declare<T>(chem_env, "0634_abab_vvoo");
    tmps["0635_abab_ovoo"] = declare<T>(chem_env, "0635_abab_ovoo");
    tmps["0636_baba_vooo"] = declare<T>(chem_env, "0636_baba_vooo");
    tmps["0637_abab_ovoo"] = declare<T>(chem_env, "0637_abab_ovoo");
    tmps["0638_abab_vvoo"] = declare<T>(chem_env, "0638_abab_vvoo");
    tmps["0639_baab_vvoo"] = declare<T>(chem_env, "0639_baab_vvoo");
    tmps["0640_bb_oo"]     = declare<T>(chem_env, "0640_bb_oo");
    tmps["0641_abab_vvoo"] = declare<T>(chem_env, "0641_abab_vvoo");
    tmps["0642_abba_ovoo"] = declare<T>(chem_env, "0642_abba_ovoo");
    tmps["0643_abab_ovoo"] = declare<T>(chem_env, "0643_abab_ovoo");
    tmps["0644_abab_ovoo"] = declare<T>(chem_env, "0644_abab_ovoo");
    tmps["0645_abab_ovoo"] = declare<T>(chem_env, "0645_abab_ovoo");
    tmps["0646_abab_vvoo"] = declare<T>(chem_env, "0646_abab_vvoo");
    tmps["0647_baab_vvoo"] = declare<T>(chem_env, "0647_baab_vvoo");
    tmps["0648_abab_ovoo"] = declare<T>(chem_env, "0648_abab_ovoo");
    tmps["0649_abab_ovoo"] = declare<T>(chem_env, "0649_abab_ovoo");
    tmps["0650_abab_ovoo"] = declare<T>(chem_env, "0650_abab_ovoo");
    tmps["0651_abab_vvoo"] = declare<T>(chem_env, "0651_abab_vvoo");
    tmps["0652_aa_vv"]     = declare<T>(chem_env, "0652_aa_vv");
    tmps["0653_baab_vvoo"] = declare<T>(chem_env, "0653_baab_vvoo");
    tmps["0654_abab_vvoo"] = declare<T>(chem_env, "0654_abab_vvoo");
    tmps["0655_abab_vvoo"] = declare<T>(chem_env, "0655_abab_vvoo");
    tmps["0656_abab_vvoo"] = declare<T>(chem_env, "0656_abab_vvoo");
    tmps["0657_abab_vvoo"] = declare<T>(chem_env, "0657_abab_vvoo");
    tmps["0658_aa_vv"]     = declare<T>(chem_env, "0658_aa_vv");
    tmps["0659_baab_vvoo"] = declare<T>(chem_env, "0659_baab_vvoo");
    tmps["0660_baab_vvoo"] = declare<T>(chem_env, "0660_baab_vvoo");
    tmps["0661_baab_vvoo"] = declare<T>(chem_env, "0661_baab_vvoo");
    tmps["0662_baba_vvoo"] = declare<T>(chem_env, "0662_baba_vvoo");
    tmps["0663_abba_vvoo"] = declare<T>(chem_env, "0663_abba_vvoo");
    tmps["0664_baab_vvoo"] = declare<T>(chem_env, "0664_baab_vvoo");
    tmps["0665_abba_vvoo"] = declare<T>(chem_env, "0665_abba_vvoo");
    tmps["0666_baab_vvoo"] = declare<T>(chem_env, "0666_baab_vvoo");
    tmps["0667_abab_vvoo"] = declare<T>(chem_env, "0667_abab_vvoo");
    tmps["0668_baab_vvoo"] = declare<T>(chem_env, "0668_baab_vvoo");
    tmps["0669_abab_vvoo"] = declare<T>(chem_env, "0669_abab_vvoo");
    tmps["0670_abab_vvoo"] = declare<T>(chem_env, "0670_abab_vvoo");
    tmps["0671_baba_vovo"] = declare<T>(chem_env, "0671_baba_vovo");
    tmps["0672_abba_vvoo"] = declare<T>(chem_env, "0672_abba_vvoo");
    tmps["0673_abba_vvoo"] = declare<T>(chem_env, "0673_abba_vvoo");
    tmps["0674_abab_vvoo"] = declare<T>(chem_env, "0674_abab_vvoo");
    tmps["0675_abab_vvoo"] = declare<T>(chem_env, "0675_abab_vvoo");
    tmps["0676_abba_vvoo"] = declare<T>(chem_env, "0676_abba_vvoo");
    tmps["0677_aa_oo"]     = declare<T>(chem_env, "0677_aa_oo");
    tmps["0678_abba_vvoo"] = declare<T>(chem_env, "0678_abba_vvoo");
    tmps["0679_abab_vvoo"] = declare<T>(chem_env, "0679_abab_vvoo");
    tmps["0680_abab_vvoo"] = declare<T>(chem_env, "0680_abab_vvoo");
    tmps["0681_baab_vvoo"] = declare<T>(chem_env, "0681_baab_vvoo");
    tmps["0682_aaaa_voov"] = declare<T>(chem_env, "0682_aaaa_voov");
    tmps["0683_baba_vvoo"] = declare<T>(chem_env, "0683_baba_vvoo");
    tmps["0684_abba_vvoo"] = declare<T>(chem_env, "0684_abba_vvoo");
    tmps["0685_abab_vooo"] = declare<T>(chem_env, "0685_abab_vooo");
    tmps["0686_abab_vooo"] = declare<T>(chem_env, "0686_abab_vooo");
    tmps["0687_abba_vooo"] = declare<T>(chem_env, "0687_abba_vooo");
    tmps["0688_abab_vooo"] = declare<T>(chem_env, "0688_abab_vooo");
    tmps["0689_abab_vooo"] = declare<T>(chem_env, "0689_abab_vooo");
    tmps["0690_abab_vooo"] = declare<T>(chem_env, "0690_abab_vooo");
    tmps["0691_abab_vooo"] = declare<T>(chem_env, "0691_abab_vooo");
    tmps["0692_abab_vooo"] = declare<T>(chem_env, "0692_abab_vooo");
    tmps["0693_baab_vvoo"] = declare<T>(chem_env, "0693_baab_vvoo");
    tmps["0694_abab_vvoo"] = declare<T>(chem_env, "0694_abab_vvoo");
    tmps["0695_abab_vvoo"] = declare<T>(chem_env, "0695_abab_vvoo");
    tmps["0696_baab_vooo"] = declare<T>(chem_env, "0696_baab_vooo");
    tmps["0697_abab_ovoo"] = declare<T>(chem_env, "0697_abab_ovoo");
    tmps["0698_baba_vooo"] = declare<T>(chem_env, "0698_baba_vooo");
    tmps["0699_abba_ovoo"] = declare<T>(chem_env, "0699_abba_ovoo");
    tmps["0700_baab_vooo"] = declare<T>(chem_env, "0700_baab_vooo");
    tmps["0701_abab_ovoo"] = declare<T>(chem_env, "0701_abab_ovoo");
    tmps["0702_baba_vooo"] = declare<T>(chem_env, "0702_baba_vooo");
    tmps["0703_baab_vooo"] = declare<T>(chem_env, "0703_baab_vooo");
    tmps["0704_baab_vooo"] = declare<T>(chem_env, "0704_baab_vooo");
    tmps["0705_baab_vooo"] = declare<T>(chem_env, "0705_baab_vooo");
    tmps["0706_abab_ovoo"] = declare<T>(chem_env, "0706_abab_ovoo");
    tmps["0707_baab_vooo"] = declare<T>(chem_env, "0707_baab_vooo");
    tmps["0708_baab_vooo"] = declare<T>(chem_env, "0708_baab_vooo");
    tmps["0709_abab_vvoo"] = declare<T>(chem_env, "0709_abab_vvoo");
    tmps["0710_abab_vvoo"] = declare<T>(chem_env, "0710_abab_vvoo");
    tmps["0711_baba_vvoo"] = declare<T>(chem_env, "0711_baba_vvoo");
    tmps["0712_baab_ovoo"] = declare<T>(chem_env, "0712_baab_ovoo");
    tmps["0713_baab_ovoo"] = declare<T>(chem_env, "0713_baab_ovoo");
    tmps["0714_baab_ovoo"] = declare<T>(chem_env, "0714_baab_ovoo");
    tmps["0715_baab_vvoo"] = declare<T>(chem_env, "0715_baab_vvoo");
    tmps["0716_abba_vvoo"] = declare<T>(chem_env, "0716_abba_vvoo");
    tmps["0717_baba_vvoo"] = declare<T>(chem_env, "0717_baba_vvoo");
    tmps["0718_abba_vvoo"] = declare<T>(chem_env, "0718_abba_vvoo");
    tmps["0719_baab_vvoo"] = declare<T>(chem_env, "0719_baab_vvoo");
    tmps["0720_baab_vvoo"] = declare<T>(chem_env, "0720_baab_vvoo");
    tmps["0721_abab_vvoo"] = declare<T>(chem_env, "0721_abab_vvoo");
    tmps["0722_abab_oooo"] = declare<T>(chem_env, "0722_abab_oooo");
    tmps["0723_aa_vo"]     = declare<T>(chem_env, "0723_aa_vo");
    tmps["0724_bb_oo"]     = declare<T>(chem_env, "0724_bb_oo");
    tmps["0725_bb_oo"]     = declare<T>(chem_env, "0725_bb_oo");
    tmps["0726_bb_vo"]     = declare<T>(chem_env, "0726_bb_vo");
    tmps["0727_abab_vooo"] = declare<T>(chem_env, "0727_abab_vooo");
    tmps["0728_abab_vooo"] = declare<T>(chem_env, "0728_abab_vooo");
    tmps["0729_abab_vooo"] = declare<T>(chem_env, "0729_abab_vooo");
    tmps["0730_abab_ovoo"] = declare<T>(chem_env, "0730_abab_ovoo");
    tmps["0731_baab_vooo"] = declare<T>(chem_env, "0731_baab_vooo");
    tmps["0732_baab_vooo"] = declare<T>(chem_env, "0732_baab_vooo");
    tmps["0733_baab_vvoo"] = declare<T>(chem_env, "0733_baab_vvoo");
    tmps["0734_abab_oooo"] = declare<T>(chem_env, "0734_abab_oooo");
    tmps["0735_baab_vooo"] = declare<T>(chem_env, "0735_baab_vooo");
    tmps["0736_abba_oooo"] = declare<T>(chem_env, "0736_abba_oooo");
    tmps["0737_baba_vooo"] = declare<T>(chem_env, "0737_baba_vooo");
    tmps["0738_baab_vooo"] = declare<T>(chem_env, "0738_baab_vooo");
    tmps["0739_baab_vooo"] = declare<T>(chem_env, "0739_baab_vooo");
    tmps["0740_abab_vvoo"] = declare<T>(chem_env, "0740_abab_vvoo");
    tmps["0741_abba_vvoo"] = declare<T>(chem_env, "0741_abba_vvoo");
    tmps["0742_abab_vvoo"] = declare<T>(chem_env, "0742_abab_vvoo");
    tmps["0743_abab_vvoo"] = declare<T>(chem_env, "0743_abab_vvoo");
    tmps["0744_baab_vvoo"] = declare<T>(chem_env, "0744_baab_vvoo");
    tmps["0745_bb_oo"]     = declare<T>(chem_env, "0745_bb_oo");
    tmps["0746_abab_vvoo"] = declare<T>(chem_env, "0746_abab_vvoo");
    tmps["0747_abba_vooo"] = declare<T>(chem_env, "0747_abba_vooo");
    tmps["0748_abab_vooo"] = declare<T>(chem_env, "0748_abab_vooo");
    tmps["0749_abab_vooo"] = declare<T>(chem_env, "0749_abab_vooo");
    tmps["0750_baab_vvoo"] = declare<T>(chem_env, "0750_baab_vvoo");
    tmps["0751_bb_vo"]     = declare<T>(chem_env, "0751_bb_vo");
    tmps["0752_bb_vo"]     = declare<T>(chem_env, "0752_bb_vo");
    tmps["0753_bb_vo"]     = declare<T>(chem_env, "0753_bb_vo");
    tmps["0754_abab_vvoo"] = declare<T>(chem_env, "0754_abab_vvoo");
    tmps["0755_abab_vvoo"] = declare<T>(chem_env, "0755_abab_vvoo");
    tmps["0756_baab_vvoo"] = declare<T>(chem_env, "0756_baab_vvoo");
    tmps["0757_abab_vvoo"] = declare<T>(chem_env, "0757_abab_vvoo");
    tmps["0758_abab_vvoo"] = declare<T>(chem_env, "0758_abab_vvoo");
    tmps["0759_abba_vvoo"] = declare<T>(chem_env, "0759_abba_vvoo");
    tmps["0760_aa_vo"]     = declare<T>(chem_env, "0760_aa_vo");
    tmps["0761_aa_vo"]     = declare<T>(chem_env, "0761_aa_vo");
    tmps["0762_aa_vo"]     = declare<T>(chem_env, "0762_aa_vo");
    tmps["0763_baba_vvoo"] = declare<T>(chem_env, "0763_baba_vvoo");
    tmps["0764_bb_oo"]     = declare<T>(chem_env, "0764_bb_oo");
    tmps["0765_bb_oo"]     = declare<T>(chem_env, "0765_bb_oo");
    tmps["0766_bb_oo"]     = declare<T>(chem_env, "0766_bb_oo");
    tmps["0767_abab_vvoo"] = declare<T>(chem_env, "0767_abab_vvoo");
    tmps["0768_abba_vvoo"] = declare<T>(chem_env, "0768_abba_vvoo");
    tmps["0769_abba_vvoo"] = declare<T>(chem_env, "0769_abba_vvoo");
    tmps["0770_abab_vvoo"] = declare<T>(chem_env, "0770_abab_vvoo");
    tmps["0771_abab_vvoo"] = declare<T>(chem_env, "0771_abab_vvoo");
    tmps["0772_abba_vvoo"] = declare<T>(chem_env, "0772_abba_vvoo");
    tmps["0773_abab_vvoo"] = declare<T>(chem_env, "0773_abab_vvoo");
    tmps["0774_baab_vvoo"] = declare<T>(chem_env, "0774_baab_vvoo");
    tmps["0775_abab_ovoo"] = declare<T>(chem_env, "0775_abab_ovoo");
    tmps["0776_abab_ovoo"] = declare<T>(chem_env, "0776_abab_ovoo");
    tmps["0777_baab_vooo"] = declare<T>(chem_env, "0777_baab_vooo");
    tmps["0778_abab_ovoo"] = declare<T>(chem_env, "0778_abab_ovoo");
    tmps["0779_abab_vvoo"] = declare<T>(chem_env, "0779_abab_vvoo");
    tmps["0780_baab_vvoo"] = declare<T>(chem_env, "0780_baab_vvoo");
    tmps["0781_baba_vvoo"] = declare<T>(chem_env, "0781_baba_vvoo");
    tmps["0782_abba_vvoo"] = declare<T>(chem_env, "0782_abba_vvoo");
    tmps["0783_abba_vooo"] = declare<T>(chem_env, "0783_abba_vooo");
    tmps["0784_abba_vooo"] = declare<T>(chem_env, "0784_abba_vooo");
    tmps["0785_abab_vooo"] = declare<T>(chem_env, "0785_abab_vooo");
    tmps["0786_abab_vooo"] = declare<T>(chem_env, "0786_abab_vooo");
    tmps["0787_abab_vooo"] = declare<T>(chem_env, "0787_abab_vooo");
    tmps["0788_abab_vooo"] = declare<T>(chem_env, "0788_abab_vooo");
    tmps["0789_baab_vvoo"] = declare<T>(chem_env, "0789_baab_vvoo");
    tmps["0790_abab_vvoo"] = declare<T>(chem_env, "0790_abab_vvoo");
    tmps["0791_abab_vvoo"] = declare<T>(chem_env, "0791_abab_vvoo");
    tmps["0792_baab_vooo"] = declare<T>(chem_env, "0792_baab_vooo");
    tmps["0793_baab_vooo"] = declare<T>(chem_env, "0793_baab_vooo");
    tmps["0794_abab_ovoo"] = declare<T>(chem_env, "0794_abab_ovoo");
    tmps["0795_baab_vooo"] = declare<T>(chem_env, "0795_baab_vooo");
    tmps["0796_baab_vooo"] = declare<T>(chem_env, "0796_baab_vooo");
    tmps["0797_abab_vvoo"] = declare<T>(chem_env, "0797_abab_vvoo");
    tmps["0798_abab_vvoo"] = declare<T>(chem_env, "0798_abab_vvoo");
    tmps["0799_abba_vvoo"] = declare<T>(chem_env, "0799_abba_vvoo");
    tmps["0800_baab_vvoo"] = declare<T>(chem_env, "0800_baab_vvoo");
    tmps["0801_abab_vvoo"] = declare<T>(chem_env, "0801_abab_vvoo");
    tmps["0802_baba_vooo"] = declare<T>(chem_env, "0802_baba_vooo");
    tmps["0803_abab_vvoo"] = declare<T>(chem_env, "0803_abab_vvoo");
    tmps["0804_baba_vooo"] = declare<T>(chem_env, "0804_baba_vooo");
    tmps["0805_abba_vvoo"] = declare<T>(chem_env, "0805_abba_vvoo");
    tmps["0806_abba_vvoo"] = declare<T>(chem_env, "0806_abba_vvoo");
    tmps["0807_abba_vvoo"] = declare<T>(chem_env, "0807_abba_vvoo");
    tmps["0808_bb_oo"]     = declare<T>(chem_env, "0808_bb_oo");
    tmps["0809_bb_oo"]     = declare<T>(chem_env, "0809_bb_oo");
    tmps["0810_aa_ov"]     = declare<T>(chem_env, "0810_aa_ov");
    tmps["0811_bb_ov"]     = declare<T>(chem_env, "0811_bb_ov");
    tmps["0812_aa_vv"]     = declare<T>(chem_env, "0812_aa_vv");
    tmps["0813_bb_vv"]     = declare<T>(chem_env, "0813_bb_vv");
    tmps["0814_baba_ovoo"] = declare<T>(chem_env, "0814_baba_ovoo");
    tmps["0815_aaaa_vovo"] = declare<T>(chem_env, "0815_aaaa_vovo");
    tmps["0816_abab_vvoo"] = declare<T>(chem_env, "0816_abab_vvoo");
    tmps["0817_abab_vvoo"] = declare<T>(chem_env, "0817_abab_vvoo");
    tmps["0818_abab_vvoo"] = declare<T>(chem_env, "0818_abab_vvoo");
    tmps["0819_abab_vvoo"] = declare<T>(chem_env, "0819_abab_vvoo");
    tmps["0820_abab_vvoo"] = declare<T>(chem_env, "0820_abab_vvoo");
    tmps["0821_abab_vvoo"] = declare<T>(chem_env, "0821_abab_vvoo");
    tmps["0822_abab_vvoo"] = declare<T>(chem_env, "0822_abab_vvoo");
    tmps["0823_abab_vvoo"] = declare<T>(chem_env, "0823_abab_vvoo");
    tmps["0824_abab_vvoo"] = declare<T>(chem_env, "0824_abab_vvoo");
    tmps["0825_abab_vvoo"] = declare<T>(chem_env, "0825_abab_vvoo");
    tmps["0826_abab_vvoo"] = declare<T>(chem_env, "0826_abab_vvoo");
    tmps["0827_abab_vvoo"] = declare<T>(chem_env, "0827_abab_vvoo");
    tmps["0828_abab_vvoo"] = declare<T>(chem_env, "0828_abab_vvoo");
    tmps["0829_abab_vvoo"] = declare<T>(chem_env, "0829_abab_vvoo");
    tmps["0830_abba_vvoo"] = declare<T>(chem_env, "0830_abba_vvoo");
    tmps["0831_abab_vvoo"] = declare<T>(chem_env, "0831_abab_vvoo");
    tmps["0832_baba_vvoo"] = declare<T>(chem_env, "0832_baba_vvoo");
    tmps["0833_abab_vvoo"] = declare<T>(chem_env, "0833_abab_vvoo");
    tmps["0834_abab_vvoo"] = declare<T>(chem_env, "0834_abab_vvoo");
    tmps["0835_abab_vvoo"] = declare<T>(chem_env, "0835_abab_vvoo");
    tmps["0836_abab_vvoo"] = declare<T>(chem_env, "0836_abab_vvoo");
    tmps["0837_abba_vvoo"] = declare<T>(chem_env, "0837_abba_vvoo");
    tmps["0838_abab_vvoo"] = declare<T>(chem_env, "0838_abab_vvoo");
    tmps["0839_bb_oo"]     = declare<T>(chem_env, "0839_bb_oo");
    tmps["0840_abab_vvoo"] = declare<T>(chem_env, "0840_abab_vvoo");
    tmps["0841_bb_oo"]     = declare<T>(chem_env, "0841_bb_oo");
    tmps["0842_abab_vooo"] = declare<T>(chem_env, "0842_abab_vooo");
    tmps["0843_baba_vooo"] = declare<T>(chem_env, "0843_baba_vooo");
    tmps["0844_baab_vooo"] = declare<T>(chem_env, "0844_baab_vooo");
    tmps["0845_baab_vvoo"] = declare<T>(chem_env, "0845_baab_vvoo");
    tmps["0846_abab_vvoo"] = declare<T>(chem_env, "0846_abab_vvoo");
    tmps["0847_abba_vvoo"] = declare<T>(chem_env, "0847_abba_vvoo");
    tmps["0848_abba_vvoo"] = declare<T>(chem_env, "0848_abba_vvoo");
    tmps["0849_abba_vvoo"] = declare<T>(chem_env, "0849_abba_vvoo");
    tmps["0850_abab_vvoo"] = declare<T>(chem_env, "0850_abab_vvoo");
    tmps["0851_abab_vvoo"] = declare<T>(chem_env, "0851_abab_vvoo");
    tmps["0852_abab_vvoo"] = declare<T>(chem_env, "0852_abab_vvoo");
    tmps["0853_abba_vvoo"] = declare<T>(chem_env, "0853_abba_vvoo");
    tmps["0854_baab_vvoo"] = declare<T>(chem_env, "0854_baab_vvoo");
    tmps["0855_abab_vvoo"] = declare<T>(chem_env, "0855_abab_vvoo");
    tmps["0856_baab_vvoo"] = declare<T>(chem_env, "0856_baab_vvoo");
    tmps["0857_abab_vvoo"] = declare<T>(chem_env, "0857_abab_vvoo");
    tmps["0858_baba_vvoo"] = declare<T>(chem_env, "0858_baba_vvoo");
    tmps["0859_abab_vvoo"] = declare<T>(chem_env, "0859_abab_vvoo");
    tmps["0860_baba_vvoo"] = declare<T>(chem_env, "0860_baba_vvoo");
    tmps["0861_abba_vvoo"] = declare<T>(chem_env, "0861_abba_vvoo");
    tmps["0862_baba_vvoo"] = declare<T>(chem_env, "0862_baba_vvoo");
    tmps["0863_baab_vvoo"] = declare<T>(chem_env, "0863_baab_vvoo");
    tmps["0864_abba_vvoo"] = declare<T>(chem_env, "0864_abba_vvoo");
    tmps["0865_baab_vvoo"] = declare<T>(chem_env, "0865_baab_vvoo");
    tmps["0866_abab_vvoo"] = declare<T>(chem_env, "0866_abab_vvoo");
    tmps["0867_abab_vvoo"] = declare<T>(chem_env, "0867_abab_vvoo");
    tmps["0868_abba_vvoo"] = declare<T>(chem_env, "0868_abba_vvoo");
    tmps["0869_abab_vvoo"] = declare<T>(chem_env, "0869_abab_vvoo");
    tmps["0870_abba_vvoo"] = declare<T>(chem_env, "0870_abba_vvoo");
    tmps["0871_abab_vvoo"] = declare<T>(chem_env, "0871_abab_vvoo");
    tmps["0872_abab_vvoo"] = declare<T>(chem_env, "0872_abab_vvoo");
    tmps["0873_baba_vvoo"] = declare<T>(chem_env, "0873_baba_vvoo");
    tmps["0874_baba_vvoo"] = declare<T>(chem_env, "0874_baba_vvoo");
    tmps["0875_abba_vvoo"] = declare<T>(chem_env, "0875_abba_vvoo");
    tmps["0876_abab_vvoo"] = declare<T>(chem_env, "0876_abab_vvoo");
    tmps["0877_abab_vvoo"] = declare<T>(chem_env, "0877_abab_vvoo");
    tmps["0878_baba_vvoo"] = declare<T>(chem_env, "0878_baba_vvoo");
    tmps["0879_baba_vvoo"] = declare<T>(chem_env, "0879_baba_vvoo");
    tmps["0880_abab_vvoo"] = declare<T>(chem_env, "0880_abab_vvoo");
    tmps["0881_baba_vvoo"] = declare<T>(chem_env, "0881_baba_vvoo");
    tmps["0882_baab_vvoo"] = declare<T>(chem_env, "0882_baab_vvoo");
    tmps["0883_abba_vvoo"] = declare<T>(chem_env, "0883_abba_vvoo");
    tmps["0884_abba_vvoo"] = declare<T>(chem_env, "0884_abba_vvoo");
    tmps["0885_abba_vvoo"] = declare<T>(chem_env, "0885_abba_vvoo");
    tmps["0886_baba_vvoo"] = declare<T>(chem_env, "0886_baba_vvoo");
    tmps["0887_baba_vvoo"] = declare<T>(chem_env, "0887_baba_vvoo");
    tmps["0888_baab_vvoo"] = declare<T>(chem_env, "0888_baab_vvoo");
    tmps["0889_baab_vvoo"] = declare<T>(chem_env, "0889_baab_vvoo");
    tmps["0890_baab_vvoo"] = declare<T>(chem_env, "0890_baab_vvoo");
    tmps["0891_baab_vvoo"] = declare<T>(chem_env, "0891_baab_vvoo");
    tmps["0892_baab_vvoo"] = declare<T>(chem_env, "0892_baab_vvoo");
    tmps["0893_abab_vvoo"] = declare<T>(chem_env, "0893_abab_vvoo");
    tmps["0894_baab_vvoo"] = declare<T>(chem_env, "0894_baab_vvoo");
    tmps["0895_abab_vvoo"] = declare<T>(chem_env, "0895_abab_vvoo");
    tmps["0896_abba_vvoo"] = declare<T>(chem_env, "0896_abba_vvoo");
    tmps["0897_baba_vvoo"] = declare<T>(chem_env, "0897_baba_vvoo");
    tmps["0898_abab_vvoo"] = declare<T>(chem_env, "0898_abab_vvoo");
    tmps["0899_abab_vvoo"] = declare<T>(chem_env, "0899_abab_vvoo");
    tmps["0900_baab_vvoo"] = declare<T>(chem_env, "0900_baab_vvoo");
    tmps["0901_baab_vvoo"] = declare<T>(chem_env, "0901_baab_vvoo");
    tmps["0902_abab_vvoo"] = declare<T>(chem_env, "0902_abab_vvoo");
    tmps["0903_baba_vvoo"] = declare<T>(chem_env, "0903_baba_vvoo");
    tmps["0904_baab_vvoo"] = declare<T>(chem_env, "0904_baab_vvoo");
    tmps["0905_baab_vvoo"] = declare<T>(chem_env, "0905_baab_vvoo");
    tmps["0906_abab_vvoo"] = declare<T>(chem_env, "0906_abab_vvoo");
    tmps["0907_abab_vvoo"] = declare<T>(chem_env, "0907_abab_vvoo");
    tmps["0908_baba_vvoo"] = declare<T>(chem_env, "0908_baba_vvoo");
    tmps["0909_abab_vvoo"] = declare<T>(chem_env, "0909_abab_vvoo");
    tmps["0910_abab_vvoo"] = declare<T>(chem_env, "0910_abab_vvoo");
    tmps["0911_abab_vvoo"] = declare<T>(chem_env, "0911_abab_vvoo");
    tmps["0912_baba_vvoo"] = declare<T>(chem_env, "0912_baba_vvoo");
    tmps["0913_abab_vvoo"] = declare<T>(chem_env, "0913_abab_vvoo");
    tmps["0914_baab_vvoo"] = declare<T>(chem_env, "0914_baab_vvoo");
    tmps["0915_abab_vvoo"] = declare<T>(chem_env, "0915_abab_vvoo");
    tmps["0916_baab_vvoo"] = declare<T>(chem_env, "0916_baab_vvoo");
    tmps["0917_baba_vvoo"] = declare<T>(chem_env, "0917_baba_vvoo");
    tmps["0918_abab_vvoo"] = declare<T>(chem_env, "0918_abab_vvoo");
    tmps["0919_baba_vooo"] = declare<T>(chem_env, "0919_baba_vooo");
    tmps["0920_abab_vvoo"] = declare<T>(chem_env, "0920_abab_vvoo");
    tmps["0921_baab_vvoo"] = declare<T>(chem_env, "0921_baab_vvoo");
    tmps["0922_abab_vvoo"] = declare<T>(chem_env, "0922_abab_vvoo");
    tmps["0923_abab_vvoo"] = declare<T>(chem_env, "0923_abab_vvoo");
    tmps["0924_abab_vvoo"] = declare<T>(chem_env, "0924_abab_vvoo");
    tmps["0925_abba_vvoo"] = declare<T>(chem_env, "0925_abba_vvoo");
    tmps["0926_abab_vvoo"] = declare<T>(chem_env, "0926_abab_vvoo");
    tmps["0927_abab_vvoo"] = declare<T>(chem_env, "0927_abab_vvoo");
    tmps["0928_abab_vvoo"] = declare<T>(chem_env, "0928_abab_vvoo");
    tmps["0929_abba_vvoo"] = declare<T>(chem_env, "0929_abba_vvoo");
    tmps["0930_abba_vvoo"] = declare<T>(chem_env, "0930_abba_vvoo");
    tmps["0931_abab_vvoo"] = declare<T>(chem_env, "0931_abab_vvoo");
    tmps["0932_baba_vvoo"] = declare<T>(chem_env, "0932_baba_vvoo");
    tmps["0933_abba_vvoo"] = declare<T>(chem_env, "0933_abba_vvoo");
    tmps["0934_abba_vvoo"] = declare<T>(chem_env, "0934_abba_vvoo");
    tmps["0935_baab_vvoo"] = declare<T>(chem_env, "0935_baab_vvoo");
    tmps["0936_abab_vvoo"] = declare<T>(chem_env, "0936_abab_vvoo");
    tmps["0937_abab_vvoo"] = declare<T>(chem_env, "0937_abab_vvoo");
    tmps["0938_baba_vvoo"] = declare<T>(chem_env, "0938_baba_vvoo");
    tmps["0939_abba_vvoo"] = declare<T>(chem_env, "0939_abba_vvoo");
    tmps["0940_baab_vvoo"] = declare<T>(chem_env, "0940_baab_vvoo");
    tmps["0941_abab_vvoo"] = declare<T>(chem_env, "0941_abab_vvoo");
    tmps["0942_baab_vvoo"] = declare<T>(chem_env, "0942_baab_vvoo");
    tmps["0943_abab_vvoo"] = declare<T>(chem_env, "0943_abab_vvoo");
    tmps["0944_baab_vvoo"] = declare<T>(chem_env, "0944_baab_vvoo");
    tmps["0945_abab_vvoo"] = declare<T>(chem_env, "0945_abab_vvoo");
    tmps["0946_baab_vvoo"] = declare<T>(chem_env, "0946_baab_vvoo");
    tmps["0947_abba_vvoo"] = declare<T>(chem_env, "0947_abba_vvoo");
    tmps["0948_abba_vvoo"] = declare<T>(chem_env, "0948_abba_vvoo");
  }

  {
    sch(scalars.at("0001")() = 0.0)(scalars.at("0002")() = 0.0)(scalars.at("0003")() = 0.0)(
      scalars.at("0004")() = 0.0)(scalars.at("0005")() = 0.0)(scalars.at("0006")() = 0.0)(
      scalars.at("0007")() = 0.0)(scalars.at("0008")() = 0.0)(scalars.at("0009")() = 0.0)(
      scalars.at("0010")() = 0.0)(scalars.at("0011")() = 0.0)(scalars.at("0012")() = 0.0)(
      scalars.at("0013")() = 0.0)(scalars.at("0014")() = 0.0)(scalars.at("0015")() = 0.0)(
      scalars.at("0016")() = 0.0)(scalars.at("0017")() = 0.0)(scalars.at("0018")() = 0.0)(
      scalars.at("0019")() = 0.0)(scalars.at("0020")() = 0.0)(scalars.at("0021")() = 0.0)(
      scalars.at("0022")() = 0.0)(scalars.at("0023")() = 0.0)(scalars.at("0024")() = 0.0)(
      scalars.at("0025")() = 0.0)(scalars.at("0026")() = 0.0)(scalars.at("0027")() = 0.0)(
      scalars.at("0028")() = 0.0)(scalars.at("0029")() = 0.0)(scalars.at("0030")() = 0.0)(
      scalars.at("0031")() = 0.0)(scalars.at("0032")() = 0.0)(scalars.at("0033")() = 0.0)(
      scalars.at("0034")() = 0.0)(scalars.at("0035")() = 0.0)(scalars.at("0036")() = 0.0)(
      scalars.at("0037")() = 0.0)(scalars.at("0038")() = 0.0)(scalars.at("0039")() = 0.0)(
      scalars.at("0040")() = 0.0)(scalars.at("0041")() = 0.0)(scalars.at("0042")() = 0.0)(
      scalars.at("0043")() = 0.0)(scalars.at("0044")() = 0.0)(scalars.at("0045")() = 0.0)(
      scalars.at("0046")() = 0.0)(scalars.at("0047")() = 0.0)(scalars.at("0048")() = 0.0)(
      scalars.at("0049")() = 0.0)(scalars.at("0050")() = 0.0)(scalars.at("0051")() = 0.0)(
      scalars.at("0052")() = 0.0)(scalars.at("0053")() = 0.0)(scalars.at("0054")() = 0.0)

      (scalars.at("0001")() = dp.at("bb_ov")(ib, ab) * t1_1p.at("bb")(ab, ib))(
        scalars.at("0002")() = dp.at("aa_ov")(ia, aa) * t1_1p.at("aa")(aa, ia))(
        scalars.at("0003")() = scalars.at("0001")())(scalars.at("0003")() += scalars.at("0002")())(
        tmps.at("bin_bb_vo")(ab, ib) = eri.at("bbbb_oovv")(jb, ib, bb, ab) * t1.at("bb")(bb, jb))(
        scalars.at("0004")() = tmps.at("bin_bb_vo")(ab, ib) * t1_1p.at("bb")(ab, ib))(
        scalars.at("0005")() =
          eri.at("bbbb_oovv")(ib, jb, ab, bb) * t2_1p.at("bbbb")(ab, bb, jb, ib))(
        tmps.at("bin_aa_vo")(aa, ia) = eri.at("abab_oovv")(ia, jb, aa, bb) * t1.at("bb")(bb, jb))(
        scalars.at("0006")() = tmps.at("bin_aa_vo")(aa, ia) * t1_1p.at("aa")(aa, ia))(
        tmps.at("bin_aa_vo")(aa, ia) =
          eri.at("abab_oovv")(ia, jb, aa, bb) * t1_1p.at("bb")(bb, jb))(
        scalars.at("0007")() = tmps.at("bin_aa_vo")(aa, ia) * t1.at("aa")(aa, ia))(
        scalars.at("0008")() =
          eri.at("abab_oovv")(ia, jb, aa, bb) * t2_1p.at("abab")(aa, bb, ia, jb))(
        tmps.at("bin_aa_vo")(aa, ia) = eri.at("aaaa_oovv")(ja, ia, ba, aa) * t1.at("aa")(ba, ja))(
        scalars.at("0009")() = tmps.at("bin_aa_vo")(aa, ia) * t1_1p.at("aa")(aa, ia))(
        scalars.at("0010")() =
          eri.at("aaaa_oovv")(ia, ja, aa, ba) * t2_1p.at("aaaa")(aa, ba, ja, ia))(
        scalars.at("0011")() = f.at("bb_ov")(ib, ab) * t1_1p.at("bb")(ab, ib))(
        scalars.at("0012")() = f.at("aa_ov")(ia, aa) * t1_1p.at("aa")(aa, ia))(
        scalars.at("0013")() = dp.at("bb_ov")(ib, ab) * t1.at("bb")(ab, ib))(
        scalars.at("0014")() = dp.at("bb_ov")(ib, ab) * t1_2p.at("bb")(ab, ib))(
        scalars.at("0015")() = dp.at("aa_ov")(ia, aa) * t1.at("aa")(aa, ia))(
        scalars.at("0016")() = dp.at("aa_ov")(ia, aa) * t1_2p.at("aa")(aa, ia))(
        scalars.at("0017")() = -0.250 * scalars.at("0005")())(scalars.at("0017")() +=
                                                              scalars.at("0008")())(
        scalars.at("0017")() += scalars.at("0015")())(scalars.at("0017")() += scalars.at("0006")())(
        scalars.at("0017")() += scalars.at("0011")())(scalars.at("0017")() += scalars.at("0007")())(
        scalars.at("0017")() += scalars.at("0012")())(scalars.at("0017")() += scalars.at("0009")())(
        scalars.at("0017")() += scalars.at("0004")())(scalars.at("0017")() += scalars.at("0013")())(
        scalars.at("0017")() -= 0.250 * scalars.at("0010")())(scalars.at("0017")() +=
                                                              2.00 * scalars.at("0016")())(
        scalars.at("0017")() += 2.00 * scalars.at("0014")())(scalars.at("0018")() = t0_1p * w0)(
        scalars.at("0019")() = t0_1p * scalars.at("0002")())(scalars.at("0020")() =
                                                               t0_2p * scalars.at("0015")())(
        scalars.at("0021")() = t0_2p * scalars.at("0013")())(scalars.at("0022")() =
                                                               t0_1p * scalars.at("0001")())(
        scalars.at("0023")() = 2.00 * scalars.at("0020")())(scalars.at("0023")() +=
                                                            2.00 * scalars.at("0021")())(
        scalars.at("0023")() += scalars.at("0017")())(scalars.at("0023")() += scalars.at("0018")())(
        scalars.at("0023")() += scalars.at("0019")())(scalars.at("0023")() += scalars.at("0022")())(
        tmps.at("bin_bb_vo")(bb, jb) = eri.at("bbbb_oovv")(jb, ib, bb, ab) * t1.at("bb")(ab, ib))(
        scalars.at("0024")() = tmps.at("bin_bb_vo")(bb, jb) * t1.at("bb")(bb, jb))(
        scalars.at("0025")() = eri.at("bbbb_oovv")(ib, jb, ab, bb) * t2.at("bbbb")(ab, bb, jb, ib))(
        tmps.at("bin_aa_vo")(aa, ia) = eri.at("abab_oovv")(ia, jb, aa, bb) * t1.at("bb")(bb, jb))(
        scalars.at("0026")() = tmps.at("bin_aa_vo")(aa, ia) * t1.at("aa")(aa, ia))(
        scalars.at("0027")() = eri.at("abab_oovv")(ia, jb, aa, bb) * t2.at("abab")(aa, bb, ia, jb))(
        tmps.at("bin_aa_vo")(ba, ja) = eri.at("aaaa_oovv")(ja, ia, ba, aa) * t1.at("aa")(aa, ia))(
        scalars.at("0028")() = tmps.at("bin_aa_vo")(ba, ja) * t1.at("aa")(ba, ja))(
        scalars.at("0029")() = eri.at("aaaa_oovv")(ia, ja, aa, ba) * t2.at("aaaa")(aa, ba, ja, ia))(
        scalars.at("0030")() = f.at("bb_ov")(ib, ab) * t1.at("bb")(ab, ib))(
        scalars.at("0031")() = f.at("aa_ov")(ia, aa) * t1.at("aa")(aa, ia))(
        scalars.at("0032")() = -2.00 * scalars.at("0024")())(scalars.at("0032")() -=
                                                             2.00 * scalars.at("0028")())(
        scalars.at("0032")() -= 4.00 * scalars.at("0026")())(scalars.at("0032")() -=
                                                             4.00 * scalars.at("0027")())(
        scalars.at("0032")() += scalars.at("0029")())(scalars.at("0032")() -=
                                                      4.00 * scalars.at("0031")())(
        scalars.at("0032")() -= 4.00 * scalars.at("0030")())(scalars.at("0032")() +=
                                                             scalars.at("0025")())(
        scalars.at("0033")() = t0_1p * scalars.at("0015")())(scalars.at("0034")() =
                                                               t0_1p * scalars.at("0013")())(
        scalars.at("0035")() = -4.00 * scalars.at("0033")())(scalars.at("0035")() +=
                                                             scalars.at("0032")())(
        scalars.at("0035")() -= 4.00 * scalars.at("0034")())(tmps.at("bin_bb_vo")(bb, jb) =
                                                               eri.at("bbbb_oovv")(jb, ib, bb, ab) *
                                                               t1_1p.at("bb")(ab, ib))(
        scalars.at("0036")() = tmps.at("bin_bb_vo")(bb, jb) * t1_1p.at("bb")(bb, jb))(
        tmps.at("bin_bb_vo")(ab, ib) = eri.at("bbbb_oovv")(jb, ib, bb, ab) * t1.at("bb")(bb, jb))(
        scalars.at("0037")() = tmps.at("bin_bb_vo")(ab, ib) * t1_2p.at("bb")(ab, ib))(
        scalars.at("0038")() =
          eri.at("bbbb_oovv")(ib, jb, ab, bb) * t2_2p.at("bbbb")(ab, bb, jb, ib))(
        tmps.at("bin_aa_vo")(aa, ia) = eri.at("abab_oovv")(ia, jb, aa, bb) * t1.at("bb")(bb, jb))(
        scalars.at("0039")() = tmps.at("bin_aa_vo")(aa, ia) * t1_2p.at("aa")(aa, ia))(
        tmps.at("bin_aa_vo")(aa, ia) =
          eri.at("abab_oovv")(ia, jb, aa, bb) * t1_2p.at("bb")(bb, jb))(
        scalars.at("0040")() = tmps.at("bin_aa_vo")(aa, ia) * t1.at("aa")(aa, ia))(
        tmps.at("bin_aa_vo")(aa, ia) =
          eri.at("abab_oovv")(ia, jb, aa, bb) * t1_1p.at("bb")(bb, jb))(
        scalars.at("0041")() = tmps.at("bin_aa_vo")(aa, ia) * t1_1p.at("aa")(aa, ia))(
        scalars.at("0042")() =
          eri.at("abab_oovv")(ia, jb, aa, bb) * t2_2p.at("abab")(aa, bb, ia, jb))(
        tmps.at("bin_aa_vo")(aa, ia) = eri.at("aaaa_oovv")(ja, ia, ba, aa) * t1.at("aa")(ba, ja))(
        scalars.at("0043")() = tmps.at("bin_aa_vo")(aa, ia) * t1_2p.at("aa")(aa, ia))(
        tmps.at("bin_aa_vo")(ba, ja) =
          eri.at("aaaa_oovv")(ja, ia, ba, aa) * t1_1p.at("aa")(aa, ia))(
        scalars.at("0044")() = tmps.at("bin_aa_vo")(ba, ja) * t1_1p.at("aa")(ba, ja))(
        scalars.at("0045")() =
          eri.at("aaaa_oovv")(ia, ja, aa, ba) * t2_2p.at("aaaa")(aa, ba, ja, ia))(
        scalars.at("0046")() = f.at("bb_ov")(ib, ab) * t1_2p.at("bb")(ab, ib))(
        scalars.at("0047")() = f.at("aa_ov")(ia, aa) * t1_2p.at("aa")(aa, ia))(
        scalars.at("0048")() = -0.250 * scalars.at("0038")())(scalars.at("0048")() -=
                                                              0.250 * scalars.at("0045")())(
        scalars.at("0048")() += scalars.at("0039")())(scalars.at("0048")() += scalars.at("0047")())(
        scalars.at("0048")() += scalars.at("0046")())(scalars.at("0048")() +=
                                                      0.50 * scalars.at("0036")())(
        scalars.at("0048")() += 0.50 * scalars.at("0044")())(scalars.at("0048")() +=
                                                             scalars.at("0037")())(
        scalars.at("0048")() += scalars.at("0041")())(scalars.at("0048")() += scalars.at("0040")())(
        scalars.at("0048")() += scalars.at("0042")())(scalars.at("0048")() += scalars.at("0043")())(
        scalars.at("0049")() = t0_2p * w0)(scalars.at("0050")() = t0_2p * scalars.at("0002")())(
        scalars.at("0051")() = t0_1p * scalars.at("0016")())(scalars.at("0052")() =
                                                               t0_1p * scalars.at("0014")())(
        scalars.at("0053")() = t0_2p * scalars.at("0001")())(scalars.at("0054")() =
                                                               0.50 * scalars.at("0048")())(
        scalars.at("0054")() += scalars.at("0049")())(scalars.at("0054")() += scalars.at("0053")())(
        scalars.at("0054")() += scalars.at("0050")())(
        scalars.at("0054")() += 0.50 * scalars.at("0052")())(scalars.at("0054")() +=
                                                             0.50 * scalars.at("0051")())

        ;
  }
}

template void exachem::cc::qed_ccsd_cs::build_tmps<double>(
  Scheduler& sch, ChemEnv& chem_env, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p);