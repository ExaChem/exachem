/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "../../qed_ccsd_cs.hpp"

template<typename T>
Tensor<T> exachem::cc::qed_ccsd_cs::declare(const TiledIndexSpace& MO, const std::string& name) {
  // extract dimension and spin block from name (e.g. if name is "0004_aaaa_vvoo", extract "aaaa"
  // and "vvoo") where "vvoo" is the dimension and "aaaa" is the spin block
  size_t      num_underscore = std::count(name.begin(), name.end(), '_');
  std::string dimstring, spin_blk;

  if(num_underscore == 1) {
    // tensor has only one underscore, e.g. "aaaa_vvoo"
    size_t pos = name.find('_');
    spin_blk   = name.substr(0, pos);
    dimstring  = name.substr(pos + 1);
  }
  else if(num_underscore == 2) {
    // tensor has two underscores, e.g. "0004_aaaa_vvoo"
    size_t pos1 = name.find('_');
    size_t pos2 = name.find('_', pos1 + 1);
    dimstring   = name.substr(pos2 + 1);
    spin_blk    = name.substr(pos1 + 1, pos2 - pos1 - 1);
  }
  else if(num_underscore == 0) return Tensor<T>{};

  // get indices
  const int otiles  = MO("occ").num_tiles();
  const int vtiles  = MO("virt").num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();

  const TiledIndexSpace Oa = {MO("occ"), range(oatiles)};
  const TiledIndexSpace Va = {MO("virt"), range(vatiles)};
  const TiledIndexSpace Ob = {MO("occ"), range(oatiles, otiles)};
  const TiledIndexSpace Vb = {MO("virt"), range(vatiles, vtiles)};

  // now, initialize for different arrangements of dimstring
  Tensor<T>                           decl_tensor;
  size_t                              ndim = dimstring.size();
  std::vector<const TiledIndexSpace*> O(ndim, nullptr);
  std::vector<const TiledIndexSpace*> V(ndim, nullptr);

  // determine spins for O and V based on spin_blk
  for(size_t i = 0; i < ndim && i < 4; ++i) {
    char s = spin_blk[i];
    if(s == 'a') {
      O[i] = &Oa;
      V[i] = &Va;
    }
    else if(s == 'b') {
      O[i] = &Ob;
      V[i] = &Vb;
    }
    else {
      throw std::runtime_error("Invalid spin character: " + std::string(1, s) +
                               " in spin block: " + spin_blk + " for tensor: " + name);
    }
  }

  // assign TiledIndexSpaces based on dimstring
  std::vector<const TiledIndexSpace*> tis;
  for(size_t i = 0; i < ndim; ++i) {
    char d = dimstring[i];
    if(d == 'o') { tis.push_back(O[i]); }
    else if(d == 'v') { tis.push_back(V[i]); }
    else {
      throw std::runtime_error("Invalid dimension character: " + std::string(1, d) +
                               " in dimstring: " + dimstring);
    }
  }

  // Set the TiledIndexSpace for the tensor. Assumes upper triangular with paired indices.
  if(ndim == 2) return {{*tis[0], *tis[1]} /*, {1, 1}*/};
  if(ndim == 4) return {{*tis[0], *tis[1], *tis[2], *tis[3]} /*, {2, 2}*/};
  if(ndim == 6) return {{*tis[0], *tis[1], *tis[2], *tis[3], *tis[4], *tis[5]} /*, {3, 3}*/};
  else
    throw std::runtime_error("Unsupported tensor dimension: " + std::to_string(ndim) +
                             " for tensor: " + name);
}

template<typename T>
void exachem::cc::qed_ccsd_cs::build_tmps(Scheduler& sch, const TiledIndexSpace& MO,
                                          TensorMap<T>& tmps, TensorMap<T>& scalars,
                                          const TensorMap<T>& f, const TensorMap<T>& eri,
                                          const TensorMap<T>& dp, const double w0,
                                          const TensorMap<T>& t1, const TensorMap<T>& t2,
                                          const double t0_1p, const TensorMap<T>& t1_1p,
                                          const TensorMap<T>& t2_1p, const double t0_2p,
                                          const TensorMap<T>& t1_2p, const TensorMap<T>& t2_2p) {
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
    tmps["bin_aa_oo"]     = declare<T>(MO, "bin_aa_oo");
    tmps["bin_aa_vo"]     = declare<T>(MO, "bin_aa_vo");
    tmps["bin_aaaa_vooo"] = declare<T>(MO, "bin_aaaa_vooo");
    tmps["bin_aabb_oooo"] = declare<T>(MO, "bin_aabb_oooo");
    tmps["bin_aabb_vooo"] = declare<T>(MO, "bin_aabb_vooo");
    tmps["bin_abab_vvoo"] = declare<T>(MO, "bin_abab_vvoo");
    tmps["bin_abba_vvvo"] = declare<T>(MO, "bin_abba_vvvo");
    tmps["bin_baab_vooo"] = declare<T>(MO, "bin_baab_vooo");
    tmps["bin_bb_oo"]     = declare<T>(MO, "bin_bb_oo");
    tmps["bin_bb_vo"]     = declare<T>(MO, "bin_bb_vo");
    tmps["bin_bbaa_vvoo"] = declare<T>(MO, "bin_bbaa_vvoo");
    tmps["bin_bbbb_vooo"] = declare<T>(MO, "bin_bbbb_vooo");
    tmps["bin_bbbb_vvoo"] = declare<T>(MO, "bin_bbbb_vvoo");

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
    tmps["0001_abab_vvoo"] = declare<T>(MO, "0001_abab_vvoo");
    tmps["0002_abab_vvoo"] = declare<T>(MO, "0002_abab_vvoo");
    tmps["0003_abba_vvoo"] = declare<T>(MO, "0003_abba_vvoo");
    tmps["0004_abba_vvoo"] = declare<T>(MO, "0004_abba_vvoo");
    tmps["0005_abab_vvoo"] = declare<T>(MO, "0005_abab_vvoo");
    tmps["0006_abab_vvoo"] = declare<T>(MO, "0006_abab_vvoo");
    tmps["0007_abab_vvoo"] = declare<T>(MO, "0007_abab_vvoo");
    tmps["0008_abab_vvoo"] = declare<T>(MO, "0008_abab_vvoo");
    tmps["0009_abab_vvoo"] = declare<T>(MO, "0009_abab_vvoo");
    tmps["0010_abab_vooo"] = declare<T>(MO, "0010_abab_vooo");
    tmps["0011_baab_vvoo"] = declare<T>(MO, "0011_baab_vvoo");
    tmps["0012_abab_vooo"] = declare<T>(MO, "0012_abab_vooo");
    tmps["0013_baab_vvoo"] = declare<T>(MO, "0013_baab_vvoo");
    tmps["0014_baab_vooo"] = declare<T>(MO, "0014_baab_vooo");
    tmps["0015_abab_vvoo"] = declare<T>(MO, "0015_abab_vvoo");
    tmps["0016_baab_vooo"] = declare<T>(MO, "0016_baab_vooo");
    tmps["0017_abab_vvoo"] = declare<T>(MO, "0017_abab_vvoo");
    tmps["0018_abab_vvoo"] = declare<T>(MO, "0018_abab_vvoo");
    tmps["0019_abab_vvoo"] = declare<T>(MO, "0019_abab_vvoo");
    tmps["0020_abab_vvoo"] = declare<T>(MO, "0020_abab_vvoo");
    tmps["0021_abba_vvoo"] = declare<T>(MO, "0021_abba_vvoo");
    tmps["0022_abab_vvoo"] = declare<T>(MO, "0022_abab_vvoo");
    tmps["0023_abab_vvoo"] = declare<T>(MO, "0023_abab_vvoo");
    tmps["0024_abab_vvoo"] = declare<T>(MO, "0024_abab_vvoo");
    tmps["0025_abab_vvoo"] = declare<T>(MO, "0025_abab_vvoo");
    tmps["0026_baab_vvoo"] = declare<T>(MO, "0026_baab_vvoo");
    tmps["0027_abab_vvoo"] = declare<T>(MO, "0027_abab_vvoo");
    tmps["0028_baab_vvoo"] = declare<T>(MO, "0028_baab_vvoo");
    tmps["0029_abab_vvoo"] = declare<T>(MO, "0029_abab_vvoo");
    tmps["0030_abab_vvoo"] = declare<T>(MO, "0030_abab_vvoo");
    tmps["0031_abba_vvoo"] = declare<T>(MO, "0031_abba_vvoo");
    tmps["0032_abba_vvoo"] = declare<T>(MO, "0032_abba_vvoo");
    tmps["0033_abba_vvoo"] = declare<T>(MO, "0033_abba_vvoo");
    tmps["0034_abab_vvoo"] = declare<T>(MO, "0034_abab_vvoo");
    tmps["0035_abab_vvoo"] = declare<T>(MO, "0035_abab_vvoo");
    tmps["0036_abab_vvoo"] = declare<T>(MO, "0036_abab_vvoo");
    tmps["0037_abab_vvoo"] = declare<T>(MO, "0037_abab_vvoo");
    tmps["0038_abab_vvoo"] = declare<T>(MO, "0038_abab_vvoo");
    tmps["0039_bb_oo"]     = declare<T>(MO, "0039_bb_oo");
    tmps["0040_bb_oo"]     = declare<T>(MO, "0040_bb_oo");
    tmps["0041_bb_oo"]     = declare<T>(MO, "0041_bb_oo");
    tmps["0042_aa_oo"]     = declare<T>(MO, "0042_aa_oo");
    tmps["0043_aa_oo"]     = declare<T>(MO, "0043_aa_oo");
    tmps["0044_aa_oo"]     = declare<T>(MO, "0044_aa_oo");
    tmps["0045_abab_vvoo"] = declare<T>(MO, "0045_abab_vvoo");
    tmps["0046_baab_vvoo"] = declare<T>(MO, "0046_baab_vvoo");
    tmps["0047_baab_vvoo"] = declare<T>(MO, "0047_baab_vvoo");
    tmps["0048_abab_vooo"] = declare<T>(MO, "0048_abab_vooo");
    tmps["0049_baab_vvoo"] = declare<T>(MO, "0049_baab_vvoo");
    tmps["0050_abab_vvoo"] = declare<T>(MO, "0050_abab_vvoo");
    tmps["0051_baab_vooo"] = declare<T>(MO, "0051_baab_vooo");
    tmps["0052_abab_vvoo"] = declare<T>(MO, "0052_abab_vvoo");
    tmps["0053_abab_vvoo"] = declare<T>(MO, "0053_abab_vvoo");
    tmps["0054_abab_vvoo"] = declare<T>(MO, "0054_abab_vvoo");
    tmps["0055_bb_vv"]     = declare<T>(MO, "0055_bb_vv");
    tmps["0056_abab_vooo"] = declare<T>(MO, "0056_abab_vooo");
    tmps["0057_baba_vooo"] = declare<T>(MO, "0057_baba_vooo");
    tmps["0058_abab_vvoo"] = declare<T>(MO, "0058_abab_vvoo");
    tmps["0059_abab_vvoo"] = declare<T>(MO, "0059_abab_vvoo");
    tmps["0060_abab_vvoo"] = declare<T>(MO, "0060_abab_vvoo");
    tmps["0061_abab_vvoo"] = declare<T>(MO, "0061_abab_vvoo");
    tmps["0062_abba_vvoo"] = declare<T>(MO, "0062_abba_vvoo");
    tmps["0063_abab_vvoo"] = declare<T>(MO, "0063_abab_vvoo");
    tmps["0064_abba_vvoo"] = declare<T>(MO, "0064_abba_vvoo");
    tmps["0065_abab_vvoo"] = declare<T>(MO, "0065_abab_vvoo");
    tmps["0066_abab_vvoo"] = declare<T>(MO, "0066_abab_vvoo");
    tmps["0067_abab_vvoo"] = declare<T>(MO, "0067_abab_vvoo");
    tmps["0068_abab_vvoo"] = declare<T>(MO, "0068_abab_vvoo");
    tmps["0069_abab_vvoo"] = declare<T>(MO, "0069_abab_vvoo");
    tmps["0070_abab_vvoo"] = declare<T>(MO, "0070_abab_vvoo");
    tmps["0071_abab_vvoo"] = declare<T>(MO, "0071_abab_vvoo");
    tmps["0072_abab_vvoo"] = declare<T>(MO, "0072_abab_vvoo");
    tmps["0073_abab_vvoo"] = declare<T>(MO, "0073_abab_vvoo");
    tmps["0074_abab_vvoo"] = declare<T>(MO, "0074_abab_vvoo");
    tmps["0075_abab_vvoo"] = declare<T>(MO, "0075_abab_vvoo");
    tmps["0076_abab_vvoo"] = declare<T>(MO, "0076_abab_vvoo");
    tmps["0077_abab_vvoo"] = declare<T>(MO, "0077_abab_vvoo");
    tmps["0078_abab_vvoo"] = declare<T>(MO, "0078_abab_vvoo");
    tmps["0079_bbbb_oovo"] = declare<T>(MO, "0079_bbbb_oovo");
    tmps["0080_abab_vooo"] = declare<T>(MO, "0080_abab_vooo");
    tmps["0081_aaaa_oovo"] = declare<T>(MO, "0081_aaaa_oovo");
    tmps["0082_baba_vooo"] = declare<T>(MO, "0082_baba_vooo");
    tmps["0083_bb_oo"]     = declare<T>(MO, "0083_bb_oo");
    tmps["0084_abab_vvoo"] = declare<T>(MO, "0084_abab_vvoo");
    tmps["0085_baab_vovo"] = declare<T>(MO, "0085_baab_vovo");
    tmps["0086_abab_vvoo"] = declare<T>(MO, "0086_abab_vvoo");
    tmps["0087_aa_oo"]     = declare<T>(MO, "0087_aa_oo");
    tmps["0088_abba_vvoo"] = declare<T>(MO, "0088_abba_vvoo");
    tmps["0089_abab_ovvo"] = declare<T>(MO, "0089_abab_ovvo");
    tmps["0090_abab_vvoo"] = declare<T>(MO, "0090_abab_vvoo");
    tmps["0091_abba_voov"] = declare<T>(MO, "0091_abba_voov");
    tmps["0092_baab_vvoo"] = declare<T>(MO, "0092_baab_vvoo");
    tmps["0093_abba_vvoo"] = declare<T>(MO, "0093_abba_vvoo");
    tmps["0094_bb_vv"]     = declare<T>(MO, "0094_bb_vv");
    tmps["0095_abab_vvoo"] = declare<T>(MO, "0095_abab_vvoo");
    tmps["0096_abab_vovo"] = declare<T>(MO, "0096_abab_vovo");
    tmps["0097_baab_vvoo"] = declare<T>(MO, "0097_baab_vvoo");
    tmps["0098_aaaa_vovo"] = declare<T>(MO, "0098_aaaa_vovo");
    tmps["0099_baba_vvoo"] = declare<T>(MO, "0099_baba_vvoo");
    tmps["0100_aa_vv"]     = declare<T>(MO, "0100_aa_vv");
    tmps["0101_baab_vvoo"] = declare<T>(MO, "0101_baab_vvoo");
    tmps["0102_aa_oo"]     = declare<T>(MO, "0102_aa_oo");
    tmps["0103_abba_vvoo"] = declare<T>(MO, "0103_abba_vvoo");
    tmps["0104_aa_vv"]     = declare<T>(MO, "0104_aa_vv");
    tmps["0105_baab_vvoo"] = declare<T>(MO, "0105_baab_vvoo");
    tmps["0106_abba_vovo"] = declare<T>(MO, "0106_abba_vovo");
    tmps["0107_baba_vvoo"] = declare<T>(MO, "0107_baba_vvoo");
    tmps["0108_abab_oooo"] = declare<T>(MO, "0108_abab_oooo");
    tmps["0109_abab_vvoo"] = declare<T>(MO, "0109_abab_vvoo");
    tmps["0110_aa_vv"]     = declare<T>(MO, "0110_aa_vv");
    tmps["0111_baab_vvoo"] = declare<T>(MO, "0111_baab_vvoo");
    tmps["0112_abab_voov"] = declare<T>(MO, "0112_abab_voov");
    tmps["0113_baba_vvoo"] = declare<T>(MO, "0113_baba_vvoo");
    tmps["0114_bbbb_vovo"] = declare<T>(MO, "0114_bbbb_vovo");
    tmps["0115_abab_vvoo"] = declare<T>(MO, "0115_abab_vvoo");
    tmps["0116_aa_oo"]     = declare<T>(MO, "0116_aa_oo");
    tmps["0117_abba_vvoo"] = declare<T>(MO, "0117_abba_vvoo");
    tmps["0118_bb_vv"]     = declare<T>(MO, "0118_bb_vv");
    tmps["0119_bb_vv"]     = declare<T>(MO, "0119_bb_vv");
    tmps["0120_bb_vv"]     = declare<T>(MO, "0120_bb_vv");
    tmps["0121_abab_vvoo"] = declare<T>(MO, "0121_abab_vvoo");
    tmps["0122_aa_oo"]     = declare<T>(MO, "0122_aa_oo");
    tmps["0123_aa_oo"]     = declare<T>(MO, "0123_aa_oo");
    tmps["0124_aa_oo"]     = declare<T>(MO, "0124_aa_oo");
    tmps["0125_abba_vvoo"] = declare<T>(MO, "0125_abba_vvoo");
    tmps["0126_baba_vovo"] = declare<T>(MO, "0126_baba_vovo");
    tmps["0127_abba_vvoo"] = declare<T>(MO, "0127_abba_vvoo");
    tmps["0128_abab_oooo"] = declare<T>(MO, "0128_abab_oooo");
    tmps["0129_abab_oooo"] = declare<T>(MO, "0129_abab_oooo");
    tmps["0130_abab_oooo"] = declare<T>(MO, "0130_abab_oooo");
    tmps["0131_abab_vvoo"] = declare<T>(MO, "0131_abab_vvoo");
    tmps["0132_aaaa_voov"] = declare<T>(MO, "0132_aaaa_voov");
    tmps["0133_baba_vvoo"] = declare<T>(MO, "0133_baba_vvoo");
    tmps["0134_abab_ovoo"] = declare<T>(MO, "0134_abab_ovoo");
    tmps["0135_abab_ovoo"] = declare<T>(MO, "0135_abab_ovoo");
    tmps["0136_abab_ovoo"] = declare<T>(MO, "0136_abab_ovoo");
    tmps["0137_abab_vvoo"] = declare<T>(MO, "0137_abab_vvoo");
    tmps["0138_bb_vo"]     = declare<T>(MO, "0138_bb_vo");
    tmps["0139_bb_vo"]     = declare<T>(MO, "0139_bb_vo");
    tmps["0140_bb_vo"]     = declare<T>(MO, "0140_bb_vo");
    tmps["0141_bb_vo"]     = declare<T>(MO, "0141_bb_vo");
    tmps["0142_bb_vo"]     = declare<T>(MO, "0142_bb_vo");
    tmps["0143_abab_vvoo"] = declare<T>(MO, "0143_abab_vvoo");
    tmps["0144_abab_vooo"] = declare<T>(MO, "0144_abab_vooo");
    tmps["0145_abab_vooo"] = declare<T>(MO, "0145_abab_vooo");
    tmps["0146_abab_vooo"] = declare<T>(MO, "0146_abab_vooo");
    tmps["0147_abab_vooo"] = declare<T>(MO, "0147_abab_vooo");
    tmps["0148_abab_vooo"] = declare<T>(MO, "0148_abab_vooo");
    tmps["0149_baab_vvoo"] = declare<T>(MO, "0149_baab_vvoo");
    tmps["0150_baba_vooo"] = declare<T>(MO, "0150_baba_vooo");
    tmps["0151_abba_ovoo"] = declare<T>(MO, "0151_abba_ovoo");
    tmps["0152_baab_vooo"] = declare<T>(MO, "0152_baab_vooo");
    tmps["0153_abab_ovoo"] = declare<T>(MO, "0153_abab_ovoo");
    tmps["0154_baab_vooo"] = declare<T>(MO, "0154_baab_vooo");
    tmps["0155_baab_vooo"] = declare<T>(MO, "0155_baab_vooo");
    tmps["0156_abab_vvoo"] = declare<T>(MO, "0156_abab_vvoo");
    tmps["0157_baab_ovoo"] = declare<T>(MO, "0157_baab_ovoo");
    tmps["0158_baab_ovoo"] = declare<T>(MO, "0158_baab_ovoo");
    tmps["0159_baab_ovoo"] = declare<T>(MO, "0159_baab_ovoo");
    tmps["0160_baab_vvoo"] = declare<T>(MO, "0160_baab_vvoo");
    tmps["0161_aa_vv"]     = declare<T>(MO, "0161_aa_vv");
    tmps["0162_baab_vvoo"] = declare<T>(MO, "0162_baab_vvoo");
    tmps["0163_aaaa_voov"] = declare<T>(MO, "0163_aaaa_voov");
    tmps["0164_baba_vvoo"] = declare<T>(MO, "0164_baba_vvoo");
    tmps["0165_baab_vooo"] = declare<T>(MO, "0165_baab_vooo");
    tmps["0166_baab_vooo"] = declare<T>(MO, "0166_baab_vooo");
    tmps["0167_baab_vooo"] = declare<T>(MO, "0167_baab_vooo");
    tmps["0168_baab_vooo"] = declare<T>(MO, "0168_baab_vooo");
    tmps["0169_baab_vooo"] = declare<T>(MO, "0169_baab_vooo");
    tmps["0170_abab_vvoo"] = declare<T>(MO, "0170_abab_vvoo");
    tmps["0171_abab_vvoo"] = declare<T>(MO, "0171_abab_vvoo");
    tmps["0172_aa_vo"]     = declare<T>(MO, "0172_aa_vo");
    tmps["0173_aa_vo"]     = declare<T>(MO, "0173_aa_vo");
    tmps["0174_aa_vo"]     = declare<T>(MO, "0174_aa_vo");
    tmps["0175_aa_vo"]     = declare<T>(MO, "0175_aa_vo");
    tmps["0176_aa_vo"]     = declare<T>(MO, "0176_aa_vo");
    tmps["0177_baba_vvoo"] = declare<T>(MO, "0177_baba_vvoo");
    tmps["0178_bb_oo"]     = declare<T>(MO, "0178_bb_oo");
    tmps["0179_bb_oo"]     = declare<T>(MO, "0179_bb_oo");
    tmps["0180_bb_oo"]     = declare<T>(MO, "0180_bb_oo");
    tmps["0181_bb_oo"]     = declare<T>(MO, "0181_bb_oo");
    tmps["0182_bb_oo"]     = declare<T>(MO, "0182_bb_oo");
    tmps["0183_abab_vvoo"] = declare<T>(MO, "0183_abab_vvoo");
    tmps["0184_abba_vvvo"] = declare<T>(MO, "0184_abba_vvvo");
    tmps["0185_abba_vvoo"] = declare<T>(MO, "0185_abba_vvoo");
    tmps["0186_abba_vooo"] = declare<T>(MO, "0186_abba_vooo");
    tmps["0187_abab_vooo"] = declare<T>(MO, "0187_abab_vooo");
    tmps["0188_abab_vooo"] = declare<T>(MO, "0188_abab_vooo");
    tmps["0189_baab_vvoo"] = declare<T>(MO, "0189_baab_vvoo");
    tmps["0190_baab_vvoo"] = declare<T>(MO, "0190_baab_vvoo");
    tmps["0191_baab_vvoo"] = declare<T>(MO, "0191_baab_vvoo");
    tmps["0192_bb_vo"]     = declare<T>(MO, "0192_bb_vo");
    tmps["0193_abab_vvoo"] = declare<T>(MO, "0193_abab_vvoo");
    tmps["0194_baab_vvoo"] = declare<T>(MO, "0194_baab_vvoo");
    tmps["0195_abab_oovo"] = declare<T>(MO, "0195_abab_oovo");
    tmps["0196_abab_oooo"] = declare<T>(MO, "0196_abab_oooo");
    tmps["0197_abab_vvoo"] = declare<T>(MO, "0197_abab_vvoo");
    tmps["0198_bb_ov"]     = declare<T>(MO, "0198_bb_ov");
    tmps["0199_bb_oo"]     = declare<T>(MO, "0199_bb_oo");
    tmps["0200_bb_oo"]     = declare<T>(MO, "0200_bb_oo");
    tmps["0201_bb_oo"]     = declare<T>(MO, "0201_bb_oo");
    tmps["0202_abab_vvoo"] = declare<T>(MO, "0202_abab_vvoo");
    tmps["0203_abba_vvoo"] = declare<T>(MO, "0203_abba_vvoo");
    tmps["0204_abab_voov"] = declare<T>(MO, "0204_abab_voov");
    tmps["0205_abba_vooo"] = declare<T>(MO, "0205_abba_vooo");
    tmps["0206_abab_vooo"] = declare<T>(MO, "0206_abab_vooo");
    tmps["0207_abab_vooo"] = declare<T>(MO, "0207_abab_vooo");
    tmps["0208_baab_vvoo"] = declare<T>(MO, "0208_baab_vvoo");
    tmps["0209_abab_vooo"] = declare<T>(MO, "0209_abab_vooo");
    tmps["0210_baab_vvoo"] = declare<T>(MO, "0210_baab_vvoo");
    tmps["0211_baab_vooo"] = declare<T>(MO, "0211_baab_vooo");
    tmps["0212_abab_vvoo"] = declare<T>(MO, "0212_abab_vvoo");
    tmps["0213_abba_oovo"] = declare<T>(MO, "0213_abba_oovo");
    tmps["0214_aa_ov"]     = declare<T>(MO, "0214_aa_ov");
    tmps["0215_aa_oo"]     = declare<T>(MO, "0215_aa_oo");
    tmps["0216_aa_oo"]     = declare<T>(MO, "0216_aa_oo");
    tmps["0217_aa_oo"]     = declare<T>(MO, "0217_aa_oo");
    tmps["0218_abba_vvoo"] = declare<T>(MO, "0218_abba_vvoo");
    tmps["0219_abab_ovoo"] = declare<T>(MO, "0219_abab_ovoo");
    tmps["0220_abab_ovoo"] = declare<T>(MO, "0220_abab_ovoo");
    tmps["0221_baab_vooo"] = declare<T>(MO, "0221_baab_vooo");
    tmps["0222_abab_ovoo"] = declare<T>(MO, "0222_abab_ovoo");
    tmps["0223_abab_vvoo"] = declare<T>(MO, "0223_abab_vvoo");
    tmps["0224_aa_vo"]     = declare<T>(MO, "0224_aa_vo");
    tmps["0225_baba_vvoo"] = declare<T>(MO, "0225_baba_vvoo");
    tmps["0226_baab_vvoo"] = declare<T>(MO, "0226_baab_vvoo");
    tmps["0227_aa_vo"]     = declare<T>(MO, "0227_aa_vo");
    tmps["0228_aa_vo"]     = declare<T>(MO, "0228_aa_vo");
    tmps["0229_aa_vo"]     = declare<T>(MO, "0229_aa_vo");
    tmps["0230_aa_vo"]     = declare<T>(MO, "0230_aa_vo");
    tmps["0231_aa_vo"]     = declare<T>(MO, "0231_aa_vo");
    tmps["0232_aa_vo"]     = declare<T>(MO, "0232_aa_vo");
    tmps["0233_aa_vo"]     = declare<T>(MO, "0233_aa_vo");
    tmps["0234_aa_vo"]     = declare<T>(MO, "0234_aa_vo");
    tmps["0235_aa_vo"]     = declare<T>(MO, "0235_aa_vo");
    tmps["0236_aa_vo"]     = declare<T>(MO, "0236_aa_vo");
    tmps["0237_aa_vo"]     = declare<T>(MO, "0237_aa_vo");
    tmps["0238_aa_vo"]     = declare<T>(MO, "0238_aa_vo");
    tmps["0239_aa_vo"]     = declare<T>(MO, "0239_aa_vo");
    tmps["0240_aa_vo"]     = declare<T>(MO, "0240_aa_vo");
    tmps["0241_aa_vo"]     = declare<T>(MO, "0241_aa_vo");
    tmps["0242_aa_vo"]     = declare<T>(MO, "0242_aa_vo");
    tmps["0243_aa_vo"]     = declare<T>(MO, "0243_aa_vo");
    tmps["0244_aa_oo"]     = declare<T>(MO, "0244_aa_oo");
    tmps["0245_aa_vo"]     = declare<T>(MO, "0245_aa_vo");
    tmps["0246_aa_vo"]     = declare<T>(MO, "0246_aa_vo");
    tmps["0247_aa_vo"]     = declare<T>(MO, "0247_aa_vo");
    tmps["0248_aa_vo"]     = declare<T>(MO, "0248_aa_vo");
    tmps["0249_aa_oo"]     = declare<T>(MO, "0249_aa_oo");
    tmps["0250_aa_vo"]     = declare<T>(MO, "0250_aa_vo");
    tmps["0251_aa_vv"]     = declare<T>(MO, "0251_aa_vv");
    tmps["0252_aa_vo"]     = declare<T>(MO, "0252_aa_vo");
    tmps["0253_aa_vo"]     = declare<T>(MO, "0253_aa_vo");
    tmps["0254_aa_vo"]     = declare<T>(MO, "0254_aa_vo");
    tmps["0255_bb_ov"]     = declare<T>(MO, "0255_bb_ov");
    tmps["0256_aa_vo"]     = declare<T>(MO, "0256_aa_vo");
    tmps["0257_abab_voov"] = declare<T>(MO, "0257_abab_voov");
    tmps["0258_aa_vo"]     = declare<T>(MO, "0258_aa_vo");
    tmps["0259_aa_vo"]     = declare<T>(MO, "0259_aa_vo");
    tmps["0260_aa_vo"]     = declare<T>(MO, "0260_aa_vo");
    tmps["0261_aa_vo"]     = declare<T>(MO, "0261_aa_vo");
    tmps["0262_abba_vovo"] = declare<T>(MO, "0262_abba_vovo");
    tmps["0263_aa_vo"]     = declare<T>(MO, "0263_aa_vo");
    tmps["0264_aaaa_voov"] = declare<T>(MO, "0264_aaaa_voov");
    tmps["0265_aa_vo"]     = declare<T>(MO, "0265_aa_vo");
    tmps["0266_aa_vo"]     = declare<T>(MO, "0266_aa_vo");
    tmps["0267_aa_vo"]     = declare<T>(MO, "0267_aa_vo");
    tmps["0268_aa_vo"]     = declare<T>(MO, "0268_aa_vo");
    tmps["0269_aa_oo"]     = declare<T>(MO, "0269_aa_oo");
    tmps["0270_aa_vo"]     = declare<T>(MO, "0270_aa_vo");
    tmps["0271_aa_vv"]     = declare<T>(MO, "0271_aa_vv");
    tmps["0272_aa_vo"]     = declare<T>(MO, "0272_aa_vo");
    tmps["0273_aaaa_voov"] = declare<T>(MO, "0273_aaaa_voov");
    tmps["0274_aa_vo"]     = declare<T>(MO, "0274_aa_vo");
    tmps["0275_aa_vv"]     = declare<T>(MO, "0275_aa_vv");
    tmps["0276_aa_vo"]     = declare<T>(MO, "0276_aa_vo");
    tmps["0277_aa_oo"]     = declare<T>(MO, "0277_aa_oo");
    tmps["0278_aa_vo"]     = declare<T>(MO, "0278_aa_vo");
    tmps["0279_aa_vo"]     = declare<T>(MO, "0279_aa_vo");
    tmps["0280_abab_voov"] = declare<T>(MO, "0280_abab_voov");
    tmps["0281_aa_vo"]     = declare<T>(MO, "0281_aa_vo");
    tmps["0282_aa_oo"]     = declare<T>(MO, "0282_aa_oo");
    tmps["0283_aa_vo"]     = declare<T>(MO, "0283_aa_vo");
    tmps["0284_aa_ov"]     = declare<T>(MO, "0284_aa_ov");
    tmps["0285_aa_vo"]     = declare<T>(MO, "0285_aa_vo");
    tmps["0286_aa_vo"]     = declare<T>(MO, "0286_aa_vo");
    tmps["0287_abab_voov"] = declare<T>(MO, "0287_abab_voov");
    tmps["0288_aa_vo"]     = declare<T>(MO, "0288_aa_vo");
    tmps["0289_aa_vo"]     = declare<T>(MO, "0289_aa_vo");
    tmps["0290_aa_vo"]     = declare<T>(MO, "0290_aa_vo");
    tmps["0291_aa_oo"]     = declare<T>(MO, "0291_aa_oo");
    tmps["0292_aa_oo"]     = declare<T>(MO, "0292_aa_oo");
    tmps["0293_aa_oo"]     = declare<T>(MO, "0293_aa_oo");
    tmps["0294_aa_oo"]     = declare<T>(MO, "0294_aa_oo");
    tmps["0295_aa_oo"]     = declare<T>(MO, "0295_aa_oo");
    tmps["0296_aa_oo"]     = declare<T>(MO, "0296_aa_oo");
    tmps["0297_aa_vo"]     = declare<T>(MO, "0297_aa_vo");
    tmps["0298_aa_vo"]     = declare<T>(MO, "0298_aa_vo");
    tmps["0299_aa_vo"]     = declare<T>(MO, "0299_aa_vo");
    tmps["0300_aa_vo"]     = declare<T>(MO, "0300_aa_vo");
    tmps["0301_aa_oo"]     = declare<T>(MO, "0301_aa_oo");
    tmps["0302_aa_vo"]     = declare<T>(MO, "0302_aa_vo");
    tmps["0303_aa_oo"]     = declare<T>(MO, "0303_aa_oo");
    tmps["0304_aa_vo"]     = declare<T>(MO, "0304_aa_vo");
    tmps["0305_aa_vo"]     = declare<T>(MO, "0305_aa_vo");
    tmps["0306_aa_oo"]     = declare<T>(MO, "0306_aa_oo");
    tmps["0307_aa_oo"]     = declare<T>(MO, "0307_aa_oo");
    tmps["0308_aa_oo"]     = declare<T>(MO, "0308_aa_oo");
    tmps["0309_aa_vo"]     = declare<T>(MO, "0309_aa_vo");
    tmps["0310_aaaa_oovo"] = declare<T>(MO, "0310_aaaa_oovo");
    tmps["0311_aa_oo"]     = declare<T>(MO, "0311_aa_oo");
    tmps["0312_aa_oo"]     = declare<T>(MO, "0312_aa_oo");
    tmps["0313_aa_oo"]     = declare<T>(MO, "0313_aa_oo");
    tmps["0314_aa_vo"]     = declare<T>(MO, "0314_aa_vo");
    tmps["0315_aa_ov"]     = declare<T>(MO, "0315_aa_ov");
    tmps["0316_aa_oo"]     = declare<T>(MO, "0316_aa_oo");
    tmps["0317_aa_oo"]     = declare<T>(MO, "0317_aa_oo");
    tmps["0318_aa_oo"]     = declare<T>(MO, "0318_aa_oo");
    tmps["0319_aa_oo"]     = declare<T>(MO, "0319_aa_oo");
    tmps["0320_aa_oo"]     = declare<T>(MO, "0320_aa_oo");
    tmps["0321_aa_vo"]     = declare<T>(MO, "0321_aa_vo");
    tmps["0322_aa_vo"]     = declare<T>(MO, "0322_aa_vo");
    tmps["0323_aa_vo"]     = declare<T>(MO, "0323_aa_vo");
    tmps["0324_aa_vo"]     = declare<T>(MO, "0324_aa_vo");
    tmps["0325_aa_vo"]     = declare<T>(MO, "0325_aa_vo");
    tmps["0326_aa_vo"]     = declare<T>(MO, "0326_aa_vo");
    tmps["0327_aa_vo"]     = declare<T>(MO, "0327_aa_vo");
    tmps["0328_aa_vo"]     = declare<T>(MO, "0328_aa_vo");
    tmps["0329_aa_vo"]     = declare<T>(MO, "0329_aa_vo");
    tmps["0330_aa_vo"]     = declare<T>(MO, "0330_aa_vo");
    tmps["0331_aa_vo"]     = declare<T>(MO, "0331_aa_vo");
    tmps["0332_aa_vo"]     = declare<T>(MO, "0332_aa_vo");
    tmps["0333_aa_vo"]     = declare<T>(MO, "0333_aa_vo");
    tmps["0334_aa_vo"]     = declare<T>(MO, "0334_aa_vo");
    tmps["0335_aa_vo"]     = declare<T>(MO, "0335_aa_vo");
    tmps["0336_aa_vo"]     = declare<T>(MO, "0336_aa_vo");
    tmps["0337_aa_vo"]     = declare<T>(MO, "0337_aa_vo");
    tmps["0338_aa_vo"]     = declare<T>(MO, "0338_aa_vo");
    tmps["0339_aa_vo"]     = declare<T>(MO, "0339_aa_vo");
    tmps["0340_aa_vo"]     = declare<T>(MO, "0340_aa_vo");
    tmps["0341_aa_vo"]     = declare<T>(MO, "0341_aa_vo");
    tmps["0342_aa_vo"]     = declare<T>(MO, "0342_aa_vo");
    tmps["0343_aa_vo"]     = declare<T>(MO, "0343_aa_vo");
    tmps["0344_aa_vo"]     = declare<T>(MO, "0344_aa_vo");
    tmps["0345_aa_oo"]     = declare<T>(MO, "0345_aa_oo");
    tmps["0346_aa_vo"]     = declare<T>(MO, "0346_aa_vo");
    tmps["0347_aa_vo"]     = declare<T>(MO, "0347_aa_vo");
    tmps["0348_aa_vo"]     = declare<T>(MO, "0348_aa_vo");
    tmps["0349_aa_vo"]     = declare<T>(MO, "0349_aa_vo");
    tmps["0350_aa_vo"]     = declare<T>(MO, "0350_aa_vo");
    tmps["0351_aa_vo"]     = declare<T>(MO, "0351_aa_vo");
    tmps["0352_aa_vo"]     = declare<T>(MO, "0352_aa_vo");
    tmps["0353_aa_vo"]     = declare<T>(MO, "0353_aa_vo");
    tmps["0354_aa_vo"]     = declare<T>(MO, "0354_aa_vo");
    tmps["0355_aa_vo"]     = declare<T>(MO, "0355_aa_vo");
    tmps["0356_aa_vo"]     = declare<T>(MO, "0356_aa_vo");
    tmps["0357_aa_vo"]     = declare<T>(MO, "0357_aa_vo");
    tmps["0358_aa_vo"]     = declare<T>(MO, "0358_aa_vo");
    tmps["0359_aa_vo"]     = declare<T>(MO, "0359_aa_vo");
    tmps["0360_aa_vo"]     = declare<T>(MO, "0360_aa_vo");
    tmps["0361_aa_vo"]     = declare<T>(MO, "0361_aa_vo");
    tmps["0362_aa_vo"]     = declare<T>(MO, "0362_aa_vo");
    tmps["0363_aa_vo"]     = declare<T>(MO, "0363_aa_vo");
    tmps["0364_aa_vo"]     = declare<T>(MO, "0364_aa_vo");
    tmps["0365_aa_vo"]     = declare<T>(MO, "0365_aa_vo");
    tmps["0366_aa_oo"]     = declare<T>(MO, "0366_aa_oo");
    tmps["0367_aa_vo"]     = declare<T>(MO, "0367_aa_vo");
    tmps["0368_aa_vo"]     = declare<T>(MO, "0368_aa_vo");
    tmps["0369_aa_vo"]     = declare<T>(MO, "0369_aa_vo");
    tmps["0370_aa_vo"]     = declare<T>(MO, "0370_aa_vo");
    tmps["0371_aa_oo"]     = declare<T>(MO, "0371_aa_oo");
    tmps["0372_aa_vo"]     = declare<T>(MO, "0372_aa_vo");
    tmps["0373_aa_vo"]     = declare<T>(MO, "0373_aa_vo");
    tmps["0374_aa_vo"]     = declare<T>(MO, "0374_aa_vo");
    tmps["0375_aa_vo"]     = declare<T>(MO, "0375_aa_vo");
    tmps["0376_aa_vo"]     = declare<T>(MO, "0376_aa_vo");
    tmps["0377_aa_vo"]     = declare<T>(MO, "0377_aa_vo");
    tmps["0378_aa_vo"]     = declare<T>(MO, "0378_aa_vo");
    tmps["0379_aa_vo"]     = declare<T>(MO, "0379_aa_vo");
    tmps["0380_aa_vo"]     = declare<T>(MO, "0380_aa_vo");
    tmps["0381_aa_vo"]     = declare<T>(MO, "0381_aa_vo");
    tmps["0382_aa_vo"]     = declare<T>(MO, "0382_aa_vo");
    tmps["0383_aa_vo"]     = declare<T>(MO, "0383_aa_vo");
    tmps["0384_aa_vo"]     = declare<T>(MO, "0384_aa_vo");
    tmps["0385_aa_vo"]     = declare<T>(MO, "0385_aa_vo");
    tmps["0386_aa_vo"]     = declare<T>(MO, "0386_aa_vo");
    tmps["0387_aa_vo"]     = declare<T>(MO, "0387_aa_vo");
    tmps["0388_aa_vo"]     = declare<T>(MO, "0388_aa_vo");
    tmps["0389_aa_vo"]     = declare<T>(MO, "0389_aa_vo");
    tmps["0390_aa_vo"]     = declare<T>(MO, "0390_aa_vo");
    tmps["0391_aa_vo"]     = declare<T>(MO, "0391_aa_vo");
    tmps["0392_aa_vo"]     = declare<T>(MO, "0392_aa_vo");
    tmps["0393_aa_vo"]     = declare<T>(MO, "0393_aa_vo");
    tmps["0394_aa_vo"]     = declare<T>(MO, "0394_aa_vo");
    tmps["0395_aa_vo"]     = declare<T>(MO, "0395_aa_vo");
    tmps["0396_aa_vo"]     = declare<T>(MO, "0396_aa_vo");
    tmps["0397_aa_vo"]     = declare<T>(MO, "0397_aa_vo");
    tmps["0398_aa_vo"]     = declare<T>(MO, "0398_aa_vo");
    tmps["0399_aa_vo"]     = declare<T>(MO, "0399_aa_vo");
    tmps["0400_baba_vvoo"] = declare<T>(MO, "0400_baba_vvoo");
    tmps["0401_abab_vvoo"] = declare<T>(MO, "0401_abab_vvoo");
    tmps["0402_baba_vvoo"] = declare<T>(MO, "0402_baba_vvoo");
    tmps["0403_abab_vvoo"] = declare<T>(MO, "0403_abab_vvoo");
    tmps["0404_abab_vvoo"] = declare<T>(MO, "0404_abab_vvoo");
    tmps["0405_abab_vvoo"] = declare<T>(MO, "0405_abab_vvoo");
    tmps["0406_abab_vvoo"] = declare<T>(MO, "0406_abab_vvoo");
    tmps["0407_abab_vvoo"] = declare<T>(MO, "0407_abab_vvoo");
    tmps["0408_abab_vvoo"] = declare<T>(MO, "0408_abab_vvoo");
    tmps["0409_abab_vvoo"] = declare<T>(MO, "0409_abab_vvoo");
    tmps["0410_abab_vvoo"] = declare<T>(MO, "0410_abab_vvoo");
    tmps["0411_abab_vvoo"] = declare<T>(MO, "0411_abab_vvoo");
    tmps["0412_abba_vvoo"] = declare<T>(MO, "0412_abba_vvoo");
    tmps["0413_abab_vvoo"] = declare<T>(MO, "0413_abab_vvoo");
    tmps["0414_abab_vvoo"] = declare<T>(MO, "0414_abab_vvoo");
    tmps["0415_abab_vvoo"] = declare<T>(MO, "0415_abab_vvoo");
    tmps["0416_abab_vvoo"] = declare<T>(MO, "0416_abab_vvoo");
    tmps["0417_abab_vvoo"] = declare<T>(MO, "0417_abab_vvoo");
    tmps["0418_abab_vvoo"] = declare<T>(MO, "0418_abab_vvoo");
    tmps["0419_abab_vvoo"] = declare<T>(MO, "0419_abab_vvoo");
    tmps["0420_abba_vvoo"] = declare<T>(MO, "0420_abba_vvoo");
    tmps["0421_abab_vvoo"] = declare<T>(MO, "0421_abab_vvoo");
    tmps["0422_abab_vvoo"] = declare<T>(MO, "0422_abab_vvoo");
    tmps["0423_abab_vvoo"] = declare<T>(MO, "0423_abab_vvoo");
    tmps["0424_baba_vvoo"] = declare<T>(MO, "0424_baba_vvoo");
    tmps["0425_baab_vvoo"] = declare<T>(MO, "0425_baab_vvoo");
    tmps["0426_abab_vvoo"] = declare<T>(MO, "0426_abab_vvoo");
    tmps["0427_baba_vvoo"] = declare<T>(MO, "0427_baba_vvoo");
    tmps["0428_abab_oooo"] = declare<T>(MO, "0428_abab_oooo");
    tmps["0429_abab_oooo"] = declare<T>(MO, "0429_abab_oooo");
    tmps["0430_abab_oooo"] = declare<T>(MO, "0430_abab_oooo");
    tmps["0431_abab_oooo"] = declare<T>(MO, "0431_abab_oooo");
    tmps["0432_aa_vo"]     = declare<T>(MO, "0432_aa_vo");
    tmps["0433_aa_vo"]     = declare<T>(MO, "0433_aa_vo");
    tmps["0434_aa_vo"]     = declare<T>(MO, "0434_aa_vo");
    tmps["0435_aa_vo"]     = declare<T>(MO, "0435_aa_vo");
    tmps["0436_aa_vo"]     = declare<T>(MO, "0436_aa_vo");
    tmps["0437_bb_oo"]     = declare<T>(MO, "0437_bb_oo");
    tmps["0438_bb_oo"]     = declare<T>(MO, "0438_bb_oo");
    tmps["0439_bb_oo"]     = declare<T>(MO, "0439_bb_oo");
    tmps["0440_bb_oo"]     = declare<T>(MO, "0440_bb_oo");
    tmps["0441_bb_oo"]     = declare<T>(MO, "0441_bb_oo");
    tmps["0442_bb_oo"]     = declare<T>(MO, "0442_bb_oo");
    tmps["0443_bb_oo"]     = declare<T>(MO, "0443_bb_oo");
    tmps["0444_bb_oo"]     = declare<T>(MO, "0444_bb_oo");
    tmps["0445_bb_oo"]     = declare<T>(MO, "0445_bb_oo");
    tmps["0446_aa_ov"]     = declare<T>(MO, "0446_aa_ov");
    tmps["0447_aa_ov"]     = declare<T>(MO, "0447_aa_ov");
    tmps["0448_aa_ov"]     = declare<T>(MO, "0448_aa_ov");
    tmps["0449_bb_vo"]     = declare<T>(MO, "0449_bb_vo");
    tmps["0450_bb_vo"]     = declare<T>(MO, "0450_bb_vo");
    tmps["0451_bb_vo"]     = declare<T>(MO, "0451_bb_vo");
    tmps["0452_bb_vo"]     = declare<T>(MO, "0452_bb_vo");
    tmps["0453_bb_vo"]     = declare<T>(MO, "0453_bb_vo");
    tmps["0454_bb_ov"]     = declare<T>(MO, "0454_bb_ov");
    tmps["0455_bb_ov"]     = declare<T>(MO, "0455_bb_ov");
    tmps["0456_bb_ov"]     = declare<T>(MO, "0456_bb_ov");
    tmps["0457_aa_vv"]     = declare<T>(MO, "0457_aa_vv");
    tmps["0458_aa_vv"]     = declare<T>(MO, "0458_aa_vv");
    tmps["0459_aa_vv"]     = declare<T>(MO, "0459_aa_vv");
    tmps["0460_aa_vv"]     = declare<T>(MO, "0460_aa_vv");
    tmps["0461_aa_vv"]     = declare<T>(MO, "0461_aa_vv");
    tmps["0462_bb_vv"]     = declare<T>(MO, "0462_bb_vv");
    tmps["0463_bb_vv"]     = declare<T>(MO, "0463_bb_vv");
    tmps["0464_bb_vv"]     = declare<T>(MO, "0464_bb_vv");
    tmps["0465_bb_vv"]     = declare<T>(MO, "0465_bb_vv");
    tmps["0466_bb_vv"]     = declare<T>(MO, "0466_bb_vv");
    tmps["0467_abab_vooo"] = declare<T>(MO, "0467_abab_vooo");
    tmps["0468_abab_vooo"] = declare<T>(MO, "0468_abab_vooo");
    tmps["0469_abab_vooo"] = declare<T>(MO, "0469_abab_vooo");
    tmps["0470_abab_vooo"] = declare<T>(MO, "0470_abab_vooo");
    tmps["0471_abab_vooo"] = declare<T>(MO, "0471_abab_vooo");
    tmps["0472_abba_vooo"] = declare<T>(MO, "0472_abba_vooo");
    tmps["0473_abab_vooo"] = declare<T>(MO, "0473_abab_vooo");
    tmps["0474_baba_ovoo"] = declare<T>(MO, "0474_baba_ovoo");
    tmps["0475_abab_vooo"] = declare<T>(MO, "0475_abab_vooo");
    tmps["0476_baab_vooo"] = declare<T>(MO, "0476_baab_vooo");
    tmps["0477_baba_vooo"] = declare<T>(MO, "0477_baba_vooo");
    tmps["0478_baab_vooo"] = declare<T>(MO, "0478_baab_vooo");
    tmps["0479_baab_vooo"] = declare<T>(MO, "0479_baab_vooo");
    tmps["0480_baab_vooo"] = declare<T>(MO, "0480_baab_vooo");
    tmps["0481_baab_vooo"] = declare<T>(MO, "0481_baab_vooo");
    tmps["0482_abab_ovoo"] = declare<T>(MO, "0482_abab_ovoo");
    tmps["0483_baba_vooo"] = declare<T>(MO, "0483_baba_vooo");
    tmps["0484_baab_vooo"] = declare<T>(MO, "0484_baab_vooo");
    tmps["0485_abab_ovoo"] = declare<T>(MO, "0485_abab_ovoo");
    tmps["0486_aaaa_voov"] = declare<T>(MO, "0486_aaaa_voov");
    tmps["0487_aaaa_vovo"] = declare<T>(MO, "0487_aaaa_vovo");
    tmps["0488_aaaa_voov"] = declare<T>(MO, "0488_aaaa_voov");
    tmps["0489_aaaa_voov"] = declare<T>(MO, "0489_aaaa_voov");
    tmps["0490_abba_oooo"] = declare<T>(MO, "0490_abba_oooo");
    tmps["0491_abab_oovo"] = declare<T>(MO, "0491_abab_oovo");
    tmps["0492_abab_oooo"] = declare<T>(MO, "0492_abab_oooo");
    tmps["0493_abab_oooo"] = declare<T>(MO, "0493_abab_oooo");
    tmps["0494_aa_vo"]     = declare<T>(MO, "0494_aa_vo");
    tmps["0495_aa_vo"]     = declare<T>(MO, "0495_aa_vo");
    tmps["0496_aa_vo"]     = declare<T>(MO, "0496_aa_vo");
    tmps["0497_bb_oo"]     = declare<T>(MO, "0497_bb_oo");
    tmps["0498_bb_oo"]     = declare<T>(MO, "0498_bb_oo");
    tmps["0499_bb_oo"]     = declare<T>(MO, "0499_bb_oo");
    tmps["0500_bb_oo"]     = declare<T>(MO, "0500_bb_oo");
    tmps["0501_bb_oo"]     = declare<T>(MO, "0501_bb_oo");
    tmps["0502_bb_oo"]     = declare<T>(MO, "0502_bb_oo");
    tmps["0503_bb_ov"]     = declare<T>(MO, "0503_bb_ov");
    tmps["0504_bb_oo"]     = declare<T>(MO, "0504_bb_oo");
    tmps["0505_bb_vo"]     = declare<T>(MO, "0505_bb_vo");
    tmps["0506_bb_vo"]     = declare<T>(MO, "0506_bb_vo");
    tmps["0507_bb_vo"]     = declare<T>(MO, "0507_bb_vo");
    tmps["0508_abba_vooo"] = declare<T>(MO, "0508_abba_vooo");
    tmps["0509_bb_ov"]     = declare<T>(MO, "0509_bb_ov");
    tmps["0510_abab_vooo"] = declare<T>(MO, "0510_abab_vooo");
    tmps["0511_abba_vooo"] = declare<T>(MO, "0511_abba_vooo");
    tmps["0512_bb_ov"]     = declare<T>(MO, "0512_bb_ov");
    tmps["0513_abab_vooo"] = declare<T>(MO, "0513_abab_vooo");
    tmps["0514_abba_vooo"] = declare<T>(MO, "0514_abba_vooo");
    tmps["0515_abba_vooo"] = declare<T>(MO, "0515_abba_vooo");
    tmps["0516_abba_voov"] = declare<T>(MO, "0516_abba_voov");
    tmps["0517_abab_vooo"] = declare<T>(MO, "0517_abab_vooo");
    tmps["0518_abba_vooo"] = declare<T>(MO, "0518_abba_vooo");
    tmps["0519_abab_vovo"] = declare<T>(MO, "0519_abab_vovo");
    tmps["0520_abab_vooo"] = declare<T>(MO, "0520_abab_vooo");
    tmps["0521_abab_vooo"] = declare<T>(MO, "0521_abab_vooo");
    tmps["0522_abab_vooo"] = declare<T>(MO, "0522_abab_vooo");
    tmps["0523_abab_vooo"] = declare<T>(MO, "0523_abab_vooo");
    tmps["0524_abba_vooo"] = declare<T>(MO, "0524_abba_vooo");
    tmps["0525_abab_vooo"] = declare<T>(MO, "0525_abab_vooo");
    tmps["0526_abab_vooo"] = declare<T>(MO, "0526_abab_vooo");
    tmps["0527_abba_voov"] = declare<T>(MO, "0527_abba_voov");
    tmps["0528_abab_vooo"] = declare<T>(MO, "0528_abab_vooo");
    tmps["0529_abab_vooo"] = declare<T>(MO, "0529_abab_vooo");
    tmps["0530_abab_vooo"] = declare<T>(MO, "0530_abab_vooo");
    tmps["0531_abab_oooo"] = declare<T>(MO, "0531_abab_oooo");
    tmps["0532_baab_vooo"] = declare<T>(MO, "0532_baab_vooo");
    tmps["0533_aaaa_oovo"] = declare<T>(MO, "0533_aaaa_oovo");
    tmps["0534_baba_vooo"] = declare<T>(MO, "0534_baba_vooo");
    tmps["0535_baab_vooo"] = declare<T>(MO, "0535_baab_vooo");
    tmps["0536_aaaa_oovo"] = declare<T>(MO, "0536_aaaa_oovo");
    tmps["0537_baba_vooo"] = declare<T>(MO, "0537_baba_vooo");
    tmps["0538_baba_vooo"] = declare<T>(MO, "0538_baba_vooo");
    tmps["0539_baab_vooo"] = declare<T>(MO, "0539_baab_vooo");
    tmps["0540_baab_vooo"] = declare<T>(MO, "0540_baab_vooo");
    tmps["0541_baab_vovo"] = declare<T>(MO, "0541_baab_vovo");
    tmps["0542_baab_vooo"] = declare<T>(MO, "0542_baab_vooo");
    tmps["0543_baab_vooo"] = declare<T>(MO, "0543_baab_vooo");
    tmps["0544_abab_ovoo"] = declare<T>(MO, "0544_abab_ovoo");
    tmps["0545_aa_ov"]     = declare<T>(MO, "0545_aa_ov");
    tmps["0546_baab_vooo"] = declare<T>(MO, "0546_baab_vooo");
    tmps["0547_abab_oooo"] = declare<T>(MO, "0547_abab_oooo");
    tmps["0548_abab_oooo"] = declare<T>(MO, "0548_abab_oooo");
    tmps["0549_abab_oooo"] = declare<T>(MO, "0549_abab_oooo");
    tmps["0550_baab_vooo"] = declare<T>(MO, "0550_baab_vooo");
    tmps["0551_aa_ov"]     = declare<T>(MO, "0551_aa_ov");
    tmps["0552_baab_vooo"] = declare<T>(MO, "0552_baab_vooo");
    tmps["0553_baab_vooo"] = declare<T>(MO, "0553_baab_vooo");
    tmps["0554_baba_vooo"] = declare<T>(MO, "0554_baba_vooo");
    tmps["0555_baab_vooo"] = declare<T>(MO, "0555_baab_vooo");
    tmps["0556_baab_vooo"] = declare<T>(MO, "0556_baab_vooo");
    tmps["0557_baba_vooo"] = declare<T>(MO, "0557_baba_vooo");
    tmps["0558_baab_vooo"] = declare<T>(MO, "0558_baab_vooo");
    tmps["0559_baab_vooo"] = declare<T>(MO, "0559_baab_vooo");
    tmps["0560_baab_vooo"] = declare<T>(MO, "0560_baab_vooo");
    tmps["0561_abab_oovo"] = declare<T>(MO, "0561_abab_oovo");
    tmps["0562_baab_vooo"] = declare<T>(MO, "0562_baab_vooo");
    tmps["0563_baba_vooo"] = declare<T>(MO, "0563_baba_vooo");
    tmps["0564_bb_vv"]     = declare<T>(MO, "0564_bb_vv");
    tmps["0565_abab_vvoo"] = declare<T>(MO, "0565_abab_vvoo");
    tmps["0566_abba_vvoo"] = declare<T>(MO, "0566_abba_vvoo");
    tmps["0567_abab_vvoo"] = declare<T>(MO, "0567_abab_vvoo");
    tmps["0568_baba_vvoo"] = declare<T>(MO, "0568_baba_vvoo");
    tmps["0569_bb_vv"]     = declare<T>(MO, "0569_bb_vv");
    tmps["0570_abab_vvoo"] = declare<T>(MO, "0570_abab_vvoo");
    tmps["0571_abab_vvoo"] = declare<T>(MO, "0571_abab_vvoo");
    tmps["0572_baab_vvoo"] = declare<T>(MO, "0572_baab_vvoo");
    tmps["0573_abab_vovo"] = declare<T>(MO, "0573_abab_vovo");
    tmps["0574_baab_vvoo"] = declare<T>(MO, "0574_baab_vvoo");
    tmps["0575_baba_vvoo"] = declare<T>(MO, "0575_baba_vvoo");
    tmps["0576_baab_vvoo"] = declare<T>(MO, "0576_baab_vvoo");
    tmps["0577_bb_oo"]     = declare<T>(MO, "0577_bb_oo");
    tmps["0578_bb_oo"]     = declare<T>(MO, "0578_bb_oo");
    tmps["0579_bb_oo"]     = declare<T>(MO, "0579_bb_oo");
    tmps["0580_abab_vvoo"] = declare<T>(MO, "0580_abab_vvoo");
    tmps["0581_baba_vvoo"] = declare<T>(MO, "0581_baba_vvoo");
    tmps["0582_baab_vvoo"] = declare<T>(MO, "0582_baab_vvoo");
    tmps["0583_abab_vvoo"] = declare<T>(MO, "0583_abab_vvoo");
    tmps["0584_baab_vvoo"] = declare<T>(MO, "0584_baab_vvoo");
    tmps["0585_abba_vvoo"] = declare<T>(MO, "0585_abba_vvoo");
    tmps["0586_abba_vvoo"] = declare<T>(MO, "0586_abba_vvoo");
    tmps["0587_abba_vvvo"] = declare<T>(MO, "0587_abba_vvvo");
    tmps["0588_abba_vvoo"] = declare<T>(MO, "0588_abba_vvoo");
    tmps["0589_baba_vvoo"] = declare<T>(MO, "0589_baba_vvoo");
    tmps["0590_abab_vvoo"] = declare<T>(MO, "0590_abab_vvoo");
    tmps["0591_abab_vvoo"] = declare<T>(MO, "0591_abab_vvoo");
    tmps["0592_baab_vvoo"] = declare<T>(MO, "0592_baab_vvoo");
    tmps["0593_abba_vvoo"] = declare<T>(MO, "0593_abba_vvoo");
    tmps["0594_baba_vvoo"] = declare<T>(MO, "0594_baba_vvoo");
    tmps["0595_bb_vv"]     = declare<T>(MO, "0595_bb_vv");
    tmps["0596_bb_vv"]     = declare<T>(MO, "0596_bb_vv");
    tmps["0597_bb_vv"]     = declare<T>(MO, "0597_bb_vv");
    tmps["0598_abab_vvoo"] = declare<T>(MO, "0598_abab_vvoo");
    tmps["0599_baba_vvoo"] = declare<T>(MO, "0599_baba_vvoo");
    tmps["0600_baab_ovoo"] = declare<T>(MO, "0600_baab_ovoo");
    tmps["0601_baab_vvoo"] = declare<T>(MO, "0601_baab_vvoo");
    tmps["0602_aaaa_vovo"] = declare<T>(MO, "0602_aaaa_vovo");
    tmps["0603_baba_vvoo"] = declare<T>(MO, "0603_baba_vvoo");
    tmps["0604_abba_vvoo"] = declare<T>(MO, "0604_abba_vvoo");
    tmps["0605_baab_vvoo"] = declare<T>(MO, "0605_baab_vvoo");
    tmps["0606_baba_vvoo"] = declare<T>(MO, "0606_baba_vvoo");
    tmps["0607_baab_vvoo"] = declare<T>(MO, "0607_baab_vvoo");
    tmps["0608_baba_vvoo"] = declare<T>(MO, "0608_baba_vvoo");
    tmps["0609_bb_vo"]     = declare<T>(MO, "0609_bb_vo");
    tmps["0610_bb_vo"]     = declare<T>(MO, "0610_bb_vo");
    tmps["0611_bb_vo"]     = declare<T>(MO, "0611_bb_vo");
    tmps["0612_bb_vo"]     = declare<T>(MO, "0612_bb_vo");
    tmps["0613_bb_vo"]     = declare<T>(MO, "0613_bb_vo");
    tmps["0614_abab_vvoo"] = declare<T>(MO, "0614_abab_vvoo");
    tmps["0615_abba_vvoo"] = declare<T>(MO, "0615_abba_vvoo");
    tmps["0616_abab_vvoo"] = declare<T>(MO, "0616_abab_vvoo");
    tmps["0617_baba_vvoo"] = declare<T>(MO, "0617_baba_vvoo");
    tmps["0618_abab_vvoo"] = declare<T>(MO, "0618_abab_vvoo");
    tmps["0619_abab_vvoo"] = declare<T>(MO, "0619_abab_vvoo");
    tmps["0620_bb_vv"]     = declare<T>(MO, "0620_bb_vv");
    tmps["0621_abab_vvoo"] = declare<T>(MO, "0621_abab_vvoo");
    tmps["0622_abab_vvoo"] = declare<T>(MO, "0622_abab_vvoo");
    tmps["0623_baba_ovoo"] = declare<T>(MO, "0623_baba_ovoo");
    tmps["0624_abab_vooo"] = declare<T>(MO, "0624_abab_vooo");
    tmps["0625_baba_ovoo"] = declare<T>(MO, "0625_baba_ovoo");
    tmps["0626_baba_vvoo"] = declare<T>(MO, "0626_baba_vvoo");
    tmps["0627_aa_vo"]     = declare<T>(MO, "0627_aa_vo");
    tmps["0628_aa_vo"]     = declare<T>(MO, "0628_aa_vo");
    tmps["0629_aa_vo"]     = declare<T>(MO, "0629_aa_vo");
    tmps["0630_aa_vo"]     = declare<T>(MO, "0630_aa_vo");
    tmps["0631_aa_vo"]     = declare<T>(MO, "0631_aa_vo");
    tmps["0632_baba_vvoo"] = declare<T>(MO, "0632_baba_vvoo");
    tmps["0633_baab_vvoo"] = declare<T>(MO, "0633_baab_vvoo");
    tmps["0634_abab_vvoo"] = declare<T>(MO, "0634_abab_vvoo");
    tmps["0635_abab_ovoo"] = declare<T>(MO, "0635_abab_ovoo");
    tmps["0636_baba_vooo"] = declare<T>(MO, "0636_baba_vooo");
    tmps["0637_abab_ovoo"] = declare<T>(MO, "0637_abab_ovoo");
    tmps["0638_abab_vvoo"] = declare<T>(MO, "0638_abab_vvoo");
    tmps["0639_baab_vvoo"] = declare<T>(MO, "0639_baab_vvoo");
    tmps["0640_bb_oo"]     = declare<T>(MO, "0640_bb_oo");
    tmps["0641_abab_vvoo"] = declare<T>(MO, "0641_abab_vvoo");
    tmps["0642_abba_ovoo"] = declare<T>(MO, "0642_abba_ovoo");
    tmps["0643_abab_ovoo"] = declare<T>(MO, "0643_abab_ovoo");
    tmps["0644_abab_ovoo"] = declare<T>(MO, "0644_abab_ovoo");
    tmps["0645_abab_ovoo"] = declare<T>(MO, "0645_abab_ovoo");
    tmps["0646_abab_vvoo"] = declare<T>(MO, "0646_abab_vvoo");
    tmps["0647_baab_vvoo"] = declare<T>(MO, "0647_baab_vvoo");
    tmps["0648_abab_ovoo"] = declare<T>(MO, "0648_abab_ovoo");
    tmps["0649_abab_ovoo"] = declare<T>(MO, "0649_abab_ovoo");
    tmps["0650_abab_ovoo"] = declare<T>(MO, "0650_abab_ovoo");
    tmps["0651_abab_vvoo"] = declare<T>(MO, "0651_abab_vvoo");
    tmps["0652_aa_vv"]     = declare<T>(MO, "0652_aa_vv");
    tmps["0653_baab_vvoo"] = declare<T>(MO, "0653_baab_vvoo");
    tmps["0654_abab_vvoo"] = declare<T>(MO, "0654_abab_vvoo");
    tmps["0655_abab_vvoo"] = declare<T>(MO, "0655_abab_vvoo");
    tmps["0656_abab_vvoo"] = declare<T>(MO, "0656_abab_vvoo");
    tmps["0657_abab_vvoo"] = declare<T>(MO, "0657_abab_vvoo");
    tmps["0658_aa_vv"]     = declare<T>(MO, "0658_aa_vv");
    tmps["0659_baab_vvoo"] = declare<T>(MO, "0659_baab_vvoo");
    tmps["0660_baab_vvoo"] = declare<T>(MO, "0660_baab_vvoo");
    tmps["0661_baab_vvoo"] = declare<T>(MO, "0661_baab_vvoo");
    tmps["0662_baba_vvoo"] = declare<T>(MO, "0662_baba_vvoo");
    tmps["0663_abba_vvoo"] = declare<T>(MO, "0663_abba_vvoo");
    tmps["0664_baab_vvoo"] = declare<T>(MO, "0664_baab_vvoo");
    tmps["0665_abba_vvoo"] = declare<T>(MO, "0665_abba_vvoo");
    tmps["0666_baab_vvoo"] = declare<T>(MO, "0666_baab_vvoo");
    tmps["0667_abab_vvoo"] = declare<T>(MO, "0667_abab_vvoo");
    tmps["0668_baab_vvoo"] = declare<T>(MO, "0668_baab_vvoo");
    tmps["0669_abab_vvoo"] = declare<T>(MO, "0669_abab_vvoo");
    tmps["0670_abab_vvoo"] = declare<T>(MO, "0670_abab_vvoo");
    tmps["0671_baba_vovo"] = declare<T>(MO, "0671_baba_vovo");
    tmps["0672_abba_vvoo"] = declare<T>(MO, "0672_abba_vvoo");
    tmps["0673_abba_vvoo"] = declare<T>(MO, "0673_abba_vvoo");
    tmps["0674_abab_vvoo"] = declare<T>(MO, "0674_abab_vvoo");
    tmps["0675_abab_vvoo"] = declare<T>(MO, "0675_abab_vvoo");
    tmps["0676_abba_vvoo"] = declare<T>(MO, "0676_abba_vvoo");
    tmps["0677_aa_oo"]     = declare<T>(MO, "0677_aa_oo");
    tmps["0678_abba_vvoo"] = declare<T>(MO, "0678_abba_vvoo");
    tmps["0679_abab_vvoo"] = declare<T>(MO, "0679_abab_vvoo");
    tmps["0680_abab_vvoo"] = declare<T>(MO, "0680_abab_vvoo");
    tmps["0681_baab_vvoo"] = declare<T>(MO, "0681_baab_vvoo");
    tmps["0682_aaaa_voov"] = declare<T>(MO, "0682_aaaa_voov");
    tmps["0683_baba_vvoo"] = declare<T>(MO, "0683_baba_vvoo");
    tmps["0684_abba_vvoo"] = declare<T>(MO, "0684_abba_vvoo");
    tmps["0685_abab_vooo"] = declare<T>(MO, "0685_abab_vooo");
    tmps["0686_abab_vooo"] = declare<T>(MO, "0686_abab_vooo");
    tmps["0687_abba_vooo"] = declare<T>(MO, "0687_abba_vooo");
    tmps["0688_abab_vooo"] = declare<T>(MO, "0688_abab_vooo");
    tmps["0689_abab_vooo"] = declare<T>(MO, "0689_abab_vooo");
    tmps["0690_abab_vooo"] = declare<T>(MO, "0690_abab_vooo");
    tmps["0691_abab_vooo"] = declare<T>(MO, "0691_abab_vooo");
    tmps["0692_abab_vooo"] = declare<T>(MO, "0692_abab_vooo");
    tmps["0693_baab_vvoo"] = declare<T>(MO, "0693_baab_vvoo");
    tmps["0694_abab_vvoo"] = declare<T>(MO, "0694_abab_vvoo");
    tmps["0695_abab_vvoo"] = declare<T>(MO, "0695_abab_vvoo");
    tmps["0696_baab_vooo"] = declare<T>(MO, "0696_baab_vooo");
    tmps["0697_abab_ovoo"] = declare<T>(MO, "0697_abab_ovoo");
    tmps["0698_baba_vooo"] = declare<T>(MO, "0698_baba_vooo");
    tmps["0699_abba_ovoo"] = declare<T>(MO, "0699_abba_ovoo");
    tmps["0700_baab_vooo"] = declare<T>(MO, "0700_baab_vooo");
    tmps["0701_abab_ovoo"] = declare<T>(MO, "0701_abab_ovoo");
    tmps["0702_baba_vooo"] = declare<T>(MO, "0702_baba_vooo");
    tmps["0703_baab_vooo"] = declare<T>(MO, "0703_baab_vooo");
    tmps["0704_baab_vooo"] = declare<T>(MO, "0704_baab_vooo");
    tmps["0705_baab_vooo"] = declare<T>(MO, "0705_baab_vooo");
    tmps["0706_abab_ovoo"] = declare<T>(MO, "0706_abab_ovoo");
    tmps["0707_baab_vooo"] = declare<T>(MO, "0707_baab_vooo");
    tmps["0708_baab_vooo"] = declare<T>(MO, "0708_baab_vooo");
    tmps["0709_abab_vvoo"] = declare<T>(MO, "0709_abab_vvoo");
    tmps["0710_abab_vvoo"] = declare<T>(MO, "0710_abab_vvoo");
    tmps["0711_baba_vvoo"] = declare<T>(MO, "0711_baba_vvoo");
    tmps["0712_baab_ovoo"] = declare<T>(MO, "0712_baab_ovoo");
    tmps["0713_baab_ovoo"] = declare<T>(MO, "0713_baab_ovoo");
    tmps["0714_baab_ovoo"] = declare<T>(MO, "0714_baab_ovoo");
    tmps["0715_baab_vvoo"] = declare<T>(MO, "0715_baab_vvoo");
    tmps["0716_abba_vvoo"] = declare<T>(MO, "0716_abba_vvoo");
    tmps["0717_baba_vvoo"] = declare<T>(MO, "0717_baba_vvoo");
    tmps["0718_abba_vvoo"] = declare<T>(MO, "0718_abba_vvoo");
    tmps["0719_baab_vvoo"] = declare<T>(MO, "0719_baab_vvoo");
    tmps["0720_baab_vvoo"] = declare<T>(MO, "0720_baab_vvoo");
    tmps["0721_abab_vvoo"] = declare<T>(MO, "0721_abab_vvoo");
    tmps["0722_abab_oooo"] = declare<T>(MO, "0722_abab_oooo");
    tmps["0723_aa_vo"]     = declare<T>(MO, "0723_aa_vo");
    tmps["0724_bb_oo"]     = declare<T>(MO, "0724_bb_oo");
    tmps["0725_bb_oo"]     = declare<T>(MO, "0725_bb_oo");
    tmps["0726_bb_vo"]     = declare<T>(MO, "0726_bb_vo");
    tmps["0727_abab_vooo"] = declare<T>(MO, "0727_abab_vooo");
    tmps["0728_abab_vooo"] = declare<T>(MO, "0728_abab_vooo");
    tmps["0729_abab_vooo"] = declare<T>(MO, "0729_abab_vooo");
    tmps["0730_abab_ovoo"] = declare<T>(MO, "0730_abab_ovoo");
    tmps["0731_baab_vooo"] = declare<T>(MO, "0731_baab_vooo");
    tmps["0732_baab_vooo"] = declare<T>(MO, "0732_baab_vooo");
    tmps["0733_baab_vvoo"] = declare<T>(MO, "0733_baab_vvoo");
    tmps["0734_abab_oooo"] = declare<T>(MO, "0734_abab_oooo");
    tmps["0735_baab_vooo"] = declare<T>(MO, "0735_baab_vooo");
    tmps["0736_abba_oooo"] = declare<T>(MO, "0736_abba_oooo");
    tmps["0737_baba_vooo"] = declare<T>(MO, "0737_baba_vooo");
    tmps["0738_baab_vooo"] = declare<T>(MO, "0738_baab_vooo");
    tmps["0739_baab_vooo"] = declare<T>(MO, "0739_baab_vooo");
    tmps["0740_abab_vvoo"] = declare<T>(MO, "0740_abab_vvoo");
    tmps["0741_abba_vvoo"] = declare<T>(MO, "0741_abba_vvoo");
    tmps["0742_abab_vvoo"] = declare<T>(MO, "0742_abab_vvoo");
    tmps["0743_abab_vvoo"] = declare<T>(MO, "0743_abab_vvoo");
    tmps["0744_baab_vvoo"] = declare<T>(MO, "0744_baab_vvoo");
    tmps["0745_bb_oo"]     = declare<T>(MO, "0745_bb_oo");
    tmps["0746_abab_vvoo"] = declare<T>(MO, "0746_abab_vvoo");
    tmps["0747_abba_vooo"] = declare<T>(MO, "0747_abba_vooo");
    tmps["0748_abab_vooo"] = declare<T>(MO, "0748_abab_vooo");
    tmps["0749_abab_vooo"] = declare<T>(MO, "0749_abab_vooo");
    tmps["0750_baab_vvoo"] = declare<T>(MO, "0750_baab_vvoo");
    tmps["0751_bb_vo"]     = declare<T>(MO, "0751_bb_vo");
    tmps["0752_bb_vo"]     = declare<T>(MO, "0752_bb_vo");
    tmps["0753_bb_vo"]     = declare<T>(MO, "0753_bb_vo");
    tmps["0754_abab_vvoo"] = declare<T>(MO, "0754_abab_vvoo");
    tmps["0755_abab_vvoo"] = declare<T>(MO, "0755_abab_vvoo");
    tmps["0756_baab_vvoo"] = declare<T>(MO, "0756_baab_vvoo");
    tmps["0757_abab_vvoo"] = declare<T>(MO, "0757_abab_vvoo");
    tmps["0758_abab_vvoo"] = declare<T>(MO, "0758_abab_vvoo");
    tmps["0759_abba_vvoo"] = declare<T>(MO, "0759_abba_vvoo");
    tmps["0760_aa_vo"]     = declare<T>(MO, "0760_aa_vo");
    tmps["0761_aa_vo"]     = declare<T>(MO, "0761_aa_vo");
    tmps["0762_aa_vo"]     = declare<T>(MO, "0762_aa_vo");
    tmps["0763_baba_vvoo"] = declare<T>(MO, "0763_baba_vvoo");
    tmps["0764_bb_oo"]     = declare<T>(MO, "0764_bb_oo");
    tmps["0765_bb_oo"]     = declare<T>(MO, "0765_bb_oo");
    tmps["0766_bb_oo"]     = declare<T>(MO, "0766_bb_oo");
    tmps["0767_abab_vvoo"] = declare<T>(MO, "0767_abab_vvoo");
    tmps["0768_abba_vvoo"] = declare<T>(MO, "0768_abba_vvoo");
    tmps["0769_abba_vvoo"] = declare<T>(MO, "0769_abba_vvoo");
    tmps["0770_abab_vvoo"] = declare<T>(MO, "0770_abab_vvoo");
    tmps["0771_abab_vvoo"] = declare<T>(MO, "0771_abab_vvoo");
    tmps["0772_abba_vvoo"] = declare<T>(MO, "0772_abba_vvoo");
    tmps["0773_abab_vvoo"] = declare<T>(MO, "0773_abab_vvoo");
    tmps["0774_baab_vvoo"] = declare<T>(MO, "0774_baab_vvoo");
    tmps["0775_abab_ovoo"] = declare<T>(MO, "0775_abab_ovoo");
    tmps["0776_abab_ovoo"] = declare<T>(MO, "0776_abab_ovoo");
    tmps["0777_baab_vooo"] = declare<T>(MO, "0777_baab_vooo");
    tmps["0778_abab_ovoo"] = declare<T>(MO, "0778_abab_ovoo");
    tmps["0779_abab_vvoo"] = declare<T>(MO, "0779_abab_vvoo");
    tmps["0780_baab_vvoo"] = declare<T>(MO, "0780_baab_vvoo");
    tmps["0781_baba_vvoo"] = declare<T>(MO, "0781_baba_vvoo");
    tmps["0782_abba_vvoo"] = declare<T>(MO, "0782_abba_vvoo");
    tmps["0783_abba_vooo"] = declare<T>(MO, "0783_abba_vooo");
    tmps["0784_abba_vooo"] = declare<T>(MO, "0784_abba_vooo");
    tmps["0785_abab_vooo"] = declare<T>(MO, "0785_abab_vooo");
    tmps["0786_abab_vooo"] = declare<T>(MO, "0786_abab_vooo");
    tmps["0787_abab_vooo"] = declare<T>(MO, "0787_abab_vooo");
    tmps["0788_abab_vooo"] = declare<T>(MO, "0788_abab_vooo");
    tmps["0789_baab_vvoo"] = declare<T>(MO, "0789_baab_vvoo");
    tmps["0790_abab_vvoo"] = declare<T>(MO, "0790_abab_vvoo");
    tmps["0791_abab_vvoo"] = declare<T>(MO, "0791_abab_vvoo");
    tmps["0792_baab_vooo"] = declare<T>(MO, "0792_baab_vooo");
    tmps["0793_baab_vooo"] = declare<T>(MO, "0793_baab_vooo");
    tmps["0794_abab_ovoo"] = declare<T>(MO, "0794_abab_ovoo");
    tmps["0795_baab_vooo"] = declare<T>(MO, "0795_baab_vooo");
    tmps["0796_baab_vooo"] = declare<T>(MO, "0796_baab_vooo");
    tmps["0797_abab_vvoo"] = declare<T>(MO, "0797_abab_vvoo");
    tmps["0798_abab_vvoo"] = declare<T>(MO, "0798_abab_vvoo");
    tmps["0799_abba_vvoo"] = declare<T>(MO, "0799_abba_vvoo");
    tmps["0800_baab_vvoo"] = declare<T>(MO, "0800_baab_vvoo");
    tmps["0801_abab_vvoo"] = declare<T>(MO, "0801_abab_vvoo");
    tmps["0802_baba_vooo"] = declare<T>(MO, "0802_baba_vooo");
    tmps["0803_abab_vvoo"] = declare<T>(MO, "0803_abab_vvoo");
    tmps["0804_baba_vooo"] = declare<T>(MO, "0804_baba_vooo");
    tmps["0805_abba_vvoo"] = declare<T>(MO, "0805_abba_vvoo");
    tmps["0806_abba_vvoo"] = declare<T>(MO, "0806_abba_vvoo");
    tmps["0807_abba_vvoo"] = declare<T>(MO, "0807_abba_vvoo");
    tmps["0808_bb_oo"]     = declare<T>(MO, "0808_bb_oo");
    tmps["0809_bb_oo"]     = declare<T>(MO, "0809_bb_oo");
    tmps["0810_aa_ov"]     = declare<T>(MO, "0810_aa_ov");
    tmps["0811_bb_ov"]     = declare<T>(MO, "0811_bb_ov");
    tmps["0812_aa_vv"]     = declare<T>(MO, "0812_aa_vv");
    tmps["0813_bb_vv"]     = declare<T>(MO, "0813_bb_vv");
    tmps["0814_baba_ovoo"] = declare<T>(MO, "0814_baba_ovoo");
    tmps["0815_aaaa_vovo"] = declare<T>(MO, "0815_aaaa_vovo");
    tmps["0816_abab_vvoo"] = declare<T>(MO, "0816_abab_vvoo");
    tmps["0817_abab_vvoo"] = declare<T>(MO, "0817_abab_vvoo");
    tmps["0818_abab_vvoo"] = declare<T>(MO, "0818_abab_vvoo");
    tmps["0819_abab_vvoo"] = declare<T>(MO, "0819_abab_vvoo");
    tmps["0820_abab_vvoo"] = declare<T>(MO, "0820_abab_vvoo");
    tmps["0821_abab_vvoo"] = declare<T>(MO, "0821_abab_vvoo");
    tmps["0822_abab_vvoo"] = declare<T>(MO, "0822_abab_vvoo");
    tmps["0823_abab_vvoo"] = declare<T>(MO, "0823_abab_vvoo");
    tmps["0824_abab_vvoo"] = declare<T>(MO, "0824_abab_vvoo");
    tmps["0825_abab_vvoo"] = declare<T>(MO, "0825_abab_vvoo");
    tmps["0826_abab_vvoo"] = declare<T>(MO, "0826_abab_vvoo");
    tmps["0827_abab_vvoo"] = declare<T>(MO, "0827_abab_vvoo");
    tmps["0828_abab_vvoo"] = declare<T>(MO, "0828_abab_vvoo");
    tmps["0829_abab_vvoo"] = declare<T>(MO, "0829_abab_vvoo");
    tmps["0830_abba_vvoo"] = declare<T>(MO, "0830_abba_vvoo");
    tmps["0831_abab_vvoo"] = declare<T>(MO, "0831_abab_vvoo");
    tmps["0832_baba_vvoo"] = declare<T>(MO, "0832_baba_vvoo");
    tmps["0833_abab_vvoo"] = declare<T>(MO, "0833_abab_vvoo");
    tmps["0834_abab_vvoo"] = declare<T>(MO, "0834_abab_vvoo");
    tmps["0835_abab_vvoo"] = declare<T>(MO, "0835_abab_vvoo");
    tmps["0836_abab_vvoo"] = declare<T>(MO, "0836_abab_vvoo");
    tmps["0837_abba_vvoo"] = declare<T>(MO, "0837_abba_vvoo");
    tmps["0838_abab_vvoo"] = declare<T>(MO, "0838_abab_vvoo");
    tmps["0839_bb_oo"]     = declare<T>(MO, "0839_bb_oo");
    tmps["0840_abab_vvoo"] = declare<T>(MO, "0840_abab_vvoo");
    tmps["0841_bb_oo"]     = declare<T>(MO, "0841_bb_oo");
    tmps["0842_abab_vooo"] = declare<T>(MO, "0842_abab_vooo");
    tmps["0843_baba_vooo"] = declare<T>(MO, "0843_baba_vooo");
    tmps["0844_baab_vooo"] = declare<T>(MO, "0844_baab_vooo");
    tmps["0845_baab_vvoo"] = declare<T>(MO, "0845_baab_vvoo");
    tmps["0846_abab_vvoo"] = declare<T>(MO, "0846_abab_vvoo");
    tmps["0847_abba_vvoo"] = declare<T>(MO, "0847_abba_vvoo");
    tmps["0848_abba_vvoo"] = declare<T>(MO, "0848_abba_vvoo");
    tmps["0849_abba_vvoo"] = declare<T>(MO, "0849_abba_vvoo");
    tmps["0850_abab_vvoo"] = declare<T>(MO, "0850_abab_vvoo");
    tmps["0851_abab_vvoo"] = declare<T>(MO, "0851_abab_vvoo");
    tmps["0852_abab_vvoo"] = declare<T>(MO, "0852_abab_vvoo");
    tmps["0853_abba_vvoo"] = declare<T>(MO, "0853_abba_vvoo");
    tmps["0854_baab_vvoo"] = declare<T>(MO, "0854_baab_vvoo");
    tmps["0855_abab_vvoo"] = declare<T>(MO, "0855_abab_vvoo");
    tmps["0856_baab_vvoo"] = declare<T>(MO, "0856_baab_vvoo");
    tmps["0857_abab_vvoo"] = declare<T>(MO, "0857_abab_vvoo");
    tmps["0858_baba_vvoo"] = declare<T>(MO, "0858_baba_vvoo");
    tmps["0859_abab_vvoo"] = declare<T>(MO, "0859_abab_vvoo");
    tmps["0860_baba_vvoo"] = declare<T>(MO, "0860_baba_vvoo");
    tmps["0861_abba_vvoo"] = declare<T>(MO, "0861_abba_vvoo");
    tmps["0862_baba_vvoo"] = declare<T>(MO, "0862_baba_vvoo");
    tmps["0863_baab_vvoo"] = declare<T>(MO, "0863_baab_vvoo");
    tmps["0864_abba_vvoo"] = declare<T>(MO, "0864_abba_vvoo");
    tmps["0865_baab_vvoo"] = declare<T>(MO, "0865_baab_vvoo");
    tmps["0866_abab_vvoo"] = declare<T>(MO, "0866_abab_vvoo");
    tmps["0867_abab_vvoo"] = declare<T>(MO, "0867_abab_vvoo");
    tmps["0868_abba_vvoo"] = declare<T>(MO, "0868_abba_vvoo");
    tmps["0869_abab_vvoo"] = declare<T>(MO, "0869_abab_vvoo");
    tmps["0870_abba_vvoo"] = declare<T>(MO, "0870_abba_vvoo");
    tmps["0871_abab_vvoo"] = declare<T>(MO, "0871_abab_vvoo");
    tmps["0872_abab_vvoo"] = declare<T>(MO, "0872_abab_vvoo");
    tmps["0873_baba_vvoo"] = declare<T>(MO, "0873_baba_vvoo");
    tmps["0874_baba_vvoo"] = declare<T>(MO, "0874_baba_vvoo");
    tmps["0875_abba_vvoo"] = declare<T>(MO, "0875_abba_vvoo");
    tmps["0876_abab_vvoo"] = declare<T>(MO, "0876_abab_vvoo");
    tmps["0877_abab_vvoo"] = declare<T>(MO, "0877_abab_vvoo");
    tmps["0878_baba_vvoo"] = declare<T>(MO, "0878_baba_vvoo");
    tmps["0879_baba_vvoo"] = declare<T>(MO, "0879_baba_vvoo");
    tmps["0880_abab_vvoo"] = declare<T>(MO, "0880_abab_vvoo");
    tmps["0881_baba_vvoo"] = declare<T>(MO, "0881_baba_vvoo");
    tmps["0882_baab_vvoo"] = declare<T>(MO, "0882_baab_vvoo");
    tmps["0883_abba_vvoo"] = declare<T>(MO, "0883_abba_vvoo");
    tmps["0884_abba_vvoo"] = declare<T>(MO, "0884_abba_vvoo");
    tmps["0885_abba_vvoo"] = declare<T>(MO, "0885_abba_vvoo");
    tmps["0886_baba_vvoo"] = declare<T>(MO, "0886_baba_vvoo");
    tmps["0887_baba_vvoo"] = declare<T>(MO, "0887_baba_vvoo");
    tmps["0888_baab_vvoo"] = declare<T>(MO, "0888_baab_vvoo");
    tmps["0889_baab_vvoo"] = declare<T>(MO, "0889_baab_vvoo");
    tmps["0890_baab_vvoo"] = declare<T>(MO, "0890_baab_vvoo");
    tmps["0891_baab_vvoo"] = declare<T>(MO, "0891_baab_vvoo");
    tmps["0892_baab_vvoo"] = declare<T>(MO, "0892_baab_vvoo");
    tmps["0893_abab_vvoo"] = declare<T>(MO, "0893_abab_vvoo");
    tmps["0894_baab_vvoo"] = declare<T>(MO, "0894_baab_vvoo");
    tmps["0895_abab_vvoo"] = declare<T>(MO, "0895_abab_vvoo");
    tmps["0896_abba_vvoo"] = declare<T>(MO, "0896_abba_vvoo");
    tmps["0897_baba_vvoo"] = declare<T>(MO, "0897_baba_vvoo");
    tmps["0898_abab_vvoo"] = declare<T>(MO, "0898_abab_vvoo");
    tmps["0899_abab_vvoo"] = declare<T>(MO, "0899_abab_vvoo");
    tmps["0900_baab_vvoo"] = declare<T>(MO, "0900_baab_vvoo");
    tmps["0901_baab_vvoo"] = declare<T>(MO, "0901_baab_vvoo");
    tmps["0902_abab_vvoo"] = declare<T>(MO, "0902_abab_vvoo");
    tmps["0903_baba_vvoo"] = declare<T>(MO, "0903_baba_vvoo");
    tmps["0904_baab_vvoo"] = declare<T>(MO, "0904_baab_vvoo");
    tmps["0905_baab_vvoo"] = declare<T>(MO, "0905_baab_vvoo");
    tmps["0906_abab_vvoo"] = declare<T>(MO, "0906_abab_vvoo");
    tmps["0907_abab_vvoo"] = declare<T>(MO, "0907_abab_vvoo");
    tmps["0908_baba_vvoo"] = declare<T>(MO, "0908_baba_vvoo");
    tmps["0909_abab_vvoo"] = declare<T>(MO, "0909_abab_vvoo");
    tmps["0910_abab_vvoo"] = declare<T>(MO, "0910_abab_vvoo");
    tmps["0911_abab_vvoo"] = declare<T>(MO, "0911_abab_vvoo");
    tmps["0912_baba_vvoo"] = declare<T>(MO, "0912_baba_vvoo");
    tmps["0913_abab_vvoo"] = declare<T>(MO, "0913_abab_vvoo");
    tmps["0914_baab_vvoo"] = declare<T>(MO, "0914_baab_vvoo");
    tmps["0915_abab_vvoo"] = declare<T>(MO, "0915_abab_vvoo");
    tmps["0916_baab_vvoo"] = declare<T>(MO, "0916_baab_vvoo");
    tmps["0917_baba_vvoo"] = declare<T>(MO, "0917_baba_vvoo");
    tmps["0918_abab_vvoo"] = declare<T>(MO, "0918_abab_vvoo");
    tmps["0919_baba_vooo"] = declare<T>(MO, "0919_baba_vooo");
    tmps["0920_abab_vvoo"] = declare<T>(MO, "0920_abab_vvoo");
    tmps["0921_baab_vvoo"] = declare<T>(MO, "0921_baab_vvoo");
    tmps["0922_abab_vvoo"] = declare<T>(MO, "0922_abab_vvoo");
    tmps["0923_abab_vvoo"] = declare<T>(MO, "0923_abab_vvoo");
    tmps["0924_abab_vvoo"] = declare<T>(MO, "0924_abab_vvoo");
    tmps["0925_abba_vvoo"] = declare<T>(MO, "0925_abba_vvoo");
    tmps["0926_abab_vvoo"] = declare<T>(MO, "0926_abab_vvoo");
    tmps["0927_abab_vvoo"] = declare<T>(MO, "0927_abab_vvoo");
    tmps["0928_abab_vvoo"] = declare<T>(MO, "0928_abab_vvoo");
    tmps["0929_abba_vvoo"] = declare<T>(MO, "0929_abba_vvoo");
    tmps["0930_abba_vvoo"] = declare<T>(MO, "0930_abba_vvoo");
    tmps["0931_abab_vvoo"] = declare<T>(MO, "0931_abab_vvoo");
    tmps["0932_baba_vvoo"] = declare<T>(MO, "0932_baba_vvoo");
    tmps["0933_abba_vvoo"] = declare<T>(MO, "0933_abba_vvoo");
    tmps["0934_abba_vvoo"] = declare<T>(MO, "0934_abba_vvoo");
    tmps["0935_baab_vvoo"] = declare<T>(MO, "0935_baab_vvoo");
    tmps["0936_abab_vvoo"] = declare<T>(MO, "0936_abab_vvoo");
    tmps["0937_abab_vvoo"] = declare<T>(MO, "0937_abab_vvoo");
    tmps["0938_baba_vvoo"] = declare<T>(MO, "0938_baba_vvoo");
    tmps["0939_abba_vvoo"] = declare<T>(MO, "0939_abba_vvoo");
    tmps["0940_baab_vvoo"] = declare<T>(MO, "0940_baab_vvoo");
    tmps["0941_abab_vvoo"] = declare<T>(MO, "0941_abab_vvoo");
    tmps["0942_baab_vvoo"] = declare<T>(MO, "0942_baab_vvoo");
    tmps["0943_abab_vvoo"] = declare<T>(MO, "0943_abab_vvoo");
    tmps["0944_baab_vvoo"] = declare<T>(MO, "0944_baab_vvoo");
    tmps["0945_abab_vvoo"] = declare<T>(MO, "0945_abab_vvoo");
    tmps["0946_baab_vvoo"] = declare<T>(MO, "0946_baab_vvoo");
    tmps["0947_abba_vvoo"] = declare<T>(MO, "0947_abba_vvoo");
    tmps["0948_abba_vvoo"] = declare<T>(MO, "0948_abba_vvoo");
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

template Tensor<double> exachem::cc::qed_ccsd_cs::declare<double>(const tamm::TiledIndexSpace& MO,
                                                                  const std::string& name);
template void           exachem::cc::qed_ccsd_cs::build_tmps<double>(
  Scheduler& sch, const TiledIndexSpace& MO, TensorMap<double>& tmps, TensorMap<double>& scalars,
  const TensorMap<double>& f, const TensorMap<double>& eri, const TensorMap<double>& dp,
  const double w0, const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p);