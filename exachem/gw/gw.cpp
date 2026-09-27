/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/gw/gw.hpp"
#include "exachem/common/constants.hpp"

namespace exachem::gw {

using exachem::constants::ha2ev;

void GWData::print() const {
  std::cout << "ipol:  " << ipol << std::endl;
  std::cout << "nmo:   " << nmo << std::endl;
  std::cout << "maxev: " << maxev << std::endl;
  std::cout << "nocc:  " << nocc << std::endl;
  std::cout << "nvir:  " << nvir << std::endl;
  std::cout << "noqp:  " << noqp << std::endl;
  std::cout << "nvqp:  " << nvqp << std::endl;
  std::cout << "nqp:   " << nqp << std::endl;
  std::cout << "lo:    " << lo << std::endl;
  std::cout << "hi:    " << hi << std::endl;
}

GWData gw_pars(const ChemEnv& chem_env, bool mrank) {
  const SystemData& sys_data   = chem_env.sys_data;
  const GWOptions&  gw_options = chem_env.ioptions.gw_options;

  GWData gwd;
  gwd.ipol = sys_data.is_unrestricted ? 2 : 1;
  gwd.nmo  = sys_data.n_occ_alpha + sys_data.n_vir_alpha;

  gwd.nocc = {sys_data.n_occ_alpha, sys_data.n_occ_beta};
  gwd.nvir = {gwd.nmo - gwd.nocc[0], gwd.nmo - gwd.nocc[1]};
  gwd.noqp.assign(2, 0);
  gwd.nvqp.assign(2, 0);
  gwd.nqp.assign(2, 0);
  gwd.lo.assign(2, 0);
  gwd.hi.assign(2, 0);

  const bool evgw   = gw_options.evgw;
  const bool evgw0  = gw_options.evgw0;
  const bool docore = gw_options.core;

  const int noqp_in[2] = {gw_options.noqpa, gw_options.noqpb};
  const int nvqp_in[2] = {gw_options.nvqpa, gw_options.nvqpb};

  for(int ispin = 0; ispin < gwd.ipol; ispin++) {
    const std::string sstr = (ispin == 0) ? "a" : "b";
    const int         nocc = gwd.nocc[ispin];

    gwd.noqp[ispin] = (noqp_in[ispin] < 0 || evgw || evgw0) ? nocc : noqp_in[ispin];
    gwd.nvqp[ispin] = (nvqp_in[ispin] < 0) ? gwd.nmo - nocc : nvqp_in[ispin];

    if(docore) {
      if(gwd.nvqp[ispin] > 0 && gwd.noqp[ispin] < nocc) {
        if(mrank) {
          std::cout << "\t Warning: nvqp" << sstr << " > 0 and noqp" << sstr
                    << " < nocc is incompatible with core" << std::endl;
          std::cout << "\t          setting nvqp" << sstr << " to 0" << std::endl;
        }
        gwd.nvqp[ispin] = 0;
      }
      gwd.lo[ispin] = 0;
      gwd.hi[ispin] = std::max(gwd.noqp[ispin] + gwd.nvqp[ispin], nocc);
    }
    else {
      gwd.lo[ispin] = nocc - gwd.noqp[ispin];
      gwd.hi[ispin] = nocc + gwd.nvqp[ispin];
    }
    gwd.nqp[ispin] = gwd.noqp[ispin] + gwd.nvqp[ispin];

    if(gwd.lo[ispin] < 0 || gwd.hi[ispin] > gwd.nmo)
      tamm_terminate("GW error: requested QP window [noqp" + sstr + ",nvqp" + sstr +
                     "] exceeds the number of occupied/virtual orbitals");
  }

  gwd.maxev = (evgw || evgw0) ? std::max(gw_options.maxev + 1, 4) : gw_options.maxev + 1;

  return gwd;
}

std::vector<int> gw_findclusters(const std::vector<double>& vals, int nqp) {
  std::vector<int> clusters;
  int              ll = 0;

  while(true) {
    int          icluster = 1;
    const double target   = vals[ll] + 0.05;

    for(int iqp = ll + 1; iqp < nqp; iqp++) {
      // HOMO and LUMO always go to different clusters
      if(vals[iqp] * vals[iqp - 1] < 0.0) break;
      if(vals[iqp] <= target) icluster += 1;
      else break;
    }
    clusters.push_back(icluster);
    ll += icluster;
    if(ll >= nqp) break;
  }
  return clusters;
}

void gw_scissor(const std::vector<double>& oldevals, std::vector<double>& newevals, int noqp,
                int nvqp, int nocc, int lo, int hi, int nmo, const std::string& spin, bool mrank) {
  // Occupied states
  if(noqp < nocc && noqp > 0) {
    double shift = 0.0;
    for(int iqp = lo; iqp < std::min(hi, nocc); iqp++) shift += newevals[iqp] - oldevals[iqp];
    shift /= noqp;

    if(mrank)
      std::cout << gw_strfmt("\t Applying %8.4f eV shift to rest of %s occupied states\n",
                             shift * ha2ev, spin.c_str())
                << std::endl;

    for(int iqp = 0; iqp < nocc; iqp++) {
      if(iqp >= lo && iqp < hi) continue;
      newevals[iqp] = oldevals[iqp] + shift;
    }
  }

  // Virtual states
  if(nvqp < nmo - nocc && nvqp > 0) {
    double shift = 0.0;
    for(int iqp = nocc; iqp < nocc + nvqp; iqp++) shift += newevals[iqp] - oldevals[iqp];
    shift /= nvqp;

    if(mrank)
      std::cout << gw_strfmt("\t Applying %8.4f eV shift to rest of %s virtual states\n",
                             shift * ha2ev, spin.c_str())
                << std::endl;

    for(int iqp = nocc + nvqp; iqp < nmo; iqp++) newevals[iqp] = oldevals[iqp] + shift;
  }
}

void gw_print_iter(int inewton, double ein, double eout, double elower, double eupper,
                   bool bracket) {
  std::cout << gw_strfmt("\t Iter: %d Ein: %12.6f Eout: %12.6f", inewton, ein * ha2ev,
                         eout * ha2ev);
  if(bracket) std::cout << gw_strfmt(" Bracket: [%12.6f, %12.6f]", elower * ha2ev, eupper * ha2ev);
  std::cout << std::endl;
}

TiledIndexSpace gw_setupMOIS(int nmo, int nocc, tamm::Tile tilesize) {
  IndexSpace MO_IS{range(0, nmo), {{"occ", {range(0, nocc)}}, {"virt", {range(nocc, nmo)}}}};

  auto split = [&](int n, std::vector<tamm::Tile>& tiles) {
    if(n <= 0) return;
    const tamm::Tile nt = static_cast<tamm::Tile>(std::ceil(1.0 * n / tilesize));
    for(tamm::Tile x = 0; x < nt; x++) tiles.push_back(n / nt + (x < (n % nt)));
  };

  std::vector<tamm::Tile> mo_tiles;
  split(nocc, mo_tiles);
  split(nmo - nocc, mo_tiles);

  return TiledIndexSpace{MO_IS, mo_tiles};
}

} // namespace exachem::gw
