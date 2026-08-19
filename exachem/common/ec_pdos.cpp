/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/ec_pdos.hpp"
#include "exachem/common/txt_utils.hpp"

double ha2ev = exachem::constants::ha2ev;

void ECPDOS::write_pdos(ChemEnv& chem_env, Matrix& S, Matrix& C_a, std::vector<double>& eps_a,
                        std::string files_prefix) {
  // Input Options
  size_t      npoints      = chem_env.ioptions.pdos_options.npoints;
  double      emin         = chem_env.ioptions.pdos_options.emin;
  double      emax         = chem_env.ioptions.pdos_options.emax;
  double      fwhm         = chem_env.ioptions.pdos_options.fwhm;
  bool        do_mod       = chem_env.ioptions.pdos_options.do_mod;
  std::string distribution = chem_env.ioptions.pdos_options.distribution;

  // Sizes and Auxiliary Objects
  size_t nmo        = C_a.cols();
  size_t noa        = chem_env.sys_data.nelectrons_alpha;
  size_t natoms     = chem_env.atoms.size();
  auto   atom2shell = chem_env.shells.atom2shell(chem_env.atoms);
  auto   shell2bf   = chem_env.shells.shell2bf();
  int    lmax       = chem_env.shells.max_l();

  // If necessary, Adjust EMIN and EMAX
  double efermi = 0.5 * (eps_a[noa] + eps_a[noa - 1]);
  if(emin == 0.0 && emax == 0.0) {
    emin = efermi - 1.0; // 1 Hartree below
    emax = efermi + 1.0;
  }

  // Set energy grid
  double              delta = (emax - emin) / (npoints - 1);
  std::vector<double> egrid(npoints);
  egrid[0] = emin;
  for(size_t igrid = 1; igrid < npoints; igrid++) egrid[igrid] = egrid[igrid - 1] + delta;

  // Atomic Overlap
  Matrix SC = S * C_a;
  Matrix pdos_atom(lmax + 1, npoints);

  // Stream
  std::ofstream pdosfile;
  std::string   filename = files_prefix + ".pdos";
  pdosfile.open(filename);
  pdos_atom.setZero();
  for(size_t imo = 0; imo < nmo; imo++) {
    std::vector<double> dos;
    if(distribution == "gaussian") { dos = gaussian_smearing(egrid, eps_a[imo], fwhm); }
    else if(distribution == "lorentzian") { dos = lorentzian_smearing(egrid, eps_a[imo], fwhm); }
    for(size_t igrid = 0; igrid < npoints; igrid++) { pdos_atom(0, igrid) += dos[igrid]; }
  }
  pdosfile << "E_Fermi = " << std::fixed << std::setprecision(6) << efermi * ha2ev << " eV"
           << std::endl;
  pdosfile << "Number of atoms = " << natoms << std::endl;
  pdosfile << npoints << std::endl;
  pdosfile << std::scientific << std::setprecision(6);
  for(size_t igrid = 0; igrid < npoints; igrid++) {
    pdosfile << std::setw(16) << egrid[igrid] * ha2ev << std::setw(16)
             << pdos_atom(0, igrid) / ha2ev << std::endl;
  }

  for(size_t iatom = 0; iatom < natoms; iatom++) {
    // Atomic Sizes
    auto   atshells = atom2shell[iatom];
    size_t nshells  = atshells.size();
    size_t sh_lo    = atshells[0];
    size_t lmax_at  = 0;
    pdosfile << "Partial DOS for atom " << chem_env.ec_atoms[iatom].esymbol << " " << iatom
             << std::endl;

    for(size_t ish = sh_lo; ish < sh_lo + nshells; ish++)
      lmax_at = std::max(lmax_at, static_cast<size_t>(chem_env.shells[ish].contr[0].l));

    pdos_atom.resize(lmax_at + 1, npoints);
    pdos_atom.setZero();
    std::vector<std::vector<size_t>> lshells(lmax_at + 1);
    for(size_t ish = sh_lo; ish < sh_lo + nshells; ish++) {
      size_t l = chem_env.shells[ish].contr[0].l;
      lshells[l].push_back(ish);
    }

    Matrix CSC_l(lmax_at + 1, nmo);
    for(size_t l = 0; l <= lmax_at; l++) {
      size_t nsh_l = lshells[l].size();
      size_t nbf_l = nsh_l * (2 * l + 1);
      Matrix C_l(nbf_l, nmo);
      Matrix S_l(nbf_l, nbf_l);
      Matrix SC_local(nbf_l, nmo);
      size_t ibf_l = 0;
      for(size_t ish_l = 0; ish_l < nsh_l; ish_l++) {
        size_t ibf_lo_l = shell2bf[lshells[l][ish_l]];
        size_t jbf_l    = 0;
        for(size_t jsh_l = 0; jsh_l < nsh_l; jsh_l++) {
          size_t jbf_lo_l = shell2bf[lshells[l][jsh_l]];
          S_l.block(ibf_l, jbf_l, 2 * l + 1, 2 * l + 1) =
            S.block(ibf_lo_l, jbf_lo_l, 2 * l + 1, 2 * l + 1);
          jbf_l += 2 * l + 1;
        }
        C_l.block(ibf_l, 0, 2 * l + 1, nmo)      = C_a.block(ibf_lo_l, 0, 2 * l + 1, nmo);
        SC_local.block(ibf_l, 0, 2 * l + 1, nmo) = SC.block(ibf_lo_l, 0, 2 * l + 1, nmo);
        ibf_l += 2 * l + 1;
      }
      if(do_mod) {
        Matrix S_l_inv = S_l.inverse();
        for(size_t imo = 0; imo < nmo; imo++) {
          CSC_l(l, imo) = SC_local.col(imo).transpose() * S_l_inv * SC_local.col(imo);
        }
      }
      else {
        for(size_t imo = 0; imo < nmo; imo++) {
          CSC_l(l, imo) = C_l.col(imo).transpose() * SC_local.col(imo);
        }
      }
    }

    for(size_t imo = 0; imo < nmo; imo++) {
      std::vector<double> smearing;
      if(distribution == "gaussian") { smearing = gaussian_smearing(egrid, eps_a[imo], fwhm); }
      else if(distribution == "lorentzian") {
        smearing = lorentzian_smearing(egrid, eps_a[imo], fwhm);
      }
      for(size_t l = 0; l <= lmax_at; l++) {
        for(size_t igrid = 0; igrid < npoints; igrid++) {
          pdos_atom(l, igrid) += smearing[igrid] * CSC_l(l, imo);
        }
      }
    }
    for(size_t igrid = 0; igrid < npoints; igrid++) {
      pdosfile << std::setw(16) << egrid[igrid] * ha2ev;
      for(size_t l = 0; l <= lmax_at; l++) pdosfile << std::setw(16) << pdos_atom(l, igrid) / ha2ev;
      pdosfile << std::endl;
    }
  }
  pdosfile.close();
}

void ECPDOS::write_pdos(ChemEnv& chem_env, Matrix& S, Matrix& C_a, std::vector<double>& eps_a,
                        Matrix& C_b, std::vector<double>& eps_b, std::string files_prefix) {
  // Input Options
  size_t      npoints      = chem_env.ioptions.pdos_options.npoints;
  double      emin         = chem_env.ioptions.pdos_options.emin;
  double      emax         = chem_env.ioptions.pdos_options.emax;
  double      fwhm         = chem_env.ioptions.pdos_options.fwhm;
  bool        do_mod       = chem_env.ioptions.pdos_options.do_mod;
  std::string distribution = chem_env.ioptions.pdos_options.distribution;

  // Sizes and Auxiliary Objects
  size_t nmo        = C_a.cols();
  size_t noa        = chem_env.sys_data.nelectrons_alpha;
  size_t nob        = chem_env.sys_data.nelectrons_beta;
  size_t natoms     = chem_env.atoms.size();
  auto   atom2shell = chem_env.shells.atom2shell(chem_env.atoms);
  auto   shell2bf   = chem_env.shells.shell2bf();
  int    lmax       = chem_env.shells.max_l();

  // If necessary, Adjust EMIN and EMAX
  double efermi_a = 0.5 * (eps_a[noa] + eps_a[noa - 1]);
  double efermi_b = 0.5 * (eps_b[nob] + eps_b[nob - 1]);
  if(emin == 0.0 && emax == 0.0) {
    emin = std::min(efermi_a, efermi_b) - 1.0; // 1 Hartree below
    emax = std::max(efermi_a, efermi_b) + 1.0;
  }

  // Set energy grid
  double              delta = (emax - emin) / (npoints - 1);
  std::vector<double> egrid(npoints);
  egrid[0] = emin;
  for(size_t igrid = 1; igrid < npoints; igrid++) egrid[igrid] = egrid[igrid - 1] + delta;

  // Atomic Overlap
  Matrix SC_a = S * C_a;
  Matrix SC_b = S * C_b;
  Matrix pdos_atom_a(lmax + 1, npoints);
  Matrix pdos_atom_b(lmax + 1, npoints);

  // Stream
  std::ofstream pdosfile;
  std::string   filename = files_prefix + ".pdos";
  pdosfile.open(filename);
  pdos_atom_a.setZero();
  pdos_atom_b.setZero();
  for(size_t imo = 0; imo < nmo; imo++) {
    std::vector<double> dos_a, dos_b;
    if(distribution == "gaussian") {
      dos_a = gaussian_smearing(egrid, eps_a[imo], fwhm);
      dos_b = gaussian_smearing(egrid, eps_b[imo], fwhm);
    }
    else if(distribution == "lorentzian") {
      dos_a = lorentzian_smearing(egrid, eps_a[imo], fwhm);
      dos_b = lorentzian_smearing(egrid, eps_b[imo], fwhm);
    }
    for(size_t igrid = 0; igrid < npoints; igrid++) {
      pdos_atom_a(0, igrid) += dos_a[igrid];
      pdos_atom_b(0, igrid) += dos_b[igrid];
    }
  }
  pdosfile << "E_Fermi Alpha = " << std::fixed << std::setprecision(6) << efermi_a * ha2ev << " eV"
           << ", E_Fermi Beta = " << efermi_b * ha2ev << " eV" << std::endl;
  pdosfile << "Number of atoms = " << natoms << std::endl;
  pdosfile << npoints << std::endl;
  pdosfile << std::scientific << std::setprecision(6);
  for(size_t igrid = 0; igrid < npoints; igrid++) {
    pdosfile << std::setw(16) << egrid[igrid] * ha2ev << std::setw(16)
             << pdos_atom_a(0, igrid) / ha2ev << std::setw(16) << pdos_atom_b(0, igrid) / ha2ev
             << std::endl;
  }

  for(size_t iatom = 0; iatom < natoms; iatom++) {
    // Atomic Sizes
    auto   atshells = atom2shell[iatom];
    size_t nshells  = atshells.size();
    size_t sh_lo    = atshells[0];
    size_t lmax_at  = 0;
    pdosfile << "Partial DOS for atom " << chem_env.ec_atoms[iatom].esymbol << " " << iatom
             << std::endl;

    for(size_t ish = sh_lo; ish < sh_lo + nshells; ish++)
      lmax_at = std::max(lmax_at, static_cast<size_t>(chem_env.shells[ish].contr[0].l));

    pdos_atom_a.resize(lmax_at + 1, npoints);
    pdos_atom_b.resize(lmax_at + 1, npoints);
    pdos_atom_a.setZero();
    pdos_atom_b.setZero();
    std::vector<std::vector<size_t>> lshells(lmax_at + 1);
    for(size_t ish = sh_lo; ish < sh_lo + nshells; ish++) {
      size_t l = chem_env.shells[ish].contr[0].l;
      lshells[l].push_back(ish);
    }

    Matrix CSC_l_a(lmax_at + 1, nmo);
    Matrix CSC_l_b(lmax_at + 1, nmo);
    for(size_t l = 0; l <= lmax_at; l++) {
      size_t nsh_l = lshells[l].size();
      size_t nbf_l = nsh_l * (2 * l + 1);
      Matrix C_l_a(nbf_l, nmo);
      Matrix C_l_b(nbf_l, nmo);
      Matrix S_l(nbf_l, nbf_l);
      Matrix SC_local_a(nbf_l, nmo);
      Matrix SC_local_b(nbf_l, nmo);
      size_t ibf_l = 0;
      for(size_t ish_l = 0; ish_l < nsh_l; ish_l++) {
        size_t ibf_lo_l = shell2bf[lshells[l][ish_l]];
        size_t jbf_l    = 0;
        for(size_t jsh_l = 0; jsh_l < nsh_l; jsh_l++) {
          size_t jbf_lo_l = shell2bf[lshells[l][jsh_l]];
          S_l.block(ibf_l, jbf_l, 2 * l + 1, 2 * l + 1) =
            S.block(ibf_lo_l, jbf_lo_l, 2 * l + 1, 2 * l + 1);
          jbf_l += 2 * l + 1;
        }
        C_l_a.block(ibf_l, 0, 2 * l + 1, nmo)      = C_a.block(ibf_lo_l, 0, 2 * l + 1, nmo);
        C_l_b.block(ibf_l, 0, 2 * l + 1, nmo)      = C_b.block(ibf_lo_l, 0, 2 * l + 1, nmo);
        SC_local_a.block(ibf_l, 0, 2 * l + 1, nmo) = SC_a.block(ibf_lo_l, 0, 2 * l + 1, nmo);
        SC_local_b.block(ibf_l, 0, 2 * l + 1, nmo) = SC_b.block(ibf_lo_l, 0, 2 * l + 1, nmo);
        ibf_l += 2 * l + 1;
      }
      if(do_mod) {
        Matrix S_l_inv = S_l.inverse();
        for(size_t imo = 0; imo < nmo; imo++) {
          CSC_l_a(l, imo) = SC_local_a.col(imo).transpose() * S_l_inv * SC_local_a.col(imo);
          CSC_l_b(l, imo) = SC_local_b.col(imo).transpose() * S_l_inv * SC_local_b.col(imo);
        }
      }
      else {
        for(size_t imo = 0; imo < nmo; imo++) {
          CSC_l_a(l, imo) = C_l_a.col(imo).transpose() * SC_local_a.col(imo);
          CSC_l_b(l, imo) = C_l_b.col(imo).transpose() * SC_local_b.col(imo);
        }
      }
    }

    for(size_t imo = 0; imo < nmo; imo++) {
      std::vector<double> smearing_a, smearing_b;
      if(distribution == "gaussian") {
        smearing_a = gaussian_smearing(egrid, eps_a[imo], fwhm);
        smearing_b = gaussian_smearing(egrid, eps_b[imo], fwhm);
      }
      else if(distribution == "lorentzian") {
        smearing_a = lorentzian_smearing(egrid, eps_a[imo], fwhm);
        smearing_b = lorentzian_smearing(egrid, eps_b[imo], fwhm);
      }
      for(size_t l = 0; l <= lmax_at; l++) {
        for(size_t igrid = 0; igrid < npoints; igrid++) {
          pdos_atom_a(l, igrid) += smearing_a[igrid] * CSC_l_a(l, imo);
          pdos_atom_b(l, igrid) += smearing_b[igrid] * CSC_l_b(l, imo);
        }
      }
    }
    for(size_t igrid = 0; igrid < npoints; igrid++) {
      pdosfile << std::setw(16) << egrid[igrid] * ha2ev;
      for(size_t l = 0; l <= lmax_at; l++)
        pdosfile << std::setw(16) << pdos_atom_a(l, igrid) / ha2ev << std::setw(16)
                 << pdos_atom_b(l, igrid) / ha2ev;
      pdosfile << std::endl;
    }
  }
  pdosfile.close();
}

std::vector<double> ECPDOS::gaussian_smearing(std::vector<double>& x, double x0, double fwhm) {
  const double        factor     = 2.0 * std::sqrt(2.0 * std::log(2.0));
  size_t              npoints    = x.size();
  double              sigma      = fwhm / factor;
  const double        twosigmasq = 2.0 * sigma * sigma;
  constexpr double    PI         = 3.14159265358979323846;
  const double        prefactor  = 1.0 / std::sqrt(PI * twosigmasq);
  std::vector<double> gaussian(npoints);
  for(size_t i = 0; i < npoints; i++)
    gaussian[i] = prefactor * std::exp(-(x[i] - x0) * (x[i] - x0) / twosigmasq);
  return gaussian;
}

std::vector<double> ECPDOS::lorentzian_smearing(std::vector<double>& x, double x0, double fwhm) {
  constexpr double    factor    = 2.0;
  size_t              npoints   = x.size();
  double              gamma     = fwhm / factor;
  double              gammasq   = gamma * gamma;
  constexpr double    PI        = 3.14159265358979323846;
  const double        prefactor = gamma / PI;
  std::vector<double> lorentzian(npoints);
  for(size_t i = 0; i < npoints; i++)
    lorentzian[i] = prefactor / ((x[i] - x0) * (x[i] - x0) + gammasq);
  return lorentzian;
}
