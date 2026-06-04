/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/ec_molden.hpp"
#include "exachem/common/txt_utils.hpp"
#include <libint2/lcao/molden.h>

bool ECMolden::check_molden(std::string moldenfile) {
  molden_exists = !moldenfile.empty();
  if(molden_exists) {
    molden_file_valid = std::filesystem::exists(moldenfile);
    if(!molden_file_valid)
      tamm_terminate("ERROR: moldenfile provided: " + moldenfile + " does not exist");
  }
  return molden_file_valid;
}

void ECMolden::write_molden(ChemEnv& chem_env, Matrix& C_a, std::vector<double>& eps_a_vec,
                            std::string files_prefix) {
  Eigen::VectorXd occs(C_a.cols());
  occs.setZero();
  const auto ndocc = chem_env.sys_data.nelectrons / 2;
  for(int i = 0; i < ndocc; i++) occs[i] = 2.0;

  std::vector<libint2::Shell> mshells_(chem_env.shells.size());
  auto                        shell2bf = chem_env.shells.shell2bf();
  Matrix                      C_copy   = C_a;
  for(size_t i = 0; i < mshells_.size(); i++) {
    mshells_[i] = chem_env.shells[i];
    for(auto& contr: mshells_[i].contr) {
      if(contr.l == 1) {
        contr.pure          = false;
        size_t ibf          = shell2bf[i];
        C_copy.row(ibf)     = C_a.row(ibf + 2);
        C_copy.row(ibf + 1) = C_a.row(ibf);
        C_copy.row(ibf + 2) = C_a.row(ibf + 1);
      }
    }
  }

  libint2::BasisSet mshells(mshells_);

  Matrix eps_a(eps_a_vec.size(), 1);
  for(Eigen::Index i = 0; i < eps_a.rows(); i++) eps_a(i, 0) = eps_a_vec[i];

  std::vector<std::string> symmetry_labels;
  std::vector<bool>        spincases;
  constexpr double         bohr2ang            = exachem::constants::bohr2ang;
  double                   coefficient_epsilon = 0.0;

  libint2::molden::Export molden_export(chem_env.atoms, mshells, C_copy, occs, eps_a,
                                        symmetry_labels, spincases, bohr2ang, coefficient_epsilon);
  std::ofstream           molden_file(files_prefix + ".molden");
  molden_export.write(molden_file);
}

void ECMolden::write_molden(ChemEnv& chem_env, Matrix& C_a, std::vector<double>& eps_a_vec,
                            Matrix& C_b, std::vector<double>& eps_b_vec, std::string files_prefix) {
  const size_t nspin  = C_a.cols();
  const size_t ntotal = C_a.cols() + C_b.cols();
  const size_t noa    = chem_env.sys_data.nelectrons_alpha;
  const size_t nob    = chem_env.sys_data.nelectrons_beta;

  std::vector<bool> spincases(ntotal, true);
  Eigen::VectorXd   occs(ntotal);
  occs.setZero();
  for(size_t i = 0; i < noa; i++) occs[i] = 1.0;
  for(size_t i = nspin; i < ntotal; i++) {
    size_t j = i - nspin;
    if(j < nob) occs[i] = 1.0;
    spincases[i] = false;
  }

  std::vector<libint2::Shell> mshells_(chem_env.shells.size());
  auto                        shell2bf = chem_env.shells.shell2bf();
  Matrix                      C_copy(C_a.rows(), ntotal);
  C_copy.leftCols(nspin)  = C_a;
  C_copy.rightCols(nspin) = C_b;
  for(size_t i = 0; i < mshells_.size(); i++) {
    mshells_[i] = chem_env.shells[i];
    for(auto& contr: mshells_[i].contr) {
      if(contr.l == 1) {
        contr.pure                         = false;
        size_t ibf                         = shell2bf[i];
        C_copy.block(ibf, 0, 1, nspin)     = C_a.row(ibf + 2);
        C_copy.block(ibf, nspin, 1, nspin) = C_b.row(ibf + 2);

        C_copy.block(ibf + 1, 0, 1, nspin)     = C_a.row(ibf);
        C_copy.block(ibf + 1, nspin, 1, nspin) = C_b.row(ibf);

        C_copy.block(ibf + 2, 0, 1, nspin)     = C_a.row(ibf + 1);
        C_copy.block(ibf + 2, nspin, 1, nspin) = C_b.row(ibf + 1);
      }
    }
  }

  libint2::BasisSet mshells(mshells_);

  Matrix eps(ntotal, 1);
  for(size_t i = 0; i < nspin; i++) eps(i, 0) = eps_a_vec[i];
  for(size_t i = nspin; i < ntotal; i++) eps(i, 0) = eps_b_vec[i - nspin];

  std::vector<std::string> symmetry_labels;
  constexpr double         bohr2ang            = exachem::constants::bohr2ang;
  double                   coefficient_epsilon = 0.0;

  libint2::molden::Export molden_export(chem_env.atoms, mshells, C_copy, occs, eps, symmetry_labels,
                                        spincases, bohr2ang, coefficient_epsilon);
  std::ofstream           molden_file(files_prefix + ".molden");
  molden_export.write(molden_file);
}

std::string ECMolden::read_option(std::string line) {
  std::istringstream       oss(line);
  std::vector<std::string> option_string{std::istream_iterator<std::string>{oss},
                                         std::istream_iterator<std::string>{}};
  // assert(option_string.size() == 2);

  return option_string[1];
}

bool ECMolden::is_comment(const std::string line) {
  auto found = false;
  if(line.find("//") != std::string::npos) {
    // found = true;
    auto fpos = line.find_first_not_of(' ');
    auto str  = line.substr(fpos, 2);
    if(str == "//") found = true;
  }
  return found;
}

bool ECMolden::is_in_line(const std::string str, const std::string line) {
  auto        found = true;
  std::string str_u = txt_utils::to_upper(str);
  std::string str_l = txt_utils::to_lower(str);

  if(is_comment(line)) found = false;
  else {
    std::istringstream       oss(line);
    std::vector<std::string> option_string{std::istream_iterator<std::string>{oss},
                                           std::istream_iterator<std::string>{}};
    for(auto& x: option_string) x.erase(std::remove(x.begin(), x.end(), ' '), x.end());

    if(std::find(option_string.begin(), option_string.end(), str_u) == option_string.end() &&
       std::find(option_string.begin(), option_string.end(), str_l) == option_string.end())
      found = false;
  }

  return found;
}

template<typename T>
void ECMolden::reorder_molden_orbitals(const bool is_spherical, const libint2::BasisSet& shells,
                                       Matrix& smat, Matrix& dmat, const bool reorder_cols,
                                       const bool reorder_rows) {
  auto   shell2bf = shells.shell2bf();
  size_t nsh      = shells.size();
  if(reorder_rows) {
    for(size_t ish = 0; ish < nsh; ish++) {
      int l   = shells[ish].contr[0].l;
      int ibf = shell2bf[ish];
      if(l == 0) { dmat.row(ibf) = smat.row(ibf); }
      else if(l == 1) {
        dmat.row(ibf)     = smat.row(ibf + 1);
        dmat.row(ibf + 1) = smat.row(ibf + 2);
        dmat.row(ibf + 2) = smat.row(ibf);
      }
      else {
        for(int m = -l, i = 0; m <= l; m++, i++) {
          int j             = 2 * std::abs(m) + (m > 0 ? -1 : 0);
          dmat.row(ibf + i) = smat.row(ibf + j);
        }
      }
    }
  }
}

// TODO: is this needed? - currently does not make a difference
libint2::BasisSet ECMolden::renormalize_libint_shells(libint2::BasisSet& shells) {
  using libint2::math::df_Kminus1;
  using std::pow;
  // const auto                  sqrt_Pi_cubed = double{5.56832799683170784528481798212};
  std::vector<libint2::Shell> rshells(shells.size());
  size_t                      sid = 0;
  for(auto it = shells.begin(); it != shells.end(); it++) {
    rshells[sid] = *it;
    sid++;
  }
#if 0 // TODO: Doesn't work - is renormalization required?
  for(auto& s: rshells) {
    const auto np = s.nprim();
    for(auto& c: s.contr) {
      EXPECTS(c.l <= 15); 
      for(auto p=0ul; p!=np; ++p) {
        EXPECTS(s.alpha[p] >= 0);
        if (s.alpha[p] != 0) {
          const auto two_alpha = 2 * s.alpha[p];
          const auto two_alpha_to_am32 = pow(two_alpha,c.l+1) * sqrt(two_alpha);
          const auto normalization_factor = sqrt(pow(2,c.l) * two_alpha_to_am32/(sqrt_Pi_cubed * df_Kminus1[2*c.l] ));

          c.coeff[p] *= normalization_factor;
        }
      }

      // need to force normalization to unity?
      if (s.do_enforce_unit_normalization()) {
        // compute the self-overlap of the , scale coefficients by its inverse square root
        double norm{0};
        for(auto p=0ul; p!=np; ++p) {
          for(decltype(p) q=0ul; q<=p; ++q) {
            auto gamma = s.alpha[p] + s.alpha[q];
            norm += (p==q ? 1 : 2) * df_Kminus1[2*c.l] * sqrt_Pi_cubed * c.coeff[p] * c.coeff[q] /
                    (pow(2,c.l) * pow(gamma,c.l+1) * sqrt(gamma));
          }
        }
        auto normalization_factor = 1 / sqrt(norm);
        for(auto p=0ul; p!=np; ++p) {
          c.coeff[p] *= normalization_factor;
        }
      }

    }
    // update max log coefficients
    s.max_ln_coeff.resize(np);
    for(auto p=0ul; p!=np; ++p) {
      double max_ln_c = - std::numeric_limits<double>::max();
      for(auto& c: s.contr) {
        max_ln_c = std::max(max_ln_c, std::log(std::abs(c.coeff[p])));
      }
      s.max_ln_coeff[p] = max_ln_c;
    }
  } // shells
#endif

  libint2::BasisSet result{rshells};

  return result;
}

void ECMolden::read_geom_molden(ChemEnv& chem_env) {
  SCFOptions&        scf_options = chem_env.ioptions.scf_options;
  std::vector<Atom>& atoms       = chem_env.atoms;

  std::string line;
  auto        is = std::ifstream(scf_options.moldenfile);

  while(line.find("[Atoms]") == std::string::npos) std::getline(is, line);

  // line at [Atoms]
  for(size_t ai = 0; ai < atoms.size(); ai++) {
    std::getline(is, line);
    std::istringstream       iss(line);
    std::vector<std::string> geom{std::istream_iterator<std::string>{iss},
                                  std::istream_iterator<std::string>{}};
    atoms[ai].x = std::stod(geom[3]);
    atoms[ai].y = std::stod(geom[4]);
    atoms[ai].z = std::stod(geom[5]);
  }
}

// TODO: is this needed? - currently does not make a difference
libint2::BasisSet ECMolden::read_basis_molden(const ChemEnv& chem_env) {
  // s_type = 0, p_type = 1, d_type = 2,
  // f_type = 3, g_type = 4
  /*For spherical
  n = 2*type + 1
  s=1,p=3,d=5,f=7,g=9
  For cartesian
  n = (type+1)*(type+2)/2
  s=1,p=3,d=6,f=10,g=15
  */

  const libint2::BasisSet& shells      = chem_env.shells;
  const SCFOptions&        scf_options = chem_env.ioptions.scf_options;

  std::string line;
  auto        is = std::ifstream(scf_options.moldenfile);

  while(line.find("GTO") == std::string::npos) { std::getline(is, line); } // end basis section

  std::vector<libint2::Shell> rshells(shells.size());
  size_t                      sid = 0;
  for(auto it = shells.begin(); it != shells.end(); it++) {
    rshells[sid] = *it;
    sid++;
  }

  bool basis_parse = true;
  int  atom_i = 0, shell_i = 0;
  while(basis_parse) {
    std::getline(is, line);
    std::istringstream       iss(line);
    std::vector<std::string> expc{std::istream_iterator<std::string>{iss},
                                  std::istream_iterator<std::string>{}};
    if(expc.size() == 0) continue;
    if(expc.size() == 2) {
      // read shell nprim 0. TODO, should expc[1]==1 ?
      if(std::stoi(expc[0]) == atom_i + 1 && std::stoi(expc[1]) == 0) { atom_i++; } // shell_i=0;
    }

    else if(expc[0] == "s" || expc[0] == "p" || expc[0] == "d" || expc[0] == "f" ||
            expc[0] == "g") {
      for(auto np = 0; np < std::stoi(expc[1]); np++) {
        std::getline(is, line);
        std::istringstream       iss(line);
        std::vector<std::string> expc_val{std::istream_iterator<std::string>{iss},
                                          std::istream_iterator<std::string>{}};
        rshells[shell_i].alpha[np]          = std::stod(expc_val[0]);
        rshells[shell_i].contr[0].coeff[np] = std::stod(expc_val[1]);
      } // nprims for shell_i
      shell_i++;
    }
    else if(line.find("[5D]") != std::string::npos) basis_parse = false;
    else if(line.find("[9G]") != std::string::npos) basis_parse = false;
    else if(line.find("[MO]") != std::string::npos) basis_parse = false;

  } // end basis parse

  libint2::BasisSet result{rshells};
  return result;
}

template<typename T>
void ECMolden::read_molden(
  ChemEnv& chem_env, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C_alpha,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C_beta) {
  SystemData&              sys_data    = chem_env.sys_data;
  const libint2::BasisSet& shells      = chem_env.shells;
  SCFOptions&              scf_options = chem_env.ioptions.scf_options;

  std::ifstream is(scf_options.moldenfile);
  if(!is.is_open()) { tamm_terminate("Could not open moldenfile"); }
  std::string line;
  size_t      n_occ_alpha = 0, n_occ_beta = 0, n_vir_alpha = 0, n_vir_beta = 0;

  size_t N      = C_alpha.rows();
  size_t Northo = C_alpha.cols();

  const bool is_spherical = (scf_options.gaussian_type == "spherical");
  const bool is_rhf       = sys_data.is_restricted;
  const bool is_uhf       = sys_data.is_unrestricted;

  Matrix eigenvecs(N, Northo);
  eigenvecs.setZero();
  Matrix eigenvecs_beta(N, Northo);
  if(is_uhf) eigenvecs_beta.setZero();

  std::getline(is, line);
  while(line.find("[MO]") == std::string::npos) { std::getline(is, line); } // end basis section

  bool   mo_end       = false;
  bool   mo_end_alpha = false;
  size_t i            = 0;
  // size_t kb = 0;
  while(!mo_end) {
    std::getline(is, line);
    if(line.find("Ene=") != std::string::npos) {
      /*evl_sorted[i] =*/(std::stod(read_option(line)));
    }
    else if(line.find("Spin=") != std::string::npos) {
      std::string spinstr       = read_option(line);
      bool        is_spin_alpha = spinstr.find("Alpha") != std::string::npos;
      bool        is_spin_beta  = spinstr.find("Beta") != std::string::npos;

      // if(is_spin_alpha) n_alpha++;
      // else if(is_spin_beta) n_beta++;
      std::getline(is, line);

      if(line.find("Occup=") != std::string::npos) {
        int occup = stoi(read_option(line));
        if(is_spin_alpha) {
          if(occup == 0) n_vir_alpha++;
          if(occup == 1) n_occ_alpha++;
          if(occup == 2) {
            n_occ_alpha++;
            n_occ_beta++;
          }
        }
        else if(is_spin_beta) {
          if(occup == 1) n_occ_beta++;
          if(occup == 0) n_vir_beta++;
        }
        mo_end = true;
      }
    }

    if(mo_end) {
      // TODO: Not all coefficients might be written
      for(size_t j = 0; j < N; j++) {
        std::getline(is, line);
        if(!mo_end_alpha) eigenvecs(j, i) = std::stod(read_option(line));
        else eigenvecs_beta(j, i) = std::stod(read_option(line));
      }
      mo_end = false;
      i++;
    }

    if(i == Northo) {
      if(is_rhf) mo_end = true;
      else if(is_uhf) {
        if(!mo_end_alpha) {
          i            = 0;
          mo_end_alpha = true;
        }
        else mo_end = true;
      }
    }
  }

  reorder_molden_orbitals<T>(is_spherical, shells, eigenvecs, C_alpha, false, true);
  if(is_uhf) reorder_molden_orbitals<T>(is_spherical, shells, eigenvecs_beta, C_beta, false, true);

  if(is_rhf) {
    n_occ_beta = n_occ_alpha;
    n_vir_beta = n_vir_alpha;
  }
  // else if(scf_options.scf_type == "rohf") {
  //     n_vir_beta = N - n_occ_beta;
  // }

  std::cout << "finished reading molden: n_occ_alpha, n_vir_alpha, n_occ_beta, n_vir_beta = "
            << n_occ_alpha << "," << n_vir_alpha << "," << n_occ_beta << "," << n_vir_beta
            << std::endl;

  EXPECTS(n_occ_alpha == (size_t) sys_data.nelectrons_alpha);
  EXPECTS(n_occ_beta == (size_t) sys_data.nelectrons_beta);
  EXPECTS(n_vir_alpha == Northo - n_occ_alpha);
  EXPECTS(n_vir_beta == Northo - n_occ_beta);
}

template void ECMolden::reorder_molden_orbitals<double>(const bool               is_spherical,
                                                        const libint2::BasisSet& shells,
                                                        Matrix& smat, Matrix& dmat,
                                                        const bool reorder_cols,
                                                        const bool reorder_rows);

template void ECMolden::read_molden<double>(
  ChemEnv&                                                                chem_env,
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C_alpha,
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C_beta);
