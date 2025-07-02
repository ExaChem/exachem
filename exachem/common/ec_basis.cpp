/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/ec_basis.hpp"
#include <unordered_set>
namespace fs = std::filesystem;

// ca_symbol represents each unique atom symbol in the geometry
void ECBasis::parse_ecp_basis_file(ExecutionContext& exc, std::string ca_symbol,
                                   std::vector<Atom>& atoms, std::vector<ECAtom>& ec_atoms,
                                   const std::string ecp_basisfile) {
  std::ifstream is(ecp_basisfile);
  std::string   line, rest;
  bool          ecp_found{false};

  do {
    if(line.find("ECP") != std::string::npos) {
      ecp_found = true;
      break;
    }
  } while(std::getline(is, line));

  if(!ecp_found) {
    std::string bfnf_msg = "ECP block not found in " + ecp_basisfile;
    tamm_terminate(bfnf_msg);
  }

  int              nelec = 0;
  std::string      amlabel;
  std::vector<int> atomlist;
  const int        ca_z = ec_atoms[0].get_atomic_number(ca_symbol);

  std::ostringstream oss;
  bool               atom_has_ecp = false;

  while(std::getline(is, line)) {
    if(line.find("END") != std::string::npos) break;

    std::istringstream iss{line};
    std::string        word;
    int                count = 0;
    while(iss >> word) count++;
    if(count == 0) continue;

    std::string elemsymbol;

    std::istringstream iss_{line};
    iss_ >> elemsymbol;

    // true if line starts with element symbol
    bool is_es = elemsymbol.find_first_not_of("0123456789") != std::string::npos;
    // TODO: Check if its valid symbol

    if(is_es && count == 3) {
      atom_has_ecp = false;
      std::string nelec_str;
      iss_ >> nelec_str >> nelec;
      atomlist.clear();
      const int es_z = ec_atoms[0].get_atomic_number(elemsymbol);

      // handle all atoms with the same element symbol represented by ca_symbol
      for(size_t i = 0; i < ec_atoms.size(); i++) {
        if(es_z == ca_z && // make sure ca_symbol atomic number matches one in ECP basis file
           txt_utils::strequal_case(ca_symbol, ec_atoms[i].esymbol)) // match labeled atom symbols
        {
          if(ec_atoms[i].has_ecp) continue;
          atom_has_ecp          = true;
          has_ecp               = true;
          ec_atoms[i].has_ecp   = true;
          ec_atoms[i].ecp_nelec = nelec;
          atoms[i].atomic_number -= nelec;
          atomlist.push_back(i);
          // iatom = i;
          // break;
        }
      }
      if(atom_has_ecp) oss << line << std::endl;
      continue;
    }
    if(is_es && count == 2) {
      iss_ >> amlabel;
      if(atom_has_ecp) oss << line << std::endl;
      continue;
    }
    if(!is_es && count == 3) {
      int am = 0;
      if(amlabel == "ul") am = -1;
      else am = lib_shell::am_symbol_to_l(amlabel[0]);

      int    ns = stoi(elemsymbol);
      double _exp;
      double _coef;
      iss_ >> _exp >> _coef;
      for(auto& iatom: atomlist) {
        ec_atoms[iatom].ecp_ams.push_back(am);
        ec_atoms[iatom].ecp_ns.push_back(ns);
        ec_atoms[iatom].ecp_exps.push_back(_exp);
        ec_atoms[iatom].ecp_coeffs.push_back(_coef);
      }
      if(atom_has_ecp) oss << line << std::endl;
    }
  }
  while(std::getline(is, line))
    ;
  if(exc.print())
    std::cout << std::endl << "ECP" << std::endl << oss.str() << "END" << std::endl << std::endl;
}

void print_basis_info(const std::vector<libint2::Atom>& atoms, const std::vector<ECAtom>& ec_atoms,
                      const libint2::BasisSet& basis) {
  std::set<std::pair<std::string, std::string>> unique_sym_basis;
  auto                                          a2s_map   = basis.atom2shell(atoms);
  const auto                                    elem_info = libint2::chemistry::get_element_info();

  for(size_t i = 0; i < ec_atoms.size(); ++i) {
    const std::string                   ec_symbol = ec_atoms[i].esymbol;
    std::pair<std::string, std::string> key       = {ec_symbol, ec_atoms[i].basis};
    if(unique_sym_basis.count(key)) continue;
    unique_sym_basis.insert(key);

    int Z = ec_atoms[i].atom.atomic_number;
    if(ec_atoms[i].is_bq) { Z = ECAtom::get_atomic_number(ec_symbol); }
    const std::string ename = ec_symbol + " (" + elem_info[Z - 1].name + ")";
    std::cout << "  " << ename << std::endl;
    std::cout << "  " << std::string(ename.length(), '-') << std::endl;
    std::cout << std::setw(18) << std::right << "Exponent" << std::setw(20) << "Coefficients"
              << std::endl;
    std::cout << std::string(6, ' ') << std::setw(16) << std::right << std::string(16, '-') << "  "
              << std::setw(20) << std::string(20, '-') << std::endl;

    // Get shells centered on this atom
    auto nshells     = a2s_map[i].size();
    auto shell_start = a2s_map[i][0];
    int  shell_count = 1;

    for(size_t s = shell_start; s < shell_start + nshells; ++s) {
      const auto& shell = basis[s];

      for(size_t c = 0; c < shell.contr.size(); ++c) {
        const auto& contr = shell.contr[c];
        std::string l_str;
        switch(contr.l) {
          case 0: l_str = "S"; break;
          case 1: l_str = "P"; break;
          case 2: l_str = "D"; break;
          case 3: l_str = "F"; break;
          case 4: l_str = "G"; break;
          case 5: l_str = "H"; break;
          case 6: l_str = "I"; break;
          case 7: l_str = "K"; break;
          default: l_str = "L" + std::to_string(contr.l); break;
        }

        for(size_t p = 0; p < shell.alpha.size(); ++p) {
          std::cout << "  " << shell_count << " " << l_str << "  " << std::uppercase
                    << std::scientific << std::setprecision(8) << std::setw(13) << shell.alpha[p]
                    << "  " << std::fixed << std::setprecision(6) << std::right << std::setw(12)
                    << contr.coeff[p] << std::endl;
        }
        shell_count++;

        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }
}

void check_basis_file(ExecutionContext& exc, bool single_basis, const ECAtom& ecatom,
                      bool ecp_check = false) {
  auto basis_set_file = std::string(DATADIR) + "/basis/" + ecatom.basis + ".g94";
  if(ecp_check) basis_set_file = std::string(DATADIR) + "/basis/" + ecatom.ecp_basis + ".ecp";
  int basis_file_exists = 0;
  if(exc.pg().rank() == 0) basis_file_exists = std::filesystem::exists(basis_set_file);
  exc.pg().broadcast(&basis_file_exists, 0);

  if(!basis_file_exists) {
    std::string err_msg = "ERROR: ";
    if(!single_basis) err_msg += "Atom " + ecatom.esymbol + " - ";
    err_msg += "basis set file " + basis_set_file + " specified does not exist.";
    tamm_terminate(err_msg);
  }
}

void ECBasis::parse_ecps(ExecutionContext& exc, std::vector<lib_atom>& atoms,
                         std::vector<ECAtom>& ec_atoms) {
  std::cout << std::endl;
  std::map<std::string, ECAtom> unique_atom_labels;
  for(auto& ecatom: ec_atoms) { unique_atom_labels.emplace(ecatom.esymbol, ecatom); }
  for(const auto& [esymbol, ecatom]: unique_atom_labels) {
    if(ecatom.ecp_basis.empty()) continue;
    // There is no global ecp basis specification for all atoms.
    // single_basis=false, ecp=true
    check_basis_file(exc, false, ecatom, true);
    auto ecp_basis_file = std::string(DATADIR) + "/basis/" + ecatom.ecp_basis + ".ecp";
    if(exc.pg().rank() == 0) std::cout << "Parsing ECP block in " << ecp_basis_file << std::endl;
    parse_ecp_basis_file(exc, esymbol, atoms, ec_atoms, ecp_basis_file);
  }
}

void ECBasis::construct_shells(ExecutionContext& exc, std::vector<lib_atom>& atoms,
                               std::vector<ECAtom>& ec_atoms) {
  std::vector<lib_shell> shell_vec;

  const bool single_basis =
    std::all_of(ec_atoms.begin() + 1, ec_atoms.end(),
                [&ec_atoms](const ECAtom& s) { return s.basis == ec_atoms[0].basis; });

  for(size_t i = 0; i < atoms.size(); i++) {
    if(!single_basis || i == 0) { check_basis_file(exc, single_basis, ec_atoms[i]); }

    // const auto        Z = atoms[i].atomic_number;
    lib_basis_set ashells(ec_atoms[i].basis, {atoms[i]}, true);
    shell_vec.insert(shell_vec.end(), ashells.begin(), ashells.end());
  }

  shells = lib_basis_set(shell_vec);
}

/* Constructors and initializer */

void ECBasis::basisset(ExecutionContext& exc, std::string basis, std::string gaussian_type,
                       std::vector<lib_atom>& atoms, std::vector<ECAtom>& ec_atoms) {
  // Initialize the Libint integrals library
  libint2::initialize(false);
  construct_shells(exc, atoms, ec_atoms);
  if(gaussian_type == "spherical") shells.set_pure(true);
  else shells.set_pure(false); // use cartesian gaussians
  // libint2::Shell::do_enforce_unit_normalization(false);
  if(exc.print()) print_basis_info(atoms, ec_atoms, shells);
}

ECBasis::ECBasis(ExecutionContext& exc, std::string basis, std::string gaussian_type,
                 std::vector<lib_atom>& atoms, std::vector<ECAtom>& ec_atoms) {
  basisset(exc, basis, gaussian_type, atoms, ec_atoms);
  parse_ecps(exc, atoms, ec_atoms);
}

void BasisFunctionCount::compute(const bool is_spherical, libint2::Shell& s) {
  // for (const auto& s: shells) {
  for(const auto& c: s.contr) {
    const int n_sph  = 2 * c.l + 1;
    const int n_cart = ((c.l + 1) * (c.l + 2)) / 2;
    if(c.l == 0) s_count += n_sph;
    if(c.l == 1) p_count += n_sph;
    if(c.l == 2) {
      if(is_spherical) d_count += n_sph;
      else d_count += n_cart;
    }
    if(c.l == 3) {
      if(is_spherical) f_count += n_sph;
      else f_count += n_cart;
    }
    if(c.l == 4) {
      if(is_spherical) g_count += n_sph;
      else g_count += n_cart;
    }
    if(c.l == 5) {
      if(is_spherical) h_count += n_sph;
      else h_count += n_cart;
    }
    if(c.l == 6) {
      if(is_spherical) i_count += n_sph;
      else i_count += n_cart;
    }
  }
  // }

  sp_count      = s_count + p_count;
  spd_count     = sp_count + d_count;
  spdf_count    = spd_count + f_count;
  spdfg_count   = spdf_count + g_count;
  spdfgh_count  = spdfg_count + h_count;
  spdfghi_count = spdfgh_count + i_count;
}

void BasisFunctionCount::print() {
  std::cout << "s = " << s_count << ", p = " << p_count << ", d = " << d_count
            << ", f = " << f_count << ", g = " << g_count << ", h = " << h_count << std::endl;
  std::cout << "sp = " << sp_count << ", spd = " << spd_count << ", spdf = " << spdf_count
            << ", spdfg = " << spdfg_count << ", spdfgh = " << spdfgh_count << std::endl;
}

void BasisSetMap::construct(std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                            bool is_spherical) {
  auto a2s_map = shells.atom2shell(atoms);
  natoms       = atoms.size();
  nshells      = shells.size();
  nbf          = shells.nbf();

  std::vector<long> shell2atom_map = shells.shell2atom(atoms);
  bf2shell                         = BasisSetMap::map_basis_function_to_shell(shells);
  shell2bf                         = BasisSetMap::map_shell_to_basis_function(shells);

  atominfo.resize(natoms);
  bf2atom.resize(nbf);
  nbf_atom.resize(natoms);
  nshells_atom.resize(natoms);
  first_bf_atom.resize(natoms);
  first_bf_shell.resize(nshells);
  first_shell_atom.resize(natoms);

  for(size_t s1 = 0; s1 != nshells; ++s1) first_bf_shell[s1] = shells[s1].size();

  std::map<int, std::string>              gaus_comp_map{{0, "s"}, {1, "p"}, {2, "d"}, {3, "f"},
                                           {4, "g"}, {5, "h"}, {6, "i"}};
  std::map<int, std::vector<std::string>> cart_comp_map{
    {1, {"x", "y", "z"}},
    {2, {"xx", "xy", "xz", "yy", "yz", "zz"}},
    {3, {"xxx", "xxy", "xxz", "xyy", "xyz", "xzz", "yyy", "yyz", "yzz", "zzz"}},
    {4,
     {"xxxx", "xxxy", "xxxz", "xxyy", "xxyz", "xxzz", "xyyy", "xyyz", "xyzz", "xzzz", "yyyy",
      "yyyz", "yyzz", "yzzz", "zzzz"}},
    {5, {"xxxxx", "xxxxy", "xxxxz", "xxxyy", "xxxyz", "xxxzz", "xxyyy",
         "xxyyz", "xxyzz", "xxzzz", "xyyyy", "xyyyz", "xyyzz", "xyzzz",
         "xzzzz", "yyyyy", "yyyyz", "yyyzz", "yyzzz", "yzzzz", "zzzzz"}},
    {6, {"xxxxxx", "xxxxxy", "xxxxxz", "xxxxyy", "xxxxyz", "xxxxzz", "xxxyyy",
         "xxxyyz", "xxxyzz", "xxxzzz", "xxyyyy", "xxyyyz", "xxyyzz", "xxyzzz",
         "xxzzzz", "xyyyyy", "xyyyyz", "xyyyzz", "xyyzzz", "xyzzzz", "xzzzzz",
         "yyyyyy", "yyyyyz", "yyyyzz", "yyyzzz", "yyzzzz", "yzzzzz", "zzzzzz"}}};

  for(size_t ai = 0; ai < natoms; ai++) {
    auto                        nshells_ai = a2s_map[ai].size();
    auto                        first      = a2s_map[ai][0];
    auto                        last       = a2s_map[ai][nshells_ai - 1];
    std::vector<libint2::Shell> atom_shells(nshells_ai);
    int                         as_index = 0;
    size_t                      atom_nbf = 0;
    first_shell_atom[ai]                 = first;
    for(auto si = first; si <= last; si++) {
      atom_shells[as_index] = shells[si];
      as_index++;
      atom_nbf += shells[si].size();
    }
    for(const auto& e: libint2::chemistry::get_element_info()) {
      if(e.Z == atoms[ai].atomic_number) {
        atominfo[ai].symbol = e.symbol;
        break;
      }
    }
    atominfo[ai].atomic_number = atoms[ai].atomic_number;
    atominfo[ai].shells        = atom_shells;
    atominfo[ai].nbf           = atom_nbf;
    atominfo[ai].nbf_lo        = 0;
    atominfo[ai].nbf_hi        = atom_nbf;
    if(ai > 0) {
      atominfo[ai].nbf_lo = atominfo[ai - 1].nbf_hi;
      atominfo[ai].nbf_hi = atominfo[ai].nbf_lo + atom_nbf;
    }

    nbf_atom[ai]      = atom_nbf;
    nshells_atom[ai]  = nshells_ai;
    first_bf_atom[ai] = atominfo[ai].nbf_lo;
    for(auto nlo = atominfo[ai].nbf_lo; nlo < atominfo[ai].nbf_hi; nlo++) bf2atom[nlo] = ai;

    int alo = atominfo[ai].nbf_lo;
    for(auto s: atominfo[ai].shells) {
      auto l = s.contr[0].l;
      if(is_spherical) {
        for(int i = 0; i < 2 * l + 1; i++) {
          std::stringstream tmps;
          if(l == 0) tmps << "";
          else if(l == 1) {
            if(i == 0) tmps << "_y";
            else if(i == 1) tmps << "_z";
            else if(i == 2) tmps << "_x";
          }
          else if(l <= 6) tmps << std::showpos << i - l;
          else NOT_IMPLEMENTED();
          bf_comp[alo] = gaus_comp_map[l] + tmps.str();
          alo++;
        }
      }
      else { // cartesian
        const auto ncfuncs  = ((l + 1) * (l + 2)) / 2;
        auto       cart_vec = cart_comp_map[l];
        for(int i = 0; i < ncfuncs; i++) {
          std::string tmps;
          if(l == 0) tmps = "";
          else if(l <= 6) tmps = "_" + cart_vec[i];
          else NOT_IMPLEMENTED();
          bf_comp[alo] = gaus_comp_map[l] + tmps;
          alo++;
        }
      }
    }
  }

  atom2shell = a2s_map;
  shell2atom = shell2atom_map;
}
