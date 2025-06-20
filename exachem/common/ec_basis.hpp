/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "exachem/common/ecatom.hpp"

using lib_basis_set = libint2::BasisSet;
using lib_shell     = libint2::Shell;
using lib_atom      = libint2::Atom;

class ECBasis {
private:
  void basisset(ExecutionContext& exc, std::string basis, std::string gaussian_type,
                std::vector<lib_atom>& atoms, std::vector<ECAtom>& ec_atoms);
  void construct_shells(ExecutionContext& exc, std::vector<lib_atom>& atoms,
                        std::vector<ECAtom>& ec_atoms);
  bool basis_has_ecp(ExecutionContext& exc, std::string basisfile);
  void ecp_check(ExecutionContext& exc, std::string basisfile, std::vector<lib_atom>& atoms,
                 std::vector<ECAtom>& ec_atoms);
  void parse_ecp(ExecutionContext& exc, std::string basisfile, std::vector<lib_atom>& atoms,
                 std::vector<ECAtom>& ec_atoms);

public:
  ECBasis() = default;
  ECBasis(ExecutionContext& exc, std::string basis, std::string basisfile,
          std::string gaussian_type, std::vector<lib_atom>& atoms, std::vector<ECAtom>& ec_atoms);

  std::string   basis_set_file;
  lib_basis_set shells;
  bool          has_ecp{false};
};

class BasisFunctionCount {
public:
  int s_count = 0;
  int p_count = 0;
  int d_count = 0;
  int f_count = 0;
  int g_count = 0;
  int h_count = 0;
  int i_count = 0;

  int sp_count      = 0;
  int spd_count     = 0;
  int spdf_count    = 0;
  int spdfg_count   = 0;
  int spdfgh_count  = 0;
  int spdfghi_count = 0;

  // s_type = 0, p_type = 1, d_type = 2,
  // f_type = 3, g_type = 4, h_type = 5, i_type = 6
  /*For spherical
  n = 2*type + 1
  s=1,p=3,d=5,f=7,g=9,h=11,i=13
  For cartesian
  n = (type+1)*(type+2)/2
  s=1,p=3,d=6,f=10,g=15,h=21,i=28
  */
  void compute(const bool is_spherical, libint2::Shell& s);

  void print();
};

class BasisSetMap {
public:
  size_t                nbf;
  size_t                natoms;
  size_t                nshells;
  std::vector<AtomInfo> atominfo; // natoms
  // map from atoms to the corresponding shells
  std::vector<std::vector<long>> atom2shell;
  // shell2atom(nshells) - the given atom a shell is assigned to.
  std::vector<long> shell2atom;
  // bf2shell(nbf)    - the given shell a function is assigned to.
  std::vector<size_t> bf2shell;
  // shell2bf(nshells) - First basis function on a given shell.
  std::vector<size_t> shell2bf;
  // bf2atom(nbf)     - the given atom a function is assigned to.
  std::vector<size_t> bf2atom;
  // nbf_atom(natoms)     - number of basis functions on a given atom.
  std::vector<size_t> nbf_atom;
  // nshells_atom(natoms)   - number of shells on a given atom.
  std::vector<size_t> nshells_atom;
  // first_bf_atom(natoms) - First basis function on a given atom.
  std::vector<size_t> first_bf_atom;
  // first_bf_shell(nshells)    - number of basis functions on a given shell.
  std::vector<size_t> first_bf_shell;
  // first_shell_atom(natoms) - First shell on a given atom.
  std::vector<size_t> first_shell_atom;
  // gaussian basis function components
  std::map<size_t, std::string> bf_comp;
  BasisSetMap() = default;
  void operator()(std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                  bool is_spherical = true) {
    construct(atoms, shells, is_spherical);
  }
  BasisSetMap(std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
              bool is_spherical = true) {
    construct(atoms, shells, is_spherical);
  }

  void                       construct(std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                                       bool is_spherical = true);
  static std::vector<size_t> map_shell_to_basis_function(const libint2::BasisSet& shells) {
    std::vector<size_t> result;
    result.reserve(shells.size());

    size_t n = 0;
    for(auto shell: shells) {
      result.push_back(n);
      n += shell.size();
    }

    return result;
  }

  static std::vector<size_t> map_basis_function_to_shell(const libint2::BasisSet& shells) {
    std::vector<size_t> result(shells.nbf());

    auto shell2bf = BasisSetMap::map_shell_to_basis_function(shells);
    for(size_t s1 = 0; s1 != shells.size(); ++s1) {
      auto bf1_first = shell2bf[s1]; // first basis function in this shell
      auto n1        = shells[s1].size();
      for(size_t f1 = 0; f1 != n1; ++f1) {
        const auto bf1 = f1 + bf1_first;
        result[bf1]    = s1;
      }
    }
    return result;
  }
};
