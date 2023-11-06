/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include <cctype>

#include "common/cutils.hpp"
#include "common/misc.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

using namespace tamm;
using libint2::Atom;
using std::cerr;
using std::cout;
using std::endl;
using std::string;

void sinfo(std::string filename, OptionsMap options_map) {
  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  auto             rank = ec.pg().rank();

  SystemData  sys_data{options_map, options_map.scf_options.scf_type};
  SCFOptions  scf_options = sys_data.options_map.scf_options;
  std::string basis       = scf_options.basis;
  int         charge      = scf_options.charge;

  libint2::initialize(false);

  std::string basis_set_file = std::string(DATADIR) + "/basis/" + basis + ".g94";

  int basis_file_exists = 0;
  if(rank == 0) basis_file_exists = std::filesystem::exists(basis_set_file);
  ec.pg().broadcast(&basis_file_exists, 0);

  if(!basis_file_exists)
    tamm_terminate("ERROR: basis set file " + basis_set_file + " does not exist");

  auto atoms    = sys_data.options_map.options.atoms;
  auto ec_atoms = sys_data.options_map.options.ec_atoms;

  libint2::BasisSet shells;
  {
    std::vector<libint2::Shell> bset_vec;
    for(size_t i = 0; i < atoms.size(); i++) {
      // const auto        Z = atoms[i].atomic_number;
      libint2::BasisSet ashells(ec_atoms[i].basis, {atoms[i]});
      bset_vec.insert(bset_vec.end(), ashells.begin(), ashells.end());
    }
    libint2::BasisSet bset(bset_vec);
    shells = std::move(bset);
  }

  shells.set_pure(true);

  const int N = shells.nbf();

  auto nelectrons = 0;
  for(size_t i = 0; i < atoms.size(); ++i) nelectrons += atoms[i].atomic_number;
  nelectrons -= charge;

  sys_data.nelectrons = nelectrons;
  EXPECTS((nelectrons + scf_options.multiplicity - 1) % 2 == 0);

  sys_data.nelectrons_alpha = (nelectrons + scf_options.multiplicity - 1) / 2;
  sys_data.nelectrons_beta  = nelectrons - sys_data.nelectrons_alpha;

  if(rank == 0) {
    std::cout << std::endl << "Number of basis functions = " << N << std::endl;
    std::cout << std::endl << "Total number of shells = " << shells.size() << std::endl;
    std::cout << std::endl << "Total number of electrons = " << nelectrons << std::endl;
    std::cout << "  # of alpha electrons    = " << sys_data.nelectrons_alpha << std::endl;
    std::cout << "  # of beta electons      = " << sys_data.nelectrons_beta << std::endl;
  }

  if(rank == 0) write_sinfo(sys_data, shells);

  ec.flush_and_sync();
}
