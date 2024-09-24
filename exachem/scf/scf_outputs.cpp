/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_outputs.hpp"
#include "exachem/common/options/parser_utils.hpp"
#include "exachem/scf/scf_matrix.hpp"

template<typename TensorType>
double exachem::scf::SCFIO::tt_trace(ExecutionContext& ec, Tensor<TensorType>& T1,
                                     Tensor<TensorType>& T2) {
  Tensor<TensorType> tensor{T1.tiled_index_spaces()}; //{tAO, tAO};
  Tensor<TensorType>::allocate(&ec, tensor);
  const TiledIndexSpace tis_ao = T1.tiled_index_spaces()[0];
  auto [mu, nu, ku]            = tis_ao.labels<3>("all");
  Scheduler{ec}(tensor(mu, nu) = T1(mu, ku) * T2(ku, nu)).execute();
  double trace = tamm::trace(tensor);
  Tensor<TensorType>::deallocate(tensor);
  return trace;
}

void exachem::scf::SCFIO::print_energies(ExecutionContext& ec, ChemEnv& chem_env,
                                         TAMMTensors& ttensors, EigenTensors& etensors,
                                         SCFVars& scf_vars, ScalapackInfo& scalapack_info) {
  const SystemData& sys_data = chem_env.sys_data;

  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;
  const bool is_ks  = sys_data.is_ks;

  double nelectrons = 0.0;
  double kinetic_1e = 0.0;
  double NE_1e      = 0.0;
  double energy_1e  = 0.0;
  double energy_2e  = 0.0;

  if(is_rhf) {
    nelectrons = tt_trace(ec, ttensors.D_last_alpha, ttensors.S1);
    kinetic_1e = tt_trace(ec, ttensors.D_last_alpha, ttensors.T1);
    NE_1e      = tt_trace(ec, ttensors.D_last_alpha, ttensors.V1);
    energy_1e  = tt_trace(ec, ttensors.D_last_alpha, ttensors.H1);
    energy_2e  = 0.5 * tt_trace(ec, ttensors.D_last_alpha, ttensors.F_alpha_tmp);

    if(is_ks) { energy_2e += scf_vars.exc; }
  }
  if(is_uhf) {
    nelectrons = tt_trace(ec, ttensors.D_last_alpha, ttensors.S1);
    kinetic_1e = tt_trace(ec, ttensors.D_last_alpha, ttensors.T1);
    NE_1e      = tt_trace(ec, ttensors.D_last_alpha, ttensors.V1);
    energy_1e  = tt_trace(ec, ttensors.D_last_alpha, ttensors.H1);
    energy_2e  = 0.5 * tt_trace(ec, ttensors.D_last_alpha, ttensors.F_alpha_tmp);
    nelectrons += tt_trace(ec, ttensors.D_last_beta, ttensors.S1);
    kinetic_1e += tt_trace(ec, ttensors.D_last_beta, ttensors.T1);
    NE_1e += tt_trace(ec, ttensors.D_last_beta, ttensors.V1);
    energy_1e += tt_trace(ec, ttensors.D_last_beta, ttensors.H1);
    energy_2e += 0.5 * tt_trace(ec, ttensors.D_last_beta, ttensors.F_beta_tmp);
    if(is_ks) { energy_2e += scf_vars.exc; }
  }

  if(ec.pg().rank() == 0) {
    std::cout << "#electrons        = " << (int) std::round(nelectrons) << endl;
    std::cout << "1e energy kinetic = " << std::setprecision(16) << kinetic_1e << endl;
    std::cout << "1e energy N-e     = " << NE_1e << endl;
    std::cout << "1e energy         = " << energy_1e << endl;
    std::cout << "2e energy         = " << energy_2e << std::endl;

    chem_env.sys_data.results["output"]["SCF"]["NE_1e"]      = NE_1e;
    chem_env.sys_data.results["output"]["SCF"]["kinetic_1e"] = kinetic_1e;
    chem_env.sys_data.results["output"]["SCF"]["energy_1e"]  = energy_1e;
    chem_env.sys_data.results["output"]["SCF"]["energy_2e"]  = energy_2e;
  }
}

void exachem::scf::SCFIO::print_mulliken(ChemEnv& chem_env, Matrix& D, Matrix& D_beta, Matrix& S) {
  std::vector<Atom>&   atoms    = chem_env.atoms;
  std::vector<ECAtom>& ec_atoms = chem_env.ec_atoms;
  libint2::BasisSet&   shells   = chem_env.shells;
  bool                 is_uhf   = chem_env.sys_data.is_unrestricted;

  BasisSetMap bsm(atoms, shells);

  const int                        natoms = ec_atoms.size();
  std::vector<double>              cs_acharge(natoms, 0);
  std::vector<double>              os_acharge(natoms, 0);
  std::vector<double>              net_acharge(natoms, 0);
  std::vector<std::vector<double>> cs_charge_shell(natoms);
  std::vector<std::vector<double>> os_charge_shell(natoms);
  std::vector<std::vector<double>> net_charge_shell(natoms);

  int j = 0;
  for(auto x = 0; x < natoms; x++) { // loop over atoms
    auto atom_shells = bsm.atominfo[x].shells;
    auto nshells     = atom_shells.size(); // #shells for atom x
    cs_charge_shell[x].resize(nshells);
    if(is_uhf) os_charge_shell[x].resize(nshells);
    for(size_t s = 0; s < nshells; s++) { // loop over each shell for atom x
      double cs_scharge = 0.0, os_scharge = 0.0;
      for(size_t si = 0; si < atom_shells[s].size(); si++) {
        for(Eigen::Index i = 0; i < S.rows(); i++) {
          const auto ds_cs = D(j, i) * S(j, i);
          cs_scharge += ds_cs;
          cs_acharge[x] += ds_cs;
          if(is_uhf) {
            const auto ds_os = D_beta(j, i) * S(j, i);
            os_scharge += ds_os;
            os_acharge[x] += ds_os;
          }
        }
        j++;
      }
      cs_charge_shell[x][s] = cs_scharge;
      if(is_uhf) os_charge_shell[x][s] = os_scharge;
    }
  }

  net_acharge      = cs_acharge;
  net_charge_shell = cs_charge_shell;
  if(is_uhf) {
    for(auto x = 0; x < natoms; x++) { // loop over atoms
      net_charge_shell[x].resize(cs_charge_shell[x].size());
      net_acharge[x] = cs_acharge[x] + os_acharge[x];
      std::transform(cs_charge_shell[x].begin(), cs_charge_shell[x].end(),
                     os_charge_shell[x].begin(), net_charge_shell[x].begin(), std::plus<double>());
    }
  }
  const auto mksp = std::string(5, ' ');

  auto print_ma = [&](const std::string dtype, std::vector<double>& acharge,
                      std::vector<std::vector<double>>& charge_shell) {
    std::cout << std::endl
              << mksp << "Mulliken analysis of the " << dtype << " density" << std::endl;
    std::cout << mksp << std::string(50, '-') << std::endl << std::endl;
    std::cout << mksp << "   Atom   " << mksp << " Charge " << mksp << "  Shell Charges  "
              << std::endl;
    std::cout << mksp << "----------" << mksp << "--------" << mksp << std::string(50, '-')
              << std::endl;

    for(int x = 0; x < natoms; x++) { // loop over atoms
      const auto Z        = ec_atoms[x].atom.atomic_number;
      const auto e_symbol = ec_atoms[x].esymbol;
      std::cout << mksp << std::setw(3) << std::right << x + 1 << " " << std::left << std::setw(2)
                << e_symbol << " " << std::setw(3) << std::right << Z << mksp << std::fixed
                << std::setprecision(2) << std::right << " " << std::setw(5) << acharge[x] << mksp
                << "  " << std::right;
      for(auto csx: charge_shell[x]) std::cout << std::setw(5) << csx << " ";
      std::cout << std::endl;
    }
  };
  print_ma("total", net_acharge, net_charge_shell);
  if(is_uhf) {
    print_ma("alpha", cs_acharge, cs_charge_shell);
    print_ma("beta", os_acharge, os_charge_shell);
  }
}

template<typename T>
void exachem::scf::SCFIO::rw_mat_disk(Tensor<T> tensor, std::string tfilename, bool profile,
                                      bool read) {
#if !defined(USE_UPCXX)
  if(read) read_from_disk<T>(tensor, tfilename, true, {}, profile);
  else write_to_disk<T>(tensor, tfilename, true, profile);
#else
  //   SCFMatrix scf_matrix;
  if((tensor.execution_context())->pg().rank() == 0) {
    if(read) {
      Matrix teig = read_scf_mat<T>(tfilename);
      eigen_to_tamm_tensor(tensor, teig);
    }
    else {
      Matrix teig = tamm_to_eigen_matrix(tensor);
      write_scf_mat<T>(teig, tfilename);
    }
  }
  tensor.execution_context()->pg().barrier();
#endif
}

void exachem::scf::SCFIO::rw_md_disk(ExecutionContext& ec, const ChemEnv& chem_env,
                                     ScalapackInfo& scalapack_info, TAMMTensors& ttensors,
                                     EigenTensors& etensors, std::string files_prefix, bool read) {
  const auto rank   = ec.pg().rank();
  const bool is_uhf = chem_env.sys_data.is_unrestricted;
  auto       debug  = chem_env.ioptions.scf_options.debug;

  std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
  std::string densityfile_alpha = files_prefix + ".alpha.density";
  std::string movecsfile_beta   = files_prefix + ".beta.movecs";
  std::string densityfile_beta  = files_prefix + ".beta.density";

  if(!read) {
#if defined(USE_SCALAPACK)
    if(scalapack_info.comm != MPI_COMM_NULL) {
      tamm::from_block_cyclic_tensor(ttensors.C_alpha_BC, ttensors.C_alpha);
      if(is_uhf) tamm::from_block_cyclic_tensor(ttensors.C_beta_BC, ttensors.C_beta);
    }
#else
    if(rank == 0) {
      eigen_to_tamm_tensor(ttensors.C_alpha, etensors.C_alpha);
      if(is_uhf) eigen_to_tamm_tensor(ttensors.C_beta, etensors.C_beta);
    }
#endif
    ec.pg().barrier();
  }

  std::vector<Tensor<TensorType>> tensor_list{ttensors.C_alpha, ttensors.D_alpha};
  std::vector<std::string>        tensor_fnames{movecsfile_alpha, densityfile_alpha};
  if(chem_env.sys_data.is_unrestricted) {
    tensor_list.insert(tensor_list.end(), {ttensors.C_beta, ttensors.D_beta});
    tensor_fnames.insert(tensor_fnames.end(), {movecsfile_beta, densityfile_beta});
  }

#if !defined(USE_UPCXX)
  if(read) {
    if(rank == 0) cout << "Reading movecs and density files from disk ... ";
    read_from_disk_group(ec, tensor_list, tensor_fnames, debug);
    if(rank == 0) cout << "done" << endl;
  }
  else write_to_disk_group(ec, tensor_list, tensor_fnames, debug);

#else
  if(read) {
    if(rank == 0) cout << "Reading movecs and density files from disk ... ";
    rw_mat_disk(ttensors.C_alpha, movecsfile_alpha, debug, true);
    rw_mat_disk(ttensors.D_alpha, densityfile_alpha, debug, true);
    if(is_uhf) {
      rw_mat_disk(ttensors.C_beta, movecsfile_beta, debug, true);
      rw_mat_disk(ttensors.D_beta, densityfile_beta, debug, true);
    }
    if(rank == 0) cout << "done" << endl;
  }
  else {
    rw_mat_disk(ttensors.C_alpha, movecsfile_alpha, debug);
    rw_mat_disk(ttensors.D_alpha, densityfile_alpha, debug);
    if(is_uhf) {
      rw_mat_disk(ttensors.C_beta, movecsfile_beta, debug);
      rw_mat_disk(ttensors.D_beta, densityfile_beta, debug);
    }
  }
#endif
}

template void exachem::scf::SCFIO::rw_mat_disk<double>(Tensor<double> tensor, std::string tfilename,
                                                       bool profile, bool read);
template double exachem::scf::SCFIO::tt_trace<double>(ExecutionContext& ec, Tensor<TensorType>& T1,
                                                      Tensor<TensorType>& T2);
