/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_outputs.hpp"
#include "exachem/common/options/parser_utils.hpp"

template<typename T>
double exachem::scf::DefaultSCFIO<T>::tt_trace(ExecutionContext& ec, Tensor<T>& T1, Tensor<T>& T2) {
  Tensor<T> tensor{T1.tiled_index_spaces()};
  Tensor<T>::allocate(&ec, tensor);
  const TiledIndexSpace tis_ao = T1.tiled_index_spaces()[0];
  auto [mu, nu, ku]            = tis_ao.labels<3>("all");
  Scheduler{ec}(tensor(mu, nu) = T1(mu, ku) * T2(ku, nu)).execute();
  double trace = tamm::trace(tensor);
  Tensor<T>::deallocate(tensor);
  return trace;
}

template<typename T>
Matrix exachem::scf::DefaultSCFIO<T>::read_scf_mat(const std::string& matfile) {
  std::string mname = fs::path(matfile).extension();
  mname.erase(0, 1); // remove "."

  auto mfile_id = H5Fopen(matfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Read attributes - reduced dims
  std::vector<int64_t> rdims(2);
  auto                 attr_dataset = H5Dopen(mfile_id, "rdims", H5P_DEFAULT);
  H5Dread(attr_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdims.data());

  Matrix mat         = Matrix::Zero(rdims[0], rdims[1]);
  auto   mdataset_id = H5Dopen(mfile_id, mname.c_str(), H5P_DEFAULT);

  /* Read the datasets. */
  H5Dread(mdataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data());

  H5Dclose(attr_dataset);
  H5Dclose(mdataset_id);
  H5Fclose(mfile_id);

  return mat;
}

template<typename T>
void exachem::scf::DefaultSCFIO<T>::write_scf_mat(Matrix& C, const std::string& matfile) {
  std::string mname = fs::path(matfile).extension();
  mname.erase(0, 1); // remove "."

  const auto  N      = C.rows();
  const auto  Northo = C.cols();
  TensorType* buf    = C.data();

  /* Create a file. */
  hid_t file_id = H5Fcreate(matfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t tsize        = N * Northo;
  hid_t   dataspace_id = H5Screate_simple(1, &tsize, NULL);

  /* Create dataset. */
  hid_t dataset_id = H5Dcreate(file_id, mname.c_str(), get_hdf5_dt<T>(), dataspace_id, H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
  /* Write the dataset. */
  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

  /* Create and write attribute information - dims */
  std::vector<int64_t> rdims{N, Northo};
  hsize_t              attr_size      = rdims.size();
  auto                 attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  auto attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT64, attr_dataspace, H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
}

template<typename T>
void exachem::scf::DefaultSCFIO<T>::print_energies(ExecutionContext& ec, ChemEnv& chem_env,
                                                   TAMMTensors<T>& ttensors, EigenTensors& etensors,
                                                   SCFData&       scf_data,
                                                   ScalapackInfo& scalapack_info) {
  const SystemData& sys_data    = chem_env.sys_data;
  const SCFOptions& scf_options = chem_env.ioptions.scf_options;

  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;
  const bool is_ks  = sys_data.is_ks;
  const bool is_qed = sys_data.is_qed;
  const bool do_qed = sys_data.do_qed;

  double nelectrons    = 0.0;
  double kinetic_1e    = 0.0;
  double NE_1e         = 0.0;
  double energy_1e     = 0.0;
  double energy_2e     = 0.0;
  double energy_qed    = 0.0;
  double energy_qed_et = 0.0;
  if(is_rhf) {
    nelectrons = tt_trace(ec, ttensors.D_last_alpha, ttensors.S1);
    kinetic_1e = tt_trace(ec, ttensors.D_last_alpha, ttensors.T1);
    NE_1e      = tt_trace(ec, ttensors.D_last_alpha, ttensors.V1);
    energy_1e  = tt_trace(ec, ttensors.D_last_alpha, ttensors.H1);
    energy_2e  = 0.5 * tt_trace(ec, ttensors.D_last_alpha, ttensors.F_alpha_tmp);

    if(is_ks) {
      if((!is_qed) || (is_qed && do_qed)) { energy_2e += scf_data.exc; }
    }

    if(is_qed) {
      if(do_qed) {
        energy_qed = tt_trace(ec, ttensors.D_last_alpha, ttensors.QED_1body);
        energy_qed += 0.5 * tt_trace(ec, ttensors.D_last_alpha, ttensors.QED_2body);
      }
      else { energy_qed = scf_data.eqed; }

      Tensor<double> X_a;

#if defined(USE_SCALAPACK)
      X_a = {scf_data.tAO, scf_data.tAO_ortho};
      Tensor<double>::allocate(&ec, X_a);
      if(scalapack_info.pg.is_valid()) { tamm::from_block_cyclic_tensor(ttensors.X_alpha, X_a); }
#else
      X_a = ttensors.X_alpha;
#endif

      energy_qed_et              = 0.0;
      std::vector<double> polvec = {0.0, 0.0, 0.0};
      Scheduler           sch{ec};
      Tensor<double>      ehf_tmp{scf_data.tAO, scf_data.tAO};
      Tensor<double>      QED_Qxx{scf_data.tAO, scf_data.tAO};
      Tensor<double>      QED_Qyy{scf_data.tAO, scf_data.tAO};
      sch.allocate(ehf_tmp, QED_Qxx, QED_Qyy).execute();

      for(int i = 0; i < sys_data.qed_nmodes; i++) {
        polvec = scf_options.qed_polvecs[i];

        // clang-format off
        sch
          (ehf_tmp("i", "j")  = X_a("i", "k") * X_a("j", "k"))
          (ehf_tmp("i", "j") -= 0.5 * ttensors.D_last_alpha("i", "j"))
          (QED_Qxx("i", "j")  = polvec[0] * ttensors.QED_Dx("i", "j"))
          (QED_Qxx("i", "j") += polvec[1] * ttensors.QED_Dy("i", "j"))
          (QED_Qxx("i", "j") += polvec[2] * ttensors.QED_Dz("i", "j"))
          (QED_Qyy("i", "j")  = QED_Qxx("i", "k") * ehf_tmp("k", "j"))
          (ehf_tmp("i", "j")  = QED_Qyy("i", "k") * QED_Qxx("k", "j"))
          .execute();
        // clang-format on

        const double coupl_strength = pow(scf_options.qed_lambdas[i], 2);
        energy_qed_et +=
          0.5 * coupl_strength * tt_trace(ec, ttensors.D_last_alpha, ttensors.ehf_tmp);
      }
      sch.deallocate(ehf_tmp, QED_Qxx, QED_Qyy).execute();

#if defined(USE_SCALAPACK)
      Tensor<double>::deallocate(X_a);
#endif
    }
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
    if(is_ks) { energy_2e += scf_data.exc; }
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

    if(is_qed) {
      std::cout << "QED energy        = " << energy_qed << std::endl;
      std::cout << "QED eT energy     = " << energy_qed_et << std::endl;
      chem_env.sys_data.results["output"]["SCF"]["energy_qed"]    = energy_qed;
      chem_env.sys_data.results["output"]["SCF"]["energy_qed_et"] = energy_qed_et;
    }
  }
}

template<typename T>
void exachem::scf::DefaultSCFIO<T>::print_mulliken(ChemEnv& chem_env, Matrix& D, Matrix& D_beta,
                                                   Matrix& S) {
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
void exachem::scf::DefaultSCFIO<T>::rw_mat_disk(Tensor<T> tensor, std::string tfilename,
                                                bool profile, bool read) {
#if !defined(USE_SERIAL_IO)
  if(read) read_from_disk<T>(tensor, tfilename, true, {}, profile);
  else write_to_disk<T>(tensor, tfilename, true, profile);
#else
  if((tensor.execution_context())->pg().rank() == 0) {
    if(read) {
      Matrix teig = this->read_scf_mat(tfilename);
      eigen_to_tamm_tensor(tensor, teig);
    }
    else {
      Matrix teig = tamm_to_eigen_matrix(tensor);
      this->write_scf_mat(teig, tfilename);
    }
  }
  tensor.execution_context()->pg().barrier();
#endif
}

template<typename T>
void exachem::scf::DefaultSCFIO<T>::rw_md_disk(ExecutionContext& ec, const ChemEnv& chem_env,
                                               ScalapackInfo&  scalapack_info,
                                               TAMMTensors<T>& ttensors, EigenTensors& etensors,
                                               std::string files_prefix, bool read) {
  const auto rank   = ec.pg().rank();
  const bool is_uhf = chem_env.sys_data.is_unrestricted;
  auto       debug  = chem_env.ioptions.scf_options.debug;

  std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
  std::string densityfile_alpha = files_prefix + ".alpha.density";
  std::string movecsfile_beta   = files_prefix + ".beta.movecs";
  std::string densityfile_beta  = files_prefix + ".beta.density";

  if(!read) {
#if defined(USE_SCALAPACK)
    if(scalapack_info.pg.is_valid()) {
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

  std::vector<Tensor<T>>   tensor_list{ttensors.C_alpha, ttensors.D_alpha};
  std::vector<std::string> tensor_fnames{movecsfile_alpha, densityfile_alpha};
  if(chem_env.sys_data.is_unrestricted) {
    tensor_list.insert(tensor_list.end(), {ttensors.C_beta, ttensors.D_beta});
    tensor_fnames.insert(tensor_fnames.end(), {movecsfile_beta, densityfile_beta});
  }

#if !defined(USE_SERIAL_IO)
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

// Explicit template instantiations
template class exachem::scf::DefaultSCFIO<double>;
template class exachem::scf::SCFIO<double>;
