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
double exachem::scf::SCFIO<T>::tt_trace(ExecutionContext& ec, const Tensor<T>& T1,
                                        const Tensor<T>& T2) const {
  Tensor<T> tensor{T1.tiled_index_spaces()};
  Tensor<T>::allocate(&ec, tensor);
  const TiledIndexSpace tis_ao = T1.tiled_index_spaces()[0];
  const auto [mu, nu, ku]      = tis_ao.labels<3>("all");
  Scheduler{ec}(tensor(mu, nu) = T1(mu, ku) * T2(ku, nu)).execute();
  const double trace = tamm::trace(tensor);
  Tensor<T>::deallocate(tensor);
  return trace;
}

template<typename T>
Matrix exachem::scf::SCFIO<T>::read_scf_mat(const std::string& matfile) const {
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
void exachem::scf::SCFIO<T>::write_scf_mat(const Matrix& C, const std::string& matfile) const {
  std::string mname = fs::path(matfile).extension();
  mname.erase(0, 1); // remove "."

  const auto        N      = C.rows();
  const auto        Northo = C.cols();
  const TensorType* buf    = C.data();

  /* Create a file. */
  const hid_t file_id = H5Fcreate(matfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  const hsize_t tsize        = N * Northo;
  hid_t         dataspace_id = H5Screate_simple(1, &tsize, NULL);

  /* Create dataset. */
  hid_t dataset_id = H5Dcreate(file_id, mname.c_str(), get_hdf5_dt<T>(), dataspace_id, H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
  /* Write the dataset. */
  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

  /* Create and write attribute information - dims */
  std::vector<int64_t> rdims{N, Northo};
  const hsize_t        attr_size      = rdims.size();
  auto                 attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  const auto           attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT64, attr_dataspace,
                                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
}

template<typename T>
void exachem::scf::SCFIO<T>::print_energies(ExecutionContext& ec, ChemEnv& chem_env,
                                            TAMMTensors<T>& ttensors, EigenTensors& etensors,
                                            const SCFData& scf_data,
                                            ScalapackInfo& scalapack_info) const {
  const SystemData& sys_data    = chem_env.sys_data;
  const SCFOptions& scf_options = chem_env.ioptions.scf_options;

  // For QED e^T energy expression
  const TiledIndexSpace& tAO       = scf_data.tAO;
  const TiledIndexSpace& tAO_ortho = scf_data.tAO_ortho;
  const auto [mu, nu, ku]          = tAO.labels<3>("all");
  const auto [mu_o, nu_o, ku_o]    = tAO_ortho.labels<3>("all");
  Tensor<T> X_a{tAO, tAO_ortho};
  Tensor<T> Sm1{tAO, tAO};
  Tensor<T> Pvir{tAO, tAO};
  Tensor<T> mu_dot_lambda{tAO, tAO};
  Tensor<T> dP{tAO, tAO};
  Tensor<T> dPd{tAO, tAO};
  Tensor<T> scalar{};

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

      Tensor<T> X_a;

#if defined(USE_SCALAPACK)
      X_a = {scf_data.tAO, scf_data.tAO_ortho};
      Tensor<T>::allocate(&ec, X_a);
      if(scalapack_info.pg.is_valid()) { tamm::from_block_cyclic_tensor(ttensors.X_alpha, X_a); }
#else
      X_a = ttensors.X_alpha;
#endif

      std::vector<double> polvec = {0.0, 0.0, 0.0};
      Scheduler           sch{ec};
      // clang-format off
      sch.allocate(Sm1, Pvir, mu_dot_lambda, dP, dPd, scalar)
      (Sm1(mu, nu) = X_a(mu, mu_o) * X_a(nu, mu_o))
      (Pvir(mu, nu) = Sm1(mu, nu))
      (Pvir(mu, nu) -= 0.5 * ttensors.D_last_alpha(mu, nu))
      .execute();
      // clang-format on

      for(int i = 0; i < sys_data.qed_nmodes; i++) {
        for(int j = 0; j < 3; j++)
          polvec[j] = scf_options.qed_lambdas[i] * scf_options.qed_polvecs[i][j];

        // clang-format off
        sch
          (mu_dot_lambda(mu, nu)  = polvec[0] * ttensors.QED_Dx(mu, nu))
          (mu_dot_lambda(mu, nu) += polvec[1] * ttensors.QED_Dy(mu, nu))
          (mu_dot_lambda(mu, nu) += polvec[2] * ttensors.QED_Dz(mu, nu))
          (dP(mu, nu)  = mu_dot_lambda(mu, ku) * Pvir(ku, nu))
          (dPd(mu, nu)  = dP(mu, ku) * mu_dot_lambda(nu, ku))
	        (scalar() = 0.5 * dPd(mu, nu) * ttensors.D_last_alpha(mu, nu))
          .execute();
        // clang-format on

        energy_qed_et += tamm::get_scalar(scalar);
      }
      sch.deallocate(Sm1, Pvir, mu_dot_lambda, dP, dPd, scalar).execute();

#if defined(USE_SCALAPACK)
      Tensor<T>::deallocate(X_a);
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
    if(is_ks) {
      if((!is_qed) || (is_qed && do_qed)) { energy_2e += scf_data.exc; }
    }
    if(is_qed) {
      if(do_qed) {
        energy_qed = tt_trace(ec, ttensors.D_last_alpha, ttensors.QED_1body);
        energy_qed += tt_trace(ec, ttensors.D_last_beta, ttensors.QED_1body);
        energy_qed += 0.5 * tt_trace(ec, ttensors.D_last_alpha, ttensors.QED_2body);
        energy_qed += 0.5 * tt_trace(ec, ttensors.D_last_beta, ttensors.QED_2body_beta);
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
      std::vector<double> polvec = {0.0, 0.0, 0.0};
      Scheduler           sch{ec};
      // clang-format off
      sch.allocate(Sm1, Pvir, mu_dot_lambda, dP, dPd, scalar)
	 (Sm1(mu, nu) = X_a(mu, mu_o) * X_a(nu, mu_o))
	 .execute();
      // clang-format on

      for(int i = 0; i < sys_data.qed_nmodes; i++) {
        for(int j = 0; j < 3; j++)
          polvec[j] = scf_options.qed_lambdas[i] * scf_options.qed_polvecs[i][j];

        // clang-format off
        sch
          (mu_dot_lambda(mu, nu)  = polvec[0] * ttensors.QED_Dx(mu, nu))
          (mu_dot_lambda(mu, nu) += polvec[1] * ttensors.QED_Dy(mu, nu))
          (mu_dot_lambda(mu, nu) += polvec[2] * ttensors.QED_Dz(mu, nu))
          // Alpha
          (Pvir(mu, nu) = Sm1(mu, nu))
          (Pvir(mu, nu) -= ttensors.D_last_alpha(mu, nu))
            (dP(mu, nu)  = mu_dot_lambda(mu, ku) * Pvir(ku, nu))
            (dPd(mu, nu)  = dP(mu, ku) * mu_dot_lambda(nu, ku))
          (scalar() = dPd(mu, nu) * ttensors.D_last_alpha(mu, nu))
          // Beta
          (Pvir(mu, nu) = Sm1(mu, nu))
          (Pvir(mu, nu) -= ttensors.D_last_beta(mu, nu))
            (dP(mu, nu)  = mu_dot_lambda(mu, ku) * Pvir(ku, nu))
            (dPd(mu, nu)  = dP(mu, ku) * mu_dot_lambda(nu, ku))
          (scalar() += dPd(mu, nu) * ttensors.D_last_beta(mu, nu))
          .execute();
        // clang-format on

        energy_qed_et += tamm::get_scalar(scalar);
      }
      sch.deallocate(Sm1, Pvir, mu_dot_lambda, dP, dPd, scalar).execute();

#if defined(USE_SCALAPACK)
      Tensor<double>::deallocate(X_a);
#endif
    }
  }

  if(ec.pg().rank() == 0) {
    std::cout << "#electrons        = " << (int) std::round(nelectrons) << endl;
    std::cout << "1e energy kinetic = " << std::setprecision(16) << kinetic_1e << endl;
    std::cout << "1e energy N-e     = " << NE_1e << endl;
    std::cout << "1e energy         = " << energy_1e << endl;
    std::cout << "2e energy         = " << energy_2e << std::endl;

    auto& scf_results         = chem_env.sys_data.results["output"]["SCF"];
    scf_results["NE_1e"]      = NE_1e;
    scf_results["kinetic_1e"] = kinetic_1e;
    scf_results["energy_1e"]  = energy_1e;
    scf_results["energy_2e"]  = energy_2e;

    if(is_qed) {
      std::cout << "QED energy        = " << energy_qed << std::endl;
      std::cout << "QED eT energy     = " << energy_qed_et << std::endl;
      scf_results["energy_qed"]    = energy_qed;
      scf_results["energy_qed_et"] = energy_qed_et;
    }
  }
}

template<typename T>
void exachem::scf::SCFIO<T>::print_multipoles(ChemEnv&                   chem_env,
                                              const std::vector<double>& multipoles) const {
  const auto& atoms = chem_env.atoms;

  std::vector<double> multipoles_nuc(20);

  for(const auto& atom: atoms) {
    multipoles_nuc[0] += atom.atomic_number;
    multipoles_nuc[1] += atom.x * atom.atomic_number;
    multipoles_nuc[2] += atom.y * atom.atomic_number;
    multipoles_nuc[3] += atom.z * atom.atomic_number;
    multipoles_nuc[4] += atom.x * atom.x * atom.atomic_number;
    multipoles_nuc[5] += atom.x * atom.y * atom.atomic_number;
    multipoles_nuc[6] += atom.x * atom.z * atom.atomic_number;
    multipoles_nuc[7] += atom.y * atom.y * atom.atomic_number;
    multipoles_nuc[8] += atom.y * atom.z * atom.atomic_number;
    multipoles_nuc[9] += atom.z * atom.z * atom.atomic_number;
    multipoles_nuc[10] += atom.x * atom.x * atom.x * atom.atomic_number;
    multipoles_nuc[11] += atom.x * atom.x * atom.y * atom.atomic_number;
    multipoles_nuc[12] += atom.x * atom.x * atom.z * atom.atomic_number;
    multipoles_nuc[13] += atom.x * atom.y * atom.y * atom.atomic_number;
    multipoles_nuc[14] += atom.x * atom.y * atom.z * atom.atomic_number;
    multipoles_nuc[15] += atom.x * atom.z * atom.z * atom.atomic_number;
    multipoles_nuc[16] += atom.y * atom.y * atom.y * atom.atomic_number;
    multipoles_nuc[17] += atom.y * atom.y * atom.z * atom.atomic_number;
    multipoles_nuc[18] += atom.y * atom.z * atom.z * atom.atomic_number;
    multipoles_nuc[19] += atom.z * atom.z * atom.z * atom.atomic_number;
  }

  std::cout << std::fixed << std::setprecision(6) << std::endl << std::endl;
  std::cout << "                      Electron        Nuclear        Total  " << std::endl;
  std::cout << "                  ------------------------------------------" << std::endl;
  std::cout << "    Monopole     " << std::setw(14) << multipoles[0] << std::setw(14)
            << multipoles_nuc[0] << std::setw(14) << multipoles[0] + multipoles_nuc[0] << std::endl;
  std::cout << std::endl;
  std::cout << "      Dipole   X " << std::setw(14) << multipoles[1] << std::setw(14)
            << multipoles_nuc[1] << std::setw(14) << multipoles[1] + multipoles_nuc[1] << std::endl;
  std::cout << "      Dipole   Y " << std::setw(14) << multipoles[2] << std::setw(14)
            << multipoles_nuc[2] << std::setw(14) << multipoles[2] + multipoles_nuc[2] << std::endl;
  std::cout << "      Dipole   Z " << std::setw(14) << multipoles[3] << std::setw(14)
            << multipoles_nuc[3] << std::setw(14) << multipoles[3] + multipoles_nuc[3] << std::endl;
  std::cout << std::endl;
  std::cout << "  Quadrupole  XX " << std::setw(14) << multipoles[4] << std::setw(14)
            << multipoles_nuc[4] << std::setw(14) << multipoles[4] + multipoles_nuc[4] << std::endl;
  std::cout << "  Quadrupole  XY " << std::setw(14) << multipoles[5] << std::setw(14)
            << multipoles_nuc[5] << std::setw(14) << multipoles[5] + multipoles_nuc[5] << std::endl;
  std::cout << "  Quadrupole  XZ " << std::setw(14) << multipoles[6] << std::setw(14)
            << multipoles_nuc[6] << std::setw(14) << multipoles[6] + multipoles_nuc[6] << std::endl;
  std::cout << "  Quadrupole  YY " << std::setw(14) << multipoles[7] << std::setw(14)
            << multipoles_nuc[7] << std::setw(14) << multipoles[7] + multipoles_nuc[7] << std::endl;
  std::cout << "  Quadrupole  YZ " << std::setw(14) << multipoles[8] << std::setw(14)
            << multipoles_nuc[8] << std::setw(14) << multipoles[8] + multipoles_nuc[8] << std::endl;
  std::cout << "  Quadrupole  ZZ " << std::setw(14) << multipoles[9] << std::setw(14)
            << multipoles_nuc[9] << std::setw(14) << multipoles[9] + multipoles_nuc[9] << std::endl;
  std::cout << std::endl;
  std::cout << "    Octupole XXX " << std::setw(14) << multipoles[10] << std::setw(14)
            << multipoles_nuc[10] << std::setw(14) << multipoles[10] + multipoles_nuc[10]
            << std::endl;
  std::cout << "    Octupole XXY " << std::setw(14) << multipoles[11] << std::setw(14)
            << multipoles_nuc[11] << std::setw(14) << multipoles[11] + multipoles_nuc[11]
            << std::endl;
  std::cout << "    Octupole XXZ " << std::setw(14) << multipoles[12] << std::setw(14)
            << multipoles_nuc[12] << std::setw(14) << multipoles[12] + multipoles_nuc[12]
            << std::endl;
  std::cout << "    Octupole XYY " << std::setw(14) << multipoles[13] << std::setw(14)
            << multipoles_nuc[13] << std::setw(14) << multipoles[13] + multipoles_nuc[13]
            << std::endl;
  std::cout << "    Octupole XYZ " << std::setw(14) << multipoles[14] << std::setw(14)
            << multipoles_nuc[14] << std::setw(14) << multipoles[14] + multipoles_nuc[14]
            << std::endl;
  std::cout << "    Octupole XZZ " << std::setw(14) << multipoles[15] << std::setw(14)
            << multipoles_nuc[15] << std::setw(14) << multipoles[15] + multipoles_nuc[15]
            << std::endl;
  std::cout << "    Octupole YYY " << std::setw(14) << multipoles[16] << std::setw(14)
            << multipoles_nuc[16] << std::setw(14) << multipoles[16] + multipoles_nuc[16]
            << std::endl;
  std::cout << "    Octupole YYZ " << std::setw(14) << multipoles[17] << std::setw(14)
            << multipoles_nuc[17] << std::setw(14) << multipoles[17] + multipoles_nuc[17]
            << std::endl;
  std::cout << "    Octupole YZZ " << std::setw(14) << multipoles[18] << std::setw(14)
            << multipoles_nuc[18] << std::setw(14) << multipoles[18] + multipoles_nuc[18]
            << std::endl;
  std::cout << "    Octupole ZZZ " << std::setw(14) << multipoles[19] << std::setw(14)
            << multipoles_nuc[19] << std::setw(14) << multipoles[19] + multipoles_nuc[19]
            << std::endl;
  std::cout << std::endl;
}

template<typename T>
void exachem::scf::SCFIO<T>::print_mulliken(ChemEnv& chem_env, const Matrix& D,
                                            const Matrix& D_beta, const Matrix& S) const {
  auto&                      atoms    = chem_env.atoms;
  const std::vector<ECAtom>& ec_atoms = chem_env.ec_atoms;
  auto&                      shells   = chem_env.shells;
  const bool                 is_uhf   = chem_env.sys_data.is_unrestricted;

  BasisSetMap bsm(atoms, shells);

  const int                   natoms = static_cast<int>(ec_atoms.size());
  std::vector<T>              cs_acharge(natoms, 0);
  std::vector<T>              os_acharge(natoms, 0);
  std::vector<T>              net_acharge(natoms, 0);
  std::vector<std::vector<T>> cs_charge_shell(natoms);
  std::vector<std::vector<T>> os_charge_shell(natoms);
  std::vector<std::vector<T>> net_charge_shell(natoms);

  int j = 0;
  for(auto x = 0; x < natoms; x++) { // loop over atoms
    const auto atom_shells = bsm.atominfo[x].shells;
    const auto nshells     = atom_shells.size(); // #shells for atom x
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
                     os_charge_shell[x].begin(), net_charge_shell[x].begin(), std::plus<T>());
    }
  }
  const auto mksp = std::string(5, ' ');

  auto print_ma = [&](const std::string& dtype, const std::vector<T>& acharge,
                      const std::vector<std::vector<T>>& charge_shell) {
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
      for(const auto csx: charge_shell[x]) std::cout << std::setw(5) << csx << " ";
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
void exachem::scf::SCFIO<T>::rw_mat_disk(Tensor<T> tensor, const std::string& tfilename,
                                         bool profile, bool read) const {
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
void exachem::scf::SCFIO<T>::rw_md_disk(ExecutionContext& ec, const ChemEnv& chem_env,
                                        ScalapackInfo& scalapack_info, TAMMTensors<T>& ttensors,
                                        EigenTensors& etensors, const std::string& files_prefix,
                                        bool read) const {
  const auto rank   = ec.pg().rank();
  const bool is_uhf = chem_env.sys_data.is_unrestricted;
  const auto debug  = chem_env.ioptions.scf_options.debug;

  const std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
  const std::string densityfile_alpha = files_prefix + ".alpha.density";
  const std::string movecsfile_beta   = files_prefix + ".beta.movecs";
  const std::string densityfile_beta  = files_prefix + ".beta.density";

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
template class exachem::scf::SCFIO<double>;
