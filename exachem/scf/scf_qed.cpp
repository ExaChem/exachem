/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_qed.hpp"

template<typename T>
void exachem::scf::DefaultSCFQed<T>::qed_functionals_setup(std::vector<double>& params,
                                                           ChemEnv&             chem_env) {
  SCFOptions& scf_options = chem_env.ioptions.scf_options;

  params[0] = 1.0;
  // SCFOptions& scf_options = scf_options;

  std::copy(scf_options.qed_lambdas.begin(), scf_options.qed_lambdas.end(), params.begin() + 1);
  if(!scf_options.qed_omegas.empty()) {
    std::copy(scf_options.qed_omegas.begin(), scf_options.qed_omegas.end(), params.begin() + 1025);
  }
  std::ofstream libxc_params("./libxc.params");
  libxc_params << std::fixed << std::setprecision(16);
  for(auto const& val: params) libxc_params << val << std::endl;
  libxc_params.close();
}

template<typename T>
void exachem::scf::DefaultSCFQed<T>::compute_QED_1body(ExecutionContext& ec, ChemEnv& chem_env,
                                                       const SCFData&  scf_data,
                                                       TAMMTensors<T>& ttensors) {
  auto [mu, nu] = scf_data.tAO.labels<2>("all");
  Scheduler sch{ec};

  SystemData&                 sys_data    = chem_env.sys_data;
  SCFOptions&                 scf_options = chem_env.ioptions.scf_options;
  std::vector<libint2::Atom>& atoms       = chem_env.atoms;

  const int                              nmodes  = sys_data.qed_nmodes;
  const std::vector<double>              lambdas = scf_options.qed_lambdas;
  const std::vector<std::vector<double>> polvecs = scf_options.qed_polvecs;

  // compute nuclear dipole operator
  std::vector<double> mu_nuc = {0.0, 0.0, 0.0};
  for(size_t i = 0; i < atoms.size(); i++) {
    mu_nuc[0] += atoms[i].x * atoms[i].atomic_number;
    mu_nuc[1] += atoms[i].y * atoms[i].atomic_number;
    mu_nuc[2] += atoms[i].z * atoms[i].atomic_number;
  }

  double s0_scaling = 0.0;
  for(int i = 0; i < nmodes; i++) {
    s0_scaling +=
      lambdas[i] *
      (mu_nuc[0] * polvecs[i][0] + mu_nuc[1] * polvecs[i][1] + mu_nuc[2] * polvecs[i][2]) /
      sys_data.nelectrons;
  }
  s0_scaling *= 0.5 * s0_scaling;

  // clang-format off
  sch
    (ttensors.QED_1body(mu,nu) = s0_scaling*ttensors.S1(mu,nu))
    (ttensors.QED_1body(mu,nu) += ttensors.QED_Qxx(mu,nu));
  // clang-format on

  for(int i = 0; i < nmodes; i++) {
    double s1_scaling =
      (mu_nuc[0] * polvecs[i][0] + mu_nuc[1] * polvecs[i][1] + mu_nuc[2] * polvecs[i][2]) *
      lambdas[i] * lambdas[i] / sys_data.nelectrons;

    // clang-format off
    sch
      (ttensors.QED_2body(mu,nu)  = polvecs[i][0] * ttensors.QED_Dx(mu,nu))
      (ttensors.QED_2body(mu,nu) += polvecs[i][1] * ttensors.QED_Dy(mu,nu))
      (ttensors.QED_2body(mu,nu) += polvecs[i][2] * ttensors.QED_Dz(mu,nu))
      (ttensors.QED_1body(mu,nu) -= s1_scaling * ttensors.QED_2body(mu,nu));
    // clang-format on
  }

  // clang-format off
  sch
    (ttensors.H1(mu,nu) += ttensors.QED_1body(mu,nu))
    .execute();
  // clang-format on
}

template<typename T>
void exachem::scf::DefaultSCFQed<T>::compute_QED_2body(ExecutionContext& ec, ChemEnv& chem_env,
                                                       const SCFData&  scf_data,
                                                       TAMMTensors<T>& ttensors) {
  auto [mu, nu, ku] = scf_data.tAO.labels<3>("all");
  Tensor<T> tensor{ttensors.QED_1body.tiled_index_spaces()}; //{tAO, tAO};
  Scheduler sch{ec};

  auto&       atoms       = chem_env.atoms;
  SystemData& sys_data    = chem_env.sys_data;
  SCFOptions& scf_options = chem_env.ioptions.scf_options;

  const int                              nmodes  = sys_data.qed_nmodes;
  const std::vector<double>              lambdas = scf_options.qed_lambdas;
  const std::vector<std::vector<double>> polvecs = scf_options.qed_polvecs;

  // compute nuclear dipole operator
  std::vector<double> mu_nuc = {0.0, 0.0, 0.0};
  for(size_t i = 0; i < atoms.size(); i++) {
    mu_nuc[0] += atoms[i].x * atoms[i].atomic_number;
    mu_nuc[1] += atoms[i].y * atoms[i].atomic_number;
    mu_nuc[2] += atoms[i].z * atoms[i].atomic_number;
  }

  // clang-format off
  sch
    .allocate(tensor)
    (ttensors.QED_2body(mu,nu) = 0.0);
  // clang-format on

  for(int i = 0; i < nmodes; i++) {
    double mu_nuc_ope = 0.0;
    mu_nuc_ope =
      (mu_nuc[0] * polvecs[i][0] + mu_nuc[1] * polvecs[i][1] + mu_nuc[2] * polvecs[i][2]) *
      lambdas[i] / sys_data.nelectrons;

    double t1 = -0.5 * pow(lambdas[i], 2);
    double t2 = 0.5 * mu_nuc_ope * lambdas[i];
    double t3 = -0.5 * pow(mu_nuc_ope, 2);

    // clang-format off
    sch
      (tensor()  = polvecs[i][0]*ttensors.QED_Dx())
      (tensor() += polvecs[i][1]*ttensors.QED_Dy())
      (tensor() += polvecs[i][2]*ttensors.QED_Dz())
      (ttensors.ehf_tmp(mu,nu)      = tensor(mu,ku)*ttensors.D_last_alpha(ku,nu))
      (ttensors.QED_2body(mu,nu)   += t1*ttensors.ehf_tmp(mu,ku)*tensor(ku,nu))
      (ttensors.QED_2body(mu,nu)   += t2*ttensors.ehf_tmp(mu,ku)*ttensors.S1(ku,nu))
      (ttensors.QED_2body(mu,nu)   += t2*ttensors.S1(mu,ku)*ttensors.ehf_tmp(nu,ku))
      (ttensors.ehf_tmp(mu,nu)      = t3*ttensors.S1(mu,ku)*ttensors.D_alpha(ku,nu))
      (ttensors.QED_2body(mu,nu)   += ttensors.ehf_tmp(mu,ku)*ttensors.S1(ku,nu));
    // clang-format on
  }

  // clang-format off
  sch
    (ttensors.F_alpha(mu,nu) += ttensors.QED_2body(mu,nu))
    .deallocate(tensor)
    .execute();
  // clang-format on
}

template<typename T>
void exachem::scf::DefaultSCFQed<T>::compute_qed_emult_ints(ExecutionContext& ec, ChemEnv& chem_env,
                                                            const SCFData&  spvars,
                                                            TAMMTensors<T>& ttensors) {
  using libint2::Atom;
  using libint2::BasisSet;
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Shell;

  // auto& atoms = chem_env.atoms;
  SystemData&              sys_data    = chem_env.sys_data;
  SCFOptions&              scf_options = chem_env.ioptions.scf_options;
  const libint2::BasisSet& shells      = chem_env.shells;

  const std::vector<Tile>&   AO_tiles       = spvars.AO_tiles;
  const std::vector<size_t>& shell_tile_map = spvars.shell_tile_map;

  const int                              nmodes  = sys_data.qed_nmodes;
  const std::vector<double>              lambdas = scf_options.qed_lambdas;
  const std::vector<std::vector<double>> polvecs = scf_options.qed_polvecs;

  Engine engine(Operator::emultipole2, max_nprim(shells), max_l(shells), 0);

  // engine.set(otype);

  // auto& buf = (engine.results());

  auto compute_qed_emult_ints_lambda = [&](const IndexVector& blockid) {
    auto bi0 = blockid[0];
    auto bi1 = blockid[1];

    const TAMM_SIZE size       = ttensors.QED_Dx.block_size(blockid);
    auto            block_dims = ttensors.QED_Dx.block_dims(blockid);
    std::vector<T>  dbufx(size);
    std::vector<T>  dbufy(size);
    std::vector<T>  dbufz(size);
    std::vector<T>  dbufQ(size);

    auto bd1 = block_dims[1];

    // cout << "blockid: [" << blockid[0] <<"," << blockid[1] << "], dims(0,1) = " <<
    //  block_dims[0] << ", " << block_dims[1] << endl;

    // auto s1 = blockid[0];
    auto                  s1range_end   = shell_tile_map[bi0];
    decltype(s1range_end) s1range_start = 0l;
    if(bi0 > 0) s1range_start = shell_tile_map[bi0 - 1] + 1;

    // cout << "s1-start,end = " << s1range_start << ", " << s1range_end << endl;
    for(auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
      // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
      // this shell
      auto n1 = shells[s1].size();

      auto                  s2range_end   = shell_tile_map[bi1];
      decltype(s2range_end) s2range_start = 0l;
      if(bi1 > 0) s2range_start = shell_tile_map[bi1 - 1] + 1;

      // cout << "s2-start,end = " << s2range_start << ", " << s2range_end << endl;

      // cout << "screend shell pair list = " << s2spl << endl;
      for(auto s2 = s2range_start; s2 <= s2range_end; ++s2) {
        // for (auto s2: spvars.obs_shellpair_list.at(s1)) {
        // auto s2 = blockid[1];
        // if (s2>s1) continue;

        if(s2 > s1) {
          auto s2spl = spvars.obs_shellpair_list.at(s2);
          if(std::find(s2spl.begin(), s2spl.end(), s1) == s2spl.end()) continue;
        }
        else {
          auto s2spl = spvars.obs_shellpair_list.at(s1);
          if(std::find(s2spl.begin(), s2spl.end(), s2) == s2spl.end()) continue;
        }

        // auto bf2 = shell2bf[s2];
        auto n2 = shells[s2].size();

        std::vector<T> tbufX(n1 * n2);
        std::vector<T> tbufY(n1 * n2);
        std::vector<T> tbufZ(n1 * n2);
        std::vector<T> tbufXX(n1 * n2);
        std::vector<T> tbufXY(n1 * n2);
        std::vector<T> tbufXZ(n1 * n2);
        std::vector<T> tbufYY(n1 * n2);
        std::vector<T> tbufYZ(n1 * n2);
        std::vector<T> tbufZZ(n1 * n2);

        // compute shell pair; return is the pointer to the buffer
        const auto& buf = engine.compute(shells[s1], shells[s2]);
        EXPECTS(buf.size() >= 10);
        if(buf[0] == nullptr) continue;
        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result

        // cout << buf[1].size() << endl;
        // cout << buf[2].size() << endl;
        // cout << buf[3].size() << endl;
        Eigen::Map<const Matrix> buf_mat_X(buf[1], n1, n2);
        Eigen::Map<Matrix>(&tbufX[0], n1, n2) = buf_mat_X;
        Eigen::Map<const Matrix> buf_mat_Y(buf[2], n1, n2);
        Eigen::Map<Matrix>(&tbufY[0], n1, n2) = buf_mat_Y;
        Eigen::Map<const Matrix> buf_mat_Z(buf[3], n1, n2);
        Eigen::Map<Matrix>(&tbufZ[0], n1, n2) = buf_mat_Z;
        Eigen::Map<const Matrix> buf_mat_XX(buf[4], n1, n2);
        Eigen::Map<Matrix>(&tbufXX[0], n1, n2) = buf_mat_XX;
        Eigen::Map<const Matrix> buf_mat_XY(buf[5], n1, n2);
        Eigen::Map<Matrix>(&tbufXY[0], n1, n2) = buf_mat_XY;
        Eigen::Map<const Matrix> buf_mat_XZ(buf[6], n1, n2);
        Eigen::Map<Matrix>(&tbufXZ[0], n1, n2) = buf_mat_XZ;
        Eigen::Map<const Matrix> buf_mat_YY(buf[7], n1, n2);
        Eigen::Map<Matrix>(&tbufYY[0], n1, n2) = buf_mat_YY;
        Eigen::Map<const Matrix> buf_mat_YZ(buf[8], n1, n2);
        Eigen::Map<Matrix>(&tbufYZ[0], n1, n2) = buf_mat_YZ;
        Eigen::Map<const Matrix> buf_mat_ZZ(buf[9], n1, n2);
        Eigen::Map<Matrix>(&tbufZZ[0], n1, n2) = buf_mat_ZZ;

        // cout << "buf_mat_X :" << buf_mat_X << endl;
        // cout << "buf_mat_Y :" << buf_mat_Y << endl;
        // cout << "buf_mat_Z :" << buf_mat_Z << endl;

        auto curshelloffset_i = 0U;
        auto curshelloffset_j = 0U;
        for(auto x = s1range_start; x < s1; x++) curshelloffset_i += AO_tiles[x];
        for(auto x = s2range_start; x < s2; x++) curshelloffset_j += AO_tiles[x];

        size_t c    = 0;
        auto   dimi = curshelloffset_i + AO_tiles[s1];
        auto   dimj = curshelloffset_j + AO_tiles[s2];

        for(size_t i = curshelloffset_i; i < dimi; i++) {
          for(size_t j = curshelloffset_j; j < dimj; j++, c++) {
            dbufx[i * bd1 + j] = tbufX[c];
            dbufy[i * bd1 + j] = tbufY[c];
            dbufz[i * bd1 + j] = tbufZ[c];
            dbufQ[i * bd1 + j] = 0.0;
            for(int k = 0; k < nmodes; k++) {
              dbufQ[i * bd1 + j] +=
                0.5 * pow(lambdas[k], 2) *
                (tbufXX[c] * pow(polvecs[k][0], 2) + tbufYY[c] * pow(polvecs[k][1], 2) +
                 tbufZZ[c] * pow(polvecs[k][2], 2) +
                 2.0 * tbufXY[c] * polvecs[k][0] * polvecs[k][1] +
                 2.0 * tbufXZ[c] * polvecs[k][0] * polvecs[k][2] +
                 2.0 * tbufYZ[c] * polvecs[k][1] * polvecs[k][2]);
            }
          }
        }
      } // s2
    }   // s1

    ttensors.QED_Dx.put(blockid, dbufx);
    ttensors.QED_Dy.put(blockid, dbufy);
    ttensors.QED_Dz.put(blockid, dbufz);
    ttensors.QED_Qxx.put(blockid, dbufQ);
  };

  block_for(ec, ttensors.QED_Dx(), compute_qed_emult_ints_lambda);
}

template class exachem::scf::DefaultSCFQed<double>;
template class exachem::scf::SCFQed<double>;
