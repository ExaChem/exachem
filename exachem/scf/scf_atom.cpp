#include "scf_guess.hpp"

const std::vector<std::vector<int>> occecp = {{0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 3, 2, 0, 1, 3, 2},
                                              {0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2},
                                              {0, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1}};
const std::vector<int>              nelecp = {0,  2,  4,  10, 12, 18, 22, 28, 30, 36, 40,  46,  48,
                                              54, 60, 62, 68, 72, 78, 80, 86, 92, 94, 100, 104, 110};
const std::vector<int>              iecp   = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                                              2, 1, 0, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0};

template<typename TensorType>
void compute_sad_guess(ExecutionContext& ec, ScalapackInfo& scalapack_info, SystemData& sys_data,
                       SCFVars& scf_vars, const std::vector<libint2::Atom>& atoms,
                       const libint2::BasisSet& shells_tot, const std::string& basis,
                       bool is_spherical, EigenTensors& etensors, TAMMTensors& ttensors, int charge,
                       int multiplicity) {
  auto ig1 = std::chrono::high_resolution_clock::now();

  Scheduler  sch{ec};
  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;

  // int neutral_charge = sys_data.nelectrons + charge;
  // double N_to_Neu  = (double)sys_data.nelectrons/neutral_charge;
  // double Na_to_Neu = (double) sys_data.nelectrons_alpha / neutral_charge;
  // double Nb_to_Neu = (double) sys_data.nelectrons_beta / neutral_charge;

  /*
  Superposition of Atomic Density
  */

  const auto rank = ec.pg().rank();
  size_t     nao  = shells_tot.nbf();

  Matrix& D_tot_a = etensors.D;
  Matrix& D_tot_b = etensors.G;

  // Get atomic occupations
  auto occs = compute_soad(atoms);

  double fock_precision = std::numeric_limits<double>::epsilon();
  auto   scf_options    = sys_data.options_map.scf_options;

  // loop over atoms
  size_t indx  = 0;
  size_t iatom = 0;

  std::map<string, int> atom_loc;

  for(const auto& k: sys_data.options_map.options.ec_atoms) {
    // const auto Z = k.atom.atomic_number;
    const auto es             = k.esymbol;
    const bool has_ecp        = k.has_ecp;
    auto       acharge        = scf_options.charge;
    auto       amultiplicity  = scf_options.multiplicity;
    bool       custom_opts    = false;
    bool       spin_polarized = false;
    bool       do_charges     = false;

    std::vector<Atom> atom;
    auto              kcopy = k.atom;
    atom.push_back(kcopy);

    // Generate local basis set
    libint2::BasisSet shells_atom(k.basis, atom);
    shells_atom.set_pure(true);
    size_t nao_atom = shells_atom.nbf();

    if(atom_loc.find(es) != atom_loc.end()) {
      int atom_indx = atom_loc[es];

      if(rank == 0) {
        D_tot_a.block(indx, indx, nao_atom, nao_atom) =
          D_tot_a.block(atom_indx, atom_indx, nao_atom, nao_atom);
        D_tot_b.block(indx, indx, nao_atom, nao_atom) =
          D_tot_b.block(atom_indx, atom_indx, nao_atom, nao_atom);
      }

      // atom_loc[es] = indx;
      indx += nao_atom;
      ++iatom;

      atom.pop_back();
      continue;
    }

    // Modify occupations if ecp is present
    if(has_ecp) {
      int ncore = k.ecp_nelec;

      // Obtain the type of ECP depending on ncore
      auto index = std::distance(nelecp.begin(), std::find(nelecp.begin(), nelecp.end(), ncore));
      if(index > nelecp.size()) tamm_terminate("Error: ECP type not compatiable");

      // Start removing electrons according to occupations
      for(size_t i = 0; i < occecp[iecp[index]].size(); ++i) {
        const int l = occecp[iecp[index]][i];
        occs(iatom, l) -= 2.0 * (2.0 * l + 1.0);
        ncore -= 2 * (2 * l + 1);
        if(ncore < 1) break;
      }
    }

    // Check if user supplied custom options
    auto guess_atom_options = scf_options.guess_atom_options;
    std::vector<std::pair<double, std::array<double, 3>>> q;
    if(guess_atom_options.find(es) != guess_atom_options.end()) {
      std::tie(acharge, amultiplicity) = guess_atom_options[es];
      custom_opts                      = true;
      // Use point charges to lift degeneracies
      for(size_t j = 0; j < atoms.size(); ++j) {
        if(j == iatom) continue;
        do_charges = true;
        q.push_back({0.05, {{atoms[j].x, atoms[j].y, atoms[j].z}}});
      }
    }

    int nelectrons_alpha_atom;
    int nelectrons_beta_atom;
    if(custom_opts) {
      int nelectrons = k.atom.atomic_number - acharge;
      if(has_ecp) nelectrons -= k.ecp_nelec;
      nelectrons_alpha_atom = (nelectrons + amultiplicity - 1) / 2;
      nelectrons_beta_atom  = nelectrons - nelectrons_alpha_atom;
    }

    auto s2bf_atom = shells_atom.shell2bf();

    std::tie(scf_vars.obs_shellpair_list_atom, scf_vars.obs_shellpair_data_atom) =
      compute_shellpairs(shells_atom);
    // if(rank == 0) cout << "compute shell pairs for present basis" << endl;

    // Get occupations
    Eigen::Vector<EigenTensorType, 4> occ_atom_b = {0.0, 0.0, 0.0, 0.0};
    Eigen::Vector<EigenTensorType, 4> occ_atom_a = {0.0, 0.0, 0.0, 0.0};
    for(int l = 0; l < 4; ++l) {
      const double norb = 2.0 * l + 1.0;                      // Number of orbitals
      const int    ndbl = occs(iatom, l) / (2 * (2 * l + 1)); // Number of doubly occupied orbitals
      occ_atom_a(l)     = ndbl * norb + std::min(occs(iatom, l) - 2 * ndbl * norb, norb);
      occ_atom_b(l)     = ndbl * norb + std::max(occs(iatom, l) - occ_atom_a(l) - ndbl * norb, 0.0);
    }
    Eigen::Vector<EigenTensorType, 4> _occ_atom_a = occ_atom_a;
    Eigen::Vector<EigenTensorType, 4> _occ_atom_b = occ_atom_b;

    // Generate Density Matrix Guess
    Matrix D_a_atom = Matrix::Zero(nao_atom, nao_atom);
    Matrix D_b_atom = Matrix::Zero(nao_atom, nao_atom);
    for(size_t ishell = 0; ishell < shells_atom.size(); ++ishell) {
      const int    l    = shells_atom[ishell].contr[0].l;
      const double norb = 2.0 * l + 1.0;

      if(l > 3) continue;                // No atom has electrons in shells with l > 3
      if(_occ_atom_a(l) < 0.1) continue; // No more electrons to be added

      double nocc_a = std::min(_occ_atom_a(l) / norb, 1.0);
      double nocc_b = std::min(_occ_atom_b(l) / norb, 1.0);
      _occ_atom_a(l) -= nocc_a * norb;
      _occ_atom_b(l) -= nocc_b * norb;

      int bf1 = s2bf_atom[ishell];
      int bf2 = bf1 + 2 * l;
      for(int ibf = bf1; ibf <= bf2; ++ibf) {
        D_a_atom(ibf, ibf) = nocc_a;
        D_b_atom(ibf, ibf) = nocc_b;
      }
    }

    if(!spin_polarized) {
      D_a_atom = 0.5 * (D_a_atom + D_b_atom);
      D_b_atom = D_a_atom;
    }

    // cout << endl << iatom << endl << D_a_atom << endl;

    tamm::Tile tile_size_atom = sys_data.options_map.scf_options.AO_tilesize;

    if(tile_size_atom < nao_atom * 0.05) tile_size_atom = std::ceil(nao_atom * 0.05);

    std::vector<Tile> AO_tiles_atom;
    for(auto s: shells_atom) AO_tiles_atom.push_back(s.size());

    tamm::Tile          est_ts_atom = 0;
    std::vector<Tile>   AO_opttiles_atom;
    std::vector<size_t> shell_tile_map_atom;
    for(auto s = 0U; s < shells_atom.size(); s++) {
      est_ts_atom += shells_atom[s].size();
      if(est_ts_atom >= tile_size_atom) {
        AO_opttiles_atom.push_back(est_ts_atom);
        shell_tile_map_atom.push_back(s); // shell id specifying tile boundary
        est_ts_atom = 0;
      }
    }
    if(est_ts_atom > 0) {
      AO_opttiles_atom.push_back(est_ts_atom);
      shell_tile_map_atom.push_back(shells_atom.size() - 1);
    }

    // if(rank == 0) cout << "compute tile info for present basis" << endl;

    IndexSpace            AO_atom{range(0, nao_atom)};
    tamm::TiledIndexSpace tAO_atom, tAOt_atom;
    tAO_atom  = {AO_atom, AO_opttiles_atom};
    tAOt_atom = {AO_atom, AO_tiles_atom};

    // compute core hamiltonian H and overlap S for the atom
    Tensor<TensorType> H_atom{tAO_atom, tAO_atom};
    Tensor<TensorType> S_atom{tAO_atom, tAO_atom};
    Tensor<TensorType> T_atom{tAO_atom, tAO_atom};
    Tensor<TensorType> V_atom{tAO_atom, tAO_atom};
    Tensor<TensorType> Q_atom{tAO_atom, tAO_atom};
    Tensor<TensorType> E_atom{tAO_atom, tAO_atom};
    Tensor<TensorType>::allocate(&ec, H_atom, S_atom, T_atom, V_atom, Q_atom, E_atom);
    Matrix H_atom_eig = Matrix::Zero(nao_atom, nao_atom);
    Matrix S_atom_eig = Matrix::Zero(nao_atom, nao_atom);

    using libint2::Operator;

    TiledIndexSpace           tAO            = scf_vars.tAO;
    const std::vector<Tile>   AO_tiles       = scf_vars.AO_tiles;
    const std::vector<size_t> shell_tile_map = scf_vars.shell_tile_map;

    scf_vars.tAO            = tAO_atom;
    scf_vars.AO_tiles       = AO_tiles_atom;
    scf_vars.shell_tile_map = shell_tile_map_atom;

    std::vector<libecpint::ECP>           ecps;
    std::vector<libecpint::GaussianShell> libecp_shells;
    if(has_ecp) {
      for(auto shell: shells_atom) {
        std::array<double, 3>    O = {shell.O[0], shell.O[1], shell.O[2]};
        libecpint::GaussianShell newshell(O, shell.contr[0].l);
        for(size_t iprim = 0; iprim < shell.alpha.size(); iprim++)
          newshell.addPrim(shell.alpha[iprim], shell.contr[0].coeff[iprim]);
        libecp_shells.push_back(newshell);
      }

      int  maxam   = *std::max_element(k.ecp_ams.begin(), k.ecp_ams.end());
      auto ecp_ams = k.ecp_ams;
      std::replace(ecp_ams.begin(), ecp_ams.end(), -1, maxam + 1);
      std::array<double, 3> O = {atom[0].x, atom[0].y, atom[0].z};
      // std::cout << O << std::endl;
      libecpint::ECP newecp(O.data());
      for(size_t iprim = 0; iprim < k.ecp_coeffs.size(); iprim++)
        newecp.addPrimitive(k.ecp_ns[iprim], ecp_ams[iprim], k.ecp_exps[iprim], k.ecp_coeffs[iprim],
                            true);
      // std::cout << "noType1: " << newecp.noType1() << std::endl;
      // std::cout << "U_l(r):  " << newecp.evaluate(0.1, 2) << std::endl;
      ecps.push_back(newecp);
    }

    compute_1body_ints(ec, scf_vars, S_atom, atom, shells_atom, Operator::overlap);
    compute_1body_ints(ec, scf_vars, T_atom, atom, shells_atom, Operator::kinetic);
    if(has_ecp) atom[0].atomic_number -= k.ecp_nelec;
    compute_1body_ints(ec, scf_vars, V_atom, atom, shells_atom, Operator::nuclear);

    if(custom_opts && do_charges) {
      compute_pchg_ints(ec, scf_vars, Q_atom, q, shells_atom, Operator::nuclear);
    }
    else { Scheduler{ec}(Q_atom() = 0.0).execute(); }

    if(has_ecp) { compute_ecp_ints<TensorType>(ec, scf_vars, E_atom, libecp_shells, ecps); }
    else { Scheduler{ec}(E_atom() = 0.0).execute(); }

    // if(rank == 0) cout << "compute one body ints" << endl;

    // clang-format off
    Scheduler{ec}
      (H_atom()  = T_atom())
      (H_atom() += V_atom())
      (H_atom() += Q_atom())
      (H_atom() += E_atom())
      .deallocate(T_atom, V_atom, Q_atom, E_atom)
      .execute();
    // clang-format on

    // if(rank == 0) cout << "compute H_atom" << endl;

    t2e_hf_helper<TensorType, 2>(ec, H_atom, H_atom_eig, "H1-H-atom");
    t2e_hf_helper<TensorType, 2>(ec, S_atom, S_atom_eig, "S1-S-atom");

    // if(rank == 0) cout << std::setprecision(6) << "H_atom: " << endl << H_atom_eig << endl
    //                                            << "S_atom: " << endl << S_atom_eig << endl;

    // Form X_atom
    size_t obs_rank;
    double S_condition_number;
    double XtX_condition_number;
    double S_condition_number_threshold = sys_data.options_map.scf_options.tol_lindep;

    assert(S_atom_eig.rows() == S_atom_eig.cols());

    // std::tie(X_atom, Xinv, obs_rank, S_condition_number, XtX_condition_number, n_illcond) =
    //     gensqrtinv(ec, S_atom_eig, false, S_condition_number_threshold);

    Tensor<TensorType> X_alpha, X_beta;
    std::tie(obs_rank, S_condition_number, XtX_condition_number) =
      gensqrtinv_atscf(ec, sys_data, scf_vars, scalapack_info, S_atom, X_alpha, X_beta, tAO_atom,
                       false, S_condition_number_threshold);
    Matrix X_atom = tamm_to_eigen_matrix(X_alpha);

    // if(rank == 0) cout << std::setprecision(6) << "X_atom: " << endl << X_atom << endl;

    // Projecting minimal basis SOAD onto basis set specified
    // double          precision = std::numeric_limits<double>::epsilon();
    const libint2::BasisSet& obs = shells_atom;

    Matrix             G_a_atom = Matrix::Zero(nao_atom, nao_atom);
    Matrix             G_b_atom = Matrix::Zero(nao_atom, nao_atom);
    Tensor<TensorType> F1tmp_atom{tAOt_atom, tAOt_atom}; // not allocated

    using libint2::BraKet;
    using libint2::Engine;
    using libint2::Operator;

    auto shell2bf = obs.shell2bf();

    // Form intial guess of Fock matrix and Density matrix for the present basis
    auto Ft_a_atom = H_atom_eig;
    auto Ft_b_atom = H_atom_eig;

    // if(rank == 0) cout << std::setprecision(6) << "Ft_a_atom: " << endl << Ft_a_atom << endl;

    Matrix C_a_atom = Matrix::Zero(nao_atom, nao_atom);
    Matrix C_b_atom = Matrix::Zero(nao_atom, nao_atom);

    // if(rank == 0) cout << std::setprecision(6) << "D_a_atom: " << endl << D_a_atom << endl;

    // Atomic SCF loop
    double rmsd_a_atom = 1.0;
    // double     rmsd_b_atom       = 1.0;
    int        iter_atom         = 0;
    Matrix     D_a_atom_last     = Matrix::Zero(nao_atom, nao_atom);
    Matrix     D_b_atom_last     = Matrix::Zero(nao_atom, nao_atom);
    auto       SchwarzK          = compute_schwarz_ints<>(ec, scf_vars, shells_atom);
    const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;

    scf_vars.tAO            = tAO;
    scf_vars.AO_tiles       = AO_tiles;
    scf_vars.shell_tile_map = shell_tile_map;

    using libint2::Engine;
    Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
    engine.set_precision(fock_precision);
    const auto& buf = engine.results();

    Tensor<TensorType> F1tmp_atom2{tAOt_atom, tAOt_atom}; // not allocated
    Tensor<TensorType> F1tmp1_a_atom2{tAO_atom, tAO_atom};
    Tensor<TensorType> F1tmp1_b_atom2{tAO_atom, tAO_atom};
    Tensor<TensorType>::allocate(&ec, F1tmp1_a_atom2, F1tmp1_b_atom2);

    do {
      ++iter_atom;
      D_a_atom_last = D_a_atom;
      D_b_atom_last = D_b_atom;

      Matrix D_shblk_norm_atom = compute_shellblock_norm(obs, D_a_atom);
      D_shblk_norm_atom += compute_shellblock_norm(obs, D_b_atom);

      Matrix G_a_atom2 = Matrix::Zero(nao_atom, nao_atom);
      Matrix G_b_atom2 = Matrix::Zero(nao_atom, nao_atom);

      auto comp_2bf_lambda_atom = [&](IndexVector blockid) {
        auto s1        = blockid[0];
        auto bf1_first = shell2bf[s1];
        auto n1        = obs[s1].size();
        auto sp12_iter = scf_vars.obs_shellpair_data_atom.at(s1).begin();

        auto s2     = blockid[1];
        auto s2spl  = scf_vars.obs_shellpair_list_atom.at(s1);
        auto s2_itr = std::find(s2spl.begin(), s2spl.end(), s2);
        if(s2_itr == s2spl.end()) return;
        auto s2_pos    = std::distance(s2spl.begin(), s2_itr);
        auto bf2_first = shell2bf[s2];
        auto n2        = obs[s2].size();
        bool do12      = obs[s1].contr[0].l == obs[s2].contr[0].l;

        std::advance(sp12_iter, s2_pos);
        const auto* sp12 = sp12_iter->get();

        const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm_atom(s1, s2) : 0.;

        for(decltype(s1) s3 = 0; s3 <= s1; ++s3) {
          auto bf3_first = shell2bf[s3];
          auto n3        = obs[s3].size();
          bool do13      = obs[s1].contr[0].l == obs[s3].contr[0].l;
          bool do23      = obs[s2].contr[0].l == obs[s3].contr[0].l;

          const auto Dnorm123 =
            do_schwarz_screen
              ? std::max(D_shblk_norm_atom(s1, s3), std::max(D_shblk_norm_atom(s2, s3), Dnorm12))
              : 0.;

          auto sp34_iter = scf_vars.obs_shellpair_data_atom.at(s3).begin();

          const auto s4_max = (s1 == s3) ? s2 : s3;
          for(const auto& s4: scf_vars.obs_shellpair_list_atom.at(s3)) {
            if(s4 > s4_max) break;
            bool do14 = obs[s1].contr[0].l == obs[s4].contr[0].l;
            bool do24 = obs[s2].contr[0].l == obs[s4].contr[0].l;
            bool do34 = obs[s3].contr[0].l == obs[s4].contr[0].l;

            const auto* sp34 = sp34_iter->get();
            ++sp34_iter;

            if(not(do12 or do34 or (do13 and do24) or (do14 and do23))) continue;

            const auto Dnorm1234 =
              do_schwarz_screen ? std::max(D_shblk_norm_atom(s1, s4),
                                           std::max(D_shblk_norm_atom(s2, s4),
                                                    std::max(D_shblk_norm_atom(s3, s4), Dnorm123)))
                                : 0.;

            if(do_schwarz_screen &&
               Dnorm1234 * SchwarzK(s1, s2) * SchwarzK(s3, s4) < fock_precision)
              continue;

            auto bf4_first = shell2bf[s4];
            auto n4        = obs[s4].size();

            auto s12_deg    = (s1 == s2) ? 1 : 2;
            auto s34_deg    = (s3 == s4) ? 1 : 2;
            auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
            auto s1234_deg  = s12_deg * s34_deg * s12_34_deg;

            engine.compute2<Operator::coulomb, libint2::BraKet::xx_xx, 0>(obs[s1], obs[s2], obs[s3],
                                                                          obs[s4], sp12, sp34);

            const auto* buf_1234 = buf[0];
            if(buf_1234 == nullptr) continue; // if all integrals screened out, skip to next quartet

            if(do12 or do34) {
              for(decltype(n1) f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for(decltype(n2) f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for(decltype(n3) f3 = 0; f3 != n3; ++f3) {
                    const auto bf3 = f3 + bf3_first;
                    for(decltype(n4) f4 = 0; f4 != n4; ++f4, ++f1234) {
                      const auto bf4               = f4 + bf4_first;
                      const auto value             = buf_1234[f1234];
                      const auto value_scal_by_deg = value * s1234_deg;
                      const auto g12 =
                        0.5 * (D_a_atom(bf3, bf4) + D_b_atom(bf3, bf4)) * value_scal_by_deg;
                      const auto g34 =
                        0.5 * (D_a_atom(bf1, bf2) + D_b_atom(bf1, bf2)) * value_scal_by_deg;
                      // alpha_part
                      G_a_atom2(bf1, bf2) += g12;
                      G_a_atom2(bf3, bf4) += g34;

                      // beta_part
                      G_b_atom2(bf1, bf2) += g12;
                      G_b_atom2(bf3, bf4) += g34;
                    }
                  }
                }
              }
            }

            if((do13 and do24) or (do14 and do23)) {
              for(decltype(n1) f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for(decltype(n2) f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for(decltype(n3) f3 = 0; f3 != n3; ++f3) {
                    const auto bf3 = f3 + bf3_first;
                    for(decltype(n4) f4 = 0; f4 != n4; ++f4, ++f1234) {
                      const auto bf4               = f4 + bf4_first;
                      const auto value             = buf_1234[f1234];
                      const auto value_scal_by_deg = value * s1234_deg;
                      // alpha_part
                      G_a_atom2(bf2, bf3) -= 0.25 * D_a_atom(bf1, bf4) * value_scal_by_deg;
                      G_a_atom2(bf2, bf4) -= 0.25 * D_a_atom(bf1, bf3) * value_scal_by_deg;
                      G_a_atom2(bf1, bf3) -= 0.25 * D_a_atom(bf2, bf4) * value_scal_by_deg;
                      G_a_atom2(bf1, bf4) -= 0.25 * D_a_atom(bf2, bf3) * value_scal_by_deg;

                      // beta_part
                      G_b_atom2(bf1, bf3) -= 0.25 * D_b_atom(bf2, bf4) * value_scal_by_deg;
                      G_b_atom2(bf1, bf4) -= 0.25 * D_b_atom(bf2, bf3) * value_scal_by_deg;
                      G_b_atom2(bf2, bf3) -= 0.25 * D_b_atom(bf1, bf4) * value_scal_by_deg;
                      G_b_atom2(bf2, bf4) -= 0.25 * D_b_atom(bf1, bf3) * value_scal_by_deg;
                    }
                  }
                }
              }
            }
          }
        }
      };

      block_for(ec, F1tmp_atom2(), comp_2bf_lambda_atom);
      // symmetrize G
      Matrix Gt_a_atom2 = 0.5 * (G_a_atom2 + G_a_atom2.transpose());
      G_a_atom2         = Gt_a_atom2;
      Gt_a_atom2.resize(0, 0);
      Matrix Gt_b_atom2 = 0.5 * (G_b_atom2 + G_b_atom2.transpose());
      G_b_atom2         = Gt_b_atom2;
      Gt_b_atom2.resize(0, 0);

      sch(F1tmp1_a_atom2() = 0.0)(F1tmp1_b_atom2() = 0.0).execute();

      eigen_to_tamm_tensor_acc(F1tmp1_a_atom2, G_a_atom2);
      eigen_to_tamm_tensor_acc(F1tmp1_b_atom2, G_b_atom2);
      ec.pg().barrier();

      Matrix F_a_atom = Matrix::Zero(nao_atom, nao_atom);
      tamm_to_eigen_tensor(F1tmp1_a_atom2, F_a_atom);
      F_a_atom += H_atom_eig;
      if(iter_atom > 1) F_a_atom -= 0.05 * S_atom_eig * D_a_atom * S_atom_eig;
      Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_alpha(X_atom.transpose() * F_a_atom *
                                                             X_atom);
      C_a_atom = X_atom * eig_solver_alpha.eigenvectors();
      D_a_atom = Matrix::Zero(nao_atom, nao_atom);

      if(spin_polarized || custom_opts) {
        Matrix F_b_atom = Matrix::Zero(nao_atom, nao_atom);
        tamm_to_eigen_tensor(F1tmp1_b_atom2, F_b_atom);
        F_b_atom += H_atom_eig;
        if(iter_atom > 1) F_b_atom -= 0.05 * S_atom_eig * D_b_atom * S_atom_eig;
        Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_beta(X_atom.transpose() * F_b_atom *
                                                              X_atom);
        C_b_atom = X_atom * eig_solver_beta.eigenvectors();
      }
      else { C_b_atom = C_a_atom; }
      D_b_atom = Matrix::Zero(nao_atom, nao_atom);

      // Use atomic occupations to get density
      // int    iocc = 0;
      // int    ideg = 0;
      // bool   deg  = false;

      Matrix occvec(nao_atom, 1);
      occvec.setZero();

      if(custom_opts) {
        D_a_atom = C_a_atom.leftCols(nelectrons_alpha_atom) *
                   C_a_atom.leftCols(nelectrons_alpha_atom).transpose();
        D_b_atom = C_b_atom.leftCols(nelectrons_beta_atom) *
                   C_b_atom.leftCols(nelectrons_beta_atom).transpose();
      }
      else {
        _occ_atom_a = occ_atom_a;
        _occ_atom_b = occ_atom_b;
        // Alpha
        for(size_t imo = 0; imo < nao_atom; ++imo) {
          // Check how many electrons are left to add
          double _nelec = _occ_atom_a.sum();
          if(_nelec < 0.1) break;

          // Check to which shell the orbital belongs to
          int                 lang = -1;
          std::vector<double> normang_a(4, 0.0);
          for(size_t ishell = 0; ishell < obs.size(); ++ishell) {
            int l   = obs[ishell].contr[0].l;
            int bf1 = shell2bf[ishell];
            int bf2 = bf1 + obs[ishell].size() - 1;
            if(l > 3) continue;
            for(int ibf = bf1; ibf <= bf2; ++ibf) {
              normang_a[l] += C_a_atom(ibf, imo) * C_a_atom(ibf, imo);
            }
            if(normang_a[l] > 0.1) {
              lang = l;
              break;
            }
          }

          // If MO is from angular momentum > f skip it
          if(lang < 0) continue;

          // Skip if no more electrons to add to this shell
          if(_occ_atom_a(lang) < 0.1) continue;

          // Distribute electrons evenly in all degenerate orbitals
          double _nocc = std::min(_occ_atom_a(lang) / (2.0 * lang + 1.0), 1.0);
          for(int j = 0; j < 2 * lang + 1; ++j) {
            _occ_atom_a(lang) -= _nocc;
            // D_a_atom += _nocc * C_a_atom.col(imo + j) * C_a_atom.col(imo + j).transpose();
            occvec(imo + j) = _nocc;
          }
          imo += 2 * lang;
        }
        D_a_atom = C_a_atom * occvec.asDiagonal() * C_a_atom.transpose();

        // Beta
        occvec.setZero();
        for(size_t imo = 0; imo < nao_atom; ++imo) {
          // Check how many electrons are left to add
          double _nelec = _occ_atom_b.sum();
          if(_nelec < 0.1) break;

          // Check to which shell the orbital belongs to
          int                 lang = -1;
          std::vector<double> normang_b(4, 0.0);
          for(size_t ishell = 0; ishell < obs.size(); ++ishell) {
            int l   = obs[ishell].contr[0].l;
            int bf1 = shell2bf[ishell];
            int bf2 = bf1 + obs[ishell].size() - 1;
            if(l > 3) continue;
            for(int ibf = bf1; ibf <= bf2; ++ibf) {
              normang_b[l] += C_b_atom(ibf, imo) * C_b_atom(ibf, imo);
            }
            if(normang_b[l] > 0.1) {
              lang = l;
              break;
            }
          }

          // If shell is from angular momentum > f skip it
          if(lang < 0) continue;

          // Skip if no more electrons to add to this shell
          if(_occ_atom_b(lang) < 0.1) continue;

          // Distribute electrons evenly in all degenerate orbitals
          double _nocc = std::min(_occ_atom_b(lang) / (2 * lang + 1), 1.0);
          for(int j = 0; j < 2 * lang + 1; ++j) {
            _occ_atom_b(lang) -= _nocc;
            occvec(imo + j) = _nocc;
            // D_b_atom += _nocc * C_b_atom.col(imo + j) * C_b_atom.col(imo + j).transpose();
          }
          imo += 2 * lang;
        }
        D_b_atom = C_b_atom * occvec.asDiagonal() * C_b_atom.transpose();

        if(!spin_polarized) {
          D_a_atom = 0.5 * (D_a_atom + D_b_atom);
          D_b_atom = D_a_atom;
        }
      }

      auto D_a_diff = D_a_atom - D_a_atom_last;
      auto D_b_diff = D_b_atom - D_b_atom_last;
      D_a_atom -= 0.3 * D_a_diff;
      D_b_atom -= 0.3 * D_b_diff;
      rmsd_a_atom = std::max(D_a_diff.norm(), D_b_diff.norm());

      // if (rank==0) cout << std::setprecision(6) << rmsd_a_atom << endl;

      if(iter_atom > 200) break;

    } while(fabs(rmsd_a_atom) > 1e-5);

    if(rank == 0) {
      D_tot_a.block(indx, indx, nao_atom, nao_atom) = D_a_atom;
      D_tot_b.block(indx, indx, nao_atom, nao_atom) = D_b_atom;
    }

    atom_loc[es] = indx;
    indx += nao_atom;
    ++iatom;

    atom.pop_back();

    Tensor<TensorType>::deallocate(F1tmp1_a_atom2, F1tmp1_b_atom2, H_atom, S_atom);
  }

  // One-shot refinement
  if(rank == 0) {
    // D_tot_a -> etensors.D
    if(is_rhf) { etensors.D += D_tot_b; }

    if(is_uhf) etensors.D_beta = D_tot_b;

    D_tot_b.setZero();
  }

  ec.pg().broadcast(etensors.D.data(), etensors.D.size(), 0);
  if(is_uhf) ec.pg().broadcast(etensors.D_beta.data(), etensors.D_beta.size(), 0);

  // if(rank==0) cout << "in sad_guess, D_tot: " << endl << D_a << endl;

  auto ig2     = std::chrono::high_resolution_clock::now();
  auto ig_time = std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();
  if(ec.print())
    std::cout << std::fixed << std::setprecision(2) << "Time taken for SAD: " << ig_time << " secs"
              << endl;
}

template void compute_sad_guess<double>(ExecutionContext& ec, ScalapackInfo& scalapack_info,
                                        SystemData& sys_data, SCFVars& scf_vars,
                                        const std::vector<libint2::Atom>& atoms,
                                        const libint2::BasisSet& shells, const std::string& basis,
                                        bool is_spherical, EigenTensors& etensors,
                                        TAMMTensors& ttensors, int charge, int multiplicity);
