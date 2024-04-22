/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "ccsd_util.hpp"
using namespace exachem::scf;
template<typename T>
void setup_full_t1t2(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& dt1_full,
                     Tensor<T>& dt2_full) {
  TiledIndexSpace O = MO("occ");
  TiledIndexSpace V = MO("virt");

  dt1_full = Tensor<T>{{V, O}, {1, 1}};
  dt2_full = Tensor<T>{{V, V, O, O}, {2, 2}};

  Tensor<TensorType>::allocate(&ec, dt1_full, dt2_full);
  // (dt1_full() = 0)
  // (dt2_full() = 0)
}

template<typename TensorType>
void update_r2(ExecutionContext& ec, LabeledTensor<TensorType> ltensor) {
  Tensor<TensorType> tensor = ltensor.tensor();

  auto lambda = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, ltensor);
    if((blockid[0] > blockid[1]) || (blockid[2] > blockid[3])) {
      const tamm::TAMM_SIZE   dsize = tensor.block_size(blockid);
      std::vector<TensorType> dbuf(dsize);
      tensor.get(blockid, dbuf);
      // func(blockid, dbuf);
      for(auto i = 0U; i < dsize; i++) dbuf[i] = 0;
      tensor.put(blockid, dbuf);
    }
  };
  block_for(ec, ltensor, lambda);
}

template<typename TensorType>
void init_diagonal(ExecutionContext& ec, LabeledTensor<TensorType> ltensor) {
  Tensor<TensorType> tensor = ltensor.tensor();
  // Defined only for NxN tensors
  EXPECTS(tensor.num_modes() == 2);

  auto lambda = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, ltensor);
    if(blockid[0] == blockid[1]) {
      const TAMM_SIZE         size = tensor.block_size(blockid);
      std::vector<TensorType> buf(size);
      tensor.get(blockid, buf);
      auto   block_dims   = tensor.block_dims(blockid);
      auto   block_offset = tensor.block_offsets(blockid);
      auto   dim          = block_dims[0];
      auto   offset       = block_offset[0];
      size_t i            = 0;
      for(auto p = offset; p < offset + dim; p++, i++) buf[i * dim + i] = 1.0;
      tensor.put(blockid, buf);
    }
  };
  block_for(ec, ltensor, lambda);
}

void iteration_print(ChemEnv& chem_env, const ProcGroup& pg, int iter, double residual,
                     double energy, double time, string cmethod) {
  if(pg.rank() == 0) {
    std::cout << std::setw(4) << std::right << iter + 1 << "     ";
    std::cout << std::setprecision(13) << std::setw(16) << std::left << residual << "  ";
    std::cout << std::fixed << std::setprecision(13) << std::right << std::setw(16) << energy
              << " ";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::string(8, ' ') << time << std::endl;

    chem_env.sys_data.results["output"][cmethod]["iter"][std::to_string(iter + 1)] = {
      {"residual", residual}, {"correlation", energy}};
    chem_env.sys_data.results["output"][cmethod]["iter"][std::to_string(iter + 1)]["performance"] =
      {{"total_time", time}};
  }
}

/**
 *
 * @tparam T
 * @param MO
 * @param p_evl_sorted
 * @return pair of residual and energy
 */
template<typename T>
std::pair<double, double> rest(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1,
                               Tensor<T>& d_r2, Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& de,
                               Tensor<T>& d_r1_residual, Tensor<T>& d_r2_residual,
                               std::vector<T>& p_evl_sorted, T zshiftl, const TAMM_SIZE& noa,
                               const TAMM_SIZE& nob, bool transpose) {
  T         residual, energy;
  Scheduler sch{ec};
  // Tensor<T> d_r1_residual{}, d_r2_residual{};
  // Tensor<T>::allocate(&ec,d_r1_residual, d_r2_residual);
  // clang-format off
  sch
    (d_r1_residual() = d_r1()  * d_r1())
    (d_r2_residual() = d_r2()  * d_r2())
    .execute();
  // clang-format on

  auto l0 = [&]() {
    T r1     = get_scalar(d_r1_residual);
    T r2     = get_scalar(d_r2_residual);
    r1       = 0.5 * std::sqrt(r1);
    r2       = 0.5 * std::sqrt(r2);
    energy   = get_scalar(de);
    residual = std::max(r1, r2);
  };

  auto l1 = [&]() { jacobi(ec, d_r1, d_t1, -1.0 * zshiftl, transpose, p_evl_sorted, noa, nob); };
  auto l2 = [&]() { jacobi(ec, d_r2, d_t2, -2.0 * zshiftl, transpose, p_evl_sorted, noa, nob); };

  l0();
  l1();
  l2();

  // Tensor<T>::deallocate(d_r1_residual, d_r2_residual);

  return {residual, energy};
}

template<typename T>
std::pair<double, double>
rest_cs(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1, Tensor<T>& d_r2,
        Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& de, Tensor<T>& d_r1_residual,
        Tensor<T>& d_r2_residual, std::vector<T>& p_evl_sorted, T zshiftl, const TAMM_SIZE& noa,
        const TAMM_SIZE& nva, bool transpose, const bool not_spin_orbital) {
  T         residual, energy;
  Scheduler sch{ec};
  // Tensor<T> d_r1_residual{}, d_r2_residual{};
  // Tensor<T>::allocate(&ec,d_r1_residual, d_r2_residual);
  // clang-format off
  sch
    (d_r1_residual() = d_r1()  * d_r1())
    (d_r2_residual() = d_r2()  * d_r2())
    .execute();
  // clang-format on

  auto l0 = [&]() {
    T r1     = get_scalar(d_r1_residual);
    T r2     = get_scalar(d_r2_residual);
    r1       = 0.5 * std::sqrt(r1);
    r2       = 0.5 * std::sqrt(r2);
    energy   = get_scalar(de);
    residual = std::max(r1, r2);
  };

  auto l1 = [&]() {
    jacobi_cs(ec, d_r1, d_t1, -1.0 * zshiftl, transpose, p_evl_sorted, noa, nva, not_spin_orbital);
  };
  auto l2 = [&]() {
    jacobi_cs(ec, d_r2, d_t2, -2.0 * zshiftl, transpose, p_evl_sorted, noa, nva, not_spin_orbital);
  };

  l0();
  l1();
  l2();

  // Tensor<T>::deallocate(d_r1_residual, d_r2_residual);
  return {residual, energy};
}

void print_ccsd_header(const bool do_print, std::string mname) {
  if(do_print) {
    if(mname.empty()) mname = "CCSD";
    const auto mksp = std::string(10, ' ');
    std::cout << std::endl << std::endl;
    std::cout << " " << mname << " iterations" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "  Iter     Residuum" << mksp << "Correlation" << mksp << "Time(s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }
}

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
             bool ccsd_restart) {
  // auto rank = ec.pg().rank();

  TiledIndexSpace O = MO("occ");
  TiledIndexSpace V = MO("virt");

  std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);

  // auto lambda2 = [&](const IndexVector& blockid, span<T> buf){
  //     if(blockid[0] != blockid[1]) {
  //         for(auto i = 0U; i < buf.size(); i++) buf[i] = 0;
  //     }
  // };

  // update_tensor(d_f1(),lambda2);

  std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;
  Tensor<T>              d_r1{{V, O}, {1, 1}};
  Tensor<T>              d_r2{{V, V, O, O}, {2, 2}};

  if(!ccsd_restart) {
    for(decltype(ndiis) i = 0; i < ndiis; i++) {
      d_r1s.push_back(Tensor<T>{{V, O}, {1, 1}});
      d_r2s.push_back(Tensor<T>{{V, V, O, O}, {2, 2}});
      d_t1s.push_back(Tensor<T>{{V, O}, {1, 1}});
      d_t2s.push_back(Tensor<T>{{V, V, O, O}, {2, 2}});
      Tensor<T>::allocate(&ec, d_r1s[i], d_r2s[i], d_t1s[i], d_t2s[i]);
    }
    Tensor<T>::allocate(&ec, d_r1, d_r2);
  }

  Tensor<T> d_t1{{V, O}, {1, 1}};
  Tensor<T> d_t2{{V, V, O, O}, {2, 2}};

  Tensor<T>::allocate(&ec, d_t1, d_t2);

  // clang-format off
  Scheduler{ec}   
  (d_t1() = 0)
  (d_t2() = 0)
  .execute();
  // clang-format on

  return std::make_tuple(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s);
}

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors_cs(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
                bool ccsd_restart) {
  // auto rank = ec.pg().rank();

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  // const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  // const int vbtiles = MO("virt_beta").num_tiles();

  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;
  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);

  // auto lambda2 = [&](const IndexVector& blockid, span<T> buf){
  //     if(blockid[0] != blockid[1]) {
  //         for(auto i = 0U; i < buf.size(); i++) buf[i] = 0;
  //     }
  // };

  // update_tensor(d_f1(),lambda2);

  std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;
  Tensor<T>              d_r1{{v_alpha, o_alpha}, {1, 1}};
  Tensor<T>              d_r2{{v_alpha, v_beta, o_alpha, o_beta}, {2, 2}};

  if(!ccsd_restart) {
    for(decltype(ndiis) i = 0; i < ndiis; i++) {
      d_r1s.push_back(Tensor<T>{{v_alpha, o_alpha}, {1, 1}});
      d_r2s.push_back(Tensor<T>{{v_alpha, v_beta, o_alpha, o_beta}, {2, 2}});
      d_t1s.push_back(Tensor<T>{{v_alpha, o_alpha}, {1, 1}});
      d_t2s.push_back(Tensor<T>{{v_alpha, v_beta, o_alpha, o_beta}, {2, 2}});
      Tensor<T>::allocate(&ec, d_r1s[i], d_r2s[i], d_t1s[i], d_t2s[i]);
    }
    Tensor<T>::allocate(&ec, d_r1, d_r2);
  }

  Tensor<T> d_t1{{v_alpha, o_alpha}, {1, 1}};
  Tensor<T> d_t2{{v_alpha, v_beta, o_alpha, o_beta}, {2, 2}};

  Tensor<T>::allocate(&ec, d_t1, d_t2);

  // clang-format off
  Scheduler{ec}   
  (d_t1() = 0)
  (d_t2() = 0)
  .execute();
  // clang-format on

  return std::make_tuple(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s);
}

void ccsd_stats(ExecutionContext& ec, double hf_energy, double residual, double energy,
                double thresh) {
  auto rank      = ec.pg().rank();
  bool ccsd_conv = residual < thresh;
  if(rank == 0) {
    std::cout << std::string(66, '-') << std::endl;
    if(ccsd_conv) {
      std::cout << " Iterations converged" << std::endl;
      std::cout.precision(15);
      std::cout << " CCSD correlation energy / hartree =" << std::setw(26) << std::right << energy
                << std::endl;
      std::cout << " CCSD total energy / hartree       =" << std::setw(26) << std::right
                << energy + hf_energy << std::endl;
    }
  }
  if(!ccsd_conv) {
    ec.pg().barrier();
    tamm_terminate("ERROR: CCSD calculation does not converge!");
  }
}

template<typename T>
V2Tensors<T> setupV2Tensors(ExecutionContext& ec, Tensor<T> cholVpr, ExecutionHW ex_hw,
                            std::vector<std::string> blocks) {
  TiledIndexSpace MO    = cholVpr.tiled_index_spaces()[0]; // MO
  TiledIndexSpace CI    = cholVpr.tiled_index_spaces()[2]; // CI
  auto [cind]           = CI.labels<1>("all");
  auto [h1, h2, h3, h4] = MO.labels<4>("occ");
  auto [p1, p2, p3, p4] = MO.labels<4>("virt");

  V2Tensors<T> v2tensors(blocks);
  v2tensors.allocate(ec, MO);
  Scheduler sch{ec};

  for(auto x: blocks) {
    // clang-format off
    if      (x == "ijab") {
      sch( v2tensors.v2ijab(h1,h2,p1,p2)      =   1.0 * cholVpr(h1,p1,cind) * cholVpr(h2,p2,cind) )
         ( v2tensors.v2ijab(h1,h2,p1,p2)     +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) );
    }

    else if (x == "iajb") {
      sch( v2tensors.v2iajb(h1,p1,h2,p2)      =   1.0 * cholVpr(h1,h2,cind) * cholVpr(p1,p2,cind) )
         ( v2tensors.v2iajb(h1,p1,h2,p2)     +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) );
    }
    else if (x == "ijka") {
      sch( v2tensors.v2ijka(h1,h2,h3,p1)      =   1.0 * cholVpr(h1,h3,cind) * cholVpr(h2,p1,cind) )
         ( v2tensors.v2ijka(h1,h2,h3,p1)     +=  -1.0 * cholVpr(h2,h3,cind) * cholVpr(h1,p1,cind) );
    }
    else if (x == "ijkl") {
      sch( v2tensors.v2ijkl(h1,h2,h3,h4)      =   1.0 * cholVpr(h1,h3,cind) * cholVpr(h2,h4,cind) )
         ( v2tensors.v2ijkl(h1,h2,h3,h4)     +=  -1.0 * cholVpr(h1,h4,cind) * cholVpr(h2,h3,cind) );
    }
    else if (x == "iabc") {
      sch( v2tensors.v2iabc(h1,p1,p2,p3)      =   1.0 * cholVpr(h1,p2,cind) * cholVpr(p1,p3,cind) )
         ( v2tensors.v2iabc(h1,p1,p2,p3)     +=  -1.0 * cholVpr(h1,p3,cind) * cholVpr(p1,p2,cind) );
    }
    else if (x == "abcd") {
      sch( v2tensors.v2abcd(p1,p2,p3,p4)      =   1.0 * cholVpr(p1,p3,cind) * cholVpr(p2,p4,cind) )
         ( v2tensors.v2abcd(p1,p2,p3,p4)     +=  -1.0 * cholVpr(p1,p4,cind) * cholVpr(p2,p3,cind) );
    }
    // clang-format on
  }

  sch.execute(ex_hw);

  return v2tensors;
}

template<typename T>
Tensor<T> setupV2(ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& CI, Tensor<T> cholVpr,
                  const tamm::Tile chol_count, ExecutionHW hw, bool anti_sym) {
  auto rank = ec.pg().rank();

  TiledIndexSpace N = MO("all");
  auto [cindex]     = CI.labels<1>("all");
  auto [p, q, r, s] = MO.labels<4>("all");

  // Spin here is defined as spin(p)=spin(r) and spin(q)=spin(s) which is not currently not
  // supported by TAMM.
  //  Tensor<T> d_a2{{N,N,N,N},{2,2}};
  // For V2, spin(p)+spin(q) == spin(r)+spin(s)
  Tensor<T> d_v2{{N, N, N, N}, {2, 2}};
  Tensor<T>::allocate(&ec, d_v2);

  auto cc_t1 = std::chrono::high_resolution_clock::now();
  // clang-format off
  Scheduler sch{ec};
  sch(d_v2(p, q, r, s)  = cholVpr(p, r, cindex) * cholVpr(q, s, cindex));
  if(anti_sym)
    sch(d_v2(p, q, r, s) += -1.0 * cholVpr(p, s, cindex) * cholVpr(q, r, cindex));
  sch.execute(hw);
  // clang-format on

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double v2_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << std::endl
              << "Time to reconstruct V2: " << std::fixed << std::setprecision(2) << v2_time
              << " secs" << std::endl;

  // Tensor<T>::deallocate(d_a2);
  return d_v2;
}

template<typename T>
void cc_print(ChemEnv& chem_env, Tensor<T> d_t1, Tensor<T> d_t2, std::string files_prefix) {
  SystemData&       sys_data     = chem_env.sys_data;
  CCSDOptions&      ccsd_options = chem_env.ioptions.ccsd_options;
  ExecutionContext& ec           = get_ec(d_t1());

  if(ccsd_options.tamplitudes.first) {
    if(ec.print()) {
      auto printtol = ccsd_options.tamplitudes.second;
      std::cout << std::endl
                << "Threshold for printing amplitudes set to: " << printtol << std::endl;
      std::cout << "T1, T2 amplitudes written to files: " << files_prefix + ".print_t1amp.txt"
                << ", " << files_prefix + ".print_t2amp.txt" << std::endl
                << std::endl;
      print_max_above_threshold(d_t1, printtol, files_prefix + ".print_t1amp.txt");
      print_max_above_threshold(d_t2, printtol, files_prefix + ".print_t2amp.txt");
    }
  }

  if(ccsd_options.ccsd_diagnostics) {
    const bool       rhf     = sys_data.is_restricted;
    const TensorType t1_norm = tamm::norm(d_t1);
    TensorType       t1_diag = std::sqrt(t1_norm * t1_norm * 1.0 / sys_data.nelectrons);
    TensorType       d1_diag{}, d2_diag{};

    tamm::TiledIndexSpace v1_tis = d_t1.tiled_index_spaces()[0];
    tamm::TiledIndexSpace o1_tis = d_t1.tiled_index_spaces()[1];
    auto [h1_1, h2_1]            = o1_tis.labels<2>("all");
    auto [p1_1, p2_1]            = v1_tis.labels<2>("all");

    tamm::TiledIndexSpace v2_tis = d_t2.tiled_index_spaces()[1];
    tamm::TiledIndexSpace o2_tis = d_t2.tiled_index_spaces()[3];
    auto [h1_2, h2_2]            = o2_tis.labels<2>("all");
    auto [p1_2, p2_2]            = v2_tis.labels<2>("all");

    Scheduler          sch{ec};
    Tensor<TensorType> d1_ij{o1_tis, o1_tis}, d1_ab{v1_tis, v1_tis};
    Tensor<TensorType> d2_ij{o1_tis, o1_tis}, d2_ab{v1_tis, v1_tis};

    if(rhf) {
      // clang-format off
      sch.allocate(d1_ij,d1_ab,d2_ij,d2_ab)
      // D1 diagnostic: Janssen, et. al Chem. Phys. Lett. 290 (1998) 423
      (d1_ab(p1_1,p2_1) = d_t1(p1_1,h1_1)*d_t1(p2_1,h1_1))
      (d1_ij(h1_1,h2_1) = d_t1(p1_1,h1_1)*d_t1(p1_1,h2_1))
      // D2 diagnostic: Nielsen, et. al Chem. Phys. Lett. 310 (1999) 568
      (d2_ab(p1_1,p2_1) = d_t2(p1_1,p1_2,h1_1,h1_2)*d_t2(p2_1,p1_2,h1_1,h1_2))
      (d2_ij(h1_1,h2_1) = d_t2(p1_1,p1_2,h1_1,h1_2)*d_t2(p1_1,p1_2,h2_1,h1_2))
      .execute(ec.exhw());
      // clang-format on
    }

    if(ec.print()) {
      auto get_diag_val = [&](const Tensor<T>& diag) {
        Matrix                                diag_eig = tamm_to_eigen_matrix(diag);
        Eigen::SelfAdjointEigenSolver<Matrix> ev_diag(diag_eig);
        auto                                  evals = ev_diag.eigenvalues();
        evals                                       = (evals.array().abs()).sqrt();
        return *(std::max_element(evals.data(), evals.data() + evals.rows()));
      };

      if(rhf) {
        d1_diag = std::max(get_diag_val(d1_ij), get_diag_val(d1_ab));
        d2_diag = std::max(get_diag_val(d2_ij), get_diag_val(d2_ab));
      }
      std::cout << std::fixed << std::setprecision(12);
      std::cout << "CC T1 diagnostic = " << t1_diag << std::endl;
      std::cout << "CC D1 diagnostic = " << d1_diag << std::endl;
      std::cout << "CC D2 diagnostic = " << d2_diag << std::endl;
    }
    if(rhf) free_tensors(d1_ij, d1_ab, d2_ij, d2_ab);
  }
}

template<typename T>
std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, TAMM_SIZE, tamm::Tile, TiledIndexSpace>
exachem::cd_svd::cd_svd_driver(ChemEnv& chem_env, ExecutionContext& ec, TiledIndexSpace& MO,
                               TiledIndexSpace& AO, Tensor<T> C_AO, Tensor<T> F_AO,
                               Tensor<T> C_beta_AO, Tensor<T> F_beta_AO, libint2::BasisSet& shells,
                               std::vector<size_t>& shell_tile_map, bool readv2,
                               std::string cholfile, bool is_dlpno, bool is_mso) {
  SystemData& sys_data        = chem_env.sys_data;
  CDOptions   cd_options      = chem_env.ioptions.cd_options;
  auto        diagtol         = cd_options.diagtol; // tolerance for the max. diagonal
  cd_options.max_cvecs_factor = 2 * std::abs(std::log10(diagtol));
  // TODO
  tamm::Tile max_cvecs = cd_options.max_cvecs_factor * sys_data.nbf;

  std::cout << std::defaultfloat;
  auto rank = ec.pg().rank();
  if(rank == 0) cd_options.print();

  TiledIndexSpace N = MO("all");

  Tensor<T> d_f1{{N, N}, {1, 1}};
  Tensor<T> lcao{AO, N};
  Tensor<T>::allocate(&ec, d_f1, lcao);

  auto      hf_t1      = std::chrono::high_resolution_clock::now();
  TAMM_SIZE chol_count = 0;

  // std::tie(V2) =
  Tensor<T> cholVpr;
  auto      itile_size = chem_env.ioptions.cd_options.itilesize;

  sys_data.freeze_atomic    = chem_env.ioptions.ccsd_options.freeze_atomic;
  sys_data.n_frozen_core    = chem_env.get_nfcore();
  sys_data.n_frozen_virtual = chem_env.ioptions.ccsd_options.freeze_virtual;
  bool do_freeze            = sys_data.n_frozen_core > 0 || sys_data.n_frozen_virtual > 0;

  std::string out_fp    = chem_env.workspace_dir;
  std::string files_dir = out_fp + chem_env.ioptions.scf_options.scf_type;
  std::string lcaofile  = files_dir + "/" + sys_data.output_file_prefix + ".lcao";

  if(!readv2) {
    two_index_transform(chem_env, ec, C_AO, F_AO, C_beta_AO, F_beta_AO, d_f1, shells, lcao,
                        is_dlpno || !is_mso);
    if(!is_dlpno)
      cholVpr = cd_svd(chem_env, ec, MO, AO, chol_count, max_cvecs, shells, lcao, is_mso);
    write_to_disk<TensorType>(lcao, lcaofile);
  }
  else {
    std::ifstream in(cholfile, std::ios::in);
    int           rstatus = 0;
    if(in.is_open()) rstatus = 1;
    if(rstatus == 1) in >> chol_count;
    else tamm_terminate("Error reading " + cholfile);

    if(rank == 0) cout << "Number of cholesky vectors to be read = " << chol_count << endl;

    if(!is_dlpno) update_sysdata(chem_env, MO, is_mso);

    IndexSpace      chol_is{range(0, chol_count)};
    TiledIndexSpace CI{chol_is, static_cast<tamm::Tile>(itile_size)};

    TiledIndexSpace N = MO("all");
    cholVpr = {{N, N, CI}, {SpinPosition::upper, SpinPosition::lower, SpinPosition::ignore}};
    if(!is_dlpno) Tensor<TensorType>::allocate(&ec, cholVpr);
    // Scheduler{ec}(cholVpr()=0).execute();
    read_from_disk(lcao, lcaofile);
  }

  auto   hf_t2 = std::chrono::high_resolution_clock::now();
  double cd_svd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

  if(rank == 0)
    std::cout << std::endl
              << "Total Time taken for Cholesky Decomposition: " << std::fixed
              << std::setprecision(2) << cd_svd_time << " secs" << std::endl;

  Tensor<T>::deallocate(C_AO, F_AO);
  if(sys_data.is_unrestricted) Tensor<T>::deallocate(C_beta_AO, F_beta_AO);

  IndexSpace      chol_is{range(0, chol_count)};
  TiledIndexSpace CI{chol_is, static_cast<tamm::Tile>(itile_size)};

  sys_data.num_chol_vectors                              = chol_count;
  sys_data.results["output"]["CD"]["n_cholesky_vectors"] = chol_count;

  if(rank == 0) sys_data.print();

  if(do_freeze) {
    TiledIndexSpace N_eff = MO("all");
    Tensor<T>       d_f1_new{{N_eff, N_eff}, {1, 1}};
    Tensor<T>::allocate(&ec, d_f1_new);
    if(rank == 0) {
      Matrix f1_eig     = tamm_to_eigen_matrix(d_f1);
      Matrix f1_new_eig = reshape_mo_matrix(chem_env, f1_eig);
      eigen_to_tamm_tensor(d_f1_new, f1_new_eig);
      f1_new_eig.resize(0, 0);
    }
    Tensor<T>::deallocate(d_f1);
    d_f1 = d_f1_new;
  }

  if(!readv2 && chem_env.ioptions.scf_options.mos_txt) {
    Scheduler   sch{ec};
    std::string hcorefile = files_dir + "/scf/" + sys_data.output_file_prefix + ".hcore";
    Tensor<T>   hcore{AO, AO};
    Tensor<T>   hcore_mo{MO, MO};
    Tensor<T>::allocate(&ec, hcore, hcore_mo);
    read_from_disk(hcore, hcorefile);

    auto [mu, nu]   = AO.labels<2>("all");
    auto [mo1, mo2] = MO.labels<2>("all");

    Tensor<T> tmp{MO, AO};
    // clang-format off
    sch.allocate(tmp)
        (tmp(mo1,nu) = lcao(mu,mo1) * hcore(mu,nu))
        (hcore_mo(mo1,mo2) = tmp(mo1,nu) * lcao(nu,mo2))
        .deallocate(tmp,hcore).execute();
    // clang-format on

    ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};
    std::string      mop_dir   = files_dir + "/mos_txt/";
    std::string      mofprefix = mop_dir + out_fp;
    if(!fs::exists(mop_dir)) fs::create_directories(mop_dir);

    Tensor<T> d_v2        = setupV2<T>(ec, MO, CI, cholVpr, chol_count);
    Tensor<T> d_f1_dense  = to_dense_tensor(ec_dense, d_f1);
    Tensor<T> lcao_dense  = to_dense_tensor(ec_dense, lcao);
    Tensor<T> d_v2_dense  = to_dense_tensor(ec_dense, d_v2);
    Tensor<T> hcore_dense = to_dense_tensor(ec_dense, hcore_mo);

    Tensor<T>::deallocate(hcore_mo, d_v2);

    print_dense_tensor(d_v2_dense, mofprefix + ".v2_mo");
    print_dense_tensor(lcao_dense, mofprefix + ".ao2mo");
    print_dense_tensor(d_f1_dense, mofprefix + ".fock_mo");
    print_dense_tensor(hcore_dense, mofprefix + ".hcore_mo");

    Tensor<T>::deallocate(hcore_dense, d_f1_dense, lcao_dense, d_v2_dense);
  }

  return std::make_tuple(cholVpr, d_f1, lcao, chol_count, max_cvecs, CI);
} // END of cd_svd_driver

void exachem::cd::cd_2e_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T = double;

  auto rank = ec.pg().rank();

  scf::scf_driver(ec, chem_env);

  libint2::BasisSet   shells         = chem_env.shells;
  Tensor<T>           C_AO           = chem_env.C_AO;
  Tensor<T>           C_beta_AO      = chem_env.C_beta_AO;
  Tensor<T>           F_AO           = chem_env.F_AO;
  Tensor<T>           F_beta_AO      = chem_env.F_beta_AO;
  TiledIndexSpace     AO_opt         = chem_env.AO_opt;
  TiledIndexSpace     AO_tis         = chem_env.AO_tis;
  std::vector<size_t> shell_tile_map = chem_env.shell_tile_map;

  SystemData  sys_data     = chem_env.sys_data;
  CCSDOptions ccsd_options = chem_env.ioptions.ccsd_options;
  if(rank == 0) ccsd_options.print();

  if(rank == 0)
    cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;

  auto [MO, total_orbitals] = cd_svd::setupMOIS(chem_env);

  std::string out_fp       = chem_env.workspace_dir;
  std::string files_dir    = out_fp + chem_env.ioptions.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + sys_data.output_file_prefix;
  std::string f1file       = files_prefix + ".f1_mo";
  std::string v2file       = files_prefix + ".cholv2";
  std::string cholfile     = files_prefix + ".cholcount";

  bool cd_restart = fs::exists(f1file) && fs::exists(v2file) && fs::exists(cholfile);

  // deallocates F_AO, C_AO
  auto [cholVpr, d_f1, lcao, chol_count, max_cvecs, CI] =
    cd_svd::cd_svd_driver<T>(chem_env, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO, shells,
                             shell_tile_map, cd_restart, cholfile);
  free_tensors(lcao);

  if(!cd_restart && ccsd_options.writet) {
    if(!fs::exists(files_dir)) fs::create_directories(files_dir);

    write_to_disk(d_f1, f1file);
    write_to_disk(cholVpr, v2file);

    if(rank == 0) {
      std::ofstream out(cholfile, std::ios::out);
      if(!out) cerr << "Error opening file " << cholfile << endl;
      out << chol_count << std::endl;
      out.close();
    }
  }

  free_tensors(d_f1, cholVpr);

  ec.flush_and_sync();
  // delete ec;
} // End of cd_2e_driver

using T = double;

template void init_diagonal(ExecutionContext& ec, LabeledTensor<T> ltensor);

template void cc_print<T>(ChemEnv& chem_env, Tensor<T> d_t1, Tensor<T> d_t2,
                          std::string files_prefix);

template void update_r2<T>(ExecutionContext& ec, LabeledTensor<T> ltensor);

template void setup_full_t1t2<T>(ExecutionContext& ec, const TiledIndexSpace& MO,
                                 Tensor<T>& dt1_full, Tensor<T>& dt2_full);

template std::pair<double, double> rest<T>(ExecutionContext& ec, const TiledIndexSpace& MO,
                                           Tensor<T>& d_r1, Tensor<T>& d_r2, Tensor<T>& d_t1,
                                           Tensor<T>& d_t2, Tensor<T>& de, Tensor<T>& d_r1_residual,
                                           Tensor<T>& d_r2_residual, std::vector<T>& p_evl_sorted,
                                           double zshiftl, const TAMM_SIZE& noa,
                                           const TAMM_SIZE& nob, bool transpose);

template std::pair<double, double>
rest_cs<T>(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1, Tensor<T>& d_r2,
           Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& de, Tensor<T>& d_r1_residual,
           Tensor<T>& d_r2_residual, std::vector<T>& p_evl_sorted, double zshiftl,
           const TAMM_SIZE& noa, const TAMM_SIZE& nva, bool transpose, const bool not_spin_orbital);

template std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>,
                    std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
                    std::vector<Tensor<T>>>
setupTensors<T>(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
                bool ccsd_restart);

template std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>,
                    std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
                    std::vector<Tensor<T>>>
setupTensors_cs<T>(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
                   bool ccsd_restart);

// template std::tuple<SystemData, double, libint2::BasisSet, std::vector<size_t>, Tensor<T>,
//                     Tensor<T>, Tensor<T>, Tensor<T>, TiledIndexSpace, TiledIndexSpace, bool>
// hartree_fock_driver<T>(ExecutionContext& ec, const string filename, ECOptions options_map);

template std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, TAMM_SIZE, tamm::Tile, TiledIndexSpace>
exachem::cd_svd::cd_svd_driver<T>(ChemEnv& chem_env, ExecutionContext& ec, TiledIndexSpace& MO,
                                  TiledIndexSpace& AO, Tensor<T> C_AO, Tensor<T> F_AO,
                                  Tensor<T> C_beta_AO, Tensor<T> F_beta_AO,
                                  libint2::BasisSet& shells, std::vector<size_t>& shell_tile_map,
                                  bool readv2, std::string cholfile, bool is_dlpno, bool is_mso);

template V2Tensors<T> setupV2Tensors<T>(ExecutionContext& ec, Tensor<T> cholVpr, ExecutionHW ex_hw,
                                        std::vector<std::string> blocks);

template Tensor<T> setupV2<T>(ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& CI,
                              Tensor<T> cholVpr, const tamm::Tile chol_count, ExecutionHW hw,
                              bool anti_sym);
