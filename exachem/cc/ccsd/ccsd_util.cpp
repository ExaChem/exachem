/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/ccsd_util.hpp"
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
std::pair<double, double> rest3(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1,
                                Tensor<T>& d_r2, Tensor<T>& d_r3, Tensor<T>& d_t1, Tensor<T>& d_t2,
                                Tensor<T>& d_t3, Tensor<T>& de, Tensor<T>& d_r1_residual,
                                Tensor<T>& d_r2_residual, std::vector<T>& p_evl_sorted, T zshiftl,
                                const TAMM_SIZE& noa, const TAMM_SIZE& nob, bool transpose) {
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
  auto l3 = [&]() { jacobi(ec, d_r3, d_t3, -2.0 * zshiftl, transpose, p_evl_sorted, noa, nob); };

  l0();
  l1();
  l2();
  l3();

  // Tensor<T>::deallocate(d_r1_residual, d_r2_residual);

  return {residual, energy};
}

template<typename T>
std::pair<double, double>
rest_qed(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1, Tensor<T>& d_r2,
         Tensor<T>& d_r1_1p, Tensor<T>& d_r2_1p, Tensor<T>& d_r1_2p, Tensor<T>& d_r2_2p,
         Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_t1_1p, Tensor<T>& d_t2_1p,
         Tensor<T>& d_t1_2p, Tensor<T>& d_t2_2p, Tensor<T>& de, Tensor<T>& d_r1_residual,
         Tensor<T>& d_r2_residual, Tensor<T>& d_r1_1p_residual, Tensor<T>& d_r2_1p_residual,
         Tensor<T>& d_r1_2p_residual, Tensor<T>& d_r2_2p_residual, std::vector<T>& p_evl_sorted,
         T zshiftl, double omega, const TAMM_SIZE& noa, const TAMM_SIZE& nob, bool transpose) {
  T         residual, energy; //, residual1, residual2, residual3, residual4;
  Scheduler sch{ec};
  // Tensor<T> d_r1_residual{}, d_r2_residual{};
  // Tensor<T>::allocate(&ec,d_r1_residual, d_r2_residual);
  // clang-format off
   sch
    (d_r1_residual()     =    d_r1() *    d_r1())
    (d_r2_residual()     =    d_r2() *    d_r2())
    (d_r1_1p_residual()  = d_r1_1p() * d_r1_1p())
    (d_r2_1p_residual()  = d_r2_1p() * d_r2_1p())
    (d_r1_2p_residual()  = d_r1_2p() * d_r1_2p())
    (d_r2_2p_residual()  = d_r2_2p() * d_r2_2p())
    .execute();

   auto l0 = [&]() {
    T r1    = 0.5 * std::sqrt(get_scalar(d_r1_residual));
    T r2    = 0.5 * std::sqrt(get_scalar(d_r2_residual));
    T r1_1p = 0.5 * std::sqrt(get_scalar(d_r1_1p_residual));
    T r2_1p = 0.5 * std::sqrt(get_scalar(d_r2_1p_residual));
    T r1_2p = 0.5 * std::sqrt(get_scalar(d_r1_2p_residual));
    T r2_2p = 0.5 * std::sqrt(get_scalar(d_r2_2p_residual));
    energy = get_scalar(de);
    residual = std::max({r1, r2, r1_1p, r2_1p, r1_2p, r2_2p});
  };

  auto l1    = [&]() { jacobi(ec, d_r1, d_t1, -1.0 * zshiftl, transpose, p_evl_sorted, noa, nob); };
  auto l2    = [&]() { jacobi(ec, d_r2, d_t2, -2.0 * zshiftl, transpose, p_evl_sorted, noa, nob); };
  auto l1_1p = [&]() { jacobi(ec, d_r1_1p, d_t1_1p, -1.0 * zshiftl - omega, transpose, p_evl_sorted, noa, nob); };
  auto l2_1p = [&]() { jacobi(ec, d_r2_1p, d_t2_1p, -2.0 * zshiftl - omega, transpose, p_evl_sorted, noa, nob); };
  auto l1_2p = [&]() { jacobi(ec, d_r1_2p, d_t1_2p, -1.0 * zshiftl - 2.0*omega, transpose, p_evl_sorted, noa, nob); };
  auto l2_2p = [&]() { jacobi(ec, d_r2_2p, d_t2_2p, -2.0 * zshiftl - 2.0*omega, transpose, p_evl_sorted, noa, nob); };


  l0();
  l1();
  l2();
  l1_1p();
  l2_1p();
  l1_2p();
  l2_2p();

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
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>,
           Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors_qed(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
                 bool ccsd_restart) {
  // auto rank = ec.pg().rank();

  TiledIndexSpace O = MO("occ");
  TiledIndexSpace V = MO("virt");

  std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);

  std::vector<Tensor<T>> d_r1s, d_r2s, d_r1_1ps, d_r2_1ps, d_r1_2ps, d_r2_2ps, d_t1s, d_t2s,
    d_t1_1ps, d_t2_1ps, d_t1_2ps, d_t2_2ps;

  Tensor<T> d_r1{{V, O}, {1, 1}};
  Tensor<T> d_r2{{V, V, O, O}, {2, 2}};
  Tensor<T> d_r0_1p{};
  Tensor<T> d_r1_1p{{V, O}, {1, 1}};
  Tensor<T> d_r2_1p{{V, V, O, O}, {2, 2}};
  Tensor<T> d_r0_2p{};
  Tensor<T> d_r1_2p{{V, O}, {1, 1}};
  Tensor<T> d_r2_2p{{V, V, O, O}, {2, 2}};

  if(!ccsd_restart) {
    for(decltype(ndiis) i = 0; i < ndiis; i++) {
      d_r1s.push_back(Tensor<T>{{V, O}, {1, 1}});
      d_r2s.push_back(Tensor<T>{{V, V, O, O}, {2, 2}});
      d_r1_1ps.push_back(Tensor<T>{{V, O}, {1, 1}});
      d_r2_1ps.push_back(Tensor<T>{{V, V, O, O}, {2, 2}});
      d_r1_2ps.push_back(Tensor<T>{{V, O}, {1, 1}});
      d_r2_2ps.push_back(Tensor<T>{{V, V, O, O}, {2, 2}});
      d_t1s.push_back(Tensor<T>{{V, O}, {1, 1}});
      d_t2s.push_back(Tensor<T>{{V, V, O, O}, {2, 2}});
      d_t1_1ps.push_back(Tensor<T>{{V, O}, {1, 1}});
      d_t2_1ps.push_back(Tensor<T>{{V, V, O, O}, {2, 2}});
      d_t1_2ps.push_back(Tensor<T>{{V, O}, {1, 1}});
      d_t2_2ps.push_back(Tensor<T>{{V, V, O, O}, {2, 2}});
      Tensor<T>::allocate(&ec, d_r1s[i], d_r2s[i], d_r1_1ps[i], d_r2_1ps[i], d_r1_2ps[i],
                          d_r2_2ps[i], d_t1s[i], d_t2s[i], d_t1_1ps[i], d_t2_1ps[i], d_t1_2ps[i],
                          d_t2_2ps[i]);
    }
    Tensor<T>::allocate(&ec, d_r1, d_r2, d_r0_1p, d_r1_1p, d_r2_1p, d_r0_2p, d_r1_2p, d_r2_2p);
  }

  Tensor<T> d_t1{{V, O}, {1, 1}};
  Tensor<T> d_t2{{V, V, O, O}, {2, 2}};
  Tensor<T> d_t1_1p{{V, O}, {1, 1}};
  Tensor<T> d_t2_1p{{V, V, O, O}, {2, 2}};
  Tensor<T> d_t1_2p{{V, O}, {1, 1}};
  Tensor<T> d_t2_2p{{V, V, O, O}, {2, 2}};

  Tensor<T>::allocate(&ec, d_t1, d_t2, d_t1_1p, d_t2_1p, d_t1_2p, d_t2_2p);

  // clang-format off
  Scheduler{ec}
  (d_t1() = 0)
  (d_t2() = 0)
  (d_t1_1p() = 0)
  (d_t2_1p() = 0)
  (d_t1_2p() = 0)
  (d_t2_2p() = 0)
  .execute();
  // clang-format on

  return std::make_tuple(p_evl_sorted, d_t1, d_t2, d_t1_1p, d_t2_1p, d_t1_2p, d_t2_2p, d_r1, d_r2,
                         d_r0_1p, d_r1_1p, d_r2_1p, d_r0_2p, d_r1_2p, d_r2_2p, d_r1s, d_r2s,
                         d_r1_1ps, d_r2_1ps, d_r1_2ps, d_r2_2ps, d_t1s, d_t2s, d_t1_1ps, d_t2_1ps,
                         d_t1_2ps, d_t2_2ps);
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
                double thresh, std::string task_str) {
  auto rank      = ec.pg().rank();
  bool ccsd_conv = residual < thresh;
  if(rank == 0) {
    std::cout << std::string(66, '-') << std::endl;
    if(ccsd_conv) {
      std::cout << " Iterations converged" << std::endl;
      std::cout.precision(15);
      std::cout << " " << task_str << " correlation energy / hartree =" << std::setw(26)
                << std::right << energy << std::endl;
      std::cout << " " << task_str << " total energy / hartree       =" << std::setw(26)
                << std::right << energy + hf_energy << std::endl;
    }
  }
  if(!ccsd_conv) {
    ec.pg().barrier();
    std::string err_msg = "[ERROR] " + task_str + " calculation does not converge!";
    tamm_terminate(err_msg);
  }
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
rest3<T>(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1, Tensor<T>& d_r2,
         Tensor<T>& d_r3, Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_t3, Tensor<T>& de,
         Tensor<T>& d_r1_residual, Tensor<T>& d_r2_residual, std::vector<T>& p_evl_sorted,
         double zshiftl, const TAMM_SIZE& noa, const TAMM_SIZE& nob, bool transpose);

template std::pair<double, double>
rest_qed<T>(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1, Tensor<T>& d_r2,
            Tensor<T>& d_r1_1p, Tensor<T>& d_r2_1p, Tensor<T>& d_r1_2p, Tensor<T>& d_r2_2p,
            Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_t1_1p, Tensor<T>& d_t2_1p,
            Tensor<T>& d_t1_2p, Tensor<T>& d_t2_2p, Tensor<T>& de, Tensor<T>& d_r1_residual,
            Tensor<T>& d_r2_residual, Tensor<T>& d_r1_1p_residual, Tensor<T>& d_r2_1p_residual,
            Tensor<T>& d_r1_2p_residual, Tensor<T>& d_r2_2p_residual, std::vector<T>& p_evl_sorted,
            double zshiftl, double omega, const TAMM_SIZE& noa, const TAMM_SIZE& nob,
            bool transpose);

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

template std::tuple<
  std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>,
  Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>,
  std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
  std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
  std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors_qed(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
                 bool ccsd_restart);

template std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>,
                    std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
                    std::vector<Tensor<T>>>
setupTensors_cs<T>(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
                   bool ccsd_restart);
