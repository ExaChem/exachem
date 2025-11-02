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

template<typename T>
Tensor<T> declare(ChemEnv& chem_env, const std::string& name) {
  auto split = [](const std::string& s, char sep) {
    std::vector<std::string> parts;
    size_t                   i = 0;
    while(i <= s.size()) {
      auto j = s.find(sep, i);
      if(j == std::string::npos) {
        parts.push_back(s.substr(i));
        break;
      }
      parts.push_back(s.substr(i, j - i));
      i = j + 1;
    }
    return parts;
  };

  auto parts = split(name, '_');
  if(parts.size() == 1) return Tensor<T>{}; // no underscore -> empty

  const std::string spin_blk  = (parts.size() == 2 ? parts[0] : parts[1]);
  const std::string dimstring = (parts.size() == 2 ? parts[1] : parts[2]);

  // special case for chol vectors (e.g. dimstring starting with 'Q')
  if(!dimstring.empty() && dimstring[0] == 'Q') return Tensor<T>{{chem_env.is_context.CI}};

  TiledIndexSpace& MO      = chem_env.is_context.MSO;
  const int        otiles  = MO("occ").num_tiles();
  const int        vtiles  = MO("virt").num_tiles();
  const int        oatiles = MO("occ_alpha").num_tiles();
  const int        vatiles = MO("virt_alpha").num_tiles();

  const TiledIndexSpace Oa{MO("occ"), range(oatiles)};
  const TiledIndexSpace Va{MO("virt"), range(vatiles)};
  const TiledIndexSpace Ob{MO("occ"), range(oatiles, otiles)};
  const TiledIndexSpace Vb{MO("virt"), range(vatiles, vtiles)};

  auto pick_space = [&](char s) -> std::pair<const TiledIndexSpace*, const TiledIndexSpace*> {
    if(s == 'a') return {&Oa, &Va};
    if(s == 'b') return {&Ob, &Vb};
    throw std::runtime_error(std::string("Invalid spin character: ") + s + " in " + name);
  };

  const bool       is_chol = (name.find('Q') != std::string::npos);
  TiledIndexSpace& CI      = chem_env.is_context.CI;

  const size_t ndim = spin_blk.size();
  if(dimstring.size() - int(is_chol) != ndim)
    throw std::runtime_error("Spin block and dimension string length mismatch for tensor: " + name);

  std::vector<const TiledIndexSpace*> tis;
  tis.reserve(ndim);
  for(size_t i = 0; i < ndim; ++i) {
    auto [Os, Vs] = pick_space(spin_blk[i]);
    char d        = dimstring[i];
    if(d == 'o') tis.push_back(Os);
    else if(d == 'v') tis.push_back(Vs);
    else
      throw std::runtime_error(std::string("Invalid dimension character: ") + d + " in " +
                               dimstring);
  }

  switch(ndim) {
    case 2: return is_chol ? Tensor<T>{{*tis[0], *tis[1], CI}} : Tensor<T>{{*tis[0], *tis[1]}};
    case 4:
      return is_chol ? Tensor<T>{{*tis[0], *tis[1], *tis[2], *tis[3], CI}}
                     : Tensor<T>{{*tis[0], *tis[1], *tis[2], *tis[3]}};
    case 6:
      return is_chol ? Tensor<T>{{*tis[0], *tis[1], *tis[2], *tis[3], *tis[4], *tis[5], CI}}
                     : Tensor<T>{{*tis[0], *tis[1], *tis[2], *tis[3], *tis[4], *tis[5]}};
    default:
      throw std::runtime_error("Unsupported tensor dimension: " + std::to_string(ndim) +
                               " for tensor: " + name);
  }
}

template<typename T>
TensorMap<T> oei_spin_blocks(Scheduler& sch, ChemEnv& chem_env, const Tensor<T>& oei,
                             bool is_chol) {
  TiledIndexSpace&       MO      = chem_env.is_context.MSO;
  const TiledIndexSpace& O       = MO("occ");
  const TiledIndexSpace& V       = MO("virt");
  const TiledIndexSpace& CI      = chem_env.is_context.CI;
  const int              otiles  = O.num_tiles();
  const int              vtiles  = V.num_tiles();
  const int              oatiles = MO("occ_alpha").num_tiles();
  const int              vatiles = MO("virt_alpha").num_tiles();

  const TiledIndexSpace Oa = {MO("occ"), range(oatiles)};
  const TiledIndexSpace Va = {MO("virt"), range(vatiles)};
  const TiledIndexSpace Ob = {MO("occ"), range(oatiles, otiles)};
  const TiledIndexSpace Vb = {MO("virt"), range(vatiles, vtiles)};

  std::vector<std::string> one_body_blocks = {"aa_oo", "aa_ov", "aa_vo", "aa_vv",
                                              "bb_oo", "bb_ov", "bb_vo", "bb_vv"};

  auto set_label = [&Oa, &Va, &Ob, &Vb, &CI](TiledIndexLabel& label, char spin, char occ) {
    if(spin == 'a') {
      if(occ == 'v') std::tie(label) = Va.labels<1>("all");
      else std::tie(label) = Oa.labels<1>("all");
    }
    else if(occ == 'v') std::tie(label) = Vb.labels<1>("all");
    else std::tie(label) = Ob.labels<1>("all");
  };

  // one body integrals
  TensorMap<T> oei_map;
  for(auto& block: one_body_blocks) {
    if(is_chol) block += 'Q';
    oei_map[block] = declare<T>(chem_env, block);
    sch.allocate(oei_map.at(block));

    TiledIndexLabel p, q, L;
    if(is_chol) std::tie(L) = CI.labels<1>("all");

    set_label(p, block[0], block[3]);
    set_label(q, block[1], block[4]);

    // clang-format off
    if (!is_chol) 
         sch (oei_map.at(block)(p,q)   = oei(p,q));
    else sch (oei_map.at(block)(p,q,L) = oei(p,q,L));
    // clang-format on
  }
  sch.execute();

  return oei_map;
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

template Tensor<double> declare<double>(ChemEnv& chem_env, const std::string& name);

template TensorMap<double> oei_spin_blocks(Scheduler& sch, ChemEnv& chem_env,
                                           const Tensor<double>& oei, bool is_chol);

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
