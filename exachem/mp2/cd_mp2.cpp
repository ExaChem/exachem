/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

//  #include "exachem/cc/ccsd/ccsd_util.hpp"

#include "exachem/mp2/cd_mp2.hpp"

namespace exachem::mp2 {
void CDMP2::initialize_index_spaces(const ChemEnv& chem_env, CDMP2State& state) {
  const TiledIndexSpace MO = chem_env.is_context.MSO;
  const TiledIndexSpace CI = chem_env.is_context.CI;

  int             otiles{0}, oatiles{0}, vatiles{0}, vtiles{0};
  TiledIndexSpace N = MO("all");
  state.O           = MO("occ");
  state.V           = MO("virt");

  otiles  = state.O.num_tiles();
  vtiles  = state.V.num_tiles();
  oatiles = MO("occ_alpha").num_tiles();
  vatiles = MO("virt_alpha").num_tiles();

  state.o_alpha = {MO("occ"), range(oatiles)};
  state.v_alpha = {MO("virt"), range(vatiles)};
  state.o_beta  = {MO("occ"), range(oatiles, otiles)};
  state.v_beta  = {MO("virt"), range(vatiles, vtiles)};

  std::tie(state.p1, state.p2) = MO.labels<2>("virt");
  std::tie(state.h1, state.h2) = MO.labels<2>("occ");
  std::tie(state.cind)         = CI.labels<1>("all");

  state.hf_energy = chem_env.scf_context.hf_energy;

} // initialize_index_spaces

void CDMP2::initialize_eigenvalues(const ChemEnv& chem_env, CDMP2State& state) {
  std::vector<T> p_evl_sorted = tamm::diagonal(chem_env.cd_context.d_f1);

  // split into occupied and virtual parts
  state.p_evl_sorted_occ.assign(chem_env.sys_data.nocc, T());
  state.p_evl_sorted_virt.assign(chem_env.sys_data.nvir, T());

  std::copy(p_evl_sorted.begin(), p_evl_sorted.begin() + chem_env.sys_data.nocc,
            state.p_evl_sorted_occ.begin());
  std::copy(p_evl_sorted.begin() + chem_env.sys_data.nocc, p_evl_sorted.end(),
            state.p_evl_sorted_virt.begin());
} // initialize_eigenvalues

//------------------------------------------------------------

void CDMP2::compute_dtmp(ExecutionContext& ec, CDMP2State& state) {
  Scheduler sch{ec};

  auto dtmp_lambda = [&state](const IndexVector& bid) {
    double            denominator = 0.0;
    const IndexVector blockid     = internal::translate_blockid(bid, state.dtmp());
    const TAMM_SIZE   size        = state.dtmp.block_size(blockid);
    std::vector<T>    buf(size);
    const auto        block_dims   = state.dtmp.block_dims(blockid);
    const auto        block_offset = state.dtmp.block_offsets(blockid);
    size_t            c            = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
        for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
          for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++, c++) {
            denominator = state.p_evl_sorted_virt[i] + state.p_evl_sorted_virt[j] -
                          state.p_evl_sorted_occ[k] - state.p_evl_sorted_occ[l];
            buf[c] = 1.0 / denominator;
          }
        }
      }
    }
    state.dtmp.put(blockid, buf);
  };
  sch.allocate(state.dtmp, state.v2ijab).execute();
  block_for(ec, state.dtmp(), dtmp_lambda);
} // compute_dtmp

void CDMP2::compute_closed_shell_mp2(ExecutionContext& ec, Scheduler& sch, CDMP2State& state) {
  // For ClosedShell O=o_alpha, V=v_alpha
  TiledIndexLabel p1_va, p2_va, h1_oa, h2_oa;
  std::tie(p1_va, p2_va) = state.v_alpha.labels<2>("all");
  std::tie(h1_oa, h2_oa) = state.o_alpha.labels<2>("all");

  Tensor<T> v2tmp_alpha{{state.o_alpha, state.o_alpha, state.v_alpha, state.v_alpha}, {2, 2}};

  // clang-format off
  Tensor<T> mp2_ea{};
  sch.allocate(mp2_ea, v2tmp_alpha)
    (v2tmp_alpha(h1_oa, h2_oa, p1_va, p2_va)  =  2.0 * state.v2ijab(h1_oa, h2_oa, p1_va, p2_va) * state.v2ijab(h1_oa, h2_oa, p1_va, p2_va))
    (v2tmp_alpha(h1_oa, h2_oa, p1_va, p2_va) += -1.0 * state.v2ijab(h1_oa, h2_oa, p1_va, p2_va) * state.v2ijab(h2_oa, h1_oa, p1_va, p2_va))
    (mp2_ea() = state.dtmp(p1_va, p2_va, h1_oa, h2_oa) * v2tmp_alpha(h1_oa, h2_oa, p1_va, p2_va))
    .deallocate(v2tmp_alpha)
    .execute(ec.exhw());
  // clang-format on
  state.mp2_energy       = -1.0 * get_scalar(mp2_ea);
  state.mp2_alpha_energy = state.mp2_energy;
  sch.deallocate(mp2_ea).execute();
} // compute_closed_shell_mp2

void CDMP2::compute_open_shell_mp2(ExecutionContext& ec, Scheduler& sch, CDMP2State& state) {
  // Open Shell expressions reference: https://pubs.acs.org/doi/10.1021/acs.jctc.6b00015
  TiledIndexLabel p1_va, p2_va, h1_oa, h2_oa;
  TiledIndexLabel p1_vb, p2_vb, h1_ob, h2_ob;
  std::tie(p1_va, p2_va) = state.v_alpha.labels<2>("all");
  std::tie(h1_oa, h2_oa) = state.o_alpha.labels<2>("all");
  std::tie(p1_vb, p2_vb) = state.v_beta.labels<2>("all");
  std::tie(h1_ob, h2_ob) = state.o_beta.labels<2>("all");

  // clang-format off
  Tensor<T> mp2_ea{};
  Tensor<T> mp2_eb{};
  Tensor<T> mp2_ec{};
  Tensor<T> v2tmp_alpha{{state.o_alpha, state.o_alpha, state.v_alpha, state.v_alpha}, {2, 2}};
  Tensor<T> v2tmp_beta{{state.o_beta, state.o_beta, state.v_beta, state.v_beta}, {2, 2}};
  Tensor<T> v2tmp_ab{{state.o_alpha, state.o_beta, state.v_alpha, state.v_beta}, {2, 2}};

  sch.allocate(mp2_ea, mp2_eb, mp2_ec, v2tmp_alpha)
    (v2tmp_alpha(h1_oa, h2_oa, p1_va, p2_va)  =  1.0 * state.v2ijab(h1_oa, h2_oa, p1_va, p2_va) * state.v2ijab(h1_oa, h2_oa, p1_va, p2_va))
    (v2tmp_alpha(h1_oa, h2_oa, p1_va, p2_va) += -1.0 * state.v2ijab(h1_oa, h2_oa, p1_va, p2_va) * state.v2ijab(h1_oa, h2_oa, p2_va, p1_va))
    (mp2_ea() = state.dtmp(p1_va, p2_va, h1_oa, h2_oa) * v2tmp_alpha(h1_oa, h2_oa, p1_va, p2_va)).deallocate(v2tmp_alpha);

  sch.allocate(v2tmp_beta)
    (v2tmp_beta(h1_ob, h2_ob, p1_vb, p2_vb)  =  1.0 * state.v2ijab(h1_ob, h2_ob, p1_vb, p2_vb) * state.v2ijab(h1_ob, h2_ob, p1_vb, p2_vb))
    (v2tmp_beta(h1_ob, h2_ob, p1_vb, p2_vb) += -1.0 * state.v2ijab(h1_ob, h2_ob, p1_vb, p2_vb) * state.v2ijab(h1_ob, h2_ob, p2_vb, p1_vb))
    (mp2_eb() = state.dtmp(p1_vb, p2_vb, h1_ob, h2_ob) * v2tmp_beta(h1_ob, h2_ob, p1_vb, p2_vb)).deallocate(v2tmp_beta);

  sch.allocate(v2tmp_ab)
    (v2tmp_ab(h1_oa, h2_ob, p1_va, p2_vb) =  state.v2ijab(h1_oa, h2_ob, p1_va, p2_vb) * state.v2ijab(h1_oa, h2_ob, p1_va, p2_vb))
    (mp2_ec() = state.dtmp(p1_va, p2_vb, h1_oa, h2_ob) * v2tmp_ab(h1_oa, h2_ob, p1_va, p2_vb)).deallocate(v2tmp_ab);
  // clang-format on

  sch.execute(ec.exhw());

  state.mp2_alpha_energy = -0.5 * get_scalar(mp2_ea);
  state.mp2_beta_energy  = -0.5 * get_scalar(mp2_eb);
  const T mp2_ab_energy  = -1.0 * get_scalar(mp2_ec);
  state.mp2_energy       = state.mp2_alpha_energy + state.mp2_beta_energy + mp2_ab_energy;
  sch.deallocate(mp2_ea, mp2_eb, mp2_ec).execute();
} // compute_open_shell_mp2

void CDMP2::run(ExecutionContext& ec, ChemEnv& chem_env) {
  cholesky_2e::cholesky_2e_driver(ec, chem_env);
  CDMP2State state;
  initialize_eigenvalues(chem_env, state);
  initialize_index_spaces(chem_env, state);
  state.dtmp   = Tensor<T>{{state.V, state.V, state.O, state.O}, {2, 2}};
  state.v2ijab = Tensor<T>{{state.O, state.O, state.V, state.V}, {2, 2}};

  const auto   rank    = ec.pg().rank();
  Tensor<T>    cholVpr = chem_env.cd_context.cholV2; // cannot be const; deallocate modifies
  const bool   is_rhf  = chem_env.sys_data.is_restricted;
  Scheduler    sch{ec};
  const double mp2_mem_req = sum_tensor_sizes(state.dtmp, cholVpr, state.v2ijab);

  if(ec.print())
    std::cout << "CPU memory required for MP2 calculation: " << std::fixed << std::setprecision(2)
              << mp2_mem_req << " GiB" << std::endl;
  check_memory_requirements(ec, mp2_mem_req);
  compute_dtmp(ec, state);

  const auto mp_t1 = std::chrono::high_resolution_clock::now();
  sch(state.v2ijab(state.h1, state.h2, state.p1, state.p2) =
        cholVpr(state.h1, state.p1, state.cind) * cholVpr(state.h2, state.p2, state.cind))
    .deallocate(cholVpr)
    .execute(ec.exhw());

  if(is_rhf) compute_closed_shell_mp2(ec, sch, state);
  else compute_open_shell_mp2(ec, sch, state);

  const auto   mp_t2 = std::chrono::high_resolution_clock::now();
  const double mp2_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((mp_t2 - mp_t1)).count();

  state.mp2_total_energy = state.hf_energy + state.mp2_energy;

  if(rank == 0) {
    if(is_rhf) std::cout << "Closed-Shell ";
    else std::cout << "Open-Shell ";
    std::cout << "MP2 correlation energy / hartree: " << std::fixed << std::setprecision(15)
              << state.mp2_energy << std::endl;
    std::cout << "MP2 total energy / hartree: " << std::fixed << std::setprecision(15)
              << state.mp2_total_energy << std::endl;
    std::cout << "Time to compute MP2 energy: " << std::fixed << std::setprecision(2) << mp2_time
              << " secs" << std::endl
              << std::endl;
  }

  sch.deallocate(state.dtmp, state.v2ijab).execute();
  ec.flush_and_sync();
  update_chemenv(chem_env, state);
}

// Free function interface preserved
void cd_mp2(ExecutionContext& ec, ChemEnv& chem_env) {
  std::cout << "Starting CD-MP2 calculation..." << std::endl;
  CDMP2 mp2; // default construct, then run
  mp2.run(ec, chem_env);
}

} // namespace exachem::mp2
