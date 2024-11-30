/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

//  #include "exachem/cc/ccsd/ccsd_util.hpp"

#include "exachem/mp2/cd_mp2.hpp"
#include <filesystem>

namespace exachem::mp2 {

void cd_mp2(ExecutionContext& ec, ChemEnv& chem_env) {
  using T   = double;
  auto rank = ec.pg().rank();

  cholesky_2e::cholesky_2e_driver(ec, chem_env);

  Scheduler sch{ec};

  SystemData       sys_data = chem_env.sys_data;
  TiledIndexSpace& MO       = chem_env.is_context.MSO;
  TiledIndexSpace& CI       = chem_env.is_context.CI;

  Tensor<T> d_f1    = chem_env.cd_context.d_f1;
  Tensor<T> cholVpr = chem_env.cd_context.cholV2;

  const bool is_rhf = sys_data.is_restricted;

  // get Eigen values
  std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);
  // split into occupied and virtual parts
  std::vector<T> p_evl_sorted_occ(sys_data.nocc);
  std::vector<T> p_evl_sorted_virt(sys_data.nvir);
  std::copy(p_evl_sorted.begin(), p_evl_sorted.begin() + sys_data.nocc, p_evl_sorted_occ.begin());
  std::copy(p_evl_sorted.begin() + sys_data.nocc, p_evl_sorted.end(), p_evl_sorted_virt.begin());

  // ############ Setup index spaces ###########
  TiledIndexSpace N = MO("all");
  TiledIndexSpace O = MO("occ");
  TiledIndexSpace V = MO("virt");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  // const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  // const int vbtiles = MO("virt_beta").num_tiles();

  TiledIndexSpace o_alpha = {MO("occ"), range(oatiles)};
  TiledIndexSpace v_alpha = {MO("virt"), range(vatiles)};
  TiledIndexSpace o_beta  = {MO("occ"), range(oatiles, otiles)};
  TiledIndexSpace v_beta  = {MO("virt"), range(vatiles, vtiles)};

  auto [p1, p2] = MO.labels<2>("virt");
  auto [h1, h2] = MO.labels<2>("occ");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
  auto [h1_ob, h2_ob] = o_beta.labels<2>("all");
  auto [cind]         = CI.labels<1>("all");

  // ############### BEGIN MP2 ###############
  Tensor<T> dtmp{{V, V, O, O}, {2, 2}};
  Tensor<T> v2ijab{{O, O, V, V}, {2, 2}};
  Tensor<T> v2tmp_alpha{{o_alpha, o_alpha, v_alpha, v_alpha}, {2, 2}};

  double mp2_mem_req = sum_tensor_sizes(dtmp, cholVpr, v2ijab);

  if(ec.print())
    std::cout << "CPU memory required for MP2 calculation: " << std::fixed << std::setprecision(2)
              << mp2_mem_req << " GiB" << std::endl;
  check_memory_requirements(ec, mp2_mem_req);

  double denominator = 0.0;
  auto   dtmp_lambda = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, dtmp());
    const TAMM_SIZE   size    = dtmp.block_size(blockid);
    std::vector<T>    buf(size);
    auto              block_dims   = dtmp.block_dims(blockid);
    auto              block_offset = dtmp.block_offsets(blockid);
    size_t            c            = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
        for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
          for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++, c++) {
            denominator = p_evl_sorted_virt[i] + p_evl_sorted_virt[j] - p_evl_sorted_occ[k] -
                          p_evl_sorted_occ[l];
            buf[c] = 1.0 / denominator;
          }
        }
      }
    }
    dtmp.put(blockid, buf);
  };

  sch.deallocate(d_f1).allocate(dtmp, v2ijab).execute();

  auto mp_t1 = std::chrono::high_resolution_clock::now();

  block_for(ec, dtmp(), dtmp_lambda);

  sch(v2ijab(h1, h2, p1, p2) = cholVpr(h1, p1, cind) * cholVpr(h2, p2, cind))
    .deallocate(cholVpr)
    .execute(ec.exhw());

  T mp2_energy{0};
  T mp2_alpha_energy{0};
  T mp2_beta_energy{0};

  if(is_rhf) {
    // For ClosedShell O=o_alpha, V=v_alpha
    // clang-format off
    Tensor<T> mp2_ea{};
    sch.allocate(mp2_ea,v2tmp_alpha)
      (v2tmp_alpha(h1_oa,h2_oa,p1_va,p2_va)  =  2.0 * v2ijab(h1_oa,h2_oa,p1_va,p2_va) * v2ijab(h1_oa,h2_oa,p1_va,p2_va))
      (v2tmp_alpha(h1_oa,h2_oa,p1_va,p2_va) += -1.0 * v2ijab(h1_oa,h2_oa,p1_va,p2_va) * v2ijab(h2_oa,h1_oa,p1_va,p2_va))
      (mp2_ea() = dtmp(p1_va,p2_va,h1_oa,h2_oa) * v2tmp_alpha(h1_oa,h2_oa,p1_va,p2_va))
      .deallocate(v2tmp_alpha)
      .execute(ec.exhw());
    // clang-format on
    mp2_energy       = -1.0 * get_scalar(mp2_ea);
    mp2_alpha_energy = mp2_energy;
    sch.deallocate(mp2_ea).execute();
  }

  else {
    // Open Shell expressions reference: https://pubs.acs.org/doi/10.1021/acs.jctc.6b00015
    // clang-format off
    Tensor<T> mp2_ea{};
    Tensor<T> mp2_eb{};
    Tensor<T> mp2_ec{};    
    Tensor<T> v2tmp_beta{{o_beta, o_beta, v_beta, v_beta}, {2, 2}};
    Tensor<T> v2tmp_ab{{o_alpha, o_beta, v_alpha, v_beta}, {2, 2}};

    sch.allocate(mp2_ea,mp2_eb,mp2_ec,v2tmp_alpha)
       (v2tmp_alpha(h1_oa,h2_oa,p1_va,p2_va)  =  1.0 * v2ijab(h1_oa,h2_oa,p1_va,p2_va) * v2ijab(h1_oa,h2_oa,p1_va,p2_va))
       (v2tmp_alpha(h1_oa,h2_oa,p1_va,p2_va) += -1.0 * v2ijab(h1_oa,h2_oa,p1_va,p2_va) * v2ijab(h1_oa,h2_oa,p2_va,p1_va))
       (mp2_ea() = dtmp(p1_va,p2_va,h1_oa,h2_oa) * v2tmp_alpha(h1_oa,h2_oa,p1_va,p2_va)).deallocate(v2tmp_alpha);

    sch.allocate(v2tmp_beta)
      (v2tmp_beta(h1_ob,h2_ob,p1_vb,p2_vb)  =  1.0 * v2ijab(h1_ob,h2_ob,p1_vb,p2_vb) * v2ijab(h1_ob,h2_ob,p1_vb,p2_vb))
      (v2tmp_beta(h1_ob,h2_ob,p1_vb,p2_vb) += -1.0 * v2ijab(h1_ob,h2_ob,p1_vb,p2_vb) * v2ijab(h1_ob,h2_ob,p2_vb,p1_vb))
      (mp2_eb() = dtmp(p1_vb,p2_vb,h1_ob,h2_ob) * v2tmp_beta(h1_ob,h2_ob,p1_vb,p2_vb)).deallocate(v2tmp_beta);

    sch.allocate(v2tmp_ab)
      (v2tmp_ab(h1_oa, h2_ob, p1_va, p2_vb) =  v2ijab(h1_oa, h2_ob, p1_va, p2_vb) * v2ijab(h1_oa, h2_ob, p1_va, p2_vb))
      (mp2_ec() = dtmp(p1_va,p2_vb,h1_oa,h2_ob) * v2tmp_ab(h1_oa, h2_ob, p1_va, p2_vb)).deallocate(v2tmp_ab);
    // clang-format on

    sch.execute(ec.exhw());

    mp2_alpha_energy = -0.5 * get_scalar(mp2_ea);
    mp2_beta_energy  = -0.5 * get_scalar(mp2_eb);
    T mp2_ab_energy  = -1.0 * get_scalar(mp2_ec);
    mp2_energy       = mp2_alpha_energy + mp2_beta_energy + mp2_ab_energy;
    sch.deallocate(mp2_ea, mp2_eb, mp2_ec).execute();
  }

  auto   mp_t2 = std::chrono::high_resolution_clock::now();
  double mp2_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((mp_t2 - mp_t1)).count();

  chem_env.mp2_context.mp2_correlation_energy = mp2_energy;
  chem_env.mp2_context.mp2_total_energy       = chem_env.scf_context.hf_energy + mp2_energy;

  if(rank == 0) {
    if(is_rhf) std::cout << "Closed-Shell ";
    else std::cout << "Open-Shell ";
    std::cout << "MP2 correlation energy / hartree: " << std::fixed << std::setprecision(15)
              << mp2_energy << std::endl;
    std::cout << "MP2 total energy / hartree: " << std::fixed << std::setprecision(15)
              << chem_env.mp2_context.mp2_total_energy << std::endl;

    std::cout << "Time to compute MP2 energy: " << std::fixed << std::setprecision(2) << mp2_time
              << " secs" << std::endl
              << std::endl;
  }

  sch.deallocate(dtmp, v2ijab).execute();

  ec.flush_and_sync();
  // delete ec;
}
} // namespace exachem::mp2
