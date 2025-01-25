/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

// clang-format off
#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"
#include "exachem/cc/ccsd_t/ccsd_t_fused_driver.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"
// clang-format on

double ccsdt_s1_t1_GetTime  = 0;
double ccsdt_s1_v2_GetTime  = 0;
double ccsdt_d1_t2_GetTime  = 0;
double ccsdt_d1_v2_GetTime  = 0;
double ccsdt_d2_t2_GetTime  = 0;
double ccsdt_d2_v2_GetTime  = 0;
double genTime              = 0;
double ccsd_t_data_per_rank = 0; // in GB

void exachem::cc::ccsd_t::ccsd_t_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T   = double;
  auto rank = ec.pg().rank();

  SystemData& sys_data = chem_env.sys_data;
  const bool  is_rhf   = sys_data.is_restricted;

  const int ccsdt_tilesize = chem_env.ioptions.ccsd_options.ccsdt_tilesize;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  std::string t_errmsg = check_memory_req(ccsdt_tilesize, sys_data.nbf);
  if(!t_errmsg.empty()) tamm_terminate(t_errmsg);
#endif

  CCContext& cc_context      = chem_env.cc_context;
  cc_context.use_subgroup    = true;
  cc_context.keep.fvt12_full = true;
  cc_context.compute.set(true, false); // compute ft12 in full, v2 is not required.
  exachem::cc::ccsd::cd_ccsd_driver(ec, chem_env);

  std::string files_dir    = chem_env.get_files_dir();
  std::string files_prefix = chem_env.get_files_prefix();

  ExecutionHW ex_hw = ec.exhw();

  TiledIndexSpace& MO           = chem_env.is_context.MSO;
  TiledIndexSpace& CI           = chem_env.is_context.CI;
  TiledIndexSpace  N            = MO("all");
  Tensor<T>        d_f1         = chem_env.cd_context.d_f1;
  Tensor<T>        cholVpr      = chem_env.cd_context.cholV2;
  Tensor<T>        d_t1         = chem_env.cc_context.d_t1_full;
  Tensor<T>        d_t2         = chem_env.cc_context.d_t2_full;
  CCSDOptions&     ccsd_options = chem_env.ioptions.ccsd_options;
  std::vector<T>&  p_evl_sorted = chem_env.cd_context.p_evl_sorted;

  bool skip_ccsd = ccsd_options.skip_ccsd;

  double& hf_energy   = chem_env.scf_context.hf_energy;
  double  corr_energy = cc_context.ccsd_correlation_energy;

  auto [MO1, total_orbitals1] = cholesky_2e::setupMOIS(ec, chem_env, true);
  TiledIndexSpace N1          = MO1("all");
  TiledIndexSpace O1          = MO1("occ");
  TiledIndexSpace V1          = MO1("virt");

  // Tensor<T> d_v2{{N,N,N,N},{2,2}};
  // Tensor<T> t_d_f1{{N1,N1},{1,1}};
  // Tensor<T> t_d_v2{{N1,N1,N1,N1}, {2,2}};

  Tensor<T>                 t_d_t1{{V1, O1}, {1, 1}};
  Tensor<T>                 t_d_t2{{V1, V1, O1, O1}, {2, 2}};
  Tensor<T>                 t_d_cv2{{N1, N1, CI}, {1, 1}};
  cholesky_2e::V2Tensors<T> v2tensors({"ijab", "ijka", "iabc"});

  T            ccsd_t_mem{};
  const double gib   = (1024 * 1024 * 1024.0);
  const double Osize = MO("occ").max_num_indices();
  const double Vsize = MO("virt").max_num_indices();
  // const double Nsize = N.max_num_indices();
  // const double cind_size = CI.max_num_indices();

  ccsd_t_mem = sum_tensor_sizes(d_f1, t_d_t1, t_d_t2);
  if(!skip_ccsd) {
    // auto v2_setup_mem = sum_tensor_sizes(d_f1,t_d_v2,t_d_cv2);
    // auto cv2_retile = (Nsize*Nsize*cind_size*8)/gib + sum_tensor_sizes(d_f1,cholVpr,t_d_cv2);
    ccsd_t_mem += sum_tensor_sizes(d_t1, d_t2); // full t1,t2

    // retiling allocates full GA versions of the t1,t2 tensors.
    ccsd_t_mem += (Osize * Vsize + Vsize * Vsize * Osize * Osize) * 8 / gib;
  }

  // const auto ccsd_t_mem_old = ccsd_t_mem + sum_tensor_sizes(t_d_v2);
  ccsd_t_mem += v2tensors.tensor_sizes(MO1);

  Index noab       = MO1("occ").num_tiles();
  Index nvab       = MO1("virt").num_tiles();
  Index cache_size = ccsd_options.cache_size;

  {
    Index noa    = MO1("occ_alpha").num_tiles();
    Index nva    = MO1("virt_alpha").num_tiles();
    auto  nranks = ec.pg().size().value();

    auto                mo_tiles = MO1.input_tile_sizes();
    std::vector<size_t> k_range;
    for(auto x: mo_tiles) k_range.push_back(x);
    size_t max_pdim = 0;
    size_t max_hdim = 0;
    for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++)
      max_pdim = std::max(max_pdim, k_range[t_p4b]);
    for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) max_hdim = std::max(max_hdim, k_range[t_h1b]);

    size_t max_d1_kernels_pertask = 9 * noa;
    size_t max_d2_kernels_pertask = 9 * nva;
    size_t size_T_s1_t1           = 9 * (max_pdim) * (max_hdim);
    size_t size_T_s1_v2           = 9 * (max_pdim * max_pdim) * (max_hdim * max_hdim);
    size_t size_T_d1_t2 = max_d1_kernels_pertask * (max_pdim * max_pdim) * (max_hdim * max_hdim);
    size_t size_T_d1_v2 = max_d1_kernels_pertask * (max_pdim) * (max_hdim * max_hdim * max_hdim);
    size_t size_T_d2_t2 = max_d2_kernels_pertask * (max_pdim * max_pdim) * (max_hdim * max_hdim);
    size_t size_T_d2_v2 = max_d2_kernels_pertask * (max_pdim * max_pdim * max_pdim) * (max_hdim);

    double extra_buf_mem_per_rank =
      size_T_s1_t1 + size_T_s1_v2 + size_T_d1_t2 + size_T_d1_v2 + size_T_d2_t2 + size_T_d2_v2;
    extra_buf_mem_per_rank     = extra_buf_mem_per_rank * 8 / gib;
    double total_extra_buf_mem = extra_buf_mem_per_rank * nranks;

    size_t cache_buf_size =
      ccsdt_tilesize * ccsdt_tilesize * ccsdt_tilesize * ccsdt_tilesize * 8; // bytes
    double cache_mem_per_rank =
      (ccsdt_tilesize * ccsdt_tilesize * 8 + cache_buf_size) * cache_size; // s1 t1+v2
    cache_mem_per_rank += (noab + nvab) * 2 * cache_size * cache_buf_size; // d1,d2 t2+v2
    cache_mem_per_rank     = cache_mem_per_rank / gib;
    double total_cache_mem = cache_mem_per_rank * nranks; // GiB

    double total_ccsd_t_mem = ccsd_t_mem + total_extra_buf_mem + total_cache_mem;
    if(rank == 0) {
      std::cout << std::string(70, '-') << std::fixed << std::setprecision(2) << std::endl;
      std::cout << "Total CPU memory required for (T) calculation = " << total_ccsd_t_mem << " GiB"
                << std::endl;
      std::cout << " -- memory required for the input tensors: " << ccsd_t_mem << " GiB"
                << std::endl;
      std::cout << " -- memory required for intermediate buffers: " << total_extra_buf_mem << " GiB"
                << std::endl;
      std::string cache_msg = " -- memory required for caching t1,t2,v2 blocks";
      if(total_cache_mem > (ccsd_t_mem + total_extra_buf_mem) / 2.0)
        cache_msg += " (set cache_size option in the input file to a lower value to reduce this "
                     "memory requirement further)";
      std::cout << cache_msg << ": " << total_cache_mem << " GiB" << std::endl;
      // std::cout << "***** old memory requirement was "
      //           << ccsd_t_mem_old + total_extra_buf_mem + total_cache_mem
      //           << " GiB (old v2 = " << sum_tensor_sizes(t_d_v2)
      //           << " GiB, new v2 = " << v2tensors.tensor_sizes(MO1) << " GiB)" << std::endl;
      std::cout << std::string(70, '-') << std::endl;
    }
    check_memory_requirements(ec, total_ccsd_t_mem);
  }

  double energy1 = 0, energy2 = 0;

  if(rank == 0) {
    auto mo_tiles = MO.input_tile_sizes();
    cout << endl << "CCSD MO Tiles = " << mo_tiles << endl;
  }

  Tensor<T>::allocate(&ec, t_d_t1, t_d_t2);

  bool ccsd_t_restart = (ccsd_options.writet || ccsd_options.readt) &&
                        fs::exists(cc_context.full_t1file) && fs::exists(cc_context.full_t2file) &&
                        fs::exists(chem_env.cd_context.f1file) &&
                        v2tensors.exist_on_disk(files_prefix);

  if(!skip_ccsd) {
    if(!ccsd_t_restart) {
      Tensor<T>::allocate(&ec, t_d_cv2);
      retile_tamm_tensor(cholVpr, t_d_cv2, "CholV2");
      free_tensors(cholVpr);

      v2tensors = cholesky_2e::setupV2Tensors<T>(ec, t_d_cv2, ex_hw, v2tensors.get_blocks());
      free_tensors(t_d_cv2);

      if(rank == 0) { cout << endl << "Retile T1,T2 tensors ... " << endl; }

      Scheduler{ec}(t_d_t1() = 0)(t_d_t2() = 0).execute();

      // d_t1, d_t2 are the full tensors
      retile_tamm_tensor(d_t1, t_d_t1);
      retile_tamm_tensor(d_t2, t_d_t2);

      // TODO: profile and re-enable as needed
      //  if(ccsd_options.writet) {
      //    ec.pg().barrier();
      //    v2tensors.write_to_disk(files_prefix);
      //    write_to_disk(t_d_t1, cc_context.full_t1file);
      //    write_to_disk(t_d_t2, cc_context.full_t2file);
      //  }
    }
    else {
      free_tensors(cholVpr);
      v2tensors.allocate(ec, MO1);
      // read_from_disk(t_d_f1,f1file);
      read_from_disk(t_d_t1, cc_context.full_t1file);
      read_from_disk(t_d_t2, cc_context.full_t2file);
      v2tensors.read_from_disk(files_prefix);
    }
    free_tensors(d_t1, d_t2);
  }
  else { // skip ccsd
    // t1,t2,cholVpr are never allocated
    v2tensors.allocate(ec, MO1);
  }

  p_evl_sorted = tamm::diagonal(d_f1);

  // cc_t1 = std::chrono::high_resolution_clock::now();

  bool is_restricted = is_rhf;

  // Given the singleton pool created by the TAMM is not used by the (T) kernel calculation.
  // We artifically destroy the pool
  tamm::reset_rmm_pool();
  tamm::reinitialize_rmm_pool();

  std::string dev_str = "[CPU]";
#if defined(USE_CUDA)
  dev_str = "[Nvidia GPU]";
#elif defined(USE_HIP)
  dev_str = "[AMD GPU]";
#elif defined(USE_DPCPP)
  dev_str = "[Intel GPU]";
#endif

  if(rank == 0) {
    if(is_restricted)
      cout << endl << dev_str << " Running Closed Shell CCSD(T) calculation" << endl;
    else cout << endl << dev_str << " Running Open Shell CCSD(T) calculation" << endl;
  }

  bool                            seq_h3b = true;
  LRUCache<Index, std::vector<T>> cache_s1t{cache_size};
  LRUCache<Index, std::vector<T>> cache_s1v{cache_size};
  LRUCache<Index, std::vector<T>> cache_d1t{cache_size * noab};
  LRUCache<Index, std::vector<T>> cache_d1v{cache_size * noab};
  LRUCache<Index, std::vector<T>> cache_d2t{cache_size * nvab};
  LRUCache<Index, std::vector<T>> cache_d2v{cache_size * nvab};

  if(rank == 0 && seq_h3b) cout << "running seq h3b loop variant..." << endl;

  std::vector<int> k_spin;
  for(tamm::Index x = 0; x < noab / 2; x++) k_spin.push_back(1);
  for(tamm::Index x = noab / 2; x < noab; x++) k_spin.push_back(2);
  for(tamm::Index x = 0; x < nvab / 2; x++) k_spin.push_back(1);
  for(tamm::Index x = nvab / 2; x < nvab; x++) k_spin.push_back(2);

  double ccsd_t_time = 0, total_t_time = 0;
  // cc_t1 = std::chrono::high_resolution_clock::now();
  std::tie(energy1, energy2, ccsd_t_time, total_t_time) = ccsd_t_fused_driver_new<T>(
    chem_env, ec, k_spin, MO1, t_d_t1, t_d_t2, v2tensors, p_evl_sorted, hf_energy + corr_energy,
    is_restricted, cache_s1t, cache_s1v, cache_d1t, cache_d1v, cache_d2t, cache_d2v, seq_h3b);

  // cc_t2 = std::chrono::high_resolution_clock::now();
  // auto ccsd_t_time =
  //     std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  energy1 = ec.pg().reduce(&energy1, ReduceOp::sum, 0);
  energy2 = ec.pg().reduce(&energy2, ReduceOp::sum, 0);

  cc_context.ccsd_st_correction_energy  = energy1;
  cc_context.ccsd_st_correlation_energy = corr_energy + energy1;
  cc_context.ccsd_st_total_energy       = hf_energy + corr_energy + energy1;

  cc_context.ccsd_pt_correction_energy  = energy2;
  cc_context.ccsd_pt_correlation_energy = corr_energy + energy2;
  cc_context.ccsd_pt_total_energy       = hf_energy + corr_energy + energy2;

  if(rank == 0 && !skip_ccsd) {
    std::cout.precision(15);
    cout << "CCSD[T] correction energy / hartree  = " << energy1 << endl;
    cout << "CCSD[T] correlation energy / hartree = " << cc_context.ccsd_st_correlation_energy
         << endl;
    cout << "CCSD[T] total energy / hartree       = " << cc_context.ccsd_st_total_energy << endl;

    cout << "CCSD(T) correction energy / hartree  = " << energy2 << endl;
    cout << "CCSD(T) correlation energy / hartree = " << cc_context.ccsd_pt_correlation_energy
         << endl;
    cout << "CCSD(T) total energy / hartree       = " << cc_context.ccsd_pt_total_energy << endl;

    sys_data.results["output"]["CCSD(T)"]["[T]Energies"]["correction"] = energy1;
    sys_data.results["output"]["CCSD(T)"]["[T]Energies"]["correlation"] =
      cc_context.ccsd_st_correlation_energy;
    sys_data.results["output"]["CCSD(T)"]["[T]Energies"]["total"] = cc_context.ccsd_st_total_energy;
    sys_data.results["output"]["CCSD(T)"]["(T)Energies"]["correction"] = energy2;
    sys_data.results["output"]["CCSD(T)"]["(T)Energies"]["correlation"] =
      cc_context.ccsd_pt_correlation_energy;
    sys_data.results["output"]["CCSD(T)"]["(T)Energies"]["total"] = cc_context.ccsd_pt_total_energy;
  }

  long double total_num_ops = 0;
  //
  if(rank == 0) {
    // std::cout << "--------------------------------------------------------------------" <<
    // std::endl;
    ccsd_t_fused_driver_calculator_ops<T>(chem_env, ec, k_spin, MO1, p_evl_sorted,
                                          hf_energy + corr_energy, is_restricted, total_num_ops,
                                          seq_h3b);
    // std::cout << "--------------------------------------------------------------------" <<
    // std::endl;
  }

  ec.pg().barrier();

  auto nranks = ec.pg().size().value();

  auto print_profile_stats = [&](const std::string& timer_type, const double g_tval,
                                 const double tval_min, const double tval_max) {
    const double tval = g_tval / nranks;
    std::cout.precision(3);
    std::cout << "   -> " << timer_type << ": " << tval << "s (" << tval * 100.0 / total_t_time
              << "%), (min,max) = (" << tval_min << "," << tval_max << ")" << std::endl;
  };

  auto comm_stats = [&](const std::string& timer_type, const double ctime) {
    double g_getTime     = ec.pg().reduce(&ctime, ReduceOp::sum, 0);
    double g_min_getTime = ec.pg().reduce(&ctime, ReduceOp::min, 0);
    double g_max_getTime = ec.pg().reduce(&ctime, ReduceOp::max, 0);

    if(rank == 0) print_profile_stats(timer_type, g_getTime, g_min_getTime, g_max_getTime);
    return g_getTime / nranks;
  };

  if(rank == 0) {
    std::cout << std::endl << "------CCSD(T) Performance------" << std::endl;
    std::cout << "Total CCSD(T) Time: " << total_t_time << std::endl;
  }
  ccsd_t_time = comm_stats("CCSD(T) Avg. Work Time", ccsd_t_time);
  if(rank == 0) {
    const double n_gflops = total_num_ops / (total_t_time * 1e9);
    const double load_imb = (1.0 - ccsd_t_time / total_t_time);
    std::cout << std::scientific << "   -> Total Number of Operations: " << total_num_ops
              << std::endl;
    std::cout << std::fixed << "   -> GFLOPS: " << n_gflops << std::endl;
    std::cout << std::fixed << "   -> Load imbalance: " << load_imb << std::endl;

    sys_data.results["output"]["CCSD(T)"]["performance"]["total_time"]     = total_t_time;
    sys_data.results["output"]["CCSD(T)"]["performance"]["gflops"]         = n_gflops;
    sys_data.results["output"]["CCSD(T)"]["performance"]["total_num_ops"]  = total_num_ops;
    sys_data.results["output"]["CCSD(T)"]["performance"]["load_imbalance"] = load_imb;
    chem_env.write_json_data();
  }

  comm_stats("S1-T1 GetTime", ccsdt_s1_t1_GetTime);
  comm_stats("S1-V2 GetTime", ccsdt_s1_v2_GetTime);
  comm_stats("D1-T2 GetTime", ccsdt_d1_t2_GetTime);
  comm_stats("D1-V2 GetTime", ccsdt_d1_v2_GetTime);
  comm_stats("D2-T2 GetTime", ccsdt_d2_t2_GetTime);
  comm_stats("D2-V2 GetTime", ccsdt_d2_v2_GetTime);

  ccsd_t_data_per_rank          = (ccsd_t_data_per_rank * 8.0) / (1024 * 1024.0 * 1024); // GB
  double g_ccsd_t_data_per_rank = ec.pg().reduce(&ccsd_t_data_per_rank, ReduceOp::sum, 0);
  if(rank == 0)
    std::cout << "   -> Data Transfer (GB): " << g_ccsd_t_data_per_rank / nranks << std::endl;

  ec.pg().barrier();

  free_tensors(t_d_t1, t_d_t2, d_f1);
  v2tensors.deallocate();

  ec.flush_and_sync();
  if(!skip_ccsd) cc_context.destroy_subgroup();
}
