/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

// clang-format off
#include "cc/ccsd/cd_ccsd_os_ann.hpp"
#include "cc/ccsd_t/ccsd_t_fused_driver.hpp"
#include "cholesky/cholesky_2e_driver.hpp"
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
  using T = double;

  auto rank = ec.pg().rank();

  scf::scf_driver(ec, chem_env);

  double              hf_energy      = chem_env.hf_energy;
  libint2::BasisSet   shells         = chem_env.shells;
  Tensor<T>           C_AO           = chem_env.C_AO;
  Tensor<T>           C_beta_AO      = chem_env.C_beta_AO;
  Tensor<T>           F_AO           = chem_env.F_AO;
  Tensor<T>           F_beta_AO      = chem_env.F_beta_AO;
  TiledIndexSpace     AO_opt         = chem_env.AO_opt;
  TiledIndexSpace     AO_tis         = chem_env.AO_tis;
  std::vector<size_t> shell_tile_map = chem_env.shell_tile_map;
  bool                scf_conv       = chem_env.no_scf;

  SystemData& sys_data = chem_env.sys_data;
  // CCSDOptions& ccsd_options   = chem_env.ioptions.ccsd_options;
  CCSDOptions& ccsd_options   = chem_env.ioptions.ccsd_options;
  const int    ccsdt_tilesize = ccsd_options.ccsdt_tilesize;

  sys_data.freeze_atomic    = chem_env.ioptions.ccsd_options.freeze_atomic;
  sys_data.n_frozen_core    = chem_env.get_nfcore();
  sys_data.n_frozen_virtual = chem_env.ioptions.ccsd_options.freeze_virtual;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  std::string t_errmsg = check_memory_req(ccsdt_tilesize, sys_data.nbf);
  if(!t_errmsg.empty()) tamm_terminate(t_errmsg);
#endif

  int nsranks = sys_data.nbf / 15;
  if(nsranks < 1) nsranks = 1;
  int ga_cnn = ec.nnodes();
  if(nsranks > ga_cnn) nsranks = ga_cnn;
  nsranks = nsranks * GA_Cluster_nprocs(0);
  int subranks[nsranks];
  for(int i = 0; i < nsranks; i++) subranks[i] = i;

#if defined(USE_UPCXX)
  upcxx::team subcomm = upcxx::world().split((rank < nsranks) ? 0 : upcxx::team::color_none, 0);
#else
  auto      world_comm = ec.pg().comm();
  MPI_Group world_group;
  MPI_Comm_group(world_comm, &world_group);
  MPI_Group subgroup;
  MPI_Group_incl(world_group, nsranks, subranks, &subgroup);
  MPI_Comm subcomm;
  MPI_Comm_create(world_comm, subgroup, &subcomm);
  MPI_Group_free(&world_group);
  MPI_Group_free(&subgroup);
#endif

  ProcGroup         sub_pg;
  ExecutionContext* sub_ec = nullptr;

  if(rank < nsranks) {
    sub_pg = ProcGroup::create_coll(subcomm);
    sub_ec = new ExecutionContext(sub_pg, DistributionKind::nw, MemoryManagerKind::ga);
  }

  Scheduler sub_sch{*sub_ec};

  ccsd_options.computeTData = true;

  auto debug     = ccsd_options.debug;
  bool skip_ccsd = ccsd_options.skip_ccsd;
  if(rank == 0) ccsd_options.print();

  if(rank == 0)
    cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;

  auto [MO, total_orbitals] = cholesky_2e::setupMOIS(ec, chem_env);

  std::string out_fp       = chem_env.workspace_dir;
  std::string files_dir    = out_fp + chem_env.ioptions.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + sys_data.output_file_prefix;
  std::string f1file       = files_prefix + ".f1_mo";
  std::string t1file       = files_prefix + ".t1amp";
  std::string t2file       = files_prefix + ".t2amp";
  std::string v2file       = files_prefix + ".cholv2";
  std::string cholfile     = files_prefix + ".cholcount";
  std::string ccsdstatus   = files_prefix + ".ccsdstatus";

  const bool is_rhf       = sys_data.is_restricted;
  bool       computeTData = ccsd_options.computeTData;

  bool ccsd_restart = ccsd_options.readt || ((fs::exists(t1file) && fs::exists(t2file) &&
                                              fs::exists(f1file) && fs::exists(v2file)));

  ExecutionHW ex_hw = ec.exhw();

  TiledIndexSpace N = MO("all");

  Tensor<T>       cholVpr, d_t1, d_t2, d_f1, lcao;
  TAMM_SIZE       chol_count;
  tamm::Tile      max_cvecs;
  TiledIndexSpace CI;

  std::vector<T> p_evl_sorted;
  double         residual = 0, corr_energy = 0;
  Tensor<T>      dt1_full, dt2_full;

  if(!skip_ccsd) {
    // deallocates F_AO, C_AO
    std::tie(cholVpr, d_f1, lcao, chol_count, max_cvecs, CI) =
      exachem::cholesky_2e::cholesky_2e_driver<T>(chem_env, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO,
                                                  F_beta_AO, shells, shell_tile_map, ccsd_restart,
                                                  cholfile);
    free_tensors(lcao);

    if(ccsd_options.writev) ccsd_options.writet = true;

    Tensor<T>              d_r1, d_r2;
    std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;

    if(is_rhf)
      std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) = setupTensors_cs(
        ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && fs::exists(ccsdstatus) && scf_conv);
    else
      std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) = setupTensors(
        ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && fs::exists(ccsdstatus) && scf_conv);

    if(ccsd_restart) {
      read_from_disk(d_f1, f1file);
      if(fs::exists(t1file) && fs::exists(t2file)) {
        read_from_disk(d_t1, t1file);
        read_from_disk(d_t2, t2file);
      }
      read_from_disk(cholVpr, v2file);
      ec.pg().barrier();
      p_evl_sorted = tamm::diagonal(d_f1);
    }

    else if(ccsd_options.writet) {
      // fs::remove_all(files_dir);
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

    if(rank == 0 && debug) {
      print_vector(p_evl_sorted, files_prefix + ".eigen_values.txt");
      cout << "Eigen values written to file: " << files_prefix + ".eigen_values.txt" << endl
           << endl;
    }

    ec.pg().barrier();

    auto cc_t1 = std::chrono::high_resolution_clock::now();

    ccsd_restart = ccsd_restart && fs::exists(ccsdstatus) && scf_conv;

    t1file = files_prefix + ".fullT1amp";
    t2file = files_prefix + ".fullT2amp";

    if(ccsd_options.writev)
      computeTData = computeTData && !fs::exists(t1file) && !fs::exists(t2file);

    if(computeTData && is_rhf) setup_full_t1t2(ec, MO, dt1_full, dt2_full);

    if(is_rhf) {
      if(ccsd_restart) {
        if(rank < nsranks) {
          const int ppn = GA_Cluster_nprocs(0);
          if(rank == 0)
            std::cout << "Executing with " << nsranks << " ranks (" << nsranks / ppn << " nodes)"
                      << std::endl;
          std::tie(residual, corr_energy) = exachem::cc::ccsd::cd_ccsd_cs_driver<T>(
            chem_env, *sub_ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
            p_evl_sorted, cholVpr, dt1_full, dt2_full, ccsd_restart, files_prefix, computeTData);
        }
        ec.pg().barrier();
      }
      else {
        std::tie(residual, corr_energy) = exachem::cc::ccsd::cd_ccsd_cs_driver<T>(
          chem_env, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
          p_evl_sorted, cholVpr, dt1_full, dt2_full, ccsd_restart, files_prefix, computeTData);
      }
    }
    else {
      if(ccsd_restart) {
        if(rank < nsranks) {
          const int ppn = GA_Cluster_nprocs(0);
          if(rank == 0)
            std::cout << "Executing with " << nsranks << " ranks (" << nsranks / ppn << " nodes)"
                      << std::endl;
          std::tie(residual, corr_energy) = cd_ccsd_os_driver<T>(
            chem_env, *sub_ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
            p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);
        }
        ec.pg().barrier();
      }
      else {
        std::tie(residual, corr_energy) = cd_ccsd_os_driver<T>(
          chem_env, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
          p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);
      }
    }

    if(computeTData && is_rhf) {
      if(ccsd_options.writev) {
        write_to_disk(dt1_full, t1file);
        write_to_disk(dt2_full, t2file);
        free_tensors(dt1_full, dt2_full);
      }
    }

    ccsd_stats(ec, hf_energy, residual, corr_energy, ccsd_options.threshold);

    if(ccsd_options.writet && !fs::exists(ccsdstatus)) {
      // write_to_disk(d_t1,t1file);
      // write_to_disk(d_t2,t2file);
      if(rank == 0) {
        std::ofstream out(ccsdstatus, std::ios::out);
        if(!out) cerr << "Error opening file " << ccsdstatus << endl;
        out << 1 << std::endl;
        out.close();
      }
    }

    if(rank < nsranks) {
      (*sub_ec).flush_and_sync();
      sub_pg.destroy_coll();
      delete sub_ec;
#ifdef USE_UPCXX
      subcomm.destroy();
#else
      MPI_Comm_free(&subcomm);
#endif
    }

    auto   cc_t2 = std::chrono::high_resolution_clock::now();
    double ccsd_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) {
      if(is_rhf)
        std::cout << std::endl
                  << "Time taken for Closed Shell Cholesky CCSD: " << std::fixed
                  << std::setprecision(2) << ccsd_time << " secs" << std::endl;
      else
        std::cout << std::endl
                  << "Time taken for Open Shell Cholesky CCSD: " << std::fixed
                  << std::setprecision(2) << ccsd_time << " secs" << std::endl;
    }

    cc_print(chem_env, d_t1, d_t2, files_prefix);

    if(!ccsd_restart) {
      free_tensors(d_r1, d_r2);
      free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
    }

    if(is_rhf) free_tensors(d_t1, d_t2);
    ec.flush_and_sync();
  }
  else { // skip ccsd
    cholesky_2e::update_sysdata(ec, chem_env, MO);
    N    = MO("all");
    d_f1 = {{N, N}, {1, 1}};
    Tensor<T>::allocate(&ec, d_f1);
    if(rank == 0) sys_data.print();

    if(rank < nsranks) {
      (*sub_ec).flush_and_sync();
      sub_pg.destroy_coll();
      delete sub_ec;
#ifdef USE_UPCXX
      subcomm.destroy();
#else
      MPI_Comm_free(&subcomm);
#endif
    }
  }

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
    if(is_rhf) ccsd_t_mem += sum_tensor_sizes(dt1_full, dt2_full);
    else ccsd_t_mem += sum_tensor_sizes(d_t1, d_t2);

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

  if(computeTData && !skip_ccsd) {
    Tensor<T>::allocate(&ec, t_d_cv2);
    retile_tamm_tensor(cholVpr, t_d_cv2, "CholV2");
    free_tensors(cholVpr);

    v2tensors = cholesky_2e::setupV2Tensors<T>(ec, t_d_cv2, ex_hw, v2tensors.get_blocks());
    if(ccsd_options.writev) {
      v2tensors.write_to_disk(files_prefix);
      v2tensors.deallocate();
    }
    free_tensors(t_d_cv2);
  }

  double energy1 = 0, energy2 = 0;

  if(rank == 0) {
    auto mo_tiles = MO.input_tile_sizes();
    cout << endl << "CCSD MO Tiles = " << mo_tiles << endl;
  }

  Tensor<T>::allocate(&ec, t_d_t1, t_d_t2);
  if(skip_ccsd || !computeTData) v2tensors.allocate(ec, MO1);

  bool ccsd_t_restart = fs::exists(t1file) && fs::exists(t2file) && fs::exists(f1file) &&
                        v2tensors.exist_on_disk(files_prefix);

  if(!ccsd_t_restart && !skip_ccsd) {
    if(!is_rhf) {
      dt1_full = d_t1;
      dt2_full = d_t2;
    }
    if(rank == 0) { cout << endl << "Retile T1,T2 tensors ... " << endl; }

    Scheduler{ec}(t_d_t1() = 0)(t_d_t2() = 0).execute();

    TiledIndexSpace O = MO("occ");
    TiledIndexSpace V = MO("virt");

    if(ccsd_options.writev) {
      Tensor<T> wd_t1{{V, O}, {1, 1}};
      Tensor<T> wd_t2{{V, V, O, O}, {2, 2}};

      read_from_disk(t_d_t1, t1file, false, wd_t1);
      read_from_disk(t_d_t2, t2file, false, wd_t2);

      ec.pg().barrier();
      write_to_disk(t_d_t1, t1file);
      write_to_disk(t_d_t2, t2file);
    }

    else {
      retile_tamm_tensor(dt1_full, t_d_t1);
      retile_tamm_tensor(dt2_full, t_d_t2);
      if(is_rhf) free_tensors(dt1_full, dt2_full);
    }
  }
  else if(ccsd_options.writev && !skip_ccsd) {
    // read_from_disk(t_d_f1,f1file);
    read_from_disk(t_d_t1, t1file);
    read_from_disk(t_d_t2, t2file);
    v2tensors.read_from_disk(files_prefix);
  }

  if(!is_rhf && !skip_ccsd) free_tensors(d_t1, d_t2);

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

  if(rank == 0 && !skip_ccsd) {
    std::cout.precision(15);
    cout << "CCSD[T] correction energy / hartree  = " << energy1 << endl;
    cout << "CCSD[T] correlation energy / hartree = " << corr_energy + energy1 << endl;
    cout << "CCSD[T] total energy / hartree       = " << hf_energy + corr_energy + energy1 << endl;

    cout << "CCSD(T) correction energy / hartree  = " << energy2 << endl;
    cout << "CCSD(T) correlation energy / hartree = " << corr_energy + energy2 << endl;
    cout << "CCSD(T) total energy / hartree       = " << hf_energy + corr_energy + energy2 << endl;

    sys_data.results["output"]["CCSD(T)"]["[T]Energies"]["correction"]  = energy1;
    sys_data.results["output"]["CCSD(T)"]["[T]Energies"]["correlation"] = corr_energy + energy1;
    sys_data.results["output"]["CCSD(T)"]["[T]Energies"]["total"] =
      hf_energy + corr_energy + energy1;
    sys_data.results["output"]["CCSD(T)"]["(T)Energies"]["correction"]  = energy2;
    sys_data.results["output"]["CCSD(T)"]["(T)Energies"]["correlation"] = corr_energy + energy2;
    sys_data.results["output"]["CCSD(T)"]["(T)Energies"]["total"] =
      hf_energy + corr_energy + energy2;
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
    chem_env.write_json_data("CCSD_T");
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
  // delete ec;
}
