/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

// clang-format off
#include "cc/cd_ccsd_os_ann.hpp"
#include "cc/ccsd_t/ccsd_t_fused_driver.hpp"
// clang-format on

void        ccsd_t_driver();
std::string filename;
double      ccsdt_s1_t1_GetTime  = 0;
double      ccsdt_s1_v2_GetTime  = 0;
double      ccsdt_d1_t2_GetTime  = 0;
double      ccsdt_d1_v2_GetTime  = 0;
double      ccsdt_d2_t2_GetTime  = 0;
double      ccsdt_d2_v2_GetTime  = 0;
double      genTime              = 0;
double      ccsd_t_data_per_rank = 0; // in GB

int main(int argc, char* argv[]) {
  if(argc < 2) {
    std::cout << "Please provide an input file!" << std::endl;
    return 1;
  }

  filename = std::string(argv[1]);
  std::ifstream testinput(filename);
  if(!testinput) {
    std::cout << "Input file provided [" << filename << "] does not exist!" << std::endl;
    return 1;
  }

  tamm::initialize(argc, argv);

  ccsd_t_driver();

  tamm::finalize();

  return 0;
}

void ccsd_t_driver() {
  // std::cout << "Input file provided = " << filename << std::endl;

  using T = double;

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  auto             rank  = ec.pg().rank();
  ExecutionHW      ex_hw = ec.exhw();
  Scheduler        sch{ec};

  json jinput;
  check_json(filename);
  auto       is = std::ifstream(filename);
  OptionsMap options_map;
  std::tie(options_map, jinput) = parse_input(is);

  SystemData sys_data{options_map, options_map.scf_options.scf_type};

#if 0
    auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, AO_opt, AO_tis,scf_conv]  
                    = hartree_fock_driver<T>(ec,filename);

    CCSDOptions& ccsd_options = sys_data.options_map.ccsd_options;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    std::string t_errmsg = check_memory_req(ccsd_options.ccsdt_tilesize,sys_data.nbf);
    if(!t_errmsg.empty()) tamm_terminate(t_errmsg);
#endif

    int nsranks = sys_data.nbf/15;
    if(nsranks < 1) nsranks=1;
    int ga_cnn = ec.nnodes();
    if(nsranks>ga_cnn) nsranks=ga_cnn;
    nsranks = nsranks * GA_Cluster_nprocs(0);
    int subranks[nsranks];
    for (int i = 0; i < nsranks; i++) subranks[i] = i;

#if defined(USE_UPCXX)
    upcxx::team subcomm = upcxx::world().split(
            (rank < nsranks) ? 0 : upcxx::team::color_none, 0);
#else
    auto world_comm = ec.pg().comm();
    MPI_Group world_group;
    MPI_Comm_group(world_comm,&world_group);
    MPI_Group subgroup;
    MPI_Group_incl(world_group,nsranks,subranks,&subgroup);
    MPI_Comm subcomm;
    MPI_Comm_create(world_comm,subgroup,&subcomm);
#endif
    
    ProcGroup sub_pg;
    ExecutionContext *sub_ec=nullptr;

    if (rank < nsranks) {
        sub_pg = ProcGroup::create_coll(subcomm);
        sub_ec = new ExecutionContext(sub_pg, DistributionKind::nw, MemoryManagerKind::ga);
    }

    Scheduler sub_sch{*sub_ec};

    //force writet on
    ccsd_options.writet = true;
    ccsd_options.computeTData = true;

    debug = ccsd_options.debug;
    bool skip_ccsd = ccsd_options.skip_ccsd;
    if(rank == 0) ccsd_options.print();

    if(rank==0) cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;
    
    auto [MO,total_orbitals] = setupMOIS(sys_data);

    std::string out_fp = sys_data.output_file_prefix+"."+ccsd_options.basis;
    std::string files_dir = out_fp+"_files/"+sys_data.options_map.scf_options.scf_type;
    std::string files_prefix = /*out_fp;*/ files_dir+"/"+out_fp;
    std::string f1file = files_prefix+".f1_mo";
    std::string t1file = files_prefix+".t1amp";
    std::string t2file = files_prefix+".t2amp";
    std::string v2file = files_prefix+".cholv2";
    std::string cholfile = files_prefix+".cholcount";
    std::string ccsdstatus = files_prefix+".ccsdstatus";
    std::string fullV2file = files_prefix+".fullV2";

    const bool is_rhf = sys_data.is_restricted;
    bool  computeTData = ccsd_options.computeTData; 

    bool ccsd_restart = ccsd_options.readt || 
        ( (fs::exists(t1file) && fs::exists(t2file)     
        && fs::exists(f1file) && fs::exists(v2file)) );

    ExecutionHW ex_hw = ec.exhw();

    TiledIndexSpace N = MO("all");

    Tensor<T> cholVpr, d_t1, d_t2, d_f1, lcao;
    TAMM_SIZE chol_count; tamm::Tile max_cvecs; 
    TiledIndexSpace CI;

    std::vector<T> p_evl_sorted;
    double residual=0, corr_energy=0;

    if(!skip_ccsd) {

    //deallocates F_AO, C_AO
    std::tie (cholVpr,d_f1,lcao,chol_count, max_cvecs, CI) = cd_svd_driver<T>
                        (sys_data, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO, shells, shell_tile_map,
                                ccsd_restart, cholfile);
    free_tensors(lcao);

    if(ccsd_options.writev) ccsd_options.writet = true;

    Tensor<T> d_r1, d_r2;
    std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;

    if(is_rhf) 
        std::tie(p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s)
                = setupTensors_cs(ec,MO,d_f1,ccsd_options.ndiis,ccsd_restart && fs::exists(ccsdstatus) && scf_conv);
    else
        std::tie(p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s)
                = setupTensors(ec,MO,d_f1,ccsd_options.ndiis,ccsd_restart && fs::exists(ccsdstatus) && scf_conv);

    if(ccsd_restart) {
        read_from_disk(d_f1,f1file);
        if(fs::exists(t1file) && fs::exists(t2file)) {
            read_from_disk(d_t1,t1file);
            read_from_disk(d_t2,t2file);
        }
        read_from_disk(cholVpr,v2file);
        ec.pg().barrier();
        p_evl_sorted = tamm::diagonal(d_f1);
    }
    
    else if(ccsd_options.writet) {
        // fs::remove_all(files_dir); 
        if(!fs::exists(files_dir)) fs::create_directories(files_dir);

        write_to_disk(d_f1,f1file);
        write_to_disk(cholVpr,v2file);

        if(rank==0){
          std::ofstream out(cholfile, std::ios::out);
          if(!out) cerr << "Error opening file " << cholfile << endl;
          out << chol_count << std::endl;
          out.close();
        }        
    }

    if(rank==0 && debug){
      print_vector(p_evl_sorted, files_prefix+".eigen_values.txt");
      cout << "Eigen values written to file: " << files_prefix+".eigen_values.txt" << endl << endl;
    }
    
    ec.pg().barrier();

    auto cc_t1 = std::chrono::high_resolution_clock::now();


    ccsd_restart = ccsd_restart && fs::exists(ccsdstatus) && scf_conv;

    t1file = files_prefix+".fullT1amp";
    t2file = files_prefix+".fullT2amp";

    if(ccsd_options.writev) 
        computeTData = computeTData && !fs::exists(fullV2file) 
                && !fs::exists(t1file) && !fs::exists(t2file);

    if(computeTData && is_rhf) 
      setup_full_t1t2(ec,MO,dt1_full,dt2_full);


    if(is_rhf) {
      if(ccsd_restart) {
          if (rank < nsranks) {
              const int ppn = GA_Cluster_nprocs(0);
              if(rank==0) std::cout << "Executing with " << nsranks << " ranks (" << nsranks/ppn << " nodes)" << std::endl; 
              std::tie(residual, corr_energy) = cd_ccsd_cs_driver<T>(
                      sys_data, *sub_ec, MO, CI, d_t1, d_t2, d_f1, 
                      d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
                      p_evl_sorted, 
                      cholVpr, ccsd_restart, files_prefix,
                      computeTData);
          }
          ec.pg().barrier();
      }
      else {
          std::tie(residual, corr_energy) = cd_ccsd_cs_driver<T>(
                  sys_data, ec, MO, CI, d_t1, d_t2, d_f1, 
                  d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
                  p_evl_sorted, 
                  cholVpr, ccsd_restart, files_prefix,
                  computeTData);
          }      
    }
    else {
      if(ccsd_restart) {
          if (rank < nsranks) {
              const int ppn = GA_Cluster_nprocs(0);
              if(rank==0) std::cout << "Executing with " << nsranks << " ranks (" << nsranks/ppn << " nodes)" << std::endl; 
              std::tie(residual, corr_energy) = cd_ccsd_os_driver<T>(
                      sys_data, *sub_ec, MO, CI, d_t1, d_t2, d_f1, 
                      d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
                      p_evl_sorted, 
                      cholVpr, ccsd_restart, files_prefix,
                      computeTData);
          }
          ec.pg().barrier();
      }
      else {
          std::tie(residual, corr_energy) = cd_ccsd_os_driver<T>(
                  sys_data, ec, MO, CI, d_t1, d_t2, d_f1, 
                  d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
                  p_evl_sorted, 
                  cholVpr, ccsd_restart, files_prefix,
                  computeTData);
          }
    }

    if(computeTData && is_rhf) {
        if(ccsd_options.writev) {
            write_to_disk(dt1_full,t1file);
            write_to_disk(dt2_full,t2file);
            free_tensors(dt1_full, dt2_full);
        }
    }

    ccsd_stats(ec, hf_energy,residual,corr_energy,ccsd_options.threshold);

    if(ccsd_options.writet && !fs::exists(ccsdstatus)) {
        // write_to_disk(d_t1,t1file);
        // write_to_disk(d_t2,t2file);
        if(rank==0){
          std::ofstream out(ccsdstatus, std::ios::out);
          if(!out) cerr << "Error opening file " << ccsdstatus << endl;
          out << 1 << std::endl;
          out.close();
        }
    }

    if (rank < nsranks) {
      (*sub_ec).flush_and_sync();
#ifdef USE_UPCXX
      subcomm.destroy();
#else
      MPI_Comm_free(&subcomm);
#endif
    }

    auto cc_t2 = std::chrono::high_resolution_clock::now();
    double ccsd_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) { 
      if(is_rhf)
        std::cout << std::endl << "Time taken for Closed Shell Cholesky CCSD: " << std::fixed << std::setprecision(2) << ccsd_time << " secs" << std::endl;
      else
        std::cout << std::endl << "Time taken for Open Shell Cholesky CCSD: " << std::fixed << std::setprecision(2) << ccsd_time << " secs" << std::endl;
    }

    cc_print(sys_data,d_t1,d_t2,files_prefix);

    if(!ccsd_restart) {
        free_tensors(d_r1,d_r2);
        free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
    }

    if(is_rhf) free_tensors(d_t1, d_t2);
    ec.flush_and_sync();

    } 
    else { //skip ccsd
        d_f1 = {{N,N},{1,1}};
        Tensor<T>::allocate(&ec,d_f1);
    }

    bool  ccsd_t_restart = fs::exists(t1file) && fs::exists(t2file) &&
                           fs::exists(f1file) && fs::exists(fullV2file);

    auto [MO1,total_orbitals1] = setupMOIS(sys_data,true);
    TiledIndexSpace N1 = MO1("all");
    TiledIndexSpace O1 = MO1("occ");
    TiledIndexSpace V1 = MO1("virt");

    // Tensor<T> d_v2{{N,N,N,N},{2,2}};
    // Tensor<T> t_d_f1{{N1,N1},{1,1}};
    Tensor<T> t_d_t1{{V1,O1},{1,1}};
    Tensor<T> t_d_t2{{V1,V1,O1,O1},{2,2}};
    Tensor<T> t_d_v2{{N1,N1,N1,N1},{2,2}};
    Tensor<T> t_d_cv2{{N1,N1,CI},{1,1}};

    T ccsd_t_mem{};
    const double gib = (1024*1024*1024.0);
    const double Osize = MO("occ").max_num_indices();
    const double Vsize = MO("virt").max_num_indices();
    // const double Nsize = N.max_num_indices();
    // const double cind_size = CI.max_num_indices();

    ccsd_t_mem = sum_tensor_sizes(d_f1,t_d_t1,t_d_t2,t_d_v2);
    if(!skip_ccsd) {
        // auto v2_setup_mem = sum_tensor_sizes(d_f1,t_d_v2,t_d_cv2);
        // auto cv2_retile = (Nsize*Nsize*cind_size*8)/gib + sum_tensor_sizes(d_f1,cholVpr,t_d_cv2);
        if(is_rhf) ccsd_t_mem += sum_tensor_sizes(dt1_full,dt2_full);
        else ccsd_t_mem += sum_tensor_sizes(d_t1,d_t2);

        //retiling allocates full GA versions of the tensors.
        ccsd_t_mem +=  (Osize*Vsize + Vsize*Vsize*Osize*Osize)*8/gib;
    }


    if(rank==0) {
        std::cout << std::string(70, '-') << std::endl;
        std::cout << "Total CPU memory required for (T) calculation = " << std::fixed << std::setprecision(2) << ccsd_t_mem << " GiB" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
    }

    if(computeTData && !skip_ccsd) {
      Tensor<T>::allocate(&ec,t_d_cv2);
      retile_tamm_tensor(cholVpr,t_d_cv2,"CholV2");
      free_tensors(cholVpr);

      t_d_v2 = setupV2<T>(ec,MO1,CI,t_d_cv2,chol_count, ex_hw);
      if(ccsd_options.writev) {
          write_to_disk(t_d_v2,fullV2file,true);
          Tensor<T>::deallocate(t_d_v2);
      }
      free_tensors(t_d_cv2);
    }

    double energy1=0, energy2=0;

    if(rank==0) {
        auto mo_tiles = MO.input_tile_sizes();
        cout << endl << "CCSD MO Tiles = " << mo_tiles << endl;   
    }

    Tensor<T>::allocate(&ec,t_d_t1,t_d_t2); //t_d_v2
    if(skip_ccsd) Tensor<T>::allocate(&ec,t_d_v2);

    if(!ccsd_t_restart && !skip_ccsd) {
        if(!is_rhf) {
          dt1_full = d_t1;
          dt2_full = d_t2;
        }        
        if(rank==0) {
            cout << endl << "Retile T1,T2 tensors ... " << endl;   
        }

        Scheduler{ec}   
        (t_d_t1() = 0)
        (t_d_t2() = 0)
        .execute();

        TiledIndexSpace O = MO("occ");
        TiledIndexSpace V = MO("virt");

        if(ccsd_options.writev) {
          // Tensor<T> wd_f1{{N,N},{1,1}};
          Tensor<T> wd_t1{{V,O},{1,1}};
          Tensor<T> wd_t2{{V,V,O,O},{2,2}};
        //   Tensor<T> wd_v2{{N,N,N,N},{2,2}};
                        
          // read_from_disk(t_d_f1,f1file,false,wd_f1);
          read_from_disk(t_d_t1,t1file,false,wd_t1);
          read_from_disk(t_d_t2,t2file,false,wd_t2);
        //   read_from_disk(t_d_v2,fullV2file,false,wd_v2);
          
          ec.pg().barrier();
          // write_to_disk(t_d_f1,f1file);
          write_to_disk(t_d_t1,t1file);
          write_to_disk(t_d_t2,t2file);
        //   write_to_disk(t_d_v2,fullV2file);
        }
        
        else {
          retile_tamm_tensor(dt1_full,t_d_t1);
          retile_tamm_tensor(dt2_full,t_d_t2);
          if(is_rhf) free_tensors(dt1_full, dt2_full);
        //   retile_tamm_tensor(d_v2,t_d_v2,"V2");
        //   free_tensors(d_v2);
        }        
    }
    else if(ccsd_options.writev && !skip_ccsd) {
        // read_from_disk(t_d_f1,f1file);
        read_from_disk(t_d_t1,t1file);
        read_from_disk(t_d_t2,t2file);
        read_from_disk(t_d_v2,fullV2file);
    }

    if(!is_rhf && !skip_ccsd) free_tensors(d_t1, d_t2);
#endif

  double energy1 = 0, energy2 = 0;
  double residual = 0, corr_energy, hf_energy = 0;
  auto   ccsd_options = options_map.ccsd_options;

  sys_data.n_occ_alpha = 12;
  sys_data.n_vir_alpha = 3;

  sys_data.n_occ_beta = sys_data.n_occ_alpha;
  sys_data.n_vir_beta = sys_data.n_vir_alpha;
  sys_data.nocc       = sys_data.n_occ_alpha + sys_data.n_occ_beta;
  sys_data.nmo        = sys_data.n_occ_alpha * 2 + sys_data.n_vir_alpha * 2;

  if(rank == 0) sys_data.print();

  auto [MO1, total_orbitals1] = setupMOIS(sys_data, true);
  TiledIndexSpace N1          = MO1("all");
  TiledIndexSpace O1          = MO1("occ");
  TiledIndexSpace V1          = MO1("virt");

  Tensor<T> d_f1{{N1, N1}, {1, 1}};
  Tensor<T> t_d_t1{{V1, O1}, {1, 1}};
  Tensor<T> t_d_t2{{V1, V1, O1, O1}, {2, 2}};
  Tensor<T> t_d_v2{{N1, N1, N1, N1}, {2, 2}};
  sch.allocate(t_d_t1, t_d_t2, t_d_v2, d_f1).execute();
  std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);

  // cc_t1 = std::chrono::high_resolution_clock::now();

  Index            noab = MO1("occ").num_tiles();
  Index            nvab = MO1("virt").num_tiles();
  std::vector<int> k_spin;
  for(tamm::Index x = 0; x < noab / 2; x++) k_spin.push_back(1);
  for(tamm::Index x = noab / 2; x < noab; x++) k_spin.push_back(2);
  for(tamm::Index x = 0; x < nvab / 2; x++) k_spin.push_back(1);
  for(tamm::Index x = nvab / 2; x < nvab; x++) k_spin.push_back(2);

  bool is_restricted = sys_data.is_restricted;

  if(rank == 0) {
    if(is_restricted) cout << endl << "Running Closed Shell CCSD(T) calculation" << endl;
    else cout << endl << "Running Open Shell CCSD(T) calculation" << endl;
  }

  bool                            seq_h3b    = true;
  Index                           cache_size = 32;
  LRUCache<Index, std::vector<T>> cache_s1t{cache_size};
  LRUCache<Index, std::vector<T>> cache_s1v{cache_size};
  LRUCache<Index, std::vector<T>> cache_d1t{cache_size * noab};
  LRUCache<Index, std::vector<T>> cache_d1v{cache_size * noab};
  LRUCache<Index, std::vector<T>> cache_d2t{cache_size * nvab};
  LRUCache<Index, std::vector<T>> cache_d2v{cache_size * nvab};

  if(rank == 0 && seq_h3b) cout << "running seq h3b loop variant..." << endl;

  double ccsd_t_time = 0, total_t_time = 0;
  // cc_t1 = std::chrono::high_resolution_clock::now();
  std::tie(energy1, energy2, ccsd_t_time, total_t_time) = ccsd_t_fused_driver_new<T>(
    sys_data, ec, k_spin, MO1, t_d_t1, t_d_t2, t_d_v2, p_evl_sorted, hf_energy + corr_energy,
    is_restricted, cache_s1t, cache_s1v, cache_d1t, cache_d1v, cache_d2t, cache_d2v, seq_h3b);

  // cc_t2 = std::chrono::high_resolution_clock::now();
  // auto ccsd_t_time =
  //     std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  energy1 = ec.pg().reduce(&energy1, ReduceOp::sum, 0);
  energy2 = ec.pg().reduce(&energy2, ReduceOp::sum, 0);

  if(rank == 0) {
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
    ccsd_t_fused_driver_calculator_ops<T>(sys_data, ec, k_spin, MO1, p_evl_sorted,
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
    write_json_data(sys_data, "CCSD_T");
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

  free_tensors(t_d_t1, t_d_t2, d_f1, t_d_v2);

  ec.flush_and_sync();
  // delete ec;
}
