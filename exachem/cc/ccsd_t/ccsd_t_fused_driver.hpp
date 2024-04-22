/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "ccsd_t_all_fused.hpp"
#else
#include "ccsd_t_all_fused_cpu.hpp"
#endif
#include "ccsd_t_common.hpp"

namespace exachem::cc::ccsd_t {

void ccsd_t_driver(ExecutionContext& ec, ChemEnv& chem_env);
}
void finalizememmodule();

/**
 *  to check if target NVIDIA GPUs can support the fully-fused kernel
 *  based on 3rd gen. tensor cores or not.
 *  - requirements: (1) arch >= 80 and (2) driver >= 11.2?
 **/
#if defined(USE_CUDA)
inline int checkCudaKernelCompatible(bool r0) {
  int            version = 0;
  cudaDeviceProp dP;

  cudaError_t cuda_arch = cudaGetDeviceProperties(&dP, 0);
  if(cuda_arch != cudaSuccess) {
    cudaError_t error = cudaGetLastError();
    if(r0) printf("CUDA error: %s", cudaGetErrorString(error));
    return cuda_arch; /* Failure */
  }
  // printf ("[%s] dP.major: %d, dP.minor: %d\n", __func__, dP.major, dP.minor);

  // the version is returned as (1000 major + 10 minior)
  cudaError_t cuda_driver = cudaRuntimeGetVersion(&version);
  if(cuda_driver != cudaSuccess) {
    cudaError_t error = cudaGetLastError();
    if(r0) printf("CUDA error: %s", cudaGetErrorString(error));
    return cuda_driver;
  }

  //
  int driver_major = version / 1000;
  int driver_minor = (version - (driver_major * 1000)) / 10;
  if(r0)
    printf("Given info.: Compatibility = %d.%d, CUDA Version = %d.%d\n", dP.major, dP.minor,
           driver_major, driver_minor);

  if(dP.major >= 8 && driver_major >= 11 && driver_minor >= 1) { return 1; }
  else { return -1; }
}
#endif

//
template<typename T>
std::tuple<T, T, double, double> ccsd_t_fused_driver_new(
  ChemEnv& chem_env, ExecutionContext& ec, std::vector<int>& k_spin, const TiledIndexSpace& MO,
  Tensor<T>& d_t1, Tensor<T>& d_t2, V2Tensors<T>& d_v2, std::vector<T>& k_evl_sorted,
  T hf_ccsd_energy, bool is_restricted, LRUCache<Index, std::vector<T>>& cache_s1t,
  LRUCache<Index, std::vector<T>>& cache_s1v, LRUCache<Index, std::vector<T>>& cache_d1t,
  LRUCache<Index, std::vector<T>>& cache_d1v, LRUCache<Index, std::vector<T>>& cache_d2t,
  LRUCache<Index, std::vector<T>>& cache_d2v, bool seq_h3b = false, bool tilesize_opt = true) {
  //
  auto rank     = ec.pg().rank().value();
  bool nodezero = rank == 0;

#if defined(USE_CUDA)
// int opt_CUDA_TC = checkCudaKernelCompatible(nodezero);
#if defined(USE_NV_TC)
  if(nodezero)
    cout << "Enabled the fully-fused kernel based on FP64 TC (Third Gen. Tensor Cores)" << endl;
#else
  if(nodezero) cout << "Enabled the fully-fused kernel based on FP64" << endl;
#endif
#endif

  Index noab = MO("occ").num_tiles();
  Index nvab = MO("virt").num_tiles();

  Index noa = MO("occ_alpha").num_tiles();
  Index nva = MO("virt_alpha").num_tiles();

  auto                mo_tiles = MO.input_tile_sizes();
  std::vector<size_t> k_range;
  std::vector<size_t> k_offset;
  size_t              sum = 0;
  for(auto x: mo_tiles) {
    k_range.push_back(x);
    k_offset.push_back(sum);
    sum += x;
  }

  if(nodezero) {
    cout << "noa,nva = " << noa << ", " << nva << endl;
    cout << "noab,nvab = " << noab << ", " << nvab << endl;
    // cout << "k_spin = " << k_spin << endl;
    // cout << "k_range = " << k_range << endl;
    cout << "MO Tiles = " << mo_tiles << endl;
  }

  // TODO replicate d_t1 L84-89 ccsd_t_gpu.F
  T              energy1 = 0.0;
  T              energy2 = 0.0;
  std::vector<T> energy_l;
  energy_l.resize(2);
  energy_l[0] = 0.0;
  energy_l[1] = 0.0;

#if defined(USE_CUDA)
  std::shared_ptr<gpuEvent_t> done_compute(new gpuEvent_t,
                                           [](gpuEvent_t* e) { CUDA_SAFE(cudaEventDestroy(*e)); });
  CUDA_SAFE(cudaEventCreateWithFlags(done_compute.get(), cudaEventDisableTiming));
  std::shared_ptr<gpuEvent_t> done_copy(new gpuEvent_t,
                                        [](gpuEvent_t* e) { CUDA_SAFE(cudaEventDestroy(*e)); });
  CUDA_SAFE(cudaEventCreateWithFlags(done_copy.get(), cudaEventDisableTiming));

  std::shared_ptr<hostEnergyReduceData_t> reduceData = std::make_shared<hostEnergyReduceData_t>();
#elif defined(USE_HIP)
  std::shared_ptr<gpuEvent_t> done_compute(new gpuEvent_t,
                                           [](gpuEvent_t* e) { HIP_SAFE(hipEventDestroy(*e)); });
  HIP_SAFE(hipEventCreateWithFlags(done_compute.get(), hipEventDisableTiming));
  std::shared_ptr<gpuEvent_t> done_copy(new gpuEvent_t,
                                        [](gpuEvent_t* e) { HIP_SAFE(hipEventDestroy(*e)); });
  HIP_SAFE(hipEventCreateWithFlags(done_copy.get(), hipEventDisableTiming));

  std::shared_ptr<hostEnergyReduceData_t> reduceData = std::make_shared<hostEnergyReduceData_t>();
#elif defined(USE_DPCPP)
  std::shared_ptr<gpuEvent_t> done_compute = std::make_shared<gpuEvent_t>();
  std::shared_ptr<gpuEvent_t> done_copy    = std::make_shared<gpuEvent_t>();

  std::shared_ptr<hostEnergyReduceData_t> reduceData = std::make_shared<hostEnergyReduceData_t>();
#endif

  AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
  ac->allocate(0);
  int64_t taskcount = 0;
  int64_t next      = ac->fetch_add(0, 1);

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  size_t max_pdim = 0;
  size_t max_hdim = 0;
  for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++)
    max_pdim = std::max(max_pdim, k_range[t_p4b]);
  for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) max_hdim = std::max(max_hdim, k_range[t_h1b]);

  size_t max_d1_kernels_pertask = 9 * noab;
  size_t max_d2_kernels_pertask = 9 * nvab;
  if(tilesize_opt) {
    max_d1_kernels_pertask = 9 * noa;
    max_d2_kernels_pertask = 9 * nva;
  }

  //
  size_t size_T_s1_t1 = 9 * (max_pdim) * (max_hdim);
  size_t size_T_s1_v2 = 9 * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d1_t2 = max_d1_kernels_pertask * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d1_v2 = max_d1_kernels_pertask * (max_pdim) * (max_hdim * max_hdim * max_hdim);
  size_t size_T_d2_t2 = max_d2_kernels_pertask * (max_pdim * max_pdim) * (max_hdim * max_hdim);
  size_t size_T_d2_v2 = max_d2_kernels_pertask * (max_pdim * max_pdim * max_pdim) * (max_hdim);

  T* df_host_pinned_s1_t1{nullptr};
  T* df_host_pinned_s1_v2{nullptr};
  T* df_host_pinned_d1_t2{nullptr};
  T* df_host_pinned_d1_v2{nullptr};
  T* df_host_pinned_d2_t2{nullptr};
  T* df_host_pinned_d2_v2{nullptr};

  int* df_simple_s1_size = static_cast<int*>(getHostMem(sizeof(int) * (6)));
  int* df_simple_s1_exec = static_cast<int*>(getHostMem(sizeof(int) * (9)));
  int* df_simple_d1_size = static_cast<int*>(getHostMem(sizeof(int) * (7 * noab)));
  int* df_simple_d1_exec = static_cast<int*>(getHostMem(sizeof(int) * (9 * noab)));
  int* df_simple_d2_size = static_cast<int*>(getHostMem(sizeof(int) * (7 * nvab)));
  int* df_simple_d2_exec = static_cast<int*>(getHostMem(sizeof(int) * (9 * nvab)));

  int* host_d1_size = static_cast<int*>(getHostMem(sizeof(int) * (noab)));
  int* host_d2_size = static_cast<int*>(getHostMem(sizeof(int) * (nvab)));

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  T* df_dev_s1_t1_all = static_cast<T*>(getGpuMem(sizeof(T) * size_T_s1_t1));
  T* df_dev_s1_v2_all = static_cast<T*>(getGpuMem(sizeof(T) * size_T_s1_v2));
  T* df_dev_d1_t2_all = static_cast<T*>(getGpuMem(sizeof(T) * size_T_d1_t2));
  T* df_dev_d1_v2_all = static_cast<T*>(getGpuMem(sizeof(T) * size_T_d1_v2));
  T* df_dev_d2_t2_all = static_cast<T*>(getGpuMem(sizeof(T) * size_T_d2_t2));
  T* df_dev_d2_v2_all = static_cast<T*>(getGpuMem(sizeof(T) * size_T_d2_v2));

  df_host_pinned_s1_t1 = static_cast<T*>(getPinnedMem(sizeof(T) * size_T_s1_t1));
  df_host_pinned_s1_v2 = static_cast<T*>(getPinnedMem(sizeof(T) * size_T_s1_v2));
  df_host_pinned_d1_t2 = static_cast<T*>(getPinnedMem(sizeof(T) * size_T_d1_t2));
  df_host_pinned_d1_v2 = static_cast<T*>(getPinnedMem(sizeof(T) * size_T_d1_v2));
  df_host_pinned_d2_t2 = static_cast<T*>(getPinnedMem(sizeof(T) * size_T_d2_t2));
  df_host_pinned_d2_v2 = static_cast<T*>(getPinnedMem(sizeof(T) * size_T_d2_v2));
#else // cpu
  df_host_pinned_s1_t1 = static_cast<T*>(getHostMem(sizeof(T) * size_T_s1_t1));
  df_host_pinned_s1_v2 = static_cast<T*>(getHostMem(sizeof(T) * size_T_s1_v2));
  df_host_pinned_d1_t2 = static_cast<T*>(getHostMem(sizeof(T) * size_T_d1_t2));
  df_host_pinned_d1_v2 = static_cast<T*>(getHostMem(sizeof(T) * size_T_d1_v2));
  df_host_pinned_d2_t2 = static_cast<T*>(getHostMem(sizeof(T) * size_T_d2_t2));
  df_host_pinned_d2_v2 = static_cast<T*>(getHostMem(sizeof(T) * size_T_d2_v2));
#endif

  size_t max_num_blocks = chem_env.ioptions.ccsd_options.ccsdt_tilesize;
  max_num_blocks        = std::ceil((max_num_blocks + 4 - 1) / 4.0);

  T* df_host_energies = static_cast<T*>(getHostMem(sizeof(T) * std::pow(max_num_blocks, 6) * 2));
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  T* df_dev_energies = static_cast<T*>(getGpuMem(sizeof(T) * std::pow(max_num_blocks, 6) * 2));
#endif

  int num_task = 0;
  if(!seq_h3b) {
    if(rank == 0) {
      std::cout << "456123 parallel 6d loop variant" << std::endl << std::endl;
      // std::cout << "tile142563,kernel,memcpy,data,total" << std::endl;
    }
    for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
      for(size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
        for(size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
          for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) { //
            for(size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
              for(size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
                if((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
                   (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
                  if((!is_restricted) || (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
                                          k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {
                    if(next == taskcount) {
                      //
                      T factor = 1.0;
                      if(is_restricted) factor = 2.0;
                      if((t_p4b == t_p5b) && (t_p5b == t_p6b)) { factor /= 6.0; }
                      else if((t_p4b == t_p5b) || (t_p5b == t_p6b)) { factor /= 2.0; }

                      if((t_h1b == t_h2b) && (t_h2b == t_h3b)) { factor /= 6.0; }
                      else if((t_h1b == t_h2b) || (t_h2b == t_h3b)) { factor /= 2.0; }

                      num_task++;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
                      ccsd_t_fully_fused_none_df_none_task<T>(
                        is_restricted, noab, nvab, rank, k_spin, k_range, k_offset, d_t1, d_t2,
                        d_v2, k_evl_sorted,
                        //
                        df_host_pinned_s1_t1, df_host_pinned_s1_v2, df_host_pinned_d1_t2,
                        df_host_pinned_d1_v2, df_host_pinned_d2_t2, df_host_pinned_d2_v2,
                        df_host_energies,
                        //
                        //
                        //
                        host_d1_size, host_d2_size,
                        //
                        df_simple_s1_size, df_simple_d1_size, df_simple_d2_size, df_simple_s1_exec,
                        df_simple_d1_exec, df_simple_d2_exec,
                        //
                        df_dev_s1_t1_all, df_dev_s1_v2_all, df_dev_d1_t2_all, df_dev_d1_v2_all,
                        df_dev_d2_t2_all, df_dev_d2_v2_all, df_dev_energies,
                        //
                        t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor, taskcount,
                        max_d1_kernels_pertask, max_d2_kernels_pertask,
                        //
                        size_T_s1_t1, size_T_s1_v2, size_T_d1_t2, size_T_d1_v2, size_T_d2_t2,
                        size_T_d2_v2,
                        //
                        energy_l,
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
                        reduceData.get(),
#endif
                        cache_s1t, cache_s1v, cache_d1t, cache_d1v, cache_d2t, cache_d2v,
                        //
                        done_compute.get(), done_copy.get());
#else
                      total_fused_ccsd_t_cpu<T>(
                        is_restricted, noab, nvab, rank, k_spin, k_range, k_offset, d_t1, d_t2,
                        d_v2, k_evl_sorted,
                        //
                        df_host_pinned_s1_t1, df_host_pinned_s1_v2, df_host_pinned_d1_t2,
                        df_host_pinned_d1_v2, df_host_pinned_d2_t2, df_host_pinned_d2_v2,
                        df_host_energies, host_d1_size, host_d2_size,
                        //
                        df_simple_s1_size, df_simple_d1_size, df_simple_d2_size, df_simple_s1_exec,
                        df_simple_d1_exec, df_simple_d2_exec,
                        //
                        t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor, taskcount,
                        max_d1_kernels_pertask, max_d2_kernels_pertask,
                        //
                        size_T_s1_t1, size_T_s1_v2, size_T_d1_t2, size_T_d1_v2, size_T_d2_t2,
                        size_T_d2_v2,
                        //
                        energy_l, cache_s1t, cache_s1v, cache_d1t, cache_d1v, cache_d2t, cache_d2v);
#endif

                      next = ac->fetch_add(0, 1);
                    }
                    taskcount++;
                  }
                }
              }
            }
          }
        }
      }
    }
  }      // parallel h3b loop
  else { // seq h3b loop
#if 1
    if(rank == 0) {
      std::cout << "14256-seq3 loop variant" << std::endl << std::endl;
      // std::cout << "tile142563,kernel,memcpy,data,total" << std::endl;
    }
    for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) { //
      for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
        for(size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
          for(size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
            for(size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
#endif
              // #if 0
              //     for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
              //     for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
              //     for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
              //     for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
              //     for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
              // #endif
              if(next == taskcount) {
                for(size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
                  if((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
                     (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
                    if((!is_restricted) || (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
                                            k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {
                      T factor = 1.0;
                      if(is_restricted) factor = 2.0;

                      //
                      if((t_p4b == t_p5b) && (t_p5b == t_p6b)) { factor /= 6.0; }
                      else if((t_p4b == t_p5b) || (t_p5b == t_p6b)) { factor /= 2.0; }

                      if((t_h1b == t_h2b) && (t_h2b == t_h3b)) { factor /= 6.0; }
                      else if((t_h1b == t_h2b) || (t_h2b == t_h3b)) { factor /= 2.0; }

                      //
                      num_task++;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
                      ccsd_t_fully_fused_none_df_none_task<T>(
                        is_restricted, noab, nvab, rank, k_spin, k_range, k_offset, d_t1, d_t2,
                        d_v2, k_evl_sorted,
                        //
                        df_host_pinned_s1_t1, df_host_pinned_s1_v2, df_host_pinned_d1_t2,
                        df_host_pinned_d1_v2, df_host_pinned_d2_t2, df_host_pinned_d2_v2,
                        df_host_energies,
                        //
                        //
                        //
                        host_d1_size, host_d2_size,
                        //
                        df_simple_s1_size, df_simple_d1_size, df_simple_d2_size, df_simple_s1_exec,
                        df_simple_d1_exec, df_simple_d2_exec,
                        //
                        df_dev_s1_t1_all, df_dev_s1_v2_all, df_dev_d1_t2_all, df_dev_d1_v2_all,
                        df_dev_d2_t2_all, df_dev_d2_v2_all, df_dev_energies,
                        //
                        t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor, taskcount,
                        max_d1_kernels_pertask, max_d2_kernels_pertask,
                        //
                        size_T_s1_t1, size_T_s1_v2, size_T_d1_t2, size_T_d1_v2, size_T_d2_t2,
                        size_T_d2_v2,
                        //
                        energy_l,
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
                        reduceData.get(),
#endif
                        cache_s1t, cache_s1v, cache_d1t, cache_d1v, cache_d2t, cache_d2v,
                        //
                        done_compute.get(), done_copy.get());
#else
            total_fused_ccsd_t_cpu<T>(
              is_restricted, noab, nvab, rank, k_spin, k_range, k_offset, d_t1, d_t2, d_v2,
              k_evl_sorted,
              //
              df_host_pinned_s1_t1, df_host_pinned_s1_v2, df_host_pinned_d1_t2,
              df_host_pinned_d1_v2, df_host_pinned_d2_t2, df_host_pinned_d2_v2, df_host_energies,
              host_d1_size, host_d2_size,
              //
              df_simple_s1_size, df_simple_d1_size, df_simple_d2_size, df_simple_s1_exec,
              df_simple_d1_exec, df_simple_d2_exec,
              //
              t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor, taskcount, max_d1_kernels_pertask,
              max_d2_kernels_pertask,
              //
              size_T_s1_t1, size_T_s1_v2, size_T_d1_t2, size_T_d1_v2, size_T_d2_t2, size_T_d2_v2,
              //
              energy_l, cache_s1t, cache_s1v, cache_d1t, cache_d1v, cache_d2t, cache_d2v);
#endif
                    }
                  }
                } // h3b

                next = ac->fetch_add(0, 1);
              }
              taskcount++;
            }
          }
        }
      }
    }
  } // end seq h3b

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  gpuDeviceSynchronize();
#endif

  energy1 = energy_l[0];
  energy2 = energy_l[1];

  freeHostMem(df_simple_s1_exec);
  freeHostMem(df_simple_s1_size);
  freeHostMem(df_simple_d1_exec);
  freeHostMem(df_simple_d1_size);
  freeHostMem(host_d1_size);
  freeHostMem(df_simple_d2_exec);
  freeHostMem(df_simple_d2_size);
  freeHostMem(host_d2_size);
  freeHostMem(df_host_energies);

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  freeGpuMem(df_dev_s1_t1_all);
  freeGpuMem(df_dev_s1_v2_all);
  freeGpuMem(df_dev_d1_t2_all);
  freeGpuMem(df_dev_d1_v2_all);
  freeGpuMem(df_dev_d2_t2_all);
  freeGpuMem(df_dev_d2_v2_all);
  freeGpuMem(df_dev_energies);

  freePinnedMem(df_host_pinned_s1_t1);
  freePinnedMem(df_host_pinned_s1_v2);
  freePinnedMem(df_host_pinned_d1_t2);
  freePinnedMem(df_host_pinned_d1_v2);
  freePinnedMem(df_host_pinned_d2_t2);
  freePinnedMem(df_host_pinned_d2_v2);

#else // cpu
freeHostMem(df_host_pinned_s1_t1);
freeHostMem(df_host_pinned_s1_v2);
freeHostMem(df_host_pinned_d1_t2);
freeHostMem(df_host_pinned_d1_v2);
freeHostMem(df_host_pinned_d2_t2);
freeHostMem(df_host_pinned_d2_v2);
#endif

  finalizememmodule();

  auto cc_t2 = std::chrono::high_resolution_clock::now();
  auto ccsd_t_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  ec.pg().barrier();
  cc_t2 = std::chrono::high_resolution_clock::now();
  auto total_t_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  //
  next = ac->fetch_add(0, 1);
  ac->deallocate();
  delete ac;

  return std::make_tuple(energy1, energy2, ccsd_t_time, total_t_time);
}

template<typename T>
void ccsd_t_fused_driver_calculator_ops(ChemEnv& chem_env, ExecutionContext& ec,
                                        std::vector<int>& k_spin, const TiledIndexSpace& MO,
                                        std::vector<T>& k_evl_sorted, double hf_ccsd_energy,
                                        bool is_restricted, long double& total_num_ops,
                                        bool seq_h3b = false) {
  auto rank = ec.pg().rank().value();

  Index noab = MO("occ").num_tiles();
  Index nvab = MO("virt").num_tiles();

  auto                mo_tiles = MO.input_tile_sizes();
  std::vector<size_t> k_range;
  std::vector<size_t> k_offset;
  size_t              sum = 0;
  for(auto x: mo_tiles) {
    k_range.push_back(x);
    k_offset.push_back(sum);
    sum += x;
  }

  //
  //  "list of tasks": size (t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b), factor,
  //
  std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, T>> list_tasks;
  if(!seq_h3b) {
    for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
      for(size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
        for(size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
          for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) { //
            for(size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
              for(size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
                //
                if((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
                   (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
                  if((!is_restricted) || (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
                                          k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {
                    //
                    double factor = 1.0;
                    if(is_restricted) factor = 2.0;
                    if((t_p4b == t_p5b) && (t_p5b == t_p6b)) { factor /= 6.0; }
                    else if((t_p4b == t_p5b) || (t_p5b == t_p6b)) { factor /= 2.0; }

                    if((t_h1b == t_h2b) && (t_h2b == t_h3b)) { factor /= 6.0; }
                    else if((t_h1b == t_h2b) || (t_h2b == t_h3b)) { factor /= 2.0; }

                    //
                    list_tasks.push_back(
                      std::make_tuple(t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor));
                  }
                }
              }
            }
          }
        }
      }
    }    // nested for loops
  }      // parallel h3b loop
  else { // seq h3b loop
    for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
      for(size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
        for(size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
          for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
            for(size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
              //
              for(size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
                if((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
                   (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
                  if((!is_restricted) || (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
                                          k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {
                    double factor = 1.0;
                    if(is_restricted) factor = 2.0;
                    //
                    if((t_p4b == t_p5b) && (t_p5b == t_p6b)) { factor /= 6.0; }
                    else if((t_p4b == t_p5b) || (t_p5b == t_p6b)) { factor /= 2.0; }

                    if((t_h1b == t_h2b) && (t_h2b == t_h3b)) { factor /= 6.0; }
                    else if((t_h1b == t_h2b) || (t_h2b == t_h3b)) { factor /= 2.0; }

                    //
                    list_tasks.push_back(
                      std::make_tuple(t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor));
                  }
                }
              } // h3b
            }
          }
        }
      }
    } // nested for loops
  }   // end seq h3b

  total_num_ops = (long double) ccsd_t_fully_fused_performance(
    is_restricted, list_tasks, rank, 1, noab, nvab, k_spin, k_range, k_offset, k_evl_sorted);
}
