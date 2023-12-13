/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "fused_common.hpp"

void dev_mem_s(size_t, size_t, size_t, size_t, size_t, size_t);
void dev_mem_d(size_t, size_t, size_t, size_t, size_t, size_t);

#define CEIL(a, b) (((a) + (b) -1) / (b))

inline void hostEnergyReduce(void* data) {
  hostEnergyReduceData_t* data_t        = (hostEnergyReduceData_t*) data;
  double*                 host_energies = data_t->host_energies;

  double final_energy_1 = 0.0;
  double final_energy_2 = 0.0;
  for(size_t i = 0; i < data_t->num_blocks; i++) {
    final_energy_1 += host_energies[i];
    final_energy_2 += host_energies[i + data_t->num_blocks];
  }

  data_t->result_energy[0] += final_energy_1 * data_t->factor;
  data_t->result_energy[1] += final_energy_2 * data_t->factor;
}

// driver for the fully-fused kernel (FP64)
template<typename T>
void fully_fused_ccsd_t_gpu(gpuStream_t& stream, size_t num_blocks, size_t base_size_h1b,
                            size_t base_size_h2b, size_t base_size_h3b, size_t base_size_p4b,
                            size_t base_size_p5b, size_t base_size_p6b,
                            //
                            T* df_dev_d1_t2_all, T* df_dev_d1_v2_all, T* df_dev_d2_t2_all,
                            T* df_dev_d2_v2_all, T* df_dev_s1_t1_all, T* df_dev_s1_v2_all,
                            //
                            int* host_d1_size, int* host_d1_exec, // used
                            int* host_d2_size, int* host_d2_exec, int* host_s1_size,
                            int* host_s1_exec,
                            //
                            size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
                            size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
                            size_t size_max_dim_s1_t1, size_t size_max_dim_s1_v2,
                            //
                            T factor,
                            //
                            T* dev_evl_sorted_h1b, T* dev_evl_sorted_h2b, T* dev_evl_sorted_h3b,
                            T* dev_evl_sorted_p4b, T* dev_evl_sorted_p5b, T* dev_evl_sorted_p6b,
                            T* partial_energies, gpuEvent_t* done_copy
#if defined(USE_DPCPP)
                            , int* s1_size, int* s1_exec, int* d1_size, int* d1_exec, int* d2_size,
                            int* d2_exec
#endif
                            );
#if defined(USE_CUDA) && defined(USE_NV_TC)
// driver for fully-fused kernel for 3rd gen. tensor core (FP64)
template<typename T>
void ccsd_t_fully_fused_nvidia_tc_fp64(gpuStream_t& stream, size_t numBlks, size_t size_h3,
                                       size_t size_h2, size_t size_h1, size_t size_p6,
                                       size_t size_p5, size_t size_p4,
                                       //
                                       T* dev_s1_t1_all, T* dev_s1_v2_all, T* dev_d1_t2_all,
                                       T* dev_d1_v2_all, T* dev_d2_t2_all, T* dev_d2_v2_all,
                                       //
                                       int* host_size_d1_h7b, int* host_size_d2_p7b,
                                       int* host_exec_s1, int* host_exec_d1, int* host_exec_d2,
                                       //
                                       size_t size_noab, size_t size_nvab,
                                       size_t size_max_dim_s1_t1, size_t size_max_dim_s1_v2,
                                       size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
                                       size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
                                       //
                                       T factor, T* dev_evl_sorted_h1b, T* dev_evl_sorted_h2b,
                                       T* dev_evl_sorted_h3b, T* dev_evl_sorted_p4b,
                                       T* dev_evl_sorted_p5b, T* dev_evl_sorted_p6b,
                                       T* dev_final_energies, gpuEvent_t* done_copy);
#endif

template<typename T>
void ccsd_t_fully_fused_none_df_none_task(
  bool is_restricted, const Index noab, const Index nvab, int64_t rank, std::vector<int>& k_spin,
  std::vector<size_t>& k_range, std::vector<size_t>& k_offset, Tensor<T>& d_t1, Tensor<T>& d_t2,
  V2Tensors<T>& d_v2, std::vector<T>& k_evl_sorted,
  //
  T* df_host_pinned_s1_t1, T* df_host_pinned_s1_v2, T* df_host_pinned_d1_t2,
  T* df_host_pinned_d1_v2, T* df_host_pinned_d2_t2, T* df_host_pinned_d2_v2, T* host_energies,
  // for new fully-fused kernel
  int* host_d1_size_h7b, int* host_d2_size_p7b,
  //
  int* df_simple_s1_size, int* df_simple_d1_size, int* df_simple_d2_size, int* df_simple_s1_exec,
  int* df_simple_d1_exec, int* df_simple_d2_exec,
  //
  T* df_dev_s1_t1_all, T* df_dev_s1_v2_all, T* df_dev_d1_t2_all, T* df_dev_d1_v2_all,
  T* df_dev_d2_t2_all, T* df_dev_d2_v2_all, T* dev_energies,
  //
  size_t t_h1b, size_t t_h2b, size_t t_h3b, size_t t_p4b, size_t t_p5b, size_t t_p6b, T factor,
  size_t taskid, size_t max_d1_kernels_pertask, size_t max_d2_kernels_pertask,
  //
  size_t size_T_s1_t1, size_t size_T_s1_v2, size_t size_T_d1_t2, size_t size_T_d1_v2,
  size_t size_T_d2_t2, size_t size_T_d2_v2,
  //
  std::vector<T>& energy_l,
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  hostEnergyReduceData_t* reduceData,
#endif
  LRUCache<Index, std::vector<T>>& cache_s1t, LRUCache<Index, std::vector<T>>& cache_s1v,
  LRUCache<Index, std::vector<T>>& cache_d1t, LRUCache<Index, std::vector<T>>& cache_d1v,
  LRUCache<Index, std::vector<T>>& cache_d2t, LRUCache<Index, std::vector<T>>& cache_d2v,
  gpuEvent_t* done_compute, gpuEvent_t* done_copy) {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  // get (round-robin) GPU stream from pool
  gpuStream_t& stream = tamm::GPUStreamPool::getInstance().getStream();
  // get GPU memory handle from pool
  auto& memPool = tamm::GPUPooledStorageManager::getInstance();
#endif

  // Index p4b,p5b,p6b,h1b,h2b,h3b;
  const size_t max_dim_s1_t1 = size_T_s1_t1 / 9;
  const size_t max_dim_s1_v2 = size_T_s1_v2 / 9;
  const size_t max_dim_d1_t2 = size_T_d1_t2 / max_d1_kernels_pertask;
  const size_t max_dim_d1_v2 = size_T_d1_v2 / max_d1_kernels_pertask;
  const size_t max_dim_d2_t2 = size_T_d2_t2 / max_d2_kernels_pertask;
  const size_t max_dim_d2_v2 = size_T_d2_v2 / max_d2_kernels_pertask;

  int df_num_s1_enabled, df_num_d1_enabled, df_num_d2_enabled;

  size_t base_size_h1b = k_range[t_h1b];
  size_t base_size_h2b = k_range[t_h2b];
  size_t base_size_h3b = k_range[t_h3b];
  size_t base_size_p4b = k_range[t_p4b];
  size_t base_size_p5b = k_range[t_p5b];
  size_t base_size_p6b = k_range[t_p6b];

  T* host_evl_sorted_h1b = &k_evl_sorted[k_offset[t_h1b]];
  T* host_evl_sorted_h2b = &k_evl_sorted[k_offset[t_h2b]];
  T* host_evl_sorted_h3b = &k_evl_sorted[k_offset[t_h3b]];
  T* host_evl_sorted_p4b = &k_evl_sorted[k_offset[t_p4b]];
  T* host_evl_sorted_p5b = &k_evl_sorted[k_offset[t_p5b]];
  T* host_evl_sorted_p6b = &k_evl_sorted[k_offset[t_p6b]];

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  T* dev_evl_sorted_h1b = static_cast<T*>(memPool.allocate(sizeof(T) * base_size_h1b));
  T* dev_evl_sorted_h2b = static_cast<T*>(memPool.allocate(sizeof(T) * base_size_h2b));
  T* dev_evl_sorted_h3b = static_cast<T*>(memPool.allocate(sizeof(T) * base_size_h3b));
  T* dev_evl_sorted_p4b = static_cast<T*>(memPool.allocate(sizeof(T) * base_size_p4b));
  T* dev_evl_sorted_p5b = static_cast<T*>(memPool.allocate(sizeof(T) * base_size_p5b));
  T* dev_evl_sorted_p6b = static_cast<T*>(memPool.allocate(sizeof(T) * base_size_p6b));

  if(!gpuEventQuery(*done_copy)) { gpuEventSynchronize(*done_copy); }
#endif

  // resets
  std::fill(df_simple_s1_exec, df_simple_s1_exec + (9), -1);
  std::fill(df_simple_d1_exec, df_simple_d1_exec + (9 * noab), -1);
  std::fill(df_simple_d2_exec, df_simple_d2_exec + (9 * nvab), -1);

  ccsd_t_data_s1_new(is_restricted, noab, nvab, k_spin, d_t1, d_t2, d_v2, k_evl_sorted, k_range,
                     t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b,
                     //
                     size_T_s1_t1, size_T_s1_v2, df_simple_s1_size, df_simple_s1_exec,
                     df_host_pinned_s1_t1, df_host_pinned_s1_v2, &df_num_s1_enabled,
                     //
                     cache_s1t, cache_s1v);

  ccsd_t_data_d1_new(is_restricted, noab, nvab, k_spin, d_t1, d_t2, d_v2, k_evl_sorted, k_range,
                     t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, max_d1_kernels_pertask,
                     //
                     size_T_d1_t2, size_T_d1_v2, df_host_pinned_d1_t2, df_host_pinned_d1_v2,
                     //
                     host_d1_size_h7b,
                     //
                     df_simple_d1_size, df_simple_d1_exec, &df_num_d1_enabled,
                     //
                     cache_d1t, cache_d1v);

  ccsd_t_data_d2_new(is_restricted, noab, nvab, k_spin, d_t1, d_t2, d_v2, k_evl_sorted, k_range,
                     t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, max_d2_kernels_pertask,
                     //
                     size_T_d2_t2, size_T_d2_v2, df_host_pinned_d2_t2, df_host_pinned_d2_v2,
                     //
                     host_d2_size_p7b,
                     //
                     df_simple_d2_size, df_simple_d2_exec, &df_num_d2_enabled,
                     //
                     cache_d2t, cache_d2v);

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  if(!gpuEventQuery(*done_compute)) { gpuEventSynchronize(*done_compute); }

  gpuMemcpyAsync<T>(dev_evl_sorted_h1b, host_evl_sorted_h1b, base_size_h1b, gpuMemcpyHostToDevice,
                    stream);
  gpuMemcpyAsync<T>(dev_evl_sorted_h2b, host_evl_sorted_h2b, base_size_h2b, gpuMemcpyHostToDevice,
                    stream);
  gpuMemcpyAsync<T>(dev_evl_sorted_h3b, host_evl_sorted_h3b, base_size_h3b, gpuMemcpyHostToDevice,
                    stream);
  gpuMemcpyAsync<T>(dev_evl_sorted_p4b, host_evl_sorted_p4b, base_size_p4b, gpuMemcpyHostToDevice,
                    stream);
  gpuMemcpyAsync<T>(dev_evl_sorted_p5b, host_evl_sorted_p5b, base_size_p5b, gpuMemcpyHostToDevice,
                    stream);
  gpuMemcpyAsync<T>(dev_evl_sorted_p6b, host_evl_sorted_p6b, base_size_p6b, gpuMemcpyHostToDevice,
                    stream);

  gpuMemcpyAsync<T>(df_dev_s1_t1_all, df_host_pinned_s1_t1, max_dim_s1_t1 * df_num_s1_enabled,
                    gpuMemcpyHostToDevice, stream);
  gpuMemcpyAsync<T>(df_dev_s1_v2_all, df_host_pinned_s1_v2, max_dim_s1_v2 * df_num_s1_enabled,
                    gpuMemcpyHostToDevice, stream);
  gpuMemcpyAsync<T>(df_dev_d1_t2_all, df_host_pinned_d1_t2, max_dim_d1_t2 * df_num_d1_enabled,
                    gpuMemcpyHostToDevice, stream);
  gpuMemcpyAsync<T>(df_dev_d1_v2_all, df_host_pinned_d1_v2, max_dim_d1_v2 * df_num_d1_enabled,
                    gpuMemcpyHostToDevice, stream);
  gpuMemcpyAsync<T>(df_dev_d2_t2_all, df_host_pinned_d2_t2, max_dim_d2_t2 * df_num_d2_enabled,
                    gpuMemcpyHostToDevice, stream);
  gpuMemcpyAsync<T>(df_dev_d2_v2_all, df_host_pinned_d2_v2, max_dim_d2_v2 * df_num_d2_enabled,
                    gpuMemcpyHostToDevice, stream);
#endif

  size_t num_blocks = CEIL(base_size_h3b, 4) * CEIL(base_size_h2b, 4) * CEIL(base_size_h1b, 4) *
                      CEIL(base_size_p6b, 4) * CEIL(base_size_p5b, 4) * CEIL(base_size_p4b, 4);

#ifdef OPT_KERNEL_TIMING
  //
  long double task_num_ops_s1    = 0;
  long double task_num_ops_d1    = 0;
  long double task_num_ops_d2    = 0;
  long double task_num_ops_total = 0;

  //
  helper_calculate_num_ops(noab, nvab, df_simple_s1_size, df_simple_d1_size, df_simple_d2_size,
                           df_simple_s1_exec, df_simple_d1_exec, df_simple_d2_exec, task_num_ops_s1,
                           task_num_ops_d1, task_num_ops_d2, total_num_ops_s1, total_num_ops_d1,
                           total_num_ops_d2);

  //
  task_num_ops_total = task_num_ops_s1 + task_num_ops_d1 + task_num_ops_d2;
#endif

#ifdef OPT_KERNEL_TIMING
  gpuEvent_t start_kernel_only, stop_kernel_only;

#if defined(USE_CUDA)
  CUDA_SAFE(cudaEventCreate(&start_kernel_only));
  CUDA_SAFE(cudaEventCreate(&stop_kernel_only));
  CUDA_SAFE(cudaEventRecord(start_kernel_only));
#elif defined(USE_HIP)
  HIP_SAFE(hipEventCreate(&start_kernel_only));
  HIP_SAFE(hipEventCreate(&stop_kernel_only));
  HIP_SAFE(hipEventRecord(start_kernel_only));
#endif

#endif // OPT_KERNEL_TIMING

#if defined(USE_DPCPP)
  int* s1_size = static_cast<int*>(memPool.allocate(sizeof(int) * (6)));
  int* s1_exec = static_cast<int*>(memPool.allocate(sizeof(int) * (9)));
  int* d1_size = static_cast<int*>(memPool.allocate(sizeof(int) * (7 * noab)));
  int* d1_exec = static_cast<int*>(memPool.allocate(sizeof(int) * (9 * noab)));
  int* d2_size = static_cast<int*>(memPool.allocate(sizeof(int) * (7 * nvab)));
  int* d2_exec = static_cast<int*>(memPool.allocate(sizeof(int) * (9 * nvab)));

  stream.memcpy(s1_size, df_simple_s1_size, sizeof(int) * (6) );
  stream.memcpy(s1_exec, df_simple_s1_exec, sizeof(int) * (9) );
  stream.memcpy(d1_size, df_simple_d1_size, sizeof(int) * (7 * noab) );
  stream.memcpy(d1_exec, df_simple_d1_exec, sizeof(int) * (9 * noab) );
  stream.memcpy(d2_size, df_simple_d2_size, sizeof(int) * (7 * nvab) );
  (*done_copy) = stream.memcpy(d2_exec, df_simple_d2_exec, sizeof(int) * (9 * nvab) );
#endif

#if defined(USE_DPCPP) || defined(USE_HIP) || (defined(USE_CUDA) && !defined(USE_NV_TC))
  fully_fused_ccsd_t_gpu(stream, num_blocks, k_range[t_h1b], k_range[t_h2b], k_range[t_h3b],
                         k_range[t_p4b], k_range[t_p5b], k_range[t_p6b],
                         //
                         df_dev_d1_t2_all, df_dev_d1_v2_all, df_dev_d2_t2_all, df_dev_d2_v2_all,
                         df_dev_s1_t1_all, df_dev_s1_v2_all,
                         //
                         //  for constant memory
                         //
                         df_simple_d1_size, df_simple_d1_exec, df_simple_d2_size, df_simple_d2_exec,
                         df_simple_s1_size, df_simple_s1_exec,
                         //
                         noab, max_dim_d1_t2, max_dim_d1_v2, nvab, max_dim_d2_t2, max_dim_d2_v2,
                         max_dim_s1_t1, max_dim_s1_v2,
                         //
                         factor,
                         //
                         dev_evl_sorted_h1b, dev_evl_sorted_h2b, dev_evl_sorted_h3b,
                         dev_evl_sorted_p4b, dev_evl_sorted_p5b, dev_evl_sorted_p6b,
                         //
                         dev_energies, done_copy
#if defined(USE_DPCPP)
                         , s1_size, s1_exec, d1_size, d1_exec, d2_size, d2_exec
#endif
                         );
#elif defined(USE_CUDA) && defined(USE_NV_TC)
  ccsd_t_fully_fused_nvidia_tc_fp64(stream, num_blocks, k_range[t_h3b], k_range[t_h2b],
                                    k_range[t_h1b], k_range[t_p6b], k_range[t_p5b], k_range[t_p4b],
                                    //
                                    df_dev_s1_t1_all, df_dev_s1_v2_all, df_dev_d1_t2_all,
                                    df_dev_d1_v2_all, df_dev_d2_t2_all, df_dev_d2_v2_all,
                                    //
                                    //  for constant memory
                                    //
                                    host_d1_size_h7b, host_d2_size_p7b, df_simple_s1_exec,
                                    df_simple_d1_exec, df_simple_d2_exec,
                                    //
                                    noab, nvab, max_dim_s1_t1, max_dim_s1_v2, max_dim_d1_t2,
                                    max_dim_d1_v2, max_dim_d2_t2, max_dim_d2_v2,
                                    //
                                    factor,
                                    //
                                    dev_evl_sorted_h1b, dev_evl_sorted_h2b, dev_evl_sorted_h3b,
                                    dev_evl_sorted_p4b, dev_evl_sorted_p5b, dev_evl_sorted_p6b,
                                    //
                                    dev_energies, done_copy);
#endif

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  gpuMemcpyAsync<T>(host_energies, dev_energies, num_blocks * 2, gpuMemcpyDeviceToHost, stream);
  reduceData->num_blocks    = num_blocks;
  reduceData->host_energies = host_energies;
  reduceData->result_energy = energy_l.data();
  reduceData->factor        = factor;
#ifdef USE_CUDA
  CUDA_SAFE(cudaLaunchHostFunc(stream, hostEnergyReduce, reduceData));
  CUDA_SAFE(cudaEventRecord(*done_compute, stream));
#elif defined(USE_HIP)
  HIP_SAFE(hipLaunchHostFunc(stream, hostEnergyReduce, reduceData));
  HIP_SAFE(hipEventRecord(*done_compute, stream));
#elif defined(USE_DPCPP)
  (*done_compute) = stream.submit(
    [&](sycl::handler& cgh) {
       cgh.host_task([=]() {
        hostEnergyReduce(reduceData);
        }); });
#endif

  //  free device mem back to pool
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_h1b), sizeof(T) * base_size_h1b);
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_h2b), sizeof(T) * base_size_h2b);
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_h3b), sizeof(T) * base_size_h3b);
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_p4b), sizeof(T) * base_size_p4b);
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_p5b), sizeof(T) * base_size_p5b);
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_p6b), sizeof(T) * base_size_p6b);
#if defined(USE_DPCPP)
  memPool.deallocate(static_cast<void*>(s1_size), sizeof(int) * (6));
  memPool.deallocate(static_cast<void*>(s1_exec), sizeof(int) * (9));
  memPool.deallocate(static_cast<void*>(d1_size), sizeof(int) * (7 * noab));
  memPool.deallocate(static_cast<void*>(d1_exec), sizeof(int) * (9 * noab));
  memPool.deallocate(static_cast<void*>(d2_size), sizeof(int) * (7 * nvab));
  memPool.deallocate(static_cast<void*>(d2_exec), sizeof(int) * (9 * nvab));
#endif
#endif // if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
}
