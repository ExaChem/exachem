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

// driver for the fully-fused kernel (FP64)
template<typename T>
void fully_fused_ccsd_t_gpu(gpuStream_t& stream_id, size_t num_blocks, size_t base_size_h1b,
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
                            T* partial_energies);

#if defined(USE_CUDA) && defined(USE_NV_TC)
// driver for fully-fused kernel for 3rd gen. tensor core (FP64)
template<typename T>
void ccsd_t_fully_fused_nvidia_tc_fp64(
  gpuStream_t& stream_id, size_t numBlks, size_t size_h3, size_t size_h2, size_t size_h1,
  size_t size_p6, size_t size_p5, size_t size_p4,
  //
  T* dev_s1_t1_all, T* dev_s1_v2_all, T* dev_d1_t2_all, T* dev_d1_v2_all, T* dev_d2_t2_all,
  T* dev_d2_v2_all,
  //
  int* host_size_d1_h7b, int* host_size_d2_p7b, int* host_exec_s1, int* host_exec_d1,
  int* host_exec_d2,
  //
  size_t size_noab, size_t size_nvab, size_t size_max_dim_s1_t1, size_t size_max_dim_s1_v2,
  size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2, size_t size_max_dim_d2_t2,
  size_t size_max_dim_d2_v2,
  //
  T factor, T* dev_evl_sorted_h1b, T* dev_evl_sorted_h2b, T* dev_evl_sorted_h3b,
  T* dev_evl_sorted_p4b, T* dev_evl_sorted_p5b, T* dev_evl_sorted_p6b, T* dev_final_energies);
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
  std::vector<T>& energy_l, LRUCache<Index, std::vector<T>>& cache_s1t,
  LRUCache<Index, std::vector<T>>& cache_s1v, LRUCache<Index, std::vector<T>>& cache_d1t,
  LRUCache<Index, std::vector<T>>& cache_d1v, LRUCache<Index, std::vector<T>>& cache_d2t,
  LRUCache<Index, std::vector<T>>& cache_d2v
#if defined(USE_DPCPP)
  ,
  gpuEvent_t& done_compute, std::vector<gpuEvent_t>& done_copy
#endif
) {
#ifdef OPT_KERNEL_TIMING
  long double total_num_ops_s1 = 0;
  long double total_num_ops_d1 = 0;
  long double total_num_ops_d2 = 0;
#endif

#ifdef OPT_ALL_TIMING
  gpuEvent_t start_init, stop_init;
  gpuEvent_t start_fused_kernel, stop_fused_kernel;
  gpuEvent_t start_pre_processing, stop_pre_processing;
  gpuEvent_t start_post_processing, stop_post_processing;
  gpuEvent_t start_collecting_data, stop_collecting_data;

  float time_ms_init            = 0.0;
  float time_ms_fused_kernel    = 0.0;
  float time_ms_pre_processing  = 0.0;
  float time_ms_post_processing = 0.0;
  float time_ms_collecting_data = 0.0;

#if defined(USE_CUDA)
  CUDA_SAFE(cudaEventCreate(&start_init));
  CUDA_SAFE(cudaEventCreate(&stop_init));
  CUDA_SAFE(cudaEventCreate(&start_fused_kernel));
  CUDA_SAFE(cudaEventCreate(&stop_fused_kernel));
  CUDA_SAFE(cudaEventCreate(&start_pre_processing));
  CUDA_SAFE(cudaEventCreate(&stop_pre_processing));
  CUDA_SAFE(cudaEventCreate(&start_post_processing));
  CUDA_SAFE(cudaEventCreate(&stop_post_processing));
  CUDA_SAFE(cudaEventCreate(&start_collecting_data));
  CUDA_SAFE(cudaEventCreate(&stop_collecting_data));
  CUDA_SAFE(cudaEventRecord(start_init));
#endif

#if defined(USE_HIP)
  HIP_SAFE(hipEventCreate(&start_init));
  HIP_SAFE(hipEventCreate(&stop_init));
  HIP_SAFE(hipEventCreate(&start_fused_kernel));
  HIP_SAFE(hipEventCreate(&stop_fused_kernel));
  HIP_SAFE(hipEventCreate(&start_pre_processing));
  HIP_SAFE(hipEventCreate(&stop_pre_processing));
  HIP_SAFE(hipEventCreate(&start_post_processing));
  HIP_SAFE(hipEventCreate(&stop_post_processing));
  HIP_SAFE(hipEventCreate(&start_collecting_data));
  HIP_SAFE(hipEventCreate(&stop_collecting_data));

  HIP_SAFE(hipEventRecord(start_init));
#endif

#endif // OPT_ALL_TIMING

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  // get GPU stream from pool
  auto&        pool   = tamm::GPUStreamPool::getInstance();
  gpuStream_t& stream = pool.getStream();

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

  //
  int df_num_s1_enabled, df_num_d1_enabled, df_num_d2_enabled;

  //
  size_t base_size_h1b = k_range[t_h1b];
  size_t base_size_h2b = k_range[t_h2b];
  size_t base_size_h3b = k_range[t_h3b];
  size_t base_size_p4b = k_range[t_p4b];
  size_t base_size_p5b = k_range[t_p5b];
  size_t base_size_p6b = k_range[t_p6b];

  //
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
#endif

#if defined(USE_CUDA)
#ifdef OPT_ALL_TIMING
  CUDA_SAFE(cudaEventRecord(stop_init));
  CUDA_SAFE(cudaEventSynchronize(stop_init));
  CUDA_SAFE(cudaEventRecord(start_collecting_data));
#endif
#endif

  // resets
  std::fill(df_simple_s1_exec, df_simple_s1_exec + (9), -1);
  std::fill(df_simple_d1_exec, df_simple_d1_exec + (9 * noab), -1);
  std::fill(df_simple_d2_exec, df_simple_d2_exec + (9 * nvab), -1);

  //
  ccsd_t_data_s1_new(is_restricted, noab, nvab, k_spin, d_t1, d_t2, d_v2, k_evl_sorted, k_range,
                     t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b,
                     //
                     size_T_s1_t1, size_T_s1_v2, df_simple_s1_size, df_simple_s1_exec,
                     df_host_pinned_s1_t1, df_host_pinned_s1_v2, &df_num_s1_enabled,
                     //
                     cache_s1t, cache_s1v);

  //
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

  //
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

#ifdef OPT_ALL_TIMING
#if defined(USE_CUDA)
  CUDA_SAFE(cudaEventRecord(stop_collecting_data));
  CUDA_SAFE(cudaEventSynchronize(stop_collecting_data));
  CUDA_SAFE(cudaEventRecord(start_pre_processing));
#elif defined(USE_HIP)
  HIP_SAFE(hipEventRecord(stop_collecting_data));
  HIP_SAFE(hipEventSynchronize(stop_collecting_data));
  HIP_SAFE(hipEventRecord(start_pre_processing));
#endif
#endif // OPT_ALL_TIMING

#if defined(USE_CUDA)
  // this is not pinned memory.
  CUDA_SAFE(cudaMemcpyAsync(dev_evl_sorted_h1b, host_evl_sorted_h1b, sizeof(T) * base_size_h1b,
                            cudaMemcpyHostToDevice, stream));
  CUDA_SAFE(cudaMemcpyAsync(dev_evl_sorted_h2b, host_evl_sorted_h2b, sizeof(T) * base_size_h2b,
                            cudaMemcpyHostToDevice, stream));
  CUDA_SAFE(cudaMemcpyAsync(dev_evl_sorted_h3b, host_evl_sorted_h3b, sizeof(T) * base_size_h3b,
                            cudaMemcpyHostToDevice, stream));
  CUDA_SAFE(cudaMemcpyAsync(dev_evl_sorted_p4b, host_evl_sorted_p4b, sizeof(T) * base_size_p4b,
                            cudaMemcpyHostToDevice, stream));
  CUDA_SAFE(cudaMemcpyAsync(dev_evl_sorted_p5b, host_evl_sorted_p5b, sizeof(T) * base_size_p5b,
                            cudaMemcpyHostToDevice, stream));
  CUDA_SAFE(cudaMemcpyAsync(dev_evl_sorted_p6b, host_evl_sorted_p6b, sizeof(T) * base_size_p6b,
                            cudaMemcpyHostToDevice, stream));

  //  new tensors
  CUDA_SAFE(cudaMemcpyAsync(df_dev_s1_t1_all, df_host_pinned_s1_t1,
                            sizeof(T) * (max_dim_s1_t1 * df_num_s1_enabled), cudaMemcpyHostToDevice,
                            stream));
  CUDA_SAFE(cudaMemcpyAsync(df_dev_s1_v2_all, df_host_pinned_s1_v2,
                            sizeof(T) * (max_dim_s1_v2 * df_num_s1_enabled), cudaMemcpyHostToDevice,
                            stream));
  CUDA_SAFE(cudaMemcpyAsync(df_dev_d1_t2_all, df_host_pinned_d1_t2,
                            sizeof(T) * (max_dim_d1_t2 * df_num_d1_enabled), cudaMemcpyHostToDevice,
                            stream));
  CUDA_SAFE(cudaMemcpyAsync(df_dev_d1_v2_all, df_host_pinned_d1_v2,
                            sizeof(T) * (max_dim_d1_v2 * df_num_d1_enabled), cudaMemcpyHostToDevice,
                            stream));
  CUDA_SAFE(cudaMemcpyAsync(df_dev_d2_t2_all, df_host_pinned_d2_t2,
                            sizeof(T) * (max_dim_d2_t2 * df_num_d2_enabled), cudaMemcpyHostToDevice,
                            stream));
  CUDA_SAFE(cudaMemcpyAsync(df_dev_d2_v2_all, df_host_pinned_d2_v2,
                            sizeof(T) * (max_dim_d2_v2 * df_num_d2_enabled), cudaMemcpyHostToDevice,
                            stream));
#elif defined(USE_HIP)
  // this is not pinned memory.
  HIP_SAFE(
    hipMemcpyHtoDAsync(dev_evl_sorted_h1b, host_evl_sorted_h1b, sizeof(T) * base_size_h1b, stream));
  HIP_SAFE(
    hipMemcpyHtoDAsync(dev_evl_sorted_h2b, host_evl_sorted_h2b, sizeof(T) * base_size_h2b, stream));
  HIP_SAFE(
    hipMemcpyHtoDAsync(dev_evl_sorted_h3b, host_evl_sorted_h3b, sizeof(T) * base_size_h3b, stream));
  HIP_SAFE(
    hipMemcpyHtoDAsync(dev_evl_sorted_p4b, host_evl_sorted_p4b, sizeof(T) * base_size_p4b, stream));
  HIP_SAFE(
    hipMemcpyHtoDAsync(dev_evl_sorted_p5b, host_evl_sorted_p5b, sizeof(T) * base_size_p5b, stream));
  HIP_SAFE(
    hipMemcpyHtoDAsync(dev_evl_sorted_p6b, host_evl_sorted_p6b, sizeof(T) * base_size_p6b, stream));

  //  new tensors
  HIP_SAFE(hipMemcpyHtoDAsync(df_dev_s1_t1_all, df_host_pinned_s1_t1,
                              sizeof(T) * (max_dim_s1_t1 * df_num_s1_enabled), stream));
  HIP_SAFE(hipMemcpyHtoDAsync(df_dev_s1_v2_all, df_host_pinned_s1_v2,
                              sizeof(T) * (max_dim_s1_v2 * df_num_s1_enabled), stream));
  HIP_SAFE(hipMemcpyHtoDAsync(df_dev_d1_t2_all, df_host_pinned_d1_t2,
                              sizeof(T) * (max_dim_d1_t2 * df_num_d1_enabled), stream));
  HIP_SAFE(hipMemcpyHtoDAsync(df_dev_d1_v2_all, df_host_pinned_d1_v2,
                              sizeof(T) * (max_dim_d1_v2 * df_num_d1_enabled), stream));
  HIP_SAFE(hipMemcpyHtoDAsync(df_dev_d2_t2_all, df_host_pinned_d2_t2,
                              sizeof(T) * (max_dim_d2_t2 * df_num_d2_enabled), stream));
  HIP_SAFE(hipMemcpyHtoDAsync(df_dev_d2_v2_all, df_host_pinned_d2_v2,
                              sizeof(T) * (max_dim_d2_v2 * df_num_d2_enabled), stream));
#elif defined(USE_DPCPP)
  // this is not pinned memory.
  done_copy[0] = stream.memcpy(dev_evl_sorted_h1b, host_evl_sorted_h1b, sizeof(T) * base_size_h1b);
  done_copy[1] = stream.memcpy(dev_evl_sorted_h2b, host_evl_sorted_h2b, sizeof(T) * base_size_h2b);
  done_copy[2] = stream.memcpy(dev_evl_sorted_h3b, host_evl_sorted_h3b, sizeof(T) * base_size_h3b);
  done_copy[3] = stream.memcpy(dev_evl_sorted_p4b, host_evl_sorted_p4b, sizeof(T) * base_size_p4b);
  done_copy[4] = stream.memcpy(dev_evl_sorted_p5b, host_evl_sorted_p5b, sizeof(T) * base_size_p5b);
  done_copy[5] = stream.memcpy(dev_evl_sorted_p6b, host_evl_sorted_p6b, sizeof(T) * base_size_p6b);

  //  new tensors
  done_copy[6]  = stream.memcpy(df_dev_s1_t1_all, df_host_pinned_s1_t1,
                                sizeof(T) * (max_dim_s1_t1 * df_num_s1_enabled));
  done_copy[7]  = stream.memcpy(df_dev_s1_v2_all, df_host_pinned_s1_v2,
                                sizeof(T) * (max_dim_s1_v2 * df_num_s1_enabled));
  done_copy[8]  = stream.memcpy(df_dev_d1_t2_all, df_host_pinned_d1_t2,
                                sizeof(T) * (max_dim_d1_t2 * df_num_d1_enabled));
  done_copy[9]  = stream.memcpy(df_dev_d1_v2_all, df_host_pinned_d1_v2,
                                sizeof(T) * (max_dim_d1_v2 * df_num_d1_enabled));
  done_copy[10] = stream.memcpy(df_dev_d2_t2_all, df_host_pinned_d2_t2,
                                sizeof(T) * (max_dim_d2_t2 * df_num_d2_enabled));
  done_copy[11] = stream.memcpy(df_dev_d2_v2_all, df_host_pinned_d2_v2,
                                sizeof(T) * (max_dim_d2_v2 * df_num_d2_enabled));
#endif

  //
#ifdef OPT_ALL_TIMING
#if defined(USE_CUDA)
  CUDA_SAFE(cudaEventRecord(stop_pre_processing));
  CUDA_SAFE(cudaEventSynchronize(stop_pre_processing));
  CUDA_SAFE(cudaEventRecord(start_fused_kernel));
#endif
#endif // OPT_ALL_TIMINGS
  //
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
                         dev_energies);
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
                                    dev_energies);
#endif

//
#ifdef OPT_KERNEL_TIMING
  CUDA_SAFE(cudaEventRecord(stop_kernel_only));
  CUDA_SAFE(cudaEventSynchronize(stop_kernel_only));

  float ms_time_kernel_only = 0.0;
  CUDA_SAFE(cudaEventElapsedTime(&ms_time_kernel_only, start_kernel_only, stop_kernel_only));
  if(rank == 0) {
    // printf ("[%s] s1: %lu, d1: %lu, d2: %lu >> total: %lu\n", __func__, task_num_ops_s1,
    // task_num_ops_d1, task_num_ops_d2, task_num_ops_total);
    printf("[ms_time_kernel_only] time: %f (ms) >> # of ops: %Lf >> %Lf GFLOPS\n",
           ms_time_kernel_only, task_num_ops_total,
           task_num_ops_total / (ms_time_kernel_only * 1000000));
  }
#endif

#if defined(USE_CUDA)
#ifdef OPT_ALL_TIMING
  CUDA_SAFE(cudaEventRecord(stop_fused_kernel));
  CUDA_SAFE(cudaEventSynchronize(stop_fused_kernel));
  CUDA_SAFE(cudaEventRecord(start_post_processing));
#endif
#endif

#if defined(USE_CUDA)
  CUDA_SAFE(cudaMemcpyAsync(host_energies, dev_energies, num_blocks * 2 * sizeof(T),
                            cudaMemcpyDeviceToHost, stream));
  CUDA_SAFE(cudaDeviceSynchronize());
#elif defined(USE_HIP)
  HIP_SAFE(hipMemcpyAsync(host_energies, dev_energies, num_blocks * 2 * sizeof(T),
                          hipMemcpyDeviceToHost, stream));
  HIP_SAFE(hipDeviceSynchronize());
#elif defined(USE_DPCPP)
  stream.memcpy(host_energies, dev_energies, num_blocks * 2 * sizeof(T), done_compute).wait();
#endif

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  T final_energy_1 = 0.0;
  T final_energy_2 = 0.0;
  for(size_t i = 0; i < num_blocks; i++) {
    final_energy_1 += host_energies[i];
    final_energy_2 += host_energies[i + num_blocks];
  }

  energy_l[0] += final_energy_1 * factor;
  energy_l[1] += final_energy_2 * factor;

  //  free device mem back to pool
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_h1b), sizeof(T) * base_size_h1b);
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_h2b), sizeof(T) * base_size_h2b);
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_h3b), sizeof(T) * base_size_h3b);
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_p4b), sizeof(T) * base_size_p4b);
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_p5b), sizeof(T) * base_size_p5b);
  memPool.deallocate(static_cast<void*>(dev_evl_sorted_p6b), sizeof(T) * base_size_p6b);
#endif

#ifdef OPT_ALL_TIMING
  CUDA_SAFE(cudaEventRecord(stop_post_processing));
  CUDA_SAFE(cudaEventSynchronize(stop_post_processing));

  CUDA_SAFE(cudaEventElapsedTime(&time_ms_init, start_init, stop_init));
  CUDA_SAFE(
    cudaEventElapsedTime(&time_ms_pre_processing, start_pre_processing, stop_pre_processing));
  CUDA_SAFE(cudaEventElapsedTime(&time_ms_fused_kernel, start_fused_kernel, stop_fused_kernel));
  CUDA_SAFE(
    cudaEventElapsedTime(&time_ms_collecting_data, start_collecting_data, stop_collecting_data));
  CUDA_SAFE(
    cudaEventElapsedTime(&time_ms_post_processing, start_post_processing, stop_post_processing));

  // if (rank == 0)
  // {
  //   int tmp_dev_id = 0;
  //   cudaGetDevice(&tmp_dev_id);
  //   printf ("[%s] performed by rank: %d with dev-id: %d ----------------------\n", __func__,
  //   rank, tmp_dev_id); printf ("[%s][df-based] time-init             : %f (ms)\n", __func__,
  //   time_ms_init); printf ("[%s][df-based] time-pre-processing   : %f (ms)\n", __func__,
  //   time_ms_pre_processing); printf ("[%s][df-based] time-fused-kernel     : %f (ms)\n",
  //   __func__, time_ms_fused_kernel); printf ("[%s][df-based] time-collecting-data  : %f (ms)\n",
  //   __func__, time_ms_collecting_data); printf ("[%s][df-based] time-post-processing  : %f
  //   (ms)\n", __func__, time_ms_post_processing); printf ("[%s]
  //   ------------------------------------------------------------\n", __func__);
  // }
  double task_memcpy_time = time_ms_init + time_ms_pre_processing + time_ms_post_processing;
  double total_task_time  = task_memcpy_time + time_ms_fused_kernel + time_ms_collecting_data;
  // 6dtaskid-142563,kernel,memcpy,data,total
  cout << std::fixed << std::setprecision(2) << t_h1b << "-" << t_p4b << "-" << t_h2b << "-"
       << t_p5b << "-" << t_p6b << "-" << t_h3b << ", " << time_ms_fused_kernel / 1e3 << ","
       << task_memcpy_time / 1e3 << "," << time_ms_collecting_data / 1e3 << ","
       << total_task_time / 1e3 << endl;
#endif
}
