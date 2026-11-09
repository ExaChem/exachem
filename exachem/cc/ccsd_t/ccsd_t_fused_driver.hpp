/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "exachem/cc/ccsd_t/ccsd_t_all_fused.hpp"
#else
#include "exachem/cc/ccsd_t/ccsd_t_all_fused_cpu.hpp"
#endif
#include "exachem/cc/ccsd_t/ccsd_t_common.hpp"

#include <atomic>
#include <cstddef>
#include <deque>
#include <span>
#include <string>
#include <thread>

namespace exachem::cc::ccsd_t {

void ccsd_t_driver(ExecutionContext& ec, ChemEnv& chem_env);
}

/**
 *  to check if target NVIDIA GPUs can support the fully-fused kernel
 *  based on 3rd gen. tensor cores or not.
 *  - requirements: (1) arch >= 80 and (2) CUDA >= 12.0
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

  int driver_major = version / 1000;
  int driver_minor = (version - (driver_major * 1000)) / 10;
  if(r0)
    printf("Given info.: Compatibility = %d.%d, CUDA Version = %d.%d\n", dP.major, dP.minor,
           driver_major, driver_minor);

  if(dP.major >= 8 && driver_major >= 12) { return 1; }
  else { return -1; }
}
#endif

/**
 * A copy of one rank's [T] and (T) energies (energy_l) at (T) checkpoint k, taken once every task
 * this rank ran below the checkpoint has added its energies. See the (T) checkpoint section of
 * CCSD_T_Fused_Driver::execute for how checkpoints work.
 *
 * On a GPU, a task's energies are added to energy_l by hostEnergyReduce, a host function queued
 * on the GPU stream after the task's kernel. The copy is queued on the same stream after them
 * (ckpt_snapshot_energies), so the rank never waits for the GPU at a checkpoint. Waiting for the
 * GPU there (gpuDeviceSynchronize) stops the CPU from preparing the next task while the GPU
 * finishes the current one.
 *
 * ready is set by the thread that runs the stream's host functions and read by the rank's main
 * thread, so it is stored with release and loaded with acquire ordering: once ready is seen, the
 * copied energies are too.
 */
template<typename T>
struct CkptSnapshot {
  size_t            k        = 0;
  const T*          energy_l = nullptr;
  T                 energy[2]{};
  std::atomic<bool> ready{false};
};

/**
 * Copies energy_l into a CkptSnapshot and marks it ready. Queued as a host function on the GPU
 * stream, or called directly on the CPU, where the energies are already up to date.
 */
template<typename T>
inline void ckpt_snapshot_energies(void* data) {
  auto* s      = static_cast<CkptSnapshot<T>*>(data);
  s->energy[0] = s->energy_l[0];
  s->energy[1] = s->energy_l[1];
  s->ready.store(true, std::memory_order_release);
}

template<typename T>
class CCSD_T_Fused_Driver {
public:
  CCSD_T_Fused_Driver() = default;

  virtual ~CCSD_T_Fused_Driver() = default;

  CCSD_T_Fused_Driver(const CCSD_T_Fused_Driver&)            = default;
  CCSD_T_Fused_Driver& operator=(const CCSD_T_Fused_Driver&) = default;
  CCSD_T_Fused_Driver(CCSD_T_Fused_Driver&&)                 = default;
  CCSD_T_Fused_Driver& operator=(CCSD_T_Fused_Driver&&)      = default;

  virtual std::tuple<T, T, double, double, size_t>
  execute(ChemEnv& chem_env, ExecutionContext& ec, std::vector<int>& k_spin,
          const TiledIndexSpace& MO, Tensor<T>& d_t1, Tensor<T>& d_t2,
          exachem::cholesky_2e::V2Tensors<T>& d_v2, std::vector<T>& k_evl_sorted, T hf_ccsd_energy,
          bool is_restricted, LRUCache<Index, std::vector<T>>& cache_s1t,
          LRUCache<Index, std::vector<T>>& cache_s1v, LRUCache<Index, std::vector<T>>& cache_d1t,
          LRUCache<Index, std::vector<T>>& cache_d1v, LRUCache<Index, std::vector<T>>& cache_d2t,
          LRUCache<Index, std::vector<T>>& cache_d2v, bool seq_h3b = false,
          bool tilesize_opt = true);

  virtual void calculate_performance_ops(ChemEnv& chem_env, ExecutionContext& ec,
                                         std::vector<int>& k_spin, const TiledIndexSpace& MO,
                                         std::vector<T>& k_evl_sorted, double hf_ccsd_energy,
                                         bool is_restricted, long double& total_num_ops,
                                         bool seq_h3b = false, size_t start_task = 0);
};

template<typename T>
std::tuple<T, T, double, double, size_t> CCSD_T_Fused_Driver<T>::execute(
  ChemEnv& chem_env, ExecutionContext& ec, std::vector<int>& k_spin, const TiledIndexSpace& MO,
  Tensor<T>& d_t1, Tensor<T>& d_t2, exachem::cholesky_2e::V2Tensors<T>& d_v2,
  std::vector<T>& k_evl_sorted, T hf_ccsd_energy, bool is_restricted,
  LRUCache<Index, std::vector<T>>& cache_s1t, LRUCache<Index, std::vector<T>>& cache_s1v,
  LRUCache<Index, std::vector<T>>& cache_d1t, LRUCache<Index, std::vector<T>>& cache_d1v,
  LRUCache<Index, std::vector<T>>& cache_d2t, LRUCache<Index, std::vector<T>>& cache_d2v,
  bool seq_h3b, bool tilesize_opt) {
  auto rank  = ec.pg().rank().value();
  bool rank0 = rank == 0;

#if defined(USE_CUDA)
// int opt_CUDA_TC = checkCudaKernelCompatible(rank0);
#if defined(USE_NV_TC)
  if(rank0) cout << "Using FP64 Tensor Cores" << endl;
#endif
#endif

  Index noab = MO("occ").num_tiles();
  Index nvab = MO("virt").num_tiles();

  Index noa = MO("occ_alpha").num_tiles();
  Index nva = MO("virt_alpha").num_tiles();

  auto                mo_tiles = MO.input_tile_sizes();
  std::vector<size_t> k_range;
  std::vector<size_t> k_offset;
  k_range.reserve(mo_tiles.size());
  k_offset.reserve(mo_tiles.size());
  size_t sum = 0;
  for(auto x: mo_tiles) {
    k_range.push_back(x);
    k_offset.push_back(sum);
    sum += x;
  }

  if(rank0) {
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

  std::unique_ptr<AtomicCounter> ac = std::make_unique<AtomicCounterGA>(ec.pg(), 1);

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  size_t max_pdim = 0;
  size_t max_hdim = 0;
  for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++)
    max_pdim = std::max(max_pdim, k_range[t_p4b]);
  for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) max_hdim = std::max(max_hdim, k_range[t_h1b]);

  size_t max_d1_kernels_pertask = 9 * noab;
  size_t max_d2_kernels_pertask = 9 * nvab;
  if(tilesize_opt) {
    // a task loops over the tiles of one spin, and the beta space can have more tiles than alpha
    max_d1_kernels_pertask = 9 * std::max(noa, noab - noa);
    max_d2_kernels_pertask = 9 * std::max(nva, nvab - nva);
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

  // Host-side integer metadata buffers. RAII via std::vector; the raw int* aliases
  // below preserve the existing call sites that expect a plain pointer.
  std::vector<int> df_simple_s1_size_v(6);
  std::vector<int> df_simple_s1_exec_v(9);
  std::vector<int> df_simple_d1_size_v(7 * noab);
  std::vector<int> df_simple_d1_exec_v(9 * noab);
  std::vector<int> df_simple_d2_size_v(7 * nvab);
  std::vector<int> df_simple_d2_exec_v(9 * nvab);
  std::vector<int> host_d1_size_v(noab);
  std::vector<int> host_d2_size_v(nvab);

  int* df_simple_s1_size = df_simple_s1_size_v.data();
  int* df_simple_s1_exec = df_simple_s1_exec_v.data();
  int* df_simple_d1_size = df_simple_d1_size_v.data();
  int* df_simple_d1_exec = df_simple_d1_exec_v.data();
  int* df_simple_d2_size = df_simple_d2_size_v.data();
  int* df_simple_d2_exec = df_simple_d2_exec_v.data();
  int* host_d1_size      = host_d1_size_v.data();
  int* host_d2_size      = host_d2_size_v.data();

  size_t max_num_blocks = chem_env.ioptions.ccsd_options.ccsdt_tilesize;
  max_num_blocks        = std::ceil((max_num_blocks + 4 - 1) / 4.0);

  // Integer arithmetic, not std::pow: std::pow returns a double, which is neither exact nor
  // guaranteed to round the same way at the matching deallocation -- and this value sizes
  // real allocations on both the GPU and CPU paths.
  size_t const num_energy_elems = max_num_blocks * max_num_blocks * max_num_blocks *
                                  max_num_blocks * max_num_blocks * max_num_blocks * 2;

  // Neither tamm::getPinnedMem nor operator new[](std::nothrow) throws on failure; both
  // return null. Unchecked, the null reaches a kernel and faults far from here.
  auto check_host_alloc = [](const void* p, std::size_t nelems, const char* what) {
    if(p == nullptr)
      tamm::tamm_terminate("[CCSD(T)] host allocation failed for " + std::string{what} + " (" +
                           std::to_string(nelems * sizeof(T)) + " bytes)");
  };

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  // the fused GPU kernels keep per-task tables for at most MAX_NOAB / MAX_NVAB tiles
  if(noab > MAX_NOAB || nvab > MAX_NVAB)
    tamm::tamm_terminate("[CCSD(T)] noab = " + std::to_string(noab) +
                         ", nvab = " + std::to_string(nvab) +
                         " exceed the GPU kernel limits (MAX_NOAB = " + std::to_string(MAX_NOAB) +
                         ", MAX_NVAB = " + std::to_string(MAX_NVAB) + ");");

  // Spans own these pool blocks; .data() feeds the kernel APIs below. The span carries its
  // length, so each deallocate matches its allocation by construction.
  auto&        memDevPool        = RMMMemoryManager::getInstance().getDeviceMemoryPool();
  std::span<T> df_dev_s1_t1_span = memDevPool.template allocate_span<T>(size_T_s1_t1);
  std::span<T> df_dev_s1_v2_span = memDevPool.template allocate_span<T>(size_T_s1_v2);
  std::span<T> df_dev_d1_t2_span = memDevPool.template allocate_span<T>(size_T_d1_t2);
  std::span<T> df_dev_d1_v2_span = memDevPool.template allocate_span<T>(size_T_d1_v2);
  std::span<T> df_dev_d2_t2_span = memDevPool.template allocate_span<T>(size_T_d2_t2);
  std::span<T> df_dev_d2_v2_span = memDevPool.template allocate_span<T>(size_T_d2_v2);
  T*           df_dev_s1_t1_all  = df_dev_s1_t1_span.data();
  T*           df_dev_s1_v2_all  = df_dev_s1_v2_span.data();
  T*           df_dev_d1_t2_all  = df_dev_d1_t2_span.data();
  T*           df_dev_d1_v2_all  = df_dev_d1_v2_span.data();
  T*           df_dev_d2_t2_all  = df_dev_d2_t2_span.data();
  T*           df_dev_d2_v2_all  = df_dev_d2_v2_span.data();

  df_host_pinned_s1_t1 = static_cast<T*>(tamm::getPinnedMem(sizeof(T) * size_T_s1_t1));
  df_host_pinned_s1_v2 = static_cast<T*>(tamm::getPinnedMem(sizeof(T) * size_T_s1_v2));
  df_host_pinned_d1_t2 = static_cast<T*>(tamm::getPinnedMem(sizeof(T) * size_T_d1_t2));
  df_host_pinned_d1_v2 = static_cast<T*>(tamm::getPinnedMem(sizeof(T) * size_T_d1_v2));
  df_host_pinned_d2_t2 = static_cast<T*>(tamm::getPinnedMem(sizeof(T) * size_T_d2_t2));
  df_host_pinned_d2_v2 = static_cast<T*>(tamm::getPinnedMem(sizeof(T) * size_T_d2_v2));

  T* df_host_energies = static_cast<T*>(tamm::getPinnedMem(sizeof(T) * num_energy_elems));

  check_host_alloc(df_host_pinned_s1_t1, size_T_s1_t1, "s1_t1");
  check_host_alloc(df_host_pinned_s1_v2, size_T_s1_v2, "s1_v2");
  check_host_alloc(df_host_pinned_d1_t2, size_T_d1_t2, "d1_t2");
  check_host_alloc(df_host_pinned_d1_v2, size_T_d1_v2, "d1_v2");
  check_host_alloc(df_host_pinned_d2_t2, size_T_d2_t2, "d2_t2");
  check_host_alloc(df_host_pinned_d2_v2, size_T_d2_v2, "d2_v2");
  check_host_alloc(df_host_energies, num_energy_elems, "energies");

#else // cpu
  df_host_pinned_s1_t1 = static_cast<T*>(operator new[](size_T_s1_t1 * sizeof(T), std::nothrow));
  df_host_pinned_s1_v2 = static_cast<T*>(operator new[](size_T_s1_v2 * sizeof(T), std::nothrow));
  df_host_pinned_d1_t2 = static_cast<T*>(operator new[](size_T_d1_t2 * sizeof(T), std::nothrow));
  df_host_pinned_d1_v2 = static_cast<T*>(operator new[](size_T_d1_v2 * sizeof(T), std::nothrow));
  df_host_pinned_d2_t2 = static_cast<T*>(operator new[](size_T_d2_t2 * sizeof(T), std::nothrow));
  df_host_pinned_d2_v2 = static_cast<T*>(operator new[](size_T_d2_v2 * sizeof(T), std::nothrow));
  T* df_host_energies = static_cast<T*>(operator new[](num_energy_elems * sizeof(T), std::nothrow));

  check_host_alloc(df_host_pinned_s1_t1, size_T_s1_t1, "s1_t1");
  check_host_alloc(df_host_pinned_s1_v2, size_T_s1_v2, "s1_v2");
  check_host_alloc(df_host_pinned_d1_t2, size_T_d1_t2, "d1_t2");
  check_host_alloc(df_host_pinned_d1_v2, size_T_d1_v2, "d1_v2");
  check_host_alloc(df_host_pinned_d2_t2, size_T_d2_t2, "d2_t2");
  check_host_alloc(df_host_pinned_d2_v2, size_T_d2_v2, "d2_v2");
  check_host_alloc(df_host_energies, num_energy_elems, "energies");
#endif

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  std::span<T> df_dev_energies_span = memDevPool.template allocate_span<T>(num_energy_elems);
  T*           df_dev_energies      = df_dev_energies_span.data();
#endif

  // int num_task = 0;
  //   if(!seq_h3b) {
  //     if(rank == 0) {
  //       std::cout << "456123 parallel 6d loop variant" << std::endl << std::endl;
  //       // std::cout << "tile142563,kernel,memcpy,data,total" << std::endl;
  //     }
  //     for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
  //       for(size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
  //         for(size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
  //           for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) { //
  //             for(size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
  //               for(size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
  //                 if((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
  //                    (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
  //                   if((!is_restricted) || (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
  //                                           k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8)
  //                                           {
  //                     if(next == taskcount) {
  //                       //
  //                       T factor = 1.0;
  //                       if(is_restricted) factor = 2.0;
  //                       if((t_p4b == t_p5b) && (t_p5b == t_p6b)) { factor /= 6.0; }
  //                       else if((t_p4b == t_p5b) || (t_p5b == t_p6b)) { factor /= 2.0; }

  //                       if((t_h1b == t_h2b) && (t_h2b == t_h3b)) { factor /= 6.0; }
  //                       else if((t_h1b == t_h2b) || (t_h2b == t_h3b)) { factor /= 2.0; }

  //                       num_task++;

  // #if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  //                       ccsd_t_fully_fused_none_df_none_task<T>(
  //                         is_restricted, noab, nvab, rank, k_spin, k_range, k_offset, d_t1, d_t2,
  //                         d_v2, k_evl_sorted,
  //                         //
  //                         df_host_pinned_s1_t1, df_host_pinned_s1_v2, df_host_pinned_d1_t2,
  //                         df_host_pinned_d1_v2, df_host_pinned_d2_t2, df_host_pinned_d2_v2,
  //                         df_host_energies,
  //                         //
  //                         //
  //                         //
  //                         host_d1_size, host_d2_size,
  //                         //
  //                         df_simple_s1_size, df_simple_d1_size, df_simple_d2_size,
  //                         df_simple_s1_exec, df_simple_d1_exec, df_simple_d2_exec,
  //                         //
  //                         df_dev_s1_t1_all, df_dev_s1_v2_all, df_dev_d1_t2_all, df_dev_d1_v2_all,
  //                         df_dev_d2_t2_all, df_dev_d2_v2_all, df_dev_energies,
  //                         //
  //                         t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor, taskcount,
  //                         max_d1_kernels_pertask, max_d2_kernels_pertask,
  //                         //
  //                         size_T_s1_t1, size_T_s1_v2, size_T_d1_t2, size_T_d1_v2, size_T_d2_t2,
  //                         size_T_d2_v2,
  //                         //
  //                         energy_l,
  // #if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  //                         reduceData.get(),
  // #endif
  //                         cache_s1t, cache_s1v, cache_d1t, cache_d1v, cache_d2t, cache_d2v,
  //                         //
  //                         done_compute, done_copy);
  // #else
  //                       total_fused_ccsd_t_cpu<T>(
  //                         is_restricted, noab, nvab, rank, k_spin, k_range, k_offset, d_t1, d_t2,
  //                         d_v2, k_evl_sorted,
  //                         //
  //                         df_host_pinned_s1_t1, df_host_pinned_s1_v2, df_host_pinned_d1_t2,
  //                         df_host_pinned_d1_v2, df_host_pinned_d2_t2, df_host_pinned_d2_v2,
  //                         df_host_energies, host_d1_size, host_d2_size,
  //                         //
  //                         df_simple_s1_size, df_simple_d1_size, df_simple_d2_size,
  //                         df_simple_s1_exec, df_simple_d1_exec, df_simple_d2_exec,
  //                         //
  //                         t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor, taskcount,
  //                         max_d1_kernels_pertask, max_d2_kernels_pertask,
  //                         //
  //                         size_T_s1_t1, size_T_s1_v2, size_T_d1_t2, size_T_d1_v2, size_T_d2_t2,
  //                         size_T_d2_v2,
  //                         //
  //                         energy_l, cache_s1t, cache_s1v, cache_d1t, cache_d1v, cache_d2t,
  //                         cache_d2v);
  // #endif

  //                       next = ac->fetch_add(0, 1);
  //                     }
  //                     taskcount++;
  //                   }
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }      // parallel h3b loop
  //   else { // seq h3b loop

  size_t total_tasks = 0;
  for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
    for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
      for(size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
        for(size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
          for(size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) { total_tasks++; }
        }
      }
    }
  }

  if(rank == 0) {
    // std::cout << "14256-seq3 loop variant" << std::endl << std::endl;
    std::cout << std::endl << "Total number of tasks = " << total_tasks << std::endl;
  }

  // Restarting (T) from a checkpoint:
  //  - writet: resume from the saved checkpoint, and keep writing new ones.
  //  - readt:  resume from the saved checkpoint, but do not write new ones.
  //  - skip_ccsd turns both off. It is a mode for timing (T) and has no real CCSD amplitudes.
  const CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;
  const bool         skip_ccsd    = ccsd_options.skip_ccsd;
  const bool         write_ckpt   = ccsd_options.writet && !skip_ccsd;
  const bool         read_ckpt    = (ccsd_options.readt || ccsd_options.writet) && !skip_ccsd;
  const double       threshold    = ccsd_options.threshold;
  const int          ts_triples   = chem_env.is_context.mso_tilesize_triples;

  if(rank0 && skip_ccsd && (ccsd_options.readt || ccsd_options.writet))
    std::cout << "--- (T) Checkpointing disabled with skip_ccsd ---" << std::endl;

  size_t start_task = 0;
  if(read_ckpt) {
    if(chem_env.run_context.contains("ccsd_t")) {
      const auto& ckpt = chem_env.run_context["ccsd_t"];
      // A task number only means the same work with the same tiling. A checkpoint from a
      // different tiling cannot be used, so stop and tell the user how to fix it.
      if(ckpt["total_tasks"].get<size_t>() != total_tasks || ckpt["noab"].get<Index>() != noab ||
         ckpt["nvab"].get<Index>() != nvab ||
         ckpt["mso_tilesize_triples"].get<int>() != ts_triples) {
        std::string err_msg = "\n[ERROR] Restarting a (T) calculation requires the same tiling\n";
        err_msg += "  - (T) checkpoint: mso_tilesize_triples = " +
                   std::to_string(ckpt["mso_tilesize_triples"].get<int>()) +
                   ", total tasks = " + std::to_string(ckpt["total_tasks"].get<size_t>()) + "\n";
        err_msg += "  - current run:    mso_tilesize_triples = " + std::to_string(ts_triples) +
                   ", total tasks = " + std::to_string(total_tasks) + "\n";
        err_msg += "  Restore the original tilesize, or remove the \"ccsd_t\" entry from the "
                   ".runcontext.json file to restart (T) from the first task";
        tamm_terminate(err_msg);
      }
      // A different CC threshold gives different CCSD amplitudes, so the saved energies do not
      // belong to this run. This is a deliberate rerun, not a mistake: drop the checkpoint and
      // start (T) from the first task.
      // Only the threshold is checked, not the amplitudes themselves. This is intentional.
      // A CCSD restart runs one more iteration (unless noscf), so a resumed (T) can mix
      // energies from slightly different amplitudes. In our tests, the difference is
      // typically ~1e-11 hartree with a CCSD threshold of 1e-6.
      const double ckpt_threshold = ckpt["threshold"].get<double>();
      if(ckpt_threshold != threshold) {
        if(rank0) {
          const auto flags = std::cout.flags();
          const auto prec  = std::cout.precision();
          std::cout << "--- (T) Checkpoint discarded: CC threshold changed (" << std::scientific
                    << std::setprecision(2) << ckpt_threshold << " -> " << threshold
                    << "), starting from the first task ---" << std::endl;
          std::cout.flags(flags);
          std::cout.precision(prec);
        }
      }
      else {
        start_task = ckpt["tasks_done"].get<size_t>();
        // Each rank sums the energies of only the tasks it runs; all ranks' sums are added up
        // on rank 0 after the loop. The tasks done before the restart are not run again, so
        // their saved total is added once, by starting rank 0 from it (the other ranks start
        // from zero).
        if(rank0) {
          energy_l[0] = ckpt["energy1"].get<T>();
          energy_l[1] = ckpt["energy2"].get<T>();
          if(start_task == total_tasks)
            std::cout << "--- (T) Checkpoint is complete: " << total_tasks << "/" << total_tasks
                      << " tasks already completed, using the saved energies ---" << std::endl;
          else
            std::cout << "--- (T) Resuming from checkpoint: " << start_task << "/" << total_tasks
                      << " tasks already completed ---" << std::endl;
        }
      }
    }
    else if(rank0)
      std::cout << "--- (T) No checkpoint found, starting from the first task ---" << std::endl;
  }

  // progress is reported every 20% of the tasks, or at every checkpoint with writet
  const size_t progress_step = 20;
  const int    task_width    = std::to_string(total_tasks).size();

  auto print_progress = [&](size_t tasks_done) {
    const size_t percentage = (tasks_done * 100) / total_tasks;
    const double elapsed    = std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::high_resolution_clock::now() - cc_t1)
                             .count();
    const auto flags = std::cout.flags();
    const auto prec  = std::cout.precision();
    std::cout << std::right << "Progress: " << std::setw(task_width) << tasks_done << "/"
              << total_tasks << " (" << std::setw(3) << percentage
              << "%) | Elapsed time: " << std::fixed << std::setprecision(1) << std::setw(8)
              << elapsed << " sec" << std::endl;
    std::cout.flags(flags);
    std::cout.precision(prec);
  };

  /*
   * (T) checkpoints (writet)
   *
   * Checkpoint k (1..20) is at task number total_tasks * k / 20, every 5% of the tasks. Every rank
   * walks every task number in the same order and runs the tasks it claimed in that order, so
   * once all ranks have passed a checkpoint's task number, every task below it is done. A
   * checkpoint saves that task number and the [T] and (T) energies summed over those tasks, and a
   * restart hands out tasks from there.
   *
   * How a checkpoint is made, without any rank waiting for another:
   *   1. When a rank passes checkpoint k, it takes a snapshot of its energies
   * (snapshot_checkpoint). On a GPU the snapshot becomes ready later, once the GPU has reached it.
   *   2. When the snapshot is ready, the rank adds it to slot k of a small GA, the two elements
   *      2(k-1) and 2(k-1)+1, and increments counter k-1 (add_to_checkpoints).
   *   3. Rank 0 writes checkpoint k to the run context once counter k-1 equals the number of
   *      ranks, i.e. slot k holds every rank's energies (save_checkpoints).
   *   4. After the task loop, every rank adds to the last checkpoint, at total_tasks, and rank 0
   *      keeps checking until it has written it.
   *
   * A collective (a reduce or a barrier) is not used: it would make every rank wait at each
   * checkpoint for the slowest task below it, and single tasks can run for a large part of (T).
   * A checkpoint is still only written once that slowest task finishes, so on systems with a few
   * long tasks the checkpoints can be written late and several at once.
   */
  const size_t nckpts          = 20;
  auto         ckpt_task       = [&](size_t k) { return total_tasks * k / nckpts; };
  auto         next_ckpt_after = [&](size_t task) {
    size_t k = 1;
    while(k < nckpts && ckpt_task(k) <= task) k++;
    return k;
  };

  // ckpt_ga holds the 20 slots and ckpt_count the 20 counters, both created once per (T) run
  const auto                       nranks         = ec.pg().size().value();
  int                              ckpt_ga        = 0;
  std::unique_ptr<AtomicCounterGA> ckpt_count     = nullptr;
  std::vector<size_t>              ckpts_added    = {}; // the checkpoints this rank added to
  size_t                           ckpts_saved    = 0;  // rank 0: the first of them not written
  double                           ckpt_wait_time = 0;  // subtracted from the (T) work time
  if(write_ckpt) {
    int64_t ckpt_size   = 2 * nckpts;
    char    ckpt_name[] = "ccsd_t_ckpt";
    ckpt_ga = NGA_Create_config64(to_ga_eltype(tensor_element_type<T>()), 1, &ckpt_size, ckpt_name,
                                  nullptr, ec.pg().ga_pg());
    GA_Zero(ckpt_ga);
    ckpt_count = std::make_unique<AtomicCounterGA>(ec.pg(), nckpts);
    ckpt_count->allocate(0);
  }

  // The host function queued for a snapshot keeps a pointer to it, so snapshots must never move:
  // a deque keeps its elements in place as more are added, a vector would not.
  std::deque<CkptSnapshot<T>> snapshots;
  size_t                      snapshots_added = 0; // the first snapshot not yet added to the GA

  /**
   * Step 1: takes this rank's snapshot for checkpoint k. On a GPU, ckpt_snapshot_energies is
   * queued on the same stream as the hostEnergyReduce calls, so it runs after the energies of
   * every task this rank launched before the checkpoint have been added. On the CPU the snapshot
   * is taken right away.
   */
  auto snapshot_checkpoint = [&](size_t k) {
    CkptSnapshot<T>& snap = snapshots.emplace_back();
    snap.k                = k;
    snap.energy_l         = energy_l.data();
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    gpuStream_t& stream = tamm::GPUStreamPool::getInstance().getStream();
#endif
#if defined(USE_CUDA)
    CUDA_SAFE(cudaLaunchHostFunc(stream.first, ckpt_snapshot_energies<T>, &snap));
#elif defined(USE_HIP)
    HIP_SAFE(hipLaunchHostFunc(stream.first, ckpt_snapshot_energies<T>, &snap));
#elif defined(USE_DPCPP)
    CkptSnapshot<T>* snap_ptr = &snap;
    stream.first.submit(
      [&](sycl::handler& cgh) { cgh.host_task([=]() { ckpt_snapshot_energies<T>(snap_ptr); }); });
#else
    ckpt_snapshot_energies<T>(&snap);
#endif
  };

  /**
   * Step 2: adds this rank's snapshots that are ready to their checkpoints, in checkpoint order,
   * and returns true if any were added. For checkpoint k:
   *  - NGA_Acc64 adds the snapshot energies to slot k, which holds the energies summed over the
   *    ranks that have added theirs so far.
   *  - GA_Init_fence starts tracking the ranks the accumulate is sent to, and GA_Fence waits until
   *    it has been applied on them: an accumulate can complete after it returns, and rank 0 must
   *    not read slot k before it holds every rank's energies.
   *  - fetch_add then increments counter k-1, the number of ranks that have added to slot k.
   */
  auto add_to_checkpoints = [&]() {
    bool added = false;
    while(snapshots_added < snapshots.size() &&
          snapshots[snapshots_added].ready.load(std::memory_order_acquire)) {
      const auto       ckpt_start = std::chrono::high_resolution_clock::now();
      CkptSnapshot<T>& snap       = snapshots[snapshots_added];
      int64_t          lo = 2 * (snap.k - 1), hi = lo + 1, ld = 1;
      T                alpha = 1.0;
      GA_Init_fence();
      NGA_Acc64(ckpt_ga, &lo, &hi, snap.energy, &ld, &alpha);
      GA_Fence();
      ckpt_count->fetch_add(snap.k - 1, 1);
      ckpts_added.push_back(snap.k);
      snapshots_added++;
      added = true;
      ckpt_wait_time += std::chrono::duration_cast<std::chrono::duration<double>>(
                          std::chrono::high_resolution_clock::now() - ckpt_start)
                          .count();
    }
    return added;
  };

  /**
   * Step 3, rank 0 only: writes the checkpoints that every rank has added its energies to, in
   * checkpoint order. Only the checkpoints rank 0 has added to can be complete, so it goes
   * through those. Each rank adds to checkpoints in order, so checkpoint k+1 can only be complete
   * once checkpoint k is, and it stops at the first one that is not. For checkpoint k:
   *  - fetch_add with an increment of 0 reads counter k-1 without changing it.
   *  - Once the counter equals the number of ranks, NGA_Get64 reads the summed energies from
   *    slot k, and the "ccsd_t" entry of the run context json is updated and written, with the
   * number of tasks done, the energies, and the tiling and threshold a restart is checked against.
   * It is called whenever rank 0 adds to a checkpoint, after each of rank 0's tasks, and at the
   * end until the last checkpoint is written.
   */
  auto save_checkpoints = [&]() {
    const auto ckpt_start = std::chrono::high_resolution_clock::now();
    while(ckpts_saved < ckpts_added.size()) {
      const size_t k = ckpts_added[ckpts_saved];
      if(ckpt_count->fetch_add(k - 1, 0) < nranks) break;
      int64_t lo = 2 * (k - 1), hi = lo + 1, ld = 1;
      T       energies[2];
      NGA_Get64(ckpt_ga, &lo, &hi, energies, &ld);
      auto& ckpt                   = chem_env.run_context["ccsd_t"];
      ckpt["total_tasks"]          = total_tasks;
      ckpt["noab"]                 = noab;
      ckpt["nvab"]                 = nvab;
      ckpt["mso_tilesize_triples"] = ts_triples;
      ckpt["threshold"]            = threshold;
      ckpt["tasks_done"]           = ckpt_task(k);
      ckpt["energy1"]              = energies[0];
      ckpt["energy2"]              = energies[1];
      chem_env.write_run_context();
      print_progress(ckpt_task(k));
      ckpts_saved++;
    }
    ckpt_wait_time += std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - ckpt_start)
                        .count();
  };

  size_t next_ckpt = next_ckpt_after(start_task);

  // tasks below start_task were done before a restart and are never handed out
  ac->allocate(start_task);
  int64_t taskcount = 0;
  int64_t next      = ac->fetch_add(0, 1);

  size_t current_task             = 0;
  size_t last_reported_percentage = 0;
  if(total_tasks > 0) {
    last_reported_percentage = (start_task * 100) / total_tasks;
    last_reported_percentage -= last_reported_percentage % progress_step;
  }

  for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
    for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
      for(size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
        for(size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
          for(size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
            // #if 0
            //     for (size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
            //     for (size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
            //     for (size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
            //     for (size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
            //     for (size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
            // #endif
            // Checkpoints (see the overview above the checkpoint setup):
            //  1. At a checkpoint's task number, take this rank's snapshot.
            //  2. Add this rank's ready snapshots to the GA. On a GPU a snapshot is ready only
            //     after the work queued before it finishes, so this is checked on every pass.
            //  3. Rank 0 writes the checkpoints that are now complete.
            if(write_ckpt) {
              if(next_ckpt < nckpts && current_task == ckpt_task(next_ckpt)) {
                snapshot_checkpoint(next_ckpt);
                next_ckpt = next_ckpt_after(current_task);
              }
              if(add_to_checkpoints() && rank0) save_checkpoints();
            }

            current_task++;
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
                    // num_task++;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
                    ccsd_t_fully_fused_none_df_none_task<T>(
                      is_restricted, noab, nvab, rank, k_spin, k_range, k_offset, d_t1, d_t2, d_v2,
                      k_evl_sorted,
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
                      done_compute, done_copy);
#else
                    total_fused_ccsd_t_cpu<T>(
                      is_restricted, noab, nvab, rank, k_spin, k_range, k_offset, d_t1, d_t2, d_v2,
                      k_evl_sorted,
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
                  }
                }
              } // h3b

              next = ac->fetch_add(0, 1);
              if(write_ckpt && rank0) save_checkpoints();
            } // next == taskcount

            // report progress every 20% of the tasks
            if(rank == 0 && !write_ckpt) {
              const size_t percentage = (current_task * 100) / total_tasks;
              if(percentage >= last_reported_percentage + progress_step) {
                last_reported_percentage = percentage - percentage % progress_step;
                print_progress(current_task);
              }
            }

            taskcount++;
          }
        }
      }
    }
  }
  // } // end seq h3b

  if(write_ckpt) {
    if(start_task < total_tasks) snapshot_checkpoint(nckpts);
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    // the only GPU wait for checkpoints: once, after the loop, so the last snapshot is taken
    gpuDeviceSynchronize();
#endif
    add_to_checkpoints();
    // Step 4: rank 0 can finish its tasks long before the others, so it keeps writing
    // checkpoints as they complete until the last one is written
    if(rank0) {
      const auto   wait_start = std::chrono::high_resolution_clock::now();
      const double wait_time  = ckpt_wait_time;
      save_checkpoints();
      // Rank 0 gets here only when it has no tasks left, so this waits in time it would
      // otherwise spend idle in the barrier below. Checking every 100 ms instead of nonstop keeps
      // it from flooding the rank that holds the counters with reads and from spinning a core;
      // the last checkpoint is written at most 100 ms late.
      while(ckpts_saved < ckpts_added.size()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        save_checkpoints();
      }
      ckpt_wait_time = wait_time + std::chrono::duration_cast<std::chrono::duration<double>>(
                                     std::chrono::high_resolution_clock::now() - wait_start)
                                     .count();
    }
    ec.pg().barrier();
    ckpt_count->deallocate();
    GA_Destroy(ckpt_ga);
  }

  if(rank == 0) std::cout << std::endl << std::endl;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  gpuDeviceSynchronize();
#endif

  energy1 = energy_l[0];
  energy2 = energy_l[1];

  // df_simple_*/host_d*_size freed automatically when their *_v vectors go out of scope

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  memDevPool.deallocate(df_dev_s1_t1_span);
  memDevPool.deallocate(df_dev_s1_v2_span);
  memDevPool.deallocate(df_dev_d1_t2_span);
  memDevPool.deallocate(df_dev_d1_v2_span);
  memDevPool.deallocate(df_dev_d2_t2_span);
  memDevPool.deallocate(df_dev_d2_v2_span);
  memDevPool.deallocate(df_dev_energies_span);

  tamm::freePinnedMem(df_host_pinned_s1_t1);
  tamm::freePinnedMem(df_host_pinned_s1_v2);
  tamm::freePinnedMem(df_host_pinned_d1_t2);
  tamm::freePinnedMem(df_host_pinned_d1_v2);
  tamm::freePinnedMem(df_host_pinned_d2_t2);
  tamm::freePinnedMem(df_host_pinned_d2_v2);
  tamm::freePinnedMem(df_host_energies);

#else // cpu
  operator delete[](df_host_pinned_s1_t1);
  operator delete[](df_host_pinned_s1_v2);
  operator delete[](df_host_pinned_d1_t2);
  operator delete[](df_host_pinned_d1_v2);
  operator delete[](df_host_pinned_d2_t2);
  operator delete[](df_host_pinned_d2_v2);
  operator delete[](df_host_energies);
#endif

  auto cc_t2 = std::chrono::high_resolution_clock::now();
  auto ccsd_t_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count() -
    ckpt_wait_time;

  ec.pg().barrier();
  cc_t2 = std::chrono::high_resolution_clock::now();
  auto total_t_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  ac->deallocate();
  // ac deleted automatically when the unique_ptr goes out of scope

  return std::make_tuple(energy1, energy2, ccsd_t_time, total_t_time, start_task);
}

template<typename T>
void CCSD_T_Fused_Driver<T>::calculate_performance_ops(
  ChemEnv& chem_env, ExecutionContext& ec, std::vector<int>& k_spin, const TiledIndexSpace& MO,
  std::vector<T>& k_evl_sorted, double hf_ccsd_energy, bool is_restricted,
  long double& total_num_ops, bool seq_h3b, size_t start_task) {
  auto rank = ec.pg().rank().value();

  Index noab = MO("occ").num_tiles();
  Index nvab = MO("virt").num_tiles();

  auto                mo_tiles = MO.input_tile_sizes();
  std::vector<size_t> k_range;
  std::vector<size_t> k_offset;
  k_range.reserve(mo_tiles.size());
  k_offset.reserve(mo_tiles.size());
  size_t sum = 0;
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
          for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
            for(size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
              for(size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
                if((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
                   (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
                  if((!is_restricted) || (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
                                          k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {
                    double factor = 1.0;
                    if(is_restricted) factor = 2.0;
                    if((t_p4b == t_p5b) && (t_p5b == t_p6b)) { factor /= 6.0; }
                    else if((t_p4b == t_p5b) || (t_p5b == t_p6b)) { factor /= 2.0; }

                    if((t_h1b == t_h2b) && (t_h2b == t_h3b)) { factor /= 6.0; }
                    else if((t_h1b == t_h2b) || (t_h2b == t_h3b)) { factor /= 2.0; }

                    list_tasks.push_back(
                      std::make_tuple(t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, factor));
                  }
                }
              }
            }
          }
        }
      }
    } // nested for loops
  } // parallel h3b loop
  else { // seq h3b loop
    // same task order as execute(), so the tasks done before a (T) restart can be skipped
    size_t taskcount = 0;
    for(size_t t_h1b = 0; t_h1b < noab; t_h1b++) {
      for(size_t t_p4b = noab; t_p4b < noab + nvab; t_p4b++) {
        for(size_t t_h2b = t_h1b; t_h2b < noab; t_h2b++) {
          for(size_t t_p5b = t_p4b; t_p5b < noab + nvab; t_p5b++) {
            for(size_t t_p6b = t_p5b; t_p6b < noab + nvab; t_p6b++) {
              if(taskcount++ < start_task) continue;
              for(size_t t_h3b = t_h2b; t_h3b < noab; t_h3b++) {
                if((k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b]) ==
                   (k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b])) {
                  if((!is_restricted) || (k_spin[t_p4b] + k_spin[t_p5b] + k_spin[t_p6b] +
                                          k_spin[t_h1b] + k_spin[t_h2b] + k_spin[t_h3b]) <= 8) {
                    double factor = 1.0;
                    if(is_restricted) factor = 2.0;

                    if((t_p4b == t_p5b) && (t_p5b == t_p6b)) { factor /= 6.0; }
                    else if((t_p4b == t_p5b) || (t_p5b == t_p6b)) { factor /= 2.0; }

                    if((t_h1b == t_h2b) && (t_h2b == t_h3b)) { factor /= 6.0; }
                    else if((t_h1b == t_h2b) || (t_h2b == t_h3b)) { factor /= 2.0; }

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
  } // end seq h3b

  total_num_ops = (long double) ccsd_t_fully_fused_performance(
    is_restricted, list_tasks, rank, 1, noab, nvab, k_spin, k_range, k_offset, k_evl_sorted);
}
