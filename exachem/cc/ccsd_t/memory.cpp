/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "ccsd_t_common.hpp"
#include <cstdlib>
#include <map>
#include <set>
using namespace std;

static map<size_t, set<void*>> free_list_gpu, free_list_host;
static map<void*, size_t>      live_ptrs_gpu, live_ptrs_host;

static void clearGpuFreeList() {
  for(map<size_t, set<void*>>::iterator it = free_list_gpu.begin(); it != free_list_gpu.end();
      ++it) {
    for(set<void*>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
#if defined(USE_CUDA)
      CUDA_SAFE(cudaFree(*it2));
#elif defined(USE_HIP)
      HIP_SAFE(hipFree(*it2));
#elif defined(USE_DPCPP)
      sycl::free(*it2, tamm::GPUStreamPool::getInstance().getStream().first);
#endif
    }
  }
  free_list_gpu.clear();
}

static void clearHostFreeList() {
  for(map<size_t, set<void*>>::iterator it = free_list_host.begin(); it != free_list_host.end();
      ++it) {
    for(set<void*>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
#if defined(USE_CUDA)
      CUDA_SAFE(cudaFreeHost(*it2));
#elif defined(USE_HIP)
      HIP_SAFE(hipHostFree(*it2));
#elif defined(USE_DPCPP)
      sycl::free(*it2, tamm::GPUStreamPool::getInstance().getStream().first);
#else
      free(*it2);
#endif
    }
  }
  free_list_host.clear();
}

static size_t num_resurrections = 0; // num_morecore=0;

static void* moreDeviceMem(size_t bytes) {
  void* ptr = nullptr;

#if defined(USE_CUDA)
  CUDA_SAFE(cudaMalloc(&ptr, bytes));
#elif defined(USE_HIP)
  HIP_SAFE(hipMalloc(&ptr, bytes));
#elif defined(USE_DPCPP)
  ptr = sycl::malloc_device(bytes, tamm::GPUStreamPool::getInstance().getStream().first);
#endif

  assert(ptr != nullptr); /*We hopefully have a pointer*/
  return ptr;
}

static void* moreHostMem(size_t bytes) {
  void* ptr = nullptr;

#if defined(USE_CUDA)
  CUDA_SAFE(cudaMallocHost(&ptr, bytes));
#elif defined(USE_HIP)
  HIP_SAFE(hipHostMalloc(&ptr, bytes));
#elif defined(USE_DPCPP)
  ptr = sycl::malloc_host(bytes, tamm::GPUStreamPool::getInstance().getStream().first);
#else
  ptr = (void*) malloc(bytes);
#endif

  assert(ptr != nullptr); /*We hopefully have a pointer*/
  return ptr;
}

static inline void* resurrect_from_free_list(map<size_t, set<void*>>& free_map, size_t bytes,
                                             map<void*, size_t>& liveset) {
  void* ptr = nullptr;
  num_resurrections += 1;
  assert(free_map.find(bytes) != free_map.end());
  /* assert(free_map.find(bytes)->second.size() > 0); */
  set<void*>& st = free_map.find(bytes)->second;
  ptr            = *st.begin();
  st.erase(ptr);
  if(st.size() == 0) free_map.erase(bytes);
  liveset[ptr] = bytes;

  return ptr;
}

void* getGpuMem(size_t bytes) {
  void* ptr = nullptr;

#ifdef NO_OPT

#if defined(USE_CUDA)
  CUDA_SAFE(cudaMalloc((void**) &ptr, bytes));
#elif defined(USE_HIP)
  HIP_SAFE(hipMalloc((void**) &ptr, bytes));
#elif defined(USE_DPCPP)
  ptr = sycl::malloc_device(bytes, tamm::GPUStreamPool::getInstance().getStream().first);
#endif

#else
  if(free_list_gpu.find(bytes) != free_list_gpu.end()) {
    set<void*>& lst = free_list_gpu.find(bytes)->second;
    if(lst.size() != 0) {
      ptr = resurrect_from_free_list(free_list_gpu, bytes, live_ptrs_gpu);
      return ptr;
    }
  }
  else {
    for(map<size_t, set<void*>>::iterator it = free_list_gpu.begin(); it != free_list_gpu.end();
        ++it) {
      if(it->first >= bytes && it->second.size() > 0) {
        ptr = resurrect_from_free_list(free_list_gpu, it->first, live_ptrs_gpu);
        return ptr;
      }
    }
  }

  ptr = moreDeviceMem(bytes);
  live_ptrs_gpu[ptr] = bytes;
#endif // NO_OPT

  return ptr;
}

void* getPinnedMem(size_t bytes) {
  void* ptr = nullptr;

#if defined(USE_CUDA)
  CUDA_SAFE(cudaMallocHost((void**) &ptr, bytes));
#elif defined(USE_HIP)
  HIP_SAFE(hipMallocHost((void**) &ptr, bytes));
#elif defined(USE_DPCPP)
  ptr = sycl::malloc_host(bytes, tamm::GPUStreamPool::getInstance().getStream().first);
#endif
  return ptr;
}

void freePinnedMem(void* ptr) {
#if defined(USE_CUDA)
  CUDA_SAFE(cudaFreeHost(ptr));
#elif defined(USE_HIP)
  HIP_SAFE(hipFreeHost(ptr));
#elif defined(USE_DPCPP)
  sycl::free(ptr, tamm::GPUStreamPool::getInstance().getStream().first);
#endif
}

void* getHostMem(size_t bytes) {
  void* ptr = nullptr;

#ifdef NO_OPT
  ptr = std::malloc(bytes);
#else  // NO_OPT
  if(free_list_host.find(bytes) != free_list_host.end()) {
    set<void*>& lst = free_list_host.find(bytes)->second;
    if(lst.size() != 0) {
      ptr = resurrect_from_free_list(free_list_host, bytes, live_ptrs_host);
      return ptr;
    }
  }
  else {
    for(map<size_t, set<void*>>::iterator it = free_list_host.begin(); it != free_list_host.end();
        ++it) {
      if(it->first >= bytes && it->second.size() > 0) {
        ptr = resurrect_from_free_list(free_list_host, it->first, live_ptrs_host);
        return ptr;
      }
    }
  }

  ptr = moreHostMem(bytes);
  live_ptrs_host[ptr] = bytes;
#endif // NO_OPT
  return ptr;
}

void freeHostMem(void* p) {
#ifdef NO_OPT
  std::free(p);
#else  // NO_OPT
  size_t bytes;
  assert(live_ptrs_host.find(p) != live_ptrs_host.end());
  bytes = live_ptrs_host[p];
  live_ptrs_host.erase(p);
  free_list_host[bytes].insert(p);
#endif // NO_OPT
}

void freeGpuMem(void* p) {
#ifdef NO_OPT

#if defined(USE_CUDA)
  CUDA_SAFE(cudaFree(p));
#elif defined(USE_HIP)
  HIP_SAFE(hipFree(p));
#elif defined(USE_DPCPP)
  sycl::free(p, tamm::GPUStreamPool::getInstance().getStream().first);
#endif // NO_OPT

#else

  size_t bytes;
  assert(live_ptrs_gpu.find(p) != live_ptrs_gpu.end());
  bytes = live_ptrs_gpu[p];
  live_ptrs_gpu.erase(p);
  free_list_gpu[bytes].insert(p);

#endif
}

void finalizememmodule() {
  /*there should be no live pointers*/
  assert(live_ptrs_gpu.size() == 0);
  assert(live_ptrs_host.size() == 0);

  /*release all freed pointers*/
  clearGpuFreeList();
  clearHostFreeList();
}
