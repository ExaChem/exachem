/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cassert>
#include <cstdio>
#include <string>

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "tamm/gpu_streams.hpp"
using tamm::gpuEvent_t;
using tamm::gpuStream_t;
#endif

#ifdef USE_CUDA
#define CHECK_ERR(x)                           \
  {                                            \
    cudaError_t err = cudaGetLastError();      \
    if(cudaSuccess != err) {                   \
      printf("%s\n", cudaGetErrorString(err)); \
      exit(1);                                 \
    }                                          \
  }

#define CUDA_SAFE(x)                                                                        \
  if(cudaSuccess != (x)) {                                                                  \
    printf("CUDA API FAILED AT LINE %d OF FILE %s errorcode: %s, %s\n", __LINE__, __FILE__, \
           cudaGetErrorName(x), cudaGetErrorString(cudaGetLastError()));                    \
    exit(100);                                                                              \
  }
#endif // USE_CUDA

#ifdef USE_HIP
#define CHECK_ERR(x)                          \
  {                                           \
    hipError_t err = hipGetLastError();       \
    if(hipSuccess != err) {                   \
      printf("%s\n", hipGetErrorString(err)); \
      exit(1);                                \
    }                                         \
  }

#define HIP_SAFE(x)                                                                        \
  if(hipSuccess != (x)) {                                                                  \
    printf("HIP API FAILED AT LINE %d OF FILE %s errorcode: %s, %s\n", __LINE__, __FILE__, \
           hipGetErrorName(x), hipGetErrorString(hipGetLastError()));                      \
    exit(100);                                                                             \
  }
#endif // USE_HIP

typedef long Integer;
// static int notset;

#define DIV_UB(x, y) ((x) / (y) + ((x) % (y) ? 1 : 0))
#define TG_MIN(x, y) ((x) < (y) ? (x) : (y))

void        initMemModule();
std::string check_memory_req(const int cc_t_ts, const int nbf);

void* getGpuMem(size_t bytes);
void* getPinnedMem(size_t bytes);
void* getHostMem(size_t bytes);
void* getPinnedMem(size_t bytes);
void  freeHostMem(void* p);
void  freePinnedMem(void* p);
void  freeGpuMem(void* p);
void  freePinnedMem(void* p);

void finalizeMemModule();

struct hostEnergyReduceData_t {
  double* result_energy;
  double* host_energies;
  size_t  num_blocks;
  double  factor;
};
