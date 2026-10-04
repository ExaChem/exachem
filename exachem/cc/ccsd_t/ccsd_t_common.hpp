/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <new>
#include <string>

// Max number of occupied (noab) / virtual (nvab) tiles supported by the fused GPU kernels;
// sizes their per-task constant-memory tables.
inline constexpr std::size_t MAX_NOAB{50};
inline constexpr std::size_t MAX_NVAB{140};

// integer ceiling division; a macro so it works in host and CUDA/HIP/SYCL device code
#define CEIL(a, b) (((a) + (b) - 1) / (b))

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "tamm/gpu_streams.hpp"
using tamm::gpuEvent_t;
using tamm::gpuStream_t;
using event_ptr_t = std::shared_ptr<tamm::gpuEvent_t>;
#endif

#ifdef USE_CUDA
#define CUDA_SAFE(x)                                                                        \
  if(cudaSuccess != (x)) {                                                                  \
    printf("CUDA API FAILED AT LINE %d OF FILE %s errorcode: %s, %s\n", __LINE__, __FILE__, \
           cudaGetErrorName(x), cudaGetErrorString(cudaGetLastError()));                    \
    exit(100);                                                                              \
  }
#endif // USE_CUDA

#ifdef USE_HIP
#define HIP_SAFE(x)                                                                        \
  if(hipSuccess != (x)) {                                                                  \
    printf("HIP API FAILED AT LINE %d OF FILE %s errorcode: %s, %s\n", __LINE__, __FILE__, \
           hipGetErrorName(x), hipGetErrorString(hipGetLastError()));                      \
    exit(100);                                                                             \
  }
#endif // USE_HIP

struct hostEnergyReduceData_t {
  double* result_energy;
  double* host_energies;
  size_t  num_blocks;
  double  factor;
};
