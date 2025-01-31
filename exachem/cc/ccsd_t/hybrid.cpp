/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

/*-------------hybrid execution------------*/

#include "exachem/cc/ccsd_t/ccsd_t_common.hpp"
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>

std::string check_memory_req(const int cc_t_ts, const int nbf) {
  size_t      total_gpu_mem{0};
  std::string errmsg = "";

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  size_t free_gpu_mem{0};
  tamm::gpuMemGetInfo(&free_gpu_mem, &total_gpu_mem);
#endif

  const size_t gpu_mem_req =
    (9.0 * (std::pow(cc_t_ts, 2) + std::pow(cc_t_ts, 4) + 2 * 2 * nbf * std::pow(cc_t_ts, 3)) * 8);
  int gpu_mem_check = 0;
  if(gpu_mem_req >= total_gpu_mem) gpu_mem_check = 1;
  if(gpu_mem_check) {
    const double gib = 1024 * 1024 * 1024.0;
    errmsg = "ERROR: GPU memory not sufficient for (T) calculation, available memory per gpu: " +
             std::to_string(total_gpu_mem / gib) +
             " GiB, required: " + std::to_string(gpu_mem_req / gib) +
             " GiB. Please set a smaller tilesize and retry";
  }

  return errmsg;
}
