/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "system_data.hpp"
using namespace tamm;

#if defined(USE_SCALAPACK)
#include <blacspp/grid.hpp>
#include <scalapackpp/block_cyclic_matrix.hpp>
#include <scalapackpp/eigenvalue_problem/sevp.hpp>
#include <scalapackpp/pblas/gemm.hpp>
#endif

// template<class T>
// inline T& unconst_cast(const T& v) {
//   return const_cast<T&>(v);
// }

inline auto sum_tensor_sizes = [](auto&&... t) {
  return ((compute_tensor_size(t) + ...) * 8) / (1024 * 1024 * 1024.0);
};

inline auto free_vec_tensors = [](auto&&... vecx) {
  (std::for_each(vecx.begin(), vecx.end(), [](auto& t) { t.deallocate(); }), ...);
};

inline auto free_tensors = [](auto&&... t) { ((t.deallocate()), ...); };

inline void check_memory_requirements(ExecutionContext& ec, double calc_mem) {
  auto minfo = ec.mem_info();
  if(calc_mem > static_cast<double>(minfo.total_cpu_mem)) {
    ec.print_mem_info();
    std::string err_msg = "ERROR: Insufficient CPU memory, required = " + std::to_string(calc_mem) +
                          "GiB, available = " + std::to_string(minfo.total_cpu_mem) + " GiB";
    tamm_terminate(err_msg);
  }
}

#if !defined(USE_SCALAPACK)
struct ScalapackInfo {
  bool use_scalapack{false};
};
#endif

#if defined(USE_SCALAPACK)
struct ScalapackInfo {
  int64_t                                         npr{}, npc{}, scalapack_nranks{};
  bool                                            use_scalapack{true};
  MPI_Comm                                        comm;
  tamm::ProcGroup                                 pg;
  tamm::ExecutionContext                          ec;
  std::unique_ptr<blacspp::Grid>                  blacs_grid;
  std::unique_ptr<scalapackpp::BlockCyclicDist2D> blockcyclic_dist;
};

MPI_Comm get_scalapack_comm(tamm::ExecutionContext& ec, int sca_nranks);

void setup_scalapack_info(SystemData& sys_data, ScalapackInfo& scalapack_info, MPI_Comm& scacomm);
#endif

// Contains node, ppn information used for creating a smaller process group from world group
struct ProcGroupData {
  int nnodes{};     // total number of nodes
  int spg_nnodes{}; // number of nodes in smaller process group
  int ppn{};        // processes per node
  int spg_nranks{}; // number of rank in smaller process group
  // #nodes used for scalapack operations can further be a subset of the smaller process group
  int scalapack_nnodes{};
  int scalapack_nranks{};

  auto unpack() {
    return std::make_tuple(nnodes, spg_nnodes, ppn, spg_nranks, scalapack_nnodes, scalapack_nranks);
  }
};

// Nbf, % of nodes, % of Nbf, nnodes from input file, (% of nodes, % of nbf) for scalapack
ProcGroupData get_spg_data(ExecutionContext& ec, const size_t N, const int node_p,
                           const int nbf_p = -1, const int node_inp = -1, const int node_p_sca = -1,
                           const int nbf_p_sca = -1);