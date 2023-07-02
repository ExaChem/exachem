/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "json_data.hpp"
using namespace tamm;

#if defined(USE_SCALAPACK)
#include <blacspp/grid.hpp>
#include <scalapackpp/block_cyclic_matrix.hpp>
#include <scalapackpp/eigenvalue_problem/sevp.hpp>
#include <scalapackpp/pblas/gemm.hpp>
#endif

inline auto sum_tensor_sizes = [](auto&&... t) {
  return ((compute_tensor_size(t) + ...) * 8) / (1024 * 1024 * 1024.0);
};

void check_memory_requirements(ExecutionContext& ec, double calc_mem) {
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

MPI_Comm get_scalapack_comm(tamm::ExecutionContext& ec, int sca_nranks) {
  int       lranks[sca_nranks];
  auto      gcomm = ec.pg().comm();
  MPI_Group wgroup;
  MPI_Comm_group(gcomm, &wgroup);

  for(int i = 0; i < sca_nranks; i++) lranks[i] = i;
  MPI_Group sca_group;
  MPI_Group_incl(wgroup, sca_nranks, lranks, &sca_group);
  MPI_Comm scacomm;
  MPI_Comm_create(gcomm, sca_group, &scacomm);
  return scacomm;
}

void setup_scalapack_info(SystemData& sys_data, ScalapackInfo& scalapack_info, MPI_Comm& scacomm) {
#if defined(USE_UPCXX)
  abort(); // Not supported with UPC++
#endif
  scalapack_info.comm = scacomm;
  if(scacomm == MPI_COMM_NULL) return;
  auto blacs_setup_st = std::chrono::high_resolution_clock::now();
  // Sanity checks
  scalapack_info.npr   = sys_data.options_map.scf_options.scalapack_np_row;
  scalapack_info.npc   = sys_data.options_map.scf_options.scalapack_np_col;
  int scalapack_nranks = scalapack_info.npr * scalapack_info.npc;

  scalapack_info.pg = ProcGroup::create_coll(scacomm);
  scalapack_info.ec =
    ExecutionContext{scalapack_info.pg, DistributionKind::dense, MemoryManagerKind::ga};

  int sca_world_size = scalapack_info.pg.size().value();

  // Default to square(ish) grid
  if(scalapack_nranks == 0) {
    int64_t npr = std::sqrt(sca_world_size);
    int64_t npc = sca_world_size / npr;
    while(npr * npc != sca_world_size) {
      npr--;
      npc = sca_world_size / npr;
    }
    scalapack_nranks   = sca_world_size;
    scalapack_info.npr = npr;
    scalapack_info.npc = npc;
  }

  EXPECTS(sca_world_size >= scalapack_nranks);

  if(not scalapack_nranks) scalapack_nranks = sca_world_size;
  std::vector<int64_t> scalapack_ranks(scalapack_nranks);
  std::iota(scalapack_ranks.begin(), scalapack_ranks.end(), 0);
  scalapack_info.scalapack_nranks = scalapack_nranks;

  if(scalapack_info.pg.rank() == 0) {
    std::cout << "scalapack_nranks = " << scalapack_nranks << std::endl;
    std::cout << "scalapack_np_row = " << scalapack_info.npr << std::endl;
    std::cout << "scalapack_np_col = " << scalapack_info.npc << std::endl;
  }

  int&       mb_ = sys_data.options_map.scf_options.scalapack_nb;
  const auto N   = sys_data.nbf_orig;
  if(mb_ > N / scalapack_info.npr) {
    mb_                                           = N / scalapack_info.npr;
    sys_data.options_map.scf_options.scalapack_nb = mb_;
    if(scalapack_info.pg.rank() == 0)
      std::cout << "WARNING: Resetting scalapack block size (scalapack_nb) to: " << mb_
                << std::endl;
  }

  scalapack_info.blacs_grid =
    std::make_unique<blacspp::Grid>(scalapack_info.pg.comm(), scalapack_info.npr,
                                    scalapack_info.npc, scalapack_ranks.data(), scalapack_info.npr);
  scalapack_info.blockcyclic_dist =
    std::make_unique<scalapackpp::BlockCyclicDist2D>(*scalapack_info.blacs_grid, mb_, mb_, 0, 0);

  auto blacs_setup_en = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> blacs_time = blacs_setup_en - blacs_setup_st;

  // if(scalapack_info.pg.rank() == 0)
  //   std::cout << std::fixed << std::setprecision(2) << std::endl
  //             << "Time for BLACS setup: " << blacs_time.count() << " secs" << std::endl;
}
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
                           const int nbf_p_sca = -1) {
  ProcGroupData pgdata;
  pgdata.nnodes = ec.nnodes();
  pgdata.ppn    = ec.ppn();

  const int ppn    = pgdata.ppn;
  const int nnodes = pgdata.nnodes;

  int spg_guessranks = std::ceil((nbf_p / 100.0) * N);
  if(node_p > 0) spg_guessranks = std::ceil((node_p / 100.0) * nnodes);
  int spg_nnodes = spg_guessranks / ppn;
  if(spg_guessranks % ppn > 0 || spg_nnodes == 0) spg_nnodes++;
  if(spg_nnodes > nnodes) spg_nnodes = nnodes;
  int spg_nranks = spg_nnodes * ppn;

  if(node_inp > nnodes) {
    const std::string errmsg = "ERROR: nnodes (" + std::to_string(node_inp) +
                               ") provided is greater than the number of nodes (" +
                               std::to_string(nnodes) + ") available!";
    tamm_terminate(errmsg);
  }

  if(node_inp > spg_nnodes) {
    spg_nnodes = nnodes;
    spg_nranks = spg_nnodes * ppn;
  }
  pgdata.spg_nnodes = spg_nnodes;
  pgdata.spg_nranks = spg_nranks;

#if defined(USE_SCALAPACK)
  // Find nearest square
  int sca_nranks = std::ceil(N * (nbf_p_sca / 100.0));
  if(node_p_sca > 0) sca_nranks = std::ceil((node_p_sca / 100.0) * nnodes);
  if(sca_nranks > spg_nranks) sca_nranks = spg_nranks;
  sca_nranks = std::pow(std::floor(std::sqrt(sca_nranks)), 2);
  if(sca_nranks == 0) sca_nranks = 1;
  int sca_nnodes = sca_nranks / ppn;
  if(sca_nranks % ppn > 0 || sca_nnodes == 0) sca_nnodes++;
  if(sca_nnodes > nnodes) sca_nnodes = nnodes;
  // if(sca_nnodes == 1) ppn = sca_nranks; // single node case
  pgdata.scalapack_nnodes = sca_nnodes;
  pgdata.scalapack_nranks = sca_nranks;
#else
  pgdata.scalapack_nnodes = spg_nnodes;
  pgdata.scalapack_nranks = spg_nranks;
#endif

  return pgdata;
}
