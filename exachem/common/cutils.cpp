/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/cutils.hpp"

#if defined(USE_SCALAPACK)

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
  MPI_Group_free(&wgroup);
  MPI_Group_free(&sca_group);
  return scacomm;
}

void setup_scalapack_info(tamm::ExecutionContext& ec, ChemEnv& chem_env,
                          ScalapackInfo& scalapack_info, ProcGroupData& pgdata) {
  SystemData& sys_data    = chem_env.sys_data;
  SCFOptions& scf_options = chem_env.ioptions.scf_options;
#if defined(USE_UPCXX)
  abort(); // Not supported with UPC++
#endif

  scalapack_info.npr = scf_options.scalapack_np_row;
  scalapack_info.npc = scf_options.scalapack_np_col;
  int sca_user_ranks = scalapack_info.npr * scalapack_info.npc;

  // node_p_sca = % of nodes, nbf_p_sca = 4 = % of nbf for scalapack
  int sca_nranks = std::ceil(sys_data.nbf_orig * (4 / 100.0));
  // if(node_p_sca > 0) sca_nranks = std::ceil((node_p_sca / 100.0) * nnodes);
  if(sca_user_ranks > 0) sca_nranks = sca_user_ranks;
  if(sca_nranks > pgdata.spg_nranks) sca_nranks = pgdata.spg_nranks;
  // Find nearest square
  sca_nranks = std::pow(std::floor(std::sqrt(sca_nranks)), 2);
  if(sca_nranks == 0) sca_nranks = 1;

  int sca_nnodes = sca_nranks / pgdata.ppn;
  if(sca_nranks % pgdata.ppn > 0 || sca_nnodes == 0) sca_nnodes++;
  // if(sca_nnodes > pgdata.spg_nnodes) sca_nnodes = pgdata.spg_nnodes;

  pgdata.scalapack_nnodes         = sca_nnodes;
  pgdata.scalapack_nranks         = sca_nranks;
  scalapack_info.scalapack_nranks = sca_nranks;

  // Always use square grid
  scalapack_info.npr = std::sqrt(sca_nranks);
  scalapack_info.npc = scalapack_info.npr;

  scalapack_info.comm = get_scalapack_comm(ec, sca_nranks);

  if(scalapack_info.comm == MPI_COMM_NULL) return;
  // auto blacs_setup_st = std::chrono::high_resolution_clock::now();

  scalapack_info.pg = ProcGroup::create_coll(scalapack_info.comm);
  scalapack_info.ec =
    ExecutionContext{scalapack_info.pg, DistributionKind::dense, MemoryManagerKind::ga};

  std::vector<int64_t> scalapack_ranks(sca_nranks);
  std::iota(scalapack_ranks.begin(), scalapack_ranks.end(), 0);

  int& mb_ = scf_options.scalapack_nb;

  if(scalapack_info.pg.rank() == 0) {
    std::cout << "scalapack_nnodes = " << sca_nnodes << std::endl;
    std::cout << "scalapack_nranks = " << sca_nranks << std::endl;
    std::cout << "scalapack_np_row = " << scalapack_info.npr << std::endl;
    std::cout << "scalapack_np_col = " << scalapack_info.npc << std::endl;
    std::cout << "scalapack_nb     = " << mb_ << std::endl;
  }

  const auto N = sys_data.nbf_orig;
  if(mb_ > N / scalapack_info.npr) {
    mb_ = std::ceil(N / scalapack_info.npr);
    // find closest power of 2
    int mb_log = (int) std::log2(mb_);
    mb_        = (int) std::pow(2, mb_log);

    scf_options.scalapack_nb = mb_;

    if(scalapack_info.pg.rank() == 0)
      std::cout << "WARNING: Resetting scalapack block size (scalapack_nb) to: " << mb_
                << std::endl;
  }

  scalapack_info.blacs_grid =
    std::make_unique<blacspp::Grid>(scalapack_info.pg.comm(), scalapack_info.npr,
                                    scalapack_info.npc, scalapack_ranks.data(), scalapack_info.npr);
  scalapack_info.blockcyclic_dist =
    std::make_unique<scalapackpp::BlockCyclicDist2D>(*scalapack_info.blacs_grid, mb_, mb_, 0, 0);

  // auto blacs_setup_en = std::chrono::high_resolution_clock::now();

  // std::chrono::duration<double> blacs_time = blacs_setup_en - blacs_setup_st;

  // if(scalapack_info.pg.rank() == 0)
  //   std::cout << std::fixed << std::setprecision(2) << std::endl
  //             << "Time for BLACS setup: " << blacs_time.count() << " secs" << std::endl;
}

#endif

// Nbf, % of nodes, % of Nbf, nnodes from input file, (% of nodes, % of nbf) for scalapack
ProcGroupData get_spg_data(ExecutionContext& ec, const size_t N, const int node_p, const int nbf_p,
                           const int node_inp) {
  ProcGroupData pgdata;
  pgdata.ppn = ec.ppn();

  const int ppn    = pgdata.ppn;
  const int nnodes = ec.nnodes();

  int spg_guessranks = std::ceil((nbf_p / 100.0) * N);
  if(node_p > 0) spg_guessranks = std::ceil((node_p / 100.0) * nnodes);
  int spg_nnodes = spg_guessranks / ppn;
  if(spg_guessranks % ppn > 0 || spg_nnodes == 0) spg_nnodes++;
  if(spg_nnodes > nnodes) spg_nnodes = nnodes;
  int spg_nranks = spg_nnodes * ppn;

  int user_nnodes = static_cast<int>(node_inp / 100) * nnodes;
  if(user_nnodes > spg_nnodes) {
    spg_nnodes = user_nnodes;
    spg_nranks = spg_nnodes * ppn;
  }
  pgdata.spg_nnodes = spg_nnodes;
  pgdata.spg_nranks = spg_nranks;

  return pgdata;
}
