/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"
#include "exachem/cc/ducc/ducc-t_ccsd.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"

#include <filesystem>
namespace fs = std::filesystem;

namespace exachem::cc::ducc {
using namespace exachem::cc::ducc::internal;

std::vector<Range> extract_ranges(const IndexVector& x) {
  std::vector<Range> result;

  if(x.empty()) return result;

  auto start = x[0];
  auto prev  = x[0];

  for(size_t i = 1; i < x.size(); ++i) {
    if(x[i] == prev + 1) {
      // still contiguous
      prev = x[i];
    }
    else {
      // end current range
      result.emplace_back(start, prev + 1);
      start = x[i];
      prev  = x[i];
    }
  }
  result.emplace_back(start, prev + 1); // add last range

  return result;
}

std::vector<std::vector<int>> split_contiguous_chunks(const std::vector<int>& vec) {
  std::vector<std::vector<int>> result;
  if(vec.empty()) return result;

  std::vector<int> current = {vec[0]};
  for(size_t i = 1; i < vec.size(); ++i) {
    if(vec[i] == vec[i - 1] + 1) { current.push_back(vec[i]); }
    else {
      result.push_back(current);
      current = {vec[i]};
    }
  }
  result.push_back(current);
  return result;
}

std::vector<Tile> tile_vector(const IndexVector& occ_int_vec, const IndexVector& virt_int_vec,
                              const std::vector<int>& subspace_bounds) {
  std::vector<int> v1, v2;

  std::transform(occ_int_vec.begin(), occ_int_vec.end(), std::back_inserter(v1),
                 [](unsigned int x) { return static_cast<int>(x); });

  std::transform(virt_int_vec.begin(), virt_int_vec.end(), std::back_inserter(v2),
                 [](unsigned int x) { return static_cast<int>(x); });

  std::vector<int> merged = v1;
  merged.insert(merged.end(), v2.begin(), v2.end());
  std::sort(merged.begin(), merged.end());

  std::vector<std::vector<int>> chunks = split_contiguous_chunks(merged);
  std::vector<Tile>             tile_sizes;

  size_t chunk_idx = 0;
  for(size_t i = 0; i + 1 < subspace_bounds.size(); ++i) {
    int sub_start = subspace_bounds[i];
    int sub_end   = subspace_bounds[i + 1];
    int cursor    = sub_start;

    while(chunk_idx < chunks.size()) {
      const auto& chunk   = chunks[chunk_idx];
      int         c_start = chunk.front();
      int         c_end   = chunk.back() + 1;

      if(c_start >= sub_end) break; // chunk is after this subspace
      if(c_end <= sub_start) {
        ++chunk_idx;
        continue;
      } // chunk is before subspace

      // Process gap before chunk
      if(cursor < c_start) {
        int gap_end = std::min(c_start, sub_end);
        if(gap_end > cursor) tile_sizes.push_back(gap_end - cursor);
        cursor = gap_end;
      }

      // Process chunk inside subspace
      int tile_start = std::max(cursor, c_start);
      int tile_end   = std::min(c_end, sub_end);
      if(tile_end > tile_start) {
        tile_sizes.push_back(tile_end - tile_start);
        cursor = tile_end;
      }

      // If chunk spills over into next subspace, stop here
      if(c_end > sub_end) break;

      ++chunk_idx;
    }

    // Final gap to end of subspace
    if(cursor < sub_end) { tile_sizes.push_back(sub_end - cursor); }
  }

  return tile_sizes;
}

std::tuple<TiledIndexSpace, TAMM_SIZE> setupMOIS_QFlow(ExecutionContext& ec, ChemEnv& chem_env,
                                                       IndexVector& occ_int_vec,
                                                       IndexVector& virt_int_vec) {
  // const int   rank         = ec.pg().rank().value();
  SystemData& sys_data     = chem_env.sys_data;
  auto        ccsd_options = chem_env.ioptions.ccsd_options;
  auto        task_options = chem_env.ioptions.task_options;

  TAMM_SIZE total_orbitals = sys_data.nmo;
  TAMM_SIZE nocc           = sys_data.nocc;
  TAMM_SIZE n_occ_alpha    = sys_data.n_occ_alpha;
  TAMM_SIZE n_occ_beta     = sys_data.n_occ_beta;
  TAMM_SIZE n_vir_alpha    = sys_data.n_vir_alpha;
  TAMM_SIZE n_vir_beta     = sys_data.n_vir_beta;

  // Active space sizes
  // 'int' (internal) are orbitals in the active space
  // 'ext' (external) are orbitals outside the active space
  // xxx_int + xxx_ext = xxx
  TAMM_SIZE occ_alpha_int = 0;
  TAMM_SIZE occ_beta_int  = 0;
  TAMM_SIZE vir_alpha_int = 0;
  TAMM_SIZE vir_beta_int  = 0;
  occ_alpha_int           = ccsd_options.nactive_oa;
  occ_beta_int            = ccsd_options.nactive_ob;
  vir_alpha_int           = ccsd_options.nactive_va;
  vir_beta_int            = ccsd_options.nactive_vb;

  Tile tce_tile                    = ccsd_options.tilesize;
  chem_env.is_context.mso_tilesize = tce_tile;

  auto vec_ranges_occ  = extract_ranges(occ_int_vec);
  auto vec_ranges_virt = extract_ranges(virt_int_vec);

  // Check if active space is allowed:
  if(task_options.ducc.first) {
    if(n_occ_alpha != n_occ_beta)
      tamm_terminate("[DUCC ERROR]: DUCC is only for closed-shell calculations");
    if(occ_alpha_int > n_occ_alpha) tamm_terminate("[DUCC ERROR]: nactive_oa > n_occ_alpha");
    if(occ_beta_int > n_occ_beta) tamm_terminate("[DUCC ERROR]: nactive_ob > n_occ_beta");
    if(vir_alpha_int > n_vir_alpha) tamm_terminate("[DUCC ERROR]: nactive_va > n_vir_alpha");
    if(vir_beta_int > n_vir_beta) tamm_terminate("[DUCC ERROR]: nactive_vb > n_vir_beta");
    if(occ_alpha_int == 0 || occ_beta_int == 0)
      tamm_terminate("[DUCC ERROR]: nactive_oa/nactive_ob cannot be 0");
    if(occ_alpha_int != occ_beta_int) tamm_terminate("[DUCC ERROR]: nactive_oa != nactive_ob");
    if(vir_alpha_int != vir_beta_int) tamm_terminate("[DUCC ERROR]: nactive_va != nactive_vb");
  }

  chem_env.is_context.mso_tilesize = 1;

  // | occ_alpha | occ_beta | virt_alpha | virt_beta |
  // | occ_alpha_ext | occ_alpha_int | occ_beta_ext | occ_beta_int | --> (cont.)
  //    --> | vir_alpha_int | vir_alpha_ext | vir_beta_int |vir_beta_ext |
  IndexSpace MO_IS{
    range(0, total_orbitals),
    {
      {"occ", {range(0, nocc)}},
      {"occ_alpha", {range(0, n_occ_alpha)}},
      {"occ_beta", {range(n_occ_alpha, nocc)}},
      {"virt", {range(nocc, total_orbitals)}},
      {"virt_alpha", {range(nocc, nocc + n_vir_alpha)}},
      {"virt_beta", {range(nocc + n_vir_alpha, total_orbitals)}},
      // Active-space index spaces
      // {"occ_alpha_ext", {range(0, occ_alpha_ext)}},
      // {"occ_beta_ext", {range(n_occ_alpha, n_occ_alpha + occ_beta_ext)}},
      // {"occ_ext", {range(0, occ_alpha_ext), range(n_occ_alpha, n_occ_alpha + occ_beta_ext)}},
      // {"occ_alpha_int", {range(occ_alpha_ext, n_occ_alpha)}},
      // {"occ_beta_int", {range(n_occ_alpha + occ_beta_ext, nocc)}},
      {"occ_int", vec_ranges_occ},
      // {"virt_alpha_int", {range(nocc, nocc + vir_alpha_int)}},
      // {"virt_beta_int", {range(nocc + n_vir_alpha, nocc + n_vir_alpha + vir_beta_int)}},
      {"virt_int", vec_ranges_virt},
      // {"virt_alpha_ext", {range(nocc + vir_alpha_int, nocc + n_vir_alpha)}},
      // {"virt_beta_ext", {range(nocc + n_vir_alpha + vir_beta_int, total_orbitals)}},
      // {"virt_ext",
      //  {range(nocc + vir_alpha_int, nocc + n_vir_alpha),
      // range(nocc + n_vir_alpha + vir_beta_int, total_orbitals)}},
      // All spin index spaces
      {"all_alpha", {range(0, n_occ_alpha), range(nocc, nocc + n_vir_alpha)}},
      {"all_beta", {range(n_occ_alpha, nocc), range(nocc + n_vir_alpha, total_orbitals)}},
    },
    {{Spin{1}, {range(0, n_occ_alpha), range(nocc, nocc + n_vir_alpha)}},
     {Spin{2}, {range(n_occ_alpha, nocc), range(nocc + n_vir_alpha, total_orbitals)}}}};

  std::vector<Tile> mo_tiles;

  std::vector<int> subspace_bounds = {0, static_cast<int>(n_occ_alpha), static_cast<int>(nocc),
                                      static_cast<int>(nocc + n_vir_alpha),
                                      static_cast<int>(total_orbitals)};

  mo_tiles = tile_vector(occ_int_vec, virt_int_vec, subspace_bounds);

  if(ec.pg().rank() == 0 && chem_env.ioptions.ccsd_options.debug) {
    std::cout << std::endl << "occ_int_vec: " << occ_int_vec << std::endl;
    std::cout << "virt_int_vec: " << virt_int_vec << std::endl;
    std::cout << "MO subsapce bounds: " << subspace_bounds << std::endl;
    std::cout << "QFlow MO Tiles = " << mo_tiles << std::endl;
  }

  TiledIndexSpace MO{MO_IS, mo_tiles};

  return std::make_tuple(MO, total_orbitals);
}

void generate_combinations(const std::vector<int>& elements, int combination_size,
                           std::vector<std::vector<int>>& result_combinations) {
  std::vector<bool> bitmask(combination_size, true); // combination_size ones
  bitmask.resize(elements.size(), false);            // followed by n - combination_size zeros

  do {
    std::vector<int> combination;
    for(size_t i = 0; i < elements.size(); ++i) {
      if(bitmask[i]) { combination.push_back(elements[i]); }
    }
    result_combinations.push_back(combination);
  } while(std::prev_permutation(bitmask.begin(), bitmask.end()));
}

void ducc_qflow_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T = double;

  CCContext& cc_context      = chem_env.cc_context;
  cc_context.keep.fvt12_full = true;
  cc_context.compute.set(true, true); // compute ft12 and v2 in full
  exachem::cc::ccsd::cd_ccsd_driver(ec, chem_env);

  // TiledIndexSpace&           MO        = chem_env.is_context.MSO;
  TiledIndexSpace&           CI        = chem_env.is_context.CI;
  Tensor<T>                  d_f1      = chem_env.cd_context.d_f1;
  Tensor<T>                  cholVpr   = chem_env.cd_context.cholV2;
  Tensor<T>                  d_t1      = chem_env.cc_context.d_t1_full;
  Tensor<T>                  d_t2      = chem_env.cc_context.d_t2_full;
  cholesky_2e::V2Tensors<T>& v2tensors = chem_env.cd_context.v2tensors;

  const int   rank = ec.pg().rank().value();
  Scheduler   sch{ec};
  ExecutionHW ex_hw = ec.exhw();

  // auto [MO_AS, total_orbitals1] = setupMOIS_QFlow(ec, chem_env, occ_int_vec, virt_int_vec);

  // const TiledIndexSpace& N = MO_AS("all");
  // const TiledIndexSpace& O = MO_AS("occ");
  // const TiledIndexSpace& V = MO_AS("virt");

  // Tensor<T> cholVpr_UT = {{N, N, CI}, {SpinPosition::upper, SpinPosition::lower,
  // SpinPosition::ignore}}; sch.allocate(cholVpr_UT).execute(); retile_tamm_tensor(cholVpr,
  // cholVpr_UT); cholesky_2e::V2Tensors<T>& v2tensors_UT = cholesky_2e::setupV2Tensors<T>(ec,
  // cholVpr_UT, ex_hw); free_tensors(cholVpr, cholVpr_UT);

  // Tensor<T> dt1_full_ut = Tensor<T>{{V, O}, {1, 1}};
  // Tensor<T> dt2_full_ut = Tensor<T>{{V, V, O, O}, {2, 2}};
  // sch.allocate(dt1_full_ut, dt2_full_ut).execute();

  // Tensor<T> d_f1_full = Tensor<T>{{N, N}, {1, 1}};
  // sch.allocate(d_f1_full).execute();
  // retile_tamm_tensor(d_f1, d_f1_full);

  const int cycles = chem_env.ioptions.ccsd_options.qflow_cycles; // Number of cycles to run

  TAMM_SIZE total_orbitals = chem_env.sys_data.nmo / 2;  // Spatial Orbitals
  TAMM_SIZE nocc_orb       = chem_env.sys_data.nocc / 2; // Occupied Spatial Orbitals
  TAMM_SIZE nvirt_orb      = total_orbitals - nocc_orb;  // Virtual Spatial Orbitals

  TAMM_SIZE occ_alpha_int = chem_env.ioptions.ccsd_options.nactive_oa;
  // TAMM_SIZE occ_beta_int  = chem_env.ioptions.ccsd_options.nactive_ob;
  TAMM_SIZE vir_alpha_int = chem_env.ioptions.ccsd_options.nactive_va;
  // TAMM_SIZE vir_beta_int  = chem_env.ioptions.ccsd_options.nactive_vb;

  // Create a list of occupied and virtual combinations
  std::vector<int> occ_list, virt_list;
  for(size_t i = 0; i < nocc_orb; ++i) occ_list.push_back(i);
  for(size_t i = nocc_orb; i < nocc_orb + nvirt_orb; ++i) virt_list.push_back(i);

  std::vector<std::vector<int>> occ_combinations;
  std::vector<std::vector<int>> virt_combinations;
  generate_combinations(occ_list, occ_alpha_int, occ_combinations);
  generate_combinations(virt_list, vir_alpha_int, virt_combinations);

  // Get the orbital energies
  std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);
  if(rank == 0) {
    std::cout << std::endl;
    std::cout << " Orbital | Energy\n";
    std::cout << "---------|--------------\n";
    for(size_t i = 0; i < p_evl_sorted.size(); ++i) {
      std::cout << std::setw(8) << i << " | " << std::fixed << std::setprecision(6) << std::setw(13)
                << p_evl_sorted[i] << "\n";
    }
    std::cout << std::endl;
  }

  // Combine occ_combinations and occ_combinations in all ways
  // The are combined with the sum of orbtital energies
  std::vector<std::pair<std::vector<int>, double>> combinations_with_energy;

  for(const auto& occ_set: occ_combinations) {
    for(const auto& virt_set: virt_combinations) {
      std::vector<int> combined;

      // First nao: original occ_set
      combined.insert(combined.end(), occ_set.begin(), occ_set.end());

      // Next nao: occ_set + nocc_orb
      for(int val: occ_set) combined.push_back(val + nocc_orb);

      // Next nav: virt_set + nocc_orb
      for(int val: virt_set) combined.push_back(val + nocc_orb);

      // Final nav: virt_set + nocc_orb + nvirt_orb
      for(int val: virt_set) combined.push_back(val + nocc_orb + nvirt_orb);

      double occ_sum = 0.0, virt_sum = 0.0;
      for(int j: occ_set) occ_sum += p_evl_sorted[j];
      for(int i: virt_set) virt_sum += p_evl_sorted[i + nocc_orb];

      double orb_e_diff = virt_sum - occ_sum;
      combinations_with_energy.emplace_back(combined, orb_e_diff);

      // if(rank == 0) {
      //   std::cout << "occ_set: [ ";
      //   for(int val: occ_set) { std::cout << val << " "; }
      //   std::cout << "] virt_set: [ ";
      //   for(int val: virt_set) { std::cout << val << " "; }
      //   std::cout << "] | spin orbital combination: [ ";
      //   for(int val: combined) { std::cout << val << " "; }
      //   std::cout << "] | associated energy: " << occ_sum << " " << virt_sum << " " << orb_e_diff
      //   << "\n";
      // }
    }
  }

  // Sort combinations based on orb_e_diff
  std::sort(combinations_with_energy.begin(), combinations_with_energy.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // Print the sorted combinations
  // if(rank == 0) {
  //   std::cout << "\nSorted combinations based on energy difference:\n";
  //   for(const auto& pair: combinations_with_energy) {
  //     std::cout << "Combination: [ ";
  //     for(int val: pair.first) { std::cout << val << " "; }
  //     std::cout << "] | Energy difference: " << pair.second << "\n";
  //   }
  // }

  // Extract sorted combinations only
  std::vector<std::vector<int>> sorted_combinations;
  for(const auto& [combo, _]: combinations_with_energy) { sorted_combinations.push_back(combo); }

  if(rank == 0) std::cout << "Total combinations: " << sorted_combinations.size() << std::endl;
  if(rank == 0) {
    std::cout << "First 5 combinations: " << std::endl;
    for(size_t i = 0; i < 5 && i < sorted_combinations.size(); ++i) {
      std::cout << "Combination: [ ";
      for(int val: sorted_combinations[i]) { std::cout << val << " "; }
      std::cout << "]\n";
    }
  }

  // Main loop
  for(int cycle = 0; cycle < cycles; ++cycle) {
    if(rank == 0) std::cout << "Cycle " << cycle + 1 << " of " << cycles << std::endl;
    for(size_t pos = 0; pos < sorted_combinations.size(); ++pos) {
      const auto& combination = sorted_combinations[pos];
      if(rank == 0) {
        std::cout << "Combination (" << pos + 1 << "/" << sorted_combinations.size() << "): ";
        for(int idx: combination) { std::cout << idx << " "; }
        std::cout << std::endl;
      }

      IndexVector occ_int_vec(combination.begin(), combination.begin() + 2 * occ_alpha_int);
      IndexVector virt_int_vec(combination.end() - 2 * vir_alpha_int, combination.end());

      // Convert pos to a string
      std::ostringstream pos_stream;
      pos_stream << pos;
      std::string pos_str = pos_stream.str();

      // Setup MO indexspace
      auto [MO_AS, total_orbitals1] = setupMOIS_QFlow(ec, chem_env, occ_int_vec, virt_int_vec);

      const TiledIndexSpace& N = MO_AS("all");
      const TiledIndexSpace& O = MO_AS("occ");
      const TiledIndexSpace& V = MO_AS("virt");

      // Retile tensors to the new MO index space
      Tensor<T> cholVpr_UT{{N, N, CI},
                           {SpinPosition::upper, SpinPosition::lower, SpinPosition::ignore}};
      sch.allocate(cholVpr_UT).execute();
      retile_tamm_tensor(cholVpr, cholVpr_UT);
      cholesky_2e::V2Tensors<T> v2tensors_UT =
        cholesky_2e::setupV2Tensors<T>(ec, cholVpr_UT, ex_hw);
      free_tensors(cholVpr_UT);

      Tensor<T> dt1_full_ut = Tensor<T>{{V, O}, {1, 1}};
      Tensor<T> dt2_full_ut = Tensor<T>{{V, V, O, O}, {2, 2}};
      sch.allocate(dt1_full_ut, dt2_full_ut).execute();

      // Initialize T1 and T2
      sch(dt1_full_ut() = 0.0)(dt2_full_ut() = 0.0).execute(ex_hw);

      Tensor<T> d_f1_UT = Tensor<T>{{N, N}, {1, 1}};
      sch.allocate(d_f1_UT).execute();
      retile_tamm_tensor(d_f1, d_f1_UT);

      // Call DUCC
      DUCC_T_CCSD_Driver<T>(chem_env, ec, MO_AS, dt1_full_ut, dt2_full_ut, d_f1_UT, v2tensors_UT,
                            occ_int_vec, virt_int_vec, pos_str);

      free_tensors(d_f1_UT, dt1_full_ut, dt2_full_ut);
    }
  }

  v2tensors.deallocate();
  free_tensors(cholVpr, d_t1, d_t2, d_f1);

  print_memory_usage<T>(ec.pg().rank().value(), "DUCC Memory Stats");

  ec.flush_and_sync();
}

template<typename T>
void DUCC_T_QFLOW_Driver(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
                         const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
                         const Tensor<T>& vtijkl, const Tensor<T>& vtijka, const Tensor<T>& vtaijb,
                         const Tensor<T>& vtijab, const Tensor<T>& vtiabc, const Tensor<T>& vtabcd,
                         ExecutionHW ex_hw, T shift, IndexVector& occ_int_vec,
                         IndexVector& virt_int_vec, string& pos_str) {
  const auto   rank   = sch.ec().pg().rank();
  const size_t nactoa = chem_env.ioptions.ccsd_options.nactive_oa;
  // const size_t nactob = chem_env.ioptions.ccsd_options.nactive_ob;
  const size_t nactva = chem_env.ioptions.ccsd_options.nactive_va;
  // const size_t nactvb = chem_env.ioptions.ccsd_options.nactive_vb;
  std::string files_prefix = chem_env.get_files_prefix("", "ducc");
  auto        rep_energy   = chem_env.scf_context.nuc_repl_energy;

  if(rank == 0) {
    std::cout << "occ_int_vec: ";
    for(const auto& val: occ_int_vec) { std::cout << val << " "; }
    std::cout << std::endl;

    std::cout << "virt_int_vec: ";
    for(const auto& val: virt_int_vec) { std::cout << val << " "; }
    std::cout << std::endl;
  }

  if(rank == 0) {
    std::cout << std::endl
              << "Size of occ_int_vec: " << occ_int_vec.size() << std::endl
              << "Size of virt_int_vec: " << virt_int_vec.size() << std::endl
              << "Total Size of occ_int_vec + virt_int_vec: "
              << occ_int_vec.size() + virt_int_vec.size() << std::endl;
  }

  int nasso = occ_int_vec.size() + virt_int_vec.size(); // Number of active-space spin orbitals
  std::vector<int> XACC_order(nasso);

  for(size_t i = 0; i < nactoa; i++) { XACC_order[i] = occ_int_vec[i]; }
  for(size_t i = 0; i < nactva; i++) { XACC_order[i + nactoa] = virt_int_vec[i]; }
  for(size_t i = 0; i < nactoa; i++) { XACC_order[i + nactoa + nactva] = occ_int_vec[nactoa + i]; }
  for(size_t i = 0; i < nactva; i++) {
    XACC_order[i + nactoa + nactva + nactoa] = virt_int_vec[nactva + i];
  }

  if(rank == 0) {
    std::cout << std::endl << "XACC order: " << std::endl;
    for(int i = 0; i < nasso; i++) { std::cout << XACC_order[i] << " "; }
    std::cout << std::endl;
  }

  std::ostringstream ham_str;
  if(rank == 0) { ham_str << std::setprecision(12) << std::endl; }

  // ij
  Matrix ftij_eigen = tamm_to_eigen_matrix(ftij);
  if(rank == 0) {
    for(size_t i = 0; i < occ_int_vec.size(); i++) {
      for(size_t j = 0; j < occ_int_vec.size(); j++) {
        T    value  = ftij_eigen(i, j);
        auto it_i   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[i]);
        auto it_j   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[j]);
        int  xacc_i = std::distance(XACC_order.begin(), it_i);
        int  xacc_j = std::distance(XACC_order.begin(), it_j);
        if(std::abs(value) > 0.00000001) {
          ham_str << "(" << value << ",0)" << xacc_i << "^ " << xacc_j << " +" << std::endl;
        }
      }
    }
  }

  // ia
  Matrix ftia_eigen = tamm_to_eigen_matrix(ftia);
  if(rank == 0) {
    for(size_t i = 0; i < occ_int_vec.size(); i++) {
      for(size_t a = 0; a < virt_int_vec.size(); a++) {
        T    value  = ftia_eigen(i, a);
        auto it_i   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[i]);
        auto it_a   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[a]);
        int  xacc_i = std::distance(XACC_order.begin(), it_i);
        int  xacc_a = std::distance(XACC_order.begin(), it_a);
        if(std::abs(value) > 0.00000001) {
          ham_str << "(" << value << ",0)" << xacc_i << "^ " << xacc_a << " +" << std::endl;
          ham_str << "(" << value << ",0)" << xacc_a << "^ " << xacc_i << " +" << std::endl;
        }
      }
    }
  }

  // ab
  Matrix ftab_eigen = tamm_to_eigen_matrix(ftab);
  if(rank == 0) {
    for(size_t a = 0; a < virt_int_vec.size(); a++) {
      for(size_t b = 0; b < virt_int_vec.size(); b++) {
        T    value  = ftab_eigen(a, b);
        auto it_a   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[a]);
        auto it_b   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[b]);
        int  xacc_a = std::distance(XACC_order.begin(), it_a);
        int  xacc_b = std::distance(XACC_order.begin(), it_b);
        if(std::abs(value) > 0.00000001) {
          ham_str << "(" << value << ",0)" << xacc_a << "^ " << xacc_b << " +" << std::endl;
        }
      }
    }
  }

  // ijkl
  Tensor4D vtijkl_eigen = tamm_to_eigen_tensor<T, 4>(vtijkl);
  if(rank == 0) {
    for(size_t i = 0; i < occ_int_vec.size(); i++) {
      for(size_t j = 0; j < occ_int_vec.size(); j++) {
        for(size_t k = 0; k < occ_int_vec.size(); k++) {
          for(size_t l = 0; l < occ_int_vec.size(); l++) {
            T    value  = vtijkl_eigen(i, j, k, l);
            auto it_i   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[i]);
            auto it_j   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[j]);
            auto it_k   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[k]);
            auto it_l   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[l]);
            int  xacc_i = std::distance(XACC_order.begin(), it_i);
            int  xacc_j = std::distance(XACC_order.begin(), it_j);
            int  xacc_k = std::distance(XACC_order.begin(), it_k);
            int  xacc_l = std::distance(XACC_order.begin(), it_l);
            if(std::abs(value) > 0.00000001) {
              ham_str << "(" << value * 0.25 << ",0)" << xacc_i << "^ " << xacc_j << "^ " << xacc_l
                      << " " << xacc_k << " +" << std::endl;
            }
          }
        }
      }
    }
  }

  // ijka
  Tensor4D vtijka_eigen = tamm_to_eigen_tensor<T, 4>(vtijka);
  if(rank == 0) {
    for(size_t i = 0; i < occ_int_vec.size(); i++) {
      for(size_t j = 0; j < occ_int_vec.size(); j++) {
        for(size_t k = 0; k < occ_int_vec.size(); k++) {
          for(size_t a = 0; a < virt_int_vec.size(); a++) {
            T    value  = vtijka_eigen(i, j, k, a);
            auto it_i   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[i]);
            auto it_j   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[j]);
            auto it_k   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[k]);
            auto it_a   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[a]);
            int  xacc_i = std::distance(XACC_order.begin(), it_i);
            int  xacc_j = std::distance(XACC_order.begin(), it_j);
            int  xacc_k = std::distance(XACC_order.begin(), it_k);
            int  xacc_a = std::distance(XACC_order.begin(), it_a);
            if(std::abs(value) > 0.00000001) {
              ham_str << "(" << value * 0.25 << ",0)" << xacc_i << "^ " << xacc_j << "^ " << xacc_a
                      << " " << xacc_k << " +" << std::endl;
              ham_str << "(" << value * 0.25 << ",0)" << xacc_j << "^ " << xacc_i << "^ " << xacc_k
                      << " " << xacc_a << " +" << std::endl;
              ham_str << "(" << value * 0.25 << ",0)" << xacc_k << "^ " << xacc_a << "^ " << xacc_j
                      << " " << xacc_i << " +" << std::endl;
              ham_str << "(" << value * 0.25 << ",0)" << xacc_a << "^ " << xacc_k << "^ " << xacc_i
                      << " " << xacc_j << " +" << std::endl;
            }
          }
        }
      }
    }
  }

  // aijb
  Tensor4D vtaijb_eigen = tamm_to_eigen_tensor<T, 4>(vtaijb);
  if(rank == 0) {
    for(size_t a = 0; a < virt_int_vec.size(); a++) {
      for(size_t i = 0; i < occ_int_vec.size(); i++) {
        for(size_t j = 0; j < occ_int_vec.size(); j++) {
          for(size_t b = 0; b < virt_int_vec.size(); b++) {
            T    value  = vtaijb_eigen(a, i, j, b);
            auto it_a   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[a]);
            auto it_i   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[i]);
            auto it_j   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[j]);
            auto it_b   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[b]);
            int  xacc_a = std::distance(XACC_order.begin(), it_a);
            int  xacc_i = std::distance(XACC_order.begin(), it_i);
            int  xacc_j = std::distance(XACC_order.begin(), it_j);
            int  xacc_b = std::distance(XACC_order.begin(), it_b);
            if(std::abs(value) > 0.00000001) {
              ham_str << "(" << value * 0.25 << ",0)" << xacc_a << "^ " << xacc_i << "^ " << xacc_b
                      << " " << xacc_j << " +" << std::endl;
              ham_str << "(" << value * 0.25 << ",0)" << xacc_j << "^ " << xacc_b << "^ " << xacc_i
                      << " " << xacc_a << " +" << std::endl;
              ham_str << "(" << value * -0.25 << ",0)" << xacc_i << "^ " << xacc_a << "^ " << xacc_b
                      << " " << xacc_j << " +" << std::endl;
              ham_str << "(" << value * -0.25 << ",0)" << xacc_a << "^ " << xacc_i << "^ " << xacc_j
                      << " " << xacc_b << " +" << std::endl;
            }
          }
        }
      }
    }
  }

  // ijab
  Tensor4D vtijab_eigen = tamm_to_eigen_tensor<T, 4>(vtijab);
  if(rank == 0) {
    for(size_t i = 0; i < occ_int_vec.size(); i++) {
      for(size_t j = 0; j < occ_int_vec.size(); j++) {
        for(size_t a = 0; a < virt_int_vec.size(); a++) {
          for(size_t b = 0; b < virt_int_vec.size(); b++) {
            T    value  = vtijab_eigen(i, j, a, b);
            auto it_i   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[i]);
            auto it_j   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[j]);
            auto it_a   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[a]);
            auto it_b   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[b]);
            int  xacc_i = std::distance(XACC_order.begin(), it_i);
            int  xacc_j = std::distance(XACC_order.begin(), it_j);
            int  xacc_a = std::distance(XACC_order.begin(), it_a);
            int  xacc_b = std::distance(XACC_order.begin(), it_b);
            if(std::abs(value) > 0.00000001) {
              ham_str << "(" << value * 0.25 << ",0)" << xacc_i << "^ " << xacc_j << "^ " << xacc_b
                      << " " << xacc_a << " +" << std::endl;
              ham_str << "(" << value * 0.25 << ",0)" << xacc_a << "^ " << xacc_b << "^ " << xacc_j
                      << " " << xacc_i << " +" << std::endl;
            }
          }
        }
      }
    }
  }

  // iabc
  Tensor4D vtiabc_eigen = tamm_to_eigen_tensor<T, 4>(vtiabc);
  if(rank == 0) {
    for(size_t i = 0; i < occ_int_vec.size(); i++) {
      for(size_t a = 0; a < virt_int_vec.size(); a++) {
        for(size_t b = 0; b < virt_int_vec.size(); b++) {
          for(size_t c = 0; c < virt_int_vec.size(); c++) {
            T    value  = vtiabc_eigen(i, a, b, c);
            auto it_i   = std::find(XACC_order.begin(), XACC_order.end(), occ_int_vec[i]);
            auto it_a   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[a]);
            auto it_b   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[b]);
            auto it_c   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[c]);
            int  xacc_i = std::distance(XACC_order.begin(), it_i);
            int  xacc_a = std::distance(XACC_order.begin(), it_a);
            int  xacc_b = std::distance(XACC_order.begin(), it_b);
            int  xacc_c = std::distance(XACC_order.begin(), it_c);
            if(std::abs(value) > 0.00000001) {
              ham_str << "(" << value * 0.25 << ",0)" << xacc_i << "^ " << xacc_a << "^ " << xacc_c
                      << " " << xacc_b << " +" << std::endl;
              ham_str << "(" << value * 0.25 << ",0)" << xacc_a << "^ " << xacc_i << "^ " << xacc_b
                      << " " << xacc_c << " +" << std::endl;
              ham_str << "(" << value * 0.25 << ",0)" << xacc_b << "^ " << xacc_c << "^ " << xacc_a
                      << " " << xacc_i << " +" << std::endl;
              ham_str << "(" << value * 0.25 << ",0)" << xacc_c << "^ " << xacc_b << "^ " << xacc_i
                      << " " << xacc_a << " +" << std::endl;
            }
          }
        }
      }
    }
  }

  // abcd
  Tensor4D vtabcd_eigen = tamm_to_eigen_tensor<T, 4>(vtabcd);
  if(rank == 0) {
    for(size_t a = 0; a < virt_int_vec.size(); a++) {
      for(size_t b = 0; b < virt_int_vec.size(); b++) {
        for(size_t c = 0; c < virt_int_vec.size(); c++) {
          for(size_t d = 0; d < virt_int_vec.size(); d++) {
            T    value  = vtabcd_eigen(a, b, c, d);
            auto it_a   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[a]);
            auto it_b   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[b]);
            auto it_c   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[c]);
            auto it_d   = std::find(XACC_order.begin(), XACC_order.end(), virt_int_vec[d]);
            int  xacc_a = std::distance(XACC_order.begin(), it_a);
            int  xacc_b = std::distance(XACC_order.begin(), it_b);
            int  xacc_c = std::distance(XACC_order.begin(), it_c);
            int  xacc_d = std::distance(XACC_order.begin(), it_d);
            if(std::abs(value) > 0.00000001) {
              ham_str << "(" << value * 0.25 << ",0)" << xacc_a << "^ " << xacc_b << "^ " << xacc_d
                      << " " << xacc_c << " +" << std::endl;
            }
          }
        }
      }
    }
  }

  // Scalar
  if(rank == 0) {
    ham_str << "(" << std::setprecision(12) << shift + rep_energy << ", 0)" << std::endl
            << std::endl;
    auto          ham_file = files_prefix + ".qflow." + pos_str;
    std::ofstream ham_fp(ham_file, std::ios::out);
    ham_fp << ham_str.str();
    ham_fp.close();
  }
}

using T = double;
template void DUCC_T_QFLOW_Driver<T>(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
                                     const Tensor<T>& ftij, const Tensor<T>& ftia,
                                     const Tensor<T>& ftab, const Tensor<T>& vtijkl,
                                     const Tensor<T>& vtijka, const Tensor<T>& vtaijb,
                                     const Tensor<T>& vtijab, const Tensor<T>& vtiabc,
                                     const Tensor<T>& vtabcd, ExecutionHW ex_hw, T shift,
                                     IndexVector& occ_int_vec, IndexVector& virt_int_vec,
                                     string& pos_str);
} // namespace exachem::cc::ducc
