/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/canonical/ccsd_canonical.hpp"
#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"
#include "exachem/cc/ducc/ducc-t_ccsd.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"

#include <filesystem>
#include <nwqsim_qflow.hpp>
#include <vqe_options.hpp>

namespace fs = std::filesystem;

namespace exachem::cc::ducc {
using namespace exachem::cc::ducc::internal;

// Helper function to generate allowed excitations for a combination
std::pair<std::set<std::pair<size_t, size_t>>, std::set<std::tuple<size_t, size_t, size_t, size_t>>>
generate_allowed_excitations(const std::vector<int>& combination, TAMM_SIZE occ_alpha_int,
                             TAMM_SIZE vir_alpha_int) {
  std::set<std::pair<size_t, size_t>>                  t1_excitations;
  std::set<std::tuple<size_t, size_t, size_t, size_t>> t2_excitations;

  // Extract indices from combination
  std::vector<size_t> occ_alpha(combination.begin(), combination.begin() + occ_alpha_int);
  std::vector<size_t> occ_beta(combination.begin() + occ_alpha_int,
                               combination.begin() + 2 * occ_alpha_int);
  std::vector<size_t> vir_alpha(combination.begin() + 2 * occ_alpha_int,
                                combination.begin() + 2 * occ_alpha_int + vir_alpha_int);
  std::vector<size_t> vir_beta(combination.begin() + 2 * occ_alpha_int + vir_alpha_int,
                               combination.end());

  // Generate single excitations (T1)
  // Alpha singles: occ_alpha -> vir_alpha
  for(size_t o: occ_alpha) {
    for(size_t v: vir_alpha) { t1_excitations.insert(std::make_pair(o, v)); }
  }

  // Beta singles: occ_beta -> vir_beta
  for(size_t o: occ_beta) {
    for(size_t v: vir_beta) { t1_excitations.insert(std::make_pair(o, v)); }
  }

  // Generate double excitations (T2)
  // Alpha-alpha: 2 occ_alpha -> 2 vir_alpha (unrestricted - all combinations)
  for(size_t i = 0; i < occ_alpha.size(); ++i) {
    for(size_t j = 0; j < occ_alpha.size(); ++j) {
      if(i == j) continue; // Skip diagonal
      for(size_t a = 0; a < vir_alpha.size(); ++a) {
        for(size_t b = 0; b < vir_alpha.size(); ++b) {
          if(a == b) continue; // Skip diagonal
          t2_excitations.insert(
            std::make_tuple(occ_alpha[i], occ_alpha[j], vir_alpha[a], vir_alpha[b]));
        }
      }
    }
  }

  // Alpha-beta mixed: All combinations with ordering occ occ virt virt
  for(size_t oa: occ_alpha) {
    for(size_t ob: occ_beta) {
      for(size_t va: vir_alpha) {
        for(size_t vb: vir_beta) {
          // abab: oa, ob, va, vb
          t2_excitations.insert(std::make_tuple(oa, ob, va, vb));
          // abba: oa, ob, vb, va
          t2_excitations.insert(std::make_tuple(oa, ob, vb, va));
          // baab: ob, oa, va, vb
          t2_excitations.insert(std::make_tuple(ob, oa, va, vb));
          // baba: ob, oa, vb, va
          t2_excitations.insert(std::make_tuple(ob, oa, vb, va));
        }
      }
    }
  }

  // Beta-beta: 2 occ_beta -> 2 vir_beta (unrestricted - all combinations)
  for(size_t i = 0; i < occ_beta.size(); ++i) {
    for(size_t j = 0; j < occ_beta.size(); ++j) {
      if(i == j) continue; // Skip diagonal
      for(size_t a = 0; a < vir_beta.size(); ++a) {
        for(size_t b = 0; b < vir_beta.size(); ++b) {
          if(a == b) continue; // Skip diagonal
          t2_excitations.insert(
            std::make_tuple(occ_beta[i], occ_beta[j], vir_beta[a], vir_beta[b]));
        }
      }
    }
  }

  return std::make_pair(t1_excitations, t2_excitations);
}

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

// void generate_combinations(const std::vector<int>& elements, int combination_size,
//                            std::vector<std::vector<int>>& result_combinations) {
//   std::vector<bool> bitmask(combination_size, true); // combination_size ones
//   bitmask.resize(elements.size(), false);            // followed by n - combination_size zeros

//   do {
//     std::vector<int> combination;
//     for(size_t i = 0; i < elements.size(); ++i) {
//       if(bitmask[i]) { combination.push_back(elements[i]); }
//     }
//     result_combinations.push_back(combination);
//   } while(std::prev_permutation(bitmask.begin(), bitmask.end()));
// }

// Function to generate combinations using iterative coverage algorithm
// Covers all 4-element combinations formed by 2 from occ_list and 2 from virt_list
// i.e. this covers all double excitations by covering all possible alpha-beta combinations
//      therefore covering all alpha-alpha and beta-beta combinations and
//      all single excitations as well.
std::vector<std::pair<std::vector<int>, double>>
generate_combinations_iterative_coverage(const std::vector<int>& occ_list, int occ_alpha_int,
                                         const std::vector<int>& virt_list, int vir_alpha_int,
                                         const std::vector<double>& p_evl_sorted,
                                         TAMM_SIZE nocc_orb, TAMM_SIZE nvirt_orb, int rank = 0) {
  using Combo = std::vector<int>;

  // Helper function to generate all combinations of k elements from a vector
  auto generate_all_combinations = [](const std::vector<int>& vec, int k,
                                      std::vector<std::vector<int>>& result) {
    std::function<void(int, std::vector<int>)> backtrack = [&](int              start,
                                                               std::vector<int> current) {
      if(current.size() == (size_t) k) {
        result.push_back(current);
        return;
      }
      for(size_t i = start; i < vec.size(); ++i) {
        current.push_back(vec[i]);
        backtrack(i + 1, current);
        current.pop_back();
      }
    };
    backtrack(0, {});
  };

  // Generate all valid 4-element combinations from 2 from list1 and 2 from list2
  auto generate_4element_combinations = [&](const std::vector<int>& list1,
                                            const std::vector<int>& list2) {
    std::vector<std::vector<int>> comb1, comb2;
    generate_all_combinations(list1, 2, comb1);
    generate_all_combinations(list2, 2, comb2);

    std::set<Combo> result;
    for(const auto& a: comb1) {
      for(const auto& b: comb2) {
        Combo merged = a;
        merged.insert(merged.end(), b.begin(), b.end());
        std::sort(merged.begin(), merged.end());
        result.insert(merged);
      }
    }
    return result;
  };

  // Generate a random subset of size k from a given list using fixed seed
  auto random_subset = [](const std::vector<int>& input, int k, std::mt19937& rng) {
    if(k > static_cast<int>(input.size())) { return input; }
    std::vector<int> shuffled = input;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    return std::vector<int>(shuffled.begin(), shuffled.begin() + k);
  };

  // Form spin orbital combination from occ_set and virt_set
  auto form_spin_orbital_combination = [&](const std::vector<int>& occ_set,
                                           const std::vector<int>& virt_set) {
    std::vector<int> combined;

    // First nao: original occ_set
    combined.insert(combined.end(), occ_set.begin(), occ_set.end());

    // Next nao: occ_set + nocc_orb
    for(int val: occ_set) combined.push_back(val + nocc_orb);

    // Next nav: virt_set + nocc_orb
    for(int val: virt_set) combined.push_back(val + nocc_orb);

    // Final nav: virt_set + nocc_orb + nvirt_orb
    for(int val: virt_set) combined.push_back(val + nocc_orb + nvirt_orb);

    return combined;
  };

  std::vector<std::pair<std::vector<int>, double>> combinations_with_energy;

  // Initialize random number generator with fixed seed for reproducibility
  std::mt19937 rng(42);

  // Generate all expected combinations for verification
  std::set<Combo> all_expected_combos = generate_4element_combinations(occ_list, virt_list);
  std::set<Combo> remaining_combos    = all_expected_combos;

  if(rank == 0) {
    std::cout << "Expected total 4-element combinations: " << all_expected_combos.size()
              << std::endl;
    std::cout << "Starting iterative coverage algorithm..." << std::endl;
  }

  // Iterative coverage loop
  while(!remaining_combos.empty()) {
    // Generate random subsets
    std::vector<int> occ_subset  = random_subset(occ_list, occ_alpha_int, rng);
    std::vector<int> virt_subset = random_subset(virt_list, vir_alpha_int, rng);

    // Generate combinations from the selected subsets
    std::set<Combo> subset_combos = generate_4element_combinations(occ_subset, virt_subset);

    // Check if any of these combinations are in the remaining set
    bool found_match = false;
    for(const auto& combo: subset_combos) {
      if(remaining_combos.count(combo) > 0) {
        found_match = true;
        break;
      }
    }

    // Only process this subset combination if it covers at least one new combination
    if(found_match) {
      // Form spin orbital combination and compute energy difference
      std::vector<int> combined = form_spin_orbital_combination(occ_subset, virt_subset);

      double occ_sum = 0.0, virt_sum = 0.0;
      for(int j: occ_subset) occ_sum += p_evl_sorted[j];
      for(int i: virt_subset) virt_sum += p_evl_sorted[i + nocc_orb];

      double orb_e_diff = virt_sum - occ_sum;

      // Sort the combined indices
      std::sort(combined.begin(), combined.end());

      combinations_with_energy.emplace_back(combined, orb_e_diff);

      // Remove all matching combinations from the remaining set
      for(const auto& combo: subset_combos) { remaining_combos.erase(combo); }
    }
  }

  // Verification step
  std::set<Combo> generated_combos;
  for(const auto& [spin_combo, energy]: combinations_with_energy) {
    // Extract the 4-element combination from the spin orbital combination
    std::vector<int> occ_part(spin_combo.begin(), spin_combo.begin() + occ_alpha_int);

    // For virt_part, need to subtract nocc_orb to get back spatial indices
    std::vector<int> virt_part;
    for(int i = 2 * occ_alpha_int; i < 2 * occ_alpha_int + vir_alpha_int; ++i) {
      virt_part.push_back(spin_combo[i] - nocc_orb);
    }

    std::set<Combo> single_combo_set = generate_4element_combinations(occ_part, virt_part);
    generated_combos.insert(single_combo_set.begin(), single_combo_set.end());
  }

  bool verification_passed = (generated_combos == all_expected_combos);
  if(rank == 0) {
    std::cout << "Coverage verification: " << (verification_passed ? "PASSED" : "FAILED")
              << std::endl;
    std::cout << "Expected: " << all_expected_combos.size()
              << ", Generated: " << generated_combos.size() << std::endl;
    std::cout << "Number of subset selections used: " << combinations_with_energy.size()
              << std::endl;
  }

  return combinations_with_energy;
}

void read_qflow_json(const bool print, std::ifstream& jread, json& jres_qflow,
                     const std::string qflow_restart_json_file) {
  try {
    jread >> jres_qflow;
  } catch(const nlohmann::json::exception& e) {
    if(print) {
      std::cerr << "[QFlow][ERROR] JSON error while reading file: " << qflow_restart_json_file
                << "\n"
                << e.what() << std::endl;
    }
  }
}

void write_qflow_results(ExecutionContext& ec, ChemEnv& chem_env) {
  if(ec.pg().rank() == 0) {
    std::string files_prefix = chem_env.get_files_prefix("", "ducc");
    const auto  cycles       = chem_env.ioptions.ccsd_options.qflow_cycles;
    auto&       qresults     = chem_env.sys_data.results["output"]["QFlow"]["results"];
    qresults.erase("energy");
    qresults.erase("converged");
    qresults.erase("amplitudes");
    qresults.erase("xacc_order");

    for(int cycle = 1; cycle <= cycles; cycle++) {
      // previous final json file will be overwritten, so write info for all cycles if they exist.
      std::string qflow_restart_json_file =
        files_prefix + "_qflow_cycle_" + std::to_string(cycle) + ".json";
      if(fs::exists(qflow_restart_json_file)) {
        json          jres_qflow;
        std::ifstream jread(qflow_restart_json_file);
        read_qflow_json(true, jread, jres_qflow, qflow_restart_json_file);
        auto cstr = "cycle" + std::to_string(cycle);
        chem_env.sys_data.results["output"]["QFlow"]["results"]["final_energies"][cstr] =
          jres_qflow["results"]["energy"];
      }
    } // cycles
    chem_env.write_json_data();
  } // rank 0
  ec.pg().barrier();
}

void ducc_qflow_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T = double;

  CCContext& cc_context      = chem_env.cc_context;
  cc_context.keep.fvt12_full = true;
  cc_context.compute.set(true, true); // compute ft12 and v2 in full

  const bool do_hubbard = chem_env.sys_data.is_hubbard;
  if(do_hubbard) { exachem::cc::ccsd_canonical::ccsd_canonical_driver(ec, chem_env); }
  else { exachem::cc::ccsd::cd_ccsd_driver(ec, chem_env); }

  TiledIndexSpace& MO      = chem_env.is_context.MSO;
  TiledIndexSpace& CI      = chem_env.is_context.CI;
  Tensor<T>        d_f1    = chem_env.cd_context.d_f1;
  Tensor<T>        cholVpr = chem_env.cd_context.cholV2;
  // Tensor<T>                  d_t1      = chem_env.cc_context.d_t1_full;
  // Tensor<T>                  d_t2      = chem_env.cc_context.d_t2_full;
  cholesky_2e::V2Tensors<T>& v2tensors = chem_env.cd_context.v2tensors;

  const int   rank = ec.pg().rank().value();
  Scheduler   sch{ec};
  ExecutionHW ex_hw   = ec.exhw();
  const bool  noprint = chem_env.ioptions.ccsd_options.noprint;
  bool        restart = true;

  // const TiledIndexSpace& N = MO_AS("all");
  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  TiledIndexLabel p1, p2, p3, p4, p5;
  TiledIndexLabel h3, h4, h5, h6;

  std::tie(p1, p2, p3, p4, p5) = MO.labels<5>("virt");
  std::tie(h3, h4, h5, h6)     = MO.labels<4>("occ");

  Tensor<T> dt1 = Tensor<T>{{V, O}, {1, 1}};
  Tensor<T> dt2 = Tensor<T>{{V, V, O, O}, {2, 2}};
  sch.allocate(dt1, dt2).execute();

  const int cycles = chem_env.ioptions.ccsd_options.qflow_cycles; // Number of cycles to run

  TAMM_SIZE total_orbitals = chem_env.sys_data.nmo / 2;  // Spatial Orbitals
  TAMM_SIZE nocc_orb       = chem_env.sys_data.nocc / 2; // Occupied Spatial Orbitals
  TAMM_SIZE nvirt_orb      = total_orbitals - nocc_orb;  // Virtual Spatial Orbitals

  TAMM_SIZE occ_alpha_int = chem_env.ioptions.ccsd_options.nactive_oa;
  // TAMM_SIZE occ_beta_int  = chem_env.ioptions.ccsd_options.nactive_ob;
  TAMM_SIZE vir_alpha_int = chem_env.ioptions.ccsd_options.nactive_va;
  // TAMM_SIZE vir_beta_int  = chem_env.ioptions.ccsd_options.nactive_vb;

  chem_env.sys_data.results["output"]["QFlow"]["results"] = json::object();

  // Get the orbital energies
  std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);
  if(rank == 0) {
    std::cout << std::endl;
    std::cout << " Orbital | Energy" << std::endl;
    std::cout << "---------|--------------" << std::endl;
    for(size_t i = 0; i < p_evl_sorted.size(); ++i) {
      std::cout << std::setw(8) << i << " | " << std::fixed << std::setprecision(6) << std::setw(13)
                << p_evl_sorted[i] << std::endl;
    }
    std::cout << std::endl;
  }

  // Create a list of occupied and virtual combinations
  std::vector<int> occ_list, virt_list;
  for(size_t i = 0; i < nocc_orb; ++i) occ_list.push_back(i);
  for(size_t i = nocc_orb; i < nocc_orb + nvirt_orb; ++i) virt_list.push_back(i);

  // Combine occ_combinations and virt_combinations in all ways
  // They are combined with the sum of orbital energies
  std::vector<std::pair<std::vector<int>, double>> combinations_with_energy =
    generate_combinations_iterative_coverage(occ_list, occ_alpha_int, virt_list, vir_alpha_int,
                                             p_evl_sorted, nocc_orb, nvirt_orb, rank);

  // std::vector<std::vector<int>> occ_combinations;
  // std::vector<std::vector<int>> virt_combinations;
  // generate_combinations(occ_list, occ_alpha_int, occ_combinations);
  // generate_combinations(virt_list, vir_alpha_int, virt_combinations);

  // for(const auto& occ_set: occ_combinations) {
  //   for(const auto& virt_set: virt_combinations) {
  //     std::vector<int> combined;

  //     // First nao: original occ_set
  //     combined.insert(combined.end(), occ_set.begin(), occ_set.end());

  //     // Next nao: occ_set + nocc_orb
  //     for(int val: occ_set) combined.push_back(val + nocc_orb);

  //     // Next nav: virt_set + nocc_orb
  //     for(int val: virt_set) combined.push_back(val + nocc_orb);

  //     // Final nav: virt_set + nocc_orb + nvirt_orb
  //     for(int val: virt_set) combined.push_back(val + nocc_orb + nvirt_orb);

  //     double occ_sum = 0.0, virt_sum = 0.0;
  //     for(int j: occ_set) occ_sum += p_evl_sorted[j];
  //     for(int i: virt_set) virt_sum += p_evl_sorted[i + nocc_orb];

  //     double orb_e_diff = virt_sum - occ_sum;
  //     combinations_with_energy.emplace_back(combined, orb_e_diff);

  //     // if(rank == 0) {
  //     //   std::cout << "occ_set: [ ";
  //     //   for(int val: occ_set) { std::cout << val << " "; }
  //     //   std::cout << "] virt_set: [ ";
  //     //   for(int val: virt_set) { std::cout << val << " "; }
  //     //   std::cout << "] | spin orbital combination: [ ";
  //     //   for(int val: combined) { std::cout << val << " "; }
  //     //   std::cout << "] | associated energy: " << occ_sum << " " << virt_sum << " " <<
  //     orb_e_diff
  //     //   << std::endl;
  //     // }
  //   }
  // }

  // Sort combinations based on orb_e_diff
  std::sort(combinations_with_energy.begin(), combinations_with_energy.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // Print the sorted combinations
  // if(rank == 0) {
  //   std::cout << std::endl << "Sorted combinations based on energy difference:" << std::endl;
  //   for(const auto& pair: combinations_with_energy) {
  //     std::cout << "Combination: [ ";
  //     for(int val: pair.first) { std::cout << val << " "; }
  //     std::cout << "] | Energy difference: " << pair.second << "" << std::endl;
  //   }
  // }

  // Extract sorted combinations only
  std::vector<std::vector<int>> sorted_combinations;
  for(const auto& [combo, _]: combinations_with_energy) { sorted_combinations.push_back(combo); }

  const size_t ncombinations  = sorted_combinations.size();
  cc_context.qf_ncombinations = ncombinations;

  if(rank == 0) std::cout << "Total combinations: " << ncombinations << std::endl;
  if(rank == 0) {
    std::cout << "First 5 combinations: " << std::endl;
    for(size_t i = 0; i < 5 && i < ncombinations; ++i) {
      std::cout << "Combination: [ ";
      for(int val: sorted_combinations[i]) { std::cout << val << " "; }
      std::cout << "]" << std::endl;
    }
  }

  std::string files_prefix = chem_env.get_files_prefix("", "ducc");

  // Predetermine allowed excitations for each combination
  std::vector<std::set<std::pair<size_t, size_t>>>                  allowed_t1s(ncombinations);
  std::vector<std::set<std::tuple<size_t, size_t, size_t, size_t>>> allowed_t2s(ncombinations);

  std::set<std::pair<size_t, size_t>>                  all_updated_t1;
  std::set<std::tuple<size_t, size_t, size_t, size_t>> all_updated_t2;

  if(rank == 0) {
    std::cout << "Predetermining allowed excitations for all combinations..." << std::endl;
  }

  for(size_t pos = 0; pos < ncombinations; ++pos) {
    const auto& combination = sorted_combinations[pos];

    auto [t1_excitations, t2_excitations] =
      generate_allowed_excitations(combination, occ_alpha_int, vir_alpha_int);

    // Only add excitations that haven't been updated yet
    for(const auto& t1: t1_excitations) {
      if(all_updated_t1.find(t1) == all_updated_t1.end()) {
        allowed_t1s[pos].insert(t1);
        all_updated_t1.insert(t1);
      }
    }

    for(const auto& t2: t2_excitations) {
      if(all_updated_t2.find(t2) == all_updated_t2.end()) {
        allowed_t2s[pos].insert(t2);
        all_updated_t2.insert(t2);
      }
    }

    if(rank == 0 && pos < 5) {
      std::cout << "Combination " << pos << ": " << allowed_t1s[pos].size() << " new T1s, "
                << allowed_t2s[pos].size() << " new T2s" << std::endl;
    }
  }

  if(rank == 0) {
    std::cout << "Total T1 excitations: " << all_updated_t1.size() << std::endl;
    std::cout << "Total T2 excitations: " << all_updated_t2.size() << std::endl;
  }

  auto      nranks = ec.pg().size().value();
  const int ppn    = ec.ppn();
  const int nnodes = ec.nnodes();
  // const bool          debug      = chem_env.ioptions.ccsd_options.debug;

  const int qflow_nprocs_pc = chem_env.ioptions.ccsd_options.qflow_nproc_pc;

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  std::string t1file_qflow = cc_context.t1file_qflow;
  std::string t2file_qflow = cc_context.t2file_qflow;

  int cycle_start = 0;

  if(restart) {
    for(int cycle = cycles; cycle >= 1; cycle--) {
      t1file_qflow = cc_context.t1file_qflow + ".cycle" + std::to_string(cycle);
      t2file_qflow = cc_context.t2file_qflow + ".cycle" + std::to_string(cycle);

      if(fs::exists(t1file_qflow) && fs::exists(t2file_qflow)) {
        if(rank == 0) std::cout << "Reading " << t2file_qflow << std::endl;
        read_from_disk(dt1, t1file_qflow);
        read_from_disk(dt2, t2file_qflow);
        cycle_start = cycle;
        if(rank == 0) {
          json        jres_qflow;
          std::string qflow_restart_json_file =
            files_prefix + "_qflow_cycle_" + std::to_string(cycle) + ".json";
          std::ifstream jread(qflow_restart_json_file);
          read_qflow_json(true, jread, jres_qflow, qflow_restart_json_file);
          auto ecycle = jres_qflow["results"]["energy"];
          std::cout << "[Restart] Cycles 1 to " << cycle << " completed." << std::endl;
          std::cout << " --> Final Energy in cycle " << cycle << ": " << ecycle << std::endl;
        }
        break;
      }
    }
  }
  ec.pg().barrier();

  // Main loop
  for(int cycle = cycle_start; cycle < cycles; ++cycle) {
    cc_context.qf_cycle = cycle + 1;
    if(rank == 0) std::cout << "Cycle " << cc_context.qf_cycle << " of " << cycles << std::endl;

    t1file_qflow = cc_context.t1file_qflow + ".cycle" + std::to_string(cc_context.qf_cycle);
    t2file_qflow = cc_context.t2file_qflow + ".cycle" + std::to_string(cc_context.qf_cycle);
    std::string qflow_restart_json_file =
      files_prefix + "_qflow_cycle_" + std::to_string(cc_context.qf_cycle) + ".json";

    /////////////////////////////////////////////////////////////////////////////////////

    int        subranks  = std::floor(nranks / ncombinations);
    const bool no_pg     = (subranks == 0 || subranks == 1);
    int        sub_nodes = 0;

    if(no_pg) {
      subranks  = nranks;
      sub_nodes = nnodes;
    }
    else {
      sub_nodes = subranks / ppn;
      if(subranks % ppn > 0 || sub_nodes == 0) sub_nodes++;
      if(sub_nodes > nnodes) sub_nodes = nnodes;
      subranks = sub_nodes * ppn;
    }

    if(qflow_nprocs_pc > 0) {
      if(nnodes > 1 && (ppn % qflow_nprocs_pc) != 0)
        tamm_terminate("[ERROR] qflow_nprocs_pc should be a muliple of user processes per node");
      if(nnodes == 1) {
        // TODO: This applies only when using GA's progress ranks runtime
        int ga_num_pr = 1;
        if(const char* ga_npr = std::getenv("GA_NUM_PROGRESS_RANKS_PER_NODE")) {
          ga_num_pr = std::atoi(ga_npr);
        }
        if(ga_num_pr > 1)
          tamm_terminate("[ERROR] use of multiple GA progress ranks for a single node qflow "
                         "calculation is not allowed");
      }
      subranks = qflow_nprocs_pc;
    }

    int num_pg = nranks / subranks;

    if(rank == 0) {
      std::cout << "Total number of process groups = " << num_pg << std::endl;
      std::cout << "Total number of combinations = " << ncombinations << std::endl;
      std::cout << "No of processes used to compute each combination = " << subranks << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////////////

    ProcGroup        sub_pg = ProcGroup::create_subgroups(ec.pg(), subranks);
    ExecutionContext sub_ec{sub_pg, DistributionKind::nw, MemoryManagerKind::ga};
    Scheduler        sub_sch{sub_ec};

    AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
    ac->allocate(0);
    int64_t taskcount = 0;
    int64_t next      = -1;
    // int     total_pi_pg = 0;

    int root_ppi = sub_ec.pg().rank().value();
    int pg_id    = rank / subranks;
    if(root_ppi == 0) next = ac->fetch_add(0, 1);
    sub_ec.pg().broadcast(&next, 0);

    const bool qflow_pg_print = root_ppi == 0 && !noprint;

    for(size_t pos = 0; pos < ncombinations; ++pos) {
      if(next == taskcount) {
        std::stringstream qfstr;
        // total_pi_pg++;
        // size_t pi = pi_tbp[pos];
        if(qflow_pg_print) {
          qfstr << std::endl << std::string(60, '=') << std::endl;
          qfstr << "Process group " << pg_id + 1 << " is executing combination " << pos + 1
                << std::endl;
          qfstr << std::string(60, '=') << std::endl;
        }

        const auto& combination = sorted_combinations[pos];
        if(root_ppi == 0) {
          qfstr << "[Cycle " << cycle + 1 << "] "
                << "Combination (" << pos + 1 << "/" << ncombinations << "): ";
          for(int idx: combination) { qfstr << idx << " "; }
          qfstr << std::endl;
        }

        IndexVector occ_int_vec(combination.begin(), combination.begin() + 2 * occ_alpha_int);
        IndexVector virt_int_vec(combination.end() - 2 * vir_alpha_int, combination.end());

        // Convert pos to a string
        std::ostringstream pos_stream;
        pos_stream << (pos + 1);
        std::string pos_str = pos_stream.str();

        // Setup MO indexspace
        auto [MO_AS, total_orbitals1] =
          setupMOIS_QFlow(sub_ec, chem_env, occ_int_vec, virt_int_vec);

        const TiledIndexSpace& N_AS = MO_AS("all");
        const TiledIndexSpace& O_AS = MO_AS("occ");
        const TiledIndexSpace& V_AS = MO_AS("virt");

        // Retile tensors to the new MO index space
        cholesky_2e::V2Tensors<T> v2tensors_sub;
        if(!do_hubbard) {
          Tensor<T> cholVpr_sub{{N_AS, N_AS, CI},
                                {SpinPosition::upper, SpinPosition::lower, SpinPosition::ignore}};
          sub_sch.allocate(cholVpr_sub).execute();
          retile_tamm_tensor(cholVpr, cholVpr_sub);
          v2tensors_sub = cholesky_2e::setupV2Tensors<T>(sub_ec, cholVpr_sub, ex_hw);
          free_tensors(cholVpr_sub);
        }
        else {
          v2tensors_sub.allocate(sub_ec, N_AS);
          retile_tamm_tensor(v2tensors.v2ijab, v2tensors_sub.v2ijab);
          retile_tamm_tensor(v2tensors.v2iajb, v2tensors_sub.v2iajb);
          retile_tamm_tensor(v2tensors.v2ijka, v2tensors_sub.v2ijka);
          retile_tamm_tensor(v2tensors.v2ijkl, v2tensors_sub.v2ijkl);
          retile_tamm_tensor(v2tensors.v2iabc, v2tensors_sub.v2iabc);
          retile_tamm_tensor(v2tensors.v2abcd, v2tensors_sub.v2abcd);
        }

        // Tensor<T> dt1_sub = Tensor<T>{{V, O}, {1, 1}};
        // Tensor<T> dt2_sub = Tensor<T>{{V, V, O, O}, {2, 2}};
        // sub_sch.allocate(dt1_sub, dt2_sub).execute();

        // Initialize T1 and T2
        // from_dense_tensor(dt1_global, dt1_sub);
        // from_dense_tensor(dt2_global, dt2_sub);
        // sub_sch(dt1_sub() = 0.0)(dt2_sub() = 0.0).execute(ex_hw);
        Tensor<T> dt1_sub =
          redistribute_tensor<T>(sub_ec, dt1, (TiledIndexSpaceVec){V_AS, O_AS}, {1, 1});
        Tensor<T> dt2_sub =
          redistribute_tensor<T>(sub_ec, dt2, (TiledIndexSpaceVec){V_AS, V_AS, O_AS, O_AS}, {2, 2});

        Tensor<T> d_f1_sub = Tensor<T>{{N_AS, N_AS}, {1, 1}};
        sub_sch.allocate(d_f1_sub).execute();
        retile_tamm_tensor(d_f1, d_f1_sub);

        // Call DUCC
        DUCCInternal<T> ducc_internal;
        ducc_internal.DUCC_T_CCSD_Driver(chem_env, sub_ec, MO_AS, dt1_sub, dt2_sub, d_f1_sub,
                                         v2tensors_sub, occ_int_vec, virt_int_vec, pos + 1, qfstr);

        std::string qflow_tmp_json_file = files_prefix + "_qflow_tmp_" + pos_str + ".json";

        if(root_ppi == 0) {
          std::ofstream res_file(qflow_tmp_json_file);
          res_file << std::setw(2) << chem_env.sys_data.results["output"]["QFlow"] << std::endl;
          // store energy of last combination of each cycle for restart.
          if(pos + 1 == ncombinations) {
            std::ofstream res_file(qflow_restart_json_file);
            res_file << std::setw(2) << chem_env.sys_data.results["output"]["QFlow"] << std::endl;
            //   chem_env.run_context["QFlow"]["results"]["energy"][cycle] = nwqsim_energy;
          }
        }
        json jres_qflow;
        sub_ec.pg().barrier();
        {
          std::ifstream jread(qflow_tmp_json_file);
          read_qflow_json(root_ppi == 0, jread, jres_qflow, qflow_tmp_json_file);
        }

        // Get QFlow results
        auto& qflow_results = jres_qflow["results"];
        bool  converged     = qflow_results.value("converged", false);
        if(!converged) {
          qfstr << "[NWQSim] VQE did not converge. Skipping this combination." << std::endl;
          if(root_ppi == 0) { std::cout << qfstr.str() << std::endl; }
          free_tensors(d_f1_sub, dt1_sub, dt2_sub, v2tensors_sub);
          if(root_ppi == 0) {
            std::filesystem::remove(qflow_tmp_json_file);
            next = ac->fetch_add(0, 1);
          }
          sub_ec.pg().broadcast(&next, 0);
          if(root_ppi == 0) taskcount++;
          sub_ec.pg().broadcast(&taskcount, 0);
          continue;
        }
        // Get XACC ordering
        std::vector<int> xacc_order;
        xacc_order = qflow_results["xacc_order"].get<std::vector<int>>();
        if(qflow_pg_print) {
          qfstr << "XACC Ordering: ";
          for(const auto& idx: xacc_order) { qfstr << idx << " "; }
          qfstr << std::endl;
        }

        for(const auto& amp: qflow_results["amplitudes"]) {
          if(amp["indices"].size() == 2) {
            size_t v      = xacc_order[amp["indices"][1]];
            size_t o      = xacc_order[amp["indices"][0]];
            auto   t1_key = std::make_pair(o, v);
            if(allowed_t1s[pos].find(t1_key) != allowed_t1s[pos].end()) {
              update_tensor_val(sub_ec, dt1, {v - chem_env.sys_data.nocc, o},
                                -1.0 * (amp["value"].get<double>()));
              update_tensor_val(sub_ec, dt1, {v - chem_env.sys_data.nocc + nvirt_orb, o + nocc_orb},
                                -1.0 * (amp["value"].get<double>()));
            }
          }
          else if(amp["indices"].size() == 4) {
            size_t v1     = xacc_order[amp["indices"][3]];
            size_t v2     = xacc_order[amp["indices"][2]];
            size_t o1     = xacc_order[amp["indices"][1]];
            size_t o2     = xacc_order[amp["indices"][0]];
            auto   t2_key = std::make_tuple(o1, o2, v1, v2);
            if(allowed_t2s[pos].find(t2_key) != allowed_t2s[pos].end()) {
              update_tensor_val(sub_ec, dt2,
                                {v1 - chem_env.sys_data.nocc, v2 - chem_env.sys_data.nocc, o1, o2},
                                -1.0 * (amp["value"].get<double>()));
              update_tensor_val(sub_ec, dt2,
                                {v2 - chem_env.sys_data.nocc, v1 - chem_env.sys_data.nocc, o1, o2},
                                1.0 * (amp["value"].get<double>()));
              update_tensor_val(sub_ec, dt2,
                                {v1 - chem_env.sys_data.nocc, v2 - chem_env.sys_data.nocc, o2, o1},
                                1.0 * (amp["value"].get<double>()));
              update_tensor_val(sub_ec, dt2,
                                {v2 - chem_env.sys_data.nocc, v1 - chem_env.sys_data.nocc, o2, o1},
                                -1.0 * (amp["value"].get<double>()));

              // if xacc_order[amp["indices"][0]] and xacc_order[amp["indices"][1]] are less than
              // occ_alpha_int that is the alpha-alpha part. The beta beta also needs to be filled
              // in.
              if((xacc_order[amp["indices"][0]] < (int) occ_alpha_int) &&
                 (xacc_order[amp["indices"][1]] < (int) occ_alpha_int)) {
                size_t v_1 =
                  xacc_order[amp["indices"][3].get<int>() + occ_alpha_int + vir_alpha_int];
                size_t v_2 =
                  xacc_order[amp["indices"][2].get<int>() + occ_alpha_int + vir_alpha_int];
                size_t o_1 =
                  xacc_order[amp["indices"][1].get<int>() + occ_alpha_int + vir_alpha_int];
                size_t o_2 =
                  xacc_order[amp["indices"][0].get<int>() + occ_alpha_int + vir_alpha_int];

                update_tensor_val(
                  sub_ec, dt2,
                  {v_1 - chem_env.sys_data.nocc, v_2 - chem_env.sys_data.nocc, o_1, o_2},
                  -1.0 * (amp["value"].get<double>()));
                update_tensor_val(
                  sub_ec, dt2,
                  {v_2 - chem_env.sys_data.nocc, v_1 - chem_env.sys_data.nocc, o_1, o_2},
                  1.0 * (amp["value"].get<double>()));
                update_tensor_val(
                  sub_ec, dt2,
                  {v_1 - chem_env.sys_data.nocc, v_2 - chem_env.sys_data.nocc, o_2, o_1},
                  1.0 * (amp["value"].get<double>()));
                update_tensor_val(
                  sub_ec, dt2,
                  {v_2 - chem_env.sys_data.nocc, v_1 - chem_env.sys_data.nocc, o_2, o_1},
                  -1.0 * (amp["value"].get<double>()));
              }
              // else fill in the b,a,b,a part.
              else {
                size_t v_1 =
                  xacc_order[amp["indices"][3].get<int>() - occ_alpha_int - vir_alpha_int];
                size_t v_2 =
                  xacc_order[amp["indices"][2].get<int>() + occ_alpha_int + vir_alpha_int];
                size_t o_1 =
                  xacc_order[amp["indices"][1].get<int>() - occ_alpha_int - vir_alpha_int];
                size_t o_2 =
                  xacc_order[amp["indices"][0].get<int>() + occ_alpha_int + vir_alpha_int];

                update_tensor_val(
                  sub_ec, dt2,
                  {v_1 - chem_env.sys_data.nocc, v_2 - chem_env.sys_data.nocc, o_1, o_2},
                  -1.0 * (amp["value"].get<double>()));
                update_tensor_val(
                  sub_ec, dt2,
                  {v_2 - chem_env.sys_data.nocc, v_1 - chem_env.sys_data.nocc, o_1, o_2},
                  1.0 * (amp["value"].get<double>()));
                update_tensor_val(
                  sub_ec, dt2,
                  {v_1 - chem_env.sys_data.nocc, v_2 - chem_env.sys_data.nocc, o_2, o_1},
                  1.0 * (amp["value"].get<double>()));
                update_tensor_val(
                  sub_ec, dt2,
                  {v_2 - chem_env.sys_data.nocc, v_1 - chem_env.sys_data.nocc, o_2, o_1},
                  -1.0 * (amp["value"].get<double>()));
              }
            }
          }
        }

        if(qflow_pg_print) {
          qfstr << "Process group " << pg_id + 1 << " finished executing combination " << pos + 1
                << std::endl;
          qfstr << std::string(60, '-') << std::endl;
        }
        sub_ec.pg().barrier();

        if(root_ppi == 0) { std::cout << qfstr.str() << std::endl; }

        free_tensors(d_f1_sub, dt1_sub, dt2_sub, v2tensors_sub);

        if(root_ppi == 0) {
          std::filesystem::remove(qflow_tmp_json_file);
          next = ac->fetch_add(0, 1);
        }
        sub_ec.pg().broadcast(&next, 0);
      } // end if next == taskcount
      if(root_ppi == 0) taskcount++;
      sub_ec.pg().broadcast(&taskcount, 0);

    } // end combinations

    sub_ec.flush_and_sync();
    // MemoryManagerGA::destroy_coll(mgr);
    sub_pg.destroy_coll();
    ac->deallocate();
    delete ac;
    ec.pg().barrier();

    if(restart) {
      write_to_disk(dt1, t1file_qflow);
      write_to_disk(dt2, t2file_qflow);
      write_qflow_results(ec, chem_env);
    } // restart

  } // end cycles

  write_qflow_results(ec, chem_env);
  auto cc_t2  = std::chrono::high_resolution_clock::now();
  auto qftime = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  if(rank == 0) { std::cout << "Total QFlow time: " << qftime << std::endl; }

  v2tensors.deallocate();
  if(!do_hubbard) free_tensors(cholVpr);
  free_tensors(d_f1, dt1, dt2);
  // Tensor<T>::deallocate(dt1_global, dt2_global);

  print_memory_usage<T>(ec.pg().rank().value(), "DUCC Memory Stats");

  ec.flush_and_sync();
}

template<typename T>
void DUCC_T_QFLOW_Driver(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
                         const Tensor<T>& ftij, const Tensor<T>& ftia, const Tensor<T>& ftab,
                         const Tensor<T>& vtijkl, const Tensor<T>& vtijka, const Tensor<T>& vtaijb,
                         const Tensor<T>& vtijab, const Tensor<T>& vtiabc, const Tensor<T>& vtabcd,
                         ExecutionHW ex_hw, T shift, IndexVector& occ_int_vec,
                         IndexVector& virt_int_vec, const int pos, std::stringstream& qfstr) {
  const auto   rank   = sch.ec().pg().rank();
  const size_t nactoa = chem_env.ioptions.ccsd_options.nactive_oa;
  // const size_t nactob = chem_env.ioptions.ccsd_options.nactive_ob;
  const size_t nactva = chem_env.ioptions.ccsd_options.nactive_va;
  // const size_t nactvb = chem_env.ioptions.ccsd_options.nactive_vb;
  SystemData& sys_data     = chem_env.sys_data;
  std::string files_prefix = chem_env.get_files_prefix("", "ducc");
  auto        rep_energy   = chem_env.scf_context.nuc_repl_energy;
  const bool  noprint      = chem_env.ioptions.ccsd_options.noprint;
  const bool  qflow_print  = rank == 0 && !noprint;

  if(qflow_print) {
    qfstr << "occ_int_vec: ";
    for(const auto& val: occ_int_vec) { qfstr << val << " "; }
    qfstr << std::endl;

    qfstr << "virt_int_vec: ";
    for(const auto& val: virt_int_vec) { qfstr << val << " "; }
    qfstr << std::endl;
  }

  if(qflow_print) {
    qfstr << std::endl
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

  if(qflow_print) {
    qfstr << std::endl << "XACC order: " << std::endl;
    for(int i = 0; i < nasso; i++) { qfstr << XACC_order[i] << " "; }
    qfstr << std::endl;
  }

  // Commented out original ham_str approach for reference
  // std::ostringstream ham_str;
  // if(qflow_print) { ham_str << std::setprecision(12) << std::endl; }

  std::vector<std::pair<std::string, std::complex<double>>> ham_terms;
  if(qflow_print) { /* No need for precision setting with new approach */
  }

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
          // ham_str << "(" << value << ",0)" << xacc_i << "^ " << xacc_j << " +" << std::endl;
          std::string term_str = std::to_string(xacc_i) + "^ " + std::to_string(xacc_j) + " +";
          ham_terms.emplace_back(term_str, std::complex<double>(value, 0));
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
          // ham_str << "(" << value << ",0)" << xacc_i << "^ " << xacc_a << " +" << std::endl;
          // ham_str << "(" << value << ",0)" << xacc_a << "^ " << xacc_i << " +" << std::endl;
          std::string term_str1 = std::to_string(xacc_i) + "^ " + std::to_string(xacc_a) + " +";
          std::string term_str2 = std::to_string(xacc_a) + "^ " + std::to_string(xacc_i) + " +";
          ham_terms.emplace_back(term_str1, std::complex<double>(value, 0));
          ham_terms.emplace_back(term_str2, std::complex<double>(value, 0));
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
          // ham_str << "(" << value << ",0)" << xacc_a << "^ " << xacc_b << " +" << std::endl;
          std::string term_str = std::to_string(xacc_a) + "^ " + std::to_string(xacc_b) + " +";
          ham_terms.emplace_back(term_str, std::complex<double>(value, 0));
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
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_i << "^ " << xacc_j << "^ " <<
              // xacc_l
              //         << " " << xacc_k << " +" << std::endl;
              std::string term_str = std::to_string(xacc_i) + "^ " + std::to_string(xacc_j) + "^ " +
                                     std::to_string(xacc_l) + " " + std::to_string(xacc_k) + " +";
              ham_terms.emplace_back(term_str, std::complex<double>(value * 0.25, 0));
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
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_i << "^ " << xacc_j << "^ " <<
              // xacc_a
              //         << " " << xacc_k << " +" << std::endl;
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_j << "^ " << xacc_i << "^ " <<
              // xacc_k
              //         << " " << xacc_a << " +" << std::endl;
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_k << "^ " << xacc_a << "^ " <<
              // xacc_j
              //         << " " << xacc_i << " +" << std::endl;
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_a << "^ " << xacc_k << "^ " <<
              // xacc_i
              //         << " " << xacc_j << " +" << std::endl;
              std::string term_str1 = std::to_string(xacc_i) + "^ " + std::to_string(xacc_j) +
                                      "^ " + std::to_string(xacc_a) + " " + std::to_string(xacc_k) +
                                      " +";
              std::string term_str2 = std::to_string(xacc_j) + "^ " + std::to_string(xacc_i) +
                                      "^ " + std::to_string(xacc_k) + " " + std::to_string(xacc_a) +
                                      " +";
              std::string term_str3 = std::to_string(xacc_k) + "^ " + std::to_string(xacc_a) +
                                      "^ " + std::to_string(xacc_j) + " " + std::to_string(xacc_i) +
                                      " +";
              std::string term_str4 = std::to_string(xacc_a) + "^ " + std::to_string(xacc_k) +
                                      "^ " + std::to_string(xacc_i) + " " + std::to_string(xacc_j) +
                                      " +";
              ham_terms.emplace_back(term_str1, std::complex<double>(value * 0.25, 0));
              ham_terms.emplace_back(term_str2, std::complex<double>(value * 0.25, 0));
              ham_terms.emplace_back(term_str3, std::complex<double>(value * 0.25, 0));
              ham_terms.emplace_back(term_str4, std::complex<double>(value * 0.25, 0));
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
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_a << "^ " << xacc_i << "^ " <<
              // xacc_b
              //         << " " << xacc_j << " +" << std::endl;
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_j << "^ " << xacc_b << "^ " <<
              // xacc_i
              //         << " " << xacc_a << " +" << std::endl;
              // ham_str << "(" << value * -0.25 << ",0)" << xacc_i << "^ " << xacc_a << "^ " <<
              // xacc_b
              //         << " " << xacc_j << " +" << std::endl;
              // ham_str << "(" << value * -0.25 << ",0)" << xacc_a << "^ " << xacc_i << "^ " <<
              // xacc_j
              //         << " " << xacc_b << " +" << std::endl;
              std::string term_str1 = std::to_string(xacc_a) + "^ " + std::to_string(xacc_i) +
                                      "^ " + std::to_string(xacc_b) + " " + std::to_string(xacc_j) +
                                      " +";
              std::string term_str2 = std::to_string(xacc_j) + "^ " + std::to_string(xacc_b) +
                                      "^ " + std::to_string(xacc_i) + " " + std::to_string(xacc_a) +
                                      " +";
              std::string term_str3 = std::to_string(xacc_i) + "^ " + std::to_string(xacc_a) +
                                      "^ " + std::to_string(xacc_b) + " " + std::to_string(xacc_j) +
                                      " +";
              std::string term_str4 = std::to_string(xacc_a) + "^ " + std::to_string(xacc_i) +
                                      "^ " + std::to_string(xacc_j) + " " + std::to_string(xacc_b) +
                                      " +";
              ham_terms.emplace_back(term_str1, std::complex<double>(value * 0.25, 0));
              ham_terms.emplace_back(term_str2, std::complex<double>(value * 0.25, 0));
              ham_terms.emplace_back(term_str3, std::complex<double>(value * -0.25, 0));
              ham_terms.emplace_back(term_str4, std::complex<double>(value * -0.25, 0));
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
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_i << "^ " << xacc_j << "^ " <<
              // xacc_b
              //         << " " << xacc_a << " +" << std::endl;
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_a << "^ " << xacc_b << "^ " <<
              // xacc_j
              //         << " " << xacc_i << " +" << std::endl;
              std::string term_str1 = std::to_string(xacc_i) + "^ " + std::to_string(xacc_j) +
                                      "^ " + std::to_string(xacc_b) + " " + std::to_string(xacc_a) +
                                      " +";
              std::string term_str2 = std::to_string(xacc_a) + "^ " + std::to_string(xacc_b) +
                                      "^ " + std::to_string(xacc_j) + " " + std::to_string(xacc_i) +
                                      " +";
              ham_terms.emplace_back(term_str1, std::complex<double>(value * 0.25, 0));
              ham_terms.emplace_back(term_str2, std::complex<double>(value * 0.25, 0));
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
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_i << "^ " << xacc_a << "^ " <<
              // xacc_c
              //         << " " << xacc_b << " +" << std::endl;
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_a << "^ " << xacc_i << "^ " <<
              // xacc_b
              //         << " " << xacc_c << " +" << std::endl;
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_b << "^ " << xacc_c << "^ " <<
              // xacc_a
              //         << " " << xacc_i << " +" << std::endl;
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_c << "^ " << xacc_b << "^ " <<
              // xacc_i
              //         << " " << xacc_a << " +" << std::endl;
              std::string term_str1 = std::to_string(xacc_i) + "^ " + std::to_string(xacc_a) +
                                      "^ " + std::to_string(xacc_c) + " " + std::to_string(xacc_b) +
                                      " +";
              std::string term_str2 = std::to_string(xacc_a) + "^ " + std::to_string(xacc_i) +
                                      "^ " + std::to_string(xacc_b) + " " + std::to_string(xacc_c) +
                                      " +";
              std::string term_str3 = std::to_string(xacc_b) + "^ " + std::to_string(xacc_c) +
                                      "^ " + std::to_string(xacc_a) + " " + std::to_string(xacc_i) +
                                      " +";
              std::string term_str4 = std::to_string(xacc_c) + "^ " + std::to_string(xacc_b) +
                                      "^ " + std::to_string(xacc_i) + " " + std::to_string(xacc_a) +
                                      " +";
              ham_terms.emplace_back(term_str1, std::complex<double>(value * 0.25, 0));
              ham_terms.emplace_back(term_str2, std::complex<double>(value * 0.25, 0));
              ham_terms.emplace_back(term_str3, std::complex<double>(value * 0.25, 0));
              ham_terms.emplace_back(term_str4, std::complex<double>(value * 0.25, 0));
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
              // ham_str << "(" << value * 0.25 << ",0)" << xacc_a << "^ " << xacc_b << "^ " <<
              // xacc_d
              //         << " " << xacc_c << " +" << std::endl;
              std::string term_str = std::to_string(xacc_a) + "^ " + std::to_string(xacc_b) + "^ " +
                                     std::to_string(xacc_d) + " " + std::to_string(xacc_c) + " +";
              ham_terms.emplace_back(term_str, std::complex<double>(value * 0.25, 0));
            }
          }
        }
      }
    }
  }

  // Scalar
  if(rank == 0) {
    // Original scalar term generation:
    // ham_str << "(" << std::setprecision(12) << shift + rep_energy << ", 0)" << std::endl
    //         << std::endl;

    // Add scalar term to the vector
    ham_terms.emplace_back("", std::complex<double>(shift + rep_energy, 0));

    if(qflow_print) {
      qfstr << std::setprecision(6) << std::endl
            << std::endl
            << "NWQSim Running QFlow ... " << std::endl;
    }

    // Setup VQE options
    vqe::vqe_options opts;
    opts.symmetry_level     = 2;
    opts.lower_bound        = -1;
    opts.upper_bound        = 1;
    opts.max_evaluations    = 1000;            // Max number of optimization iterations
    opts.absolute_tolerance = 1e-5;            // -1 for no absolute tolerance
    opts.optimizer          = nlopt::LD_LBFGS; // Use a derivative-based optimizer
    // Note: initial_parameters can be set if needed, but may be optional for default initialization
    // opts.initial_parameters = {0.1, 0.2, 0.3, ...}; // Must match parameter count for the ansatz
    // (UCCSD)

    auto [nwqsim_energy, vqe_converged, nwqsim_parameters] =
      qflow_nwqsim(ham_terms, nactoa * 2, "CPU", opts);

    sys_data.results["output"]["QFlow"]["results"]["converged"] = vqe_converged;

    if(vqe_converged) {
      // Store results in sys_data.results for use elsewhere in the code
      sys_data.results["output"]["QFlow"]["results"]["energy"] = nwqsim_energy;

      // Store amplitudes in sys_data.results
      json amplitudes_json = json::array();
      for(const auto& pair: nwqsim_parameters) {
        const std::vector<int>& vec = pair.first;
        double                  val = pair.second;

        json amp_entry;
        amp_entry["indices"] = vec;
        amp_entry["value"]   = val;
        amplitudes_json.push_back(amp_entry);
      }
      sys_data.results["output"]["QFlow"]["results"]["amplitudes"] = amplitudes_json;

      // Store XACC_order in sys_data.results
      sys_data.results["output"]["QFlow"]["results"]["xacc_order"] = XACC_order;

      qfstr << std::setprecision(10) << "Final Energy: " << nwqsim_energy << "" << std::endl;

      if(qflow_print) {
        // Print the vector of pairs
        qfstr << "Amplitudes:" << std::endl;
        for(const auto& pair: nwqsim_parameters) {
          const std::vector<int>& vec = pair.first;
          double                  val = pair.second;

          qfstr << "  [";
          for(size_t i = 0; i < vec.size(); ++i) {
            qfstr << vec[i];
            if(i < vec.size() - 1) qfstr << ", ";
          }
          qfstr << "] -> " << val << "" << std::endl;
        }
        qfstr << std::endl
              << "NWQSim finished executing combination " << pos << std::endl
              << std::endl;
      }
    } // vqe_converged
  }   // rank==0
}

using T = double;
template void DUCC_T_QFLOW_Driver<T>(Scheduler& sch, ChemEnv& chem_env, const TiledIndexSpace& MO,
                                     const Tensor<T>& ftij, const Tensor<T>& ftia,
                                     const Tensor<T>& ftab, const Tensor<T>& vtijkl,
                                     const Tensor<T>& vtijka, const Tensor<T>& vtaijb,
                                     const Tensor<T>& vtijab, const Tensor<T>& vtiabc,
                                     const Tensor<T>& vtabcd, ExecutionHW ex_hw, T shift,
                                     IndexVector& occ_int_vec, IndexVector& virt_int_vec,
                                     const int pos, std::stringstream& qfstr);
} // namespace exachem::cc::ducc
