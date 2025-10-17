/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "tamm/tamm.hpp"
// using namespace tamm;

extern double ccsdt_d1_t2_GetTime;
extern double ccsdt_d1_v2_GetTime;
extern double ccsd_t_data_per_rank;

template<typename T>
class CCSD_T_Fused_Doubles1 {
public:
  // Constructor
  CCSD_T_Fused_Doubles1()  = default;
  ~CCSD_T_Fused_Doubles1() = default;

  // Copy constructor
  CCSD_T_Fused_Doubles1(const CCSD_T_Fused_Doubles1&) = default;
  // Copy assignment operator
  CCSD_T_Fused_Doubles1& operator=(const CCSD_T_Fused_Doubles1&) = default;

  // Move constructor
  CCSD_T_Fused_Doubles1(CCSD_T_Fused_Doubles1&&) = default;
  // Move assignment operator
  CCSD_T_Fused_Doubles1& operator=(CCSD_T_Fused_Doubles1&&) = default;

  // doubles1 data driver
  void ccsd_t_data_d1_new(bool is_restricted,
                          // ExecutionContext& ec, const TiledIndexSpace& MO,
                          const Index noab, const Index nvab, std::vector<int>& k_spin,
                          // std::vector<size_t>& k_offset,
                          Tensor<T>& d_t1, Tensor<T>& d_t2,
                          exachem::cholesky_2e::V2Tensors<T>& d_v2, std::vector<T>& k_evl_sorted,
                          std::vector<size_t>& k_range, size_t t_h1b, size_t t_h2b, size_t t_h3b,
                          size_t t_p4b, size_t t_p5b, size_t t_p6b, size_t max_d1_kernels_pertask,
                          //
                          size_t size_T_d1_t2, size_t size_T_d1_v2, T* df_T_d1_t2, T* df_T_d1_v2,
                          //
                          int* host_d1_size_h7b,
                          //
                          int* df_simple_d1_size, int* df_simple_d1_exec, int* df_num_d1_enabled,
                          //
                          LRUCache<Index, std::vector<T>>& cache_d1t,
                          LRUCache<Index, std::vector<T>>& cache_d1v) {
    size_t abuf_size1 = size_T_d1_t2; // k_abuf1.size();
    size_t bbuf_size1 = size_T_d1_v2; // k_bbuf1.size();

    std::tuple<Index, Index, Index, Index, Index, Index> a3_d1[] = {
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p5b, t_p6b, t_p4b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p5b, t_p6b, t_p4b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p5b, t_p6b, t_p4b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p4b, t_p6b, t_p5b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p6b, t_p5b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p4b, t_p6b, t_p5b, t_h3b, t_h1b, t_h2b)};

    for(auto ia6 = 0; ia6 < 8; ia6++) {
      if(std::get<0>(a3_d1[ia6]) != 0) {
        for(auto ja6 = ia6 + 1; ja6 < 9; ja6++) { // TODO: ja6 start ?
          if(a3_d1[ia6] == a3_d1[ja6]) { a3_d1[ja6] = std::make_tuple(0, 0, 0, 0, 0, 0); }
        }
      }
    }

    const size_t max_dima = abuf_size1 / max_d1_kernels_pertask;
    const size_t max_dimb = bbuf_size1 / max_d1_kernels_pertask;

    for(Index h7b = 0; h7b < noab; h7b++) {
      df_simple_d1_size[0 + (h7b) *7] = (int) k_range[t_h1b];
      df_simple_d1_size[1 + (h7b) *7] = (int) k_range[t_h2b];
      df_simple_d1_size[2 + (h7b) *7] = (int) k_range[t_h3b];
      df_simple_d1_size[3 + (h7b) *7] = (int) k_range[h7b];
      df_simple_d1_size[4 + (h7b) *7] = (int) k_range[t_p4b];
      df_simple_d1_size[5 + (h7b) *7] = (int) k_range[t_p5b];
      df_simple_d1_size[6 + (h7b) *7] = (int) k_range[t_p6b];

      host_d1_size_h7b[h7b] = (int) k_range[h7b];
    }

    std::vector<bool> ia6_enabled(9 * noab, false);

    // ia6 -- compute which variants are enabled
    for(auto ia6 = 0; ia6 < 9; ia6++) {
      auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d1[ia6];

      if(!((p4b <= p5b) && (h2b <= h3b) && p4b != 0)) { continue; }
      if(is_restricted &&
         !(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] + k_spin[h1b] + k_spin[h2b] + k_spin[h3b] !=
           12)) {
        continue;
      }
      if(!(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] == k_spin[h1b] + k_spin[h2b] + k_spin[h3b])) {
        continue;
      }

      for(Index h7b = 0; h7b < noab; h7b++) {
        if(!(k_spin[p4b] + k_spin[p5b] == k_spin[h1b] + k_spin[h7b])) { continue; }
        if(!(k_range[p4b] > 0 && k_range[p5b] > 0 && k_range[p6b] > 0 && k_range[h1b] > 0 &&
             k_range[h2b] > 0 && k_range[h3b] > 0)) {
          continue;
        }
        if(!(h7b <= p6b)) continue;
        ia6_enabled[ia6 * noab + h7b] = true;
      } // end h7b

    } // end ia6

    // ia6 -- compute sizes and permutations
    int idx_offset = 0;
    for(auto ia6 = 0; ia6 < 9; ia6++) {
      auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d1[ia6];

      auto ref_p456_h123 = std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
      auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
      auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
      auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
      auto cur_p645_h123 = std::make_tuple(p6b, p4b, p5b, h1b, h2b, h3b);
      auto cur_p645_h213 = std::make_tuple(p6b, p4b, p5b, h2b, h1b, h3b);
      auto cur_p645_h231 = std::make_tuple(p6b, p4b, p5b, h2b, h3b, h1b);
      auto cur_p465_h123 = std::make_tuple(p4b, p6b, p5b, h1b, h2b, h3b);
      auto cur_p465_h213 = std::make_tuple(p4b, p6b, p5b, h2b, h1b, h3b);
      auto cur_p465_h231 = std::make_tuple(p4b, p6b, p5b, h2b, h3b, h1b);

      for(Index h7b = 0; h7b < noab; h7b++) {
        if(!ia6_enabled[ia6 * noab + h7b]) continue;

        if(ref_p456_h123 == cur_p456_h123) {
          // df_d1_exec[0 + (h7b + (ia6) * noab) * 9] = d1b++;
          df_simple_d1_exec[0 + h7b * 9] = idx_offset;
          // sd_t_d1_exec[0 + (h7b + (ia6) * noab) * 9] = d1b++;
        }

        if(ref_p456_h123 == cur_p456_h213) {
          // df_d1_exec[1 + (h7b + (ia6) * noab) * 9] = d1b++;
          df_simple_d1_exec[1 + h7b * 9] = idx_offset;
          // sd_t_d1_exec[1 + (h7b + (ia6) * noab) * 9] = d1b++;
        }
        if(ref_p456_h123 == cur_p456_h231) {
          // df_d1_exec[2 + (h7b + (ia6) * noab) * 9] = d1b++;
          df_simple_d1_exec[2 + h7b * 9] = idx_offset;
          // sd_t_d1_exec[2 + (h7b + (ia6) * noab) * 9] = d1b++;
        }
        if(ref_p456_h123 == cur_p645_h123) {
          // df_d1_exec[3 + (h7b + (ia6) * noab) * 9] = d1b++;
          df_simple_d1_exec[3 + h7b * 9] = idx_offset;
          // sd_t_d1_exec[3 + (h7b + (ia6) * noab) * 9] = d1b++;
        }
        if(ref_p456_h123 == cur_p645_h213) {
          // df_d1_exec[4 + (h7b + (ia6) * noab) * 9] = d1b++;
          df_simple_d1_exec[4 + h7b * 9] = idx_offset;
          // sd_t_d1_exec[4 + (h7b + (ia6) * noab) * 9] = d1b++;
        }
        if(ref_p456_h123 == cur_p645_h231) {
          // df_d1_exec[5 + (h7b + (ia6) * noab) * 9] = d1b++;
          df_simple_d1_exec[5 + h7b * 9] = idx_offset;
          // sd_t_d1_exec[5 + (h7b + (ia6) * noab) * 9] = d1b++;
        }
        if(ref_p456_h123 == cur_p465_h123) {
          // df_d1_exec[6 + (h7b + (ia6) * noab) * 9] = d1b++;
          df_simple_d1_exec[6 + h7b * 9] = idx_offset;
          // sd_t_d1_exec[6 + (h7b + (ia6) * noab) * 9] = d1b++;
        }
        if(ref_p456_h123 == cur_p465_h213) {
          // df_d1_exec[7 + (h7b + (ia6) * noab) * 9] = d1b++;
          df_simple_d1_exec[7 + h7b * 9] = idx_offset;
          // sd_t_d1_exec[7 + (h7b + (ia6) * noab) * 9] = d1b++;
        }
        if(ref_p456_h123 == cur_p465_h231) {
          // df_d1_exec[8 + (h7b + (ia6) * noab) * 9] = d1b++;
          df_simple_d1_exec[8 + h7b * 9] = idx_offset;
          // sd_t_d1_exec[8 + (h7b + (ia6) * noab) * 9] = d1b++;
        }

        //
        idx_offset++;
      } // h7b
    }   // end ia6

    // ia6 -- get for t2
    //  d1b = 0;
    idx_offset = 0;
    // printf ("[%s] ------------------------------------------------------\n", __func__);
    for(auto ia6 = 0; ia6 < 9; ia6++) {
      auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d1[ia6];

      // auto abuf_start = d1b * max_dima;
      for(Index h7b = 0; h7b < noab; h7b++) {
        if(!ia6_enabled[ia6 * noab + h7b]) continue;

        size_t dim_common = k_range[h7b];
        size_t dima_sort  = k_range[p4b] * k_range[p5b] * k_range[h1b];
        size_t dima       = dim_common * dima_sort;

        std::vector<T> k_a(dima);
        std::vector<T> k_a_sort(dima);

        IndexVector a_bids_minus_sidx = {p4b - noab, p5b - noab, h7b, h1b};
        auto [hit, value]             = cache_d1t.log_access(a_bids_minus_sidx);

        if(hit) {
          // if (false) {
          // std::copy(value.begin(), value.end(), k_abuf1.begin() + d1b * max_dima);
          // d1b += value.size() / max_dima;
          k_a_sort = value;
        }
        else {
          if(h7b < h1b) {
            {
              TimerGuard tg_total{&ccsdt_d1_t2_GetTime};
              ccsd_t_data_per_rank += dima;
              d_t2.get({p4b - noab, p5b - noab, h7b, h1b},
                       k_a); // h1b,h7b,p5b-noab,p4b-noab
            }
            int perm[4] = {3, 1, 0, 2}; // 3,1,0,2
            int size[4] = {(int) k_range[p4b], (int) k_range[p5b], (int) k_range[h7b],
                           (int) k_range[h1b]};

            auto plan = hptt::create_plan(perm, 4, -1.0, &k_a[0], size, NULL, 0, &k_a_sort[0], NULL,
                                          hptt::ESTIMATE, 1, NULL, true);
            plan->execute();
          }
          if(h1b <= h7b) {
            {
              TimerGuard tg_total{&ccsdt_d1_t2_GetTime};
              ccsd_t_data_per_rank += dima;
              d_t2.get({p4b - noab, p5b - noab, h1b, h7b},
                       k_a); // h7b,h1b,p5b-noab,p4b-noab
            }
            int perm[4] = {2, 1, 0, 3}; // 2,1,0,3
            int size[4] = {(int) k_range[p4b], (int) k_range[p5b], (int) k_range[h1b],
                           (int) k_range[h7b]};

            auto plan = hptt::create_plan(perm, 4, 1.0, &k_a[0], size, NULL, 0, &k_a_sort[0], NULL,
                                          hptt::ESTIMATE, 1, NULL, true);
            plan->execute();
          }
          value = k_a_sort;
        }

        {
          // to get a unique t2 according to ia6 and noab.
          std::copy(k_a_sort.begin(), k_a_sort.end(), df_T_d1_t2 + (idx_offset * max_dima));
        }

        idx_offset++;
      } // h7b
    }   // end ia6

    idx_offset = 0;
    for(auto ia6 = 0; ia6 < 9; ia6++) {
      auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d1[ia6];

      for(Index h7b = 0; h7b < noab; h7b++) {
        if(!ia6_enabled[ia6 * noab + h7b]) continue;

        size_t dim_common = k_range[h7b];
        size_t dimb_sort  = k_range[p6b] * k_range[h2b] * k_range[h3b];
        size_t dimb       = dim_common * dimb_sort;

        std::vector<T> k_b_sort(dimb);

        IndexVector b_bids_minus_sidx = {h2b, h3b, h7b, p6b - noab};
        auto [hit, value]             = cache_d1v.log_access(b_bids_minus_sidx);

        if(hit) { k_b_sort = value; }
        else {
          std::vector<T> k_b(dimb);
          {
            TimerGuard tg_total{&ccsdt_d1_v2_GetTime};
            ccsd_t_data_per_rank += dimb;
            d_v2.v2ijka.get({h2b, h3b, h7b, p6b - noab}, k_b); // h7b,p6b,h2b,h3b

            int perm[4] = {2, 3, 0, 1};
            int size[4] = {(int) k_range[h2b], (int) k_range[h3b], (int) k_range[h7b],
                           (int) k_range[p6b]};

            auto plan = hptt::create_plan(perm, 4, 1.0, &k_b[0], size, NULL, 0, &k_b_sort[0], NULL,
                                          hptt::ESTIMATE, 1, NULL, true);
            plan->execute();
          }
          value = k_b_sort;
        }

        {
          // to get a unique v2 according to ia6 and noab
          // printf ("[%s] to get d1_v2 based on ia=%2lu, noab=%2lu >> linearized: %d (%d)\n",
          // __func__, ia6, h7b, idx_offset, idx_offset * max_dimb);
          std::copy(k_b_sort.begin(), k_b_sort.end(), df_T_d1_v2 + (idx_offset * max_dimb));
        }

        idx_offset++;
      } // h7b
    }   // end ia6

    *df_num_d1_enabled = idx_offset;
  } // ccsd_t_data_d1_new

  void ccsd_t_data_d1_info_only(bool is_restricted, const Index noab, const Index nvab,
                                std::vector<int>& k_spin, std::vector<T>& k_evl_sorted,
                                std::vector<size_t>& k_range, size_t t_h1b, size_t t_h2b,
                                size_t t_h3b, size_t t_p4b, size_t t_p5b, size_t t_p6b,
                                //
                                int* df_simple_d1_size, int* df_simple_d1_exec,
                                int* num_enabled_kernels, size_t& comm_data_elems) {
    std::tuple<Index, Index, Index, Index, Index, Index> a3_d1[] = {
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p5b, t_p6b, t_p4b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p5b, t_p6b, t_p4b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p5b, t_p6b, t_p4b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p4b, t_p6b, t_p5b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p6b, t_p5b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p4b, t_p6b, t_p5b, t_h3b, t_h1b, t_h2b)};

    for(auto ia6 = 0; ia6 < 8; ia6++) {
      if(std::get<0>(a3_d1[ia6]) != 0) {
        for(auto ja6 = ia6 + 1; ja6 < 9; ja6++) { // TODO: ja6 start ?
          if(a3_d1[ia6] == a3_d1[ja6]) { a3_d1[ja6] = std::make_tuple(0, 0, 0, 0, 0, 0); }
        }
      }
    }

    for(Index h7b = 0; h7b < noab; h7b++) {
      df_simple_d1_size[0 + (h7b) *7] = (int) k_range[t_h1b];
      df_simple_d1_size[1 + (h7b) *7] = (int) k_range[t_h2b];
      df_simple_d1_size[2 + (h7b) *7] = (int) k_range[t_h3b];
      df_simple_d1_size[3 + (h7b) *7] = (int) k_range[h7b];
      df_simple_d1_size[4 + (h7b) *7] = (int) k_range[t_p4b];
      df_simple_d1_size[5 + (h7b) *7] = (int) k_range[t_p5b];
      df_simple_d1_size[6 + (h7b) *7] = (int) k_range[t_p6b];
    }

    std::vector<bool> ia6_enabled(9 * noab, false);

    // ia6 -- compute which variants are enabled
    for(auto ia6 = 0; ia6 < 9; ia6++) {
      auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d1[ia6];

      if(!((p4b <= p5b) && (h2b <= h3b) && p4b != 0)) { continue; }
      if(is_restricted &&
         !(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] + k_spin[h1b] + k_spin[h2b] + k_spin[h3b] !=
           12)) {
        continue;
      }
      if(!(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] == k_spin[h1b] + k_spin[h2b] + k_spin[h3b])) {
        continue;
      }

      for(Index h7b = 0; h7b < noab; h7b++) {
        if(!(k_spin[p4b] + k_spin[p5b] == k_spin[h1b] + k_spin[h7b])) { continue; }
        if(!(k_range[p4b] > 0 && k_range[p5b] > 0 && k_range[p6b] > 0 && k_range[h1b] > 0 &&
             k_range[h2b] > 0 && k_range[h3b] > 0)) {
          continue;
        }
        if(!(h7b <= p6b)) continue;
        ia6_enabled[ia6 * noab + h7b] = true;
      } // end h7b

    } // end ia6

    // ia6 -- compute sizes and permutations
    int                             idx_offset = 0;
    std::vector<std::array<int, 9>> detailed_stats(noab);

    for(Index idx_noab = 0; idx_noab < noab; idx_noab++) {
      detailed_stats[idx_noab][0] = 0;
      detailed_stats[idx_noab][1] = 0;
      detailed_stats[idx_noab][2] = 0;
      detailed_stats[idx_noab][3] = 0;
      detailed_stats[idx_noab][4] = 0;
      detailed_stats[idx_noab][5] = 0;
      detailed_stats[idx_noab][6] = 0;
      detailed_stats[idx_noab][7] = 0;
      detailed_stats[idx_noab][8] = 0;
    }

    for(auto ia6 = 0; ia6 < 9; ia6++) {
      auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d1[ia6];

      auto ref_p456_h123 = std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
      auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
      auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
      auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
      auto cur_p645_h123 = std::make_tuple(p6b, p4b, p5b, h1b, h2b, h3b);
      auto cur_p645_h213 = std::make_tuple(p6b, p4b, p5b, h2b, h1b, h3b);
      auto cur_p645_h231 = std::make_tuple(p6b, p4b, p5b, h2b, h3b, h1b);
      auto cur_p465_h123 = std::make_tuple(p4b, p6b, p5b, h1b, h2b, h3b);
      auto cur_p465_h213 = std::make_tuple(p4b, p6b, p5b, h2b, h1b, h3b);
      auto cur_p465_h231 = std::make_tuple(p4b, p6b, p5b, h2b, h3b, h1b);

      for(Index h7b = 0; h7b < noab; h7b++) {
        detailed_stats[h7b][0] = 0;
        detailed_stats[h7b][1] = 0;
        detailed_stats[h7b][2] = 0;
        detailed_stats[h7b][3] = 0;
        detailed_stats[h7b][4] = 0;
        detailed_stats[h7b][5] = 0;
        detailed_stats[h7b][6] = 0;
        detailed_stats[h7b][7] = 0;
        detailed_stats[h7b][8] = 0;

        // int idx_new_offset = 0;
        if(!ia6_enabled[ia6 * noab + h7b]) continue;

        size_t dim_common = k_range[h7b];
        size_t dima_sort  = k_range[p4b] * k_range[p5b] * k_range[h1b];
        size_t dimb_sort  = k_range[p6b] * k_range[h2b] * k_range[h3b];
        comm_data_elems += dim_common * (dima_sort + dimb_sort);

        if(ref_p456_h123 == cur_p456_h123) {
          df_simple_d1_exec[0 + h7b * 9] = idx_offset;
          *num_enabled_kernels           = *num_enabled_kernels + 1;
          // idx_new_offset++;
          detailed_stats[h7b][0] = detailed_stats[h7b][0] + 1;
        }
        if(ref_p456_h123 == cur_p456_h213) {
          df_simple_d1_exec[1 + h7b * 9] = idx_offset;
          *num_enabled_kernels           = *num_enabled_kernels + 1;
          // idx_new_offset++;
          detailed_stats[h7b][1] = detailed_stats[h7b][1] + 1;
        }
        if(ref_p456_h123 == cur_p456_h231) {
          df_simple_d1_exec[2 + h7b * 9] = idx_offset;
          *num_enabled_kernels           = *num_enabled_kernels + 1;
          // idx_new_offset++;
          detailed_stats[h7b][2] = detailed_stats[h7b][2] + 1;
        }
        if(ref_p456_h123 == cur_p645_h123) {
          df_simple_d1_exec[3 + h7b * 9] = idx_offset;
          *num_enabled_kernels           = *num_enabled_kernels + 1;
          // idx_new_offset++;
          detailed_stats[h7b][3] = detailed_stats[h7b][3] + 1;
        }
        if(ref_p456_h123 == cur_p645_h213) {
          df_simple_d1_exec[4 + h7b * 9] = idx_offset;
          *num_enabled_kernels           = *num_enabled_kernels + 1;
          // idx_new_offset++;
          detailed_stats[h7b][4] = detailed_stats[h7b][4] + 1;
        }
        if(ref_p456_h123 == cur_p645_h231) {
          df_simple_d1_exec[5 + h7b * 9] = idx_offset;
          *num_enabled_kernels           = *num_enabled_kernels + 1;
          // idx_new_offset++;
          detailed_stats[h7b][5] = detailed_stats[h7b][5] + 1;
        }
        if(ref_p456_h123 == cur_p465_h123) {
          df_simple_d1_exec[6 + h7b * 9] = idx_offset;
          *num_enabled_kernels           = *num_enabled_kernels + 1;
          // idx_new_offset++;
          detailed_stats[h7b][6] = detailed_stats[h7b][6] + 1;
        }
        if(ref_p456_h123 == cur_p465_h213) {
          df_simple_d1_exec[7 + h7b * 9] = idx_offset;
          *num_enabled_kernels           = *num_enabled_kernels + 1;
          // idx_new_offset++;
          detailed_stats[h7b][7] = detailed_stats[h7b][7] + 1;
        }
        if(ref_p456_h123 == cur_p465_h231) {
          df_simple_d1_exec[8 + h7b * 9] = idx_offset;
          *num_enabled_kernels           = *num_enabled_kernels + 1;
          // idx_new_offset++;
          detailed_stats[h7b][8] = detailed_stats[h7b][8] + 1;
        }

        //
        idx_offset++;
        // printf ("[%s] d1, noab=%2d, #: %d >> %d,%d,%d,%d,%d,%d,%d,%d,%d\n", __func__, h7b,
        // idx_new_offset, detailed_stats[h7b][0], detailed_stats[h7b][1], detailed_stats[h7b][2],
        // detailed_stats[h7b][3], detailed_stats[h7b][4], detailed_stats[h7b][5],
        // detailed_stats[h7b][6], detailed_stats[h7b][7], detailed_stats[h7b][8]);
      } // h7b
    }   // end ia6
  }     // ccsd_t_data_d1_info_only

}; // class CCSD_T_Fused_Doubles1

// Backward compatibility: free function wrappers
template<typename T>
void ccsd_t_data_d1_new(bool is_restricted,
                        // ExecutionContext& ec, const TiledIndexSpace& MO,
                        const Index noab, const Index nvab, std::vector<int>& k_spin,
                        // std::vector<size_t>& k_offset,
                        Tensor<T>& d_t1, Tensor<T>& d_t2, exachem::cholesky_2e::V2Tensors<T>& d_v2,
                        std::vector<T>& k_evl_sorted, std::vector<size_t>& k_range, size_t t_h1b,
                        size_t t_h2b, size_t t_h3b, size_t t_p4b, size_t t_p5b, size_t t_p6b,
                        size_t max_d1_kernels_pertask,
                        //
                        size_t size_T_d1_t2, size_t size_T_d1_v2, T* df_T_d1_t2, T* df_T_d1_v2,
                        //
                        int* host_d1_size_h7b,
                        //
                        int* df_simple_d1_size, int* df_simple_d1_exec, int* df_num_d1_enabled,
                        //
                        LRUCache<Index, std::vector<T>>& cache_d1t,
                        LRUCache<Index, std::vector<T>>& cache_d1v) {
  CCSD_T_Fused_Doubles1<T> ccsd_t_obj;
  ccsd_t_obj.ccsd_t_data_d1_new(is_restricted, noab, nvab, k_spin, d_t1, d_t2, d_v2, k_evl_sorted,
                                k_range, t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b,
                                max_d1_kernels_pertask, size_T_d1_t2, size_T_d1_v2, df_T_d1_t2,
                                df_T_d1_v2, host_d1_size_h7b, df_simple_d1_size, df_simple_d1_exec,
                                df_num_d1_enabled, cache_d1t, cache_d1v);
}

template<typename T>
void ccsd_t_data_d1_info_only(bool is_restricted, const Index noab, const Index nvab,
                              std::vector<int>& k_spin, std::vector<T>& k_evl_sorted,
                              std::vector<size_t>& k_range, size_t t_h1b, size_t t_h2b,
                              size_t t_h3b, size_t t_p4b, size_t t_p5b, size_t t_p6b,
                              //
                              int* df_simple_d1_size, int* df_simple_d1_exec,
                              int* num_enabled_kernels, size_t& comm_data_elems) {
  CCSD_T_Fused_Doubles1<T> ccsd_t_obj;
  ccsd_t_obj.ccsd_t_data_d1_info_only(is_restricted, noab, nvab, k_spin, k_evl_sorted, k_range,
                                      t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, df_simple_d1_size,
                                      df_simple_d1_exec, num_enabled_kernels, comm_data_elems);
}
