/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "fused_common.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#define CEIL(a, b) (((a) + (b) -1) / (b))

template<typename T>
void total_fused_ccsd_t_cpu(
  bool is_restricted, const Index noab, const Index nvab, int64_t rank, std::vector<int>& k_spin,
  std::vector<size_t>& k_range, std::vector<size_t>& k_offset, Tensor<T>& d_t1, Tensor<T>& d_t2,
  exachem::cholesky_2e::V2Tensors<T>& d_v2, std::vector<T>& k_evl_sorted,
  //
  T* df_host_pinned_s1_t1, T* df_host_pinned_s1_v2, T* df_host_pinned_d1_t2,
  T* df_host_pinned_d1_v2, T* df_host_pinned_d2_t2, T* df_host_pinned_d2_v2, T* host_energies,
  // for new fully-fused kernel
  int* host_d1_size_h7b, int* host_d2_size_p7b,
  //
  int* df_simple_s1_size, int* df_simple_d1_size, int* df_simple_d2_size, int* df_simple_s1_exec,
  int* df_simple_d1_exec, int* df_simple_d2_exec,
  //
  size_t t_h1b, size_t t_h2b, size_t t_h3b, size_t t_p4b, size_t t_p5b, size_t t_p6b, double factor,
  size_t taskid, size_t max_d1_kernels_pertask, size_t max_d2_kernels_pertask,
  //
  size_t size_T_s1_t1, size_t size_T_s1_v2, size_t size_T_d1_t2, size_t size_T_d1_v2,
  size_t size_T_d2_t2, size_t size_T_d2_v2,
  //
  std::vector<double>& energy_l, LRUCache<Index, std::vector<T>>& cache_s1t,
  LRUCache<Index, std::vector<T>>& cache_s1v, LRUCache<Index, std::vector<T>>& cache_d1t,
  LRUCache<Index, std::vector<T>>& cache_d1v, LRUCache<Index, std::vector<T>>& cache_d2t,
  LRUCache<Index, std::vector<T>>& cache_d2v)

{
  size_t base_size_h1b = k_range[t_h1b];
  size_t base_size_h2b = k_range[t_h2b];
  size_t base_size_h3b = k_range[t_h3b];
  size_t base_size_p4b = k_range[t_p4b];
  size_t base_size_p5b = k_range[t_p5b];
  size_t base_size_p6b = k_range[t_p6b];

  const size_t max_dim_s1_t1 = size_T_s1_t1 / 9;
  const size_t max_dim_s1_v2 = size_T_s1_v2 / 9;
  const size_t max_dim_d1_t2 = size_T_d1_t2 / max_d1_kernels_pertask;
  const size_t max_dim_d1_v2 = size_T_d1_v2 / max_d1_kernels_pertask;
  const size_t max_dim_d2_t2 = size_T_d2_t2 / max_d2_kernels_pertask;
  const size_t max_dim_d2_v2 = size_T_d2_v2 / max_d2_kernels_pertask;

  int df_num_s1_enabled;
  int df_num_d1_enabled;
  int df_num_d2_enabled;

  double* host_evl_sorted_h1b = &k_evl_sorted[k_offset[t_h1b]];
  double* host_evl_sorted_h2b = &k_evl_sorted[k_offset[t_h2b]];
  double* host_evl_sorted_h3b = &k_evl_sorted[k_offset[t_h3b]];
  double* host_evl_sorted_p4b = &k_evl_sorted[k_offset[t_p4b]];
  double* host_evl_sorted_p5b = &k_evl_sorted[k_offset[t_p5b]];
  double* host_evl_sorted_p6b = &k_evl_sorted[k_offset[t_p6b]];

  std::fill(df_simple_s1_exec, df_simple_s1_exec + (9), -1);
  std::fill(df_simple_d1_exec, df_simple_d1_exec + (9 * noab), -1);
  std::fill(df_simple_d2_exec, df_simple_d2_exec + (9 * nvab), -1);

  //
  ccsd_t_data_s1_new(is_restricted, noab, nvab, k_spin, d_t1, d_t2, d_v2, k_evl_sorted, k_range,
                     t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b,
                     //
                     size_T_s1_t1, size_T_s1_v2, df_simple_s1_size, df_simple_s1_exec,
                     df_host_pinned_s1_t1, df_host_pinned_s1_v2, &df_num_s1_enabled,
                     //
                     cache_s1t, cache_s1v);

  //
  ccsd_t_data_d1_new(is_restricted, noab, nvab, k_spin, d_t1, d_t2, d_v2, k_evl_sorted, k_range,
                     t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, max_d1_kernels_pertask,
                     //
                     size_T_d1_t2, size_T_d1_v2, df_host_pinned_d1_t2, df_host_pinned_d1_v2,
                     host_d1_size_h7b, df_simple_d1_size, df_simple_d1_exec, &df_num_d1_enabled,
                     //
                     cache_d1t, cache_d1v);

  //
  ccsd_t_data_d2_new(is_restricted, noab, nvab, k_spin, d_t1, d_t2, d_v2, k_evl_sorted, k_range,
                     t_h1b, t_h2b, t_h3b, t_p4b, t_p5b, t_p6b, max_d2_kernels_pertask,
                     //
                     size_T_d2_t2, size_T_d2_v2, df_host_pinned_d2_t2, df_host_pinned_d2_v2,
                     host_d2_size_p7b, df_simple_d2_size, df_simple_d2_exec, &df_num_d2_enabled,
                     //
                     cache_d2t, cache_d2v);

  //
  size_t size_tensor_t3 =
    base_size_h3b * base_size_h2b * base_size_h1b * base_size_p6b * base_size_p5b * base_size_p4b;

  //
  double* host_t3_d = (double*) malloc(sizeof(double) * size_tensor_t3);
  double* host_t3_s = (double*) malloc(sizeof(double) * size_tensor_t3);

  for(size_t i = 0; i < size_tensor_t3; i++) {
    host_t3_d[i] = 0.000;
    host_t3_s[i] = 0.000;
  }

  //
  // for (size_t idx_ia6 = 0; idx_ia6 < 9; idx_ia6++){
  // d1
  for(size_t idx_noab = 0; idx_noab < noab; idx_noab++) {
    int flag_d1_1 = (int) df_simple_d1_exec[0 + (idx_noab) *9];
    int flag_d1_2 = (int) df_simple_d1_exec[1 + (idx_noab) *9];
    int flag_d1_3 = (int) df_simple_d1_exec[2 + (idx_noab) *9];
    int flag_d1_4 = (int) df_simple_d1_exec[3 + (idx_noab) *9];
    int flag_d1_5 = (int) df_simple_d1_exec[4 + (idx_noab) *9];
    int flag_d1_6 = (int) df_simple_d1_exec[5 + (idx_noab) *9];
    int flag_d1_7 = (int) df_simple_d1_exec[6 + (idx_noab) *9];
    int flag_d1_8 = (int) df_simple_d1_exec[7 + (idx_noab) *9];
    int flag_d1_9 = (int) df_simple_d1_exec[8 + (idx_noab) *9];

    int d1_base_size_h1b = (int) df_simple_d1_size[0 + (idx_noab) *7];
    int d1_base_size_h2b = (int) df_simple_d1_size[1 + (idx_noab) *7];
    int d1_base_size_h3b = (int) df_simple_d1_size[2 + (idx_noab) *7];
    int d1_base_size_h7b = (int) df_simple_d1_size[3 + (idx_noab) *7];
    int d1_base_size_p4b = (int) df_simple_d1_size[4 + (idx_noab) *7];
    int d1_base_size_p5b = (int) df_simple_d1_size[5 + (idx_noab) *7];
    int d1_base_size_p6b = (int) df_simple_d1_size[6 + (idx_noab) *7];

    double* host_d1_t2_1 = df_host_pinned_d1_t2 + max_dim_d1_t2 * flag_d1_1;
    double* host_d1_v2_1 = df_host_pinned_d1_v2 + max_dim_d1_v2 * flag_d1_1;
    double* host_d1_t2_2 = df_host_pinned_d1_t2 + max_dim_d1_t2 * flag_d1_2;
    double* host_d1_v2_2 = df_host_pinned_d1_v2 + max_dim_d1_v2 * flag_d1_2;
    double* host_d1_t2_3 = df_host_pinned_d1_t2 + max_dim_d1_t2 * flag_d1_3;
    double* host_d1_v2_3 = df_host_pinned_d1_v2 + max_dim_d1_v2 * flag_d1_3;
    double* host_d1_t2_4 = df_host_pinned_d1_t2 + max_dim_d1_t2 * flag_d1_4;
    double* host_d1_v2_4 = df_host_pinned_d1_v2 + max_dim_d1_v2 * flag_d1_4;
    double* host_d1_t2_5 = df_host_pinned_d1_t2 + max_dim_d1_t2 * flag_d1_5;
    double* host_d1_v2_5 = df_host_pinned_d1_v2 + max_dim_d1_v2 * flag_d1_5;
    double* host_d1_t2_6 = df_host_pinned_d1_t2 + max_dim_d1_t2 * flag_d1_6;
    double* host_d1_v2_6 = df_host_pinned_d1_v2 + max_dim_d1_v2 * flag_d1_6;
    double* host_d1_t2_7 = df_host_pinned_d1_t2 + max_dim_d1_t2 * flag_d1_7;
    double* host_d1_v2_7 = df_host_pinned_d1_v2 + max_dim_d1_v2 * flag_d1_7;
    double* host_d1_t2_8 = df_host_pinned_d1_t2 + max_dim_d1_t2 * flag_d1_8;
    double* host_d1_v2_8 = df_host_pinned_d1_v2 + max_dim_d1_v2 * flag_d1_8;
    double* host_d1_t2_9 = df_host_pinned_d1_t2 + max_dim_d1_t2 * flag_d1_9;
    double* host_d1_v2_9 = df_host_pinned_d1_v2 + max_dim_d1_v2 * flag_d1_9;

#ifdef _OPENMP
#pragma omp parallel for collapse(6)
#endif
    for(int t3_h3 = 0; t3_h3 < d1_base_size_h3b; t3_h3++)
      for(int t3_h2 = 0; t3_h2 < d1_base_size_h2b; t3_h2++)
        for(int t3_h1 = 0; t3_h1 < d1_base_size_h1b; t3_h1++)
          for(int t3_p6 = 0; t3_p6 < d1_base_size_p6b; t3_p6++)
            for(int t3_p5 = 0; t3_p5 < d1_base_size_p5b; t3_p5++)
              for(int t3_p4 = 0; t3_p4 < d1_base_size_p4b; t3_p4++) {
                int t3_idx =
                  t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) *d1_base_size_p5b) *
                                                       d1_base_size_p6b) *
                                              d1_base_size_h1b) *
                                     d1_base_size_h2b) *
                            d1_base_size_h3b;

                for(int t3_h7 = 0; t3_h7 < d1_base_size_h7b; t3_h7++) {
                  // sd1_1:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h1] * v2[h3,h2,p6,h7]
                  if(flag_d1_1 >= 0) {
                    host_t3_d[t3_idx] -=
                      host_d1_t2_1[t3_h7 + (t3_p4 + (t3_p5 + (t3_h1) *d1_base_size_p5b) *
                                                      d1_base_size_p4b) *
                                             d1_base_size_h7b] *
                      host_d1_v2_1[t3_h3 + (t3_h2 + (t3_p6 + (t3_h7) *d1_base_size_p6b) *
                                                      d1_base_size_h2b) *
                                             d1_base_size_h3b];
                  }

                  // sd1_2:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p5,h2] * v2[h3,h1,p6,h7]
                  if(flag_d1_2 >= 0) {
                    host_t3_d[t3_idx] +=
                      host_d1_t2_2[t3_h7 + (t3_p4 + (t3_p5 + (t3_h2) *d1_base_size_p5b) *
                                                      d1_base_size_p4b) *
                                             d1_base_size_h7b] *
                      host_d1_v2_2[t3_h3 + (t3_h1 + (t3_p6 + (t3_h7) *d1_base_size_p6b) *
                                                      d1_base_size_h1b) *
                                             d1_base_size_h3b];
                  }

                  // sd1_3:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h3] * v2[h2,h1,p6,h7]
                  if(flag_d1_3 >= 0) {
                    host_t3_d[t3_idx] -=
                      host_d1_t2_3[t3_h7 + (t3_p4 + (t3_p5 + (t3_h3) *d1_base_size_p5b) *
                                                      d1_base_size_p4b) *
                                             d1_base_size_h7b] *
                      host_d1_v2_3[t3_h2 + (t3_h1 + (t3_p6 + (t3_h7) *d1_base_size_p6b) *
                                                      d1_base_size_h1b) *
                                             d1_base_size_h2b];
                  }

                  // sd1_4:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h1] * v2[h3,h2,p4,h7]
                  if(flag_d1_4 >= 0) {
                    host_t3_d[t3_idx] -=
                      host_d1_t2_4[t3_h7 + (t3_p5 + (t3_p6 + (t3_h1) *d1_base_size_p6b) *
                                                      d1_base_size_p5b) *
                                             d1_base_size_h7b] *
                      host_d1_v2_4[t3_h3 + (t3_h2 + (t3_p4 + (t3_h7) *d1_base_size_p4b) *
                                                      d1_base_size_h2b) *
                                             d1_base_size_h3b];
                  }

                  // sd1_5:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p5,p6,h2] * v2[h3,h1,p4,h7]
                  if(flag_d1_5 >= 0) {
                    host_t3_d[t3_idx] +=
                      host_d1_t2_5[t3_h7 + (t3_p5 + (t3_p6 + (t3_h2) *d1_base_size_p6b) *
                                                      d1_base_size_p5b) *
                                             d1_base_size_h7b] *
                      host_d1_v2_5[t3_h3 + (t3_h1 + (t3_p4 + (t3_h7) *d1_base_size_p4b) *
                                                      d1_base_size_h1b) *
                                             d1_base_size_h3b];
                  }

                  // sd1_6:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h3] * v2[h2,h1,p4,h7]
                  if(flag_d1_6 >= 0) {
                    host_t3_d[t3_idx] -=
                      host_d1_t2_6[t3_h7 + (t3_p5 + (t3_p6 + (t3_h3) *d1_base_size_p6b) *
                                                      d1_base_size_p5b) *
                                             d1_base_size_h7b] *
                      host_d1_v2_6[t3_h2 + (t3_h1 + (t3_p4 + (t3_h7) *d1_base_size_p4b) *
                                                      d1_base_size_h1b) *
                                             d1_base_size_h2b];
                  }

                  // sd1_7:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h1] * v2[h3,h2,p5,h7]
                  if(flag_d1_7 >= 0) {
                    host_t3_d[t3_idx] +=
                      host_d1_t2_7[t3_h7 + (t3_p4 + (t3_p6 + (t3_h1) *d1_base_size_p6b) *
                                                      d1_base_size_p4b) *
                                             d1_base_size_h7b] *
                      host_d1_v2_7[t3_h3 + (t3_h2 + (t3_p5 + (t3_h7) *d1_base_size_p5b) *
                                                      d1_base_size_h2b) *
                                             d1_base_size_h3b];
                  }

                  // sd1_8:  t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p6,h2] * v2[h3,h1,p5,h7]
                  if(flag_d1_8 >= 0) {
                    host_t3_d[t3_idx] -=
                      host_d1_t2_8[t3_h7 + (t3_p4 + (t3_p6 + (t3_h2) *d1_base_size_p6b) *
                                                      d1_base_size_p4b) *
                                             d1_base_size_h7b] *
                      host_d1_v2_8[t3_h3 + (t3_h1 + (t3_p5 + (t3_h7) *d1_base_size_p5b) *
                                                      d1_base_size_h1b) *
                                             d1_base_size_h3b];
                  }

                  // sd1_9:  t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h3] * v2[h2,h1,p5,h7]
                  if(flag_d1_9 >= 0) {
                    host_t3_d[t3_idx] +=
                      host_d1_t2_9[t3_h7 + (t3_p4 + (t3_p6 + (t3_h3) *d1_base_size_p6b) *
                                                      d1_base_size_p4b) *
                                             d1_base_size_h7b] *
                      host_d1_v2_9[t3_h2 + (t3_h1 + (t3_p5 + (t3_h7) *d1_base_size_p5b) *
                                                      d1_base_size_h1b) *
                                             d1_base_size_h2b];
                  }
                }
              }
  }

  // d2
  for(size_t idx_nvab = 0; idx_nvab < nvab; idx_nvab++) {
    int flag_d2_1 = (int) df_simple_d2_exec[0 + (idx_nvab) *9];
    int flag_d2_2 = (int) df_simple_d2_exec[1 + (idx_nvab) *9];
    int flag_d2_3 = (int) df_simple_d2_exec[2 + (idx_nvab) *9];
    int flag_d2_4 = (int) df_simple_d2_exec[3 + (idx_nvab) *9];
    int flag_d2_5 = (int) df_simple_d2_exec[4 + (idx_nvab) *9];
    int flag_d2_6 = (int) df_simple_d2_exec[5 + (idx_nvab) *9];
    int flag_d2_7 = (int) df_simple_d2_exec[6 + (idx_nvab) *9];
    int flag_d2_8 = (int) df_simple_d2_exec[7 + (idx_nvab) *9];
    int flag_d2_9 = (int) df_simple_d2_exec[8 + (idx_nvab) *9];

    int d2_base_size_h1b = (int) df_simple_d2_size[0 + (idx_nvab) *7];
    int d2_base_size_h2b = (int) df_simple_d2_size[1 + (idx_nvab) *7];
    int d2_base_size_h3b = (int) df_simple_d2_size[2 + (idx_nvab) *7];
    int d2_base_size_p4b = (int) df_simple_d2_size[3 + (idx_nvab) *7];
    int d2_base_size_p5b = (int) df_simple_d2_size[4 + (idx_nvab) *7];
    int d2_base_size_p6b = (int) df_simple_d2_size[5 + (idx_nvab) *7];
    int d2_base_size_p7b = (int) df_simple_d2_size[6 + (idx_nvab) *7];

    double* host_d2_t2_1 = df_host_pinned_d2_t2 + max_dim_d2_t2 * flag_d2_1;
    double* host_d2_v2_1 = df_host_pinned_d2_v2 + max_dim_d2_v2 * flag_d2_1;
    double* host_d2_t2_2 = df_host_pinned_d2_t2 + max_dim_d2_t2 * flag_d2_2;
    double* host_d2_v2_2 = df_host_pinned_d2_v2 + max_dim_d2_v2 * flag_d2_2;
    double* host_d2_t2_3 = df_host_pinned_d2_t2 + max_dim_d2_t2 * flag_d2_3;
    double* host_d2_v2_3 = df_host_pinned_d2_v2 + max_dim_d2_v2 * flag_d2_3;
    double* host_d2_t2_4 = df_host_pinned_d2_t2 + max_dim_d2_t2 * flag_d2_4;
    double* host_d2_v2_4 = df_host_pinned_d2_v2 + max_dim_d2_v2 * flag_d2_4;
    double* host_d2_t2_5 = df_host_pinned_d2_t2 + max_dim_d2_t2 * flag_d2_5;
    double* host_d2_v2_5 = df_host_pinned_d2_v2 + max_dim_d2_v2 * flag_d2_5;
    double* host_d2_t2_6 = df_host_pinned_d2_t2 + max_dim_d2_t2 * flag_d2_6;
    double* host_d2_v2_6 = df_host_pinned_d2_v2 + max_dim_d2_v2 * flag_d2_6;
    double* host_d2_t2_7 = df_host_pinned_d2_t2 + max_dim_d2_t2 * flag_d2_7;
    double* host_d2_v2_7 = df_host_pinned_d2_v2 + max_dim_d2_v2 * flag_d2_7;
    double* host_d2_t2_8 = df_host_pinned_d2_t2 + max_dim_d2_t2 * flag_d2_8;
    double* host_d2_v2_8 = df_host_pinned_d2_v2 + max_dim_d2_v2 * flag_d2_8;
    double* host_d2_t2_9 = df_host_pinned_d2_t2 + max_dim_d2_t2 * flag_d2_9;
    double* host_d2_v2_9 = df_host_pinned_d2_v2 + max_dim_d2_v2 * flag_d2_9;

#ifdef _OPENMP
#pragma omp parallel for collapse(6)
#endif
    for(int t3_h3 = 0; t3_h3 < d2_base_size_h3b; t3_h3++)
      for(int t3_h2 = 0; t3_h2 < d2_base_size_h2b; t3_h2++)
        for(int t3_h1 = 0; t3_h1 < d2_base_size_h1b; t3_h1++)
          for(int t3_p6 = 0; t3_p6 < d2_base_size_p6b; t3_p6++)
            for(int t3_p5 = 0; t3_p5 < d2_base_size_p5b; t3_p5++)
              for(int t3_p4 = 0; t3_p4 < d2_base_size_p4b; t3_p4++) {
                int t3_idx =
                  t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) *d2_base_size_p5b) *
                                                       d2_base_size_p6b) *
                                              d2_base_size_h1b) *
                                     d2_base_size_h2b) *
                            d2_base_size_h3b;

                for(int t3_p7 = 0; t3_p7 < d2_base_size_p7b; t3_p7++) {
                  // sd2_1:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]
                  if(flag_d2_1 >= 0) {
                    host_t3_d[t3_idx] -=
                      host_d2_t2_1[t3_p7 + (t3_p4 + (t3_h1 + (t3_h2) *d2_base_size_h1b) *
                                                      d2_base_size_p4b) *
                                             d2_base_size_p7b] *
                      host_d2_v2_1[t3_p7 + (t3_h3 + (t3_p6 + (t3_p5) *d2_base_size_p6b) *
                                                      d2_base_size_h3b) *
                                             d2_base_size_p7b];
                  }

                  // sd2_2:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h2,h3] * v2[p7,h1,p6,p5]
                  if(flag_d2_2 >= 0) {
                    host_t3_d[t3_idx] -=
                      host_d2_t2_2[t3_p7 + (t3_p4 + (t3_h2 + (t3_h3) *d2_base_size_h2b) *
                                                      d2_base_size_p4b) *
                                             d2_base_size_p7b] *
                      host_d2_v2_2[t3_p7 + (t3_h1 + (t3_p6 + (t3_p5) *d2_base_size_p6b) *
                                                      d2_base_size_h1b) *
                                             d2_base_size_p7b];
                  }

                  // sd2_3:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p4,h1,h3] * v2[p7,h2,p6,p5]
                  if(flag_d2_3 >= 0) {
                    host_t3_d[t3_idx] +=
                      host_d2_t2_3[t3_p7 + (t3_p4 + (t3_h1 + (t3_h3) *d2_base_size_h1b) *
                                                      d2_base_size_p4b) *
                                             d2_base_size_p7b] *
                      host_d2_v2_3[t3_p7 + (t3_h2 + (t3_p6 + (t3_p5) *d2_base_size_p6b) *
                                                      d2_base_size_h2b) *
                                             d2_base_size_p7b];
                  }

                  // sd2_4:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h1,h2] * v2[p7,h3,p6,p4]
                  if(flag_d2_4 >= 0) {
                    host_t3_d[t3_idx] +=
                      host_d2_t2_4[t3_p7 + (t3_p5 + (t3_h1 + (t3_h2) *d2_base_size_h1b) *
                                                      d2_base_size_p5b) *
                                             d2_base_size_p7b] *
                      host_d2_v2_4[t3_p7 + (t3_h3 + (t3_p6 + (t3_p4) *d2_base_size_p6b) *
                                                      d2_base_size_h3b) *
                                             d2_base_size_p7b];
                  }

                  // sd2_5:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h2,h3] * v2[p7,h1,p6,p4]
                  if(flag_d2_5 >= 0) {
                    host_t3_d[t3_idx] +=
                      host_d2_t2_5[t3_p7 + (t3_p5 + (t3_h2 + (t3_h3) *d2_base_size_h2b) *
                                                      d2_base_size_p5b) *
                                             d2_base_size_p7b] *
                      host_d2_v2_5[t3_p7 + (t3_h1 + (t3_p6 + (t3_p4) *d2_base_size_p6b) *
                                                      d2_base_size_h1b) *
                                             d2_base_size_p7b];
                  }

                  // sd2_6:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p5,h1,h3] * v2[p7,h2,p6,p4]
                  if(flag_d2_6 >= 0) {
                    host_t3_d[t3_idx] -=
                      host_d2_t2_6[t3_p7 + (t3_p5 + (t3_h1 + (t3_h3) *d2_base_size_h1b) *
                                                      d2_base_size_p5b) *
                                             d2_base_size_p7b] *
                      host_d2_v2_6[t3_p7 + (t3_h2 + (t3_p6 + (t3_p4) *d2_base_size_p6b) *
                                                      d2_base_size_h2b) *
                                             d2_base_size_p7b];
                  }

                  // sd2_7:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h1,h2] * v2[p7,h3,p5,p4]
                  if(flag_d2_7 >= 0) {
                    host_t3_d[t3_idx] -=
                      host_d2_t2_7[t3_p7 + (t3_p6 + (t3_h1 + (t3_h2) *d2_base_size_h1b) *
                                                      d2_base_size_p6b) *
                                             d2_base_size_p7b] *
                      host_d2_v2_7[t3_p7 + (t3_h3 + (t3_p5 + (t3_p4) *d2_base_size_p5b) *
                                                      d2_base_size_h3b) *
                                             d2_base_size_p7b];
                  }

                  // sd2_8:  t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h2,h3] * v2[p7,h1,p5,p4]
                  if(flag_d2_8 >= 0) {
                    host_t3_d[t3_idx] -=
                      host_d2_t2_8[t3_p7 + (t3_p6 + (t3_h2 + (t3_h3) *d2_base_size_h2b) *
                                                      d2_base_size_p6b) *
                                             d2_base_size_p7b] *
                      host_d2_v2_8[t3_p7 + (t3_h1 + (t3_p5 + (t3_p4) *d2_base_size_p5b) *
                                                      d2_base_size_h1b) *
                                             d2_base_size_p7b];
                  }

                  // sd2_9:  t3[h3,h2,h1,p6,p5,p4] += t2[p7,p6,h1,h3] * v2[p7,h2,p5,p4]
                  if(flag_d2_9 >= 0) {
                    host_t3_d[t3_idx] +=
                      host_d2_t2_9[t3_p7 + (t3_p6 + (t3_h1 + (t3_h3) *d2_base_size_h1b) *
                                                      d2_base_size_p6b) *
                                             d2_base_size_p7b] *
                      host_d2_v2_9[t3_p7 + (t3_h2 + (t3_p5 + (t3_p4) *d2_base_size_p5b) *
                                                      d2_base_size_h2b) *
                                             d2_base_size_p7b];
                  }
                }
              }
  }

  // s1
  {
    // 	flags
    int flag_s1_1 = (int) df_simple_s1_exec[0];
    int flag_s1_2 = (int) df_simple_s1_exec[1];
    int flag_s1_3 = (int) df_simple_s1_exec[2];
    int flag_s1_4 = (int) df_simple_s1_exec[3];
    int flag_s1_5 = (int) df_simple_s1_exec[4];
    int flag_s1_6 = (int) df_simple_s1_exec[5];
    int flag_s1_7 = (int) df_simple_s1_exec[6];
    int flag_s1_8 = (int) df_simple_s1_exec[7];
    int flag_s1_9 = (int) df_simple_s1_exec[8];

    int s1_base_size_h1b = (int) df_simple_s1_size[0];
    int s1_base_size_h2b = (int) df_simple_s1_size[1];
    int s1_base_size_h3b = (int) df_simple_s1_size[2];
    int s1_base_size_p4b = (int) df_simple_s1_size[3];
    int s1_base_size_p5b = (int) df_simple_s1_size[4];
    int s1_base_size_p6b = (int) df_simple_s1_size[5];

    double* host_s1_t2;
    double* host_s1_v2;

#ifdef _OPENMP
#pragma omp parallel for collapse(6)
#endif
    for(int t3_h3 = 0; t3_h3 < s1_base_size_h3b; t3_h3++)
      for(int t3_h2 = 0; t3_h2 < s1_base_size_h2b; t3_h2++)
        for(int t3_h1 = 0; t3_h1 < s1_base_size_h1b; t3_h1++)
          for(int t3_p6 = 0; t3_p6 < s1_base_size_p6b; t3_p6++)
            for(int t3_p5 = 0; t3_p5 < s1_base_size_p5b; t3_p5++)
              for(int t3_p4 = 0; t3_p4 < s1_base_size_p4b; t3_p4++) {
                int t3_idx =
                  t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (t3_p4) *s1_base_size_p5b) *
                                                       s1_base_size_p6b) *
                                              s1_base_size_h1b) *
                                     s1_base_size_h2b) *
                            s1_base_size_h3b;

                //  s1_1: t3[h3,h2,h1,p6,p5,p4] += t1[p4,h1] * v2[h3,h2,p6,p5]
                host_s1_t2 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_1;
                host_s1_v2 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_1;

                if(flag_s1_1 >= 0) {
                  host_t3_s[t3_idx] +=
                    host_s1_t2[t3_p4 + (t3_h1) *s1_base_size_p4b] *
                    host_s1_v2[t3_h3 +
                               (t3_h2 + (t3_p6 + (t3_p5) *s1_base_size_p6b) * s1_base_size_h2b) *
                                 s1_base_size_h3b];
                }

                // s1_2: t3[h3,h2,h1,p6,p5,p4] -= t1[p4,h2] * v2[h3,h1,p6,p5]
                host_s1_t2 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_2;
                host_s1_v2 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_2;

                if(flag_s1_2 >= 0) {
                  host_t3_s[t3_idx] -=
                    host_s1_t2[t3_p4 + (t3_h2) *s1_base_size_p4b] *
                    host_s1_v2[t3_h3 +
                               (t3_h1 + (t3_p6 + (t3_p5) *s1_base_size_p6b) * s1_base_size_h1b) *
                                 s1_base_size_h3b];
                }

                // s1_3: t3[h3,h2,h1,p6,p5,p4] += t1[p4,h3] * v2[h2,h1,p6,p5]
                host_s1_t2 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_3;
                host_s1_v2 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_3;

                if(flag_s1_3 >= 0) {
                  host_t3_s[t3_idx] +=
                    host_s1_t2[t3_p4 + (t3_h3) *s1_base_size_p4b] *
                    host_s1_v2[t3_h2 +
                               (t3_h1 + (t3_p6 + (t3_p5) *s1_base_size_p6b) * s1_base_size_h1b) *
                                 s1_base_size_h2b];
                }

                // s1_4:   t3[h3,h2,h1,p6,p5,p4] -= t1[p5,h1] * v2[h3,h2,p6,p4]
                host_s1_t2 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_4;
                host_s1_v2 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_4;

                if(flag_s1_4 >= 0) {
                  host_t3_s[t3_idx] -=
                    host_s1_t2[t3_p5 + (t3_h1) *s1_base_size_p5b] *
                    host_s1_v2[t3_h3 +
                               (t3_h2 + (t3_p6 + (t3_p4) *s1_base_size_p6b) * s1_base_size_h2b) *
                                 s1_base_size_h3b];
                }

                // s1_5:   t3[h3,h2,h1,p6,p5,p4] += t1[p5,h2] * v2[h3,h1,p6,p4]
                host_s1_t2 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_5;
                host_s1_v2 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_5;

                if(flag_s1_5 >= 0) {
                  host_t3_s[t3_idx] +=
                    host_s1_t2[t3_p5 + (t3_h2) *s1_base_size_p5b] *
                    host_s1_v2[t3_h3 +
                               (t3_h1 + (t3_p6 + (t3_p4) *s1_base_size_p6b) * s1_base_size_h1b) *
                                 s1_base_size_h3b];
                }

                // s1_6:   t3[h3,h2,h1,p6,p5,p4] -= t1[p5,h3] * v2[h2,h1,p6,p4]
                host_s1_t2 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_6;
                host_s1_v2 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_6;

                if(flag_s1_6 >= 0) {
                  host_t3_s[t3_idx] -=
                    host_s1_t2[t3_p5 + (t3_h3) *s1_base_size_p5b] *
                    host_s1_v2[t3_h2 +
                               (t3_h1 + (t3_p6 + (t3_p4) *s1_base_size_p6b) * s1_base_size_h1b) *
                                 s1_base_size_h2b];
                }

                // s1_7:   t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h1] * v2[h3,h2,p5,p4]
                host_s1_t2 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_7;
                host_s1_v2 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_7;

                if(flag_s1_7 >= 0) {
                  host_t3_s[t3_idx] +=
                    host_s1_t2[t3_p6 + (t3_h1) *s1_base_size_p6b] *
                    host_s1_v2[t3_h3 +
                               (t3_h2 + (t3_p5 + (t3_p4) *s1_base_size_p5b) * s1_base_size_h2b) *
                                 s1_base_size_h3b];
                }

                // s1_8:   t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h2] * v2[h3,h1,p5,p4]
                host_s1_t2 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_8;
                host_s1_v2 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_8;

                if(flag_s1_8 >= 0) {
                  host_t3_s[t3_idx] -=
                    host_s1_t2[t3_p6 + (t3_h2) *s1_base_size_p6b] *
                    host_s1_v2[t3_h3 +
                               (t3_h1 + (t3_p5 + (t3_p4) *s1_base_size_p5b) * s1_base_size_h1b) *
                                 s1_base_size_h3b];
                }

                // s1_9:   t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h3] * v2[h2,h1,p5,p4]
                host_s1_t2 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_9;
                host_s1_v2 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_9;

                if(flag_s1_9 >= 0) {
                  host_t3_s[t3_idx] +=
                    host_s1_t2[t3_p6 + (t3_h3) *s1_base_size_p6b] *
                    host_s1_v2[t3_h2 +
                               (t3_h1 + (t3_p5 + (t3_p4) *s1_base_size_p5b) * s1_base_size_h1b) *
                                 s1_base_size_h2b];
                }
              }
  }
  //} //idx_ia6

  //
  //  to calculate energies--- E(4) and E(5)
  //
  double final_energy_1 = 0.0;
  double final_energy_2 = 0.0;

  int size_idx_h1 = (int) base_size_h1b;
  int size_idx_h2 = (int) base_size_h2b;
  int size_idx_h3 = (int) base_size_h3b;
  int size_idx_p4 = (int) base_size_p4b;
  int size_idx_p5 = (int) base_size_p5b;
  int size_idx_p6 = (int) base_size_p6b;

  //
  for(int idx_p4 = 0; idx_p4 < size_idx_p4; idx_p4++)
    for(int idx_p5 = 0; idx_p5 < size_idx_p5; idx_p5++)
      for(int idx_p6 = 0; idx_p6 < size_idx_p6; idx_p6++)
        for(int idx_h1 = 0; idx_h1 < size_idx_h1; idx_h1++)
          for(int idx_h2 = 0; idx_h2 < size_idx_h2; idx_h2++)
            for(int idx_h3 = 0; idx_h3 < size_idx_h3; idx_h3++) {
              //
              int idx_t3 =
                idx_h3 +
                (idx_h2 + (idx_h1 + (idx_p6 + (idx_p5 + (idx_p4) *size_idx_p5) * size_idx_p6) *
                                      size_idx_h1) *
                            size_idx_h2) *
                  size_idx_h3;

              //
              double inner_factor = (host_evl_sorted_h3b[idx_h3] + host_evl_sorted_h2b[idx_h2] +
                                     host_evl_sorted_h1b[idx_h1] - host_evl_sorted_p6b[idx_p6] -
                                     host_evl_sorted_p5b[idx_p5] - host_evl_sorted_p4b[idx_p4]);
              //
              final_energy_1 += factor * host_t3_d[idx_t3] * (host_t3_d[idx_t3]) / inner_factor;
              final_energy_2 +=
                factor * host_t3_d[idx_t3] * (host_t3_d[idx_t3] + host_t3_s[idx_t3]) / inner_factor;
            }

  energy_l[0] += final_energy_1;

  energy_l[1] += final_energy_2;

  free(host_t3_d);
  free(host_t3_s);

  // printf ("E(4): %.14f, E(5): %.14f\n", host_energy_4, host_energy_5);
  //  printf
  //  ("========================================================================================\n");
}
