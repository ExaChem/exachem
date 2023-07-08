/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "ccsd_t_all_fused_doubles1.hpp"
#include "ccsd_t_all_fused_doubles2.hpp"
#include "ccsd_t_all_fused_singles.hpp"
#include "ccsd_t_common.hpp"

inline void helper_calculate_num_ops(const Index noab, const Index nvab, int* df_simple_s1_size,
                                     int* df_simple_d1_size, int* df_simple_d2_size,
                                     int* df_simple_s1_exec, int* df_simple_d1_exec,
                                     int* df_simple_d2_exec, long double& task_num_ops_s1,
                                     long double& task_num_ops_d1, long double& task_num_op_sd2,
                                     long double& total_num_ops_s1, long double& total_num_ops_d1,
                                     long double& total_num_ops_d2);

template<typename T>
long double ccsd_t_fully_fused_performance(
  bool                                                                        is_restricted,
  std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, T>>& list_tasks,
  int64_t rank, int task_stride, const Index noab, const Index nvab, std::vector<int>& k_spin,
  std::vector<size_t>& k_range, std::vector<size_t>& k_offset, std::vector<T>& k_evl_sorted) {
  //
  long double total_num_ops_s1 = 0;
  long double total_num_ops_d1 = 0;
  long double total_num_ops_d2 = 0;
  // long double total_num_ops_total = 0;

  //
  int* df_simple_s1_size = (int*) malloc(sizeof(int) * (6));
  int* df_simple_s1_exec = (int*) malloc(sizeof(int) * (9));
  int* df_simple_d1_size = (int*) malloc(sizeof(int) * (7 * noab));
  int* df_simple_d1_exec = (int*) malloc(sizeof(int) * (9 * noab));
  int* df_simple_d2_size = (int*) malloc(sizeof(int) * (7 * nvab));
  int* df_simple_d2_exec = (int*) malloc(sizeof(int) * (9 * nvab));

  //
  unsigned int init_id = rank;
  // unsigned int offset_current = 0;
  // unsigned int offset_next    = 1;

  for(unsigned int current_id = init_id; current_id < list_tasks.size();
      current_id += task_stride) {
    //
    long double task_num_ops_s1 = 0;
    long double task_num_ops_d1 = 0;
    long double task_num_ops_d2 = 0;
    // long double task_num_ops_total = 0;
    size_t total_comm_data = 0;

    //
    int num_s1_enabled_kernels = 0;
    int num_d1_enabled_kernels = 0;
    int num_d2_enabled_kernels = 0;

    //
    size_t t_h1b = std::get<0>(list_tasks[(int) current_id]);
    size_t t_h2b = std::get<1>(list_tasks[(int) current_id]);
    size_t t_h3b = std::get<2>(list_tasks[(int) current_id]);
    size_t t_p4b = std::get<3>(list_tasks[(int) current_id]);
    size_t t_p5b = std::get<4>(list_tasks[(int) current_id]);
    size_t t_p6b = std::get<5>(list_tasks[(int) current_id]);

    //
    std::fill(df_simple_s1_exec, df_simple_s1_exec + (9), -1);
    std::fill(df_simple_d1_exec, df_simple_d1_exec + (9 * noab), -1);
    std::fill(df_simple_d2_exec, df_simple_d2_exec + (9 * nvab), -1);

    ccsd_t_data_s1_info_only(is_restricted, noab, nvab, k_spin, k_evl_sorted, k_range, t_h1b, t_h2b,
                             t_h3b, t_p4b, t_p5b, t_p6b, df_simple_s1_size, df_simple_s1_exec,
                             &num_s1_enabled_kernels, total_comm_data);

    ccsd_t_data_d1_info_only(is_restricted, noab, nvab, k_spin, k_evl_sorted, k_range, t_h1b, t_h2b,
                             t_h3b, t_p4b, t_p5b, t_p6b, df_simple_d1_size, df_simple_d1_exec,
                             &num_d1_enabled_kernels, total_comm_data);

    ccsd_t_data_d2_info_only(is_restricted, noab, nvab, k_spin, k_evl_sorted, k_range, t_h1b, t_h2b,
                             t_h3b, t_p4b, t_p5b, t_p6b, df_simple_d2_size, df_simple_d2_exec,
                             &num_d2_enabled_kernels, total_comm_data);

    // printf ("[%s] each_kernel[%4d] >> total: %lu\n", __func__, current_id, total_comm_data);

    // size_t total_enabled_kernels = num_s1_enabled_kernels + num_d1_enabled_kernels +
    // num_d2_enabled_kernels; printf ("[%s] each_kernel[%4d] s1: %d, d1: %d, d2: %d >> total:
    // %lu\n", __func__, current_id, num_s1_enabled_kernels, num_d1_enabled_kernels,
    // num_d2_enabled_kernels, total_enabled_kernels);

    //
    helper_calculate_num_ops(noab, nvab, df_simple_s1_size, df_simple_d1_size, df_simple_d2_size,
                             df_simple_s1_exec, df_simple_d1_exec, df_simple_d2_exec,
                             task_num_ops_s1, task_num_ops_d1, task_num_ops_d2, total_num_ops_s1,
                             total_num_ops_d1, total_num_ops_d2);

    // long double total_each = task_num_ops_s1 + task_num_ops_d1 + task_num_ops_d2;
    // printf ("[%s] each task >> s1: %lu, d1: %lu, d2: %lu >> total: %lu\n", __func__,
    // task_num_ops_s1, task_num_ops_d1, task_num_ops_d2, total_each);
  }

  //
  long double total_overall = total_num_ops_s1 + total_num_ops_d1 + total_num_ops_d2;
  // printf ("[%s] total task >> s1: %lu, d1: %lu, d2: %lu >> total: %lu\n", __func__,
  // total_num_ops_s1, total_num_ops_d1, total_num_ops_d2, total_overall);

  //
  return total_overall;
}

//
inline void helper_calculate_num_ops(const Index noab, const Index nvab, int* df_simple_s1_size,
                                     int* df_simple_d1_size, int* df_simple_d2_size,
                                     int* df_simple_s1_exec, int* df_simple_d1_exec,
                                     int* df_simple_d2_exec, long double& task_num_ops_s1,
                                     long double& task_num_ops_d1, long double& task_num_ops_d2,
                                     long double& total_num_ops_s1, long double& total_num_ops_d1,
                                     long double& total_num_ops_d2) {
  //
  //  s1
  //
  long double num_ops_s1 = 0;
  long double num_ops_d1 = 0;
  long double num_ops_d2 = 0;
  {
    long double base_num_ops_s1_per_eq =
      ((long double) df_simple_s1_size[0]) * ((long double) df_simple_s1_size[1]) *
      ((long double) df_simple_s1_size[2]) * ((long double) df_simple_s1_size[3]) *
      ((long double) df_simple_s1_size[4]) * ((long double) df_simple_s1_size[5]) * 2;

#if 0
    printf ("[%s] s1: h1,h2,h3,p4,p5,p6=%2d,%2d,%2d,%2d,%2d,%2d\n", __func__, df_simple_s1_size[0], df_simple_s1_size[1], df_simple_s1_size[2], 
                                                                              df_simple_s1_size[3], df_simple_s1_size[4], df_simple_s1_size[5]);
#endif
#if 0
    printf ("[%s] s1: %2d,%2d,%2d/%2d,%2d,%2d/%2d,%2d,%2d\n", __func__, df_simple_s1_exec[0], df_simple_s1_exec[1], df_simple_s1_exec[2], 
                                                                        df_simple_s1_exec[3], df_simple_s1_exec[4], df_simple_s1_exec[5],
                                                                        df_simple_s1_exec[6], df_simple_s1_exec[7], df_simple_s1_exec[8]);
    printf ("[%s] s1: eq: %lu\n", __func__, base_num_ops_s1_per_eq);
#endif

    if(df_simple_s1_exec[0] >= 0) num_ops_s1 += base_num_ops_s1_per_eq;
    if(df_simple_s1_exec[1] >= 0) num_ops_s1 += base_num_ops_s1_per_eq;
    if(df_simple_s1_exec[2] >= 0) num_ops_s1 += base_num_ops_s1_per_eq;
    if(df_simple_s1_exec[3] >= 0) num_ops_s1 += base_num_ops_s1_per_eq;
    if(df_simple_s1_exec[4] >= 0) num_ops_s1 += base_num_ops_s1_per_eq;
    if(df_simple_s1_exec[5] >= 0) num_ops_s1 += base_num_ops_s1_per_eq;
    if(df_simple_s1_exec[6] >= 0) num_ops_s1 += base_num_ops_s1_per_eq;
    if(df_simple_s1_exec[7] >= 0) num_ops_s1 += base_num_ops_s1_per_eq;
    if(df_simple_s1_exec[8] >= 0) num_ops_s1 += base_num_ops_s1_per_eq;
  }

  //
  //  d1
  //
  {
    for(Index idx_noab = 0; idx_noab < noab; idx_noab++) {
      long double base_num_ops_d1_per_eq = ((long double) df_simple_d1_size[0 + (idx_noab) *7]) *
                                           ((long double) df_simple_d1_size[1 + (idx_noab) *7]) *
                                           ((long double) df_simple_d1_size[2 + (idx_noab) *7]) *
                                           ((long double) df_simple_d1_size[3 + (idx_noab) *7]) *
                                           ((long double) df_simple_d1_size[4 + (idx_noab) *7]) *
                                           ((long double) df_simple_d1_size[5 + (idx_noab) *7]) *
                                           ((long double) df_simple_d1_size[6 + (idx_noab) *7]) * 2;

#if 0
      printf ("[%s] d1 with noab: %d >> h1,h2,h3,h7,p4,p5,p6=%2d,%2d,%2d,%2d,%2d,%2d,%2d\n", __func__, idx_noab, 
      df_simple_d1_size[0 + (idx_noab) * 7], df_simple_d1_size[1 + (idx_noab) * 7], df_simple_d1_size[2 + (idx_noab) * 7], 
                                      df_simple_d1_size[3 + (idx_noab) * 7], df_simple_d1_size[4 + (idx_noab) * 7], df_simple_d1_size[5 + (idx_noab) * 7], df_simple_d1_size[6 + (idx_noab) * 7]);

      printf ("[%s] d1: %2d,%2d,%2d/%2d,%2d,%2d/%2d,%2d,%2d\n", __func__, df_simple_d1_exec[0 + (idx_noab) * 9], df_simple_d1_exec[1 + (idx_noab) * 9], df_simple_d1_exec[2 + (idx_noab) * 9], 
                                                                          df_simple_d1_exec[3 + (idx_noab) * 9], df_simple_d1_exec[4 + (idx_noab) * 9], df_simple_d1_exec[5 + (idx_noab) * 9],
                                                                          df_simple_d1_exec[6 + (idx_noab) * 9], df_simple_d1_exec[7 + (idx_noab) * 9], df_simple_d1_exec[8 + (idx_noab) * 9]);

      printf ("[%s] d1 with noab: %d >> %lu\n", __func__, idx_noab, base_num_ops_d1_per_eq);
#endif

      if(df_simple_d1_exec[0 + (idx_noab) *9] >= 0) num_ops_d1 += base_num_ops_d1_per_eq;
      if(df_simple_d1_exec[1 + (idx_noab) *9] >= 0) num_ops_d1 += base_num_ops_d1_per_eq;
      if(df_simple_d1_exec[2 + (idx_noab) *9] >= 0) num_ops_d1 += base_num_ops_d1_per_eq;
      if(df_simple_d1_exec[3 + (idx_noab) *9] >= 0) num_ops_d1 += base_num_ops_d1_per_eq;
      if(df_simple_d1_exec[4 + (idx_noab) *9] >= 0) num_ops_d1 += base_num_ops_d1_per_eq;
      if(df_simple_d1_exec[5 + (idx_noab) *9] >= 0) num_ops_d1 += base_num_ops_d1_per_eq;
      if(df_simple_d1_exec[6 + (idx_noab) *9] >= 0) num_ops_d1 += base_num_ops_d1_per_eq;
      if(df_simple_d1_exec[7 + (idx_noab) *9] >= 0) num_ops_d1 += base_num_ops_d1_per_eq;
      if(df_simple_d1_exec[8 + (idx_noab) *9] >= 0) num_ops_d1 += base_num_ops_d1_per_eq;
    }
  }

  //
  //  d2
  //
  {
    for(Index idx_nvab = 0; idx_nvab < nvab; idx_nvab++) {
      long double base_num_ops_d2_per_eq = ((long double) df_simple_d2_size[0 + (idx_nvab) *7]) *
                                           ((long double) df_simple_d2_size[1 + (idx_nvab) *7]) *
                                           ((long double) df_simple_d2_size[2 + (idx_nvab) *7]) *
                                           ((long double) df_simple_d2_size[3 + (idx_nvab) *7]) *
                                           ((long double) df_simple_d2_size[4 + (idx_nvab) *7]) *
                                           ((long double) df_simple_d2_size[5 + (idx_nvab) *7]) *
                                           ((long double) df_simple_d2_size[6 + (idx_nvab) *7]) * 2;

#if 0
    printf ("[%s] d2 with noab: %d >> h1,h2,h3,p4,p5,p6,p7=%2d,%2d,%2d,%2d,%2d,%2d,%2d\n", __func__, idx_nvab, 
    df_simple_d2_size[0 + (idx_nvab) * 7], df_simple_d2_size[1 + (idx_nvab) * 7], df_simple_d2_size[2 + (idx_nvab) * 7], 
                                    df_simple_d2_size[3 + (idx_nvab) * 7], df_simple_d2_size[4 + (idx_nvab) * 7], df_simple_d2_size[5 + (idx_nvab) * 7], df_simple_d2_size[6 + (idx_nvab) * 7]);


    printf ("[%s] d2: %2d,%2d,%2d/%2d,%2d,%2d/%2d,%2d,%2d\n", __func__, df_simple_d2_exec[0 + (idx_nvab) * 9], df_simple_d2_exec[1 + (idx_nvab) * 9], df_simple_d2_exec[2 + (idx_nvab) * 9], 
                                                                        df_simple_d2_exec[3 + (idx_nvab) * 9], df_simple_d2_exec[4 + (idx_nvab) * 9], df_simple_d2_exec[5 + (idx_nvab) * 9],
                                                                        df_simple_d2_exec[6 + (idx_nvab) * 9], df_simple_d2_exec[7 + (idx_nvab) * 9], df_simple_d2_exec[8 + (idx_nvab) * 9]);

    printf ("[%s] d2 with noab: %d >> %lu\n", __func__, idx_nvab, base_num_ops_d2_per_eq);
#endif

      if(df_simple_d2_exec[0 + (idx_nvab) *9] >= 0) num_ops_d2 += base_num_ops_d2_per_eq;
      if(df_simple_d2_exec[1 + (idx_nvab) *9] >= 0) num_ops_d2 += base_num_ops_d2_per_eq;
      if(df_simple_d2_exec[2 + (idx_nvab) *9] >= 0) num_ops_d2 += base_num_ops_d2_per_eq;
      if(df_simple_d2_exec[3 + (idx_nvab) *9] >= 0) num_ops_d2 += base_num_ops_d2_per_eq;
      if(df_simple_d2_exec[4 + (idx_nvab) *9] >= 0) num_ops_d2 += base_num_ops_d2_per_eq;
      if(df_simple_d2_exec[5 + (idx_nvab) *9] >= 0) num_ops_d2 += base_num_ops_d2_per_eq;
      if(df_simple_d2_exec[6 + (idx_nvab) *9] >= 0) num_ops_d2 += base_num_ops_d2_per_eq;
      if(df_simple_d2_exec[7 + (idx_nvab) *9] >= 0) num_ops_d2 += base_num_ops_d2_per_eq;
      if(df_simple_d2_exec[8 + (idx_nvab) *9] >= 0) num_ops_d2 += base_num_ops_d2_per_eq;
    }
  }

  // printf ("[%s][# of ops] s1: %lu, d1: %lu, d2: %lu\n", __func__, num_ops_s1, num_ops_d1,
  // num_ops_d2);

  //
  task_num_ops_s1 = num_ops_s1;
  task_num_ops_d1 = num_ops_d1;
  task_num_ops_d2 = num_ops_d2;

  //
  total_num_ops_s1 += num_ops_s1;
  total_num_ops_d1 += num_ops_d1;
  total_num_ops_d2 += num_ops_d2;
}
