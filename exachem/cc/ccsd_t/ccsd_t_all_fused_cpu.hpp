/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccsd_t/fused_common.hpp"

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// t3 index slots, in t3[h3,h2,h1,p6,p5,p4] order
enum { S_H3 = 0, S_H2, S_H1, S_P6, S_P5, S_P4 };

// One d1/d2 term:  t3[h3,h2,h1,p6,p5,p4] += alpha * sum_k t2[..] * v2[..]
// computed as a GEMM into scratch C[v2 free idx, t2 free idx] (column-major), followed by a
// permuted accumulate into t3. t2 is always stored with the contracted index k first;
// v2 has it last for d1 (op_v2 = NoTrans) and first for d2 (op_v2 = Trans).
// t2_slots/v2_slots list the t3 slots of the free indices of t2/v2 in their storage order.
template<typename T>
inline void ccsd_t_cpu_gemm_acc(T alpha, int size_k, const T* t2, const int (&t2_slots)[3],
                                const T* v2, const int (&v2_slots)[3], blas::Op op_v2,
                                const int (&dims)[6], std::vector<T>& scratch, T* t3) {
  const int64_t m = (int64_t) dims[v2_slots[0]] * dims[v2_slots[1]] * dims[v2_slots[2]];
  const int64_t n = (int64_t) dims[t2_slots[0]] * dims[t2_slots[1]] * dims[t2_slots[2]];
  if(scratch.size() < (size_t) (m * n)) scratch.resize(m * n);
  T* C = scratch.data();

  const int64_t lda = (op_v2 == blas::Op::NoTrans) ? m : size_k;
  blas::gemm(blas::Layout::ColMajor, op_v2, blas::Op::NoTrans, m, n, size_k, alpha, v2, lda, t2,
             size_k, T{0}, C, m);

  // stride of each t3 slot within C
  const int c_order[6] = {v2_slots[0], v2_slots[1], v2_slots[2],
                          t2_slots[0], t2_slots[1], t2_slots[2]};
  int64_t   s[6];
  int64_t   stride = 1;
  for(int i = 0; i < 6; i++) {
    s[c_order[i]] = stride;
    stride *= dims[c_order[i]];
  }

  const int size_h3 = dims[S_H3], size_h2 = dims[S_H2], size_h1 = dims[S_H1];
  const int size_p6 = dims[S_P6], size_p5 = dims[S_P5], size_p4 = dims[S_P4];

  // h3 innermost: t3 is stride-1 in h3
#ifdef _OPENMP
#pragma omp parallel for collapse(6)
#endif
  for(int t3_p4 = 0; t3_p4 < size_p4; t3_p4++)
    for(int t3_p5 = 0; t3_p5 < size_p5; t3_p5++)
      for(int t3_p6 = 0; t3_p6 < size_p6; t3_p6++)
        for(int t3_h1 = 0; t3_h1 < size_h1; t3_h1++)
          for(int t3_h2 = 0; t3_h2 < size_h2; t3_h2++)
            for(int t3_h3 = 0; t3_h3 < size_h3; t3_h3++) {
              const int64_t t3_idx =
                t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (int64_t) t3_p4 * size_p5) * size_p6) *
                                            size_h1) *
                                   size_h2) *
                          size_h3;
              const int64_t c_idx = t3_h3 * s[S_H3] + t3_h2 * s[S_H2] + t3_h1 * s[S_H1] +
                                    t3_p6 * s[S_P6] + t3_p5 * s[S_P5] + t3_p4 * s[S_P4];
              t3[t3_idx] += C[c_idx];
            }
}

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
  std::vector<double> host_t3_d_v(size_tensor_t3, 0.0);
  std::vector<double> host_t3_s_v(size_tensor_t3, 0.0);
  double*             host_t3_d = host_t3_d_v.data();
  double*             host_t3_s = host_t3_s_v.data();

  //
  // for (size_t idx_ia6 = 0; idx_ia6 < 9; idx_ia6++){
  // d1:  t3[h3,h2,h1,p6,p5,p4] +/-= sum_h7 t2[h7,..] * v2[..,h7]
  // each term = GEMM into scratch + permuted accumulate (see ccsd_t_cpu_gemm_acc)
  std::vector<double> host_scratch_v(size_tensor_t3);

  // {sign, t2 free slots, v2 free slots} for sd1_1 .. sd1_9
  struct GemmTerm {
    double sign;
    int    t2_slots[3];
    int    v2_slots[3];
  };
  static constexpr GemmTerm d1_terms[9] = {
    {-1.0, {S_P4, S_P5, S_H1}, {S_H3, S_H2, S_P6}}, // sd1_1: t2[h7,p4,p5,h1] * v2[h3,h2,p6,h7]
    {+1.0, {S_P4, S_P5, S_H2}, {S_H3, S_H1, S_P6}}, // sd1_2: t2[h7,p4,p5,h2] * v2[h3,h1,p6,h7]
    {-1.0, {S_P4, S_P5, S_H3}, {S_H2, S_H1, S_P6}}, // sd1_3: t2[h7,p4,p5,h3] * v2[h2,h1,p6,h7]
    {-1.0, {S_P5, S_P6, S_H1}, {S_H3, S_H2, S_P4}}, // sd1_4: t2[h7,p5,p6,h1] * v2[h3,h2,p4,h7]
    {+1.0, {S_P5, S_P6, S_H2}, {S_H3, S_H1, S_P4}}, // sd1_5: t2[h7,p5,p6,h2] * v2[h3,h1,p4,h7]
    {-1.0, {S_P5, S_P6, S_H3}, {S_H2, S_H1, S_P4}}, // sd1_6: t2[h7,p5,p6,h3] * v2[h2,h1,p4,h7]
    {+1.0, {S_P4, S_P6, S_H1}, {S_H3, S_H2, S_P5}}, // sd1_7: t2[h7,p4,p6,h1] * v2[h3,h2,p5,h7]
    {-1.0, {S_P4, S_P6, S_H2}, {S_H3, S_H1, S_P5}}, // sd1_8: t2[h7,p4,p6,h2] * v2[h3,h1,p5,h7]
    {+1.0, {S_P4, S_P6, S_H3}, {S_H2, S_H1, S_P5}}, // sd1_9: t2[h7,p4,p6,h3] * v2[h2,h1,p5,h7]
  };

  for(size_t idx_noab = 0; idx_noab < noab; idx_noab++) {
    const int* d1_size = df_simple_d1_size + idx_noab * 7; // h1,h2,h3,h7,p4,p5,p6
    const int  dims[6] = {d1_size[2], d1_size[1], d1_size[0], d1_size[6], d1_size[5], d1_size[4]};
    const int  d1_base_size_h7b = d1_size[3];

    for(int k = 0; k < 9; k++) {
      int flag = df_simple_d1_exec[k + idx_noab * 9];
      if(flag < 0) continue;
      ccsd_t_cpu_gemm_acc<double>(d1_terms[k].sign, d1_base_size_h7b,
                                  df_host_pinned_d1_t2 + max_dim_d1_t2 * flag, d1_terms[k].t2_slots,
                                  df_host_pinned_d1_v2 + max_dim_d1_v2 * flag, d1_terms[k].v2_slots,
                                  blas::Op::NoTrans, dims, host_scratch_v, host_t3_d);
    }
  }

  // d2:  t3[h3,h2,h1,p6,p5,p4] +/-= sum_p7 t2[p7,..] * v2[p7,..]
  static constexpr GemmTerm d2_terms[9] = {
    {-1.0, {S_P4, S_H1, S_H2}, {S_H3, S_P6, S_P5}}, // sd2_1: t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]
    {-1.0, {S_P4, S_H2, S_H3}, {S_H1, S_P6, S_P5}}, // sd2_2: t2[p7,p4,h2,h3] * v2[p7,h1,p6,p5]
    {+1.0, {S_P4, S_H1, S_H3}, {S_H2, S_P6, S_P5}}, // sd2_3: t2[p7,p4,h1,h3] * v2[p7,h2,p6,p5]
    {+1.0, {S_P5, S_H1, S_H2}, {S_H3, S_P6, S_P4}}, // sd2_4: t2[p7,p5,h1,h2] * v2[p7,h3,p6,p4]
    {+1.0, {S_P5, S_H2, S_H3}, {S_H1, S_P6, S_P4}}, // sd2_5: t2[p7,p5,h2,h3] * v2[p7,h1,p6,p4]
    {-1.0, {S_P5, S_H1, S_H3}, {S_H2, S_P6, S_P4}}, // sd2_6: t2[p7,p5,h1,h3] * v2[p7,h2,p6,p4]
    {-1.0, {S_P6, S_H1, S_H2}, {S_H3, S_P5, S_P4}}, // sd2_7: t2[p7,p6,h1,h2] * v2[p7,h3,p5,p4]
    {-1.0, {S_P6, S_H2, S_H3}, {S_H1, S_P5, S_P4}}, // sd2_8: t2[p7,p6,h2,h3] * v2[p7,h1,p5,p4]
    {+1.0, {S_P6, S_H1, S_H3}, {S_H2, S_P5, S_P4}}, // sd2_9: t2[p7,p6,h1,h3] * v2[p7,h2,p5,p4]
  };

  for(size_t idx_nvab = 0; idx_nvab < nvab; idx_nvab++) {
    const int* d2_size = df_simple_d2_size + idx_nvab * 7; // h1,h2,h3,p4,p5,p6,p7
    const int  dims[6] = {d2_size[2], d2_size[1], d2_size[0], d2_size[5], d2_size[4], d2_size[3]};
    const int  d2_base_size_p7b = d2_size[6];

    for(int k = 0; k < 9; k++) {
      int flag = df_simple_d2_exec[k + idx_nvab * 9];
      if(flag < 0) continue;
      ccsd_t_cpu_gemm_acc<double>(d2_terms[k].sign, d2_base_size_p7b,
                                  df_host_pinned_d2_t2 + max_dim_d2_t2 * flag, d2_terms[k].t2_slots,
                                  df_host_pinned_d2_v2 + max_dim_d2_v2 * flag, d2_terms[k].v2_slots,
                                  blas::Op::Trans, dims, host_scratch_v, host_t3_d);
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

    double* host_s1_t1_1 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_1;
    double* host_s1_v2_1 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_1;
    double* host_s1_t1_2 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_2;
    double* host_s1_v2_2 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_2;
    double* host_s1_t1_3 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_3;
    double* host_s1_v2_3 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_3;
    double* host_s1_t1_4 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_4;
    double* host_s1_v2_4 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_4;
    double* host_s1_t1_5 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_5;
    double* host_s1_v2_5 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_5;
    double* host_s1_t1_6 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_6;
    double* host_s1_v2_6 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_6;
    double* host_s1_t1_7 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_7;
    double* host_s1_v2_7 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_7;
    double* host_s1_t1_8 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_8;
    double* host_s1_v2_8 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_8;
    double* host_s1_t1_9 = df_host_pinned_s1_t1 + max_dim_s1_t1 * flag_s1_9;
    double* host_s1_v2_9 = df_host_pinned_s1_v2 + max_dim_s1_v2 * flag_s1_9;

    // h3 innermost: t3 is stride-1 in h3
#ifdef _OPENMP
#pragma omp parallel for collapse(6)
#endif
    for(int t3_p4 = 0; t3_p4 < s1_base_size_p4b; t3_p4++)
      for(int t3_p5 = 0; t3_p5 < s1_base_size_p5b; t3_p5++)
        for(int t3_p6 = 0; t3_p6 < s1_base_size_p6b; t3_p6++)
          for(int t3_h1 = 0; t3_h1 < s1_base_size_h1b; t3_h1++)
            for(int t3_h2 = 0; t3_h2 < s1_base_size_h2b; t3_h2++)
              for(int t3_h3 = 0; t3_h3 < s1_base_size_h3b; t3_h3++) {
                int64_t t3_idx =
                  t3_h3 + (t3_h2 + (t3_h1 + (t3_p6 + (t3_p5 + (int64_t) t3_p4 * s1_base_size_p5b) *
                                                       s1_base_size_p6b) *
                                              s1_base_size_h1b) *
                                     s1_base_size_h2b) *
                            s1_base_size_h3b;

                //  s1_1: t3[h3,h2,h1,p6,p5,p4] += t1[p4,h1] * v2[h3,h2,p6,p5]
                if(flag_s1_1 >= 0) {
                  host_t3_s[t3_idx] +=
                    host_s1_t1_1[t3_p4 + (t3_h1) *s1_base_size_p4b] *
                    host_s1_v2_1[t3_h3 +
                                 (t3_h2 + (t3_p6 + (t3_p5) *s1_base_size_p6b) * s1_base_size_h2b) *
                                   s1_base_size_h3b];
                }

                // s1_2: t3[h3,h2,h1,p6,p5,p4] -= t1[p4,h2] * v2[h3,h1,p6,p5]
                if(flag_s1_2 >= 0) {
                  host_t3_s[t3_idx] -=
                    host_s1_t1_2[t3_p4 + (t3_h2) *s1_base_size_p4b] *
                    host_s1_v2_2[t3_h3 +
                                 (t3_h1 + (t3_p6 + (t3_p5) *s1_base_size_p6b) * s1_base_size_h1b) *
                                   s1_base_size_h3b];
                }

                // s1_3: t3[h3,h2,h1,p6,p5,p4] += t1[p4,h3] * v2[h2,h1,p6,p5]
                if(flag_s1_3 >= 0) {
                  host_t3_s[t3_idx] +=
                    host_s1_t1_3[t3_p4 + (t3_h3) *s1_base_size_p4b] *
                    host_s1_v2_3[t3_h2 +
                                 (t3_h1 + (t3_p6 + (t3_p5) *s1_base_size_p6b) * s1_base_size_h1b) *
                                   s1_base_size_h2b];
                }

                // s1_4:   t3[h3,h2,h1,p6,p5,p4] -= t1[p5,h1] * v2[h3,h2,p6,p4]
                if(flag_s1_4 >= 0) {
                  host_t3_s[t3_idx] -=
                    host_s1_t1_4[t3_p5 + (t3_h1) *s1_base_size_p5b] *
                    host_s1_v2_4[t3_h3 +
                                 (t3_h2 + (t3_p6 + (t3_p4) *s1_base_size_p6b) * s1_base_size_h2b) *
                                   s1_base_size_h3b];
                }

                // s1_5:   t3[h3,h2,h1,p6,p5,p4] += t1[p5,h2] * v2[h3,h1,p6,p4]
                if(flag_s1_5 >= 0) {
                  host_t3_s[t3_idx] +=
                    host_s1_t1_5[t3_p5 + (t3_h2) *s1_base_size_p5b] *
                    host_s1_v2_5[t3_h3 +
                                 (t3_h1 + (t3_p6 + (t3_p4) *s1_base_size_p6b) * s1_base_size_h1b) *
                                   s1_base_size_h3b];
                }

                // s1_6:   t3[h3,h2,h1,p6,p5,p4] -= t1[p5,h3] * v2[h2,h1,p6,p4]
                if(flag_s1_6 >= 0) {
                  host_t3_s[t3_idx] -=
                    host_s1_t1_6[t3_p5 + (t3_h3) *s1_base_size_p5b] *
                    host_s1_v2_6[t3_h2 +
                                 (t3_h1 + (t3_p6 + (t3_p4) *s1_base_size_p6b) * s1_base_size_h1b) *
                                   s1_base_size_h2b];
                }

                // s1_7:   t3[h3,h2,h1,p6,p5,p4] += t1[p6,h1] * v2[h3,h2,p5,p4]
                if(flag_s1_7 >= 0) {
                  host_t3_s[t3_idx] +=
                    host_s1_t1_7[t3_p6 + (t3_h1) *s1_base_size_p6b] *
                    host_s1_v2_7[t3_h3 +
                                 (t3_h2 + (t3_p5 + (t3_p4) *s1_base_size_p5b) * s1_base_size_h2b) *
                                   s1_base_size_h3b];
                }

                // s1_8:   t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h2] * v2[h3,h1,p5,p4]
                if(flag_s1_8 >= 0) {
                  host_t3_s[t3_idx] -=
                    host_s1_t1_8[t3_p6 + (t3_h2) *s1_base_size_p6b] *
                    host_s1_v2_8[t3_h3 +
                                 (t3_h1 + (t3_p5 + (t3_p4) *s1_base_size_p5b) * s1_base_size_h1b) *
                                   s1_base_size_h3b];
                }

                // s1_9:   t3[h3,h2,h1,p6,p5,p4] += t1[p6,h3] * v2[h2,h1,p5,p4]
                if(flag_s1_9 >= 0) {
                  host_t3_s[t3_idx] +=
                    host_s1_t1_9[t3_p6 + (t3_h3) *s1_base_size_p6b] *
                    host_s1_v2_9[t3_h2 +
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
              int64_t idx_t3 =
                idx_h3 + (idx_h2 + (idx_h1 + (idx_p6 + (idx_p5 + (int64_t) idx_p4 * size_idx_p5) *
                                                         size_idx_p6) *
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

  // host_t3_d/host_t3_s freed automatically when the *_v vectors go out of scope

  // printf ("E(4): %.14f, E(5): %.14f\n", host_energy_4, host_energy_5);
  //  printf
  //  ("========================================================================================\n");
}
