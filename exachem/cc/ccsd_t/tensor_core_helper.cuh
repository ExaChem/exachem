/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

constexpr int pad = 3;

constexpr int WARP_M = 4;
constexpr int WARP_N = 2;

constexpr int INST_M = 8;
constexpr int INST_N = 8;
constexpr int INST_K = 4;

constexpr int MMA_M = INST_M * WARP_M;
constexpr int MMA_N = INST_N * WARP_N;
constexpr int MMA_K = INST_K;

struct WarpRegisterMapping {
  int lane_id;
  int group_id;
  int thread_id_in_group;

  __device__ WarpRegisterMapping(int thread_id):
    lane_id(thread_id & 31), group_id(lane_id >> 2), thread_id_in_group(lane_id & 3) {}
};

struct MmaOperandA {
  using reg_type = double;
  reg_type reg[WARP_M];

  template<int lda>
  __device__ inline void load(void* smem, int tile_k, int tile_m, const WarpRegisterMapping& wrm) {
    reg_type* A = reinterpret_cast<reg_type*>(smem);
    // column
    int k = tile_k * MMA_K + wrm.thread_id_in_group;

#pragma unroll
    for(int i = 0; i < WARP_M; i++) {
      int m  = tile_m * MMA_M + i + WARP_M * wrm.group_id;
      reg[i] = -1 * A[k + lda * m];
    }
  }

  template<int lda>
  __device__ inline void load_plus(void* smem, int tile_k, int tile_m,
                                   const WarpRegisterMapping& wrm) {
    reg_type* A = reinterpret_cast<reg_type*>(smem);
    // column
    int k = tile_k * MMA_K + wrm.thread_id_in_group;

#pragma unroll
    for(int i = 0; i < WARP_M; i++) {
      int m  = tile_m * MMA_M + i + WARP_M * wrm.group_id;
      reg[i] = A[k + lda * m];
    }
  }
};

struct MmaOperandB {
  using reg_type = double;
  reg_type reg[WARP_N];

  template<int ldb>
  __device__ inline void load(void* smem, int tile_k, int tile_n, const WarpRegisterMapping& wrm) {
    reg_type* B = reinterpret_cast<reg_type*>(smem);
    // row
    int k = tile_k * MMA_K + wrm.thread_id_in_group;

#pragma unroll
    for(int i = 0; i < WARP_N; i++) {
      int n  = tile_n * MMA_N + 2 * WARP_N * (wrm.group_id / 2) + 2 * i + (wrm.group_id % 2);
      reg[i] = B[k + ldb * n];
    }
  }
};

struct MmaOperandC {
  using reg_type = double;
  reg_type reg[WARP_M * WARP_N * 2];

  __device__ MmaOperandC() {
#pragma unroll
    for(int i = 0; i < WARP_M * WARP_N * 2; i++) { reg[i] = 0; }
  }

  __device__ inline int col_index(int tile_n, int idx_n, int idx_p,
                                  const WarpRegisterMapping& wrm) {
    return tile_n * MMA_N + 2 * idx_n + WARP_N * 2 * wrm.thread_id_in_group + idx_p;
  }

  __device__ inline int row_index(int tile_m, int idx_m, const WarpRegisterMapping& wrm) {
    return tile_m * MMA_M + idx_m + WARP_M * wrm.group_id;
  }

  __device__ inline reg_type get(int idx_n, int idx_m, int idx_p) {
    return reg[idx_m * WARP_N * 2 + idx_n * 2 + idx_p];
  }

  // Tranpose accessors
  __device__ inline reg_type get_t(int idx_n, int idx_m, int idx_p) {
    return reg[idx_m * 2 + WARP_M * idx_n * 2 + idx_p];
  }
};

__device__ void mma(MmaOperandC& op_c, const MmaOperandA& op_a, const MmaOperandB& op_b) {
#pragma unroll
  for(int n_iter = 0; n_iter < WARP_N; n_iter++) {
#pragma unroll
    for(int m_iter = 0; m_iter < WARP_M; m_iter++) {
      int c_iter = (m_iter * WARP_N + n_iter) * 2;
      asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                   : "+d"(op_c.reg[c_iter + 0]), "+d"(op_c.reg[c_iter + 1])
                   : "d"(op_a.reg[m_iter]), "d"(op_b.reg[n_iter]));
    }
  }
}

__device__ void mma_t(MmaOperandC& op_c, const MmaOperandA& op_a, const MmaOperandB& op_b) {
#pragma unroll
  for(int n_iter = 0; n_iter < WARP_N; n_iter++) {
#pragma unroll
    for(int m_iter = 0; m_iter < WARP_M; m_iter++) {
      int c_iter = (m_iter + WARP_M * n_iter) * 2;
      asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                   : "+d"(op_c.reg[c_iter + 0]), "+d"(op_c.reg[c_iter + 1])
                   : "d"(op_a.reg[m_iter]), "d"(op_b.reg[n_iter]));
    }
  }
}
