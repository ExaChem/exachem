/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

/*
  This file contains Fully-Fused (T) Kernels for
  3rd. Generation Tensor Cores (FP64)
*/
// (1) Pure FP64
#include "exachem/cc/ccsd_t/ccsd_t_common.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define NUM_IA6_LOOPS 9
#define NUM_D1_EQUATIONS 9
#define NUM_D2_EQUATIONS 9
#define NUM_S1_EQUATIONS 9
#define NUM_D1_INDEX 7
#define NUM_D2_INDEX 7
#define NUM_S1_INDEX 6

// (2) 3rd. Generation Tensor Cores (FP64)
#if defined(USE_NV_TC)
#include "tensor_core_helper.cuh"
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#endif

using namespace std;

#define CUCHK(call)                                                                     \
  {                                                                                     \
    cudaError_t err = call;                                                             \
    if(cudaSuccess != err) {                                                            \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                                 \
      fflush(stderr);                                                                   \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  }

//
#define SIZE_TILE_P7 16
#define SIZE_TILE_H3 4
#define SIZE_TILE_P4 4
#define SIZE_TILE_H2 4
#define SIZE_TILE_H1 4
#define SIZE_TILE_P6 4
#define SIZE_TILE_P5 4

#define SIZE_UNIT_INT SIZE_TILE_P7

#define NUM_INDEX 6
#define CEIL(a, b) (((a) + (b) -1) / (b))

#define PAD 3
#define STAGE_ALIGN 32
#define SINGLE_STAGE_SIZE (64 * (PAD + 16))
#define STAGE_OFFSET ((SINGLE_STAGE_SIZE + STAGE_ALIGN - 1) / STAGE_ALIGN) * STAGE_ALIGN

#define NUM_STAGE 2

#define NUM_ENERGY 2
#define FULL_MASK 0xffffffff

#define TEST_ENABLE_RT

//
//      helpers
//
#define MAX_NOAB 50
#define MAX_NVAB 140

// 9 * (1 + MAX_NOAB + MAX_NVAB) + (MAX_NOAB + MAX_NVAB) * sizeof(int) <= 64KB
__constant__ int const_s1_exec[9];
__constant__ int const_d1_exec[9 * MAX_NOAB];
__constant__ int const_d2_exec[9 * MAX_NVAB];

__constant__ int const_d1_h7b[MAX_NOAB];
__constant__ int const_d2_p7b[MAX_NVAB];

//------------------------------------------------------------------------------ device helper
// fuctions
__device__ inline void zero_shared(double* smem, const int start_row, const int num_rows) {
  const int t_id    = threadIdx.y * blockDim.x + threadIdx.x;
  const int col_idx = t_id % 64;
  const int row_idx = t_id / 64;
  const int row_inc = blockDim.x * blockDim.y / 64;
  for(int i = row_idx + start_row; i < num_rows; i += row_inc) {
    smem[col_idx * (16 + PAD) + i] = 0.0;
  }
}

#if defined(USE_NV_TC)

// fixed (reg_x, reg_y)
__device__ inline void rt_store_fixed(double* smem, const int idx_x_1, const int idx_x_2,
                                      const int idx_y_1, const int idx_y_2, MmaOperandC& op_c) {
#pragma unroll 4
  for(int i = 0; i < 4; i++) {
#pragma unroll 4
    for(int j = 0; j < 4; j++) {
      smem[idx_x_1 + (idx_x_2 + (j) *4) * 4 + (idx_y_1 + (idx_y_2 + (i) *4) * 4) * 65] =
        op_c.reg[j + (i) *4];
    }
  }
}

// fixed (reg_x, reg_y)
__device__ inline void rt_load_fixed(double* smem, const int idx_x_1, const int idx_x_2,
                                     const int idx_y_1, const int idx_y_2, MmaOperandC& op_c) {
#pragma unroll 4
  for(int i = 0; i < 4; i++) {
#pragma unroll 4
    for(int j = 0; j < 4; j++) {
      op_c.reg[j + (i) *4] =
        smem[idx_x_1 + (idx_x_2 + (j) *4) * 4 + (idx_y_1 + (idx_y_2 + (i) *4) * 4) * 65];
    }
  }
}

#include "exachem/cc/ccsd_t/ccsd_t_g2s_device_functions.cu"

//------------------------------------------------------------------------------
// created by tc_gen_code_Kernel()
template<typename T>
__global__ __launch_bounds__(256, 3) void fully_fused_kernel_ccsd_t_nvidia_tc_fp64(
  int size_noab, int size_nvab,
  // common
  int size_max_dim_s1_t1, int size_max_dim_s1_v2, int size_max_dim_d1_t2, int size_max_dim_d1_v2,
  int size_max_dim_d2_t2, int size_max_dim_d2_v2,
  //
  T* __restrict__ dev_s1_t1_all, T* __restrict__ dev_s1_v2_all, T* __restrict__ dev_d1_t2_all,
  T* __restrict__ dev_d1_v2_all, T* __restrict__ dev_d2_t2_all, T* __restrict__ dev_d2_v2_all,
  //
  T* dev_energy, const T* dev_evl_sorted_h3b, const T* dev_evl_sorted_h2b,
  const T* dev_evl_sorted_h1b, const T* dev_evl_sorted_p6b, const T* dev_evl_sorted_p5b,
  const T* dev_evl_sorted_p4b,
  //
  const int size_h3, const int size_h2, const int size_h1, const int size_p6, const int size_p5,
  const int size_p4, const int numBlk_h3, const int numBlk_h2, const int numBlk_h1,
  const int numBlk_p6, const int numBlk_p5, const int numBlk_p4) {
  auto grid  = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  // For Shared Memory,
  const int           lda = 16 + PAD;
  extern __shared__ T sm_block[];
  T*                  sm_a = reinterpret_cast<T*>(sm_block) + 0 * STAGE_OFFSET;
  T*                  sm_b = reinterpret_cast<T*>(sm_block) + NUM_STAGE * STAGE_OFFSET;

#pragma unroll
  for(int i = 0; i < NUM_STAGE; i++) {
    zero_shared(sm_a + STAGE_OFFSET * i, 0, 16);
    zero_shared(sm_b + STAGE_OFFSET * i, 0, 16);
  }
  block.sync();

  // Allocate shared storage for a N-stage cuda::pipeline:
  cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

  const int           thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  const int           warp_id   = thread_id / 32; // 0:7
  WarpRegisterMapping wrm(thread_id);

  const int tile_m = warp_id % 2; // 0:1
  const int tile_n = warp_id / 2; // 0:3

  MmaOperandC op_c;
  MmaOperandC op_c_s;

  int internal_upperbound = 0;
  int internal_offset;

  //
  //  based on sd2_1
  //
  int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2.
  int idx_h2 = threadIdx.x / SIZE_TILE_P6;
  int idx_h1 = threadIdx.y % SIZE_TILE_H1;
  int idx_h3 = threadIdx.y / SIZE_TILE_H1;

  int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
  int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

  int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
  tmp_blkIdx     = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

  int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
  tmp_blkIdx     = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

  int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
  tmp_blkIdx     = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

  int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
  tmp_blkIdx     = tmp_blkIdx % (numBlk_h3);

  int blk_idx_h3 = tmp_blkIdx;

  // need to support partial tiles
  int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
  rng_h3 = ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3) ? SIZE_TILE_H3
                                                                     : (size_h3 % SIZE_TILE_H3);
  rng_h2 = ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2) ? SIZE_TILE_H2
                                                                     : (size_h2 % SIZE_TILE_H2);
  rng_h1 = ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1) ? SIZE_TILE_H1
                                                                     : (size_h1 % SIZE_TILE_H1);
  rng_p6 = ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6) ? SIZE_TILE_P6
                                                                     : (size_p6 % SIZE_TILE_P6);
  rng_p5 = ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5) ? SIZE_TILE_P5
                                                                     : (size_p5 % SIZE_TILE_P5);
  rng_p4 = ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4) ? SIZE_TILE_P4
                                                                     : (size_p4 % SIZE_TILE_P4);

  //
  // const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

  // TB_X(p4,h3), TB_Y(h1,h2)
  T partial_inner_factor = 0.0;
  if(idx_p6 < rng_p4 && idx_h2 < rng_h3 && idx_h1 < rng_h1 && idx_h3 < rng_h2)
    partial_inner_factor = dev_evl_sorted_h3b[blk_idx_h3 * SIZE_TILE_H3 + idx_h2] +
                           dev_evl_sorted_h2b[blk_idx_h2 * SIZE_TILE_H2 + idx_h3] +
                           dev_evl_sorted_h1b[blk_idx_h1 * SIZE_TILE_H1 + idx_h1] -
                           dev_evl_sorted_p4b[blk_idx_p4 * SIZE_TILE_P4 + idx_p6];

    //
#if 1
  // sd2_1: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5] --> TB_X(p6,h2), TB_Y(h1,h3),
  // REG_X,Y(p5,p4)
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    int size_p7   = const_d2_p7b[iter_nvab];
    int flag_d2_1 = const_d2_exec[0 + (iter_nvab) *9];

    const size_t num_batches      = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

    if(flag_d2_1 >= 0) {
      T* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_1;
      T* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_1;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_p7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_h2) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_t2_1<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,
                                         blk_idx_h2, idx_h1, blk_idx_h1, size_h1, idx_h3,
                                         blk_idx_p4, size_p4, size_p7, threadIdx.x + l_fetch,
                                         rng_p4, pipeline);
          }

          if((idx_h3 < rng_h3) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_v2_1<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2,
                                         blk_idx_p5, blk_idx_p6, size_p6, idx_h1, blk_idx_h3,
                                         size_h3, idx_h3, size_p7, threadIdx.x + l_fetch, rng_p5,
                                         pipeline);
          }
          pipeline.producer_commit();
        }

        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }

  // sd1_6:     t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h3] * v2[h2,h1,p4,h7]
  // sd1_6':    t3[h3,h2,h1,p6,p5,p4] -= v2[h2,h1,p4,h7] * t2[h7,p5,p6,h3] --> TB_X(p6,h2),
  // TB_Y(h1,h3), REG_X,Y(p5,p4)
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    //
    int size_h7   = const_d1_h7b[iter_noab];
    int flag_d1_6 = const_d1_exec[5 + (iter_noab) *9];

    const size_t num_batches      = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

    if(flag_d1_6 >= 0) {
      T* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_6;
      T* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_6;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_h7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h3) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_t2_6<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2,
                                         blk_idx_h3, idx_h3, blk_idx_p6, size_p6, idx_h1,
                                         blk_idx_p5, size_p5, size_h7, threadIdx.x + l_fetch,
                                         rng_p5, pipeline);
          }

          if((idx_h2 < rng_h1) && (idx_p6 < rng_h2) &&
             threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_v2_6<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2,
                                         blk_idx_p4, size_p4, blk_idx_h1, size_h1, idx_h2,
                                         blk_idx_h2, size_h2, idx_p6, threadIdx.y + l_fetch, rng_p4,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }
#endif

  // rt #1
  // from TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p5,p4) // d2_1
  // to         TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4) // d2_2
#ifdef TEST_ENABLE_RT
  rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c);
  block.sync();
  rt_load_fixed(sm_block, idx_p6, idx_h1, idx_h3, idx_h2, op_c);
  block.sync();
#endif

#if 1
  // sd2_2: t3[h3,h2,h1,p6,p5,p4] -= t2[p7,p4,h2,h3] * v2[p7,h1,p6,p5]
  // t2[p7,ry,h2,h3] * v2[p7,h1,p6,rx] -> TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4)
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    int size_p7   = const_d2_p7b[iter_nvab];
    int flag_d2_2 = const_d2_exec[1 + (iter_nvab) *9];

    const size_t num_batches      = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

    if(flag_d2_2 >= 0) {
      T* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_2;
      T* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_2;

      //
      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_p7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h2) && (idx_h1 < rng_h3) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_t2_2<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,
                                         blk_idx_h3, idx_h1, blk_idx_h2, size_h2, idx_h3,
                                         blk_idx_p4, size_p4, size_p7, threadIdx.x + l_fetch,
                                         rng_p4, pipeline);
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_v2_2<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2,
                                         blk_idx_p5, blk_idx_p6, size_p6, idx_h1, blk_idx_h1,
                                         size_h1, idx_h3, size_p7, threadIdx.x + l_fetch, rng_p5,
                                         pipeline);
          }
          pipeline.producer_commit();
        }

        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }

  // sd1_4:     t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h1] * v2[h3,h2,p4,h7]
  // sd1_4':    t3[h3,h2,h1,p6,p5,p4] -= v2[h3,h2,p4,h7] * t2[h7,p5,p6,h1] --> TB_X(p6,h3),
  // TB_Y(h2,h1), REG_X,Y(p5,p4)
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    //
    int size_h7   = const_d1_h7b[iter_noab];
    int flag_d1_4 = const_d1_exec[3 + (iter_noab) *9];

    const size_t num_batches      = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

    if(flag_d1_4 >= 0) {
      T* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_4;
      T* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_4;

      //
      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_h7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_t2_4<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2,
                                         blk_idx_h1, idx_h3, blk_idx_p6, size_p6, idx_h1,
                                         blk_idx_p5, size_p5, size_h7, threadIdx.x + l_fetch,
                                         rng_p5, pipeline);
          }

          if((idx_h2 < rng_h2) && (idx_p6 < rng_h3) &&
             threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_v2_4<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2,
                                         blk_idx_p4, size_p4, blk_idx_h2, size_h2, idx_h2,
                                         blk_idx_h3, size_h3, idx_p6, threadIdx.y + l_fetch, rng_p4,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }
#endif

  // rt         TB_X(p6,h2), TB_Y(h1,h3) #2
  // from TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4) // d2_2
  // to         TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4) // d2_3
#ifdef TEST_ENABLE_RT
  rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3,
                 op_c); // x=[(p6,h3),(p5)], y=[(h2,h1),(p4)] -> x=[(p4,h2),(p6)], y=[(h1,h3),(p5)]
  block.sync();
  rt_load_fixed(sm_block, idx_p6, idx_h2, idx_h3, idx_h1, op_c);
  block.sync();
#endif

#if 1
  // sd2_3: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p4,h1,h3] * v2[p7,h2,p6,p5]
  // t2[p7,ry,h1,h3] * v2[p7,h2,p6,rx] -> TB_X(p6,h3), TB_Y(h1,h2)
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    int size_p7   = const_d2_p7b[iter_nvab];
    int flag_d2_3 = const_d2_exec[2 + (iter_nvab) *9];

    const size_t num_batches      = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

    if(flag_d2_3 >= 0) {
      T* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_3;
      T* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_3;

      //
      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_p7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_h3) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_t2_3<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,
                                         blk_idx_h3, idx_h1, blk_idx_h1, size_h1, idx_h3,
                                         blk_idx_p4, size_p4, size_p7, threadIdx.x + l_fetch,
                                         rng_p4, pipeline);
          }

          if((idx_h3 < rng_h2) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_v2_3<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2,
                                         blk_idx_p5, blk_idx_p6, size_p6, idx_h1, blk_idx_h2,
                                         size_h2, idx_h3, size_p7, threadIdx.x + l_fetch, rng_p5,
                                         pipeline);
          }
          pipeline.producer_commit();
        }

        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }

  // sd1_5:     t3[h3,h2,h1,p6,p5,p4] += t2[h7,p5,p6,h2] * v2[h3,h1,p4,h7]
  // sd1_5':    t3[h3,h2,h1,p6,p5,p4] += v2[h3,h1,p4,h7] * t2[h7,p5,p6,h2] --> TB_X(p6,h3),
  // TB_Y(h1,h2),  REG_X,Y(p5,p4)
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    //
    int size_h7   = const_d1_h7b[iter_noab];
    int flag_d1_5 = const_d1_exec[4 + (iter_noab) *9];

    const size_t num_batches      = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

    if(flag_d1_5 >= 0) {
      T* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_5;
      T* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_5;

      //
      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_h7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h2) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_t2_5<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2,
                                         blk_idx_h2, idx_h3, blk_idx_p6, size_p6, idx_h1,
                                         blk_idx_p5, size_p5, size_h7, threadIdx.x + l_fetch,
                                         rng_p5, pipeline);
          }

          if((idx_h2 < rng_h1) && (idx_p6 < rng_h3) &&
             threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_v2_5<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2,
                                         blk_idx_p4, size_p4, blk_idx_h1, size_h1, idx_h2,
                                         blk_idx_h3, size_h3, idx_p6, threadIdx.y + l_fetch, rng_p4,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }
#endif

  // rt         TB_X(p6,h2), TB_Y(h1,h3) #3
  // from TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4) // d2_3
  // to         TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p4,p5) // d2_4
#ifdef TEST_ENABLE_RT
  rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3,
                 op_c); // x=[(p6,h3),(p5)], y=[(h1,h2),(p4)] -> x=[(p4,h2),(p6)], y=[(h1,h3),(p5)]
  block.sync();
#pragma unroll 4
  for(int i = 0; i < 4; i++) { // p5
#pragma unroll 4
    for(int j = 0; j < 4; j++) { // p4
      op_c.reg[j + (i) *4] =
        sm_block[idx_p6 + (idx_h3 + (i) *4) * 4 + (idx_h1 + (idx_h2 + (j) *4) * 4) * 65];
    }
  }
  block.sync();
#endif

#if 1
  //
  // sd2_4: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h1,h2] * v2[p7,h3,p6,p4] -> TB_X(p6,h2), TB_Y(h1,h3),
  // REG_X,Y(p4,p5)
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    int size_p7   = const_d2_p7b[iter_nvab];
    int flag_d2_4 = const_d2_exec[3 + (iter_nvab) *9];

    const size_t num_batches      = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

    if(flag_d2_4 >= 0) {
      T* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_4;
      T* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_4;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_p7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_h2) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_t2_4<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,
                                         blk_idx_h2, idx_h1, blk_idx_h1, size_h1, idx_h3,
                                         blk_idx_p5, size_p5, // reg_y: p5
                                         size_p7, threadIdx.x + l_fetch, rng_p5, pipeline);
          }

          if((idx_h3 < rng_h3) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_v2_4<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2,
                                         blk_idx_p4, // reg_x: p4
                                         blk_idx_p6, size_p6, idx_h1, blk_idx_h3, size_h3, idx_h3,
                                         size_p7, threadIdx.x + l_fetch, rng_p4, pipeline);
          }
          pipeline.producer_commit();
        }

        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }

  // sd1_9:     t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h3] * v2[h2,h1,p5,h7]
  // sd1_9':    t3[h3,h2,h1,p6,p5,p4] += v2[h2,h1,p5,h7] * t2[h7,p4,p6,h3] --> TB_X(p6,h2),
  // TB_Y(h1,h3), REG_X,Y(p4,p5)
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    //
    int size_h7   = const_d1_h7b[iter_noab];
    int flag_d1_9 = const_d1_exec[8 + (iter_noab) *9];

    const size_t num_batches      = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

    if(flag_d1_9 >= 0) {
      T* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_9;
      T* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_9;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_h7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h3) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_t2_9<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2,
                                         blk_idx_h3, idx_h3, blk_idx_p6, size_p6, idx_h1,
                                         blk_idx_p4, size_p4, size_h7, threadIdx.x + l_fetch,
                                         rng_p4, pipeline);
          }

          if((idx_h2 < rng_h1) && (idx_p6 < rng_h2) &&
             threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_v2_9<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2,
                                         blk_idx_p5, size_p5, blk_idx_h1, size_h1, idx_h2,
                                         blk_idx_h2, size_h2, idx_p6, threadIdx.y + l_fetch, rng_p5,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }
#endif

  // rt         TB_X(p6,h2), TB_Y(h1,h3) #4
  // from TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p4,p5) // sd2_4
  // to         TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5) // sd2_5
#ifdef TEST_ENABLE_RT
  rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c);
  block.sync();
  rt_load_fixed(sm_block, idx_p6, idx_h1, idx_h3, idx_h2, op_c);
  block.sync();
#endif

#if 1
  // sd2_5: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h2,h3] * v2[p7,h1,p6,p4] (pending)
  // [1] t2[p7,rx,h2,h3] * v2[p7,h1,ry,p4] -> TB_X(h3,p4), TB_Y(h1,h2)
  // [2] t2[p7,ry,h2,h3] * v2[p7,h1,rx,p4] -> TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5)
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    int size_p7   = const_d2_p7b[iter_nvab];
    int flag_d2_5 = const_d2_exec[4 + (iter_nvab) *9];

    const size_t num_batches      = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

    if(flag_d2_5 >= 0) {
      T* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_5;
      T* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_5;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_p7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h2) && (idx_h1 < rng_h3) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_t2_5<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,
                                         blk_idx_h3, idx_h1, blk_idx_h2, size_h2, idx_h3,
                                         blk_idx_p5, size_p5, size_p7, threadIdx.x + l_fetch,
                                         rng_p5, pipeline);
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_v2_5<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2,
                                         blk_idx_p4, blk_idx_p6, size_p6, idx_h1, blk_idx_h1,
                                         size_h1, idx_h3, size_p7, threadIdx.x + l_fetch, rng_p4,
                                         pipeline);
          }
          pipeline.producer_commit();
        }

        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }

  // sd1_7:     t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h1] * v2[h3,h2,p5,h7]
  // sd1_7':    t3[h3,h2,h1,p6,p5,p4] += v2[h3,h2,p5,h7] * t2[h7,p4,p6,h1] --> TB_X(p6,h3),
  // TB_Y(h2,h1), REG_X,Y(p4,p5)
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    //
    int size_h7   = const_d1_h7b[iter_noab];
    int flag_d1_7 = const_d1_exec[6 + (iter_noab) *9];

    const size_t num_batches      = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

    if(flag_d1_7 >= 0) {
      T* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_7;
      T* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_7;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_h7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_t2_7<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2,
                                         blk_idx_h1, idx_h3, blk_idx_p6, size_p6, idx_h1,
                                         blk_idx_p4, size_p4, size_h7, threadIdx.x + l_fetch,
                                         rng_p4, pipeline);
          }

          if((idx_h2 < rng_h2) && (idx_p6 < rng_h3) &&
             threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_v2_7<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2,
                                         blk_idx_p5, size_p5, blk_idx_h2, size_h2, idx_h2,
                                         blk_idx_h3, size_h3, idx_p6, threadIdx.y + l_fetch, rng_p5,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }
#endif

  // rt         TB_X(p6,h2), TB_Y(h1,h3) #5
  // from TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5) // sd2_5
  // to         TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p4,p5) // sd2_6
#ifdef TEST_ENABLE_RT
  rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c);
  block.sync();
  rt_load_fixed(sm_block, idx_p6, idx_h2, idx_h3, idx_h1, op_c);
  block.sync();
#endif

#if 1
  // sd2_6: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p5,h1,h3] * v2[p7,h2,p6,p4]
  // [1] t2[p7,rx,h1,h3] * v2[p7,h2,ry,p4] -> TB_X(h3,p4), TB_Y(h2,h3)
  // [2] t2[p7,ry,h1,h3] * v2[p7,h2,rx,p4] -> TB_X(p6,h3), TB_Y(h1,h2) <----
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    int size_p7   = const_d2_p7b[iter_nvab];
    int flag_d2_6 = const_d2_exec[5 + (iter_nvab) *9];

    const size_t num_batches      = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

    if(flag_d2_6 >= 0) {
      T* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_6;
      T* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_6;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_p7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_h3) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_t2_6<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,
                                         blk_idx_h3, idx_h1, blk_idx_h1, size_h1, idx_h3,
                                         blk_idx_p5, size_p5, size_p7, threadIdx.x + l_fetch,
                                         rng_p5, pipeline);
          }

          if((idx_h3 < rng_h2) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_v2_6<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2,
                                         blk_idx_p4, blk_idx_p6, size_p6, idx_h1, blk_idx_h2,
                                         size_h2, idx_h3, size_p7, threadIdx.x + l_fetch, rng_p4,
                                         pipeline);
          }
          pipeline.producer_commit();
        }

        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }

  // sd1_8: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p6,h2] * v2[h3,h1,p5,h7]
  // sd1_8: t3[h3,h2,h1,p6,p5,p4] -= v2[h3,h1,p5,h7] * t2[h7,p4,p6,h2] --> TB_X(p6,h3), TB_Y(h1,h2),
  // REG_X,Y(p4,p5)
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    //
    int size_h7   = const_d1_h7b[iter_noab];
    int flag_d1_8 = const_d1_exec[7 + (iter_noab) *9];

    const size_t num_batches      = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

    if(flag_d1_8 >= 0) {
      T* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_8;
      T* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_8;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_h7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h2) && (idx_h1 < rng_p6) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_t2_8<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2,
                                         blk_idx_h2, idx_h3, blk_idx_p6, size_p6, idx_h1,
                                         blk_idx_p4, size_p4, size_h7, threadIdx.x + l_fetch,
                                         rng_p4, pipeline);
          }

          if((idx_h2 < rng_h1) && (idx_p6 < rng_h3) &&
             threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_v2_8<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2,
                                         blk_idx_p5, size_p5, blk_idx_h1, size_h1, idx_h2,
                                         blk_idx_h3, size_h3, idx_p6, threadIdx.y + l_fetch, rng_p5,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }
#endif

  // rt         TB_X(p6,h2), TB_Y(h1,h3) #6
  // from TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p4,p5)
  // to         TB_X(p4,h2), TB_Y(h1,h3), REG_X,Y(p5,p6)
#ifdef TEST_ENABLE_RT
  rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3,
                 op_c); // x=[(p4,h3),(p6)], y=[(h1,h2),(p5)] -> x=[(p4,h2),(p5)], y=[(h1,h3),(p6)]
  block.sync();
#pragma unroll 4
  for(int i = 0; i < 4; i++) { // p6
#pragma unroll 4
    for(int j = 0; j < 4; j++) { // p5
      op_c.reg[j + (i) *4] =
        sm_block[i + (idx_h3 + (idx_p6) *4) * 4 + (idx_h1 + (idx_h2 + (j) *4) * 4) * 65];
    }
  }
  block.sync();
#endif

  //----------------------------------------------------------------------------
  // >> REG_X,Y(p5,p6)
  // sd2_7 && sd1_3
  // sd2_8 && sd1_1
  // sd2_9 && sd1_2
  //----------------------------------------------------------------------------
#if 1
  // sd2_7:     t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h1,h2] * v2[p7,h3,p5,p4] --> TB_X(p4,h2),
  // TB_Y(h1,h3)
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    int size_p7   = const_d2_p7b[iter_nvab];
    int flag_d2_7 = const_d2_exec[6 + (iter_nvab) *9];

    const size_t num_batches      = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

    if(flag_d2_7 >= 0) {
      T* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_7;
      T* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_7;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_p7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_h2) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_t2_7<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,
                                         blk_idx_h2, idx_h1, blk_idx_h1, size_h1, idx_h3,
                                         blk_idx_p6, size_p6, size_p7, threadIdx.x + l_fetch,
                                         rng_p6, pipeline);
          }

          if((idx_h3 < rng_h3) && (idx_h1 < rng_p4) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_v2_7<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2,
                                         blk_idx_p4, idx_h1, blk_idx_p5, size_p5, blk_idx_h3,
                                         size_h3, idx_h3, size_p7, threadIdx.x + l_fetch, rng_p5,
                                         pipeline);
          }
          pipeline.producer_commit();
        }

        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }

  // sd1_3:     t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h3] * v2[h2,h1,p6,h7]
  // sd1_3':    t3[h3,h2,h1,p6,p5,p4] -= v2[h2,h1,p6,h7] * t2[h7,p4,p5,h3] --> TB_X(h3,h1),
  // TB_Y(h2,p4) sd1_3'': t3[h3,h2,h1,p6,p5,p4] -= v2[h2,h1,p6,h7] * t2[h7,p4,p5,h3] -->
  // TB_X(p4,h2), TB_X(h1,h3)
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    //
    int size_h7   = const_d1_h7b[iter_noab];
    int flag_d1_3 = const_d1_exec[2 + (iter_noab) *9];

    const size_t num_batches      = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

    if(flag_d1_3 >= 0) {
      T* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_3;
      T* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_3;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_h7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h3) && (idx_h1 < rng_p4) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_t2_3<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2,
                                         blk_idx_h3, idx_h3, blk_idx_p5, size_p5, blk_idx_p4,
                                         size_p4, idx_h1, size_h7, threadIdx.x + l_fetch, rng_p5,
                                         pipeline);
          }

          if((idx_h2 < rng_h1) && (idx_p6 < rng_h2) &&
             threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_v2_3<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2,
                                         blk_idx_p6, size_p6, blk_idx_h1, size_h1, idx_h2,
                                         blk_idx_h2, size_h2, idx_p6, threadIdx.y + l_fetch, rng_p6,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }
#endif

  // rt         TB_X(p6,h2), TB_Y(h1,h3) #7
  // from TB_X(p4,h2), TB_Y(h1,h3)
  // to         TB_X(p4,h3), TB_Y(h2,h1)
#ifdef TEST_ENABLE_RT
  rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c);
  block.sync();
  rt_load_fixed(sm_block, idx_p6, idx_h1, idx_h3, idx_h2, op_c);
  block.sync();
#endif

#if 1
  // sd2_8: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h2,h3] * v2[p7,h1,p5,p4]
  // t2[p7,ry,h2,h3] * v2[p7,h1,rx,p4] -> TB_X(p4,h3), TB_Y(h2,h1)
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    int size_p7   = const_d2_p7b[iter_nvab];
    int flag_d2_8 = const_d2_exec[7 + (iter_nvab) *9];

    const size_t num_batches      = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

    if(flag_d2_8 >= 0) {
      T* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_8;
      T* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_8;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_p7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h2) && (idx_h1 < rng_h3) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_t2_8<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,
                                         blk_idx_h3, idx_h1, blk_idx_h2, size_h2, idx_h3,
                                         blk_idx_p6, size_p6, size_p7, threadIdx.x + l_fetch,
                                         rng_p6, pipeline);
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_p4) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_v2_8<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2,
                                         blk_idx_p4, idx_h1, blk_idx_p5, size_p5, blk_idx_h1,
                                         size_h1, idx_h3, size_p7, threadIdx.x + l_fetch, rng_p5,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();

        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }

  // sd1_1:     t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h1] * v2[h3,h2,p6,h7]
  // sd1_1':    t3[h3,h2,h1,p6,p5,p4] -= v2[h3,h2,p6,h7] * t2[h7,p4,p5,h1] --> TB_X(h1,h2),
  // TB_Y(h3,p4), REG_X,Y(p5,p6) sd1_1'': t3[h3,h2,h1,p6,p5,p4] -= v2[h3,h2,p6,h7] * t2[h7,p4,p5,h1]
  // --> TB_X(p4,h3), TB_Y(h2,h1), REG_X,Y(p5,p6)
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    //
    int size_h7   = const_d1_h7b[iter_noab];
    int flag_d1_1 = const_d1_exec[0 + (iter_noab) *9];

    const size_t num_batches      = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

    if(flag_d1_1 >= 0) {
      T* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_1;
      T* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_1;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_h7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_p4) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_t2_1<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2,
                                         blk_idx_h1, idx_h3, blk_idx_p5, size_p5, blk_idx_p4,
                                         size_p4, idx_h1, size_h7, threadIdx.x + l_fetch, rng_p5,
                                         pipeline);
          }

          if((idx_h2 < rng_h2) && (idx_p6 < rng_h3) &&
             threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_v2_1<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2,
                                         blk_idx_p6, size_p6, blk_idx_h2, size_h2, idx_h2,
                                         blk_idx_h3, size_h3, idx_p6, threadIdx.y + l_fetch, rng_p6,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }
#endif

  // rt         TB_X(p6,h2), TB_Y(h1,h3) #8
  // from TB_X(p4,h3), TB_Y(h2,h1)
  // to         TB_X(p4,h3), TB_Y(h1,h2)
#ifdef TEST_ENABLE_RT
  rt_store_fixed(sm_block, idx_p6, idx_h2, idx_h1, idx_h3, op_c);
  block.sync();
  rt_load_fixed(sm_block, idx_p6, idx_h2, idx_h3, idx_h1, op_c);
  block.sync();
#endif

#if 1
  // sd2_9: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p6,h1,h3] * v2[p7,h2,p5,p4]
  // TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6)
  for(int iter_nvab = 0; iter_nvab < size_nvab; iter_nvab++) {
    int size_p7   = const_d2_p7b[iter_nvab];
    int flag_d2_9 = const_d2_exec[8 + (iter_nvab) *9];

    const size_t num_batches      = (size_p7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_p7 + 3) / 4) * 4;

    if(flag_d2_9 >= 0) {
      T* tmp_dev_d2_t2 = dev_d2_t2_all + size_max_dim_d2_t2 * flag_d2_9;
      T* tmp_dev_d2_v2 = dev_d2_v2_all + size_max_dim_d2_v2 * flag_d2_9;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_p7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_p7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h1) && (idx_h1 < rng_h3) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_t2_9<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d2_t2,
                                         blk_idx_h3, idx_h1, blk_idx_h1, size_h1, idx_h3,
                                         blk_idx_p6, size_p6, size_p7, threadIdx.x + l_fetch,
                                         rng_p6, pipeline);
          }

          if((idx_h3 < rng_h2) && (idx_h1 < rng_p4) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d2_v2_9<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d2_v2,
                                         blk_idx_p4, idx_h1, blk_idx_p5, size_p5, blk_idx_h2,
                                         size_h2, idx_h3, size_p7, threadIdx.x + l_fetch, rng_p5,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }

  // sd1_2:     t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p5,h2] * v2[h3,h1,p6,h7]
  // sd1_2':    t3[h3,h2,h1,p6,p5,p4] += v2[h3,h1,p6,h7] * t2[h7,p4,p5,h2] --> TB_X(p2,h1),
  // TB_Y(h3,p4), REG_X,Y(p5,p6) sd1_2'':   t3[h3,h2,h1,p6,p5,p4] += v2[h3,h1,p6,h7] *
  // t2[h7,p4,p5,h2] --> TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6)
  for(int iter_noab = 0; iter_noab < size_noab; iter_noab++) {
    //
    int size_h7   = const_d1_h7b[iter_noab];
    int flag_d1_2 = const_d1_exec[1 + (iter_noab) *9];

    const size_t num_batches      = (size_h7 + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;
    const size_t size_internal_up = ((size_h7 + 3) / 4) * 4;

    if(flag_d1_2 >= 0) {
      T* tmp_dev_d1_t2 = dev_d1_t2_all + size_max_dim_d1_t2 * flag_d1_2;
      T* tmp_dev_d1_v2 = dev_d1_v2_all + size_max_dim_d1_v2 * flag_d1_2;

      internal_upperbound = 0;
#pragma unroll 1
      for(size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
#pragma unroll 1
        for(; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE);
            ++fetch_batch) {
          pipeline.producer_acquire();

          const int    l_fetch    = fetch_batch * SIZE_UNIT_INT;
          const size_t shared_idx = fetch_batch % NUM_STAGE;
          // internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
          internal_offset = (l_fetch + SIZE_UNIT_INT) - size_h7;
          block.sync();

          if(internal_offset > 0) {
            const int start_row = size_h7 - l_fetch;
            const int max_row   = ((start_row + 3) / 4) * 4;
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx, start_row,
                        max_row); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx, start_row, max_row);
            block.sync();
          }

          if((idx_h3 < rng_h2) && (idx_h1 < rng_p4) &&
             threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_t2_2<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, tmp_dev_d1_t2,
                                         blk_idx_h2, idx_h3, blk_idx_p5, size_p5, blk_idx_p4,
                                         size_p4, idx_h1, size_h7, threadIdx.x + l_fetch, rng_p5,
                                         pipeline);
          }

          if((idx_h2 < rng_h1) && (idx_p6 < rng_h3) &&
             threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
            g2s_d1_v2_2<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, tmp_dev_d1_v2,
                                         blk_idx_p6, size_p6, blk_idx_h1, size_h1, idx_h2,
                                         blk_idx_h3, size_h3, idx_p6, threadIdx.y + l_fetch, rng_p6,
                                         pipeline);
          }
          pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        const int max_iter = (size_internal_up - (compute_batch * SIZE_UNIT_INT)) / 4;
#pragma unroll 1
        for(int ll = 0; ll < 4 && ll < max_iter; ll++) {
          MmaOperandA op_a;
          op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
          MmaOperandB op_b;
          op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
          mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
      }
      block.sync();
    }
  }
#endif

  //----------------------------------------------------------------------------
  //
  //    S (Singles)
  //
  //----------------------------------------------------------------------------

  //    flags
  int flag_s1_1 = const_s1_exec[0];
  int flag_s1_2 = const_s1_exec[1];
  int flag_s1_3 = const_s1_exec[2];
  int flag_s1_4 = const_s1_exec[3];
  int flag_s1_5 = const_s1_exec[4];
  int flag_s1_6 = const_s1_exec[5];
  int flag_s1_7 = const_s1_exec[6];
  int flag_s1_8 = const_s1_exec[7];
  int flag_s1_9 = const_s1_exec[8];

  //----------------------------------------------------------------------------
  // [1] TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6) by TB_X(p6,h2) and TB_Y(h1,h3)
  // [2] TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4) by TB_X(p6,h2) and TB_Y(h1,h3)
  //----------------------------------------------------------------------------
  //                                                                                                                             t1[ry,h1] * v2[h3,h2,p6,rx]
  //    s1_1: t3[h3,h2,h1,p6,p5,p4] += t1[p4,h1] * v2[h3,h2,p6,p5]
  if(flag_s1_1 >= 0) {
    T* dev_s1_t1_1 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_1;
    T* dev_s1_v2_1 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_1;

    //
    if(idx_h1 == 0 && idx_h3 == 0) {
      if(idx_p6 < rng_p4 && idx_h2 < rng_h1) {
        const int smem_idx = max(min(idx_p6 + (idx_h2) *SIZE_TILE_P4, 15), 0);
        sm_a[smem_idx]     = dev_s1_t1_1[blk_idx_p4 * SIZE_TILE_P4 + idx_p6 +
                                     (blk_idx_h1 * SIZE_TILE_H1 + idx_h2) * size_p4];
      }
    }

    if(idx_p6 < rng_h3 && idx_h2 < rng_h2 && idx_h1 < rng_p6 && idx_h3 < rng_p5) {
      sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) *SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3] =
        dev_s1_v2_1[blk_idx_h3 * SIZE_TILE_H3 + idx_p6 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h2 +
                     (blk_idx_p6 * SIZE_TILE_P6 + idx_h1 +
                      (blk_idx_p5 * SIZE_TILE_P5 + idx_h3) * size_p6) *
                       size_h2) *
                      size_h3];
    }
    block.sync();

    // TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6) by TB_X(p6(p4),h2(h3)) and TB_Y(h1(h1),h3(h2))
    op_c_s.reg[0 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[1 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[2 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[3 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];

    op_c_s.reg[0 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[1 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[2 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[3 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];

    op_c_s.reg[0 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[1 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[2 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[3 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];

    op_c_s.reg[0 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[1 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[2 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];
    op_c_s.reg[3 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h1) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h3 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H2) * SIZE_TILE_H3];

    block.sync();
  }

  // s1_2: t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h2] * v2[h3,h1,p6,p5]
  if(flag_s1_2 >= 0) {
    //
    T* dev_s1_t1_2 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_2;
    T* dev_s1_v2_2 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_2;

    if(threadIdx.y == 0) {
      if(idx_p6 < rng_p4 && idx_h2 < rng_h2) {
        sm_a[idx_p6 + (idx_h2) *SIZE_TILE_P4] =
          dev_s1_t1_2[blk_idx_p4 * SIZE_TILE_P4 + idx_p6 +
                      (blk_idx_h2 * SIZE_TILE_H2 + idx_h2) * size_p4];
      }
    }

    if(idx_p6 < rng_h3 && idx_h2 < rng_h1 && idx_h1 < rng_p6 && idx_h3 < rng_p5) {
      sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2] =
        dev_s1_v2_2[blk_idx_h3 * SIZE_TILE_H3 + idx_p6 +
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h2 +
                     (blk_idx_p6 * SIZE_TILE_P6 + idx_h1 +
                      (blk_idx_p5 * SIZE_TILE_P5 + idx_h3) * size_p6) *
                       size_h1) *
                      size_h3];
    }
    block.sync();

    //    TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4)
    // by TB_X(p6,h2), TB_Y(h1,h3)
    op_c_s.reg[0 + (0) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (0) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (0) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (0) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (1) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (1) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (1) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (1) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (2) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (2) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (2) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (2) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (3) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (3) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (3) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (3) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h3) *SIZE_TILE_P4] *
      sm_b[idx_h2 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    block.sync();
  }

  // s1_3: t3[h3,h2,h1,p6,p5,p4] += t2[p4,h3] * v2[h2,h1,p6,p5]
  if(flag_s1_3 >= 0) {
    //
    T* dev_s1_t1_3 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_3;
    T* dev_s1_v2_3 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_3;

    if(threadIdx.y == 0) {
      if(idx_p6 < rng_p4 && idx_h2 < rng_h3) {
        sm_a[idx_p6 + (idx_h2) *SIZE_TILE_P4] =
          dev_s1_t1_3[blk_idx_p4 * SIZE_TILE_P4 + idx_p6 +
                      (blk_idx_h3 * SIZE_TILE_H3 + idx_h2) * size_p4];
      }
    }

    if(idx_p6 < rng_h2 && idx_h2 < rng_h1 && idx_h1 < rng_p6 && idx_h3 < rng_p5) {
      sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2] =
        dev_s1_v2_3[blk_idx_h2 * SIZE_TILE_H2 + idx_p6 +
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h2 +
                     (blk_idx_p6 * SIZE_TILE_P6 + idx_h1 +
                      (blk_idx_p5 * SIZE_TILE_P5 + idx_h3) * size_p6) *
                       size_h1) *
                      size_h2];
    }
    block.sync();

    // TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4)
    // by TB_X(p6,h2), TB_Y(h1,h3)
    op_c_s.reg[0 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (0) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (1) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (2) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h2) *SIZE_TILE_P4] *
      sm_b[idx_h3 + (idx_h1 + (idx_p6 + (3) * SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    block.sync();
  }

  // from TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4)
  // to         TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6)
#ifdef TEST_ENABLE_RT
  rt_store_fixed(
    sm_block, idx_p6, idx_h2, idx_h1, idx_h3,
    op_c_s); // x=[(p6,h3),(p5)], y=[(h1,h2),(p4)] -> x=[(p4,h3),(p5)], y=[(h1,h2),(p6)]
  block.sync();
#pragma unroll 4
  for(int i = 0; i < 4; i++) { // p6
#pragma unroll 4
    for(int j = 0; j < 4; j++) { // p5
      op_c_s.reg[j + (i) *4] =
        sm_block[i + (idx_h2 + (j) *4) * 4 + (idx_h1 + (idx_h3 + (idx_p6) *4) * 4) * 65];
    }
  }
  block.sync();
#endif

  // s1_4: t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h1] * v2[h3,h2,p6,p4]
  if(flag_s1_4 >= 0) {
    T* dev_s1_t1_4 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_4;
    T* dev_s1_v2_4 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_4;

    if(threadIdx.y == 0) {
      if(idx_p6 < rng_p5 && idx_h2 < rng_h1) {
        sm_a[idx_p6 + (idx_h2) *SIZE_TILE_P5] =
          dev_s1_t1_4[blk_idx_p5 * SIZE_TILE_P5 + idx_p6 +
                      (blk_idx_h1 * SIZE_TILE_H1 + idx_h2) * size_p5];
      }
    }

    if(idx_p6 < rng_h3 && idx_h2 < rng_h2 && idx_h1 < rng_p6 && idx_h3 < rng_p4) {
      sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2] =
        dev_s1_v2_4[blk_idx_h3 * SIZE_TILE_H3 + idx_p6 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h2 +
                     (blk_idx_p6 * SIZE_TILE_P6 + idx_h1 +
                      (blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p6) *
                       size_h2) *
                      size_h3];
    }
    block.sync();

    //          TB_X(p4,h3), TB_Y(h1,h2)
    //          REG_X,Y(p5,p6)
    // by TB_X(p6,h2), TB_Y(h1,h3)
    op_c_s.reg[0 + (0) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (0) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (0) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (0) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (1) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (1) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (1) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (1) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (2) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (2) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (2) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (2) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (3) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (3) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (3) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (3) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h1) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    block.sync();
  }

  // s1_5: t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h2] * v2[h3,h1,p6,p4]
  if(flag_s1_5 >= 0) {
    T* dev_s1_t1_5 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_5;
    T* dev_s1_v2_5 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_5;

    if(threadIdx.y == 0) {
      if(idx_p6 < rng_p5 && idx_h2 < rng_h2) {
        sm_a[idx_p6 + (idx_h2) *SIZE_TILE_P5] =
          dev_s1_t1_5[blk_idx_p5 * SIZE_TILE_P5 + idx_p6 +
                      (blk_idx_h2 * SIZE_TILE_H2 + idx_h2) * size_p5];
      }
    }

    if(idx_p6 < rng_h3 && idx_h2 < rng_h1 && idx_h1 < rng_p6 && idx_h3 < rng_p4) {
      sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2] =
        dev_s1_v2_5[blk_idx_h3 * SIZE_TILE_H3 + idx_p6 +
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h2 +
                     (blk_idx_p6 * SIZE_TILE_P6 + idx_h1 +
                      (blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p6) *
                       size_h1) *
                      size_h3];
    }
    block.sync();

    //          TB_X(p4,h3), TB_Y(h1,h2)
    //          REG_X,Y(p5,p6)
    // by TB_X(p6,h2), TB_Y(h1,h3)
    op_c_s.reg[0 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (0) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (0) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (0) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (1) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (1) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (1) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (2) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (2) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (2) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (3) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (3) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (3) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h3) *SIZE_TILE_P5] *
      sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    block.sync();
  }

  // s1_6: t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h3] * v2[h2,h1,p6,p4]
  if(flag_s1_6 >= 0) {
    T* dev_s1_t1_6 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_6;
    T* dev_s1_v2_6 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_6;

    if(threadIdx.y == 0) {
      if(idx_p6 < rng_p5 && idx_h2 < rng_h3) {
        sm_a[idx_p6 + (idx_h2) *SIZE_TILE_P5] =
          dev_s1_t1_6[blk_idx_p5 * SIZE_TILE_P5 + idx_p6 +
                      (blk_idx_h3 * SIZE_TILE_H3 + idx_h2) * size_p5];
      }
    }

    if(idx_p6 < rng_h2 && idx_h2 < rng_h1 && idx_h1 < rng_p6 && idx_h3 < rng_p4) {
      sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2] =
        dev_s1_v2_6[blk_idx_h2 * SIZE_TILE_H2 + idx_p6 +
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h2 +
                     (blk_idx_p6 * SIZE_TILE_P6 + idx_h1 +
                      (blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p6) *
                       size_h1) *
                      size_h2];
    }
    block.sync();

    //          TB_X(p4,h3), TB_Y(h1,h2)
    //          REG_X,Y(p5,p6)
    // by TB_X(p6,h2), TB_Y(h1,h3)
    op_c_s.reg[0 + (0) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (0) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (0) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (0) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (1) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (1) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (1) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (1) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (2) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (2) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (2) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (2) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[0 + (3) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (3) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (3) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (3) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h2) *SIZE_TILE_P5] *
      sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P6) * SIZE_TILE_H1) * SIZE_TILE_H2];

    //
    block.sync();
  }

  // s1_7: t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h1] * v2[h3,h2,p5,p4]
  if(flag_s1_7 >= 0) {
    T* dev_s1_t1_7 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_7;
    T* dev_s1_v2_7 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_7;

    if(threadIdx.y == 0) {
      if(idx_p6 < rng_p6 && idx_h2 < rng_h1) {
        sm_a[idx_p6 + (idx_h2) *SIZE_TILE_P6] =
          dev_s1_t1_7[blk_idx_p6 * SIZE_TILE_P6 + idx_p6 +
                      (blk_idx_h1 * SIZE_TILE_H1 + idx_h2) * size_p6];
      }
    }

    if(idx_p6 < rng_h3 && idx_h2 < rng_h2 && idx_h1 < rng_p5 && idx_h3 < rng_p4) {
      sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) *SIZE_TILE_P5) * SIZE_TILE_H2) * SIZE_TILE_H3] =
        dev_s1_v2_7[blk_idx_h3 * SIZE_TILE_H3 + idx_p6 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h2 +
                     (blk_idx_p5 * SIZE_TILE_P5 + idx_h1 +
                      (blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p5) *
                       size_h2) *
                      size_h3];
    }
    block.sync();

    //          TB_X(p4,h3), TB_Y(h1,h2)
    //          REG_X,Y(p5,p6)
    // by TB_X(p6,h2) and TB_Y(h1,h3)
    op_c_s.reg[0 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[0 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[0 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[0 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[1 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[2 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[3 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h1) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h3 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    block.sync();
  }

  // s1_8: t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h2] * v2[h3,h1,p5,p4]
  if(flag_s1_8 >= 0) {
    T* dev_s1_t1_8 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_8;
    T* dev_s1_v2_8 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_8;

    if(threadIdx.y == 0) {
      if(idx_p6 < rng_p6 && idx_h2 < rng_h2) {
        sm_a[idx_p6 + (idx_h2) *SIZE_TILE_P6] =
          dev_s1_t1_8[blk_idx_p6 * SIZE_TILE_P6 + idx_p6 +
                      (blk_idx_h2 * SIZE_TILE_H2 + idx_h2) * size_p6];
      }
    }

    if(idx_p6 < rng_h3 && idx_h2 < rng_h1 && idx_h1 < rng_p5 && idx_h3 < rng_p4) {
      sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H3] =
        dev_s1_v2_8[blk_idx_h3 * SIZE_TILE_H3 + idx_p6 +
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h2 +
                     (blk_idx_p5 * SIZE_TILE_P5 + idx_h1 +
                      (blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p5) *
                       size_h1) *
                      size_h3];
    }
    block.sync();

    //
    // TB_X(p4,h3), TB_Y(h1,h2) (target)
    // TB_X(p6,h2), TB_Y(h1,h3) (index)
    // REG_X,Y(p5,p6)
    op_c_s.reg[0 + (0) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[0 + (1) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[0 + (2) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[0 + (3) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[1 + (0) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (1) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (2) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (3) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[2 + (0) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (1) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (2) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (3) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[3 + (0) * SIZE_TILE_P5] -=
      sm_a[0 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (1) * SIZE_TILE_P5] -=
      sm_a[1 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (2) * SIZE_TILE_P5] -=
      sm_a[2 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (3) * SIZE_TILE_P5] -=
      sm_a[3 + (idx_h3) *SIZE_TILE_P6] *
      sm_b[idx_h2 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

    block.sync();
  }

  //            TB_X(p4,h3), TB_Y(h1,h2)
  //            REG_X,Y(p5,p6)
  // by TB_X(p6,h2) and TB_Y(h1,h3)
  //                                                                                                                             t1[ry,h3] * v2[h2,h1,rx,p4]
  //    s1_9: t3[h3,h2,h1,p6,p5,p4] -= t1[p6,h3] * v2[h2,h1,p5,p4]
  if(flag_s1_9 >= 0) {
    T* dev_s1_t1_9 = dev_s1_t1_all + size_max_dim_s1_t1 * flag_s1_9;
    T* dev_s1_v2_9 = dev_s1_v2_all + size_max_dim_s1_v2 * flag_s1_9;

    if(threadIdx.y == 0) {
      if(idx_p6 < rng_p6 && idx_h2 < rng_h3) {
        sm_a[idx_p6 + (idx_h2) *SIZE_TILE_P6] =
          dev_s1_t1_9[blk_idx_p6 * SIZE_TILE_P6 + idx_p6 +
                      (blk_idx_h3 * SIZE_TILE_H3 + idx_h2) * size_p6];
      }
    }

    if(idx_p6 < rng_h2 && idx_h2 < rng_h1 && idx_h1 < rng_p5 && idx_h3 < rng_p4) {
      sm_b[idx_p6 + (idx_h2 + (idx_h1 + (idx_h3) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2] =
        dev_s1_v2_9[blk_idx_h2 * SIZE_TILE_H2 + idx_p6 +
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h2 +
                     (blk_idx_p5 * SIZE_TILE_P5 + idx_h1 +
                      (blk_idx_p4 * SIZE_TILE_P4 + idx_h3) * size_p5) *
                       size_h1) *
                      size_h2];
    }
    block.sync();

    op_c_s.reg[0 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[0 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[0 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[0 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (0 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[1 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[1 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (1 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[2 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[2 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (2 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];

    op_c_s.reg[3 + (0) * SIZE_TILE_P5] +=
      sm_a[0 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (1) * SIZE_TILE_P5] +=
      sm_a[1 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (2) * SIZE_TILE_P5] +=
      sm_a[2 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    op_c_s.reg[3 + (3) * SIZE_TILE_P5] +=
      sm_a[3 + (idx_h2) *SIZE_TILE_P6] *
      sm_b[idx_h3 + (idx_h1 + (3 + (idx_p6) *SIZE_TILE_P5) * SIZE_TILE_H1) * SIZE_TILE_H2];
    block.sync();
  }

#if 1
  block.sync();

  if(threadIdx.y == 0) {
    if(idx_h2 < rng_p6 && idx_p6 < rng_p5) {
      sm_a[idx_h2 * 4 + idx_p6] = (dev_evl_sorted_p5b[blk_idx_p5 * SIZE_TILE_P5 + idx_p6] +
                                   dev_evl_sorted_p6b[blk_idx_p6 * SIZE_TILE_P6 + idx_h2]);
    }
  }
  block.sync();
#endif
  //
  //    kernel x(idx_p6,idx_h2), y(idx_h1,idx_h3)
  //
  T energy_1 = 0.0;
  T energy_2 = 0.0;
  if(idx_p6 < rng_p4 && idx_h2 < rng_h3 && idx_h1 < rng_h1 && idx_h3 < rng_h2) {
#pragma unroll 4
    for(int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
#pragma unroll 4
      for(int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
        //
        if(idx_reg_y < rng_p6 && idx_reg_x < rng_p5) {
#if 1
          T inner_factor = (partial_inner_factor - sm_a[idx_reg_y * 4 + idx_reg_x]);
          T temp         = op_c.reg[idx_reg_y * 4 + idx_reg_x] / inner_factor;
          energy_1 += temp * op_c.reg[idx_reg_y * 4 + idx_reg_x];
          energy_2 +=
            temp * (op_c.reg[idx_reg_y * 4 + idx_reg_x] + op_c_s.reg[idx_reg_y * 4 + idx_reg_x]);
#else
          T inner_factor = partial_inner_factor -
                           dev_evl_sorted_p5b[blk_idx_p5 * SIZE_TILE_P5 + idx_reg_x] -
                           dev_evl_sorted_p6b[blk_idx_p6 * SIZE_TILE_P6 + idx_reg_y];
          energy_1 += op_c.reg[idx_reg_y * 4 + idx_reg_x] * op_c.reg[idx_reg_y * 4 + idx_reg_x] /
                      inner_factor;
          energy_2 +=
            op_c.reg[idx_reg_y * 4 + idx_reg_x] *
            (op_c.reg[idx_reg_y * 4 + idx_reg_x] + op_c_s.reg[idx_reg_y * 4 + idx_reg_x]) /
            inner_factor;
#endif
        }
      }
    }
  }
  __syncthreads();

  //
  //  to partially reduce the energies--- E(4) and E(5)
  //  a warp: 32 -(1)-> 16 -(2)-> 8 -(3)-> 4 -(4)-> 2
  //
  for(int offset = 16; offset > 0; offset /= 2) {
    energy_1 += __shfl_down_sync(FULL_MASK, energy_1, offset);
    energy_2 += __shfl_down_sync(FULL_MASK, energy_2, offset);
  }

  if(threadIdx.x == 0 && threadIdx.y % 2 == 0) {
    sm_a[threadIdx.y / 2] = energy_1;
    sm_b[threadIdx.y / 2] = energy_2;
    // atomicAdd(&dev_energy[0], energy_1);
    // atomicAdd(&dev_energy[1], energy_2);
  }
  __syncthreads();

  //
  T final_energy_1 = 0.0;
  T final_energy_2 = 0.0;
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    for(int i = 0; i < 8; i++) {
      final_energy_1 += sm_a[i];
      final_energy_2 += sm_b[i];
    }

    //
    //  [TODO] atomicAdd vs. Memcpy
    //
#if 0
    atomicAdd(&dev_energy[0], final_energy_1);
    atomicAdd(&dev_energy[1], final_energy_2);
#else
    dev_energy[blockIdx.x] = final_energy_1;
    dev_energy[blockIdx.x + gridDim.x] = final_energy_2;
#endif
  }
}

// #define DEBUG_PRINT_KERNEL_TIME
/**
 *      @brief the driver of the fully-fused kernel for CCSD(T)
 **/
template<typename T>
void ccsd_t_fully_fused_nvidia_tc_fp64(
  gpuStream_t& stream_id, size_t numBlks, size_t size_h3, size_t size_h2, size_t size_h1,
  size_t size_p6, size_t size_p5, size_t size_p4,
  //
  T* dev_s1_t1_all, T* dev_s1_v2_all, T* dev_d1_t2_all, T* dev_d1_v2_all, T* dev_d2_t2_all,
  T* dev_d2_v2_all,
  //
  int* host_size_d1_h7b, int* host_size_d2_p7b, int* host_exec_s1, int* host_exec_d1,
  int* host_exec_d2,
  //
  size_t size_noab, size_t size_nvab, size_t size_max_dim_s1_t1, size_t size_max_dim_s1_v2,
  size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2, size_t size_max_dim_d2_t2,
  size_t size_max_dim_d2_v2,
  //
  T* dev_evl_sorted_h1b, T* dev_evl_sorted_h2b, T* dev_evl_sorted_h3b, T* dev_evl_sorted_p4b,
  T* dev_evl_sorted_p5b, T* dev_evl_sorted_p6b, T* dev_energies, event_ptr_t done_copy) {
  //
  //    constant memories
  //
  cudaMemcpyToSymbolAsync(const_d1_h7b, host_size_d1_h7b, sizeof(int) * size_noab, 0,
                          cudaMemcpyHostToDevice, stream_id.first);
  cudaMemcpyToSymbolAsync(const_d2_p7b, host_size_d2_p7b, sizeof(int) * size_nvab, 0,
                          cudaMemcpyHostToDevice, stream_id.first);

  cudaMemcpyToSymbolAsync(const_s1_exec, host_exec_s1, sizeof(int) * (9), 0, cudaMemcpyHostToDevice,
                          stream_id.first);
  cudaMemcpyToSymbolAsync(const_d1_exec, host_exec_d1, sizeof(int) * (9 * size_noab), 0,
                          cudaMemcpyHostToDevice, stream_id.first);
  cudaMemcpyToSymbolAsync(const_d2_exec, host_exec_d2, sizeof(int) * (9 * size_nvab), 0,
                          cudaMemcpyHostToDevice, stream_id.first);

  CUDA_SAFE(cudaEventRecord(*done_copy, stream_id.first));

  // printf ("[new] s1: %d,%d,%d/%d,%d,%d/%d,%d,%d\n", host_exec_s1[0], host_exec_s1[1],
  // host_exec_s1[2], host_exec_s1[3], host_exec_s1[4], host_exec_s1[5], host_exec_s1[6],
  // host_exec_s1[7], host_exec_s1[8]); for (int i = 0; i < (int)size_noab; i++) {
  //    printf ("[new] noab: %d, d1: %d,%d,%d/%d,%d,%d/%d,%d,%d\n", i,
  //    host_exec_d1[0 + (i) * 9], host_exec_d1[1 + (i) * 9], host_exec_d1[2 + (i) * 9],
  //    host_exec_d1[3 + (i) * 9], host_exec_d1[4 + (i) * 9], host_exec_d1[5 + (i) * 9],
  //    host_exec_d1[6 + (i) * 9], host_exec_d1[7 + (i) * 9], host_exec_d1[8 + (i) * 9]);
  // }

  // for (int i = 0; i < (int)size_nvab; i++) {
  //    printf ("[new] nvab: %d, d2: %d,%d,%d/%d,%d,%d/%d,%d,%d\n", i,
  //    host_exec_d2[0 + (i) * 9], host_exec_d2[1 + (i) * 9], host_exec_d2[2 + (i) * 9],
  //    host_exec_d2[3 + (i) * 9], host_exec_d2[4 + (i) * 9], host_exec_d2[5 + (i) * 9],
  //    host_exec_d2[6 + (i) * 9], host_exec_d2[7 + (i) * 9], host_exec_d2[8 + (i) * 9]);
  // }

  //
  dim3 gridsize_1(numBlks);
  dim3 blocksize_1(16, 16);

  //
  // printf ("[%s] called with # blocks: %d\n", __func__, numBlks);

#ifdef DEBUG_PRINT_KERNEL_TIME
  cudaEvent_t start_kernel;
  cudaEvent_t stop_kernel;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  cudaEventRecord(start_kernel);
#endif

  // T host_energies_zero[2] = {0.0, 0.0};
  // cudaMemcpyAsync(dev_energies, host_energies_zero, sizeof(T) * 2, cudaMemcpyHostToDevice,
  // stream_id.first);

  //
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  // int maxbytes = 98304; // 96 KB
  // int maxbytes = 196608; // 192 KB
  // int maxbytes = 135168; // 132 KB
  // CUCHK(cudaFuncSetAttribute(fused_kernel_d2, cudaFuncAttributeMaxDynamicSharedMemorySize,
  // maxbytes));
  fully_fused_kernel_ccsd_t_nvidia_tc_fp64<T>
    <<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, stream_id.first>>>(
      (int) size_noab, (int) size_nvab,
      //
      (int) size_max_dim_s1_t1, (int) size_max_dim_s1_v2, (int) size_max_dim_d1_t2,
      (int) size_max_dim_d1_v2, (int) size_max_dim_d2_t2, (int) size_max_dim_d2_v2,
      //
      dev_s1_t1_all, dev_s1_v2_all, dev_d1_t2_all, dev_d1_v2_all, dev_d2_t2_all, dev_d2_v2_all,
      //
      dev_energies, dev_evl_sorted_h3b, dev_evl_sorted_h2b, dev_evl_sorted_h1b, dev_evl_sorted_p6b,
      dev_evl_sorted_p5b, dev_evl_sorted_p4b,
      //
      (int) size_h3, (int) size_h2, (int) size_h1, (int) size_p6, (int) size_p5, (int) size_p4,
      CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1),
      CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4));
  CUCHK(cudaGetLastError());

#ifdef DEBUG_PRINT_KERNEL_TIME
  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);
  float kernel_ms = 0;
  cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  printf("[%s] kernel: %f (ms)\n", __func__, kernel_ms);
#endif

  // T host_energies[2];
  // CUCHK(cudaMemcpy(host_energies, dev_energies, sizeof(T) * NUM_ENERGY,
  // cudaMemcpyDeviceToHost));

  // *final_energy_4 = factor * host_energies[0];
  // *final_energy_5 = factor * host_energies[1];
  // printf ("[%s] (gpu) energy: %.10f, %.10f\n", __func__, *final_energy_4, *final_energy_5);
}
// end of (2) 3rd. Generation Tensor Cores (FP64)

// explicit template instantiation
template void ccsd_t_fully_fused_nvidia_tc_fp64<double>(
  gpuStream_t& stream_id, size_t numBlks, size_t size_h3, size_t size_h2, size_t size_h1,
  size_t size_p6, size_t size_p5, size_t size_p4,
  //
  double* dev_s1_t1_all, double* dev_s1_v2_all, double* dev_d1_t2_all, double* dev_d1_v2_all,
  double* dev_d2_t2_all, double* dev_d2_v2_all,
  //
  int* host_size_d1_h7b, int* host_size_d2_p7b, int* host_exec_s1, int* host_exec_d1,
  int* host_exec_d2,
  //
  size_t size_noab, size_t size_nvab, size_t size_max_dim_s1_t1, size_t size_max_dim_s1_v2,
  size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2, size_t size_max_dim_d2_t2,
  size_t size_max_dim_d2_v2,
  //
  double* dev_evl_sorted_h1b, double* dev_evl_sorted_h2b, double* dev_evl_sorted_h3b,
  double* dev_evl_sorted_p4b, double* dev_evl_sorted_p5b, double* dev_evl_sorted_p6b,
  double* dev_energies, event_ptr_t done_copy);

#endif // USE_NV_TC
