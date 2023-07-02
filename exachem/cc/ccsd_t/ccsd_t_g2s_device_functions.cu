/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

//------------------------------------------------------------------------------
// sd1_1: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h1] * v2[h3,h2,p6,h7] with diff. order
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_1(double* smem, const double* __restrict__ gmem, const int blk_idx_h1,
            const int offset_h1, const int blk_idx_p5, const int size_p5, const int blk_idx_p4,
            const int size_p4, const int offset_p4, const int size_h7, const int offset_h7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem + offset_h7 +
        (blk_idx_p4 * SIZE_TILE_P4 + offset_p4 +
         (blk_idx_p5 * SIZE_TILE_P5 + ll + (blk_idx_h1 * SIZE_TILE_H1 + offset_h1) * size_p5) *
           size_p4) *
          size_h7,
      sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_1(double* smem, const double* __restrict__ gmem, const int blk_idx_p6,
            const int size_p6,                                            // p6	(2)
            const int blk_idx_h2, const int size_h2, const int offset_h2, // h2	(1)
            const int blk_idx_h3, const int size_h3, const int offset_h3, // h3 (0)
            const int offset_h7,                                          // h7	(3)
            const int length_p6, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p6; ll++) {
    cuda::memcpy_async(smem + threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
                         (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
                          (blk_idx_p6 * SIZE_TILE_P6 + ll + (offset_h7) *size_p6) * size_h2) *
                           size_h3,
                       sizeof(double), pipe);
  }
}

// sd1_2: t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p5,h2] * v2[h3,h1,p6,h7] with diff. order
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_2(double* smem, const double* __restrict__ gmem, const int blk_idx_h2,
            const int offset_h2, const int blk_idx_p5, const int size_p5, const int blk_idx_p4,
            const int size_p4, const int offset_p4, const int size_h7, const int offset_h7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem + offset_h7 +
        (blk_idx_p4 * SIZE_TILE_P4 + offset_p4 +
         (blk_idx_p5 * SIZE_TILE_P5 + ll + (blk_idx_h2 * SIZE_TILE_H2 + offset_h2) * size_p5) *
           size_p4) *
          size_h7,
      sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_2(double* smem, const double* __restrict__ gmem, const int blk_idx_p6,
            const int size_p6,                                            // p6	(2)
            const int blk_idx_h1, const int size_h1, const int offset_h1, // h1	(1)
            const int blk_idx_h3, const int size_h3, const int offset_h3, // h3 (0)
            const int offset_h7,                                          // h7	(3)
            const int length_p6, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p6; ll++) {
    cuda::memcpy_async(smem + threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
                         (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                          (blk_idx_p6 * SIZE_TILE_P6 + ll + (offset_h7) *size_p6) * size_h1) *
                           size_h3,
                       sizeof(double), pipe);
  }
}

// sd1_3: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h3] * v2[h2,h1,p6,h7]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_3(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
            const int offset_h3, const int blk_idx_p5, const int size_p5, const int blk_idx_p4,
            const int size_p4, const int offset_p4, const int size_h7, const int offset_h7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem + offset_h7 +
        (blk_idx_p4 * SIZE_TILE_P4 + offset_p4 +
         (blk_idx_p5 * SIZE_TILE_P5 + ll + (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_p5) *
           size_p4) *
          size_h7,
      sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_3(double* smem, const double* __restrict__ gmem, const int blk_idx_p6,
            const int size_p6,                                            // p6	(2)
            const int blk_idx_h1, const int size_h1, const int offset_h1, // h2	(1)
            const int blk_idx_h2, const int size_h2, const int offset_h2, // h3 (0)
            const int offset_h7,                                          // h7	(3)
            const int length_p6, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p6; ll++) {
    cuda::memcpy_async(smem + threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
                         (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                          (blk_idx_p6 * SIZE_TILE_P6 + ll + (offset_h7) *size_p6) * size_h1) *
                           size_h2,
                       sizeof(double), pipe);
  }
}

// sd1_4: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h1] * v2[h3,h2,p4,h7]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_4(double* smem, const double* __restrict__ gmem, const int blk_idx_h1,
            const int offset_h1, const int blk_idx_p6, const int size_p6, const int offset_p6,
            const int blk_idx_p5, const int size_p5, const int size_h7, const int offset_h7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + offset_h7 +
                         (blk_idx_p5 * SIZE_TILE_P5 + ll +
                          (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 +
                           (blk_idx_h1 * SIZE_TILE_H1 + offset_h1) * size_p6) *
                            size_p5) *
                           size_h7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_4_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_h1,
                   const int offset_h1, const int blk_idx_p6, const int size_p6,
                   const int offset_p6, const int blk_idx_p5, const int size_p5, const int size_h7,
                   const int offset_h7, const int length_p5) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    smem[threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride] =
      gmem[offset_h7 + (blk_idx_p5 * SIZE_TILE_P5 + ll +
                        (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 +
                         (blk_idx_h1 * SIZE_TILE_H1 + offset_h1) * size_p6) *
                          size_p5) *
                         size_h7];
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_4(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
            const int size_p4,                                            // p6	(2)
            const int blk_idx_h2, const int size_h2, const int offset_h2, // h2	(1)
            const int blk_idx_h3, const int size_h3, const int offset_h3, // h3 (0)
            const int offset_h7,                                          // h7	(3)
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(smem + threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
                         (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
                          (blk_idx_p4 * SIZE_TILE_P4 + ll + (offset_h7) *size_p4) * size_h2) *
                           size_h3,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_4_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
                   const int size_p4,                                            // p6	(2)
                   const int blk_idx_h2, const int size_h2, const int offset_h2, // h2	(1)
                   const int blk_idx_h3, const int size_h3, const int offset_h3, // h3 (0)
                   const int offset_h7,                                          // h7	(3)
                   const int length_p4) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    smem[threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride + ll * smem_i_stride] =
      gmem[blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
           (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
            (blk_idx_p4 * SIZE_TILE_P4 + ll + (offset_h7) *size_p4) * size_h2) *
             size_h3];
  }
}

// sd1_5: t3[h3,h2,h1,p6,p5,p4] += t2[h7,p5,p6,h2] * v2[h3,h1,p4,h7]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_5(double* smem, const double* __restrict__ gmem, const int blk_idx_h2,
            const int offset_h2, const int blk_idx_p6, const int size_p6, const int offset_p6,
            const int blk_idx_p5, const int size_p5, const int size_h7, const int offset_h7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + offset_h7 +
                         (blk_idx_p5 * SIZE_TILE_P5 + ll +
                          (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 +
                           (blk_idx_h2 * SIZE_TILE_H2 + offset_h2) * size_p6) *
                            size_p5) *
                           size_h7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_5(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
            const int size_p4,                                            // p6	(2)
            const int blk_idx_h1, const int size_h1, const int offset_h1, // h2	(1)
            const int blk_idx_h3, const int size_h3, const int offset_h3, // h3 (0)
            const int offset_h7,                                          // h7	(3)
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(smem + threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
                         (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                          (blk_idx_p4 * SIZE_TILE_P4 + ll + (offset_h7) *size_p4) * size_h1) *
                           size_h3,
                       sizeof(double), pipe);
  }
}

// sd1_6: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h3] * v2[h2,h1,p4,h7]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_6(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
            const int offset_h3, const int blk_idx_p6, const int size_p6, const int offset_p6,
            const int blk_idx_p5, const int size_p5, const int size_h7, const int offset_h7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + offset_h7 +
                         (blk_idx_p5 * SIZE_TILE_P5 + ll +
                          (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 +
                           (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_p6) *
                            size_p5) *
                           size_h7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_6(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
            const int size_p4,                                            // p6	(2)
            const int blk_idx_h1, const int size_h1, const int offset_h1, // h2	(1)
            const int blk_idx_h2, const int size_h2, const int offset_h2, // h3 (0)
            const int offset_h7,                                          // h7	(3)
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(smem + threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
                         (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                          (blk_idx_p4 * SIZE_TILE_P4 + ll + (offset_h7) *size_p4) * size_h1) *
                           size_h2,
                       sizeof(double), pipe);
  }
}

// sd1_7: t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h1] * v2[h3,h2,p5,h7]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_7(double* smem, const double* __restrict__ gmem, const int blk_idx_h1,
            const int offset_h1, const int blk_idx_p6, const int size_p6, const int offset_p6,
            const int blk_idx_p4, const int size_p4, const int size_h7, const int offset_h7,
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + offset_h7 +
                         (blk_idx_p4 * SIZE_TILE_P4 + ll +
                          (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 +
                           (blk_idx_h1 * SIZE_TILE_H1 + offset_h1) * size_p6) *
                            size_p4) *
                           size_h7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_7(double* smem, const double* __restrict__ gmem, const int blk_idx_p5,
            const int size_p5,                                            // p6	(2)
            const int blk_idx_h2, const int size_h2, const int offset_h2, // h2	(1)
            const int blk_idx_h3, const int size_h3, const int offset_h3, // h3 (0)
            const int offset_h7,                                          // h7	(3)
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(smem + threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
                         (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
                          (blk_idx_p5 * SIZE_TILE_P5 + ll + (offset_h7) *size_p5) * size_h2) *
                           size_h3,
                       sizeof(double), pipe);
  }
}

// sd1_8: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p6,h2] * v2[h3,h1,p5,h7]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_8(double* smem, const double* __restrict__ gmem, const int blk_idx_h2,
            const int offset_h2, const int blk_idx_p6, const int size_p6, const int offset_p6,
            const int blk_idx_p4, const int size_p4, const int size_h7, const int offset_h7,
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + offset_h7 +
                         (blk_idx_p4 * SIZE_TILE_P4 + ll +
                          (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 +
                           (blk_idx_h2 * SIZE_TILE_H2 + offset_h2) * size_p6) *
                            size_p4) *
                           size_h7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_8(double* smem, const double* __restrict__ gmem, const int blk_idx_p5,
            const int size_p5,                                            // p6	(2)
            const int blk_idx_h1, const int size_h1, const int offset_h1, // h2	(1)
            const int blk_idx_h3, const int size_h3, const int offset_h3, // h3 (0)
            const int offset_h7,                                          // h7	(3)
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(smem + threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
                         (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                          (blk_idx_p5 * SIZE_TILE_P5 + ll + (offset_h7) *size_p5) * size_h1) *
                           size_h3,
                       sizeof(double), pipe);
  }
}

// sd1_9: t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h3] * v2[h2,h1,p5,h7]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_9(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
            const int offset_h3, const int blk_idx_p6, const int size_p6, const int offset_p6,
            const int blk_idx_p4, const int size_p4, const int size_h7, const int offset_h7,
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + offset_h7 +
                         (blk_idx_p4 * SIZE_TILE_P4 + ll +
                          (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 +
                           (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_p6) *
                            size_p4) *
                           size_h7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_t2_9_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
                   const int offset_h3, const int blk_idx_p6, const int size_p6,
                   const int offset_p6, const int blk_idx_p4, const int size_p4, const int size_h7,
                   const int offset_h7, const int length_p4) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    smem[threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride] =
      gmem[offset_h7 + (blk_idx_p4 * SIZE_TILE_P4 + ll +
                        (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 +
                         (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_p6) *
                          size_p4) *
                         size_h7];
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_9(double* smem, const double* __restrict__ gmem, const int blk_idx_p5,
            const int size_p5,                                            // p6	(2)
            const int blk_idx_h1, const int size_h1, const int offset_h1, // h2	(1)
            const int blk_idx_h2, const int size_h2, const int offset_h2, // h3 (0)
            const int offset_h7,                                          // h7	(3)
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(smem + threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
                         (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                          (blk_idx_p5 * SIZE_TILE_P5 + ll + (offset_h7) *size_p5) * size_h1) *
                           size_h2,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d1_v2_9_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_p5,
                   const int size_p5,                                            // p6	(2)
                   const int blk_idx_h1, const int size_h1, const int offset_h1, // h2	(1)
                   const int blk_idx_h2, const int size_h2, const int offset_h2, // h3 (0)
                   const int offset_h7,                                          // h7	(3)
                   const int length_p5) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    smem[threadIdx.y * smem_x_stride + threadIdx.x * smem_y_stride + ll * smem_i_stride] =
      gmem[blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
           (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
            (blk_idx_p5 * SIZE_TILE_P5 + ll + (offset_h7) *size_p5) * size_h1) *
             size_h2];
  }
}

//------------------------------------------------------------------------------ device g2s
// functions
// sd2_1: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_1(double* smem, const double* __restrict__ gmem, const int blk_idx_h2,
            const int offset_h2, const int blk_idx_h1, const int size_h1, const int offset_h1,
            const int blk_idx_p4, const int size_p4, const int size_p7, const int offset_p7,
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem +
                         (blk_idx_p4 * SIZE_TILE_P4 + ll +
                          (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                           (blk_idx_h2 * SIZE_TILE_H2 + offset_h2) * size_h1) *
                            size_p4) *
                           size_p7 +
                         (offset_p7),
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_1_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_h2,
                   const int offset_h2, const int blk_idx_h1, const int size_h1,
                   const int offset_h1, const int blk_idx_p4, const int size_p4, const int size_p7,
                   const int offset_p7, const int length_p4) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    smem[threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride] =
      gmem[(blk_idx_p4 * SIZE_TILE_P4 + ll +
            (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
             (blk_idx_h2 * SIZE_TILE_H2 + offset_h2) * size_h1) *
              size_p4) *
             size_p7 +
           (offset_p7)];
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_1(double* smem, const double* __restrict__ gmem, const int blk_idx_p5,
            const int blk_idx_p6, const int size_p6, const int offset_p6, const int blk_idx_h3,
            const int size_h3, const int offset_h3, const int size_p7, const int offset_p7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem +
        (blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
         (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 + (blk_idx_p5 * SIZE_TILE_P5 + ll) * size_p6) *
           size_h3) *
          size_p7 +
        offset_p7,
      sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_1_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_p5,
                   const int blk_idx_p6, const int size_p6, const int offset_p6,
                   const int blk_idx_h3, const int size_h3, const int offset_h3, const int size_p7,
                   const int offset_p7, const int length_p5) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    smem[threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride] =
      gmem[(blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
            (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 + (blk_idx_p5 * SIZE_TILE_P5 + ll) * size_p6) *
              size_h3) *
             size_p7 +
           offset_p7];
  }
}

// sd2_2: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h2,h3] * v2[p7,h1,p6,p5]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_2(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
            const int offset_h3, const int blk_idx_h2, const int size_h2, const int offset_h2,
            const int blk_idx_p4, const int size_p4, const int size_p7, const int offset_p7,
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem +
                         (blk_idx_p4 * SIZE_TILE_P4 + ll +
                          (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
                           (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_h2) *
                            size_p4) *
                           size_p7 +
                         (offset_p7),
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_2_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
                   const int offset_h3, const int blk_idx_h2, const int size_h2,
                   const int offset_h2, const int blk_idx_p4, const int size_p4, const int size_p7,
                   const int offset_p7, const int length_p4) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    smem[threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride] =
      gmem[(blk_idx_p4 * SIZE_TILE_P4 + ll +
            (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
             (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_h2) *
              size_p4) *
             size_p7 +
           (offset_p7)];
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_2(double* smem, const double* __restrict__ gmem, const int blk_idx_p5,
            const int blk_idx_p6, const int size_p6, const int offset_p6, const int blk_idx_h1,
            const int size_h1, const int offset_h1, const int size_p7, const int offset_p7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem +
        (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
         (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 + (blk_idx_p5 * SIZE_TILE_P5 + ll) * size_p6) *
           size_h1) *
          size_p7 +
        offset_p7,
      sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_2_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_p5,
                   const int blk_idx_p6, const int size_p6, const int offset_p6,
                   const int blk_idx_h1, const int size_h1, const int offset_h1, const int size_p7,
                   const int offset_p7, const int length_p5) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    smem[threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride] =
      gmem[(blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
            (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 + (blk_idx_p5 * SIZE_TILE_P5 + ll) * size_p6) *
              size_h1) *
             size_p7 +
           offset_p7];
  }
}

// sd2_3: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p4,h1,h3] * v2[p7,h2,p6,p5]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_3(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
            const int offset_h3, const int blk_idx_h1, const int size_h1, const int offset_h1,
            const int blk_idx_p4, const int size_p4, const int size_p7, const int offset_p7,
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem +
                         (blk_idx_p4 * SIZE_TILE_P4 + ll +
                          (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                           (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_h1) *
                            size_p4) *
                           size_p7 +
                         (offset_p7),
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_3_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
                   const int offset_h3, const int blk_idx_h1, const int size_h1,
                   const int offset_h1, const int blk_idx_p4, const int size_p4, const int size_p7,
                   const int offset_p7, const int length_p4) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    smem[threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride] =
      gmem[(blk_idx_p4 * SIZE_TILE_P4 + ll +
            (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
             (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_h1) *
              size_p4) *
             size_p7 +
           (offset_p7)];
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_3(double* smem, const double* __restrict__ gmem, const int blk_idx_p5,
            const int blk_idx_p6, const int size_p6, const int offset_p6, const int blk_idx_h2,
            const int size_h2, const int offset_h2, const int size_p7, const int offset_p7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem +
        (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
         (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 + (blk_idx_p5 * SIZE_TILE_P5 + ll) * size_p6) *
           size_h2) *
          size_p7 +
        offset_p7,
      sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_3_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_p5,
                   const int blk_idx_p6, const int size_p6, const int offset_p6,
                   const int blk_idx_h2, const int size_h2, const int offset_h2, const int size_p7,
                   const int offset_p7, const int length_p5) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    smem[threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride] =
      gmem[(blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
            (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 + (blk_idx_p5 * SIZE_TILE_P5 + ll) * size_p6) *
              size_h2) *
             size_p7 +
           offset_p7];
  }
}

// sd2_4: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5(r),h1,h2] * v2[p7,h3,p6(r),p4]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_4(double* smem, const double* __restrict__ gmem, const int blk_idx_h2,
            const int offset_h2, const int blk_idx_h1, const int size_h1, const int offset_h1,
            const int blk_idx_p5, const int size_p5, const int size_p7, const int offset_p7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem +
                         (blk_idx_p5 * SIZE_TILE_P5 + ll +
                          (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                           (blk_idx_h2 * SIZE_TILE_H2 + offset_h2) * size_h1) *
                            size_p5) *
                           size_p7 +
                         offset_p7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_4_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_h2,
                   const int offset_h2, const int blk_idx_h1, const int size_h1,
                   const int offset_h1, const int blk_idx_p5, const int size_p5, const int size_p7,
                   const int offset_p7, const int length_p5) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    smem[threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride] =
      gmem[(blk_idx_p5 * SIZE_TILE_P5 + ll +
            (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
             (blk_idx_h2 * SIZE_TILE_H2 + offset_h2) * size_h1) *
              size_p5) *
             size_p7 +
           offset_p7];
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_4(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
            const int blk_idx_p6, const int size_p6, const int offset_p6, const int blk_idx_h3,
            const int size_h3, const int offset_h3, const int size_p7, const int offset_p7,
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem +
        (blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
         (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 + (blk_idx_p4 * SIZE_TILE_P4 + ll) * size_p6) *
           size_h3) *
          size_p7 +
        offset_p7,
      sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_4_normal(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
                   const int blk_idx_p6, const int size_p6, const int offset_p6,
                   const int blk_idx_h3, const int size_h3, const int offset_h3, const int size_p7,
                   const int offset_p7, const int length_p4) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    smem[threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride] =
      gmem[(blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
            (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 + (blk_idx_p4 * SIZE_TILE_P4 + ll) * size_p6) *
              size_h3) *
             size_p7 +
           offset_p7];
  }
}

// sd2_5: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h2,h3] * v2[p7,h1,p6,p4]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_5(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
            const int offset_h3, const int blk_idx_h2, const int size_h2, const int offset_h2,
            const int blk_idx_p5, const int size_p5, const int size_p7, const int offset_p7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem +
                         (blk_idx_p5 * SIZE_TILE_P5 + ll +
                          (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
                           (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_h2) *
                            size_p5) *
                           size_p7 +
                         offset_p7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_5(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
            const int blk_idx_p6, const int size_p6, const int offset_p6, const int blk_idx_h1,
            const int size_h1, const int offset_h1, const int size_p7, const int offset_p7,
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem +
        (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
         (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 + (blk_idx_p4 * SIZE_TILE_P4 + ll) * size_p6) *
           size_h1) *
          size_p7 +
        offset_p7,
      sizeof(double), pipe);
  }
}

// sd2_6: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p5,h1,h3] * v2[p7,h2,p6,p4]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_6(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
            const int offset_h3, const int blk_idx_h1, const int size_h1, const int offset_h1,
            const int blk_idx_p5, const int size_p5, const int size_p7, const int offset_p7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem +
                         (blk_idx_p5 * SIZE_TILE_P5 + ll +
                          (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                           (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_h1) *
                            size_p5) *
                           size_p7 +
                         offset_p7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_6(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
            const int blk_idx_p6, const int size_p6, const int offset_p6, const int blk_idx_h2,
            const int size_h2, const int offset_h2, const int size_p7, const int offset_p7,
            const int length_p4, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p4; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem +
        (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
         (blk_idx_p6 * SIZE_TILE_P6 + offset_p6 + (blk_idx_p4 * SIZE_TILE_P4 + ll) * size_p6) *
           size_h2) *
          size_p7 +
        offset_p7,
      sizeof(double), pipe);
  }
}

// sd2_7: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h1,h2] * v2[p7,h3,p5,p4]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_7(double* smem, const double* __restrict__ gmem, const int blk_idx_h2,
            const int offset_h2, const int blk_idx_h1, const int size_h1, const int offset_h1,
            const int blk_idx_p6, const int size_p6, const int size_p7, const int offset_p7,
            const int length_p6, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p6; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem +
                         (blk_idx_p6 * SIZE_TILE_P6 + ll +
                          (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                           (blk_idx_h2 * SIZE_TILE_H2 + offset_h2) * size_h1) *
                            size_p6) *
                           size_p7 +
                         offset_p7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_7(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
            const int offset_p4, const int blk_idx_p5, const int size_p5, const int blk_idx_h3,
            const int size_h3, const int offset_h3, const int size_p7, const int offset_p7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem +
        (blk_idx_h3 * SIZE_TILE_H3 + offset_h3 +
         (blk_idx_p5 * SIZE_TILE_P5 + ll + (blk_idx_p4 * SIZE_TILE_P4 + offset_p4) * size_p5) *
           size_h3) *
          size_p7 +
        offset_p7,
      sizeof(double), pipe);
  }
}

// sd2_8: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h2,h3] * v2[p7,h1,p5,p4]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_8(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
            const int offset_h3, const int blk_idx_h2, const int size_h2, const int offset_h2,
            const int blk_idx_p6, const int size_p6, const int size_p7, const int offset_p7,
            const int length_p6, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p6; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem +
                         (blk_idx_p6 * SIZE_TILE_P6 + ll +
                          (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
                           (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_h2) *
                            size_p6) *
                           size_p7 +
                         offset_p7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_8(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
            const int offset_p4, const int blk_idx_p5, const int size_p5, const int blk_idx_h1,
            const int size_h1, const int offset_h1, const int size_p7, const int offset_p7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem +
        (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
         (blk_idx_p5 * SIZE_TILE_P5 + ll + (blk_idx_p4 * SIZE_TILE_P4 + offset_p4) * size_p5) *
           size_h1) *
          size_p7 +
        offset_p7,
      sizeof(double), pipe);
  }
}

// sd2_9: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p6,h1,h3] * v2[p7,h2,p5,p4]
template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_t2_9(double* smem, const double* __restrict__ gmem, const int blk_idx_h3,
            const int offset_h3, const int blk_idx_h1, const int size_h1, const int offset_h1,
            const int blk_idx_p6, const int size_p6, const int size_p7, const int offset_p7,
            const int length_p6, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p6; ll++) {
    cuda::memcpy_async(smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride +
                         ll * smem_i_stride,
                       gmem + offset_p7 +
                         (blk_idx_p6 * SIZE_TILE_P6 + ll +
                          (blk_idx_h1 * SIZE_TILE_H1 + offset_h1 +
                           (blk_idx_h3 * SIZE_TILE_H3 + offset_h3) * size_h1) *
                            size_p6) *
                           size_p7,
                       sizeof(double), pipe);
  }
}

template<int smem_i_stride, int smem_x_stride, int smem_y_stride>
__device__ inline void
g2s_d2_v2_9(double* smem, const double* __restrict__ gmem, const int blk_idx_p4,
            const int offset_p4, const int blk_idx_p5, const int size_p5, const int blk_idx_h2,
            const int size_h2, const int offset_h2, const int size_p7, const int offset_p7,
            const int length_p5, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
#pragma unroll 4
  for(int ll = 0; ll < length_p5; ll++) {
    cuda::memcpy_async(
      smem + threadIdx.x * smem_x_stride + threadIdx.y * smem_y_stride + ll * smem_i_stride,
      gmem +
        (blk_idx_h2 * SIZE_TILE_H2 + offset_h2 +
         (blk_idx_p5 * SIZE_TILE_P5 + ll + (blk_idx_p4 * SIZE_TILE_P4 + offset_p4) * size_p5) *
           size_h2) *
          size_p7 +
        offset_p7,
      sizeof(double), pipe);
  }
}