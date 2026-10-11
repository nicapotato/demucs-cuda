#ifndef DEMUCS_CUDA_COMMON_CUH
#define DEMUCS_CUDA_COMMON_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "backend.h"

extern "C" {
[[noreturn]] void dc_die(const char *fmt, ...);
uint16_t dc_f32_to_f16(float f);
}

cudaStream_t dk_stream(void);

#define DK_CHECK(call)                                                                     \
  do {                                                                                     \
    cudaError_t dk_err_ = (call);                                                          \
    if (dk_err_ != cudaSuccess) {                                                          \
      dc_die("CUDA error %s at %s:%d: %s", cudaGetErrorName(dk_err_), __FILE__, __LINE__,  \
             cudaGetErrorString(dk_err_));                                                 \
    }                                                                                      \
  } while (0)

#define DK_LAUNCH_CHECK() DK_CHECK(cudaGetLastError())

static inline unsigned dk_blocks(int64_t n, int threads) {
  int64_t b = (n + threads - 1) / threads;
  if (b > 65535LL * 32) {
    b = 65535LL * 32; /* grid-stride loops cover the rest */
  }
  return (unsigned)(b < 1 ? 1 : b);
}

#define DK_GRID_STRIDE(i, n)                                                               \
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n);                \
       i += (int64_t)blockDim.x * gridDim.x)

static __device__ __forceinline__ __half dk_to_h(float v) { return __float2half_rn(v); }

static __device__ __forceinline__ float dk_gelu(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

static __device__ __forceinline__ float dk_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

#endif
