/* Attention helpers: scaled row softmax to fp16 and head split / transpose. */
#include "cuda_common.cuh"

#include <math.h>

static __device__ float block_reduce(float v, float *sh, bool is_max) {
  for (int o = 16; o > 0; o >>= 1) {
    float u = __shfl_xor_sync(0xffffffffu, v, o);
    v = is_max ? fmaxf(v, u) : v + u;
  }
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) {
    sh[wid] = v;
  }
  __syncthreads();
  int nw = (blockDim.x + 31) >> 5;
  v = lane < nw ? sh[lane] : (is_max ? -INFINITY : 0.0f);
  for (int o = 16; o > 0; o >>= 1) {
    float u = __shfl_xor_sync(0xffffffffu, v, o);
    v = is_max ? fmaxf(v, u) : v + u;
  }
  return v;
}

__global__ void k_softmax_to_half(const float *__restrict__ s, int64_t lds,
                                  __half *__restrict__ p, int64_t ldp, int64_t L, float scale) {
  __shared__ float sh[32];
  const float *row = s + (int64_t)blockIdx.x * lds;
  __half *out = p + (int64_t)blockIdx.x * ldp;
  float mx = -INFINITY;
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    mx = fmaxf(mx, row[j] * scale);
  }
  mx = block_reduce(mx, sh, true);
  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    sum += expf(row[j] * scale - mx);
  }
  float inv = 1.0f / block_reduce(sum, sh, false);
  for (int64_t j = threadIdx.x; j < ldp; j += blockDim.x) {
    out[j] = dk_to_h(j < L ? expf(row[j] * scale - mx) * inv : 0.0f);
  }
}

extern "C" void dk_softmax_to_half(const float *s, int64_t lds, dc_half *p, int64_t ldp,
                                   int64_t rows, int64_t L, float scale) {
  if (rows == 0) {
    return;
  }
  if (rows > 0x7fffffff) {
    dc_die("softmax: too many rows");
  }
  k_softmax_to_half<<<(unsigned)rows, 256, 0, dk_stream()>>>(s, lds, (__half *)p, ldp, L,
                                                              scale);
  DK_LAUNCH_CHECK();
}

/* transpose=0: y [B][H][T][hdp]; transpose=1: y [B][H][hd][Tp] (padding zeroed) */
__global__ void k_split_heads(const float *__restrict__ x, __half *__restrict__ y, int64_t T,
                              int H, int hd, int hdp, int64_t Tp, int transpose, int64_t n) {
  int64_t D = (int64_t)H * hd;
  DK_GRID_STRIDE(i, n) {
    int64_t t, d, bh;
    if (!transpose) {
      d = i % hdp;
      t = (i / hdp) % T;
      bh = i / (hdp * T);
    } else {
      t = i % Tp;
      d = (i / Tp) % hd;
      bh = i / (Tp * hd);
    }
    int64_t b = bh / H, h = bh % H;
    bool valid = transpose ? t < T : d < hd;
    y[i] = dk_to_h(valid ? x[(b * T + t) * D + h * hd + d] : 0.0f);
  }
}

extern "C" void dk_split_heads_half(const float *x, dc_half *y, int64_t B, int64_t T, int H,
                                    int hd, int hdp, int64_t Tp, int transpose) {
  int64_t n = transpose ? B * H * (int64_t)hd * Tp : B * H * T * (int64_t)hdp;
  if (n == 0) {
    return;
  }
  k_split_heads<<<dk_blocks(n, 256), 256, 0, dk_stream()>>>(x, (__half *)y, T, H, hd, hdp, Tp,
                                                             transpose, n);
  DK_LAUNCH_CHECK();
}
