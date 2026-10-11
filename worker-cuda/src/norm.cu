/* Row statistics and normalisation kernels (all fp32). */
#include "cuda_common.cuh"

#define ST_THREADS 256
#define ST_CHUNK 16384

/* grow-only scratch; safe to reuse because all work is ordered on one stream */
static float *scratch_floats(int slot, size_t n) {
  static float *buf[2] = {nullptr, nullptr};
  static size_t cap[2] = {0, 0};
  if (n > cap[slot]) {
    if (buf[slot]) {
      DK_CHECK(cudaStreamSynchronize(dk_stream()));
      DK_CHECK(cudaFree(buf[slot]));
    }
    size_t want = n < 4096 ? 4096 : n * 2;
    DK_CHECK(cudaMalloc((void **)&buf[slot], want * sizeof(float)));
    cap[slot] = want;
  }
  return buf[slot];
}

/* block-wide sum; every thread receives the result */
static __device__ float block_sum(float v, float *sh) {
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) {
    sh[wid] = v;
  }
  __syncthreads();
  int nw = (blockDim.x + 31) >> 5;
  v = lane < nw ? sh[lane] : 0.0f;
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}

/* per (row, chunk): count, mean and M2 with a two-pass over the chunk */
__global__ void k_chunk_stats(const float *__restrict__ x, int64_t L, int nchunks,
                              float *__restrict__ cmean, float *__restrict__ cm2) {
  __shared__ float sh[32];
  int64_t row = blockIdx.x / nchunks;
  int chunk = blockIdx.x % nchunks;
  int64_t c0 = (int64_t)chunk * ST_CHUNK;
  int64_t c1 = c0 + ST_CHUNK < L ? c0 + ST_CHUNK : L;
  const float *xr = x + row * L;
  float s = 0.0f;
  for (int64_t i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
    s += xr[i];
  }
  float n = (float)(c1 - c0);
  float mean = block_sum(s, sh) / n;
  float q = 0.0f;
  for (int64_t i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
    float d = xr[i] - mean;
    q += d * d;
  }
  float m2 = block_sum(q, sh);
  if (threadIdx.x == 0) {
    cmean[blockIdx.x] = mean;
    cm2[blockIdx.x] = m2;
  }
}

/* Chan et al. pairwise combination of the chunk statistics, in double */
__global__ void k_combine_stats(const float *__restrict__ cmean, const float *__restrict__ cm2,
                                int64_t rows, int64_t L, int nchunks, float *__restrict__ mean,
                                float *__restrict__ var) {
  int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  double n = 0.0, m = 0.0, m2 = 0.0;
  for (int c = 0; c < nchunks; c++) {
    int64_t c0 = (int64_t)c * ST_CHUNK;
    double nb = (double)((c0 + ST_CHUNK < L ? c0 + ST_CHUNK : L) - c0);
    double mb = cmean[r * nchunks + c], qb = cm2[r * nchunks + c];
    double nn = n + nb, d = mb - m;
    m += d * nb / nn;
    m2 += qb + d * d * n * nb / nn;
    n = nn;
  }
  mean[r] = (float)m;
  var[r] = (float)(m2 / n);
}

extern "C" void dk_row_stats(const float *x, int64_t rows, int64_t L, float *mean, float *var) {
  if (rows == 0) {
    return;
  }
  int nchunks = (int)((L + ST_CHUNK - 1) / ST_CHUNK);
  int64_t nblk = rows * nchunks;
  if (nblk > 0x7fffffff) {
    dc_die("row_stats: too many rows");
  }
  float *tmp = scratch_floats(0, (size_t)nblk * 2);
  int threads = L < ST_THREADS ? ((int)L + 31) / 32 * 32 : ST_THREADS;
  k_chunk_stats<<<(unsigned)nblk, threads, 0, dk_stream()>>>(x, L, nchunks, tmp, tmp + nblk);
  DK_LAUNCH_CHECK();
  k_combine_stats<<<dk_blocks(rows, 128), 128, 0, dk_stream()>>>(tmp, tmp + nblk, rows, L,
                                                                 nchunks, mean, var);
  DK_LAUNCH_CHECK();
}

__global__ void k_groupnorm(const float *__restrict__ x, float *__restrict__ y,
                            const float *__restrict__ mean, const float *__restrict__ var,
                            const float *__restrict__ w, const float *__restrict__ b, int64_t C,
                            int64_t S, int64_t G, float eps, int act, int64_t n) {
  int64_t cpg = C / G;
  DK_GRID_STRIDE(i, n) {
    int64_t t = i / S;
    int64_t c = t % C, bi = t / C;
    int64_t g = c / cpg;
    float v = (x[i] - mean[bi * G + g]) * rsqrtf(var[bi * G + g] + eps) * w[c] + b[c];
    y[i] = act == 1 ? dk_gelu(v) : v;
  }
}

__global__ void k_groupnorm_glu(const float *__restrict__ x, float *__restrict__ y,
                                const float *__restrict__ mean, const float *__restrict__ var,
                                const float *__restrict__ w, const float *__restrict__ b,
                                int64_t C, int64_t S, int64_t G, float eps, int64_t n) {
  int64_t cpg = C / G, H = C / 2;
  DK_GRID_STRIDE(i, n) {
    int64_t s = i % S, t = i / S;
    int64_t c = t % H, bi = t / H;
    int64_t ga = c / cpg, gb = (c + H) / cpg;
    float xa = x[(bi * C + c) * S + s], xb = x[(bi * C + c + H) * S + s];
    float na = (xa - mean[bi * G + ga]) * rsqrtf(var[bi * G + ga] + eps) * w[c] + b[c];
    float nb = (xb - mean[bi * G + gb]) * rsqrtf(var[bi * G + gb] + eps) * w[c + H] + b[c + H];
    y[i] = na * dk_sigmoid(nb);
  }
}

extern "C" void dk_groupnorm_apply(const float *x, float *y, const float *mean,
                                   const float *var, const float *w, const float *b, int64_t B,
                                   int64_t C, int64_t S, int64_t G, float eps, int act) {
  if (act == 2) {
    int64_t n = B * (C / 2) * S;
    if (n) {
      k_groupnorm_glu<<<dk_blocks(n, 256), 256, 0, dk_stream()>>>(x, y, mean, var, w, b, C, S,
                                                                   G, eps, n);
    }
  } else {
    int64_t n = B * C * S;
    if (n) {
      k_groupnorm<<<dk_blocks(n, 256), 256, 0, dk_stream()>>>(x, y, mean, var, w, b, C, S, G,
                                                               eps, act, n);
    }
  }
  DK_LAUNCH_CHECK();
}

__global__ void k_norm_last(const float *__restrict__ x, float *__restrict__ y,
                            const float *__restrict__ mean, const float *__restrict__ var,
                            const float *__restrict__ w, const float *__restrict__ b, int64_t L,
                            int64_t C, float eps, int64_t n) {
  DK_GRID_STRIDE(i, n) {
    int64_t r = i / L, c = (i % L) % C;
    float v = (x[i] - mean[r]) * rsqrtf(var[r] + eps);
    if (w) {
      v *= w[c];
    }
    if (b) {
      v += b[c];
    }
    y[i] = v;
  }
}

extern "C" void dk_norm_apply_last(const float *x, float *y, const float *mean,
                                   const float *var, const float *w, const float *b,
                                   int64_t rows, int64_t L, int64_t C, float eps) {
  int64_t n = rows * L;
  if (n) {
    k_norm_last<<<dk_blocks(n, 256), 256, 0, dk_stream()>>>(x, y, mean, var, w, b, L, C, eps,
                                                             n);
    DK_LAUNCH_CHECK();
  }
}

__global__ void k_affine_rows(const float *__restrict__ x, float *__restrict__ y,
                              const float *__restrict__ mean, const float *__restrict__ var,
                              int64_t L, float eps, int denorm, int64_t n) {
  DK_GRID_STRIDE(i, n) {
    int64_t r = i / L;
    float sd = sqrtf(var[r]) + eps, m = mean[r];
    y[i] = denorm ? x[i] * sd + m : (x[i] - m) / sd;
  }
}

extern "C" void dk_affine_rows(const float *x, float *y, const float *mean, const float *var,
                               int64_t rows, int64_t L, float eps, int denorm) {
  int64_t n = rows * L;
  if (n) {
    k_affine_rows<<<dk_blocks(n, 256), 256, 0, dk_stream()>>>(x, y, mean, var, L, eps, denorm,
                                                               n);
    DK_LAUNCH_CHECK();
  }
}

/* one block per row; the row stays in registers between the passes (C <= 32 * 128) */
#define LN_THREADS 128
#define LN_PER_THREAD 32

__global__ void k_layernorm(const float *__restrict__ x, float *__restrict__ y,
                            const float *__restrict__ w, const float *__restrict__ b, int64_t C,
                            float eps) {
  __shared__ float sh[32];
  const float *xr = x + (int64_t)blockIdx.x * C;
  float *yr = y + (int64_t)blockIdx.x * C;
  float v[LN_PER_THREAD];
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < LN_PER_THREAD; j++) {
    int64_t c = threadIdx.x + (int64_t)j * LN_THREADS;
    v[j] = c < C ? xr[c] : 0.0f;
    s += v[j];
  }
  float mean = block_sum(s, sh) / (float)C;
  float q = 0.0f;
#pragma unroll
  for (int j = 0; j < LN_PER_THREAD; j++) {
    int64_t c = threadIdx.x + (int64_t)j * LN_THREADS;
    float d = c < C ? v[j] - mean : 0.0f;
    q += d * d;
  }
  float rs = rsqrtf(block_sum(q, sh) / (float)C + eps);
#pragma unroll
  for (int j = 0; j < LN_PER_THREAD; j++) {
    int64_t c = threadIdx.x + (int64_t)j * LN_THREADS;
    if (c < C) {
      float o = (v[j] - mean) * rs;
      if (w) {
        o *= w[c];
      }
      if (b) {
        o += b[c];
      }
      yr[c] = o;
    }
  }
}

extern "C" void dk_layernorm(const float *x, float *y, const float *w, const float *b,
                             int64_t rows, int64_t C, float eps) {
  if (rows == 0) {
    return;
  }
  if (C > (int64_t)LN_THREADS * LN_PER_THREAD || rows > 0x7fffffff) {
    float *stats = scratch_floats(1, (size_t)rows * 2);
    dk_row_stats(x, rows, C, stats, stats + rows);
    dk_norm_apply_last(x, y, stats, stats + rows, w, b, rows, C, C, eps);
    return;
  }
  k_layernorm<<<(unsigned)rows, LN_THREADS, 0, dk_stream()>>>(x, y, w, b, C, eps);
  DK_LAUNCH_CHECK();
}
