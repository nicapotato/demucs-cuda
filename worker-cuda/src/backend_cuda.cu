/* CUDA runtime, elementwise and convolution-operand kernels. */
#include "cuda_common.cuh"

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DK_HAVE_NVTX 1
#endif
#include <stdio.h>
#include <string.h>

static cudaStream_t g_stream = nullptr;
static char g_name[256] = "cuda";

cudaStream_t dk_stream(void) { return g_stream; }

extern "C" int dk_init(int device, char *name, size_t name_cap) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    snprintf(name, name_cap, "no NVIDIA GPU / driver found (%s)",
             err != cudaSuccess ? cudaGetErrorString(err) : "0 devices");
    return -1;
  }
  if (device < 0 || device >= count) {
    snprintf(name, name_cap, "device %d out of range (%d devices)", device, count);
    return -1;
  }
  cudaDeviceProp prop;
  DK_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major * 10 + prop.minor < 75) {
    snprintf(name, name_cap, "%s (sm_%d%d) is older than Turing; sm_75+ required", prop.name,
             prop.major, prop.minor);
    return -1;
  }
  err = cudaSetDevice(device);
  if (err == cudaSuccess) {
    err = cudaFree(0);
  }
  if (err != cudaSuccess) {
    snprintf(name, name_cap, "cannot create a CUDA context on %s: %s (driver too old?)",
             prop.name, cudaGetErrorString(err));
    return -1;
  }
  DK_CHECK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  snprintf(g_name, sizeof(g_name), "%.200s sm_%d%d %.1fGiB", prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (double)(1u << 30));
  snprintf(name, name_cap, "%s", g_name);
  return 0;
}

extern "C" const char *dk_backend_name(void) { return "cuda"; }

extern "C" void dk_sync(void) { DK_CHECK(cudaStreamSynchronize(g_stream)); }

extern "C" void *dk_raw_malloc(size_t bytes) {
  void *p = nullptr;
  if (cudaMalloc(&p, bytes ? bytes : 1) != cudaSuccess) {
    (void)cudaGetLastError();
    return nullptr;
  }
  return p;
}

extern "C" void dk_raw_free(void *p) {
  if (p) {
    DK_CHECK(cudaStreamSynchronize(g_stream));
    DK_CHECK(cudaFree(p));
  }
}

extern "C" void dk_memcpy_h2d(void *dst, const void *src, size_t bytes) {
  if (bytes) {
    DK_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, g_stream));
    DK_CHECK(cudaStreamSynchronize(g_stream));
  }
}

extern "C" void dk_memcpy_d2h(void *dst, const void *src, size_t bytes) {
  if (bytes) {
    DK_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, g_stream));
    DK_CHECK(cudaStreamSynchronize(g_stream));
  }
}

extern "C" void dk_memcpy_d2d(void *dst, const void *src, size_t bytes) {
  if (bytes) {
    DK_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, g_stream));
  }
}

extern "C" void dk_memset0(void *dst, size_t bytes) {
  if (bytes) {
    DK_CHECK(cudaMemsetAsync(dst, 0, bytes, g_stream));
  }
}

extern "C" int dk_mem_info(size_t *free_bytes, size_t *total_bytes) {
  return cudaMemGetInfo(free_bytes, total_bytes) == cudaSuccess ? 0 : -1;
}

#ifdef DK_HAVE_NVTX
extern "C" void dk_range_push(const char *name) { nvtxRangePushA(name); }

extern "C" void dk_range_pop(void) { nvtxRangePop(); }
#else
extern "C" void dk_range_push(const char *name) { (void)name; }

extern "C" void dk_range_pop(void) {}
#endif

extern "C" dc_half dk_pack_half(float v, int weight) {
  (void)weight;
  return dc_f32_to_f16(v);
}

/* ---------------------------------------------------------------- strided */

struct DkIdx {
  int ndim;
  int64_t shape[DK_MAX_DIM];
  int64_t a[DK_MAX_DIM];
  int64_t b[DK_MAX_DIM];
};

static int64_t strided_total(const DkStrided *p) {
  int64_t n = 1;
  for (int i = 0; i < p->ndim; i++) {
    n *= p->shape[i];
  }
  return n;
}

static DkIdx to_idx(const DkStrided *p) {
  DkIdx q;
  memset(&q, 0, sizeof(q));
  q.ndim = p->ndim;
  for (int i = 0; i < p->ndim; i++) {
    q.shape[i] = p->shape[i];
    q.a[i] = p->a_strides[i];
    q.b[i] = p->b_strides[i];
  }
  return q;
}

static __device__ __forceinline__ void unravel(const DkIdx &q, int64_t e, int64_t &ao,
                                               int64_t &bo) {
  ao = 0;
  bo = 0;
  for (int d = q.ndim - 1; d >= 0; d--) {
    int64_t s = q.shape[d];
    int64_t i = e % s;
    e /= s;
    ao += i * q.a[d];
    bo += i * q.b[d];
  }
}

__global__ void k_strided_copy(DkIdx q, int64_t n, const float *__restrict__ src,
                               float *__restrict__ dst, int accumulate) {
  DK_GRID_STRIDE(e, n) {
    int64_t ao, bo;
    unravel(q, e, ao, bo);
    if (accumulate) {
      dst[bo] += src[ao];
    } else {
      dst[bo] = src[ao];
    }
  }
}

extern "C" void dk_strided_copy(const DkStrided *p, const float *src, float *dst,
                                int accumulate) {
  int64_t n = strided_total(p);
  if (n == 0) {
    return;
  }
  k_strided_copy<<<dk_blocks(n, 256), 256, 0, g_stream>>>(to_idx(p), n, src, dst, accumulate);
  DK_LAUNCH_CHECK();
}

__global__ void k_binary(int op, DkIdx q, int64_t n, const float *__restrict__ a,
                         const float *__restrict__ b, float *__restrict__ y) {
  DK_GRID_STRIDE(e, n) {
    int64_t ao, bo;
    unravel(q, e, ao, bo);
    float x = a[ao], z = b[bo], r;
    switch (op) {
    case DK_ADD:
      r = x + z;
      break;
    case DK_SUB:
      r = x - z;
      break;
    case DK_MUL:
      r = x * z;
      break;
    case DK_DIV:
      r = x / z;
      break;
    default:
      r = x > z ? x : z;
      break;
    }
    y[e] = r;
  }
}

extern "C" void dk_binary(DkBinOp op, const DkStrided *p, const float *a, const float *b,
                          float *y) {
  int64_t n = strided_total(p);
  if (n == 0) {
    return;
  }
  k_binary<<<dk_blocks(n, 256), 256, 0, g_stream>>>((int)op, to_idx(p), n, a, b, y);
  DK_LAUNCH_CHECK();
}

/* ---------------------------------------------------------- elementwise */

__global__ void k_unary(int op, const float *__restrict__ x, float *__restrict__ y, int64_t n,
                        float alpha, float beta) {
  DK_GRID_STRIDE(i, n) {
    float v = x[i];
    switch (op) {
    case DK_UN_AFFINE:
      v = v * alpha + beta;
      break;
    case DK_UN_GELU:
      v = dk_gelu(v);
      break;
    case DK_UN_RELU:
      v = v > 0.0f ? v : 0.0f;
      break;
    default:
      v = dk_sigmoid(v);
      break;
    }
    y[i] = v;
  }
}

extern "C" void dk_unary(DkUnOp op, const float *x, float *y, int64_t n, float alpha,
                         float beta) {
  if (n == 0) {
    return;
  }
  k_unary<<<dk_blocks(n, 256), 256, 0, g_stream>>>((int)op, x, y, n, alpha, beta);
  DK_LAUNCH_CHECK();
}

__global__ void k_fill(float *y, int64_t n, float v) {
  DK_GRID_STRIDE(i, n) { y[i] = v; }
}

extern "C" void dk_fill(float *y, int64_t n, float v) {
  if (n == 0) {
    return;
  }
  k_fill<<<dk_blocks(n, 256), 256, 0, g_stream>>>(y, n, v);
  DK_LAUNCH_CHECK();
}

__global__ void k_glu(const float *__restrict__ x, float *__restrict__ y, int64_t outer,
                      int64_t hi, int64_t n) {
  DK_GRID_STRIDE(i, n) {
    int64_t o = i / hi, r = i % hi;
    const float *base = x + o * 2 * hi;
    y[i] = base[r] * dk_sigmoid(base[hi + r]);
  }
}

extern "C" void dk_glu(const float *x, float *y, int64_t outer, int64_t half, int64_t inner) {
  int64_t hi = half * inner, n = outer * hi;
  if (n == 0) {
    return;
  }
  k_glu<<<dk_blocks(n, 256), 256, 0, g_stream>>>(x, y, outer, hi, n);
  DK_LAUNCH_CHECK();
}

__global__ void k_channel_bias(float *y, const float *__restrict__ bias, int64_t C,
                               int64_t inner, int64_t n) {
  DK_GRID_STRIDE(i, n) { y[i] += bias[(i / inner) % C]; }
}

extern "C" void dk_channel_bias(float *y, const float *bias, int64_t outer, int64_t C,
                                int64_t inner) {
  int64_t n = outer * C * inner;
  if (n == 0) {
    return;
  }
  k_channel_bias<<<dk_blocks(n, 256), 256, 0, g_stream>>>(y, bias, C, inner, n);
  DK_LAUNCH_CHECK();
}

__global__ void k_scale_add(float *y, const float *__restrict__ x,
                            const float *__restrict__ scale, int64_t C, int64_t inner,
                            int64_t n) {
  DK_GRID_STRIDE(i, n) { y[i] += x[i] * scale[(i / inner) % C]; }
}

extern "C" void dk_scale_add(float *y, const float *x, const float *scale, int64_t outer,
                             int64_t C, int64_t inner) {
  int64_t n = outer * C * inner;
  if (n == 0) {
    return;
  }
  k_scale_add<<<dk_blocks(n, 256), 256, 0, g_stream>>>(y, x, scale, C, inner, n);
  DK_LAUNCH_CHECK();
}

__global__ void k_reflect_pad(const float *__restrict__ x, float *__restrict__ y, int64_t L,
                              int64_t left, int64_t Lo, int64_t n) {
  DK_GRID_STRIDE(i, n) {
    int64_t r = i / Lo, j = i % Lo - left;
    if (j < 0) {
      j = -j;
    } else if (j >= L) {
      j = 2 * L - 2 - j;
    }
    y[i] = x[r * L + j];
  }
}

extern "C" void dk_reflect_pad(const float *x, float *y, int64_t rows, int64_t L, int64_t left,
                               int64_t right) {
  int64_t Lo = L + left + right, n = rows * Lo;
  if (n == 0) {
    return;
  }
  k_reflect_pad<<<dk_blocks(n, 256), 256, 0, g_stream>>>(x, y, L, left, Lo, n);
  DK_LAUNCH_CHECK();
}

/* ------------------------------------------------------ GEMM operands */

__global__ void k_to_half_rows(const float *__restrict__ x, __half *__restrict__ y, int64_t C,
                               int64_t Cp, int64_t n) {
  DK_GRID_STRIDE(i, n) {
    int64_t r = i / Cp, c = i % Cp;
    y[i] = dk_to_h(c < C ? x[r * C + c] : 0.0f);
  }
}

extern "C" void dk_to_half_rows(const float *x, dc_half *y, int64_t rows, int64_t C,
                                int64_t Cp) {
  int64_t n = rows * Cp;
  if (n == 0) {
    return;
  }
  k_to_half_rows<<<dk_blocks(n, 256), 256, 0, g_stream>>>(x, (__half *)y, C, Cp, n);
  DK_LAUNCH_CHECK();
}

#define DK_TILE 32
#define DK_TROWS 8

/* x [batch][C][P] -> y [batch][P][Cp] through a shared tile (coalesced both ways) */
__global__ void k_transpose_to_half(const float *__restrict__ x, __half *__restrict__ y,
                                    int64_t C, int64_t P, int64_t Cp) {
  __shared__ float tile[DK_TILE][DK_TILE + 1];
  int64_t b = blockIdx.z;
  int64_t p0 = (int64_t)blockIdx.x * DK_TILE, c0 = (int64_t)blockIdx.y * DK_TILE;
  const float *xb = x + b * C * P;
  for (int j = threadIdx.y; j < DK_TILE; j += DK_TROWS) {
    int64_t c = c0 + j, p = p0 + threadIdx.x;
    tile[j][threadIdx.x] = (c < C && p < P) ? xb[c * P + p] : 0.0f;
  }
  __syncthreads();
  __half *yb = y + b * P * Cp;
  for (int j = threadIdx.y; j < DK_TILE; j += DK_TROWS) {
    int64_t p = p0 + j, c = c0 + threadIdx.x;
    if (p < P && c < Cp) {
      yb[p * Cp + c] = dk_to_h(tile[threadIdx.x][j]);
    }
  }
}

extern "C" void dk_transpose_to_half(const float *x, dc_half *y, int64_t batch, int64_t C,
                                     int64_t P, int64_t Cp) {
  if (batch == 0 || P == 0) {
    return;
  }
  dim3 grid((unsigned)((P + DK_TILE - 1) / DK_TILE), (unsigned)((Cp + DK_TILE - 1) / DK_TILE),
            (unsigned)batch);
  if (batch > 65535 || grid.y > 65535) {
    dc_die("transpose_to_half: batch %lld / channels %lld too large", (long long)batch,
           (long long)Cp);
  }
  k_transpose_to_half<<<grid, dim3(DK_TILE, DK_TROWS), 0, g_stream>>>(x, (__half *)y, C, P,
                                                                        Cp);
  DK_LAUNCH_CHECK();
}

/* col [nb][np][Kp]: tile of 32 positions x 32 reduction indices per block; reads run
 * along positions (contiguous in x for stride 1), writes along k. */
__global__ void k_im2col(const float *__restrict__ x, __half *__restrict__ col, DkConvGeom g,
                         int64_t n0, int64_t p0, int64_t np) {
  __shared__ float tile[DK_TILE][DK_TILE + 1];
  int64_t bi = blockIdx.z;
  int64_t pt = (int64_t)blockIdx.x * DK_TILE, kt = (int64_t)blockIdx.y * DK_TILE;
  int64_t K = (int64_t)g.KH * g.KW * g.C;
  int64_t plane = (int64_t)g.H * g.W;
  const float *xb = x + (n0 + bi) * g.C * plane;
  for (int j = threadIdx.y; j < DK_TILE; j += DK_TROWS) {
    int64_t k = kt + j, pi = pt + threadIdx.x;
    float v = 0.0f;
    if (k < K && pi < np) {
      int64_t p = p0 + pi;
      int ho = (int)(p / g.Wo), wo = (int)(p % g.Wo);
      int c = (int)(k % g.C);
      int kk = (int)(k / g.C);
      int kh = kk / g.KW, kw = kk % g.KW;
      int h = ho * g.SH - g.PH + kh * g.DH;
      int w = wo * g.SW - g.PW + kw * g.DW;
      if (h >= 0 && h < g.H && w >= 0 && w < g.W) {
        v = xb[c * plane + (int64_t)h * g.W + w];
      }
    }
    tile[j][threadIdx.x] = v;
  }
  __syncthreads();
  __half *cb = col + bi * np * g.Kp;
  for (int j = threadIdx.y; j < DK_TILE; j += DK_TROWS) {
    int64_t pi = pt + j, k = kt + threadIdx.x;
    if (pi < np && k < g.Kp) {
      cb[pi * g.Kp + k] = dk_to_h(tile[threadIdx.x][j]);
    }
  }
}

extern "C" void dk_im2col_half(const float *x, dc_half *col, const DkConvGeom *g, int64_t n0,
                               int64_t nb, int64_t p0, int64_t np) {
  if (nb == 0 || np == 0) {
    return;
  }
  dim3 grid((unsigned)((np + DK_TILE - 1) / DK_TILE),
            (unsigned)((g->Kp + DK_TILE - 1) / DK_TILE), (unsigned)nb);
  if (nb > 65535 || grid.y > 65535) {
    dc_die("im2col: %lld items / Kp %lld too large", (long long)nb, (long long)g->Kp);
  }
  k_im2col<<<grid, dim3(DK_TILE, DK_TROWS), 0, g_stream>>>(x, (__half *)col, *g, n0, p0, np);
  DK_LAUNCH_CHECK();
}

__global__ void k_col2im_tr(const float *__restrict__ cols, float *__restrict__ y,
                            const float *__restrict__ bias, int O, DkConvGeom g, int64_t n) {
  int64_t P = (int64_t)g.H * g.W;
  int64_t M = (int64_t)O * g.KH * g.KW;
  DK_GRID_STRIDE(i, n) {
    int wo = (int)(i % g.Wo);
    int64_t t = i / g.Wo;
    int ho = (int)(t % g.Ho);
    t /= g.Ho;
    int o = (int)(t % O);
    int64_t bi = t / O;
    const float *cb = cols + bi * M * P;
    float acc = bias ? bias[o] : 0.0f;
    int hf = ho + g.PH, wf = wo + g.PW;
    for (int kh = 0; kh < g.KH; kh++) {
      int hs = hf - kh;
      if (hs < 0 || hs % g.SH) {
        continue;
      }
      int h = hs / g.SH;
      if (h >= g.H) {
        continue;
      }
      for (int kw = 0; kw < g.KW; kw++) {
        int ws = wf - kw;
        if (ws < 0 || ws % g.SW) {
          continue;
        }
        int w = ws / g.SW;
        if (w >= g.W) {
          continue;
        }
        acc += cb[((int64_t)(o * g.KH + kh) * g.KW + kw) * P + (int64_t)h * g.W + w];
      }
    }
    y[i] = acc;
  }
}

extern "C" void dk_col2im_tr(const float *cols, float *y, const float *bias, int64_t nb, int O,
                             const DkConvGeom *g) {
  int64_t n = nb * O * (int64_t)g->Ho * g->Wo;
  if (n == 0) {
    return;
  }
  k_col2im_tr<<<dk_blocks(n, 256), 256, 0, g_stream>>>(cols, y, bias, O, *g, n);
  DK_LAUNCH_CHECK();
}
