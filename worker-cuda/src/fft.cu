/* STFT / iSTFT / overlap-add. One thread block per frame runs an in-place radix-2
 * FFT in shared memory (n_fft <= 4096 -> 32 KiB); twiddles come from a table
 * computed in double on the host. */
#include "cuda_common.cuh"

#include <math.h>
#include <stdlib.h>

#define FFT_THREADS 512
#define FFT_MAX_N 4096

static float2 *g_tw = nullptr;
static int g_tw_n = 0;
static int g_tw_bits = 0;

static const float2 *twiddles(int n) {
  if (n > FFT_MAX_N || n < 2 || (n & (n - 1))) {
    dc_die("fft size %d must be a power of two <= %d", n, FFT_MAX_N);
  }
  if (g_tw_n != n) {
    float2 *h = (float2 *)malloc((size_t)n / 2 * sizeof(float2));
    for (int k = 0; k < n / 2; k++) {
      h[k] = make_float2((float)cos(-2.0 * 3.14159265358979323846 * k / n),
                         (float)sin(-2.0 * 3.14159265358979323846 * k / n));
    }
    if (g_tw) {
      DK_CHECK(cudaStreamSynchronize(dk_stream()));
      DK_CHECK(cudaFree(g_tw));
    }
    DK_CHECK(cudaMalloc((void **)&g_tw, (size_t)n / 2 * sizeof(float2)));
    DK_CHECK(cudaMemcpy(g_tw, h, (size_t)n / 2 * sizeof(float2), cudaMemcpyHostToDevice));
    free(h);
    g_tw_n = n;
    g_tw_bits = 0;
    while ((1 << g_tw_bits) < n) {
      g_tw_bits++;
    }
  }
  return g_tw;
}

static __device__ __forceinline__ int bitrev(int i, int bits) {
  return (int)(__brev((unsigned)i) >> (32 - bits));
}

/* forward DFT of bit-reversed input held in buf */
static __device__ void fft_smem(float2 *buf, const float2 *__restrict__ tw, int n) {
  for (int half = 1; half < n; half <<= 1) {
    int step = n / (2 * half);
    for (int j = threadIdx.x; j < n / 2; j += blockDim.x) {
      int pos = j & (half - 1);
      int i0 = (j / half) * 2 * half + pos;
      int i1 = i0 + half;
      float2 w = __ldg(&tw[pos * step]);
      float2 a = buf[i0], b = buf[i1];
      float tr = w.x * b.x - w.y * b.y;
      float ti = w.x * b.y + w.y * b.x;
      buf[i1] = make_float2(a.x - tr, a.y - ti);
      buf[i0] = make_float2(a.x + tr, a.y + ti);
    }
    __syncthreads();
  }
}

__global__ void k_stft(const float *__restrict__ sig, float *__restrict__ out, int64_t L,
                       int n, int bits, int hop, int frame0, int nframes, int nbins,
                       const float *__restrict__ window, const float2 *__restrict__ tw) {
  extern __shared__ float2 buf[];
  int64_t r = blockIdx.x / nframes;
  int f = blockIdx.x % nframes;
  int64_t start = (int64_t)(frame0 + f) * hop;
  const float *s = sig + r * L;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int64_t t = start + i;
    buf[bitrev(i, bits)] = make_float2(t < L ? s[t] * window[i] : 0.0f, 0.0f);
  }
  __syncthreads();
  fft_smem(buf, tw, n);
  float *ore = out + r * 2 * (int64_t)nbins * nframes;
  float *oim = ore + (int64_t)nbins * nframes;
  for (int k = threadIdx.x; k < nbins; k += blockDim.x) {
    ore[(int64_t)k * nframes + f] = buf[k].x;
    oim[(int64_t)k * nframes + f] = buf[k].y;
  }
}

extern "C" void dk_stft(const float *sig, float *out, int64_t R, int64_t L, int n_fft, int hop,
                        int frame0, int nframes, int nbins, const float *window) {
  if (R == 0 || nframes == 0) {
    return;
  }
  const float2 *tw = twiddles(n_fft);
  int64_t blocks = R * nframes;
  if (blocks > 0x7fffffff) {
    dc_die("stft: too many frames");
  }
  k_stft<<<(unsigned)blocks, FFT_THREADS, (size_t)n_fft * sizeof(float2), dk_stream()>>>(
      sig, out, L, n_fft, g_tw_bits, hop, frame0, nframes, nbins, window, tw);
  DK_LAUNCH_CHECK();
}

/* irfft as conj(FFT(conj(X))) / n with the Hermitian extension of the kept bins */
__global__ void k_istft_frames(const float *__restrict__ spec, float *__restrict__ frames,
                               int nbins, int nframes, int pad_frames, int n, int bits,
                               const float *__restrict__ window,
                               const float2 *__restrict__ tw) {
  extern __shared__ float2 buf[];
  int NF = nframes + 2 * pad_frames;
  int64_t r = blockIdx.x / NF;
  int f = blockIdx.x % NF;
  float *dst = frames + (r * NF + f) * (int64_t)n;
  int fi = f - pad_frames;
  if (fi < 0 || fi >= nframes) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      dst[i] = 0.0f;
    }
    return;
  }
  const float *sre = spec + r * 2 * (int64_t)nbins * nframes;
  const float *sim = sre + (int64_t)nbins * nframes;
  for (int k = threadIdx.x; k < n; k += blockDim.x) {
    int kk = k <= n / 2 ? k : n - k;
    float xr = 0.0f, xi = 0.0f;
    if (kk < nbins) {
      xr = sre[(int64_t)kk * nframes + fi];
      xi = sim[(int64_t)kk * nframes + fi];
    }
    if (k > n / 2) {
      xi = -xi;
    }
    buf[bitrev(k, bits)] = make_float2(xr, -xi);
  }
  __syncthreads();
  fft_smem(buf, tw, n);
  float inv_n = 1.0f / (float)n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    dst[i] = buf[i].x * inv_n * window[i];
  }
}

extern "C" void dk_istft_frames(const float *spec, float *frames, int64_t R, int nbins,
                                int nframes, int pad_frames, int n_fft, const float *window) {
  int64_t blocks = R * (nframes + 2 * pad_frames);
  if (blocks == 0) {
    return;
  }
  if (blocks > 0x7fffffff) {
    dc_die("istft: too many frames");
  }
  const float2 *tw = twiddles(n_fft);
  k_istft_frames<<<(unsigned)blocks, FFT_THREADS, (size_t)n_fft * sizeof(float2),
                   dk_stream()>>>(spec, frames, nbins, nframes, pad_frames, n_fft, g_tw_bits,
                                  window, tw);
  DK_LAUNCH_CHECK();
}

__global__ void k_ola(const float *__restrict__ frames, float *__restrict__ out, int nframes,
                      int n, int hop, const float *__restrict__ window, int64_t start,
                      int64_t length, int64_t total) {
  int64_t out_len = (int64_t)hop * (nframes - 1) + n;
  DK_GRID_STRIDE(e, total) {
    int64_t r = e / length, t = e % length;
    int64_t idx = start + t;
    float num = 0.0f, den = 0.0f;
    if (idx < out_len) {
      int64_t f_hi = idx / hop;
      if (f_hi > nframes - 1) {
        f_hi = nframes - 1;
      }
      int64_t f_lo = idx - n + 1 <= 0 ? 0 : (idx - n + hop) / hop;
      for (int64_t f = f_lo; f <= f_hi; f++) {
        int64_t i = idx - f * hop;
        if (i < 0 || i >= n) {
          continue;
        }
        num += frames[(r * nframes + f) * n + i];
        den += window[i] * window[i];
      }
    }
    out[e] = den > 0.0f ? num / den : 0.0f;
  }
}

extern "C" void dk_ola(const float *frames, float *out, int64_t R, int nframes, int n_fft,
                       int hop, const float *window, int64_t start, int64_t length) {
  int64_t total = R * length;
  if (total == 0) {
    return;
  }
  k_ola<<<dk_blocks(total, 256), 256, 0, dk_stream()>>>(frames, out, nframes, n_fft, hop,
                                                         window, start, length, total);
  DK_LAUNCH_CHECK();
}
