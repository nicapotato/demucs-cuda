/* Portable CPU implementation of the kernel layer. Reference for tests and a
 * development path on machines without CUDA. fp16 operands are rounded exactly
 * like the GPU path so numerics match. */
#include "backend.h"
#include "util.h"

#include <math.h>

#if defined(DC_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

#ifdef DC_HALF_FLOAT
/* DC_ROUND_WEIGHTS=1 / DC_ROUND_ACT=1 re-enable fp16 rounding per operand class */
static int round_flag(int weight) {
  static int cached[2] = {-1, -1};
  if (cached[weight] < 0) {
    const char *e = getenv(weight ? "DC_ROUND_WEIGHTS" : "DC_ROUND_ACT");
    cached[weight] = e && e[0] == '1';
  }
  return cached[weight];
}
dc_half dk_pack_half(float v, int weight) {
  return round_flag(weight) ? dc_f16_to_f32(dc_f32_to_f16(v)) : v;
}
static inline float from_half(dc_half h) { return h; }
#else
dc_half dk_pack_half(float v, int weight) {
  (void)weight;
  return dc_f32_to_f16(v);
}
static inline float from_half(dc_half h) { return dc_f16_to_f32(h); }
#endif
static inline dc_half to_half(float v) { return dk_pack_half(v, 0); }

int dk_init(int device, char *name, size_t name_cap) {
  (void)device;
  if (name && name_cap) {
    snprintf(name, name_cap, "cpu");
  }
  return 0;
}

const char *dk_backend_name(void) { return "cpu"; }
void dk_sync(void) {}
void *dk_raw_malloc(size_t bytes) { return malloc(bytes ? bytes : 1); }
void dk_raw_free(void *p) { free(p); }
void dk_memcpy_h2d(void *dst, const void *src, size_t bytes) { memcpy(dst, src, bytes); }
void dk_memcpy_d2h(void *dst, const void *src, size_t bytes) { memcpy(dst, src, bytes); }
void dk_memcpy_d2d(void *dst, const void *src, size_t bytes) { memmove(dst, src, bytes); }
void dk_memset0(void *dst, size_t bytes) { memset(dst, 0, bytes); }

int dk_mem_info(size_t *free_bytes, size_t *total_bytes) {
  *free_bytes = (size_t)16 << 30;
  *total_bytes = (size_t)16 << 30;
  return 0;
}

void dk_range_push(const char *name) { (void)name; }

void dk_range_pop(void) {}

static int64_t total_elems(const DkStrided *p) {
  int64_t n = 1;
  for (int i = 0; i < p->ndim; i++) {
    n *= p->shape[i];
  }
  return n;
}

void dk_strided_copy(const DkStrided *p, const float *src, float *dst, int accumulate) {
  int64_t n = total_elems(p);
  int64_t idx[DK_MAX_DIM] = {0};
  int64_t ao = 0, bo = 0;
  int last = p->ndim - 1;
  for (int64_t e = 0; e < n;) {
    int64_t len = p->shape[last];
    int64_t as = p->a_strides[last], bs = p->b_strides[last];
    if (accumulate) {
      for (int64_t i = 0; i < len; i++) {
        dst[bo + i * bs] += src[ao + i * as];
      }
    } else {
      for (int64_t i = 0; i < len; i++) {
        dst[bo + i * bs] = src[ao + i * as];
      }
    }
    e += len;
    for (int d = last - 1; d >= 0; d--) {
      idx[d]++;
      ao += p->a_strides[d];
      bo += p->b_strides[d];
      if (idx[d] < p->shape[d]) {
        break;
      }
      ao -= p->a_strides[d] * p->shape[d];
      bo -= p->b_strides[d] * p->shape[d];
      idx[d] = 0;
    }
  }
}

static inline float binop(DkBinOp op, float a, float b) {
  switch (op) {
  case DK_ADD:
    return a + b;
  case DK_SUB:
    return a - b;
  case DK_MUL:
    return a * b;
  case DK_DIV:
    return a / b;
  default:
    return a > b ? a : b;
  }
}

void dk_binary(DkBinOp op, const DkStrided *p, const float *a, const float *b, float *y) {
  int64_t n = total_elems(p);
  int64_t idx[DK_MAX_DIM] = {0};
  int64_t ao = 0, bo = 0;
  int last = p->ndim - 1;
  for (int64_t e = 0; e < n;) {
    int64_t len = p->shape[last];
    int64_t as = p->a_strides[last], bs = p->b_strides[last];
    for (int64_t i = 0; i < len; i++) {
      y[e + i] = binop(op, a[ao + i * as], b[bo + i * bs]);
    }
    e += len;
    for (int d = last - 1; d >= 0; d--) {
      idx[d]++;
      ao += p->a_strides[d];
      bo += p->b_strides[d];
      if (idx[d] < p->shape[d]) {
        break;
      }
      ao -= p->a_strides[d] * p->shape[d];
      bo -= p->b_strides[d] * p->shape[d];
      idx[d] = 0;
    }
  }
}

static inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void dk_unary(DkUnOp op, const float *x, float *y, int64_t n, float alpha, float beta) {
  for (int64_t i = 0; i < n; i++) {
    float v = x[i];
    switch (op) {
    case DK_UN_AFFINE:
      y[i] = v * alpha + beta;
      break;
    case DK_UN_GELU:
      y[i] = gelu(v);
      break;
    case DK_UN_RELU:
      y[i] = v > 0.0f ? v : 0.0f;
      break;
    default:
      y[i] = sigmoid(v);
      break;
    }
  }
}

void dk_fill(float *y, int64_t n, float v) {
  for (int64_t i = 0; i < n; i++) {
    y[i] = v;
  }
}

void dk_glu(const float *x, float *y, int64_t outer, int64_t half, int64_t inner) {
  for (int64_t o = 0; o < outer; o++) {
    const float *a = x + o * 2 * half * inner;
    const float *b = a + half * inner;
    float *out = y + o * half * inner;
    for (int64_t i = 0; i < half * inner; i++) {
      out[i] = a[i] * sigmoid(b[i]);
    }
  }
}

void dk_channel_bias(float *y, const float *bias, int64_t outer, int64_t C, int64_t inner) {
  for (int64_t o = 0; o < outer; o++) {
    for (int64_t c = 0; c < C; c++) {
      float bv = bias[c];
      float *row = y + (o * C + c) * inner;
      for (int64_t i = 0; i < inner; i++) {
        row[i] += bv;
      }
    }
  }
}

void dk_scale_add(float *y, const float *x, const float *scale, int64_t outer, int64_t C,
                  int64_t inner) {
  for (int64_t o = 0; o < outer; o++) {
    for (int64_t c = 0; c < C; c++) {
      float s = scale[c];
      int64_t base = (o * C + c) * inner;
      for (int64_t i = 0; i < inner; i++) {
        y[base + i] += x[base + i] * s;
      }
    }
  }
}

void dk_reflect_pad(const float *x, float *y, int64_t rows, int64_t L, int64_t left,
                    int64_t right) {
  int64_t Lo = L + left + right;
  for (int64_t r = 0; r < rows; r++) {
    const float *src = x + r * L;
    float *dst = y + r * Lo;
    for (int64_t i = 0; i < Lo; i++) {
      int64_t j = i - left;
      if (j < 0) {
        j = -j;
      } else if (j >= L) {
        j = 2 * L - 2 - j;
      }
      dst[i] = src[j];
    }
  }
}

void dk_row_stats(const float *x, int64_t rows, int64_t L, float *mean, float *var) {
  for (int64_t r = 0; r < rows; r++) {
    const float *row = x + r * L;
    double s = 0.0;
    for (int64_t i = 0; i < L; i++) {
      s += row[i];
    }
    double m = s / (double)L;
    double ss = 0.0;
    for (int64_t i = 0; i < L; i++) {
      double d = row[i] - m;
      ss += d * d;
    }
    mean[r] = (float)m;
    var[r] = (float)(ss / (double)L);
  }
}

void dk_groupnorm_apply(const float *x, float *y, const float *mean, const float *var,
                        const float *w, const float *b, int64_t B, int64_t C, int64_t S,
                        int64_t G, float eps, int act) {
  int64_t cpg = C / G;
  if (act == 2) {
    int64_t H = C / 2;
    for (int64_t bi = 0; bi < B; bi++) {
      for (int64_t c = 0; c < H; c++) {
        int64_t ga = c / cpg, gb = (c + H) / cpg;
        float ra = 1.0f / sqrtf(var[bi * G + ga] + eps);
        float rb = 1.0f / sqrtf(var[bi * G + gb] + eps);
        float ma = mean[bi * G + ga], mb = mean[bi * G + gb];
        const float *xa = x + (bi * C + c) * S;
        const float *xb = x + (bi * C + c + H) * S;
        float *out = y + (bi * H + c) * S;
        for (int64_t s = 0; s < S; s++) {
          float na = (xa[s] - ma) * ra * w[c] + b[c];
          float nb = (xb[s] - mb) * rb * w[c + H] + b[c + H];
          out[s] = na * sigmoid(nb);
        }
      }
    }
    return;
  }
  for (int64_t bi = 0; bi < B; bi++) {
    for (int64_t c = 0; c < C; c++) {
      int64_t g = c / cpg;
      float rs = 1.0f / sqrtf(var[bi * G + g] + eps);
      float m = mean[bi * G + g];
      const float *xr = x + (bi * C + c) * S;
      float *out = y + (bi * C + c) * S;
      for (int64_t s = 0; s < S; s++) {
        float n = (xr[s] - m) * rs * w[c] + b[c];
        out[s] = act == 1 ? gelu(n) : n;
      }
    }
  }
}

void dk_norm_apply_last(const float *x, float *y, const float *mean, const float *var,
                        const float *w, const float *b, int64_t rows, int64_t L, int64_t C,
                        float eps) {
  for (int64_t r = 0; r < rows; r++) {
    float rs = 1.0f / sqrtf(var[r] + eps);
    float m = mean[r];
    for (int64_t j = 0; j < L; j++) {
      int64_t c = j % C;
      float n = (x[r * L + j] - m) * rs;
      if (w) {
        n *= w[c];
      }
      if (b) {
        n += b[c];
      }
      y[r * L + j] = n;
    }
  }
}

void dk_affine_rows(const float *x, float *y, const float *mean, const float *var,
                    int64_t rows, int64_t L, float eps, int denorm) {
  for (int64_t r = 0; r < rows; r++) {
    float sd = sqrtf(var[r]) + eps;
    float m = mean[r];
    for (int64_t j = 0; j < L; j++) {
      float v = x[r * L + j];
      y[r * L + j] = denorm ? v * sd + m : (v - m) / sd;
    }
  }
}

void dk_layernorm(const float *x, float *y, const float *w, const float *b, int64_t rows,
                  int64_t C, float eps) {
  for (int64_t r = 0; r < rows; r++) {
    float m, v;
    dk_row_stats(x + r * C, 1, C, &m, &v);
    dk_norm_apply_last(x + r * C, y + r * C, &m, &v, w, b, 1, C, C, eps);
  }
}

void dk_to_half_rows(const float *x, dc_half *y, int64_t rows, int64_t C, int64_t Cp) {
  for (int64_t r = 0; r < rows; r++) {
    for (int64_t c = 0; c < Cp; c++) {
      y[r * Cp + c] = c < C ? to_half(x[r * C + c]) : 0;
    }
  }
}

void dk_transpose_to_half(const float *x, dc_half *y, int64_t batch, int64_t C, int64_t P,
                          int64_t Cp) {
  for (int64_t bi = 0; bi < batch; bi++) {
    const float *xb = x + bi * C * P;
    dc_half *yb = y + bi * P * Cp;
    for (int64_t p = 0; p < P; p++) {
      for (int64_t c = 0; c < Cp; c++) {
        yb[p * Cp + c] = c < C ? to_half(xb[c * P + p]) : 0;
      }
    }
  }
}

void dk_im2col_half(const float *x, dc_half *col, const DkConvGeom *g, int64_t n0, int64_t nb,
                    int64_t p0, int64_t np) {
  int64_t K = (int64_t)g->KH * g->KW * g->C;
  int64_t plane = (int64_t)g->H * g->W;
  for (int64_t bi = 0; bi < nb; bi++) {
    const float *xb = x + (n0 + bi) * g->C * plane;
    for (int64_t pi = 0; pi < np; pi++) {
      int64_t p = p0 + pi;
      int ho = (int)(p / g->Wo), wo = (int)(p % g->Wo);
      dc_half *dst = col + (bi * np + pi) * g->Kp;
      for (int64_t k = 0; k < g->Kp; k++) {
        float v = 0.0f;
        if (k < K) {
          int c = (int)(k % g->C);
          int kk = (int)(k / g->C);
          int kh = kk / g->KW, kw = kk % g->KW;
          int h = ho * g->SH - g->PH + kh * g->DH;
          int w = wo * g->SW - g->PW + kw * g->DW;
          if (h >= 0 && h < g->H && w >= 0 && w < g->W) {
            v = xb[c * plane + (int64_t)h * g->W + w];
          }
        }
        dst[k] = to_half(v);
      }
    }
  }
}

void dk_col2im_tr(const float *cols, float *y, const float *bias, int64_t nb, int O,
                  const DkConvGeom *g) {
  int64_t P = (int64_t)g->H * g->W;
  int64_t M = (int64_t)O * g->KH * g->KW;
  for (int64_t bi = 0; bi < nb; bi++) {
    const float *cb = cols + bi * M * P;
    for (int o = 0; o < O; o++) {
      for (int ho = 0; ho < g->Ho; ho++) {
        for (int wo = 0; wo < g->Wo; wo++) {
          float acc = bias ? bias[o] : 0.0f;
          int hf = ho + g->PH, wf = wo + g->PW;
          for (int kh = 0; kh < g->KH; kh++) {
            int hs = hf - kh;
            if (hs < 0 || hs % g->SH) {
              continue;
            }
            int h = hs / g->SH;
            if (h >= g->H) {
              continue;
            }
            for (int kw = 0; kw < g->KW; kw++) {
              int ws = wf - kw;
              if (ws < 0 || ws % g->SW) {
                continue;
              }
              int w = ws / g->SW;
              if (w >= g->W) {
                continue;
              }
              acc += cb[((int64_t)(o * g->KH + kh) * g->KW + kw) * P + (int64_t)h * g->W + w];
            }
          }
          y[((bi * O + o) * g->Ho + ho) * (int64_t)g->Wo + wo] = acc;
        }
      }
    }
  }
}

static float *half_rows_to_float(const dc_half *src, int64_t rows, int64_t K, int64_t ld) {
  float *out = (float *)malloc((size_t)(rows * K) * sizeof(float) + 1);
  if (!out) {
    dc_die("oom gemm staging");
  }
  for (int64_t r = 0; r < rows; r++) {
    for (int64_t k = 0; k < K; k++) {
      out[r * K + k] = from_half(src[r * ld + k]);
    }
  }
  return out;
}

void dk_gemm_half(int64_t M, int64_t N, int64_t K, float alpha, const dc_half *A, int64_t lda,
                  int64_t strideA, const dc_half *B, int64_t ldb, int64_t strideB, float beta,
                  float *C, int64_t ldc, int64_t strideC, int64_t batch) {
  float *Af = NULL;
  const dc_half *A_cached = NULL;
  for (int64_t bi = 0; bi < batch; bi++) {
    const dc_half *Ab = A + bi * strideA;
    if (!Af || Ab != A_cached) {
      free(Af);
      Af = half_rows_to_float(Ab, M, K, lda);
      A_cached = Ab;
    }
    float *Bf = half_rows_to_float(B + bi * strideB, N, K, ldb);
    float *Cb = C + bi * strideC;
#if defined(DC_USE_ACCELERATE)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)M, (int)N, (int)K, alpha, Af,
                (int)K, Bf, (int)K, beta, Cb, (int)ldc);
#else
    for (int64_t m = 0; m < M; m++) {
      const float *ar = Af + m * K;
      for (int64_t n = 0; n < N; n++) {
        const float *br = Bf + n * K;
        float acc = 0.0f;
        for (int64_t k = 0; k < K; k++) {
          acc += ar[k] * br[k];
        }
        float prev = beta != 0.0f ? beta * Cb[m * ldc + n] : 0.0f;
        Cb[m * ldc + n] = alpha * acc + prev;
      }
    }
#endif
    free(Bf);
  }
  free(Af);
}

void dk_softmax_to_half(const float *s, int64_t lds, dc_half *p, int64_t ldp, int64_t rows,
                        int64_t L, float scale) {
  for (int64_t r = 0; r < rows; r++) {
    const float *row = s + r * lds;
    float mx = -INFINITY;
    for (int64_t j = 0; j < L; j++) {
      float v = row[j] * scale;
      mx = v > mx ? v : mx;
    }
    double sum = 0.0;
    for (int64_t j = 0; j < L; j++) {
      sum += expf(row[j] * scale - mx);
    }
    float inv = (float)(1.0 / sum);
    for (int64_t j = 0; j < ldp; j++) {
      p[r * ldp + j] = j < L ? to_half(expf(row[j] * scale - mx) * inv) : 0;
    }
  }
}

void dk_split_heads_half(const float *x, dc_half *y, int64_t B, int64_t T, int H, int hd,
                         int hdp, int64_t Tp, int transpose) {
  int64_t D = (int64_t)H * hd;
  for (int64_t b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      if (!transpose) {
        dc_half *dst = y + ((b * H + h) * T) * hdp;
        for (int64_t t = 0; t < T; t++) {
          for (int d = 0; d < hdp; d++) {
            dst[t * hdp + d] = d < hd ? to_half(x[(b * T + t) * D + h * hd + d]) : 0;
          }
        }
      } else {
        dc_half *dst = y + ((b * H + h) * hd) * Tp;
        for (int d = 0; d < hd; d++) {
          for (int64_t t = 0; t < Tp; t++) {
            dst[d * Tp + t] = t < T ? to_half(x[(b * T + t) * D + h * hd + d]) : 0;
          }
        }
      }
    }
  }
}

/* ------------------------------------------------------------------- FFT */

typedef struct {
  int n;
  float *tw_re; /* exp(-2 pi i k / n), k < n/2, computed in double */
  float *tw_im;
  int *rev;
} CpuFft;

static CpuFft g_fft;

static const CpuFft *fft_plan(int n) {
  if (g_fft.n == n) {
    return &g_fft;
  }
  if (n <= 1 || (n & (n - 1))) {
    dc_die("fft size %d must be a power of two", n);
  }
  free(g_fft.tw_re);
  free(g_fft.tw_im);
  free(g_fft.rev);
  g_fft.n = n;
  g_fft.tw_re = (float *)malloc((size_t)n / 2 * sizeof(float));
  g_fft.tw_im = (float *)malloc((size_t)n / 2 * sizeof(float));
  g_fft.rev = (int *)malloc((size_t)n * sizeof(int));
  for (int k = 0; k < n / 2; k++) {
    g_fft.tw_re[k] = (float)cos(-2.0 * DC_PI * k / n);
    g_fft.tw_im[k] = (float)sin(-2.0 * DC_PI * k / n);
  }
  int bits = 0;
  while ((1 << bits) < n) {
    bits++;
  }
  for (int i = 0; i < n; i++) {
    int r = 0;
    for (int b = 0; b < bits; b++) {
      r |= ((i >> b) & 1) << (bits - 1 - b);
    }
    g_fft.rev[i] = r;
  }
  return &g_fft;
}

/* in-place forward DFT of bit-reversed input */
static void fft_radix2(const CpuFft *pl, float *re, float *im) {
  int n = pl->n;
  for (int half = 1; half < n; half <<= 1) {
    int step = n / (2 * half);
    for (int j = 0; j < n / 2; j++) {
      int pos = j % half;
      int i0 = (j / half) * 2 * half + pos;
      int i1 = i0 + half;
      float wr = pl->tw_re[pos * step], wi = pl->tw_im[pos * step];
      float tr = wr * re[i1] - wi * im[i1];
      float ti = wr * im[i1] + wi * re[i1];
      re[i1] = re[i0] - tr;
      im[i1] = im[i0] - ti;
      re[i0] += tr;
      im[i0] += ti;
    }
  }
}

void dk_stft(const float *sig, float *out, int64_t R, int64_t L, int n_fft, int hop,
             int frame0, int nframes, int nbins, const float *window) {
  const CpuFft *pl = fft_plan(n_fft);
  float *re = (float *)malloc((size_t)n_fft * sizeof(float));
  float *im = (float *)malloc((size_t)n_fft * sizeof(float));
  for (int64_t r = 0; r < R; r++) {
    for (int f = 0; f < nframes; f++) {
      int64_t start = (int64_t)(frame0 + f) * hop;
      for (int i = 0; i < n_fft; i++) {
        int64_t t = start + i;
        float v = t < L ? sig[r * L + t] * window[i] : 0.0f;
        re[pl->rev[i]] = v;
        im[pl->rev[i]] = 0.0f;
      }
      fft_radix2(pl, re, im);
      float *ore = out + r * 2 * (int64_t)nbins * nframes;
      float *oim = ore + (int64_t)nbins * nframes;
      for (int k = 0; k < nbins; k++) {
        ore[(int64_t)k * nframes + f] = re[k];
        oim[(int64_t)k * nframes + f] = im[k];
      }
    }
  }
  free(re);
  free(im);
}

void dk_istft_frames(const float *spec, float *frames, int64_t R, int nbins, int nframes,
                     int pad_frames, int n_fft, const float *window) {
  const CpuFft *pl = fft_plan(n_fft);
  int NF = nframes + 2 * pad_frames;
  float *re = (float *)malloc((size_t)n_fft * sizeof(float));
  float *im = (float *)malloc((size_t)n_fft * sizeof(float));
  float inv_n = 1.0f / (float)n_fft;
  for (int64_t r = 0; r < R; r++) {
    const float *sre = spec + r * 2 * (int64_t)nbins * nframes;
    const float *sim = sre + (int64_t)nbins * nframes;
    for (int f = 0; f < NF; f++) {
      float *dst = frames + (r * NF + f) * (int64_t)n_fft;
      int fi = f - pad_frames;
      if (fi < 0 || fi >= nframes) {
        memset(dst, 0, (size_t)n_fft * sizeof(float));
        continue;
      }
      /* conj of the Hermitian-extended spectrum, then forward FFT */
      for (int k = 0; k < n_fft; k++) {
        int kk = k <= n_fft / 2 ? k : n_fft - k;
        float xr = 0.0f, xi = 0.0f;
        if (kk < nbins) {
          xr = sre[(int64_t)kk * nframes + fi];
          xi = sim[(int64_t)kk * nframes + fi];
        }
        if (k > n_fft / 2) {
          xi = -xi;
        }
        re[pl->rev[k]] = xr;
        im[pl->rev[k]] = -xi;
      }
      fft_radix2(pl, re, im);
      for (int i = 0; i < n_fft; i++) {
        dst[i] = re[i] * inv_n * window[i];
      }
    }
  }
  free(re);
  free(im);
}

void dk_ola(const float *frames, float *out, int64_t R, int nframes, int n_fft, int hop,
            const float *window, int64_t start, int64_t length) {
  int64_t out_len = (int64_t)hop * (nframes - 1) + n_fft;
  for (int64_t r = 0; r < R; r++) {
    for (int64_t t = 0; t < length; t++) {
      int64_t idx = start + t;
      float num = 0.0f, den = 0.0f;
      if (idx < out_len) {
        int64_t f_hi = idx / hop;
        if (f_hi > nframes - 1) {
          f_hi = nframes - 1;
        }
        int64_t f_lo = idx - n_fft + 1 <= 0 ? 0 : (idx - n_fft + hop) / hop;
        for (int64_t f = f_lo; f <= f_hi; f++) {
          int64_t i = idx - f * hop;
          if (i < 0 || i >= n_fft) {
            continue;
          }
          num += frames[(r * nframes + f) * n_fft + i];
          den += window[i] * window[i];
        }
      }
      out[r * length + t] = den > 0.0f ? num / den : 0.0f;
    }
  }
}
