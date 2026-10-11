/* Backend kernels vs naive double-precision host references.
 * Sizes are deliberately odd so K padding, partial tiles and fallback paths run. */
#include "ops.h"

#include <math.h>

static int g_fail = 0;
static uint32_t g_rng = 12345u;

static float frand(void) {
  g_rng = g_rng * 1664525u + 1013904223u;
  return (float)((g_rng >> 8) & 0xffffff) / (float)0x800000 - 1.0f;
}

static float *rand_buf(size_t n) {
  float *p = (float *)malloc(n * sizeof(float));
  for (size_t i = 0; i < n; i++) {
    p[i] = frand();
  }
  return p;
}

static void check(const char *name, DcTensor got, const double *ref, int64_t n, double gate) {
  if (dc_numel(got) != n) {
    printf("%-26s numel %lld != %lld FAIL\n", name, (long long)dc_numel(got), (long long)n);
    g_fail = 1;
    return;
  }
  float *p = dc_to_host(got);
  double num = 0, den = 0, mx = 0;
  for (int64_t i = 0; i < n; i++) {
    double d = (double)p[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
    mx = fabs(d) > mx ? fabs(d) : mx;
  }
  free(p);
  double rel = den > 0 ? sqrt(num / den) : sqrt(num);
  int ok = rel <= gate && !isnan(rel);
  printf("%-26s rel_l2=%.3e max_abs=%.3e %s\n", name, rel, mx, ok ? "ok" : "FAIL");
  g_fail |= !ok;
}

static void test_conv1d(int N, int C, int L, int O, int K, int s, int p, int d, int bias) {
  float *x = rand_buf((size_t)N * C * L), *w = rand_buf((size_t)O * K * C), *b = rand_buf(O);
  int xs[3] = {N, C, L}, ws[3] = {O, K, C}, bs[1] = {O};
  DcTensor xt = dc_from_host(x, xs, 3);
  DcWeight W, B;
  dc_weight_from_host(&W, "t.conv.weight", w, ws, 3);
  dc_weight_from_host(&B, "t.conv.bias", b, bs, 1);
  DcTensor y = dc_conv1d(xt, &W, bias ? &B : NULL, s, p, d);
  int Lo = (L + 2 * p - d * (K - 1) - 1) / s + 1;
  double *r = (double *)calloc((size_t)N * O * Lo, sizeof(double));
  for (int n = 0; n < N; n++)
    for (int o = 0; o < O; o++)
      for (int t = 0; t < Lo; t++) {
        double acc = bias ? b[o] : 0.0;
        for (int k = 0; k < K; k++) {
          int it = t * s - p + k * d;
          if (it < 0 || it >= L) continue;
          for (int c = 0; c < C; c++) acc += (double)w[(o * K + k) * C + c] * x[(n * C + c) * L + it];
        }
        r[((size_t)n * O + o) * Lo + t] = acc;
      }
  char name[64];
  snprintf(name, sizeof(name), "conv1d %dx%dx%d k%d s%d", N, C, L, K, s);
  check(name, y, r, (int64_t)N * O * Lo, 3e-3);
  dc_free(xt);
  dc_free(y);
  dc_weight_release(&W);
  dc_weight_release(&B);
  free(x);
  free(w);
  free(b);
  free(r);
}

static void test_conv2d(int N, int C, int H, int Wd, int O, int KH, int KW, int sh, int sw,
                        int ph, int pw) {
  float *x = rand_buf((size_t)N * C * H * Wd), *w = rand_buf((size_t)O * KH * KW * C);
  float *b = rand_buf(O);
  int xs[4] = {N, C, H, Wd}, ws[4] = {O, KH, KW, C}, bs[1] = {O};
  DcTensor xt = dc_from_host(x, xs, 4);
  DcWeight W, B;
  dc_weight_from_host(&W, "t.conv.weight", w, ws, 4);
  dc_weight_from_host(&B, "t.conv.bias", b, bs, 1);
  DcTensor y = dc_conv2d(xt, &W, &B, sh, sw, ph, pw);
  int Ho = (H + 2 * ph - KH) / sh + 1, Wo = (Wd + 2 * pw - KW) / sw + 1;
  double *r = (double *)calloc((size_t)N * O * Ho * Wo, sizeof(double));
  for (int n = 0; n < N; n++)
    for (int o = 0; o < O; o++)
      for (int i = 0; i < Ho; i++)
        for (int j = 0; j < Wo; j++) {
          double acc = b[o];
          for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
              int ih = i * sh - ph + kh, iw = j * sw - pw + kw;
              if (ih < 0 || ih >= H || iw < 0 || iw >= Wd) continue;
              for (int c = 0; c < C; c++)
                acc += (double)w[((o * KH + kh) * KW + kw) * C + c] *
                       x[(((size_t)n * C + c) * H + ih) * Wd + iw];
            }
          r[(((size_t)n * O + o) * Ho + i) * Wo + j] = acc;
        }
  check("conv2d", y, r, (int64_t)N * O * Ho * Wo, 3e-3);
  dc_free(xt);
  dc_free(y);
  dc_weight_release(&W);
  dc_weight_release(&B);
  free(x);
  free(w);
  free(b);
  free(r);
}

static void test_conv_tr2d(int N, int C, int H, int Wd, int O, int KH, int KW, int sh, int sw) {
  float *x = rand_buf((size_t)N * C * H * Wd), *w = rand_buf((size_t)O * KH * KW * C);
  float *b = rand_buf(O);
  int xs[4] = {N, C, H, Wd}, ws[4] = {O, KH, KW, C}, bs[1] = {O};
  DcTensor xt = dc_from_host(x, xs, 4);
  DcWeight W, B;
  dc_weight_from_host(&W, "t.conv_tr.weight", w, ws, 4);
  dc_weight_from_host(&B, "t.conv_tr.bias", b, bs, 1);
  DcTensor y = dc_conv_tr2d(xt, &W, &B, sh, sw);
  int Ho = (H - 1) * sh + KH, Wo = (Wd - 1) * sw + KW;
  double *r = (double *)calloc((size_t)N * O * Ho * Wo, sizeof(double));
  for (int n = 0; n < N; n++)
    for (int o = 0; o < O; o++) {
      for (int i = 0; i < Ho * Wo; i++) r[((size_t)n * O + o) * Ho * Wo + i] = b[o];
      for (int h = 0; h < H; h++)
        for (int ww = 0; ww < Wd; ww++)
          for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
              double acc = 0;
              for (int c = 0; c < C; c++)
                acc += (double)w[((o * KH + kh) * KW + kw) * C + c] *
                       x[(((size_t)n * C + c) * H + h) * Wd + ww];
              r[(((size_t)n * O + o) * Ho + h * sh + kh) * Wo + ww * sw + kw] += acc;
            }
    }
  char name[64];
  snprintf(name, sizeof(name), "conv_tr2d k%dx%d s%dx%d", KH, KW, sh, sw);
  check(name, y, r, (int64_t)N * O * Ho * Wo, 3e-3);

  if (KH == 1 && H == 1 && sh == 1) {
    /* the same weights as a conv_tr1d */
    int xs3[3] = {N, C, Wd}, ws3[3] = {O, KW, C};
    DcTensor x3 = dc_reshape(xt, xs3, 3);
    DcWeight W1;
    dc_weight_from_host(&W1, "t.conv_tr.weight", w, ws3, 3);
    DcTensor y1 = dc_conv_tr1d(x3, &W1, &B, sw);
    check("conv_tr1d", y1, r, (int64_t)N * O * Ho * Wo, 3e-3);
    dc_free(x3);
    dc_free(y1);
    dc_weight_release(&W1);
  }
  dc_free(xt);
  dc_free(y);
  dc_weight_release(&W);
  dc_weight_release(&B);
  free(x);
  free(w);
  free(b);
  free(r);
}

static void test_linear(int R, int IN, int OUT) {
  float *x = rand_buf((size_t)R * IN), *w = rand_buf((size_t)OUT * IN), *b = rand_buf(OUT);
  int xs[2] = {R, IN}, ws[2] = {OUT, IN}, bs[1] = {OUT};
  DcTensor xt = dc_from_host(x, xs, 2);
  DcWeight W, B;
  dc_weight_from_host(&W, "t.weight", w, ws, 2);
  dc_weight_from_host(&B, "t.bias", b, bs, 1);
  DcTensor y = dc_linear(xt, &W, &B);
  double *r = (double *)malloc((size_t)R * OUT * sizeof(double));
  for (int i = 0; i < R; i++)
    for (int o = 0; o < OUT; o++) {
      double acc = b[o];
      for (int k = 0; k < IN; k++) acc += (double)x[(size_t)i * IN + k] * w[(size_t)o * IN + k];
      r[(size_t)i * OUT + o] = acc;
    }
  char name[64];
  snprintf(name, sizeof(name), "linear %dx%d->%d", R, IN, OUT);
  check(name, y, r, (int64_t)R * OUT, 3e-3);
  dc_free(xt);
  dc_free(y);
  dc_weight_release(&W);
  dc_weight_release(&B);
  free(x);
  free(w);
  free(b);
  free(r);
}

static void test_attention(int B, int Tq, int Tk, int D, int H) {
  float *q = rand_buf((size_t)B * Tq * D), *k = rand_buf((size_t)B * Tk * D);
  float *v = rand_buf((size_t)B * Tk * D);
  int qs[3] = {B, Tq, D}, ks[3] = {B, Tk, D};
  DcTensor qt = dc_from_host(q, qs, 3), kt = dc_from_host(k, ks, 3), vt = dc_from_host(v, ks, 3);
  DcTensor y = dc_attention(qt, kt, vt, H);
  int hd = D / H;
  double *r = (double *)calloc((size_t)B * Tq * D, sizeof(double));
  double *s = (double *)malloc((size_t)Tk * sizeof(double));
  for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++)
      for (int i = 0; i < Tq; i++) {
        double mx = -1e300, sum = 0;
        for (int j = 0; j < Tk; j++) {
          double acc = 0;
          for (int e = 0; e < hd; e++)
            acc += (double)q[((size_t)b * Tq + i) * D + h * hd + e] *
                   k[((size_t)b * Tk + j) * D + h * hd + e];
          s[j] = acc / sqrt((double)hd);
          mx = s[j] > mx ? s[j] : mx;
        }
        for (int j = 0; j < Tk; j++) {
          s[j] = exp(s[j] - mx);
          sum += s[j];
        }
        for (int e = 0; e < hd; e++) {
          double acc = 0;
          for (int j = 0; j < Tk; j++) acc += s[j] / sum * v[((size_t)b * Tk + j) * D + h * hd + e];
          r[((size_t)b * Tq + i) * D + h * hd + e] = acc;
        }
      }
  char name[64];
  snprintf(name, sizeof(name), "attention %d/%d h%d", Tq, Tk, H);
  check(name, y, r, (int64_t)B * Tq * D, 5e-3);
  dc_free(qt);
  dc_free(kt);
  dc_free(vt);
  dc_free(y);
  free(q);
  free(k);
  free(v);
  free(r);
  free(s);
}

static double gelu_ref(double x) { return 0.5 * x * (1.0 + erf(x / sqrt(2.0))); }

static void test_groupnorm(int B, int C, int L, int G, int act) {
  float *x = rand_buf((size_t)B * C * L), *w = rand_buf(C), *bb = rand_buf(C);
  for (size_t i = 0; i < (size_t)B * C * L; i++) x[i] = x[i] * 3.0f + 0.5f;
  int xs[3] = {B, C, L}, cs[1] = {C};
  DcTensor xt = dc_from_host(x, xs, 3), wt = dc_from_host(w, cs, 1), bt = dc_from_host(bb, cs, 1);
  DcTensor y = dc_groupnorm(xt, wt, bt, G, 1e-5f, act);
  int Cg = C / G;
  double *n = (double *)malloc((size_t)B * C * L * sizeof(double));
  for (int b = 0; b < B; b++)
    for (int g = 0; g < G; g++) {
      double mean = 0, var = 0;
      size_t base = ((size_t)b * C + (size_t)g * Cg) * L, cnt = (size_t)Cg * L;
      for (size_t i = 0; i < cnt; i++) mean += x[base + i];
      mean /= (double)cnt;
      for (size_t i = 0; i < cnt; i++) var += (x[base + i] - mean) * (x[base + i] - mean);
      var /= (double)cnt;
      for (int c = g * Cg; c < (g + 1) * Cg; c++)
        for (int t = 0; t < L; t++) {
          size_t i = ((size_t)b * C + c) * L + t;
          n[i] = (x[i] - mean) / sqrt(var + 1e-5) * w[c] + bb[c];
        }
    }
  int Co = act == 2 ? C / 2 : C;
  double *r = (double *)malloc((size_t)B * Co * L * sizeof(double));
  for (int b = 0; b < B; b++)
    for (int c = 0; c < Co; c++)
      for (int t = 0; t < L; t++) {
        double a = n[((size_t)b * C + c) * L + t];
        if (act == 1) a = gelu_ref(a);
        if (act == 2) a = a / (1.0 + exp(-n[((size_t)b * C + c + Co) * L + t]));
        r[((size_t)b * Co + c) * L + t] = a;
      }
  char name[64];
  snprintf(name, sizeof(name), "groupnorm g%d act%d", G, act);
  check(name, y, r, (int64_t)B * Co * L, 1e-4);
  dc_free(xt);
  dc_free(wt);
  dc_free(bt);
  dc_free(y);
  free(x);
  free(w);
  free(bb);
  free(n);
  free(r);
}

static void test_layernorm(int R, int C) {
  float *x = rand_buf((size_t)R * C), *w = rand_buf(C), *b = rand_buf(C);
  int xs[2] = {R, C}, cs[1] = {C};
  DcTensor xt = dc_from_host(x, xs, 2), wt = dc_from_host(w, cs, 1), bt = dc_from_host(b, cs, 1);
  DcTensor y = dc_layer_norm(xt, wt, bt, 1e-5f);
  double *r = (double *)malloc((size_t)R * C * sizeof(double));
  for (int i = 0; i < R; i++) {
    double mean = 0, var = 0;
    for (int c = 0; c < C; c++) mean += x[(size_t)i * C + c];
    mean /= C;
    for (int c = 0; c < C; c++) var += (x[(size_t)i * C + c] - mean) * (x[(size_t)i * C + c] - mean);
    var /= C;
    for (int c = 0; c < C; c++)
      r[(size_t)i * C + c] = (x[(size_t)i * C + c] - mean) / sqrt(var + 1e-5) * w[c] + b[c];
  }
  check("layernorm", y, r, (int64_t)R * C, 1e-5);
  dc_free(xt);
  dc_free(wt);
  dc_free(bt);
  dc_free(y);
  free(x);
  free(w);
  free(b);
  free(r);
}

static void test_tensor_ops(void) {
  int A = 3, Bd = 5, Cd = 7;
  float *x = rand_buf((size_t)A * Bd * Cd);
  int s[3] = {A, Bd, Cd};
  DcTensor t = dc_from_host(x, s, 3);
  int ax[3] = {2, 0, 1};
  DcTensor tt = dc_transpose(t, ax, 3);
  double *r = (double *)malloc((size_t)A * Bd * Cd * sizeof(double));
  for (int c = 0; c < Cd; c++)
    for (int a = 0; a < A; a++)
      for (int b = 0; b < Bd; b++) r[((size_t)c * A + a) * Bd + b] = x[((size_t)a * Bd + b) * Cd + c];
  check("transpose", tt, r, (int64_t)A * Bd * Cd, 0.0);

  DcTensor pr = dc_pad1d(t, 3, 4, 1);
  int Lp = Cd + 7;
  double *rp = (double *)malloc((size_t)A * Bd * Lp * sizeof(double));
  for (int row = 0; row < A * Bd; row++)
    for (int j = 0; j < Lp; j++) {
      int src = j - 3;
      if (src < 0) src = -src;
      if (src >= Cd) src = 2 * (Cd - 1) - src;
      rp[(size_t)row * Lp + j] = x[(size_t)row * Cd + src];
    }
  check("reflect pad", pr, rp, (int64_t)A * Bd * Lp, 0.0);

  DcTensor items[2] = {t, t};
  DcTensor cat = dc_concat(items, 2, 1);
  double *rc = (double *)malloc((size_t)A * 2 * Bd * Cd * sizeof(double));
  for (int a = 0; a < A; a++)
    for (int b = 0; b < 2 * Bd; b++)
      for (int c = 0; c < Cd; c++)
        rc[((size_t)a * 2 * Bd + b) * Cd + c] = x[((size_t)a * Bd + (b % Bd)) * Cd + c];
  check("concat", cat, rc, (int64_t)A * 2 * Bd * Cd, 0.0);

  int bs[3] = {1, Bd, 1};
  float *bv = rand_buf(Bd);
  DcTensor bt = dc_from_host(bv, bs, 3);
  DcTensor sum = dc_add(t, bt);
  for (int a = 0; a < A; a++)
    for (int b = 0; b < Bd; b++)
      for (int c = 0; c < Cd; c++)
        r[((size_t)a * Bd + b) * Cd + c] = (double)x[((size_t)a * Bd + b) * Cd + c] + bv[b];
  check("broadcast add", sum, r, (int64_t)A * Bd * Cd, 1e-7);

  dc_free(t);
  dc_free(tt);
  dc_free(pr);
  dc_free(cat);
  dc_free(bt);
  dc_free(sum);
  free(x);
  free(bv);
  free(r);
  free(rp);
  free(rc);
}

int main(void) {
  char dev[256];
  if (dk_init(0, dev, sizeof(dev)) != 0) {
    dc_die("no device: %s", dev);
  }
  printf("backend %s (%s)\n", dk_backend_name(), dev);
  test_tensor_ops();
  test_linear(37, 50, 29);
  test_linear(130, 384, 1536);
  test_conv1d(2, 13, 101, 17, 8, 4, 2, 1, 1);
  test_conv1d(1, 24, 64, 6, 3, 1, 4, 4, 1);
  test_conv1d(3, 48, 77, 96, 1, 1, 0, 1, 0);
  test_conv2d(2, 4, 64, 9, 24, 8, 1, 4, 1, 2, 0);
  test_conv2d(1, 9, 11, 13, 10, 3, 3, 1, 1, 1, 1);
  test_conv_tr2d(2, 24, 16, 9, 12, 8, 1, 4, 1);
  test_conv_tr2d(2, 19, 1, 23, 11, 1, 8, 1, 4);
  test_attention(2, 33, 47, 64, 8);
  test_attention(1, 300, 300, 384, 8);
  test_groupnorm(2, 12, 37, 1, 0);
  test_groupnorm(2, 12, 37, 1, 1);
  test_groupnorm(2, 12, 37, 1, 2);
  test_groupnorm(2, 16, 33, 4, 1);
  test_layernorm(45, 384);
  printf(g_fail ? "test_kernels: FAIL\n" : "test_kernels: OK\n");
  return g_fail;
}
