#include "ops.h"

#include <math.h>

static size_t g_scratch_budget = (size_t)256 << 20;

void dc_ops_set_scratch_budget(size_t bytes) {
  if (bytes >= ((size_t)16 << 20)) {
    g_scratch_budget = bytes;
  }
}

static void check_gemm_weight(const DcWeight *wt, DcWeightKind kind, int ndim_min) {
  if (!wt || wt->kind != kind || wt->ndim < ndim_min) {
    dc_die("weight '%s' has unexpected kind/rank for this op", wt ? wt->name : "(null)");
  }
}

static const float *bias_ptr(const DcWeight *b) {
  if (!b) {
    return NULL;
  }
  if (b->kind != DC_W_F32) {
    dc_die("bias '%s' must be fp32", b->name);
  }
  return (const float *)b->dptr;
}

/* --------------------------------------------------------------- conv core */

static DcTensor conv_core(const float *x, int N, DkConvGeom g, const DcWeight *wt,
                          const DcWeight *bias, int O) {
  int64_t P = (int64_t)g.Ho * g.Wo;
  int oshape[4] = {N, O, g.Ho, g.Wo};
  DcTensor y = dc_new(oshape, 4);
  if (P == 0 || N == 0) {
    return y;
  }
  int64_t per_pos = g.Kp * (int64_t)sizeof(dc_half);
  int64_t max_pos = (int64_t)(g_scratch_budget / (size_t)per_pos);
  if (max_pos < 64) {
    max_pos = 64;
  }
  float *yd = dc_data(y);
  if (P <= max_pos) {
    int64_t nb_max = max_pos / P;
    if (nb_max > 65535) {
      nb_max = 65535;
    }
    int64_t nb_cap = nb_max < N ? nb_max : N;
    DcScratch col = dc_scratch((size_t)(nb_cap * P * per_pos));
    for (int64_t n0 = 0; n0 < N; n0 += nb_cap) {
      int64_t nb = N - n0 < nb_cap ? N - n0 : nb_cap;
      dk_im2col_half(x, (dc_half *)col.ptr, &g, n0, nb, 0, P);
      dk_gemm_half(O, P, g.Kp, 1.0f, (const dc_half *)wt->dptr, wt->kp, 0,
                   (const dc_half *)col.ptr, g.Kp, P * g.Kp, 0.0f, yd + n0 * O * P, P, O * P,
                   nb);
    }
    dc_scratch_free(&col);
  } else {
    DcScratch col = dc_scratch((size_t)(max_pos * per_pos));
    for (int64_t n = 0; n < N; n++) {
      for (int64_t p0 = 0; p0 < P; p0 += max_pos) {
        int64_t np = P - p0 < max_pos ? P - p0 : max_pos;
        dk_im2col_half(x, (dc_half *)col.ptr, &g, n, 1, p0, np);
        dk_gemm_half(O, np, g.Kp, 1.0f, (const dc_half *)wt->dptr, wt->kp, 0,
                     (const dc_half *)col.ptr, g.Kp, 0, 0.0f, yd + n * O * P + p0, P, 0, 1);
      }
    }
    dc_scratch_free(&col);
  }
  const float *bp = bias_ptr(bias);
  if (bp) {
    dk_channel_bias(yd, bp, N, O, P);
  }
  return y;
}

DcTensor dc_conv1d(DcTensor x, const DcWeight *weight, const DcWeight *bias, int stride,
                   int padding, int dilation) {
  check_gemm_weight(weight, DC_W_GEMM, 3);
  if (x.ndim != 3) {
    dc_die("conv1d expects [N, C, L]");
  }
  int O = weight->shape[0], K = weight->shape[1], C = weight->shape[2];
  if (x.shape[1] != C) {
    dc_print_shape("conv1d x", x);
    dc_die("conv1d '%s': channel mismatch %d vs %d", weight->name, x.shape[1], C);
  }
  DkConvGeom g;
  memset(&g, 0, sizeof(g));
  g.C = C;
  g.H = 1;
  g.W = x.shape[2];
  g.KH = 1;
  g.KW = K;
  g.SH = 1;
  g.SW = stride;
  g.PH = 0;
  g.PW = padding;
  g.DH = 1;
  g.DW = dilation;
  g.Ho = 1;
  g.Wo = (g.W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
  g.Kp = weight->kp;
  DcTensor y4 = conv_core(dc_data(x), x.shape[0], g, weight, bias, O);
  int s3[3] = {x.shape[0], O, g.Wo};
  DcTensor y = dc_reshape(y4, s3, 3);
  dc_free(y4);
  return y;
}

DcTensor dc_conv2d(DcTensor x, const DcWeight *weight, const DcWeight *bias, int sh, int sw,
                   int ph, int pw) {
  check_gemm_weight(weight, DC_W_GEMM, 4);
  if (x.ndim != 4) {
    dc_die("conv2d expects [N, C, H, W]");
  }
  int O = weight->shape[0], KH = weight->shape[1], KW = weight->shape[2], C = weight->shape[3];
  if (x.shape[1] != C) {
    dc_print_shape("conv2d x", x);
    dc_die("conv2d '%s': channel mismatch %d vs %d", weight->name, x.shape[1], C);
  }
  DkConvGeom g;
  memset(&g, 0, sizeof(g));
  g.C = C;
  g.H = x.shape[2];
  g.W = x.shape[3];
  g.KH = KH;
  g.KW = KW;
  g.SH = sh;
  g.SW = sw;
  g.PH = ph;
  g.PW = pw;
  g.DH = 1;
  g.DW = 1;
  g.Ho = (g.H + 2 * ph - (KH - 1) - 1) / sh + 1;
  g.Wo = (g.W + 2 * pw - (KW - 1) - 1) / sw + 1;
  g.Kp = weight->kp;
  return conv_core(dc_data(x), x.shape[0], g, weight, bias, O);
}

static DcTensor conv_tr_core(const float *x, int N, DkConvGeom g, const DcWeight *wt,
                             const DcWeight *bias, int O) {
  int64_t P = (int64_t)g.H * g.W;
  int64_t M = (int64_t)O * g.KH * g.KW;
  int oshape[4] = {N, O, g.Ho, g.Wo};
  DcTensor y = dc_new(oshape, 4);
  if (N == 0 || P == 0) {
    return y;
  }
  int64_t per_item = P * g.Kp * (int64_t)sizeof(dc_half) + M * P * (int64_t)sizeof(float);
  int64_t nb_cap = (int64_t)(g_scratch_budget / (size_t)per_item);
  if (nb_cap < 1) {
    nb_cap = 1;
  }
  if (nb_cap > N) {
    nb_cap = N;
  }
  if (nb_cap > 65535) {
    nb_cap = 65535;
  }
  DcScratch xt = dc_scratch((size_t)(nb_cap * P * g.Kp) * sizeof(dc_half));
  DcScratch cols = dc_scratch((size_t)(nb_cap * M * P) * sizeof(float));
  const float *bp = bias_ptr(bias);
  int64_t out_item = (int64_t)O * g.Ho * g.Wo;
  for (int64_t n0 = 0; n0 < N; n0 += nb_cap) {
    int64_t nb = N - n0 < nb_cap ? N - n0 : nb_cap;
    dk_transpose_to_half(x + n0 * g.C * P, (dc_half *)xt.ptr, nb, g.C, P, g.Kp);
    dk_gemm_half(M, P, g.Kp, 1.0f, (const dc_half *)wt->dptr, wt->kp, 0,
                 (const dc_half *)xt.ptr, g.Kp, P * g.Kp, 0.0f, (float *)cols.ptr, P, M * P, nb);
    dk_col2im_tr((const float *)cols.ptr, dc_data(y) + n0 * out_item, bp, nb, O, &g);
  }
  dc_scratch_free(&xt);
  dc_scratch_free(&cols);
  return y;
}

DcTensor dc_conv_tr1d(DcTensor x, const DcWeight *weight, const DcWeight *bias, int stride) {
  check_gemm_weight(weight, DC_W_GEMM_TR, 3);
  if (x.ndim != 3) {
    dc_die("conv_tr1d expects [N, C, L]");
  }
  int O = weight->shape[0], K = weight->shape[1], C = weight->shape[2];
  if (x.shape[1] != C) {
    dc_die("conv_tr1d '%s': channel mismatch %d vs %d", weight->name, x.shape[1], C);
  }
  DkConvGeom g;
  memset(&g, 0, sizeof(g));
  g.C = C;
  g.H = 1;
  g.W = x.shape[2];
  g.KH = 1;
  g.KW = K;
  g.SH = 1;
  g.SW = stride;
  g.DH = g.DW = 1;
  g.Ho = 1;
  g.Wo = (g.W - 1) * stride + K;
  g.Kp = weight->kp;
  DcTensor y4 = conv_tr_core(dc_data(x), x.shape[0], g, weight, bias, O);
  int s3[3] = {x.shape[0], O, g.Wo};
  DcTensor y = dc_reshape(y4, s3, 3);
  dc_free(y4);
  return y;
}

DcTensor dc_conv_tr2d(DcTensor x, const DcWeight *weight, const DcWeight *bias, int sh,
                      int sw) {
  check_gemm_weight(weight, DC_W_GEMM_TR, 4);
  if (x.ndim != 4) {
    dc_die("conv_tr2d expects [N, C, H, W]");
  }
  int O = weight->shape[0], KH = weight->shape[1], KW = weight->shape[2], C = weight->shape[3];
  if (x.shape[1] != C) {
    dc_die("conv_tr2d '%s': channel mismatch %d vs %d", weight->name, x.shape[1], C);
  }
  DkConvGeom g;
  memset(&g, 0, sizeof(g));
  g.C = C;
  g.H = x.shape[2];
  g.W = x.shape[3];
  g.KH = KH;
  g.KW = KW;
  g.SH = sh;
  g.SW = sw;
  g.DH = g.DW = 1;
  g.Ho = (g.H - 1) * sh + KH;
  g.Wo = (g.W - 1) * sw + KW;
  g.Kp = weight->kp;
  return conv_tr_core(dc_data(x), x.shape[0], g, weight, bias, O);
}

DcTensor dc_linear(DcTensor x, const DcWeight *weight, const DcWeight *bias) {
  check_gemm_weight(weight, DC_W_GEMM, 2);
  int out = weight->shape[0], in = weight->shape[1];
  if (x.shape[x.ndim - 1] != in) {
    dc_die("linear '%s': input dim %d vs %d", weight->name, x.shape[x.ndim - 1], in);
  }
  int64_t R = dc_numel(x) / in;
  int oshape[DC_MAX_DIM];
  for (int i = 0; i < x.ndim; i++) {
    oshape[i] = x.shape[i];
  }
  oshape[x.ndim - 1] = out;
  DcTensor y = dc_new(oshape, x.ndim);
  DcScratch xh = dc_scratch((size_t)(R * weight->kp) * sizeof(dc_half));
  dk_to_half_rows(dc_data(x), (dc_half *)xh.ptr, R, in, weight->kp);
  dk_gemm_half(R, out, weight->kp, 1.0f, (const dc_half *)xh.ptr, weight->kp, 0,
               (const dc_half *)weight->dptr, weight->kp, 0, 0.0f, dc_data(y), out, 0, 1);
  dc_scratch_free(&xh);
  const float *bp = bias_ptr(bias);
  if (bp) {
    dk_channel_bias(dc_data(y), bp, R, out, 1);
  }
  return y;
}

/* --------------------------------------------------------------- prefixed */

DcTensor dc_conv1d_pref(DcTensor x, const DcWeights *w, const char *prefix, int stride,
                        int padding, int dilation) {
  return dc_conv1d(x, dc_wget(w, "%s.conv.weight", prefix), dc_wget(w, "%s.conv.bias", prefix),
                   stride, padding, dilation);
}

DcTensor dc_conv2d_pref(DcTensor x, const DcWeights *w, const char *prefix, int sh, int sw,
                        int ph, int pw) {
  return dc_conv2d(x, dc_wget(w, "%s.conv.weight", prefix), dc_wget(w, "%s.conv.bias", prefix),
                   sh, sw, ph, pw);
}

DcTensor dc_convtr1d_pref(DcTensor x, const DcWeights *w, const char *prefix, int stride) {
  return dc_conv_tr1d(x, dc_wget(w, "%s.conv.weight", prefix),
                      dc_wget(w, "%s.conv.bias", prefix), stride);
}

DcTensor dc_convtr2d_pref(DcTensor x, const DcWeights *w, const char *prefix, int sh, int sw) {
  return dc_conv_tr2d(x, dc_wget(w, "%s.conv.weight", prefix),
                      dc_wget(w, "%s.conv.bias", prefix), sh, sw);
}

DcTensor dc_linear_pref(DcTensor x, const DcWeights *w, const char *prefix) {
  return dc_linear(x, dc_wget(w, "%s.weight", prefix), dc_wfind(w, "%s.bias", prefix));
}

DcTensor dc_ln_pref(DcTensor x, const DcWeights *w, const char *prefix) {
  DcTensor wt = dc_wt(w, "%s.weight", prefix);
  DcTensor bt = dc_wt(w, "%s.bias", prefix);
  DcTensor y = dc_layer_norm(x, wt, bt, 1e-5f);
  dc_free(wt);
  dc_free(bt);
  return y;
}

/* ------------------------------------------------------------------ norms */

DcTensor dc_layer_norm(DcTensor x, DcTensor weight, DcTensor bias, float eps) {
  int C = x.shape[x.ndim - 1];
  DcTensor y = dc_new(x.shape, x.ndim);
  dk_layernorm(dc_data(x), dc_data(y), dc_valid(weight) ? dc_data(weight) : NULL,
               dc_valid(bias) ? dc_data(bias) : NULL, dc_numel(x) / C, C, eps);
  return y;
}

DcTensor dc_groupnorm(DcTensor x, DcTensor weight, DcTensor bias, int groups, float eps,
                      int act) {
  int B = x.shape[0], C = x.shape[1];
  if (C % groups != 0) {
    dc_die("groupnorm: channels %d not divisible by groups %d", C, groups);
  }
  int64_t S = dc_numel(x) / ((int64_t)B * C);
  int sstat[1] = {B * groups};
  DcTensor mean = dc_new(sstat, 1);
  DcTensor var = dc_new(sstat, 1);
  dk_row_stats(dc_data(x), (int64_t)B * groups, (int64_t)(C / groups) * S, dc_data(mean),
               dc_data(var));
  int oshape[DC_MAX_DIM];
  for (int i = 0; i < x.ndim; i++) {
    oshape[i] = x.shape[i];
  }
  if (act == 2) {
    if (C % 2) {
      dc_die("groupnorm+glu: odd channels");
    }
    oshape[1] = C / 2;
  }
  DcTensor y = dc_new(oshape, x.ndim);
  dk_groupnorm_apply(dc_data(x), dc_data(y), dc_data(mean), dc_data(var), dc_data(weight),
                     dc_data(bias), B, C, S, groups, eps, act);
  dc_free(mean);
  dc_free(var);
  return y;
}

DcTensor dc_groupnorm_last(DcTensor x, DcTensor weight, DcTensor bias, float eps) {
  int B = x.shape[0];
  int C = x.shape[x.ndim - 1];
  int64_t L = dc_numel(x) / B;
  int sstat[1] = {B};
  DcTensor mean = dc_new(sstat, 1);
  DcTensor var = dc_new(sstat, 1);
  dk_row_stats(dc_data(x), B, L, dc_data(mean), dc_data(var));
  DcTensor y = dc_new(x.shape, x.ndim);
  dk_norm_apply_last(dc_data(x), dc_data(y), dc_data(mean), dc_data(var), dc_data(weight),
                     dc_data(bias), B, L, C, eps);
  dc_free(mean);
  dc_free(var);
  return y;
}

void dc_scale_add_inplace(DcTensor y, DcTensor x, DcTensor scale, int axis) {
  if (axis < 0) {
    axis += x.ndim;
  }
  int64_t outer = 1, inner = 1;
  for (int i = 0; i < axis; i++) {
    outer *= x.shape[i];
  }
  for (int i = axis + 1; i < x.ndim; i++) {
    inner *= x.shape[i];
  }
  if (dc_numel(y) != dc_numel(x) || dc_numel(scale) != x.shape[axis]) {
    dc_die("scale_add: shape mismatch");
  }
  dk_scale_add(dc_data(y), dc_data(x), dc_data(scale), outer, x.shape[axis], inner);
}

void dc_batch_stats(DcTensor x, DcTensor *mean, DcTensor *var) {
  int B = x.shape[0];
  int s[1] = {B};
  *mean = dc_new(s, 1);
  *var = dc_new(s, 1);
  dk_row_stats(dc_data(x), B, dc_numel(x) / B, dc_data(*mean), dc_data(*var));
}

DcTensor dc_batch_normalize(DcTensor x, DcTensor mean, DcTensor var, float eps, int denorm) {
  int B = x.shape[0];
  if (dc_numel(mean) != B) {
    dc_die("batch_normalize: stats for %lld rows, tensor has %d", (long long)dc_numel(mean), B);
  }
  DcTensor y = dc_new(x.shape, x.ndim);
  dk_affine_rows(dc_data(x), dc_data(y), dc_data(mean), dc_data(var), B, dc_numel(x) / B, eps,
                 denorm);
  return y;
}

/* -------------------------------------------------------------- attention */

DcTensor dc_attention(DcTensor q, DcTensor k, DcTensor v, int nheads) {
  int B = q.shape[0], Tq = q.shape[1], D = q.shape[2];
  int Tk = k.shape[1];
  if (D % nheads) {
    dc_die("attention dim %d not divisible by %d heads", D, nheads);
  }
  int hd = D / nheads;
  int hdp = (int)dc_round_up(hd, 8);
  int64_t Tkp = dc_round_up(Tk, 8);
  int64_t BH = (int64_t)B * nheads;
  DcScratch qh = dc_scratch((size_t)(BH * Tq * hdp) * sizeof(dc_half));
  DcScratch kh = dc_scratch((size_t)(BH * Tk * hdp) * sizeof(dc_half));
  DcScratch vt = dc_scratch((size_t)(BH * hd * Tkp) * sizeof(dc_half));
  dk_split_heads_half(dc_data(q), (dc_half *)qh.ptr, B, Tq, nheads, hd, hdp, 0, 0);
  dk_split_heads_half(dc_data(k), (dc_half *)kh.ptr, B, Tk, nheads, hd, hdp, 0, 0);
  dk_split_heads_half(dc_data(v), (dc_half *)vt.ptr, B, Tk, nheads, hd, hdp, Tkp, 1);

  int oshape[4] = {B, nheads, Tq, hd};
  DcTensor o = dc_new(oshape, 4);
  int64_t per_bh = (int64_t)Tq * Tk * (int64_t)sizeof(float) +
                   (int64_t)Tq * Tkp * (int64_t)sizeof(dc_half);
  int64_t nbh_cap = (int64_t)(g_scratch_budget / (size_t)per_bh);
  if (nbh_cap < 1) {
    nbh_cap = 1;
  }
  if (nbh_cap > BH) {
    nbh_cap = BH;
  }
  DcScratch sc = dc_scratch((size_t)(nbh_cap * Tq * Tk) * sizeof(float));
  DcScratch pr = dc_scratch((size_t)(nbh_cap * Tq * Tkp) * sizeof(dc_half));
  float scale = 1.0f / sqrtf((float)hd);
  for (int64_t b0 = 0; b0 < BH; b0 += nbh_cap) {
    int64_t nb = BH - b0 < nbh_cap ? BH - b0 : nbh_cap;
    dk_gemm_half(Tq, Tk, hdp, 1.0f, (const dc_half *)qh.ptr + b0 * Tq * hdp, hdp,
                 (int64_t)Tq * hdp, (const dc_half *)kh.ptr + b0 * Tk * hdp, hdp,
                 (int64_t)Tk * hdp, 0.0f, (float *)sc.ptr, Tk, (int64_t)Tq * Tk, nb);
    dk_softmax_to_half((const float *)sc.ptr, Tk, (dc_half *)pr.ptr, Tkp, nb * Tq, Tk, scale);
    dk_gemm_half(Tq, hd, Tkp, 1.0f, (const dc_half *)pr.ptr, Tkp, (int64_t)Tq * Tkp,
                 (const dc_half *)vt.ptr + b0 * hd * Tkp, Tkp, (int64_t)hd * Tkp, 0.0f,
                 dc_data(o) + b0 * Tq * hd, hd, (int64_t)Tq * hd, nb);
  }
  dc_scratch_free(&sc);
  dc_scratch_free(&pr);
  dc_scratch_free(&qh);
  dc_scratch_free(&kh);
  dc_scratch_free(&vt);
  int tax[4] = {0, 2, 1, 3};
  DcTensor ot = dc_transpose(o, tax, 4);
  dc_free(o);
  int fshape[3] = {B, Tq, D};
  DcTensor out = dc_reshape(ot, fshape, 3);
  dc_free(ot);
  return out;
}
