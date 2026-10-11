#include "transformer.h"

#include <math.h>

/* x + a * gamma (gamma optional) */
static DcTensor residual(DcTensor x, DcTensor a, const DcWeights *w, const char *gamma) {
  if (!dc_w_has(w, "%s.scale", gamma)) {
    return dc_add(x, a);
  }
  DcTensor scale = dc_wt(w, "%s.scale", gamma);
  DcTensor y = dc_ref(x);
  dc_make_unique(&y);
  dc_scale_add_inplace(y, a, scale, -1);
  dc_free(scale);
  return y;
}

static DcTensor mha_pref(DcTensor q, DcTensor kv, const DcWeights *w, const char *prefix,
                         int nheads) {
  char p[256];
  snprintf(p, sizeof(p), "%s.query_proj", prefix);
  DcTensor qq = dc_linear_pref(q, w, p);
  snprintf(p, sizeof(p), "%s.key_proj", prefix);
  DcTensor kk = dc_linear_pref(kv, w, p);
  snprintf(p, sizeof(p), "%s.value_proj", prefix);
  DcTensor vv = dc_linear_pref(kv, w, p);
  DcTensor attn = dc_attention(qq, kk, vv, nheads);
  dc_free(qq);
  dc_free(kk);
  dc_free(vv);
  snprintf(p, sizeof(p), "%s.out_proj", prefix);
  DcTensor out = dc_linear_pref(attn, w, p);
  dc_free(attn);
  return out;
}

static DcTensor feed_forward(DcTensor x, const DcWeights *w, const char *prefix,
                             const char *norm, const DcConfig *cfg) {
  char p[192];
  snprintf(p, sizeof(p), "%s.%s", prefix, norm);
  DcTensor xn = dc_ln_pref(x, w, p);
  snprintf(p, sizeof(p), "%s.linear1", prefix);
  DcTensor h = dc_linear_pref(xn, w, p);
  dc_free(xn);
  DcTensor ha = cfg->t_gelu ? dc_gelu(h) : dc_relu(h);
  dc_free(h);
  snprintf(p, sizeof(p), "%s.linear2", prefix);
  DcTensor h2 = dc_linear_pref(ha, w, p);
  dc_free(ha);
  snprintf(p, sizeof(p), "%s.gamma_2", prefix);
  DcTensor y = residual(x, h2, w, p);
  dc_free(h2);
  return y;
}

static DcTensor maybe_norm_out(DcTensor x, const DcWeights *w, const char *prefix) {
  if (!dc_w_has(w, "%s.norm_out.gn.weight", prefix)) {
    return dc_ref(x);
  }
  DcTensor nw = dc_wt(w, "%s.norm_out.gn.weight", prefix);
  DcTensor nb = dc_wt(w, "%s.norm_out.gn.bias", prefix);
  DcTensor y = dc_groupnorm_last(x, nw, nb, 1e-5f);
  dc_free(nw);
  dc_free(nb);
  return y;
}

static DcTensor self_attn_layer(DcTensor x, const DcWeights *w, const char *prefix,
                                const DcConfig *cfg) {
  char p[192];
  snprintf(p, sizeof(p), "%s.norm1", prefix);
  DcTensor xn = dc_ln_pref(x, w, p);
  snprintf(p, sizeof(p), "%s.attn", prefix);
  DcTensor a = mha_pref(xn, xn, w, p, cfg->t_heads);
  dc_free(xn);
  snprintf(p, sizeof(p), "%s.gamma_1", prefix);
  DcTensor x1 = residual(x, a, w, p);
  dc_free(a);
  DcTensor y = feed_forward(x1, w, prefix, "norm2", cfg);
  dc_free(x1);
  DcTensor yo = maybe_norm_out(y, w, prefix);
  dc_free(y);
  return yo;
}

static DcTensor cross_attn_layer(DcTensor q, DcTensor k, const DcWeights *w, const char *prefix,
                                 const DcConfig *cfg) {
  char p[192];
  snprintf(p, sizeof(p), "%s.norm1", prefix);
  DcTensor qn = dc_ln_pref(q, w, p);
  snprintf(p, sizeof(p), "%s.norm2", prefix);
  DcTensor kn = dc_ln_pref(k, w, p);
  snprintf(p, sizeof(p), "%s.cross_attn", prefix);
  DcTensor a = mha_pref(qn, kn, w, p, cfg->t_heads);
  dc_free(qn);
  dc_free(kn);
  snprintf(p, sizeof(p), "%s.gamma_1", prefix);
  DcTensor x = residual(q, a, w, p);
  dc_free(a);
  DcTensor y = feed_forward(x, w, prefix, "norm3", cfg);
  dc_free(x);
  DcTensor yo = maybe_norm_out(y, w, prefix);
  dc_free(y);
  return yo;
}

/* 1D sinusoidal embedding [1, T, C]: cat(cos(phase), sin(phase)) */
static DcTensor sin_embedding(int length, int dim, float max_period, float weight) {
  if (dim % 2 != 0) {
    dc_die("sin embed dim must be even");
  }
  int half = dim / 2;
  float *pe = (float *)malloc((size_t)length * (size_t)dim * sizeof(float));
  for (int t = 0; t < length; t++) {
    for (int j = 0; j < half; j++) {
      float den = powf(max_period, (float)j / (float)(half - 1));
      float phase = (float)t / den;
      pe[(size_t)t * dim + j] = cosf(phase) * weight;
      pe[(size_t)t * dim + half + j] = sinf(phase) * weight;
    }
  }
  int shape[3] = {1, length, dim};
  DcTensor r = dc_from_host(pe, shape, 3);
  free(pe);
  return r;
}

/* 2D sinusoidal embedding laid out per token (t * Fr + f): [1, T1*Fr, C] */
static DcTensor sin_embedding_2d_tokens(int d_model, int height, int width, float max_period,
                                        float weight) {
  if (d_model % 4 != 0) {
    dc_die("2d sin embed dim must be divisible by 4");
  }
  int half = d_model / 2;
  int nterm = half / 2;
  float *div = (float *)malloc((size_t)nterm * sizeof(float));
  float sc = -logf(max_period) / (float)half;
  for (int j = 0; j < nterm; j++) {
    div[j] = expf((float)(2 * j) * sc);
  }
  float *pe = (float *)malloc((size_t)width * height * d_model * sizeof(float));
  for (int t = 0; t < width; t++) {
    for (int f = 0; f < height; f++) {
      float *row = pe + ((size_t)t * height + f) * d_model;
      for (int c = 0; c < d_model; c++) {
        int cc = c < half ? c : c - half;
        float pos = c < half ? (float)t : (float)f;
        float ang = pos * div[cc / 2];
        row[c] = ((cc % 2) == 0 ? sinf(ang) : cosf(ang)) * weight;
      }
    }
  }
  int shape[3] = {1, width * height, d_model};
  DcTensor r = dc_from_host(pe, shape, 3);
  free(pe);
  free(div);
  return r;
}

void dc_crosstransformer(DcTensor *x_io, DcTensor *xt_io, const DcWeights *w,
                         const DcConfig *cfg) {
  DcTensor x = *x_io;
  DcTensor xt = *xt_io;
  int B = x.shape[0], C = x.shape[1], Fr = x.shape[2], T1 = x.shape[3];
  int tax[4] = {0, 3, 2, 1};
  DcTensor xtf = dc_transpose(x, tax, 4);
  int fshape[3] = {B, T1 * Fr, C};
  DcTensor xf = dc_reshape(xtf, fshape, 3);
  dc_free(xtf);
  DcTensor xn = dc_ln_pref(xf, w, "crosstransformer.norm_in");
  dc_free(xf);
  DcTensor pe = sin_embedding_2d_tokens(C, Fr, T1, cfg->t_max_period, cfg->t_weight_pos_embed);
  DcTensor cur_x = dc_add(xn, pe);
  dc_free(xn);
  dc_free(pe);

  int Ct = xt.shape[1], T2 = xt.shape[2];
  int tax2[3] = {0, 2, 1};
  DcTensor xtt = dc_transpose(xt, tax2, 3);
  DcTensor xtn = dc_ln_pref(xtt, w, "crosstransformer.norm_in_t");
  dc_free(xtt);
  DcTensor pos = sin_embedding(T2, Ct, cfg->t_max_period, cfg->t_weight_pos_embed);
  DcTensor cur_xt = dc_add(xtn, pos);
  dc_free(xtn);
  dc_free(pos);

  int classic_parity = cfg->t_cross_first ? 1 : 0;
  for (int idx = 0; idx < cfg->t_layers; idx++) {
    char lp[128], lpt[128];
    snprintf(lp, sizeof(lp), "crosstransformer.layers.%d", idx);
    snprintf(lpt, sizeof(lpt), "crosstransformer.layers_t.%d", idx);
    DcTensor nx, nxt;
    if (idx % 2 == classic_parity) {
      nx = self_attn_layer(cur_x, w, lp, cfg);
      nxt = self_attn_layer(cur_xt, w, lpt, cfg);
    } else {
      nx = cross_attn_layer(cur_x, cur_xt, w, lp, cfg);
      nxt = cross_attn_layer(cur_xt, cur_x, w, lpt, cfg);
    }
    dc_free(cur_x);
    dc_free(cur_xt);
    cur_x = nx;
    cur_xt = nxt;
  }
  int rshape[4] = {B, T1, Fr, C};
  DcTensor xr = dc_reshape(cur_x, rshape, 4);
  DcTensor xout = dc_transpose(xr, tax, 4);
  dc_free(xr);
  DcTensor xtout = dc_transpose(cur_xt, tax2, 3);
  dc_free(cur_x);
  dc_free(cur_xt);
  dc_free(*x_io);
  dc_free(*xt_io);
  *x_io = xout;
  *xt_io = xtout;
}
