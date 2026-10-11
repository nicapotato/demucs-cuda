#include "layers.h"

static DcTensor freq_to_ncl(DcTensor y) {
  int B = y.shape[0], C = y.shape[1], Fr = y.shape[2], T = y.shape[3];
  int tax[4] = {0, 2, 1, 3};
  DcTensor yt = dc_transpose(y, tax, 4);
  int r[3] = {B * Fr, C, T};
  DcTensor yr = dc_reshape(yt, r, 3);
  dc_free(yt);
  return yr;
}

static DcTensor ncl_to_freq(DcTensor yd, int B, int Fr, int C, int T) {
  int r4[4] = {B, Fr, C, T};
  DcTensor yb = dc_reshape(yd, r4, 4);
  int tax[4] = {0, 2, 1, 3};
  DcTensor y2 = dc_transpose(yb, tax, 4);
  dc_free(yb);
  return y2;
}

/* DConv residual branch: conv(k=3, dilation 2^d) -> GN(1)+GELU -> conv 1x1 ->
 * GN(1)+GLU -> layer scale, added to the input. */
DcTensor dc_dconv(DcTensor x, const DcWeights *w, const char *prefix, int depth) {
  DcTensor y = dc_ref(x);
  for (int d = 0; d < depth; d++) {
    int dilation = 1 << d;
    char p0[192], p3[192];
    snprintf(p0, sizeof(p0), "%s.layers.%d.layers.0", prefix, d);
    snprintf(p3, sizeof(p3), "%s.layers.%d.layers.3", prefix, d);
    DcTensor h = dc_conv1d_pref(y, w, p0, 1, dilation, dilation);
    DcTensor w1 = dc_wt(w, "%s.layers.%d.layers.1.weight", prefix, d);
    DcTensor b1 = dc_wt(w, "%s.layers.%d.layers.1.bias", prefix, d);
    DcTensor h1 = dc_groupnorm(h, w1, b1, 1, 1e-5f, 1);
    dc_free(h);
    DcTensor h2 = dc_conv1d_pref(h1, w, p3, 1, 0, 1);
    dc_free(h1);
    DcTensor w2 = dc_wt(w, "%s.layers.%d.layers.4.weight", prefix, d);
    DcTensor b2 = dc_wt(w, "%s.layers.%d.layers.4.bias", prefix, d);
    DcTensor h3 = dc_groupnorm(h2, w2, b2, 1, 1e-5f, 2);
    dc_free(h2);
    DcTensor scale = dc_wt(w, "%s.layers.%d.layers.6.scale", prefix, d);
    dc_make_unique(&y);
    dc_scale_add_inplace(y, h3, scale, 1);
    dc_free(h3);
    dc_free(w1);
    dc_free(b1);
    dc_free(w2);
    dc_free(b2);
    dc_free(scale);
  }
  return y;
}

static DcTensor dconv_any(DcTensor y, const DcWeights *w, const DcLayerSpec *spec) {
  char dp[160];
  snprintf(dp, sizeof(dp), "%s.dconv", spec->prefix);
  if (spec->freq) {
    int B = y.shape[0], C = y.shape[1], Fr = y.shape[2], T = y.shape[3];
    DcTensor yr = freq_to_ncl(y);
    DcTensor yd = dc_dconv(yr, w, dp, spec->dconv_depth);
    dc_free(yr);
    DcTensor y2 = ncl_to_freq(yd, B, Fr, C, T);
    dc_free(yd);
    return y2;
  }
  return dc_dconv(y, w, dp, spec->dconv_depth);
}

static DcTensor gn_act(DcTensor z, const DcWeights *w, const char *prefix, const char *norm,
                       int groups, int act) {
  DcTensor nw = dc_wt(w, "%s.%s.weight", prefix, norm);
  DcTensor nb = dc_wt(w, "%s.%s.bias", prefix, norm);
  DcTensor y = dc_groupnorm(z, nw, nb, groups, 1e-5f, act);
  dc_free(nw);
  dc_free(nb);
  return y;
}

DcTensor dc_henc(DcTensor x_in, DcTensor inject, const DcWeights *w, const DcLayerSpec *spec) {
  DcTensor x = dc_ref(x_in);
  if (!spec->freq && x.ndim == 4) {
    int r[3] = {x.shape[0], x.shape[1] * x.shape[2], x.shape[3]};
    DcTensor xr = dc_reshape(x, r, 3);
    dc_free(x);
    x = xr;
  }
  if (!spec->freq) {
    int le = x.shape[x.ndim - 1];
    if (le % spec->stride != 0) {
      DcTensor xp = dc_pad1d(x, 0, spec->stride - (le % spec->stride), 0);
      dc_free(x);
      x = xp;
    }
  }
  char convp[160];
  snprintf(convp, sizeof(convp), "%s.conv", spec->prefix);
  DcTensor y;
  if (spec->freq) {
    y = dc_conv2d_pref(x, w, convp, spec->stride, 1, spec->pad, 0);
  } else {
    y = dc_conv1d_pref(x, w, convp, spec->stride, spec->pad, 1);
  }
  dc_free(x);
  if (spec->empty) {
    return y;
  }
  if (dc_valid(inject)) {
    DcTensor inj = dc_ref(inject);
    if (inj.ndim == 3 && y.ndim == 4) {
      DcTensor e = dc_expand_dims(inj, 2);
      dc_free(inj);
      inj = e;
    }
    DcTensor y2 = dc_add(y, inj);
    dc_free(y);
    dc_free(inj);
    y = y2;
  }
  DcTensor y2 = spec->fused_norm1 ? gn_act(y, w, spec->prefix, "norm1", spec->norm_groups, 1)
                                  : dc_gelu(y);
  dc_free(y);
  y = y2;
  if (spec->dconv) {
    DcTensor yd = dconv_any(y, w, spec);
    dc_free(y);
    y = yd;
  }
  if (spec->rewrite) {
    char rp[160];
    snprintf(rp, sizeof(rp), "%s.rewrite", spec->prefix);
    DcTensor z = spec->freq ? dc_conv2d_pref(y, w, rp, 1, 1, spec->context, spec->context)
                            : dc_conv1d_pref(y, w, rp, 1, spec->context, 1);
    DcTensor z2 = spec->fused_norm2 ? gn_act(z, w, spec->prefix, "norm2", spec->norm_groups, 2)
                                    : dc_glu_axis1(z);
    dc_free(z);
    dc_free(y);
    y = z2;
  }
  return y;
}

DcTensor dc_hdec(DcTensor x_in, DcTensor skip, int length, const DcWeights *w,
                 const DcLayerSpec *spec, DcTensor *pre_out) {
  DcTensor x = dc_ref(x_in);
  if (spec->freq && x.ndim == 3) {
    int C = x.shape[1];
    if (C % spec->chin != 0) {
      dc_die("hdec freq reshape %d not divisible by chin %d", C, spec->chin);
    }
    int r[4] = {x.shape[0], spec->chin, C / spec->chin, x.shape[2]};
    DcTensor xr = dc_reshape(x, r, 4);
    dc_free(x);
    x = xr;
  }
  DcTensor y;
  if (!spec->empty) {
    if (!dc_valid(skip)) {
      dc_die("hdec skip is required when empty=0");
    }
    DcTensor xs = dc_add(x, skip);
    dc_free(x);
    x = xs;
    if (spec->rewrite) {
      char rp[160];
      snprintf(rp, sizeof(rp), "%s.rewrite", spec->prefix);
      DcTensor z;
      if (spec->freq) {
        z = spec->context_freq ? dc_conv2d_pref(x, w, rp, 1, 1, spec->context, spec->context)
                               : dc_conv2d_pref(x, w, rp, 1, 1, 0, spec->context);
      } else {
        z = dc_conv1d_pref(x, w, rp, 1, spec->context, 1);
      }
      y = spec->fused_norm1 ? gn_act(z, w, spec->prefix, "norm1", spec->norm_groups, 2)
                            : dc_glu_axis1(z);
      dc_free(z);
    } else {
      y = dc_ref(x);
    }
    if (spec->dconv) {
      DcTensor yd = dconv_any(y, w, spec);
      dc_free(y);
      y = yd;
    }
  } else {
    y = dc_ref(x);
  }
  if (pre_out) {
    *pre_out = dc_ref(y);
  }
  char cp[160];
  snprintf(cp, sizeof(cp), "%s.conv_tr", spec->prefix);
  DcTensor z = spec->freq ? dc_convtr2d_pref(y, w, cp, spec->stride, 1)
                          : dc_convtr1d_pref(y, w, cp, spec->stride);
  /* decoder norm2 is Identity when norm=False (typical HTDemucs) */
  if (spec->norm && dc_w_has(w, "%s.norm2.weight", spec->prefix)) {
    DcTensor zn = gn_act(z, w, spec->prefix, "norm2", spec->norm_groups, 0);
    dc_free(z);
    z = zn;
  }
  if (spec->freq) {
    if (spec->pad) {
      DcTensor z2 = dc_slice_axis(z, 2, spec->pad, z.shape[2] - spec->pad);
      dc_free(z);
      z = z2;
    }
  } else {
    DcTensor z2 = dc_slice_axis(z, -1, spec->pad, spec->pad + length);
    dc_free(z);
    z = z2;
  }
  if (!spec->last) {
    DcTensor zg = dc_gelu(z);
    dc_free(z);
    z = zg;
  }
  dc_free(x);
  dc_free(y);
  return z;
}
