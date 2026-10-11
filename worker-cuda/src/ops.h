#ifndef DEMUCS_CUDA_OPS_H
#define DEMUCS_CUDA_OPS_H

#include "weights.h"

/* Scratch budget for im2col / attention score buffers. */
void dc_ops_set_scratch_budget(size_t bytes);

/* x [N, C, L]; weight "<prefix>.weight" [O, K, C] */
DcTensor dc_conv1d(DcTensor x, const DcWeight *weight, const DcWeight *bias, int stride,
                   int padding, int dilation);
/* x [N, C, H, W]; weight [O, KH, KW, C] */
DcTensor dc_conv2d(DcTensor x, const DcWeight *weight, const DcWeight *bias, int sh, int sw,
                   int ph, int pw);
/* x [N, C, L]; weight [O, K, C]; output length (L-1)*stride + K (no padding) */
DcTensor dc_conv_tr1d(DcTensor x, const DcWeight *weight, const DcWeight *bias, int stride);
/* x [N, C, H, W]; weight [O, KH, KW, C] */
DcTensor dc_conv_tr2d(DcTensor x, const DcWeight *weight, const DcWeight *bias, int sh,
                      int sw);
/* x [..., in]; weight [out, in] */
DcTensor dc_linear(DcTensor x, const DcWeight *weight, const DcWeight *bias);

/* prefixed helpers: "<prefix>.conv.weight" / "<prefix>.conv.bias" */
DcTensor dc_conv1d_pref(DcTensor x, const DcWeights *w, const char *prefix, int stride,
                        int padding, int dilation);
DcTensor dc_conv2d_pref(DcTensor x, const DcWeights *w, const char *prefix, int sh, int sw,
                        int ph, int pw);
DcTensor dc_convtr1d_pref(DcTensor x, const DcWeights *w, const char *prefix, int stride);
DcTensor dc_convtr2d_pref(DcTensor x, const DcWeights *w, const char *prefix, int sh, int sw);
/* "<prefix>.weight" / "<prefix>.bias" */
DcTensor dc_linear_pref(DcTensor x, const DcWeights *w, const char *prefix);
DcTensor dc_ln_pref(DcTensor x, const DcWeights *w, const char *prefix);

DcTensor dc_layer_norm(DcTensor x, DcTensor weight, DcTensor bias, float eps);
/* channel-first group norm on [B, C, ...]; act 0 none, 1 gelu, 2 glu */
DcTensor dc_groupnorm(DcTensor x, DcTensor weight, DcTensor bias, int groups, float eps,
                      int act);
/* single group over all non-batch dims, affine on the last (channel) dim */
DcTensor dc_groupnorm_last(DcTensor x, DcTensor weight, DcTensor bias, float eps);
/* y[o, c, i] += x[o, c, i] * scale[c] where C is axis `axis` */
void dc_scale_add_inplace(DcTensor y, DcTensor x, DcTensor scale, int axis);
/* per-batch mean and biased variance over all non-batch dims ([B] each) */
void dc_batch_stats(DcTensor x, DcTensor *mean, DcTensor *var);
/* (x - mean) / (std + eps)  or  x * (std + eps) + mean, stats per batch row */
DcTensor dc_batch_normalize(DcTensor x, DcTensor mean, DcTensor var, float eps, int denorm);

/* projected q [B, Tq, D], k/v [B, Tk, D] -> softmax(q k^T / sqrt(hd)) v, [B, Tq, D] */
DcTensor dc_attention(DcTensor q, DcTensor k, DcTensor v, int nheads);

#endif
