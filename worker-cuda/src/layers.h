#ifndef DEMUCS_CUDA_LAYERS_H
#define DEMUCS_CUDA_LAYERS_H

#include "ops.h"

typedef struct {
  char prefix[128];
  int chin;
  int chout;
  int kernel;
  int stride;
  int pad; /* kernel//4 or 0 */
  int freq;
  int empty;
  int last;
  int rewrite;
  int dconv;
  int norm; /* outer groupnorm enabled */
  int fused_norm1;
  int fused_norm2;
  int context;
  int context_freq;
  int norm_groups;
  int dconv_depth;
  int dconv_comp;
  float dconv_init;
} DcLayerSpec;

DcTensor dc_dconv(DcTensor x, const DcWeights *w, const char *prefix, int depth);
/* inject may be empty */
DcTensor dc_henc(DcTensor x, DcTensor inject, const DcWeights *w, const DcLayerSpec *spec);
/* returns z; if pre_out non-NULL stores the pre-conv_tr activation */
DcTensor dc_hdec(DcTensor x, DcTensor skip, int length, const DcWeights *w,
                 const DcLayerSpec *spec, DcTensor *pre_out);

#endif
