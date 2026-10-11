#ifndef DEMUCS_CUDA_HTDEMUCS_H
#define DEMUCS_CUDA_HTDEMUCS_H

#include "stft.h"
#include "transformer.h"

#define DC_MAX_DEPTH 8

typedef struct {
  DcWeights *w;
  DcStft stft;
  DcLayerSpec encoder[DC_MAX_DEPTH];
  DcLayerSpec tencoder[DC_MAX_DEPTH];
  DcLayerSpec decoder[DC_MAX_DEPTH];
  DcLayerSpec tdecoder[DC_MAX_DEPTH];
  int n_enc;
  int n_tenc;
  int n_dec;
  int n_tdec;
} DcModel;

void dc_model_init(DcModel *m, DcWeights *w);
void dc_model_free(DcModel *m);
/* mix [B, C, T] -> stems [B, S, C, T] */
DcTensor dc_htdemucs_forward(DcModel *m, DcTensor mix);
/* adds the frequency embedding after encoder 0 */
DcTensor dc_add_freq_emb(DcModel *m, DcTensor x);

#endif
