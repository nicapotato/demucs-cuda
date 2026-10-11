#ifndef DEMUCS_CUDA_STFT_H
#define DEMUCS_CUDA_STFT_H

#include "tensor.h"

typedef struct {
  int n_fft;
  int hop;
  DcTensor window; /* periodic Hann [n_fft] */
} DcStft;

DcStft dc_stft_new(int n_fft, int hop);
void dc_stft_free(DcStft *st);

/* HTDemucs _spec + _magnitude(cac): mix [B, C, T] -> [B, 2C, n_fft/2, ceil(T/hop)]
 * with channel 2c = real, 2c+1 = imag */
DcTensor dc_htdemucs_spec_cac(DcStft *st, DcTensor mix);
/* HTDemucs _mask(cac) + _ispec: m [B, S, 2C, F, N] -> [B, S, C, length] */
DcTensor dc_htdemucs_ispec_cac(DcStft *st, DcTensor m, int length);

#endif
