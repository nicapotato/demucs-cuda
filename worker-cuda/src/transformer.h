#ifndef DEMUCS_CUDA_TRANSFORMER_H
#define DEMUCS_CUDA_TRANSFORMER_H

#include "layers.h"

/* x [B, C, Fr, T1] and xt [B, Ct, T2] are replaced in place */
void dc_crosstransformer(DcTensor *x_io, DcTensor *xt_io, const DcWeights *w,
                         const DcConfig *cfg);

#endif
