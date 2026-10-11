#ifndef DEMUCS_CUDA_WEIGHTS_H
#define DEMUCS_CUDA_WEIGHTS_H

#include "config.h"
#include "tensor.h"

typedef enum {
  DC_W_F32 = 0, /* biases, norms, scales, embeddings: fp32 tensor */
  DC_W_GEMM,    /* conv / linear: fp16 A operand [rows][kp] */
  DC_W_GEMM_TR, /* transposed conv: fp16 A operand [O*K][kp] */
} DcWeightKind;

typedef struct {
  char name[112];
  int ndim;
  int shape[DC_MAX_DIM]; /* original MLX layout, e.g. conv [O, K, I] */
  DcWeightKind kind;
  void *dptr;
  int64_t rows; /* GEMM A rows */
  int64_t kp;   /* GEMM A row stride (reduction length padded to 8) */
  DcTensor t;   /* DC_W_F32 only */
} DcWeight;

typedef struct {
  DcWeight *items;
  int n;
  int *table;
  int table_cap;
  void *block;
  size_t block_bytes;
  DcConfig cfg;
} DcWeights;

/* Loads F32 / F16 / BF16 safetensors in MLX layout (worker-c export) */
void dc_weights_load(DcWeights *w, const char *safetensors_path, const char *json_path);
void dc_weights_free(DcWeights *w);

const DcWeight *dc_wfind(const DcWeights *w, const char *fmt, ...);
const DcWeight *dc_wget(const DcWeights *w, const char *fmt, ...); /* dies if missing */
/* new reference to an fp32 weight tensor; caller frees */
DcTensor dc_wt(const DcWeights *w, const char *fmt, ...);
bool dc_w_has(const DcWeights *w, const char *fmt, ...);

/* standalone weight packed with the same rules as the loader (tests) */
void dc_weight_from_host(DcWeight *it, const char *name, const float *data, const int *shape,
                         int ndim);
void dc_weight_release(DcWeight *it);

/* generic safetensors access (tests / reference tensors): fp32 device tensors */
typedef struct DcStFile DcStFile;
DcStFile *dc_st_open(const char *path); /* NULL if unreadable */
DcTensor dc_st_get(DcStFile *f, const char *name); /* empty tensor if missing */
void dc_st_close(DcStFile *f);

#endif
