#ifndef DEMUCS_CUDA_TENSOR_H
#define DEMUCS_CUDA_TENSOR_H

#include "backend.h"
#include "util.h"

/* Contiguous fp32 device tensor. Buffers are refcounted; every op returns a new
 * tensor and never mutates its inputs (except the explicit *_inplace helpers). */
typedef struct DcBuf DcBuf;

typedef struct {
  DcBuf *buf;
  int ndim;
  int shape[DC_MAX_DIM];
} DcTensor;

/* caching device allocator: freed blocks are reused, cudaMalloc only on growth.
 * *got receives the block size, which must be passed back to dc_dev_free. */
void *dc_dev_alloc(size_t bytes, size_t *got);
void dc_dev_free(void *p, size_t got);

/* scratch buffers for GEMM operands */
typedef struct {
  void *ptr;
  size_t bytes;
} DcScratch;
DcScratch dc_scratch(size_t bytes);
void dc_scratch_free(DcScratch *s);
void dc_dev_release_cache(void);
size_t dc_dev_peak_bytes(void);
size_t dc_dev_live_bytes(void);

static inline DcTensor dc_empty(void) {
  DcTensor t;
  memset(&t, 0, sizeof(t));
  return t;
}
static inline int dc_valid(DcTensor t) { return t.buf != NULL; }

DcTensor dc_new(const int *shape, int ndim);
DcTensor dc_zeros(const int *shape, int ndim);
DcTensor dc_full(const int *shape, int ndim, float v);
DcTensor dc_from_host(const float *data, const int *shape, int ndim);
float *dc_to_host(DcTensor t); /* malloc'd copy, synchronises */
DcTensor dc_ref(DcTensor t);
void dc_free(DcTensor t);
/* tensor over memory owned elsewhere (e.g. the weight block); never freed */
DcTensor dc_wrap_external(float *ptr, const int *shape, int ndim);
/* ensure *t is the sole owner of its buffer (copies if shared) */
void dc_make_unique(DcTensor *t);

float *dc_data(DcTensor t);
int64_t dc_numel(DcTensor t);
int dc_dim(DcTensor t, int axis);
void dc_print_shape(const char *name, DcTensor t);

DcTensor dc_reshape(DcTensor t, const int *shape, int ndim); /* view; one -1 allowed */
DcTensor dc_transpose(DcTensor t, const int *axes, int naxes);
DcTensor dc_slice(DcTensor t, const int *start, const int *stop);
DcTensor dc_slice_axis(DcTensor t, int axis, int start, int stop);
DcTensor dc_pad_axis(DcTensor t, int axis, int left, int right); /* zeros */
DcTensor dc_pad1d(DcTensor t, int left, int right, int reflect);
DcTensor dc_concat(const DcTensor *items, int n, int axis);
DcTensor dc_broadcast_to(DcTensor t, const int *shape, int ndim);
DcTensor dc_center_trim(DcTensor t, int ref_len);
DcTensor dc_squeeze_axis(DcTensor t, int axis);
DcTensor dc_expand_dims(DcTensor t, int axis);

DcTensor dc_add(DcTensor a, DcTensor b);
DcTensor dc_sub(DcTensor a, DcTensor b);
DcTensor dc_mul(DcTensor a, DcTensor b);
DcTensor dc_div(DcTensor a, DcTensor b);
DcTensor dc_affine(DcTensor a, float alpha, float beta);
DcTensor dc_gelu(DcTensor a);
DcTensor dc_relu(DcTensor a);
DcTensor dc_glu_axis1(DcTensor a);

/* dst[..., start:start+src.shape] += src  (dst must be exclusively owned) */
void dc_slice_add_inplace(DcTensor dst, DcTensor src, const int *start);
void dc_slice_set_inplace(DcTensor dst, DcTensor src, const int *start);

#endif
