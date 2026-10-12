#include "tensor.h"

struct DcBuf {
  float *ptr;
  size_t bytes; /* 0 for external memory (weights) that the allocator does not own */
  int refs;
};

/* ---------------------------------------------------------------- allocator */

typedef struct {
  void *ptr;
  size_t bytes;
} DcBlock;

static DcBlock *g_free_blocks;
static int g_free_n;
static int g_free_cap;
static size_t g_live_bytes;
static size_t g_cached_bytes;
static size_t g_peak_bytes;

static size_t round_alloc(size_t bytes) {
  if (bytes == 0) {
    bytes = 1;
  }
  if (bytes < ((size_t)1 << 20)) {
    return (bytes + 511) & ~(size_t)511;
  }
  size_t g = (size_t)2 << 20;
  return (bytes + g - 1) / g * g;
}

void dc_dev_release_cache(void) {
  for (int i = 0; i < g_free_n; i++) {
    dk_raw_free(g_free_blocks[i].ptr);
  }
  g_free_n = 0;
  g_cached_bytes = 0;
}

void *dc_dev_alloc(size_t bytes, size_t *got) {
  size_t want = round_alloc(bytes);
  int best = -1;
  for (int i = 0; i < g_free_n; i++) {
    size_t b = g_free_blocks[i].bytes;
    if (b >= want && b <= want + want / 4 + ((size_t)2 << 20) &&
        (best < 0 || b < g_free_blocks[best].bytes)) {
      best = i;
      if (b == want) {
        break;
      }
    }
  }
  void *p;
  if (best >= 0) {
    p = g_free_blocks[best].ptr;
    want = g_free_blocks[best].bytes;
    g_cached_bytes -= want;
    g_free_blocks[best] = g_free_blocks[--g_free_n];
  } else {
    p = dk_raw_malloc(want);
    if (!p) {
      dc_dev_release_cache();
      p = dk_raw_malloc(want);
    }
    if (!p) {
      size_t fb = 0, tb = 0;
      dk_mem_info(&fb, &tb);
      dc_die("out of device memory allocating %.1f MB (live %.1f MB, free %.1f MB of %.1f MB). "
             "Try a smaller batch: -b 1",
             (double)want / 1048576.0, (double)g_live_bytes / 1048576.0,
             (double)fb / 1048576.0, (double)tb / 1048576.0);
    }
  }
  g_live_bytes += want;
  if (g_live_bytes + g_cached_bytes > g_peak_bytes) {
    g_peak_bytes = g_live_bytes + g_cached_bytes;
  }
  *got = want;
  return p;
}

void dc_dev_free(void *p, size_t got) {
  if (!p) {
    return;
  }
  g_live_bytes -= got;
  if (g_free_n == g_free_cap) {
    g_free_cap = g_free_cap ? g_free_cap * 2 : 256;
    g_free_blocks = (DcBlock *)realloc(g_free_blocks, (size_t)g_free_cap * sizeof(DcBlock));
    if (!g_free_blocks) {
      dc_die("oom allocator table");
    }
  }
  g_free_blocks[g_free_n].ptr = p;
  g_free_blocks[g_free_n].bytes = got;
  g_free_n++;
  g_cached_bytes += got;
}

DcScratch dc_scratch(size_t bytes) {
  DcScratch s;
  s.ptr = dc_dev_alloc(bytes, &s.bytes);
  return s;
}

void dc_scratch_free(DcScratch *s) {
  dc_dev_free(s->ptr, s->bytes);
  s->ptr = NULL;
  s->bytes = 0;
}

size_t dc_dev_peak_bytes(void) { return g_peak_bytes; }
size_t dc_dev_live_bytes(void) { return g_live_bytes; }

/* ------------------------------------------------------------------ tensors */

static int64_t shape_numel(const int *shape, int ndim) {
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) {
    n *= shape[i];
  }
  return n;
}

DcTensor dc_new(const int *shape, int ndim) {
  if (ndim < 0 || ndim > DC_MAX_DIM) {
    dc_die("bad ndim %d", ndim);
  }
  DcTensor t;
  memset(&t, 0, sizeof(t));
  t.ndim = ndim;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] < 0) {
      dc_die("negative dim %d", shape[i]);
    }
    t.shape[i] = shape[i];
  }
  DcBuf *b = (DcBuf *)malloc(sizeof(DcBuf));
  if (!b) {
    dc_die("oom tensor header");
  }
  size_t bytes = (size_t)shape_numel(shape, ndim) * sizeof(float);
  b->ptr = (float *)dc_dev_alloc(bytes, &b->bytes);
  b->refs = 1;
  t.buf = b;
  return t;
}

DcTensor dc_zeros(const int *shape, int ndim) {
  DcTensor t = dc_new(shape, ndim);
  dk_memset0(t.buf->ptr, (size_t)dc_numel(t) * sizeof(float));
  return t;
}

DcTensor dc_full(const int *shape, int ndim, float v) {
  DcTensor t = dc_new(shape, ndim);
  dk_fill(t.buf->ptr, dc_numel(t), v);
  return t;
}

DcTensor dc_from_host(const float *data, const int *shape, int ndim) {
  DcTensor t = dc_new(shape, ndim);
  dk_memcpy_h2d(t.buf->ptr, data, (size_t)dc_numel(t) * sizeof(float));
  return t;
}

float *dc_to_host(DcTensor t) {
  size_t n = (size_t)dc_numel(t);
  float *h = (float *)malloc(n * sizeof(float) + 1);
  if (!h) {
    dc_die("oom host copy (%zu floats)", n);
  }
  dk_memcpy_d2h(h, t.buf->ptr, n * sizeof(float));
  return h;
}

DcTensor dc_ref(DcTensor t) {
  if (t.buf) {
    t.buf->refs++;
  }
  return t;
}

void dc_free(DcTensor t) {
  if (!t.buf) {
    return;
  }
  if (--t.buf->refs == 0) {
    if (t.buf->bytes) {
      dc_dev_free(t.buf->ptr, t.buf->bytes);
    }
    free(t.buf);
  }
}

void dc_make_unique(DcTensor *t) {
  if (!t->buf || (t->buf->refs == 1 && t->buf->bytes)) {
    return;
  }
  DcTensor c = dc_new(t->shape, t->ndim);
  dk_memcpy_d2d(dc_data(c), dc_data(*t), (size_t)dc_numel(c) * sizeof(float));
  dc_free(*t);
  *t = c;
}

DcTensor dc_wrap_external(float *ptr, const int *shape, int ndim) {
  DcTensor t;
  memset(&t, 0, sizeof(t));
  t.ndim = ndim;
  for (int i = 0; i < ndim; i++) {
    t.shape[i] = shape[i];
  }
  DcBuf *b = (DcBuf *)malloc(sizeof(DcBuf));
  if (!b) {
    dc_die("oom tensor header");
  }
  b->ptr = ptr;
  b->bytes = 0;
  b->refs = 1;
  t.buf = b;
  return t;
}

float *dc_data(DcTensor t) {
  if (!t.buf) {
    dc_die("dc_data on empty tensor");
  }
  return t.buf->ptr;
}

int64_t dc_numel(DcTensor t) { return shape_numel(t.shape, t.ndim); }

int dc_dim(DcTensor t, int axis) {
  if (axis < 0) {
    axis += t.ndim;
  }
  if (axis < 0 || axis >= t.ndim) {
    dc_die("axis %d out of range for ndim %d", axis, t.ndim);
  }
  return t.shape[axis];
}

void dc_print_shape(const char *name, DcTensor t) {
  fprintf(stderr, "%s [", name);
  for (int i = 0; i < t.ndim; i++) {
    fprintf(stderr, "%s%d", i ? "," : "", t.shape[i]);
  }
  fprintf(stderr, "]\n");
}

static void contiguous_strides(const int *shape, int ndim, int64_t *strides) {
  int64_t s = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = s;
    s *= shape[i];
  }
}

DcTensor dc_reshape(DcTensor t, const int *shape, int ndim) {
  int s[DC_MAX_DIM] = {0};
  int infer = -1;
  int64_t known = 1;
  for (int i = 0; i < ndim; i++) {
    s[i] = shape[i];
    if (s[i] == -1) {
      if (infer >= 0) {
        dc_die("reshape: more than one -1");
      }
      infer = i;
    } else {
      known *= s[i];
    }
  }
  int64_t n = dc_numel(t);
  if (infer >= 0) {
    if (known == 0 || n % known != 0) {
      dc_die("reshape: cannot infer dim");
    }
    s[infer] = (int)(n / known);
  }
  if (shape_numel(s, ndim) != n) {
    dc_print_shape("reshape from", t);
    dc_die("reshape: element count mismatch (%lld vs %lld)", (long long)shape_numel(s, ndim),
           (long long)n);
  }
  DcTensor r = dc_ref(t);
  r.ndim = ndim;
  for (int i = 0; i < ndim; i++) {
    r.shape[i] = s[i];
  }
  return r;
}

/* Merge adjacent dims that are contiguous in both stride sets. */
static void coalesce(DkStrided *p) {
  int out = 0;
  for (int i = 0; i < p->ndim; i++) {
    if (p->shape[i] == 1) {
      continue;
    }
    if (out > 0 && p->a_strides[out - 1] == p->a_strides[i] * p->shape[i] &&
        p->b_strides[out - 1] == p->b_strides[i] * p->shape[i]) {
      p->shape[out - 1] *= p->shape[i];
      p->a_strides[out - 1] = p->a_strides[i];
      p->b_strides[out - 1] = p->b_strides[i];
      continue;
    }
    p->shape[out] = p->shape[i];
    p->a_strides[out] = p->a_strides[i];
    p->b_strides[out] = p->b_strides[i];
    out++;
  }
  if (out == 0) {
    p->shape[0] = 1;
    p->a_strides[0] = 1;
    p->b_strides[0] = 1;
    out = 1;
  }
  p->ndim = out;
}

DcTensor dc_transpose(DcTensor t, const int *axes, int naxes) {
  if (naxes != t.ndim) {
    dc_die("transpose: %d axes for ndim %d", naxes, t.ndim);
  }
  int64_t src_strides[DC_MAX_DIM];
  contiguous_strides(t.shape, t.ndim, src_strides);
  int oshape[DC_MAX_DIM];
  DkStrided p;
  memset(&p, 0, sizeof(p));
  p.ndim = naxes;
  for (int i = 0; i < naxes; i++) {
    int a = axes[i] < 0 ? axes[i] + t.ndim : axes[i];
    oshape[i] = t.shape[a];
    p.shape[i] = t.shape[a];
    p.a_strides[i] = src_strides[a];
  }
  contiguous_strides(oshape, naxes, p.b_strides);
  DcTensor r = dc_new(oshape, naxes);
  coalesce(&p);
  dk_strided_copy(&p, dc_data(t), dc_data(r), 0);
  return r;
}

DcTensor dc_slice(DcTensor t, const int *start, const int *stop) {
  int64_t src_strides[DC_MAX_DIM];
  contiguous_strides(t.shape, t.ndim, src_strides);
  int oshape[DC_MAX_DIM];
  int64_t offset = 0;
  DkStrided p;
  memset(&p, 0, sizeof(p));
  p.ndim = t.ndim;
  for (int i = 0; i < t.ndim; i++) {
    if (start[i] < 0 || stop[i] > t.shape[i] || stop[i] < start[i]) {
      dc_print_shape("slice of", t);
      dc_die("slice: bad range [%d, %d) on axis %d", start[i], stop[i], i);
    }
    oshape[i] = stop[i] - start[i];
    offset += (int64_t)start[i] * src_strides[i];
    p.shape[i] = oshape[i];
    p.a_strides[i] = src_strides[i];
  }
  contiguous_strides(oshape, t.ndim, p.b_strides);
  DcTensor r = dc_new(oshape, t.ndim);
  if (dc_numel(r) > 0) {
    coalesce(&p);
    dk_strided_copy(&p, dc_data(t) + offset, dc_data(r), 0);
  }
  return r;
}

DcTensor dc_slice_axis(DcTensor t, int axis, int start, int stop) {
  if (axis < 0) {
    axis += t.ndim;
  }
  int st[DC_MAX_DIM], sp[DC_MAX_DIM];
  for (int i = 0; i < t.ndim; i++) {
    st[i] = 0;
    sp[i] = t.shape[i];
  }
  st[axis] = start;
  sp[axis] = stop;
  return dc_slice(t, st, sp);
}

static void slice_write(DcTensor dst, DcTensor src, const int *start, int accumulate) {
  if (dst.ndim != src.ndim) {
    dc_die("slice write: ndim mismatch");
  }
  int64_t dst_strides[DC_MAX_DIM];
  contiguous_strides(dst.shape, dst.ndim, dst_strides);
  DkStrided p;
  memset(&p, 0, sizeof(p));
  p.ndim = src.ndim;
  contiguous_strides(src.shape, src.ndim, p.a_strides);
  int64_t offset = 0;
  for (int i = 0; i < src.ndim; i++) {
    if (start[i] < 0 || start[i] + src.shape[i] > dst.shape[i]) {
      dc_die("slice write out of range on axis %d", i);
    }
    p.shape[i] = src.shape[i];
    p.b_strides[i] = dst_strides[i];
    offset += (int64_t)start[i] * dst_strides[i];
  }
  if (dc_numel(src) == 0) {
    return;
  }
  coalesce(&p);
  dk_strided_copy(&p, dc_data(src), dc_data(dst) + offset, accumulate);
}

void dc_slice_add_inplace(DcTensor dst, DcTensor src, const int *start) {
  slice_write(dst, src, start, 1);
}

void dc_slice_set_inplace(DcTensor dst, DcTensor src, const int *start) {
  slice_write(dst, src, start, 0);
}

DcTensor dc_pad_axis(DcTensor t, int axis, int left, int right) {
  if (axis < 0) {
    axis += t.ndim;
  }
  if (left == 0 && right == 0) {
    return dc_ref(t);
  }
  int oshape[DC_MAX_DIM];
  int start[DC_MAX_DIM];
  for (int i = 0; i < t.ndim; i++) {
    oshape[i] = t.shape[i];
    start[i] = 0;
  }
  oshape[axis] += left + right;
  start[axis] = left;
  DcTensor r = dc_zeros(oshape, t.ndim);
  slice_write(r, t, start, 0);
  return r;
}

DcTensor dc_pad1d(DcTensor t, int left, int right, int reflect) {
  if (!reflect) {
    return dc_pad_axis(t, -1, left, right);
  }
  int length = t.shape[t.ndim - 1];
  int pl = left, pr = right;
  DcTensor x = dc_ref(t);
  int max_pad = pl > pr ? pl : pr;
  if (length <= max_pad) {
    int extra_pad = max_pad - length + 1;
    int extra_right = pr < extra_pad ? pr : extra_pad;
    int extra_left = extra_pad - extra_right;
    pl -= extra_left;
    pr -= extra_right;
    DcTensor padded = dc_pad_axis(x, -1, extra_left, extra_right);
    dc_free(x);
    x = padded;
    length = x.shape[x.ndim - 1];
  }
  if (pl == 0 && pr == 0) {
    return x;
  }
  int oshape[DC_MAX_DIM];
  for (int i = 0; i < x.ndim; i++) {
    oshape[i] = x.shape[i];
  }
  oshape[x.ndim - 1] = length + pl + pr;
  DcTensor r = dc_new(oshape, x.ndim);
  int64_t rows = dc_numel(x) / length;
  dk_reflect_pad(dc_data(x), dc_data(r), rows, length, pl, pr);
  dc_free(x);
  return r;
}

DcTensor dc_concat(const DcTensor *items, int n, int axis) {
  int nd = items[0].ndim;
  if (axis < 0) {
    axis += nd;
  }
  int oshape[DC_MAX_DIM];
  for (int i = 0; i < nd; i++) {
    oshape[i] = items[0].shape[i];
  }
  oshape[axis] = 0;
  for (int k = 0; k < n; k++) {
    oshape[axis] += items[k].shape[axis];
  }
  DcTensor r = dc_new(oshape, nd);
  int start[DC_MAX_DIM] = {0};
  for (int k = 0; k < n; k++) {
    slice_write(r, items[k], start, 0);
    start[axis] += items[k].shape[axis];
  }
  return r;
}

DcTensor dc_broadcast_to(DcTensor t, const int *shape, int ndim) {
  if (ndim < t.ndim) {
    dc_die("broadcast_to: lower rank");
  }
  int64_t ts[DC_MAX_DIM];
  contiguous_strides(t.shape, t.ndim, ts);
  DkStrided p;
  memset(&p, 0, sizeof(p));
  p.ndim = ndim;
  for (int i = 0; i < ndim; i++) {
    int ti = i - (ndim - t.ndim);
    p.shape[i] = shape[i];
    if (ti < 0) {
      p.a_strides[i] = 0;
    } else if (t.shape[ti] == shape[i]) {
      p.a_strides[i] = ts[ti];
    } else if (t.shape[ti] == 1) {
      p.a_strides[i] = 0;
    } else {
      dc_die("broadcast_to: incompatible dim %d (%d -> %d)", i, t.shape[ti], shape[i]);
    }
  }
  contiguous_strides(shape, ndim, p.b_strides);
  DcTensor r = dc_new(shape, ndim);
  coalesce(&p);
  dk_strided_copy(&p, dc_data(t), dc_data(r), 0);
  return r;
}

DcTensor dc_center_trim(DcTensor t, int ref_len) {
  int length = t.shape[t.ndim - 1];
  int delta = length - ref_len;
  if (delta < 0) {
    dc_die("center_trim: tensor shorter than reference (%d < %d)", length, ref_len);
  }
  if (delta == 0) {
    return dc_ref(t);
  }
  int start = delta / 2;
  return dc_slice_axis(t, -1, start, length - (delta - start));
}

DcTensor dc_squeeze_axis(DcTensor t, int axis) {
  if (axis < 0) {
    axis += t.ndim;
  }
  if (t.shape[axis] != 1) {
    dc_die("squeeze: axis %d has size %d", axis, t.shape[axis]);
  }
  int s[DC_MAX_DIM];
  int k = 0;
  for (int i = 0; i < t.ndim; i++) {
    if (i != axis) {
      s[k++] = t.shape[i];
    }
  }
  return dc_reshape(t, s, k);
}

DcTensor dc_expand_dims(DcTensor t, int axis) {
  if (axis < 0) {
    axis += t.ndim + 1;
  }
  int s[DC_MAX_DIM];
  int k = 0;
  for (int i = 0; i <= t.ndim; i++) {
    if (i == axis) {
      s[k++] = 1;
    }
    if (i < t.ndim) {
      s[k++] = t.shape[i];
    }
  }
  return dc_reshape(t, s, t.ndim + 1);
}

static DcTensor binary(DkBinOp op, DcTensor a, DcTensor b) {
  int nd = a.ndim > b.ndim ? a.ndim : b.ndim;
  int64_t as[DC_MAX_DIM], bs[DC_MAX_DIM];
  contiguous_strides(a.shape, a.ndim, as);
  contiguous_strides(b.shape, b.ndim, bs);
  int oshape[DC_MAX_DIM];
  DkStrided p;
  memset(&p, 0, sizeof(p));
  p.ndim = nd;
  for (int i = 0; i < nd; i++) {
    int ai = i - (nd - a.ndim);
    int bi = i - (nd - b.ndim);
    int ad = ai >= 0 ? a.shape[ai] : 1;
    int bd = bi >= 0 ? b.shape[bi] : 1;
    if (ad != bd && ad != 1 && bd != 1) {
      dc_print_shape("a", a);
      dc_print_shape("b", b);
      dc_die("binary op: incompatible shapes");
    }
    oshape[i] = ad > bd ? ad : bd;
    p.shape[i] = oshape[i];
    p.a_strides[i] = (ai >= 0 && ad != 1) ? as[ai] : 0;
    p.b_strides[i] = (bi >= 0 && bd != 1) ? bs[bi] : 0;
  }
  DcTensor y = dc_new(oshape, nd);
  if (dc_numel(y) > 0) {
    dk_binary(op, &p, dc_data(a), dc_data(b), dc_data(y));
  }
  return y;
}

DcTensor dc_add(DcTensor a, DcTensor b) { return binary(DK_ADD, a, b); }
DcTensor dc_sub(DcTensor a, DcTensor b) { return binary(DK_SUB, a, b); }
DcTensor dc_mul(DcTensor a, DcTensor b) { return binary(DK_MUL, a, b); }
DcTensor dc_div(DcTensor a, DcTensor b) { return binary(DK_DIV, a, b); }

static DcTensor unary(DkUnOp op, DcTensor a, float alpha, float beta) {
  DcTensor y = dc_new(a.shape, a.ndim);
  dk_unary(op, dc_data(a), dc_data(y), dc_numel(a), alpha, beta);
  return y;
}

DcTensor dc_affine(DcTensor a, float alpha, float beta) {
  return unary(DK_UN_AFFINE, a, alpha, beta);
}
DcTensor dc_gelu(DcTensor a) { return unary(DK_UN_GELU, a, 1.0f, 0.0f); }
DcTensor dc_relu(DcTensor a) { return unary(DK_UN_RELU, a, 1.0f, 0.0f); }

DcTensor dc_glu_axis1(DcTensor a) {
  int C = a.shape[1];
  if (C % 2 != 0) {
    dc_die("glu: odd channel count %d", C);
  }
  int s[DC_MAX_DIM];
  for (int i = 0; i < a.ndim; i++) {
    s[i] = a.shape[i];
  }
  s[1] = C / 2;
  DcTensor y = dc_new(s, a.ndim);
  int64_t inner = 1;
  for (int i = 2; i < a.ndim; i++) {
    inner *= a.shape[i];
  }
  dk_glu(dc_data(a), dc_data(y), a.shape[0], C / 2, inner);
  return y;
}
