#include "weights.h"

/* ------------------------------------------------- safetensors header parse */

typedef struct {
  char name[112];
  char dtype[8];
  int ndim;
  int shape[DC_MAX_DIM];
  uint64_t begin, end;
} StEntry;

typedef struct {
  const char *p;
  const char *end;
} Cursor;

static void skip_ws(Cursor *c) {
  while (c->p < c->end && (*c->p == ' ' || *c->p == '\n' || *c->p == '\r' || *c->p == '\t')) {
    c->p++;
  }
}

static int expect(Cursor *c, char ch) {
  skip_ws(c);
  if (c->p < c->end && *c->p == ch) {
    c->p++;
    return 1;
  }
  return 0;
}

static void parse_string(Cursor *c, char *out, size_t cap) {
  skip_ws(c);
  if (c->p >= c->end || *c->p != '"') {
    dc_die("safetensors header: expected string");
  }
  c->p++;
  size_t n = 0;
  while (c->p < c->end && *c->p != '"') {
    char ch = *c->p++;
    if (ch == '\\' && c->p < c->end) {
      ch = *c->p++;
    }
    if (n + 1 < cap) {
      out[n++] = ch;
    }
  }
  if (c->p >= c->end) {
    dc_die("safetensors header: unterminated string");
  }
  c->p++;
  out[n] = 0;
}

static uint64_t parse_uint(Cursor *c) {
  skip_ws(c);
  uint64_t v = 0;
  int any = 0;
  while (c->p < c->end && *c->p >= '0' && *c->p <= '9') {
    v = v * 10 + (uint64_t)(*c->p++ - '0');
    any = 1;
  }
  if (!any) {
    dc_die("safetensors header: expected integer");
  }
  return v;
}

static void skip_value(Cursor *c) {
  skip_ws(c);
  if (c->p >= c->end) {
    return;
  }
  if (*c->p == '"') {
    char tmp[4];
    parse_string(c, tmp, sizeof(tmp));
    return;
  }
  if (*c->p == '{' || *c->p == '[') {
    int depth = 0;
    int in_str = 0;
    for (; c->p < c->end; c->p++) {
      char ch = *c->p;
      if (in_str) {
        if (ch == '\\') {
          c->p++;
        } else if (ch == '"') {
          in_str = 0;
        }
        continue;
      }
      if (ch == '"') {
        in_str = 1;
      } else if (ch == '{' || ch == '[') {
        depth++;
      } else if (ch == '}' || ch == ']') {
        if (--depth == 0) {
          c->p++;
          return;
        }
      }
    }
    return;
  }
  while (c->p < c->end && *c->p != ',' && *c->p != '}' && *c->p != ']') {
    c->p++;
  }
}

static int parse_header(const char *json, size_t len, StEntry **out) {
  Cursor c = {json, json + len};
  if (!expect(&c, '{')) {
    dc_die("safetensors header: expected object");
  }
  int cap = 1024, n = 0;
  StEntry *items = (StEntry *)calloc((size_t)cap, sizeof(StEntry));
  while (1) {
    skip_ws(&c);
    if (expect(&c, '}')) {
      break;
    }
    char key[112];
    parse_string(&c, key, sizeof(key));
    if (!expect(&c, ':')) {
      dc_die("safetensors header: expected ':'");
    }
    if (strcmp(key, "__metadata__") == 0) {
      skip_value(&c);
    } else {
      if (n == cap) {
        cap *= 2;
        items = (StEntry *)realloc(items, (size_t)cap * sizeof(StEntry));
      }
      StEntry *e = &items[n++];
      memset(e, 0, sizeof(*e));
      snprintf(e->name, sizeof(e->name), "%s", key);
      if (!expect(&c, '{')) {
        dc_die("safetensors header: expected tensor object for %s", key);
      }
      while (!expect(&c, '}')) {
        char field[32];
        parse_string(&c, field, sizeof(field));
        expect(&c, ':');
        if (strcmp(field, "dtype") == 0) {
          parse_string(&c, e->dtype, sizeof(e->dtype));
        } else if (strcmp(field, "shape") == 0) {
          expect(&c, '[');
          while (!expect(&c, ']')) {
            if (e->ndim >= DC_MAX_DIM) {
              dc_die("tensor %s rank too high", key);
            }
            e->shape[e->ndim++] = (int)parse_uint(&c);
            expect(&c, ',');
          }
        } else if (strcmp(field, "data_offsets") == 0) {
          expect(&c, '[');
          e->begin = parse_uint(&c);
          expect(&c, ',');
          e->end = parse_uint(&c);
          expect(&c, ']');
        } else {
          skip_value(&c);
        }
        expect(&c, ',');
      }
    }
    expect(&c, ',');
  }
  *out = items;
  return n;
}

/* ------------------------------------------------------------ lookup table */

static uint32_t fnv1a(const char *s) {
  uint32_t h = 2166136261u;
  for (; *s; s++) {
    h = (h ^ (uint8_t)*s) * 16777619u;
  }
  return h;
}

static void table_build(DcWeights *w) {
  w->table_cap = 1;
  while (w->table_cap < w->n * 2) {
    w->table_cap <<= 1;
  }
  w->table = (int *)malloc((size_t)w->table_cap * sizeof(int));
  for (int i = 0; i < w->table_cap; i++) {
    w->table[i] = -1;
  }
  for (int i = 0; i < w->n; i++) {
    uint32_t h = fnv1a(w->items[i].name) & (uint32_t)(w->table_cap - 1);
    while (w->table[h] >= 0) {
      h = (h + 1) & (uint32_t)(w->table_cap - 1);
    }
    w->table[h] = i;
  }
}

static const DcWeight *table_find(const DcWeights *w, const char *key) {
  uint32_t h = fnv1a(key) & (uint32_t)(w->table_cap - 1);
  while (w->table[h] >= 0) {
    const DcWeight *it = &w->items[w->table[h]];
    if (strcmp(it->name, key) == 0) {
      return it;
    }
    h = (h + 1) & (uint32_t)(w->table_cap - 1);
  }
  return NULL;
}

/* ------------------------------------------------------------------- load */

static float load_scalar(const unsigned char *src, const char *dtype, size_t i) {
  if (strcmp(dtype, "F32") == 0) {
    float v;
    memcpy(&v, src + i * 4, 4);
    return v;
  }
  if (strcmp(dtype, "F16") == 0) {
    uint16_t h;
    memcpy(&h, src + i * 2, 2);
    return dc_f16_to_f32(h);
  }
  if (strcmp(dtype, "BF16") == 0) {
    uint16_t h;
    memcpy(&h, src + i * 2, 2);
    uint32_t x = (uint32_t)h << 16;
    float v;
    memcpy(&v, &x, 4);
    return v;
  }
  dc_die("unsupported safetensors dtype %s", dtype);
}

static size_t dtype_size(const char *dtype) {
  if (strcmp(dtype, "F32") == 0) {
    return 4;
  }
  if (strcmp(dtype, "F16") == 0 || strcmp(dtype, "BF16") == 0) {
    return 2;
  }
  dc_die("unsupported safetensors dtype %s", dtype);
}

static int is_gemm_weight(const StEntry *e) {
  if (e->ndim < 2) {
    return 0;
  }
  if (strstr(e->name, "embedding")) {
    return 0;
  }
  return 1;
}

void dc_weights_load(DcWeights *w, const char *safetensors_path, const char *json_path) {
  memset(w, 0, sizeof(*w));
  size_t flen = 0;
  unsigned char *file = (unsigned char *)dc_read_file(safetensors_path, &flen);
  if (!file) {
    dc_die("cannot read weights %s", safetensors_path);
  }
  if (flen < 8) {
    dc_die("weights file too small: %s", safetensors_path);
  }
  uint64_t hlen = 0;
  for (int i = 7; i >= 0; i--) {
    hlen = (hlen << 8) | file[i];
  }
  if (8 + hlen > flen) {
    dc_die("corrupt safetensors header in %s", safetensors_path);
  }
  StEntry *entries = NULL;
  int n = parse_header((const char *)file + 8, (size_t)hlen, &entries);
  const unsigned char *data = file + 8 + hlen;
  size_t data_len = flen - 8 - (size_t)hlen;

  w->items = (DcWeight *)calloc((size_t)n, sizeof(DcWeight));
  w->n = n;
  size_t total = 0;
  for (int i = 0; i < n; i++) {
    StEntry *e = &entries[i];
    DcWeight *it = &w->items[i];
    snprintf(it->name, sizeof(it->name), "%s", e->name);
    it->ndim = e->ndim;
    memcpy(it->shape, e->shape, sizeof(e->shape));
    size_t numel = 1;
    for (int d = 0; d < e->ndim; d++) {
      numel *= (size_t)e->shape[d];
    }
    if (e->end < e->begin || e->end > data_len ||
        (e->end - e->begin) != numel * dtype_size(e->dtype)) {
      dc_die("bad data offsets for %s", e->name);
    }
    size_t bytes;
    if (is_gemm_weight(e)) {
      if (strstr(e->name, "conv_tr")) {
        /* [O, KH(, KW), I] -> rows (o, k), reduction over I */
        it->kind = DC_W_GEMM_TR;
        int64_t I = e->shape[e->ndim - 1];
        it->rows = (int64_t)(numel / (size_t)I);
        it->kp = dc_round_up(I, 8);
      } else {
        /* [O, ...] -> rows o, reduction over the flattened rest */
        it->kind = DC_W_GEMM;
        it->rows = e->shape[0];
        it->kp = dc_round_up((int64_t)(numel / (size_t)e->shape[0]), 8);
      }
      bytes = (size_t)(it->rows * it->kp) * sizeof(dc_half);
    } else {
      it->kind = DC_W_F32;
      bytes = numel * sizeof(float);
    }
    total += (bytes + 255) & ~(size_t)255;
  }

  unsigned char *staging = (unsigned char *)calloc(1, total ? total : 1);
  if (!staging) {
    dc_die("oom weight staging (%zu bytes)", total);
  }
  size_t off = 0;
  size_t *offsets = (size_t *)malloc((size_t)n * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    StEntry *e = &entries[i];
    DcWeight *it = &w->items[i];
    const unsigned char *src = data + e->begin;
    offsets[i] = off;
    size_t bytes;
    if (it->kind == DC_W_F32) {
      size_t numel = (e->end - e->begin) / dtype_size(e->dtype);
      float *dst = (float *)(staging + off);
      for (size_t j = 0; j < numel; j++) {
        dst[j] = load_scalar(src, e->dtype, j);
      }
      bytes = numel * sizeof(float);
    } else {
      int64_t K = it->kind == DC_W_GEMM_TR ? e->shape[e->ndim - 1]
                                           : (int64_t)((e->end - e->begin) /
                                                       dtype_size(e->dtype) / (size_t)it->rows);
      dc_half *dst = (dc_half *)(staging + off);
      for (int64_t r = 0; r < it->rows; r++) {
        for (int64_t k = 0; k < K; k++) {
          dst[r * it->kp + k] = dk_pack_half(load_scalar(src, e->dtype, (size_t)(r * K + k)), 1);
        }
      }
      bytes = (size_t)(it->rows * it->kp) * sizeof(dc_half);
    }
    off += (bytes + 255) & ~(size_t)255;
  }
  free(file);

  w->block = dk_raw_malloc(total ? total : 1);
  if (!w->block) {
    dc_die("out of device memory uploading weights (%.1f MB)", (double)total / 1048576.0);
  }
  w->block_bytes = total;
  dk_memcpy_h2d(w->block, staging, total);
  free(staging);
  for (int i = 0; i < n; i++) {
    DcWeight *it = &w->items[i];
    it->dptr = (unsigned char *)w->block + offsets[i];
    if (it->kind == DC_W_F32) {
      it->t = dc_wrap_external((float *)it->dptr, it->shape, it->ndim);
    }
  }
  free(offsets);
  free(entries);
  table_build(w);
  dc_config_load_json(&w->cfg, json_path);
}

void dc_weights_free(DcWeights *w) {
  if (!w) {
    return;
  }
  for (int i = 0; i < w->n; i++) {
    if (w->items[i].kind == DC_W_F32) {
      dc_free(w->items[i].t);
    }
  }
  if (w->block) {
    dk_raw_free(w->block);
  }
  free(w->items);
  free(w->table);
  memset(w, 0, sizeof(*w));
}

static void format_key(char *buf, size_t cap, const char *fmt, va_list ap) {
  if (vsnprintf(buf, cap, fmt, ap) >= (int)cap) {
    dc_die("weight key overflow");
  }
}

const DcWeight *dc_wfind(const DcWeights *w, const char *fmt, ...) {
  char key[256];
  va_list ap;
  va_start(ap, fmt);
  format_key(key, sizeof(key), fmt, ap);
  va_end(ap);
  return table_find(w, key);
}

const DcWeight *dc_wget(const DcWeights *w, const char *fmt, ...) {
  char key[256];
  va_list ap;
  va_start(ap, fmt);
  format_key(key, sizeof(key), fmt, ap);
  va_end(ap);
  const DcWeight *it = table_find(w, key);
  if (!it) {
    dc_die("missing weight '%s'", key);
  }
  return it;
}

DcTensor dc_wt(const DcWeights *w, const char *fmt, ...) {
  char key[256];
  va_list ap;
  va_start(ap, fmt);
  format_key(key, sizeof(key), fmt, ap);
  va_end(ap);
  const DcWeight *it = table_find(w, key);
  if (!it) {
    dc_die("missing weight '%s'", key);
  }
  if (it->kind != DC_W_F32) {
    dc_die("weight '%s' is a packed GEMM operand, not an fp32 tensor", key);
  }
  return dc_ref(it->t);
}

bool dc_w_has(const DcWeights *w, const char *fmt, ...) {
  char key[256];
  va_list ap;
  va_start(ap, fmt);
  format_key(key, sizeof(key), fmt, ap);
  va_end(ap);
  return table_find(w, key) != NULL;
}

void dc_weight_from_host(DcWeight *it, const char *name, const float *data, const int *shape,
                         int ndim) {
  memset(it, 0, sizeof(*it));
  StEntry e;
  memset(&e, 0, sizeof(e));
  snprintf(e.name, sizeof(e.name), "%s", name);
  e.ndim = ndim;
  memcpy(e.shape, shape, (size_t)ndim * sizeof(int));
  snprintf(it->name, sizeof(it->name), "%s", name);
  it->ndim = ndim;
  memcpy(it->shape, shape, (size_t)ndim * sizeof(int));
  size_t numel = 1;
  for (int d = 0; d < ndim; d++) {
    numel *= (size_t)shape[d];
  }
  if (!is_gemm_weight(&e)) {
    it->kind = DC_W_F32;
    it->t = dc_from_host(data, shape, ndim);
    it->dptr = dc_data(it->t);
    return;
  }
  int64_t K;
  if (strstr(name, "conv_tr")) {
    it->kind = DC_W_GEMM_TR;
    K = shape[ndim - 1];
    it->rows = (int64_t)(numel / (size_t)K);
  } else {
    it->kind = DC_W_GEMM;
    it->rows = shape[0];
    K = (int64_t)(numel / (size_t)shape[0]);
  }
  it->kp = dc_round_up(K, 8);
  size_t bytes = (size_t)(it->rows * it->kp) * sizeof(dc_half);
  dc_half *h = (dc_half *)calloc(1, bytes);
  for (int64_t r = 0; r < it->rows; r++) {
    for (int64_t k = 0; k < K; k++) {
      h[r * it->kp + k] = dk_pack_half(data[r * K + k], 1);
    }
  }
  it->dptr = dk_raw_malloc(bytes);
  if (!it->dptr) {
    dc_die("oom packing weight %s", name);
  }
  dk_memcpy_h2d(it->dptr, h, bytes);
  free(h);
}

void dc_weight_release(DcWeight *it) {
  if (it->kind == DC_W_F32) {
    dc_free(it->t);
  } else if (it->dptr) {
    dk_raw_free(it->dptr);
  }
  memset(it, 0, sizeof(*it));
}

/* ------------------------------------------------------- generic reader */

struct DcStFile {
  unsigned char *file;
  StEntry *entries;
  int n;
  const unsigned char *data;
  size_t data_len;
};

DcStFile *dc_st_open(const char *path) {
  size_t flen = 0;
  unsigned char *file = (unsigned char *)dc_read_file(path, &flen);
  if (!file) {
    return NULL;
  }
  uint64_t hlen = 0;
  for (int i = 7; i >= 0 && flen >= 8; i--) {
    hlen = (hlen << 8) | file[i];
  }
  if (flen < 8 || 8 + hlen > flen) {
    dc_die("corrupt safetensors file %s", path);
  }
  DcStFile *f = (DcStFile *)calloc(1, sizeof(DcStFile));
  f->file = file;
  f->n = parse_header((const char *)file + 8, (size_t)hlen, &f->entries);
  f->data = file + 8 + hlen;
  f->data_len = flen - 8 - (size_t)hlen;
  return f;
}

DcTensor dc_st_get(DcStFile *f, const char *name) {
  for (int i = 0; i < f->n; i++) {
    StEntry *e = &f->entries[i];
    if (strcmp(e->name, name) != 0) {
      continue;
    }
    size_t numel = (e->end - e->begin) / dtype_size(e->dtype);
    if (e->end > f->data_len) {
      dc_die("bad data offsets for %s", name);
    }
    float *tmp = (float *)malloc((numel ? numel : 1) * sizeof(float));
    for (size_t j = 0; j < numel; j++) {
      tmp[j] = load_scalar(f->data + e->begin, e->dtype, j);
    }
    int shape1[1] = {1};
    DcTensor t = e->ndim ? dc_from_host(tmp, e->shape, e->ndim) : dc_from_host(tmp, shape1, 1);
    free(tmp);
    return t;
  }
  return dc_empty();
}

void dc_st_close(DcStFile *f) {
  if (!f) {
    return;
  }
  free(f->file);
  free(f->entries);
  free(f);
}
