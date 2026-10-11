#include "apply.h"

#include <math.h>

/* CPython 3 MT19937 — matches random.Random(seed).randint. */
#define DC_MT_N 624
#define DC_MT_M 397
#define DC_MT_MATRIX_A 0x9908b0dfUL
#define DC_MT_UPPER 0x80000000UL
#define DC_MT_LOWER 0x7fffffffUL

typedef struct {
  uint32_t mt[DC_MT_N];
  int mti;
} DcPyRand;

static void mt_init_genrand(DcPyRand *r, uint32_t s) {
  r->mt[0] = s;
  for (int i = 1; i < DC_MT_N; i++) {
    r->mt[i] = (1812433253UL * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + (uint32_t)i);
  }
  r->mti = DC_MT_N;
}

static void mt_init_by_array(DcPyRand *r, const uint32_t *init_key, int key_length) {
  mt_init_genrand(r, 19650218UL);
  int i = 1;
  int j = 0;
  int k = DC_MT_N > key_length ? DC_MT_N : key_length;
  for (; k; k--) {
    r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1664525UL)) + init_key[j] +
               (uint32_t)j;
    i++;
    j++;
    if (i >= DC_MT_N) {
      r->mt[0] = r->mt[DC_MT_N - 1];
      i = 1;
    }
    if (j >= key_length) {
      j = 0;
    }
  }
  for (k = DC_MT_N - 1; k; k--) {
    r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1566083941UL)) - (uint32_t)i;
    i++;
    if (i >= DC_MT_N) {
      r->mt[0] = r->mt[DC_MT_N - 1];
      i = 1;
    }
  }
  r->mt[0] = 0x80000000UL;
}

static void py_seed(DcPyRand *r, uint64_t n) {
  uint32_t key[2];
  int nkey = 0;
  if (n == 0) {
    key[nkey++] = 0;
  } else {
    while (n && nkey < 2) {
      key[nkey++] = (uint32_t)(n & 0xffffffffu);
      n >>= 32;
    }
  }
  mt_init_by_array(r, key, nkey);
}

static uint32_t mt_genrand_int32(DcPyRand *r) {
  uint32_t y;
  static const uint32_t mag01[2] = {0x0UL, DC_MT_MATRIX_A};
  if (r->mti >= DC_MT_N) {
    int kk;
    for (kk = 0; kk < DC_MT_N - DC_MT_M; kk++) {
      y = (r->mt[kk] & DC_MT_UPPER) | (r->mt[kk + 1] & DC_MT_LOWER);
      r->mt[kk] = r->mt[kk + DC_MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    for (; kk < DC_MT_N - 1; kk++) {
      y = (r->mt[kk] & DC_MT_UPPER) | (r->mt[kk + 1] & DC_MT_LOWER);
      r->mt[kk] = r->mt[kk + (DC_MT_M - DC_MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    y = (r->mt[DC_MT_N - 1] & DC_MT_UPPER) | (r->mt[0] & DC_MT_LOWER);
    r->mt[DC_MT_N - 1] = r->mt[DC_MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
    r->mti = 0;
  }
  y = r->mt[r->mti++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);
  return y;
}

static int py_randint(DcPyRand *r, int lo, int hi) {
  if (hi < lo) {
    dc_die("randint range");
  }
  uint32_t n = (uint32_t)(hi - lo + 1);
  int k = 0;
  for (uint32_t t = n; t; t >>= 1) {
    k++;
  }
  uint32_t v;
  do {
    v = k == 0 ? 0 : (mt_genrand_int32(r) >> (32 - k));
  } while (v >= n);
  return lo + (int)v;
}

static void note_progress(const DcApplyOpts *opts, int *done, int total) {
  (*done)++;
  if (opts->on_progress) {
    opts->on_progress(*done, total, opts->progress_user);
  }
}

static float *ola_weight(int segment_length, float transition_power) {
  int half = segment_length / 2;
  float *w = (float *)malloc((size_t)segment_length * sizeof(float));
  float wmax = 0.0f;
  for (int i = 0; i < half; i++) {
    w[i] = (float)(i + 1);
  }
  for (int i = half; i < segment_length; i++) {
    w[i] = (float)(segment_length - i);
  }
  for (int i = 0; i < segment_length; i++) {
    wmax = w[i] > wmax ? w[i] : wmax;
  }
  for (int i = 0; i < segment_length; i++) {
    w[i] = powf(w[i] / wmax, transition_power);
  }
  return w;
}

static DcTensor padded_chunk(DcTensor mix, int offset, int chunk_len, int target_len) {
  int total = mix.shape[mix.ndim - 1];
  int delta = target_len - chunk_len;
  if (delta < 0) {
    dc_die("padded_chunk target < chunk");
  }
  int start = offset - delta / 2;
  int end = start + target_len;
  int correct_start = start < 0 ? 0 : start;
  int correct_end = end > total ? total : end;
  int pad_left = correct_start - start;
  int pad_right = end - correct_end;
  DcTensor sl = dc_slice_axis(mix, mix.ndim - 1, correct_start, correct_end);
  if (pad_left || pad_right) {
    DcTensor p = dc_pad1d(sl, pad_left, pad_right, 0);
    dc_free(sl);
    sl = p;
  }
  return sl;
}

/* out[..., off:off+n] += center_trim(y, n) * weight[0:n] */
static void accumulate(DcTensor out, DcTensor y, DcTensor wdev, int off, int n) {
  DcTensor ct = dc_center_trim(y, n);
  DcTensor wsl = dc_slice_axis(wdev, 0, 0, n);
  int ws[4] = {1, 1, 1, n};
  DcTensor wr = dc_reshape(wsl, ws, 4);
  DcTensor upd = dc_mul(ct, wr);
  int starts[4] = {0, 0, 0, off};
  dc_slice_add_inplace(out, upd, starts);
  dc_free(ct);
  dc_free(wsl);
  dc_free(wr);
  dc_free(upd);
}

#define DC_MAX_BATCH 32

DcTensor dc_apply_model(DcModel *m, DcTensor mix, const DcApplyOpts *opts) {
  int batch = mix.shape[0];
  int channels = mix.shape[1];
  int length = mix.shape[2];
  int sources = m->w->cfg.n_sources;

  if (opts->shifts > 0) {
    int max_shift = (int)(0.5f * (float)m->w->cfg.samplerate);
    DcTensor padded = padded_chunk(mix, 0, length, length + 2 * max_shift);
    DcTensor acc = dc_empty();
    DcPyRand pr;
    py_seed(&pr, opts->use_seed ? (uint64_t)opts->seed : 1u);
    for (int i = 0; i < opts->shifts; i++) {
      int offset = py_randint(&pr, 0, max_shift);
      DcApplyOpts inner = *opts;
      inner.shifts = 0;
      inner.on_progress = (i == 0) ? opts->on_progress : NULL;
      DcTensor shifted =
          padded_chunk(padded, offset, length + max_shift - offset, length + max_shift - offset);
      DcTensor out = dc_apply_model(m, shifted, &inner);
      DcTensor trimmed = dc_slice_axis(out, 3, max_shift - offset, out.shape[3]);
      if (!dc_valid(acc)) {
        acc = trimmed;
      } else {
        DcTensor a2 = dc_add(acc, trimmed);
        dc_free(acc);
        dc_free(trimmed);
        acc = a2;
      }
      dc_free(shifted);
      dc_free(out);
    }
    DcTensor avg = dc_affine(acc, 1.0f / (float)opts->shifts, 0.0f);
    dc_free(padded);
    dc_free(acc);
    return avg;
  }

  if (!opts->split) {
    DcTensor padded = padded_chunk(mix, 0, length, m->w->cfg.valid_length);
    DcTensor out = dc_htdemucs_forward(m, padded);
    DcTensor trim = dc_center_trim(out, length);
    dc_free(padded);
    dc_free(out);
    if (opts->on_progress) {
      opts->on_progress(1, 1, opts->progress_user);
    }
    return trim;
  }

  float segment = opts->segment > 0 ? opts->segment : m->w->cfg.segment;
  int segment_length = (int)((float)m->w->cfg.samplerate * segment);
  int stride = (int)((1.0f - opts->overlap) * (float)segment_length);
  if (stride <= 0) {
    dc_die("invalid overlap %f", opts->overlap);
  }
  int n_off = 0;
  for (int off = 0; off < length; off += stride) {
    n_off++;
  }
  int *offsets = (int *)malloc((size_t)(n_off > 0 ? n_off : 1) * sizeof(int));
  n_off = 0;
  for (int off = 0; off < length; off += stride) {
    offsets[n_off++] = off;
  }

  float *weight = ola_weight(segment_length, opts->transition_power);
  int wsh[1] = {segment_length};
  DcTensor wdev = dc_from_host(weight, wsh, 1);
  double *sum_weight = (double *)calloc((size_t)length, sizeof(double));
  int oshape[4] = {batch, sources, channels, length};
  DcTensor out = dc_zeros(oshape, 4);
  int std_valid = m->w->cfg.valid_length;
  int segments_done = 0;

  int max_b = opts->batch_size;
  max_b = max_b > DC_MAX_BATCH ? DC_MAX_BATCH : (max_b < 1 ? 1 : max_b);
  DcTensor batch_in[DC_MAX_BATCH];
  int batch_idx[DC_MAX_BATCH];
  int nb = 0;

  for (int i = 0; i <= n_off; i++) {
    int is_full = i < n_off && (length - offsets[i]) >= segment_length;
    int flush = !is_full;
    if (is_full) {
      batch_in[nb] = padded_chunk(mix, offsets[i], segment_length, std_valid);
      batch_idx[nb] = i;
      nb++;
      flush = nb >= max_b;
    }
    if (flush && nb > 0) {
      DcTensor flat = nb == 1 ? dc_ref(batch_in[0]) : dc_concat(batch_in, nb, 0);
      for (int k = 0; k < nb; k++) {
        dc_free(batch_in[k]);
      }
      DcTensor yflat = dc_htdemucs_forward(m, flat);
      dc_free(flat);
      for (int k = 0; k < nb; k++) {
        DcTensor yk = dc_slice_axis(yflat, 0, k * batch, (k + 1) * batch);
        int off = offsets[batch_idx[k]];
        accumulate(out, yk, wdev, off, segment_length);
        dc_free(yk);
        for (int t = 0; t < segment_length; t++) {
          sum_weight[off + t] += weight[t];
        }
        note_progress(opts, &segments_done, n_off);
      }
      dc_free(yflat);
      nb = 0;
    }
    if (i < n_off && !is_full) {
      int this_len = length - offsets[i];
      int valid = this_len < std_valid ? std_valid : this_len;
      DcTensor ch = padded_chunk(mix, offsets[i], this_len, valid);
      DcTensor y = dc_htdemucs_forward(m, ch);
      dc_free(ch);
      accumulate(out, y, wdev, offsets[i], this_len);
      dc_free(y);
      for (int t = 0; t < this_len; t++) {
        sum_weight[offsets[i] + t] += weight[t];
      }
      note_progress(opts, &segments_done, n_off);
    }
  }

  float *inv = (float *)malloc((size_t)length * sizeof(float));
  for (int t = 0; t < length; t++) {
    inv[t] = sum_weight[t] > 0.0 ? (float)(1.0 / sum_weight[t]) : 0.0f;
  }
  int ish[4] = {1, 1, 1, length};
  DcTensor invd = dc_from_host(inv, ish, 4);
  DcTensor y = dc_mul(out, invd);
  dc_free(invd);
  dc_free(out);
  dc_free(wdev);
  free(inv);
  free(sum_weight);
  free(weight);
  free(offsets);
  return y;
}
