#include "htdemucs.h"

static void fill_spec(DcLayerSpec *sp, const char *prefix, int chin, int chout, int kernel,
                      int stride, int freq, int empty, int last, int pad, int rewrite, int dconv,
                      int norm, int context, int context_freq, const DcConfig *cfg) {
  memset(sp, 0, sizeof(*sp));
  snprintf(sp->prefix, sizeof(sp->prefix), "%s", prefix);
  sp->chin = chin;
  sp->chout = chout;
  sp->kernel = kernel;
  sp->stride = stride;
  sp->pad = pad ? kernel / 4 : 0;
  sp->freq = freq;
  sp->empty = empty;
  sp->last = last;
  sp->rewrite = rewrite && !empty;
  sp->dconv = dconv && !empty;
  sp->norm = norm;
  sp->fused_norm1 = norm && !empty;
  sp->fused_norm2 = norm && rewrite && !empty;
  sp->context = context;
  sp->context_freq = context_freq;
  sp->norm_groups = cfg->norm_groups;
  sp->dconv_depth = cfg->dconv_depth;
  sp->dconv_comp = cfg->dconv_comp;
  sp->dconv_init = cfg->dconv_init;
}

void dc_model_init(DcModel *m, DcWeights *w) {
  memset(m, 0, sizeof(*m));
  m->w = w;
  const DcConfig *cfg = &w->cfg;
  if (cfg->t_layers > 0 && cfg->bottom_channels) {
    dc_die("bottom_channels != 0 is not supported");
  }
  if (cfg->depth > DC_MAX_DEPTH) {
    dc_die("model depth %d exceeds %d", cfg->depth, DC_MAX_DEPTH);
  }
  m->stft = dc_stft_new(cfg->nfft, cfg->hop_length);

  int chin = cfg->audio_channels;
  int chin_z = chin * (cfg->cac ? 2 : 1);
  int chout = cfg->channels_time > 0 ? cfg->channels_time : cfg->channels;
  int chout_z = cfg->channels;
  int freqs = cfg->nfft / 2;
  DcLayerSpec dec_tmp[DC_MAX_DEPTH];
  DcLayerSpec tdec_tmp[DC_MAX_DEPTH];
  int n_dec = 0, n_tdec = 0;

  for (int index = 0; index < cfg->depth; index++) {
    int norm = index >= cfg->norm_starts;
    int freq = freqs > 1;
    int stri = cfg->stride;
    int ker = cfg->kernel_size;
    int pad = 1;
    int last_freq = 0;
    if (!freq) {
      ker = cfg->time_stride * 2;
      stri = cfg->time_stride;
    }
    if (freq && freqs <= cfg->kernel_size) {
      ker = freqs;
      pad = 0;
      last_freq = 1;
    }
    char name[64];
    snprintf(name, sizeof(name), "encoder.%d", m->n_enc);
    fill_spec(&m->encoder[m->n_enc++], name, chin_z, chout_z, ker, stri, freq, 0, 0, pad,
              cfg->rewrite, cfg->dconv_mode & 1, norm, cfg->context_enc, 1, cfg);
    if (freq) {
      snprintf(name, sizeof(name), "tencoder.%d", m->n_tenc);
      fill_spec(&m->tencoder[m->n_tenc++], name, chin, chout, cfg->kernel_size, cfg->stride, 0,
                last_freq, 0, 1, cfg->rewrite, cfg->dconv_mode & 1, norm, cfg->context_enc, 1,
                cfg);
    }
    fill_spec(&dec_tmp[n_dec++], "decoder", chout_z, chin_z, ker, stri, freq, 0, index == 0,
              pad, cfg->rewrite, cfg->dconv_mode & 2, norm, cfg->context, 1, cfg);
    if (freq) {
      fill_spec(&tdec_tmp[n_tdec++], "tdecoder", chout, chin, cfg->kernel_size, cfg->stride, 0,
                last_freq, index == 0, 1, cfg->rewrite, cfg->dconv_mode & 2, norm,
                cfg->context, 1, cfg);
    }
    if (index == 0) {
      chin = cfg->audio_channels * cfg->n_sources;
      chin_z = chin * (cfg->cac ? 2 : 1);
    }
    chin = chout;
    chin_z = chout_z;
    chout = cfg->growth * chout;
    chout_z = cfg->growth * chout_z;
    if (freq) {
      freqs = freqs <= cfg->kernel_size ? 1 : freqs / cfg->stride;
    }
  }
  /* decoder.insert(0): construction order is reversed in the weight names */
  m->n_dec = n_dec;
  m->n_tdec = n_tdec;
  for (int i = 0; i < n_dec; i++) {
    m->decoder[i] = dec_tmp[n_dec - 1 - i];
    snprintf(m->decoder[i].prefix, sizeof(m->decoder[i].prefix), "decoder.%d", i);
  }
  for (int i = 0; i < n_tdec; i++) {
    m->tdecoder[i] = tdec_tmp[n_tdec - 1 - i];
    snprintf(m->tdecoder[i].prefix, sizeof(m->tdecoder[i].prefix), "tdecoder.%d", i);
  }
}

void dc_model_free(DcModel *m) {
  if (m) {
    dc_stft_free(&m->stft);
  }
}

DcTensor dc_add_freq_emb(DcModel *m, DcTensor x) {
  const DcConfig *cfg = &m->w->cfg;
  if (cfg->freq_emb == 0.0f || !dc_w_has(m->w, "freq_emb.embedding.weight")) {
    return dc_ref(x);
  }
  int Fr = x.shape[2];
  DcTensor wemb = dc_wt(m->w, "freq_emb.embedding.weight"); /* [n_freq, C] */
  DcTensor emb = dc_slice_axis(wemb, 0, 0, Fr);
  dc_free(wemb);
  int tax[2] = {1, 0};
  DcTensor embt = dc_transpose(emb, tax, 2); /* [C, Fr] */
  dc_free(emb);
  DcTensor scaled = dc_affine(embt, cfg->freq_emb * cfg->emb_scale, 0.0f);
  dc_free(embt);
  int s4[4] = {1, scaled.shape[0], Fr, 1};
  DcTensor e4 = dc_reshape(scaled, s4, 4);
  dc_free(scaled);
  DcTensor y = dc_add(x, e4);
  dc_free(e4);
  return y;
}

DcTensor dc_htdemucs_forward(DcModel *m, DcTensor mix_in) {
  const DcConfig *cfg = &m->w->cfg;
  DcTensor mix = dc_ref(mix_in);
  int length = mix.shape[mix.ndim - 1];
  int length_pre_pad = 0;
  int training_length = cfg->valid_length;
  if (cfg->use_train_segment && length < training_length) {
    length_pre_pad = length;
    DcTensor p = dc_pad1d(mix, 0, training_length - length, 0);
    dc_free(mix);
    mix = p;
  }
  dk_range_push("spec");
  DcTensor mag = dc_htdemucs_spec_cac(&m->stft, mix);
  dk_range_pop();
  int Fq = mag.shape[2], T = mag.shape[3];
  DcTensor mean, var;
  dc_batch_stats(mag, &mean, &var);
  DcTensor x = dc_batch_normalize(mag, mean, var, 1e-5f, 0);
  dc_free(mag);
  DcTensor meant, vart;
  dc_batch_stats(mix, &meant, &vart);
  DcTensor xt = dc_batch_normalize(mix, meant, vart, 1e-5f, 0);
  dc_free(mix);

  DcTensor saved[DC_MAX_DEPTH], saved_t[DC_MAX_DEPTH];
  int lengths[DC_MAX_DEPTH], lengths_t[DC_MAX_DEPTH];
  int n_saved = 0, n_saved_t = 0, n_len = 0, n_lent = 0;

  dk_range_push("encoders");
  for (int idx = 0; idx < m->n_enc; idx++) {
    lengths[n_len++] = x.shape[x.ndim - 1];
    DcTensor inject = dc_empty();
    if (idx < m->n_tenc) {
      lengths_t[n_lent++] = xt.shape[xt.ndim - 1];
      DcTensor xt2 = dc_henc(xt, dc_empty(), m->w, &m->tencoder[idx]);
      dc_free(xt);
      xt = xt2;
      if (!m->tencoder[idx].empty) {
        saved_t[n_saved_t++] = dc_ref(xt);
      } else {
        inject = xt;
      }
    }
    DcTensor x2 = dc_henc(x, inject, m->w, &m->encoder[idx]);
    dc_free(x);
    x = x2;
    if (idx == 0) {
      DcTensor x3 = dc_add_freq_emb(m, x);
      dc_free(x);
      x = x3;
    }
    saved[n_saved++] = dc_ref(x);
  }
  dk_range_pop();

  if (cfg->t_layers > 0) {
    dk_range_push("crosstransformer");
    dc_crosstransformer(&x, &xt, m->w, cfg);
    dk_range_pop();
  }

  dk_range_push("decoders");
  int offset = m->n_enc - m->n_tdec;
  for (int idx = 0; idx < m->n_dec; idx++) {
    if (n_saved <= 0 || n_len <= 0) {
      dc_die("decoder skip underflow");
    }
    DcTensor skip = saved[--n_saved];
    int len = lengths[--n_len];
    DcTensor pre = dc_empty();
    DcTensor x2 = dc_hdec(x, skip, len, m->w, &m->decoder[idx], &pre);
    dc_free(x);
    dc_free(skip);
    x = x2;
    if (idx >= offset) {
      DcLayerSpec *tdec = &m->tdecoder[idx - offset];
      if (n_lent <= 0) {
        dc_die("tdecoder length underflow");
      }
      int length_t = lengths_t[--n_lent];
      DcTensor xt2;
      if (tdec->empty) {
        DcTensor pre0 = dc_slice_axis(pre, 2, 0, 1);
        DcTensor pre_s = dc_squeeze_axis(pre0, 2);
        dc_free(pre0);
        xt2 = dc_hdec(pre_s, dc_empty(), length_t, m->w, tdec, NULL);
        dc_free(pre_s);
      } else {
        if (n_saved_t <= 0) {
          dc_die("tdecoder skip underflow");
        }
        DcTensor skip_t = saved_t[--n_saved_t];
        xt2 = dc_hdec(xt, skip_t, length_t, m->w, tdec, NULL);
        dc_free(skip_t);
      }
      dc_free(xt);
      xt = xt2;
    }
    dc_free(pre);
  }
  dk_range_pop();
  if (n_saved || n_saved_t || n_lent) {
    dc_die("skip connections not fully consumed (%d %d %d)", n_saved, n_saved_t, n_lent);
  }

  int B = x.shape[0];
  int S = cfg->n_sources;
  int Ctot = x.shape[1];
  if (Ctot % S != 0) {
    dc_die("decoder channels %d not divisible by sources %d", Ctot, S);
  }
  int r5[5] = {B, S, Ctot / S, Fq, T};
  DcTensor xr = dc_reshape(x, r5, 5);
  dc_free(x);
  DcTensor xden = dc_batch_normalize(xr, mean, var, 1e-5f, 1);
  dc_free(xr);
  dk_range_push("ispec");
  DcTensor xi = dc_htdemucs_ispec_cac(&m->stft, xden, training_length);
  dk_range_pop();
  dc_free(xden);

  int actual = xt.shape[xt.ndim - 1];
  int r4[4] = {B, S, xt.shape[1] / S, actual};
  DcTensor xtr = dc_reshape(xt, r4, 4);
  dc_free(xt);
  DcTensor xtd = dc_batch_normalize(xtr, meant, vart, 1e-5f, 1);
  dc_free(xtr);
  DcTensor xtrim = dc_center_trim(xi, xtd.shape[3]);
  dc_free(xi);
  DcTensor y = dc_add(xtd, xtrim);
  dc_free(xtd);
  dc_free(xtrim);
  int keep = length_pre_pad ? length_pre_pad : training_length;
  if (y.shape[3] > keep) {
    DcTensor y2 = dc_slice_axis(y, 3, 0, keep);
    dc_free(y);
    y = y2;
  }
  dc_free(mean);
  dc_free(var);
  dc_free(meant);
  dc_free(vart);
  return y;
}
