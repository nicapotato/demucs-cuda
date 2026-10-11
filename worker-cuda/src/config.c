#include "config.h"

static char *load_file(const char *path, size_t *len_out) {
  char *buf = (char *)dc_read_file(path, len_out);
  if (!buf) {
    dc_die("cannot read config %s", path);
  }
  return buf;
}

static const char *find_key(const char *json, const char *key) {
  char pat[128];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char *p = strstr(json, pat);
  if (!p) {
    return NULL;
  }
  p = strchr(p + strlen(pat), ':');
  if (!p) {
    return NULL;
  }
  p++;
  while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') {
    p++;
  }
  return p;
}

static int parse_int_key(const char *json, const char *key, int fallback) {
  const char *p = find_key(json, key);
  if (!p) {
    return fallback;
  }
  if (strncmp(p, "true", 4) == 0) {
    return 1;
  }
  if (strncmp(p, "false", 5) == 0) {
    return 0;
  }
  if (strncmp(p, "null", 4) == 0) {
    return fallback;
  }
  return (int)strtol(p, NULL, 10);
}

static float parse_float_key(const char *json, const char *key, float fallback) {
  const char *p = find_key(json, key);
  if (!p || strncmp(p, "null", 4) == 0) {
    return fallback;
  }
  return strtof(p, NULL);
}

static void parse_string_key(const char *json, const char *key, char *out,
                             size_t cap, const char *fallback) {
  const char *p = find_key(json, key);
  if (!p || *p != '"') {
    snprintf(out, cap, "%s", fallback);
    return;
  }
  p++;
  size_t i = 0;
  while (*p && *p != '"' && i + 1 < cap) {
    out[i++] = *p++;
  }
  out[i] = 0;
}

static int parse_string_array(const char *json, const char *key, char out[][32],
                              int maxn) {
  const char *p = find_key(json, key);
  if (!p || *p != '[') {
    return 0;
  }
  p++;
  int n = 0;
  while (*p && *p != ']' && n < maxn) {
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == ',' || *p == '\r') {
      p++;
    }
    if (*p == ']') {
      break;
    }
    if (*p != '"') {
      break;
    }
    p++;
    int i = 0;
    while (*p && *p != '"' && i + 1 < 32) {
      out[n][i++] = *p++;
    }
    out[n][i] = 0;
    if (*p == '"') {
      p++;
    }
    n++;
  }
  return n;
}

static const char *kwargs_blob(const char *json) {
  const char *p = find_key(json, "kwargs");
  return p;
}

void dc_config_defaults(DcConfig *c) {
  memset(c, 0, sizeof(*c));
  snprintf(c->model, sizeof(c->model), "htdemucs_6s");
  const char *src[] = {"drums", "bass", "other", "vocals", "guitar", "piano"};
  c->n_sources = 6;
  for (int i = 0; i < 6; i++) {
    snprintf(c->sources[i], sizeof(c->sources[i]), "%s", src[i]);
  }
  c->samplerate = 44100;
  c->segment = 7.8f;
  c->nfft = 4096;
  c->hop_length = 1024;
  c->audio_channels = 2;
  c->channels = 48;
  c->channels_time = 0;
  c->growth = 2;
  c->depth = 4;
  c->kernel_size = 8;
  c->stride = 4;
  c->time_stride = 2;
  c->context = 1;
  c->context_enc = 0;
  c->norm_groups = 4;
  c->norm_starts = 4;
  c->dconv_mode = 1;
  c->dconv_depth = 2;
  c->dconv_comp = 8;
  c->dconv_init = 1e-3f;
  c->rewrite = 1;
  c->cac = 1;
  c->freq_emb = 0.2f;
  c->emb_scale = 10.0f;
  c->emb_smooth = 1;
  c->t_layers = 5;
  c->t_heads = 8;
  c->t_hidden_scale = 4.0f;
  snprintf(c->t_emb, sizeof(c->t_emb), "sin");
  c->t_max_period = 10000.0f;
  c->t_weight_pos_embed = 1.0f;
  c->t_norm_in = 1;
  c->t_norm_first = 1;
  c->t_norm_out = 1;
  c->t_layer_scale = 1;
  c->t_gelu = 1;
  c->t_cross_first = 0;
  c->bottom_channels = 0;
  c->use_train_segment = 1;
  c->wiener_iters = 0;
  c->valid_length = (int)(c->segment * (float)c->samplerate);
}

void dc_config_load_json(DcConfig *c, const char *path) {
  dc_config_defaults(c);
  char *json = load_file(path, NULL);
  parse_string_key(json, "model", c->model, sizeof(c->model), c->model);
  int ns = parse_string_array(json, "sources", c->sources, DC_MAX_SOURCES);
  if (ns > 0) {
    c->n_sources = ns;
  }
  c->samplerate = parse_int_key(json, "samplerate", c->samplerate);
  c->segment = parse_float_key(json, "segment", c->segment);
  c->nfft = parse_int_key(json, "nfft", c->nfft);
  c->hop_length = parse_int_key(json, "hop_length", c->hop_length);
  c->audio_channels = parse_int_key(json, "audio_channels", c->audio_channels);
  c->channels = parse_int_key(json, "channels", c->channels);
  c->depth = parse_int_key(json, "depth", c->depth);
  c->kernel_size = parse_int_key(json, "kernel_size", c->kernel_size);
  c->stride = parse_int_key(json, "stride", c->stride);
  c->context = parse_int_key(json, "context", c->context);
  c->cac = parse_int_key(json, "cac", c->cac);
  c->use_train_segment =
      parse_int_key(json, "use_train_segment", c->use_train_segment);
  c->wiener_iters = parse_int_key(json, "wiener_iters", c->wiener_iters);
  c->valid_length = parse_int_key(json, "valid_length", c->valid_length);
  c->channels_time = parse_int_key(json, "channels_time", c->channels_time);
  c->growth = parse_int_key(json, "growth", c->growth);
  c->time_stride = parse_int_key(json, "time_stride", c->time_stride);
  c->context_enc = parse_int_key(json, "context_enc", c->context_enc);
  c->norm_groups = parse_int_key(json, "norm_groups", c->norm_groups);
  c->norm_starts = parse_int_key(json, "norm_starts", c->norm_starts);
  c->dconv_mode = parse_int_key(json, "dconv_mode", c->dconv_mode);
  c->dconv_depth = parse_int_key(json, "dconv_depth", c->dconv_depth);
  c->dconv_comp = parse_int_key(json, "dconv_comp", c->dconv_comp);
  c->dconv_init = parse_float_key(json, "dconv_init", c->dconv_init);
  c->rewrite = parse_int_key(json, "rewrite", c->rewrite);
  c->freq_emb = parse_float_key(json, "freq_emb", c->freq_emb);
  c->emb_scale = parse_float_key(json, "emb_scale", c->emb_scale);
  c->emb_smooth = parse_int_key(json, "emb_smooth", c->emb_smooth);
  c->t_layers = parse_int_key(json, "t_layers", c->t_layers);
  c->t_heads = parse_int_key(json, "t_heads", c->t_heads);
  c->t_hidden_scale = parse_float_key(json, "t_hidden_scale", c->t_hidden_scale);
  parse_string_key(json, "t_emb", c->t_emb, sizeof(c->t_emb), c->t_emb);
  c->t_max_period = parse_float_key(json, "t_max_period", c->t_max_period);
  c->t_weight_pos_embed =
      parse_float_key(json, "t_weight_pos_embed", c->t_weight_pos_embed);
  c->t_norm_in = parse_int_key(json, "t_norm_in", c->t_norm_in);
  c->t_norm_first = parse_int_key(json, "t_norm_first", c->t_norm_first);
  c->t_norm_out = parse_int_key(json, "t_norm_out", c->t_norm_out);
  c->t_layer_scale = parse_int_key(json, "t_layer_scale", c->t_layer_scale);
  c->t_gelu = parse_int_key(json, "t_gelu", c->t_gelu);
  c->t_cross_first = parse_int_key(json, "t_cross_first", c->t_cross_first);
  c->bottom_channels = parse_int_key(json, "bottom_channels", c->bottom_channels);

  const char *kw = kwargs_blob(json);
  if (kw) {
    c->channels = parse_int_key(kw, "channels", c->channels);
    c->channels_time = parse_int_key(kw, "channels_time", c->channels_time);
    c->growth = parse_int_key(kw, "growth", c->growth);
    c->depth = parse_int_key(kw, "depth", c->depth);
    c->nfft = parse_int_key(kw, "nfft", c->nfft);
    c->kernel_size = parse_int_key(kw, "kernel_size", c->kernel_size);
    c->stride = parse_int_key(kw, "stride", c->stride);
    c->time_stride = parse_int_key(kw, "time_stride", c->time_stride);
    c->context = parse_int_key(kw, "context", c->context);
    c->context_enc = parse_int_key(kw, "context_enc", c->context_enc);
    c->norm_groups = parse_int_key(kw, "norm_groups", c->norm_groups);
    c->norm_starts = parse_int_key(kw, "norm_starts", c->norm_starts);
    c->dconv_mode = parse_int_key(kw, "dconv_mode", c->dconv_mode);
    c->dconv_depth = parse_int_key(kw, "dconv_depth", c->dconv_depth);
    c->dconv_comp = parse_int_key(kw, "dconv_comp", c->dconv_comp);
    c->dconv_init = parse_float_key(kw, "dconv_init", c->dconv_init);
    c->rewrite = parse_int_key(kw, "rewrite", c->rewrite);
    c->cac = parse_int_key(kw, "cac", c->cac);
    c->freq_emb = parse_float_key(kw, "freq_emb", c->freq_emb);
    c->emb_scale = parse_float_key(kw, "emb_scale", c->emb_scale);
    c->emb_smooth = parse_int_key(kw, "emb_smooth", c->emb_smooth);
    c->t_layers = parse_int_key(kw, "t_layers", c->t_layers);
    c->t_heads = parse_int_key(kw, "t_heads", c->t_heads);
    c->t_hidden_scale = parse_float_key(kw, "t_hidden_scale", c->t_hidden_scale);
    parse_string_key(kw, "t_emb", c->t_emb, sizeof(c->t_emb), c->t_emb);
    c->t_max_period = parse_float_key(kw, "t_max_period", c->t_max_period);
    c->t_weight_pos_embed =
        parse_float_key(kw, "t_weight_pos_embed", c->t_weight_pos_embed);
    c->t_norm_in = parse_int_key(kw, "t_norm_in", c->t_norm_in);
    c->t_norm_first = parse_int_key(kw, "t_norm_first", c->t_norm_first);
    c->t_norm_out = parse_int_key(kw, "t_norm_out", c->t_norm_out);
    c->t_layer_scale = parse_int_key(kw, "t_layer_scale", c->t_layer_scale);
    c->t_gelu = parse_int_key(kw, "t_gelu", c->t_gelu);
    c->t_cross_first = parse_int_key(kw, "t_cross_first", c->t_cross_first);
    c->bottom_channels = parse_int_key(kw, "bottom_channels", c->bottom_channels);
    c->use_train_segment =
        parse_int_key(kw, "use_train_segment", c->use_train_segment);
    c->wiener_iters = parse_int_key(kw, "wiener_iters", c->wiener_iters);
    /* kwargs["segment"] may be the YAML fraction string "39/5"; keep the
       float already parsed from the top-level "segment" field. */
    c->samplerate = parse_int_key(kw, "samplerate", c->samplerate);
  }
  c->hop_length = c->nfft / 4;
  if (c->valid_length <= 0) {
    c->valid_length = (int)(c->segment * (float)c->samplerate);
  }
  if (c->wiener_iters != 0) {
    dc_die("C worker requires wiener_iters=0 (got %d)", c->wiener_iters);
  }
  if (c->n_sources <= 0 || c->n_sources > DC_MAX_SOURCES) {
    dc_die("bad n_sources %d", c->n_sources);
  }
  free(json);
}

int dc_config_valid_length(const DcConfig *c) {
  if (!c->use_train_segment) {
    dc_die("use_train_segment=0 is not supported in the C worker");
  }
  return c->valid_length;
}
