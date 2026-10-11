#ifndef DEMUCS_CUDA_CONFIG_H
#define DEMUCS_CUDA_CONFIG_H

#include "util.h"

typedef struct {
  char model[64];
  char sources[DC_MAX_SOURCES][32];
  int n_sources;
  int samplerate;
  float segment;
  int nfft;
  int hop_length;
  int audio_channels;
  int channels;
  int channels_time;
  int growth;
  int depth;
  int kernel_size;
  int stride;
  int time_stride;
  int context;
  int context_enc;
  int norm_groups;
  int norm_starts;
  int dconv_mode;
  int dconv_depth;
  int dconv_comp;
  float dconv_init;
  int rewrite;
  int cac;
  float freq_emb;
  float emb_scale;
  int emb_smooth;
  int t_layers;
  int t_heads;
  float t_hidden_scale;
  char t_emb[16];
  float t_max_period;
  float t_weight_pos_embed;
  int t_norm_in;
  int t_norm_first;
  int t_norm_out;
  int t_layer_scale;
  int t_gelu;
  int t_cross_first;
  int bottom_channels;
  int use_train_segment;
  int wiener_iters;
  int valid_length;
} DcConfig;

void dc_config_defaults(DcConfig *c);
void dc_config_load_json(DcConfig *c, const char *path);
int dc_config_valid_length(const DcConfig *c);

#endif
