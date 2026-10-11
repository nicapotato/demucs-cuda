/* Golden comparison against the MLX/PyTorch reference tensors.
 * Every stage is fed the golden output of the previous stage (teacher forcing),
 * so a failure points at the layer that introduced it; then the full forward
 * pass is compared on the final stems. Gates are on relative L2 error, sized for
 * fp16 GEMM operands with fp32 accumulation (observed: stages <= 8e-4, stems ~5e-4).
 *
 * Goldens must come from a reference without demucs_mlx's fused GroupNorm Metal
 * kernels (scripts/make_reference_tensors.py): those race on their shared
 * reduction buffer and corrupt the high-resolution frequency layers.
 *
 * usage: test_compare WEIGHTS.safetensors CONFIG.json GOLDENS.safetensors */
#include "htdemucs.h"

#include <math.h>

#define STAGE_GATE 2e-3
#define STEMS_GATE 2e-3

static int g_fail = 0;

static double check(const char *name, DcTensor pred, DcTensor ref, double gate) {
  if (!dc_valid(ref)) {
    printf("%-28s (no golden)\n", name);
    return 0.0;
  }
  if (dc_numel(pred) != dc_numel(ref)) {
    printf("%-28s SHAPE MISMATCH ", name);
    dc_print_shape("pred", pred);
    dc_print_shape("ref", ref);
    g_fail = 1;
    return 1e9;
  }
  float *p = dc_to_host(pred);
  float *r = dc_to_host(ref);
  int64_t n = dc_numel(ref);
  double num = 0.0, den = 0.0, max_abs = 0.0;
  for (int64_t i = 0; i < n; i++) {
    double d = (double)p[i] - (double)r[i];
    num += d * d;
    den += (double)r[i] * (double)r[i];
    max_abs = fabs(d) > max_abs ? fabs(d) : max_abs;
  }
  double rel = den > 0 ? sqrt(num / den) : sqrt(num);
  int ok = rel <= gate && !isnan(rel);
  if (!ok && ref.ndim >= 3 && getenv("DC_COMPARE_VERBOSE")) {
    /* error energy per index of each of the last three axes */
    for (int ax = ref.ndim - 3; ax < ref.ndim; ax++) {
      int len = ref.shape[ax];
      int64_t inner = 1;
      for (int d = ax + 1; d < ref.ndim; d++) {
        inner *= ref.shape[d];
      }
      double *e = (double *)calloc((size_t)len, sizeof(double));
      for (int64_t i = 0; i < n; i++) {
        double d = (double)p[i] - (double)r[i];
        e[(i / inner) % len] += d * d;
      }
      int worst[4] = {0, 0, 0, 0};
      for (int k = 0; k < 4; k++) {
        double best = -1;
        for (int j = 0; j < len; j++) {
          int used = 0;
          for (int q = 0; q < k; q++) {
            used |= worst[q] == j;
          }
          if (!used && e[j] > best) {
            best = e[j];
            worst[k] = j;
          }
        }
      }
      printf("    axis %d (len %d): worst idx %d %d %d %d, share %.2f\n", ax, len, worst[0],
             worst[1], worst[2], worst[3], e[worst[0]] / (num > 0 ? num : 1));
      free(e);
    }
  }
  free(p);
  free(r);
  printf("%-28s rel_l2=%.3e max_abs=%.3e gate=%.0e %s\n", name, rel, max_abs, gate,
         ok ? "ok" : "FAIL");
  if (!ok) {
    g_fail = 1;
  }
  return rel;
}

static DcTensor golden(DcStFile *g, const char *fmt, int idx) {
  char key[64];
  snprintf(key, sizeof(key), fmt, idx);
  DcTensor t = dc_st_get(g, key);
  if (!dc_valid(t)) {
    dc_die("golden %s missing", key);
  }
  return t;
}

int main(int argc, char **argv) {
  const char *weights = argc > 1 ? argv[1] : "models/htdemucs_6s.safetensors";
  const char *config = argc > 2 ? argv[2] : "models/htdemucs_6s.json";
  const char *goldens = argc > 3 ? argv[3] : "goldens/htdemucs_6s/goldens.safetensors";
  char dev[256];
  if (dk_init(0, dev, sizeof(dev)) != 0) {
    dc_die("no device: %s", dev);
  }
  printf("backend %s (%s)\n", dk_backend_name(), dev);

  DcWeights w;
  dc_weights_load(&w, weights, config);
  DcModel m;
  dc_model_init(&m, &w);
  DcStFile *g = dc_st_open(goldens);
  if (!g) {
    dc_die("cannot read goldens %s", goldens);
  }

  /* spectrogram + normalisation (fp32 path) */
  DcTensor mix0 = dc_st_get(g, "mix");
  DcTensor mix = dc_expand_dims(mix0, 0);
  double t0 = dc_now_s();
  DcTensor mag = dc_htdemucs_spec_cac(&m.stft, mix);
  dk_sync();
  double t_spec = dc_now_s() - t0;
  DcTensor ref = dc_st_get(g, "mag");
  check("mag", mag, ref, 1e-5);
  dc_free(ref);
  DcTensor mean, var;
  dc_batch_stats(mag, &mean, &var);
  DcTensor xn = dc_batch_normalize(mag, mean, var, 1e-5f, 0);
  ref = dc_st_get(g, "x_norm");
  check("x_norm", xn, ref, 1e-5);
  dc_free(ref);
  DcTensor meant, vart;
  dc_batch_stats(mix, &meant, &vart);
  DcTensor xtn = dc_batch_normalize(mix, meant, vart, 1e-5f, 0);
  ref = dc_st_get(g, "xt_norm");
  check("xt_norm", xtn, ref, 1e-5);
  dc_free(ref);

  /* encoders (teacher forced) */
  for (int i = 0; i < m.n_enc; i++) {
    DcTensor in = i == 0 ? dc_ref(xn) : golden(g, "enc_%d", i - 1);
    DcTensor out = dc_henc(in, dc_empty(), m.w, &m.encoder[i]);
    if (i == 0) {
      DcTensor o2 = dc_add_freq_emb(&m, out);
      dc_free(out);
      out = o2;
    }
    char name[32];
    snprintf(name, sizeof(name), "enc_%d", i);
    ref = golden(g, "enc_%d", i);
    check(name, out, ref, STAGE_GATE);
    dc_free(ref);
    dc_free(in);
    dc_free(out);
  }
  for (int i = 0; i < m.n_tenc; i++) {
    DcTensor in = i == 0 ? dc_ref(xtn) : golden(g, "tenc_%d", i - 1);
    DcTensor out = dc_henc(in, dc_empty(), m.w, &m.tencoder[i]);
    char name[32];
    snprintf(name, sizeof(name), "tenc_%d", i);
    ref = golden(g, "tenc_%d", i);
    check(name, out, ref, STAGE_GATE);
    dc_free(ref);
    dc_free(in);
    dc_free(out);
  }

  /* cross transformer from the golden bottleneck */
  DcTensor gx = golden(g, "enc_%d", m.n_enc - 1);
  DcTensor gxt = golden(g, "tenc_%d", m.n_tenc - 1);
  t0 = dc_now_s();
  dc_crosstransformer(&gx, &gxt, m.w, &w.cfg);
  dk_sync();
  double t_tf = dc_now_s() - t0;
  DcTensor rx = dc_st_get(g, "transformer_x");
  DcTensor rxt = dc_st_get(g, "transformer_xt");
  check("transformer_x", gx, rx, STAGE_GATE);
  check("transformer_xt", gxt, rxt, STAGE_GATE);
  dc_free(gx);
  dc_free(gxt);

  /* decoders (teacher forced) */
  int tlens[DC_MAX_DEPTH];
  for (int i = 0; i < m.n_tenc; i++) {
    DcTensor tin = i == 0 ? dc_ref(xtn) : golden(g, "tenc_%d", i - 1);
    tlens[i] = tin.shape[tin.ndim - 1];
    dc_free(tin);
  }
  for (int i = 0; i < m.n_dec; i++) {
    DcTensor in = i == 0 ? dc_ref(rx) : golden(g, "dec_%d", i - 1);
    DcTensor skip = golden(g, "enc_%d", m.n_enc - 1 - i);
    DcTensor out = dc_hdec(in, skip, mag.shape[3], m.w, &m.decoder[i], NULL);
    char name[32];
    snprintf(name, sizeof(name), "dec_%d", i);
    ref = golden(g, "dec_%d", i);
    check(name, out, ref, STAGE_GATE);
    dc_free(ref);
    dc_free(in);
    dc_free(skip);
    dc_free(out);
  }
  for (int i = 0; i < m.n_tdec; i++) {
    DcTensor in = i == 0 ? dc_ref(rxt) : golden(g, "tdec_%d", i - 1);
    DcTensor skip = golden(g, "tenc_%d", m.n_tenc - 1 - i);
    DcTensor out = dc_hdec(in, skip, tlens[m.n_tenc - 1 - i], m.w, &m.tdecoder[i], NULL);
    char name[32];
    snprintf(name, sizeof(name), "tdec_%d", i);
    ref = golden(g, "tdec_%d", i);
    check(name, out, ref, STAGE_GATE);
    dc_free(ref);
    dc_free(in);
    dc_free(skip);
    dc_free(out);
  }
  dc_free(rx);
  dc_free(rxt);

  /* full forward */
  t0 = dc_now_s();
  DcTensor y = dc_htdemucs_forward(&m, mix);
  dk_sync();
  double t_fwd = dc_now_s() - t0;
  ref = dc_st_get(g, "stems");
  double rel = check("stems (end to end)", y, ref, STEMS_GATE);
  printf("stems SNR vs reference: %.1f dB\n", -20.0 * log10(rel > 0 ? rel : 1e-12));
  printf("timing spec=%.3fs transformer=%.3fs forward=%.3fs peak_mem=%.2fGiB\n", t_spec, t_tf,
         t_fwd, (double)dc_dev_peak_bytes() / (double)(1u << 30));
  dc_free(ref);
  dc_free(y);

  dc_free(mix0);
  dc_free(mix);
  dc_free(mag);
  dc_free(xn);
  dc_free(xtn);
  dc_free(mean);
  dc_free(var);
  dc_free(meant);
  dc_free(vart);
  dc_st_close(g);
  dc_model_free(&m);
  dc_weights_free(&w);
  printf(g_fail ? "test_compare: FAIL\n" : "test_compare: OK\n");
  return g_fail;
}
