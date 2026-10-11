#include "apply.h"
#include "audio.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#define DC_SOURCES_JSON "[\"drums\",\"bass\",\"other\",\"vocals\",\"guitar\",\"piano\"]"
/* rough peak device memory per batch item for htdemucs at the training segment */
#define DC_BYTES_PER_BATCH_ITEM ((size_t)640 << 20)
#define DC_DEVICE_RESERVE ((size_t)768 << 20)

static int g_json = 0;

static void emit(const char *fmt, ...) {
  if (!g_json) {
    return;
  }
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
  fputc('\n', stdout);
  fflush(stdout);
}

typedef struct {
  const char *track_json;
} DcProgressCtx;

static void on_progress(int done, int total, void *user) {
  DcProgressCtx *ctx = (DcProgressCtx *)user;
  float pct = total > 0 ? (100.0f * (float)done / (float)total) : 0.0f;
  emit("{\"event\":\"separating\",\"track\":\"%s\",\"done\":%d,\"total\":%d,\"pct\":%.2f}",
       ctx->track_json, done, total, pct);
}

static void strip_ext(char *name) {
  char *dot = strrchr(name, '.');
  if (dot && dot != name) {
    *dot = 0;
  }
}

/* first directory containing <model>.fp16.safetensors or <model>.safetensors */
static int find_model(const char *model, char *weights, char *config, size_t cap) {
  char exe[DC_PATH_MAX] = {0};
  char exe_models[DC_PATH_MAX] = {0};
  char exe_up_models[DC_PATH_MAX] = {0};
  if (dc_exe_dir(exe, sizeof(exe)) == 0) {
    dc_path_join(exe_models, sizeof(exe_models), exe, "models");
    char up[DC_PATH_MAX];
    dc_path_join(up, sizeof(up), exe, "..");
    dc_path_join(exe_up_models, sizeof(exe_up_models), up, "models");
  }
  const char *dirs[] = {getenv("DEMUCS_CUDA_CACHE"), getenv("DEMUCS_MLX_CACHE"),
                        getenv("DEMUCS_TORCH_CACHE"), exe_models, exe_up_models, "models"};
  const char *suffixes[] = {".fp16.safetensors", ".safetensors"};
  for (size_t d = 0; d < sizeof(dirs) / sizeof(dirs[0]); d++) {
    if (!dirs[d] || !dirs[d][0]) {
      continue;
    }
    for (int s = 0; s < 2; s++) {
      char name[256];
      snprintf(name, sizeof(name), "%s%s", model, suffixes[s]);
      dc_path_join(weights, cap, dirs[d], name);
      if (!dc_file_exists(weights)) {
        continue;
      }
      snprintf(name, sizeof(name), "%s.json", model);
      dc_path_join(config, cap, dirs[d], name);
      if (dc_file_exists(config)) {
        return 1;
      }
    }
  }
  return 0;
}

static void usage(void) {
  fprintf(stderr,
          "usage: demucs_cuda_worker [flags] track...\n"
          "  -n MODEL  --weights FILE.safetensors --config FILE.json\n"
          "  -o DIR --track-name NAME --overlap F --shifts N --seed N\n"
          "  -b N --no-split --progress-json --mp3 --mp3-bitrate N --device N\n"
          "  --prefetch-tracks N is accepted and ignored (GUI compatibility)\n"
          "models are searched in DEMUCS_CUDA_CACHE, DEMUCS_MLX_CACHE, DEMUCS_TORCH_CACHE,\n"
          "<exe>/models, <exe>/../models and ./models\n");
}

static int run(int argc, char **argv) {
  double t_all = dc_now_s();
  const char *out_dir = "separated";
  const char *track_name = NULL;
  const char *model_name = "htdemucs_6s";
  const char *weights = NULL;
  const char *config = NULL;
  float overlap = 0.25f;
  int shifts = 1;
  int seed = 0;
  int use_seed = 0;
  int batch_size = 8;
  int split = 1;
  int write_mp3 = 0;
  int mp3_bitrate = 320;
  int device = 0;
  const char *tracks[64];
  int ntracks = 0;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    int more = i + 1 < argc;
    if (strcmp(a, "-o") == 0 && more) {
      out_dir = argv[++i];
    } else if (strcmp(a, "--track-name") == 0 && more) {
      track_name = argv[++i];
    } else if ((strcmp(a, "-n") == 0 || strcmp(a, "--name") == 0) && more) {
      model_name = argv[++i];
    } else if (strcmp(a, "--weights") == 0 && more) {
      weights = argv[++i];
    } else if (strcmp(a, "--config") == 0 && more) {
      config = argv[++i];
    } else if (strcmp(a, "--overlap") == 0 && more) {
      overlap = strtof(argv[++i], NULL);
    } else if (strcmp(a, "--shifts") == 0 && more) {
      shifts = (int)strtol(argv[++i], NULL, 10);
    } else if (strcmp(a, "--seed") == 0 && more) {
      seed = (int)strtol(argv[++i], NULL, 10);
      use_seed = 1;
    } else if ((strcmp(a, "-b") == 0 || strcmp(a, "--batch-size") == 0) && more) {
      batch_size = (int)strtol(argv[++i], NULL, 10);
    } else if (strcmp(a, "--device") == 0 && more) {
      device = (int)strtol(argv[++i], NULL, 10);
    } else if (strcmp(a, "--no-split") == 0) {
      split = 0;
    } else if (strcmp(a, "--progress-json") == 0) {
      g_json = 1;
      dc_set_json_errors(1);
    } else if (strcmp(a, "--list-models") == 0) {
      printf("htdemucs_6s\n");
      return 0;
    } else if (strcmp(a, "--prefetch-tracks") == 0 && more) {
      ++i;
    } else if (strcmp(a, "--mp3") == 0) {
      write_mp3 = 1;
    } else if (strcmp(a, "--mp3-bitrate") == 0 && more) {
      mp3_bitrate = (int)strtol(argv[++i], NULL, 10);
    } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
      usage();
      return 0;
    } else if (a[0] == '-' && a[1]) {
      dc_die("unknown flag %s", a);
    } else {
      if (ntracks >= 64) {
        dc_die("too many tracks");
      }
      tracks[ntracks++] = a;
    }
  }
  if (ntracks == 0) {
    usage();
    return 2;
  }
  if (track_name && ntracks != 1) {
    dc_die("--track-name requires exactly one track");
  }
  if (mp3_bitrate <= 0) {
    dc_die("--mp3-bitrate must be > 0");
  }

  char wbuf[DC_PATH_MAX], cbuf[DC_PATH_MAX];
  if (!weights || !config) {
    if (!find_model(model_name, wbuf, cbuf, sizeof(wbuf))) {
      dc_die("model %s not found (expected %s.fp16.safetensors + %s.json in a models "
             "directory; see --help)",
             model_name, model_name, model_name);
    }
    weights = weights ? weights : wbuf;
    config = config ? config : cbuf;
  }

  emit("{\"event\":\"status\",\"stage\":\"loading_model\",\"model\":\"%s\",\"device\":\"%s\"}",
       model_name, dk_backend_name());

  double t0 = dc_now_s();
  char gpu_name[256];
  if (dk_init(device, gpu_name, sizeof(gpu_name)) != 0) {
    dc_die("no usable %s device (%s)", dk_backend_name(), gpu_name);
  }
  DcWeights wts;
  dc_weights_load(&wts, weights, config);
  DcModel model;
  dc_model_init(&model, &wts);
  dk_sync();
  dc_time_log("load_model", t0);

  size_t free_b = 0, total_b = 0;
  if (dk_mem_info(&free_b, &total_b) == 0 && free_b > DC_DEVICE_RESERVE) {
    int cap = (int)((free_b - DC_DEVICE_RESERVE) / DC_BYTES_PER_BATCH_ITEM);
    cap = cap < 1 ? 1 : cap;
    if (batch_size > cap) {
      fprintf(stderr, "demucs-cuda: batch size %d -> %d (%.1f GiB free)\n", batch_size, cap,
              (double)free_b / (double)(1u << 30));
      batch_size = cap;
    }
  }
  if (dc_timing_enabled()) {
    fprintf(stderr, "dc_device %s free=%.2fGiB total=%.2fGiB batch=%d\n", gpu_name,
            (double)free_b / (double)(1u << 30), (double)total_b / (double)(1u << 30),
            batch_size);
  }

  char gpu_json[512];
  dc_json_escape(gpu_json, sizeof(gpu_json), gpu_name);
  emit("{\"event\":\"status\",\"stage\":\"model_ready\",\"model\":\"%s\",\"device\":\"%s\","
       "\"gpu\":\"%s\",\"sources\":" DC_SOURCES_JSON "}",
       model_name, dk_backend_name(), gpu_json);

  DcApplyOpts opts;
  memset(&opts, 0, sizeof(opts));
  opts.shifts = shifts;
  opts.split = split;
  opts.overlap = overlap;
  opts.transition_power = 1.0f;
  opts.segment = wts.cfg.segment;
  opts.batch_size = batch_size;
  opts.seed = (unsigned)seed;
  opts.use_seed = use_seed;

  for (int t = 0; t < ntracks; t++) {
    char stem[256];
    snprintf(stem, sizeof(stem), "%s", track_name ? track_name : dc_path_basename(tracks[t]));
    if (!track_name) {
      strip_ext(stem);
    }
    char dest[DC_PATH_MAX];
    dc_path_join(dest, sizeof(dest), out_dir, stem);
    dc_mkdir_p(dest);
    char track_json[2 * DC_PATH_MAX], dest_json[2 * DC_PATH_MAX];
    dc_json_escape(track_json, sizeof(track_json), tracks[t]);
    dc_json_escape(dest_json, sizeof(dest_json), dest);
    emit("{\"event\":\"loading\",\"track\":\"%s\",\"out\":\"%s\"}", track_json, dest_json);

    DcProgressCtx pctx = {track_json};
    opts.on_progress = on_progress;
    opts.progress_user = &pctx;
    emit("{\"event\":\"separating\",\"track\":\"%s\",\"done\":0,\"total\":0,\"pct\":0.00}",
         track_json);

    t0 = dc_now_s();
    DcAudio wav = dc_audio_read(tracks[t]);
    if (wav.sample_rate != wts.cfg.samplerate) {
      fprintf(stderr, "demucs-cuda: resampling %d -> %d\n", wav.sample_rate,
              wts.cfg.samplerate);
      dc_audio_resample(&wav, wts.cfg.samplerate);
    }
    int ishape[3] = {1, wav.channels, wav.frames};
    DcTensor mix = dc_from_host(wav.data, ishape, 3);
    dc_time_log("audio_read", t0);

    t0 = dc_now_s();
    DcTensor stems = dc_apply_model(&model, mix, &opts);
    dk_sync();
    dc_time_log("apply", t0);
    if (dc_timing_enabled()) {
      fprintf(stderr, "dc_memory peak=%.2fGiB\n",
              (double)dc_dev_peak_bytes() / (double)(1u << 30));
    }

    t0 = dc_now_s();
    float *host = dc_to_host(stems); /* [1, S, C, T] */
    int S = stems.shape[1], C = stems.shape[2], T = stems.shape[3];
    for (int si = 0; si < S && si < wts.cfg.n_sources; si++) {
      const float *planar = host + (size_t)si * C * T;
      char fname[96], path[DC_PATH_MAX];
      snprintf(fname, sizeof(fname), "%s.%s", wts.cfg.sources[si], write_mp3 ? "mp3" : "wav");
      dc_path_join(path, sizeof(path), dest, fname);
      if (write_mp3) {
        dc_mp3_write(path, planar, C, T, wts.cfg.samplerate, mp3_bitrate);
      } else {
        dc_wav_write_pcm16(path, planar, C, T, wts.cfg.samplerate);
      }
      char path_json[2 * DC_PATH_MAX];
      dc_json_escape(path_json, sizeof(path_json), path);
      emit("{\"event\":\"writing\",\"stem\":\"%s\",\"path\":\"%s\"}", wts.cfg.sources[si],
           path_json);
    }
    free(host);
    dc_time_log(write_mp3 ? "mp3_write" : "wav_write", t0);
    emit("{\"event\":\"done\",\"track\":\"%s\",\"out\":\"%s\",\"sources\":" DC_SOURCES_JSON
         ",\"format\":\"%s\"}",
         track_json, dest_json, write_mp3 ? "mp3" : "wav");
    dc_free(mix);
    dc_free(stems);
    dc_audio_free(&wav);
  }

  dc_model_free(&model);
  dc_weights_free(&wts);
  dc_time_log("total", t_all);
  return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t **wargv) {
  char **argv = (char **)calloc((size_t)argc + 1, sizeof(char *));
  for (int i = 0; i < argc; i++) {
    int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
    argv[i] = (char *)malloc((size_t)(n > 0 ? n : 1));
    if (n <= 0 || !WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, argv[i], n, NULL, NULL)) {
      argv[i][0] = 0;
    }
  }
  SetConsoleOutputCP(CP_UTF8);
  return run(argc, argv);
}
#else
int main(int argc, char **argv) { return run(argc, argv); }
#endif
