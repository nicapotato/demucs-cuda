#include "audio.h"

#include <lame/lame.h>
#include <math.h>

#define DR_WAV_IMPLEMENTATION
#define DR_WAV_NO_STDIO
#include "dr_wav.h"
#define DR_MP3_IMPLEMENTATION
#define DR_MP3_NO_STDIO
#include "dr_mp3.h"
#define DR_FLAC_IMPLEMENTATION
#define DR_FLAC_NO_STDIO
#include "dr_flac.h"

#define STB_VORBIS_NO_STDIO
#define STB_VORBIS_NO_PUSHDATA_API
#include "stb_vorbis.c"
#undef L
#undef R
#undef C

#ifdef _WIN32
#include "audio_mf.h"
#endif

void dc_audio_free(DcAudio *a) {
  if (!a) {
    return;
  }
  free(a->data);
  memset(a, 0, sizeof(*a));
}

/* interleaved [frames][channels] -> planar stereo */
static DcAudio from_interleaved(const float *pcm, int channels, int64_t frames, int sr,
                                const char *path) {
  if (channels <= 0 || frames <= 0 || sr <= 0) {
    dc_die("decoded audio is empty or invalid: %s", path);
  }
  if (frames > (1 << 30)) {
    dc_die("audio too long: %s", path);
  }
  DcAudio a;
  a.channels = 2;
  a.frames = (int)frames;
  a.sample_rate = sr;
  a.data = (float *)malloc((size_t)2 * (size_t)frames * sizeof(float));
  if (!a.data) {
    dc_die("oom audio buffer (%lld frames)", (long long)frames);
  }
  int c1 = channels > 1 ? 1 : 0;
  for (int64_t t = 0; t < frames; t++) {
    a.data[t] = pcm[t * channels];
    a.data[frames + t] = pcm[t * channels + c1];
  }
  return a;
}

static int has_magic(const unsigned char *b, size_t n, size_t off, const char *m) {
  size_t k = strlen(m);
  return n >= off + k && memcmp(b + off, m, k) == 0;
}

static int try_wav(const void *buf, size_t n, const char *path, DcAudio *out) {
  unsigned int ch = 0, sr = 0;
  drwav_uint64 frames = 0;
  float *pcm = drwav_open_memory_and_read_pcm_frames_f32(buf, n, &ch, &sr, &frames, NULL);
  if (!pcm) {
    return 0;
  }
  *out = from_interleaved(pcm, (int)ch, (int64_t)frames, (int)sr, path);
  drwav_free(pcm, NULL);
  return 1;
}

static int try_flac(const void *buf, size_t n, const char *path, DcAudio *out) {
  unsigned int ch = 0, sr = 0;
  drflac_uint64 frames = 0;
  float *pcm = drflac_open_memory_and_read_pcm_frames_f32(buf, n, &ch, &sr, &frames, NULL);
  if (!pcm) {
    return 0;
  }
  *out = from_interleaved(pcm, (int)ch, (int64_t)frames, (int)sr, path);
  drflac_free(pcm, NULL);
  return 1;
}

static int try_mp3(const void *buf, size_t n, const char *path, DcAudio *out) {
  drmp3_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  drmp3_uint64 frames = 0;
  float *pcm = drmp3_open_memory_and_read_pcm_frames_f32(buf, n, &cfg, &frames, NULL);
  if (!pcm) {
    return 0;
  }
  if (frames == 0) {
    drmp3_free(pcm, NULL);
    return 0;
  }
  *out = from_interleaved(pcm, (int)cfg.channels, (int64_t)frames, (int)cfg.sampleRate, path);
  drmp3_free(pcm, NULL);
  return 1;
}

static int try_vorbis(const void *buf, size_t n, const char *path, DcAudio *out) {
  int ch = 0, sr = 0;
  short *pcm16 = NULL;
  int frames = stb_vorbis_decode_memory((const unsigned char *)buf, (int)n, &ch, &sr, &pcm16);
  if (frames <= 0 || !pcm16) {
    free(pcm16);
    return 0;
  }
  float *pcm = (float *)malloc((size_t)frames * (size_t)ch * sizeof(float));
  if (!pcm) {
    dc_die("oom vorbis buffer");
  }
  for (size_t i = 0; i < (size_t)frames * (size_t)ch; i++) {
    pcm[i] = (float)pcm16[i] / 32768.0f;
  }
  free(pcm16);
  *out = from_interleaved(pcm, ch, frames, sr, path);
  free(pcm);
  return 1;
}

#ifdef _WIN32
static int try_media_foundation(const char *path, DcAudio *out) {
  float *pcm = NULL;
  int ch = 0, sr = 0;
  long long frames = 0;
  if (!dc_mf_decode(path, &pcm, &ch, &frames, &sr)) {
    return 0;
  }
  *out = from_interleaved(pcm, ch, (int64_t)frames, sr, path);
  free(pcm);
  return 1;
}
#endif

DcAudio dc_audio_read(const char *path) {
  size_t n = 0;
  unsigned char *buf = (unsigned char *)dc_read_file(path, &n);
  if (!buf) {
    dc_die("cannot read audio file %s", path);
  }
  DcAudio a;
  memset(&a, 0, sizeof(a));
  int ok = 0;
  if ((has_magic(buf, n, 0, "RIFF") || has_magic(buf, n, 0, "RIFX") ||
       has_magic(buf, n, 0, "RF64") || has_magic(buf, n, 0, "FORM")) ) {
    ok = try_wav(buf, n, path, &a);
  } else if (has_magic(buf, n, 0, "fLaC")) {
    ok = try_flac(buf, n, path, &a);
  } else if (has_magic(buf, n, 0, "OggS")) {
    ok = try_vorbis(buf, n, path, &a) || try_flac(buf, n, path, &a);
  } else if (!has_magic(buf, n, 4, "ftyp")) {
    ok = try_mp3(buf, n, path, &a);
  }
  free(buf);
#ifdef _WIN32
  if (!ok) {
    ok = try_media_foundation(path, &a);
  }
#endif
  if (!ok) {
    dc_die("unsupported or corrupt audio file: %s", path);
  }
  return a;
}

void dc_audio_resample(DcAudio *a, int target_sr) {
  if (!a || !a->data || target_sr <= 0) {
    dc_die("dc_audio_resample: bad arguments");
  }
  if (a->sample_rate == target_sr) {
    return;
  }
  int64_t nf = ((int64_t)a->frames * target_sr) / a->sample_rate;
  if (nf <= 0 || nf > (1 << 30)) {
    dc_die("dc_audio_resample: bad output length %lld", (long long)nf);
  }
  float *out = (float *)malloc((size_t)a->channels * (size_t)nf * sizeof(float));
  if (!out) {
    dc_die("oom resample buffer");
  }
  double ratio = (double)a->sample_rate / (double)target_sr;
  for (int c = 0; c < a->channels; c++) {
    const float *src = a->data + (size_t)c * a->frames;
    float *dst = out + (size_t)c * nf;
    for (int64_t t = 0; t < nf; t++) {
      double st = (double)t * ratio;
      int i0 = (int)st;
      i0 = i0 >= a->frames ? a->frames - 1 : i0;
      int i1 = i0 + 1 >= a->frames ? a->frames - 1 : i0 + 1;
      float frac = (float)(st - (double)i0);
      dst[t] = src[i0] + frac * (src[i1] - src[i0]);
    }
  }
  free(a->data);
  a->data = out;
  a->frames = (int)nf;
  a->sample_rate = target_sr;
}

static void put_u16le(unsigned char *p, uint16_t v) {
  p[0] = (unsigned char)(v & 0xff);
  p[1] = (unsigned char)(v >> 8);
}
static void put_u32le(unsigned char *p, uint32_t v) {
  for (int i = 0; i < 4; i++) {
    p[i] = (unsigned char)((v >> (8 * i)) & 0xff);
  }
}

/* save_audio(clip='rescale'): wav / max(1.01 * peak, 1) */
static float prevent_clip_scale(const float *planar, int channels, int frames) {
  float peak = 0.0f;
  for (size_t i = 0; i < (size_t)channels * (size_t)frames; i++) {
    float v = fabsf(planar[i]);
    peak = v > peak ? v : peak;
  }
  float scale = 1.01f * peak;
  return scale < 1.0f ? 1.0f : scale;
}

static int16_t s16_from_float(float v, float scale) {
  v /= scale;
  v = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
  int iv = (int)lrintf(v * 32767.0f);
  iv = iv > 32767 ? 32767 : (iv < -32768 ? -32768 : iv);
  return (int16_t)iv;
}

static void interleave_s16(int16_t *tile, const float *planar, int channels, int frames, int t0,
                           int n, float scale) {
  for (int t = 0; t < n; t++) {
    for (int c = 0; c < channels; c++) {
      tile[t * channels + c] =
          s16_from_float(planar[(size_t)c * (size_t)frames + (size_t)(t0 + t)], scale);
    }
  }
}

#define DC_TILE_FRAMES 65536

void dc_wav_write_pcm16(const char *path, const float *planar, int channels, int frames,
                        int sample_rate) {
  FILE *fp = dc_fopen(path, "wb");
  if (!fp) {
    dc_die("cannot write %s", path);
  }
  uint32_t data_bytes = (uint32_t)channels * (uint32_t)frames * 2u;
  unsigned char hdr[44];
  memset(hdr, 0, sizeof(hdr));
  memcpy(hdr, "RIFF", 4);
  put_u32le(hdr + 4, 36 + data_bytes);
  memcpy(hdr + 8, "WAVEfmt ", 8);
  put_u32le(hdr + 16, 16);
  put_u16le(hdr + 20, 1);
  put_u16le(hdr + 22, (uint16_t)channels);
  put_u32le(hdr + 24, (uint32_t)sample_rate);
  put_u32le(hdr + 28, (uint32_t)sample_rate * (uint32_t)channels * 2u);
  put_u16le(hdr + 32, (uint16_t)(channels * 2));
  put_u16le(hdr + 34, 16);
  memcpy(hdr + 36, "data", 4);
  put_u32le(hdr + 40, data_bytes);
  if (fwrite(hdr, 1, 44, fp) != 44) {
    dc_die("wav header write failed %s", path);
  }
  float scale = prevent_clip_scale(planar, channels, frames);
  int16_t *tile = (int16_t *)malloc((size_t)DC_TILE_FRAMES * channels * sizeof(int16_t));
  if (!tile) {
    dc_die("oom wav encode tile");
  }
  for (int t0 = 0; t0 < frames; t0 += DC_TILE_FRAMES) {
    int n = frames - t0 < DC_TILE_FRAMES ? frames - t0 : DC_TILE_FRAMES;
    interleave_s16(tile, planar, channels, frames, t0, n, scale);
    size_t nbytes = (size_t)n * channels * 2u;
    if (fwrite(tile, 1, nbytes, fp) != nbytes) {
      dc_die("wav data write failed %s", path);
    }
  }
  free(tile);
  fclose(fp);
}

void dc_mp3_write(const char *path, const float *planar, int channels, int frames,
                  int sample_rate, int bitrate_kbps) {
  if (channels != 1 && channels != 2) {
    dc_die("dc_mp3_write: channels=%d, expected 1 or 2", channels);
  }
  if (frames <= 0 || sample_rate <= 0 || bitrate_kbps <= 0) {
    dc_die("dc_mp3_write: bad frames/rate/bitrate");
  }
  lame_t gfp = lame_init();
  if (!gfp) {
    dc_die("lame_init failed for %s", path);
  }
  lame_set_in_samplerate(gfp, sample_rate);
  lame_set_num_channels(gfp, channels);
  lame_set_brate(gfp, bitrate_kbps);
  lame_set_quality(gfp, 2);
  lame_set_mode(gfp, channels == 1 ? MONO : STEREO);
  if (lame_init_params(gfp) < 0) {
    lame_close(gfp);
    dc_die("lame_init_params failed for %s", path);
  }
  FILE *fp = dc_fopen(path, "wb");
  if (!fp) {
    lame_close(gfp);
    dc_die("cannot write %s", path);
  }
  float scale = prevent_clip_scale(planar, channels, frames);
  int16_t *tile = (int16_t *)malloc((size_t)DC_TILE_FRAMES * channels * sizeof(int16_t));
  /* LAME: mp3buf_size >= 1.25*samples + 7200 */
  int mp3_cap = (int)((double)DC_TILE_FRAMES * 1.25) + 7200;
  unsigned char *mp3buf = (unsigned char *)malloc((size_t)mp3_cap);
  if (!tile || !mp3buf) {
    dc_die("oom mp3 encode buffers");
  }
  int failed = 0;
  for (int t0 = 0; t0 < frames && !failed; t0 += DC_TILE_FRAMES) {
    int n = frames - t0 < DC_TILE_FRAMES ? frames - t0 : DC_TILE_FRAMES;
    interleave_s16(tile, planar, channels, frames, t0, n, scale);
    int nb = channels == 2 ? lame_encode_buffer_interleaved(gfp, tile, n, mp3buf, mp3_cap)
                           : lame_encode_buffer(gfp, tile, NULL, n, mp3buf, mp3_cap);
    failed = nb < 0 || (nb > 0 && fwrite(mp3buf, 1, (size_t)nb, fp) != (size_t)nb);
  }
  if (!failed) {
    int nb = lame_encode_flush(gfp, mp3buf, mp3_cap);
    failed = nb < 0 || (nb > 0 && fwrite(mp3buf, 1, (size_t)nb, fp) != (size_t)nb);
  }
  free(tile);
  free(mp3buf);
  fclose(fp);
  lame_close(gfp);
  if (failed) {
    remove(path);
    dc_die("mp3 encode/write failed for %s", path);
  }
}
