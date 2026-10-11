#ifndef DEMUCS_CUDA_AUDIO_H
#define DEMUCS_CUDA_AUDIO_H

#include "util.h"

/* planar float audio: data[c * frames + t] */
typedef struct {
  float *data;
  int channels;
  int frames;
  int sample_rate;
} DcAudio;

/* wav/aiff/mp3/flac/ogg (+ m4a and anything Media Foundation decodes on
 * Windows). Output is always stereo: mono is duplicated, extra channels dropped. */
DcAudio dc_audio_read(const char *path);
void dc_audio_free(DcAudio *a);
void dc_audio_resample(DcAudio *a, int target_sr);

void dc_wav_write_pcm16(const char *path, const float *planar, int channels, int frames,
                        int sample_rate);
void dc_mp3_write(const char *path, const float *planar, int channels, int frames,
                  int sample_rate, int bitrate_kbps);

#endif
