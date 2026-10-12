#ifndef DEMUCS_CUDA_AUDIO_MF_H
#define DEMUCS_CUDA_AUDIO_MF_H

#ifdef __cplusplus
extern "C" {
#endif

/* Decode any Media Foundation-supported file to interleaved float PCM (caller frees).
 * Returns 1 on success, 0 on failure. Windows only. */
int dc_mf_decode(const char *path_utf8, float **pcm_out, int *channels, long long *frames,
                 int *sample_rate);

#ifdef __cplusplus
}
#endif

#endif
