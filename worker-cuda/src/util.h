#ifndef DEMUCS_CUDA_UTIL_H
#define DEMUCS_CUDA_UTIL_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DC_PI 3.14159265358979323846
#define DC_MAX_DIM 8
#define DC_MAX_SOURCES 8
#define DC_PATH_MAX 1024

#if defined(_MSC_VER)
#define DC_NORETURN __declspec(noreturn)
#else
#define DC_NORETURN _Noreturn
#endif

DC_NORETURN void dc_die(const char *fmt, ...);
/* when enabled, dc_die also prints {"event":"error"} on stdout */
void dc_set_json_errors(int enabled);
void dc_json_escape(char *out, size_t cap, const char *s);

double dc_now_s(void);
int dc_timing_enabled(void);
void dc_time_log(const char *stage, double t0);

/* UTF-8 paths on every platform (wide APIs on Windows). */
FILE *dc_fopen(const char *path, const char *mode);
int dc_file_exists(const char *path);
void dc_mkdir_p(const char *path);
void *dc_read_file(const char *path, size_t *len_out);
/* Directory containing the running executable, without trailing separator. */
int dc_exe_dir(char *out, size_t cap);
void dc_path_join(char *out, size_t cap, const char *a, const char *b);
const char *dc_path_basename(const char *path);

static inline int64_t dc_round_up(int64_t v, int64_t m) { return (v + m - 1) / m * m; }

uint16_t dc_f32_to_f16(float f);
float dc_f16_to_f32(uint16_t h);

#endif
