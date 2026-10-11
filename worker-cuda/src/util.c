#include "util.h"

#include <math.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

static int g_json_errors = 0;

void dc_set_json_errors(int enabled) { g_json_errors = enabled; }

void dc_json_escape(char *out, size_t cap, const char *s) {
  size_t o = 0;
  for (; s && *s && o + 7 < cap; s++) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') {
      out[o++] = '\\';
      out[o++] = (char)c;
    } else if (c < 0x20) {
      o += (size_t)snprintf(out + o, cap - o, "\\u%04x", c);
    } else {
      out[o++] = (char)c;
    }
  }
  out[o] = 0;
}

DC_NORETURN void dc_die(const char *fmt, ...) {
  char msg[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);
  fprintf(stderr, "demucs-cuda: %s\n", msg);
  fflush(stderr);
  if (g_json_errors) {
    char esc[2048];
    dc_json_escape(esc, sizeof(esc), msg);
    fprintf(stdout, "{\"event\":\"error\",\"message\":\"%s\"}\n", esc);
    fflush(stdout);
  }
  exit(1);
}

double dc_now_s(void) {
#ifdef _WIN32
  static LARGE_INTEGER freq;
  LARGE_INTEGER now;
  if (freq.QuadPart == 0) {
    QueryPerformanceFrequency(&freq);
  }
  QueryPerformanceCounter(&now);
  return (double)now.QuadPart / (double)freq.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

int dc_timing_enabled(void) {
  static int cached = -1;
  if (cached < 0) {
    const char *e = getenv("DC_TIMING");
    cached = e && e[0] && e[0] != '0';
  }
  return cached;
}

void dc_time_log(const char *stage, double t0) {
  if (dc_timing_enabled()) {
    fprintf(stderr, "dc_timing %s %.3fs\n", stage, dc_now_s() - t0);
  }
}

#ifdef _WIN32
static wchar_t *utf8_to_wide(const char *s) {
  int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
  if (n <= 0) {
    return NULL;
  }
  wchar_t *w = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
  if (w && MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n) <= 0) {
    free(w);
    return NULL;
  }
  return w;
}
#endif

FILE *dc_fopen(const char *path, const char *mode) {
#ifdef _WIN32
  wchar_t *wp = utf8_to_wide(path);
  wchar_t *wm = utf8_to_wide(mode);
  FILE *fp = (wp && wm) ? _wfopen(wp, wm) : NULL;
  free(wp);
  free(wm);
  return fp;
#else
  return fopen(path, mode);
#endif
}

int dc_file_exists(const char *path) {
  if (!path || !path[0]) {
    return 0;
  }
#ifdef _WIN32
  wchar_t *wp = utf8_to_wide(path);
  DWORD attr = wp ? GetFileAttributesW(wp) : INVALID_FILE_ATTRIBUTES;
  free(wp);
  return attr != INVALID_FILE_ATTRIBUTES;
#else
  return access(path, F_OK) == 0;
#endif
}

static void mkdir_one(const char *p) {
#ifdef _WIN32
  wchar_t *wp = utf8_to_wide(p);
  if (wp) {
    CreateDirectoryW(wp, NULL);
    free(wp);
  }
#else
  mkdir(p, 0755);
#endif
}

void dc_mkdir_p(const char *path) {
  char buf[DC_PATH_MAX];
  snprintf(buf, sizeof(buf), "%s", path);
  for (char *p = buf + 1; *p; p++) {
    if (*p == '/' || *p == '\\') {
      char c = *p;
      *p = 0;
      if (!(p - buf == 2 && buf[1] == ':')) { /* skip bare drive "C:" */
        mkdir_one(buf);
      }
      *p = c;
    }
  }
  mkdir_one(buf);
}

void *dc_read_file(const char *path, size_t *len_out) {
  FILE *fp = dc_fopen(path, "rb");
  if (!fp) {
    return NULL;
  }
#ifdef _WIN32
  _fseeki64(fp, 0, SEEK_END);
  int64_t n = _ftelli64(fp);
#else
  fseeko(fp, 0, SEEK_END);
  int64_t n = (int64_t)ftello(fp);
#endif
  if (n < 0) {
    fclose(fp);
    return NULL;
  }
  rewind(fp);
  unsigned char *buf = (unsigned char *)malloc((size_t)n + 1);
  if (!buf) {
    fclose(fp);
    dc_die("oom reading %s (%lld bytes)", path, (long long)n);
  }
  if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
    fclose(fp);
    free(buf);
    return NULL;
  }
  buf[n] = 0;
  fclose(fp);
  if (len_out) {
    *len_out = (size_t)n;
  }
  return buf;
}

int dc_exe_dir(char *out, size_t cap) {
  char exe[DC_PATH_MAX];
  exe[0] = 0;
#ifdef _WIN32
  wchar_t wexe[DC_PATH_MAX];
  DWORD n = GetModuleFileNameW(NULL, wexe, DC_PATH_MAX);
  if (n == 0 || n >= DC_PATH_MAX ||
      WideCharToMultiByte(CP_UTF8, 0, wexe, -1, exe, (int)sizeof(exe), NULL, NULL) <= 0) {
    return -1;
  }
#elif defined(__APPLE__)
  uint32_t n = sizeof(exe);
  if (_NSGetExecutablePath(exe, &n) != 0) {
    return -1;
  }
#else
  ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (n <= 0) {
    return -1;
  }
  exe[n] = 0;
#endif
  snprintf(out, cap, "%s", exe);
  char *slash = strrchr(out, '/');
  char *bslash = strrchr(out, '\\');
  if (!slash || (bslash && bslash > slash)) {
    slash = bslash;
  }
  if (slash) {
    *slash = 0;
  } else {
    snprintf(out, cap, ".");
  }
  return 0;
}

void dc_path_join(char *out, size_t cap, const char *a, const char *b) {
#ifdef _WIN32
  snprintf(out, cap, "%s\\%s", a, b);
#else
  snprintf(out, cap, "%s/%s", a, b);
#endif
}

const char *dc_path_basename(const char *path) {
  const char *slash = strrchr(path, '/');
  const char *bslash = strrchr(path, '\\');
  if (!slash || (bslash && bslash > slash)) {
    slash = bslash;
  }
  return slash ? slash + 1 : path;
}

uint16_t dc_f32_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t exp = (x >> 23) & 0xffu;
  uint32_t mant = x & 0x7fffffu;
  if (exp == 0xffu) {
    return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
  }
  int32_t e = (int32_t)exp - 127 + 15;
  if (e >= 31) {
    return (uint16_t)(sign | 0x7c00u);
  }
  if (e <= 0) {
    if (e < -10) {
      return (uint16_t)sign;
    }
    mant |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - e);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    uint32_t mid = 1u << (shift - 1u);
    if (rem > mid || (rem == mid && (half & 1u))) {
      half++;
    }
    return (uint16_t)(sign | half);
  }
  uint32_t half = sign | ((uint32_t)e << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) {
    half++; /* carry into exponent is the correct rounding */
  }
  return (uint16_t)half;
}

float dc_f16_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
  uint32_t mant = (uint32_t)h & 0x3ffu;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {
      int e = -1;
      do {
        e++;
        mant <<= 1;
      } while (!(mant & 0x400u));
      mant &= 0x3ffu;
      x = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    x = sign | 0x7f800000u | (mant << 13);
  } else {
    x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &x, 4);
  return f;
}
