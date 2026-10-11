/* STFT kernel vs a naive DFT, and the HTDemucs spec -> ispec round trip. */
#include "stft.h"

#include <math.h>

static int g_fail = 0;

static void report(const char *name, double rel, double gate) {
  int ok = rel <= gate && !isnan(rel);
  printf("%-30s rel_l2=%.3e gate=%.0e %s\n", name, rel, gate, ok ? "ok" : "FAIL");
  g_fail |= !ok;
}

static void test_dft(int n_fft, int hop, int L, int nframes) {
  int R = 2;
  int nbins = n_fft / 2 + 1;
  float *sig = (float *)malloc((size_t)R * L * sizeof(float));
  for (int r = 0; r < R; r++)
    for (int t = 0; t < L; t++)
      sig[(size_t)r * L + t] = (float)(sin(0.013 * t * (r + 1)) + 0.3 * cos(0.31 * t) +
                                       0.05 * ((t * 7919 % 101) / 50.0 - 1.0));
  DcStft st = dc_stft_new(n_fft, hop);
  int ss[2] = {R, L};
  DcTensor s = dc_from_host(sig, ss, 2);
  int os[4] = {R, 2, nbins, nframes};
  DcTensor out = dc_new(os, 4);
  dk_stft(dc_data(s), dc_data(out), R, L, n_fft, hop, 0, nframes, nbins, dc_data(st.window));
  float *got = dc_to_host(out);
  double num = 0, den = 0;
  for (int r = 0; r < R; r++)
    for (int f = 0; f < nframes; f++)
      for (int k = 0; k < nbins; k++) {
        double re = 0, im = 0;
        for (int n = 0; n < n_fft; n++) {
          int t = f * hop + n;
          if (t >= L) break;
          double wv = 0.5 - 0.5 * cos(2.0 * DC_PI * n / n_fft);
          double a = -2.0 * DC_PI * (double)k * n / n_fft;
          re += sig[(size_t)r * L + t] * wv * cos(a);
          im += sig[(size_t)r * L + t] * wv * sin(a);
        }
        double gr = got[(((size_t)r * 2 + 0) * nbins + k) * nframes + f];
        double gi = got[(((size_t)r * 2 + 1) * nbins + k) * nframes + f];
        num += (gr - re) * (gr - re) + (gi - im) * (gi - im);
        den += re * re + im * im;
      }
  char name[64];
  snprintf(name, sizeof(name), "stft vs dft n_fft=%d", n_fft);
  report(name, sqrt(num / den), 1e-5);
  free(got);
  free(sig);
  dc_free(s);
  dc_free(out);
  dc_stft_free(&st);
}

static void test_roundtrip(int T) {
  int C = 2;
  float *mix = (float *)malloc((size_t)C * T * sizeof(float));
  /* band-limited: HTDemucs drops the Nyquist bin, so white noise cannot round trip */
  for (int c = 0; c < C; c++)
    for (int t = 0; t < T; t++) {
      double v = 0;
      for (int k = 1; k <= 12; k++) {
        v += sin(t * 0.0371 * k * (c + 1) + 0.7 * k) / k;
      }
      mix[(size_t)c * T + t] = (float)(0.3 * v);
    }
  DcStft st = dc_stft_new(4096, 1024);
  int ms[3] = {1, C, T};
  DcTensor m = dc_from_host(mix, ms, 3);
  DcTensor z = dc_htdemucs_spec_cac(&st, m); /* [1, 2C, F, N] */
  int r5[5] = {1, 1, 2 * C, z.shape[2], z.shape[3]};
  DcTensor z5 = dc_reshape(z, r5, 5);
  DcTensor y = dc_htdemucs_ispec_cac(&st, z5, T); /* [1, 1, C, T] */
  float *got = dc_to_host(y);
  /* HTDemucs drops the two outer STFT frames on each side, so the first and last
   * ~n_fft/2 samples are not reconstructible by design; check the interior. */
  double num = 0, den = 0, worst = 0;
  int worst_i = 0;
  for (int i = 0; i < C * T; i++) {
    int t = i % T;
    if (t < 4096 || t >= T - 4096) {
      continue;
    }
    double d = (double)got[i] - mix[i];
    num += d * d;
    den += (double)mix[i] * mix[i];
    if (fabs(d) > worst) {
      worst = fabs(d);
      worst_i = i;
    }
  }
  if (sqrt(num / den) > 1e-5) {
    printf("  worst |err|=%.3e at channel %d t=%d (got %.5f want %.5f)\n", worst, worst_i / T,
           worst_i % T, got[worst_i], mix[worst_i]);
  }
  char name[64];
  snprintf(name, sizeof(name), "spec/ispec round trip T=%d", T);
  report(name, sqrt(num / den), 1e-5);
  free(got);
  free(mix);
  dc_free(m);
  dc_free(z);
  dc_free(z5);
  dc_free(y);
  dc_stft_free(&st);
}

int main(void) {
  char dev[256];
  if (dk_init(0, dev, sizeof(dev)) != 0) {
    dc_die("no device: %s", dev);
  }
  printf("backend %s (%s)\n", dk_backend_name(), dev);
  test_dft(64, 16, 400, 20);
  test_dft(4096, 1024, 4096 * 3, 9);
  test_roundtrip(343980);
  test_roundtrip(44100);
  printf(g_fail ? "test_stft: FAIL\n" : "test_stft: OK\n");
  return g_fail;
}
