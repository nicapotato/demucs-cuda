#include "stft.h"

#include <math.h>

DcStft dc_stft_new(int n_fft, int hop) {
  DcStft st;
  st.n_fft = n_fft;
  st.hop = hop;
  float *w = (float *)malloc((size_t)n_fft * sizeof(float));
  if (!w) {
    dc_die("oom hann window");
  }
  for (int i = 0; i < n_fft; i++) {
    w[i] = (float)(0.5 - 0.5 * cos(2.0 * DC_PI * (double)i / (double)n_fft));
  }
  int shape[1] = {n_fft};
  st.window = dc_from_host(w, shape, 1);
  free(w);
  return st;
}

void dc_stft_free(DcStft *st) {
  if (st) {
    dc_free(st->window);
    st->window = dc_empty();
  }
}

DcTensor dc_htdemucs_spec_cac(DcStft *st, DcTensor mix) {
  int B = mix.shape[0], C = mix.shape[1], T = mix.shape[2];
  int hl = st->hop;
  int le = (T + hl - 1) / hl;
  int pad = (hl / 2) * 3;
  int extra = pad + le * hl - T;
  DcTensor xp = dc_pad1d(mix, pad, extra, 1);
  /* torch.stft(center=True) reflect padding */
  DcTensor xc = dc_pad1d(xp, st->n_fft / 2, st->n_fft / 2, 1);
  dc_free(xp);
  int L = xc.shape[2];
  int nbins = st->n_fft / 2; /* HTDemucs drops the Nyquist bin */
  int oshape[4] = {B, 2 * C, nbins, le};
  DcTensor z = dc_new(oshape, 4);
  dk_stft(dc_data(xc), dc_data(z), (int64_t)B * C, L, st->n_fft, hl, 2, le, nbins,
          dc_data(st->window));
  dc_free(xc);
  return z;
}

DcTensor dc_htdemucs_ispec_cac(DcStft *st, DcTensor m, int length) {
  if (m.ndim != 5 || m.shape[2] % 2) {
    dc_print_shape("ispec input", m);
    dc_die("ispec expects [B, S, 2C, F, N]");
  }
  int B = m.shape[0], S = m.shape[1], C = m.shape[2] / 2, F = m.shape[3], N = m.shape[4];
  int64_t R = (int64_t)B * S * C;
  int pad_frames = 2;
  int NF = N + 2 * pad_frames;
  int fshape[3] = {(int)R, NF, st->n_fft};
  DcTensor frames = dc_new(fshape, 3);
  dk_istft_frames(dc_data(m), dc_data(frames), R, F, N, pad_frames, st->n_fft,
                  dc_data(st->window));
  int pad = (st->hop / 2) * 3;
  int oshape[4] = {B, S, C, length};
  DcTensor y = dc_new(oshape, 4);
  dk_ola(dc_data(frames), dc_data(y), R, NF, st->n_fft, st->hop, dc_data(st->window),
         (int64_t)st->n_fft / 2 + pad, length);
  dc_free(frames);
  return y;
}
