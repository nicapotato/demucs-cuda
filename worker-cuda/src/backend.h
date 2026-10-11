#ifndef DEMUCS_CUDA_BACKEND_H
#define DEMUCS_CUDA_BACKEND_H

/* Kernel layer. Implemented by backend_cuda.cu (+ gemm/norm/fft .cu files) and
 * by backend_cpu.c (portable reference used for tests and macOS development).
 * All pointers are device pointers; activations are fp32, GEMM operands fp16.
 * Every call is asynchronous on the backend's single stream. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef DC_HALF_FLOAT
/* CPU numerics experiments: GEMM operands kept in fp32 (see backend_cpu.c) */
typedef float dc_half;
#else
typedef uint16_t dc_half;
#endif
/* float -> GEMM operand; `weight` selects the DC_HALF_FLOAT rounding switch */
dc_half dk_pack_half(float v, int weight);

#define DK_MAX_DIM 8

/* runtime */
int dk_init(int device, char *name, size_t name_cap);
const char *dk_backend_name(void);
void dk_sync(void);
void *dk_raw_malloc(size_t bytes); /* NULL on out-of-memory */
void dk_raw_free(void *p);
void dk_memcpy_h2d(void *dst, const void *src, size_t bytes);
void dk_memcpy_d2h(void *dst, const void *src, size_t bytes);
void dk_memcpy_d2d(void *dst, const void *src, size_t bytes);
void dk_memset0(void *dst, size_t bytes);
int dk_mem_info(size_t *free_bytes, size_t *total_bytes);
/* profiler ranges (NVTX on CUDA, no-ops elsewhere) */
void dk_range_push(const char *name);
void dk_range_pop(void);

/* strided elementwise */
typedef struct {
  int ndim;
  int64_t shape[DK_MAX_DIM];
  int64_t a_strides[DK_MAX_DIM]; /* src for copy */
  int64_t b_strides[DK_MAX_DIM]; /* dst for copy */
} DkStrided;

/* dst[idx . b_strides] (+)= src[idx . a_strides] over every index of shape */
void dk_strided_copy(const DkStrided *p, const float *src, float *dst, int accumulate);

typedef enum { DK_ADD = 0, DK_SUB, DK_MUL, DK_DIV, DK_MAX } DkBinOp;
/* y contiguous over shape; a and b addressed through their strides (0 = broadcast) */
void dk_binary(DkBinOp op, const DkStrided *p, const float *a, const float *b, float *y);

typedef enum { DK_UN_AFFINE = 0, DK_UN_GELU, DK_UN_RELU, DK_UN_SIGMOID } DkUnOp;
/* AFFINE: y = x * alpha + beta */
void dk_unary(DkUnOp op, const float *x, float *y, int64_t n, float alpha, float beta);
void dk_fill(float *y, int64_t n, float v);
/* x [outer, 2*half, inner] -> y [outer, half, inner] = a * sigmoid(b) */
void dk_glu(const float *x, float *y, int64_t outer, int64_t half, int64_t inner);
/* y[o, c, i] += bias[c] */
void dk_channel_bias(float *y, const float *bias, int64_t outer, int64_t C, int64_t inner);
/* y[o, c, i] += x[o, c, i] * scale[c]   (layer scale + residual) */
void dk_scale_add(float *y, const float *x, const float *scale, int64_t outer, int64_t C,
                  int64_t inner);
/* reflect padding of rows of length L (numpy "reflect", requires L > left, right) */
void dk_reflect_pad(const float *x, float *y, int64_t rows, int64_t L, int64_t left,
                    int64_t right);

/* normalisation */
/* per contiguous row of length L: mean and biased variance (fp32, two-pass) */
void dk_row_stats(const float *x, int64_t rows, int64_t L, float *mean, float *var);
/* channel-first group norm apply on x [B, C, S] with row stats over [B, G].
 * act: 0 none, 1 gelu, 2 glu over channels (y has C/2 channels). */
void dk_groupnorm_apply(const float *x, float *y, const float *mean, const float *var,
                        const float *w, const float *b, int64_t B, int64_t C, int64_t S,
                        int64_t G, float eps, int act);
/* rows of length L, affine channel = j % C (w/b may be NULL) */
void dk_norm_apply_last(const float *x, float *y, const float *mean, const float *var,
                        const float *w, const float *b, int64_t rows, int64_t L, int64_t C,
                        float eps);
/* denorm=0: y = (x - mean) / (sqrt(var) + eps); denorm=1: y = x * (sqrt(var) + eps) + mean */
void dk_affine_rows(const float *x, float *y, const float *mean, const float *var,
                    int64_t rows, int64_t L, float eps, int denorm);
/* fused layer norm over last dim C */
void dk_layernorm(const float *x, float *y, const float *w, const float *b, int64_t rows,
                  int64_t C, float eps);

/* GEMM operand preparation (fp16, K padded with zeros to Kp) */
/* y[r][c] = x[r][c] for c < C else 0;  y has row stride Cp */
void dk_to_half_rows(const float *x, dc_half *y, int64_t rows, int64_t C, int64_t Cp);
/* x [batch][C][P] -> y [batch][P][Cp] */
void dk_transpose_to_half(const float *x, dc_half *y, int64_t batch, int64_t C, int64_t P,
                          int64_t Cp);

typedef struct {
  int C, H, W;    /* input channels / height / width */
  int KH, KW;     /* kernel */
  int SH, SW;     /* stride */
  int PH, PW;     /* padding (conv) or output crop (transposed) */
  int DH, DW;     /* dilation */
  int Ho, Wo;     /* output spatial size */
  int64_t Kp;     /* padded reduction length */
} DkConvGeom;

/* x [N][C][H][W]; col [nb][np][Kp] with k = (kh*KW + kw)*C + c, for items
 * n0..n0+nb and output positions p0..p0+np (p = ho*Wo + wo) */
void dk_im2col_half(const float *x, dc_half *col, const DkConvGeom *g, int64_t n0, int64_t nb,
                    int64_t p0, int64_t np);
/* cols [nb][O*KH*KW][H*W] -> y [nb][O][Ho][Wo] (+ bias) for transposed convolution
 * with output crop (PH, PW): y[o][ho][wo] gathers full-position (ho+PH, wo+PW). */
void dk_col2im_tr(const float *cols, float *y, const float *bias, int64_t nb, int O,
                  const DkConvGeom *g);

/* C[b] = alpha * A[b] . B[b]^T + beta * C[b]
 * A [M][K] row-major (lda), B [N][K] row-major (ldb), C [M][N] fp32 (ldc).
 * K, lda, ldb must be multiples of 8. */
void dk_gemm_half(int64_t M, int64_t N, int64_t K, float alpha, const dc_half *A, int64_t lda,
                  int64_t strideA, const dc_half *B, int64_t ldb, int64_t strideB, float beta,
                  float *C, int64_t ldc, int64_t strideC, int64_t batch);

/* attention helpers */
/* softmax(scale * s) per row of length L (row stride lds), written as fp16 with
 * row stride ldp; columns L..ldp-1 are zeroed */
void dk_softmax_to_half(const float *s, int64_t lds, dc_half *p, int64_t ldp, int64_t rows,
                        int64_t L, float scale);
/* x [B][T][H*hd] -> transpose=0: y [B][H][T][hdp];  transpose=1: y [B][H][hd][Tp] */
void dk_split_heads_half(const float *x, dc_half *y, int64_t B, int64_t T, int H, int hd,
                         int hdp, int64_t Tp, int transpose);

/* spectral */
/* sig [R][L]; frames f = frame0..frame0+nframes-1 start at f*hop; Hann-windowed rFFT.
 * out [R][2][nbins][nframes]  (real plane, imag plane) */
void dk_stft(const float *sig, float *out, int64_t R, int64_t L, int n_fft, int hop,
             int frame0, int nframes, int nbins, const float *window);
/* spec [R][2][nbins][nframes] -> windowed time frames [R][nframes + 2*pad_frames][n_fft];
 * missing bins and the pad frames are zero */
void dk_istft_frames(const float *spec, float *frames, int64_t R, int nbins, int nframes,
                     int pad_frames, int n_fft, const float *window);
/* out[r][t] = OLA(frames)[start + t] / OLA(window^2)[start + t] */
void dk_ola(const float *frames, float *out, int64_t R, int nframes, int n_fft, int hop,
            const float *window, int64_t start, int64_t length);

#ifdef __cplusplus
}
#endif

#endif
