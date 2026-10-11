/* C[b] = alpha * A[b] . B[b]^T + beta * C[b];  A [M][K] fp16, B [N][K] fp16, C fp32.
 * CUTLASS tensor-core kernels (fp32 accumulation): Sm80 multistage for sm_80+,
 * Sm75 for Turing. The threadblock tile narrows with M because the conv weights in
 * DConv have very few output channels. A plain tiled kernel covers anything the
 * tensor-core path cannot take (misaligned pointers, huge batch). */
#include "cuda_common.cuh"

#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_batched.h"

namespace {

using ElemIn = cutlass::half_t;
using ElemAcc = float;
using Epilogue = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;

template <typename Arch, typename TB, typename Warp, typename Inst, int Stages>
using BatchedGemm = cutlass::gemm::device::GemmBatched<
    ElemIn, cutlass::layout::RowMajor, ElemIn, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, ElemAcc, cutlass::arch::OpClassTensorOp, Arch, TB, Warp, Inst,
    Epilogue, cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages, 8, 8>;

using Inst80 = cutlass::gemm::GemmShape<16, 8, 16>;
using Inst75 = cutlass::gemm::GemmShape<16, 8, 8>;

using G80_128 = BatchedGemm<cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
                            cutlass::gemm::GemmShape<64, 64, 32>, Inst80, 3>;
using G80_64 = BatchedGemm<cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 128, 32>,
                           cutlass::gemm::GemmShape<32, 64, 32>, Inst80, 3>;
using G80_32 = BatchedGemm<cutlass::arch::Sm80, cutlass::gemm::GemmShape<32, 128, 32>,
                           cutlass::gemm::GemmShape<32, 32, 32>, Inst80, 3>;
using G75_128 = BatchedGemm<cutlass::arch::Sm75, cutlass::gemm::GemmShape<128, 128, 32>,
                            cutlass::gemm::GemmShape<64, 64, 32>, Inst75, 2>;
using G75_64 = BatchedGemm<cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 128, 32>,
                           cutlass::gemm::GemmShape<32, 64, 32>, Inst75, 2>;
using G75_32 = BatchedGemm<cutlass::arch::Sm75, cutlass::gemm::GemmShape<32, 128, 32>,
                           cutlass::gemm::GemmShape<32, 32, 32>, Inst75, 2>;

int g_sm = -1;

int device_sm(void) {
  if (g_sm < 0) {
    int dev = 0, major = 0, minor = 0;
    DK_CHECK(cudaGetDevice(&dev));
    DK_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    DK_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    g_sm = major * 10 + minor;
  }
  return g_sm;
}

template <typename Gemm>
bool run_cutlass(int64_t M, int64_t N, int64_t K, float alpha, const dc_half *A, int64_t lda,
                 int64_t strideA, const dc_half *B, int64_t ldb, int64_t strideB, float beta,
                 float *C, int64_t ldc, int64_t strideC, int64_t batch) {
  typename Gemm::Arguments args(
      cutlass::gemm::GemmCoord((int)M, (int)N, (int)K),
      {reinterpret_cast<const ElemIn *>(A), lda}, strideA,
      {reinterpret_cast<const ElemIn *>(B), ldb}, strideB, {C, ldc}, strideC, {C, ldc}, strideC,
      {alpha, beta}, (int)batch);
  Gemm op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) {
    return false;
  }
  cutlass::Status st = op(args, nullptr, dk_stream());
  if (st != cutlass::Status::kSuccess) {
    dc_die("CUTLASS GEMM failed (%s) M=%lld N=%lld K=%lld batch=%lld",
           cutlassGetStatusString(st), (long long)M, (long long)N, (long long)K,
           (long long)batch);
  }
  DK_LAUNCH_CHECK();
  return true;
}

/* generic fallback: 64x64 tile, 16-deep K slices, 4x4 outputs per thread */
#define FB_T 64
#define FB_K 16

__global__ void k_gemm_fallback(int64_t M, int64_t N, int64_t K, float alpha,
                                const __half *__restrict__ A, int64_t lda, int64_t strideA,
                                const __half *__restrict__ B, int64_t ldb, int64_t strideB,
                                float beta, float *__restrict__ C, int64_t ldc,
                                int64_t strideC) {
  __shared__ float As[FB_K][FB_T + 1];
  __shared__ float Bs[FB_K][FB_T + 1];
  int64_t b = blockIdx.z;
  A += b * strideA;
  B += b * strideB;
  C += b * strideC;
  int64_t m0 = (int64_t)blockIdx.y * FB_T, n0 = (int64_t)blockIdx.x * FB_T;
  int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  float acc[4][4] = {};
  for (int64_t k0 = 0; k0 < K; k0 += FB_K) {
    for (int i = threadIdx.x; i < FB_K * FB_T; i += 256) {
      int kk = i % FB_K, r = i / FB_K;
      int64_t m = m0 + r, n = n0 + r, k = k0 + kk;
      As[kk][r] = (m < M && k < K) ? __half2float(A[m * lda + k]) : 0.0f;
      Bs[kk][r] = (n < N && k < K) ? __half2float(B[n * ldb + k]) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < FB_K; kk++) {
      float a[4], bv[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        a[i] = As[kk][ty * 4 + i];
        bv[i] = Bs[kk][tx * 4 + i];
      }
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] += a[i] * bv[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int64_t m = m0 + ty * 4 + i;
    if (m >= M) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int64_t n = n0 + tx * 4 + j;
      if (n < N) {
        float prev = beta != 0.0f ? beta * C[m * ldc + n] : 0.0f;
        C[m * ldc + n] = alpha * acc[i][j] + prev;
      }
    }
  }
}

void run_fallback(int64_t M, int64_t N, int64_t K, float alpha, const dc_half *A, int64_t lda,
                  int64_t strideA, const dc_half *B, int64_t ldb, int64_t strideB, float beta,
                  float *C, int64_t ldc, int64_t strideC, int64_t batch) {
  for (int64_t b0 = 0; b0 < batch; b0 += 65535) {
    int64_t nb = batch - b0 < 65535 ? batch - b0 : 65535;
    dim3 grid((unsigned)((N + FB_T - 1) / FB_T), (unsigned)((M + FB_T - 1) / FB_T),
              (unsigned)nb);
    k_gemm_fallback<<<grid, 256, 0, dk_stream()>>>(
        M, N, K, alpha, (const __half *)A + b0 * strideA, lda, strideA,
        (const __half *)B + b0 * strideB, ldb, strideB, beta, C + b0 * strideC, ldc, strideC);
    DK_LAUNCH_CHECK();
  }
}

bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

bool force_fallback(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("DC_GEMM_FALLBACK");
    v = e && e[0] == '1';
  }
  return v != 0;
}

}  // namespace

extern "C" void dk_gemm_half(int64_t M, int64_t N, int64_t K, float alpha, const dc_half *A,
                             int64_t lda, int64_t strideA, const dc_half *B, int64_t ldb,
                             int64_t strideB, float beta, float *C, int64_t ldc,
                             int64_t strideC, int64_t batch) {
  if (M == 0 || N == 0 || batch == 0) {
    return;
  }
  bool tc = (K % 8 == 0) && (lda % 8 == 0) && (ldb % 8 == 0) && (strideA % 8 == 0) &&
            (strideB % 8 == 0) && aligned16(A) && aligned16(B) && batch <= 65535 &&
            M < (1LL << 31) && N < (1LL << 31) && K < (1LL << 31) &&
            !force_fallback();
  if (tc) {
    bool ok;
    if (device_sm() >= 80) {
      ok = M > 64   ? run_cutlass<G80_128>(M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
                                         beta, C, ldc, strideC, batch)
           : M > 32 ? run_cutlass<G80_64>(M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
                                          beta, C, ldc, strideC, batch)
                    : run_cutlass<G80_32>(M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
                                          beta, C, ldc, strideC, batch);
    } else {
      ok = M > 64   ? run_cutlass<G75_128>(M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
                                         beta, C, ldc, strideC, batch)
           : M > 32 ? run_cutlass<G75_64>(M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
                                          beta, C, ldc, strideC, batch)
                    : run_cutlass<G75_32>(M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
                                          beta, C, ldc, strideC, batch);
    }
    if (ok) {
      return;
    }
  }
  run_fallback(M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch);
}
