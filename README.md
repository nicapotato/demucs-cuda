# demucs-cuda

Inference-only [HTDemucs](https://github.com/facebookresearch/demucs) (`htdemucs_6s`) for
Windows x86-64 + NVIDIA, written in C and CUDA. No Python, no PyTorch, no cuBLAS/cuDNN/cuFFT:
GEMMs are CUTLASS templates compiled into the exe, everything else is a handful of custom kernels,
and cudart is linked statically. The only runtime dependency is the NVIDIA driver.

| | torch worker (demucs-windows) | this worker |
|---|---|---|
| worker | PyInstaller onedir, ~2.5 GB | `demucs_cuda_worker.exe`, single file |
| weights | fp32 torch checkpoint, 110 MB | fp16 safetensors, 55 MB |

The worker is a drop-in for the torch worker in [demucs-ui-app](../demucs-ui-app): same CLI and
`--progress-json` event stream.

## Layout

```
worker-cuda/
  src/            model (C): htdemucs.c layers.c transformer.c apply.c stft.c ops.c ...
                  backend.h: the kernel interface, implemented twice:
                    backend_cuda.cu gemm.cu norm.cu attn.cu fft.cu   (NVIDIA)
                    backend_cpu.c                                    (reference, macOS)
  tests/          test_stft, test_kernels, test_compare
  third_party/    cutlass (submodule), dr_libs, stb_vorbis
scripts/          export_fp16_safetensors.py, make_reference_tensors.py, package_release.py
```

Numerics: GEMM operands (conv / linear / attention weights and activations) are fp16, and all
accumulation, normalization, softmax statistics and the STFT run in fp32. Convs are im2col + GEMM;
transposed convs are GEMM + col2im. The 4096-point STFT is a shared-memory radix-2 FFT.

## Build (Windows)

Requirements: Visual Studio 2022 (C++ workload), CUDA Toolkit 12.8+, CMake 3.24+, Ninja, and
vcpkg for lame (mp3 output).

```powershell
git submodule update --init
vcpkg install mp3lame:x64-windows-static
# from an "x64 Native Tools" prompt:
cmake -S worker-cuda -B worker-cuda/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDC_BACKEND=cuda `
  -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows-static
cmake --build worker-cuda/build
```

`-DDC_CUDA_ARCHS="86"` builds a single architecture (much faster to compile while iterating).
The default `75;80;86;89;120` covers Turing through Blackwell, with PTX for forward compatibility.
GPUs older than Turing (sm_75) are rejected at startup.

## Weights

```sh
make models   # mac: ../demucs-mac-silicon/models/htdemucs_6s.safetensors -> models/htdemucs_6s.fp16.safetensors
```

The worker looks for `htdemucs_6s.fp16.safetensors` (or an fp32 `htdemucs_6s.safetensors`, which
it converts at load time) plus `htdemucs_6s.json` in `DEMUCS_CUDA_CACHE`, `DEMUCS_MLX_CACHE`,
`DEMUCS_TORCH_CACHE`, `<exe>/models`, `<exe>/../models`, then `./models`.

## Usage

```
demucs_cuda_worker [-n htdemucs_6s] [-o DIR] [--track-name NAME] [-b N] [--shifts N]
                   [--overlap 0.25] [--mp3 [--mp3-bitrate 320]] [--progress-json] track...
```

Input: wav, flac, mp3 and ogg are decoded in-process (dr_libs / stb_vorbis). On Windows,
Media Foundation is used as a fallback for anything else (m4a/aac/wma). Output:
`DIR/htdemucs_6s/<track>/{drums,bass,other,vocals,guitar,piano}.wav` (16-bit) or `.mp3`.

`-b` is capped automatically to fit free VRAM (about 640 MB per batch item; 8 GB cards run -b 8).
`DC_TIMING=1` logs per-phase timings and the device memory peak to stderr.

## Tests

The CPU backend runs the same model code with the same fp16 operand rounding, so most validation
happens on a Mac:

```sh
make test           # test_stft (vs naive DFT, spec/ispec round trip), test_kernels (vs double refs)
make test-compare   # per-layer + end-to-end comparison against reference tensors
```

On a CUDA machine, the same tests are built next to the worker:

```powershell
worker-cuda\build\test_kernels.exe
worker-cuda\build\test_stft.exe
worker-cuda\build\test_compare.exe models\htdemucs_6s.fp16.safetensors models\htdemucs_6s.json goldens\htdemucs_6s\goldens.safetensors
```

`test_compare` feeds each stage the reference output of the previous one, so a failure names the
layer at fault. Current results with fp16 operands: every stage ≤ 8e-4 relative L2 error and stems
4.7e-4 (66.5 dB SNR) against the fp32 reference. With fp32 operands (`-DDC_HALF_FLOAT=ON`, cpu only)
every stage matches to ~1e-6.

### Reference tensors: avoid the demucs_mlx Metal kernels

`scripts/make_reference_tensors.py` runs demucs-mac-silicon's `dump_goldens.py` with
`demucs_mlx.metal_kernels.HAS_METAL = False`. The fused GroupNorm+GELU / GroupNorm+GLU Metal
kernels reuse their threadgroup reduction buffer for the mean and the variance without a barrier
between reading the mean and overwriting it, which corrupts the many-small-group DConv layers in
the high-resolution frequency branch (encoder 0/1, decoder 2/3). Goldens produced with those
kernels are not a valid reference (and the MLX app's output is affected by the same race).

## Profiling

```powershell
nsys profile -o demucs --trace=cuda,nvtx worker-cuda\build\demucs_cuda_worker.exe -b 8 track.wav
ncu --set full -k regex:"Gemm|k_im2col|k_groupnorm" -c 20 worker-cuda\build\demucs_cuda_worker.exe track.wav
```

Expect the CUTLASS GEMMs (conv im2col GEMMs and the transformer) to dominate. Next steps if the
timeline shows launch gaps or im2col traffic: capture one segment batch in a CUDA Graph (the
allocator already reuses blocks, so addresses are stable after the first batch), and replace
im2col with implicit-GEMM convolution for the large frequency-branch layers.

## Releases

```sh
make models-release   # once per MODELS_RELEASE_TAG (project.conf): uploads htdemucs-6s-fp16.zip
make release-watch    # windows-latest builds the exe, packages demucs-cuda-windows-worker-vX.Y.Z.zip
make ci-watch
```

The worker zip unpacks to `worker\demucs_cuda_worker.exe` + `models\`, next to the
demucs-ui-app exe.

## License

MIT (see `LICENSE`). CUTLASS is BSD-3-Clause; dr_libs and stb_vorbis are public domain; lame is LGPL.
