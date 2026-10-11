#!/usr/bin/env python3
"""Convert the worker-c MLX-layout fp32 safetensors to the fp16 file the CUDA worker ships.

Tensors the worker uses as GEMM operands (every rank >= 2 tensor except embeddings)
are stored as F16 with round-to-nearest-even, exactly what the C loader would do to
an fp32 file, so both inputs give bit-identical device weights. Biases, norms, layer
scales and the frequency embedding stay F32 (they are a fraction of a percent of the
parameters). numpy is the only dependency.

usage: export_fp16_safetensors.py IN.safetensors IN.json OUT_DIR [--name htdemucs_6s]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": None}


def read_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    raw = path.read_bytes()
    (hlen,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + hlen])
    meta = header.pop("__metadata__", {}) or {}
    base = 8 + hlen
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        a, b = info["data_offsets"]
        buf = raw[base + a : base + b]
        dt = info["dtype"]
        if dt == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        elif dt in DTYPES:
            arr = np.frombuffer(buf, dtype=DTYPES[dt])
        else:
            raise SystemExit(f"unsupported dtype {dt} for {name}")
        out[name] = arr.reshape(info["shape"]).astype(np.float32)
    return out, meta


def is_gemm_weight(name: str, arr: np.ndarray) -> bool:
    return arr.ndim >= 2 and "embedding" not in name


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], meta: dict[str, str]) -> None:
    header: dict[str, object] = {"__metadata__": meta}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        dt = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16"}[arr.dtype]
        data = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    hjson = json.dumps(header, separators=(",", ":")).encode()
    hjson += b" " * ((8 - len(hjson) % 8) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for blob in blobs:
            f.write(blob)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--name", default="htdemucs_6s")
    args = ap.parse_args(argv)

    tensors, meta = read_safetensors(args.weights)
    out: dict[str, np.ndarray] = {}
    n16 = n32 = 0
    for name, arr in tensors.items():
        if is_gemm_weight(name, arr):
            out[name] = arr.astype(np.float16)
            n16 += arr.size
        else:
            out[name] = arr
            n32 += arr.size
    if not np.all(np.isfinite(np.concatenate([v.astype(np.float32).ravel() for v in out.values()]))):
        raise SystemExit("non-finite value after fp16 conversion (weight out of fp16 range)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dst = args.out_dir / f"{args.name}.fp16.safetensors"
    meta = {str(k): str(v) for k, v in meta.items()}
    meta.update({"format": "demucs-cuda-fp16", "source_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest()})
    write_safetensors(dst, out, meta)
    shutil.copyfile(args.config, args.out_dir / f"{args.name}.json")
    print(
        f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB; {n16 / 1e6:.2f}M fp16 + {n32 / 1e3:.1f}k fp32 params, "
        f"{len(out)} tensors)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
