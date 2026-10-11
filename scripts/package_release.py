#!/usr/bin/env python3
"""Assemble release-assets/ for the native CUDA worker.

  demucs-cuda-windows-worker-vX.Y.Z.zip   worker/demucs_cuda_worker.exe + models/ + licenses
  htdemucs-6s-fp16.zip                    models/htdemucs_6s.fp16.safetensors + .json

The worker zip unpacks next to the UI exe (worker\\ and models\\ siblings), which is
where demucs-ui-app looks for both.
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_version() -> str:
    v = os.environ.get("VERSION", "").strip()
    if v:
        return v
    for line in (ROOT / "project.conf").read_text().splitlines():
        if line.startswith("VERSION="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("VERSION missing from project.conf")


def add(z: zipfile.ZipFile, src: Path, arc: str) -> None:
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    z.write(src, arc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=ROOT / "worker-cuda" / "build" / "demucs_cuda_worker.exe")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--out", type=Path, default=ROOT / "release-assets")
    ap.add_argument("--name", default="htdemucs_6s")
    args = ap.parse_args(argv)

    version = read_version()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = args.models / f"{args.name}.fp16.safetensors"
    config = args.models / f"{args.name}.json"
    licenses = [
        (ROOT / "LICENSE", "LICENSE.txt"),
        (ROOT / "worker-cuda" / "third_party" / "cutlass" / "LICENSE.txt", "licenses/CUTLASS.txt"),
    ]

    worker_zip = args.out / f"demucs-cuda-windows-worker-v{version}.zip"
    with zipfile.ZipFile(worker_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        add(z, args.exe, "worker/demucs_cuda_worker.exe")
        add(z, weights, f"models/{weights.name}")
        add(z, config, f"models/{config.name}")
        for src, arc in licenses:
            if src.is_file():
                z.write(src, arc)

    models_zip = args.out / "htdemucs-6s-fp16.zip"
    with zipfile.ZipFile(models_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        add(z, weights, f"models/{weights.name}")
        add(z, config, f"models/{config.name}")

    for p in (worker_zip, models_zip):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")
    exe_mb = args.exe.stat().st_size / 1e6
    print(f"worker exe: {exe_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
