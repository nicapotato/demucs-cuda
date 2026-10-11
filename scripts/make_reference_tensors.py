#!/usr/bin/env python3
"""Regenerate the per-layer reference tensors used by worker-cuda/tests/test_compare.

Runs demucs-mac-silicon's scripts/dump_goldens.py with demucs_mlx's custom Metal
kernels disabled. Those fused GroupNorm+GELU/GLU kernels reuse their shared
reduction buffer between the mean and variance passes without a barrier, so the
frequency-branch DConv outputs (many small groups) are corrupted run to run; the
pure-MLX fallbacks give a deterministic fp32 reference.

usage (macOS, demucs-mac-silicon venv):
  make_reference_tensors.py --mac-silicon ../demucs-mac-silicon -o goldens/htdemucs_6s
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mac-silicon", type=Path, default=Path(__file__).resolve().parents[2] / "demucs-mac-silicon")
    ap.add_argument("-o", "--out", type=Path, default=Path("goldens/htdemucs_6s"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    root = args.mac_silicon.resolve()
    script = root / "scripts" / "dump_goldens.py"
    if not script.is_file():
        raise SystemExit(f"{script} not found (pass --mac-silicon)")
    sys.path.insert(0, str(root))
    import demucs_mlx.metal_kernels as mk  # noqa: E402

    mk.HAS_METAL = False
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(script), "-o", str(out), "--seed", str(args.seed)]
    os.chdir(root)
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
