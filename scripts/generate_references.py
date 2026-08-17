#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Generate the PyTorch reference tensors for the pytorch_parity suite.

The consumer is crates/nn-reftest/tests/pytorch_parity.rs, which loads
`test_data/references/<group>/<case>/{input_N.npy,output_0.npy}` and compares
nn DynTensor ops against the PyTorch-computed outputs. When the directory is
absent every test in that suite prints a skip line and returns (graceful skip,
no #[ignore] — ait#341), so this script is what turns the suite from
skip-as-green into a real parity gate.

The output tree is gitignored (`test_data/references/` — regenerate at will);
determinism comes from the fixed torch seed, not from committing tensors.

Contract (each case mirrors one test in pytorch_parity.rs — keep in lockstep):

  elementwise/{relu,gelu,silu,sigmoid,tanh,exp,softmax,log}
      input_0.npy, output_0.npy.
      gelu uses approximate='none' (erf) — the harness calls nn's gelu_erf().
      softmax is over dim=-1. log stores the RAW x as input_0; the harness
      replicates log(abs(x) + 1e-6) itself.
  matmul/{basic,batched}       input_0.npy @ input_1.npy -> output_0.npy
  binary/{add,sub,mul,div,add_broadcast}
      input_0.npy (op) input_1.npy -> output_0.npy. div's denominator is kept
      away from zero (|b| >= 0.5) so the 1e-4 band is meaningful.
  reduction/{sum_dim1,mean_dim1,max_dim1}   keepdim=True over dim 1
  shape/transpose              x.transpose(1, 2) on shape [2, 3, 4]
  shape/reshape                x.reshape(6, 4) on shape [2, 3, 4]
  shape/cat                    torch.cat([input_0, input_1], dim=0)
  norm/layernorm               input_0.npy, output_0.npy, metadata.json
                               (weight/bias lists + eps for nn's LayerNorm)

Every .npy is written float32, C-order, NumPy format v1.0 — the exact subset
nn_reftest::load_npy parses — and the script re-opens each written header to
prove it (fail-closed) before reporting success.

Usage: python3 scripts/generate_references.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit(
        "generate_references.py: numpy is required.\n"
        "  Install: python3 -m pip install numpy"
    )

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit(
        "generate_references.py: PyTorch is required — the whole point of the\n"
        "pytorch_parity suite is that the references come from torch, not from\n"
        "a reimplementation.\n"
        "  Install: python3 -m pip install torch\n"
        "Until it runs, crates/nn-reftest/tests/pytorch_parity.rs skips every test."
    )

SEED = 20260615  # fixed: references are deterministic per torch version


def default_out_dir() -> Path:
    """Repo-root test_data/references — the path pytorch_parity.rs computes."""
    return Path(__file__).resolve().parent.parent / "test_data" / "references"


def save_npy(path: Path, tensor: torch.Tensor) -> None:
    """Write float32 C-order NumPy v1.0 — the subset the Rust loader parses."""
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr, allow_pickle=False)


def write_case(root: Path, group: str, case: str,
               inputs: list[torch.Tensor], output: torch.Tensor,
               metadata: dict | None = None) -> list[Path]:
    d = root / group / case
    written = []
    for i, t in enumerate(inputs):
        p = d / f"input_{i}.npy"
        save_npy(p, t)
        written.append(p)
    p = d / "output_0.npy"
    save_npy(p, output)
    written.append(p)
    if metadata is not None:
        mp = d / "metadata.json"
        mp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        written.append(mp)
    return written


def generate(root: Path) -> list[Path]:
    torch.manual_seed(SEED)
    written: list[Path] = []

    def case(group: str, name: str, inputs: list[torch.Tensor],
             output: torch.Tensor, metadata: dict | None = None) -> None:
        written.extend(write_case(root, group, name, inputs, output, metadata))

    # ── elementwise ────────────────────────────────────────────────────────
    x = torch.randn(2, 3, 4)
    case("elementwise", "relu", [x], torch.relu(x))
    x = torch.randn(2, 3, 4)
    # PyTorch default approximate='none' (erf) — harness compares gelu_erf().
    case("elementwise", "gelu", [x], F.gelu(x))
    x = torch.randn(2, 3, 4)
    case("elementwise", "silu", [x], F.silu(x))
    x = torch.randn(2, 3, 4)
    case("elementwise", "sigmoid", [x], torch.sigmoid(x))
    x = torch.randn(2, 3, 4)
    case("elementwise", "tanh", [x], torch.tanh(x))
    x = torch.randn(2, 3, 4)
    case("elementwise", "exp", [x], torch.exp(x))
    x = torch.randn(2, 3, 4)
    case("elementwise", "softmax", [x], torch.softmax(x, dim=-1))
    # log: input_0 is the RAW x; the harness applies abs + 1e-6 itself.
    x = torch.randn(2, 3, 4)
    case("elementwise", "log", [x], torch.log(torch.abs(x) + 1e-6))

    # ── matmul ─────────────────────────────────────────────────────────────
    a, b = torch.randn(4, 5), torch.randn(5, 6)
    case("matmul", "basic", [a, b], a @ b)
    a, b = torch.randn(2, 3, 4), torch.randn(2, 4, 5)
    case("matmul", "batched", [a, b], torch.matmul(a, b))

    # ── binary ─────────────────────────────────────────────────────────────
    a, b = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    case("binary", "add", [a, b], a + b)
    a, b = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    case("binary", "sub", [a, b], a - b)
    a, b = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    case("binary", "mul", [a, b], a * b)
    # Denominator bounded away from zero: |b| in [0.5, 1.5], random sign.
    a = torch.randn(2, 3, 4)
    b = (torch.rand(2, 3, 4) + 0.5) * torch.where(
        torch.rand(2, 3, 4) < 0.5, -1.0, 1.0
    )
    case("binary", "div", [a, b], a / b)
    # NumPy-style right-aligned broadcast, same rule nn's .add() implements.
    a, b = torch.randn(2, 3, 4), torch.randn(1, 3, 1)
    case("binary", "add_broadcast", [a, b], a + b)

    # ── reduction (keepdim over dim 1) ─────────────────────────────────────
    x = torch.randn(2, 3, 4)
    case("reduction", "sum_dim1", [x], x.sum(dim=1, keepdim=True))
    x = torch.randn(2, 3, 4)
    case("reduction", "mean_dim1", [x], x.mean(dim=1, keepdim=True))
    x = torch.randn(2, 3, 4)
    case("reduction", "max_dim1", [x], x.max(dim=1, keepdim=True).values)

    # ── shape ──────────────────────────────────────────────────────────────
    x = torch.randn(2, 3, 4)  # harness documents exactly [2, 3, 4]
    case("shape", "transpose", [x], x.transpose(1, 2).contiguous())
    x = torch.randn(2, 3, 4)
    case("shape", "reshape", [x], x.reshape(6, 4))
    a, b = torch.randn(2, 3, 4), torch.randn(1, 3, 4)
    case("shape", "cat", [a, b], torch.cat([a, b], dim=0))

    # ── norm/layernorm ─────────────────────────────────────────────────────
    x = torch.randn(2, 5, 8)
    normalized = 8
    weight = torch.randn(normalized) * 0.5 + 1.0  # non-trivial affine
    bias = torch.randn(normalized) * 0.1
    eps = 1e-5
    out = F.layer_norm(x, (normalized,), weight, bias, eps)
    case(
        "norm", "layernorm", [x], out,
        metadata={
            "weight": [float(v) for v in weight],
            "bias": [float(v) for v in bias],
            "eps": eps,
        },
    )

    return written


def verify(root: Path, written: list[Path]) -> None:
    """Fail-closed re-read: every file must exist and every .npy header must be
    the exact subset nn_reftest::load_npy accepts (v1.0/v2.0, <f4, C-order)."""
    problems = []
    for p in written:
        if not p.is_file():
            problems.append(f"missing: {p}")
            continue
        if p.suffix != ".npy":
            continue
        with p.open("rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                problems.append(f"{p}: npy version {version} (loader takes 1.0/2.0)")
                continue
            if fortran:
                problems.append(f"{p}: fortran_order=True (loader is C-order only)")
            if dtype != np.dtype("<f4"):
                problems.append(f"{p}: dtype {dtype} (expected <f4)")
        arr = np.load(p, allow_pickle=False)
        if arr.size == 0:
            problems.append(f"{p}: empty tensor")
    if problems:
        sys.exit(
            "generate_references.py: self-check FAILED:\n  "
            + "\n  ".join(problems)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch reference tensors for pytorch_parity.rs"
    )
    parser.add_argument(
        "--out", type=Path, default=default_out_dir(),
        help="output directory (default: <repo>/test_data/references)",
    )
    args = parser.parse_args()

    written = generate(args.out)
    verify(args.out, written)
    n_cases = len({p.parent for p in written})
    print(
        f"generate_references.py: wrote {len(written)} files across {n_cases} "
        f"cases under {args.out}\n"
        f"  torch {torch.__version__}, numpy {np.__version__}, seed {SEED}\n"
        f"  run the suite: cargo nextest run -p nn-reftest --test pytorch_parity"
    )


if __name__ == "__main__":
    main()
