#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# SPDX-License-Identifier: Apache-2.0
#
# Verified lane over the nn targets that verify with NO refuted row, each held
# to a proved floor. Additive: it does not replace or narrow the release
# gate's `targo trust check --allow-l0-gaps --workspace` (publish/config.sh),
# which still covers every first-party unit. That workspace step takes hours
# and most crates still carry refuted rows, so it cannot tell you when one of
# these targets regresses; this lane can, in a few minutes.
#
# Each row runs the gate's own verified command (advisory lane,
# `--allow-l0-gaps`) on one target, then reads `verification_gate.counts` from
# the report and FAILS the row when:
#   * targo exits non-zero (a refutation, a refusal, or a crash);
#   * `failed` > 0;
#   * `proved` < the row's floor. Only `proved` counts toward the floor:
#     unknown, timed-out, runtime-checked and assumed rows never do;
#   * the report is not this row's own package as the primary proof unit, its
#     gate decision is neither `pass` nor `conditional-pass`, or its coverage is
#     incomplete (a report for the wrong unit, or one that skipped functions,
#     must not satisfy a floor).
# A pass here is the advisory lane's CONDITIONAL pass: the unknown rows are
# still unknown. It is not a strict/certify result.
#
# Floors are set at about 80% of the lowest `proved` measured for the row
# (targo 1.99.0-dev 321aaeda7, -j 4, on a loaded host where per-function
# budgets expire and `proved` moves by a few rows between runs). Raise a floor
# when a change proves more; never lower one to turn a row green -- fix the
# regression instead.
#
# Usage: scripts/verify-ready.sh [report-root]
#   report-root defaults to "$CARGO_TARGET_DIR/verify-ready" (or
#   target/verify-ready); each row writes its report under <report-root>/<row>.
#   NN_VERIFY_JOBS overrides the job count (default 4).
set -euo pipefail

cd "$(dirname "$0")/.."

# The verified lane takes its configuration from targo alone. Ambient compiler
# flags would change what is verified (or switch the verifier off), so they are
# cleared rather than inherited.
unset TRUSTFLAGS RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_ENCODED_TRUSTFLAGS \
  CARGO_BUILD_RUSTFLAGS

# Rows: <package> <target-selector> <proved-floor>
ROWS=(
  "nn-macros --lib 2"
  "nn-optimize --bins 16"
  "nn-cli --bins 22"
  "nn-qwen3 --lib 34"
  "nn-glm5 --lib 20"
  "nn-optim --lib 47"
)

report_root="${1:-${CARGO_TARGET_DIR:-target}/verify-ready}"
jobs="${NN_VERIFY_JOBS:-4}"

# Development verifies the locked graph. The public snapshot ships no
# Cargo.lock (publish/manifest.txt), and `--locked` refuses outright without
# one, so there the lane resolves fresh, exactly as the release gate does.
locked=()
if [[ -f Cargo.lock ]]; then
  locked=(--locked)
else
  echo "verify-ready: no Cargo.lock; resolving fresh (not a reproducibility check)" >&2
fi

status=0
summary=()
for row in "${ROWS[@]}"; do
  read -r pkg selector floor <<<"$row"
  name="${pkg}${selector/--/.}"
  dir="$report_root/$name"
  rm -rf "${dir:?}"
  mkdir -p "$dir"
  echo "verify-ready: $name (proved floor $floor)" >&2

  rc=0
  rustup run trust targo trust check --allow-l0-gaps ${locked[@]+"${locked[@]}"} -j "$jobs" \
    -p "$pkg" "$selector" --report-dir "$dir" || rc=$?

  verdict=$(python3 - "$dir/report.json" "$floor" "$rc" "$pkg" <<'PY'
import json, sys

path, floor, rc, pkg = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
try:
    with open(path) as fh:
        report = json.load(fh)
    gate = report["verification_gate"]
    counts = gate["counts"]
    unit = str(report["crate_name"])
    decision = gate["decision"]
    coverage = gate["coverage"]
    proved = int(counts["proved"])
    failed = int(counts["failed"])
    unknown = int(counts["unknown"])
    total = int(counts["total"])
except (OSError, ValueError, KeyError, TypeError) as err:
    print(f"FAIL no readable verification_gate in {path} ({err!r}); targo rc={rc}")
    sys.exit(1)

line = f"total={total} proved={proved} failed={failed} unknown={unknown} floor={floor} rc={rc}"
problems = []
if f'package="{pkg}"' not in unit or 'proof_unit_role="primary"' not in unit:
    problems.append(f"report is not the primary unit of {pkg}: {unit[:120]}")
if decision not in ("pass", "conditional-pass"):
    problems.append(f"gate decision {decision!r}")
if coverage.get("coverage_complete") is not True:
    problems.append(f"coverage incomplete: {coverage}")
if rc != 0:
    problems.append(f"targo exited {rc}")
if failed > 0:
    problems.append(f"{failed} refuted row(s)")
if proved < floor:
    problems.append(f"proved {proved} < floor {floor}")
if problems:
    print(f"FAIL {line}: " + "; ".join(problems))
    sys.exit(1)
print(f"PASS {line}")
PY
  ) || status=1
  summary+=("$name: $verdict")
done

echo "verify-ready summary (advisory lane; unknown rows are not proved):" >&2
for line in "${summary[@]}"; do
  echo "  $line" >&2
done
exit "$status"
