#!/usr/bin/env bash
# Memory-capped nn test runner — the OOM-safe way to run the workspace tests.
# Prevents the 2026-06-15 OOM (see CLAUDE.md): caps binary-level parallelism
# (-j2), in-binary thread parallelism (--test-threads=1), and per-process
# virtual memory (ulimit -v). The #[ignore]'d giant GPU tests stay skipped.
#
# The test command at the bottom is the unverified test lane of the toolchain
# rust-toolchain.toml pins: Trust's targo (owner directive 2026-09-28), in
# development and in the public snapshot alike.
#
# Usage: scripts/test-capped.sh [extra test args...]
set -euo pipefail

# cargo-nextest honours .config/nextest.toml's heavy group but cannot drive
# targo: it re-enters `$CARGO test --no-run`, which targo refuses without a
# lane (measured 2026-09-28).

# Per-process virtual-memory cap (KB). 24 GB leaves headroom on a 128 GB box and
# is far below any single heavy test's blow-up, so a runaway test dies instead of
# panicking the kernel. If the host cannot enforce it, no tests may start.
if ! ulimit -v $((24 * 1024 * 1024)); then
    echo "test-capped: virtual-memory cap unavailable; no tests started" >&2
    exit 75
fi

exec targo --unverified test --workspace -j2 "$@" -- --test-threads=1
