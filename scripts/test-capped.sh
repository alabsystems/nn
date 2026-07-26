#!/usr/bin/env bash
# Memory-capped nn test runner — stopgap when cargo-nextest isn't installed.
# Prevents the 2026-06-15 OOM (see CLAUDE.md): caps binary-level parallelism
# (-j2), in-binary thread parallelism (--test-threads=1), and per-process
# virtual memory (ulimit -v). The #[ignore]'d giant GPU tests stay skipped.
#
# Usage: scripts/test-capped.sh [extra cargo test args...]
set -euo pipefail

# Prefer nextest if available (honors .config/nextest.toml heavy group).
if command -v cargo-nextest >/dev/null 2>&1; then
  exec cargo nextest run "$@"
fi

# Per-process virtual-memory cap (KB). 24 GB leaves headroom on a 128 GB box and
# is far below any single heavy test's blow-up, so a runaway test dies instead of
# panicking the kernel. Adjust for smaller machines.
ulimit -v $((24 * 1024 * 1024)) || true

exec cargo test --workspace -j2 "$@" -- --test-threads=1
