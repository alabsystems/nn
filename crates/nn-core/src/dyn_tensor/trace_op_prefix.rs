// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Node-name prefix lookup for `TraceOp` variants.
//!
//! Delegates to `TraceOp::canonical_name()` — the authoritative source
//! for op names (#2134).

use super::TraceOp;

/// Returns a short prefix for naming traced nodes.
#[allow(unreachable_patterns)] // #[non_exhaustive] catch-all for future variants
pub(super) fn op_prefix(op: &TraceOp) -> &str {
    op.canonical_name()
}
