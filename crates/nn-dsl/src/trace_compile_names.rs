// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Human-readable names for `TraceOp` variants.
//!
//! Delegates to `TraceOp::canonical_name()` (nn-core) as the single
//! source of truth. See #2134.

use nn_core::dyn_tensor::trace::TraceOp;

/// Returns a human-readable name for a `TraceOp`.
///
/// Delegates to [`TraceOp::canonical_name()`] to eliminate the prior
/// `_ => "unknown"` catch-all that returned degraded names for 20+ variants.
pub(crate) fn op_name(op: &TraceOp) -> String {
    op.canonical_name().to_string()
}
