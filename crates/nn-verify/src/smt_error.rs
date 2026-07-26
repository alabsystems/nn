// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SMT-specific errors, mapped into `VerifyError`.
//!
//! Extracted from `ay/error.rs` to be always-available without the `ay-smt`
//! feature flag (#859). The analytical bounds functions use these error types
//! but don't depend on `ay-bindings`.
//!
//! ## Non-finite value variants
//!
//! Four variants catch NaN/Inf at different stages of the ay SMT pipeline:
//!
//! - **`NonFiniteLiteral`** — During `real_from_f64` translation of a constant
//!   expression node. Catches NaN/Inf that survived constant folding.
//! - **`NonFiniteConstantParam`** — During `translate_kernel` input validation.
//!   Catches non-finite values in caller-provided constant parameters.
//! - **`NonFiniteBound`** — In `finalize_query` when analytical or heuristic
//!   output bounds are NaN/Inf. Prevents unsound SMT assertions.
//! - **`NonFiniteInputBound`** — In `compute_output_bounds_heuristic` when
//!   caller-provided input bounds are NaN/Inf.
//!
//! Each has defense-in-depth purpose: upstream callers validate, but these guards
//! catch anything that slips through. See `error.rs` module docs for the full
//! cross-enum taxonomy (#502).

use thiserror::Error;

/// Errors specific to the ay SMT verification path.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SmtError {
    #[error("unsupported IR operation for SMT translation: {op_description}")]
    UnsupportedOp {
        /// Debug representation of the unsupported IR node kind.
        op_description: String,
    },

    #[error("ay solver error: {reason}")]
    SolverError {
        /// Solver failure reason (pass-through from ay or internal).
        reason: String,
    },

    #[error("kernel has no parameters")]
    NoParameters,

    #[error("non-finite literal value: {0}")]
    NonFiniteLiteral(f64),

    /// A constant kernel parameter is NaN or infinite.
    ///
    /// `index` is the **kernel parameter index** (0-based position in the
    /// kernel's parameter list, where param 0 is typically the symbolic
    /// variable). For example, in `snake(x, alpha)`, a non-finite `alpha`
    /// reports `index: 1`.
    #[error("non-finite constant parameter at index {index}: {value}")]
    NonFiniteConstantParam { index: usize, value: f64 },

    #[error("value too large for real encoding (|val * 1e6| > i64::MAX): {0}")]
    ValueTooLargeForRealEncoding(f64),

    #[error("inverted bounds: lower ({lower}) > upper ({upper})")]
    InvertedBounds { lower: f64, upper: f64 },

    #[error("invalid alpha for Snake bounds: {0} (must be > 0)")]
    InvalidSnakeAlpha(f64),

    #[error("non-finite output bounds: lower={lower}, upper={upper}")]
    NonFiniteBound { lower: f64, upper: f64 },

    #[error("non-finite input bounds: lower={lower}, upper={upper}")]
    NonFiniteInputBound { lower: f64, upper: f64 },

    #[error(
        "constant_params count mismatch: kernel has {ir_count} params, \
         expected at most {expected} constants but got {provided}"
    )]
    ParamCountMismatch {
        ir_count: usize,
        expected: usize,
        provided: usize,
    },

    #[error(
        "variable_bounds count mismatch: bindings have {num_variables} Variable entries \
         but variable_bounds has {bounds_len} entries"
    )]
    VariableBoundsMismatch {
        num_variables: usize,
        bounds_len: usize,
    },

    #[error("IR validation failed: {0}")]
    IrValidation(#[from] nn_dsl::ir::IRError),

    #[error("index out of bounds: {context} (index {index}, length {length})")]
    IndexOutOfBounds {
        context: &'static str,
        index: usize,
        length: usize,
    },
}

#[cfg(test)]
#[path = "smt_error_tests.rs"]
mod tests;
