// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for nn-verify.
//!
//! ## Non-finite value error taxonomy
//!
//! Both `VerifyError` and [`SmtError`](crate::smt_error::SmtError) include variants for
//! non-finite (NaN/Inf) values caught at different pipeline stages. Each variant
//! carries context-specific information about *where* the non-finite value was
//! detected:
//!
//! | Variant | Enum | Catches |
//! |---------|------|---------|
//! | `NonFiniteConstant` | `VerifyError` | Constant-fold producing NaN/Inf in graph ops |
//! | `NonFiniteInputMetadata` | `VerifyError` | NaN/Inf in caller-provided metadata (e.g. eps) |
//! | `InvalidInputBounds` | `VerifyError` | Inverted or non-finite scalar input bounds |
//! | `NonFiniteLiteral` | `SmtError` | NaN/Inf literal during SMT translation |
//! | `NonFiniteConstantParam` | `SmtError` | Non-finite constant kernel parameter in ay path |
//! | `NonFiniteBound` | `SmtError` | Non-finite *output* bounds for SMT assertion |
//! | `NonFiniteInputBound` | `SmtError` | Non-finite *input* bounds for SMT assertion |
//!
//! To catch "any non-finite error" in caller code, match on `VerifyError::Smt(_)`
//! and the four `SmtError::NonFinite*` variants, plus the three `VerifyError`
//! variants directly. This granularity is intentional — each variant triggers
//! different diagnostic messages. See #502 for consolidation discussion.
//!
//! ## `ParamCountMismatch` variants
//!
//! `VerifyError::ParamCountMismatch` has 2 fields (`ir_count`, `provided`) while
//! `SmtError::ParamCountMismatch` has 3 fields (`ir_count`, `expected`, `provided`).
//! The difference exists because the SMT path distinguishes the total IR param count
//! from the *expected constant count* (total minus the symbolic variable), while the
//! NY path only needs the total count vs provided count.

use thiserror::Error;

/// Structural errors: graph translation, shapes, and bounds conversion.
///
/// Never matched by callers — only constructed and propagated.
/// Grouped to reduce `VerifyError` variant count (see #464).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StructuralError {
    /// Fusion subgraph parameter wiring does not match.
    #[error("fusion parameter mismatch: {context}")]
    FusionParam { context: String },

    /// Shape or axis constraint violated during graph translation.
    #[error("shape/axis constraint violation: {context}")]
    ShapeConstraint { context: String },

    /// Failed to construct an ndarray shape from tensor dimensions.
    #[error("array shape construction failed: {reason}")]
    Shape { reason: String },

    /// A `Variable` binding lacks a node name required for multi-input graphs.
    #[error("Variable binding at input index {input_idx} is missing a node name")]
    MissingNodeName { input_idx: usize },

    /// Bounds propagation produced NaN or Inf diff bounds.
    #[error("propagation produced non-finite diff bounds: lower={lower}, upper={upper}")]
    NonFiniteBounds { lower: f32, upper: f32 },

    /// Bounds format conversion failed between NY and nn types.
    #[error("bounds type conversion failed: {0}")]
    BoundsConversion(String),
}

/// Errors from the verification pipeline.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VerifyError {
    /// IR node kind has no NY layer equivalent.
    #[error("unsupported IR operation for NY translation: {0}")]
    UnsupportedOp(String),

    /// Dimension or size computation overflowed during graph translation.
    #[error("dimension overflow in {op}: {context}")]
    DimensionOverflow { op: &'static str, context: String },

    /// Weight tensor failed validation (empty, wrong shape, non-finite).
    #[error("weight validation failed for {op}: {reason}")]
    WeightValidation { op: &'static str, reason: String },

    /// Internal error during IR-to-NY graph translation.
    #[error("internal graph translation error: {context}")]
    InternalTranslationError { context: String },

    /// Error from the SMT verification / analytical bounds path.
    #[error("SMT verification error: {0}")]
    Smt(#[from] crate::smt_error::SmtError),

    /// Caller provided a different number of constant params than the IR declares.
    #[error("kernel parameter count mismatch: IR has {ir_count} params, got {provided} values")]
    ParamCountMismatch { ir_count: usize, provided: usize },

    /// Error from the NY bounds propagation library.
    #[cfg(feature = "ny")]
    #[error("NY error: {0}")]
    Ny(#[from] ny_api::NyError),

    /// Kernel IR failed structural validation before verification.
    #[error("IR validation failed: {0}")]
    IrValidation(#[from] nn_dsl::ir::IRError),

    /// Tensor-level IR failed structural validation.
    #[error("tensor IR validation failed: {0}")]
    TensorIrValidation(#[from] nn_dsl::tensor_ir::TensorIRError),

    /// Input bounds are inverted (lower > upper) or non-finite.
    #[error("invalid input bounds: lower ({lower}) > upper ({upper})")]
    InvalidInputBounds { lower: f32, upper: f32 },

    /// Constant-folding produced NaN or Inf.
    #[error("constant folding produced non-finite value ({value}) in {context}")]
    NonFiniteConstant { value: f32, context: String },

    /// Number of variable bounds does not match the number of `Variable` bindings.
    #[error("variable bounds count mismatch: {variable_count} Variable bindings but {bounds_count} bounds provided")]
    VariableBoundsMismatch {
        variable_count: usize,
        bounds_count: usize,
    },

    /// Multi-variable verification requires at least one `Variable` binding.
    #[error("at least one Variable binding is required for multi-variable verification")]
    NoVariableBindings,

    /// Verification threshold is NaN, Inf, or negative.
    #[error("invalid threshold value: {value} (must be finite and non-negative)")]
    InvalidThreshold { value: f32 },

    /// Caller required sound verification but the result used heuristic (IBP) bounds.
    #[error("soundness required but verification used heuristic approximations for kernel `{kernel_name}`")]
    SoundnessRequired { kernel_name: String },

    /// Graph translation, shape, or bounds conversion error.
    #[error("structural error: {0}")]
    Structural(#[from] StructuralError),

    /// JSON serialization/deserialization of status files failed.
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Filesystem I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Rust-to-IR lowering failed.
    #[error("kernel lowering failed: {0}")]
    LowerError(#[from] nn_dsl::LowerError),

    /// Caller-provided metadata (e.g. epsilon) contains NaN or Inf.
    #[error("non-finite value in input metadata: {context}")]
    NonFiniteInputMetadata { context: String },

    /// Generic invalid input (catch-all for pre-condition violations).
    #[error("invalid verification input: {0}")]
    InvalidInput(String),

    /// Bounds propagation failed (parallel or sequential path).
    #[error("propagation failed: {0}")]
    PropagationFailed(String),

    /// Single-input `trace_to_graph_model()` called on a graph with multiple
    /// variable inputs. Use `trace_to_graph_model_multi_input()` instead.
    #[error(
        "trace_to_graph_model() found {count} variable inputs but expects exactly 1; \
         use trace_to_graph_model_multi_input() for multi-input models"
    )]
    MultipleVariableInputs { count: usize },

    /// Certificate validation failed.
    #[error("invalid certificate: {reason}")]
    InvalidCertificate { reason: String },

    /// Graph or network has no nodes — cannot decompose or verify.
    #[error("graph has no nodes")]
    EmptyGraph,
}

impl From<VerifyError> for nn_core::TensorError {
    fn from(e: VerifyError) -> Self {
        let msg = e.to_string();
        Self::backend_failure_with_source(
            nn_core::BackendDomain::Verification,
            nn_core::BackendErrorKind::Other,
            msg,
            e,
        )
    }
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
