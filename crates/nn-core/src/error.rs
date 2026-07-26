// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for nn

use std::backtrace::Backtrace;

use thiserror::Error;

/// Result type for nn operations
pub type Result<T> = std::result::Result<T, TensorError>;

/// Wrapper around [`std::backtrace::Backtrace`] that hides the type from
/// thiserror v2's automatic `provide()` generation (which requires the
/// unstable `error_generic_member_access` feature).
///
/// `Backtrace::capture()` is a no-op (<5ns) when `RUST_BACKTRACE` is not
/// set. The cost is only paid on the error path when the developer opts in.
#[derive(Debug)]
pub struct ErrorTrace(Backtrace);

impl ErrorTrace {
    /// Capture a backtrace at the current call site.
    fn capture() -> Self {
        Self(Backtrace::capture())
    }

    /// Access the inner `Backtrace`.
    pub fn inner(&self) -> &Backtrace {
        &self.0
    }
}

impl std::fmt::Display for ErrorTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// Domain tag for backend errors that cannot use `#[from]` due to
/// circular dependency constraints (nn-core is a leaf crate).
///
/// Backend crates provide `From<BackendError> for TensorError` impls
/// that map into `BackendFailure { domain, message }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BackendDomain {
    /// Generic device-level error.
    Device,
    /// CPU backend (signal processing, reference implementations).
    Cpu,
    /// Apple Metal GPU backend.
    Metal,
    /// NVIDIA CUDA GPU backend.
    Cuda,
    /// Vulkan/SPIR-V GPU backend.
    Vulkan,
    /// Apple Neural Engine backend.
    Ane,
    /// Interval bounds computation.
    Bounds,
    /// Formal verification pipeline (NY, ay).
    Verification,
    /// Whisper speech-to-text model.
    Whisper,
    /// Qwen3 LLM model.
    Qwen3,
    /// GLM-4/5 (ChatGLM) LLM model.
    Glm5,
    /// Kokoro TTS model.
    Kokoro,
}

/// Classification of backend errors for programmatic recovery.
///
/// Callers can match on `BackendErrorKind` to decide whether to retry
/// (e.g. `OutOfMemory` → shrink batch), fail fast (`KernelCompile` → bug),
/// or fall back (`Other` → generic handler).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BackendErrorKind {
    /// GPU or CPU memory allocation failed (arena overflow, buffer create).
    OutOfMemory,
    /// Kernel/shader compilation failed (MSL, PTX, SPIR-V pipeline).
    KernelCompile,
    /// Solver or computation timed out.
    Timeout,
    /// GPU dispatch, readback, or execution failed.
    DispatchFailed,
    /// Unclassified or domain-specific error.
    Other,
}

/// Errors that can occur in nn
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TensorError {
    /// Tensor shapes do not match for the attempted operation.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        /// Backtrace captured at error creation (no-op unless RUST_BACKTRACE=1).
        bt: ErrorTrace,
    },

    /// Tensor rank (number of dimensions) does not match.
    #[error("Rank mismatch: expected {expected} dimensions, got {actual}")]
    RankMismatch { expected: usize, actual: usize },

    /// Shape is malformed (e.g. zero-length dimension where not allowed).
    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    /// A dimension index exceeds the tensor's rank.
    #[error("Dimension {dim} out of range for rank {rank}")]
    DimensionOutOfRange { dim: usize, rank: usize },

    /// A convolution parameter is invalid.
    #[error("Conv error: {param} = {value} is invalid ({reason})")]
    ConvParameterInvalid {
        param: &'static str,
        value: usize,
        reason: &'static str,
    },

    /// A scalar value is outside the valid range for the operation.
    #[error("Value out of range: {description}")]
    ValueOutOfRange { description: &'static str },

    /// A dtype conversion failed because a value cannot be represented in the target type.
    #[error("Dtype conversion {source_dtype} → {target_dtype}: {reason}")]
    DtypeConversion {
        source_dtype: crate::DType,
        target_dtype: crate::DType,
        reason: String,
    },

    /// An embedding lookup index exceeds the vocabulary size.
    #[error("Embedding index {index} out of range for vocab size {vocab_size}")]
    EmbeddingIndexOutOfRange { index: usize, vocab_size: usize },

    /// An operation requires a non-empty dimension but received a zero-length one.
    #[error("Zero-length dimension: axis {axis} has size 0 (operation: {operation})")]
    ZeroLengthDimension {
        axis: usize,
        operation: &'static str,
    },

    /// Requested device backend is not yet implemented.
    #[error("Device allocation unavailable: {device} backend not yet implemented")]
    DeviceAllocationUnavailable { device: crate::Device },

    /// Cross-device tensor transfer is not yet implemented.
    #[error("Device transfer unavailable: {from_device} -> {target} transfer not yet implemented")]
    DeviceTransferUnavailable {
        from_device: crate::Device,
        target: crate::Device,
        /// Backtrace captured at error creation (no-op unless RUST_BACKTRACE=1).
        bt: ErrorTrace,
    },

    /// Error from a specific backend crate (Metal, CUDA, etc.).
    ///
    /// When converted from a typed backend error via `From`, the original error
    /// is preserved in `source` for programmatic recovery and `anyhow` chain
    /// display. Use `error.source()` to access the original typed error, and
    /// `error.source().unwrap().downcast_ref::<MetalError>()` to recover it.
    #[error("{domain:?} error: {message}")]
    BackendFailure {
        domain: BackendDomain,
        kind: BackendErrorKind,
        message: String,
        /// Original typed backend error, preserved for source chaining.
        /// `None` when constructed via `backend_failure()` (string-only path).
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
        /// Backtrace captured at error creation (no-op unless RUST_BACKTRACE=1).
        bt: ErrorTrace,
    },

    /// Provided data length does not match the shape's element count.
    #[error("Data length mismatch: shape requires {expected} elements, got {actual}")]
    DataLengthMismatch { expected: usize, actual: usize },

    /// Shape dimension product overflows `usize`.
    #[error("Dimension product overflow: dimensions {dims:?} exceed usize::MAX")]
    DimensionOverflow { dims: Vec<usize> },

    /// Operation received a tensor with an unexpected data type.
    #[error("Data type mismatch: expected {expected}, got {actual}")]
    DTypeMismatch {
        expected: crate::DType,
        actual: crate::DType,
        /// Backtrace captured at error creation (no-op unless RUST_BACKTRACE=1).
        bt: ErrorTrace,
    },

    /// Insufficient memory for the requested allocation.
    #[error("Out of memory: requested {requested} bytes, available {available}")]
    OutOfMemory { requested: usize, available: usize },

    /// Interval bounds are malformed (e.g. lower > upper, non-finite).
    #[error("Invalid bounds: {0}")]
    InvalidBounds(String),

    /// Operation is not supported for this tensor configuration.
    #[error("Operation not supported: {0}")]
    Unsupported(String),

    /// Named tensor not found in weight store (VarBuilder, safetensors, etc.).
    #[error("Tensor not found: {name}")]
    TensorNotFound { name: String },

    /// Tensor data contains non-finite values (NaN or Inf).
    #[error("Non-finite data: {count} NaN/Inf values in tensor '{name}'")]
    NonFiniteData { name: String, count: usize },

    /// Computation graph topology violation: a node references an input that
    /// has not appeared earlier in the graph.
    #[error(
        "Topology error: node '{node_name}' at index {index} references \
         input_id {missing_input} which has not appeared earlier in the graph"
    )]
    TopologyError {
        node_name: String,
        index: usize,
        missing_input: u64,
    },

    /// Weight data extraction failed during trace recording.
    ///
    /// `to_weight_ref()` could not convert tensor data to f32 weight storage.
    /// Previously fell through silently to shape-only `WeightRef::from_shape()`.
    #[error("Weight conversion failed: dtype={dtype}, device={device}")]
    WeightConversionFailed {
        dtype: crate::DType,
        device: crate::Device,
    },

    /// Filesystem I/O error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl TensorError {
    /// Convenience constructor for `ShapeMismatch` with backtrace capture.
    pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::ShapeMismatch {
            expected,
            actual,
            bt: ErrorTrace::capture(),
        }
    }

    /// Convenience constructor for `DTypeMismatch` with backtrace capture.
    pub fn dtype_mismatch(expected: crate::DType, actual: crate::DType) -> Self {
        Self::DTypeMismatch {
            expected,
            actual,
            bt: ErrorTrace::capture(),
        }
    }

    /// Convenience constructor for `BackendFailure` with backtrace capture.
    ///
    /// The original typed error is not preserved — use
    /// [`backend_failure_with_source`](Self::backend_failure_with_source) when
    /// converting from a typed backend error to preserve the source chain.
    pub fn backend_failure(domain: BackendDomain, kind: BackendErrorKind, message: String) -> Self {
        Self::BackendFailure {
            domain,
            kind,
            message,
            source: None,
            bt: ErrorTrace::capture(),
        }
    }

    /// Convenience constructor for `BackendFailure` that preserves the original
    /// typed error as a source chain.
    ///
    /// Callers can recover the original error via `error.source()` and downcast:
    /// ```ignore
    /// if let Some(metal_err) = tensor_err.source().and_then(|s| s.downcast_ref::<MetalError>()) {
    ///     // handle typed MetalError
    /// }
    /// ```
    pub fn backend_failure_with_source(
        domain: BackendDomain,
        kind: BackendErrorKind,
        message: String,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::BackendFailure {
            domain,
            kind,
            message,
            source: Some(Box::new(source)),
            bt: ErrorTrace::capture(),
        }
    }

    /// Convenience constructor for `DeviceTransferUnavailable` with backtrace capture.
    pub fn device_transfer(from_device: crate::Device, target: crate::Device) -> Self {
        Self::DeviceTransferUnavailable {
            from_device,
            target,
            bt: ErrorTrace::capture(),
        }
    }

    /// Returns the [`BackendErrorKind`] if this is a `BackendFailure`.
    ///
    /// Callers can use this to decide whether to retry (`OutOfMemory`),
    /// fail fast (`KernelCompile`), or fall back (`Other`).
    pub fn backend_error_kind(&self) -> Option<BackendErrorKind> {
        match self {
            Self::BackendFailure { kind, .. } => Some(*kind),
            _ => None,
        }
    }

    /// Returns the backtrace if this error variant captured one.
    ///
    /// Only `ShapeMismatch`, `DTypeMismatch`, `DeviceTransferUnavailable`,
    /// and `BackendFailure` carry backtraces. Returns `None` for all other
    /// variants.
    pub fn backtrace(&self) -> Option<&Backtrace> {
        match self {
            Self::ShapeMismatch { bt, .. } => Some(bt.inner()),
            Self::DTypeMismatch { bt, .. } => Some(bt.inner()),
            Self::DeviceTransferUnavailable { bt, .. } => Some(bt.inner()),
            Self::BackendFailure { bt, .. } => Some(bt.inner()),
            _ => None,
        }
    }
}

/// Validate that a dimension index is within the tensor's rank.
///
/// Returns `Err(TensorError::DimensionOutOfRange)` if `dim >= rank`.
pub fn check_dim(dim: usize, rank: usize) -> Result<()> {
    if dim >= rank {
        return Err(TensorError::DimensionOutOfRange { dim, rank });
    }
    Ok(())
}

impl From<ndarray::ShapeError> for TensorError {
    fn from(e: ndarray::ShapeError) -> Self {
        Self::InvalidShape(e.to_string())
    }
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
