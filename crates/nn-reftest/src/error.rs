// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for reference tensor comparison.

use thiserror::Error;

/// Errors arising from reference tensor comparison operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReftestError {
    /// Reference and candidate tensors have different shapes.
    #[error("shape mismatch for '{name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// A named tensor was not found in the reference trace.
    #[error("tensor not found in reference trace: '{0}'")]
    TensorNotFound(String),

    /// Reference and candidate traces have different checkpoint counts.
    #[error(
        "trace length mismatch: reference has {reference} checkpoints, candidate has {candidate}"
    )]
    TraceLengthMismatch { reference: usize, candidate: usize },

    /// A tensor contains zero elements.
    #[error("empty tensor: '{0}'")]
    EmptyTensor(String),

    /// The tensor dtype cannot be converted to f32.
    #[error("unsupported dtype for f32 conversion: {0}")]
    UnsupportedDtype(String),

    /// Failed to parse a safetensors file.
    #[error("safetensors parse error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    /// Filesystem I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Raw byte length does not match the expected `numel * bytes_per_element`.
    #[error("tensor data length mismatch: expected {expected} bytes, got {actual} bytes")]
    DataLengthMismatch { expected: usize, actual: usize },

    /// Error propagated from `nn-core` tensor operations.
    #[cfg(feature = "nn-core")]
    #[error("nn-core tensor error: {0}")]
    Core(#[from] nn_core::TensorError),

    /// Shape dimension product overflows `usize`.
    #[error("shape product overflow: {0:?}")]
    ShapeProductOverflow(Vec<usize>),

    /// Flat data length does not match the shape's element count.
    #[error("element count mismatch for '{name}': shape {shape:?} expects {expected} elements, got {actual}")]
    ElementCountMismatch {
        name: String,
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    /// `numel * bytes_per_element` overflows `usize`.
    #[error("byte count overflow: {numel} elements * {bytes_per_element} bytes/element")]
    ByteCountOverflow {
        numel: usize,
        bytes_per_element: usize,
    },

    /// An f64 value is non-finite or exceeds `f32::MAX` magnitude.
    #[error("f64 value {value} at index {index} is not representable as f32 (non-finite or |value| > f32::MAX)")]
    F64OutOfF32Range { value: f64, index: usize },

    /// NPY file does not start with the `\x93NUMPY` magic bytes.
    #[error("invalid NPY magic bytes (expected \\x93NUMPY)")]
    NpyBadMagic,

    /// NPY file uses a version other than 1.0 or 2.0.
    #[error("unsupported NPY version {major}.{minor} (only 1.0 and 2.0 supported)")]
    NpyUnsupportedVersion { major: u8, minor: u8 },

    /// NPY header could not be parsed.
    #[error("NPY header parse error: {0}")]
    NpyHeaderParse(String),

    /// NPY file uses a dtype not supported for f32 conversion.
    #[error("unsupported NPY dtype: {0}")]
    NpyUnsupportedDtype(String),

    /// NPY file uses Fortran (column-major) order, which is unsupported.
    #[error("NPY Fortran order not supported (only C order)")]
    NpyFortranOrder,

    /// An integer value exceeds 2^24 and cannot be losslessly cast to f32.
    #[error(
        "integer value {value} at index {index} loses precision when cast to f32 (|value| > 2^24)"
    )]
    IntPrecisionLoss { value: i64, index: usize },

    /// Invalid spectral comparison configuration.
    #[cfg(feature = "spectral")]
    #[error("spectral config error: {0}")]
    SpectralConfig(String),
}
