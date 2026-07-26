// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Types for safetensors weight loading: `WeightError`, `TensorInfo`, `convert_dtype`.
//!
//! Extracted from `safetensors.rs` (#1575) to keep files under 400 lines.

use nn_core::DType;
use thiserror::Error;

use crate::error::MetalError;

/// Errors from safetensors weight loading.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WeightError {
    #[error("failed to open weight file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse safetensors: {0}")]
    Parse(#[from] safetensors::SafeTensorError),
    #[error("Metal error: {0}")]
    Metal(#[from] MetalError),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("unsupported dtype: {0:?}")]
    UnsupportedDtype(safetensors::Dtype),
    #[error("shape product overflow: {0:?}")]
    ShapeOverflow(Vec<usize>),
    #[error("tensor data range overflow: offset + byte_len overflows usize for tensor {name}")]
    TensorDataOverflow { name: String },
    #[error(
        "tensor data out of bounds: tensor {name} at offset {offset} + {byte_len} bytes exceeds buffer size {buffer_size}"
    )]
    TensorDataOutOfBounds {
        name: String,
        offset: usize,
        byte_len: usize,
        buffer_size: usize,
    },
    #[error("dtype mismatch for tensor {name}: expected {expected:?}, got {actual:?}")]
    DtypeMismatch {
        name: String,
        expected: DType,
        actual: DType,
    },
    #[error("shape mismatch for tensor {name}: expected rank {expected_rank} with dims {expected_dims:?}, got {actual_dims:?}")]
    ShapeMismatch {
        name: String,
        expected_rank: usize,
        expected_dims: Vec<usize>,
        actual_dims: Vec<usize>,
    },
}

impl From<WeightError> for nn_core::TensorError {
    fn from(e: WeightError) -> Self {
        // Convert inner MetalError directly to avoid double "Metal error:" prefix
        if let WeightError::Metal(inner) = e {
            return inner.into();
        }
        let msg = e.to_string();
        Self::backend_failure_with_source(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::Other,
            msg,
            e,
        )
    }
}

/// Metadata for a single tensor within the weight map.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TensorInfo {
    /// Byte offset of this tensor's data within the Metal buffer.
    pub offset: usize,
    /// Byte length of this tensor's data.
    pub byte_len: usize,
    /// Data type.
    pub dtype: DType,
    /// Shape dimensions.
    pub shape: Vec<usize>,
}

impl TensorInfo {
    /// Number of elements in this tensor.
    ///
    /// Uses checked arithmetic to prevent silent overflow from malformed
    /// safetensors shape dimensions.
    #[must_use = "returns a Result that may contain an error"]
    pub fn numel(&self) -> Result<usize, WeightError> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| WeightError::ShapeOverflow(self.shape.clone()))
    }
}

/// Convert safetensors dtype to nn DType.
pub(super) fn convert_dtype(dt: safetensors::Dtype) -> Result<DType, WeightError> {
    match dt {
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F64 => Ok(DType::F64),
        safetensors::Dtype::I32 => Ok(DType::I32),
        safetensors::Dtype::I64 => Ok(DType::I64),
        safetensors::Dtype::U8 => Ok(DType::U8),
        safetensors::Dtype::BOOL => Ok(DType::Bool),
        other => Err(WeightError::UnsupportedDtype(other)),
    }
}
