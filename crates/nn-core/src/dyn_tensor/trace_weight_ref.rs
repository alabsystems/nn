// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight data captured during tracing (flat f32 + shape).

/// Weight data captured during tracing (flat f32 + shape).
///
/// Use [`data()`](Self::data) and [`shape()`](Self::shape) accessors.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WeightRef {
    /// Flat f32 data (converted from original dtype).
    pub(crate) data: Vec<f32>,
    /// Original shape of the weight tensor.
    pub(crate) shape: Vec<usize>,
}

impl WeightRef {
    /// Create a new weight reference from flat data and shape.
    ///
    /// Returns an error if `data` is non-empty and its length does not
    /// match the product of `shape` dimensions (shape-only refs with
    /// empty data are allowed via [`from_shape`](Self::from_shape)).
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, crate::error::TensorError> {
        if !data.is_empty() {
            let expected = crate::tensor::checked_dim_product(&shape)?;
            if data.len() != expected {
                return Err(crate::error::TensorError::shape_mismatch(
                    shape,
                    vec![data.len()],
                ));
            }
        }
        Ok(Self { data, shape })
    }

    /// Create a weight reference without validation.
    ///
    /// For internal use where data/shape consistency is already guaranteed
    /// (e.g., extracted from a validated `DynTensor`).
    pub(crate) fn new_unchecked(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Create a shape-only weight reference (no data).
    ///
    /// Used as a last-resort fallback when actual data extraction fails
    /// (e.g., unsupported dtype). Prefer `DynTensor::to_weight_ref()`
    /// which attempts to capture actual weight data first.
    pub fn from_shape(shape: &[usize]) -> Self {
        Self {
            data: Vec::new(),
            shape: shape.to_vec(),
        }
    }

    /// Flat f32 weight data.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Shape of the weight tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns `true` if this is a shape-only placeholder (no actual data).
    ///
    /// A placeholder has a non-empty shape with non-zero product but no
    /// data. This occurs when `from_shape()` is used as a fallback for
    /// weight extraction failures. Empty shape (absent optional param)
    /// and zero-dim shapes (product 0) are not placeholders.
    #[must_use]
    pub fn is_placeholder(&self) -> bool {
        self.data.is_empty() && !self.shape.is_empty() && self.shape.iter().all(|&d| d > 0)
    }
}
