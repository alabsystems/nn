// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Quantized tensor storage and dequantization for [`DynTensor`].
//!
//! Supports GGUF-style block quantization formats (Q4_0, Q4_1, Q8_0) that
//! store weights in compressed form and dequantize to f32 on demand during
//! operations. This enables loading GGUF models without materializing all
//! weights as f32 upfront.
//!
//! Dequantization routines are self-contained (no dependency on nn-gguf)
//! so that nn-core remains a leaf crate.

use crate::tensor::checked_dim_product;
use crate::{DType, Result, TensorError};
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

use super::{DynTensor, TensorStorage};

/// Quantization format identifier.
///
/// Each variant corresponds to a GGUF/GGML block quantization scheme.
/// Block size is 32 elements for all currently supported formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QuantType {
    /// Q4_0: 4-bit symmetric quantization with f16 block scale.
    ///
    /// Block layout (18 bytes per 32 elements):
    /// - 2 bytes: f16 scale
    /// - 16 bytes: 32 x 4-bit signed values (packed 2 per byte)
    ///
    /// Dequant: `val = scale * (nibble - 8)`
    Q4_0,

    /// Q4_1: 4-bit asymmetric quantization with f16 scale and minimum.
    ///
    /// Block layout (20 bytes per 32 elements):
    /// - 2 bytes: f16 scale (`d`)
    /// - 2 bytes: f16 minimum (`m`)
    /// - 16 bytes: 32 x 4-bit unsigned values (packed 2 per byte)
    ///
    /// Dequant: `val = d * nibble + m`
    Q4_1,

    /// Q8_0: 8-bit symmetric quantization with f16 block scale.
    ///
    /// Block layout (34 bytes per 32 elements):
    /// - 2 bytes: f16 scale
    /// - 32 bytes: 32 x 8-bit signed values
    ///
    /// Dequant: `val = scale * q`
    Q8_0,
}

impl QuantType {
    /// Number of elements per quantization block.
    #[must_use]
    pub fn block_size(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => 32,
        }
    }

    /// Bytes per quantization block.
    #[must_use]
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => 18, // 2 (scale) + 16 (4-bit * 32 values)
            Self::Q4_1 => 20, // 2 (d) + 2 (m) + 16
            Self::Q8_0 => 34, // 2 (scale) + 32 (8-bit * 32 values)
        }
    }

    /// Expected byte length for `num_elements` quantized values.
    ///
    /// Returns `None` if `num_elements` is not a multiple of the block size.
    #[must_use]
    pub fn expected_bytes(self, num_elements: usize) -> Option<usize> {
        if num_elements == 0 {
            return Some(0);
        }
        if !num_elements.is_multiple_of(self.block_size()) {
            return None;
        }
        let num_blocks = num_elements / self.block_size();
        num_blocks.checked_mul(self.block_bytes())
    }
}

impl std::fmt::Display for QuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
        }
    }
}

/// Quantized tensor storage: raw block-quantized bytes plus metadata.
///
/// The data is stored as-is from the GGUF file (or any source producing
/// GGML-format blocks). Dequantization to f32 happens on demand via
/// [`QuantizedStorage::dequantize`].
#[derive(Debug, Clone)]
pub struct QuantizedStorage {
    /// The raw quantized block data.
    data: Arc<Vec<u8>>,
    /// Logical tensor shape (e.g., `[out_features, in_features]`).
    shape: Vec<usize>,
    /// Quantization format.
    quant_type: QuantType,
}

impl QuantizedStorage {
    /// Create a new quantized storage, validating that the byte length matches
    /// the shape and quantization format.
    pub fn new(data: Vec<u8>, shape: &[usize], quant_type: QuantType) -> Result<Self> {
        let num_elements = checked_dim_product(shape)?;
        let expected = quant_type.expected_bytes(num_elements).ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "element count {num_elements} is not a multiple of {quant_type} block size {}",
                quant_type.block_size()
            ))
        })?;
        if data.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            data: Arc::new(data),
            shape: shape.to_vec(),
            quant_type,
        })
    }

    /// Logical tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Quantization format.
    #[must_use]
    pub fn quant_type(&self) -> QuantType {
        self.quant_type
    }

    /// Raw quantized bytes.
    #[must_use]
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }

    /// Dequantize to an f32 ndarray.
    pub fn dequantize(&self) -> Result<ArrayD<f32>> {
        let num_elements = checked_dim_product(&self.shape)?;
        let floats = match self.quant_type {
            QuantType::Q4_0 => dequantize_q4_0(&self.data, num_elements),
            QuantType::Q4_1 => dequantize_q4_1(&self.data, num_elements),
            QuantType::Q8_0 => dequantize_q8_0(&self.data, num_elements),
        };
        debug_assert_eq!(floats.len(), num_elements);
        ArrayD::from_shape_vec(IxDyn(&self.shape), floats)
            .map_err(|e| TensorError::InvalidShape(format!("dequantize reshape failed: {e}")))
    }
}

// -- Dequantization routines --------------------------------------------------
// Self-contained implementations matching GGML/llama.cpp format specs.
// These duplicate the logic in nn-gguf::dequant but avoid a cross-crate
// dependency (nn-core is a leaf crate).

/// Dequantize Q4_0 block data to f32.
///
/// Q4_0: 18 bytes per block of 32 elements.
/// Layout: `[f16 scale][16 bytes: 32 x 4-bit signed, packed 2/byte]`
/// Dequant: `val = scale * (nibble - 8)`
fn dequantize_q4_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 32;
    let bytes_per_block = 18;
    let num_blocks = num_elements / block_size;
    let mut output = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let b = block_idx * bytes_per_block;
        let scale = half::f16::from_le_bytes([data[b], data[b + 1]]).to_f32();

        for j in 0..16 {
            let byte = data[b + 2 + j];
            let lo = i32::from(byte & 0x0F) - 8;
            let hi = i32::from((byte >> 4) & 0x0F) - 8;
            output.push(scale * lo as f32);
            output.push(scale * hi as f32);
        }
    }
    output
}

/// Dequantize Q4_1 block data to f32.
///
/// Q4_1: 20 bytes per block of 32 elements.
/// Layout: `[f16 d][f16 m][16 bytes: 32 x 4-bit unsigned, packed 2/byte]`
/// Dequant: `val = d * nibble + m`
fn dequantize_q4_1(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 32;
    let bytes_per_block = 20;
    let num_blocks = num_elements / block_size;
    let mut output = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let b = block_idx * bytes_per_block;
        let d = half::f16::from_le_bytes([data[b], data[b + 1]]).to_f32();
        let m = half::f16::from_le_bytes([data[b + 2], data[b + 3]]).to_f32();

        for j in 0..16 {
            let byte = data[b + 4 + j];
            let lo = f32::from(byte & 0x0F);
            let hi = f32::from((byte >> 4) & 0x0F);
            output.push(d * lo + m);
            output.push(d * hi + m);
        }
    }
    output
}

/// Dequantize Q8_0 block data to f32.
///
/// Q8_0: 34 bytes per block of 32 elements.
/// Layout: `[f16 scale][32 bytes: 32 x i8 values]`
/// Dequant: `val = scale * q`
fn dequantize_q8_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_size = 32;
    let bytes_per_block = 34;
    let num_blocks = num_elements / block_size;
    let mut output = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let b = block_idx * bytes_per_block;
        let scale = half::f16::from_le_bytes([data[b], data[b + 1]]).to_f32();

        for j in 0..32 {
            let q = data[b + 2 + j] as i8;
            output.push(scale * f32::from(q));
        }
    }
    output
}

// -- DynTensor integration ----------------------------------------------------

impl DynTensor {
    /// Create a quantized tensor from raw block-quantized bytes.
    ///
    /// The tensor stores data in compressed form. Arithmetic operations
    /// automatically dequantize to f32 via [`dequantize`](Self::dequantize).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The element count (product of `shape`) is not a multiple of the
    ///   quantization block size (32 for Q4_0/Q4_1/Q8_0).
    /// - The byte length of `data` does not match the expected size for
    ///   the shape and quantization format.
    pub fn from_quantized(data: &[u8], quant_type: QuantType, shape: &[usize]) -> Result<Self> {
        let qs = QuantizedStorage::new(data.to_vec(), shape, quant_type)?;
        Ok(Self {
            dims: shape.to_vec(),
            dtype: DType::F32, // Logical dtype is f32 (dequantized output type)
            storage: TensorStorage::Quantized(Arc::new(qs)),
            trace_node_id: None,
        })
    }

    /// Dequantize this tensor to f32.
    ///
    /// - If the tensor is quantized, dequantizes the raw blocks and returns
    ///   a CPU f32 tensor.
    /// - If the tensor is already non-quantized, returns a clone (no-op).
    pub fn dequantize(&self) -> Result<Self> {
        match &self.storage {
            TensorStorage::Quantized(qs) => {
                let arr = qs.dequantize()?;
                Self::from_cpu_f32(arr)
            }
            _ => Ok(self.clone()),
        }
    }

    /// Returns `true` if this tensor holds quantized (compressed) data.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        matches!(&self.storage, TensorStorage::Quantized(_))
    }

    /// Access the quantized storage metadata, if this tensor is quantized.
    #[must_use]
    pub fn quantized_storage(&self) -> Option<&QuantizedStorage> {
        match &self.storage {
            TensorStorage::Quantized(qs) => Some(qs.as_ref()),
            _ => None,
        }
    }

    /// If this tensor is quantized, dequantize it to CPU f32. Otherwise
    /// return a borrowed reference to `self`.
    ///
    /// This is the standard entry point for operations that need to work
    /// with quantized tensors: call `auto_dequantize()` at the top of
    /// the operation, then proceed with the returned (non-quantized) tensor.
    pub(crate) fn auto_dequantize(&self) -> Result<std::borrow::Cow<'_, Self>> {
        if self.is_quantized() {
            Ok(std::borrow::Cow::Owned(self.dequantize()?))
        } else {
            Ok(std::borrow::Cow::Borrowed(self))
        }
    }
}
