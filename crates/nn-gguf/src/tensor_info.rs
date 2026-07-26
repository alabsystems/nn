// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GGUF tensor info (per-tensor metadata in the header).

use std::io::Read;

use crate::dequant::GgufDType;
use crate::error::GgufError;
use crate::header::{read_string, read_u32, read_u64, MAX_DIMENSIONS, MAX_TENSOR_BYTE_SIZE};

/// Maximum number of dimensions a tensor may have.
///
/// Real tensors have at most 5-6 dimensions (batch, channels, depth, height,
/// width, time). 8 is generous. An uncapped `n_dims` from a crafted GGUF
/// file could attempt a ~32 GB allocation via `Vec::with_capacity`.
const MAX_TENSOR_DIMS: u32 = 8;

/// Information about a single tensor stored in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g., "blk.0.attn_q.weight").
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Shape (dimension sizes). GGUF stores in row-major order.
    pub shape: Vec<u64>,
    /// Quantization/data type.
    pub dtype: GgufDType,
    /// Byte offset of this tensor's data from the start of the data section.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Parse a single tensor info entry.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, GgufError> {
        let name = read_string(reader)?;
        let n_dims = read_u32(reader)?;
        if n_dims > MAX_DIMENSIONS {
            return Err(GgufError::DimensionCountExceeded {
                n_dims,
                max: MAX_DIMENSIONS,
            });
        }

        if n_dims > MAX_TENSOR_DIMS {
            return Err(GgufError::DimensionCountExceeded {
                n_dims,
                max: MAX_TENSOR_DIMS,
            });
        }

        let mut shape = Vec::with_capacity(n_dims as usize);
        for i in 0..n_dims {
            let dim = read_u64(reader)?;
            if dim == 0 {
                return Err(GgufError::ZeroDimension {
                    name,
                    dim_index: i as usize,
                });
            }
            shape.push(dim);
        }

        let type_id = read_u32(reader)?;
        let dtype =
            GgufDType::from_u32(type_id).ok_or(GgufError::UnknownMetadataType { type_id })?;

        let offset = read_u64(reader)?;

        Ok(Self {
            name,
            n_dims,
            shape,
            dtype,
            offset,
        })
    }

    /// Total number of elements in this tensor. Returns an error if the
    /// product overflows u64.
    pub fn checked_num_elements(&self) -> Result<u64, GgufError> {
        let mut product: u64 = 1;
        for &dim in &self.shape {
            product = product
                .checked_mul(dim)
                .ok_or_else(|| GgufError::ElementCountOverflow {
                    name: self.name.clone(),
                })?;
        }
        Ok(product.max(1))
    }

    /// Total number of elements (legacy convenience, wraps on overflow).
    ///
    /// **Prefer [`checked_num_elements`](Self::checked_num_elements)** which
    /// returns `Err` on overflow instead of silently wrapping.
    #[deprecated(note = "use checked_num_elements() for safe overflow handling")]
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product::<u64>().max(1)
    }

    /// Total byte size with overflow checking and allocation cap.
    ///
    /// Returns an error if the product overflows u64, or if the resulting
    /// byte size exceeds `MAX_TENSOR_BYTE_SIZE` (8 GiB). This prevents
    /// crafted GGUF files from triggering extreme memory allocation.
    pub fn checked_byte_size(&self) -> Result<u64, GgufError> {
        let n = self.checked_num_elements()?;
        let bs = self.dtype.block_size() as u64;
        let ts = self.dtype.type_size() as u64;
        if bs == 0 {
            return Err(GgufError::ByteSizeOverflow {
                name: self.name.clone(),
                elements: n,
                type_size: ts,
                block_size: bs,
            });
        }
        let num_blocks = n / bs;
        let byte_size = num_blocks
            .checked_mul(ts)
            .ok_or_else(|| GgufError::ByteSizeOverflow {
                name: self.name.clone(),
                elements: n,
                type_size: ts,
                block_size: bs,
            })?;
        if byte_size > MAX_TENSOR_BYTE_SIZE {
            return Err(GgufError::TensorTooLarge {
                name: self.name.clone(),
                byte_size,
                max: MAX_TENSOR_BYTE_SIZE,
            });
        }
        Ok(byte_size)
    }

    /// Total byte size of this tensor's data in the file (wraps on overflow).
    ///
    /// **Prefer [`checked_byte_size`](Self::checked_byte_size)** which returns
    /// `Err` on overflow and enforces the 8 GiB allocation cap.
    #[deprecated(note = "use checked_byte_size() for safe overflow handling")]
    #[allow(deprecated)]
    pub fn byte_size(&self) -> u64 {
        let n = self.num_elements();
        let bs = self.dtype.block_size() as u64;
        let ts = self.dtype.type_size() as u64;
        (n / bs) * ts
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_num_elements() {
        let info = GgufTensorInfo {
            name: "test".into(),
            n_dims: 2,
            shape: vec![4, 8],
            dtype: GgufDType::F32,
            offset: 0,
        };
        assert_eq!(info.num_elements(), 32);
    }

    #[test]
    fn test_byte_size_f32() {
        let info = GgufTensorInfo {
            name: "test".into(),
            n_dims: 1,
            shape: vec![100],
            dtype: GgufDType::F32,
            offset: 0,
        };
        assert_eq!(info.byte_size(), 400); // 100 * 4 bytes
    }

    #[test]
    fn test_byte_size_q4_0() {
        let info = GgufTensorInfo {
            name: "test".into(),
            n_dims: 1,
            shape: vec![256],
            dtype: GgufDType::Q4_0,
            offset: 0,
        };
        // 256 elements / 32 per block = 8 blocks * 18 bytes = 144
        assert_eq!(info.byte_size(), 144);
    }
}
