// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor layout information and reshaping utilities for GGUF files.
//!
//! Provides [`TensorLayout`] (per-tensor shape, quantization, offset metadata),
//! [`LayoutMap`] (name-to-layout mapping built during GGUF parsing), and
//! reshaping utilities that validate element-count compatibility before
//! returning reinterpreted data.

use std::collections::HashMap;

use crate::dequant::GgufDType;
use crate::error::GgufError;
use crate::reader::GgufFile;
use crate::tensor_info::GgufTensorInfo;

/// Layout metadata for a single tensor in a GGUF file.
///
/// Tracks the logical shape, quantization type, and byte offset so that
/// callers can reason about tensor storage without re-parsing the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    /// Logical shape (dimension sizes). Follows GGUF row-major convention.
    pub shape: Vec<usize>,
    /// Quantization / data type.
    pub quant_type: GgufDType,
    /// Byte offset of this tensor's data from the start of the data section.
    pub byte_offset: u64,
}

impl TensorLayout {
    /// Total number of logical elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Total byte size of this tensor's data in the file.
    pub fn byte_size(&self) -> usize {
        compute_byte_size(&self.shape, self.quant_type)
    }
}

/// A map from tensor name to [`TensorLayout`].
///
/// Built from a parsed [`GgufFile`] via [`LayoutMap::from_gguf`], or
/// constructed incrementally via [`LayoutMap::insert`].
#[derive(Debug, Clone)]
pub struct LayoutMap {
    inner: HashMap<String, TensorLayout>,
}

impl LayoutMap {
    /// Create an empty layout map.
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Create a layout map with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(capacity),
        }
    }

    /// Build a `LayoutMap` from a parsed GGUF file.
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let mut map = Self::with_capacity(gguf.tensors.len());
        for (name, info) in &gguf.tensors {
            map.inner.insert(name.clone(), TensorLayout::from(info));
        }
        map
    }

    /// Insert a layout entry.
    pub fn insert(&mut self, name: String, layout: TensorLayout) {
        self.inner.insert(name, layout);
    }

    /// Look up a layout by tensor name.
    pub fn get(&self, name: &str) -> Option<&TensorLayout> {
        self.inner.get(name)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate over all (name, layout) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorLayout)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// All tensor names in the map.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.inner.keys().map(String::as_str)
    }
}

impl Default for LayoutMap {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&GgufTensorInfo> for TensorLayout {
    fn from(info: &GgufTensorInfo) -> Self {
        Self {
            shape: info.shape.iter().map(|&d| d as usize).collect(),
            quant_type: info.dtype,
            byte_offset: info.offset,
        }
    }
}

/// Compute the storage byte size for a tensor with the given shape and
/// quantization type.
///
/// For non-quantized types (F32, F16, BF16, I8, I16, I32, I64, F64),
/// this is simply `num_elements * bytes_per_element`.
///
/// For block-quantized types (Q4_0, Q8_0, etc.), this is
/// `(num_elements / block_size) * bytes_per_block`.
///
/// # Panics
///
/// Does not panic. Returns 0 for types with unknown `type_size`.
pub fn compute_byte_size(shape: &[usize], quant_type: GgufDType) -> usize {
    let num_elements: usize = shape.iter().product::<usize>().max(1);
    let block_size = quant_type.block_size();
    let type_size = quant_type.type_size();

    if block_size == 0 || type_size == 0 {
        return 0;
    }

    // For block-quantized types, elements must be a multiple of block_size.
    // Mimic GgufTensorInfo::byte_size: (n / block_size) * type_size.
    (num_elements / block_size) * type_size
}

/// Validate and reshape tensor data from one logical shape to another.
///
/// The source and target shapes must have the same total element count.
/// For non-quantized types, the raw bytes are simply returned as-is (the
/// byte layout is identical regardless of logical shape). For quantized
/// types, the target shape must also satisfy block alignment constraints.
///
/// # Errors
///
/// - [`GgufError::ReshapeElementMismatch`] if the element counts differ.
/// - [`GgufError::QuantBlockAlignment`] if a quantized type's target shape
///   has an innermost dimension not divisible by the block size.
pub fn reshape_tensor_data(
    data: &[u8],
    from: &TensorLayout,
    to_shape: &[usize],
) -> Result<Vec<u8>, GgufError> {
    let source_elements = from.num_elements();
    let target_elements: usize = to_shape.iter().product::<usize>().max(1);

    if source_elements != target_elements {
        return Err(GgufError::ReshapeElementMismatch {
            from_count: source_elements,
            to_count: target_elements,
        });
    }

    let block_size = from.quant_type.block_size();

    // For block-quantized types, verify the total element count is divisible
    // by the block size (it always should be if the source was valid, but
    // the target shape could introduce a mismatch if someone unflatten-ed
    // incorrectly).
    if block_size > 1 && !target_elements.is_multiple_of(block_size) {
        return Err(GgufError::QuantBlockAlignment {
            block_size,
            element_count: target_elements,
        });
    }

    // The raw byte layout is identical — quantized blocks are a flat sequence
    // regardless of logical shape. Return a copy of the data.
    Ok(data.to_vec())
}

/// Compute the flattened (1-D) shape for a tensor.
pub fn flatten_shape(shape: &[usize]) -> Vec<usize> {
    vec![shape.iter().product::<usize>().max(1)]
}

/// Unflatten a 1-D shape into a target shape, validating element count.
///
/// # Errors
///
/// Returns [`GgufError::ReshapeElementMismatch`] if the total element count
/// of `flat_shape` does not match `target_shape`.
pub fn unflatten_shape(
    flat_shape: &[usize],
    target_shape: &[usize],
) -> Result<Vec<usize>, GgufError> {
    let flat_elements: usize = flat_shape.iter().product::<usize>().max(1);
    let target_elements: usize = target_shape.iter().product::<usize>().max(1);

    if flat_elements != target_elements {
        return Err(GgufError::ReshapeElementMismatch {
            from_count: flat_elements,
            to_count: target_elements,
        });
    }

    Ok(target_shape.to_vec())
}

/// Compute the transposed shape (swap the last two dimensions).
///
/// For tensors with fewer than 2 dimensions, returns the shape unchanged.
pub fn transpose_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() < 2 {
        return shape.to_vec();
    }
    let mut result = shape.to_vec();
    let n = result.len();
    result.swap(n - 2, n - 1);
    result
}

/// Transpose tensor data by swapping the last two dimensions (via copy).
///
/// This performs an actual data rearrangement, not just a shape change.
/// Only supported for non-quantized element types where `block_size == 1`.
///
/// # Errors
///
/// - [`GgufError::QuantBlockAlignment`] if the type is block-quantized
///   (transpose requires element-level reordering, which is incompatible
///   with block quantization).
pub fn transpose_tensor_data(
    data: &[u8],
    layout: &TensorLayout,
) -> Result<(Vec<u8>, Vec<usize>), GgufError> {
    let block_size = layout.quant_type.block_size();
    if block_size > 1 {
        return Err(GgufError::QuantBlockAlignment {
            block_size,
            element_count: layout.num_elements(),
        });
    }

    if layout.shape.len() < 2 {
        return Ok((data.to_vec(), layout.shape.clone()));
    }

    let elem_bytes = layout.quant_type.type_size();
    if elem_bytes == 0 {
        return Ok((data.to_vec(), layout.shape.clone()));
    }

    let ndims = layout.shape.len();
    let rows = layout.shape[ndims - 2];
    let cols = layout.shape[ndims - 1];

    // Number of matrices (product of all dimensions except the last two).
    let batch: usize = layout.shape[..ndims - 2].iter().product::<usize>().max(1);
    let matrix_elems = rows * cols;
    let matrix_bytes = matrix_elems * elem_bytes;

    let mut output = vec![0u8; data.len()];

    for b in 0..batch {
        let base = b * matrix_bytes;
        for r in 0..rows {
            for c in 0..cols {
                let src_offset = base + (r * cols + c) * elem_bytes;
                let dst_offset = base + (c * rows + r) * elem_bytes;
                output[dst_offset..dst_offset + elem_bytes]
                    .copy_from_slice(&data[src_offset..src_offset + elem_bytes]);
            }
        }
    }

    let new_shape = transpose_shape(&layout.shape);
    Ok((output, new_shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // compute_byte_size
    // ---------------------------------------------------------------

    #[test]
    fn test_compute_byte_size_f32() {
        // 100 F32 elements = 100 * 4 = 400 bytes.
        assert_eq!(compute_byte_size(&[100], GgufDType::F32), 400);
        // 2-D: [4, 8] = 32 elements = 128 bytes.
        assert_eq!(compute_byte_size(&[4, 8], GgufDType::F32), 128);
        // 3-D: [2, 3, 4] = 24 elements = 96 bytes.
        assert_eq!(compute_byte_size(&[2, 3, 4], GgufDType::F32), 96);
    }

    #[test]
    fn test_compute_byte_size_f16() {
        // 100 F16 elements = 100 * 2 = 200 bytes.
        assert_eq!(compute_byte_size(&[100], GgufDType::F16), 200);
        assert_eq!(compute_byte_size(&[4, 8], GgufDType::F16), 64);
    }

    #[test]
    fn test_compute_byte_size_q4_0() {
        // Q4_0: block_size=32, type_size=18 bytes per block.
        // 256 elements / 32 = 8 blocks * 18 = 144 bytes.
        assert_eq!(compute_byte_size(&[256], GgufDType::Q4_0), 144);
        // 1024 elements / 32 = 32 blocks * 18 = 576 bytes.
        assert_eq!(compute_byte_size(&[1024], GgufDType::Q4_0), 576);
        // 2-D: [8, 32] = 256 elements → same as 1-D.
        assert_eq!(compute_byte_size(&[8, 32], GgufDType::Q4_0), 144);
    }

    #[test]
    fn test_compute_byte_size_q8_0() {
        // Q8_0: block_size=32, type_size=34 bytes per block.
        // 256 elements / 32 = 8 blocks * 34 = 272 bytes.
        assert_eq!(compute_byte_size(&[256], GgufDType::Q8_0), 272);
        // 32 elements / 32 = 1 block * 34 = 34 bytes.
        assert_eq!(compute_byte_size(&[32], GgufDType::Q8_0), 34);
    }

    #[test]
    fn test_compute_byte_size_empty_shape() {
        // Empty shape → product is 1 (scalar), but max(1) keeps it as 1.
        // 1 F32 element = 4 bytes.
        assert_eq!(compute_byte_size(&[], GgufDType::F32), 4);
    }

    // ---------------------------------------------------------------
    // reshape_tensor_data
    // ---------------------------------------------------------------

    #[test]
    fn test_reshape_same_elements() {
        let layout = TensorLayout {
            shape: vec![4, 8],
            quant_type: GgufDType::F32,
            byte_offset: 0,
        };
        let data = vec![0u8; 128]; // 32 f32 elements * 4 bytes
        let result = reshape_tensor_data(&data, &layout, &[2, 16]).unwrap();
        assert_eq!(result.len(), 128);
    }

    #[test]
    fn test_reshape_flatten() {
        let layout = TensorLayout {
            shape: vec![4, 8],
            quant_type: GgufDType::F32,
            byte_offset: 0,
        };
        let data = vec![0u8; 128];
        let flat = flatten_shape(&layout.shape);
        let result = reshape_tensor_data(&data, &layout, &flat).unwrap();
        assert_eq!(result.len(), 128);
    }

    #[test]
    fn test_reshape_element_count_mismatch() {
        let layout = TensorLayout {
            shape: vec![4, 8],
            quant_type: GgufDType::F32,
            byte_offset: 0,
        };
        let data = vec![0u8; 128];
        let result = reshape_tensor_data(&data, &layout, &[5, 7]);
        assert!(result.is_err());
        match result.unwrap_err() {
            GgufError::ReshapeElementMismatch {
                from_count,
                to_count,
            } => {
                assert_eq!(from_count, 32);
                assert_eq!(to_count, 35);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_reshape_quantized_block_alignment() {
        // Q4_0 has block_size=32. Reshaping 64 elements is fine (64 % 32 == 0).
        let layout = TensorLayout {
            shape: vec![64],
            quant_type: GgufDType::Q4_0,
            byte_offset: 0,
        };
        let data = vec![0u8; 36]; // 2 blocks * 18 bytes
        let result = reshape_tensor_data(&data, &layout, &[2, 32]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_quantized_block_alignment_error() {
        // Attempt to reshape 64 Q4_0 elements into [3, ...] which doesn't
        // change element count — but we test with mismatched count.
        let layout = TensorLayout {
            shape: vec![64],
            quant_type: GgufDType::Q4_0,
            byte_offset: 0,
        };
        // Create a shape with same element count but trigger mismatch.
        let data = vec![0u8; 36];
        let result = reshape_tensor_data(&data, &layout, &[3, 21]);
        assert!(result.is_err());
        // 3*21=63 != 64 → element mismatch, not block alignment.
        match result.unwrap_err() {
            GgufError::ReshapeElementMismatch {
                from_count,
                to_count,
            } => {
                assert_eq!(from_count, 64);
                assert_eq!(to_count, 63);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    // ---------------------------------------------------------------
    // LayoutMap
    // ---------------------------------------------------------------

    #[test]
    fn test_layout_map_construction_and_lookup() {
        let mut map = LayoutMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        map.insert(
            "weight".to_string(),
            TensorLayout {
                shape: vec![768, 256],
                quant_type: GgufDType::F32,
                byte_offset: 0,
            },
        );
        map.insert(
            "bias".to_string(),
            TensorLayout {
                shape: vec![768],
                quant_type: GgufDType::F16,
                byte_offset: 786432,
            },
        );

        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());

        let weight = map.get("weight").expect("should find weight");
        assert_eq!(weight.shape, vec![768, 256]);
        assert_eq!(weight.quant_type, GgufDType::F32);
        assert_eq!(weight.byte_offset, 0);
        assert_eq!(weight.num_elements(), 768 * 256);
        assert_eq!(weight.byte_size(), 768 * 256 * 4);

        let bias = map.get("bias").expect("should find bias");
        assert_eq!(bias.shape, vec![768]);
        assert_eq!(bias.quant_type, GgufDType::F16);
        assert_eq!(bias.num_elements(), 768);
        assert_eq!(bias.byte_size(), 768 * 2);

        assert!(map.get("nonexistent").is_none());
    }

    #[test]
    fn test_layout_map_iter_and_names() {
        let mut map = LayoutMap::new();
        map.insert(
            "a".to_string(),
            TensorLayout {
                shape: vec![10],
                quant_type: GgufDType::F32,
                byte_offset: 0,
            },
        );
        map.insert(
            "b".to_string(),
            TensorLayout {
                shape: vec![20],
                quant_type: GgufDType::F32,
                byte_offset: 40,
            },
        );

        let mut names: Vec<&str> = map.names().collect();
        names.sort_unstable();
        assert_eq!(names, vec!["a", "b"]);

        let entries: Vec<_> = map.iter().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_layout_from_tensor_info() {
        let info = GgufTensorInfo {
            name: "test".into(),
            n_dims: 2,
            shape: vec![4, 8],
            dtype: GgufDType::Q8_0,
            offset: 1024,
        };
        let layout = TensorLayout::from(&info);
        assert_eq!(layout.shape, vec![4, 8]);
        assert_eq!(layout.quant_type, GgufDType::Q8_0);
        assert_eq!(layout.byte_offset, 1024);
    }

    // ---------------------------------------------------------------
    // flatten / unflatten / transpose helpers
    // ---------------------------------------------------------------

    #[test]
    fn test_flatten_shape() {
        assert_eq!(flatten_shape(&[4, 8]), vec![32]);
        assert_eq!(flatten_shape(&[2, 3, 4]), vec![24]);
        assert_eq!(flatten_shape(&[100]), vec![100]);
    }

    #[test]
    fn test_unflatten_shape() {
        let result = unflatten_shape(&[24], &[2, 3, 4]).unwrap();
        assert_eq!(result, vec![2, 3, 4]);

        let err = unflatten_shape(&[24], &[2, 5, 4]);
        assert!(err.is_err());
    }

    #[test]
    fn test_transpose_shape() {
        assert_eq!(transpose_shape(&[4, 8]), vec![8, 4]);
        assert_eq!(transpose_shape(&[2, 3, 4]), vec![2, 4, 3]);
        assert_eq!(transpose_shape(&[100]), vec![100]);
        assert_eq!(transpose_shape(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_transpose_tensor_data_f32() {
        // 2x3 matrix of f32: [[1,2,3],[4,5,6]]
        // Transposed: [[1,4],[2,5],[3,6]]
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let layout = TensorLayout {
            shape: vec![2, 3],
            quant_type: GgufDType::F32,
            byte_offset: 0,
        };

        let (result_data, result_shape) = transpose_tensor_data(&data, &layout).unwrap();
        assert_eq!(result_shape, vec![3, 2]);

        // Read back f32 values from the transposed data.
        let result_values: Vec<f32> = result_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Expected transposed: [1, 4, 2, 5, 3, 6]
        assert_eq!(result_values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_quantized_rejected() {
        let layout = TensorLayout {
            shape: vec![32, 64],
            quant_type: GgufDType::Q4_0,
            byte_offset: 0,
        };
        let data = vec![0u8; compute_byte_size(&[32, 64], GgufDType::Q4_0)];
        let result = transpose_tensor_data(&data, &layout);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_1d_noop() {
        let layout = TensorLayout {
            shape: vec![8],
            quant_type: GgufDType::F32,
            byte_offset: 0,
        };
        let data = vec![0u8; 32];
        let (result_data, result_shape) = transpose_tensor_data(&data, &layout).unwrap();
        assert_eq!(result_shape, vec![8]);
        assert_eq!(result_data, data);
    }

    #[test]
    fn test_transpose_batched_f32() {
        // Batch of 2 matrices, each 2x2: [[[1,2],[3,4]],[[5,6],[7,8]]]
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let layout = TensorLayout {
            shape: vec![2, 2, 2],
            quant_type: GgufDType::F32,
            byte_offset: 0,
        };

        let (result_data, result_shape) = transpose_tensor_data(&data, &layout).unwrap();
        assert_eq!(result_shape, vec![2, 2, 2]);

        let result_values: Vec<f32> = result_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Transpose swaps last two dims:
        // Batch 0: [[1,2],[3,4]] → [[1,3],[2,4]]
        // Batch 1: [[5,6],[7,8]] → [[5,7],[6,8]]
        assert_eq!(result_values, vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
    }
}
