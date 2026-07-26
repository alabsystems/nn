// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Padding operations for [`DynTensor`]: zero, constant, and reflection padding.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{Dim, DynTensor};
use crate::error::{Result, TensorError};
use ndarray::{ArrayD, IxDyn, SliceInfoElem};

use super::checked_buffer_len;

impl DynTensor {
    /// 1-D zero-padding. Pads the last dimension.
    ///
    /// Input shape: `[..., length]`
    /// Output shape: `[..., pad_left + length + pad_right]`
    pub fn pad1d(&self, pad_left: usize, pad_right: usize) -> Result<Self> {
        if self.rank() == 0 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }

        // GPU path: delegate to pad_with_zeros which uses zeros + cat on GPU.
        if self.device().is_gpu() {
            let last = self.rank() - 1;
            return self.pad_with_zeros(last, pad_left, pad_right);
        }

        if pad_left == 0 && pad_right == 0 {
            return Ok(self.clone());
        }

        let tracing = trace::is_tracing();
        let input_dtype = self.dtype;
        let input_c = self.contiguous()?;
        let arr = input_c.to_f32_array()?;
        let old_shape = arr.shape();
        let last = old_shape.len() - 1;
        let old_len = old_shape[last];
        let new_len = old_len + pad_left + pad_right;

        let mut new_shape: Vec<usize> = old_shape.to_vec();
        new_shape[last] = new_len;

        let mut result_arr = ArrayD::<f32>::zeros(IxDyn(&new_shape));

        // Copy source data with offset
        let prefix_size = checked_buffer_len(&old_shape[..last], "pad1d: prefix_size")?;
        let src = arr.as_slice().ok_or_else(|| {
            TensorError::InvalidShape("pad1d: input not contiguous after contiguous()".into())
        })?;
        let dst = result_arr.as_slice_mut().ok_or_else(|| {
            TensorError::InvalidShape("pad1d: output array not contiguous after allocation".into())
        })?;

        for i in 0..prefix_size {
            let src_start = i * old_len;
            let dst_start = i * new_len + pad_left;
            dst[dst_start..dst_start + old_len]
                .copy_from_slice(&src[src_start..src_start + old_len]);
        }

        let mut result = Self::from_f32_result(result_arr, input_dtype)?;
        // Record ConstantPadNd trace op so the trace-to-graph translator can
        // reconstruct the padding for NY IBP propagation.
        if tracing {
            let input_ids = Self::trace_input_ids(&[self])?;
            // PyTorch convention: [last_dim_left, last_dim_right].
            let padding = vec![pad_left, pad_right];
            if let Some(id) = trace::record_op(
                TraceOp::ConstantPadNd {
                    padding,
                    value: 0.0,
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Zero-pad a specific dimension on both sides.
    ///
    /// Equivalent to candle's `Tensor::pad_with_zeros(dim, left, right)`.
    ///
    /// `pad_with_zeros(dim, left, right)` creates a new tensor where dimension
    /// `dim` is extended by `left` zeros prepended and `right` zeros appended.
    ///
    /// # Examples
    /// ```text
    /// // x: [B, C, T] → pad_with_zeros(2, 3, 0) → [B, C, 3+T]  (causal left-pad)
    /// // x: [B, C, T] → pad_with_zeros(2, 1, 1) → [B, C, T+2]  (symmetric pad)
    /// ```
    pub fn pad_with_zeros(&self, dim: impl Dim, left: usize, right: usize) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        if left == 0 && right == 0 {
            return Ok(self.clone());
        }
        // GPU path: decompose into zeros + cat (GPU has no ndarray fast path).
        if self.device().is_gpu() {
            let mut parts: Vec<Self> = Vec::with_capacity(3);
            if left > 0 {
                let mut pad_shape = self.dims().to_vec();
                pad_shape[dim] = left;
                parts.push(Self::zeros(&pad_shape, self.dtype(), &self.device())?);
            }
            parts.push(self.clone());
            if right > 0 {
                let mut pad_shape = self.dims().to_vec();
                pad_shape[dim] = right;
                parts.push(Self::zeros(&pad_shape, self.dtype(), &self.device())?);
            }
            let refs: Vec<&Self> = parts.iter().collect();
            return Self::cat(&refs, dim);
        }
        let input_dtype = self.dtype;
        let input_c = self.contiguous()?;
        let arr = input_c.to_f32_array()?;
        let old_shape = arr.shape();
        let old_dim_len = old_shape[dim];
        let new_dim_len = old_dim_len + left + right;

        let mut new_shape: Vec<usize> = old_shape.to_vec();
        new_shape[dim] = new_dim_len;

        let mut result = ArrayD::<f32>::zeros(IxDyn(&new_shape));

        // Copy source data into the padded region using slice assignment.
        // Build an ndarray SliceInfo to select [0..all, ..., left..left+old_dim_len, ..., 0..all]
        let slices: Vec<SliceInfoElem> = (0..self.rank())
            .map(|d| {
                if d == dim {
                    SliceInfoElem::Slice {
                        start: left as isize,
                        end: Some((left + old_dim_len) as isize),
                        step: 1,
                    }
                } else {
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
            })
            .collect();
        result.slice_mut(slices.as_slice()).assign(&arr);

        let mut out = Self::from_f32_result(result, input_dtype)?;
        // Record ConstantPadNd trace op so trace-to-graph can reconstruct
        // the padding for NY IBP propagation (#2987).
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            // PyTorch ConstantPadNd convention: pairs from last dim to first.
            // Only the padded dimension has non-zero entries.
            let rank = self.rank();
            let num_pairs = rank; // one (left, right) pair per dimension
            let mut padding = vec![0usize; num_pairs * 2];
            // Dimension `dim` maps to pair index `rank - 1 - dim` in PyTorch convention.
            // Each pair occupies [2*pair_idx, 2*pair_idx+1] = (left, right).
            let pair_idx = rank - 1 - dim;
            padding[2 * pair_idx] = left;
            padding[2 * pair_idx + 1] = right;
            if let Some(id) = trace::record_op(
                TraceOp::ConstantPadNd {
                    padding,
                    value: 0.0,
                },
                &input_ids,
                out.dims(),
                out.dtype(),
            ) {
                out.set_trace_id(id);
            }
        }
        Ok(out)
    }

    /// 1-D reflection padding on the last dimension.
    ///
    /// Mirrors samples at the boundary (excluding the boundary element itself),
    /// matching PyTorch `nn.ReflectionPad1d((pad_left, pad_right))`.
    ///
    /// Requires `pad_left < dim_len` and `pad_right < dim_len`.
    ///
    /// ```text
    /// // [a, b, c, d, e] with pad_left=2, pad_right=1
    /// // → [c, b, a, b, c, d, e, d]
    /// ```
    pub fn reflection_pad1d(&self, pad_left: usize, pad_right: usize) -> Result<Self> {
        if self.rank() == 0 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if pad_left == 0 && pad_right == 0 {
            return Ok(self.clone());
        }
        let last = self.rank() - 1;
        let dim_len = self.dims()[last];
        if pad_left >= dim_len || pad_right >= dim_len {
            return Err(TensorError::InvalidShape(format!(
                "reflection_pad1d: padding ({pad_left}, {pad_right}) must be \
                 less than input size {dim_len} along last dim"
            )));
        }

        // Decompose into narrow + flip + cat. Works on both CPU and GPU.
        let mut parts: Vec<Self> = Vec::with_capacity(3);

        if pad_left > 0 {
            // Elements at indices [1..=pad_left], reversed.
            let left_slice = self.narrow(last, 1, pad_left)?;
            parts.push(left_slice.flip(last)?);
        }

        parts.push(self.clone());

        if pad_right > 0 {
            // Elements at indices [dim_len-pad_right-1..dim_len-2], reversed.
            let right_slice = self.narrow(last, dim_len - pad_right - 1, pad_right)?;
            parts.push(right_slice.flip(last)?);
        }

        let refs: Vec<&Self> = parts.iter().collect();
        Self::cat(&refs, last)
    }

    /// 2-D reflection padding on the last two dimensions.
    ///
    /// Mirrors samples at the boundary (excluding the boundary element itself),
    /// matching PyTorch `nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))`.
    ///
    /// Requires `pad_left < W`, `pad_right < W`, `pad_top < H`, `pad_bottom < H`
    /// where H and W are the last two dimension sizes.
    ///
    /// ```text
    /// // For 2D input [H, W], pad_left=1, pad_right=1, pad_top=1, pad_bottom=1:
    /// // [[a, b, c],      [[e, d, e, f, e],
    /// //  [d, e, f],  →    [b, a, b, c, b],
    /// //  [g, h, i]]       [e, d, e, f, e],
    /// //                   [h, g, h, i, h],
    /// //                   [e, d, e, f, e]]
    /// ```
    pub fn reflection_pad2d(
        &self,
        pad_left: usize,
        pad_right: usize,
        pad_top: usize,
        pad_bottom: usize,
    ) -> Result<Self> {
        if self.rank() < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }
        if pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0 {
            return Ok(self.clone());
        }

        let rank = self.rank();
        let h_dim = rank - 2;
        let w_dim = rank - 1;
        let h_len = self.dims()[h_dim];
        let w_len = self.dims()[w_dim];

        if pad_left >= w_len || pad_right >= w_len {
            return Err(TensorError::InvalidShape(format!(
                "reflection_pad2d: horizontal padding ({pad_left}, {pad_right}) must be \
                 less than input width {w_len}"
            )));
        }
        if pad_top >= h_len || pad_bottom >= h_len {
            return Err(TensorError::InvalidShape(format!(
                "reflection_pad2d: vertical padding ({pad_top}, {pad_bottom}) must be \
                 less than input height {h_len}"
            )));
        }

        // Pad width (last dim) first, then height (second-to-last dim).
        // Decompose into narrow + flip + cat, same pattern as reflection_pad1d.
        let width_padded = if pad_left > 0 || pad_right > 0 {
            let mut parts: Vec<Self> = Vec::with_capacity(3);
            if pad_left > 0 {
                let left_slice = self.narrow(w_dim, 1, pad_left)?;
                parts.push(left_slice.flip(w_dim)?);
            }
            parts.push(self.clone());
            if pad_right > 0 {
                let right_slice = self.narrow(w_dim, w_len - pad_right - 1, pad_right)?;
                parts.push(right_slice.flip(w_dim)?);
            }
            let refs: Vec<&Self> = parts.iter().collect();
            Self::cat(&refs, w_dim)?
        } else {
            self.clone()
        };

        // Pad height (second-to-last dim).
        let result = if pad_top > 0 || pad_bottom > 0 {
            let mut parts: Vec<Self> = Vec::with_capacity(3);
            if pad_top > 0 {
                let top_slice = width_padded.narrow(h_dim, 1, pad_top)?;
                parts.push(top_slice.flip(h_dim)?);
            }
            parts.push(width_padded);
            if pad_bottom > 0 {
                let padded_h = h_len; // original H before height padding
                let bottom_slice =
                    parts
                        .last()
                        .unwrap()
                        .narrow(h_dim, padded_h - pad_bottom - 1, pad_bottom)?;
                parts.push(bottom_slice.flip(h_dim)?);
            }
            let refs: Vec<&Self> = parts.iter().collect();
            Self::cat(&refs, h_dim)?
        } else {
            width_padded
        };

        // Record trace op if tracing is active.
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::ReflectionPad2d {
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom,
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                let mut traced = result;
                traced.set_trace_id(id);
                return Ok(traced);
            }
        }

        Ok(result)
    }

    /// N-D constant padding with a fill value.
    ///
    /// `padding` follows PyTorch convention: pairs from the last dimension to
    /// the first. For example, `&[1, 2]` pads the last dim with 1 on the left
    /// and 2 on the right. `&[1, 2, 3, 4]` additionally pads the second-to-last
    /// dim with 3 on the top and 4 on the bottom.
    ///
    /// Matches PyTorch `nn.functional.pad(input, padding, mode='constant', value=value)`.
    pub fn constant_pad_nd(&self, padding: &[usize], value: f64) -> Result<Self> {
        if !padding.len().is_multiple_of(2) {
            return Err(TensorError::InvalidShape(
                "constant_pad_nd: padding must have even length (pairs of left/right)".into(),
            ));
        }
        let num_padded_dims = padding.len() / 2;
        if num_padded_dims > self.rank() {
            return Err(TensorError::InvalidShape(format!(
                "constant_pad_nd: padding specifies {} dims but tensor has rank {}",
                num_padded_dims,
                self.rank()
            )));
        }

        // Check if all padding is zero — shortcut.
        if padding.iter().all(|&p| p == 0) {
            return Ok(self.clone());
        }

        let rank = self.rank();
        let input_dtype = self.dtype;
        let val_f32 = crate::dyn_tensor::checked_f64_to_f32(value, "constant_pad_nd() value")?;
        let input_c = self.contiguous()?;
        let arr = input_c.to_f32_array()?;
        let old_shape = arr.shape();

        // Build the new shape.
        let mut new_shape: Vec<usize> = old_shape.to_vec();
        for pair_idx in 0..num_padded_dims {
            let dim = rank - 1 - pair_idx;
            let pad_left = padding[2 * pair_idx];
            let pad_right = padding[2 * pair_idx + 1];
            new_shape[dim] = old_shape[dim] + pad_left + pad_right;
        }

        // Create output filled with the constant value.
        let mut result_arr = ArrayD::<f32>::from_elem(IxDyn(&new_shape), val_f32);

        // Copy source data into the padded region using slice assignment.
        let slices: Vec<SliceInfoElem> = (0..rank)
            .map(|d| {
                let pair_idx_from_end = rank - 1 - d;
                if pair_idx_from_end < num_padded_dims {
                    let pad_left = padding[2 * pair_idx_from_end];
                    let old_dim = old_shape[d];
                    SliceInfoElem::Slice {
                        start: pad_left as isize,
                        end: Some((pad_left + old_dim) as isize),
                        step: 1,
                    }
                } else {
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
            })
            .collect();
        result_arr.slice_mut(slices.as_slice()).assign(&arr);

        let mut result = Self::from_f32_result(result_arr, input_dtype)?;

        // Record trace op.
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::ConstantPadNd {
                    padding: padding.to_vec(),
                    value,
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }

        Ok(result)
    }
}
