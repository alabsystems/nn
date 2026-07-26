// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! im2col (unfold) operations for GEMM-based convolution backward passes.
//!
//! `im2col_1d` unfolds a 3D input `[B, C, L]` into a column matrix
//! `[B, C*K, L_out]` where each spatial column contains the receptive field
//! for one output position. Combined with `matmul`, this computes the kernel
//! gradient as a GEMM instead of nested CPU loops.
//!
//! All operations compose existing DynTensor ops (pad, narrow, gather, cat, reshape)
//! that have native GPU dispatch, so im2col works on any device without explicit
//! CPU transfers.

use crate::dyn_tensor::DynTensor;
use crate::error::{Result, TensorError};

use super::conv1d_out_len;

impl DynTensor {
    /// Unfold a 1D input into column matrix for GEMM-based convolution.
    ///
    /// Input shape: `[batch, channels, length]`
    /// Output shape: `[batch, channels * kernel_size, out_length]`
    ///
    /// Each column `output[:, :, t]` contains the flattened receptive field patch
    /// for output position `t`, matching the im2col layout used by GEMM-based
    /// convolution implementations.
    ///
    /// This is device-agnostic: composes pad, narrow, gather, and cat ops that
    /// all have native GPU dispatch.
    pub fn im2col_1d(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self> {
        let dims = self.dims();
        if dims.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: dims.len(),
            });
        }
        let in_len = dims[2];
        let out_len = conv1d_out_len(in_len, kernel_size, padding, stride, dilation)?;

        // Pad input along spatial dimension (dim 2)
        let padded = if padding > 0 {
            self.pad1d(padding, padding)?
        } else {
            self.clone()
        };
        let padded_len = padded.dims()[2];

        // Build index tensor for strided selection: [0, stride, 2*stride, ...]
        // Shape: [out_len] — used for index_select on the spatial dim.
        let needs_stride_select = stride > 1;
        let indices = if needs_stride_select {
            let idx_data: Vec<u32> = (0..out_len)
                .map(|t| {
                    u32::try_from(t * stride).map_err(|_| {
                        TensorError::InvalidShape(format!(
                            "im2col_1d: stride index {} overflows u32",
                            t * stride
                        ))
                    })
                })
                .collect::<Result<Vec<u32>>>()?;
            Some(Self::from_vec_u32(
                idx_data,
                &[out_len],
                &self.device(),
            )?)
        } else {
            None
        };

        // For each kernel position ki, extract the spatial slice at offset ki*dilation.
        // If stride > 1, gather every stride-th element. Otherwise, narrow directly.
        let mut columns = Vec::with_capacity(kernel_size);
        for ki in 0..kernel_size {
            let offset = ki * dilation;
            if needs_stride_select {
                // With stride: narrow to range [offset, offset + (out_len-1)*stride + 1),
                // then gather at stride positions.
                let range_len = (out_len - 1) * stride + 1;
                if offset + range_len > padded_len {
                    return Err(TensorError::InvalidShape(format!(
                        "im2col_1d: patch exceeds padded length (offset={offset}, range_len={range_len}, padded_len={padded_len})"
                    )));
                }
                let sliced = padded.narrow(2, offset, range_len)?;
                // index_select along dim 2 to pick every stride-th element
                let idx = indices.as_ref().ok_or_else(|| {
                    TensorError::InvalidShape(
                        "im2col_1d: stride indices missing (internal error)".into(),
                    )
                })?;
                let col = sliced.index_select(idx, 2)?;
                columns.push(col);
            } else {
                // stride == 1: narrow directly to [offset, offset + out_len)
                if offset + out_len > padded_len {
                    return Err(TensorError::InvalidShape(format!(
                        "im2col_1d: patch exceeds padded length (offset={offset}, out_len={out_len}, padded_len={padded_len})"
                    )));
                }
                let col = padded.narrow(2, offset, out_len)?;
                columns.push(col);
            }
        }

        // Each column is [B, C, L_out]. We want [B, C*K, L_out].
        // For K=1, just return the single column.
        if kernel_size == 1 {
            return columns.into_iter().next().ok_or_else(|| {
                TensorError::InvalidShape("im2col_1d: empty columns (internal error)".into())
            });
        }

        // Interleave: for proper im2col layout, columns should be ordered as
        // [c0_k0, c0_k1, ..., c0_kK, c1_k0, c1_k1, ..., c1_kK, ...]
        // But cat along dim 1 gives [c0_k0..cC_k0, c0_k1..cC_k1, ...] (channels first).
        // We need to reshape to separate channels and kernel positions, then transpose.

        // Stack all K columns: cat along dim 1 gives [B, C*K, L_out] but with wrong ordering
        // (all channels for k=0, then all channels for k=1, etc.)
        // Correct ordering: channels interleaved with kernel positions.
        //
        // Approach: reshape each column from [B, C, L_out] to [B, C, 1, L_out],
        // cat along dim 2 gives [B, C, K, L_out], then reshape to [B, C*K, L_out].
        let batch = dims[0];
        let channels = dims[1];

        let reshaped_cols: Vec<Self> = columns
            .iter()
            .map(|col| col.reshape([batch, channels, 1, out_len]))
            .collect::<Result<Vec<_>>>()?;

        let stacked = Self::cat(&reshaped_cols, 2)?; // [B, C, K, L_out]
        stacked.reshape([batch, channels * kernel_size, out_len])
    }

    /// Unfold a 2D input into column matrix for GEMM-based convolution.
    ///
    /// Input shape: `[batch, channels, height, width]`
    /// Output shape: `[batch, channels * kH * kW, H_out * W_out]`
    ///
    /// Device-agnostic: composes pad, narrow, gather, cat, and reshape ops.
    pub fn im2col_2d(
        &self,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self> {
        let dims = self.dims();
        if dims.len() != 4 {
            return Err(TensorError::RankMismatch {
                expected: 4,
                actual: dims.len(),
            });
        }
        let (batch, channels, in_h, in_w) = (dims[0], dims[1], dims[2], dims[3]);

        let out_h = conv1d_out_len(in_h, kernel_h, padding, stride, dilation)?;
        let out_w = conv1d_out_len(in_w, kernel_w, padding, stride, dilation)?;

        // Pad input along H and W (dims 2, 3)
        let padded = if padding > 0 {
            self.pad_with_zeros(2, padding, padding)?
                .pad_with_zeros(3, padding, padding)?
        } else {
            self.clone()
        };
        // Build stride indices for H and W if needed
        let needs_h_stride = stride > 1;
        let needs_w_stride = stride > 1;
        let h_indices = if needs_h_stride {
            let idx: Vec<u32> = (0..out_h)
                .map(|t| {
                    u32::try_from(t * stride).map_err(|_| {
                        TensorError::InvalidShape(format!(
                            "im2col_2d: H stride index {} overflows u32",
                            t * stride
                        ))
                    })
                })
                .collect::<Result<Vec<u32>>>()?;
            Some(Self::from_vec_u32(idx, &[out_h], &self.device())?)
        } else {
            None
        };
        let w_indices = if needs_w_stride {
            let idx: Vec<u32> = (0..out_w)
                .map(|t| {
                    u32::try_from(t * stride).map_err(|_| {
                        TensorError::InvalidShape(format!(
                            "im2col_2d: W stride index {} overflows u32",
                            t * stride
                        ))
                    })
                })
                .collect::<Result<Vec<u32>>>()?;
            Some(Self::from_vec_u32(idx, &[out_w], &self.device())?)
        } else {
            None
        };

        // For each (kh, kw), extract the spatial patch and collect.
        let mut columns = Vec::with_capacity(kernel_h * kernel_w);
        for kh in 0..kernel_h {
            let h_offset = kh * dilation;
            for kw in 0..kernel_w {
                let w_offset = kw * dilation;

                // Extract H slice
                let col = if needs_h_stride {
                    let h_range = (out_h - 1) * stride + 1;
                    let h_idx = h_indices.as_ref().ok_or_else(|| {
                        TensorError::InvalidShape(
                            "im2col_2d: h stride indices missing (internal error)".into(),
                        )
                    })?;
                    padded
                        .narrow(2, h_offset, h_range)?
                        .index_select(h_idx, 2)?
                } else {
                    padded.narrow(2, h_offset, out_h)?
                };

                // Extract W slice
                let col = if needs_w_stride {
                    let w_range = (out_w - 1) * stride + 1;
                    let w_idx = w_indices.as_ref().ok_or_else(|| {
                        TensorError::InvalidShape(
                            "im2col_2d: w stride indices missing (internal error)".into(),
                        )
                    })?;
                    col.narrow(3, w_offset, w_range)?.index_select(w_idx, 3)?
                } else {
                    col.narrow(3, w_offset, out_w)?
                };

                // col is [B, C, out_h, out_w]
                columns.push(col);
            }
        }

        let kk = kernel_h * kernel_w;

        // Reshape each column from [B, C, out_h, out_w] to [B, C, 1, out_h * out_w]
        let reshaped_cols: Vec<Self> = columns
            .iter()
            .map(|col| col.reshape([batch, channels, 1, out_h * out_w]))
            .collect::<Result<Vec<_>>>()?;

        // Cat along dim 2: [B, C, kH*kW, out_h*out_w]
        let stacked = Self::cat(&reshaped_cols, 2)?;
        // Reshape to [B, C*kH*kW, out_h*out_w]
        stacked.reshape([batch, channels * kk, out_h * out_w])
    }

    /// Helper: compute conv1d output length for backward gradient computation.
    ///
    /// Returns the output length for given conv parameters, used by GEMM-based
    /// backward rules.
    pub fn conv1d_output_len(
        input_len: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<usize> {
        conv1d_out_len(input_len, kernel_size, padding, stride, dilation)
    }
}

#[cfg(test)]
#[path = "im2col_tests.rs"]
mod tests;
