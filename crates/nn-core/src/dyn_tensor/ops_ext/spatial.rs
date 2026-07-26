// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spatial and mask operations for [`DynTensor`] — pixel shuffle, triu/tril, grid sample.
//!
//! Upsample ops (nearest 1D/2D, bilinear 2D) extracted to `spatial_upsample.rs`.
//! Grid sample (bilinear 2D at arbitrary coordinates) in `spatial_grid_sample.rs`.

#[path = "spatial_upsample.rs"]
mod upsample;

#[path = "spatial_grid_sample.rs"]
pub(super) mod grid_sample;

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::tensor::checked_dim_product;
use crate::{Result, TensorError};

impl DynTensor {
    /// Sub-pixel convolution: rearranges channels into spatial dimensions.
    ///
    /// Input: `[B, C * r², H, W]`. Output: `[B, C, H * r, W * r]`.
    /// Equivalent to `reshape → permute → reshape`:
    ///   `[B, C, r, r, H, W] → [B, C, H, r, W, r] → [B, C, H*r, W*r]`.
    ///
    /// Matches PyTorch `nn.PixelShuffle(upscale_factor)`.
    pub fn pixel_shuffle(&self, upscale_factor: usize) -> Result<Self> {
        if upscale_factor == 0 {
            return Err(TensorError::InvalidShape(
                "pixel_shuffle: upscale_factor must be > 0".into(),
            ));
        }
        if self.rank() < 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: self.rank(),
            });
        }
        let shape = self.dims();
        let rank = shape.len();
        let c_total = shape[rank - 3];
        let h = shape[rank - 2];
        let w = shape[rank - 1];
        let r2 = upscale_factor * upscale_factor;
        if !c_total.is_multiple_of(r2) {
            return Err(TensorError::InvalidShape(format!(
                "pixel_shuffle: channel dim {c_total} must be divisible by r²={r2}"
            )));
        }
        let c = c_total / r2;
        let r = upscale_factor;
        // Build intermediate shapes.
        let mut shape6: Vec<usize> = shape[..rank - 3].to_vec();
        shape6.extend_from_slice(&[c, r, r, h, w]);
        // Permute: move r dims after their spatial counterparts.
        // [batch..., C, r, r, H, W] → [batch..., C, H, r, W, r]
        let batch_dims = rank - 3;
        let mut perm: Vec<usize> = (0..batch_dims).collect();
        perm.push(batch_dims); // C
        perm.push(batch_dims + 3); // H
        perm.push(batch_dims + 1); // r (height)
        perm.push(batch_dims + 4); // W
        perm.push(batch_dims + 2); // r (width)
        let reshaped = self.reshape(&shape6)?;
        let permuted = reshaped.permute(&perm)?;
        // Final reshape to merge spatial dims.
        let mut final_shape: Vec<usize> = shape[..rank - 3].to_vec();
        final_shape.extend_from_slice(&[c, h * r, w * r]);
        permuted.contiguous()?.reshape(&final_shape)
    }

    /// Inverse sub-pixel convolution: rearranges spatial dimensions into channels.
    ///
    /// Input: `[B, C, H * r, W * r]`. Output: `[B, C * r², H, W]`.
    /// Inverse of [`pixel_shuffle`](Self::pixel_shuffle).
    ///
    /// Matches PyTorch `nn.PixelUnshuffle(downscale_factor)`.
    pub fn pixel_unshuffle(&self, downscale_factor: usize) -> Result<Self> {
        if downscale_factor == 0 {
            return Err(TensorError::InvalidShape(
                "pixel_unshuffle: downscale_factor must be > 0".into(),
            ));
        }
        if self.rank() < 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: self.rank(),
            });
        }
        let shape = self.dims();
        let rank = shape.len();
        let c = shape[rank - 3];
        let h = shape[rank - 2];
        let w = shape[rank - 1];
        let r = downscale_factor;
        if !h.is_multiple_of(r) || !w.is_multiple_of(r) {
            return Err(TensorError::InvalidShape(format!(
                "pixel_unshuffle: H={h} and W={w} must be divisible by r={r}"
            )));
        }
        let out_h = h / r;
        let out_w = w / r;
        // [batch..., C, H, W] → [batch..., C, out_h, r, out_w, r]
        let batch_dims = rank - 3;
        let mut shape6: Vec<usize> = shape[..rank - 3].to_vec();
        shape6.extend_from_slice(&[c, out_h, r, out_w, r]);
        // Permute: [batch..., C, out_h, r, out_w, r] → [batch..., C, r, r, out_h, out_w]
        let mut perm: Vec<usize> = (0..batch_dims).collect();
        perm.push(batch_dims); // C
        perm.push(batch_dims + 2); // r (height)
        perm.push(batch_dims + 4); // r (width)
        perm.push(batch_dims + 1); // out_h
        perm.push(batch_dims + 3); // out_w
        let reshaped = self.reshape(&shape6)?;
        let permuted = reshaped.permute(&perm)?;
        let mut final_shape: Vec<usize> = shape[..rank - 3].to_vec();
        final_shape.extend_from_slice(&[c * r * r, out_h, out_w]);
        permuted.contiguous()?.reshape(&final_shape)
    }

    /// Upper-triangular: zero out elements below the k-th diagonal.
    ///
    /// Operates on the last two dimensions. `diagonal = 0` is the main diagonal,
    /// positive values move up, negative values move down.
    /// Matches `torch.triu(input, diagonal=k)` / candle `Tensor::triu(diagonal)`.
    pub fn triu(&self, diagonal: i64) -> Result<Self> {
        // Suppress internal where_cond trace from GPU path; record composite Triu instead.
        let mut result = if trace::is_tracing() {
            trace::with_trace_suppressed(|| self.triangular_mask(diagonal, false))?
        } else {
            self.triangular_mask(diagonal, false)?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Triu { diagonal },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Lower-triangular: zero out elements above the k-th diagonal.
    ///
    /// Operates on the last two dimensions. `diagonal = 0` is the main diagonal,
    /// positive values move up, negative values move down.
    /// Matches `torch.tril(input, diagonal=k)` / candle `Tensor::tril(diagonal)`.
    pub fn tril(&self, diagonal: i64) -> Result<Self> {
        // Suppress internal where_cond trace from GPU path; record composite Tril instead.
        let mut result = if trace::is_tracing() {
            trace::with_trace_suppressed(|| self.triangular_mask(diagonal, true))?
        } else {
            self.triangular_mask(diagonal, true)?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Tril { diagonal },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Shared implementation for triu/tril.
    /// `lower = true` keeps lower triangle (tril), `lower = false` keeps upper (triu).
    fn triangular_mask(&self, diagonal: i64, lower: bool) -> Result<Self> {
        if self.rank() < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }
        let shape = self.dims();
        let rank = shape.len();
        let rows = shape[rank - 2];
        let cols = shape[rank - 1];
        // Build U8 keep-mask on CPU (cheap: 1 byte per element).
        let outer = checked_dim_product(&shape[..rank - 2])?;
        let rc = rows
            .checked_mul(cols)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;
        let alloc = outer
            .checked_mul(rc)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;
        let mut mask_data = vec![1u8; alloc];
        for batch in 0..outer {
            for r in 0..rows {
                // Use i128 to avoid overflow in (r as i64) + diagonal when
                // diagonal is near i64::MAX/MIN. Both r (usize) and diagonal
                // (i64) fit in i128 without loss.
                let threshold = (r as i128) + i128::from(diagonal);
                for c in 0..cols {
                    let ci = c as i128;
                    let zero = if lower {
                        ci > threshold
                    } else {
                        ci < threshold
                    };
                    if zero {
                        mask_data[batch * rc + r * cols + c] = 0;
                    }
                }
            }
        }
        let mask = Self::from_cpu_u8(ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(shape),
            mask_data,
        )?)?;
        // GPU path: transfer small mask to GPU, use where_cond to avoid full
        // data tensor CPU round-trip.
        if self.device().is_gpu() {
            let gpu_mask = mask.to_device(&self.device())?;
            let zeros = Self::zeros(shape, self.dtype, &self.device())?;
            return gpu_mask.where_cond(self, &zeros);
        }
        // CPU path: apply mask directly.
        let input_dtype = self.dtype;
        let arr = self.to_f32_array()?;
        let mask_arr = mask.as_cpu_u8()?;
        let result = ndarray::Zip::from(&arr)
            .and(&mask_arr)
            .map_collect(|&v, &m| if m != 0 { v } else { 0.0 });
        Self::from_f32_result(result, input_dtype)
    }
}
