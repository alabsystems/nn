// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Grid sample operation for [`DynTensor`] — bilinear interpolation at arbitrary 2D coordinates.
//!
//! Matches PyTorch `F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros',
//! align_corners=False)`. Required by deformable attention (Zhu et al., 2021).
//!
//! Extracted from `spatial.rs` for file-size compliance.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::tensor::checked_dim_product;
use crate::{Result, TensorError};

/// Padding mode for out-of-bounds grid coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GridSamplePaddingMode {
    /// Out-of-bounds positions return 0.
    Zeros,
    /// Out-of-bounds coordinates are clamped to the border pixel.
    Border,
}

impl DynTensor {
    /// Sample from a 4D input at arbitrary 2D coordinates using bilinear interpolation.
    ///
    /// - `input`: `[B, C, H_in, W_in]` — the feature map to sample from.
    /// - `grid`: `[B, H_out, W_out, 2]` — normalized coordinates in `[-1, 1]`.
    ///   The last dimension contains `(x, y)` pairs where `(-1, -1)` is top-left
    ///   and `(1, 1)` is bottom-right (matching PyTorch convention).
    ///
    /// Returns: `[B, C, H_out, W_out]`.
    ///
    /// When `align_corners = true`, `-1` and `1` map exactly to corner pixels.
    /// When `align_corners = false`, `-1` and `1` map to the outside edges.
    ///
    /// Matches PyTorch `F.grid_sample(input, grid, mode='bilinear', ...)`.
    pub fn grid_sample(
        &self,
        grid: &Self,
        padding_mode: GridSamplePaddingMode,
        align_corners: bool,
    ) -> Result<Self> {
        // Validate input shape: [B, C, H_in, W_in]
        if self.rank() != 4 {
            return Err(TensorError::InvalidShape(format!(
                "grid_sample: input must be rank 4 [B, C, H, W], got rank {}",
                self.rank()
            )));
        }
        // Validate grid shape: [B, H_out, W_out, 2]
        if grid.rank() != 4 || grid.dim(3)? != 2 {
            return Err(TensorError::InvalidShape(format!(
                "grid_sample: grid must be rank 4 [B, H_out, W_out, 2], got {:?}",
                grid.dims()
            )));
        }
        let batch = self.dim(0)?;
        let channels = self.dim(1)?;
        let in_h = self.dim(2)?;
        let in_w = self.dim(3)?;
        let grid_b = grid.dim(0)?;
        let out_h = grid.dim(1)?;
        let out_w = grid.dim(2)?;

        if batch != grid_b {
            return Err(TensorError::InvalidShape(format!(
                "grid_sample: batch mismatch input={batch} vs grid={grid_b}"
            )));
        }
        if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
            return Err(TensorError::InvalidShape(
                "grid_sample: spatial dimensions must be > 0".into(),
            ));
        }

        // GPU path: CPU round-trip. Suppress tracing during round-trip (CPU
        // copies have no trace IDs), then record the composite GridSample op.
        if self.device().is_gpu() {
            let original_device = self.device();
            let mut result = trace::with_trace_suppressed(|| {
                let cpu_input = self.to_device(&crate::Device::Cpu)?;
                let cpu_grid = grid.to_device(&crate::Device::Cpu)?;
                let result = cpu_input.grid_sample(&cpu_grid, padding_mode, align_corners)?;
                result.to_device(&original_device)
            })?;
            if trace::is_tracing() {
                let input_ids = Self::trace_input_ids(&[self, grid])?;
                if let Some(id) = trace::record_op(
                    TraceOp::GridSample {
                        padding_mode,
                        align_corners,
                    },
                    &input_ids,
                    result.dims(),
                    result.dtype(),
                ) {
                    result.set_trace_id(id);
                }
            }
            return Ok(result);
        }

        let input_dtype = self.dtype;
        let input_data = self.to_f32_array()?;
        let grid_data = grid.to_f32_array()?;
        let input_flat: Vec<f32> = input_data.iter().copied().collect();
        let grid_flat: Vec<f32> = grid_data.iter().copied().collect();

        let chw = checked_dim_product(&[channels, in_h, in_w])?;
        let hw_in = in_h
            .checked_mul(in_w)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: self.dims().to_vec(),
            })?;
        let hw_out_2 = out_h
            .checked_mul(out_w)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: grid.dims().to_vec(),
            })?;
        let alloc = batch
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(out_h))
            .and_then(|v| v.checked_mul(out_w))
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: vec![batch, channels, out_h, out_w],
            })?;

        let mut output = Vec::with_capacity(alloc);

        for b in 0..batch {
            let input_batch = &input_flat[b * chw..(b + 1) * chw];
            let grid_batch = &grid_flat[b * hw_out_2..(b + 1) * hw_out_2];

            for c in 0..channels {
                let channel_offset = c * hw_in;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let grid_idx = (oh * out_w + ow) * 2;
                        let gx = f64::from(grid_batch[grid_idx]);
                        let gy = f64::from(grid_batch[grid_idx + 1]);

                        // Unnormalize grid coordinates from [-1, 1] to pixel indices.
                        let (ix, iy) = unnormalize(gx, gy, in_w, in_h, align_corners);

                        let val = bilinear_sample(
                            input_batch,
                            channel_offset,
                            in_h,
                            in_w,
                            ix,
                            iy,
                            padding_mode,
                        );
                        output.push(val);
                    }
                }
            }
        }

        let mut result = Self::from_f32_result(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[batch, channels, out_h, out_w]),
                output,
            )?,
            input_dtype,
        )?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, grid])?;
            if let Some(id) = trace::record_op(
                TraceOp::GridSample {
                    padding_mode,
                    align_corners,
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

/// Convert normalized grid coordinates `[-1, 1]` to pixel indices.
fn unnormalize(gx: f64, gy: f64, w: usize, h: usize, align_corners: bool) -> (f64, f64) {
    if align_corners {
        // -1 → 0, 1 → size-1
        let ix = (gx + 1.0) * 0.5 * (w as f64 - 1.0);
        let iy = (gy + 1.0) * 0.5 * (h as f64 - 1.0);
        (ix, iy)
    } else {
        // -1 → -0.5, 1 → size-0.5
        let ix = ((gx + 1.0) * w as f64 - 1.0) * 0.5;
        let iy = ((gy + 1.0) * h as f64 - 1.0) * 0.5;
        (ix, iy)
    }
}

/// Bilinear interpolation at fractional pixel coordinates.
fn bilinear_sample(
    channel_data: &[f32],
    channel_offset: usize,
    in_h: usize,
    in_w: usize,
    ix: f64,
    iy: f64,
    padding_mode: GridSamplePaddingMode,
) -> f32 {
    // NaN/Inf defense: non-finite coordinates produce wrong results via
    // saturating `NaN.floor() as i64 == 0`. NaN.clamp() also returns NaN,
    // so even Border mode does not protect. Return 0.0 (zero-padding).
    if !ix.is_finite() || !iy.is_finite() {
        return 0.0;
    }

    let (ix, iy) = match padding_mode {
        GridSamplePaddingMode::Border => {
            let ix = ix.clamp(0.0, (in_w as f64) - 1.0);
            let iy = iy.clamp(0.0, (in_h as f64) - 1.0);
            (ix, iy)
        }
        GridSamplePaddingMode::Zeros => (ix, iy),
    };

    let x0 = ix.floor() as i64;
    let y0 = iy.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let wx = (ix - x0 as f64) as f32;
    let wy = (iy - y0 as f64) as f32;

    let v00 = safe_pixel(
        channel_data,
        channel_offset,
        y0,
        x0,
        in_h,
        in_w,
        padding_mode,
    );
    let v01 = safe_pixel(
        channel_data,
        channel_offset,
        y0,
        x1,
        in_h,
        in_w,
        padding_mode,
    );
    let v10 = safe_pixel(
        channel_data,
        channel_offset,
        y1,
        x0,
        in_h,
        in_w,
        padding_mode,
    );
    let v11 = safe_pixel(
        channel_data,
        channel_offset,
        y1,
        x1,
        in_h,
        in_w,
        padding_mode,
    );

    v00 * (1.0 - wy) * (1.0 - wx) + v01 * (1.0 - wy) * wx + v10 * wy * (1.0 - wx) + v11 * wy * wx
}

/// Fetch a pixel value with bounds checking.
fn safe_pixel(
    channel_data: &[f32],
    channel_offset: usize,
    y: i64,
    x: i64,
    in_h: usize,
    in_w: usize,
    padding_mode: GridSamplePaddingMode,
) -> f32 {
    if y < 0 || x < 0 || y >= in_h as i64 || x >= in_w as i64 {
        match padding_mode {
            GridSamplePaddingMode::Zeros => 0.0,
            GridSamplePaddingMode::Border => {
                let cy = (y.max(0) as usize).min(in_h - 1);
                let cx = (x.max(0) as usize).min(in_w - 1);
                channel_data[channel_offset + cy * in_w + cx]
            }
        }
    } else {
        channel_data[channel_offset + y as usize * in_w + x as usize]
    }
}
