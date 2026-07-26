// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Upsample and resize operations for [`DynTensor`] — nearest-neighbor 1D/2D,
//! bilinear 2D upsample, and bilinear resize to absolute target dimensions.
//!
//! Extracted from `spatial.rs` for file-size compliance (#1342).

use crate::dyn_tensor::gpu::gpu_backend_dispatch;
use crate::dyn_tensor::trace::{self, TraceOp, TraceUpsampleMode};
use crate::dyn_tensor::DynTensor;
use crate::tensor::checked_dim_product;
use crate::{Result, TensorError};

/// Overflow-safe `dim * scale`, returning `DimensionOverflow` on overflow.
fn checked_scale(dim: usize, scale: usize, dims: &[usize]) -> Result<usize> {
    dim.checked_mul(scale)
        .ok_or_else(|| TensorError::DimensionOverflow {
            dims: dims.to_vec(),
        })
}

impl DynTensor {
    /// Nearest-neighbor 1D upsample by integer factor along last dimension.
    ///
    /// Input: `[..., T]`. Output: `[..., T * factor]`.
    /// Each element is repeated `factor` times along the last axis.
    pub fn upsample_nearest_1d(&self, factor: usize) -> Result<Self> {
        if factor == 0 {
            return Err(TensorError::InvalidShape(
                "upsample factor must be > 0".into(),
            ));
        }
        if factor == 1 {
            return Ok(self.clone());
        }
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::Upsample1d { factor }),
            || self.upsample_nearest_1d_impl(factor),
        )
    }

    /// Implementation body for nearest-neighbor 1D upsample. Called within
    /// trace suppression so decomposed ops are not individually recorded.
    fn upsample_nearest_1d_impl(&self, factor: usize) -> Result<Self> {
        // GPU path: decompose as unsqueeze → expand → reshape.
        // unsqueeze/reshape are metadata-only; expand uses broadcast_add on GPU.
        if self.device().is_gpu() {
            // [.., T] → [.., T, 1] → [.., T, factor] → [.., T*factor]
            let rank = self.rank();
            let t = self.dim(rank - 1)?;
            let expanded = self.unsqueeze(rank)?.expand(&{
                let mut s = self.dims().to_vec();
                s.push(factor);
                s
            })?;
            let mut out_shape = self.dims().to_vec();
            out_shape[rank - 1] =
                t.checked_mul(factor)
                    .ok_or_else(|| TensorError::DimensionOverflow {
                        dims: self.dims().to_vec(),
                    })?;
            return expanded.reshape(&out_shape);
        }
        let input_dtype = self.dtype;
        let arr = self.to_f32_array()?;
        let shape = arr.shape();
        let rank = shape.len();
        if rank == 0 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }
        let t = shape[rank - 1];
        let outer = checked_dim_product(&shape[..rank - 1])?;
        let t_out = t
            .checked_mul(factor)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;
        let alloc = outer
            .checked_mul(t_out)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;
        let flat: Vec<f32> = arr.iter().copied().collect();
        let mut out = Vec::with_capacity(alloc);
        for chunk in flat.chunks_exact(t) {
            for &val in chunk {
                for _ in 0..factor {
                    out.push(val);
                }
            }
        }
        let mut new_shape: Vec<usize> = shape.to_vec();
        new_shape[rank - 1] = t_out;
        Self::from_f32_result(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), out)?,
            input_dtype,
        )
    }

    /// Nearest-neighbor 2D upsample by integer scale factors along the last two dimensions.
    ///
    /// Input: `[..., H, W]`. Output: `[..., H * scale_h, W * scale_w]`.
    /// Each element is replicated in a `scale_h × scale_w` block.
    /// Matches PyTorch `F.interpolate(mode='nearest', scale_factor=(scale_h, scale_w))`.
    pub fn upsample_nearest_2d(&self, scale_h: usize, scale_w: usize) -> Result<Self> {
        if scale_h == 0 || scale_w == 0 {
            return Err(TensorError::InvalidShape(
                "upsample scale factors must be > 0".into(),
            ));
        }
        if self.rank() < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }
        if scale_h == 1 && scale_w == 1 {
            return Ok(self.clone());
        }
        trace::traced_forward(
            &[self],
            || {
                Ok(TraceOp::Upsample2d {
                    mode: TraceUpsampleMode::Nearest,
                    scale_h: scale_h as f64,
                    scale_w: scale_w as f64,
                })
            },
            || {
                if self.device().is_gpu() {
                    self.upsample_nearest_2d_gpu(scale_h, scale_w)
                } else {
                    self.upsample_nearest_2d_cpu(scale_h, scale_w)
                }
            },
        )
    }

    /// GPU path: decompose as unsqueeze → expand → reshape, twice (H then W).
    /// All ops are GPU-native (unsqueeze/reshape are metadata-only, expand uses broadcast).
    fn upsample_nearest_2d_gpu(&self, scale_h: usize, scale_w: usize) -> Result<Self> {
        let rank = self.rank();
        let in_h = self.dim(rank - 2)?;
        let in_w = self.dim(rank - 1)?;
        let out_h = checked_scale(in_h, scale_h, self.dims())?;
        let out_w = checked_scale(in_w, scale_w, self.dims())?;

        // Step 1: Repeat along H.
        // [.., H, W] → [.., H, 1, W] → [.., H, scale_h, W] → [.., H*scale_h, W]
        let mut expand_h = self.dims().to_vec();
        expand_h.insert(rank - 1, scale_h);
        let t = self.unsqueeze(rank - 1)?.expand(&expand_h)?;
        let mut mid = self.dims().to_vec();
        mid[rank - 2] = out_h;
        let t = t.reshape(&mid)?;

        // Step 2: Repeat along W.
        // [.., H*scale_h, W] → [.., W, 1] → [.., W, scale_w] → [.., W*scale_w]
        let mut expand_w = mid.clone();
        expand_w.push(scale_w);
        let t = t.unsqueeze(rank)?.expand(&expand_w)?;
        let mut out_shape = self.dims().to_vec();
        out_shape[rank - 2] = out_h;
        out_shape[rank - 1] = out_w;
        t.reshape(&out_shape)
    }

    /// CPU path: direct nearest-neighbor replication.
    fn upsample_nearest_2d_cpu(&self, scale_h: usize, scale_w: usize) -> Result<Self> {
        let shape = self.dims();
        let rank = shape.len();
        let in_h = shape[rank - 2];
        let in_w = shape[rank - 1];
        let out_h = checked_scale(in_h, scale_h, shape)?;
        let out_w = checked_scale(in_w, scale_w, shape)?;
        let outer = checked_dim_product(&shape[..rank - 2])?;
        let input_dtype = self.dtype;
        let arr = self.to_f32_array()?;
        let flat: Vec<f32> = arr.iter().copied().collect();
        let hw = in_h
            .checked_mul(in_w)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;
        let alloc = outer
            .checked_mul(out_h)
            .and_then(|v| v.checked_mul(out_w))
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;
        let mut out = Vec::with_capacity(alloc);
        for batch in flat.chunks_exact(hw) {
            for oh in 0..out_h {
                let ih = oh / scale_h;
                for ow in 0..out_w {
                    out.push(batch[ih * in_w + ow / scale_w]);
                }
            }
        }
        let mut new_shape = shape.to_vec();
        new_shape[rank - 2] = out_h;
        new_shape[rank - 1] = out_w;
        Self::from_f32_result(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), out)?,
            input_dtype,
        )
    }

    /// Bilinear 2D upsample by float scale factors along the last two dimensions.
    ///
    /// Input: `[..., H, W]`. Output: `[..., out_h, out_w]` where
    /// `out_h = H * scale_h`, `out_w = W * scale_w` (rounded to nearest integer).
    ///
    /// When `align_corners = true`, corner pixels of input and output are aligned:
    ///   `src_idx = out_idx * (in_size - 1) / (out_size - 1)`
    /// When `align_corners = false`, indices are computed as:
    ///   `src_idx = (out_idx + 0.5) / scale - 0.5`
    ///
    /// Matches PyTorch `F.interpolate(mode='bilinear', align_corners=...)`.
    pub fn upsample_bilinear_2d(
        &self,
        scale_h: f64,
        scale_w: f64,
        align_corners: bool,
    ) -> Result<Self> {
        if !scale_h.is_finite() || !scale_w.is_finite() || scale_h <= 0.0 || scale_w <= 0.0 {
            return Err(TensorError::InvalidShape(
                "upsample_bilinear_2d: scale factors must be finite and > 0".into(),
            ));
        }
        if self.rank() < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }
        trace::traced_forward(
            &[self],
            || {
                Ok(TraceOp::Upsample2d {
                    mode: TraceUpsampleMode::Bilinear,
                    scale_h,
                    scale_w,
                })
            },
            || self.upsample_bilinear_2d_compute(scale_h, scale_w, align_corners),
        )
    }

    /// Bilinear 2D upsample to explicit output dimensions.
    ///
    /// Input: `[..., H, W]`. Output: `[..., out_h, out_w]`.
    ///
    /// When `align_corners = true`, corner pixels of input and output are aligned:
    ///   `src_idx = out_idx * (in_size - 1) / (out_size - 1)`
    /// When `align_corners = false`, indices are computed as:
    ///   `src_idx = (out_idx + 0.5) * in_size / out_size - 0.5`
    ///
    /// This matches PyTorch `F.interpolate(size=(out_h, out_w), mode='bilinear',
    /// align_corners=...)`. Preferred over scale-factor variant when exact output
    /// dimensions are known (e.g., FPN/PAN matching feature map sizes).
    pub fn upsample_bilinear_2d_to_size(
        &self,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
    ) -> Result<Self> {
        if out_h == 0 || out_w == 0 {
            return Err(TensorError::InvalidShape(
                "upsample_bilinear_2d_to_size: output dimensions must be > 0".into(),
            ));
        }
        if self.rank() < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }
        let shape = self.dims();
        let rank = shape.len();
        let in_h = shape[rank - 2];
        let in_w = shape[rank - 1];
        // Compute effective scale for trace recording.
        let scale_h = out_h as f64 / in_h.max(1) as f64;
        let scale_w = out_w as f64 / in_w.max(1) as f64;
        trace::traced_forward(
            &[self],
            || {
                Ok(TraceOp::Upsample2d {
                    mode: TraceUpsampleMode::Bilinear,
                    scale_h,
                    scale_w,
                })
            },
            || self.bilinear_2d_core(out_h, out_w, align_corners),
        )
    }

    /// Compute body for bilinear 2D upsample. Called within trace suppression.
    fn upsample_bilinear_2d_compute(
        &self,
        scale_h: f64,
        scale_w: f64,
        align_corners: bool,
    ) -> Result<Self> {
        let shape = self.dims();
        let rank = shape.len();
        let in_h = shape[rank - 2];
        let in_w = shape[rank - 1];
        let out_h_f = (in_h as f64 * scale_h).round();
        let out_w_f = (in_w as f64 * scale_w).round();
        // Guard against f64-to-usize saturation: reject dimensions that would
        // exceed isize::MAX (Rust's allocation limit).
        let dim_limit = isize::MAX as f64;
        if out_h_f <= 0.0 || out_w_f <= 0.0 || out_h_f > dim_limit || out_w_f > dim_limit {
            return Err(TensorError::InvalidShape(
                "upsample_bilinear_2d: output dimensions must be > 0 and within allocation limits"
                    .into(),
            ));
        }
        let out_h = out_h_f as usize;
        let out_w = out_w_f as usize;
        self.bilinear_2d_core(out_h, out_w, align_corners)
    }

    /// Core bilinear interpolation logic shared by scale-factor and output-size APIs.
    ///
    /// Implements the standard bilinear interpolation for `[..., H, W]` tensors.
    /// Coordinate mapping follows PyTorch `F.interpolate` conventions:
    /// - `align_corners=true`: `src = dst * (in_size - 1) / (out_size - 1)`
    /// - `align_corners=false`: `src = (dst + 0.5) * in_size / out_size - 0.5`
    fn bilinear_2d_core(&self, out_h: usize, out_w: usize, align_corners: bool) -> Result<Self> {
        let shape = self.dims();
        let rank = shape.len();
        let in_h = shape[rank - 2];
        let in_w = shape[rank - 1];
        // GPU path: CPU round-trip (no native GPU upsample_bilinear_2d yet).
        if self.device().is_gpu() {
            let original_device = self.device();
            let cpu_input = self.to_device(&crate::Device::Cpu)?;
            let result = cpu_input.bilinear_2d_core(out_h, out_w, align_corners)?;
            return result.to_device(&original_device);
        }
        let outer = checked_dim_product(&shape[..rank - 2])?;
        let input_dtype = self.dtype;
        let arr = self.to_f32_array()?;
        let flat: Vec<f32> = arr.iter().copied().collect();
        let hw = in_h
            .checked_mul(in_w)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;
        let alloc = outer
            .checked_mul(out_h)
            .and_then(|v| v.checked_mul(out_w))
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;
        let mut out = Vec::with_capacity(alloc);
        for batch in flat.chunks_exact(hw) {
            for oh in 0..out_h {
                let src_y = bilinear_coord(oh, in_h, out_h, align_corners);
                let y0 = (src_y.floor() as usize).min(in_h - 1);
                let y1 = (y0 + 1).min(in_h - 1);
                let wy = (src_y - y0 as f64) as f32;
                for ow in 0..out_w {
                    let src_x = bilinear_coord(ow, in_w, out_w, align_corners);
                    let x0 = (src_x.floor() as usize).min(in_w - 1);
                    let x1 = (x0 + 1).min(in_w - 1);
                    let wx = (src_x - x0 as f64) as f32;
                    let v00 = batch[y0 * in_w + x0];
                    let v01 = batch[y0 * in_w + x1];
                    let v10 = batch[y1 * in_w + x0];
                    let v11 = batch[y1 * in_w + x1];
                    let val = v00 * (1.0 - wy) * (1.0 - wx)
                        + v01 * (1.0 - wy) * wx
                        + v10 * wy * (1.0 - wx)
                        + v11 * wy * wx;
                    out.push(val);
                }
            }
        }
        let mut new_shape = shape.to_vec();
        new_shape[rank - 2] = out_h;
        new_shape[rank - 1] = out_w;
        Self::from_f32_result(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), out)?,
            input_dtype,
        )
    }

    /// Bilinear interpolation resize to absolute target dimensions.
    ///
    /// Input: `[N, C, H_in, W_in]` or `[C, H_in, W_in]` (rank 3 or 4).
    /// Output: same batch/channel dims with `[target_h, target_w]`.
    ///
    /// Coordinate mapping: `src = (dst + 0.5) * (in_size / out_size) - 0.5`,
    /// clamped to `[0, in_size - 1]`. Matches PyTorch `F.interpolate(mode='bilinear',
    /// align_corners=False)`. Supports upscaling and downscaling.
    pub fn resize_bilinear(&self, target_h: usize, target_w: usize) -> Result<Self> {
        let rank = self.rank();
        if !(3..=4).contains(&rank) {
            return Err(TensorError::RankMismatch {
                expected: 3, // "3 or 4" — report minimum
                actual: rank,
            });
        }
        if target_h == 0 || target_w == 0 {
            return Err(TensorError::InvalidShape(
                "resize_bilinear: target dimensions must be > 0".into(),
            ));
        }
        let shape = self.dims();
        let in_h = shape[rank - 2];
        let in_w = shape[rank - 1];
        if in_h == 0 || in_w == 0 {
            return Err(TensorError::InvalidShape(
                "resize_bilinear: input spatial dimensions must be > 0".into(),
            ));
        }
        // Identity resize: no computation needed.
        if in_h == target_h && in_w == target_w {
            return Ok(self.clone());
        }
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::ResizeBilinear { target_h, target_w }),
            || self.resize_bilinear_compute(target_h, target_w),
        )
    }

    /// Compute body for bilinear resize. Called within trace suppression.
    fn resize_bilinear_compute(&self, target_h: usize, target_w: usize) -> Result<Self> {
        // GPU path: try native GPU kernel, fall back to CPU round-trip.
        if self.device().is_gpu() {
            if let Some(result) =
                gpu_backend_dispatch(|b| b.resize_bilinear(self, target_h, target_w))
            {
                return result;
            }
            // Fallback: CPU round-trip.
            let original_device = self.device();
            let cpu_input = self.to_device(&crate::Device::Cpu)?;
            let result = cpu_input.resize_bilinear_compute(target_h, target_w)?;
            return result.to_device(&original_device);
        }
        let shape = self.dims();
        let rank = shape.len();
        let in_h = shape[rank - 2];
        let in_w = shape[rank - 1];
        let outer = checked_dim_product(&shape[..rank - 2])?;
        let input_dtype = self.dtype;
        let arr = self.to_f32_array()?;
        let flat: Vec<f32> = arr.iter().copied().collect();
        let hw = in_h
            .checked_mul(in_w)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;
        let alloc = outer
            .checked_mul(target_h)
            .and_then(|v| v.checked_mul(target_w))
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;
        let scale_y = in_h as f64 / target_h as f64;
        let scale_x = in_w as f64 / target_w as f64;
        let mut out = Vec::with_capacity(alloc);
        for batch in flat.chunks_exact(hw) {
            for oh in 0..target_h {
                let src_y = resize_bilinear_coord(oh, scale_y, in_h);
                let y0 = src_y.floor() as usize;
                let y1 = (y0 + 1).min(in_h - 1);
                let y0 = y0.min(in_h - 1);
                let wy = (src_y - y0 as f64) as f32;
                for ow in 0..target_w {
                    let src_x = resize_bilinear_coord(ow, scale_x, in_w);
                    let x0 = src_x.floor() as usize;
                    let x1 = (x0 + 1).min(in_w - 1);
                    let x0 = x0.min(in_w - 1);
                    let wx = (src_x - x0 as f64) as f32;
                    let v00 = batch[y0 * in_w + x0];
                    let v01 = batch[y0 * in_w + x1];
                    let v10 = batch[y1 * in_w + x0];
                    let v11 = batch[y1 * in_w + x1];
                    let val = v00 * (1.0 - wy) * (1.0 - wx)
                        + v01 * (1.0 - wy) * wx
                        + v10 * wy * (1.0 - wx)
                        + v11 * wy * wx;
                    out.push(val);
                }
            }
        }
        let mut new_shape = shape.to_vec();
        new_shape[rank - 2] = target_h;
        new_shape[rank - 1] = target_w;
        Self::from_f32_result(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), out)?,
            input_dtype,
        )
    }
}

/// Bilinear interpolation coordinate mapping matching PyTorch `F.interpolate`.
///
/// - `align_corners=true`: `src = dst * (in_size - 1) / (out_size - 1)`
///   Corner pixels of input and output are aligned exactly.
/// - `align_corners=false`: `src = (dst + 0.5) * in_size / out_size - 0.5`
///   Half-pixel-center mapping -- pixel centers are uniformly spaced.
///
/// Result is clamped to `[0, in_size - 1]`.
fn bilinear_coord(dst: usize, in_size: usize, out_size: usize, align_corners: bool) -> f64 {
    let src = if align_corners && out_size > 1 {
        dst as f64 * (in_size as f64 - 1.0) / (out_size as f64 - 1.0)
    } else if align_corners {
        // out_size == 1: map to center (0.0 for single output pixel).
        0.0
    } else {
        (dst as f64 + 0.5) * (in_size as f64) / (out_size as f64) - 0.5
    };
    src.clamp(0.0, (in_size - 1) as f64)
}

/// Half-pixel-center coordinate mapping for bilinear resize.
///
/// `src = (dst + 0.5) * scale - 0.5`, clamped to `[0, in_size - 1]`.
/// `scale = in_size / out_size` (precomputed by caller).
///
/// This is a free function (not a method) so it can be used by the Kani harness.
fn resize_bilinear_coord(dst: usize, scale: f64, in_size: usize) -> f64 {
    let src = (dst as f64 + 0.5) * scale - 0.5;
    src.clamp(0.0, (in_size - 1) as f64)
}

#[cfg(kani)]
mod kani_proofs {
    /// Proves the bilinear resize coordinate mapping produces finite values
    /// and stays within `[0, in_size - 1]` for all valid dimension combinations.
    ///
    /// Domain: dst in [0, 8191], in_size in [1, 8192], out_size in [1, 8192].
    /// Covers all practical image resize scenarios.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn resize_bilinear_coord_is_finite_and_bounded() {
        let dst: usize = kani::any();
        let in_size: usize = kani::any();
        let out_size: usize = kani::any();

        // Constrain to valid positive dimensions and dst < out_size.
        kani::assume(in_size >= 1 && in_size <= 8192);
        kani::assume(out_size >= 1 && out_size <= 8192);
        kani::assume(dst < out_size);

        let scale = in_size as f64 / out_size as f64;
        let result = super::resize_bilinear_coord(dst, scale, in_size);

        // Result must be finite.
        assert!(result.is_finite(), "coordinate must be finite");
        // Result must be in [0, in_size - 1].
        assert!(result >= 0.0, "coordinate must be >= 0");
        assert!(
            result <= (in_size - 1) as f64,
            "coordinate must be <= in_size - 1"
        );
    }
}
