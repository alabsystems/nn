// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU dispatch for dpdf image preprocessing operations.
//!
//! [`DpdfImagePreprocessMetal`] wraps a [`DpdfPreprocessConfig`] and dispatches
//! image preprocessing (resize, normalize, padding, HWC-to-CHW transpose)
//! through Metal GPU via `DynTensor` ops. Falls back to CPU when the Metal
//! backend is not available.
//!
//! # Pipeline
//!
//! ```text
//! Input [H, W, 3] or [3, H, W]
//!   -> gpu_hwc_to_chw (if HWC)
//!   -> gpu_resize_bilinear(target_h, target_w)
//!   -> gpu_letterbox_pad (if Letterbox padding)
//!   -> gpu_normalize(mean, std, scale_factor)
//!   -> Output [3, final_h, final_w]
//! ```
//!
//! Part of #3908.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, DType, Result, TensorError};
use nn_models::dpdf_image_preprocess::{DpdfPreprocessConfig, PaddingMode};

/// GPU-accelerated image preprocessing for dpdf document models.
///
/// Holds a [`DpdfPreprocessConfig`] and dispatches preprocessing operations
/// through Metal GPU when available. All operations use `DynTensor` ops which
/// dispatch to the registered Metal backend or fall back to CPU transparently.
///
/// # Example
///
/// ```rust,no_run
/// # use nn_metal::DpdfImagePreprocessMetal;
/// # use nn_models::dpdf_image_preprocess::DpdfPreprocessConfig;
/// let config = DpdfPreprocessConfig::for_doclayout_yolo();
/// let preprocessor = DpdfImagePreprocessMetal::new(config);
/// // let output = preprocessor.preprocess_image(&input_tensor).unwrap();
/// ```
pub struct DpdfImagePreprocessMetal {
    /// Model-specific preprocessing configuration.
    config: DpdfPreprocessConfig,
}

impl DpdfImagePreprocessMetal {
    /// Create a new GPU-accelerated preprocessor from the given config.
    #[must_use]
    pub fn new(config: DpdfPreprocessConfig) -> Self {
        Self { config }
    }

    /// Access the underlying preprocessing configuration.
    #[must_use]
    pub fn config(&self) -> &DpdfPreprocessConfig {
        &self.config
    }

    /// Per-channel normalization on GPU: `(pixel * scale - mean) / std`.
    ///
    /// Input must be CHW `[3, H, W]` or batched `[1, 3, H, W]`.
    /// Mean and std tensors are broadcast across spatial dimensions.
    ///
    /// # Errors
    ///
    /// Returns error on shape mismatch or GPU dispatch failure.
    pub fn gpu_normalize(
        &self,
        input: &DynTensor,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Result<DynTensor> {
        let is_batched = input.rank() == 4;

        // Scale pixel values.
        let scaled = input.mul_scalar(f64::from(self.config.scale_factor))?;

        // Build mean/std tensors shaped for broadcast.
        let device = input.device();
        let (mean_t, std_t) = if is_batched {
            let m = DynTensor::from_vec(mean.to_vec(), &[1, 3, 1, 1], &device)?;
            let s = DynTensor::from_vec(std.to_vec(), &[1, 3, 1, 1], &device)?;
            (m, s)
        } else {
            let m = DynTensor::from_vec(mean.to_vec(), &[3, 1, 1], &device)?;
            let s = DynTensor::from_vec(std.to_vec(), &[3, 1, 1], &device)?;
            (m, s)
        };

        let normed = scaled.broadcast_sub(&mean_t)?;
        let normed = normed.broadcast_div(&std_t)?;
        Ok(normed)
    }

    /// Bilinear resize on GPU to target spatial dimensions.
    ///
    /// Input must be CHW `[3, H, W]` or `[1, 3, H, W]`.
    /// Uses `DynTensor::resize_bilinear` which dispatches to the Metal
    /// backend when available.
    ///
    /// # Errors
    ///
    /// Returns error on invalid shape or GPU dispatch failure.
    pub fn gpu_resize_bilinear(
        &self,
        input: &DynTensor,
        target_h: u32,
        target_w: u32,
    ) -> Result<DynTensor> {
        input.resize_bilinear(target_h as usize, target_w as usize)
    }

    /// Letterbox padding on GPU: center the image in a larger canvas filled
    /// with `fill` value.
    ///
    /// Input must be CHW `[3, H, W]` or `[1, 3, H, W]`. The output has
    /// spatial dimensions `(target_h, target_w)` with the input centered
    /// and borders filled with `fill`.
    ///
    /// Implementation concatenates fill-value strips along H and W axes
    /// using `DynTensor::cat`, which dispatches to GPU when available.
    ///
    /// # Errors
    ///
    /// Returns error on shape mismatch or if the input is larger than the
    /// target dimensions.
    pub fn gpu_letterbox_pad(
        &self,
        input: &DynTensor,
        target_h: u32,
        target_w: u32,
        fill: f32,
    ) -> Result<DynTensor> {
        let dims = input.dims();
        let (c_idx, h_idx, w_idx) = match dims.len() {
            3 => (0, 1, 2),
            4 => (1, 2, 3),
            _ => {
                return Err(TensorError::InvalidShape(format!(
                    "gpu_letterbox_pad: expected rank 3 or 4, got {}",
                    dims.len()
                )));
            }
        };

        let in_h = dims[h_idx];
        let in_w = dims[w_idx];
        let th = target_h as usize;
        let tw = target_w as usize;

        if in_h > th || in_w > tw {
            return Err(TensorError::InvalidShape(format!(
                "gpu_letterbox_pad: input ({in_h}x{in_w}) larger than target ({th}x{tw})"
            )));
        }

        // If already at target size, return as-is.
        if in_h == th && in_w == tw {
            return Ok(input.clone());
        }

        let pad_h = th - in_h;
        let pad_w = tw - in_w;
        let top = pad_h / 2;
        let bottom = pad_h - top;
        let left = pad_w / 2;
        let right = pad_w - left;

        let device = input.device();
        let channels = dims[c_idx];

        // Step 1: Pad along H dimension via concat.
        let h_padded = if top > 0 || bottom > 0 {
            let mut parts: Vec<DynTensor> = Vec::with_capacity(3);
            if top > 0 {
                let shape: Vec<usize> = if dims.len() == 4 {
                    vec![dims[0], channels, top, in_w]
                } else {
                    vec![channels, top, in_w]
                };
                parts.push(DynTensor::full(&shape, f64::from(fill), DType::F32, &device)?);
            }
            parts.push(input.clone());
            if bottom > 0 {
                let shape: Vec<usize> = if dims.len() == 4 {
                    vec![dims[0], channels, bottom, in_w]
                } else {
                    vec![channels, bottom, in_w]
                };
                parts.push(DynTensor::full(&shape, f64::from(fill), DType::F32, &device)?);
            }
            let refs: Vec<&DynTensor> = parts.iter().collect();
            DynTensor::cat(&refs, h_idx)?
        } else {
            input.clone()
        };

        // Step 2: Pad along W dimension via concat.
        if left > 0 || right > 0 {
            let mut parts: Vec<DynTensor> = Vec::with_capacity(3);
            if left > 0 {
                let shape: Vec<usize> = if dims.len() == 4 {
                    vec![dims[0], channels, th, left]
                } else {
                    vec![channels, th, left]
                };
                parts.push(DynTensor::full(&shape, f64::from(fill), DType::F32, &device)?);
            }
            parts.push(h_padded);
            if right > 0 {
                let shape: Vec<usize> = if dims.len() == 4 {
                    vec![dims[0], channels, th, right]
                } else {
                    vec![channels, th, right]
                };
                parts.push(DynTensor::full(&shape, f64::from(fill), DType::F32, &device)?);
            }
            let refs: Vec<&DynTensor> = parts.iter().collect();
            DynTensor::cat(&refs, w_idx)
        } else {
            Ok(h_padded)
        }
    }

    /// Transpose HWC `[H, W, 3]` to CHW `[3, H, W]` on GPU.
    ///
    /// If the input is already CHW (first dim is 3 and last dim is not 3,
    /// or explicitly rank-4 BCHW), it is returned unchanged.
    ///
    /// # Errors
    ///
    /// Returns error on invalid shape or permutation failure.
    pub fn gpu_hwc_to_chw(input: &DynTensor) -> Result<DynTensor> {
        let dims = input.dims();
        match dims.len() {
            3 => {
                if dims[2] == 3 && dims[0] != 3 {
                    // HWC -> CHW
                    input.permute([2, 0, 1])
                } else if dims[0] == 3 {
                    // Already CHW
                    Ok(input.clone())
                } else {
                    Err(TensorError::InvalidShape(format!(
                        "gpu_hwc_to_chw: expected [H,W,3] or [3,H,W], got {dims:?}"
                    )))
                }
            }
            4 => {
                if dims[3] == 3 && dims[1] != 3 {
                    // BHWC -> BCHW
                    input.permute([0, 3, 1, 2])
                } else if dims[1] == 3 {
                    // Already BCHW
                    Ok(input.clone())
                } else {
                    Err(TensorError::InvalidShape(format!(
                        "gpu_hwc_to_chw: expected [B,H,W,3] or [B,3,H,W], got {dims:?}"
                    )))
                }
            }
            _ => Err(TensorError::InvalidShape(format!(
                "gpu_hwc_to_chw: expected rank 3 or 4, got {}",
                dims.len()
            ))),
        }
    }

    /// Full preprocessing pipeline using GPU ops.
    ///
    /// # Pipeline
    /// 1. Ensure tensor is on GPU (upload from CPU if needed).
    /// 2. Convert HWC to CHW if needed.
    /// 3. Compute target dimensions (respecting `maintain_aspect`).
    /// 4. Bilinear resize to target dimensions.
    /// 5. Apply letterbox padding if configured.
    /// 6. Per-channel normalize: `(pixel * scale - mean) / std`.
    ///
    /// Returns a CHW tensor `[3, final_h, final_w]`.
    ///
    /// # Errors
    ///
    /// Returns error on invalid input shape, zero dimensions, or GPU failure.
    pub fn preprocess_image(&self, input: &DynTensor) -> Result<DynTensor> {
        // Step 1: Ensure on GPU.
        let device = Device::metal();
        let img = if input.device().is_gpu() {
            input.clone()
        } else {
            input.to_device(&device)?
        };

        // Step 2: Convert to CHW.
        let img = Self::gpu_hwc_to_chw(&img)?;

        // Handle batched input: squeeze batch dim, will re-add if needed.
        let (img, was_batched) = if img.rank() == 4 {
            let dims = img.dims();
            if dims[0] != 1 {
                return Err(TensorError::InvalidShape(format!(
                    "dpdf preprocess: batch size must be 1, got {}",
                    dims[0]
                )));
            }
            (img.squeeze(0)?, true)
        } else {
            (img, false)
        };

        // Now img is [3, H, W].
        let img_dims = img.dims();
        if img_dims[0] != 3 {
            return Err(TensorError::InvalidShape(format!(
                "dpdf preprocess: expected 3 channels, got {}",
                img_dims[0]
            )));
        }
        let src_h = img_dims[1] as u32;
        let src_w = img_dims[2] as u32;

        // Step 3: Compute target dimensions.
        let (resize_h, resize_w) =
            nn_models::dpdf_image_preprocess::compute_resize_dims(
                src_h,
                src_w,
                self.config.target_height,
                self.config.target_width,
                self.config.maintain_aspect,
            );

        // Step 4: Bilinear resize.
        let img = self.gpu_resize_bilinear(&img, resize_h, resize_w)?;

        // Step 5: Apply padding mode.
        let img = match &self.config.padding_mode {
            PaddingMode::Letterbox { fill_value } => {
                let fill_scaled = fill_value * self.config.scale_factor;
                self.gpu_letterbox_pad(
                    &img,
                    self.config.target_height,
                    self.config.target_width,
                    fill_scaled,
                )?
            }
            PaddingMode::CenterCrop => {
                // Center-crop: already resized to fit, now crop center.
                let target_h = self.config.target_height as usize;
                let target_w = self.config.target_width as usize;
                let cur_dims = img.dims();
                let cur_h = cur_dims[1];
                let cur_w = cur_dims[2];

                let offset_y = cur_h.saturating_sub(target_h) / 2;
                let offset_x = cur_w.saturating_sub(target_w) / 2;

                // Narrow along H and W dimensions.
                let img = img.narrow(1, offset_y, target_h)?;
                img.narrow(2, offset_x, target_w)?
            }
            PaddingMode::None => img,
            _ => {
                return Err(TensorError::InvalidShape(
                    "dpdf preprocess: unsupported padding mode".to_string(),
                ));
            }
        };

        // Step 6: Per-channel normalize.
        let img = self.gpu_normalize(&img, self.config.mean, self.config.std)?;

        // Re-add batch dim if input was batched.
        if was_batched {
            img.unsqueeze(0)
        } else {
            Ok(img)
        }
    }
}

#[cfg(test)]
#[path = "dpdf_image_preprocess_metal_tests.rs"]
mod tests;
