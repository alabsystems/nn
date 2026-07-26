// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor-based image preprocessing pipeline for ViT inference.
//!
//! [`ImagePreprocessor`] takes a `DynTensor` of pixel values and applies:
//! 1. Bilinear resize to target `(H, W)` dimensions
//! 2. Rescale by `rescale_factor` (typically `1/255` for uint8 pixel range)
//! 3. Per-channel normalization: `(x - mean) / std`
//! 4. Ensure CHW layout `[C, H, W]` (or `[B, C, H, W]` for batched input)
//!
//! Unlike [`ImageProcessor`](super::ImageProcessor), which accepts raw `&[u8]`
//! bytes, `ImagePreprocessor` operates on already-loaded `DynTensor` values
//! and uses `DynTensor::resize_bilinear` for the resize step.
//!
//! # Factory presets
//!
//! | Method | Model family | Size | Mean | Std | Rescale |
//! |--------|-------------|------|------|-----|---------|
//! | [`siglip2`](ImagePreprocessor::siglip2) | SigLIP2, CLIP | 384 | `[0.5; 3]` | `[0.5; 3]` | 1/255 |
//! | [`vit_base`](ImagePreprocessor::vit_base) | ViT-B/16, DeiT | 224 | ImageNet | ImageNet | 1/255 |
//! | [`qwen_vl`](ImagePreprocessor::qwen_vl) | Qwen2.5-VL | 448 | ImageNet | ImageNet | 1/255 |

use crate::dyn_tensor::DynTensor;
use crate::error::{Result, TensorError};

/// ImageNet normalization constants (torchvision standard).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// SigLIP2 normalization constants (symmetric `[0.5]`).
const SIGLIP2_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SIGLIP2_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Tensor-based image preprocessing pipeline for vision model inference.
///
/// Applies bilinear resize, rescale, per-channel normalization, and CHW
/// layout enforcement to `DynTensor` inputs. Designed for ViT, SigLIP2,
/// Qwen2.5-VL, and other vision encoders that expect normalized `[B, C, H, W]`
/// float tensors.
///
/// # Example
/// ```no_run
/// # use nn_core::layers::vision::ImagePreprocessor;
/// # use nn_core::dyn_tensor::DynTensor;
/// # use nn_core::Device;
/// let preprocessor = ImagePreprocessor::vit_base();
/// // [3, 480, 640] float tensor with pixel values in [0, 255] -> resized to [3, 224, 224].
/// let image = DynTensor::full(&[3, 480, 640], 128.0, nn_core::DType::F32, &Device::Cpu).unwrap();
/// let normalized = preprocessor.preprocess(&image).unwrap();
/// assert_eq!(normalized.dims(), &[3, 224, 224]);
/// ```
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    /// Target spatial dimensions (H, W) for bilinear resize.
    target_height: usize,
    target_width: usize,
    /// Per-channel mean for normalization.
    mean: [f32; 3],
    /// Per-channel standard deviation for normalization.
    std: [f32; 3],
    /// Rescale factor applied before normalization (typically 1/255).
    rescale_factor: f32,
}

impl ImagePreprocessor {
    /// Create a new `ImagePreprocessor` with custom parameters.
    ///
    /// # Arguments
    /// - `height`, `width`: target spatial dimensions for bilinear resize.
    /// - `mean`, `std`: per-channel normalization constants.
    /// - `rescale_factor`: multiplied with pixel values before normalization (e.g., `1.0/255.0`).
    ///
    /// # Errors
    /// Returns an error if any std value is zero (would cause division by zero).
    pub fn new(
        height: usize,
        width: usize,
        mean: [f32; 3],
        std: [f32; 3],
        rescale_factor: f32,
    ) -> Result<Self> {
        for (i, &s) in std.iter().enumerate() {
            if s == 0.0 {
                return Err(TensorError::InvalidShape(format!(
                    "ImagePreprocessor: std[{i}] must be non-zero"
                )));
            }
        }
        Ok(Self {
            target_height: height,
            target_width: width,
            mean,
            std,
            rescale_factor,
        })
    }

    /// SigLIP2 preset: 384x384, mean/std `[0.5; 3]`, rescale 1/255.
    ///
    /// Maps uint8 `[0, 255]` to `[-1, 1]` after rescale + normalization.
    /// Used by SigLIP2, some CLIP variants.
    pub fn siglip2() -> Self {
        // SAFETY: std values are all 0.5 (non-zero), so `new` cannot fail.
        Self::new(384, 384, SIGLIP2_MEAN, SIGLIP2_STD, 1.0 / 255.0)
            .expect("siglip2 preset has valid std")
    }

    /// ViT-Base/16 preset: 224x224, ImageNet mean/std, rescale 1/255.
    ///
    /// Standard for ViT-B/16, ViT-L/16, DeiT, and most torchvision-pretrained
    /// vision transformers.
    pub fn vit_base() -> Self {
        Self::new(224, 224, IMAGENET_MEAN, IMAGENET_STD, 1.0 / 255.0)
            .expect("vit_base preset has valid std")
    }

    /// Qwen2.5-VL preset: 448x448, ImageNet mean/std, rescale 1/255.
    ///
    /// Used by Qwen2-VL and Qwen2.5-VL vision-language models.
    pub fn qwen_vl() -> Self {
        Self::new(448, 448, IMAGENET_MEAN, IMAGENET_STD, 1.0 / 255.0)
            .expect("qwen_vl preset has valid std")
    }

    /// Target height for the preprocessor.
    #[must_use]
    pub fn target_height(&self) -> usize {
        self.target_height
    }

    /// Target width for the preprocessor.
    #[must_use]
    pub fn target_width(&self) -> usize {
        self.target_width
    }

    /// Per-channel mean used for normalization.
    #[must_use]
    pub fn mean(&self) -> &[f32; 3] {
        &self.mean
    }

    /// Per-channel std used for normalization.
    #[must_use]
    pub fn std_dev(&self) -> &[f32; 3] {
        &self.std
    }

    /// Rescale factor applied before normalization.
    #[must_use]
    pub fn rescale_factor(&self) -> f32 {
        self.rescale_factor
    }

    /// Preprocess a tensor for vision model inference.
    ///
    /// Accepts tensors in either layout:
    /// - **CHW**: `[C, H, W]` or `[B, C, H, W]` (channel dimension must be 3)
    /// - **HWC**: `[H, W, C]` or `[B, H, W, C]` (last dimension must be 3)
    ///
    /// HWC inputs are automatically transposed to CHW before processing.
    ///
    /// # Pipeline
    /// 1. Ensure CHW layout (transpose HWC to CHW if needed).
    /// 2. Resize: bilinear interpolation to `(target_height, target_width)`.
    /// 3. Rescale: `x = x * rescale_factor`
    /// 4. Normalize per-channel: `x[c] = (x[c] - mean[c]) / std[c]`
    ///
    /// # Errors
    /// - `InvalidShape` if rank is not 3 or 4, or channel dimension is not 3.
    pub fn preprocess(&self, image: &DynTensor) -> Result<DynTensor> {
        let dims = image.dims();

        // Determine layout and extract channel count.
        let (x, channels, is_batched) = match dims.len() {
            3 => {
                // Could be [C, H, W] or [H, W, C].
                if dims[2] == 3 && dims[0] != 3 {
                    // HWC: [H, W, 3] -> transpose to [3, H, W]
                    let transposed = image.permute([2, 0, 1])?;
                    (transposed.clone(), transposed.dims()[0], false)
                } else {
                    // CHW: [C, H, W]
                    (image.clone(), dims[0], false)
                }
            }
            4 => {
                // Could be [B, C, H, W] or [B, H, W, C].
                if dims[3] == 3 && dims[1] != 3 {
                    // BHWC: [B, H, W, 3] -> transpose to [B, 3, H, W]
                    let transposed = image.permute([0, 3, 1, 2])?;
                    (transposed.clone(), transposed.dims()[1], true)
                } else {
                    // BCHW: [B, C, H, W]
                    (image.clone(), dims[1], true)
                }
            }
            _ => {
                return Err(TensorError::InvalidShape(format!(
                    "ImagePreprocessor: expected rank 3 or 4, got rank {}",
                    dims.len(),
                )));
            }
        };
        if channels != 3 {
            return Err(TensorError::InvalidShape(format!(
                "ImagePreprocessor: expected 3 channels, got {channels}"
            )));
        }

        // Step 1: Bilinear resize to target dimensions.
        let x_dims = x.dims();
        let (in_h, in_w) = if is_batched {
            (x_dims[2], x_dims[3])
        } else {
            (x_dims[1], x_dims[2])
        };
        let x = if in_h != self.target_height || in_w != self.target_width {
            x.resize_bilinear(self.target_height, self.target_width)?
        } else {
            x
        };

        // Step 2: Rescale pixel values.
        let x = x.mul_scalar(f64::from(self.rescale_factor))?;

        // Step 3: Per-channel normalization.
        // Build mean and std tensors with shape [1, 3, 1, 1] (batched) or [3, 1, 1] (unbatched)
        // for correct broadcasting against [B, C, H, W] or [C, H, W].
        let device = image.device();
        let (mean_t, std_t) = if is_batched {
            let mean_t = DynTensor::from_vec(self.mean.to_vec(), &[1, 3, 1, 1], &device)?;
            let std_t = DynTensor::from_vec(self.std.to_vec(), &[1, 3, 1, 1], &device)?;
            (mean_t, std_t)
        } else {
            let mean_t = DynTensor::from_vec(self.mean.to_vec(), &[3, 1, 1], &device)?;
            let std_t = DynTensor::from_vec(self.std.to_vec(), &[3, 1, 1], &device)?;
            (mean_t, std_t)
        };

        let x = x.broadcast_sub(&mean_t)?;
        let x = x.broadcast_div(&std_t)?;

        Ok(x)
    }
}

#[cfg(test)]
#[path = "image_preprocess_tests.rs"]
mod tests;
