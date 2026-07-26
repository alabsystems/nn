// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Image preprocessing pipeline for vision models.
//!
//! [`ImageProcessor`] converts raw pixel data (u8 HWC) into the `[B, 3, H, W]`
//! normalized float tensors that vision models expect. Pure Rust, no external
//! image library dependencies.
//!
//! Supports:
//! - Bilinear resize to arbitrary target dimensions
//! - Per-channel mean/std normalization
//! - HWC -> CHW layout conversion
//! - Preset configurations for ImageNet, SigLIP2, and raw `[0,1]` normalization

use crate::dyn_tensor::DynTensor;
use crate::error::{Result, TensorError};
use crate::Device;

/// ImageNet normalization constants (torchvision standard).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// SigLIP2 normalization constants (symmetric `[0.5]`).
const SIGLIP2_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SIGLIP2_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Image preprocessing pipeline: resize, normalize, HWC->CHW for vision models.
///
/// Converts raw RGB pixel data into the `[1, 3, H, W]` float tensors that
/// vision encoders (ViT, SigLIP2, ResNet, etc.) consume.
///
/// # Example
/// ```no_run
/// # use nn_core::layers::vision::ImageProcessor;
/// # use nn_core::Device;
/// let processor = ImageProcessor::imagenet(224);
/// let rgb_pixels: Vec<u8> = vec![128; 640 * 480 * 3]; // 640x480 RGB
/// let tensor = processor.process(&rgb_pixels, 480, 640, 3, &Device::Cpu).unwrap();
/// assert_eq!(tensor.dims(), &[1, 3, 224, 224]);
/// ```
#[derive(Debug, Clone)]
pub struct ImageProcessor {
    target_height: usize,
    target_width: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl ImageProcessor {
    /// Create a new `ImageProcessor` with custom target size and normalization.
    ///
    /// # Arguments
    /// - `height` / `width`: target spatial dimensions (must be non-zero).
    /// - `mean` / `std`: per-channel normalization: `(pixel - mean) / std`.
    pub fn new(height: usize, width: usize, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            target_height: height,
            target_width: width,
            mean,
            std,
        }
    }

    /// ImageNet preset: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
    ///
    /// Standard for ResNet, ViT, DeiT, and most torchvision-pretrained models.
    pub fn imagenet(size: usize) -> Self {
        Self::new(size, size, IMAGENET_MEAN, IMAGENET_STD)
    }

    /// SigLIP2 preset: `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`.
    ///
    /// Maps `[0, 1]` float range to `[-1, 1]`. Used by SigLIP2 and some CLIP variants.
    pub fn siglip2(size: usize) -> Self {
        Self::new(size, size, SIGLIP2_MEAN, SIGLIP2_STD)
    }

    /// Normalize-only preset: `mean=[0, 0, 0]`, `std=[1, 1, 1]`.
    ///
    /// Divides by 255 to get `[0, 1]` range, no further normalization.
    pub fn normalize_only(height: usize, width: usize) -> Self {
        Self::new(height, width, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }

    /// Target height for the output tensor.
    pub fn target_height(&self) -> usize {
        self.target_height
    }

    /// Target width for the output tensor.
    pub fn target_width(&self) -> usize {
        self.target_width
    }

    /// Per-channel mean used for normalization.
    pub fn mean(&self) -> &[f32; 3] {
        &self.mean
    }

    /// Per-channel std used for normalization.
    pub fn std_dev(&self) -> &[f32; 3] {
        &self.std
    }

    /// Process raw RGB pixels into a model-ready tensor.
    ///
    /// # Arguments
    /// - `pixels`: raw pixel data in HWC (row-major) layout.
    /// - `height`, `width`: source image dimensions.
    /// - `channels`: number of channels (must be 3 for RGB).
    /// - `device`: target device for the output tensor.
    ///
    /// # Pipeline
    /// 1. Validate input dimensions.
    /// 2. Convert u8 `[H, W, 3]` to f32, divide by 255.
    /// 3. Bilinear resize to `[target_H, target_W, 3]`.
    /// 4. Per-channel normalize: `(pixel - mean) / std`.
    /// 5. Transpose HWC -> CHW: `[H, W, 3]` -> `[3, H, W]`.
    /// 6. Add batch dimension: `[1, 3, H, W]`.
    ///
    /// # Errors
    /// Returns `TensorError::InvalidShape` if dimensions are zero, channels != 3,
    /// or pixel buffer is too short.
    pub fn process(
        &self,
        pixels: &[u8],
        height: usize,
        width: usize,
        channels: usize,
        device: &Device,
    ) -> Result<DynTensor> {
        // Validate inputs.
        if height == 0 || width == 0 {
            return Err(TensorError::InvalidShape(
                "image height and width must be non-zero".into(),
            ));
        }
        if channels != 3 {
            return Err(TensorError::InvalidShape(format!(
                "expected 3 channels (RGB), got {channels}"
            )));
        }
        let expected_len = height
            .checked_mul(width)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| TensorError::InvalidShape("image dimensions overflow usize".into()))?;
        if pixels.len() < expected_len {
            return Err(TensorError::InvalidShape(format!(
                "pixel buffer length {} too short for {height}x{width}x3 = {expected_len}",
                pixels.len(),
            )));
        }

        // Step 1: Convert u8 -> f32 [0.0, 1.0] in HWC layout.
        let float_hwc: Vec<f32> = pixels[..expected_len]
            .iter()
            .map(|&b| f32::from(b) / 255.0)
            .collect();

        // Step 2: Bilinear resize [H, W, 3] -> [target_H, target_W, 3].
        let resized = if height == self.target_height && width == self.target_width {
            float_hwc
        } else {
            bilinear_resize_f32(
                &float_hwc,
                width,
                height,
                self.target_width,
                self.target_height,
                3,
            )
        };

        // Step 3+4: Normalize per-channel and transpose HWC -> CHW.
        let th = self.target_height;
        let tw = self.target_width;
        let pixels_per_channel = th * tw;
        let mut chw = vec![0.0f32; 3 * pixels_per_channel];
        for c in 0..3 {
            let inv_std = 1.0 / self.std[c];
            let mean_c = self.mean[c];
            for i in 0..pixels_per_channel {
                let val = resized[i * 3 + c];
                chw[c * pixels_per_channel + i] = (val - mean_c) * inv_std;
            }
        }

        // Step 5: Create tensor [1, 3, target_H, target_W].
        DynTensor::from_vec(chw, &[1, 3, th, tw], device)
    }

    /// Process a `DynTensor` that is already f32 in HWC layout.
    ///
    /// Expects input shape `[H, W, 3]` or `[B, H, W, 3]`. Applies resize,
    /// normalize, and HWC->CHW conversion. Returns `[B, 3, target_H, target_W]`.
    ///
    /// # Errors
    /// Returns an error if the input rank is not 3 or 4, or the channel
    /// dimension is not 3.
    pub fn process_tensor(&self, input: &DynTensor, device: &Device) -> Result<DynTensor> {
        let dims = input.dims();
        let (batch, height, width, channels) = match dims.len() {
            3 => (1, dims[0], dims[1], dims[2]),
            4 => (dims[0], dims[1], dims[2], dims[3]),
            _ => {
                return Err(TensorError::InvalidShape(format!(
                    "expected [H, W, 3] or [B, H, W, 3], got rank {}",
                    dims.len(),
                )));
            }
        };
        if channels != 3 {
            return Err(TensorError::InvalidShape(format!(
                "expected 3 channels, got {channels}"
            )));
        }
        if height == 0 || width == 0 {
            return Err(TensorError::InvalidShape(
                "image height and width must be non-zero".into(),
            ));
        }

        let flat = input.to_flat_vec::<f32>()?;
        let th = self.target_height;
        let tw = self.target_width;
        let src_pixels = height * width * 3;
        let dst_pixels_per_channel = th * tw;
        let mut all_chw = Vec::with_capacity(batch * 3 * dst_pixels_per_channel);

        for b in 0..batch {
            let src = &flat[b * src_pixels..(b + 1) * src_pixels];

            // Resize.
            let resized = if height == th && width == tw {
                src.to_vec()
            } else {
                bilinear_resize_f32(src, width, height, tw, th, 3)
            };

            // Normalize + HWC -> CHW.
            for c in 0..3 {
                let inv_std = 1.0 / self.std[c];
                let mean_c = self.mean[c];
                for i in 0..dst_pixels_per_channel {
                    let val = resized[i * 3 + c];
                    all_chw.push((val - mean_c) * inv_std);
                }
            }
        }

        DynTensor::from_vec(all_chw, &[batch, 3, th, tw], device)
    }
}

/// Bilinear resize for an f32 HWC image.
///
/// Center-aligned pixel mapping (matches OpenCV `INTER_LINEAR` and PyTorch
/// `F.interpolate(align_corners=False)` conventions).
///
/// # Arguments
/// - `data`: source pixels, flat `[src_h, src_w, channels]` in row-major HWC order.
/// - `src_w`, `src_h`: source width and height.
/// - `dst_w`, `dst_h`: destination width and height (must be non-zero).
/// - `channels`: number of channels per pixel.
fn bilinear_resize_f32(
    data: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    channels: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; dst_w * dst_h * channels];
    let scale_x = src_w as f64 / dst_w as f64;
    let scale_y = src_h as f64 / dst_h as f64;

    for y in 0..dst_h {
        for x in 0..dst_w {
            // Center-aligned mapping: dst center -> src center.
            let src_x = (x as f64 + 0.5) * scale_x - 0.5;
            let src_y = (y as f64 + 0.5) * scale_y - 0.5;

            let x0 = (src_x.floor() as isize).max(0) as usize;
            let y0 = (src_y.floor() as isize).max(0) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);

            let fx = (src_x - x0 as f64).clamp(0.0, 1.0);
            let fy = (src_y - y0 as f64).clamp(0.0, 1.0);

            for c in 0..channels {
                let p00 = f64::from(data[(y0 * src_w + x0) * channels + c]);
                let p10 = f64::from(data[(y0 * src_w + x1) * channels + c]);
                let p01 = f64::from(data[(y1 * src_w + x0) * channels + c]);
                let p11 = f64::from(data[(y1 * src_w + x1) * channels + c]);

                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;

                out[(y * dst_w + x) * channels + c] = val as f32;
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "image_processor_tests.rs"]
mod tests;
