// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Image preprocessing pipeline for dpdf document models.
//!
//! [`DpdfPreprocessConfig`] describes model-specific image transformations:
//! resize, letterbox/crop padding, scale, and per-channel normalization.
//! Each dpdf model has a factory preset that encodes its exact input contract.
//!
//! # Per-model presets
//!
//! | Preset | Resolution | Padding | Normalization |
//! |--------|-----------|---------|---------------|
//! | [`for_granite_docling`] | 384x384 | None | `[0.5; 3]` mean/std |
//! | [`for_doclayout_yolo`] | 1024x1024 | Letterbox(114) | scale 1/255 |
//! | [`for_paddle_ocr_detect`] | 960 max side | None | ImageNet |
//! | [`for_paddle_ocr_recognize`] | 48x320 | None | ImageNet |
//! | [`for_table_transformer`] | 800 shortest | None | ImageNet |
//! | [`for_qwen3_vl`] | dynamic | None | `[0.5; 3]` mean/std |
//! | [`for_glm_ocr`] | 1120x1120 max | None | `[0.5; 3]` mean/std |
//! | [`for_rt_detr`] | 640x640 | None | ImageNet |
//!
//! [`for_granite_docling`]: DpdfPreprocessConfig::for_granite_docling
//! [`for_doclayout_yolo`]: DpdfPreprocessConfig::for_doclayout_yolo
//! [`for_paddle_ocr_detect`]: DpdfPreprocessConfig::for_paddle_ocr_detect
//! [`for_paddle_ocr_recognize`]: DpdfPreprocessConfig::for_paddle_ocr_recognize
//! [`for_table_transformer`]: DpdfPreprocessConfig::for_table_transformer
//! [`for_qwen3_vl`]: DpdfPreprocessConfig::for_qwen3_vl
//! [`for_glm_ocr`]: DpdfPreprocessConfig::for_glm_ocr
//! [`for_rt_detr`]: DpdfPreprocessConfig::for_rt_detr

/// ImageNet normalization constants (torchvision standard).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Symmetric normalization constants (SigLIP2 / Granite / GLM style).
const SYMMETRIC_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SYMMETRIC_STD: [f32; 3] = [0.5, 0.5, 0.5];

// ---------------------------------------------------------------------------
// Padding mode
// ---------------------------------------------------------------------------

/// Padding strategy applied after resizing to preserve aspect ratio.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PaddingMode {
    /// No padding: resize directly to target dimensions, distorting aspect
    /// ratio if needed.
    None,

    /// Letterbox: resize to fit within target, fill remaining area with a
    /// constant value.
    Letterbox {
        /// Pixel fill value in `[0, 255]` range (applied before scaling).
        fill_value: f32,
    },

    /// Center-crop: resize shortest side to target, crop the center.
    CenterCrop,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Image preprocessing configuration for a dpdf document model.
///
/// Encodes target resolution, normalization parameters, padding strategy,
/// and aspect-ratio handling. Use per-model factory constructors (e.g.
/// [`for_granite_docling`](Self::for_granite_docling)) for standard presets.
#[derive(Debug, Clone, PartialEq)]
pub struct DpdfPreprocessConfig {
    /// Target image height in pixels.
    pub target_height: u32,
    /// Target image width in pixels.
    pub target_width: u32,
    /// Per-channel normalization mean (R, G, B).
    pub mean: [f32; 3],
    /// Per-channel normalization std (R, G, B).
    pub std: [f32; 3],
    /// Padding strategy after aspect-preserving resize.
    pub padding_mode: PaddingMode,
    /// Multiplicative scale factor applied to raw pixels before
    /// normalization (default `1.0 / 255.0`).
    pub scale_factor: f32,
    /// Whether to preserve the aspect ratio during resize. When `true`,
    /// the image is resized so that the larger (or smaller, depending on
    /// model) side matches the target, and the remaining area is handled
    /// by `padding_mode`.
    pub maintain_aspect: bool,
    /// Minimum total pixel count for dynamic-resolution models (Qwen3-VL).
    /// Zero means unused.
    pub min_pixels: u32,
    /// Maximum total pixel count for dynamic-resolution models (Qwen3-VL).
    /// Zero means unused.
    pub max_pixels: u32,
    /// Patch size for vision-transformer patch embedding (Qwen3-VL).
    /// Zero means unused.
    pub patch_size: u32,
}

impl DpdfPreprocessConfig {
    // -- Per-model presets ---------------------------------------------------

    /// Granite-Docling-258M: 384x384, symmetric `[0.5; 3]` normalization.
    ///
    /// SigLIP2 vision encoder with 384-pixel square input. Symmetric mean/std
    /// maps `[0, 1]` to `[-1, 1]`.
    #[must_use]
    pub fn for_granite_docling() -> Self {
        Self {
            target_height: 384,
            target_width: 384,
            mean: SYMMETRIC_MEAN,
            std: SYMMETRIC_STD,
            padding_mode: PaddingMode::None,
            scale_factor: 1.0 / 255.0,
            maintain_aspect: false,
            min_pixels: 0,
            max_pixels: 0,
            patch_size: 0,
        }
    }

    /// DocLayout-YOLO: 1024x1024 with letterbox padding (fill 114/255).
    ///
    /// Standard YOLO letterbox: resize to fit within 1024x1024 while
    /// preserving aspect ratio, fill borders with gray (114).
    #[must_use]
    pub fn for_doclayout_yolo() -> Self {
        Self {
            target_height: 1024,
            target_width: 1024,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            padding_mode: PaddingMode::Letterbox { fill_value: 114.0 },
            scale_factor: 1.0 / 255.0,
            maintain_aspect: true,
            min_pixels: 0,
            max_pixels: 0,
            patch_size: 0,
        }
    }

    /// PaddleOCR text detection: 960 max side, ImageNet normalization.
    ///
    /// Resizes the longer side to 960 while preserving aspect ratio.
    /// Uses standard ImageNet mean/std normalization.
    #[must_use]
    pub fn for_paddle_ocr_detect() -> Self {
        Self {
            target_height: 960,
            target_width: 960,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            padding_mode: PaddingMode::None,
            scale_factor: 1.0 / 255.0,
            maintain_aspect: true,
            min_pixels: 0,
            max_pixels: 0,
            patch_size: 0,
        }
    }

    /// PaddleOCR text recognition: 48x320, ImageNet normalization.
    ///
    /// Fixed height 48, max width 320. Recognizes text in a single-line crop.
    #[must_use]
    pub fn for_paddle_ocr_recognize() -> Self {
        Self {
            target_height: 48,
            target_width: 320,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            padding_mode: PaddingMode::None,
            scale_factor: 1.0 / 255.0,
            maintain_aspect: true,
            min_pixels: 0,
            max_pixels: 0,
            patch_size: 0,
        }
    }

    /// Table Transformer (DETR): 800 shortest side, ImageNet normalization.
    ///
    /// Resizes the shortest side to 800, capping the longest side at 1333.
    /// Standard torchvision DETR preprocessing.
    #[must_use]
    pub fn for_table_transformer() -> Self {
        Self {
            target_height: 800,
            target_width: 800,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            padding_mode: PaddingMode::None,
            scale_factor: 1.0 / 255.0,
            maintain_aspect: true,
            min_pixels: 0,
            max_pixels: 0,
            patch_size: 0,
        }
    }

    /// Qwen3-VL: dynamic resolution with patch-size constraints.
    ///
    /// Resolution is dynamically chosen from `min_pixels..max_pixels`
    /// such that both height and width are multiples of `patch_size * 2`.
    /// Uses symmetric `[0.5; 3]` normalization.
    #[must_use]
    pub fn for_qwen3_vl() -> Self {
        Self {
            target_height: 0,
            target_width: 0,
            mean: SYMMETRIC_MEAN,
            std: SYMMETRIC_STD,
            padding_mode: PaddingMode::None,
            scale_factor: 1.0 / 255.0,
            maintain_aspect: true,
            min_pixels: 256 * 28 * 28,
            max_pixels: 1280 * 28 * 28,
            patch_size: 28,
        }
    }

    /// RT-DETRv2 (Heron): 640x640, ImageNet normalization.
    ///
    /// Standard RT-DETR preprocessing: resize to 640x640 without maintaining
    /// aspect ratio (model expects exact square input). Uses ImageNet
    /// normalization as with other DETR-family models.
    #[must_use]
    pub fn for_rt_detr() -> Self {
        Self {
            target_height: 640,
            target_width: 640,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            padding_mode: PaddingMode::None,
            scale_factor: 1.0 / 255.0,
            maintain_aspect: false,
            min_pixels: 0,
            max_pixels: 0,
            patch_size: 0,
        }
    }

    /// GLM-OCR 0.9B: 1120x1120 max, symmetric `[0.5; 3]` normalization.
    ///
    /// Resize longest side to 1120 while preserving aspect ratio.
    #[must_use]
    pub fn for_glm_ocr() -> Self {
        Self {
            target_height: 1120,
            target_width: 1120,
            mean: SYMMETRIC_MEAN,
            std: SYMMETRIC_STD,
            padding_mode: PaddingMode::None,
            scale_factor: 1.0 / 255.0,
            maintain_aspect: true,
            min_pixels: 0,
            max_pixels: 0,
            patch_size: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocessing functions
// ---------------------------------------------------------------------------

/// Result of preprocessing: normalized pixel data in CHW layout with
/// the final spatial dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessResult {
    /// Normalized pixel data in CHW order: `[C, H, W]` where C=3.
    pub data: Vec<f32>,
    /// Height of the preprocessed image.
    pub height: u32,
    /// Width of the preprocessed image.
    pub width: u32,
    /// Channels (always 3).
    pub channels: u32,
}

/// Preprocess raw pixel data (f32, HWC layout) according to `config`.
///
/// # Pipeline
/// 1. Compute target dimensions (respecting `maintain_aspect`).
/// 2. Apply padding mode (letterbox / center-crop / none).
/// 3. Scale by `config.scale_factor`.
/// 4. Per-channel normalize: `(pixel * scale - mean) / std`.
/// 5. Convert HWC to CHW layout.
///
/// # Arguments
/// - `pixels`: raw pixel data in HWC (row-major) layout, values in `[0, 255]`.
/// - `src_height`, `src_width`: source image dimensions.
/// - `config`: preprocessing configuration.
///
/// # Returns
/// A [`PreprocessResult`] containing normalized CHW data and final dimensions.
///
/// # Errors
/// Returns `None` if input dimensions are zero or the pixel buffer is too short.
#[must_use]
pub fn preprocess(
    pixels: &[f32],
    src_height: u32,
    src_width: u32,
    config: &DpdfPreprocessConfig,
) -> Option<PreprocessResult> {
    if src_height == 0 || src_width == 0 {
        return None;
    }
    let expected_len = (src_height as usize) * (src_width as usize) * 3;
    if pixels.len() < expected_len {
        return None;
    }

    // Step 1: Compute resize dimensions.
    let (resize_h, resize_w) = compute_resize_dims(
        src_height,
        src_width,
        config.target_height,
        config.target_width,
        config.maintain_aspect,
    );

    // Step 2: Apply padding mode.
    let (final_h, final_w, padded) = match &config.padding_mode {
        PaddingMode::Letterbox { fill_value } => {
            let params = compute_letterbox_params(
                resize_h,
                resize_w,
                config.target_height,
                config.target_width,
            );
            let filled = apply_letterbox(
                pixels,
                src_height,
                src_width,
                resize_h,
                resize_w,
                &params,
                *fill_value * config.scale_factor,
            );
            (config.target_height, config.target_width, filled)
        }
        PaddingMode::CenterCrop => {
            let cropped = apply_center_crop(
                pixels,
                src_height,
                src_width,
                config.target_height,
                config.target_width,
            );
            (config.target_height, config.target_width, cropped)
        }
        PaddingMode::None => {
            // Simple bilinear resize (conceptual — store source pixels
            // scaled to resize dims). For a real implementation this would
            // invoke bilinear interpolation; here we pass through scaled
            // pixels for the normalized pipeline.
            let resized = simple_resize_hwc(pixels, src_height, src_width, resize_h, resize_w);
            (resize_h, resize_w, resized)
        }
    };

    // Step 3+4+5: Scale, normalize per-channel, convert HWC → CHW.
    let fh = final_h as usize;
    let fw = final_w as usize;
    let pixels_per_channel = fh * fw;
    let mut chw = vec![0.0f32; 3 * pixels_per_channel];

    for c in 0..3 {
        let inv_std = 1.0 / config.std[c];
        let mean_c = config.mean[c];
        let sf = config.scale_factor;
        for i in 0..pixels_per_channel {
            let val = padded[i * 3 + c];
            chw[c * pixels_per_channel + i] = (val * sf - mean_c) * inv_std;
        }
    }

    Some(PreprocessResult {
        data: chw,
        height: final_h,
        width: final_w,
        channels: 3,
    })
}

/// Compute resize dimensions respecting aspect ratio constraints.
///
/// When `maintain_aspect` is true, the image is scaled so its longer side
/// does not exceed the corresponding target dimension. When false, the
/// target dimensions are returned directly.
#[must_use]
pub fn compute_resize_dims(
    src_height: u32,
    src_width: u32,
    target_height: u32,
    target_width: u32,
    maintain_aspect: bool,
) -> (u32, u32) {
    if !maintain_aspect || target_height == 0 || target_width == 0 {
        return (target_height.max(1), target_width.max(1));
    }

    let scale_h = f64::from(target_height) / f64::from(src_height);
    let scale_w = f64::from(target_width) / f64::from(src_width);
    let scale = scale_h.min(scale_w);

    let new_h = (f64::from(src_height) * scale).round() as u32;
    let new_w = (f64::from(src_width) * scale).round() as u32;

    (new_h.max(1), new_w.max(1))
}

/// Compute letterbox padding parameters: offsets and padded dimensions.
///
/// Returns `(top_pad, left_pad, bottom_pad, right_pad)` such that the
/// resized image of `(resize_h, resize_w)` is centered within
/// `(target_h, target_w)`.
#[must_use]
pub fn compute_letterbox_params(
    resize_h: u32,
    resize_w: u32,
    target_h: u32,
    target_w: u32,
) -> LetterboxParams {
    let pad_h = target_h.saturating_sub(resize_h);
    let pad_w = target_w.saturating_sub(resize_w);
    let top = pad_h / 2;
    let left = pad_w / 2;
    LetterboxParams {
        top,
        left,
        bottom: pad_h - top,
        right: pad_w - left,
    }
}

/// Letterbox padding parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LetterboxParams {
    /// Padding rows at the top.
    pub top: u32,
    /// Padding columns on the left.
    pub left: u32,
    /// Padding rows at the bottom.
    pub bottom: u32,
    /// Padding columns on the right.
    pub right: u32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Nearest-neighbor resize for HWC f32 data.
///
/// This is a simple (non-interpolating) resize used for the preprocessing
/// pipeline. Production code would use bilinear interpolation via
/// `ImageProcessor` in nn-core.
fn simple_resize_hwc(data: &[f32], src_h: u32, src_w: u32, dst_h: u32, dst_w: u32) -> Vec<f32> {
    let dh = dst_h as usize;
    let dw = dst_w as usize;
    let sh = src_h as usize;
    let sw = src_w as usize;
    let mut out = vec![0.0f32; dh * dw * 3];

    for y in 0..dh {
        let src_y = (y * sh / dh).min(sh.saturating_sub(1));
        for x in 0..dw {
            let src_x = (x * sw / dw).min(sw.saturating_sub(1));
            for c in 0..3 {
                out[(y * dw + x) * 3 + c] = data[(src_y * sw + src_x) * 3 + c];
            }
        }
    }
    out
}

/// Apply letterbox padding: resize source into center of target-sized canvas.
fn apply_letterbox(
    src_pixels: &[f32],
    src_h: u32,
    src_w: u32,
    resize_h: u32,
    resize_w: u32,
    params: &LetterboxParams,
    fill_value: f32,
) -> Vec<f32> {
    let target_h = (resize_h + params.top + params.bottom) as usize;
    let target_w = (resize_w + params.left + params.right) as usize;
    let mut canvas = vec![fill_value; target_h * target_w * 3];

    // Resize source to (resize_h, resize_w).
    let resized = simple_resize_hwc(src_pixels, src_h, src_w, resize_h, resize_w);

    // Copy resized image into the center of the canvas.
    let top = params.top as usize;
    let left = params.left as usize;
    let rw = resize_w as usize;
    let rh = resize_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            for c in 0..3 {
                canvas[((y + top) * target_w + (x + left)) * 3 + c] = resized[(y * rw + x) * 3 + c];
            }
        }
    }
    canvas
}

/// Apply center-crop: resize so shortest side matches target, then crop center.
fn apply_center_crop(
    src_pixels: &[f32],
    src_h: u32,
    src_w: u32,
    target_h: u32,
    target_w: u32,
) -> Vec<f32> {
    // Scale so that the shortest side matches the target.
    let scale_h = f64::from(target_h) / f64::from(src_h);
    let scale_w = f64::from(target_w) / f64::from(src_w);
    let scale = scale_h.max(scale_w);
    let scaled_h = (f64::from(src_h) * scale).round() as u32;
    let scaled_w = (f64::from(src_w) * scale).round() as u32;

    let resized = simple_resize_hwc(src_pixels, src_h, src_w, scaled_h, scaled_w);

    // Crop center.
    let offset_y = (scaled_h.saturating_sub(target_h) / 2) as usize;
    let offset_x = (scaled_w.saturating_sub(target_w) / 2) as usize;
    let th = target_h as usize;
    let tw = target_w as usize;
    let sw = scaled_w as usize;

    let mut cropped = vec![0.0f32; th * tw * 3];
    for y in 0..th {
        for x in 0..tw {
            for c in 0..3 {
                cropped[(y * tw + x) * 3 + c] =
                    resized[((y + offset_y) * sw + (x + offset_x)) * 3 + c];
            }
        }
    }
    cropped
}

#[cfg(test)]
#[path = "dpdf_image_preprocess_tests.rs"]
mod tests;
