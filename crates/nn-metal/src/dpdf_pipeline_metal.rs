// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated dpdf document inference pipeline.
//!
//! Wraps [`DpdfPipeline`] and [`DocLayoutYolo`] with Metal GPU dispatch for
//! model forward passes. Image preprocessing (resize, normalize, HWC->CHW)
//! runs on GPU via standard DynTensor ops. NMS and reading-order computation
//! remain on CPU (inherently sequential, no GPU benefit).
//!
//! # Architecture
//!
//! ```text
//! Image -> preprocess_image (GPU, gpu_scope) -> DocLayoutYolo::forward (GPU+CPU readback)
//!       -> NMS (CPU) -> DpdfPipeline::build_page (CPU) -> PageOutput
//! ```
//!
//! # GPU Hardening (Part of #4317)
//!
//! - **Preprocessing** runs inside [`with_gpu_scope`](crate::with_gpu_scope) to
//!   batch permute + resize + normalize into one command buffer commit.
//! - **Model forward** runs with [`NanCheckPolicy::Skip`] to avoid per-layer
//!   flush+readback cycles from `check_output_finite`. DocLayoutYolo's
//!   `decode_detections` does CPU readback (`to_flat_vec`) for NMS, so the
//!   forward MUST NOT be wrapped in `with_gpu_scope` (nn_engineering.md rule:
//!   "GpuScope must NOT wrap functions that do CPU readback").
//! - **Multi-page** processing reuses GPU buffers via the always-on default
//!   [`ActivationArena`](crate::ActivationArena) (auto-grow, reset on flush).
//!
//! Part of #3890.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_core::{Device, Result, TensorError};
use nn_models::doclayout_yolo::{DocLayoutYolo, INPUT_SIZE};
use nn_models::dpdf_pipeline::{DpdfPipeline, DocumentOutput, PageOutput, PipelineConfig};

/// ImageNet channel means for normalization (RGB order).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet channel standard deviations for normalization (RGB order).
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// GPU-accelerated dpdf document inference pipeline.
///
/// Wraps [`DpdfPipeline`] (post-processing) and [`DocLayoutYolo`] (layout
/// detection model) with Metal GPU dispatch for all model forward passes.
/// Falls back to CPU for ops without GPU kernels.
///
/// # GPU Buffer Reuse
///
/// The default Metal [`ActivationArena`](crate::ActivationArena) is always-on
/// with auto-grow. GPU activation buffers are automatically reused between
/// pages via arena reset on each command buffer flush. No explicit
/// `with_arena` call is needed for typical usage.
///
/// For advanced control (e.g., pre-allocating a specific arena size), callers
/// can use [`ensure_default_arena_capacity`](crate::ensure_default_arena_capacity)
/// before calling [`process_document`](Self::process_document).
pub struct DpdfPipelineMetal {
    /// Layout detection model (GPU-resident weights).
    model: DocLayoutYolo,
    /// Post-processing pipeline (NMS, reading order, markdown export).
    pipeline: DpdfPipeline,
}

impl DpdfPipelineMetal {
    /// Create a new GPU-accelerated dpdf pipeline.
    ///
    /// The `model` should be loaded with weights on the Metal GPU device.
    /// `config` controls NMS thresholds and OCR parameters.
    #[must_use]
    pub fn new(model: DocLayoutYolo, config: PipelineConfig) -> Self {
        Self {
            model,
            pipeline: DpdfPipeline::new(config),
        }
    }

    /// Access the underlying pipeline configuration.
    #[must_use]
    pub fn config(&self) -> &PipelineConfig {
        self.pipeline.config()
    }

    /// Access the layout detection model.
    #[must_use]
    pub fn model(&self) -> &DocLayoutYolo {
        &self.model
    }

    /// Process a single page image on GPU.
    ///
    /// # Arguments
    ///
    /// - `image`: page image tensor in HWC format `[H, W, 3]` or CHW format
    ///   `[3, H, W]` or batched `[1, 3, H, W]`, float values in `[0, 1]`.
    ///
    /// # Pipeline
    ///
    /// 1. Preprocess: resize to model input size, normalize, ensure `[1, 3, H, W]`
    /// 2. Model inference on GPU (DocLayoutYolo backbone -> neck -> head)
    /// 3. NMS + reading order on CPU (sequential, no GPU benefit)
    ///
    /// # Errors
    ///
    /// Returns error on invalid image shape, GPU dispatch failure, or model error.
    pub fn process_page(&self, image: &DynTensor) -> Result<PageOutput> {
        let (orig_h, orig_w) = extract_image_hw(image)?;

        // Step 1: Preprocess on GPU inside a gpu_scope to batch permute +
        // resize + normalize into one command buffer commit. Preprocessing is
        // pure GPU work with no CPU readback, so gpu_scope is safe here.
        let preprocessed = crate::gpu_scope::with_gpu_scope(|| {
            self.preprocess_image(image)
        })?;

        // Step 2: Run model on GPU with NaN checks skipped.
        //
        // Skip per-layer NaN checks to avoid flush+readback cycles that can
        // cause Metal GPU timeouts (same pattern as GraniteDocling fix in
        // #4317). DocLayoutYolo::forward calls decode_detections which does
        // CPU readback (to_flat_vec) for NMS, so we must NOT wrap it in
        // with_gpu_scope (nn_engineering.md rule: "GpuScope must NOT wrap
        // functions that do CPU readback").
        //
        // The model output is Vec<Detection> (CPU-side f32 values from NMS),
        // so output validation is implicit -- NaN confidence values would be
        // filtered by the threshold comparison below (NaN fails `>=`).
        let detections = with_nan_check_policy(NanCheckPolicy::Skip, || {
            self.model.forward(&preprocessed)
        })?;

        // Step 3: NMS + reading order on CPU.
        // Detections are already Vec<Detection> on CPU from the model head.
        let regions: Vec<_> = detections
            .iter()
            .filter(|d| d.confidence >= self.pipeline.config().layout_conf_threshold)
            .map(|d| {
                DpdfPipeline::classify_detection(
                    d.class_id as usize,
                    [d.x1, d.y1, d.x2, d.y2],
                    d.confidence,
                )
            })
            .collect();

        Ok(self.pipeline.build_page(regions, orig_w, orig_h))
    }

    /// Process a multi-page document.
    ///
    /// Processes each page sequentially through [`process_page`](Self::process_page).
    /// GPU activation buffers are reused between pages via the always-on
    /// default [`ActivationArena`](crate::ActivationArena) (auto-grow, reset
    /// on each command buffer flush).
    ///
    /// # Errors
    ///
    /// Returns error if any page fails processing.
    pub fn process_document(&self, pages: &[DynTensor]) -> Result<DocumentOutput> {
        let page_outputs: Result<Vec<PageOutput>> = pages
            .iter()
            .map(|page| self.process_page(page))
            .collect();
        Ok(DocumentOutput {
            pages: page_outputs?,
        })
    }

    /// Preprocess image for model input.
    ///
    /// 1. Ensure GPU device
    /// 2. Convert HWC -> CHW if needed
    /// 3. Add batch dimension if needed
    /// 4. Resize to model input size (bilinear)
    /// 5. Normalize with ImageNet mean/std
    fn preprocess_image(&self, image: &DynTensor) -> Result<DynTensor> {
        let device = Device::metal();
        let input_size = INPUT_SIZE;

        // Ensure tensor is on GPU.
        let img = if image.device().is_gpu() {
            image.clone()
        } else {
            image.to_device(&device)?
        };

        // Handle shape variants: HWC [H, W, 3] -> CHW [3, H, W].
        let img = match img.rank() {
            3 => {
                let dims = img.dims();
                if dims[2] == 3 {
                    // HWC -> CHW: permute [H, W, C] -> [C, H, W]
                    img.permute([2, 0, 1])?
                } else if dims[0] == 3 {
                    // Already CHW
                    img
                } else {
                    return Err(TensorError::InvalidShape(format!(
                        "dpdf preprocess: expected [H,W,3] or [3,H,W], got {dims:?}"
                    )));
                }
            }
            4 => {
                // [B, 3, H, W] -- already batched CHW
                let dims = img.dims();
                if dims[0] != 1 {
                    return Err(TensorError::InvalidShape(format!(
                        "dpdf preprocess: batch size must be 1, got {}",
                        dims[0]
                    )));
                }
                if dims[1] != 3 {
                    return Err(TensorError::InvalidShape(format!(
                        "dpdf preprocess: expected 3 channels, got {}",
                        dims[1]
                    )));
                }
                // Remove batch dim, will re-add below.
                img.squeeze(0)?
            }
            _ => {
                return Err(TensorError::InvalidShape(format!(
                    "dpdf preprocess: expected rank 3 or 4, got {}",
                    img.rank()
                )));
            }
        };

        // Now img is [3, H, W]. Add batch dim -> [1, 3, H, W].
        let img = img.unsqueeze(0)?;

        // Resize to model input size via bilinear interpolation.
        // Uses GPU kernel when available, CPU fallback otherwise.
        let img = img.resize_bilinear(input_size, input_size)?;

        // Normalize: (x - mean) / std per channel.
        // Build mean/std tensors shaped [1, 3, 1, 1] for broadcast.
        let mean = DynTensor::from_vec(
            IMAGENET_MEAN.to_vec(),
            &[1, 3, 1, 1],
            &img.device(),
        )?;
        let std_dev = DynTensor::from_vec(
            IMAGENET_STD.to_vec(),
            &[1, 3, 1, 1],
            &img.device(),
        )?;

        let img = img.broadcast_sub(&mean)?;
        let img = img.broadcast_div(&std_dev)?;

        Ok(img)
    }
}

/// Extract (height, width) from an image tensor of rank 3 or 4.
fn extract_image_hw(image: &DynTensor) -> Result<(usize, usize)> {
    let dims = image.dims();
    match dims.len() {
        3 => {
            // [H, W, 3] or [3, H, W]
            if dims[2] == 3 {
                Ok((dims[0], dims[1]))
            } else if dims[0] == 3 {
                Ok((dims[1], dims[2]))
            } else {
                Err(TensorError::InvalidShape(format!(
                    "extract_image_hw: expected [H,W,3] or [3,H,W], got {dims:?}"
                )))
            }
        }
        4 => {
            // [B, 3, H, W]
            Ok((dims[2], dims[3]))
        }
        _ => Err(TensorError::InvalidShape(format!(
            "extract_image_hw: expected rank 3 or 4, got {dims:?}"
        ))),
    }
}

#[cfg(test)]
#[path = "dpdf_pipeline_metal_tests.rs"]
mod tests;
