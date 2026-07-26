// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Anchor-free detection head for YOLO-style object detection.
//!
//! YOLOv8-style `Detect` module with decoupled classification and regression
//! branches. Each scale gets independent prediction heads. The regression
//! branch predicts Distribution Focal Loss (DFL) bin distributions over
//! bounding box offsets; the classification branch predicts per-class scores.
//!
//! ```text
//! Per scale:
//!   features ─┬─ cls_convs → Conv(1x1) → cls_pred  [B, num_classes, H, W]
//!             └─ reg_convs → Conv(1x1) → reg_pred  [B, 4 * reg_max, H, W]
//! ```
//!
//! Post-processing decodes the DFL distributions into (x1, y1, x2, y2) boxes
//! using an integral over the distribution bins, then applies [`nms`](super::nms).

use crate::dyn_tensor::DynTensor;
use crate::error::{Result, TensorError};
use crate::layers::{Activation, Conv2d, Conv2dConfig, Module};
use crate::var_builder::VarBuilder;
use crate::{DType, Device};

use super::nms::Detection;
use super::ConvBnAct;

/// Anchor-free detection head — decoupled classification + regression.
///
/// Input: N feature maps from the PAN neck, each `[B, C_i, H_i, W_i]`.
/// Output: per-scale raw predictions (classification logits + bbox regression).
///
/// # Architecture
///
/// Each scale has:
/// - 2 stacked ConvBnAct (3×3) for classification features
/// - 2 stacked ConvBnAct (3×3) for regression features
/// - 1×1 Conv2d projecting to `num_classes` (cls) and `4 * reg_max` (reg)
///
/// # Weight names
///
/// Per scale `i`:
/// - `"cls_convs.{i}.0.*"`, `"cls_convs.{i}.1.*"` — cls feature convs
/// - `"reg_convs.{i}.0.*"`, `"reg_convs.{i}.1.*"` — reg feature convs
/// - `"cls_pred.{i}.*"` — classification projection (1×1)
/// - `"reg_pred.{i}.*"` — regression projection (1×1)
#[derive(Clone, Debug)]
pub struct DetectHead {
    cls_convs: Vec<[ConvBnAct; 2]>,
    reg_convs: Vec<[ConvBnAct; 2]>,
    cls_preds: Vec<Conv2d>,
    reg_preds: Vec<Conv2d>,
    num_classes: usize,
    reg_max: usize,
}

/// Raw detection output from a single scale.
#[derive(Debug)]
pub struct ScaleOutput {
    /// Classification logits: `[B, num_classes, H, W]`.
    pub cls_logits: DynTensor,
    /// Regression predictions: `[B, 4 * reg_max, H, W]`.
    pub reg_preds: DynTensor,
}

impl DetectHead {
    /// Create from pre-loaded components.
    ///
    /// All vectors must have the same length (one entry per detection scale).
    pub fn new(
        cls_convs: Vec<[ConvBnAct; 2]>,
        reg_convs: Vec<[ConvBnAct; 2]>,
        cls_preds: Vec<Conv2d>,
        reg_preds: Vec<Conv2d>,
        num_classes: usize,
        reg_max: usize,
    ) -> Result<Self> {
        let n = cls_convs.len();
        if reg_convs.len() != n || cls_preds.len() != n || reg_preds.len() != n {
            return Err(TensorError::shape_mismatch(vec![n], vec![reg_convs.len()]));
        }
        Ok(Self {
            cls_convs,
            reg_convs,
            cls_preds,
            reg_preds,
            num_classes,
            reg_max,
        })
    }

    /// Load a detection head from a VarBuilder.
    ///
    /// - `in_channels`: channel count for each input scale
    /// - `num_classes`: number of object classes
    /// - `reg_max`: DFL distribution bins (typically 16)
    /// - `hidden`: intermediate channel count for conv branches (typically 256)
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: &[usize],
        num_classes: usize,
        reg_max: usize,
        hidden: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let n_scales = in_channels.len();
        let mut cls_convs = Vec::with_capacity(n_scales);
        let mut reg_convs = Vec::with_capacity(n_scales);
        let mut cls_preds = Vec::with_capacity(n_scales);
        let mut reg_preds = Vec::with_capacity(n_scales);

        for (i, &in_c) in in_channels.iter().enumerate() {
            // Classification branch
            let cc0 = ConvBnAct::load(
                vb.pp(format!("cls_convs.{i}.0")),
                in_c,
                hidden,
                3,
                1,
                Some(Activation::Silu),
            )?;
            let cc1 = ConvBnAct::load(
                vb.pp(format!("cls_convs.{i}.1")),
                hidden,
                hidden,
                3,
                1,
                Some(Activation::Silu),
            )?;
            cls_convs.push([cc0, cc1]);

            // Regression branch
            let rc0 = ConvBnAct::load(
                vb.pp(format!("reg_convs.{i}.0")),
                in_c,
                hidden,
                3,
                1,
                Some(Activation::Silu),
            )?;
            let rc1 = ConvBnAct::load(
                vb.pp(format!("reg_convs.{i}.1")),
                hidden,
                hidden,
                3,
                1,
                Some(Activation::Silu),
            )?;
            reg_convs.push([rc0, rc1]);

            // 1×1 projection heads
            let cls_cfg = Conv2dConfig::new(0, 1, 1);
            let cls_w = vb.pp(format!("cls_pred.{i}"));
            let cls_proj = Conv2d::load(&cls_w, hidden, num_classes, 1, cls_cfg)?;
            cls_preds.push(cls_proj);

            let reg_cfg = Conv2dConfig::new(0, 1, 1);
            let reg_w = vb.pp(format!("reg_pred.{i}"));
            let reg_proj = Conv2d::load(&reg_w, hidden, 4 * reg_max, 1, reg_cfg)?;
            reg_preds.push(reg_proj);
        }

        Ok(Self {
            cls_convs,
            reg_convs,
            cls_preds,
            reg_preds,
            num_classes,
            reg_max,
        })
    }

    /// Number of object classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// DFL regression bin count.
    #[must_use]
    pub fn reg_max(&self) -> usize {
        self.reg_max
    }

    /// Number of detection scales.
    #[must_use]
    pub fn num_scales(&self) -> usize {
        self.cls_convs.len()
    }

    /// Forward pass on multi-scale features.
    ///
    /// - `features`: slice of tensors `[B, C_i, H_i, W_i]`, one per scale.
    ///   Length must match `num_scales()`.
    ///
    /// Returns per-scale `ScaleOutput` with classification logits and
    /// regression predictions.
    pub fn forward_multi(&self, features: &[&DynTensor]) -> Result<Vec<ScaleOutput>> {
        if features.len() != self.cls_convs.len() {
            return Err(TensorError::shape_mismatch(
                vec![self.cls_convs.len()],
                vec![features.len()],
            ));
        }

        let mut outputs = Vec::with_capacity(features.len());
        for (i, feat) in features.iter().enumerate() {
            // Classification branch
            let cls_feat = self.cls_convs[i][0].forward(feat)?;
            let cls_feat = self.cls_convs[i][1].forward(&cls_feat)?;
            let cls_logits = self.cls_preds[i].forward(&cls_feat)?;

            // Regression branch
            let reg_feat = self.reg_convs[i][0].forward(feat)?;
            let reg_feat = self.reg_convs[i][1].forward(&reg_feat)?;
            let reg_preds = self.reg_preds[i].forward(&reg_feat)?;

            outputs.push(ScaleOutput {
                cls_logits,
                reg_preds,
            });
        }
        Ok(outputs)
    }

    /// Decode DFL regression predictions into bounding box distances.
    ///
    /// Takes `[B, 4 * reg_max, H, W]` regression output and returns
    /// `[B, 4, H, W]` decoded box distances (left, top, right, bottom).
    ///
    /// Uses softmax over each group of `reg_max` bins followed by integral
    /// (weighted sum with bin indices).
    pub fn decode_dfl(&self, reg: &DynTensor) -> Result<DynTensor> {
        let shape = reg.dims().to_vec();
        if shape.len() != 4 {
            return Err(TensorError::shape_mismatch(vec![0, 0, 0, 0], shape));
        }
        let [b, _c, h, w] = [shape[0], shape[1], shape[2], shape[3]];
        let rm = self.reg_max;

        // Reshape to [B, 4, reg_max, H*W] for softmax over dim 2
        let x = reg.reshape([b, 4, rm, h * w])?;
        let x = x.softmax(2)?;

        // Create bin indices [0, 1, ..., reg_max-1] on same device as input
        let indices: Vec<f32> = (0..rm).map(|i| i as f32).collect();
        let idx = DynTensor::from_vec(indices, &[1, 1, rm, 1], &reg.device())?;

        // Weighted sum: integral over distribution bins → [B, 4, H*W]
        let x = x.broadcast_mul(&idx)?;
        let x = x.sum_keepdim(2)?;
        let x = x.squeeze(2)?;
        x.reshape([b, 4, h, w])
    }

    /// Decode multi-scale detection outputs into final bounding boxes.
    ///
    /// Complete anchor-free decoding pipeline:
    /// 1. DFL decode each scale's regression output
    /// 2. Convert distance predictions to absolute boxes via anchor grids
    /// 3. Apply sigmoid to classification logits
    /// 4. Filter by confidence threshold and apply NMS
    ///
    /// # Arguments
    ///
    /// - `scale_outputs`: raw outputs from [`forward_multi`](Self::forward_multi)
    /// - `strides`: per-scale stride values (e.g., `[8, 16, 32]`)
    /// - `img_size`: `(height, width)` of the original input image
    /// - `confidence_threshold`: minimum confidence to keep a detection
    /// - `iou_threshold`: IoU threshold for NMS suppression
    ///
    /// Returns filtered [`Detection`] objects in descending confidence order.
    pub fn decode_detections(
        &self,
        scale_outputs: &[ScaleOutput],
        strides: &[usize],
        img_size: (usize, usize),
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Detection>> {
        if scale_outputs.len() != strides.len() {
            return Err(TensorError::shape_mismatch(
                vec![strides.len()],
                vec![scale_outputs.len()],
            ));
        }

        let (img_h, img_w) = img_size;
        let img_w_f = img_w as f32;
        let img_h_f = img_h as f32;
        let mut all_detections = Vec::new();

        for (scale, stride) in scale_outputs.iter().zip(strides.iter()) {
            let cls_shape = scale.cls_logits.dims().to_vec();
            if cls_shape.len() != 4 {
                return Err(TensorError::shape_mismatch(vec![0, 0, 0, 0], cls_shape));
            }
            let b = cls_shape[0];
            let h = cls_shape[2];
            let w = cls_shape[3];
            let stride_f = *stride as f32;

            // DFL decode: [B, 4*reg_max, H, W] -> [B, 4, H, W]
            let dist = self.decode_dfl(&scale.reg_preds)?;
            // Sigmoid on cls logits: [B, num_classes, H, W]
            let cls_scores = scale.cls_logits.sigmoid()?;

            // Generate anchor grid centers for this scale
            let (grid_x, grid_y) = make_anchor_grid(h, w, &scale.cls_logits.device())?;

            // Flatten spatial dims for extraction
            let dist_flat = dist.reshape([b, 4, h * w])?;
            let cls_flat = cls_scores.reshape([b, self.num_classes, h * w])?;
            let gx_flat = grid_x.reshape([1, 1, h * w])?;
            let gy_flat = grid_y.reshape([1, 1, h * w])?;

            // Extract distance channels: [B, 1, H*W] each
            let d_left = dist_flat.narrow(1, 0, 1)?;
            let d_top = dist_flat.narrow(1, 1, 1)?;
            let d_right = dist_flat.narrow(1, 2, 1)?;
            let d_bottom = dist_flat.narrow(1, 3, 1)?;

            // dist2bbox: convert (left, top, right, bottom) distances to
            // absolute (x1, y1, x2, y2) pixel coordinates.
            //   x1 = (grid_x + 0.5 - d_left) * stride
            //   y1 = (grid_y + 0.5 - d_top)  * stride
            //   x2 = (grid_x + 0.5 + d_right) * stride
            //   y2 = (grid_y + 0.5 + d_bottom) * stride
            let dev = dist.device();
            let half = DynTensor::full(&[1, 1, 1], 0.5, DType::F32, &dev)?;
            let stride_t = DynTensor::full(&[1, 1, 1], f64::from(stride_f), DType::F32, &dev)?;

            let cx = gx_flat.broadcast_add(&half)?;
            let cy = gy_flat.broadcast_add(&half)?;

            let x1 = cx.broadcast_sub(&d_left)?.broadcast_mul(&stride_t)?;
            let y1 = cy.broadcast_sub(&d_top)?.broadcast_mul(&stride_t)?;
            let x2 = cx.broadcast_add(&d_right)?.broadcast_mul(&stride_t)?;
            let y2 = cy.broadcast_add(&d_bottom)?.broadcast_mul(&stride_t)?;

            // Extract raw CPU data
            let x1_v = x1.to_flat_vec::<f32>()?;
            let y1_v = y1.to_flat_vec::<f32>()?;
            let x2_v = x2.to_flat_vec::<f32>()?;
            let y2_v = y2.to_flat_vec::<f32>()?;
            let cls_v = cls_flat.to_flat_vec::<f32>()?;

            let spatial = h * w;
            for bi in 0..b {
                for pos in 0..spatial {
                    // Find best class score at this anchor position
                    let mut best_class = 0u32;
                    let mut best_score = 0.0f32;
                    for c in 0..self.num_classes {
                        let idx = bi * self.num_classes * spatial + c * spatial + pos;
                        let score = cls_v[idx];
                        if score > best_score {
                            best_score = score;
                            best_class = c as u32;
                        }
                    }

                    if best_score < confidence_threshold {
                        continue;
                    }

                    let box_idx = bi * spatial + pos;
                    all_detections.push(Detection {
                        x1: x1_v[box_idx].clamp(0.0, img_w_f),
                        y1: y1_v[box_idx].clamp(0.0, img_h_f),
                        x2: x2_v[box_idx].clamp(0.0, img_w_f),
                        y2: y2_v[box_idx].clamp(0.0, img_h_f),
                        confidence: best_score,
                        class_id: best_class,
                    });
                }
            }
        }

        super::nms::nms(&all_detections, confidence_threshold, iou_threshold)
    }
}

/// Generate anchor grid coordinates for a single detection scale.
///
/// Returns `(grid_x, grid_y)` tensors each of shape `[1, 1, H, W]`
/// containing integer grid coordinates (0-indexed).
///
/// The actual center in pixel space is `(grid_coord + 0.5) * stride`.
pub fn make_anchor_grid(h: usize, w: usize, device: &Device) -> Result<(DynTensor, DynTensor)> {
    if h == 0 || w == 0 {
        return Err(TensorError::ValueOutOfRange {
            description: "make_anchor_grid: height and width must be > 0",
        });
    }

    let mut gx = Vec::with_capacity(h * w);
    let mut gy = Vec::with_capacity(h * w);
    for row in 0..h {
        for col in 0..w {
            gx.push(col as f32);
            gy.push(row as f32);
        }
    }

    let grid_x = DynTensor::from_vec(gx, &[1, 1, h, w], device)?;
    let grid_y = DynTensor::from_vec(gy, &[1, 1, h, w], device)?;
    Ok((grid_x, grid_y))
}

#[cfg(test)]
#[path = "detect_head_tests.rs"]
mod tests;
