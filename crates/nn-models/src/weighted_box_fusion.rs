// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weighted Box Fusion (WBF) for ensemble detection merging.
//!
//! WBF merges overlapping bounding boxes from multiple models (or multiple
//! runs of the same model) into refined consensus detections. Unlike NMS
//! which suppresses lower-confidence duplicates, WBF computes a weighted
//! average of all boxes that overlap sufficiently, producing tighter,
//! more accurate bounding boxes.
//!
//! # Algorithm
//!
//! For each model's detections:
//! 1. Sort all detections by confidence (descending).
//! 2. For each detection, find existing clusters with IoU >= threshold.
//! 3. If a matching cluster exists, add the detection to that cluster.
//! 4. If no matching cluster, create a new cluster.
//! 5. For each cluster, compute the fused box as the confidence-weighted
//!    average of all member boxes.
//! 6. Compute the fused confidence as a function of cluster membership
//!    and per-model weights.
//!
//! # Usage
//!
//! ```rust
//! use nn_models::weighted_box_fusion::{WeightedBoxFusion, WbfConfig, ScoredBox};
//!
//! // Two models producing overlapping detections.
//! let model_a = vec![
//!     ScoredBox::new(0, 0.95, [0.1, 0.1, 0.5, 0.5]),
//!     ScoredBox::new(1, 0.80, [0.6, 0.6, 0.9, 0.9]),
//! ];
//! let model_b = vec![
//!     ScoredBox::new(0, 0.90, [0.12, 0.09, 0.48, 0.52]),
//! ];
//!
//! let config = WbfConfig::default();
//! let fused = WeightedBoxFusion::fuse(
//!     &[&model_a, &model_b],
//!     &[1.0, 1.0],
//!     &config,
//! );
//! // fused[0] is a refined box averaging the two model_a/model_b boxes for class 0.
//! ```
//!
//! Reference: Solovyev et al. 2021, "Weighted boxes fusion: Ensembling boxes
//! from different object detection models", Image and Vision Computing, Vol 107.

use crate::dpdf_postprocess::compute_iou;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Weighted Box Fusion.
#[derive(Debug, Clone)]
pub struct WbfConfig {
    /// IoU threshold for merging boxes into a cluster (default 0.55).
    pub iou_threshold: f32,
    /// Minimum fused confidence to keep a detection (default 0.0).
    pub conf_threshold: f32,
    /// Whether to allow fusing boxes of different classes (default false).
    /// When false (recommended), only same-class boxes are fused.
    pub allow_cross_class: bool,
}

impl Default for WbfConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.55,
            conf_threshold: 0.0,
            allow_cross_class: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Scored box (detection from a single model)
// ---------------------------------------------------------------------------

/// A single detection with class, confidence, and bounding box.
///
/// Bounding box coordinates are in `[x1, y1, x2, y2]` format, normalized
/// to `[0, 1]` (or pixel coordinates — WBF is coordinate-agnostic).
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredBox {
    /// Predicted class ID.
    pub class_id: u32,
    /// Detection confidence in `[0, 1]`.
    pub confidence: f32,
    /// Bounding box `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    /// Index of the source model (set internally during fusion).
    pub(crate) model_idx: usize,
}

impl ScoredBox {
    /// Create a new scored box.
    #[must_use]
    pub fn new(class_id: u32, confidence: f32, bbox: [f32; 4]) -> Self {
        Self {
            class_id,
            confidence,
            bbox,
            model_idx: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal cluster type
// ---------------------------------------------------------------------------

/// A cluster of overlapping boxes that will be fused.
struct BoxCluster {
    /// All boxes in this cluster, with their model weights.
    members: Vec<(ScoredBox, f32)>,
    /// Class ID (all members have the same class when `allow_cross_class` is false).
    class_id: u32,
    /// Current fused bounding box (updated incrementally).
    fused_bbox: [f32; 4],
    /// Sum of weights * confidence for normalization.
    weight_sum: f32,
}

impl BoxCluster {
    fn new(first: ScoredBox, model_weight: f32) -> Self {
        let fused_bbox = first.bbox;
        let class_id = first.class_id;
        let weight = model_weight * first.confidence;
        Self {
            members: vec![(first, model_weight)],
            class_id,
            fused_bbox,
            weight_sum: weight,
        }
    }

    fn add(&mut self, sb: ScoredBox, model_weight: f32) {
        let w = model_weight * sb.confidence;
        self.members.push((sb, model_weight));
        self.weight_sum += w;
        self.recompute_fused_bbox();
    }

    /// Recompute the fused bbox as the confidence-weighted average of all members.
    fn recompute_fused_bbox(&mut self) {
        let mut total_w = 0.0f32;
        let mut x1 = 0.0f32;
        let mut y1 = 0.0f32;
        let mut x2 = 0.0f32;
        let mut y2 = 0.0f32;
        for (sb, mw) in &self.members {
            let w = mw * sb.confidence;
            x1 += sb.bbox[0] * w;
            y1 += sb.bbox[1] * w;
            x2 += sb.bbox[2] * w;
            y2 += sb.bbox[3] * w;
            total_w += w;
        }
        if total_w > 0.0 {
            self.fused_bbox = [x1 / total_w, y1 / total_w, x2 / total_w, y2 / total_w];
        }
    }

    /// Compute the fused confidence score.
    ///
    /// Standard WBF formula: `(sum of weighted confidences) * min(T, N) / T`
    /// where T = number of models, N = number of contributing models in this
    /// cluster.
    fn fused_confidence(&self, num_models: usize) -> f32 {
        let t = num_models.max(1) as f32;
        let n = self.num_contributing_models().min(num_models) as f32;
        // Average weighted confidence, scaled by model coverage.
        let avg_conf = self.weight_sum / self.members.len().max(1) as f32;
        avg_conf * n / t
    }

    /// Count how many distinct models contributed to this cluster.
    fn num_contributing_models(&self) -> usize {
        let mut seen = Vec::new();
        for (sb, _) in &self.members {
            if !seen.contains(&sb.model_idx) {
                seen.push(sb.model_idx);
            }
        }
        seen.len()
    }
}

// ---------------------------------------------------------------------------
// Weighted Box Fusion
// ---------------------------------------------------------------------------

/// Weighted Box Fusion: merge overlapping detections from multiple models.
pub struct WeightedBoxFusion;

impl WeightedBoxFusion {
    /// Fuse detections from multiple models.
    ///
    /// # Arguments
    ///
    /// - `model_detections`: slice of per-model detection lists.
    /// - `model_weights`: per-model importance weights (e.g., `[1.0, 1.0]`
    ///   for equal weighting, `[2.0, 1.0]` to favor the first model).
    /// - `config`: WBF parameters (IoU threshold, confidence threshold).
    ///
    /// # Returns
    ///
    /// Fused detection list, sorted by descending confidence.
    ///
    /// # Panics
    ///
    /// Panics if `model_detections.len() != model_weights.len()`.
    #[must_use]
    pub fn fuse(
        model_detections: &[&[ScoredBox]],
        model_weights: &[f32],
        config: &WbfConfig,
    ) -> Vec<ScoredBox> {
        assert_eq!(
            model_detections.len(),
            model_weights.len(),
            "WeightedBoxFusion: model_detections and model_weights must have the same length"
        );

        let num_models = model_detections.len();
        if num_models == 0 {
            return Vec::new();
        }

        // 1. Collect all detections with model index and sort by confidence descending.
        let mut all_boxes: Vec<(ScoredBox, f32)> = Vec::new();
        for (model_idx, (dets, &weight)) in model_detections
            .iter()
            .zip(model_weights.iter())
            .enumerate()
        {
            for det in *dets {
                let mut sb = det.clone();
                sb.model_idx = model_idx;
                all_boxes.push((sb, weight));
            }
        }
        all_boxes.sort_by(|a, b| {
            b.0.confidence
                .partial_cmp(&a.0.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 2. Build clusters.
        let mut clusters: Vec<BoxCluster> = Vec::new();

        for (sb, weight) in all_boxes {
            let mut best_cluster_idx: Option<usize> = None;
            let mut best_iou = 0.0f32;

            for (idx, cluster) in clusters.iter().enumerate() {
                // Class check (unless cross-class fusion is allowed).
                if !config.allow_cross_class && cluster.class_id != sb.class_id {
                    continue;
                }
                let iou = compute_iou(&sb.bbox, &cluster.fused_bbox);
                if iou >= config.iou_threshold && iou > best_iou {
                    best_iou = iou;
                    best_cluster_idx = Some(idx);
                }
            }

            match best_cluster_idx {
                Some(idx) => {
                    clusters[idx].add(sb, weight);
                }
                None => {
                    clusters.push(BoxCluster::new(sb, weight));
                }
            }
        }

        // 3. Convert clusters to fused detections.
        let mut fused: Vec<ScoredBox> = clusters
            .iter()
            .map(|c| ScoredBox {
                class_id: c.class_id,
                confidence: c.fused_confidence(num_models),
                bbox: c.fused_bbox,
                model_idx: 0,
            })
            .filter(|sb| sb.confidence >= config.conf_threshold)
            .collect();

        // 4. Sort by descending confidence.
        fused.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        fused
    }

    /// Convenience: fuse detections from exactly two models with equal weights.
    #[must_use]
    pub fn fuse_pair(
        detections_a: &[ScoredBox],
        detections_b: &[ScoredBox],
        config: &WbfConfig,
    ) -> Vec<ScoredBox> {
        Self::fuse(&[detections_a, detections_b], &[1.0, 1.0], config)
    }
}

// ---------------------------------------------------------------------------
// Normalize confidence scores across heterogeneous models
// ---------------------------------------------------------------------------

/// Normalize detection confidences from heterogeneous models to a common
/// scale using quantile-based calibration.
///
/// Each model may have different confidence distributions (e.g., YOLO scores
/// are typically higher than DETR scores). This function maps each model's
/// confidences to `[0, 1]` using min-max normalization within each model,
/// then applies optional temperature scaling.
///
/// # Arguments
///
/// - `model_detections`: mutable per-model detection lists.
/// - `temperature`: scaling factor applied after normalization (default 1.0).
///   Values > 1.0 flatten the distribution, < 1.0 sharpen it.
pub fn normalize_confidences(model_detections: &mut [&mut [ScoredBox]], temperature: f32) {
    for dets in model_detections.iter_mut() {
        if dets.is_empty() {
            continue;
        }

        let min_conf = dets
            .iter()
            .map(|d| d.confidence)
            .fold(f32::INFINITY, f32::min);
        let max_conf = dets
            .iter()
            .map(|d| d.confidence)
            .fold(f32::NEG_INFINITY, f32::max);

        let range = max_conf - min_conf;
        if range <= 0.0 {
            // All confidences are the same — normalize to 1.0.
            for d in dets.iter_mut() {
                d.confidence = 1.0;
            }
            continue;
        }

        let inv_temp = 1.0 / temperature.max(1e-6);
        for d in dets.iter_mut() {
            let normalized = (d.confidence - min_conf) / range;
            // Temperature-scaled sigmoid.
            d.confidence = 1.0 / (1.0 + (-(normalized * 2.0 - 1.0) * inv_temp).exp());
        }
    }
}

#[cfg(test)]
#[path = "weighted_box_fusion_tests.rs"]
mod tests;
