// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Table cell detection post-processing: NMS, box decoding, confidence
//! filtering for UniTable and Table Transformer outputs.
//!
//! This module converts raw model outputs (logits + box predictions) into
//! a clean set of cell-level detections suitable for downstream structure
//! parsing in [`super::table_structure`].
//!
//! # Pipeline
//!
//! 1. **Box decoding**: Convert `(cx, cy, w, h)` normalized predictions to
//!    `[x1, y1, x2, y2]` pixel coordinates.
//! 2. **Confidence filtering**: Remove predictions below a threshold.
//! 3. **Non-Maximum Suppression (NMS)**: Deduplicate overlapping boxes,
//!    keeping only the highest-confidence detection per spatial cluster.
//! 4. **Class assignment**: Apply softmax-argmax to logits for class IDs.

use nn_core::layers::vision::Detection;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for table cell post-processing.
#[derive(Debug, Clone)]
pub struct TableCellPostProcessConfig {
    /// Minimum confidence score to keep a detection (default 0.5).
    pub confidence_threshold: f32,
    /// IoU threshold for NMS suppression (default 0.5).
    pub nms_iou_threshold: f32,
    /// Image width in pixels for box denormalization.
    pub image_width: f32,
    /// Image height in pixels for box denormalization.
    pub image_height: f32,
}

impl TableCellPostProcessConfig {
    /// Create a config for the given image dimensions.
    #[must_use]
    pub fn new(image_width: f32, image_height: f32) -> Self {
        Self {
            confidence_threshold: 0.5,
            nms_iou_threshold: 0.5,
            image_width,
            image_height,
        }
    }
}

// ---------------------------------------------------------------------------
// Box decoding
// ---------------------------------------------------------------------------

/// Decode DETR-style normalized `(cx, cy, w, h)` boxes to `[x1, y1, x2, y2]`
/// in pixel coordinates.
///
/// `boxes` is a flat slice of `[num_queries * 4]` values in `(cx, cy, w, h)`
/// format, each in `[0, 1]`. Returns the same number of boxes in
/// `[x1, y1, x2, y2]` pixel coordinates.
#[must_use]
pub fn decode_boxes(boxes: &[[f32; 4]], image_width: f32, image_height: f32) -> Vec<[f32; 4]> {
    boxes
        .iter()
        .map(|b| cxcywh_to_xyxy(b, image_width, image_height))
        .collect()
}

/// Convert a single `(cx, cy, w, h)` box to `[x1, y1, x2, y2]` pixels.
#[must_use]
fn cxcywh_to_xyxy(b: &[f32; 4], img_w: f32, img_h: f32) -> [f32; 4] {
    let cx = b[0] * img_w;
    let cy = b[1] * img_h;
    let w = b[2] * img_w;
    let h = b[3] * img_h;
    let half_w = w * 0.5;
    let half_h = h * 0.5;
    [
        (cx - half_w).max(0.0),
        (cy - half_h).max(0.0),
        (cx + half_w).min(img_w),
        (cy + half_h).min(img_h),
    ]
}

// ---------------------------------------------------------------------------
// Softmax + argmax for class assignment
// ---------------------------------------------------------------------------

/// Compute softmax probabilities from raw logits for a single query.
///
/// Returns `(class_id, confidence)` where `class_id` is the argmax
/// (excluding the last "no-object" class) and `confidence` is the
/// corresponding softmax probability.
#[must_use]
pub fn classify_logits(logits: &[f32]) -> (u32, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }

    // Softmax with numerical stability (subtract max).
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();

    if !exp_sum.is_finite() || exp_sum <= 0.0 {
        return (0, 0.0);
    }

    // Argmax over all classes except the last (no-object).
    let num_real_classes = if logits.len() > 1 {
        logits.len() - 1
    } else {
        logits.len()
    };

    let mut best_class = 0u32;
    let mut best_prob = 0.0f32;
    for (i, &logit) in logits.iter().take(num_real_classes).enumerate() {
        let prob = (logit - max_val).exp() / exp_sum;
        if prob > best_prob {
            best_prob = prob;
            best_class = i as u32;
        }
    }

    (best_class, best_prob)
}

// ---------------------------------------------------------------------------
// Non-Maximum Suppression
// ---------------------------------------------------------------------------

/// Apply class-aware NMS to a set of detections.
///
/// Detections are sorted by confidence (descending). For each detection,
/// all lower-confidence detections of the same class whose IoU exceeds
/// `iou_threshold` are suppressed.
#[must_use]
pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    let mut sorted: Vec<Detection> = detections.to_vec();
    sorted.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; sorted.len()];
    let mut result = Vec::with_capacity(sorted.len());

    for i in 0..sorted.len() {
        if suppressed[i] {
            continue;
        }
        result.push(sorted[i]);
        for j in (i + 1)..sorted.len() {
            if suppressed[j] {
                continue;
            }
            if sorted[i].class_id != sorted[j].class_id {
                continue;
            }
            let iou = compute_iou_det(&sorted[i], &sorted[j]);
            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    result
}

/// Compute IoU between two `Detection` instances.
fn compute_iou_det(a: &Detection, b: &Detection) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        return 0.0;
    }
    inter_area / union_area
}

// ---------------------------------------------------------------------------
// Full post-processing pipeline
// ---------------------------------------------------------------------------

/// Full post-processing: decode boxes, classify, filter, NMS.
///
/// - `raw_logits`: `[num_queries, num_classes]` flattened row-major.
/// - `raw_boxes`: `[num_queries, 4]` in `(cx, cy, w, h)` normalized format.
/// - `num_classes`: number of classes including the no-object class.
///
/// Returns the surviving [`Detection`] set after confidence filtering and NMS.
#[must_use]
pub fn postprocess_table_detections(
    raw_logits: &[f32],
    raw_boxes: &[[f32; 4]],
    num_classes: usize,
    config: &TableCellPostProcessConfig,
) -> Vec<Detection> {
    if num_classes == 0 || raw_boxes.is_empty() {
        return Vec::new();
    }

    let num_queries = raw_boxes.len();
    let expected_logit_len = num_queries.checked_mul(num_classes).unwrap_or(0);
    if raw_logits.len() != expected_logit_len {
        return Vec::new();
    }

    // 1. Decode boxes to pixel coordinates.
    let decoded_boxes = decode_boxes(raw_boxes, config.image_width, config.image_height);

    // 2. Classify each query and build detections.
    let mut detections = Vec::with_capacity(num_queries);
    for (q, bbox) in decoded_boxes.iter().enumerate() {
        let logit_start = q * num_classes;
        let logit_end = logit_start + num_classes;
        let query_logits = &raw_logits[logit_start..logit_end];

        let (class_id, confidence) = classify_logits(query_logits);

        // Skip low-confidence and "no-object" class (last class).
        if confidence < config.confidence_threshold {
            continue;
        }
        if class_id as usize >= num_classes.saturating_sub(1) {
            continue;
        }

        detections.push(Detection {
            x1: bbox[0],
            y1: bbox[1],
            x2: bbox[2],
            y2: bbox[3],
            confidence,
            class_id,
        });
    }

    // 3. NMS.
    nms(&detections, config.nms_iou_threshold)
}

// ---------------------------------------------------------------------------
// Clamp helpers
// ---------------------------------------------------------------------------

/// Clamp a set of detections to image boundaries.
pub fn clamp_detections(detections: &mut [Detection], image_width: f32, image_height: f32) {
    for det in detections.iter_mut() {
        det.x1 = det.x1.max(0.0).min(image_width);
        det.y1 = det.y1.max(0.0).min(image_height);
        det.x2 = det.x2.max(0.0).min(image_width);
        det.y2 = det.y2.max(0.0).min(image_height);
    }
}

#[cfg(test)]
#[path = "table_cell_postprocess_tests.rs"]
mod tests;
