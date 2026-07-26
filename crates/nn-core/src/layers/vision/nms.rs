// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Non-Maximum Suppression (NMS) for object detection post-processing.
//!
//! Pure algorithmic utility — no learned parameters, no GPU dispatch.
//! Operates on CPU f32 data extracted from detection model outputs.

use crate::error::{Result, TensorError};

/// A single detection bounding box with class confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Top-left x coordinate.
    pub x1: f32,
    /// Top-left y coordinate.
    pub y1: f32,
    /// Bottom-right x coordinate.
    pub x2: f32,
    /// Bottom-right y coordinate.
    pub y2: f32,
    /// Detection confidence score (0..1).
    pub confidence: f32,
    /// Predicted class index.
    pub class_id: u32,
}

impl Detection {
    /// Compute the area of this bounding box.
    ///
    /// Returns 0.0 for degenerate boxes where x2 <= x1 or y2 <= y1.
    #[must_use]
    pub fn area(&self) -> f32 {
        let w = (self.x2 - self.x1).max(0.0);
        let h = (self.y2 - self.y1).max(0.0);
        w * h
    }
}

/// Compute Intersection over Union (IoU) between two bounding boxes.
///
/// Returns 0.0 if either box has zero area or boxes don't overlap.
#[must_use]
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let union_area = a.area() + b.area() - inter_area;
    if union_area <= 0.0 {
        return 0.0;
    }
    inter_area / union_area
}

/// Apply Non-Maximum Suppression to a list of detections.
///
/// 1. Filter detections below `confidence_threshold`.
/// 2. Sort remaining by confidence (descending).
/// 3. Greedily keep the highest-confidence detection and suppress all
///    remaining detections of the **same class** with IoU > `iou_threshold`.
///
/// Returns the surviving detections in descending confidence order.
///
/// # Errors
///
/// Returns `ValueOutOfRange` if thresholds are not in `[0, 1]` or not finite.
pub fn nms(
    detections: &[Detection],
    confidence_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    if !confidence_threshold.is_finite() || !(0.0..=1.0).contains(&confidence_threshold) {
        return Err(TensorError::ValueOutOfRange {
            description: "nms: confidence_threshold must be in [0, 1]",
        });
    }
    if !iou_threshold.is_finite() || !(0.0..=1.0).contains(&iou_threshold) {
        return Err(TensorError::ValueOutOfRange {
            description: "nms: iou_threshold must be in [0, 1]",
        });
    }

    // Filter by confidence
    let mut candidates: Vec<Detection> = detections
        .iter()
        .filter(|d| d.confidence >= confidence_threshold)
        .copied()
        .collect();

    // Sort by confidence descending (NaN-safe: treat NaN as lowest)
    candidates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::with_capacity(candidates.len());
    let mut suppressed = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(candidates[i]);
        // Suppress lower-confidence boxes of the same class with high IoU
        for j in (i + 1)..candidates.len() {
            if suppressed[j] {
                continue;
            }
            if candidates[j].class_id == candidates[i].class_id
                && iou(&candidates[i], &candidates[j]) > iou_threshold
            {
                suppressed[j] = true;
            }
        }
    }

    Ok(keep)
}

#[cfg(test)]
#[path = "nms_tests.rs"]
mod tests;
