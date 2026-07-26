// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Document region post-processing: merging, deduplication, and confidence
//! filtering for dpdf pipeline output.
//!
//! This module provides utilities to refine raw detection outputs from
//! [`super::dpdf_pipeline`] by removing low-confidence detections, merging
//! overlapping same-class regions, deduplicating near-identical results
//! (e.g., from multiple models), and fusing results from different model
//! sources with configurable priority.

use crate::dpdf_pipeline::DocumentRegion;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the full post-processing pipeline.
#[derive(Debug, Clone)]
pub struct PostProcessConfig {
    /// IoU threshold for merging overlapping same-class regions (default 0.5).
    pub merge_iou: f32,
    /// Similarity threshold for deduplication (default 0.9).
    pub dedup_similarity: f32,
    /// Minimum confidence to keep a region (default 0.3).
    pub min_confidence: f32,
    /// Whether to enable multi-model fusion (default true).
    pub enable_model_fusion: bool,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            merge_iou: 0.5,
            dedup_similarity: 0.9,
            min_confidence: 0.3,
            enable_model_fusion: true,
        }
    }
}

/// Priority ordering for multi-model fusion.
///
/// When regions from different model sources overlap, higher-priority
/// sources take precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionPriority {
    /// DocLayout-YOLO: general layout detection (highest priority for
    /// structural elements).
    DocLayout,
    /// Table Transformer: specialised table detection.
    TableTransformer,
    /// OCR model: text-level bounding boxes (lowest structural priority).
    Ocr,
}

// ---------------------------------------------------------------------------
// IoU computation
// ---------------------------------------------------------------------------

/// Compute Intersection over Union between two `[x1, y1, x2, y2]` bounding
/// boxes.
///
/// Returns 0.0 if either box has zero area or the boxes do not overlap.
#[must_use]
pub fn compute_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_x1 = a[0].max(b[0]);
    let inter_y1 = a[1].max(b[1]);
    let inter_x2 = a[2].min(b[2]);
    let inter_y2 = a[3].min(b[3]);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        return 0.0;
    }
    inter_area / union_area
}

// ---------------------------------------------------------------------------
// Merge overlapping same-class regions
// ---------------------------------------------------------------------------

/// Merge overlapping same-class regions whose IoU exceeds `iou_threshold`.
///
/// When two regions of the same class overlap beyond the threshold, they are
/// merged into one region whose bounding box is the enclosing union and
/// whose confidence is the maximum of the two. Regions of different classes
/// are never merged.
///
/// The algorithm is greedy: it iterates in order, merging the first
/// qualifying pair found, then restarts. This converges because each merge
/// strictly reduces the region count.
pub fn merge_overlapping_regions(regions: &mut Vec<DocumentRegion>, iou_threshold: f32) {
    // Greedy merge: restart whenever a merge happens.
    loop {
        let mut merged = false;
        'outer: for i in 0..regions.len() {
            for j in (i + 1)..regions.len() {
                if regions[i].class_name() != regions[j].class_name() {
                    continue;
                }
                let iou = compute_iou(&regions[i].bbox(), &regions[j].bbox());
                if iou > iou_threshold {
                    let merged_region = merge_two(&regions[i], &regions[j]);
                    regions[i] = merged_region;
                    regions.remove(j);
                    merged = true;
                    break 'outer;
                }
            }
        }
        if !merged {
            break;
        }
    }
}

/// Merge two regions of the same class into one enclosing region.
fn merge_two(a: &DocumentRegion, b: &DocumentRegion) -> DocumentRegion {
    let ba = a.bbox();
    let bb = b.bbox();
    let merged_bbox = [
        ba[0].min(bb[0]),
        ba[1].min(bb[1]),
        ba[2].max(bb[2]),
        ba[3].max(bb[3]),
    ];
    let merged_conf = a.confidence().max(b.confidence());
    // Re-classify using the class name of `a` (same class guaranteed by caller).
    rebuild_region(a, merged_bbox, merged_conf)
}

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

/// Remove near-duplicate regions whose IoU exceeds `similarity_threshold`.
///
/// Among duplicates (same class, high IoU), the region with higher confidence
/// is kept. This is useful after fusing results from multiple models that may
/// detect the same region.
pub fn deduplicate_regions(regions: &mut Vec<DocumentRegion>, similarity_threshold: f32) {
    let mut suppressed = vec![false; regions.len()];
    // Sort by confidence descending so we keep higher-confidence detections.
    regions.sort_by(|a, b| {
        b.confidence()
            .partial_cmp(&a.confidence())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for i in 0..regions.len() {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..regions.len() {
            if suppressed[j] {
                continue;
            }
            if regions[i].class_name() == regions[j].class_name() {
                let iou = compute_iou(&regions[i].bbox(), &regions[j].bbox());
                if iou > similarity_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }
    // Remove suppressed in reverse order to preserve indices.
    for i in (0..suppressed.len()).rev() {
        if suppressed[i] {
            regions.remove(i);
        }
    }
}

// ---------------------------------------------------------------------------
// Confidence filtering
// ---------------------------------------------------------------------------

/// Remove regions whose confidence is below `min_confidence`.
pub fn filter_by_confidence(regions: &mut Vec<DocumentRegion>, min_confidence: f32) {
    regions.retain(|r| r.confidence() >= min_confidence);
}

// ---------------------------------------------------------------------------
// Multi-model fusion
// ---------------------------------------------------------------------------

/// Fuse results from multiple detection models with priority ordering.
///
/// Combines regions from DocLayout-YOLO (structural layout), Table
/// Transformer (specialised table detection), and OCR (text boxes) into
/// a single list. When regions from different sources overlap significantly
/// (IoU > 0.5), the higher-priority source wins.
///
/// Priority: `DocLayout` > `TableTransformer` > `Ocr`.
#[must_use]
pub fn fuse_model_results(
    doclayout: &[DocumentRegion],
    table_det: &[DocumentRegion],
    ocr: &[DocumentRegion],
) -> Vec<DocumentRegion> {
    // Start with highest-priority source and add lower-priority regions
    // only if they don't overlap significantly with existing ones.
    let mut fused: Vec<DocumentRegion> = doclayout.to_vec();

    let fusion_iou_threshold = 0.5;

    for region in table_det {
        if !overlaps_any(region, &fused, fusion_iou_threshold) {
            fused.push(region.clone());
        }
    }

    for region in ocr {
        if !overlaps_any(region, &fused, fusion_iou_threshold) {
            fused.push(region.clone());
        }
    }

    fused
}

/// Check whether `candidate` overlaps any region in `existing` above `threshold`.
fn overlaps_any(candidate: &DocumentRegion, existing: &[DocumentRegion], threshold: f32) -> bool {
    let bbox = candidate.bbox();
    existing
        .iter()
        .any(|r| compute_iou(&bbox, &r.bbox()) > threshold)
}

// ---------------------------------------------------------------------------
// Bounding-box refinement
// ---------------------------------------------------------------------------

/// Clamp all region bounding boxes to lie within the image boundaries.
///
/// Coordinates are clamped to `[0, image_width]` for x and `[0, image_height]`
/// for y. Degenerate boxes (where x2 <= x1 or y2 <= y1 after clamping) are
/// left as-is — callers should filter them separately if needed.
pub fn refine_bboxes(regions: &mut [DocumentRegion], image_width: usize, image_height: usize) {
    let w = image_width as f32;
    let h = image_height as f32;
    for region in regions.iter_mut() {
        let bbox = region.bbox();
        let clamped = [
            bbox[0].max(0.0).min(w),
            bbox[1].max(0.0).min(h),
            bbox[2].max(0.0).min(w),
            bbox[3].max(0.0).min(h),
        ];
        if clamped != bbox {
            *region = rebuild_region(region, clamped, region.confidence());
        }
    }
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

/// Apply the full post-processing pipeline to a set of regions.
///
/// Steps (in order):
/// 1. Filter by confidence (`min_confidence`).
/// 2. Merge overlapping same-class regions (`merge_iou`).
/// 3. Deduplicate near-identical regions (`dedup_similarity`).
pub fn postprocess(regions: &mut Vec<DocumentRegion>, config: &PostProcessConfig) {
    filter_by_confidence(regions, config.min_confidence);
    merge_overlapping_regions(regions, config.merge_iou);
    deduplicate_regions(regions, config.dedup_similarity);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Rebuild a `DocumentRegion` with a new bounding box and confidence,
/// preserving the variant type and content fields.
#[allow(unreachable_patterns)] // `DocumentRegion` is `#[non_exhaustive]`
fn rebuild_region(source: &DocumentRegion, bbox: [f32; 4], confidence: f32) -> DocumentRegion {
    match source {
        DocumentRegion::Text { content, .. } => DocumentRegion::Text {
            content: content.clone(),
            bbox,
            confidence,
        },
        DocumentRegion::Table { cells, .. } => DocumentRegion::Table {
            cells: cells.clone(),
            bbox,
            confidence,
        },
        DocumentRegion::Figure { caption, .. } => DocumentRegion::Figure {
            caption: caption.clone(),
            bbox,
            confidence,
        },
        DocumentRegion::Formula { latex, .. } => DocumentRegion::Formula {
            latex: latex.clone(),
            bbox,
            confidence,
        },
        DocumentRegion::SectionHeader { content, .. } => DocumentRegion::SectionHeader {
            content: content.clone(),
            bbox,
            confidence,
        },
        DocumentRegion::PageHeader { content, .. } => DocumentRegion::PageHeader {
            content: content.clone(),
            bbox,
            confidence,
        },
        DocumentRegion::PageFooter { content, .. } => DocumentRegion::PageFooter {
            content: content.clone(),
            bbox,
            confidence,
        },
        DocumentRegion::Caption { content, .. } => DocumentRegion::Caption {
            content: content.clone(),
            bbox,
            confidence,
        },
        DocumentRegion::ListItem { content, .. } => DocumentRegion::ListItem {
            content: content.clone(),
            bbox,
            confidence,
        },
        DocumentRegion::Footnote { content, .. } => DocumentRegion::Footnote {
            content: content.clone(),
            bbox,
            confidence,
        },
        // `#[non_exhaustive]` catch-all: fall back to Text region.
        _ => DocumentRegion::Text {
            content: String::new(),
            bbox,
            confidence,
        },
    }
}

#[cfg(test)]
#[path = "dpdf_postprocess_tests.rs"]
mod tests;
