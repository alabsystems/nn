// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dpdf_pipeline::DocumentRegion;

// ---------------------------------------------------------------------------
// Helper: quickly build a Text region
// ---------------------------------------------------------------------------
fn text_region(bbox: [f32; 4], confidence: f32) -> DocumentRegion {
    DocumentRegion::Text {
        content: String::new(),
        bbox,
        confidence,
    }
}

fn table_region(bbox: [f32; 4], confidence: f32) -> DocumentRegion {
    DocumentRegion::Table {
        cells: Vec::new(),
        bbox,
        confidence,
    }
}

fn header_region(bbox: [f32; 4], confidence: f32) -> DocumentRegion {
    DocumentRegion::SectionHeader {
        content: "heading".to_string(),
        bbox,
        confidence,
    }
}

// ===========================================================================
// compute_iou
// ===========================================================================

#[test]
fn test_compute_iou_identical_boxes() {
    let b = [10.0, 10.0, 50.0, 50.0];
    let iou = compute_iou(&b, &b);
    assert!(
        (iou - 1.0).abs() < 1e-6,
        "identical boxes should have IoU ~1.0, got {iou}"
    );
}

#[test]
fn test_compute_iou_no_overlap() {
    let a = [0.0, 0.0, 10.0, 10.0];
    let b = [20.0, 20.0, 30.0, 30.0];
    let iou = compute_iou(&a, &b);
    assert!(
        iou.abs() < 1e-6,
        "non-overlapping boxes should have IoU ~0.0, got {iou}"
    );
}

#[test]
fn test_compute_iou_partial_overlap() {
    // Box A: 0,0 -> 20,20 (area = 400)
    // Box B: 10,10 -> 30,30 (area = 400)
    // Intersection: 10,10 -> 20,20 (area = 100)
    // Union: 400 + 400 - 100 = 700
    // IoU = 100 / 700 ~ 0.1429
    let a = [0.0, 0.0, 20.0, 20.0];
    let b = [10.0, 10.0, 30.0, 30.0];
    let iou = compute_iou(&a, &b);
    let expected = 100.0 / 700.0;
    assert!(
        (iou - expected).abs() < 1e-5,
        "expected IoU ~{expected}, got {iou}"
    );
}

#[test]
fn test_compute_iou_contained_box() {
    // B is fully inside A.
    let a = [0.0, 0.0, 100.0, 100.0]; // area = 10000
    let b = [20.0, 20.0, 40.0, 40.0]; // area = 400
    let iou = compute_iou(&a, &b);
    // intersection = 400, union = 10000 + 400 - 400 = 10000
    let expected = 400.0 / 10000.0;
    assert!(
        (iou - expected).abs() < 1e-5,
        "expected IoU ~{expected}, got {iou}"
    );
}

#[test]
fn test_compute_iou_zero_area_box() {
    let a = [10.0, 10.0, 10.0, 10.0]; // zero-area
    let b = [0.0, 0.0, 20.0, 20.0];
    assert!(compute_iou(&a, &b).abs() < 1e-6);
}

#[test]
fn test_compute_iou_symmetric() {
    let a = [5.0, 5.0, 25.0, 25.0];
    let b = [15.0, 10.0, 35.0, 30.0];
    let iou_ab = compute_iou(&a, &b);
    let iou_ba = compute_iou(&b, &a);
    assert!(
        (iou_ab - iou_ba).abs() < 1e-6,
        "IoU should be symmetric: {iou_ab} vs {iou_ba}"
    );
}

// ===========================================================================
// filter_by_confidence
// ===========================================================================

#[test]
fn test_filter_by_confidence_removes_low() {
    let mut regions = vec![
        text_region([0.0, 0.0, 10.0, 10.0], 0.1),
        text_region([10.0, 10.0, 20.0, 20.0], 0.5),
        text_region([20.0, 20.0, 30.0, 30.0], 0.9),
    ];
    filter_by_confidence(&mut regions, 0.3);
    assert_eq!(regions.len(), 2, "low-confidence region should be removed");
    assert!(
        regions.iter().all(|r| r.confidence() >= 0.3),
        "all remaining should have confidence >= 0.3"
    );
}

#[test]
fn test_filter_by_confidence_keeps_equal() {
    let mut regions = vec![text_region([0.0, 0.0, 10.0, 10.0], 0.3)];
    filter_by_confidence(&mut regions, 0.3);
    assert_eq!(regions.len(), 1, "region at threshold should be kept");
}

#[test]
fn test_filter_by_confidence_empty_input() {
    let mut regions: Vec<DocumentRegion> = vec![];
    filter_by_confidence(&mut regions, 0.5);
    assert!(regions.is_empty());
}

// ===========================================================================
// merge_overlapping_regions
// ===========================================================================

#[test]
fn test_merge_overlapping_same_class() {
    // Two text regions that overlap significantly.
    let mut regions = vec![
        text_region([0.0, 0.0, 20.0, 20.0], 0.8),
        text_region([5.0, 5.0, 25.0, 25.0], 0.7),
    ];
    // IoU of these two: intersection = 15*15 = 225, union = 400+400-225 = 575
    // IoU ~ 0.391. Use threshold 0.3 to trigger merge.
    merge_overlapping_regions(&mut regions, 0.3);
    assert_eq!(
        regions.len(),
        1,
        "overlapping same-class regions should merge"
    );
    // Merged bbox should be enclosing union.
    let bbox = regions[0].bbox();
    assert!((bbox[0] - 0.0).abs() < 1e-6);
    assert!((bbox[1] - 0.0).abs() < 1e-6);
    assert!((bbox[2] - 25.0).abs() < 1e-6);
    assert!((bbox[3] - 25.0).abs() < 1e-6);
    // Confidence should be max.
    assert!((regions[0].confidence() - 0.8).abs() < 1e-6);
}

#[test]
fn test_merge_does_not_merge_different_classes() {
    let mut regions = vec![
        text_region([0.0, 0.0, 20.0, 20.0], 0.8),
        table_region([0.0, 0.0, 20.0, 20.0], 0.8),
    ];
    merge_overlapping_regions(&mut regions, 0.1);
    assert_eq!(regions.len(), 2, "different classes should not merge");
}

#[test]
fn test_merge_below_threshold_no_merge() {
    let mut regions = vec![
        text_region([0.0, 0.0, 10.0, 10.0], 0.8),
        text_region([50.0, 50.0, 60.0, 60.0], 0.7),
    ];
    merge_overlapping_regions(&mut regions, 0.5);
    assert_eq!(regions.len(), 2, "non-overlapping regions should not merge");
}

// ===========================================================================
// deduplicate_regions
// ===========================================================================

#[test]
fn test_deduplicate_removes_near_duplicates() {
    // Near-identical boxes, same class.
    let mut regions = vec![
        text_region([0.0, 0.0, 100.0, 100.0], 0.9),
        text_region([1.0, 1.0, 101.0, 101.0], 0.7),
    ];
    // IoU is very high (close to 1.0). Use threshold 0.8.
    deduplicate_regions(&mut regions, 0.8);
    assert_eq!(regions.len(), 1, "near-duplicate should be removed");
    assert!(
        (regions[0].confidence() - 0.9).abs() < 1e-6,
        "higher-confidence region should survive"
    );
}

#[test]
fn test_deduplicate_keeps_different_classes() {
    let mut regions = vec![
        text_region([0.0, 0.0, 100.0, 100.0], 0.9),
        table_region([0.0, 0.0, 100.0, 100.0], 0.9),
    ];
    deduplicate_regions(&mut regions, 0.8);
    assert_eq!(
        regions.len(),
        2,
        "different classes should not be deduplicated"
    );
}

#[test]
fn test_deduplicate_preserves_distinct() {
    let mut regions = vec![
        text_region([0.0, 0.0, 10.0, 10.0], 0.9),
        text_region([50.0, 50.0, 60.0, 60.0], 0.8),
    ];
    deduplicate_regions(&mut regions, 0.9);
    assert_eq!(
        regions.len(),
        2,
        "non-overlapping same-class regions are not duplicates"
    );
}

// ===========================================================================
// fuse_model_results
// ===========================================================================

#[test]
fn test_fuse_model_results_priority() {
    let doclayout = vec![text_region([0.0, 0.0, 100.0, 100.0], 0.9)];
    let table_det = vec![table_region([0.0, 0.0, 100.0, 100.0], 0.95)];
    let ocr: Vec<DocumentRegion> = vec![];

    let fused = fuse_model_results(&doclayout, &table_det, &ocr);
    // DocLayout wins — the table region overlaps and is lower priority.
    assert_eq!(
        fused.len(),
        1,
        "overlapping lower-priority region should be dropped"
    );
    assert_eq!(fused[0].class_name(), "text");
}

#[test]
fn test_fuse_model_results_non_overlapping() {
    let doclayout = vec![text_region([0.0, 0.0, 50.0, 50.0], 0.9)];
    let table_det = vec![table_region([60.0, 60.0, 120.0, 120.0], 0.95)];
    let ocr = vec![text_region([200.0, 200.0, 250.0, 250.0], 0.6)];

    let fused = fuse_model_results(&doclayout, &table_det, &ocr);
    assert_eq!(
        fused.len(),
        3,
        "non-overlapping regions from all sources should be kept"
    );
}

#[test]
fn test_fuse_empty_sources() {
    let fused = fuse_model_results(&[], &[], &[]);
    assert!(fused.is_empty());
}

// ===========================================================================
// refine_bboxes
// ===========================================================================

#[test]
fn test_refine_bboxes_clamps_to_image() {
    let mut regions = vec![text_region([-10.0, -5.0, 200.0, 300.0], 0.9)];
    refine_bboxes(&mut regions, 100, 150);
    let bbox = regions[0].bbox();
    assert!((bbox[0] - 0.0).abs() < 1e-6, "x1 clamped to 0");
    assert!((bbox[1] - 0.0).abs() < 1e-6, "y1 clamped to 0");
    assert!((bbox[2] - 100.0).abs() < 1e-6, "x2 clamped to width");
    assert!((bbox[3] - 150.0).abs() < 1e-6, "y2 clamped to height");
}

#[test]
fn test_refine_bboxes_no_change_when_inside() {
    let mut regions = vec![text_region([10.0, 20.0, 80.0, 90.0], 0.8)];
    refine_bboxes(&mut regions, 100, 100);
    let bbox = regions[0].bbox();
    assert!((bbox[0] - 10.0).abs() < 1e-6);
    assert!((bbox[1] - 20.0).abs() < 1e-6);
    assert!((bbox[2] - 80.0).abs() < 1e-6);
    assert!((bbox[3] - 90.0).abs() < 1e-6);
}

// ===========================================================================
// postprocess (full pipeline)
// ===========================================================================

#[test]
fn test_postprocess_full_pipeline() {
    let mut regions = vec![
        // Low confidence — will be filtered out.
        text_region([0.0, 0.0, 10.0, 10.0], 0.1),
        // Two overlapping text regions — will be merged.
        text_region([50.0, 50.0, 150.0, 150.0], 0.8),
        text_region([55.0, 55.0, 155.0, 155.0], 0.7),
        // Distinct header — should survive.
        header_region([0.0, 0.0, 200.0, 30.0], 0.95),
    ];
    let config = PostProcessConfig {
        min_confidence: 0.3,
        merge_iou: 0.3,
        dedup_similarity: 0.9,
        enable_model_fusion: false,
    };
    postprocess(&mut regions, &config);
    // 1 low-conf removed, 2 text merged into 1, header survives = 2 total.
    assert_eq!(
        regions.len(),
        2,
        "expected 2 regions after postprocess, got {}",
        regions.len()
    );
}

#[test]
fn test_postprocess_default_config() {
    let mut regions = vec![text_region([10.0, 10.0, 50.0, 50.0], 0.5)];
    let config = PostProcessConfig::default();
    postprocess(&mut regions, &config);
    assert_eq!(regions.len(), 1, "single valid region should survive");
}

// ===========================================================================
// PostProcessConfig defaults
// ===========================================================================

#[test]
fn test_config_defaults() {
    let cfg = PostProcessConfig::default();
    assert!((cfg.merge_iou - 0.5).abs() < 1e-6);
    assert!((cfg.dedup_similarity - 0.9).abs() < 1e-6);
    assert!((cfg.min_confidence - 0.3).abs() < 1e-6);
    assert!(cfg.enable_model_fusion);
}

// ===========================================================================
// FusionPriority
// ===========================================================================

#[test]
fn test_fusion_priority_equality() {
    assert_eq!(FusionPriority::DocLayout, FusionPriority::DocLayout);
    assert_ne!(FusionPriority::DocLayout, FusionPriority::Ocr);
}

// ===========================================================================
// rebuild_region preserves content
// ===========================================================================

#[test]
fn test_rebuild_preserves_content() {
    let region = DocumentRegion::SectionHeader {
        content: "Important".to_string(),
        bbox: [0.0, 0.0, 100.0, 100.0],
        confidence: 0.9,
    };
    let rebuilt = rebuild_region(&region, [10.0, 10.0, 90.0, 90.0], 0.95);
    match rebuilt {
        DocumentRegion::SectionHeader {
            content,
            bbox,
            confidence,
        } => {
            assert_eq!(content, "Important");
            assert!((bbox[0] - 10.0).abs() < 1e-6);
            assert!((confidence - 0.95).abs() < 1e-6);
        }
        _ => panic!("expected SectionHeader variant"),
    }
}
