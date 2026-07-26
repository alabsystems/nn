// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Weighted Box Fusion (WBF).

use super::*;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

#[test]
fn test_wbf_empty_input() {
    let config = WbfConfig::default();
    let fused = WeightedBoxFusion::fuse(&[], &[], &config);
    assert!(fused.is_empty());
}

#[test]
fn test_wbf_single_model_no_overlap() {
    let dets = vec![
        ScoredBox::new(0, 0.9, [0.0, 0.0, 0.1, 0.1]),
        ScoredBox::new(1, 0.8, [0.5, 0.5, 0.9, 0.9]),
    ];
    let config = WbfConfig::default();
    let fused = WeightedBoxFusion::fuse(&[&dets], &[1.0], &config);
    // Non-overlapping boxes remain separate.
    assert_eq!(fused.len(), 2);
    assert_eq!(fused[0].class_id, 0); // Higher confidence first.
    assert_eq!(fused[1].class_id, 1);
}

#[test]
fn test_wbf_two_models_overlapping_same_class() {
    let model_a = vec![ScoredBox::new(0, 0.9, [0.1, 0.1, 0.5, 0.5])];
    let model_b = vec![ScoredBox::new(0, 0.8, [0.12, 0.09, 0.48, 0.52])];

    let config = WbfConfig {
        iou_threshold: 0.3,
        ..Default::default()
    };
    let fused = WeightedBoxFusion::fuse(&[&model_a, &model_b], &[1.0, 1.0], &config);

    // Should merge into one detection.
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].class_id, 0);
    // Fused bbox should be between the two inputs.
    let fb = &fused[0].bbox;
    assert!(fb[0] > 0.09 && fb[0] < 0.13, "x1={}", fb[0]);
    assert!(fb[2] > 0.47 && fb[2] < 0.51, "x2={}", fb[2]);
}

#[test]
fn test_wbf_different_classes_not_fused() {
    let model_a = vec![ScoredBox::new(0, 0.9, [0.1, 0.1, 0.5, 0.5])];
    let model_b = vec![ScoredBox::new(1, 0.9, [0.1, 0.1, 0.5, 0.5])];

    let config = WbfConfig {
        iou_threshold: 0.3,
        allow_cross_class: false,
        ..Default::default()
    };
    let fused = WeightedBoxFusion::fuse(&[&model_a, &model_b], &[1.0, 1.0], &config);

    // Different classes: two separate clusters.
    assert_eq!(fused.len(), 2);
}

#[test]
fn test_wbf_cross_class_allowed() {
    let model_a = vec![ScoredBox::new(0, 0.9, [0.1, 0.1, 0.5, 0.5])];
    let model_b = vec![ScoredBox::new(1, 0.8, [0.1, 0.1, 0.5, 0.5])];

    let config = WbfConfig {
        iou_threshold: 0.3,
        allow_cross_class: true,
        ..Default::default()
    };
    let fused = WeightedBoxFusion::fuse(&[&model_a, &model_b], &[1.0, 1.0], &config);

    // Cross-class allowed: merged into one (takes class of first/highest conf).
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].class_id, 0); // Class from the higher-confidence box.
}

#[test]
fn test_wbf_weighted_models() {
    // Model A has weight 2.0, Model B has weight 1.0.
    // Both see the same object; A's box should dominate.
    let model_a = vec![ScoredBox::new(0, 0.9, [0.1, 0.1, 0.5, 0.5])];
    let model_b = vec![ScoredBox::new(0, 0.9, [0.2, 0.2, 0.6, 0.6])];

    let config = WbfConfig {
        iou_threshold: 0.2,
        ..Default::default()
    };
    let fused = WeightedBoxFusion::fuse(&[&model_a, &model_b], &[2.0, 1.0], &config);

    assert_eq!(fused.len(), 1);
    // With equal confidence but weight 2:1, the fused box is (2*A + 1*B) / 3.
    // x1 = (2*0.1 + 1*0.2) / 3 ≈ 0.133
    let fb = &fused[0].bbox;
    assert!(
        approx_eq(fb[0], (2.0 * 0.1 + 0.2) / 3.0, 0.01),
        "x1={}",
        fb[0]
    );
}

#[test]
fn test_wbf_confidence_threshold() {
    let dets = vec![ScoredBox::new(0, 0.3, [0.1, 0.1, 0.5, 0.5])];

    let config = WbfConfig {
        conf_threshold: 0.5,
        ..Default::default()
    };
    let fused = WeightedBoxFusion::fuse(&[&dets], &[1.0], &config);

    // Single-model single-box cluster: conf = 0.3 * 1/1 = 0.3 < 0.5 threshold.
    assert!(fused.is_empty());
}

#[test]
fn test_wbf_pair_convenience() {
    let a = vec![ScoredBox::new(0, 0.9, [0.1, 0.1, 0.5, 0.5])];
    let b = vec![ScoredBox::new(0, 0.8, [0.12, 0.09, 0.48, 0.52])];

    let config = WbfConfig {
        iou_threshold: 0.3,
        ..Default::default()
    };
    let fused = WeightedBoxFusion::fuse_pair(&a, &b, &config);
    assert_eq!(fused.len(), 1);
}

#[test]
fn test_wbf_three_models() {
    let a = vec![ScoredBox::new(0, 0.95, [0.10, 0.10, 0.50, 0.50])];
    let b = vec![ScoredBox::new(0, 0.90, [0.11, 0.09, 0.49, 0.51])];
    let c = vec![ScoredBox::new(0, 0.85, [0.12, 0.11, 0.48, 0.49])];

    let config = WbfConfig {
        iou_threshold: 0.3,
        ..Default::default()
    };
    let fused = WeightedBoxFusion::fuse(&[&a, &b, &c], &[1.0, 1.0, 1.0], &config);
    assert_eq!(fused.len(), 1);
    // All three contribute: confidence should reflect full coverage.
    assert!(fused[0].confidence > 0.8, "conf={}", fused[0].confidence);
}

#[test]
fn test_wbf_sorted_by_confidence() {
    let dets = vec![
        ScoredBox::new(0, 0.5, [0.0, 0.0, 0.1, 0.1]),
        ScoredBox::new(1, 0.9, [0.5, 0.5, 0.9, 0.9]),
        ScoredBox::new(2, 0.7, [0.3, 0.3, 0.4, 0.4]),
    ];
    let config = WbfConfig::default();
    let fused = WeightedBoxFusion::fuse(&[&dets], &[1.0], &config);
    assert_eq!(fused.len(), 3);
    // Should be sorted descending by confidence.
    assert!(fused[0].confidence >= fused[1].confidence);
    assert!(fused[1].confidence >= fused[2].confidence);
}

#[test]
fn test_normalize_confidences_basic() {
    let mut dets = vec![
        ScoredBox::new(0, 0.2, [0.0, 0.0, 0.1, 0.1]),
        ScoredBox::new(1, 0.8, [0.5, 0.5, 0.9, 0.9]),
    ];
    normalize_confidences(&mut [&mut dets], 1.0);
    // After normalization, the relative order should be preserved.
    assert!(
        dets[1].confidence > dets[0].confidence,
        "Higher original confidence should remain higher: {} vs {}",
        dets[0].confidence,
        dets[1].confidence
    );
    // Both should be in (0, 1).
    assert!(dets[0].confidence > 0.0 && dets[0].confidence < 1.0);
    assert!(dets[1].confidence > 0.0 && dets[1].confidence < 1.0);
}

#[test]
fn test_normalize_confidences_single_value() {
    let mut dets = vec![ScoredBox::new(0, 0.5, [0.0, 0.0, 0.1, 0.1])];
    normalize_confidences(&mut [&mut dets], 1.0);
    // Single value → all normalized to 1.0.
    assert!(approx_eq(dets[0].confidence, 1.0, 1e-5));
}

#[test]
fn test_normalize_confidences_empty() {
    let mut dets: Vec<ScoredBox> = vec![];
    normalize_confidences(&mut [&mut dets], 1.0);
    assert!(dets.is_empty());
}
