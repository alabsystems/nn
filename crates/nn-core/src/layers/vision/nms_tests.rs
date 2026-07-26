// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::{iou, nms, Detection};

fn det(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32, class_id: u32) -> Detection {
    Detection {
        x1,
        y1,
        x2,
        y2,
        confidence,
        class_id,
    }
}

#[test]
fn test_iou_identical_boxes() {
    let a = det(0.0, 0.0, 10.0, 10.0, 0.9, 0);
    assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
}

#[test]
fn test_iou_no_overlap() {
    let a = det(0.0, 0.0, 10.0, 10.0, 0.9, 0);
    let b = det(20.0, 20.0, 30.0, 30.0, 0.8, 0);
    assert!((iou(&a, &b)).abs() < 1e-6);
}

#[test]
fn test_iou_partial_overlap() {
    let a = det(0.0, 0.0, 10.0, 10.0, 0.9, 0);
    let b = det(5.0, 5.0, 15.0, 15.0, 0.8, 0);
    // Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
    let expected = 25.0 / 175.0;
    assert!((iou(&a, &b) - expected).abs() < 1e-6);
}

#[test]
fn test_iou_degenerate_box() {
    let a = det(0.0, 0.0, 10.0, 10.0, 0.9, 0);
    let b = det(5.0, 5.0, 5.0, 5.0, 0.8, 0); // zero area
    assert!((iou(&a, &b)).abs() < 1e-6);
}

#[test]
fn test_nms_basic_suppression() {
    let dets = vec![
        det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
        det(1.0, 1.0, 11.0, 11.0, 0.8, 0), // high IoU with first, same class -> suppressed
        det(50.0, 50.0, 60.0, 60.0, 0.7, 0), // no overlap -> kept
    ];
    let result = nms(&dets, 0.5, 0.5).unwrap();
    assert_eq!(result.len(), 2);
    assert!((result[0].confidence - 0.9).abs() < 1e-6);
    assert!((result[1].confidence - 0.7).abs() < 1e-6);
}

#[test]
fn test_nms_different_classes_not_suppressed() {
    let dets = vec![
        det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
        det(1.0, 1.0, 11.0, 11.0, 0.8, 1), // high IoU but different class -> kept
    ];
    let result = nms(&dets, 0.5, 0.5).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn test_nms_confidence_filter() {
    let dets = vec![
        det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
        det(50.0, 50.0, 60.0, 60.0, 0.1, 0), // below threshold -> filtered
    ];
    let result = nms(&dets, 0.5, 0.5).unwrap();
    assert_eq!(result.len(), 1);
    assert!((result[0].confidence - 0.9).abs() < 1e-6);
}

#[test]
fn test_nms_empty_input() {
    let result = nms(&[], 0.5, 0.5).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_nms_invalid_confidence_threshold() {
    let result = nms(&[], 1.5, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_nms_invalid_iou_threshold() {
    let result = nms(&[], 0.5, -0.1);
    assert!(result.is_err());
}

#[test]
fn test_nms_ordering_preserved() {
    let dets = vec![
        det(0.0, 0.0, 10.0, 10.0, 0.3, 0),
        det(20.0, 20.0, 30.0, 30.0, 0.9, 0),
        det(40.0, 40.0, 50.0, 50.0, 0.6, 0),
    ];
    let result = nms(&dets, 0.1, 0.5).unwrap();
    assert_eq!(result.len(), 3);
    // Should be sorted by confidence descending
    assert!(result[0].confidence >= result[1].confidence);
    assert!(result[1].confidence >= result[2].confidence);
}
