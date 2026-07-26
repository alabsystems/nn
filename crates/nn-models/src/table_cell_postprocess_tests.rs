// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_cxcywh_to_xyxy_basic() {
    let bbox = cxcywh_to_xyxy(&[0.5, 0.5, 0.4, 0.2], 100.0, 200.0);
    assert!((bbox[0] - 30.0).abs() < 1e-3);
    assert!((bbox[1] - 80.0).abs() < 1e-3);
    assert!((bbox[2] - 70.0).abs() < 1e-3);
    assert!((bbox[3] - 120.0).abs() < 1e-3);
}

#[test]
fn test_cxcywh_to_xyxy_clamped() {
    // Box that would extend beyond image boundaries.
    let bbox = cxcywh_to_xyxy(&[0.0, 0.0, 0.4, 0.2], 100.0, 200.0);
    assert!((bbox[0] - 0.0).abs() < 1e-3); // clamped to 0
    assert!((bbox[1] - 0.0).abs() < 1e-3); // clamped to 0
}

#[test]
fn test_decode_boxes_multiple() {
    let boxes = vec![[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]];
    let decoded = decode_boxes(&boxes, 200.0, 200.0);
    assert_eq!(decoded.len(), 2);
    assert!((decoded[0][0] - 80.0).abs() < 1e-3); // 0.5*200 - 0.1*200
}

#[test]
fn test_decode_boxes_empty() {
    let decoded = decode_boxes(&[], 100.0, 100.0);
    assert!(decoded.is_empty());
}

#[test]
fn test_classify_logits_basic() {
    // 3 classes: logits [2.0, 0.5, -1.0]. Class 0 should win.
    let (class_id, confidence) = classify_logits(&[2.0, 0.5, -1.0]);
    assert_eq!(class_id, 0);
    assert!(confidence > 0.5);
}

#[test]
fn test_classify_logits_no_object_excluded() {
    // 3 classes where last is no-object. Logits: [0.1, 0.2, 10.0]
    // No-object has highest logit but should not be selected.
    let (class_id, _confidence) = classify_logits(&[0.1, 0.2, 10.0]);
    // class_id should be 0 or 1 (the real classes), not 2.
    assert!(class_id < 2);
}

#[test]
fn test_classify_logits_empty() {
    let (class_id, confidence) = classify_logits(&[]);
    assert_eq!(class_id, 0);
    assert!((confidence - 0.0).abs() < 1e-6);
}

#[test]
fn test_classify_logits_single_class() {
    let (class_id, confidence) = classify_logits(&[5.0]);
    assert_eq!(class_id, 0);
    assert!((confidence - 1.0).abs() < 1e-3);
}

#[test]
fn test_nms_no_suppression() {
    let dets = vec![
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            confidence: 0.9,
            class_id: 0,
        },
        Detection {
            x1: 50.0,
            y1: 50.0,
            x2: 60.0,
            y2: 60.0,
            confidence: 0.8,
            class_id: 0,
        },
    ];
    let result = nms(&dets, 0.5);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_nms_suppression() {
    let dets = vec![
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            confidence: 0.9,
            class_id: 0,
        },
        Detection {
            x1: 1.0,
            y1: 1.0,
            x2: 11.0,
            y2: 11.0,
            confidence: 0.8,
            class_id: 0,
        },
    ];
    let result = nms(&dets, 0.3);
    // High overlap -- second detection should be suppressed.
    assert_eq!(result.len(), 1);
    assert!((result[0].confidence - 0.9).abs() < 1e-6);
}

#[test]
fn test_nms_different_classes_not_suppressed() {
    let dets = vec![
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            confidence: 0.9,
            class_id: 0,
        },
        Detection {
            x1: 1.0,
            y1: 1.0,
            x2: 11.0,
            y2: 11.0,
            confidence: 0.8,
            class_id: 1,
        },
    ];
    let result = nms(&dets, 0.3);
    // Different classes should not suppress each other.
    assert_eq!(result.len(), 2);
}

#[test]
fn test_nms_empty() {
    let result = nms(&[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn test_postprocess_table_detections_basic() {
    let config = TableCellPostProcessConfig::new(100.0, 100.0);
    // 2 queries, 3 classes (row, column, no-object).
    let logits = vec![
        5.0, 0.1, -2.0, // query 0: class 0 (row)
        0.1, 5.0, -2.0, // query 1: class 1 (column)
    ];
    let boxes = vec![[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]];
    let result = postprocess_table_detections(&logits, &boxes, 3, &config);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_postprocess_table_detections_filters_no_object() {
    let config = TableCellPostProcessConfig::new(100.0, 100.0);
    // 1 query, 3 classes. No-object class has highest logit.
    let logits = vec![-5.0, -5.0, 10.0];
    let boxes = vec![[0.5, 0.5, 0.2, 0.2]];
    let result = postprocess_table_detections(&logits, &boxes, 3, &config);
    // Should be filtered out because the predicted class is no-object.
    assert!(result.is_empty());
}

#[test]
fn test_postprocess_table_detections_mismatched_lengths() {
    let config = TableCellPostProcessConfig::new(100.0, 100.0);
    let logits = vec![1.0, 2.0]; // 2 values
    let boxes = vec![[0.5, 0.5, 0.2, 0.2]]; // 1 query * 3 classes = 3, but got 2
    let result = postprocess_table_detections(&logits, &boxes, 3, &config);
    assert!(result.is_empty());
}

#[test]
fn test_clamp_detections() {
    let mut dets = vec![Detection {
        x1: -5.0,
        y1: -3.0,
        x2: 150.0,
        y2: 250.0,
        confidence: 0.9,
        class_id: 0,
    }];
    clamp_detections(&mut dets, 100.0, 200.0);
    assert!((dets[0].x1 - 0.0).abs() < 1e-6);
    assert!((dets[0].y1 - 0.0).abs() < 1e-6);
    assert!((dets[0].x2 - 100.0).abs() < 1e-6);
    assert!((dets[0].y2 - 200.0).abs() < 1e-6);
}

#[test]
fn test_compute_iou_det_no_overlap() {
    let a = Detection {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
        confidence: 0.9,
        class_id: 0,
    };
    let b = Detection {
        x1: 20.0,
        y1: 20.0,
        x2: 30.0,
        y2: 30.0,
        confidence: 0.8,
        class_id: 0,
    };
    assert!((compute_iou_det(&a, &b) - 0.0).abs() < 1e-6);
}

#[test]
fn test_compute_iou_det_full_overlap() {
    let a = Detection {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
        confidence: 0.9,
        class_id: 0,
    };
    let b = Detection {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
        confidence: 0.8,
        class_id: 0,
    };
    assert!((compute_iou_det(&a, &b) - 1.0).abs() < 1e-6);
}
