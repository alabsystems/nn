// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DocLayout-YOLO model builder.

use super::*;
use nn_core::layers::vision::{nms_filter, Detection};

#[test]
fn test_config_defaults() {
    let cfg = DocLayoutYoloConfig::default();
    assert_eq!(cfg.input_channels, 3);
    assert_eq!(cfg.backbone_channels, [16, 32, 64, 128, 256]);
    assert_eq!(cfg.num_classes, NUM_CLASSES);
    assert_eq!(cfg.reg_max, REG_MAX);
    assert!((cfg.conf_threshold - 0.25).abs() < 1e-6);
    assert!((cfg.iou_threshold - 0.45).abs() < 1e-6);
}

#[test]
fn test_class_names_count() {
    assert_eq!(CLASS_NAMES.len(), NUM_CLASSES);
    assert_eq!(CLASS_NAMES.len(), 10);
}

#[test]
fn test_class_names_content() {
    assert_eq!(CLASS_NAMES[0], "caption");
    assert_eq!(CLASS_NAMES[1], "footnote");
    assert_eq!(CLASS_NAMES[2], "formula");
    assert_eq!(CLASS_NAMES[3], "list-item");
    assert_eq!(CLASS_NAMES[4], "page-footer");
    assert_eq!(CLASS_NAMES[5], "page-header");
    assert_eq!(CLASS_NAMES[6], "picture");
    assert_eq!(CLASS_NAMES[7], "section-header");
    assert_eq!(CLASS_NAMES[8], "table");
    assert_eq!(CLASS_NAMES[9], "text");
}

#[test]
fn test_backbone_channel_progression() {
    let cfg = DocLayoutYoloConfig::default();
    let c = cfg.backbone_channels;
    // Each stage doubles channels: 16 → 32 → 64 → 128 → 256
    assert_eq!(c[0], 16);
    assert_eq!(c[1], 32);
    assert_eq!(c[2], 64);
    assert_eq!(c[3], 128);
    assert_eq!(c[4], 256);
    for i in 1..c.len() {
        assert_eq!(c[i], c[i - 1] * 2, "stage {i} should be 2x stage {}", i - 1);
    }
}

#[test]
fn test_neck_channels() {
    let cfg = DocLayoutYoloConfig::default();
    // Neck takes the last 3 backbone stages (P3=64, P4=128, P5=256)
    assert_eq!(cfg.neck_channels(), [64, 128, 256]);
}

#[test]
fn test_feature_map_strides() {
    assert_eq!(STRIDES, [8, 16, 32]);
}

#[test]
fn test_feature_map_sizes_at_default_input() {
    // For INPUT_SIZE=800:
    // P3 stride 8: 800/8 = 100
    // P4 stride 16: 800/16 = 50
    // P5 stride 32: 800/32 = 25
    let sizes: Vec<usize> = STRIDES.iter().map(|s| INPUT_SIZE / s).collect();
    assert_eq!(sizes, vec![100, 50, 25]);
}

#[test]
fn test_detection_output_format() {
    let det = Detection {
        x1: 10.0,
        y1: 20.0,
        x2: 100.0,
        y2: 200.0,
        confidence: 0.95,
        class_id: 8, // "table"
    };
    assert!((det.x1 - 10.0).abs() < f32::EPSILON);
    assert!((det.y1 - 20.0).abs() < f32::EPSILON);
    assert!((det.x2 - 100.0).abs() < f32::EPSILON);
    assert!((det.y2 - 200.0).abs() < f32::EPSILON);
    assert!((det.confidence - 0.95).abs() < f32::EPSILON);
    assert_eq!(det.class_id, 8);
    assert!(det.area() > 0.0);
}

#[test]
fn test_conf_threshold_filtering_via_nms() {
    let dets = vec![
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 50.0,
            y2: 50.0,
            confidence: 0.9,
            class_id: 0,
        },
        Detection {
            x1: 60.0,
            y1: 60.0,
            x2: 110.0,
            y2: 110.0,
            confidence: 0.3,
            class_id: 1,
        },
        Detection {
            x1: 120.0,
            y1: 120.0,
            x2: 170.0,
            y2: 170.0,
            confidence: 0.1,
            class_id: 2,
        },
    ];
    // High threshold keeps fewer detections
    let kept_high = nms_filter(&dets, 0.5, 0.5).unwrap();
    let kept_low = nms_filter(&dets, 0.05, 0.5).unwrap();
    assert!(kept_high.len() <= kept_low.len());
    assert_eq!(kept_high.len(), 1); // only 0.9 survives
    assert_eq!(kept_low.len(), 3); // all survive
}

#[test]
fn test_nms_iou_threshold_suppresses_overlap() {
    // Two highly overlapping boxes of the same class
    let dets = vec![
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            confidence: 0.9,
            class_id: 0,
        },
        Detection {
            x1: 5.0,
            y1: 5.0,
            x2: 105.0,
            y2: 105.0,
            confidence: 0.8,
            class_id: 0,
        },
    ];
    // Strict IoU threshold → suppress the second
    let kept = nms_filter(&dets, 0.1, 0.3).unwrap();
    assert_eq!(kept.len(), 1);
    assert!((kept[0].confidence - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_label_detections_maps_class_ids() {
    let dets = vec![
        Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 50.0,
            y2: 50.0,
            confidence: 0.9,
            class_id: 0,
        },
        Detection {
            x1: 10.0,
            y1: 10.0,
            x2: 60.0,
            y2: 60.0,
            confidence: 0.8,
            class_id: 8,
        },
        Detection {
            x1: 20.0,
            y1: 20.0,
            x2: 70.0,
            y2: 70.0,
            confidence: 0.7,
            class_id: 9,
        },
    ];
    let labeled = DocLayoutYolo::label_detections(&dets);
    assert_eq!(labeled.len(), 3);
    assert_eq!(labeled[0].0, "caption");
    assert_eq!(labeled[1].0, "table");
    assert_eq!(labeled[2].0, "text");
}

#[test]
fn test_label_detections_skips_invalid_class() {
    let dets = vec![Detection {
        x1: 0.0,
        y1: 0.0,
        x2: 50.0,
        y2: 50.0,
        confidence: 0.9,
        class_id: 99,
    }];
    let labeled = DocLayoutYolo::label_detections(&dets);
    assert_eq!(labeled.len(), 0); // class_id 99 out of range
}

#[test]
fn test_10_class_output_range() {
    // All 10 class IDs produce valid labels
    for class_id in 0..NUM_CLASSES {
        let dets = vec![Detection {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            confidence: 0.5,
            class_id: class_id as u32,
        }];
        let labeled = DocLayoutYolo::label_detections(&dets);
        assert_eq!(
            labeled.len(),
            1,
            "class_id {class_id} should produce a label"
        );
        assert_eq!(labeled[0].0, CLASS_NAMES[class_id]);
    }
}

#[test]
fn test_config_custom_overrides() {
    let cfg = DocLayoutYoloConfig {
        input_channels: 1,
        backbone_channels: [8, 16, 32, 64, 128],
        num_classes: 5,
        reg_max: 8,
        conf_threshold: 0.5,
        iou_threshold: 0.6,
    };
    assert_eq!(cfg.input_channels, 1);
    assert_eq!(cfg.num_classes, 5);
    assert_eq!(cfg.neck_channels(), [32, 64, 128]);
}

#[test]
fn test_input_size_constant() {
    assert_eq!(INPUT_SIZE, 800);
}

#[test]
fn test_strides_are_powers_of_two() {
    for s in STRIDES {
        assert!(s.is_power_of_two(), "stride {s} should be a power of 2");
    }
}
