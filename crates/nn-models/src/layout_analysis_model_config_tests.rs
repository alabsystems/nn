// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Multi-scale detection config tests
// ---------------------------------------------------------------------------

#[test]
fn test_multiscale_defaults_valid() {
    let cfg = MultiScaleDetectionConfig::default();
    assert_eq!(cfg.strides, vec![8, 16, 32]);
    assert_eq!(cfg.head_channels, 256);
    assert_eq!(cfg.reg_max, 16);
    cfg.validate().expect("multiscale defaults should be valid");
}

#[test]
fn test_multiscale_empty_strides_rejected() {
    let cfg = MultiScaleDetectionConfig {
        strides: vec![],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_multiscale_zero_stride_rejected() {
    let cfg = MultiScaleDetectionConfig {
        strides: vec![0, 16, 32],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_multiscale_non_power_of_two_stride_rejected() {
    let cfg = MultiScaleDetectionConfig {
        strides: vec![6, 16, 32],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_multiscale_non_increasing_strides_rejected() {
    let cfg = MultiScaleDetectionConfig {
        strides: vec![16, 8, 32],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_multiscale_duplicate_strides_rejected() {
    let cfg = MultiScaleDetectionConfig {
        strides: vec![8, 8, 32],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_multiscale_zero_head_channels_rejected() {
    let cfg = MultiScaleDetectionConfig {
        head_channels: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_multiscale_zero_reg_max_rejected() {
    let cfg = MultiScaleDetectionConfig {
        reg_max: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_total_anchors_computation() {
    let cfg = MultiScaleDetectionConfig::default();
    // input_size=800: 800/8=100, 800/16=50, 800/32=25
    // 100*100 + 50*50 + 25*25 = 10000 + 2500 + 625 = 13125
    assert_eq!(cfg.total_anchors(800), 13125);
}

// ---------------------------------------------------------------------------
// PAN neck config tests
// ---------------------------------------------------------------------------

#[test]
fn test_pan_neck_defaults_valid() {
    let cfg = PanNeckConfig::default();
    cfg.validate().expect("PAN neck defaults should be valid");
}

#[test]
fn test_pan_neck_empty_channels_rejected() {
    let cfg = PanNeckConfig {
        backbone_channels: vec![],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_pan_neck_zero_channel_rejected() {
    let cfg = PanNeckConfig {
        backbone_channels: vec![64, 0, 256],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_pan_neck_zero_output_rejected() {
    let cfg = PanNeckConfig {
        output_channels: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_pan_neck_zero_csp_depth_rejected() {
    let cfg = PanNeckConfig {
        csp_depth: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Document preprocess config tests
// ---------------------------------------------------------------------------

#[test]
fn test_preprocess_defaults_valid() {
    let cfg = DocumentPreprocessConfig::default();
    assert_eq!(cfg.input_size, 800);
    assert!(cfg.letterbox);
    cfg.validate().expect("preprocess defaults should be valid");
}

#[test]
fn test_preprocess_zero_input_size_rejected() {
    let cfg = DocumentPreprocessConfig {
        input_size: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_preprocess_zero_std_rejected() {
    let cfg = DocumentPreprocessConfig {
        std: [0.0, 0.224, 0.225],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_preprocess_negative_std_rejected() {
    let cfg = DocumentPreprocessConfig {
        std: [0.229, -0.1, 0.225],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_preprocess_nan_mean_rejected() {
    let cfg = DocumentPreprocessConfig {
        mean: [f32::NAN, 0.456, 0.406],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_preprocess_zero_max_dimension_rejected() {
    let cfg = DocumentPreprocessConfig {
        max_dimension: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Composite layout analysis config tests
// ---------------------------------------------------------------------------

#[test]
fn test_preset_nano_valid() {
    let cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_nano();
    assert_eq!(cfg.num_classes, 10);
    assert_eq!(cfg.class_names.len(), 10);
    assert_eq!(cfg.input_channels, 3);
    cfg.validate().expect("nano preset should be valid");
}

#[test]
fn test_preset_small_valid() {
    let cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_small();
    assert_eq!(cfg.num_classes, 10);
    assert_eq!(cfg.preprocess.input_size, 1024);
    cfg.validate().expect("small preset should be valid");
}

#[test]
fn test_composite_zero_input_channels_rejected() {
    let mut cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_nano();
    cfg.input_channels = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_empty_backbone_rejected() {
    let mut cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_nano();
    cfg.backbone_channels = vec![];
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_zero_classes_rejected() {
    let mut cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_nano();
    cfg.num_classes = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_class_names_mismatch_rejected() {
    let mut cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_nano();
    cfg.class_names.push("extra".into());
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_propagates_neck_error() {
    let mut cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_nano();
    cfg.neck.output_channels = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_propagates_detection_error() {
    let mut cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_nano();
    cfg.detection.strides = vec![];
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_propagates_preprocess_error() {
    let mut cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_nano();
    cfg.preprocess.input_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_total_anchors_from_composite() {
    let cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_nano();
    // input_size=800 with strides [8, 16, 32]
    assert_eq!(cfg.total_anchors(), 13125);
}

#[test]
fn test_dfl_output_dim() {
    let cfg = LayoutAnalysisModelConfig::preset_doclayout_yolo_nano();
    // 4 * reg_max(16) = 64
    assert_eq!(cfg.dfl_output_dim(), 64);
}
