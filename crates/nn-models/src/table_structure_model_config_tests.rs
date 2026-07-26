// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Backbone config tests
// ---------------------------------------------------------------------------

#[test]
fn test_backbone_resnet18_defaults() {
    let cfg = TableBackboneConfig::resnet18();
    assert_eq!(cfg.variant, BackboneVariant::ResNet18);
    assert_eq!(cfg.output_channels, 512);
    assert_eq!(cfg.input_channels, 3);
    assert!(cfg.freeze_backbone);
    assert!(cfg.pretrained);
    cfg.validate().expect("resnet18 defaults should be valid");
}

#[test]
fn test_backbone_resnet50_defaults() {
    let cfg = TableBackboneConfig::resnet50();
    assert_eq!(cfg.variant, BackboneVariant::ResNet50);
    assert_eq!(cfg.output_channels, 2048);
    cfg.validate().expect("resnet50 defaults should be valid");
}

#[test]
fn test_backbone_zero_input_channels_rejected() {
    let cfg = TableBackboneConfig {
        input_channels: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_backbone_zero_output_channels_rejected() {
    let cfg = TableBackboneConfig {
        output_channels: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_backbone_zero_dilation_rejected() {
    let mut cfg = TableBackboneConfig {
        dilation_rates: [0, 1],
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    cfg.dilation_rates = [1, 0];
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Decoder config tests
// ---------------------------------------------------------------------------

#[test]
fn test_decoder_defaults_valid() {
    let cfg = TableDecoderConfig::default();
    assert_eq!(cfg.hidden_dim, 256);
    assert_eq!(cfg.num_heads, 8);
    assert_eq!(cfg.num_layers, 6);
    assert_eq!(cfg.num_queries, 125);
    cfg.validate().expect("decoder defaults should be valid");
}

#[test]
fn test_decoder_hidden_dim_not_divisible_by_heads_rejected() {
    let cfg = TableDecoderConfig {
        hidden_dim: 100,
        num_heads: 7,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_decoder_zero_layers_rejected() {
    let cfg = TableDecoderConfig {
        num_layers: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_decoder_zero_queries_rejected() {
    let cfg = TableDecoderConfig {
        num_queries: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_decoder_invalid_dropout_rejected() {
    let mut cfg = TableDecoderConfig {
        dropout: 1.5,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    cfg.dropout = -0.1;
    assert!(cfg.validate().is_err());

    cfg.dropout = f32::NAN;
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Post-processing config tests
// ---------------------------------------------------------------------------

#[test]
fn test_postprocess_defaults_valid() {
    let cfg = TablePostProcessConfig::default();
    cfg.validate()
        .expect("postprocess defaults should be valid");
}

#[test]
fn test_postprocess_invalid_confidence_rejected() {
    let cfg = TablePostProcessConfig {
        table_confidence: 2.0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_postprocess_invalid_structure_confidence_rejected() {
    let cfg = TablePostProcessConfig {
        structure_confidence: -0.1,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_postprocess_invalid_nms_iou_rejected() {
    let cfg = TablePostProcessConfig {
        nms_iou_threshold: 1.5,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_postprocess_zero_max_detections_rejected() {
    let cfg = TablePostProcessConfig {
        max_detections: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Composite config tests
// ---------------------------------------------------------------------------

#[test]
fn test_preset_detection_valid() {
    let cfg = TableStructureModelConfig::preset_detection();
    assert_eq!(cfg.num_classes, 2);
    assert_eq!(cfg.input_size, 800);
    cfg.validate().expect("detection preset should be valid");
}

#[test]
fn test_preset_structure_valid() {
    let cfg = TableStructureModelConfig::preset_structure();
    assert_eq!(cfg.num_classes, 6);
    cfg.validate().expect("structure preset should be valid");
}

#[test]
fn test_composite_zero_classes_rejected() {
    let mut cfg = TableStructureModelConfig::preset_detection();
    cfg.num_classes = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_zero_input_size_rejected() {
    let mut cfg = TableStructureModelConfig::preset_detection();
    cfg.input_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_feature_sequence_length_computation() {
    let cfg = TableStructureModelConfig::preset_detection();
    // 800 / 32 = 25, 25 * 25 = 625
    assert_eq!(cfg.feature_sequence_length(), 625);
}

#[test]
fn test_composite_propagates_backbone_error() {
    let mut cfg = TableStructureModelConfig::preset_detection();
    cfg.backbone.input_channels = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_propagates_decoder_error() {
    let mut cfg = TableStructureModelConfig::preset_detection();
    cfg.decoder.num_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_propagates_postprocess_error() {
    let mut cfg = TableStructureModelConfig::preset_detection();
    cfg.postprocess.max_detections = 0;
    assert!(cfg.validate().is_err());
}
