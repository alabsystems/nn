// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Field detection head tests
// ---------------------------------------------------------------------------

#[test]
fn test_field_head_funsd_preset() {
    let cfg = FieldDetectionHeadConfig::preset_funsd();
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.num_labels, 7);
    assert!(!cfg.use_crf);
    cfg.validate().expect("FUNSD preset should be valid");
}

#[test]
fn test_field_head_cord_preset() {
    let cfg = FieldDetectionHeadConfig::preset_cord();
    assert_eq!(cfg.num_labels, 30);
    cfg.validate().expect("CORD preset should be valid");
}

#[test]
fn test_field_head_zero_hidden_size_rejected() {
    let cfg = FieldDetectionHeadConfig {
        hidden_size: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_field_head_zero_labels_rejected() {
    let cfg = FieldDetectionHeadConfig {
        num_labels: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_field_head_invalid_dropout_rejected() {
    let mut cfg = FieldDetectionHeadConfig {
        classifier_dropout: -0.5,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    cfg.classifier_dropout = 1.1;
    assert!(cfg.validate().is_err());

    cfg.classifier_dropout = f32::NAN;
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Value extraction head tests
// ---------------------------------------------------------------------------

#[test]
fn test_value_head_defaults_valid() {
    let cfg = ValueExtractionHeadConfig::default();
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.biaffine_dim, 128);
    assert_eq!(cfg.max_links, 64);
    cfg.validate().expect("value head defaults should be valid");
}

#[test]
fn test_value_head_zero_hidden_rejected() {
    let cfg = ValueExtractionHeadConfig {
        hidden_size: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_value_head_biaffine_exceeds_hidden_rejected() {
    let cfg = ValueExtractionHeadConfig {
        biaffine_dim: 1024, // greater than hidden_size=768
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_value_head_zero_max_links_rejected() {
    let cfg = ValueExtractionHeadConfig {
        max_links: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_value_head_invalid_threshold_rejected() {
    let cfg = ValueExtractionHeadConfig {
        link_threshold: 1.5,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_value_head_zero_span_width_with_embedding_rejected() {
    let cfg = ValueExtractionHeadConfig {
        use_width_embedding: true,
        max_span_width: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_value_head_zero_span_width_without_embedding_ok() {
    let cfg = ValueExtractionHeadConfig {
        use_width_embedding: false,
        max_span_width: 0,
        ..Default::default()
    };
    cfg.validate()
        .expect("zero span width without embedding should be ok");
}

// ---------------------------------------------------------------------------
// Composite model config tests
// ---------------------------------------------------------------------------

#[test]
fn test_preset_funsd_valid() {
    let cfg = FormFieldModelConfig::preset_funsd();
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.num_layers, 12);
    assert_eq!(cfg.num_heads, 12);
    assert!(cfg.value_head.is_none());
    cfg.validate().expect("FUNSD preset should be valid");
}

#[test]
fn test_preset_funsd_with_linking_valid() {
    let cfg = FormFieldModelConfig::preset_funsd_with_linking();
    assert!(cfg.value_head.is_some());
    cfg.validate()
        .expect("FUNSD with linking preset should be valid");
}

#[test]
fn test_preset_cord_valid() {
    let cfg = FormFieldModelConfig::preset_cord();
    assert_eq!(cfg.field_head.num_labels, 30);
    cfg.validate().expect("CORD preset should be valid");
}

#[test]
fn test_composite_zero_hidden_rejected() {
    let mut cfg = FormFieldModelConfig::preset_funsd();
    cfg.hidden_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_hidden_not_divisible_by_heads_rejected() {
    let mut cfg = FormFieldModelConfig::preset_funsd();
    cfg.hidden_size = 100;
    cfg.num_heads = 7;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_zero_layers_rejected() {
    let mut cfg = FormFieldModelConfig::preset_funsd();
    cfg.num_layers = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_image_not_divisible_by_patch_rejected() {
    let mut cfg = FormFieldModelConfig::preset_funsd();
    cfg.image_size = 225; // not divisible by 16
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_value_head_hidden_mismatch_rejected() {
    let mut cfg = FormFieldModelConfig::preset_funsd_with_linking();
    if let Some(ref mut vh) = cfg.value_head {
        vh.hidden_size = 512; // mismatch with backbone's 768
    }
    assert!(cfg.validate().is_err());
}

#[test]
fn test_visual_seq_len_computation() {
    let cfg = FormFieldModelConfig::preset_funsd();
    // 224 / 16 = 14, 14 * 14 = 196
    assert_eq!(cfg.visual_seq_len(), 196);
}

#[test]
fn test_composite_propagates_field_head_error() {
    let mut cfg = FormFieldModelConfig::preset_funsd();
    cfg.field_head.num_labels = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_composite_propagates_value_head_error() {
    let mut cfg = FormFieldModelConfig::preset_funsd_with_linking();
    if let Some(ref mut vh) = cfg.value_head {
        vh.biaffine_dim = 0;
    }
    assert!(cfg.validate().is_err());
}
