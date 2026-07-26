// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Table Transformer (DETR) model builder.

use super::*;
use nn_core::DType;

// ---------------------------------------------------------------------------
// Configuration tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_preset_detection() {
    let cfg = TableTransformerConfig::preset_detection();
    assert_eq!(cfg.num_classes, 2);
    assert_eq!(cfg.hidden_dim, 256);
    assert_eq!(cfg.num_heads, 8);
    assert_eq!(cfg.num_encoder_layers, 6);
    assert_eq!(cfg.num_decoder_layers, 6);
    assert_eq!(cfg.num_queries, 125);
    assert_eq!(cfg.ffn_dim, 2048);
    cfg.validate().expect("detection preset should be valid");
}

#[test]
fn test_config_preset_structure() {
    let cfg = TableTransformerConfig::preset_structure();
    assert_eq!(cfg.num_classes, 6);
    assert_eq!(cfg.hidden_dim, 256);
    cfg.validate().expect("structure preset should be valid");
}

#[test]
fn test_config_validate_zero_hidden_dim() {
    let mut cfg = TableTransformerConfig::preset_detection();
    cfg.hidden_dim = 0;
    assert!(cfg.validate().is_err(), "zero hidden_dim should fail");
}

#[test]
fn test_config_validate_non_divisible_heads() {
    let mut cfg = TableTransformerConfig::preset_detection();
    cfg.num_heads = 7; // 256 % 7 != 0
    assert!(
        cfg.validate().is_err(),
        "hidden_dim not divisible by num_heads should fail"
    );
}

#[test]
fn test_config_validate_zero_queries() {
    let mut cfg = TableTransformerConfig::preset_detection();
    cfg.num_queries = 0;
    assert!(cfg.validate().is_err(), "zero num_queries should fail");
}

// ---------------------------------------------------------------------------
// Positional encoding tests
// ---------------------------------------------------------------------------

#[test]
fn test_sinusoidal_2d_pos_encoding_shape() {
    let pe =
        sinusoidal_2d_pos_encoding(4, 6, 256, &Device::Cpu).expect("pos encoding should succeed");
    assert_eq!(pe.dims(), &[24, 256]); // 4*6=24 positions, 256 dims
}

#[test]
fn test_sinusoidal_2d_pos_encoding_bounded() {
    let pe =
        sinusoidal_2d_pos_encoding(3, 3, 64, &Device::Cpu).expect("pos encoding should succeed");
    let vals = pe.to_flat_vec::<f32>().expect("should convert to vec");
    for v in &vals {
        assert!(
            v.abs() <= 1.0 + 1e-6,
            "sin/cos values should be in [-1, 1], got {v}"
        );
        assert!(v.is_finite(), "pos encoding values must be finite");
    }
}

#[test]
fn test_sinusoidal_2d_pos_encoding_1x1() {
    let pe = sinusoidal_2d_pos_encoding(1, 1, 8, &Device::Cpu).expect("1x1 should succeed");
    assert_eq!(pe.dims(), &[1, 8]);
    // All sin(0)=0, cos(0)=1 for position 0
    let vals = pe.to_flat_vec::<f32>().expect("should convert to vec");
    // First element: sin(0)=0
    assert!(vals[0].abs() < 1e-6, "sin(0) should be 0, got {}", vals[0]);
}

// ---------------------------------------------------------------------------
// Class name tests
// ---------------------------------------------------------------------------

#[test]
fn test_detection_class_names() {
    assert_eq!(DETECTION_CLASSES.len(), 2);
    assert_eq!(DETECTION_CLASSES[0], "table");
    assert_eq!(DETECTION_CLASSES[1], "no-object");
}

#[test]
fn test_structure_class_names() {
    assert_eq!(STRUCTURE_CLASSES.len(), 6);
    assert_eq!(STRUCTURE_CLASSES[0], "table");
    assert_eq!(STRUCTURE_CLASSES[1], "row");
    assert_eq!(STRUCTURE_CLASSES[2], "column");
    assert_eq!(STRUCTURE_CLASSES[3], "spanning-cell");
    assert_eq!(STRUCTURE_CLASSES[4], "projected-row-header");
    assert_eq!(STRUCTURE_CLASSES[5], "no-object");
}

// ---------------------------------------------------------------------------
// Constant tests
// ---------------------------------------------------------------------------

#[test]
fn test_constants_consistency() {
    assert_eq!(HIDDEN_DIM, 256);
    assert_eq!(NUM_HEADS, 8);
    assert_eq!(
        HIDDEN_DIM % NUM_HEADS,
        0,
        "hidden_dim must be divisible by num_heads"
    );
    assert_eq!(NUM_ENCODER_LAYERS, 6);
    assert_eq!(NUM_DECODER_LAYERS, 6);
    assert_eq!(NUM_QUERIES, 125);
    assert_eq!(FFN_DIM, 2048);
    assert_eq!(BACKBONE_OUT_CHANNELS, 512);
}

// ---------------------------------------------------------------------------
// Output struct tests
// ---------------------------------------------------------------------------

#[test]
fn test_output_struct_fields() {
    let logits = DynTensor::zeros(&[1, 125, 3], DType::F32, &Device::Cpu).expect("logits tensor");
    let boxes = DynTensor::zeros(&[1, 125, 4], DType::F32, &Device::Cpu).expect("boxes tensor");
    let output = TableTransformerOutput { logits, boxes };
    assert_eq!(output.logits.dims(), &[1, 125, 3]);
    assert_eq!(output.boxes.dims(), &[1, 125, 4]);
}

#[test]
fn test_output_batch_dimension() {
    let b = 4;
    let nq = 100;
    let logits = DynTensor::zeros(&[b, nq, 7], DType::F32, &Device::Cpu).expect("logits tensor");
    let boxes = DynTensor::zeros(&[b, nq, 4], DType::F32, &Device::Cpu).expect("boxes tensor");
    let output = TableTransformerOutput { logits, boxes };
    assert_eq!(output.logits.dim(0).unwrap(), b);
    assert_eq!(output.boxes.dim(0).unwrap(), b);
}
