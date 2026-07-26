// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PaddleOCR-VL-1.5 vision encoder.

use super::*;
use nn_core::{DType, Device};

#[test]
fn test_vision_config_defaults() {
    let cfg = PaddleOcrVlVisionConfig::default();
    assert_eq!(cfg.hidden_size, 1152);
    assert_eq!(cfg.intermediate_size, 4304);
    assert_eq!(cfg.num_hidden_layers, 27);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.patch_size, 14);
    assert_eq!(cfg.image_size, 384);
    assert_eq!(cfg.spatial_merge_size, 2);
    assert_eq!(cfg.merge_output_size, 1024);
}

#[test]
fn test_vision_config_validates() {
    let cfg = PaddleOcrVlVisionConfig::default();
    cfg.validate().expect("default config should be valid");
}

#[test]
fn test_vision_config_head_dim() {
    let cfg = PaddleOcrVlVisionConfig::default();
    assert_eq!(cfg.head_dim(), 72); // 1152 / 16 = 72
}

#[test]
fn test_vision_config_rejects_indivisible_heads() {
    let cfg = PaddleOcrVlVisionConfig {
        num_attention_heads: 7, // 1152 % 7 != 0
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_vision_config_rejects_zero_layers() {
    let cfg = PaddleOcrVlVisionConfig {
        num_hidden_layers: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_vision_config_rejects_zero_merge() {
    let cfg = PaddleOcrVlVisionConfig {
        spatial_merge_size: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_vision_encoder_forward_shape() {
    // [1, 3, 392, 392] -> patches: 392/14 = 28 per side
    // After merge (2x2): 14*14 = 196 merged tokens
    // Output: [1, 196, 1024]
    let cfg = PaddleOcrVlVisionConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder =
        PaddleOcrVlVisionEncoder::load(&vb, cfg).expect("vision encoder should load from zeros");

    // 392 is divisible by 28 (14*2)
    let input =
        DynTensor::zeros(&[1, 3, 392, 392], DType::F32, &Device::Cpu).expect("should create input");
    let output = encoder.forward(&input).expect("forward should succeed");
    // 392/14 = 28 patches per side, after 2x2 merge: 14*14 = 196
    assert_eq!(output.dims(), &[1, 196, 1024]);
}

#[test]
fn test_vision_encoder_rejects_wrong_channels() {
    let cfg = PaddleOcrVlVisionConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = PaddleOcrVlVisionEncoder::load(&vb, cfg).unwrap();

    let input = DynTensor::zeros(&[1, 1, 392, 392], DType::F32, &Device::Cpu).unwrap();
    assert!(encoder.forward(&input).is_err());
}

#[test]
fn test_vision_encoder_rejects_non_divisible_image() {
    let cfg = PaddleOcrVlVisionConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = PaddleOcrVlVisionEncoder::load(&vb, cfg).unwrap();

    // 400 is not divisible by 28
    let input = DynTensor::zeros(&[1, 3, 400, 400], DType::F32, &Device::Cpu).unwrap();
    assert!(encoder.forward(&input).is_err());
}

#[test]
fn test_vision_constants_consistency() {
    let cfg = PaddleOcrVlVisionConfig::default();
    assert_eq!(VISION_HIDDEN, cfg.hidden_size);
    assert_eq!(VISION_INTERMEDIATE, cfg.intermediate_size);
    assert_eq!(VISION_LAYERS, cfg.num_hidden_layers);
    assert_eq!(VISION_HEADS, cfg.num_attention_heads);
    assert_eq!(VISION_HEAD_DIM, cfg.head_dim());
    assert_eq!(VISION_PATCH_SIZE, cfg.patch_size);
    assert_eq!(VISION_IMAGE_SIZE, cfg.image_size);
    assert_eq!(SPATIAL_MERGE_SIZE, cfg.spatial_merge_size);
    assert_eq!(MERGE_OUTPUT_DIM, cfg.merge_output_size);
}

#[test]
fn test_vision_encoder_load_from_zeros() {
    let cfg = PaddleOcrVlVisionConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = PaddleOcrVlVisionEncoder::load(&vb, cfg);
    assert!(encoder.is_ok(), "should load from zero-weight VarBuilder");
}
