// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for FireRed-OCR model builder.

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Config preset tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_preset_2b_values() {
    let cfg = FireRedOcrConfig::preset_2b();
    assert_eq!(cfg.base_config.hidden_size, 1536);
    assert_eq!(cfg.base_config.num_heads, 12);
    assert_eq!(cfg.base_config.num_kv_heads, 2);
    assert_eq!(cfg.base_config.intermediate_size, 8960);
    assert_eq!(cfg.base_config.num_layers, 28);
    assert_eq!(cfg.base_config.vocab_size, 151936);
    assert_eq!(cfg.max_output_tokens, 4096);
    assert_eq!(cfg.ocr_mode, OcrMode::FullPage);
}

#[test]
fn test_config_convenience_accessors() {
    let cfg = FireRedOcrConfig::preset_2b();
    assert_eq!(cfg.hidden_size(), 1536);
    assert_eq!(cfg.num_layers(), 28);
    assert_eq!(cfg.vocab_size(), 151936);
    assert_eq!(cfg.head_dim(), 128); // 1536 / 12
    assert_eq!(cfg.gqa_ratio(), 6); // 12 / 2
}

#[test]
fn test_config_validate_ok() {
    let cfg = FireRedOcrConfig::preset_2b();
    cfg.validate().expect("default 2B config should be valid");
}

#[test]
fn test_config_validate_zero_max_output_tokens() {
    let mut cfg = FireRedOcrConfig::preset_2b();
    cfg.max_output_tokens = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_base_error_propagates() {
    let mut cfg = FireRedOcrConfig::preset_2b();
    cfg.base_config.num_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_inherits_from_qwen3_vl_2b() {
    let firered = FireRedOcrConfig::preset_2b();
    let qwen3 = Qwen3VLConfig::preset_2b();

    // Inherited fields match (except vocab_size which FireRed overrides)
    assert_eq!(firered.base_config.hidden_size, qwen3.hidden_size);
    assert_eq!(firered.base_config.num_heads, qwen3.num_heads);
    assert_eq!(firered.base_config.num_kv_heads, qwen3.num_kv_heads);
    assert_eq!(
        firered.base_config.intermediate_size,
        qwen3.intermediate_size
    );
    assert_eq!(firered.base_config.num_layers, qwen3.num_layers);
    assert_eq!(firered.base_config.vision_hidden, qwen3.vision_hidden);
    assert_eq!(firered.base_config.vision_heads, qwen3.vision_heads);
    assert_eq!(firered.base_config.vision_layers, qwen3.vision_layers);
    assert_eq!(
        firered.base_config.vision_patch_size,
        qwen3.vision_patch_size
    );
    assert_eq!(
        firered.base_config.vision_temporal_patch,
        qwen3.vision_temporal_patch
    );
}

#[test]
fn test_config_vocab_size_differs_from_qwen3_vl() {
    let firered = FireRedOcrConfig::preset_2b();
    let qwen3 = Qwen3VLConfig::preset_2b();
    // FireRed-OCR uses 151936 vs Qwen3-VL's 152064
    assert_eq!(firered.base_config.vocab_size, 151936);
    assert_ne!(firered.base_config.vocab_size, qwen3.vocab_size);
}

// ---------------------------------------------------------------------------
// OcrMode tests
// ---------------------------------------------------------------------------

#[test]
fn test_ocr_mode_default_is_full_page() {
    assert_eq!(OcrMode::default(), OcrMode::FullPage);
}

#[test]
fn test_ocr_mode_variants_distinct() {
    assert_ne!(OcrMode::FullPage, OcrMode::RegionCrop);
    assert_ne!(OcrMode::RegionCrop, OcrMode::LineLevel);
    assert_ne!(OcrMode::FullPage, OcrMode::LineLevel);
}

#[test]
fn test_ocr_mode_debug() {
    let debug = format!("{:?}", OcrMode::FullPage);
    assert_eq!(debug, "FullPage");
    let debug = format!("{:?}", OcrMode::RegionCrop);
    assert_eq!(debug, "RegionCrop");
    let debug = format!("{:?}", OcrMode::LineLevel);
    assert_eq!(debug, "LineLevel");
}

// ---------------------------------------------------------------------------
// Forward signature / shape tests
// ---------------------------------------------------------------------------

#[test]
fn test_forward_text_only_shape() {
    let cfg = FireRedOcrConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = FireRedOcr::load(&vb, cfg.clone()).expect("model should load from zeros");

    let text_ids: Vec<usize> = (0..10).collect();
    let output = model
        .forward(None, &text_ids)
        .expect("text-only forward should succeed");
    assert_eq!(output.logits.dims(), &[1, 10, cfg.vocab_size()]);
}

#[test]
fn test_forward_with_vision_shape() {
    let cfg = FireRedOcrConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = FireRedOcr::load(&vb, cfg.clone()).expect("model should load from zeros");

    // Vision features: [1, 64, 1280]
    let vis = DynTensor::zeros(
        &[1, 64, cfg.base_config.vision_hidden],
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let text_ids: Vec<usize> = (0..10).collect();
    let output = model
        .forward(Some(&vis), &text_ids)
        .expect("vision+text forward should succeed");
    // 64 vision tokens + 10 text tokens = 74
    assert_eq!(output.logits.dims(), &[1, 74, cfg.vocab_size()]);
}

// ---------------------------------------------------------------------------
// Model accessor tests
// ---------------------------------------------------------------------------

#[test]
fn test_model_accessors() {
    let cfg = FireRedOcrConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = FireRedOcr::load(&vb, cfg).expect("model should load");

    assert_eq!(model.config().hidden_size(), 1536);
    assert_eq!(model.config().num_layers(), 28);
    assert_eq!(model.ocr_mode(), OcrMode::FullPage);
    assert_eq!(model.max_output_tokens(), 4096);
    assert_eq!(model.inner().decoder_layers().len(), 28);
}

#[test]
fn test_create_cache() {
    let cfg = FireRedOcrConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = FireRedOcr::load(&vb, cfg).expect("model should load");

    let cache = model.create_cache();
    assert_eq!(cache.num_layers(), 28);
}

// ---------------------------------------------------------------------------
// decode_ocr_tokens tests
// ---------------------------------------------------------------------------

#[test]
fn test_decode_ocr_tokens_basic() {
    let tokens = vec![10, 20, 30];
    let text = FireRedOcr::decode_ocr_tokens(&tokens, None);
    assert_eq!(text, "10,20,30");
}

#[test]
fn test_decode_ocr_tokens_with_eos() {
    let tokens = vec![10, 20, 999, 30];
    let text = FireRedOcr::decode_ocr_tokens(&tokens, Some(999));
    assert_eq!(text, "10,20");
}

#[test]
fn test_decode_ocr_tokens_empty() {
    let tokens: Vec<usize> = vec![];
    let text = FireRedOcr::decode_ocr_tokens(&tokens, None);
    assert_eq!(text, "");
}

#[test]
fn test_decode_ocr_tokens_eos_at_start() {
    let tokens = vec![999, 10, 20];
    let text = FireRedOcr::decode_ocr_tokens(&tokens, Some(999));
    assert_eq!(text, "");
}

// ---------------------------------------------------------------------------
// Debug output test
// ---------------------------------------------------------------------------

#[test]
fn test_debug_output() {
    let cfg = FireRedOcrConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = FireRedOcr::load(&vb, cfg).unwrap();
    let debug = format!("{model:?}");
    assert!(debug.contains("FireRedOcr"));
    assert!(debug.contains("FullPage"));
    assert!(debug.contains("4096"));
}
