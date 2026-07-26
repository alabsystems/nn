// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PaddleOCR-VL-1.5 model builder.

use super::*;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Config tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_defaults() {
    let cfg = PaddleOcrVlConfig::default_vl();
    assert_eq!(cfg.vision.hidden_size, 1152);
    assert_eq!(cfg.vision.intermediate_size, 4304);
    assert_eq!(cfg.vision.num_hidden_layers, 27);
    assert_eq!(cfg.vision.num_attention_heads, 16);
    assert_eq!(cfg.vision.head_dim(), 72);
    assert_eq!(cfg.vision.patch_size, 14);
    assert_eq!(cfg.vision.image_size, 384);
    assert_eq!(cfg.vision.spatial_merge_size, 2);
    assert_eq!(cfg.vision.merge_output_size, 1024);
    assert_eq!(cfg.decoder_hidden, 1024);
    assert_eq!(cfg.decoder_intermediate, 3072);
    assert_eq!(cfg.num_decoder_layers, 18);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.num_kv_heads, 2);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.vocab_size, 103_424);
    assert_eq!(cfg.mrope_section, [16, 24, 24]);
}

#[test]
fn test_config_validates() {
    let cfg = PaddleOcrVlConfig::default_vl();
    cfg.validate().expect("default config should be valid");
}

#[test]
fn test_gqa_ratio() {
    let cfg = PaddleOcrVlConfig::default_vl();
    // 16 Q heads / 2 KV heads = 8
    assert_eq!(cfg.gqa_ratio(), 8);
}

#[test]
fn test_config_rejects_indivisible_heads() {
    let mut cfg = PaddleOcrVlConfig::default_vl();
    cfg.num_heads = 16;
    cfg.num_kv_heads = 3; // 16 % 3 != 0
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_zero_decoder_layers() {
    let mut cfg = PaddleOcrVlConfig::default_vl();
    cfg.num_decoder_layers = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_zero_vocab() {
    let mut cfg = PaddleOcrVlConfig::default_vl();
    cfg.vocab_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_zero_head_dim() {
    let mut cfg = PaddleOcrVlConfig::default_vl();
    cfg.head_dim = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_zero_decoder_hidden() {
    let mut cfg = PaddleOcrVlConfig::default_vl();
    cfg.decoder_hidden = 0;
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Constants consistency
// ---------------------------------------------------------------------------

#[test]
fn test_constants_match_config() {
    let cfg = PaddleOcrVlConfig::default_vl();
    assert_eq!(DECODER_HIDDEN, cfg.decoder_hidden);
    assert_eq!(DECODER_INTERMEDIATE, cfg.decoder_intermediate);
    assert_eq!(NUM_DECODER_LAYERS, cfg.num_decoder_layers);
    assert_eq!(NUM_HEADS, cfg.num_heads);
    assert_eq!(NUM_KV_HEADS, cfg.num_kv_heads);
    assert_eq!(HEAD_DIM, cfg.head_dim);
    assert_eq!(VOCAB_SIZE, cfg.vocab_size);
    assert_eq!(RMS_NORM_EPS, cfg.rms_norm_eps);
    assert_eq!(ROPE_THETA, cfg.rope_theta);
    assert_eq!(MAX_POSITION_EMBEDDINGS, cfg.max_position_embeddings);
    assert_eq!(MROPE_SECTION, cfg.mrope_section);
}

// ---------------------------------------------------------------------------
// MropePositionIds tests
// ---------------------------------------------------------------------------

#[test]
fn test_mrope_text_positions() {
    let pos = MropePositionIds::text(5, 3);
    assert_eq!(pos.len(), 3);
    assert!(!pos.is_empty());
    let [t, h, w] = pos.axes();
    assert_eq!(t, &[5, 6, 7]);
    assert_eq!(h, &[5, 6, 7]);
    assert_eq!(w, &[5, 6, 7]);
}

#[test]
fn test_mrope_mismatched_lengths_fails() {
    let result = MropePositionIds::new(vec![0, 1], vec![0], vec![0, 1]);
    assert!(result.is_err());
}

#[test]
fn test_mrope_max_position() {
    let pos = MropePositionIds::new(vec![0, 1], vec![5, 3], vec![2, 4]).expect("valid positions");
    assert_eq!(pos.max_position(), 5);
}

#[test]
fn test_mrope_continuation() {
    let prefill = MropePositionIds::text(0, 4);
    let cont = prefill
        .continuation_from_cache_len(4, 1)
        .expect("continuation");
    assert_eq!(cont.len(), 1);
    let [t, _h, _w] = cont.axes();
    assert_eq!(t, &[4]);
}

// ---------------------------------------------------------------------------
// Model load test (zero-weight VarBuilder)
// ---------------------------------------------------------------------------

#[test]
fn test_full_model_load_from_zeros() {
    let cfg = PaddleOcrVlConfig::default_vl();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = PaddleOcrVl::load(&vb, cfg);
    assert!(model.is_ok(), "should load from zero-weight VarBuilder");
}

// ---------------------------------------------------------------------------
// Vision encoder shape test (through full model)
// ---------------------------------------------------------------------------

#[test]
fn test_vision_encode_shape() {
    let cfg = PaddleOcrVlConfig::default_vl();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = PaddleOcrVl::load(&vb, cfg).expect("model load");

    // 392 = 14 * 28, divisible by 28 (patch_size * spatial_merge_size)
    let image =
        DynTensor::zeros(&[1, 3, 392, 392], DType::F32, &Device::Cpu).expect("create image");
    let visual = model.vision_encode(&image).expect("vision encode");
    // 392/14 = 28 patches per side, merge 2x2: 14*14 = 196
    assert_eq!(visual.dims(), &[1, 196, 1024]);
}

// ---------------------------------------------------------------------------
// Decoder shape test
// ---------------------------------------------------------------------------

#[test]
fn test_decoder_forward_shape() {
    let cfg = PaddleOcrVlConfig::default_vl();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = PaddleOcrVl::load(&vb, cfg.clone()).expect("model load");

    let input = DynTensor::zeros(&[1, 5, cfg.decoder_hidden], DType::F32, &Device::Cpu)
        .expect("create input");
    let pos = MropePositionIds::text(0, 5);
    let hidden = model
        .decoder_forward(&input, &pos, None)
        .expect("decoder forward");
    assert_eq!(hidden.dims(), &[1, 5, cfg.decoder_hidden]);
}

// ---------------------------------------------------------------------------
// LM head shape test
// ---------------------------------------------------------------------------

#[test]
fn test_lm_head_forward_shape() {
    let cfg = PaddleOcrVlConfig::default_vl();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = PaddleOcrVl::load(&vb, cfg.clone()).expect("model load");

    let hidden = DynTensor::zeros(&[1, 3, cfg.decoder_hidden], DType::F32, &Device::Cpu)
        .expect("create hidden");
    let logits = model.lm_head_forward(&hidden).expect("lm head forward");
    assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
}
