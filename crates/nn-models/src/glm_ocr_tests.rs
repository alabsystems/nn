// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GLM-OCR 0.9B model builder.

use super::*;
use nn_core::{DType, Device};

#[test]
fn test_config_preset_900m() {
    let cfg = GlmOcrConfig::preset_900m();
    assert_eq!(cfg.hidden_size, 1536);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.num_kv_heads, 4);
    assert_eq!(cfg.intermediate_size, 4096);
    assert_eq!(cfg.num_layers, 24);
    assert_eq!(cfg.vocab_size, 65024);
    assert_eq!(cfg.vision_hidden, 768);
    assert_eq!(cfg.vision_layers, 12);
    assert_eq!(cfg.mtp_depth, 3);
    assert_eq!(cfg.image_size, 384);
    assert_eq!(cfg.patch_size, 16);
}

#[test]
fn test_num_patches_calculation() {
    let cfg = GlmOcrConfig::preset_900m();
    // 384 / 16 = 24 patches per side, 24 * 24 = 576 total
    assert_eq!(cfg.num_patches(), 576);
}

#[test]
fn test_gqa_head_config() {
    let cfg = GlmOcrConfig::preset_900m();
    // 16 Q heads, 4 KV heads -> ratio 4
    assert_eq!(cfg.gqa_ratio(), 4);
    // 1536 / 16 = 96
    assert_eq!(cfg.head_dim(), 96);
}

#[test]
fn test_config_validate_ok() {
    let cfg = GlmOcrConfig::preset_900m();
    cfg.validate().expect("default config should be valid");
}

#[test]
fn test_config_validate_zero_heads() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.num_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_zero_kv_heads() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.num_kv_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_hidden_not_divisible_by_heads() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.hidden_size = 100;
    cfg.num_heads = 16;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_heads_not_divisible_by_kv_heads() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.num_heads = 16;
    cfg.num_kv_heads = 5; // 16 % 5 != 0
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_zero_patch_size() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.patch_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_image_not_divisible() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.image_size = 385; // not divisible by 16
    assert!(cfg.validate().is_err());
}

#[test]
fn test_head_dim_zero_heads_returns_zero() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.num_heads = 0;
    assert_eq!(cfg.head_dim(), 0);
}

#[test]
fn test_gqa_ratio_zero_kv_heads_returns_zero() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.num_kv_heads = 0;
    assert_eq!(cfg.gqa_ratio(), 0);
}

#[test]
fn test_num_patches_zero_patch_size() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.patch_size = 0;
    assert_eq!(cfg.num_patches(), 0);
}

#[test]
fn test_constants_consistency() {
    let cfg = GlmOcrConfig::preset_900m();
    assert_eq!(HIDDEN, cfg.hidden_size);
    assert_eq!(NUM_HEADS, cfg.num_heads);
    assert_eq!(NUM_KV_HEADS, cfg.num_kv_heads);
    assert_eq!(INTERMEDIATE, cfg.intermediate_size);
    assert_eq!(NUM_LAYERS, cfg.num_layers);
    assert_eq!(VOCAB_SIZE, cfg.vocab_size);
    assert_eq!(VISION_HIDDEN, cfg.vision_hidden);
    assert_eq!(VISION_LAYERS, cfg.vision_layers);
    assert_eq!(MTP_DEPTH, cfg.mtp_depth);
    assert_eq!(IMAGE_SIZE, cfg.image_size);
    assert_eq!(PATCH_SIZE, cfg.patch_size);
}

#[test]
fn test_decoder_layer_shape_via_varbuilder() {
    let cfg = GlmOcrConfig::preset_900m();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

    let layer_vb = vb.pp("model").pp("layers").pp("0");
    let layer =
        GlmDecoderLayer::load(&layer_vb, &cfg).expect("decoder layer should load from zeros");

    // Forward: [1, 10, 1536] -> [1, 10, 1536]
    let input =
        DynTensor::zeros(&[1, 10, 1536], DType::F32, &Device::Cpu).expect("should create input");
    let output = layer.forward(&input, None).expect("forward should succeed");
    assert_eq!(output.dims(), &[1, 10, 1536]);
}

#[test]
fn test_swiglu_ffn_shape() {
    let cfg = GlmOcrConfig::preset_900m();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let mlp_vb = vb.pp("mlp");
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;

    let gate = Linear::new(mlp_vb.get(&[i, h], "gate_proj.weight").unwrap(), None).unwrap();
    let up = Linear::new(mlp_vb.get(&[i, h], "up_proj.weight").unwrap(), None).unwrap();
    let down = Linear::new(mlp_vb.get(&[h, i], "down_proj.weight").unwrap(), None).unwrap();

    let input = DynTensor::zeros(&[1, 5, h], DType::F32, &Device::Cpu).unwrap();
    let g = gate.forward(&input).unwrap().silu().unwrap();
    let u = up.forward(&input).unwrap();
    let result = down.forward(&g.broadcast_mul(&u).unwrap()).unwrap();
    assert_eq!(result.dims(), &[1, 5, h]);
}

#[test]
fn test_vision_output_shape() {
    let cfg = GlmOcrConfig::preset_900m();
    let siglip_cfg = cfg.to_siglip2_config().unwrap();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(vb.pp("vision_model"), &siglip_cfg)
        .expect("encoder load should succeed");

    let image = DynTensor::zeros(
        &[1, 3, cfg.image_size, cfg.image_size],
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let output = encoder
        .forward(&image, PoolingStrategy::None)
        .expect("vision forward should succeed");
    assert_eq!(output.dims(), &[1, 576, 768]);
}

#[test]
fn test_vision_projection_shape() {
    let cfg = GlmOcrConfig::preset_900m();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let proj_vb = vb.pp("vision_projection");
    let w = proj_vb
        .get(&[cfg.hidden_size, cfg.vision_hidden], "weight")
        .unwrap();
    let proj = Linear::new(w, None).unwrap();

    let input = DynTensor::zeros(&[1, 576, 768], DType::F32, &Device::Cpu).unwrap();
    let output = proj.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 576, 1536]);
}

#[test]
fn test_full_forward_shape() {
    // End-to-end: image [1, 3, 384, 384] + 10 text tokens -> logits + mtp_logits
    let cfg = GlmOcrConfig::preset_900m();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = GlmOcr::load(&vb, cfg.clone()).expect("full model should load from zeros");

    let image = DynTensor::zeros(
        &[1, 3, cfg.image_size, cfg.image_size],
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let text_ids: Vec<usize> = (0..10).collect();
    let output = model
        .forward(&image, &text_ids)
        .expect("forward should succeed");

    // 576 vision patches + 10 text tokens = 586 total sequence length
    assert_eq!(output.logits.dims(), &[1, 586, cfg.vocab_size]);

    // MTP logits: [1, 586, 3, 65024]
    let mtp = output.mtp_logits.expect("MTP logits should be present");
    assert_eq!(mtp.dims(), &[1, 586, 3, cfg.vocab_size]);
}

#[test]
fn test_full_forward_no_mtp() {
    // Model with mtp_depth=0 should have no MTP logits
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.mtp_depth = 0;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = GlmOcr::load(&vb, cfg.clone()).expect("model without MTP should load");

    let image = DynTensor::zeros(
        &[1, 3, cfg.image_size, cfg.image_size],
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let text_ids: Vec<usize> = (0..5).collect();
    let output = model
        .forward(&image, &text_ids)
        .expect("forward should succeed");

    assert_eq!(output.logits.dims(), &[1, 581, cfg.vocab_size]);
    assert!(output.mtp_logits.is_none());
}

#[test]
fn test_model_accessors() {
    let cfg = GlmOcrConfig::preset_900m();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = GlmOcr::load(&vb, cfg).expect("full model should load");

    assert_eq!(model.config().hidden_size, 1536);
    assert_eq!(model.decoder_layers().len(), 24);
    assert!(model.mtp_heads().is_some());
    assert_eq!(model.mtp_heads().unwrap().num_predict_tokens(), 3);
    // vision_encoder accessor should work
    let _ = model.vision_encoder();
}

#[test]
fn test_generate_with_mtp_requires_mtp_heads() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.mtp_depth = 0;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = GlmOcr::load(&vb, cfg.clone()).unwrap();

    let image = DynTensor::zeros(
        &[1, 3, cfg.image_size, cfg.image_size],
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let result = model.generate_with_mtp(&image, &[0, 1, 2], 10);
    assert!(result.is_err());
}

#[test]
fn test_causal_mask_shape() {
    // Verify the causal mask has the right shape for the combined sequence
    let seq_len = 586; // 576 + 10
    let mask = nn_core::layers::causal_mask_dtype(seq_len, DType::F32, &Device::Cpu)
        .expect("mask should be created");
    assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);
}

#[test]
fn test_debug_output() {
    let cfg = GlmOcrConfig::preset_900m();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = GlmOcr::load(&vb, cfg).unwrap();
    let debug = format!("{model:?}");
    assert!(debug.contains("GlmOcr"));
    assert!(debug.contains("has_mtp: true"));
    assert!(debug.contains("layers: 24"));
}
