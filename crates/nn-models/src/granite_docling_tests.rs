// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Granite-Docling-258M model builder.

use super::*;
use nn_core::{DType, Device};

#[test]
fn test_config_defaults() {
    let cfg = GraniteDoclingConfig::default_258m();
    assert_eq!(cfg.image_size, 512);
    assert_eq!(cfg.patch_size, 16);
    assert_eq!(cfg.vision_hidden, 768);
    assert_eq!(cfg.vision_heads, 12);
    assert_eq!(cfg.vision_layers, 12);
    assert_eq!(cfg.decoder_hidden, 768);
    assert_eq!(cfg.decoder_heads, 12);
    assert_eq!(cfg.decoder_kv_heads, 4);
    assert_eq!(cfg.decoder_intermediate, 2048);
    assert_eq!(cfg.decoder_layers, 12);
    assert_eq!(cfg.vocab_size, 49152);
}

#[test]
fn test_num_patches_calculation() {
    let cfg = GraniteDoclingConfig::default_258m();
    // 512 / 16 = 32 patches per side, 32 * 32 = 1024 total
    assert_eq!(cfg.num_patches(), 1024);
    assert_eq!(NUM_PATCHES, 1024);
}

#[test]
fn test_gqa_head_config() {
    let cfg = GraniteDoclingConfig::default_258m();
    // 12 Q heads, 4 KV heads → ratio 3 (each KV head serves 3 Q heads)
    assert_eq!(cfg.decoder_heads, 12);
    assert_eq!(cfg.decoder_kv_heads, 4);
    assert_eq!(cfg.decoder_heads / cfg.decoder_kv_heads, 3);
    assert_eq!(cfg.head_dim(), 64); // 768 / 12 = 64
}

#[test]
fn test_config_validate_ok() {
    let cfg = GraniteDoclingConfig::default_258m();
    cfg.validate().expect("default config should be valid");
}

#[test]
fn test_config_validate_bad_patch() {
    let mut cfg = GraniteDoclingConfig::default_258m();
    cfg.patch_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_image_not_divisible() {
    let mut cfg = GraniteDoclingConfig::default_258m();
    cfg.image_size = 500; // not divisible by 16
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_hidden_not_divisible_by_heads() {
    let mut cfg = GraniteDoclingConfig::default_258m();
    cfg.decoder_hidden = 100;
    cfg.decoder_heads = 12;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_heads_not_divisible_by_kv_heads() {
    let mut cfg = GraniteDoclingConfig::default_258m();
    cfg.decoder_heads = 12;
    cfg.decoder_kv_heads = 5; // 12 % 5 != 0
    assert!(cfg.validate().is_err());
}

#[test]
fn test_decoder_layer_shape_via_varbuilder() {
    // Build a single decoder layer with zero weights and verify shapes propagate.
    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

    // Construct a layer under "model.layers.0"
    let layer_vb = vb.pp("model").pp("layers").pp("0");
    let layer =
        GraniteDecoderLayer::load(&layer_vb, &cfg).expect("decoder layer should load from zeros");

    // Forward: [1, 10, 768] → [1, 10, 768]
    let input =
        DynTensor::zeros(&[1, 10, 768], DType::F32, &Device::Cpu).expect("should create input");
    let output = layer.forward(&input, None).expect("forward should succeed");
    assert_eq!(output.dims(), &[1, 10, 768]);
}

#[test]
fn test_swiglu_ffn_shape() {
    // Verify SwiGLU MLP preserves dimensions: [B, S, 768] → [B, S, 768].
    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let mlp_vb = vb.pp("mlp");
    let h = cfg.decoder_hidden;
    let i = cfg.decoder_intermediate;

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
    // SigLIP2 encoder: [1, 3, 512, 512] → [1, 1024, 768]
    let cfg = GraniteDoclingConfig::default_258m();
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
    assert_eq!(output.dims(), &[1, 1024, 768]);
}

#[test]
fn test_projection_shape() {
    // Vision projection: [1, 1024, 768] → [1, 1024, 768] (same dims for 258M)
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let proj_vb = vb.pp("multi_modal_projector").pp("linear");
    let w = proj_vb.get(&[768, 768], "weight").unwrap();
    let proj = Linear::new(w, None).unwrap();

    let input = DynTensor::zeros(&[1, 1024, 768], DType::F32, &Device::Cpu).unwrap();
    let output = proj.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 1024, 768]);
}

#[test]
fn test_full_forward_shape() {
    // End-to-end: image [1, 3, 512, 512] + 10 text tokens → [1, 1034, 49152] logits
    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = GraniteDocling::load(&vb, cfg.clone()).expect("full model should load from zeros");

    let image = DynTensor::zeros(
        &[1, 3, cfg.image_size, cfg.image_size],
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let text_ids: Vec<usize> = (0..10).collect();
    let logits = model
        .forward(&image, &text_ids)
        .expect("forward should succeed");
    // 1024 vision patches + 10 text tokens = 1034 total sequence length
    assert_eq!(logits.dims(), &[1, 1034, cfg.vocab_size]);
}

#[test]
fn test_causal_attention_mask_shape() {
    // Verify the causal mask has the right shape for the combined sequence
    let seq_len = 1034; // 1024 + 10
    let mask = nn_core::layers::causal_mask_dtype(seq_len, DType::F32, &Device::Cpu)
        .expect("mask should be created");
    assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);
}

#[test]
fn test_constants_consistency() {
    // Verify module-level constants match default config.
    let cfg = GraniteDoclingConfig::default_258m();
    assert_eq!(IMAGE_SIZE, cfg.image_size);
    assert_eq!(PATCH_SIZE, cfg.patch_size);
    assert_eq!(NUM_PATCHES, cfg.num_patches());
    assert_eq!(VISION_HIDDEN, cfg.vision_hidden);
    assert_eq!(VISION_HEADS, cfg.vision_heads);
    assert_eq!(VISION_LAYERS, cfg.vision_layers);
    assert_eq!(DECODER_HIDDEN, cfg.decoder_hidden);
    assert_eq!(DECODER_HEADS, cfg.decoder_heads);
    assert_eq!(DECODER_KV_HEADS, cfg.decoder_kv_heads);
    assert_eq!(DECODER_INTERMEDIATE, cfg.decoder_intermediate);
    assert_eq!(DECODER_LAYERS, cfg.decoder_layers);
    assert_eq!(VOCAB_SIZE, cfg.vocab_size);
}
