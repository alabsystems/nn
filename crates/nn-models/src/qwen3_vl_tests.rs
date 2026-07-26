// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Qwen3-VL multimodal model builder.

use super::*;
use nn_core::layers::Linear;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Config preset tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_2b_defaults() {
    let cfg = Qwen3VLConfig::preset_2b();
    assert_eq!(cfg.hidden_size, 1536);
    assert_eq!(cfg.num_heads, 12);
    assert_eq!(cfg.num_kv_heads, 2);
    assert_eq!(cfg.intermediate_size, 8960);
    assert_eq!(cfg.num_layers, 28);
    assert_eq!(cfg.vocab_size, 152064);
    assert_eq!(cfg.vision_hidden, 1280);
    assert_eq!(cfg.vision_heads, 16);
    assert_eq!(cfg.vision_layers, 32);
    assert_eq!(cfg.vision_patch_size, 14);
    assert_eq!(cfg.vision_temporal_patch, 2);
    assert!(!cfg.is_moe());
}

#[test]
fn test_config_7b_defaults() {
    let cfg = Qwen3VLConfig::preset_7b();
    assert_eq!(cfg.hidden_size, 3584);
    assert_eq!(cfg.num_heads, 28);
    assert_eq!(cfg.num_kv_heads, 4);
    assert_eq!(cfg.intermediate_size, 18944);
    assert_eq!(cfg.num_layers, 28);
    assert_eq!(cfg.vocab_size, 152064);
    assert!(!cfg.is_moe());
    cfg.validate().expect("7B config should be valid");
}

#[test]
fn test_config_8b_defaults() {
    let cfg = Qwen3VLConfig::preset_8b();
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_heads, 32);
    assert_eq!(cfg.num_kv_heads, 8);
    assert_eq!(cfg.intermediate_size, 11008);
    assert_eq!(cfg.num_layers, 32);
    assert_eq!(cfg.vocab_size, 152064);
    assert_eq!(cfg.vision_hidden, 1152);
    assert_eq!(cfg.vision_heads, 16);
    assert_eq!(cfg.vision_layers, 27);
    assert_eq!(cfg.vision_patch_size, 16);
    assert_eq!(cfg.vision_temporal_patch, 2);
    assert!(!cfg.is_moe());
    cfg.validate().expect("8B config should be valid");
}

#[test]
fn test_config_8b_gqa_ratio() {
    let cfg = Qwen3VLConfig::preset_8b();
    assert_eq!(cfg.gqa_ratio(), 4); // 32 / 8
    assert_eq!(cfg.head_dim(), 128); // 4096 / 32
}

#[test]
fn test_config_30b_moe() {
    let cfg = Qwen3VLConfig::preset_30b_a3b();
    assert!(cfg.is_moe());
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.active_experts, 8);
    assert_eq!(cfg.num_layers, 48);
    assert_eq!(cfg.intermediate_size, 2560);
    cfg.validate().expect("30B-A3B config should be valid");
}

#[test]
fn test_vision_patch_calculation() {
    let cfg = Qwen3VLConfig::preset_2b();
    // Temporal patch = 2, spatial patch = 14x14
    assert_eq!(cfg.vision_temporal_patch, 2);
    assert_eq!(cfg.vision_patch_size, 14);
    // Conv3d kernel volume: 2 * 14 * 14 = 392
    let kernel_volume = cfg.vision_temporal_patch * cfg.vision_patch_size * cfg.vision_patch_size;
    assert_eq!(kernel_volume, 392);
}

#[test]
fn test_vision_8b_patch_calculation() {
    let cfg = Qwen3VLConfig::preset_8b();
    assert_eq!(cfg.vision_temporal_patch, 2);
    assert_eq!(cfg.vision_patch_size, 16);
    // Conv3d kernel volume: 2 * 16 * 16 = 512
    let kernel_volume = cfg.vision_temporal_patch * cfg.vision_patch_size * cfg.vision_patch_size;
    assert_eq!(kernel_volume, 512);
}

#[test]
fn test_gqa_ratio() {
    let cfg = Qwen3VLConfig::preset_2b();
    // 12 Q heads / 2 KV heads = GQA ratio of 6
    assert_eq!(cfg.gqa_ratio(), 6);
    assert_eq!(cfg.head_dim(), 128); // 1536 / 12

    let cfg_7b = Qwen3VLConfig::preset_7b();
    assert_eq!(cfg_7b.gqa_ratio(), 7); // 28 / 4
    assert_eq!(cfg_7b.head_dim(), 128); // 3584 / 28
}

#[test]
fn test_swiglu_expansion() {
    let cfg = Qwen3VLConfig::preset_2b();
    // SwiGLU: 1536 -> 8960 -> 1536
    assert_eq!(cfg.hidden_size, 1536);
    assert_eq!(cfg.intermediate_size, 8960);
    let ratio = cfg.intermediate_size as f64 / cfg.hidden_size as f64;
    // SwiGLU effective expansion should be in [4, 8] range
    assert!((4.0..=8.0).contains(&ratio), "expansion ratio: {ratio:.2}");
}

#[test]
fn test_vocab_size() {
    // All Qwen3-VL variants share the same vocab size
    assert_eq!(Qwen3VLConfig::preset_2b().vocab_size, 152064);
    assert_eq!(Qwen3VLConfig::preset_7b().vocab_size, 152064);
    assert_eq!(Qwen3VLConfig::preset_8b().vocab_size, 152064);
    assert_eq!(Qwen3VLConfig::preset_30b_a3b().vocab_size, 152064);
}

#[test]
fn test_vision_merger_shape() {
    // Vision hidden (1280) -> decoder hidden (1536) for 2B
    let cfg = Qwen3VLConfig::preset_2b();
    assert_eq!(cfg.vision_hidden, 1280);
    assert_eq!(cfg.hidden_size, 1536);
    assert_ne!(cfg.vision_hidden, cfg.hidden_size);

    // Build merger with zero weights and verify shape
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let w = vb
        .get(&[cfg.hidden_size, cfg.vision_hidden], "weight")
        .unwrap();
    let merger = Linear::new(w, None).unwrap();

    let input = DynTensor::zeros(&[1, 64, cfg.vision_hidden], DType::F32, &Device::Cpu).unwrap();
    let output = merger.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 64, cfg.hidden_size]);
}

#[test]
fn test_num_layers() {
    assert_eq!(Qwen3VLConfig::preset_2b().num_layers, 28);
    assert_eq!(Qwen3VLConfig::preset_7b().num_layers, 28);
    assert_eq!(Qwen3VLConfig::preset_8b().num_layers, 32);
    assert_eq!(Qwen3VLConfig::preset_30b_a3b().num_layers, 48);
}

#[test]
fn test_decoder_layer_shape() {
    let cfg = Qwen3VLConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let layer_vb = vb.pp("model").pp("layers").pp("0");
    let layer =
        Qwen3DecoderLayer::load(&layer_vb, &cfg).expect("decoder layer should load from zeros");

    // Forward: [1, 10, 1536] -> [1, 10, 1536]
    let input = DynTensor::zeros(&[1, 10, cfg.hidden_size], DType::F32, &Device::Cpu)
        .expect("should create input");
    let output = layer.forward(&input, None).expect("forward should succeed");
    assert_eq!(output.dims(), &[1, 10, cfg.hidden_size]);
}

#[test]
fn test_forward_text_only_shape() {
    let cfg = Qwen3VLConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3VL::load(&vb, cfg.clone()).expect("model should load from zeros");

    assert!(!model.has_vision_encoder());

    // Text-only: 10 tokens -> [1, 10, vocab_size]
    let text_ids: Vec<usize> = (0..10).collect();
    let logits = model
        .forward(None, &text_ids)
        .expect("text-only forward should succeed");
    assert_eq!(logits.dims(), &[1, 10, cfg.vocab_size]);
}

#[test]
fn test_forward_with_vision_shape() {
    let cfg = Qwen3VLConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3VL::load(&vb, cfg.clone()).expect("model should load from zeros");

    // Vision features: [1, 64, 1280] (pre-encoded)
    let vis = DynTensor::zeros(&[1, 64, cfg.vision_hidden], DType::F32, &Device::Cpu).unwrap();
    let text_ids: Vec<usize> = (0..10).collect();
    let logits = model
        .forward(Some(&vis), &text_ids)
        .expect("vision+text forward should succeed");
    // 64 vision tokens + 10 text tokens = 74 total
    assert_eq!(logits.dims(), &[1, 74, cfg.vocab_size]);
}

// ---------------------------------------------------------------------------
// Validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_validate_ok() {
    Qwen3VLConfig::preset_2b()
        .validate()
        .expect("2B config should validate");
}

#[test]
fn test_config_validate_bad_heads() {
    let mut cfg = Qwen3VLConfig::preset_2b();
    cfg.num_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_hidden_not_divisible() {
    let mut cfg = Qwen3VLConfig::preset_2b();
    cfg.hidden_size = 100;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_moe_no_active() {
    let mut cfg = Qwen3VLConfig::preset_30b_a3b();
    cfg.active_experts = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_moe_active_exceeds_total() {
    let mut cfg = Qwen3VLConfig::preset_30b_a3b();
    cfg.active_experts = 200;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_vision_shared_across_2b_7b_30b() {
    // 2B/7B/30B share the same vision encoder config (SigLIP2-style)
    let c2 = Qwen3VLConfig::preset_2b();
    let c7 = Qwen3VLConfig::preset_7b();
    let c30 = Qwen3VLConfig::preset_30b_a3b();
    assert_eq!(c2.vision_hidden, c7.vision_hidden);
    assert_eq!(c7.vision_hidden, c30.vision_hidden);
    assert_eq!(c2.vision_heads, c7.vision_heads);
    assert_eq!(c2.vision_layers, c7.vision_layers);
    assert_eq!(c2.vision_patch_size, c7.vision_patch_size);
}

#[test]
fn test_8b_vision_different_from_2b() {
    // 8B has a different vision encoder (ViT-1152) vs 2B (ViT-1280)
    let c2 = Qwen3VLConfig::preset_2b();
    let c8 = Qwen3VLConfig::preset_8b();
    assert_ne!(c2.vision_hidden, c8.vision_hidden);
    assert_ne!(c2.vision_patch_size, c8.vision_patch_size);
    assert_ne!(c2.vision_layers, c8.vision_layers);
    assert_eq!(c8.vision_hidden, 1152);
    assert_eq!(c8.vision_patch_size, 16);
    assert_eq!(c8.vision_layers, 27);
}

// ---------------------------------------------------------------------------
// Vision encoder tests (8B)
// ---------------------------------------------------------------------------

#[test]
fn test_vision_encoder_load_from_zeros() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = Qwen3VlVisionEncoder::load(&vb).expect("vision encoder should load from zeros");
    let debug_str = format!("{encoder:?}");
    assert!(debug_str.contains("Qwen3VlVisionEncoder"));
    assert!(debug_str.contains("num_layers"));
}

#[test]
fn test_vision_encoder_shape_64x64() {
    // 64x64 image with patch_size=16 -> 4x4 grid = 16 tokens
    // After 2x2 merge -> 2x2 = 4 merged tokens
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = Qwen3VlVisionEncoder::load(&vb).expect("vision encoder should load from zeros");

    let image =
        DynTensor::zeros(&[1, 3, 64, 64], DType::F32, &Device::Cpu).expect("should create image");
    let output = encoder
        .forward(&image)
        .expect("vision encoder forward should succeed");

    // Main features: [1, 4, 4096]
    assert_eq!(output.features.dims(), &[1, 4, 4096]);

    // DeepStack: 3 tensors, each [1, 4, 4096]
    assert_eq!(output.deepstack_features.len(), 3);
    for ds in &output.deepstack_features {
        assert_eq!(ds.dims(), &[1, 4, 4096]);
    }
}

#[test]
fn test_vision_encoder_rejects_non_divisible_image() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = Qwen3VlVisionEncoder::load(&vb).expect("vision encoder should load from zeros");

    // 50x50 is not divisible by 32 (patch_size * spatial_merge_size)
    let image =
        DynTensor::zeros(&[1, 3, 50, 50], DType::F32, &Device::Cpu).expect("should create image");
    assert!(encoder.forward(&image).is_err());
}

#[test]
fn test_vision_encoder_rejects_wrong_channels() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = Qwen3VlVisionEncoder::load(&vb).expect("vision encoder should load from zeros");

    // 4 channels instead of 3
    let image =
        DynTensor::zeros(&[1, 4, 64, 64], DType::F32, &Device::Cpu).expect("should create image");
    assert!(encoder.forward(&image).is_err());
}

#[test]
fn test_deepstack_index_validation() {
    // DeepStack indexes [8, 16, 24] must be within [0, 27)
    for &idx in &vision::DEEPSTACK_INDEXES {
        assert!(
            idx < vision::VISION_NUM_LAYERS,
            "DeepStack index {idx} out of range for {} layers",
            vision::VISION_NUM_LAYERS
        );
    }
}

#[test]
fn test_deepstack_inject_layers_count() {
    assert_eq!(DEEPSTACK_INJECT_LAYERS, 3);
    // Must fit within 8B decoder depth
    let cfg = Qwen3VLConfig::preset_8b();
    assert!(DEEPSTACK_INJECT_LAYERS <= cfg.num_layers);
}

#[test]
fn test_patch_merger_shape() {
    use vision::PatchMerger;

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

    // Main merger (pre-shuffle norm on hidden_size=1152)
    let main_merger = PatchMerger::load(&vb.pp("main"), false).expect("main merger should load");
    let input = DynTensor::zeros(&[1, 16, 1152], DType::F32, &Device::Cpu).unwrap();
    let output = main_merger
        .forward(&input, 4, 4)
        .expect("main merger forward should succeed");
    // 4x4 grid -> 2x2 merged = 4 tokens, output dim 4096
    assert_eq!(output.dims(), &[1, 4, 4096]);

    // DeepStack merger (post-shuffle norm on merged_dim=4608)
    let ds_merger = PatchMerger::load(&vb.pp("ds"), true).expect("deepstack merger should load");
    let output = ds_merger
        .forward(&input, 4, 4)
        .expect("deepstack merger forward should succeed");
    assert_eq!(output.dims(), &[1, 4, 4096]);
}

#[test]
fn test_vision_rope_construction() {
    use vision::VisionRoPE;

    // 4x4 grid with merge_size 2 -> 16 patch tokens
    let rope = VisionRoPE::new(4, 4, &Device::Cpu).expect("should build VisionRoPE for 4x4 grid");
    let cos_dims = rope.cos.dims();
    assert_eq!(cos_dims[0], 1); // batch
    assert_eq!(cos_dims[1], 1); // heads
    assert_eq!(cos_dims[2], 16); // seq_len = 4*4
    assert_eq!(cos_dims[3], 36); // rope_dim = head_dim/2 = 72/2
}

#[test]
fn test_model_has_no_vision_encoder_by_default() {
    let cfg = Qwen3VLConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3VL::load(&vb, cfg).expect("model should load");
    assert!(!model.has_vision_encoder());
    assert!(model.vision_encoder().is_none());
}

#[test]
fn test_vision_encode_without_encoder_errors() {
    let cfg = Qwen3VLConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3VL::load(&vb, cfg).expect("model should load");

    let image = DynTensor::zeros(&[1, 3, 64, 64], DType::F32, &Device::Cpu).unwrap();
    assert!(model.vision_encode(&image).is_err());
}

#[test]
fn test_forward_with_deepstack_shape() {
    let cfg = Qwen3VLConfig::preset_2b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3VL::load(&vb, cfg.clone()).expect("model should load");

    // Vision features: [1, 64, 1536] (already projected to decoder hidden)
    let vis = DynTensor::zeros(&[1, 64, cfg.vision_hidden], DType::F32, &Device::Cpu).unwrap();
    // DeepStack features: 3 x [1, 74, 1536]
    let ds: Vec<DynTensor> = (0..3)
        .map(|_| DynTensor::zeros(&[1, 74, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap())
        .collect();
    let text_ids: Vec<usize> = (0..10).collect();
    let logits = model
        .forward_with_deepstack(Some(&vis), &text_ids, Some(&ds))
        .expect("forward_with_deepstack should succeed");
    assert_eq!(logits.dims(), &[1, 74, cfg.vocab_size]);
}

#[test]
fn test_encoder_decoder_hidden_size_match_8b() {
    // Vision encoder main merger outputs OUT_HIDDEN_SIZE which must
    // match the 8B decoder hidden size for DeepStack injection.
    let cfg = Qwen3VLConfig::preset_8b();
    assert_eq!(vision::OUT_HIDDEN_SIZE, cfg.hidden_size);
}

#[test]
fn test_load_with_vision_encoder_from_zeros() {
    let cfg = Qwen3VLConfig::preset_8b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3VL::load_with_vision_encoder(&vb, cfg)
        .expect("model with vision encoder should load from zeros");

    assert!(model.has_vision_encoder());
    assert!(model.vision_encoder().is_some());
}

#[test]
fn test_end_to_end_8b_zero_weight() {
    // Load full 8B model from zeros, encode image, run through decoder
    let cfg = Qwen3VLConfig::preset_8b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3VL::load_with_vision_encoder(&vb, cfg.clone())
        .expect("model with vision encoder should load from zeros");

    // Encode a 64x64 image
    let image = DynTensor::zeros(&[1, 3, 64, 64], DType::F32, &Device::Cpu).unwrap();
    let vision_output = model
        .vision_encode(&image)
        .expect("vision_encode should succeed");

    assert_eq!(vision_output.features.dims()[2], cfg.hidden_size);
    assert_eq!(vision_output.deepstack_features.len(), 3);

    // Run decoder with vision features + DeepStack
    let text_ids: Vec<usize> = (0..5).collect();
    let logits = model
        .forward_with_deepstack(
            Some(&vision_output.features),
            &text_ids,
            Some(&vision_output.deepstack_features),
        )
        .expect("decoder forward with deepstack should succeed");

    // 4 vision tokens + 5 text tokens = 9 total
    let vision_tokens = vision_output.features.dim(1).unwrap();
    assert_eq!(logits.dims(), &[1, vision_tokens + 5, cfg.vocab_size]);
}

#[test]
fn test_8b_model_debug() {
    let cfg = Qwen3VLConfig::preset_8b();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3VL::load_with_vision_encoder(&vb, cfg).expect("model should load");
    let debug_str = format!("{model:?}");
    assert!(debug_str.contains("has_vision_encoder: true"));
    assert!(debug_str.contains("Qwen3VL"));
}
