// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for vision module components: VitEncoder, VitEncoderBlock,
//! PatchEmbedding, SigLip2VisionEncoder, VitConfig, SigLip2Config,
//! Qwen2VLVitConfig, PoolingStrategy, window attention, RotaryEmbedding2d,
//! and multi-head attention variants.

use crate::dyn_tensor::DynTensor;
use crate::layers::attention::{
    window_partition, window_unpartition, AttentionMode, RotaryEmbedding2d,
};
use crate::layers::vision::{PoolingStrategy, SigLip2Config, WindowVitConfig};
use crate::layers::{
    Conv2d, Conv2dConfig, LayerNorm, Linear, Module, Qwen2VLVitConfig, Qwen3VLVitConfig, VitConfig,
    VitEncoder, VitEncoderBlock,
};
use crate::test_prng::rand_f32_vec;
use crate::var_builder::VarBuilder;
use crate::{DType, Device};

// =============================================================================
// Helpers
// =============================================================================

/// Create a DynTensor with deterministic pseudo-random data.
fn rand_tensor(seed: u64, dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data = rand_f32_vec(seed, numel, -0.5, 0.5);
    DynTensor::from_vec(data, dims, &Device::Cpu).unwrap()
}

/// Small deterministic weight data for testing.
fn det_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
        .collect()
}

/// Create a small VitConfig for testing.
fn small_vit_config(use_cls: bool) -> VitConfig {
    VitConfig::new(3, 32, 2, 4, 64, 4, 8, 1e-6, use_cls).expect("valid test config")
}

/// Create a Linear layer for testing.
fn make_linear(out: usize, inp: usize, seed: f32) -> Linear {
    let w = DynTensor::from_vec(det_data(out * inp, seed), &[out, inp], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(det_data(out, seed + 100.0), &[out], &Device::Cpu).unwrap();
    Linear::new(w, Some(b)).unwrap()
}

/// Create a LayerNorm for testing (weight ~1.0, bias ~0.0).
fn make_layer_norm(d: usize, seed: f32) -> LayerNorm {
    let w_data: Vec<f32> = (0..d)
        .map(|i| 1.0 + det_data(1, seed + i as f32)[0] * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..d)
        .map(|i| det_data(1, seed + 50.0 + i as f32)[0] * 0.01)
        .collect();
    let w = DynTensor::from_vec(w_data, &[d], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(b_data, &[d], &Device::Cpu).unwrap();
    LayerNorm::new(w, b, 1e-6).unwrap()
}

/// Build a VitEncoderBlock manually.
fn make_encoder_block(d: usize, num_heads: usize, ff_dim: usize) -> VitEncoderBlock {
    VitEncoderBlock::new(
        make_layer_norm(d, 1.0),
        make_linear(3 * d, d, 2.0),
        make_linear(d, d, 3.0),
        make_layer_norm(d, 4.0),
        make_linear(ff_dim, d, 5.0),
        make_linear(d, ff_dim, 6.0),
        num_heads,
        d / num_heads,
    )
    .unwrap()
}

/// Build a VitEncoderBlock with window attention support.
fn make_encoder_block_with_window(
    d: usize,
    num_heads: usize,
    ff_dim: usize,
    window_size: usize,
) -> VitEncoderBlock {
    VitEncoderBlock::new_with_window(
        make_layer_norm(d, 1.0),
        make_linear(3 * d, d, 2.0),
        make_linear(d, d, 3.0),
        make_layer_norm(d, 4.0),
        make_linear(ff_dim, d, 5.0),
        make_linear(d, ff_dim, 6.0),
        num_heads,
        d / num_heads,
        window_size,
    )
    .unwrap()
}

/// Create a PatchEmbedding from manual weights.
fn make_patch_embed(config: &VitConfig) -> crate::layers::PatchEmbedding {
    let p = config.patch_size;
    let c = config.num_channels;
    let d = config.hidden_size;
    let n = d * c * p * p;
    let w = DynTensor::from_vec(det_data(n, 1.0), &[d, c, p, p], &Device::Cpu).unwrap();
    let proj = Conv2d::new(
        w,
        None,
        Conv2dConfig {
            stride: p,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
    )
    .unwrap();
    crate::layers::PatchEmbedding::new(proj, d).unwrap()
}

/// Build a full VitEncoder via VarBuilder::zeros (the load path).
fn make_encoder(config: &VitConfig) -> VitEncoder {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    VitEncoder::load(&vb, config).unwrap()
}

/// Create a small image tensor: [B, C, H, W].
fn make_image(batch: usize, config: &VitConfig, seed: u64) -> DynTensor {
    let n = batch * config.num_channels * config.image_size * config.image_size;
    let data = rand_f32_vec(seed, n, -1.0, 1.0);
    DynTensor::from_vec(
        data,
        &[
            batch,
            config.num_channels,
            config.image_size,
            config.image_size,
        ],
        &Device::Cpu,
    )
    .unwrap()
}

// =============================================================================
// 1. VitConfig validation and computation
// =============================================================================

#[test]
fn test_vit_config_new_valid() {
    let config = VitConfig::new(3, 64, 4, 8, 256, 16, 224, 1e-5, true).unwrap();
    assert_eq!(config.num_patches(), (224 / 16) * (224 / 16));
    assert_eq!(config.seq_len(), 14 * 14 + 1);
}

#[test]
fn test_vit_config_seq_len_no_cls() {
    let config = VitConfig::new(3, 64, 4, 8, 256, 16, 224, 1e-5, false).unwrap();
    assert_eq!(config.seq_len(), 196);
    assert_eq!(config.num_patches(), 196);
}

#[test]
fn test_vit_config_seq_len_with_cls() {
    let config = VitConfig::new(3, 64, 4, 8, 256, 16, 224, 1e-5, true).unwrap();
    assert_eq!(config.seq_len(), 197);
}

#[test]
fn test_vit_config_reject_zero_patch_size() {
    let err = VitConfig::new(3, 64, 4, 8, 256, 0, 224, 1e-5, false).unwrap_err();
    assert!(err.to_string().contains("patch_size"));
}

#[test]
fn test_vit_config_reject_zero_image_size() {
    let err = VitConfig::new(3, 64, 4, 8, 256, 16, 0, 1e-5, false).unwrap_err();
    assert!(err.to_string().contains("image_size"));
}

#[test]
fn test_vit_config_reject_image_not_divisible_by_patch() {
    let err = VitConfig::new(3, 64, 4, 8, 256, 16, 225, 1e-5, false).unwrap_err();
    assert!(err.to_string().contains("divisible"));
}

#[test]
fn test_vit_config_reject_zero_hidden() {
    let err = VitConfig::new(3, 0, 4, 8, 256, 16, 224, 1e-5, false).unwrap_err();
    assert!(err.to_string().contains("hidden_size"));
}

#[test]
fn test_vit_config_reject_zero_channels() {
    let err = VitConfig::new(0, 64, 4, 8, 256, 16, 224, 1e-5, false).unwrap_err();
    assert!(err.to_string().contains("num_channels"));
}

#[test]
fn test_vit_config_reject_zero_intermediate() {
    let err = VitConfig::new(3, 64, 4, 8, 0, 16, 224, 1e-5, false).unwrap_err();
    assert!(err.to_string().contains("intermediate_size"));
}

#[test]
fn test_vit_config_reject_heads_not_dividing_hidden() {
    let err = VitConfig::new(3, 64, 4, 3, 256, 16, 224, 1e-5, false).unwrap_err();
    assert!(err.to_string().contains("divisible"));
}

#[test]
fn test_vit_config_num_patches_small() {
    let config = small_vit_config(false);
    assert_eq!(config.num_patches(), 4);
}

// =============================================================================
// 2. VitEncoderBlock
// =============================================================================

#[test]
fn test_encoder_block_forward_shape() {
    let d = 32;
    let block = make_encoder_block(d, 4, 64);
    let x = rand_tensor(1, &[2, 5, d]);
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 5, d]);
}

#[test]
fn test_encoder_block_single_token() {
    let d = 16;
    let block = make_encoder_block(d, 2, 32);
    let x = rand_tensor(2, &[1, 1, d]);
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 1, d]);
    let data = out.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_encoder_block_module_trait() {
    let d = 32;
    let block = make_encoder_block(d, 4, 64);
    let x = rand_tensor(3, &[1, 4, d]);
    let out = Module::forward(&block, &x).unwrap();
    assert_eq!(out.dims(), &[1, 4, d]);
}

#[test]
fn test_encoder_block_residual_connection() {
    let d = 32;
    let block = make_encoder_block(d, 4, 64);
    let x = rand_tensor(4, &[1, 4, d]);
    let out = block.forward(&x).unwrap();
    let x_data = x.as_cpu_f32().unwrap();
    let out_data = out.as_cpu_f32().unwrap();
    assert!(out_data.iter().all(|v| v.is_finite()));
    let diff: f32 = x_data
        .iter()
        .zip(out_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.0, "output should differ from input");
}

#[test]
fn test_encoder_block_batch_independence() {
    let d = 32;
    let block = make_encoder_block(d, 4, 64);
    let x_batch = rand_tensor(5, &[2, 4, d]);
    let out_batch = block.forward(&x_batch).unwrap();

    let x0 = x_batch.narrow(0, 0, 1).unwrap();
    let x1 = x_batch.narrow(0, 1, 1).unwrap();
    let out0 = block.forward(&x0).unwrap();
    let out1 = block.forward(&x1).unwrap();

    let batch_data = out_batch.as_cpu_f32().unwrap();
    let single0_data = out0.as_cpu_f32().unwrap();
    let single1_data = out1.as_cpu_f32().unwrap();

    for i in 0..(4 * d) {
        let b_val = batch_data.as_slice().unwrap()[i];
        let s_val = single0_data.as_slice().unwrap()[i];
        assert!(
            (b_val - s_val).abs() < 1e-5,
            "mismatch at {i}: batch={b_val}, single={s_val}"
        );
    }
    for i in 0..(4 * d) {
        let b_val = batch_data.as_slice().unwrap()[4 * d + i];
        let s_val = single1_data.as_slice().unwrap()[i];
        assert!(
            (b_val - s_val).abs() < 1e-5,
            "mismatch at {i}: batch={b_val}, single={s_val}"
        );
    }
}

#[test]
fn test_encoder_block_window_size_accessor() {
    let block = make_encoder_block(32, 4, 64);
    assert_eq!(block.window_size(), None);

    let block_w = make_encoder_block_with_window(32, 4, 64, 4);
    assert_eq!(block_w.window_size(), Some(4));
}

// =============================================================================
// 3. VitEncoderBlock with spatial window attention
// =============================================================================

#[test]
fn test_encoder_block_forward_with_spatial_global() {
    let d = 32;
    let block = make_encoder_block(d, 4, 64);
    let x = rand_tensor(10, &[1, 16, d]);
    let out = block
        .forward_with_spatial(&x, 4, 4, AttentionMode::Global)
        .unwrap();
    assert_eq!(out.dims(), &[1, 16, d]);
}

#[test]
fn test_encoder_block_forward_with_spatial_window() {
    let d = 32;
    let block = make_encoder_block_with_window(d, 4, 64, 2);
    let x = rand_tensor(11, &[1, 16, d]);
    let out = block
        .forward_with_spatial(&x, 4, 4, AttentionMode::Window)
        .unwrap();
    assert_eq!(out.dims(), &[1, 16, d]);
    let data = out.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_encoder_block_forward_with_spatial_no_window_configured() {
    let d = 32;
    let block = make_encoder_block(d, 4, 64);
    let x = rand_tensor(12, &[1, 16, d]);
    let out = block
        .forward_with_spatial(&x, 4, 4, AttentionMode::Window)
        .unwrap();
    assert_eq!(out.dims(), &[1, 16, d]);
}

#[test]
fn test_encoder_block_forward_with_spatial_shape_mismatch() {
    let d = 32;
    let block = make_encoder_block_with_window(d, 4, 64, 2);
    let x = rand_tensor(13, &[1, 16, d]);
    let err = block
        .forward_with_spatial(&x, 3, 5, AttentionMode::Window)
        .unwrap_err();
    assert!(err.to_string().to_lowercase().contains("shape"));
}

// =============================================================================
// 4. PatchEmbedding
// =============================================================================

#[test]
fn test_patch_embed_output_shape_extended() {
    let config = small_vit_config(false);
    let pe = make_patch_embed(&config);
    let img = make_image(2, &config, 100);
    let out = pe.forward(&img).unwrap();
    assert_eq!(out.dims(), &[2, 4, 32]);
}

#[test]
fn test_patch_embed_single_batch() {
    let config = small_vit_config(false);
    let pe = make_patch_embed(&config);
    let img = make_image(1, &config, 200);
    let out = pe.forward(&img).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

#[test]
fn test_patch_embed_stride_correctness() {
    let config = VitConfig::new(3, 16, 1, 2, 32, 8, 32, 1e-6, false).unwrap();
    let pe = make_patch_embed(&config);
    let n = 3 * 32 * 32;
    let img = DynTensor::from_vec(
        rand_f32_vec(300, n, -1.0, 1.0),
        &[1, 3, 32, 32],
        &Device::Cpu,
    )
    .unwrap();
    let out = pe.forward(&img).unwrap();
    assert_eq!(out.dims(), &[1, 16, 16]);
}

#[test]
fn test_patch_embed_module_trait() {
    let config = small_vit_config(false);
    let pe = make_patch_embed(&config);
    let img = make_image(1, &config, 400);
    let out = Module::forward(&pe, &img).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

#[test]
fn test_patch_embed_reject_zero_hidden() {
    let w = DynTensor::from_vec(vec![0.0f32; 12], &[1, 3, 2, 2], &Device::Cpu).unwrap();
    let proj = Conv2d::new(
        w,
        None,
        Conv2dConfig {
            stride: 2,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
    )
    .unwrap();
    let err = crate::layers::PatchEmbedding::new(proj, 0).unwrap_err();
    assert!(err.to_string().contains("hidden_size"));
}

// =============================================================================
// 5. VitEncoder: forward pass shapes and pooling
// =============================================================================

#[test]
fn test_vit_encoder_load_zeros_no_cls() {
    let config = small_vit_config(false);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 500);
    let out = encoder.forward(&img, PoolingStrategy::None).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

#[test]
fn test_vit_encoder_load_zeros_with_cls() {
    let config = small_vit_config(true);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 501);
    let out = encoder.forward(&img, PoolingStrategy::None).unwrap();
    assert_eq!(out.dims(), &[1, 5, 32]);
}

#[test]
fn test_vit_encoder_cls_pooling() {
    let config = small_vit_config(true);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 502);
    let out = encoder.forward(&img, PoolingStrategy::Cls).unwrap();
    assert_eq!(out.dims(), &[1, 32]);
}

#[test]
fn test_vit_encoder_mean_pooling_no_cls() {
    let config = small_vit_config(false);
    let encoder = make_encoder(&config);
    let img = make_image(2, &config, 503);
    let out = encoder.forward(&img, PoolingStrategy::Mean).unwrap();
    assert_eq!(out.dims(), &[2, 32]);
}

#[test]
fn test_vit_encoder_mean_pooling_with_cls() {
    let config = small_vit_config(true);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 504);
    let out = encoder.forward(&img, PoolingStrategy::Mean).unwrap();
    assert_eq!(out.dims(), &[1, 32]);
}

#[test]
fn test_vit_encoder_no_pooling() {
    let config = small_vit_config(true);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 505);
    let out = encoder.forward(&img, PoolingStrategy::None).unwrap();
    assert_eq!(out.dims(), &[1, 5, 32]);
}

#[test]
fn test_vit_encoder_cls_pooling_rejected_without_cls_token() {
    let config = small_vit_config(false);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 506);
    let err = encoder
        .forward(&img, PoolingStrategy::Cls)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Cls pooling"), "error: {err}");
}

#[test]
fn test_vit_encoder_module_trait() {
    let config = small_vit_config(false);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 507);
    let out = Module::forward(&encoder, &img).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

#[test]
fn test_vit_encoder_output_finite() {
    let config = small_vit_config(true);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 508);
    for pooling in [
        PoolingStrategy::None,
        PoolingStrategy::Cls,
        PoolingStrategy::Mean,
    ] {
        let out = encoder.forward(&img, pooling).unwrap();
        let data = out.as_cpu_f32().unwrap();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "Non-finite in {pooling:?}"
        );
    }
}

#[test]
fn test_vit_encoder_config_accessor() {
    let config = small_vit_config(true);
    let encoder = make_encoder(&config);
    assert_eq!(encoder.config().hidden_size, 32);
    assert_eq!(encoder.config().num_layers, 2);
    assert!(encoder.config().use_cls_token);
}

#[test]
fn test_vit_encoder_blocks_accessor() {
    let config = small_vit_config(false);
    let encoder = make_encoder(&config);
    assert_eq!(encoder.blocks().len(), 2);
}

// =============================================================================
// 6. VitEncoder: deepstack feature extraction
// =============================================================================

#[test]
fn test_vit_encoder_deepstack_single_layer() {
    let config = small_vit_config(false);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 600);
    let features = encoder.forward_deepstack(&img, &[0]).unwrap();
    assert_eq!(features.len(), 1);
    assert_eq!(features[0].dims(), &[1, 4, 32]);
}

#[test]
fn test_vit_encoder_deepstack_multiple_layers() {
    let config = small_vit_config(false);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 601);
    let features = encoder.forward_deepstack(&img, &[0, 1]).unwrap();
    assert_eq!(features.len(), 2);
    for f in &features {
        assert_eq!(f.dims(), &[1, 4, 32]);
    }
}

#[test]
fn test_vit_encoder_deepstack_out_of_range() {
    let config = small_vit_config(false);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 602);
    let err = encoder
        .forward_deepstack(&img, &[5])
        .unwrap_err()
        .to_string();
    assert!(err.contains("out of range"), "error: {err}");
}

#[test]
fn test_vit_encoder_deepstack_empty_indices() {
    let config = small_vit_config(false);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 603);
    let err = encoder
        .forward_deepstack(&img, &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty"), "error: {err}");
}

#[test]
fn test_vit_encoder_deepstack_with_cls() {
    let config = small_vit_config(true);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 604);
    let features = encoder.forward_deepstack(&img, &[1]).unwrap();
    assert_eq!(features.len(), 1);
    assert_eq!(features[0].dims(), &[1, 5, 32]);
}

// =============================================================================
// 7. VitEncoder: position embedding interpolation
// =============================================================================

#[test]
fn test_vit_encoder_position_interp_same_size() {
    let config = small_vit_config(false);
    let encoder = make_encoder(&config);
    let img = make_image(1, &config, 700);
    let out = encoder.forward(&img, PoolingStrategy::None).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

// =============================================================================
// 8. PoolingStrategy variants
// =============================================================================

#[test]
fn test_pooling_strategy_eq() {
    assert_eq!(PoolingStrategy::Cls, PoolingStrategy::Cls);
    assert_eq!(PoolingStrategy::Mean, PoolingStrategy::Mean);
    assert_eq!(PoolingStrategy::None, PoolingStrategy::None);
    assert_ne!(PoolingStrategy::Cls, PoolingStrategy::Mean);
    assert_ne!(PoolingStrategy::Mean, PoolingStrategy::None);
    assert_ne!(PoolingStrategy::None, PoolingStrategy::Cls);
}

#[test]
fn test_pooling_strategy_clone() {
    let p = PoolingStrategy::Mean;
    let p2 = p;
    assert_eq!(p, p2);
}

#[test]
fn test_pooling_strategy_debug() {
    let s = format!("{:?}", PoolingStrategy::Cls);
    assert!(s.contains("Cls"));
    let s = format!("{:?}", PoolingStrategy::Mean);
    assert!(s.contains("Mean"));
    let s = format!("{:?}", PoolingStrategy::None);
    assert!(s.contains("None"));
}

// =============================================================================
// 9. SigLip2Config
// =============================================================================

#[test]
fn test_siglip2_config_base_patch16() {
    let config = SigLip2Config::base_patch16(224).unwrap();
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_layers, 12);
    assert_eq!(config.num_heads, 12);
    assert_eq!(config.intermediate_size, 3072);
    assert_eq!(config.patch_size, 16);
    assert_eq!(config.image_size, 224);
}

#[test]
fn test_siglip2_config_to_vit_config() {
    let config = SigLip2Config::base_patch16(224).unwrap();
    let vit = config.to_vit_config().unwrap();
    assert_eq!(vit.hidden_size, 768);
    assert!(!vit.use_cls_token);
    assert_eq!(vit.num_patches(), (224 / 16) * (224 / 16));
}

#[test]
fn test_siglip2_config_new_validation() {
    let _config = SigLip2Config::new(3, 64, 2, 4, 128, 8, 32, 1e-6).unwrap();

    let err = SigLip2Config::new(3, 0, 2, 4, 128, 8, 32, 1e-6).unwrap_err();
    assert!(err.to_string().contains("hidden_size"));

    let err = SigLip2Config::new(3, 64, 2, 3, 128, 8, 32, 1e-6).unwrap_err();
    assert!(err.to_string().contains("divisible"));
}

// =============================================================================
// 10. SigLip2VisionEncoder via VarBuilder::zeros
// =============================================================================

#[test]
fn test_siglip2_encoder_load_zeros() {
    let config = SigLip2Config::new(3, 32, 2, 4, 64, 4, 8, 1e-6).unwrap();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = crate::layers::SigLip2VisionEncoder::load(&vb, &config).unwrap();
    assert_eq!(encoder.config().hidden_size, 32);
}

#[test]
fn test_siglip2_encoder_forward_no_pooling() {
    let config = SigLip2Config::new(3, 32, 2, 4, 64, 4, 8, 1e-6).unwrap();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = crate::layers::SigLip2VisionEncoder::load(&vb, &config).unwrap();
    let img = make_image(1, &config.to_vit_config().unwrap(), 800);
    let out = encoder.forward(&img, PoolingStrategy::None).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

#[test]
fn test_siglip2_encoder_forward_mean_pooling() {
    let config = SigLip2Config::new(3, 32, 2, 4, 64, 4, 8, 1e-6).unwrap();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = crate::layers::SigLip2VisionEncoder::load(&vb, &config).unwrap();
    let img = make_image(1, &config.to_vit_config().unwrap(), 801);
    let out = encoder.forward(&img, PoolingStrategy::Mean).unwrap();
    assert_eq!(out.dims(), &[1, 32]);
}

#[test]
fn test_siglip2_encoder_cls_pooling_rejected() {
    let config = SigLip2Config::new(3, 32, 2, 4, 64, 4, 8, 1e-6).unwrap();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = crate::layers::SigLip2VisionEncoder::load(&vb, &config).unwrap();
    let img = make_image(1, &config.to_vit_config().unwrap(), 802);
    let err = encoder
        .forward(&img, PoolingStrategy::Cls)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Cls pooling"), "error: {err}");
}

#[test]
fn test_siglip2_encoder_deepstack() {
    let config = SigLip2Config::new(3, 32, 2, 4, 64, 4, 8, 1e-6).unwrap();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = crate::layers::SigLip2VisionEncoder::load(&vb, &config).unwrap();
    let img = make_image(1, &config.to_vit_config().unwrap(), 803);
    let features = encoder.forward_deepstack(&img, &[0, 1]).unwrap();
    assert_eq!(features.len(), 2);
    for f in &features {
        assert_eq!(f.dims(), &[1, 4, 32]);
    }
}

#[test]
fn test_siglip2_encoder_deepstack_empty_rejected() {
    let config = SigLip2Config::new(3, 32, 2, 4, 64, 4, 8, 1e-6).unwrap();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = crate::layers::SigLip2VisionEncoder::load(&vb, &config).unwrap();
    let img = make_image(1, &config.to_vit_config().unwrap(), 804);
    let err = encoder
        .forward_deepstack(&img, &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty"), "error: {err}");
}

#[test]
fn test_siglip2_encoder_deepstack_out_of_range() {
    let config = SigLip2Config::new(3, 32, 2, 4, 64, 4, 8, 1e-6).unwrap();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = crate::layers::SigLip2VisionEncoder::load(&vb, &config).unwrap();
    let img = make_image(1, &config.to_vit_config().unwrap(), 805);
    let err = encoder
        .forward_deepstack(&img, &[10])
        .unwrap_err()
        .to_string();
    assert!(err.contains("out of range"), "error: {err}");
}

#[test]
fn test_siglip2_encoder_module_trait() {
    let config = SigLip2Config::new(3, 32, 2, 4, 64, 4, 8, 1e-6).unwrap();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let encoder = crate::layers::SigLip2VisionEncoder::load(&vb, &config).unwrap();
    let img = make_image(1, &config.to_vit_config().unwrap(), 806);
    let out = Module::forward(&encoder, &img).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

// =============================================================================
// 11. Qwen2VLVitConfig
// =============================================================================

#[test]
fn test_qwen2vl_config_new_valid() {
    let config = Qwen2VLVitConfig::new(3, 64, 4, 4, 128, 14, 2, 1e-6, 7, vec![]).unwrap();
    assert_eq!(config.hidden_size, 64);
    assert_eq!(config.num_layers, 4);
    assert_eq!(config.window_size, 7);
    assert_eq!(config.head_dim(), 16);
}

#[test]
fn test_qwen2vl_config_qwen25_vl_7b() {
    let config = Qwen2VLVitConfig::qwen25_vl_7b().unwrap();
    assert_eq!(config.hidden_size, 1280);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_heads, 16);
    assert_eq!(config.window_size, 14);
    assert_eq!(config.head_dim(), 80);
}

#[test]
fn test_qwen2vl_config_is_window_layer_default() {
    let config = Qwen2VLVitConfig::new(3, 64, 4, 4, 128, 14, 2, 1e-6, 7, vec![]).unwrap();
    assert!(!config.is_window_layer(0));
    assert!(config.is_window_layer(1));
    assert!(!config.is_window_layer(2));
    assert!(config.is_window_layer(3));
}

#[test]
fn test_qwen2vl_config_is_window_layer_explicit() {
    let config = Qwen2VLVitConfig::new(3, 64, 4, 4, 128, 14, 2, 1e-6, 7, vec![0, 2]).unwrap();
    assert!(config.is_window_layer(0));
    assert!(!config.is_window_layer(1));
    assert!(config.is_window_layer(2));
    assert!(!config.is_window_layer(3));
}

#[test]
fn test_qwen2vl_config_reject_zero_patch_size() {
    let err = Qwen2VLVitConfig::new(3, 64, 4, 4, 128, 0, 2, 1e-6, 7, vec![]).unwrap_err();
    assert!(err.to_string().contains("patch_size"));
}

#[test]
fn test_qwen2vl_config_reject_zero_window_size() {
    let err = Qwen2VLVitConfig::new(3, 64, 4, 4, 128, 14, 2, 1e-6, 0, vec![]).unwrap_err();
    assert!(err.to_string().contains("window_size"));
}

#[test]
fn test_qwen2vl_config_reject_window_layer_out_of_range() {
    let err = Qwen2VLVitConfig::new(3, 64, 4, 4, 128, 14, 2, 1e-6, 7, vec![10]).unwrap_err();
    assert!(err.to_string().contains("window_layers"));
}

#[test]
fn test_qwen2vl_config_reject_zero_temporal_patch() {
    let err = Qwen2VLVitConfig::new(3, 64, 4, 4, 128, 14, 0, 1e-6, 7, vec![]).unwrap_err();
    assert!(err.to_string().contains("temporal_patch_size"));
}

// =============================================================================
// 12. Qwen3VLVitConfig
// =============================================================================

#[test]
fn test_qwen3vl_config_qwen3_vl_2b() {
    let config = Qwen3VLVitConfig::qwen3_vl_2b().unwrap();
    assert_eq!(config.hidden_size, 1280);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.global_every_n, 4);
    assert_eq!(config.deepstack_layers, vec![7, 15, 23, 31]);
    assert_eq!(config.deepstack_output_size, 1536);
}

#[test]
fn test_qwen3vl_config_is_global_layer() {
    let config = Qwen3VLVitConfig::qwen3_vl_2b().unwrap();
    assert!(!config.is_global_layer(0));
    assert!(!config.is_global_layer(1));
    assert!(!config.is_global_layer(2));
    assert!(config.is_global_layer(3));
    assert!(!config.is_global_layer(4));
    assert!(config.is_global_layer(7));
}

#[test]
fn test_qwen3vl_config_is_window_layer() {
    let config = Qwen3VLVitConfig::qwen3_vl_2b().unwrap();
    assert!(config.is_window_layer(0));
    assert!(config.is_window_layer(1));
    assert!(config.is_window_layer(2));
    assert!(!config.is_window_layer(3));
    assert!(config.is_window_layer(4));
}

#[test]
fn test_qwen3vl_config_window_pattern() {
    let config = Qwen3VLVitConfig::qwen3_vl_2b().unwrap();
    let pattern = config.window_pattern();
    assert_eq!(pattern.len(), 32);
    assert!(!pattern[3]);
    assert!(pattern[0]);
}

#[test]
fn test_qwen3vl_config_global_every_n_zero() {
    let config = Qwen3VLVitConfig::new(3, 64, 4, 4, 128, 14, 2, 1e-6, 7, 0, vec![], 0).unwrap();
    for i in 0..4 {
        assert!(!config.is_global_layer(i));
        assert!(config.is_window_layer(i));
    }
}

#[test]
fn test_qwen3vl_config_reject_deepstack_out_of_range() {
    let err = Qwen3VLVitConfig::new(3, 64, 4, 4, 128, 14, 2, 1e-6, 7, 4, vec![10], 64).unwrap_err();
    assert!(err.to_string().contains("deepstack_layers"));
}

#[test]
fn test_qwen3vl_config_reject_zero_deepstack_output_with_layers() {
    let err =
        Qwen3VLVitConfig::new(3, 64, 4, 4, 128, 14, 2, 1e-6, 7, 4, vec![0, 1], 0).unwrap_err();
    assert!(err.to_string().contains("deepstack_output_size"));
}

// =============================================================================
// 13. Window attention: partition/unpartition roundtrip
// =============================================================================

#[test]
fn test_window_partition_roundtrip_exact() {
    let d = 16;
    let x = rand_tensor(1300, &[1, 16, d]);
    let (windowed, ph, pw) = window_partition(&x, 4, 4, 2).unwrap();
    assert_eq!(windowed.dims(), &[4, 4, d]);
    assert_eq!(ph, 4);
    assert_eq!(pw, 4);

    let recovered = window_unpartition(&windowed, 4, 4, ph, pw, 2, 1).unwrap();
    assert_eq!(recovered.dims(), &[1, 16, d]);

    let x_data = x.as_cpu_f32().unwrap();
    let r_data = recovered.as_cpu_f32().unwrap();
    for i in 0..(16 * d) {
        let a = x_data.as_slice().unwrap()[i];
        let b = r_data.as_slice().unwrap()[i];
        assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
    }
}

#[test]
fn test_window_partition_roundtrip_with_padding() {
    let d = 8;
    let x = rand_tensor(1301, &[1, 15, d]);
    let (windowed, ph, pw) = window_partition(&x, 3, 5, 2).unwrap();
    assert_eq!(ph, 4);
    assert_eq!(pw, 6);

    let recovered = window_unpartition(&windowed, 3, 5, ph, pw, 2, 1).unwrap();
    assert_eq!(recovered.dims(), &[1, 15, d]);

    let x_data = x.as_cpu_f32().unwrap();
    let r_data = recovered.as_cpu_f32().unwrap();
    for i in 0..(15 * d) {
        let a = x_data.as_slice().unwrap()[i];
        let b = r_data.as_slice().unwrap()[i];
        assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
    }
}

#[test]
fn test_window_partition_batch() {
    let d = 8;
    let x = rand_tensor(1302, &[2, 16, d]);
    let (windowed, ph, pw) = window_partition(&x, 4, 4, 2).unwrap();
    assert_eq!(windowed.dims(), &[8, 4, d]);
    let recovered = window_unpartition(&windowed, 4, 4, ph, pw, 2, 2).unwrap();
    assert_eq!(recovered.dims(), &[2, 16, d]);
}

#[test]
fn test_window_partition_reject_zero_window_size() {
    let x = rand_tensor(1303, &[1, 4, 8]);
    let err = window_partition(&x, 2, 2, 0).unwrap_err();
    assert!(err.to_string().contains("window_size"));
}

#[test]
fn test_window_partition_reject_shape_mismatch() {
    let x = rand_tensor(1304, &[1, 10, 8]);
    let err = window_partition(&x, 3, 4, 2).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("shape"));
}

#[test]
fn test_window_unpartition_reject_zero_window_size() {
    let w = rand_tensor(1305, &[1, 4, 8]);
    let err = window_unpartition(&w, 2, 2, 2, 2, 0, 1).unwrap_err();
    assert!(err.to_string().contains("window_size"));
}

#[test]
fn test_window_partition_single_window() {
    let d = 8;
    let x = rand_tensor(1306, &[1, 9, d]);
    let (windowed, ph, pw) = window_partition(&x, 3, 3, 3).unwrap();
    assert_eq!(windowed.dims(), &[1, 9, d]);
    assert_eq!(ph, 3);
    assert_eq!(pw, 3);
}

// =============================================================================
// 14. RotaryEmbedding2d
// =============================================================================

#[test]
fn test_rope2d_new_valid() {
    let rope = RotaryEmbedding2d::new(16, 64, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), 16);
    assert_eq!(rope.max_position(), 64);
}

#[test]
fn test_rope2d_reject_non_multiple_of_4() {
    let err = RotaryEmbedding2d::new(15, 64, 10000.0, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("multiple of 4"));
}

#[test]
fn test_rope2d_reject_zero_head_dim() {
    let err = RotaryEmbedding2d::new(0, 64, 10000.0, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("multiple of 4"));
}

#[test]
fn test_rope2d_reject_zero_max_position() {
    let err = RotaryEmbedding2d::new(16, 0, 10000.0, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("max_position"));
}

#[test]
fn test_rope2d_reject_negative_base() {
    let err = RotaryEmbedding2d::new(16, 64, -1.0, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("base"));
}

#[test]
fn test_rope2d_reject_nan_base() {
    let err = RotaryEmbedding2d::new(16, 64, f64::NAN, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("base"));
}

#[test]
fn test_rope2d_apply_shape_preserved() {
    let rope = RotaryEmbedding2d::new(16, 64, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(1400, &[2, 9, 16]);
    let h_pos: Vec<usize> = (0..9).map(|i| i / 3).collect();
    let w_pos: Vec<usize> = (0..9).map(|i| i % 3).collect();
    let out = rope.apply(&x, &h_pos, &w_pos).unwrap();
    assert_eq!(out.dims(), &[2, 9, 16]);
}

#[test]
fn test_rope2d_apply_output_finite() {
    let rope = RotaryEmbedding2d::new(16, 64, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(1401, &[1, 4, 16]);
    let h_pos = vec![0, 0, 1, 1];
    let w_pos = vec![0, 1, 0, 1];
    let out = rope.apply(&x, &h_pos, &w_pos).unwrap();
    let data = out.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_rope2d_position_zero_identity() {
    let rope = RotaryEmbedding2d::new(8, 64, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(1402, &[1, 1, 8]);
    let out = rope.apply(&x, &[0], &[0]).unwrap();
    let x_data = x.as_cpu_f32().unwrap();
    let o_data = out.as_cpu_f32().unwrap();
    for i in 0..8 {
        let a = x_data.as_slice().unwrap()[i];
        let b = o_data.as_slice().unwrap()[i];
        assert!(
            (a - b).abs() < 1e-5,
            "At position (0,0), expected identity: x[{i}]={a}, out[{i}]={b}"
        );
    }
}

#[test]
fn test_rope2d_reject_position_out_of_range() {
    let rope = RotaryEmbedding2d::new(8, 4, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(1403, &[1, 1, 8]);
    let err = rope.apply(&x, &[5], &[0]).unwrap_err();
    assert!(err.to_string().contains("position"));
}

#[test]
fn test_rope2d_reject_wrong_head_dim() {
    let rope = RotaryEmbedding2d::new(8, 64, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(1404, &[1, 1, 12]);
    let err = rope.apply(&x, &[0], &[0]).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("shape"));
}

#[test]
fn test_rope2d_reject_mismatched_position_len() {
    let rope = RotaryEmbedding2d::new(8, 64, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(1405, &[1, 4, 8]);
    let err = rope.apply(&x, &[0, 1], &[0, 1, 2, 3]).unwrap_err();
    assert!(err.to_string().contains("length") || err.to_string().contains("mismatch"));
}

#[test]
fn test_rope2d_different_positions_produce_different_outputs() {
    let rope = RotaryEmbedding2d::new(8, 64, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(1406, &[1, 1, 8]);
    let out0 = rope.apply(&x, &[0], &[0]).unwrap();
    let out1 = rope.apply(&x, &[1], &[2]).unwrap();
    let d0 = out0.as_cpu_f32().unwrap();
    let d1 = out1.as_cpu_f32().unwrap();
    let diff: f32 = d0.iter().zip(d1.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-6,
        "different positions should produce different outputs"
    );
}

// =============================================================================
// 15. Multi-head attention with various head counts and dimensions
// =============================================================================

#[test]
fn test_mha_2_heads() {
    let d = 16;
    let block = make_encoder_block(d, 2, 32);
    let x = rand_tensor(1500, &[1, 4, d]);
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, d]);
}

#[test]
fn test_mha_8_heads() {
    let d = 64;
    let block = make_encoder_block(d, 8, 128);
    let x = rand_tensor(1501, &[1, 4, d]);
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, d]);
}

#[test]
fn test_mha_1_head() {
    let d = 16;
    let block = make_encoder_block(d, 1, 32);
    let x = rand_tensor(1502, &[1, 4, d]);
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, d]);
    let data = out.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mha_large_head_dim() {
    let d = 128;
    let block = make_encoder_block(d, 2, 256);
    let x = rand_tensor(1503, &[1, 4, d]);
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, d]);
}

#[test]
fn test_mha_many_tokens() {
    let d = 32;
    let block = make_encoder_block(d, 4, 64);
    let x = rand_tensor(1504, &[1, 64, d]);
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 64, d]);
    let data = out.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

// =============================================================================
// 16. VarBuilder::zeros weight loading patterns
// =============================================================================

#[test]
fn test_vit_encoder_load_zeros_all_variants() {
    for use_cls in [false, true] {
        let config = small_vit_config(use_cls);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let encoder = VitEncoder::load(&vb, &config).unwrap();
        assert_eq!(encoder.config().use_cls_token, use_cls);
        assert_eq!(encoder.blocks().len(), config.num_layers);
    }
}

#[test]
fn test_siglip2_encoder_load_zeros_different_sizes() {
    for &(d, heads) in &[(32, 4), (64, 8)] {
        let config = SigLip2Config::new(3, d, 1, heads, d * 2, 4, 8, 1e-6).unwrap();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let encoder = crate::layers::SigLip2VisionEncoder::load(&vb, &config).unwrap();
        assert_eq!(encoder.config().hidden_size, d);
    }
}

// =============================================================================
// 17. WindowVitConfig
// =============================================================================

#[test]
fn test_window_vit_config_alternating() {
    let vit = small_vit_config(false);
    let wconfig = WindowVitConfig::alternating(vit, 2).unwrap();
    assert_eq!(wconfig.window_size, 2);
    assert!(!wconfig.window_pattern[0]);
    assert!(wconfig.window_pattern[1]);
}

#[test]
fn test_window_vit_config_all_window() {
    let vit = small_vit_config(false);
    let wconfig = WindowVitConfig::all_window(vit, 2).unwrap();
    assert!(wconfig.window_pattern.iter().all(|&w| w));
}

#[test]
fn test_window_vit_config_all_global() {
    let vit = small_vit_config(false);
    let wconfig = WindowVitConfig::all_global(vit, 2).unwrap();
    assert!(wconfig.window_pattern.iter().all(|&w| !w));
}

// =============================================================================
// 18. sinusoidal_2d positional encoding
// =============================================================================

#[test]
fn test_sinusoidal_2d_shape() {
    let pos = crate::layers::attention::sinusoidal_2d(4, 4, 16, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(pos.dims(), &[16, 16]);
}

#[test]
fn test_sinusoidal_2d_reject_non_multiple_of_4() {
    let err = crate::layers::attention::sinusoidal_2d(4, 4, 15, 10000.0, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("multiple of 4"));
}

#[test]
fn test_sinusoidal_2d_reject_zero_dim() {
    let err = crate::layers::attention::sinusoidal_2d(4, 4, 0, 10000.0, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("multiple of 4"));
}

#[test]
fn test_sinusoidal_2d_reject_zero_height() {
    let err = crate::layers::attention::sinusoidal_2d(0, 4, 16, 10000.0, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("height"));
}

#[test]
fn test_sinusoidal_2d_reject_negative_temperature() {
    let err = crate::layers::attention::sinusoidal_2d(4, 4, 16, -1.0, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("temperature"));
}

#[test]
fn test_sinusoidal_2d_output_finite() {
    let pos = crate::layers::attention::sinusoidal_2d(3, 5, 8, 10000.0, &Device::Cpu).unwrap();
    let data = pos.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_sinusoidal_2d_different_positions_differ() {
    let pos = crate::layers::attention::sinusoidal_2d(2, 2, 8, 10000.0, &Device::Cpu).unwrap();
    let data = pos.as_cpu_f32().unwrap();
    let row0: Vec<f32> = (0..8).map(|i| data.as_slice().unwrap()[i]).collect();
    let row3: Vec<f32> = (0..8)
        .map(|i| data.as_slice().unwrap()[3 * 8 + i])
        .collect();
    let diff: f32 = row0
        .iter()
        .zip(row3.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "different positions should have different encodings"
    );
}

// =============================================================================
// 19. VitEncoderBlock construction validation
// =============================================================================

#[test]
fn test_encoder_block_new_reject_zero_heads() {
    let result = VitEncoderBlock::new(
        make_layer_norm(32, 1.0),
        make_linear(96, 32, 2.0),
        make_linear(32, 32, 3.0),
        make_layer_norm(32, 4.0),
        make_linear(64, 32, 5.0),
        make_linear(32, 64, 6.0),
        0,
        8,
    );
    assert!(result.is_err());
}

#[test]
fn test_encoder_block_new_reject_zero_head_dim() {
    let result = VitEncoderBlock::new(
        make_layer_norm(32, 1.0),
        make_linear(96, 32, 2.0),
        make_linear(32, 32, 3.0),
        make_layer_norm(32, 4.0),
        make_linear(64, 32, 5.0),
        make_linear(32, 64, 6.0),
        4,
        0,
    );
    assert!(result.is_err());
}

#[test]
fn test_encoder_block_new_with_window_reject_zero_window() {
    let result = VitEncoderBlock::new_with_window(
        make_layer_norm(32, 1.0),
        make_linear(96, 32, 2.0),
        make_linear(32, 32, 3.0),
        make_layer_norm(32, 4.0),
        make_linear(64, 32, 5.0),
        make_linear(32, 64, 6.0),
        4,
        8,
        0,
    );
    assert!(result.is_err());
}

// =============================================================================
// 20. VitEncoderBlock::load via VarBuilder::zeros
// =============================================================================

#[test]
fn test_encoder_block_load_zeros() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let block = VitEncoderBlock::load(&vb, 32, 4, 64, 1e-6).unwrap();
    let x = rand_tensor(2000, &[1, 4, 32]);
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

#[test]
fn test_encoder_block_load_reject_zero_intermediate() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let err = VitEncoderBlock::load(&vb, 32, 4, 0, 1e-6).unwrap_err();
    assert!(err.to_string().contains("intermediate_size"));
}
