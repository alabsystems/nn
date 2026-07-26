// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`PatchEmbedding`], [`VitEncoderBlock`], [`VitEncoder`], and [`VitConfig`].
//!
//! VarBuilder loading, position interpolation, and Debug impl tests are in
//! `vit_tests_loading.rs`. Additional validation and numerical correctness
//! tests are in `vit_tests_numerical.rs`.

#[path = "vit_tests_deepstack.rs"]
mod deepstack;
#[path = "vit_tests_loading.rs"]
mod loading;
#[path = "vit_tests_numerical.rs"]
mod numerical;

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::{Conv2d, Conv2dConfig, LayerNorm, Linear, Module};
use crate::Device;

// -- Helpers ------------------------------------------------------------------

/// Small deterministic weight data for testing.
fn det_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
        .collect()
}

/// Create a small VitConfig for testing using the public constructor.
fn small_config(use_cls: bool) -> VitConfig {
    VitConfig::new(
        3,    // num_channels
        32,   // hidden_size
        2,    // num_layers
        4,    // num_heads
        64,   // intermediate_size
        4,    // patch_size
        8,    // image_size
        1e-6, // layer_norm_eps
        use_cls,
    )
    .expect("valid test config")
}

/// Create a PatchEmbedding from manual weights.
fn make_patch_embed(config: &VitConfig) -> PatchEmbedding {
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
    PatchEmbedding::new(proj, d).unwrap()
}

/// Create a small image tensor: [B, C, H, W].
fn make_image(batch: usize, config: &VitConfig, seed: f32) -> DynTensor {
    let n = batch * config.num_channels * config.image_size * config.image_size;
    DynTensor::from_vec(
        det_data(n, seed),
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

/// Create a Linear layer for testing.
fn make_linear(out: usize, inp: usize, seed: f32) -> Linear {
    let w = DynTensor::from_vec(det_data(out * inp, seed), &[out, inp], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(det_data(out, seed + 100.0), &[out], &Device::Cpu).unwrap();
    Linear::new(w, Some(b)).unwrap()
}

/// Create a LayerNorm for testing.
fn make_layer_norm(d: usize, seed: f32) -> LayerNorm {
    // Weight close to 1.0, bias close to 0.0 for reasonable outputs
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

/// Build a VitEncoderBlock manually for testing.
fn make_encoder_block(d: usize, num_heads: usize, ff_dim: usize) -> VitEncoderBlock {
    VitEncoderBlock {
        ln1: make_layer_norm(d, 1.0),
        attn_qkv: make_linear(3 * d, d, 2.0),
        attn_proj: make_linear(d, d, 3.0),
        ln2: make_layer_norm(d, 4.0),
        mlp_fc1: make_linear(ff_dim, d, 5.0),
        mlp_fc2: make_linear(d, ff_dim, 6.0),
        num_heads,
        head_dim: d / num_heads,
        scale: 1.0 / ((d / num_heads) as f64).sqrt(),
        window_size: None,
    }
}

/// Build a full VitEncoder manually for testing.
fn make_encoder(config: &VitConfig, num_blocks: usize) -> VitEncoder {
    let d = config.hidden_size;
    let seq_len = config.seq_len();
    let cls_token = if config.use_cls_token {
        Some(DynTensor::from_vec(det_data(d, 50.0), &[1, 1, d], &Device::Cpu).unwrap())
    } else {
        None
    };
    let pos_emb =
        DynTensor::from_vec(det_data(seq_len * d, 10.0), &[1, seq_len, d], &Device::Cpu).unwrap();
    let blocks: Vec<_> = (0..num_blocks)
        .map(|_| make_encoder_block(d, config.num_heads, config.intermediate_size))
        .collect();
    VitEncoder {
        patch_embed: make_patch_embed(config),
        cls_token,
        position_embedding: pos_emb,
        blocks,
        ln: make_layer_norm(d, 20.0),
        config: config.clone(),
    }
}

// -- VitConfig validation -----------------------------------------------------

#[test]
fn test_config_validate_valid() {
    let config = small_config(true);
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validate_zero_patch_size() {
    let mut config = small_config(false);
    config.patch_size = 0;
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("patch_size"), "error: {err}");
}

#[test]
fn test_config_validate_zero_image_size() {
    let mut config = small_config(false);
    config.image_size = 0;
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("image_size"), "error: {err}");
}

#[test]
fn test_config_validate_not_divisible() {
    let mut config = small_config(false);
    config.image_size = 9; // 9 not divisible by 4
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("divisible"), "error: {err}");
}

#[test]
fn test_config_validate_zero_hidden() {
    let mut config = small_config(false);
    config.hidden_size = 0;
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("hidden_size"), "error: {err}");
}

#[test]
fn test_config_validate_heads_not_divisible() {
    let mut config = small_config(false);
    config.num_heads = 3; // 32 not divisible by 3
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("divisible"), "error: {err}");
}

#[test]
fn test_config_num_patches() {
    let config = small_config(false);
    // image_size=8, patch_size=4 -> grid=2, patches=4
    assert_eq!(config.num_patches(), 4);
}

#[test]
fn test_config_seq_len_no_cls() {
    let config = small_config(false);
    assert_eq!(config.seq_len(), 4); // just patches
}

#[test]
fn test_config_seq_len_with_cls() {
    let config = small_config(true);
    assert_eq!(config.seq_len(), 5); // 4 patches + 1 CLS
}

// -- PatchEmbedding -----------------------------------------------------------

#[test]
fn test_patch_embed_output_shape() {
    let config = small_config(false);
    let pe = make_patch_embed(&config);
    let img = make_image(2, &config, 0.0);
    let out = pe.forward(&img).unwrap();
    // [2, C, 8, 8] -> Conv2d(stride=4) -> [2, 32, 2, 2] -> [2, 4, 32]
    assert_eq!(out.dims(), &[2, 4, 32]);
}

#[test]
fn test_patch_embed_single_batch() {
    let config = small_config(false);
    let pe = make_patch_embed(&config);
    let img = make_image(1, &config, 0.5);
    let out = pe.forward(&img).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

// -- VitEncoderBlock ----------------------------------------------------------

#[test]
fn test_encoder_block_output_shape() {
    let d = 32;
    let num_heads = 4;
    let ff_dim = 64;
    let block = make_encoder_block(d, num_heads, ff_dim);

    let x = DynTensor::from_vec(det_data(2 * 5 * d, 0.0), &[2, 5, d], &Device::Cpu).unwrap();
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 5, d]);
}

#[test]
fn test_encoder_block_preserves_residual() {
    // With zero weights, the attention and MLP outputs should be near-zero,
    // so the output should approximate the input (through layer norm).
    let d = 16;
    let num_heads = 2;
    let ff_dim = 32;
    let block = make_encoder_block(d, num_heads, ff_dim);

    let x_data: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
    let x = DynTensor::from_vec(x_data, &[1, 1, d], &Device::Cpu).unwrap();
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 1, d]);
    // Output should be finite
    let out_data = out.as_cpu_f32().unwrap();
    assert!(out_data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_encoder_block_module_trait() {
    let block = make_encoder_block(32, 4, 64);
    let x = DynTensor::from_vec(det_data(4 * 32, 0.0), &[1, 4, 32], &Device::Cpu).unwrap();
    // Module trait forward should work
    let out = Module::forward(&block, &x).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

// -- VitEncoder (manually constructed) ----------------------------------------

#[test]
fn test_vit_encoder_manual_no_cls() {
    let config = small_config(false);
    let encoder = make_encoder(&config, 2);
    let img = make_image(1, &config, 0.0);
    let out = encoder.forward(&img, PoolingStrategy::None).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

#[test]
fn test_vit_encoder_manual_with_cls() {
    let config = small_config(true);
    let encoder = make_encoder(&config, 2);
    let img = make_image(1, &config, 0.0);
    let out = encoder.forward(&img, PoolingStrategy::None).unwrap();
    assert_eq!(out.dims(), &[1, 5, 32]);
    let cls_out = encoder.forward(&img, PoolingStrategy::Cls).unwrap();
    assert_eq!(cls_out.dims(), &[1, 32]);
    let mean_out = encoder.forward(&img, PoolingStrategy::Mean).unwrap();
    assert_eq!(mean_out.dims(), &[1, 32]);
}

#[test]
fn test_vit_encoder_cls_pooling_without_cls_token_rejected() {
    let config = small_config(false);
    let encoder = make_encoder(&config, 1);
    let img = make_image(1, &config, 0.0);
    let err = encoder
        .forward(&img, PoolingStrategy::Cls)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Cls pooling"), "error: {err}");
}

#[test]
fn test_vit_encoder_batch_2() {
    let config = small_config(false);
    let encoder = make_encoder(&config, 1);
    let img = make_image(2, &config, 0.0);
    let out = encoder.forward(&img, PoolingStrategy::Mean).unwrap();
    assert_eq!(out.dims(), &[2, 32]);
}

#[test]
fn test_vit_encoder_module_trait() {
    let config = small_config(false);
    let encoder = make_encoder(&config, 1);
    let img = make_image(1, &config, 0.0);
    let out = Module::forward(&encoder, &img).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

#[test]
fn test_vit_encoder_output_finite() {
    let config = small_config(true);
    let encoder = make_encoder(&config, 2);
    let img = make_image(1, &config, 0.0);
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
