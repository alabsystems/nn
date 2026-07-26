// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for WindowAttentionConfig, WindowMultiHeadAttention, and AttentionMode.

use super::{AttentionMode, WindowAttentionConfig, WindowMultiHeadAttention};
use crate::dyn_tensor::DynTensor;
use crate::layers::vision::VitEncoderBlock;
use crate::layers::{LayerNorm, Linear};
use crate::{DType, Device};

// -- WindowAttentionConfig tests ----------------------------------------------

#[test]
fn test_window_attention_config_valid() {
    let config = WindowAttentionConfig::new(14, 8, 80).unwrap();
    assert_eq!(config.window_size, 14);
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.head_dim, 80);
    assert_eq!(config.hidden_size(), 640);
}

#[test]
fn test_window_attention_config_zero_window_size() {
    let err = WindowAttentionConfig::new(0, 8, 80);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("window_size"), "msg: {msg}");
}

#[test]
fn test_window_attention_config_zero_heads() {
    let err = WindowAttentionConfig::new(14, 0, 80);
    assert!(err.is_err());
}

#[test]
fn test_window_attention_config_zero_head_dim() {
    let err = WindowAttentionConfig::new(14, 8, 0);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("head_dim"), "msg: {msg}");
}

// -- WindowMultiHeadAttention tests -------------------------------------------

fn make_linear(out_features: usize, in_features: usize, seed: f32) -> Linear {
    let n = out_features * in_features;
    let w_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.001).sin() * 0.1)
        .collect();
    let b_data: Vec<f32> = (0..out_features)
        .map(|i| ((i as f32 + seed * 2.0) * 0.003).sin() * 0.01)
        .collect();
    let w = DynTensor::from_vec(w_data, &[out_features, in_features], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(b_data, &[out_features], &Device::Cpu).unwrap();
    Linear::new(w, Some(b)).unwrap()
}

fn make_window_mha(d: usize, num_heads: usize, window_size: usize) -> WindowMultiHeadAttention {
    let config = WindowAttentionConfig::new(window_size, num_heads, d / num_heads).unwrap();
    let qkv = make_linear(3 * d, d, 1.0);
    let out_proj = make_linear(d, d, 2.0);
    WindowMultiHeadAttention::new(qkv, out_proj, config).unwrap()
}

#[test]
fn test_window_mha_output_shape() {
    let d = 32;
    let num_heads = 4;
    let ws = 2;
    let (b, h, w) = (1, 4, 4);
    let mha = make_window_mha(d, num_heads, ws);

    let x = DynTensor::ones(&[b, h * w, d], DType::F32, &Device::Cpu).unwrap();
    let out = mha.forward(&x, h, w).unwrap();

    // Output shape must match input shape
    assert_eq!(out.dims(), &[b, h * w, d]);
}

#[test]
fn test_window_mha_output_shape_batch2() {
    let d = 16;
    let num_heads = 2;
    let ws = 2;
    let (b, h, w) = (2, 4, 4);
    let mha = make_window_mha(d, num_heads, ws);

    let x = DynTensor::ones(&[b, h * w, d], DType::F32, &Device::Cpu).unwrap();
    let out = mha.forward(&x, h, w).unwrap();

    assert_eq!(out.dims(), &[b, h * w, d]);
}

#[test]
fn test_window_mha_single_window() {
    // When window_size == grid_size, window attention degenerates to global
    let d = 16;
    let num_heads = 2;
    let (b, h, w) = (1, 4, 4);
    let ws = 4; // one window covers the entire grid
    let mha = make_window_mha(d, num_heads, ws);

    let x = DynTensor::ones(&[b, h * w, d], DType::F32, &Device::Cpu).unwrap();
    let out = mha.forward(&x, h, w).unwrap();

    assert_eq!(out.dims(), &[b, h * w, d]);
}

#[test]
fn test_window_mha_seq_len_mismatch() {
    let d = 16;
    let num_heads = 2;
    let ws = 2;
    let mha = make_window_mha(d, num_heads, ws);

    let x = DynTensor::ones(&[1, 10, d], DType::F32, &Device::Cpu).unwrap();
    // height * width = 3 * 4 = 12 != 10
    let err = mha.forward(&x, 3, 4);
    assert!(err.is_err());
}

// -- VitEncoderBlock with window attention tests ------------------------------

fn make_layer_norm(d: usize, seed: f32) -> LayerNorm {
    let w_data: Vec<f32> = (0..d)
        .map(|i| 1.0 + ((i as f32 + seed) * 0.01).sin() * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..d)
        .map(|i| ((i as f32 + seed * 2.0) * 0.01).sin() * 0.01)
        .collect();
    let w = DynTensor::from_vec(w_data, &[d], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(b_data, &[d], &Device::Cpu).unwrap();
    LayerNorm::new(w, b, 1e-6).unwrap()
}

fn make_window_encoder_block(
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

fn make_global_encoder_block(d: usize, num_heads: usize, ff_dim: usize) -> VitEncoderBlock {
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

#[test]
fn test_vit_block_forward_with_spatial_global() {
    let d = 32;
    let block = make_global_encoder_block(d, 4, 64);
    let (b, h, w) = (1, 4, 4);
    let x = DynTensor::ones(&[b, h * w, d], DType::F32, &Device::Cpu).unwrap();

    // Global mode should work on any block
    let out = block
        .forward_with_spatial(&x, h, w, AttentionMode::Global)
        .unwrap();
    assert_eq!(out.dims(), &[b, h * w, d]);
}

#[test]
fn test_vit_block_forward_with_spatial_window() {
    let d = 32;
    let block = make_window_encoder_block(d, 4, 64, 2);
    let (b, h, w) = (1, 4, 4);
    let x = DynTensor::ones(&[b, h * w, d], DType::F32, &Device::Cpu).unwrap();

    let out = block
        .forward_with_spatial(&x, h, w, AttentionMode::Window)
        .unwrap();
    assert_eq!(out.dims(), &[b, h * w, d]);
}

#[test]
fn test_vit_block_window_mode_on_global_block_falls_back() {
    // A block without window_size should fall back to global attention
    // even when called with AttentionMode::Window.
    let d = 32;
    let block = make_global_encoder_block(d, 4, 64);
    let (b, h, w) = (1, 4, 4);
    let x = DynTensor::ones(&[b, h * w, d], DType::F32, &Device::Cpu).unwrap();

    // Window mode on a global block just uses global attention (no error)
    let out = block
        .forward_with_spatial(&x, h, w, AttentionMode::Window)
        .unwrap();
    assert_eq!(out.dims(), &[b, h * w, d]);
}

#[test]
fn test_window_vs_global_equivalence_full_window() {
    // When window_size covers the entire grid, window attention should
    // produce the same output as global attention (same weights).
    let d = 16;
    let num_heads = 2;
    let ff_dim = 32;
    let (b, h, w) = (1, 4, 4);
    let ws = 4; // window == entire grid

    let block = make_window_encoder_block(d, num_heads, ff_dim, ws);

    let data: Vec<f32> = (0..(b * h * w * d))
        .map(|i| ((i as f32) * 0.01).sin() * 0.1)
        .collect();
    let x = DynTensor::from_vec(data, &[b, h * w, d], &Device::Cpu).unwrap();

    let out_global = block
        .forward_with_spatial(&x, h, w, AttentionMode::Global)
        .unwrap();
    let out_window = block
        .forward_with_spatial(&x, h, w, AttentionMode::Window)
        .unwrap();

    // They share the same weights, so when window == full grid, results match.
    let g = out_global.to_flat_vec::<f32>().unwrap();
    let w_vec = out_window.to_flat_vec::<f32>().unwrap();
    assert_eq!(g.len(), w_vec.len());
    for (i, (gv, wv)) in g.iter().zip(w_vec.iter()).enumerate() {
        assert!(
            (gv - wv).abs() < 1e-4,
            "Mismatch at {i}: global={gv}, window={wv}"
        );
    }
}

#[test]
fn test_new_with_window_zero_size_error() {
    let err = VitEncoderBlock::new_with_window(
        make_layer_norm(16, 1.0),
        make_linear(48, 16, 2.0),
        make_linear(16, 16, 3.0),
        make_layer_norm(16, 4.0),
        make_linear(32, 16, 5.0),
        make_linear(16, 32, 6.0),
        2,
        8,
        0, // window_size = 0 is invalid
    );
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("window_size"), "msg: {msg}");
}

// -- AttentionMode tests ------------------------------------------------------

#[test]
fn test_attention_mode_eq() {
    assert_eq!(AttentionMode::Global, AttentionMode::Global);
    assert_eq!(AttentionMode::Window, AttentionMode::Window);
    assert_ne!(AttentionMode::Global, AttentionMode::Window);
}

#[test]
fn test_attention_mode_debug() {
    let dbg = format!("{:?}", AttentionMode::Window);
    assert!(dbg.contains("Window"));
}
