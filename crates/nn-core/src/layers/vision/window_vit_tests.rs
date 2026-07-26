// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`WindowVitEncoderBlock`] and [`WindowVitEncoder`] (#2421).

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::attention::{window_partition, window_unpartition};
use crate::layers::{LayerNorm, Linear};
use crate::Device;

// -- Helpers ------------------------------------------------------------------

fn det_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
        .collect()
}

fn make_linear(out: usize, inp: usize, seed: f32) -> Linear {
    let w = DynTensor::from_vec(det_data(out * inp, seed), &[out, inp], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(det_data(out, seed + 100.0), &[out], &Device::Cpu).unwrap();
    Linear::new(w, Some(b)).unwrap()
}

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

fn make_window_block(
    d: usize,
    num_heads: usize,
    ff_dim: usize,
    window_size: Option<usize>,
) -> WindowVitEncoderBlock {
    let inner = make_encoder_block(d, num_heads, ff_dim);
    WindowVitEncoderBlock::new(inner, window_size).unwrap()
}

// -- WindowVitConfig ----------------------------------------------------------

#[test]
fn test_window_vit_config_valid() {
    let vit = VitConfig::new(3, 32, 4, 4, 64, 4, 8, 1e-6, false).unwrap();
    let config = WindowVitConfig::new(vit, 2, vec![true, false, true, false]).unwrap();
    assert_eq!(config.window_size, 2);
    assert_eq!(config.window_pattern, vec![true, false, true, false]);
}

#[test]
fn test_window_vit_config_alternating() {
    let vit = VitConfig::new(3, 32, 6, 4, 64, 4, 8, 1e-6, false).unwrap();
    let config = WindowVitConfig::alternating(vit, 2).unwrap();
    // Odd indices = window attention
    assert_eq!(
        config.window_pattern,
        vec![false, true, false, true, false, true]
    );
}

#[test]
fn test_window_vit_config_zero_window_size() {
    let vit = VitConfig::new(3, 32, 2, 4, 64, 4, 8, 1e-6, false).unwrap();
    let err = WindowVitConfig::new(vit, 0, vec![true, false]).unwrap_err();
    assert!(format!("{err:?}").contains("window_size must be > 0"));
}

#[test]
fn test_window_vit_config_pattern_length_mismatch() {
    let vit = VitConfig::new(3, 32, 4, 4, 64, 4, 8, 1e-6, false).unwrap();
    let err = WindowVitConfig::new(vit, 2, vec![true, false]).unwrap_err();
    assert!(format!("{err:?}").contains("window_pattern length"));
}

// -- WindowVitEncoderBlock ----------------------------------------------------

#[test]
fn test_window_block_global_attention_shape() {
    let d = 32;
    let block = make_window_block(d, 4, 64, None);
    assert!(!block.uses_window_attention());

    // [B=1, S=16, D=32] — 4x4 spatial grid
    let x = DynTensor::from_vec(det_data(16 * d, 0.0), &[1, 16, d], &Device::Cpu).unwrap();
    let out = block.forward_spatial(&x, 4, 4).unwrap();
    assert_eq!(out.dims(), &[1, 16, d]);
}

#[test]
fn test_window_block_window_attention_shape() {
    let d = 32;
    let block = make_window_block(d, 4, 64, Some(2));
    assert!(block.uses_window_attention());

    // [B=1, S=16, D=32] — 4x4 spatial grid, window_size=2
    let x = DynTensor::from_vec(det_data(16 * d, 0.0), &[1, 16, d], &Device::Cpu).unwrap();
    let out = block.forward_spatial(&x, 4, 4).unwrap();
    assert_eq!(out.dims(), &[1, 16, d]);
}

#[test]
fn test_window_block_window_attention_batch() {
    let d = 32;
    let block = make_window_block(d, 4, 64, Some(2));

    // batch=2, 4x4 grid
    let x = DynTensor::from_vec(det_data(2 * 16 * d, 0.0), &[2, 16, d], &Device::Cpu).unwrap();
    let out = block.forward_spatial(&x, 4, 4).unwrap();
    assert_eq!(out.dims(), &[2, 16, d]);
}

#[test]
fn test_window_block_with_padding() {
    let d = 16;
    let block = make_window_block(d, 2, 32, Some(4));

    // 3x3 grid needs padding to 4x4 for window_size=4
    let x = DynTensor::from_vec(det_data(9 * d, 0.0), &[1, 9, d], &Device::Cpu).unwrap();
    let out = block.forward_spatial(&x, 3, 3).unwrap();
    // Output should be unpadded back to [1, 9, D]
    assert_eq!(out.dims(), &[1, 9, d]);
}

#[test]
fn test_window_block_output_finite() {
    let d = 32;
    let block = make_window_block(d, 4, 64, Some(2));

    let x = DynTensor::from_vec(det_data(16 * d, 0.0), &[1, 16, d], &Device::Cpu).unwrap();
    let out = block.forward_spatial(&x, 4, 4).unwrap();
    let data = out.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_window_block_seq_len_mismatch() {
    let d = 16;
    let block = make_window_block(d, 2, 32, Some(2));

    // seq_len=10 but height*width = 3*3 = 9
    let x = DynTensor::from_vec(det_data(10 * d, 0.0), &[1, 10, d], &Device::Cpu).unwrap();
    let err = block.forward_spatial(&x, 3, 3).unwrap_err();
    assert!(format!("{err:?}").contains("ShapeMismatch"));
}

#[test]
fn test_window_block_zero_window_size_rejected() {
    let inner = make_encoder_block(32, 4, 64);
    let err = WindowVitEncoderBlock::new(inner, Some(0)).unwrap_err();
    assert!(format!("{err:?}").contains("window_size must be > 0"));
}

// -- window_partition -> attention -> window_unpartition shape test ------------

#[test]
fn test_partition_attention_unpartition_shapes() {
    // Simulate the full window attention data flow manually.
    let (b, h, w, d, ws) = (2, 4, 4, 16, 2);
    let x =
        DynTensor::from_vec(det_data(b * h * w * d, 0.0), &[b, h * w, d], &Device::Cpu).unwrap();

    // Step 1: Partition
    let (windowed, ph, pw) = window_partition(&x, h, w, ws).unwrap();
    // 2 batches * 4 windows = 8 window-batches, each with 4 tokens
    assert_eq!(windowed.dims(), &[b * 4, ws * ws, d]);

    // Step 2: Run encoder block within each window
    let block = make_encoder_block(d, 2, 32);
    let windowed_out = block.forward(&windowed).unwrap();
    assert_eq!(windowed_out.dims(), &[b * 4, ws * ws, d]);

    // Step 3: Unpartition
    let recovered = window_unpartition(&windowed_out, h, w, ph, pw, ws, b).unwrap();
    assert_eq!(recovered.dims(), &[b, h * w, d]);
}

// -- WindowVitEncoder (alternating pattern) -----------------------------------

#[test]
fn test_window_encoder_alternating_pattern() {
    let d = 32;
    let num_heads = 4;
    let ff_dim = 64;
    let num_layers = 4;
    let ws = 2;
    // Build alternating blocks: [global, window, global, window]
    let blocks: Vec<WindowVitEncoderBlock> = (0..num_layers)
        .map(|i| {
            let inner = make_encoder_block(d, num_heads, ff_dim);
            let window = if i % 2 == 1 { Some(ws) } else { None };
            WindowVitEncoderBlock::new(inner, window).unwrap()
        })
        .collect();

    assert!(!blocks[0].uses_window_attention());
    assert!(blocks[1].uses_window_attention());
    assert!(!blocks[2].uses_window_attention());
    assert!(blocks[3].uses_window_attention());

    // Run all blocks sequentially on a 4x4 spatial grid
    let x = DynTensor::from_vec(det_data(16 * d, 0.0), &[1, 16, d], &Device::Cpu).unwrap();
    let (grid_h, grid_w) = (4, 4);
    let mut h = x;
    for block in &blocks {
        h = block.forward_spatial(&h, grid_h, grid_w).unwrap();
    }
    assert_eq!(h.dims(), &[1, 16, d]);
    let data = h.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_window_encoder_all_global() {
    let d = 32;
    // All blocks global — equivalent to standard ViT
    let blocks: Vec<WindowVitEncoderBlock> = (0..3)
        .map(|_| {
            let inner = make_encoder_block(d, 4, 64);
            WindowVitEncoderBlock::new(inner, None).unwrap()
        })
        .collect();

    let x = DynTensor::from_vec(det_data(16 * d, 0.0), &[1, 16, d], &Device::Cpu).unwrap();
    let mut h = x;
    for block in &blocks {
        h = block.forward_spatial(&h, 4, 4).unwrap();
    }
    assert_eq!(h.dims(), &[1, 16, d]);
}

#[test]
fn test_window_encoder_all_window() {
    let d = 32;
    // All blocks use window attention
    let blocks: Vec<WindowVitEncoderBlock> = (0..3)
        .map(|_| {
            let inner = make_encoder_block(d, 4, 64);
            WindowVitEncoderBlock::new(inner, Some(2)).unwrap()
        })
        .collect();

    let x = DynTensor::from_vec(det_data(16 * d, 0.0), &[1, 16, d], &Device::Cpu).unwrap();
    let mut h = x;
    for block in &blocks {
        h = block.forward_spatial(&h, 4, 4).unwrap();
    }
    assert_eq!(h.dims(), &[1, 16, d]);
}

#[test]
fn test_window_encoder_inner_accessor() {
    let block = make_window_block(32, 4, 64, Some(2));
    let inner = block.inner();
    assert_eq!(inner.num_heads, 4);
    assert_eq!(inner.head_dim, 8);
}

#[test]
fn test_window_encoder_non_square_grid() {
    let d = 16;
    // 2x8 = 16 tokens, window_size=2 (exactly divisible in both dims)
    let block = make_window_block(d, 2, 32, Some(2));
    let x = DynTensor::from_vec(det_data(16 * d, 0.0), &[1, 16, d], &Device::Cpu).unwrap();
    let out = block.forward_spatial(&x, 2, 8).unwrap();
    assert_eq!(out.dims(), &[1, 16, d]);
}

#[test]
fn test_window_vit_config_propagates_vit_validation() {
    // Bad vit config (hidden_size not divisible by num_heads)
    let vit = VitConfig::new(3, 32, 2, 3, 64, 4, 8, 1e-6, false);
    assert!(vit.is_err()); // VitConfig itself catches this
}

// -- Qwen3-VL every_nth_global pattern (#3857) --------------------------------

#[test]
fn test_window_vit_config_every_nth_global_pattern() {
    let vit = VitConfig::new(3, 32, 8, 4, 64, 4, 8, 1e-6, false).unwrap();
    let config = WindowVitConfig::every_nth_global(vit, 2, 4).unwrap();
    // With global_every_n=4 and 8 layers:
    // layers 3, 7 are global (false); rest are window (true)
    assert_eq!(
        config.window_pattern,
        vec![true, true, true, false, true, true, true, false]
    );
}

#[test]
fn test_window_vit_config_every_nth_global_every_2() {
    let vit = VitConfig::new(3, 32, 6, 4, 64, 4, 8, 1e-6, false).unwrap();
    let config = WindowVitConfig::every_nth_global(vit, 2, 2).unwrap();
    // global_every_n=2: layers 1, 3, 5 are global
    assert_eq!(
        config.window_pattern,
        vec![true, false, true, false, true, false]
    );
}

#[test]
fn test_window_vit_config_every_nth_global_zero_rejected() {
    let vit = VitConfig::new(3, 32, 4, 4, 64, 4, 8, 1e-6, false).unwrap();
    let err = WindowVitConfig::every_nth_global(vit, 2, 0).unwrap_err();
    assert!(format!("{err:?}").contains("global_every_n must be > 0"));
}

#[test]
fn test_window_vit_config_all_window() {
    let vit = VitConfig::new(3, 32, 4, 4, 64, 4, 8, 1e-6, false).unwrap();
    let config = WindowVitConfig::all_window(vit, 2).unwrap();
    assert_eq!(config.window_pattern, vec![true, true, true, true]);
}

#[test]
fn test_window_vit_config_all_global() {
    let vit = VitConfig::new(3, 32, 4, 4, 64, 4, 8, 1e-6, false).unwrap();
    let config = WindowVitConfig::all_global(vit, 2).unwrap();
    assert_eq!(config.window_pattern, vec![false, false, false, false]);
}

#[test]
fn test_qwen3_vl_style_encoder_every_4th_global() {
    // Simulate Qwen3-VL's every-4th-global pattern with small dimensions.
    let d = 32;
    let num_heads = 4;
    let ff_dim = 64;
    let num_layers = 8;
    let ws = 2;

    // Build blocks: layers 3, 7 global; rest window
    let blocks: Vec<WindowVitEncoderBlock> = (0..num_layers)
        .map(|i| {
            let inner = make_encoder_block(d, num_heads, ff_dim);
            let window = if (i + 1) % 4 != 0 { Some(ws) } else { None };
            WindowVitEncoderBlock::new(inner, window).unwrap()
        })
        .collect();

    // Verify pattern: 3 window, 1 global, 3 window, 1 global
    assert!(blocks[0].uses_window_attention());
    assert!(blocks[1].uses_window_attention());
    assert!(blocks[2].uses_window_attention());
    assert!(!blocks[3].uses_window_attention()); // global
    assert!(blocks[4].uses_window_attention());
    assert!(blocks[5].uses_window_attention());
    assert!(blocks[6].uses_window_attention());
    assert!(!blocks[7].uses_window_attention()); // global

    // Run all blocks on a 4x4 spatial grid
    let x = DynTensor::from_vec(det_data(16 * d, 0.0), &[1, 16, d], &Device::Cpu).unwrap();
    let (grid_h, grid_w) = (4, 4);
    let mut h = x;
    for block in &blocks {
        h = block.forward_spatial(&h, grid_h, grid_w).unwrap();
    }
    assert_eq!(h.dims(), &[1, 16, d]);
    let data = h.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_qwen3_vl_variable_resolution_grid() {
    // Qwen3-VL supports variable-resolution images, producing different
    // spatial grids. Test with non-square grids at Qwen3-VL-like pattern.
    let d = 32;
    let num_heads = 4;
    let ff_dim = 64;
    let ws = 2;

    // 4 layers: every-4th-global with period 4 => layer 3 is global
    let blocks: Vec<WindowVitEncoderBlock> = (0..4)
        .map(|i| {
            let inner = make_encoder_block(d, num_heads, ff_dim);
            let window = if (i + 1) % 4 != 0 { Some(ws) } else { None };
            WindowVitEncoderBlock::new(inner, window).unwrap()
        })
        .collect();

    // 6x4 = 24 tokens (non-square, divisible by window_size=2)
    let (grid_h, grid_w) = (6, 4);
    let seq_len = grid_h * grid_w;
    let x = DynTensor::from_vec(
        det_data(seq_len * d, 0.0),
        &[1, seq_len, d],
        &Device::Cpu,
    )
    .unwrap();
    let mut h = x;
    for block in &blocks {
        h = block.forward_spatial(&h, grid_h, grid_w).unwrap();
    }
    assert_eq!(h.dims(), &[1, seq_len, d]);

    // 3x5 = 15 tokens (non-square, needs padding for window_size=2)
    let (grid_h2, grid_w2) = (3, 5);
    let seq_len2 = grid_h2 * grid_w2;
    let x2 = DynTensor::from_vec(
        det_data(seq_len2 * d, 0.0),
        &[1, seq_len2, d],
        &Device::Cpu,
    )
    .unwrap();
    let mut h2 = x2;
    for block in &blocks {
        h2 = block.forward_spatial(&h2, grid_h2, grid_w2).unwrap();
    }
    assert_eq!(h2.dims(), &[1, seq_len2, d]);
    let data2 = h2.as_cpu_f32().unwrap();
    assert!(data2.iter().all(|v| v.is_finite()));
}
