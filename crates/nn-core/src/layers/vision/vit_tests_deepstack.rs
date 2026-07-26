// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`VitEncoder::forward_deepstack`] and end-to-end DeepStack fusion.

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::vision::DeepStackFusion;
use crate::layers::{LayerNorm, Linear};
use crate::Device;

// -- Helpers (reuse from vit_tests) -------------------------------------------

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

fn make_patch_embed(config: &VitConfig) -> PatchEmbedding {
    use crate::layers::{Conv2d, Conv2dConfig};
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

fn make_test_encoder(config: &VitConfig, num_blocks: usize) -> VitEncoder {
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

fn small_config(use_cls: bool) -> VitConfig {
    VitConfig::new(3, 32, 4, 4, 64, 4, 8, 1e-6, use_cls).expect("valid test config")
}

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

fn make_fusion(input_hidden: usize, num_layers: usize, output_hidden: usize) -> DeepStackFusion {
    let concat_dim = num_layers * input_hidden;
    let proj = make_linear(output_hidden, concat_dim, 42.0);
    DeepStackFusion::new(proj, input_hidden, num_layers, output_hidden).unwrap()
}

// -- VitEncoder::forward_deepstack -------------------------------------------

#[test]
fn test_forward_deepstack_shape() {
    let config = small_config(false);
    let encoder = make_test_encoder(&config, 4);
    let img = make_image(1, &config, 0.0);

    let outputs = encoder.forward_deepstack(&img, &[0, 2, 3]).unwrap();
    assert_eq!(outputs.len(), 3);
    for out in &outputs {
        assert_eq!(out.dims(), &[1, 4, 32]); // [B, num_patches, D]
    }
}

#[test]
fn test_forward_deepstack_all_layers() {
    let config = small_config(false);
    let encoder = make_test_encoder(&config, 3);
    let img = make_image(1, &config, 0.0);

    let outputs = encoder.forward_deepstack(&img, &[0, 1, 2]).unwrap();
    assert_eq!(outputs.len(), 3);
}

#[test]
fn test_forward_deepstack_empty_indices() {
    let config = VitConfig::new(3, 32, 2, 4, 64, 4, 8, 1e-6, false).unwrap();
    let encoder = make_test_encoder(&config, 2);
    let img = make_image(1, &config, 0.0);

    let err = encoder
        .forward_deepstack(&img, &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty"), "error: {err}");
}

#[test]
fn test_forward_deepstack_index_out_of_range() {
    let config = VitConfig::new(3, 32, 2, 4, 64, 4, 8, 1e-6, false).unwrap();
    let encoder = make_test_encoder(&config, 2);
    let img = make_image(1, &config, 0.0);

    let err = encoder
        .forward_deepstack(&img, &[0, 5])
        .unwrap_err()
        .to_string();
    assert!(err.contains("out of range"), "error: {err}");
}

#[test]
fn test_forward_deepstack_order_preserved() {
    let config = small_config(false);
    let encoder = make_test_encoder(&config, 4);
    let img = make_image(1, &config, 0.0);

    // Request in reverse order: results should match requested order
    let rev = encoder.forward_deepstack(&img, &[3, 1, 0]).unwrap();
    let fwd = encoder.forward_deepstack(&img, &[0, 1, 3]).unwrap();

    // rev[0] (layer 3) should equal fwd[2] (layer 3)
    let rev_0: Vec<f32> = rev[0].to_flat_vec::<f32>().unwrap();
    let fwd_2: Vec<f32> = fwd[2].to_flat_vec::<f32>().unwrap();
    assert_eq!(rev_0.len(), fwd_2.len());
    for (a, b) in rev_0.iter().zip(fwd_2.iter()) {
        assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
    }
}

#[test]
fn test_forward_deepstack_with_cls() {
    let config = small_config(true);
    let encoder = make_test_encoder(&config, 2);
    let img = make_image(1, &config, 0.0);

    let outputs = encoder.forward_deepstack(&img, &[0, 1]).unwrap();
    assert_eq!(outputs.len(), 2);
    // With CLS: seq_len = 4 patches + 1 CLS = 5
    assert_eq!(outputs[0].dims(), &[1, 5, 32]);
}

#[test]
fn test_forward_deepstack_batch_2() {
    let config = small_config(false);
    let encoder = make_test_encoder(&config, 3);
    let img = make_image(2, &config, 0.0);

    let outputs = encoder.forward_deepstack(&img, &[1, 2]).unwrap();
    assert_eq!(outputs.len(), 2);
    for out in &outputs {
        assert_eq!(out.dims(), &[2, 4, 32]);
    }
}

// -- End-to-end: VitEncoder + DeepStackFusion --------------------------------

#[test]
fn test_end_to_end_deepstack_fusion() {
    let hidden = 32;
    let num_extract = 3;
    let output = 48;

    let config = small_config(false);
    let encoder = make_test_encoder(&config, 4);
    let fusion = make_fusion(hidden, num_extract, output);

    let img = make_image(1, &config, 0.0);

    // Extract from layers 0, 2, 3
    let intermediates = encoder.forward_deepstack(&img, &[0, 2, 3]).unwrap();
    let fused = fusion.forward_multi(&intermediates).unwrap();
    assert_eq!(fused.dims(), &[1, 4, output]);

    let data = fused.as_cpu_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}
