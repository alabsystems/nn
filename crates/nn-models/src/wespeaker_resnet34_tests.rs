// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for WeSpeaker ResNet34 speaker embedding model.

use super::*;
use nn_core::test_utils::cpu;
use nn_core::var_builder::VarBuilder;

/// Create a deterministic test tensor with values in [0.1, 1.1).
fn test_tensor(dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| (i as f32 * 0.0073).sin().abs() + 0.1)
        .collect();
    DynTensor::from_vec(data, dims, &cpu()).expect("test tensor")
}

fn make_vb() -> VarBuilder {
    VarBuilder::zeros(nn_core::DType::F32, &cpu())
}

#[test]
fn test_basic_block_no_downsample() {
    let vb = make_vb();
    let block = BasicBlock::load(&vb, 32, 32, 1).unwrap();
    let x = test_tensor(&[1, 32, 40, 80]);
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 32, 40, 80]);
}

#[test]
fn test_basic_block_with_downsample() {
    let vb = make_vb();
    let block = BasicBlock::load(&vb, 32, 64, 2).unwrap();
    let x = test_tensor(&[1, 32, 40, 80]);
    let out = block.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 64, 20, 40]);
}

#[test]
fn test_resnet_layer_stride1() {
    let vb = make_vb();
    let layer = ResNetLayer::load(&vb, 32, 32, 3, 1).unwrap();
    let x = test_tensor(&[1, 32, 40, 80]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 32, 40, 80]);
}

#[test]
fn test_resnet_layer_stride2() {
    let vb = make_vb();
    let layer = ResNetLayer::load(&vb, 32, 64, 4, 2).unwrap();
    let x = test_tensor(&[1, 32, 40, 80]);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 64, 20, 40]);
}

#[test]
fn test_tstp_pooling_shape() {
    // Simulated layer4 output: [B, 256, T/8, 10].
    let x = test_tensor(&[2, 256, 37, 10]);
    let out = tstp_pool(&x).unwrap();
    // Mean and std of [B, 2560, 37] → cat → [B, 5120].
    assert_eq!(out.dims(), &[2, 5120]);
}

#[test]
fn test_tstp_pooling_single_frame() {
    // Edge case: T=1 → std uses n=1.0 denominator (no div-by-zero).
    let x = test_tensor(&[1, 256, 1, 10]);
    let out = tstp_pool(&x).unwrap();
    assert_eq!(out.dims(), &[1, 5120]);
}

#[test]
fn test_wespeaker_resnet34_forward_shape() {
    let vb = make_vb();
    let model = WeSpeakerResNet34::load(&vb).unwrap();

    // Standard input: 300 frames of 80-bin fbank.
    let input = test_tensor(&[1, 1, 300, 80]);
    let output = model.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 256]);
}

#[test]
fn test_wespeaker_resnet34_batch() {
    let vb = make_vb();
    let model = WeSpeakerResNet34::load(&vb).unwrap();

    let input = test_tensor(&[4, 1, 200, 80]);
    let output = model.forward(&input).unwrap();
    assert_eq!(output.dims(), &[4, 256]);
}

#[test]
fn test_wespeaker_resnet34_wrong_rank() {
    let vb = make_vb();
    let model = WeSpeakerResNet34::load(&vb).unwrap();

    let input = test_tensor(&[1, 300, 80]);
    let err = model.forward(&input).unwrap_err();
    assert!(err.to_string().contains("Rank"), "err: {err}");
}

#[test]
fn test_wespeaker_resnet34_wrong_channels() {
    let vb = make_vb();
    let model = WeSpeakerResNet34::load(&vb).unwrap();

    let input = test_tensor(&[1, 3, 300, 80]);
    let err = model.forward(&input).unwrap_err();
    assert!(err.to_string().contains("Shape"), "err: {err}");
}

#[test]
fn test_wespeaker_resnet34_wrong_freq() {
    let vb = make_vb();
    let model = WeSpeakerResNet34::load(&vb).unwrap();

    let input = test_tensor(&[1, 1, 300, 40]);
    let err = model.forward(&input).unwrap_err();
    assert!(err.to_string().contains("Shape"), "err: {err}");
}

#[test]
fn test_embed_dim() {
    let vb = make_vb();
    let model = WeSpeakerResNet34::load(&vb).unwrap();
    assert_eq!(model.embed_dim(), 256);
}

#[test]
fn test_variable_length_input() {
    let vb = make_vb();
    let model = WeSpeakerResNet34::load(&vb).unwrap();
    // Different temporal lengths → same embedding dimension.
    let short = test_tensor(&[1, 1, 80, 80]);
    let long = test_tensor(&[1, 1, 400, 80]);
    let emb_short = model.forward(&short).unwrap();
    let emb_long = model.forward(&long).unwrap();
    assert_eq!(emb_short.dims(), &[1, 256]);
    assert_eq!(emb_long.dims(), &[1, 256]);
}
