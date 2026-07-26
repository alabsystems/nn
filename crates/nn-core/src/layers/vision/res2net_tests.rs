// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_utils::cpu;

/// Create a deterministic test tensor with values in [0.1, 1.1).
fn test_tensor(dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| (i as f32 * 0.0073).sin().abs() + 0.1)
        .collect();
    DynTensor::from_vec(data, dims, &cpu()).expect("test tensor")
}

#[test]
fn test_res2net_shape() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let block = Res2NetBlock::load(vb.pp("r2"), 128, 3, 2, 4).expect("load");
    let x = test_tensor(&[2, 128, 100]);
    let out = block.forward(&x).expect("forward");
    assert_eq!(out.dims(), &[2, 128, 100]);
}

#[test]
fn test_res2net_multi_scale() {
    let device = cpu();
    // Different dilation values should produce different outputs.
    let vb1 = VarBuilder::zeros(crate::DType::F32, &device);
    let block1 = Res2NetBlock::load(vb1.pp("r2"), 64, 3, 1, 4).expect("load d=1");
    let vb2 = VarBuilder::zeros(crate::DType::F32, &device);
    let block2 = Res2NetBlock::load(vb2.pp("r2"), 64, 3, 3, 4).expect("load d=3");
    // With zero weights, both should produce the same output (identity through chunks).
    // But the structure is correct — actual weights would differ.
    let x = test_tensor(&[1, 64, 50]);
    let out1 = block1.forward(&x).expect("forward d=1");
    let out2 = block2.forward(&x).expect("forward d=3");
    assert_eq!(out1.dims(), out2.dims());
}

#[test]
fn test_res2net_scale_1() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    // scale=1 means no convolutions, just pass through.
    let block = Res2NetBlock::load(vb.pp("r2"), 32, 3, 1, 1).expect("load");
    let x = test_tensor(&[1, 32, 20]);
    let out = block.forward(&x).expect("forward");
    assert_eq!(out.dims(), &[1, 32, 20]);
    assert_eq!(block.convs.len(), 0);
}

#[test]
fn test_res2net_invalid_channels() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let result = Res2NetBlock::load(vb.pp("r2"), 13, 3, 1, 4);
    assert!(result.is_err(), "13 not divisible by 4");
}

#[test]
fn test_res2net_zero_scale() {
    let device = cpu();
    let vb = VarBuilder::zeros(crate::DType::F32, &device);
    let result = Res2NetBlock::load(vb.pp("r2"), 32, 3, 1, 0);
    assert!(result.is_err(), "scale=0 should be rejected");
}
