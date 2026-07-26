// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nn_core::test_utils::cpu;

/// Create a deterministic test tensor with values in [0.1, 1.1).
fn test_tensor(dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| (i as f32 * 0.0073).sin().abs() + 0.1)
        .collect();
    DynTensor::from_vec(data, dims, &cpu()).expect("test tensor")
}

#[test]
fn test_se_res2_block_shape() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let block = SERes2Block::load(vb.pp("b"), 512, 512, 3, 2, 8, 128).expect("load");
    let x = test_tensor(&[2, 512, 100]);
    let out = block.forward(&x).expect("forward");
    assert_eq!(out.dims(), &[2, 512, 100]);
}

#[test]
fn test_se_res2_block_shortcut() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    // Different in/out channels triggers shortcut path.
    let block = SERes2Block::load(vb.pp("b"), 256, 512, 3, 2, 8, 128).expect("load");
    assert!(block.shortcut.is_some());
    let x = test_tensor(&[1, 256, 50]);
    let out = block.forward(&x).expect("forward");
    assert_eq!(out.dims(), &[1, 512, 50]);
}

#[test]
fn test_se_res2_block_same_channels_no_shortcut() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let block = SERes2Block::load(vb.pp("b"), 128, 128, 3, 1, 4, 32).expect("load");
    assert!(block.shortcut.is_none());
}

#[test]
fn test_se_res2_block_dilation() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    // Different dilation values should produce same output shape.
    let b1 = SERes2Block::load(vb.pp("b1"), 64, 64, 3, 2, 4, 16).expect("load d=2");
    let vb2 = VarBuilder::zeros(nn_core::DType::F32, &device);
    let b2 = SERes2Block::load(vb2.pp("b2"), 64, 64, 3, 4, 4, 16).expect("load d=4");
    let x = test_tensor(&[1, 64, 40]);
    let out1 = b1.forward(&x).expect("forward d=2");
    let out2 = b2.forward(&x).expect("forward d=4");
    assert_eq!(out1.dims(), out2.dims());
}

#[test]
fn test_se_res2_block_zero_channels_rejected() {
    let device = cpu();
    let vb = VarBuilder::zeros(nn_core::DType::F32, &device);
    let result = SERes2Block::load(vb.pp("b"), 0, 64, 3, 1, 4, 16);
    let err = result.unwrap_err();
    assert!(
        matches!(err, TensorError::ZeroLengthDimension { .. }),
        "expected ZeroLengthDimension, got: {err:?}"
    );
}
