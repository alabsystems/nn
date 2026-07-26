#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Validation error-path tests for fused GPU RoPE.
//!
//! Covers the 5 distinct error returns in `gpu_rope`:
//! 1. Non-f32 dtype on any of the 3 inputs
//! 2. Rank < 2
//! 3. Odd or zero head_dim
//! 4. cos shape mismatch
//! 5. sin shape mismatch
//!
//! Gap identified by P1 proof_coverage audit — zero GPU-level error path
//! coverage existed before this file.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::init;

/// Non-f32 input tensor returns DTypeMismatch.
#[test]
fn test_gpu_rope_rejects_non_f32_input() {
    init();
    let half = 2;
    let seq_len = 3;
    let head_dim = 4;

    // Create U32 tensor — gpu_rope should reject it
    let x = DynTensor::from_vec_u32(
        vec![1u32; seq_len * head_dim],
        &[seq_len, head_dim],
        &Device::metal(),
    )
    .unwrap();
    let cos = DynTensor::from_vec(
        vec![1.0f32; seq_len * half],
        &[seq_len, half],
        &Device::metal(),
    )
    .unwrap();
    let sin = DynTensor::from_vec(
        vec![0.0f32; seq_len * half],
        &[seq_len, half],
        &Device::metal(),
    )
    .unwrap();

    let result = nn_core::layers::rope(&x, &cos, &sin);
    assert!(result.is_err(), "Expected DTypeMismatch for U32 input");
}

/// Rank-1 tensor (1D) returns InvalidShape.
#[test]
fn test_gpu_rope_rejects_rank1() {
    init();
    let head_dim = 4;

    let x = DynTensor::from_vec(vec![1.0f32; head_dim], &[head_dim], &Device::metal()).unwrap();
    let cos = DynTensor::from_vec(
        vec![1.0f32; head_dim / 2],
        &[1, head_dim / 2],
        &Device::metal(),
    )
    .unwrap();
    let sin = DynTensor::from_vec(
        vec![0.0f32; head_dim / 2],
        &[1, head_dim / 2],
        &Device::metal(),
    )
    .unwrap();

    let result = nn_core::layers::rope(&x, &cos, &sin);
    assert!(result.is_err(), "Expected error for rank-1 input");
}

/// Odd head_dim returns InvalidShape.
#[test]
fn test_gpu_rope_rejects_odd_head_dim() {
    init();
    let seq_len = 2;
    let odd_dim = 5; // odd, not valid

    let x = DynTensor::from_vec(
        vec![1.0f32; seq_len * odd_dim],
        &[seq_len, odd_dim],
        &Device::metal(),
    )
    .unwrap();
    // cos/sin shape would be [2, 2] for head_dim=5 but that's floor(5/2)=2
    let cos =
        DynTensor::from_vec(vec![1.0f32; seq_len * 2], &[seq_len, 2], &Device::metal()).unwrap();
    let sin =
        DynTensor::from_vec(vec![0.0f32; seq_len * 2], &[seq_len, 2], &Device::metal()).unwrap();

    let result = nn_core::layers::rope(&x, &cos, &sin);
    assert!(result.is_err(), "Expected error for odd head_dim");
}

/// cos shape mismatch returns ShapeMismatch.
#[test]
fn test_gpu_rope_rejects_cos_shape_mismatch() {
    init();
    let seq_len = 3;
    let head_dim = 4;
    let half = head_dim / 2;

    let x = DynTensor::from_vec(
        vec![1.0f32; seq_len * head_dim],
        &[seq_len, head_dim],
        &Device::metal(),
    )
    .unwrap();
    // Wrong cos shape: [seq_len, head_dim] instead of [seq_len, half]
    let cos = DynTensor::from_vec(
        vec![1.0f32; seq_len * head_dim],
        &[seq_len, head_dim],
        &Device::metal(),
    )
    .unwrap();
    let sin = DynTensor::from_vec(
        vec![0.0f32; seq_len * half],
        &[seq_len, half],
        &Device::metal(),
    )
    .unwrap();

    let result = nn_core::layers::rope(&x, &cos, &sin);
    assert!(
        result.is_err(),
        "Expected ShapeMismatch for wrong cos shape"
    );
}

/// sin with wrong last dim (not half of head_dim) returns error.
#[test]
fn test_gpu_rope_rejects_sin_wrong_last_dim() {
    init();
    let seq_len = 3;
    let head_dim = 4;
    let half = head_dim / 2;

    let x = DynTensor::from_vec(
        vec![1.0f32; seq_len * head_dim],
        &[seq_len, head_dim],
        &Device::metal(),
    )
    .unwrap();
    let cos = DynTensor::from_vec(
        vec![1.0f32; seq_len * half],
        &[seq_len, half],
        &Device::metal(),
    )
    .unwrap();
    // Wrong sin last dim: 3 instead of 2 (half of head_dim=4)
    let sin =
        DynTensor::from_vec(vec![0.0f32; seq_len * 3], &[seq_len, 3], &Device::metal()).unwrap();

    let result = nn_core::layers::rope(&x, &cos, &sin);
    assert!(
        result.is_err(),
        "Expected error for sin with wrong last dim"
    );
}

/// Multi-batch 4D tensor (batch=2, heads=4) exercises batch flattening.
#[test]
fn test_gpu_rope_multi_batch_multi_head() {
    init();
    let head_dim = 4;
    let max_seq = 16;
    let batch = 2;
    let heads = 4;
    let seq_len = 3;
    let base = 10000.0;

    let n = batch * heads * seq_len * head_dim;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.05).collect();

    // CPU reference
    let cpu_rope =
        nn_core::layers::RotaryEmbedding::new(head_dim, max_seq, base, &Device::Cpu).unwrap();
    let cpu_x =
        DynTensor::from_vec(data, &[batch, heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let cpu_out = cpu_rope.apply(&cpu_x, 0).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // GPU
    let gpu_rope =
        nn_core::layers::RotaryEmbedding::new(head_dim, max_seq, base, &Device::metal()).unwrap();
    let gpu_x = cpu_x.to_device(&Device::metal()).unwrap();
    let gpu_out = gpu_rope.apply(&gpu_x, 0).unwrap();
    assert_eq!(gpu_out.device(), Device::metal());

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    crate::test_common::assert_close(&gpu_vals, &cpu_vals, 1e-5, "rope_multi_batch_head");
}
