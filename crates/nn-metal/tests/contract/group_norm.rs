// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for GroupNorm(groups=1) decomposition:
//! GPU output of GroupNorm g1 within NY IBP bounds.
//!
//! GroupNorm(1) decomposes into:
//!   Reshape [C, T] → [1, C*T]
//!   → Reduce(mean) → Broadcast → Sub → Square → Reduce(var) → Broadcast
//!   → BroadcastEps → Add → Rsqrt → Mul → Reshape [1, C*T] → [C, T]
//!   → optional per-channel affine (gamma * x + beta)
//!
//! IBP bounds through this 10-op decomposition are vacuously wide (>1e6)
//! due to correlation loss at each step (#697). The contract test validates
//! GPU output falls within these wide-but-finite proved bounds.
//!
//! Part of #709 AC1.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

// ===========================================================================
// GroupNorm(1) GPU contract tests
// ===========================================================================

/// Small GroupNorm g1: input [4, 8], eps=1e-5, no affine.
/// Exercises the full decomposed norm pipeline: Reshape + 10 primitives + Reshape.
/// Part of #709 AC1.
#[test]
fn test_group_norm_g1_gpu_within_bounds_small() {
    let (channels, time_len) = (4, 8);
    let eps = 1e-5_f32;

    let mut b = TensorBlockBuilder::new("group_norm_g1_small");
    let x = b.add_input("x", &[channels, time_len]);
    let eps_node = b.add_input("eps", &[1]);
    let out = b.add_group_norm_g1(x, eps_node, None, None, channels, time_len);
    let def = b.build(out).expect("valid graph");

    let bindings = vec![
        TensorParamBinding::Variable,            // x
        TensorParamBinding::ConstantScalar(eps), // eps
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GroupNorm graph must build");
    let lower_in = ArrayD::from_elem(IxDyn(&[1, channels, time_len]), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[1, channels, time_len]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through GroupNorm");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    // Squeeze leading dimension if present.
    let squeeze = |arr: &ArrayD<f32>| -> ArrayD<f32> {
        if arr.shape().first() == Some(&1) {
            let new_shape: Vec<usize> = arr.shape()[1..].to_vec();
            let flat: Vec<f32> = arr.iter().copied().collect();
            ArrayD::from_shape_vec(IxDyn(&new_shape), flat).expect("squeeze reshape")
        } else {
            arr.clone()
        }
    };
    let proved_lo = squeeze(proved_lo);
    let proved_hi = squeeze(proved_hi);

    assert_eq!(
        proved_lo.shape(),
        &[channels, time_len],
        "output bounds shape"
    );

    // IBP bounds for decomposed GroupNorm are vacuously wide (#697).
    // Verify they are finite but wide.
    let max_width = proved_lo
        .iter()
        .zip(proved_hi.iter())
        .map(|(l, u)| u - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width.is_finite(),
        "IBP bounds must be finite, got {max_width}"
    );

    // Run on Metal GPU.
    let cache = metal_setup();
    let data = rand_f32_vec(0x6E01_0001, channels * time_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);
    inputs.insert("eps", vec![eps]);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("GroupNorm GPU dispatch");
    assert_eq!(gpu_out.len(), channels * time_len, "output length");

    assert_gpu_within_bounds("group_norm_g1_small", &gpu_out, &proved_lo, &proved_hi);
}

/// GroupNorm g1 with affine: input [4, 8], gamma=2.0, beta=0.5.
/// Exercises the full pipeline including per-channel broadcast+mul and broadcast+add.
/// Part of #709 AC1.
#[test]
fn test_group_norm_g1_affine_gpu_within_bounds() {
    let (channels, time_len) = (4, 8);
    let eps = 1e-5_f32;
    let gamma_val = 2.0_f32;
    let beta_val = 0.5_f32;

    let mut b = TensorBlockBuilder::new("group_norm_g1_affine");
    let x = b.add_input("x", &[channels, time_len]);
    let eps_node = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[channels]);
    let beta = b.add_input("beta", &[channels]);
    let out = b.add_group_norm_g1(x, eps_node, Some(gamma), Some(beta), channels, time_len);
    let def = b.build(out).expect("valid graph");

    let bindings = vec![
        TensorParamBinding::Variable,                  // x
        TensorParamBinding::ConstantScalar(eps),       // eps
        TensorParamBinding::ConstantScalar(gamma_val), // gamma (uniform)
        TensorParamBinding::ConstantScalar(beta_val),  // beta (uniform)
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("affine GroupNorm graph must build");
    let lower_in = ArrayD::from_elem(IxDyn(&[1, channels, time_len]), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[1, channels, time_len]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through affine GroupNorm");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    let squeeze = |arr: &ArrayD<f32>| -> ArrayD<f32> {
        if arr.shape().first() == Some(&1) {
            let new_shape: Vec<usize> = arr.shape()[1..].to_vec();
            let flat: Vec<f32> = arr.iter().copied().collect();
            ArrayD::from_shape_vec(IxDyn(&new_shape), flat).expect("squeeze reshape")
        } else {
            arr.clone()
        }
    };
    let proved_lo = squeeze(proved_lo);
    let proved_hi = squeeze(proved_hi);

    assert_eq!(proved_lo.shape(), &[channels, time_len]);

    let max_width = proved_lo
        .iter()
        .zip(proved_hi.iter())
        .map(|(l, u)| u - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width.is_finite(),
        "affine IBP bounds must be finite, got {max_width}"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0x6E02_0001, channels * time_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);
    inputs.insert("eps", vec![eps]);
    // Per-channel gamma/beta: uniform values for all channels.
    inputs.insert("gamma", vec![gamma_val; channels]);
    inputs.insert("beta", vec![beta_val; channels]);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("affine GroupNorm GPU dispatch");
    assert_eq!(gpu_out.len(), channels * time_len);

    assert_gpu_within_bounds("group_norm_g1_affine", &gpu_out, &proved_lo, &proved_hi);
}

/// Dvoice-scale GroupNorm g1: input [48, 16], no affine.
/// Tests at production channel counts matching Demucs encoder blocks.
/// Part of #709 AC1.
#[test]
fn test_group_norm_g1_gpu_within_bounds_dvoice() {
    let (channels, time_len) = (48, 16);
    let eps = 1e-5_f32;

    let mut b = TensorBlockBuilder::new("group_norm_g1_dvoice");
    let x = b.add_input("x", &[channels, time_len]);
    let eps_node = b.add_input("eps", &[1]);
    let out = b.add_group_norm_g1(x, eps_node, None, None, channels, time_len);
    let def = b.build(out).expect("valid graph");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(eps),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GroupNorm graph must build");
    let lower_in = ArrayD::from_elem(IxDyn(&[1, channels, time_len]), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[1, channels, time_len]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through GroupNorm");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    let squeeze = |arr: &ArrayD<f32>| -> ArrayD<f32> {
        if arr.shape().first() == Some(&1) {
            let new_shape: Vec<usize> = arr.shape()[1..].to_vec();
            let flat: Vec<f32> = arr.iter().copied().collect();
            ArrayD::from_shape_vec(IxDyn(&new_shape), flat).expect("squeeze reshape")
        } else {
            arr.clone()
        }
    };
    let proved_lo = squeeze(proved_lo);
    let proved_hi = squeeze(proved_hi);

    assert_eq!(proved_lo.shape(), &[channels, time_len]);

    let max_width = proved_lo
        .iter()
        .zip(proved_hi.iter())
        .map(|(l, u)| u - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width.is_finite(),
        "dvoice IBP bounds must be finite, got {max_width}"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0xDA5D_6E01, channels * time_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);
    inputs.insert("eps", vec![eps]);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("dvoice GroupNorm GPU dispatch");
    assert_eq!(gpu_out.len(), channels * time_len);

    assert_gpu_within_bounds("group_norm_g1_dvoice", &gpu_out, &proved_lo, &proved_hi);
}
