// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal↔IR differential testing: reduction + structural kernels.
//!
//! Split from `metal_ir_parity.rs` (#548) — tests for multi-step tensor
//! dispatch kernels (reduce→broadcast→elementwise chains) and structural
//! kernels (reshape→axis_select→broadcast→elementwise→stack).
//!
//! # Coverage
//!
//! Reduction kernels:
//! - K2 InstanceNorm: `instance_norm_ref()` — reduce→broadcast→elementwise chains
//! - K5 RMSNorm: `rms_norm_ref()` — reduce→broadcast→elementwise chains
//! - K7 LayerNorm: `layer_norm_ref()` — reduce→broadcast→elementwise chains
//!
//! Structural kernels:
//! - K6 RoPE: `rope_rotate_ref()` — reshape→axis_select→broadcast→elementwise→stack

#![cfg(target_os = "macos")]

use nn_dsl::{
    build_instance_norm_decomposed, build_layer_norm_decomposed, build_rms_norm_decomposed,
    build_rope_rotate_kernel, instance_norm_ref, layer_norm_ref, rms_norm_ref, rope_rotate_ref,
    ScalarType,
};
use nn_metal::{execute_tensor_dispatch, PipelineCache};
use std::collections::HashMap;

#[path = "../common/metal_helpers.rs"]
mod metal_helpers;
use metal_helpers::{assert_metal_cpu_parity, assert_within_budget, metal_setup, rand_f32_vec};

/// Helper: execute a decomposed tensor kernel on Metal via the production
/// `execute_tensor_dispatch` pipeline.
fn run_tensor_dispatch(
    cache: &PipelineCache,
    kernel: &nn_dsl::TensorKernelDef,
    inputs: HashMap<&str, Vec<f32>>,
) -> Vec<f32> {
    // Ensure global MetalBackend is initialized for DynTensor GPU dispatch.
    let _ = nn_metal::MetalBackend::init();
    execute_tensor_dispatch(cache, kernel, ScalarType::F32, &inputs)
        .expect("execute_tensor_dispatch")
}

// ===========================================================================
// K2 InstanceNorm: instance_norm_ref(x, b, c, t, eps)
// ===========================================================================

#[test]
fn test_k2_instance_norm_metal_ir_parity() {
    let cache = metal_setup();

    let (b, c, t) = (1, 4, 32);
    let total = b * c * t; // 128 elements
    let eps = 1e-5_f32;

    let kernel = build_instance_norm_decomposed(b, c, t).expect("build K2 InstanceNorm");

    // CPU reference
    let x_data = rand_f32_vec(0xA1B2_C3D4, total, -3.0, 3.0);
    let cpu_out = instance_norm_ref(&x_data, b, c, t, eps).expect("instance_norm_ref");

    // GPU via production execute_tensor_dispatch
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);
    let gpu_out = run_tensor_dispatch(&cache, &kernel, inputs);

    assert_eq!(gpu_out.len(), cpu_out.len(), "K2 output length mismatch");
    assert_metal_cpu_parity("instance_norm", &gpu_out, &cpu_out);
    assert_within_budget("instance_norm", &gpu_out, &cpu_out, &[]);
}

// ===========================================================================
// K5 RMSNorm: rms_norm_ref(x, weight, n, hidden, eps)
// ===========================================================================

#[test]
fn test_k5_rms_norm_metal_ir_parity() {
    let cache = metal_setup();

    let (n, hidden) = (4, 32);
    let total = n * hidden; // 128 elements
    let eps = 1e-5_f32;

    let kernel = build_rms_norm_decomposed(n, hidden).expect("build K5 RMSNorm");

    // CPU reference
    let x_data = rand_f32_vec(0xE5F6_A7B8, total, -2.0, 2.0);
    let weight = vec![1.0_f32; hidden]; // Identity weights
    let cpu_out = rms_norm_ref(&x_data, &weight, n, hidden, eps).expect("rms_norm_ref");

    // GPU via production execute_tensor_dispatch
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);
    inputs.insert("weight", weight);
    let gpu_out = run_tensor_dispatch(&cache, &kernel, inputs);

    assert_eq!(gpu_out.len(), cpu_out.len(), "K5 output length mismatch");
    assert_metal_cpu_parity("rms_norm", &gpu_out, &cpu_out);
    assert_within_budget("rms_norm", &gpu_out, &cpu_out, &[]);
}

/// K5 RMSNorm with non-trivial weights: exercises weight broadcasting across
/// the reduce dimension. Identity weights (1.0) are mathematically transparent
/// and cannot catch broadcasting bugs (#328).
#[test]
fn test_k5_rms_norm_nontrivial_weights_parity() {
    let cache = metal_setup();

    let (n, hidden) = (4, 32);
    let total = n * hidden;
    let eps = 1e-5_f32;

    let kernel = build_rms_norm_decomposed(n, hidden).expect("build K5 RMSNorm");

    let x_data = rand_f32_vec(0xA1B2_C3D4, total, -2.0, 2.0);
    let weight = rand_f32_vec(0xD4E5_F6A7, hidden, 0.1, 3.0);
    let cpu_out = rms_norm_ref(&x_data, &weight, n, hidden, eps).expect("rms_norm_ref");

    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);
    inputs.insert("weight", weight);
    let gpu_out = run_tensor_dispatch(&cache, &kernel, inputs);

    assert_eq!(
        gpu_out.len(),
        cpu_out.len(),
        "K5 nontrivial output length mismatch"
    );
    assert_metal_cpu_parity("rms_norm_nontrivial", &gpu_out, &cpu_out);
    assert_within_budget("rms_norm_nontrivial", &gpu_out, &cpu_out, &[]);
}

// ===========================================================================
// K7 LayerNorm: layer_norm_ref(x, gamma, beta, n, hidden, eps)
// ===========================================================================

#[test]
fn test_k7_layer_norm_metal_ir_parity() {
    let cache = metal_setup();

    let (n, hidden) = (4, 32);
    let total = n * hidden; // 128 elements
    let eps = 1e-5_f32;

    let kernel = build_layer_norm_decomposed(n, hidden).expect("build K7 LayerNorm");

    // CPU reference
    let x_data = rand_f32_vec(0xC9DA_EB0C, total, -2.0, 2.0);
    let gamma = vec![1.0_f32; hidden]; // Identity scale
    let beta_vec = vec![0.0_f32; hidden]; // Zero bias
    let cpu_out =
        layer_norm_ref(&x_data, &gamma, &beta_vec, n, hidden, eps).expect("layer_norm_ref");

    // GPU via production execute_tensor_dispatch
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);
    inputs.insert("gamma", gamma);
    inputs.insert("beta", beta_vec);
    let gpu_out = run_tensor_dispatch(&cache, &kernel, inputs);

    assert_eq!(gpu_out.len(), cpu_out.len(), "K7 output length mismatch");
    assert_metal_cpu_parity("layer_norm", &gpu_out, &cpu_out);
    assert_within_budget("layer_norm", &gpu_out, &cpu_out, &[]);
}

/// K7 LayerNorm with non-trivial gamma/beta: exercises affine broadcasting.
/// Identity gamma (1.0) and zero beta (0.0) are mathematically transparent
/// and cannot catch broadcasting bugs (#328).
#[test]
fn test_k7_layer_norm_nontrivial_weights_parity() {
    let cache = metal_setup();

    let (n, hidden) = (4, 32);
    let total = n * hidden;
    let eps = 1e-5_f32;

    let kernel = build_layer_norm_decomposed(n, hidden).expect("build K7 LayerNorm");

    let x_data = rand_f32_vec(0xB2C3_D4E5, total, -2.0, 2.0);
    let gamma = rand_f32_vec(0xE5F6_A7B8, hidden, 0.1, 3.0);
    let beta_vec = rand_f32_vec(0xF6A7_B8C9, hidden, -1.0, 1.0);
    let cpu_out =
        layer_norm_ref(&x_data, &gamma, &beta_vec, n, hidden, eps).expect("layer_norm_ref");

    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);
    inputs.insert("gamma", gamma);
    inputs.insert("beta", beta_vec);
    let gpu_out = run_tensor_dispatch(&cache, &kernel, inputs);

    assert_eq!(
        gpu_out.len(),
        cpu_out.len(),
        "K7 nontrivial output length mismatch"
    );
    assert_metal_cpu_parity("layer_norm_nontrivial", &gpu_out, &cpu_out);
    assert_within_budget("layer_norm_nontrivial", &gpu_out, &cpu_out, &[]);
}

// ===========================================================================
// K6 RoPE: rope_rotate_ref(x, freqs, bh, seq_len, head_dim)
// ===========================================================================

#[test]
fn test_k6_rope_metal_ir_parity() {
    let cache = metal_setup();

    let (bh, seq_len, head_dim) = (2, 8, 16);
    let x_total = bh * seq_len * head_dim; // 256 elements
    let freq_total = seq_len * (head_dim / 2); // 64 elements

    let kernel = build_rope_rotate_kernel(bh, seq_len, head_dim).expect("build K6 RoPE");

    // CPU reference
    let x_data = rand_f32_vec(0xF1E2_D3C4, x_total, -2.0, 2.0);
    let freqs = rand_f32_vec(0xB5A6_9788, freq_total, 0.0, std::f32::consts::TAU);
    let cpu_out = rope_rotate_ref(&x_data, &freqs, bh, seq_len, head_dim).expect("rope_rotate_ref");

    // GPU via production execute_tensor_dispatch
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("freqs", freqs);
    let gpu_out = run_tensor_dispatch(&cache, &kernel, inputs);

    assert_eq!(gpu_out.len(), cpu_out.len(), "K6 output length mismatch");
    assert_metal_cpu_parity("rope_rotate", &gpu_out, &cpu_out);
    assert_within_budget("rope_rotate", &gpu_out, &cpu_out, &[]);
}
