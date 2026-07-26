// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for decomposed InstanceNorm1d:
//! GPU output of the full Reduce → Broadcast → Elementwise pipeline checked
//! against native NY IBP bounds.
//!
//! Unlike `contract_norm.rs` which tests the per-element scalar kernel with
//! pre-computed mean/variance, these tests exercise the **complete decomposed
//! pipeline** including:
//! 1. Reduction pass (mean, variance computation)
//! 2. Broadcast alignment (statistics → full tensor shape)
//! 3. Normalization elementwise: `(x - mean) * rsqrt(var + eps)`
//!
//! The soundness loop: NY IBP bounds (native layer) ⊇ GPU output
//! (decomposed steps).
//!
//! Part of #696.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::{build_instance_norm_decomposed, instance_norm_ref, ScalarType};
use nn_metal::execute_tensor_dispatch;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------

/// Prove IBP bounds for decomposed InstanceNorm using native NY layer.
///
/// The `build_instance_norm_decomposed` kernel has 2 inputs:
///   - "x" (Variable): shape [B, C, T]
///   - "eps" (ConstantScalar): shape [1]
///
/// Returns (proved_lower, proved_upper) arrays over the output shape [B, C, T].
fn prove_instance_norm_bounds(
    b: usize,
    c: usize,
    t: usize,
    eps: f32,
    input_lo: f32,
    input_hi: f32,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let kernel = build_instance_norm_decomposed(b, c, t).expect("build decomposed InstanceNorm");

    let bindings = vec![
        TensorParamBinding::Variable,            // x
        TensorParamBinding::ConstantScalar(eps), // eps
    ];

    let graph = tensor_kernel_to_graph(&kernel, &bindings).expect("InstanceNorm graph must build");

    let lower_in = ArrayD::from_elem(IxDyn(&[b, c, t]), input_lo);
    let upper_in = ArrayD::from_elem(IxDyn(&[b, c, t]), input_hi);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");

    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through InstanceNorm");
    let (lo, hi) = output_bounds.lower_upper();

    assert!(
        lo.iter().all(|v| v.is_finite()),
        "proved lower must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "proved upper must be finite"
    );
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }

    (lo.to_owned(), hi.to_owned())
}

// ===========================================================================
// AC1 + AC2: Full-tensor GPU contract test exercising reduction pass
// ===========================================================================

/// Small decomposed InstanceNorm: B=1, C=2, T=8.
/// Exercises: 2 reductions (mean, variance), 3 broadcasts, 5 elementwise ops.
/// The tensor has 16 elements across 2 channels — reduction computes per-channel
/// mean and variance, NOT pre-computed constants.
#[test]
fn test_instance_norm_decomposed_gpu_within_bounds_small() {
    let b = 1;
    let c = 2;
    let t = 8;
    let total = b * c * t;
    let eps = 1e-5_f32;

    let kernel = build_instance_norm_decomposed(b, c, t).expect("build decomposed InstanceNorm");

    let (proved_lo, proved_hi) = prove_instance_norm_bounds(b, c, t, eps, -1.0, 1.0);
    assert_eq!(proved_lo.shape(), &[b, c, t], "output bounds shape");

    // GPU dispatch with random inputs within [-1, 1]
    let cache = metal_setup();
    let x_data = rand_f32_vec(0x1A5D_0001, total, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("InstanceNorm GPU dispatch");
    assert_eq!(gpu_out.len(), total, "output length");

    assert_gpu_within_bounds("instance_norm_small", &gpu_out, &proved_lo, &proved_hi);
}

/// Medium decomposed InstanceNorm: B=1, C=4, T=32.
/// Matches the dimensions used in the existing differential test
/// (`tensor_dispatch.rs::test_k2_instance_norm_tensor_dispatch`).
/// Also validates GPU output matches CPU reference within precision budget.
#[test]
fn test_instance_norm_decomposed_gpu_within_bounds_medium() {
    let b = 1;
    let c = 4;
    let t = 32;
    let total = b * c * t;
    let eps = 1e-5_f32;

    let kernel = build_instance_norm_decomposed(b, c, t).expect("build decomposed InstanceNorm");

    let (proved_lo, proved_hi) = prove_instance_norm_bounds(b, c, t, eps, -3.0, 3.0);
    assert_eq!(proved_lo.shape(), &[b, c, t]);

    let cache = metal_setup();
    let x_data = rand_f32_vec(0x1A5D_0002, total, -3.0, 3.0);

    // CPU reference for differential check
    let cpu_out = instance_norm_ref(&x_data, b, c, t, eps).expect("CPU reference");

    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("InstanceNorm GPU dispatch");
    assert_eq!(gpu_out.len(), total);

    // AC3: GPU output within native NY IBP bounds
    assert_gpu_within_bounds("instance_norm_medium", &gpu_out, &proved_lo, &proved_hi);

    // Differential check: GPU ≈ CPU within precision budget
    let max_diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "GPU-CPU max diff {max_diff} exceeds tolerance"
    );
}

/// B=2 batch: exercises reduction across batch dimension.
/// InstanceNorm normalizes per (batch, channel) slice, so B=2 tests that
/// each batch element is normalized independently.
#[test]
fn test_instance_norm_decomposed_gpu_within_bounds_batched() {
    let b = 2;
    let c = 4;
    let t = 16;
    let total = b * c * t;
    let eps = 1e-5_f32;

    let kernel = build_instance_norm_decomposed(b, c, t).expect("build decomposed InstanceNorm");

    let (proved_lo, proved_hi) = prove_instance_norm_bounds(b, c, t, eps, -2.0, 2.0);
    assert_eq!(proved_lo.shape(), &[b, c, t]);

    let cache = metal_setup();
    let x_data = rand_f32_vec(0x1A5D_0003, total, -2.0, 2.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("batched InstanceNorm GPU dispatch");
    assert_eq!(gpu_out.len(), total);

    assert_gpu_within_bounds("instance_norm_batched", &gpu_out, &proved_lo, &proved_hi);
}

// ===========================================================================
// AC4: Dvoice-scale test
// ===========================================================================

/// Dvoice-realistic: B=1, C=48, T=16 (Demucs encoder block dimensions).
/// Tests at production channel counts where precision accumulation through
/// the reduction pass matters most.
#[test]
fn test_instance_norm_decomposed_gpu_within_bounds_dvoice() {
    let b = 1;
    let c = 48;
    let t = 16;
    let total = b * c * t;
    let eps = 1e-5_f32;

    let kernel = build_instance_norm_decomposed(b, c, t).expect("build decomposed InstanceNorm");

    let (proved_lo, proved_hi) = prove_instance_norm_bounds(b, c, t, eps, -1.0, 1.0);
    assert_eq!(proved_lo.shape(), &[b, c, t]);

    // Note: IBP bounds for InstanceNorm at dvoice scale (48 channels, T=16)
    // are typically vacuous due to bound blow-up through the rsqrt(var + eps)
    // division. This is a known limitation of IBP for normalization layers.
    // The contract test still validates that GPU output falls within the proved
    // (albeit wide) bounds, and we rely on the differential check for precision.
    let max_width = proved_lo
        .iter()
        .zip(proved_hi.iter())
        .map(|(l, u)| u - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width.is_finite(),
        "IBP bounds must be finite even if wide"
    );

    let cache = metal_setup();
    let x_data = rand_f32_vec(0xDA5E_0001, total, -1.0, 1.0);

    let cpu_out = instance_norm_ref(&x_data, b, c, t, eps).expect("CPU reference");

    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("dvoice InstanceNorm GPU dispatch");
    assert_eq!(gpu_out.len(), total);

    // AC3: GPU output within native NY IBP bounds
    assert_gpu_within_bounds("instance_norm_dvoice", &gpu_out, &proved_lo, &proved_hi);

    // Differential: GPU ≈ CPU (primary precision check at dvoice scale)
    let max_diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "dvoice GPU-CPU max diff {max_diff} exceeds tolerance"
    );
}

/// Wider temporal dimension: B=1, C=4, T=128.
/// Tests reduction accuracy over a longer time axis where floating-point
/// accumulation in mean/variance computation is more demanding.
#[test]
fn test_instance_norm_decomposed_gpu_within_bounds_long_temporal() {
    let b = 1;
    let c = 4;
    let t = 128;
    let total = b * c * t;
    let eps = 1e-5_f32;

    let kernel = build_instance_norm_decomposed(b, c, t).expect("build decomposed InstanceNorm");

    let (proved_lo, proved_hi) = prove_instance_norm_bounds(b, c, t, eps, -2.0, 2.0);
    assert_eq!(proved_lo.shape(), &[b, c, t]);

    let cache = metal_setup();
    let x_data = rand_f32_vec(0x1A5D_0005, total, -2.0, 2.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", x_data);
    inputs.insert("eps", vec![eps]);

    let gpu_out = execute_tensor_dispatch(&cache, &kernel, ScalarType::F32, &inputs)
        .expect("long temporal InstanceNorm GPU dispatch");
    assert_eq!(gpu_out.len(), total);

    assert_gpu_within_bounds(
        "instance_norm_long_temporal",
        &gpu_out,
        &proved_lo,
        &proved_hi,
    );
}
