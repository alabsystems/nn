// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Analytical bounds verification for NativeOp GPU kernels.
//!
//! NativeOps bypass the IR decomposition path that NY operates on,
//! meaning NY cannot verify their execution. This test suite provides
//! the next-best assurance: analytical bounds derived from mathematical
//! properties of each operation, verified against actual GPU output.
//!
//! ## Analytical Bounds Summary
//!
//! | NativeOp        | Bound                                                          |
//! |-----------------|----------------------------------------------------------------|
//! | InstanceNorm    | `|out[i]| ≤ sqrt(T)` where T = spatial dim                     |
//! | AdainSnake      | `|out| ≤ (1+|γ|_max)·√T + |β|_max + 1/α_min`                  |
//! | AdainLeakyRelu  | `|out| ≤ max(1,slope)·((1+|γ|_max)·√T + |β|_max)`             |
//! | AdaLayerNorm    | `|out| ≤ (1+|γ|_max)·(|w|_max·√H + |b|_max) + |β|_max`       |
//! | Cumsum          | `out[k] ∈ [(k+1)·x_min, (k+1)·x_max]`                         |
//!
//! ## Relationship to NY
//!
//! These bounds are analytically derived, not machine-verified. They serve as
//! correctness evidence until NativeOps can be decomposed into NY's
//! layer vocabulary. See #2506 for the tracking issue.
//!
//! Part of #2506 (NativeOps bypass NY).
//! Part of #2218 (Kokoro epic).

use nn_core::dyn_tensor::trace::{ComputationGraph, KokoroFusedOp, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{compile_and_run, create_input_buffer, input_node};

// -- Shared bound-verification helper -----------------------------------------

/// Verify every element in `result` is finite and within `[-bound, bound]`.
///
/// Returns the maximum absolute value observed (for diagnostic logging).
fn assert_within_analytical_bound(label: &str, result: &[f32], bound: f32, tol: f32) -> f32 {
    let mut max_abs: f32 = 0.0;
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "{label}[{i}]: non-finite output {val}");
        let abs_val = val.abs();
        max_abs = max_abs.max(abs_val);
        assert!(
            abs_val <= bound + tol,
            "{label}[{i}]: |{val}| = {abs_val} > bound {bound}"
        );
    }
    max_abs
}

// =============================================================================
// InstanceNorm bounds: |normed[i]| ≤ sqrt(T)
//
// Proof sketch: For T values with sample mean μ and population variance σ²,
// (x_i - μ)² ≤ Σ(x_j - μ)² = T·σ², so |x_i - μ| ≤ √(T·σ²).
// After normalization: |normed_i| = |x_i - μ| / √(σ² + ε) ≤ √(T·σ²/(σ² + ε)) ≤ √T.
// =============================================================================

/// InstanceNorm NativeOp: every output element must satisfy |out| ≤ √T.
#[test]
fn test_instance_norm_nativeop_analytical_bound() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (2, 4, 32);
    let eps = 1e-5_f64;
    let bound = (time as f32).sqrt();

    let input_data =
        super::test_utils::rand_f32_vec(0xB00D_0001, batch * channels * time, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "instance_norm_bounds".into(),
            TraceOp::InstanceNorm { eps },
            vec![0],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * channels * time,
    );

    let max_abs = assert_within_analytical_bound("instance_norm", &result, bound, 1e-5);
    eprintln!("instance_norm_bounds: max |out| = {max_abs:.4}, bound √T = {bound:.4}");
}

/// InstanceNorm with near-constant input: one extreme outlier per channel.
///
/// When T-1 elements are identical and 1 outlier differs, the outlier's
/// normalized value approaches ±√(T-1). This is the tightest case.
#[test]
fn test_instance_norm_nativeop_bound_tight_case() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (1, 2, 16);
    let eps = 1e-5_f64;
    let bound = (time as f32).sqrt();

    // Build input where each channel has T-1 zeros and one large outlier.
    let mut input_data = vec![0.0_f32; batch * channels * time];
    for c in 0..channels {
        input_data[c * time] = 10.0; // outlier at position 0
    }

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "instance_norm_tight".into(),
            TraceOp::InstanceNorm { eps },
            vec![0],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * channels * time,
    );

    // Verify √T bound (loose) and √(T-1) bound (tight) both hold.
    let max_abs = assert_within_analytical_bound("instance_norm_tight", &result, bound, 1e-5);
    let tight_bound = ((time - 1) as f32).sqrt();
    assert!(
        max_abs <= tight_bound + 1e-4,
        "instance_norm_tight: max |out| = {max_abs} > √(T-1) = {tight_bound}"
    );
    eprintln!(
        "instance_norm_tight: max |out| = {max_abs:.4}, bound √T = {bound:.4}, \
         tight bound √(T-1) = {tight_bound:.4}"
    );
}

// =============================================================================
// AdainSnake bounds: |out| ≤ (1 + |γ|_max) · √T + |β|_max + 1/α_min
//
// Composition: normed = InstanceNorm(x), y = (1+γ)·normed + β, out = y + sin²(α·y)/α.
// |normed| ≤ √T, |y| ≤ (1+|γ|)·√T + |β|, sin²(·) ∈ [0,1], so |out - y| ≤ 1/|α|.
// =============================================================================

/// AdainSnake NativeOp: verify analytical bound based on input parameter ranges.
#[test]
fn test_adain_snake_nativeop_analytical_bound() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (1, 4, 32);
    let eps = 1e-5_f64;

    let gamma_range = 0.3_f32;
    let beta_range = 0.2_f32;
    let alpha_min = 0.5_f32;

    let alpha_data = super::test_utils::rand_f32_vec(0xB00D_1001, channels, alpha_min, 2.0);
    let input_data =
        super::test_utils::rand_f32_vec(0xB00D_1002, batch * channels * time, -3.0, 3.0);
    let gamma_data =
        super::test_utils::rand_f32_vec(0xB00D_1003, batch * channels, -gamma_range, gamma_range);
    let beta_data =
        super::test_utils::rand_f32_vec(0xB00D_1004, batch * channels, -beta_range, beta_range);

    let alpha = WeightRef::new(alpha_data.clone(), vec![channels]).expect("alpha");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        input_node(1, &[batch, channels, 1]),
        input_node(2, &[batch, channels, 1]),
        TraceNode::new(
            3,
            "adain_snake_bounds".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake { alpha, eps }),
            vec![0, 1, 2],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &input_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * channels * time,
    );

    // Compute analytical bound.
    let norm_bound = (time as f32).sqrt();
    let actual_alpha_min = alpha_data.iter().copied().fold(f32::INFINITY, f32::min);
    let actual_gamma_max = gamma_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let actual_beta_max = beta_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

    let y_bound = (1.0 + actual_gamma_max) * norm_bound + actual_beta_max;
    let snake_addition = 1.0 / actual_alpha_min;
    let total_bound = y_bound + snake_addition;

    let max_abs = assert_within_analytical_bound("adain_snake", &result, total_bound, 1e-3);
    eprintln!(
        "adain_snake_bounds: max |out| = {max_abs:.4}, analytical bound = {total_bound:.4} \
         (norm={norm_bound:.2}, y={y_bound:.2}, snake_add={snake_addition:.2})"
    );
}

// =============================================================================
// AdainLeakyRelu bounds: |out| ≤ max(1, slope) · ((1 + |γ|_max) · √T + |β|_max)
//
// Composition: normed = InstanceNorm(x), y = (1+γ)·normed + β,
// out = y if y ≥ 0, else slope·y.
// For 0 < slope ≤ 1: |out| ≤ |y|. For slope > 1: |out| ≤ slope·|y|.
// =============================================================================

/// AdainLeakyRelu NativeOp: verify analytical bound.
#[test]
fn test_adain_leaky_relu_nativeop_analytical_bound() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (2, 8, 16);
    let eps = 1e-5_f64;
    let slope = 0.2_f64;

    let gamma_range = 0.5_f32;
    let beta_range = 0.3_f32;

    let input_data =
        super::test_utils::rand_f32_vec(0xB00D_2001, batch * channels * time, -2.0, 2.0);
    let gamma_data =
        super::test_utils::rand_f32_vec(0xB00D_2002, batch * channels, -gamma_range, gamma_range);
    let beta_data =
        super::test_utils::rand_f32_vec(0xB00D_2003, batch * channels, -beta_range, beta_range);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        input_node(1, &[batch, channels, 1]),
        input_node(2, &[batch, channels, 1]),
        TraceNode::new(
            3,
            "adain_leaky_relu_bounds".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu { eps, slope }),
            vec![0, 1, 2],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &input_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * channels * time,
    );

    // Compute analytical bound.
    let norm_bound = (time as f32).sqrt();
    let actual_gamma_max = gamma_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let actual_beta_max = beta_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let y_bound = (1.0 + actual_gamma_max) * norm_bound + actual_beta_max;
    let slope_factor = f32::max(1.0, slope as f32);
    let total_bound = slope_factor * y_bound;

    let max_abs = assert_within_analytical_bound("adain_leaky_relu", &result, total_bound, 1e-3);
    eprintln!(
        "adain_leaky_relu_bounds: max |out| = {max_abs:.4}, analytical bound = {total_bound:.4} \
         (norm={norm_bound:.2}, y={y_bound:.2}, slope_factor={slope_factor})"
    );
}

// =============================================================================
// AdaLayerNorm bounds:
// |out| ≤ (1 + |γ|_max) · (|w|_max · √H + |b|_max) + |β|_max
//
// LayerNorm normalizes over hidden dim H: |normed| ≤ √H.
// After LN affine: |ln| ≤ |w|·√H + |b|.
// After adaptive affine: |out| ≤ (1+|γ|)·|ln| + |β|.
// =============================================================================

/// AdaLayerNorm NativeOp: verify analytical bound.
#[test]
fn test_ada_layer_norm_nativeop_analytical_bound() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, time, hidden) = (2, 8, 32);
    let eps = 1e-5_f64;

    let x_data = super::test_utils::rand_f32_vec(0xB00D_3001, batch * time * hidden, -2.0, 2.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xB00D_3002, batch * hidden, -0.4, 0.4);
    let beta_data = super::test_utils::rand_f32_vec(0xB00D_3003, batch * hidden, -0.3, 0.3);
    let norm_w_data = super::test_utils::rand_f32_vec(0xB00D_3004, hidden, 0.8, 1.2);
    let norm_b_data = super::test_utils::rand_f32_vec(0xB00D_3005, hidden, -0.1, 0.1);

    let norm_weight = WeightRef::new(norm_w_data.clone(), vec![hidden]).expect("norm_weight");
    let norm_bias = WeightRef::new(norm_b_data.clone(), vec![hidden]).expect("norm_bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, time, hidden]),
        input_node(1, &[batch, 1, hidden]),
        input_node(2, &[batch, 1, hidden]),
        TraceNode::new(
            3,
            "ada_layer_norm_bounds".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdaLayerNorm {
                norm_weight,
                norm_bias,
                eps,
            }),
            vec![0, 1, 2],
            vec![batch, time, hidden],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * time * hidden,
    );

    // Compute analytical bound.
    let norm_bound = (hidden as f32).sqrt();
    let w_max = norm_w_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let b_max = norm_b_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let ln_bound = w_max * norm_bound + b_max;

    let gamma_max = gamma_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let beta_max = beta_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let total_bound = (1.0 + gamma_max) * ln_bound + beta_max;

    let max_abs = assert_within_analytical_bound("ada_layer_norm", &result, total_bound, 1e-3);
    eprintln!(
        "ada_layer_norm_bounds: max |out| = {max_abs:.4}, analytical bound = {total_bound:.4} \
         (norm=√{hidden}={norm_bound:.2}, ln={ln_bound:.2}, gamma_max={gamma_max:.2})"
    );
}

// =============================================================================
// Cumsum bounds: out[k] ∈ [(k+1)·x_min, (k+1)·x_max]
//
// cumsum(x)[k] = x[0] + x[1] + ... + x[k]. If x[i] ∈ [a, b] for all i,
// then cumsum[k] ∈ [(k+1)·min(a,b), (k+1)·max(a,b)] when a ≤ b.
// The last element's bound is [T·a, T·b].
// =============================================================================

/// Cumsum NativeOp: verify per-element prefix-sum bound.
#[test]
fn test_cumsum_nativeop_analytical_bound() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (1, 2, 64);
    let input_lo = -1.0_f32;
    let input_hi = 1.0_f32;

    let input_data =
        super::test_utils::rand_f32_vec(0xB00D_4001, batch * channels * time, input_lo, input_hi);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "cumsum_bounds".into(),
            TraceOp::Cumsum { dim: 2 },
            vec![0],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * channels * time,
    );

    // Verify per-element analytical bound.
    // For cumsum along dim=2 (time axis), element [b, c, t] is the sum of
    // input[b, c, 0..=t]. With inputs in [lo, hi], cumsum[t] ∈ [(t+1)*lo, (t+1)*hi].
    let mut max_violation: f32 = 0.0;
    for b in 0..batch {
        for c in 0..channels {
            for t in 0..time {
                let idx = (b * channels + c) * time + t;
                let val = result[idx];
                let prefix_len = (t + 1) as f32;
                let lo_bound = prefix_len * input_lo;
                let hi_bound = prefix_len * input_hi;

                assert!(
                    val.is_finite(),
                    "cumsum[{b},{c},{t}]: non-finite output {val}"
                );

                // Allow small FP tolerance proportional to prefix length.
                let tol = prefix_len * 1e-5;
                let violation = f32::max(lo_bound - tol - val, val - hi_bound - tol);
                if violation > 0.0 {
                    max_violation = max_violation.max(violation);
                }
                assert!(
                    val >= lo_bound - tol && val <= hi_bound + tol,
                    "cumsum[{b},{c},{t}]: {val} outside [{lo_bound}, {hi_bound}]"
                );
            }
        }
    }
    let last_lo = (time as f32) * input_lo;
    let last_hi = (time as f32) * input_hi;
    eprintln!(
        "cumsum_bounds: all elements within prefix-sum bounds. \
         Last-element bound: [{last_lo}, {last_hi}]. Max violation margin: {max_violation:.2e}"
    );
}

/// Cumsum along dim=0: verify bounds for a different axis.
#[test]
fn test_cumsum_nativeop_analytical_bound_dim0() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (16, 8);
    let input_lo = -0.5_f32;
    let input_hi = 0.5_f32;

    let input_data = super::test_utils::rand_f32_vec(0xB00D_4010, rows * cols, input_lo, input_hi);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "cumsum_dim0_bounds".into(),
            TraceOp::Cumsum { dim: 0 },
            vec![0],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        rows * cols,
    );

    // Cumsum along dim=0: result[r, c] = sum(input[0..=r, c]).
    // Bound: result[r, c] ∈ [(r+1)*lo, (r+1)*hi].
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let val = result[idx];
            assert!(val.is_finite(), "cumsum_dim0[{r},{c}]: non-finite {val}");
            let prefix_len = (r + 1) as f32;
            let lo_bound = prefix_len * input_lo;
            let hi_bound = prefix_len * input_hi;
            let tol = prefix_len * 1e-5;
            assert!(
                val >= lo_bound - tol && val <= hi_bound + tol,
                "cumsum_dim0[{r},{c}]: {val} outside [{lo_bound}, {hi_bound}]"
            );
        }
    }
    eprintln!("cumsum_dim0_bounds: all elements within prefix-sum bounds");
}
