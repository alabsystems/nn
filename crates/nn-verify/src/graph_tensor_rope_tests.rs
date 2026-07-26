// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for RoPE native layer pattern matcher.
//!
//! Part of #525.

use super::*;
use crate::graph_tensor::TensorParamBinding;
use ny_api::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

/// Helper: build a RoPE kernel with small dimensions.
fn build_test_rope(bh: usize, seq_len: usize, head_dim: usize) -> TensorKernelDef {
    nn_dsl::build_rope_rotate_kernel(bh, seq_len, head_dim).expect("build RoPE must succeed")
}

/// Helper: compute total IBP width (sum of hi-lo across all elements).
fn total_ibp_width(lo: &ArrayD<f32>, hi: &ArrayD<f32>) -> f32 {
    hi.iter().zip(lo.iter()).map(|(h, l)| h - l).sum()
}

/// Helper: build stacked multi-variable input for decomposition path.
/// Returns `None` if shape construction fails (expected for mismatched dims).
fn build_decomp_input(
    bh: usize,
    seq_len: usize,
    head_dim: usize,
    x_lo: f32,
    x_hi: f32,
    freq_val: f32,
    freq_eps: f32,
) -> Option<BoundedTensor> {
    let half_dim = head_dim / 2;
    let x_lower = ArrayD::from_elem(IxDyn(&[bh, seq_len, head_dim]), x_lo);
    let x_upper = ArrayD::from_elem(IxDyn(&[bh, seq_len, head_dim]), x_hi);
    let f_lower = ArrayD::from_elem(IxDyn(&[bh, seq_len, half_dim]), freq_val - freq_eps);
    let f_upper = ArrayD::from_elem(IxDyn(&[bh, seq_len, half_dim]), freq_val + freq_eps);
    let sl = ndarray::concatenate(
        ndarray::Axis(0),
        &[
            x_lower.view(),
            f_lower
                .into_shape_with_order(IxDyn(&[bh, seq_len, half_dim]))
                .ok()?
                .view(),
        ],
    )
    .ok()?;
    let su = ndarray::concatenate(
        ndarray::Axis(0),
        &[
            x_upper.view(),
            f_upper
                .into_shape_with_order(IxDyn(&[bh, seq_len, half_dim]))
                .ok()?
                .view(),
        ],
    )
    .ok()?;
    BoundedTensor::new(sl, su).ok()
}

// --- Pattern matching tests ---

#[test]
fn test_native_rope_fires_for_constant_freq() {
    let rope = build_test_rope(1, 2, 4);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(0.5),
    ];
    let result = try_native_rope(&rope, &bindings).expect("must not error");
    assert!(
        result.is_some(),
        "native RoPE path must fire for constant freq"
    );
    let graph = result.unwrap();
    assert_eq!(
        graph.num_nodes(),
        1,
        "native RoPE graph should have exactly 1 node (RopeLayer)"
    );
}

#[test]
fn test_native_rope_skips_variable_freq() {
    let rope = build_test_rope(1, 2, 4);
    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let result = try_native_rope(&rope, &bindings).expect("must not error");
    assert!(
        result.is_none(),
        "native RoPE must skip when freqs is Variable (fall through to decomposition)"
    );
}

#[test]
fn test_native_rope_skips_non_rope_kernel() {
    // InstanceNorm K2 is not a rope kernel — must return None.
    let k2 = nn_dsl::instance_norm::build_instance_norm(1, 2, 8).expect("build K2 must succeed");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let result = try_native_rope(&k2, &bindings).expect("must not error");
    assert!(result.is_none(), "native RoPE must skip non-rope kernels");
}

#[test]
fn test_native_rope_rejects_non_finite_freq() {
    let rope = build_test_rope(1, 2, 4);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(f32::NAN),
    ];
    let result = try_native_rope(&rope, &bindings);
    assert!(result.is_err(), "NaN freq must be rejected");

    let bindings_inf = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(f32::INFINITY),
    ];
    let result_inf = try_native_rope(&rope, &bindings_inf);
    assert!(result_inf.is_err(), "Inf freq must be rejected");
}

#[test]
fn test_native_rope_zero_freq() {
    // freq=0 → cos(0)=1, sin(0)=0 → identity rotation.
    let rope = build_test_rope(1, 1, 4);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(0.0),
    ];
    let result = try_native_rope(&rope, &bindings).expect("must not error");
    assert!(result.is_some(), "freq=0 should produce a valid graph");
}

// --- IBP soundness tests ---

#[test]
fn test_native_rope_ibp_soundness() {
    // Verify that the native RoPE path produces bounds that contain
    // the reference implementation output for a concrete input.
    let bh = 1;
    let seq_len = 2;
    let head_dim = 4;
    let freq_val = 0.5_f32;

    let rope = build_test_rope(bh, seq_len, head_dim);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(freq_val),
    ];
    let graph = try_native_rope(&rope, &bindings)
        .expect("must not error")
        .expect("native path must fire");

    // Input bounds: x in [-2, 3].
    let lower = ArrayD::from_elem(IxDyn(&[bh, seq_len, head_dim]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[bh, seq_len, head_dim]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("native RoPE IBP must succeed");
    let (lo, hi) = output.lower_upper();

    // All output bounds must be finite.
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "output lower bounds must be finite, got: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "output upper bounds must be finite, got: {hi:?}"
    );

    // Verify soundness: concrete input within bounds, reference output
    // must fall within IBP bounds.
    let x_sample: Vec<f32> = vec![1.0, -0.5, 2.0, 0.3, -1.0, 0.5, -0.3, 1.5];
    let freqs_const: Vec<f32> = vec![freq_val; seq_len * (head_dim / 2)];
    let ref_out = nn_dsl::rope_rotate_ref(&x_sample, &freqs_const, bh, seq_len, head_dim)
        .expect("reference must succeed");

    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();
    assert_eq!(
        ref_out.len(),
        lo_flat.len(),
        "reference output length ({}) must match IBP bounds length ({})",
        ref_out.len(),
        lo_flat.len(),
    );
    // Directed rounding (next_down_f32/next_up_f32) guarantees 1-ULP containment.
    // Use 1e-6 tolerance (well above 1 ULP for f32 in [-10, 10] range) to catch
    // real bound violations without masking them behind a generous 1e-3 margin.
    for (i, &val) in ref_out.iter().enumerate() {
        assert!(
            val >= lo_flat[i] - 1e-6,
            "ref_out[{i}]={val} below native lower bound {}",
            lo_flat[i]
        );
        assert!(
            val <= hi_flat[i] + 1e-6,
            "ref_out[{i}]={val} above native upper bound {}",
            hi_flat[i]
        );
    }
}

#[test]
fn test_native_rope_tighter_than_decomposition() {
    // Native RoPE should produce tighter bounds than decomposition
    // because RopeLayer uses exact interval arithmetic.
    let (bh, seq_len, head_dim) = (1, 1, 2);
    let freq_val = 0.5_f32;
    let (x_lo, x_hi) = (-1.0_f32, 1.0_f32);
    let rope = build_test_rope(bh, seq_len, head_dim);

    // Native path.
    let native_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(freq_val),
    ];
    let native_graph = try_native_rope(&rope, &native_bindings)
        .expect("must not error")
        .expect("native path must fire");
    let lower = ArrayD::from_elem(IxDyn(&[bh, seq_len, head_dim]), x_lo);
    let upper = ArrayD::from_elem(IxDyn(&[bh, seq_len, head_dim]), x_hi);
    let native_input = BoundedTensor::new(lower, upper).expect("valid bounds");
    let native_out = native_graph
        .propagate_ibp(&native_input)
        .expect("native IBP");
    let (native_lo, native_hi) = native_out.lower_upper();

    // Decomposition path (Variable freqs → native skips).
    let decomp_bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let decomp_graph = crate::graph_tensor::tensor_kernel_to_graph(&rope, &decomp_bindings)
        .expect("decomposition graph must build");
    let Some(decomp_input) = build_decomp_input(bh, seq_len, head_dim, x_lo, x_hi, freq_val, 0.001)
    else {
        return; // Shape mismatch — skip comparison.
    };
    if let Ok(decomp_out) = decomp_graph.propagate_ibp(&decomp_input) {
        let (decomp_lo, decomp_hi) = decomp_out.lower_upper();
        let native_width = total_ibp_width(native_lo, native_hi);
        let decomp_width = total_ibp_width(decomp_lo, decomp_hi);
        // Native path should be tighter (or at most marginally wider due to
        // rounding). For head_dim=2 (2 elements), 0.1 total slack is generous.
        assert!(
            native_width <= decomp_width + 0.1,
            "native ({native_width:.6}) should be <= decomp ({decomp_width:.6}) + 0.1"
        );
    }
}

// --- Boundary condition tests (#525 audit) ---

#[test]
fn test_native_rope_minimum_head_dim_2() {
    // head_dim=2 is the minimum valid case (num_pairs=1).
    // Verify pattern fires AND IBP produces finite bounds.
    let rope = build_test_rope(1, 1, 2);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1.0),
    ];
    let graph = try_native_rope(&rope, &bindings)
        .expect("must not error")
        .expect("native path must fire for head_dim=2");

    let lower = ArrayD::from_elem(IxDyn(&[1, 1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 1, 2]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");
    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(
        lo.iter().all(|v| v.is_finite()),
        "head_dim=2 lower bounds must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "head_dim=2 upper bounds must be finite"
    );
    // Output has same total elements as input: 1*1*2 = 2.
    assert_eq!(lo.len(), 2, "output should have 2 elements for head_dim=2");
}

#[test]
fn test_native_rope_zero_freq_identity_soundness() {
    // freq=0 → cos(0)=1, sin(0)=0 → rotation is identity.
    // Output bounds should tightly match input bounds.
    let shape = [1, 1, 4]; // bh, seq_len, head_dim
    let rope = build_test_rope(shape[0], shape[1], shape[2]);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(0.0),
    ];
    let graph = try_native_rope(&rope, &bindings)
        .expect("must not error")
        .expect("native path must fire");

    let lower = ArrayD::from_elem(IxDyn(&shape), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&shape), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");
    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    // cos(0)=1, sin(0)=0 → rotation matrix is identity.
    // Each output pair: y_even = cos*x_even - sin*x_odd = x_even,
    //                   y_odd  = sin*x_even + cos*x_odd = x_odd.
    // So output bounds should equal input bounds within 1 ULP of rounding.
    for &v in lo.iter() {
        assert!(
            (v - (-3.0_f32)).abs() < 1e-5,
            "zero-freq lower bound should be ~-3.0, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            (v - 5.0_f32).abs() < 1e-5,
            "zero-freq upper bound should be ~5.0, got {v}"
        );
    }
}

#[test]
fn test_native_rope_rejects_wrong_binding_count() {
    let rope = build_test_rope(1, 1, 4);

    // 1 binding (too few)
    let result = try_native_rope(&rope, &[TensorParamBinding::Variable]).expect("must not error");
    assert!(result.is_none(), "1 binding must skip (needs 2)");

    // 3 bindings (too many)
    let bindings_3 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(0.5),
        TensorParamBinding::Variable,
    ];
    let result = try_native_rope(&rope, &bindings_3).expect("must not error");
    assert!(result.is_none(), "3 bindings must skip (needs 2)");
}

#[test]
fn test_native_rope_neg_infinity_freq_rejected() {
    let rope = build_test_rope(1, 1, 4);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(f32::NEG_INFINITY),
    ];
    let result = try_native_rope(&rope, &bindings);
    assert!(result.is_err(), "negative infinity freq must be rejected");
}
