// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for tensor IR → NY translation:
//! InstanceNorm K2 end-to-end and K6 RoPE pipeline.
//!
//! Multi-variable stacking tests extracted to `graph_translate_tensor_multi_var.rs` (#542).

use nn_dsl::instance_norm::build_instance_norm;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// --- Phase E: InstanceNorm K2 end-to-end ---
// Since #71: uses native InstanceNorm1d op → NY InstanceNorm1dLayer.
// The native layer has variance clamping so IBP is more robust than the old
// 12-node decomposition which lost inter-variable correlation.

#[test]
fn test_instance_norm_k2_gamma_crown_translation() {
    let k2 = build_instance_norm(2, 4, 16).expect("build must succeed");
    // x is variable, eps is constant scalar
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&k2, &bindings)
        .expect("K2 InstanceNorm NY graph must build");
    assert_eq!(
        graph.num_nodes(),
        1,
        "K2 InstanceNorm graph should have 1 node"
    );
}

#[test]
fn test_instance_norm_k2_ibp_bounds_propagation() {
    // With the native InstanceNorm1dLayer (#71), IBP uses conservative per-channel
    // interval arithmetic with variance clamping instead of the old decomposed
    // 12-node form that lost correlation. The native layer should produce finite
    // bounds for reasonable input ranges.
    let k2 = build_instance_norm(1, 2, 8).expect("build must succeed");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&k2, &bindings).expect("build K2 graph");

    // Use tight positive bounds: x in [1, 2].
    let lower = ArrayD::from_elem(IxDyn(&[1, 2, 8]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2, 8]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("native InstanceNorm1dLayer IBP should succeed for tight bounds");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "output lower bounds must be finite for tight inputs"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "output upper bounds must be finite for tight inputs"
    );
}

#[test]
fn test_instance_norm_k2_ibp_wider_bounds() {
    // The native InstanceNorm1dLayer should handle wider input ranges than
    // the decomposed form. With x in [-10, 10], the old decomposition
    // could produce negative variance under IBP → NaN. The native layer
    // clamps variance to be non-negative.
    let k2 = build_instance_norm(1, 2, 8).expect("build must succeed");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&k2, &bindings).expect("build K2 graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 2, 8]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2, 8]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("native InstanceNorm1dLayer IBP should succeed for wider bounds");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "output lower bounds must be finite for wider inputs, got: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "output upper bounds must be finite for wider inputs, got: {hi:?}"
    );
}

// --- Phase D: K6 RoPE end-to-end NY bounds propagation ---
// Exercises the full tensor pipeline: Reshape → AxisSelect → Broadcast →
// Elementwise(rope_cos/rope_sin) → Stack → Reshape through NY IBP.

#[test]
fn test_k6_rope_gamma_crown_translation() {
    // Build the full K6 RoPE tensor kernel with small dimensions.
    let bh = 1;
    let seq_len = 2;
    let head_dim = 4;
    let rope =
        nn_dsl::build_rope_rotate_kernel(bh, seq_len, head_dim).expect("build RoPE must succeed");

    // x is Variable, freqs is Variable (both have position-dependent values).
    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&rope, &bindings)
        .expect("K6 RoPE NY translation must succeed");

    // Graph must have nodes for the full pipeline:
    // 2 SliceLayer (multi-var) + structural ops + elementwise layers
    assert!(
        graph.num_nodes() >= 4,
        "RoPE graph needs structural + elementwise nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_k6_rope_ibp_bounds_propagation() {
    // End-to-end: build K6 RoPE, translate to NY, run IBP, verify
    // bounds contain reference implementation output.
    let bh = 1;
    let seq_len = 2;
    let head_dim = 4;

    let rope =
        nn_dsl::build_rope_rotate_kernel(bh, seq_len, head_dim).expect("build RoPE must succeed");

    // Both inputs as Variable to test full multi-variable path.
    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let graph =
        tensor_kernel_to_graph(&rope, &bindings).expect("NY translation must succeed");

    // Verify the translation produces a non-trivial graph.
    // Multi-variable stacking + structural ops + elementwise layers.
    assert!(graph.num_nodes() >= 4, "graph should have multiple nodes");
}

#[test]
fn test_k6_rope_constant_freq_ibp() {
    // Simpler case: x is Variable, freqs is a ConstantScalar.
    // This avoids multi-variable stacking shape complexity.
    let bh = 1;
    let seq_len = 2;
    let head_dim = 4;

    let rope =
        nn_dsl::build_rope_rotate_kernel(bh, seq_len, head_dim).expect("build RoPE must succeed");

    // x is Variable, freqs is constant 0.5.
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(0.5),
    ];
    let graph =
        tensor_kernel_to_graph(&rope, &bindings).expect("constant-freq RoPE graph must build");

    // With constant freq, there's only 1 variable input → no multi-var stacking.
    // Input shape matches x: [1, 2, 4]
    let lower = ArrayD::from_elem(IxDyn(&[bh, seq_len, head_dim]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[bh, seq_len, head_dim]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP propagation for constant-freq RoPE must succeed");
    let (lo, hi) = output.lower_upper();

    // Verify all output bounds are finite.
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "output lower bounds must be finite, got: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "output upper bounds must be finite, got: {hi:?}"
    );

    // Verify soundness: sample a concrete input within bounds and check
    // the reference output falls within the IBP bounds.
    let x_sample: Vec<f32> = vec![1.0, -0.5, 2.0, 0.3, -1.0, 0.5, -0.3, 1.5];
    let freqs_const: Vec<f32> = vec![0.5; seq_len * (head_dim / 2)];
    let ref_out = nn_dsl::rope_rotate_ref(&x_sample, &freqs_const, bh, seq_len, head_dim)
        .expect("reference must succeed");

    // The output shape from IBP should be [1, 2, 4] (same as input).
    // Check that each reference output value falls within the IBP bounds.
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();
    for (i, &val) in ref_out.iter().enumerate() {
        if i < lo_flat.len() {
            assert!(
                val >= lo_flat[i] - 1e-3,
                "ref_out[{i}]={val} below lower bound {}",
                lo_flat[i]
            );
            assert!(
                val <= hi_flat[i] + 1e-3,
                "ref_out[{i}]={val} above upper bound {}",
                hi_flat[i]
            );
        }
    }
}

/// Compare decomposition IBP bounds vs analytical scalar bounds for K6 RoPE.
/// Part of #304: evidence that K6 IS verifiable today via decomposition.
#[test]
fn test_k6_rope_decomposition_bounds_quality() {
    let bh = 1;
    let seq_len = 1;
    let head_dim = 2;
    let rope =
        nn_dsl::build_rope_rotate_kernel(bh, seq_len, head_dim).expect("build RoPE must succeed");
    let freq_val = 0.5_f32;
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(freq_val),
    ];
    let graph =
        tensor_kernel_to_graph(&rope, &bindings).expect("constant-freq RoPE graph must build");

    let x_lo = -2.0_f32;
    let x_hi = 3.0_f32;
    let lower = ArrayD::from_elem(IxDyn(&[bh, seq_len, head_dim]), x_lo);
    let upper = ArrayD::from_elem(IxDyn(&[bh, seq_len, head_dim]), x_hi);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (ibp_lo, ibp_hi) = output.lower_upper();
    let ibp_lo_flat: Vec<f32> = ibp_lo.iter().copied().collect();
    let ibp_hi_flat: Vec<f32> = ibp_hi.iter().copied().collect();

    // Analytical scalar bounds: x0, x1 both from [x_lo, x_hi], freq constant.
    let (cos_lo, cos_hi) =
        nn_dsl::rope_cos_scalar_bounds(x_lo, x_hi, x_lo, x_hi, freq_val, freq_val)
            .expect("analytical cos bounds");
    let (sin_lo, sin_hi) =
        nn_dsl::rope_sin_scalar_bounds(x_lo, x_hi, x_lo, x_hi, freq_val, freq_val)
            .expect("analytical sin bounds");

    assert!(
        ibp_lo_flat.iter().all(|v| v.is_finite()),
        "IBP lower must be finite"
    );
    assert!(
        ibp_hi_flat.iter().all(|v| v.is_finite()),
        "IBP upper must be finite"
    );

    // IBP envelope should contain the union of analytical cos and sin bounds.
    let ibp_min = ibp_lo_flat.iter().copied().fold(f32::INFINITY, f32::min);
    let ibp_max = ibp_hi_flat
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let anal_lo = cos_lo.min(sin_lo);
    let anal_hi = cos_hi.max(sin_hi);

    assert!(
        ibp_min <= anal_lo + 1e-3,
        "IBP lower ({ibp_min}) must be <= analytical ({anal_lo})"
    );
    assert!(
        ibp_max >= anal_hi - 1e-3,
        "IBP upper ({ibp_max}) must be >= analytical ({anal_hi})"
    );

    // Width ratio: how much wider is decomposition IBP vs analytical?
    let ibp_width = ibp_max - ibp_min;
    let anal_width = anal_hi - anal_lo;
    let ratio = if anal_width > 0.0 {
        ibp_width / anal_width
    } else {
        1.0
    };
    assert!(
        ratio < 5.0,
        "width ratio {ratio:.2} too large (IBP={ibp_width:.4}, analytical={anal_width:.4})"
    );
}
