// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: GroupNorm(groups=1) decomposition via Reshape + primitives.
//!
//! Part of #642, #697: verifies decomposition soundness for NY bounds propagation.
//! GroupNorm g1 is now decomposed into Reduce/Broadcast/Elementwise primitives
//! (no InstanceNorm1d nodes).
//!
//! Numerical correctness and pipeline tests are in `graph_translate_group_norm_numerical.rs`.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a GroupNorm(1, C) decomposition kernel using the block builder.
fn group_norm_g1_kernel(channels: usize, time_len: usize) -> nn_dsl::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("group_norm_g1_test");
    let x = b.add_input("x", &[channels, time_len]);
    let eps = b.add_input("eps", &[1]);

    let out = b.add_group_norm_g1(x, eps, None, None, channels, time_len);
    b.build(out).expect("valid graph")
}

/// Build a GroupNorm(1, C) with affine parameters.
fn group_norm_g1_affine_kernel(channels: usize, time_len: usize) -> nn_dsl::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("group_norm_g1_affine_test");
    let x = b.add_input("x", &[channels, time_len]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[channels]);
    let beta = b.add_input("beta", &[channels]);

    let out = b.add_group_norm_g1(x, eps, Some(gamma), Some(beta), channels, time_len);
    b.build(out).expect("valid graph")
}

#[test]
fn test_group_norm_g1_validates() {
    // build() now auto-validates, so just check it succeeds.
    let _def = group_norm_g1_kernel(4, 8);
}

#[test]
fn test_group_norm_g1_node_count() {
    let def = group_norm_g1_kernel(4, 8);
    // 2 inputs + reshape + 10 norm primitives + reshape = 14 nodes
    assert_eq!(
        def.nodes.len(),
        14,
        "no-affine group_norm should have 14 nodes"
    );
}

#[test]
fn test_group_norm_g1_affine_node_count() {
    let def = group_norm_g1_affine_kernel(4, 8);
    // 4 inputs + reshape + 10 norm primitives + reshape
    // + broadcast(gamma) + binary_mul + broadcast(beta) + binary_add = 20 nodes
    assert_eq!(
        def.nodes.len(),
        20,
        "affine group_norm should have 20 nodes"
    );
}

#[test]
fn test_group_norm_g1_output_shape() {
    let def = group_norm_g1_kernel(4, 8);
    let output = &def.nodes[def.output.index()];
    assert_eq!(
        output.shape,
        vec![4, 8],
        "output shape should match input [C, T]"
    );
}

#[test]
fn test_group_norm_g1_builds_gamma_crown_graph() {
    let def = group_norm_g1_kernel(2, 4);
    let bindings = [
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("GroupNorm(1) decomposition should build NY graph");

    // Graph should have nodes for Reshape + decomposed norm primitives + Reshape.
    assert!(
        graph.num_nodes() >= 2,
        "graph should have multiple nodes for decomposed GroupNorm"
    );
}

#[test]
fn test_group_norm_g1_ibp_bounds_finite() {
    // GroupNorm normalizes to zero mean, unit variance (approximately).
    // IBP bounds should be finite for finite inputs.
    let def = group_norm_g1_kernel(2, 4);
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GroupNorm graph should build");

    // Input bounds: x ∈ [-1, 1] for all elements
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // All output bounds should be finite
    for &val in lo.iter() {
        assert!(val.is_finite(), "lower bound {val} must be finite");
    }
    for &val in hi.iter() {
        assert!(val.is_finite(), "upper bound {val} must be finite");
    }

    // Decomposed GroupNorm (10 primitive ops) produces wider IBP bounds than
    // native InstanceNorm1d because each Reduce/Broadcast/Elementwise step
    // independently widens the interval. This is IBP's known weakness on
    // multi-op chains, not a bug; CROWN propagation gives tighter results.
    let max_width = output.max_width();
    assert!(
        max_width.is_finite(),
        "output width must be finite, got {max_width}"
    );
    // The IBP bound is loose but the engine has been tightened over time (the
    // old "vacuously wide >1e6" expectation is stale; observed ~1.3e3 here).
    // The true GroupNorm(g=1) output envelope over n = C*T = 8 elements is
    // |z| <= sqrt(n) ~= 2.83, so any sound bound must be at least that wide;
    // the decomposed IBP over-approximates it but stays well under a generous
    // looseness ceiling. We assert a sound interval rather than a brittle exact
    // value (a further ny norm-envelope tightening may shrink this further).
    let envelope = (2 * 4) as f32; // n = C*T; sqrt(n) is the true z-score scale
    assert!(
        max_width >= envelope.sqrt(),
        "IBP width must cover the true |z| <= sqrt(n) ~= {} envelope, got {max_width}",
        envelope.sqrt()
    );
    assert!(
        max_width <= 1e5,
        "IBP width should stay under the decomposed-norm looseness ceiling, got {max_width}"
    );
}

#[test]
fn test_group_norm_g1_affine_validates() {
    // build() now auto-validates, so just check it succeeds.
    let _def = group_norm_g1_affine_kernel(4, 8);
}

#[test]
fn test_group_norm_g1_affine_output_shape() {
    let def = group_norm_g1_affine_kernel(4, 8);
    let output = &def.nodes[def.output.index()];
    assert_eq!(
        output.shape,
        vec![4, 8],
        "affine output shape should match input [C, T]"
    );
}

#[test]
fn test_group_norm_g1_affine_builds_gamma_crown_graph() {
    let def = group_norm_g1_affine_kernel(2, 4);
    let bindings = [
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantScalar(1.0),  // gamma (uniform)
        TensorParamBinding::ConstantScalar(0.0),  // beta (zero)
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("GroupNorm(1) affine decomposition should build NY graph");

    // Affine graph has more layers: Reshape + InstanceNorm + Reshape + MulBinary + Add
    assert!(
        graph.num_nodes() >= 4,
        "affine graph should have nodes for decomposed GroupNorm + affine"
    );
}

#[test]
fn test_group_norm_g1_affine_ibp_bounds_finite() {
    let channels = 2;
    let time_len = 4;
    let def = group_norm_g1_affine_kernel(channels, time_len);

    // gamma=2.0, beta=0.5: output = 2.0 * normalized + 0.5
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(2.0),
        TensorParamBinding::ConstantScalar(0.5),
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("affine GroupNorm graph should build");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[channels, time_len]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[channels, time_len]), 1.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    for &val in lo.iter() {
        assert!(val.is_finite(), "affine lower bound {val} must be finite");
    }
    for &val in hi.iter() {
        assert!(val.is_finite(), "affine upper bound {val} must be finite");
    }

    // Decomposed affine GroupNorm has wider IBP bounds than non-affine because
    // gamma=2 amplifies the decomposed norm width (~2x the non-affine width).
    let max_width = output.max_width();
    assert!(
        max_width.is_finite(),
        "affine output width must be finite, got {max_width}"
    );
    // The old "vacuously wide >1e6" expectation is stale; the engine tightened
    // (observed ~2.5e3 here). The true affine envelope over n = C*T = 8 with
    // gamma=2 is |gamma|*sqrt(n) ~= 2*2.83 = 5.66, so a sound bound is at least
    // that wide; the decomposed IBP over-approximates but stays under a generous
    // looseness ceiling. Assert a sound interval, not a brittle exact value (a
    // further ny norm-envelope tightening may shrink this further).
    let n = (2 * 4) as f32; // C*T
    let affine_envelope = 2.0 * n.sqrt(); // |gamma| * sqrt(n)
    assert!(
        max_width >= affine_envelope,
        "affine IBP width must cover |gamma|*sqrt(n) ~= {affine_envelope} envelope, got {max_width}"
    );
    assert!(
        max_width <= 2e5,
        "affine IBP width should stay under the decomposed-norm looseness ceiling, got {max_width}"
    );
}
