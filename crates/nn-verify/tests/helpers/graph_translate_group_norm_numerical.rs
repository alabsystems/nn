// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Numerical correctness and pipeline integration tests for GroupNorm(1) decomposition.
//!
//! Extracted from `graph_translate_group_norm.rs` to stay under the 500-line limit.
//! Part of #705 (numerical correctness) and #703 (pipeline API).

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    tensor_kernel_to_graph, verify_tensor_and_record, BoundedTensor, TensorParamBinding,
    VerifyStatus,
};
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

// ---------------------------------------------------------------------------
// Numerical correctness tests (#705)
// ---------------------------------------------------------------------------

/// AC1: GroupNorm(1) on point input (all elements identical) should output
/// exactly 0.0 — normalized = (x - mean) / sqrt(var + eps) = 0 when all x equal.
#[test]
fn test_group_norm_g1_point_input_zero_output() {
    let def = group_norm_g1_kernel(2, 4);
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // All elements = 0.5 (point input: lower == upper)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), 0.5f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 0.5f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // For identical elements: mean = 0.5, var = 0, normalized = 0
    // IBP on a point input should produce exact (or near-exact) results.
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            l.abs() < 1e-3,
            "lower bound should be ~0.0 for constant input, got {l}"
        );
        assert!(
            u.abs() < 1e-3,
            "upper bound should be ~0.0 for constant input, got {u}"
        );
    }
    assert!(
        output.max_width() < 1e-3,
        "point input should give near-zero width, got {}",
        output.max_width()
    );
}

/// AC1: GroupNorm(1) with affine on point input: output = gamma * 0 + beta = beta.
#[test]
fn test_group_norm_g1_affine_point_input_equals_beta() {
    let channels = 2;
    let time_len = 4;
    let def = group_norm_g1_affine_kernel(channels, time_len);
    let gamma = 2.0f32;
    let beta = 0.5f32;
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(gamma),
        TensorParamBinding::ConstantScalar(beta),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Point input: all 1.0
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[channels, time_len]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[channels, time_len]), 1.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // normalized = 0, output = gamma * 0 + beta = 0.5
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            (l - beta).abs() < 1e-2,
            "lower should be ~{beta} for constant input, got {l}"
        );
        assert!(
            (u - beta).abs() < 1e-2,
            "upper should be ~{beta} for constant input, got {u}"
        );
    }
}

/// AC2: Decomposed GroupNorm IBP bounds are vacuously wide (±20B) for [-1, 1]
/// input due to IBP losing correlations at each primitive op (design doc #697).
/// Rather than asserting tight bounds (which would be incorrect for IBP on
/// decomposed norms), we verify:
/// 1. Bounds are symmetric around 0 (GroupNorm centers output)
/// 2. Width is finite but documented as very wide
/// 3. Affine gamma=2 produces ~2x the width of non-affine
#[test]
fn test_group_norm_g1_ibp_bounds_symmetry_and_scaling() {
    let def = group_norm_g1_kernel(2, 4);
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // All output elements should have identical bounds (uniform symmetric input).
    let lo0 = lo[[0, 0]];
    let hi0 = hi[[0, 0]];
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert_eq!(
            l, lo0,
            "all lower bounds should be identical for uniform input"
        );
        assert_eq!(
            u, hi0,
            "all upper bounds should be identical for uniform input"
        );
    }

    // IBP bounds should be symmetric around 0 for symmetric input.
    // GroupNorm centers the output, so for input [-1, 1], output center is 0.
    assert!(
        (lo0 + hi0).abs() < 1.0,
        "bounds should be approximately symmetric: lo={lo0}, hi={hi0}, sum={}",
        lo0 + hi0
    );

    // Record the non-affine width for comparison with affine.
    let non_affine_width = output.max_width();

    // Affine: gamma=2 should roughly double the width.
    let def2 = group_norm_g1_affine_kernel(2, 4);
    let bindings2 = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(2.0),
        TensorParamBinding::ConstantScalar(0.5),
    ];
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("graph");
    let input2 = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32),
    )
    .expect("valid bounds");
    let output2 = graph2.propagate_ibp(&input2).expect("IBP propagation");
    let affine_width = output2.max_width();

    // gamma=2 should approximately double the bounds width.
    // Allow some tolerance for the additive beta=0.5 contribution.
    let ratio = affine_width / non_affine_width;
    assert!(
        ratio > 1.8 && ratio < 2.2,
        "gamma=2 should ~double width: non_affine={non_affine_width}, affine={affine_width}, ratio={ratio}"
    );
}

/// AC1 (#714): GroupNorm(1) with ConstantTensor bindings for gamma/beta.
/// Previously failed with "BinaryMul does not support weight tensor operands".
/// ConstantTensor → WeightTensor in graph translation, now handled by
/// MulConstantLayer/AddConstantLayer with per-channel weight arrays.
#[test]
fn test_group_norm_g1_affine_constant_tensor_bindings() {
    let channels = 4;
    let time_len = 8;
    let def = group_norm_g1_affine_kernel(channels, time_len);

    // Per-channel gamma/beta as ConstantTensor (not uniform ConstantScalar).
    let gamma_data: Vec<f32> = (0..channels).map(|i| 1.0 + 0.5 * i as f32).collect();
    let beta_data: Vec<f32> = (0..channels).map(|i| 0.1 * i as f32).collect();

    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[channels]), gamma_data).expect("gamma shape"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[channels]), beta_data).expect("beta shape"),
        ),
    ];

    // This must succeed — previously returned UnsupportedOp.
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("ConstantTensor affine GroupNorm graph");

    assert!(
        graph.num_nodes() >= 4,
        "graph should have multiple nodes, got {}",
        graph.num_nodes()
    );

    // IBP propagation must succeed with per-channel weights.
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[channels, time_len]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[channels, time_len]), 1.0f32),
    )
    .expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through affine GroupNorm");
    let (lo, hi) = output.lower_upper();

    for &val in lo.iter() {
        assert!(val.is_finite(), "lower bound must be finite, got {val}");
    }
    for &val in hi.iter() {
        assert!(val.is_finite(), "upper bound must be finite, got {val}");
    }
}

/// AC3: Constant-input test (all elements = 1.0). GroupNorm normalizes to 0.
/// This is the strongest numerical test: exact expected output.
#[test]
fn test_group_norm_g1_constant_input_normalizes_to_zero() {
    let def = group_norm_g1_kernel(4, 8);
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Constant input: all 1.0
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4, 8]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[4, 8]), 1.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // GroupNorm(1, constant=1.0): mean=1.0, var=0.0
    // normalized = (1.0 - 1.0) / sqrt(0.0 + eps) = 0.0
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            l.abs() < 1e-3,
            "lower bound should be ~0 for constant input, got {l}"
        );
        assert!(
            u.abs() < 1e-3,
            "upper bound should be ~0 for constant input, got {u}"
        );
    }
}

// ---------------------------------------------------------------------------
// Pipeline API tests (#703)
// ---------------------------------------------------------------------------

#[test]
fn test_group_norm_g1_verify_tensor_pipeline() {
    let def = group_norm_g1_kernel(2, 4);
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32),
    )
    .expect("valid bounds");

    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(&mut status, &def, &bindings, &input, None)
        .expect("tensor pipeline should succeed");

    // Verification result should be finite.
    assert!(
        result.verification.is_finite,
        "output bounds must be finite"
    );
    assert_eq!(result.num_variables, 1);

    // Output tensor bounds should match input shape [C, T].
    let (lo, hi) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[2, 4]);
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} <= upper {u}");
    }

    // Status should have an entry for this kernel.
    assert!(
        status.kernel("group_norm_g1_test").is_some(),
        "status should contain entry for kernel"
    );
}

#[test]
fn test_group_norm_g1_affine_verify_tensor_pipeline() {
    let channels = 2;
    let time_len = 4;
    let def = group_norm_g1_affine_kernel(channels, time_len);
    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(2.0),
        TensorParamBinding::ConstantScalar(0.5),
    ];
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[channels, time_len]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[channels, time_len]), 1.0f32),
    )
    .expect("valid bounds");

    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("group_norm_g1_affine"),
    )
    .expect("affine tensor pipeline should succeed");

    assert!(
        result.verification.is_finite,
        "affine output bounds must be finite"
    );
    assert_eq!(result.num_variables, 1);

    // Custom status key should be used.
    assert!(
        status.kernel("group_norm_g1_affine").is_some(),
        "status should use custom key"
    );
}
