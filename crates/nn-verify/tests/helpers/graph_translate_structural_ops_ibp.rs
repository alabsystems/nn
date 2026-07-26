// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IBP bounds propagation tests for structural tensor ops (Reshape, AxisSelect, Stack).
//!
//! Extracted from `graph_translate_structural_ops.rs` — verifies that IBP bounds
//! propagate correctly through structural ops, not just that graph nodes are created.
//!
//! Strengthened for #1684: exact tolerances for identity ops, output shape checks,
//! width preservation, per-element positional verification for Stack,
//! overflow/unbounded guards, constant-path tests.
//!
//! Advanced Reshape and Concat IBP tests extracted to
//! `graph_translate_structural_ops_ibp_reshape_concat.rs` for 500-line compliance.
//!
//! Stack IBP tests extracted to
//! `graph_translate_structural_ops_ibp_stack.rs` for 500-line compliance.
//!
//! Part of #1678, #1684, #1693.

use super::common;

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Reshape IBP
// ---------------------------------------------------------------------------

/// Reshape [2,6] → [2,3,2]: IBP bounds must propagate unchanged through Reshape
/// (same data, different shape). Input bounds [-3, 7] should appear in output
/// with exact preservation (Reshape is a metadata-only op).
#[test]
fn test_reshape_variable_ibp_bounds() {
    let def = TensorKernelDef::new(
        "reshape_ibp",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 6],
                },
                vec![2, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(0),
                    target_shape: vec![2, 3, 2],
                },
                vec![2, 3, 2],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph =
        tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("Reshape build");

    let lower = ArrayD::from_elem(IxDyn(&[2, 6]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 6]), 7.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through Reshape");

    // Structural: output bounds are valid (finite, lower <= upper).
    common::assert_bounds_valid(&output);

    // Reshape is identity for data — bounds must be exact, not approximate.
    let (lo, hi) = output.lower_upper();
    let eps = 1e-6;
    assert!(
        lo.iter().all(|&v| (v - (-3.0)).abs() < eps),
        "Reshape lower should be exactly -3, got min={:.6}",
        lo.iter().copied().reduce(f32::min).unwrap()
    );
    assert!(
        hi.iter().all(|&v| (v - 7.0).abs() < eps),
        "Reshape upper should be exactly 7, got max={:.6}",
        hi.iter().copied().reduce(f32::max).unwrap()
    );

    // Width preservation: input width = 10.0, output width should match.
    assert!(
        (output.max_width() - 10.0).abs() < eps,
        "Reshape max_width should be exactly 10.0, got {:.6}",
        output.max_width()
    );

    // No overflow or unbounded values after a pure reshape.
    assert!(!output.has_overflow(), "Reshape must not produce overflow");
    assert!(
        !output.has_unbounded(),
        "Reshape must not produce unbounded values"
    );
}

/// Reshape constant: constant [6] → [2,3] should fold to constant output.
#[test]
fn test_reshape_constant_ibp_bounds() {
    let def = TensorKernelDef::new(
        "reshape_const_ibp",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "c".to_string(),
                    shape: vec![6],
                },
                vec![6],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(0),
                    target_shape: vec![2, 3],
                },
                vec![2, 3],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(4.0)])
        .expect("Reshape constant build");

    // For a constant input, the bounds are the constant value ± tolerance.
    let lower = ArrayD::from_elem(IxDyn(&[6]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[6]), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through constant Reshape");
    common::assert_bounds_valid(&output);

    // Constant path: bounds should be finite and non-overflowing.
    assert!(!output.has_overflow(), "constant reshape must not overflow");
    assert!(
        !output.has_unbounded(),
        "constant reshape must not be unbounded"
    );
}

// ---------------------------------------------------------------------------
// AxisSelect IBP
// ---------------------------------------------------------------------------

/// AxisSelect axis=1, index=2 on [4,8]: IBP bounds from input [-2, 5] should
/// propagate through Slice+Squeeze with exact preservation (data subsetting only).
#[test]
fn test_axis_select_variable_ibp_bounds() {
    let def = TensorKernelDef::new(
        "axis_select_ibp",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 8],
                },
                vec![4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 1,
                    index: 2,
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph =
        tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("AxisSelect build");

    let lower = ArrayD::from_elem(IxDyn(&[4, 8]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4, 8]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through AxisSelect");

    // Structural validity.
    common::assert_bounds_valid(&output);

    // AxisSelect selects one slice — identity for data, bounds must be exact.
    let (lo, hi) = output.lower_upper();
    let eps = 1e-6;
    assert!(
        lo.iter().all(|&v| (v - (-2.0)).abs() < eps),
        "AxisSelect lower should be exactly -2, got min={:.6}",
        lo.iter().copied().reduce(f32::min).unwrap()
    );
    assert!(
        hi.iter().all(|&v| (v - 5.0).abs() < eps),
        "AxisSelect upper should be exactly 5, got max={:.6}",
        hi.iter().copied().reduce(f32::max).unwrap()
    );

    // Width preservation: input width = 7.0, output should match.
    assert!(
        (output.max_width() - 7.0).abs() < eps,
        "AxisSelect max_width should be exactly 7.0, got {:.6}",
        output.max_width()
    );

    assert!(
        !output.has_overflow(),
        "AxisSelect must not produce overflow"
    );
    assert!(
        !output.has_unbounded(),
        "AxisSelect must not produce unbounded values"
    );
}

/// AxisSelect on a higher-rank tensor: axis=2 on [3,4,5] with variable input.
/// Verifies dimension reduction works correctly for non-trivial axis.
#[test]
fn test_axis_select_higher_rank_ibp_bounds() {
    let def = TensorKernelDef::new(
        "axis_select_3d_ibp",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![3, 4, 5],
                },
                vec![3, 4, 5],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(0),
                    axis: 2,
                    index: 0,
                },
                vec![3, 4],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph =
        tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("AxisSelect 3D build");

    let lower = ArrayD::from_elem(IxDyn(&[3, 4, 5]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3, 4, 5]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3D AxisSelect");
    common::assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let eps = 1e-6;
    // Identity data subsetting: bounds must be exact.
    assert!(
        lo.iter().all(|&v| (v - (-10.0)).abs() < eps),
        "3D AxisSelect lower should be exactly -10"
    );
    assert!(
        hi.iter().all(|&v| (v - 10.0).abs() < eps),
        "3D AxisSelect upper should be exactly 10"
    );

    // Width preservation.
    assert!(
        (output.max_width() - 20.0).abs() < eps,
        "3D AxisSelect max_width should be exactly 20.0"
    );
}

// Stack IBP tests extracted to `graph_translate_structural_ops_ibp_stack.rs`
// for 500-line compliance. Part of #1693.
