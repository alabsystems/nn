// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IBP bounds propagation tests for Reshape (advanced) and Concat structural ops.
//!
//! Extracted from `graph_translate_structural_ops_ibp.rs` for 500-line compliance.
//! Part B: advanced reshape (flatten, non-uniform) and Concat IBP tests.

use super::common;

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Reshape: flatten 3D → 1D. Verifies total element count is preserved
/// and bounds are exactly maintained through a non-trivial reshape.
#[test]
fn test_reshape_flatten_ibp_bounds() {
    let def = TensorKernelDef::new(
        "reshape_flatten_ibp",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 3, 4],
                },
                vec![2, 3, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(0),
                    target_shape: vec![24],
                },
                vec![24],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("Reshape flatten build");

    let lower = ArrayD::from_elem(IxDyn(&[2, 3, 4]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through flatten Reshape");
    common::assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let eps = 1e-6;

    // Bounds must be exactly preserved through flatten.
    assert!(
        lo.iter().all(|&v| (v - (-5.0)).abs() < eps),
        "flatten lower should be exactly -5"
    );
    assert!(
        hi.iter().all(|&v| (v - 5.0).abs() < eps),
        "flatten upper should be exactly 5"
    );

    // Width = 10.0, exactly preserved.
    assert!(
        (output.max_width() - 10.0).abs() < eps,
        "flatten max_width should be exactly 10.0"
    );

    assert!(!output.has_overflow(), "flatten must not overflow");
    assert!(!output.has_unbounded(), "flatten must not be unbounded");
}

/// Reshape with non-uniform per-element bounds to verify element-wise preservation.
/// Each position in the input has a unique lower/upper derived from its flat index.
#[test]
fn test_reshape_non_uniform_bounds_preserved() {
    let input_shape = [2, 6];
    let output_shape = vec![3, 4];
    let n = input_shape.iter().product::<usize>();

    let def = TensorKernelDef::new(
        "reshape_nonunif",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: input_shape.to_vec(),
                },
                input_shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(0),
                    target_shape: output_shape.clone(),
                },
                output_shape,
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("Reshape non-uniform build");

    // Each element i has bounds [-(i+1), (i+1)].
    let lower_data: Vec<f32> = (0..n).map(|i| -(i as f32 + 1.0)).collect();
    let upper_data: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
    let lower = ArrayD::from_shape_vec(IxDyn(&input_shape), lower_data).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&input_shape), upper_data).unwrap();
    let input = BoundedTensor::new(lower, upper).expect("non-uniform bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through non-uniform Reshape");
    common::assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let eps = 1e-5;

    // After reshape, element at flat index i should still have bounds [-(i+1), (i+1)].
    // Check a few specific positions in the reshaped output.
    // Flat index 0 → [0,0] in [3,4]: bounds [-1, 1]
    assert!(
        (lo[[0, 0]] - (-1.0)).abs() < eps,
        "position [0,0] lower should be -1, got {}",
        lo[[0, 0]]
    );
    assert!(
        (hi[[0, 0]] - 1.0).abs() < eps,
        "position [0,0] upper should be 1, got {}",
        hi[[0, 0]]
    );

    // Flat index 11 → [2,3] in [3,4]: bounds [-12, 12]
    assert!(
        (lo[[2, 3]] - (-12.0)).abs() < eps,
        "position [2,3] lower should be -12, got {}",
        lo[[2, 3]]
    );
    assert!(
        (hi[[2, 3]] - 12.0).abs() < eps,
        "position [2,3] upper should be 12, got {}",
        hi[[2, 3]]
    );

    // Max width should be the widest element: 2*12 = 24.
    assert!(
        (output.max_width() - 24.0).abs() < eps,
        "max_width should be 24.0, got {}",
        output.max_width()
    );
}

// ---------------------------------------------------------------------------
// Concat IBP (AC4: previously zero test coverage)
// ---------------------------------------------------------------------------

/// Concat: two variable [3,4] inputs along axis=1 → [3,8].
/// Variable 0 bounds [-1, 2], variable 1 bounds [5, 10]. IBP must preserve
/// both ranges in the output (concat joins data, not transforms it).
#[test]
fn test_concat_two_variables_ibp_bounds() {
    let def = TensorKernelDef::new(
        "concat_ibp",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![3, 4],
                },
                vec![3, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![3, 4],
                },
                vec![3, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Concat {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 1,
                },
                vec![3, 8],
            ),
        ],
        TensorNodeId::new(2),
    );
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("Concat build");

    // Multi-variable: stacked along dim 0 → input shape [2, 3, 4]
    // Variable 0 (a): bounds [-1, 2]  (width = 3)
    // Variable 1 (b): bounds [5, 10]  (width = 5)
    let mut lower = ArrayD::zeros(IxDyn(&[2, 3, 4]));
    let mut upper = ArrayD::zeros(IxDyn(&[2, 3, 4]));
    lower.slice_mut(ndarray::s![0, .., ..]).fill(-1.0f32);
    upper.slice_mut(ndarray::s![0, .., ..]).fill(2.0f32);
    lower.slice_mut(ndarray::s![1, .., ..]).fill(5.0f32);
    upper.slice_mut(ndarray::s![1, .., ..]).fill(10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("stacked bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through Concat");

    common::assert_bounds_valid(&output);
    assert!(!output.has_overflow(), "Concat must not produce overflow");
    assert!(
        !output.has_unbounded(),
        "Concat must not produce unbounded values"
    );

    let (lo, hi) = output.lower_upper();

    // Output should include both variable ranges.
    let out_lo_min = lo.iter().copied().reduce(f32::min).unwrap();
    let out_hi_max = hi.iter().copied().reduce(f32::max).unwrap();
    assert!(
        out_lo_min <= -0.9,
        "Concat lower should include var 0 range, got {out_lo_min:.4}"
    );
    assert!(
        out_hi_max >= 9.9,
        "Concat upper should include var 1 range, got {out_hi_max:.4}"
    );

    // ConcatLayer IBP is exact concatenation — no widening — max_width = max(3, 5) = 5.
    assert!(
        output.max_width() <= 5.1,
        "Concat max_width should not exceed widest variable width (5.0), got {:.4}",
        output.max_width()
    );
}

/// Concat with non-uniform per-element bounds (AC3 for concat path).
/// Variable 0 has element-varying bounds, variable 1 has uniform bounds.
/// Verifies the output doesn't flatten all bounds to a single range.
#[test]
fn test_concat_non_uniform_bounds_ibp() {
    let def = TensorKernelDef::new(
        "concat_nonunif_ibp",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![1, 4],
                },
                vec![1, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![1, 4],
                },
                vec![1, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Concat {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 1,
                },
                vec![1, 8],
            ),
        ],
        TensorNodeId::new(2),
    );
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("Concat non-uniform build");

    // Multi-variable: stacked along dim 0 → input shape [2, 1, 4]
    // Variable 0 (a): non-uniform bounds, element i has [-i, i+1]
    // Variable 1 (b): uniform bounds [10, 20]
    let mut lower = ArrayD::zeros(IxDyn(&[2, 1, 4]));
    let mut upper = ArrayD::zeros(IxDyn(&[2, 1, 4]));
    for i in 0..4 {
        lower[[0, 0, i]] = -(i as f32);
        upper[[0, 0, i]] = (i + 1) as f32;
    }
    lower.slice_mut(ndarray::s![1, .., ..]).fill(10.0f32);
    upper.slice_mut(ndarray::s![1, .., ..]).fill(20.0f32);
    let input = BoundedTensor::new(lower, upper).expect("stacked bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through non-uniform Concat");
    common::assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();

    // Output must include the tight range from variable 0 (some elements near 0)
    // and the wide range from variable 1 (10-20).
    let out_lo_min = lo.iter().copied().reduce(f32::min).unwrap();
    let out_hi_max = hi.iter().copied().reduce(f32::max).unwrap();
    assert!(
        out_lo_min <= -2.9,
        "non-uniform Concat: lower should include var 0 element at index 3 (>= -3). Got {out_lo_min:.4}."
    );
    assert!(
        out_hi_max >= 19.9,
        "non-uniform Concat: upper should include var 1 range (<= 20). Got {out_hi_max:.4}."
    );
}
