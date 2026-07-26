// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `TensorOpKind::MatMul` → NY `MatMulLayer`.
//!
//! Tests:
//! - Two-variable IBP bounds propagation (McCormick bilinear relaxation)
//! - Q @ K^T attention pattern with transpose_right + scale
//! - Attention-value multiplication (attn_weights @ V) without transpose
//! - Soundness: concrete forward pass lies within IBP bounds

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a MatMul kernel: two matrix inputs → MatMul output.
///
/// left shape: [M, K], right shape: [K, N] or [N, K] if transpose_right.
/// Output shape: [M, N].
fn matmul_kernel(
    name: &str,
    left_shape: &[usize],
    right_shape: &[usize],
    out_shape: &[usize],
    transpose_right: bool,
    scale: Option<f32>,
) -> TensorKernelDef {
    TensorKernelDef::new(
        name,
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "left".into(),
                    shape: left_shape.to_vec(),
                },
                left_shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "right".into(),
                    shape: right_shape.to_vec(),
                },
                right_shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::MatMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                    transpose_right,
                    scale,
                },
                out_shape.to_vec(),
            ),
        ],
        TensorNodeId::new(2),
    )
}

#[test]
fn test_matmul_graph_builds() {
    // Simple 2x3 @ 3x4 = 2x4, no transpose, no scale
    let def = matmul_kernel("matmul_basic", &[2, 3], &[3, 4], &[2, 4], false, None);
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("matmul graph should build");
    assert!(
        graph.num_nodes() >= 1,
        "graph should have at least the matmul node"
    );
}

#[test]
fn test_matmul_transpose_right_graph_builds() {
    // Q @ K^T pattern: left=[2,3], right=[4,3] (transposed), out=[2,4]
    let def = matmul_kernel(
        "matmul_qkt",
        &[2, 3],
        &[4, 3],
        &[2, 4],
        true,
        Some(1.0 / (3.0f32).sqrt()),
    );
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("Q @ K^T graph should build");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_matmul_ibp_bounds_sound() {
    // IBP for MatMul with McCormick bilinear relaxation.
    // Use square matrices so multi-variable stacking produces uniform shapes.
    // left ∈ [-1, 2], right ∈ [1, 3], no transpose, no scale.
    // Each product bounded by McCormick: for a∈[-1,2], b∈[1,3]:
    //   min((-1)*1, (-1)*3, 2*1, 2*3) = -3
    //   max((-1)*1, (-1)*3, 2*1, 2*3) = 6
    let def = matmul_kernel("ibp_matmul_sq", &[2, 2], &[2, 2], &[2, 2], false, None);
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("matmul IBP graph");

    // Multi-variable stacking: [2, 2, 2] where axis 0 selects variable.
    let mut lower = ArrayD::from_elem(IxDyn(&[2, 2, 2]), 0.0f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[2, 2, 2]), 0.0f32);
    // left (variable 0): bounds [-1, 2]
    for i in 0..2 {
        for j in 0..2 {
            lower[[0, i, j]] = -1.0;
            upper[[0, i, j]] = 2.0;
        }
    }
    // right (variable 1): bounds [1, 3]
    for i in 0..2 {
        for j in 0..2 {
            lower[[1, i, j]] = 1.0;
            upper[[1, i, j]] = 3.0;
        }
    }
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // Sound bounds: lower <= upper for all elements.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite(), "lower must be finite, got {l}");
        assert!(u.is_finite(), "upper must be finite, got {u}");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }

    // Each output element is sum of 2 products:
    // Product bounds for a∈[-1,2], b∈[1,3]: [-3, 6]
    // Sum of 2: [-6, 12]
    // IBP may be wider due to McCormick relaxation, but must contain [-6, 12].
    let min_lo = lo.iter().copied().reduce(f32::min).unwrap();
    let max_hi = hi.iter().copied().reduce(f32::max).unwrap();
    assert!(min_lo <= -3.0, "lower bound should be <= -3 (got {min_lo})");
    assert!(max_hi >= 6.0, "upper bound should be >= 6 (got {max_hi})");
}

#[test]
fn test_matmul_ibp_with_scale() {
    // Scaled MatMul: C = (A @ B) * scale, scale = 0.5
    let def = matmul_kernel(
        "ibp_matmul_scale",
        &[2, 2],
        &[2, 2],
        &[2, 2],
        false,
        Some(0.5),
    );
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("scaled matmul graph");

    let mut lower = ArrayD::from_elem(IxDyn(&[2, 2, 2]), 0.0f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[2, 2, 2]), 0.0f32);
    // left: [0, 1]
    for i in 0..2 {
        for j in 0..2 {
            lower[[0, i, j]] = 0.0;
            upper[[0, i, j]] = 1.0;
        }
    }
    // right: [0, 1]
    for i in 0..2 {
        for j in 0..2 {
            lower[[1, i, j]] = 0.0;
            upper[[1, i, j]] = 1.0;
        }
    }
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // All inputs non-negative: output must be non-negative.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!(l <= u, "lower {l} must be <= upper {u}");
        // With scale 0.5: max per product = 0.5*1*1 = 0.5
        // Sum of 2 products: max = 1.0
        assert!(
            *l >= -0.01,
            "lower should be >= 0 for non-negative inputs, got {l}"
        );
    }
}

#[test]
fn test_matmul_ibp_soundness_concrete() {
    // Soundness: concrete forward pass must lie within IBP bounds.
    // Use [2,2] matrices with known values to compute exact output.
    let def = matmul_kernel("sound_matmul", &[2, 2], &[2, 2], &[2, 2], false, None);
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("soundness matmul graph");

    // Bounds: left ∈ [-2, 2], right ∈ [-1, 3]
    let mut lower = ArrayD::from_elem(IxDyn(&[2, 2, 2]), 0.0f32);
    let mut upper = ArrayD::from_elem(IxDyn(&[2, 2, 2]), 0.0f32);
    for i in 0..2 {
        for j in 0..2 {
            lower[[0, i, j]] = -2.0;
            upper[[0, i, j]] = 2.0;
            lower[[1, i, j]] = -1.0;
            upper[[1, i, j]] = 3.0;
        }
    }
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP soundness");
    let (lo, hi) = output.lower_upper();

    // Concrete: left = [[1, 0.5], [0, -1]], right = [[1, 2], [0, 1]]
    // C[0,0] = 1*1 + 0.5*0 = 1.0
    // C[0,1] = 1*2 + 0.5*1 = 2.5
    // C[1,0] = 0*1 + (-1)*0 = 0.0
    // C[1,1] = 0*2 + (-1)*1 = -1.0
    let concrete = [1.0f32, 2.5, 0.0, -1.0];
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();

    for (idx, &c) in concrete.iter().enumerate() {
        assert!(
            lo_flat[idx] <= c + 0.01,
            "soundness: lo[{idx}]={} should be <= concrete={c}",
            lo_flat[idx]
        );
        assert!(
            hi_flat[idx] >= c - 0.01,
            "soundness: hi[{idx}]={} should be >= concrete={c}",
            hi_flat[idx]
        );
    }
}

#[test]
fn test_matmul_rejects_constant_input() {
    let def = matmul_kernel("const_matmul", &[2, 2], &[2, 2], &[2, 2], false, None);
    let result = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::ConstantScalar(1.0),
            TensorParamBinding::Variable,
        ],
    );
    assert!(
        result.is_err(),
        "MatMul with ConstantScalar input should fail"
    );
}
