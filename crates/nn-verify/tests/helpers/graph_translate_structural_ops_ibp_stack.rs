// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IBP bounds propagation tests for Stack tensor op.
//!
//! Extracted from `graph_translate_structural_ops_ibp.rs` for 500-line compliance.
//! Tests verify that IBP bounds propagate correctly through Stack operations,
//! including multi-variable inputs with asymmetric bound ranges.
//!
//! Part of #1693.

use super::common;

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Stack: two [4,6] variable inputs along axis=2 → [4,6,2].
/// Input 0 bounds [-1,3], input 1 bounds [10,20].
/// Checks: validity, both ranges present, no overflow/unbounded, width bounded.
#[test]
fn test_stack_two_variables_ibp_bounds() {
    let def = TensorKernelDef::new(
        "stack_ibp",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Stack {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 2,
                },
                vec![4, 6, 2],
            ),
        ],
        TensorNodeId::new(2),
    );
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("Stack build");

    // Multi-variable: stacked along dim 0 → input shape [2, 4, 6]
    // Variable 0 (a): bounds [-1, 3]  (width = 4)
    // Variable 1 (b): bounds [10, 20] (width = 10)
    let mut lower = ArrayD::zeros(IxDyn(&[2, 4, 6]));
    let mut upper = ArrayD::zeros(IxDyn(&[2, 4, 6]));
    lower.slice_mut(ndarray::s![0, .., ..]).fill(-1.0f32);
    upper.slice_mut(ndarray::s![0, .., ..]).fill(3.0f32);
    lower.slice_mut(ndarray::s![1, .., ..]).fill(10.0f32);
    upper.slice_mut(ndarray::s![1, .., ..]).fill(20.0f32);
    let input = BoundedTensor::new(lower, upper).expect("stacked bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through Stack");

    common::assert_bounds_valid(&output);
    assert!(!output.has_overflow(), "Stack must not produce overflow");
    assert!(
        !output.has_unbounded(),
        "Stack must not produce unbounded values"
    );

    let (lo, hi) = output.lower_upper();

    // Output should include both variable ranges.
    let out_lo_min = lo.iter().copied().reduce(f32::min).unwrap();
    let out_hi_max = hi.iter().copied().reduce(f32::max).unwrap();
    assert!(
        out_lo_min <= -0.9,
        "Stack output lower should include variable 0 range (>= -1). Got min={out_lo_min:.4}."
    );
    assert!(
        out_hi_max >= 19.9,
        "Stack output upper should include variable 1 range (<= 20). Got max={out_hi_max:.4}."
    );

    // Width bounded by widest input variable's width (10.0 from variable 1).
    // ConcatLayer IBP is exact concatenation — no widening — so max_width = max(4, 10) = 10.
    assert!(
        output.max_width() <= 10.1,
        "Stack max_width should not exceed widest variable width (10.0), got {:.4}",
        output.max_width()
    );
}

/// Stack with asymmetric bounds: variable 0 has tight bounds [0, 0.1],
/// variable 1 has wide bounds [-100, 100]. Verify the output doesn't
/// incorrectly merge or average the ranges.
#[test]
fn test_stack_asymmetric_width_ibp_bounds() {
    let def = TensorKernelDef::new(
        "stack_asymmetric_ibp",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "tight".to_string(),
                    shape: vec![3],
                },
                vec![3],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "wide".to_string(),
                    shape: vec![3],
                },
                vec![3],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Stack {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 1,
                },
                vec![3, 2],
            ),
        ],
        TensorNodeId::new(2),
    );
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("Stack asymmetric build");

    // Multi-variable input: dim 0 stacks variables, shape [2, 3].
    // Variable 0: tight bounds [0.0, 0.1] (width=0.1)
    // Variable 1: wide bounds [-100, 100] (width=200)
    let mut lower = ArrayD::zeros(IxDyn(&[2, 3]));
    let mut upper = ArrayD::zeros(IxDyn(&[2, 3]));
    lower.slice_mut(ndarray::s![0, ..]).fill(0.0f32);
    upper.slice_mut(ndarray::s![0, ..]).fill(0.1f32);
    lower.slice_mut(ndarray::s![1, ..]).fill(-100.0f32);
    upper.slice_mut(ndarray::s![1, ..]).fill(100.0f32);
    let input = BoundedTensor::new(lower, upper).expect("stacked bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through asymmetric Stack");
    common::assert_bounds_valid(&output);
    assert!(!output.has_overflow(), "asymmetric Stack must not overflow");

    let (lo, hi) = output.lower_upper();

    // The output must contain both the tight and wide ranges.
    // At minimum, lower must reach ≤ -99.0 (wide variable lower) and
    // upper must reach ≥ 99.0 (wide variable upper).
    let out_lo_min = lo.iter().copied().reduce(f32::min).unwrap();
    let out_hi_max = hi.iter().copied().reduce(f32::max).unwrap();
    assert!(
        out_lo_min <= -99.0,
        "asymmetric Stack must include wide lower, got {out_lo_min:.4}"
    );
    assert!(
        out_hi_max >= 99.0,
        "asymmetric Stack must include wide upper, got {out_hi_max:.4}"
    );

    // Width bounded by widest input variable's width (200.0 from variable 1).
    // ConcatLayer IBP is exact concatenation — no widening beyond input ranges.
    assert!(
        output.max_width() <= 200.1,
        "asymmetric Stack max_width should not exceed widest variable width (200.0), got {:.4}",
        output.max_width()
    );
}
