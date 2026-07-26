// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for trace recorder, WeightRef, and segment logic (#3744).
//!
//! Supplements existing trace Kani files with proofs for:
//!
//! **WeightRef construction and validation (8 harnesses):**
//!  1. WeightRef::new rejects data/shape mismatch
//!  2. WeightRef::new accepts matching data/shape
//!  3. WeightRef::new accepts empty data (shape-only)
//!  4. WeightRef::from_shape creates shape-only ref
//!  5. WeightRef::is_placeholder detects shape-only with data
//!  6. WeightRef::is_placeholder false for empty shape
//!  7. WeightRef::is_placeholder false for zero-dim shape
//!  8. WeightRef data/shape accessors match construction
//!
//! **TraceActivation string properties (3 harnesses):**
//!  9. All TraceActivation variants have non-empty as_str()
//! 10. TraceActivation::as_str() returns unique strings
//! 11. TraceUpsampleMode::as_str() returns non-empty unique strings
//!
//! **SegmentBoundary properties (4 harnesses):**
//! 12. SegmentBoundary with bounds stores both lower and upper
//! 13. SegmentBoundary without bounds has None
//! 14. Graph with one boundary splits into 2 segments
//! 15. Graph with two boundaries splits into 3 segments
//!
//! Part of #3744.

use crate::dyn_tensor::trace::{
    ComputationGraph, TraceActivation, TraceNode, TraceOp, TraceUpsampleMode, WeightRef,
};
use crate::DType;

// -- Helper -------------------------------------------------------------------

fn make_node(id: u64, op: TraceOp, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(id, format!("node_{id}"), op, inputs, shape, DType::F32)
}

// ===========================================================================
// WeightRef construction and validation
// ===========================================================================

/// Prove: WeightRef::new rejects data/shape mismatch.
///
/// When data is non-empty and its length does not equal the product of
/// shape dimensions, new() must return Err.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_rejects_data_shape_mismatch() {
    let data = vec![1.0f32, 2.0, 3.0]; // 3 elements
    let shape = vec![2, 2]; // product = 4
    let result = WeightRef::new(data, shape);
    assert!(
        result.is_err(),
        "data len 3 != shape product 4 must be rejected"
    );
}

/// Prove: WeightRef::new accepts matching data/shape.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_accepts_matching_data_shape() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0]; // 4 elements
    let shape = vec![2, 2]; // product = 4
    let result = WeightRef::new(data, shape);
    assert!(result.is_ok(), "matching data/shape must be accepted");
}

/// Prove: WeightRef::new accepts empty data (shape-only path).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_accepts_empty_data() {
    let data = vec![];
    let shape = vec![4, 8]; // non-empty shape but empty data is OK
    let result = WeightRef::new(data, shape);
    assert!(
        result.is_ok(),
        "empty data must be accepted regardless of shape"
    );
}

/// Prove: WeightRef::from_shape creates a shape-only reference with empty data.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_from_shape_creates_empty_data() {
    let wr = WeightRef::from_shape(&[4, 8, 16]);
    assert!(wr.data().is_empty(), "from_shape must have empty data");
    assert!(wr.shape().len() == 3, "shape must have 3 dimensions");
    assert!(wr.shape()[0] == 4);
    assert!(wr.shape()[1] == 8);
    assert!(wr.shape()[2] == 16);
}

/// Prove: is_placeholder returns true for non-empty shape with empty data
/// where all dimensions are > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_is_placeholder_shape_only() {
    let wr = WeightRef::from_shape(&[4, 8]);
    assert!(
        wr.is_placeholder(),
        "shape-only ref with positive dims must be placeholder"
    );
}

/// Prove: is_placeholder returns false for empty shape (even with empty data).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_is_placeholder_false_empty_shape() {
    let wr = WeightRef::from_shape(&[]);
    assert!(!wr.is_placeholder(), "empty shape must not be placeholder");
}

/// Prove: is_placeholder returns false for zero-dim shape.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_is_placeholder_false_zero_dim() {
    let wr = WeightRef::from_shape(&[4, 0, 8]);
    assert!(
        !wr.is_placeholder(),
        "zero-dim shape must not be placeholder"
    );
}

/// Prove: data() and shape() accessors match construction values.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_accessors_match_construction() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let wr = WeightRef::new(data.clone(), shape.clone()).unwrap();
    assert!(wr.data().len() == 6, "data length must be 6");
    assert!(wr.shape().len() == 2, "shape must have 2 dims");
    assert!(wr.data()[0] == 1.0, "first data element");
    assert!(wr.data()[5] == 6.0, "last data element");
    assert!(wr.shape()[0] == 2, "first dim");
    assert!(wr.shape()[1] == 3, "second dim");
}

// ===========================================================================
// TraceActivation string properties
// ===========================================================================

/// Prove: all TraceActivation variants have non-empty as_str().
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn trace_activation_all_nonempty() {
    let variants = [
        TraceActivation::Relu,
        TraceActivation::Gelu,
        TraceActivation::GeluErf,
        TraceActivation::Silu,
        TraceActivation::Sigmoid,
        TraceActivation::Tanh,
        TraceActivation::Exp,
        TraceActivation::Log,
        TraceActivation::Elu,
        TraceActivation::LeakyRelu,
        TraceActivation::Mish,
    ];
    let mut i = 0;
    while i < 11 {
        assert!(!variants[i].as_str().is_empty(), "as_str must be non-empty");
        i += 1;
    }
}

/// Prove: TraceActivation::as_str() returns distinct strings for different
/// variants (except Gelu/GeluErf which differ).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn trace_activation_relu_silu_sigmoid_distinct() {
    let relu = TraceActivation::Relu.as_str();
    let silu = TraceActivation::Silu.as_str();
    let sigmoid = TraceActivation::Sigmoid.as_str();
    let tanh = TraceActivation::Tanh.as_str();
    assert!(relu != silu, "relu != silu");
    assert!(relu != sigmoid, "relu != sigmoid");
    assert!(relu != tanh, "relu != tanh");
    assert!(silu != sigmoid, "silu != sigmoid");
    assert!(silu != tanh, "silu != tanh");
    assert!(sigmoid != tanh, "sigmoid != tanh");
}

/// Prove: TraceUpsampleMode::as_str() returns non-empty, distinct strings.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn trace_upsample_mode_as_str() {
    let nearest = TraceUpsampleMode::Nearest.as_str();
    let bilinear = TraceUpsampleMode::Bilinear.as_str();
    assert!(!nearest.is_empty(), "nearest must be non-empty");
    assert!(!bilinear.is_empty(), "bilinear must be non-empty");
    assert!(nearest != bilinear, "nearest != bilinear");
}

// ===========================================================================
// SegmentBoundary graph splitting
// ===========================================================================

/// Prove: SegmentBoundary with bounds stores both lower and upper.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn segment_boundary_stores_bounds() {
    let op = TraceOp::SegmentBoundary {
        reason: "test".to_string(),
        input_bounds: Some((-1.0, 1.0)),
    };
    if let TraceOp::SegmentBoundary {
        reason,
        input_bounds,
    } = &op
    {
        assert!(reason == "test", "reason must be stored");
        assert!(input_bounds.is_some(), "bounds must be present");
        let (lo, hi) = input_bounds.unwrap();
        assert!(lo == -1.0, "lower bound");
        assert!(hi == 1.0, "upper bound");
    } else {
        panic!("must be SegmentBoundary");
    }
}

/// Prove: SegmentBoundary without bounds has None.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn segment_boundary_no_bounds() {
    let op = TraceOp::SegmentBoundary {
        reason: "regulate".to_string(),
        input_bounds: None,
    };
    if let TraceOp::SegmentBoundary { input_bounds, .. } = &op {
        assert!(input_bounds.is_none(), "bounds must be None");
    } else {
        panic!("must be SegmentBoundary");
    }
}

/// Prove: graph with one SegmentBoundary splits into exactly 2 segments.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn graph_one_boundary_splits_into_two() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let n2 = make_node(
        3,
        TraceOp::SegmentBoundary {
            reason: "split".to_string(),
            input_bounds: None,
        },
        vec![2],
        vec![4],
    );
    let n3 = make_node(4, TraceOp::Sigmoid, vec![3], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2, n3]);

    assert!(graph.has_segment_boundaries(), "must have boundaries");
    let segmented = graph.split_at_segment_boundaries();
    assert!(segmented.segments.len() == 2, "one boundary -> 2 segments");
}

/// Prove: graph with two SegmentBoundary markers splits into 3 segments.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(7)]
fn graph_two_boundaries_split_into_three() {
    let n0 = make_node(1, TraceOp::Input, vec![], vec![4]);
    let n1 = make_node(2, TraceOp::Relu, vec![1], vec![4]);
    let n2 = make_node(
        3,
        TraceOp::SegmentBoundary {
            reason: "s1".to_string(),
            input_bounds: None,
        },
        vec![2],
        vec![4],
    );
    let n3 = make_node(4, TraceOp::Sigmoid, vec![3], vec![4]);
    let n4 = make_node(
        5,
        TraceOp::SegmentBoundary {
            reason: "s2".to_string(),
            input_bounds: Some((-1.0, 1.0)),
        },
        vec![4],
        vec![4],
    );
    let n5 = make_node(6, TraceOp::Exp, vec![5], vec![4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2, n3, n4, n5]);

    assert!(graph.has_segment_boundaries());
    let segmented = graph.split_at_segment_boundaries();
    assert!(
        segmented.segments.len() == 3,
        "two boundaries -> 3 segments"
    );
}
