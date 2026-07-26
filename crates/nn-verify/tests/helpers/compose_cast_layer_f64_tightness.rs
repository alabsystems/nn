// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CastLayer compose tests: IBP/CROWN propagation through ToDtype operations.
//!
//! Tests that ToDtype operations (CastLayer for upcasts, Clamp for downcasts)
//! translate correctly through the trace_to_graph pipeline AND produce sound
//! IBP/CROWN bounds when propagated through the resulting GraphNetwork.
//!
//! Part of #4316: CastLayer for ToDtype verification + f64 evaluation.

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_verify::{propagate_with_crown_fallback, trace_to_graph_model, BoundedTensor, PropMethod};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Graph construction helpers
// ---------------------------------------------------------------------------

/// Build a graph: Input -> op -> output.
fn graph_with_unary_op(op: TraceOp, shape: Vec<usize>) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(1, "op_0".into(), op, vec![0], shape, DType::F32),
    ])
}

/// Build a graph: Input -> Relu -> ToDtype -> output.
fn graph_relu_then_cast(target_dtype: DType, shape: Vec<usize>) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "cast_0".into(),
            TraceOp::ToDtype { target_dtype },
            vec![1],
            shape,
            DType::F32,
        ),
    ])
}

/// Build a graph: Input -> ToDtype -> Relu -> output.
fn graph_cast_then_relu(target_dtype: DType, shape: Vec<usize>) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "cast_0".into(),
            TraceOp::ToDtype { target_dtype },
            vec![0],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![1],
            shape,
            DType::F32,
        ),
    ])
}

/// Build: Input -> Relu -> ToDtype(F16) -> ToDtype(F32) -> Relu -> output.
fn graph_relu_f16_roundtrip(shape: Vec<usize>) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "cast_f16".into(),
            TraceOp::ToDtype {
                target_dtype: DType::F16,
            },
            vec![1],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            3,
            "cast_f32".into(),
            TraceOp::ToDtype {
                target_dtype: DType::F32,
            },
            vec![2],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            4,
            "relu_1".into(),
            TraceOp::Relu,
            vec![3],
            shape,
            DType::F32,
        ),
    ])
}

// ---------------------------------------------------------------------------
// 1. CastLayer (F32 identity) IBP propagation
// ---------------------------------------------------------------------------

/// CastLayer (F32 upcast = identity) propagates IBP bounds unchanged.
#[test]
fn test_castlayer_f32_ibp_propagation() {
    let shape = vec![2, 4];
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::F32,
        },
        shape.clone(),
    );

    let result = trace_to_graph_model(&graph).expect("F32 CastLayer translate");
    assert_eq!(result.dtype_cast_count, 0, "F32 upcast = no cast count");

    let input = uniform_bounds(&shape, 2.0);
    let output = result
        .graph
        .propagate_ibp(&input)
        .expect("IBP through CastLayer");
    assert_bounds_valid(&output);

    let (in_lo, in_hi) = input.lower_upper();
    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-5;
    for (i, ((&il, &ih), (&ol, &oh))) in in_lo
        .iter()
        .zip(in_hi.iter())
        .zip(out_lo.iter().zip(out_hi.iter()))
        .enumerate()
    {
        assert!(
            (il - ol).abs() < eps,
            "CastLayer lower[{i}]: input={il}, output={ol}"
        );
        assert!(
            (ih - oh).abs() < eps,
            "CastLayer upper[{i}]: input={ih}, output={oh}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. CastLayer (F64 upcast) IBP propagation
// ---------------------------------------------------------------------------

/// CastLayer (F64 upcast) works as identity for IBP bounds.
#[test]
fn test_castlayer_f64_ibp_propagation() {
    let shape = vec![2, 4];
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::F64,
        },
        shape.clone(),
    );

    let result = trace_to_graph_model(&graph).expect("F64 CastLayer translate");
    assert_eq!(result.dtype_cast_count, 0, "F64 upcast = no cast count");

    let input = uniform_bounds(&shape, 5.0);
    let output = result
        .graph
        .propagate_ibp(&input)
        .expect("IBP through F64 CastLayer");
    assert_bounds_valid(&output);

    let (in_lo, in_hi) = input.lower_upper();
    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-5;
    for (&il, &ol) in in_lo.iter().zip(out_lo.iter()) {
        assert!((il - ol).abs() < eps, "F64 Cast lower: {il} vs {ol}");
    }
    for (&ih, &oh) in in_hi.iter().zip(out_hi.iter()) {
        assert!((ih - oh).abs() < eps, "F64 Cast upper: {ih} vs {oh}");
    }
}

// ---------------------------------------------------------------------------
// 3. F16 downcast (Clamp) IBP propagation
// ---------------------------------------------------------------------------

/// F16 downcast Clamp: inactive when inputs are within F16 range.
#[test]
fn test_f16_downcast_ibp_clamp_within_range() {
    let shape = vec![2, 4];
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::F16,
        },
        shape.clone(),
    );

    let result = trace_to_graph_model(&graph).expect("F16 downcast translate");
    assert_eq!(result.dtype_cast_count, 1, "F16 downcast = 1 cast count");

    let input = uniform_bounds(&shape, 100.0);
    let output = result
        .graph
        .propagate_ibp(&input)
        .expect("IBP through F16 Clamp");
    assert_bounds_valid(&output);

    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-3;
    for &v in out_lo.iter() {
        assert!(
            (v - (-100.0)).abs() < eps,
            "F16 clamp inactive: lower should be -100, got {v}"
        );
    }
    for &v in out_hi.iter() {
        assert!(
            (v - 100.0).abs() < eps,
            "F16 clamp inactive: upper should be 100, got {v}"
        );
    }
}

/// F16 downcast clamps inputs exceeding F16 representable range.
#[test]
fn test_f16_downcast_ibp_clamp_exceeds_range() {
    let shape = vec![1, 2];
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::F16,
        },
        shape.clone(),
    );

    let result = trace_to_graph_model(&graph).expect("F16 downcast translate");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&shape), -100000.0_f32),
        ArrayD::from_elem(IxDyn(&shape), 100000.0_f32),
    )
    .expect("valid bounds");

    let output = result
        .graph
        .propagate_ibp(&input)
        .expect("IBP through F16 Clamp");
    assert_bounds_valid(&output);

    let (out_lo, out_hi) = output.lower_upper();
    let f16_max = 65504.0_f32;
    let eps = 1.0;
    for &v in out_lo.iter() {
        assert!(v >= -f16_max - eps, "F16 clamp: lower {v} >= -{f16_max}");
    }
    for &v in out_hi.iter() {
        assert!(v <= f16_max + eps, "F16 clamp: upper {v} <= {f16_max}");
    }
}

// ---------------------------------------------------------------------------
// 4. CastLayer in pipeline: Relu -> Cast(F32) -> output
// ---------------------------------------------------------------------------

/// CastLayer (F32) after Relu: output matches Relu-only bounds.
#[test]
fn test_relu_then_castlayer_f32_ibp() {
    let shape = vec![2, 3];

    let graph_with = graph_relu_then_cast(DType::F32, shape.clone());
    let result_with = trace_to_graph_model(&graph_with).expect("Relu+Cast translate");
    let input = uniform_bounds(&shape, 2.0);
    let out_with = result_with
        .graph
        .propagate_ibp(&input)
        .expect("IBP with cast");
    assert_bounds_valid(&out_with);

    let graph_no = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            shape,
            DType::F32,
        ),
    ]);
    let result_no = trace_to_graph_model(&graph_no).expect("Relu-only translate");
    let out_no = result_no
        .graph
        .propagate_ibp(&input)
        .expect("IBP without cast");
    assert_bounds_valid(&out_no);

    let (lo_with, hi_with) = out_with.lower_upper();
    let (lo_no, hi_no) = out_no.lower_upper();
    let eps = 1e-5;
    for (i, ((&lw, &ln), (&hw, &hn))) in lo_with
        .iter()
        .zip(lo_no.iter())
        .zip(hi_with.iter().zip(hi_no.iter()))
        .enumerate()
    {
        assert!(
            (lw - ln).abs() < eps,
            "Relu+Cast lower[{i}]={lw} vs Relu={ln}"
        );
        assert!(
            (hw - hn).abs() < eps,
            "Relu+Cast upper[{i}]={hw} vs Relu={hn}"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. CastLayer before activation: Cast(F32) -> Relu
// ---------------------------------------------------------------------------

/// CastLayer (F32) before Relu: transparent, bounds match Relu-only.
#[test]
fn test_castlayer_f32_then_relu_ibp() {
    let shape = vec![2, 3];
    let graph = graph_cast_then_relu(DType::F32, shape.clone());
    let result = trace_to_graph_model(&graph).expect("Cast+Relu translate");
    let input = uniform_bounds(&shape, 2.0);
    let output = result.graph.propagate_ibp(&input).expect("IBP Cast+Relu");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let eps = 1e-4;
    for &v in lo.iter() {
        assert!(v >= -eps, "Cast+Relu lower >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 2.0 + eps, "Cast+Relu upper <= 2, got {v}");
    }
}

// ---------------------------------------------------------------------------
// 6. F16 roundtrip: Relu -> F16 (Clamp) -> F32 (Cast) -> Relu
// ---------------------------------------------------------------------------

/// F16 roundtrip: Clamp + CastLayer compose correctly.
#[test]
fn test_f16_roundtrip_ibp() {
    let shape = vec![2, 3];
    let graph = graph_relu_f16_roundtrip(shape.clone());
    let result = trace_to_graph_model(&graph).expect("F16 roundtrip translate");
    assert_eq!(result.dtype_cast_count, 1, "1 F16 downcast counted");

    let input = uniform_bounds(&shape, 3.0);
    let output = result
        .graph
        .propagate_ibp(&input)
        .expect("IBP F16 roundtrip");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let eps = 1e-3;
    for &v in lo.iter() {
        assert!(v >= -eps, "F16 roundtrip lower >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 3.0 + eps, "F16 roundtrip upper <= 3, got {v}");
    }
}

// ---------------------------------------------------------------------------
// 7. CROWN propagation through CastLayer
// ---------------------------------------------------------------------------

/// CROWN through CastLayer (F32) pipeline: coefficients pass unchanged.
#[test]
fn test_castlayer_crown_propagation() {
    let shape = vec![2, 3];
    let graph = graph_cast_then_relu(DType::F32, shape.clone());
    let result = trace_to_graph_model(&graph).expect("Cast+Relu translate");
    let input = uniform_bounds(&shape, 2.0);

    let ibp_output = result
        .graph
        .propagate_ibp(&input)
        .expect("IBP through Cast+Relu");
    assert_bounds_valid(&ibp_output);

    let (method, crown_output, _fallback) =
        propagate_with_crown_fallback(&result.graph, &input).expect("CROWN Cast+Relu");
    assert_bounds_valid(&crown_output);

    if matches!(method, PropMethod::Crown) {
        let (crown_lo, crown_hi) = crown_output.lower_upper();
        let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
        let eps = 1e-4;
        for (i, (&cl, &il)) in crown_lo.iter().zip(ibp_lo.iter()).enumerate() {
            assert!(cl >= il - eps, "CROWN lower[{i}] {cl} >= IBP lower {il}");
        }
        for (i, (&cu, &iu)) in crown_hi.iter().zip(ibp_hi.iter()).enumerate() {
            assert!(cu <= iu + eps, "CROWN upper[{i}] {cu} <= IBP upper {iu}");
        }
    }
}

// ---------------------------------------------------------------------------
// 8. BF16 downcast (Clamp) IBP propagation
// ---------------------------------------------------------------------------

/// BF16 downcast Clamp: inactive for typical input ranges.
#[test]
fn test_bf16_downcast_ibp_clamp() {
    let shape = vec![2, 4];
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::BF16,
        },
        shape.clone(),
    );

    let result = trace_to_graph_model(&graph).expect("BF16 downcast translate");
    assert_eq!(result.dtype_cast_count, 1, "BF16 downcast = 1 cast count");

    let input = uniform_bounds(&shape, 100.0);
    let output = result
        .graph
        .propagate_ibp(&input)
        .expect("IBP through BF16 Clamp");
    assert_bounds_valid(&output);

    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-3;
    for &v in out_lo.iter() {
        assert!(
            (v - (-100.0)).abs() < eps,
            "BF16 clamp inactive: lower should be -100, got {v}"
        );
    }
    for &v in out_hi.iter() {
        assert!(
            (v - 100.0).abs() < eps,
            "BF16 clamp inactive: upper should be 100, got {v}"
        );
    }
}
