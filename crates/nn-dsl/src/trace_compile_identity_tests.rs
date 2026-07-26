// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kernel identity, cumsum/repeat_interleave, and powf compile tests.
//!
//! Extracted from `trace_compile_tests.rs` to keep files under 1000 lines.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::trace_compile::{compile_trace, CompiledStep, NativeOpKind};

// -- Helpers (shared with parent test module) ----------------------------------

fn graph_from_nodes(nodes: Vec<TraceNode>) -> ComputationGraph {
    ComputationGraph::from_nodes(nodes)
}

fn input_node(id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

fn unary_node(id: u64, name: &str, op: TraceOp, input_id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![input_id],
        shape.to_vec(),
        DType::F32,
    )
}

fn binary_node(
    id: u64,
    name: &str,
    op: TraceOp,
    lhs_id: u64,
    rhs_id: u64,
    shape: &[usize],
) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![lhs_id, rhs_id],
        shape.to_vec(),
        DType::F32,
    )
}

// -- Cumsum / RepeatInterleave compilation ------------------------------------

#[test]
fn test_compile_cumsum_1d() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "cumsum_0".into(),
            TraceOp::Cumsum { dim: 0 },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("cumsum should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[0], CompiledStep::InputForward));
    match &steps[1] {
        CompiledStep::NativeOp { op, weight_data } => {
            assert!(matches!(op, NativeOpKind::Cumsum { dim: 0, .. }));
            assert!(weight_data.is_empty());
        }
        other => panic!("expected NativeOp Cumsum, got {other:?}"),
    }
}

#[test]
fn test_compile_cumsum_2d() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 3]),
        TraceNode::new(
            1,
            "cumsum_0".into(),
            TraceOp::Cumsum { dim: 1 },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("cumsum 2d should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::NativeOp { .. }));
}

#[test]
fn test_compile_cumsum_single_element() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1]),
        TraceNode::new(
            1,
            "cumsum_0".into(),
            TraceOp::Cumsum { dim: 0 },
            vec![0],
            vec![1],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("cumsum single should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::NativeOp { .. }));
}

#[test]
fn test_compile_repeat_interleave() {
    // Two-input form: always RuntimeOp (counts are data-dependent). Fixes #2452.
    let graph = graph_from_nodes(vec![
        input_node(0, &[3]),
        input_node(1, &[3]),
        TraceNode::new(
            2,
            "repeat_interleave_0".into(),
            TraceOp::RepeatInterleave { dim: 0 },
            vec![0, 1],
            vec![6],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("repeat_interleave should compile");
    assert_eq!(steps.len(), 3);
    assert!(
        matches!(steps[2], CompiledStep::RuntimeOp { .. }),
        "two-input repeat_interleave should emit RuntimeOp"
    );
}

#[test]
fn test_compile_repeat_interleave_2d() {
    // Two-input form: always RuntimeOp (counts are data-dependent). Fixes #2452.
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 3]),
        input_node(1, &[3]),
        TraceNode::new(
            2,
            "repeat_interleave_0".into(),
            TraceOp::RepeatInterleave { dim: 1 },
            vec![0, 1],
            vec![2, 6],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("repeat_interleave 2d should compile");
    assert_eq!(steps.len(), 3);
    assert!(matches!(steps[2], CompiledStep::RuntimeOp { .. }));
}

#[test]
fn test_compile_repeat_interleave_variable_emits_runtime_op() {
    // Variable repeats: output dim not divisible by input dim -> RuntimeOp (#2234).
    let graph = graph_from_nodes(vec![
        input_node(0, &[3]),
        input_node(1, &[3]),
        TraceNode::new(
            2,
            "repeat_interleave_0".into(),
            TraceOp::RepeatInterleave { dim: 0 },
            vec![0, 1],
            vec![7],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("variable repeats should compile as RuntimeOp");
    assert_eq!(steps.len(), 3);
    assert!(
        matches!(&steps[2], CompiledStep::RuntimeOp { .. }),
        "expected RuntimeOp for variable repeats, got {:?}",
        &steps[2]
    );
}

// -- Kernel identity verification ---------------------------------------------

/// Extract the kernel def name from a `CompiledStep::Dispatch`.
fn dispatch_name(step: &CompiledStep) -> &str {
    match step {
        CompiledStep::Dispatch { kernel, .. } => kernel.name(),
        other => panic!("expected Dispatch, got: {other:?}"),
    }
}

/// Verify that distinct activation ops produce distinct kernel names.
#[test]
fn test_kernel_identity_activations_are_distinct() {
    let ops = [
        ("relu", TraceOp::Relu),
        ("gelu", TraceOp::Gelu),
        ("gelu_erf", TraceOp::GeluErf),
        ("silu", TraceOp::Silu),
        ("sigmoid", TraceOp::Sigmoid),
        ("tanh", TraceOp::Tanh),
    ];

    let mut names: Vec<(&str, String)> = Vec::new();
    for (label, op) in &ops {
        let graph = graph_from_nodes(vec![
            input_node(0, &[8]),
            unary_node(1, &format!("{label}_0"), op.clone(), 0, &[8]),
        ]);
        let steps = compile_trace(&graph).unwrap_or_else(|e| panic!("{label} failed: {e}"));
        let name = dispatch_name(&steps[1]).to_string();
        names.push((label, name));
    }

    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            assert_ne!(
                names[i].1, names[j].1,
                "{} and {} produced the same kernel name '{}' — wrong dispatch",
                names[i].0, names[j].0, names[i].1
            );
        }
    }
}

/// Verify that distinct binary ops produce distinct kernel names.
#[test]
fn test_kernel_identity_binary_ops_are_distinct() {
    let ops = [
        ("add", TraceOp::Add),
        ("sub", TraceOp::Sub),
        ("mul", TraceOp::Mul),
    ];

    let mut names: Vec<(&str, String)> = Vec::new();
    for (label, op) in &ops {
        let graph = graph_from_nodes(vec![
            input_node(0, &[4]),
            input_node(1, &[4]),
            binary_node(2, &format!("{label}_0"), op.clone(), 0, 1, &[4]),
        ]);
        let steps = compile_trace(&graph).unwrap_or_else(|e| panic!("{label} failed: {e}"));
        let name = dispatch_name(&steps[2]).to_string();
        names.push((label, name));
    }

    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            assert_ne!(
                names[i].1, names[j].1,
                "{} and {} produced the same kernel name '{}' — wrong dispatch",
                names[i].0, names[j].0, names[i].1
            );
        }
    }
}

// -- Powf compile tests -------------------------------------------------------

/// compile_powf must produce a Dispatch step (not Passthrough or error).
/// Uses exponent 3.0 (not 2.0 or 0.5 which are dispatch-guarded to sqr/sqrt).
#[test]
fn test_compile_powf_basic() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "powf_0".into(),
            TraceOp::Powf { exponent: 3.0 },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("powf should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert_eq!(kernel.name(), "powf");
            assert!(
                weight_data.contains_key("exponent"),
                "weight_data must contain 'exponent'"
            );
            let exp_ref = &weight_data["exponent"];
            assert_eq!(exp_ref.data(), &[3.0f32]);
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

/// compile_powf decomposes to abs -> log -> mul -> exp plus sign correction.
/// Uses exponent 3.0 (not 0.5 which is dispatch-guarded to sqrt).
#[test]
fn test_compile_powf_graph_structure() {
    use crate::tensor_ir::TensorOpKind;

    let graph = graph_from_nodes(vec![
        input_node(0, &[8]),
        TraceNode::new(
            1,
            "powf_0".into(),
            TraceOp::Powf { exponent: 3.0 },
            vec![0],
            vec![8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("powf should compile");
    let kernel = match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert!(
                weight_data.contains_key("zero_const"),
                "odd integer powf must materialize a negative-base compare threshold"
            );
            assert!(
                weight_data.contains_key("one_const"),
                "odd integer powf must materialize a select mask inverse"
            );
            kernel
        }
        other => panic!("expected Dispatch, got {other:?}"),
    };
    let has_abs = kernel.def().nodes.iter().any(
        |n| matches!(&n.kind, TensorOpKind::Elementwise { kernel, .. } if kernel.name == "abs"),
    );
    let has_log = kernel.def().nodes.iter().any(
        |n| matches!(&n.kind, TensorOpKind::Elementwise { kernel, .. } if kernel.name == "log"),
    );
    let has_exp = kernel.def().nodes.iter().any(
        |n| matches!(&n.kind, TensorOpKind::Elementwise { kernel, .. } if kernel.name == "exp"),
    );
    let has_cmp_lt = kernel.def().nodes.iter().any(
        |n| matches!(&n.kind, TensorOpKind::Elementwise { kernel, .. } if kernel.name == "cmp_lt"),
    );
    assert!(
        kernel.def().nodes.len() >= 6,
        "powf decomposition should have >= 6 IR nodes, got {}",
        kernel.def().nodes.len()
    );
    assert!(has_abs, "powf decomposition must contain an Abs kernel");
    assert!(has_log, "powf decomposition must contain a Log kernel");
    assert!(has_exp, "powf decomposition must contain an Exp kernel");
    assert!(
        has_cmp_lt,
        "odd integer powf must contain a negative-base compare mask"
    );
}

/// compile_powf must handle fractional exponents via abs/log/exp and NaN fill.
/// Uses exponent 1.5 (not 0.5 which is dispatch-guarded to sqrt).
#[test]
fn test_compile_powf_fractional_exponent() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "powf_0".into(),
            TraceOp::Powf { exponent: 1.5 },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("powf(1.5) should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            let exp_val = weight_data["exponent"].data();
            assert_eq!(exp_val, &[1.5f32]);
            assert!(
                weight_data.contains_key("neg_one_const"),
                "fractional powf must materialize a finite constant that generates NaN at runtime"
            );
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

/// compile_powf preserves output shape from the trace node.
#[test]
fn test_compile_powf_preserves_shape() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 3, 4]),
        TraceNode::new(
            1,
            "powf_0".into(),
            TraceOp::Powf { exponent: 3.0 },
            vec![0],
            vec![2, 3, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("powf should compile");
    let kernel = match &steps[1] {
        CompiledStep::Dispatch { kernel, .. } => kernel,
        other => panic!("expected Dispatch, got {other:?}"),
    };
    let out_node = &kernel.def().nodes[kernel.def().output.index()];
    assert_eq!(out_node.shape, vec![2, 3, 4]);
}

/// Powf(0.0) is a constant-ones fast path matching `f32::powf`.
#[test]
fn test_compile_powf_zero_dispatches_to_constant_ones() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "powf_0".into(),
            TraceOp::Powf { exponent: 0.0 },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("powf(0.0) should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::ConstantValue { value, shape } => {
            assert_eq!(*value, 1.0);
            assert_eq!(shape, &[4]);
        }
        other => panic!("expected ConstantValue, got {other:?}"),
    }
}

/// Powf(1.0) is an identity fast path.
#[test]
fn test_compile_powf_one_dispatches_to_identity() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "powf_0".into(),
            TraceOp::Powf { exponent: 1.0 },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("powf(1.0) should compile");
    assert_eq!(steps.len(), 2);
    assert!(
        matches!(&steps[1], CompiledStep::IdentityPassthrough),
        "Powf(1.0) must compile to IdentityPassthrough"
    );
}

// -- Powf dispatch guard tests (#2751) ----------------------------------------

/// Powf(2.0) is dispatch-guarded to sqr (avoids NaN from log domain error).
#[test]
fn test_compile_powf_2_dispatches_to_sqr() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "powf_0".into(),
            TraceOp::Powf { exponent: 2.0 },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("powf(2.0) should compile");
    assert_eq!(steps.len(), 2);
    assert_eq!(
        dispatch_name(&steps[1]),
        "sqr",
        "Powf(2.0) must dispatch to sqr kernel"
    );
}

/// Powf(0.5) is dispatch-guarded to sqrt (avoids NaN from log domain error).
#[test]
fn test_compile_powf_05_dispatches_to_sqrt() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "powf_0".into(),
            TraceOp::Powf { exponent: 0.5 },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("powf(0.5) should compile");
    assert_eq!(steps.len(), 2);
    assert_eq!(
        dispatch_name(&steps[1]),
        "sqrt",
        "Powf(0.5) must dispatch to sqrt kernel"
    );
}
