// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the constant folding graph pass.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::constant_fold;

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

fn const_node(id: u64, value: f64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("const_{id}"),
        TraceOp::Constant { value },
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

fn binary_node(id: u64, name: &str, op: TraceOp, lhs: u64, rhs: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![lhs, rhs],
        shape.to_vec(),
        DType::F32,
    )
}

fn unary_node(id: u64, name: &str, op: TraceOp, input: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![input],
        shape.to_vec(),
        DType::F32,
    )
}

// -- Constant-constant folding ------------------------------------------------

#[test]
fn test_fold_constant_add() {
    // Constant(2.0) + Constant(3.0) → Constant(5.0)
    let graph = ComputationGraph::from_nodes(vec![
        const_node(0, 2.0, &[4]),
        const_node(1, 3.0, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    let nodes = folded.nodes();
    assert_eq!(nodes.len(), 3);
    match nodes[2].op() {
        TraceOp::Constant { value } => {
            assert!((value - 5.0).abs() < 1e-10, "expected 5.0, got {value}");
        }
        other => panic!("expected Constant, got {other:?}"),
    }
}

#[test]
fn test_fold_constant_mul() {
    // Constant(3.0) * Constant(4.0) → Constant(12.0)
    let graph = ComputationGraph::from_nodes(vec![
        const_node(0, 3.0, &[4]),
        const_node(1, 4.0, &[4]),
        binary_node(2, "mul_0", TraceOp::Mul, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    match folded.nodes()[2].op() {
        TraceOp::Constant { value } => assert!((value - 12.0).abs() < 1e-10),
        other => panic!("expected Constant, got {other:?}"),
    }
}

#[test]
fn test_fold_constant_unary_exp() {
    // Exp(Constant(0.0)) → Constant(1.0)
    let graph = ComputationGraph::from_nodes(vec![
        const_node(0, 0.0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
    ]);
    let folded = constant_fold(&graph);
    match folded.nodes()[1].op() {
        TraceOp::Constant { value } => assert!((value - 1.0).abs() < 1e-10),
        other => panic!("expected Constant, got {other:?}"),
    }
}

#[test]
fn test_fold_chain_constant_constant() {
    // Constant(2) + Constant(3) = Constant(5), then Constant(5) * Constant(2) = Constant(10)
    let graph = ComputationGraph::from_nodes(vec![
        const_node(0, 2.0, &[4]),
        const_node(1, 3.0, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
        binary_node(3, "mul_0", TraceOp::Mul, 2, 0, &[4]),
    ]);
    let folded = constant_fold(&graph);
    // Node 2 folded to 5.0, then node 3 = 5.0 * 2.0 = 10.0.
    match folded.nodes()[3].op() {
        TraceOp::Constant { value } => assert!((value - 10.0).abs() < 1e-10),
        other => panic!("expected Constant(10.0), got {other:?}"),
    }
}

#[test]
fn test_no_fold_nan_result() {
    // Constant(0.0) / Constant(0.0) produces NaN — should NOT be folded.
    let graph = ComputationGraph::from_nodes(vec![
        const_node(0, 0.0, &[4]),
        const_node(1, 0.0, &[4]),
        binary_node(2, "div_0", TraceOp::Div, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    // Node 2 should remain as Div (not folded to NaN constant).
    assert!(
        matches!(folded.nodes()[2].op(), TraceOp::Div),
        "NaN result should not be folded"
    );
}

#[test]
fn test_no_fold_inf_result() {
    // Constant(1.0) / Constant(0.0) produces Inf — should NOT be folded.
    let graph = ComputationGraph::from_nodes(vec![
        const_node(0, 1.0, &[4]),
        const_node(1, 0.0, &[4]),
        binary_node(2, "div_0", TraceOp::Div, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    assert!(
        matches!(folded.nodes()[2].op(), TraceOp::Div),
        "Inf result should not be folded"
    );
}

// -- Identity simplification --------------------------------------------------

#[test]
fn test_simplify_add_zero_rhs() {
    // Input + Constant(0) → forward Input
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        const_node(1, 0.0, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    // Node 2 should be a Reshape (passthrough) forwarding input 0.
    match folded.nodes()[2].op() {
        TraceOp::Reshape { .. } => {
            assert_eq!(folded.nodes()[2].inputs(), &[0], "should forward input_0");
        }
        other => panic!("expected Reshape passthrough, got {other:?}"),
    }
}

#[test]
fn test_simplify_add_zero_lhs() {
    // Constant(0) + Input → forward Input
    let graph = ComputationGraph::from_nodes(vec![
        const_node(0, 0.0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    match folded.nodes()[2].op() {
        TraceOp::Reshape { .. } => {
            assert_eq!(folded.nodes()[2].inputs(), &[1]);
        }
        other => panic!("expected Reshape passthrough, got {other:?}"),
    }
}

#[test]
fn test_simplify_mul_one() {
    // Input * Constant(1) → forward Input
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        const_node(1, 1.0, &[4]),
        binary_node(2, "mul_0", TraceOp::Mul, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    match folded.nodes()[2].op() {
        TraceOp::Reshape { .. } => {
            assert_eq!(folded.nodes()[2].inputs(), &[0]);
        }
        other => panic!("expected Reshape passthrough, got {other:?}"),
    }
}

#[test]
fn test_simplify_mul_zero() {
    // Input * Constant(0) → Constant(0)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        const_node(1, 0.0, &[4]),
        binary_node(2, "mul_0", TraceOp::Mul, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    match folded.nodes()[2].op() {
        TraceOp::Constant { value } => {
            assert_eq!(*value, 0.0);
        }
        other => panic!("expected Constant(0.0), got {other:?}"),
    }
}

#[test]
fn test_simplify_div_one() {
    // Input / Constant(1) → forward Input
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        const_node(1, 1.0, &[4]),
        binary_node(2, "div_0", TraceOp::Div, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    match folded.nodes()[2].op() {
        TraceOp::Reshape { .. } => {
            assert_eq!(folded.nodes()[2].inputs(), &[0]);
        }
        other => panic!("expected Reshape passthrough, got {other:?}"),
    }
}

#[test]
fn test_simplify_sub_zero() {
    // Input - Constant(0) → forward Input
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        const_node(1, 0.0, &[4]),
        binary_node(2, "sub_0", TraceOp::Sub, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    match folded.nodes()[2].op() {
        TraceOp::Reshape { .. } => {
            assert_eq!(folded.nodes()[2].inputs(), &[0]);
        }
        other => panic!("expected Reshape passthrough, got {other:?}"),
    }
}

// -- Mixed: non-constant inputs preserved ------------------------------------

#[test]
fn test_non_constant_inputs_preserved() {
    // Input + Input should not be folded.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    assert!(
        matches!(folded.nodes()[2].op(), TraceOp::Add),
        "non-constant add should be preserved"
    );
}

#[test]
fn test_remap_propagates_through_chain() {
    // Input(0) + Constant(0)(1) → simplify to Input(0)
    // Then result(2) * Constant(2.0)(3) should reference Input(0) directly.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        const_node(1, 0.0, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
        const_node(3, 2.0, &[4]),
        binary_node(4, "mul_0", TraceOp::Mul, 2, 3, &[4]),
    ]);
    let folded = constant_fold(&graph);
    // Node 2: add(input, 0) → Reshape forwarding 0.
    // Node 4: mul(2, 3) → inputs remapped from [2, 3] to [0, 3].
    let node4_inputs = folded.nodes()[4].inputs();
    assert_eq!(
        node4_inputs[0], 0,
        "first input should be remapped to input_0"
    );
    assert_eq!(node4_inputs[1], 3, "second input should remain const_3");
}

// -- Dispatch impact measurement (#1815, #3083) --------------------------------

/// Constant folding eliminates standalone constant-constant dispatches.
///
/// Graph: Const(3.0) → Const(2.0) → Mul(const, const) → Relu → output
///
/// Without CF: Mul(Const, Const) is a fusible op that chains with Relu → 1 fused
/// dispatch (2-op chain). Both ops dispatch to GPU.
/// With CF: Mul(3.0, 2.0) → Const(6.0). Relu(Const(6.0)) → Const(6.0) since relu(6)=6.
/// Entire graph folds to constants. Zero dispatches.
///
/// Also tests: in a mixed graph (some constants, some inputs), CF reduces
/// dispatches by folding the constant subgraph while leaving the variable
/// subgraph intact.
#[test]
fn test_constant_fold_eliminates_constant_subgraph_dispatches() {
    use crate::trace_compile::{
        compile_trace_to_plan_with_fusion, compile_trace_with_fusion, CompiledStep,
    };

    let shape = &[1, 4];

    // Mixed graph: constant subgraph (nodes 0-4) + variable subgraph (nodes 5-7).
    // The constant subgraph's output is consumed by a binary op with the variable.
    let graph = ComputationGraph::from_nodes(vec![
        // Constant subgraph: Const(3) * Const(2) = 6, then Relu(6) = 6.
        const_node(0, 3.0, shape),
        const_node(1, 2.0, shape),
        binary_node(2, "const_mul", TraceOp::Mul, 0, 1, shape),
        TraceNode::new(
            3,
            "relu_const".into(),
            TraceOp::Relu,
            vec![2],
            shape.to_vec(),
            DType::F32,
        ),
        // Variable subgraph: Input → sigmoid
        input_node(4, shape),
        TraceNode::new(
            5,
            "sigmoid".into(),
            TraceOp::Sigmoid,
            vec![4],
            shape.to_vec(),
            DType::F32,
        ),
        // Combine: variable_result * constant_subgraph_result
        binary_node(6, "mul_combine", TraceOp::Mul, 5, 3, shape),
    ]);

    // Without CF (fusion only):
    // Chain: const_mul(0,1) → relu(2) → ... — this chain has fan-in from const_mul
    // The fusion engine sees Mul(0,1), Relu(2), and then Mul(5,3) can't chain from
    // sigmoid because sigmoid has use_count=1 but its consumer is mul_combine(6).
    // Actual: multiple dispatches for the constant + variable subgraphs.
    let without_cf_steps = compile_trace_with_fusion(&graph).unwrap();
    let without_cf_dispatches = without_cf_steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();

    // With CF: Const(3)*Const(2)→6, Relu(6)→6. Graph reduces to:
    // Input → sigmoid → mul(sigmoid, Const(6)) → 1 fused dispatch (sigmoid→mul).
    let plan = compile_trace_to_plan_with_fusion(&graph).unwrap();
    let with_cf_dispatches = plan
        .steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();

    assert!(
        with_cf_dispatches < without_cf_dispatches,
        "CF should reduce dispatches: with={with_cf_dispatches}, without={without_cf_dispatches}"
    );
}

/// Pure constant subgraph folds to zero dispatches.
///
/// Graph: Const(1.0) → Exp → Const(0.5) → Add → output
/// Without CF: Exp + Add = 2-op fused chain (1 dispatch).
/// With CF: exp(1.0) ≈ 2.718, 2.718 + 0.5 ≈ 3.218. All folded. Zero dispatches.
#[test]
fn test_constant_fold_eliminates_pure_constant_chain() {
    use crate::trace_compile::{compile_trace_to_plan_with_fusion, CompiledStep};

    let shape = &[1, 4];
    let graph = ComputationGraph::from_nodes(vec![
        const_node(0, 1.0, shape),
        TraceNode::new(
            1,
            "exp".into(),
            TraceOp::Exp,
            vec![0],
            shape.to_vec(),
            DType::F32,
        ),
        const_node(2, 0.5, shape),
        binary_node(3, "add", TraceOp::Add, 1, 2, shape),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).unwrap();
    let dispatches = plan
        .steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();
    let constant_steps = plan
        .steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::ConstantValue { .. }))
        .count();

    // After folding: exp(1.0) → Const(e), Const(e) + Const(0.5) → Const(e + 0.5).
    // Zero dispatches — entire graph is constant.
    assert_eq!(
        dispatches, 0,
        "pure constant chain should fold to zero dispatches"
    );
    assert!(
        constant_steps > 0,
        "should have at least one ConstantValue step"
    );
}

// -- Output preservation ------------------------------------------------------

#[test]
fn test_output_node_preserved_after_fold() {
    // Graph output should be preserved after folding.
    let graph = ComputationGraph::from_nodes(vec![
        const_node(0, 2.0, &[4]),
        const_node(1, 3.0, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let folded = constant_fold(&graph);
    let output = folded.output_node().expect("should have output");
    assert_eq!(output.id(), 2);
    match output.op() {
        TraceOp::Constant { value } => assert!((value - 5.0).abs() < 1e-10),
        other => panic!("output should be folded to Constant(5.0), got {other:?}"),
    }
}
