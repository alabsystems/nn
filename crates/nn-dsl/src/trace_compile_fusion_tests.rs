// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for elementwise chain fusion in [`compile_trace_with_fusion`].

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::compile_trace_with_fusion;
use crate::ir::{IRNodeKind, KernelDef};
use crate::trace_compile::CompiledStep;
use crate::TensorOpKind;

// -- Helpers ------------------------------------------------------------------

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

/// Count how many Dispatch steps (GPU kernel launches) are in the compiled plan.
fn count_dispatches(steps: &[CompiledStep]) -> usize {
    steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count()
}

/// Count how many non-dispatch, non-passthrough steps are in the compiled
/// plan (inputs + identity passthroughs / fusion placeholders).
fn count_non_dispatches(steps: &[CompiledStep]) -> usize {
    steps
        .iter()
        .filter(|s| {
            matches!(
                s,
                CompiledStep::InputForward | CompiledStep::IdentityPassthrough
            )
        })
        .count()
}

/// Count input nodes in a TensorKernelDef by inspecting its IR nodes.
fn count_def_inputs(def: &crate::TensorKernelDef) -> usize {
    def.nodes
        .iter()
        .filter(|n| matches!(&n.kind, TensorOpKind::Input { .. }))
        .count()
}

fn constant_node(id: u64, value: f64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("const_{id}"),
        TraceOp::Constant { value },
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

/// Extract the scalar `KernelDef` from the Elementwise node inside a
/// compiled dispatch step. Returns `None` if not found.
fn extract_scalar_kernel(step: &CompiledStep) -> Option<&KernelDef> {
    match step {
        CompiledStep::Dispatch { kernel, .. } => {
            for node in &kernel.def().nodes {
                if let TensorOpKind::Elementwise { kernel, .. } = &node.kind {
                    return Some(kernel);
                }
            }
            None
        }
        _ => None,
    }
}

/// Count Param nodes in a scalar KernelDef.
fn count_kernel_params(kernel: &KernelDef) -> usize {
    kernel
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, IRNodeKind::Param(_)))
        .count()
}

// -- Tests: is_fusible_elementwise coverage -----------------------------------

#[test]
fn test_single_unary_exp_no_fusion() {
    // A single fusible op compiles individually (no fusion benefit).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[0], CompiledStep::InputForward));
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
    assert_eq!(count_dispatches(&steps), 1);
}

// -- Tests: chain detection and fusion ----------------------------------------

#[test]
fn test_fuse_two_unary_ops() {
    // exp → relu on same shape should fuse into one dispatch.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[8]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[8]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();

    // 3 steps total: InputForward, IdentityPassthrough(placeholder for exp), Dispatch(fused)
    assert_eq!(steps.len(), 3);
    assert!(matches!(steps[0], CompiledStep::InputForward));
    assert!(matches!(steps[1], CompiledStep::IdentityPassthrough)); // placeholder
    assert!(matches!(steps[2], CompiledStep::Dispatch { .. }));
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_fuse_three_unary_ops() {
    // exp → sqrt → tanh should fuse into one dispatch.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(2, "sqrt_0", TraceOp::Sqrt, 1, &[4]),
        unary_node(3, "tanh_0", TraceOp::Tanh, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    // Identity(input), Identity(exp), Identity(sqrt), Dispatch(fused)
    assert_eq!(count_dispatches(&steps), 1);
    assert_eq!(count_non_dispatches(&steps), 3);
}

#[test]
fn test_fused_dispatch_name_contains_chain_length() {
    // Verify the fused dispatch has a name like "fused_exp_x3".
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
        unary_node(3, "sigmoid_0", TraceOp::Sigmoid, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    match &steps[3] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert!(
                kernel.name().contains("x3"),
                "fused dispatch name should contain chain length: {}",
                kernel.name()
            );
            assert!(
                kernel.name().starts_with("fused_"),
                "fused dispatch name should start with 'fused_': {}",
                kernel.name()
            );
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

// -- Tests: binary ops in chains ----------------------------------------------

#[test]
fn test_fuse_binary_add_then_relu() {
    // Two inputs → add → relu should fuse add+relu.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
        unary_node(3, "relu_0", TraceOp::Relu, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    // 4 steps: Identity(in0), Identity(in1), Identity(add placeholder), Dispatch(fused add+relu)
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 1);

    // The fused kernel should have 2 external inputs (the two graph inputs).
    match &steps[3] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(count_def_inputs(kernel.def()), 2);
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_fuse_mul_then_add() {
    // x * y + z pattern: mul → add with input(3) between them in topology.
    // Non-consecutive detection skips input(3) and fuses mul→add.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "mul_0", TraceOp::Mul, 0, 1, &[4]),
        input_node(3, &[4]),
        binary_node(4, "add_0", TraceOp::Add, 2, 3, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 5);
    // Chain: mul(2)→add(4), non-consecutive but fused into 1 dispatch.
    assert_eq!(count_dispatches(&steps), 1);
}

// -- Tests: fan-out breaks fusion ---------------------------------------------

#[test]
fn test_fanout_breaks_chain() {
    // exp → (relu, sigmoid): exp has fan-out of 2, so chains break.
    // exp feeds both relu and sigmoid.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
        unary_node(3, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    // exp has use_count=2, so no fusion. Each op compiled individually.
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 3); // exp, relu, sigmoid
    assert_eq!(count_non_dispatches(&steps), 1); // just the input
}

// -- Tests: shape mismatch breaks chain ---------------------------------------

#[test]
fn test_shape_mismatch_breaks_chain() {
    // Two consecutive fusible ops with different output shapes → no fusion.
    // Use separate subgraphs so each op is independently compilable.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        input_node(2, &[8]),
        unary_node(3, "relu_0", TraceOp::Relu, 2, &[8]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    // exp alone (chain of 1) and relu alone (chain of 1), no fusion.
    assert_eq!(count_dispatches(&steps), 2);
}

// -- Tests: non-fusible ops break chain ---------------------------------------

#[test]
fn test_non_fusible_op_breaks_chain() {
    // exp → reshape → relu: reshape is not fusible, breaks the chain.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 3]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[2, 3]),
        TraceNode::new(
            2,
            "reshape_0".into(),
            TraceOp::Reshape {
                target_shape: vec![6],
            },
            vec![1],
            vec![6],
            DType::F32,
        ),
        unary_node(3, "relu_0", TraceOp::Relu, 2, &[6]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    // exp compiled individually, reshape is Passthrough, relu compiled individually.
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 2); // exp, relu
    assert!(matches!(steps[2], CompiledStep::Passthrough { .. }));
}

// -- Tests: Neg with weight_data ----------------------------------------------

#[test]
fn test_neg_in_chain_no_weight_data() {
    // neg → relu: scalar composition uses Literal(0.0) for neg, no weight tensor.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "neg_0", TraceOp::Neg, 0, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(count_dispatches(&steps), 1);

    // Scalar composition inlines Neg as Literal(0) + Sub — no weight_data needed.
    match &steps[2] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(
                weight_data.is_empty(),
                "fused chain with Neg should have no weight_data (uses IR Literal)"
            );
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_neg_standalone_no_fusion() {
    // Single neg — compiled individually, should still work (no fusion path).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "neg_0", TraceOp::Neg, 0, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[0], CompiledStep::InputForward));
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));

    // The standalone neg should also have weight_data (from compile_neg).
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(
                weight_data.contains_key("zero"),
                "standalone neg should have 'zero' weight"
            );
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

// -- Tests: all fusible ops compile without error -----------------------------

#[test]
fn test_all_unary_fusible_ops() {
    // Verify every unary fusible op can appear in a fused chain.
    let unary_ops = vec![
        TraceOp::Exp,
        TraceOp::Log,
        TraceOp::Sqrt,
        TraceOp::Sqr,
        TraceOp::Abs,
        TraceOp::Neg,
        TraceOp::Recip,
        TraceOp::Sin,
        TraceOp::Cos,
        TraceOp::Floor,
        TraceOp::Round,
        TraceOp::Fract,
        TraceOp::Tanh,
        TraceOp::Relu,
        TraceOp::Gelu,
        TraceOp::GeluErf,
        TraceOp::Sigmoid,
        TraceOp::Silu,
    ];

    for (i, op) in unary_ops.into_iter().enumerate() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, &format!("op_{i}"), op.clone(), 0, &[4]),
            unary_node(2, "relu_tail", TraceOp::Relu, 1, &[4]),
        ]);
        let result = compile_trace_with_fusion(&graph);
        assert!(
            result.is_ok(),
            "fusion failed for unary op index {i}: {:?}",
            result.err()
        );
        let steps = result.unwrap();
        // Should be fused: Identity(input), Identity(placeholder), Dispatch(fused)
        assert_eq!(
            count_dispatches(&steps),
            1,
            "expected 1 fused dispatch for unary op index {i}"
        );
    }
}

#[test]
fn test_all_binary_fusible_ops() {
    // Verify every binary fusible op can appear in a fused chain.
    let binary_ops = vec![
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
    ];

    for (i, op) in binary_ops.into_iter().enumerate() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            input_node(1, &[4]),
            binary_node(2, &format!("op_{i}"), op, 0, 1, &[4]),
            unary_node(3, "relu_tail", TraceOp::Relu, 2, &[4]),
        ]);
        let result = compile_trace_with_fusion(&graph);
        assert!(
            result.is_ok(),
            "fusion failed for binary op index {i}: {:?}",
            result.err()
        );
        let steps = result.unwrap();
        assert_eq!(
            count_dispatches(&steps),
            1,
            "expected 1 fused dispatch for binary op index {i}"
        );
    }
}

// -- Tests: unsupported op in chain -------------------------------------------

#[test]
fn test_unsupported_op_returns_error() {
    // A Custom op is not fusible and returns UnsupportedTraceOp.
    // This is handled by compile_node, not the fusion code directly,
    // but verify the integration works.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "custom_0".into(),
            TraceOp::Custom {
                name: "unknown_op".into(),
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let result = compile_trace_with_fusion(&graph);
    assert!(result.is_err());
}

// -- Tests: empty and trivial graphs ------------------------------------------

#[test]
fn test_empty_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert!(steps.is_empty());
}

#[test]
fn test_input_only_graph() {
    let graph = ComputationGraph::from_nodes(vec![input_node(0, &[4])]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 1);
    assert!(matches!(steps[0], CompiledStep::InputForward));
}

// -- Tests: mixed fusible and non-fusible ops ---------------------------------

#[test]
fn test_mixed_fusible_and_nonfusible() {
    // input → exp → relu → reduce_sum → sigmoid
    // exp+relu should fuse, reduce_sum breaks chain, sigmoid standalone.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4, 8]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4, 8]),
        TraceNode::new(
            3,
            "reduce_0".into(),
            TraceOp::ReduceSum {
                dim: 1,
                keepdim: false,
            },
            vec![2],
            vec![4],
            DType::F32,
        ),
        unary_node(4, "sigmoid_0", TraceOp::Sigmoid, 3, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 5);
    // Fused exp+relu = 1 dispatch, reduce_sum = 1 dispatch, sigmoid = 1 dispatch
    assert_eq!(count_dispatches(&steps), 3);
    // Identity: input + exp placeholder = 2
    assert_eq!(count_non_dispatches(&steps), 2);
}

// -- Tests: step count alignment with node count ------------------------------

#[test]
fn test_step_count_equals_node_count() {
    // The number of compiled steps must always equal the number of graph nodes,
    // because CompiledModel's edge_map relies on this 1:1 correspondence.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
        unary_node(3, "sigmoid_0", TraceOp::Sigmoid, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(
        steps.len(),
        4,
        "step count must equal node count for edge_map alignment"
    );
}

// -- Tests: error paths for malformed graphs ---------------------------------

#[test]
fn test_fusion_missing_external_input_returns_error() {
    // A fusible node references a graph node ID that doesn't exist in the graph.
    // The topology validation should catch this before fusion even runs.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        // exp references input 0 (exists) — fine
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        // add references exp(1) and a non-existent node 999
        binary_node(2, "add_0", TraceOp::Add, 1, 999, &[4]),
    ]);
    let result = compile_trace_with_fusion(&graph);
    assert!(
        result.is_err(),
        "fusion with missing external input should error, not silently substitute shape"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("input_id 999") || err_msg.contains("missing input"),
        "error should reference the missing node: {err_msg}"
    );
}

// -- Tests: Silu + GeluErf fused chains ---------------------------------------

#[test]
fn test_fuse_silu_then_relu() {
    // silu -> relu should fuse. Silu decomposes to sigmoid(x)*x within the
    // fused builder, but the chain should still produce a single dispatch.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "silu_0", TraceOp::Silu, 0, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_gelu_erf_fuses_in_chain() {
    // GeluErf decomposes to A&S 7.1.26 polynomial in scalar IR.
    // gelu_erf → mul: fused into 1 dispatch.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        unary_node(2, "gelu_erf_0", TraceOp::GeluErf, 0, &[4]),
        binary_node(3, "mul_0", TraceOp::Mul, 2, 1, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 1); // gelu_erf + mul fused
}

// -- Tests: fused chain with external inputs from both sides ------------------

#[test]
fn test_fused_chain_external_inputs() {
    // input0 → exp → add(exp_out, input1) → relu
    // The fused chain is exp→add→relu. add has an external input (input1).
    // BUT input1 is at index 1 in topological order, exp is at index 2.
    // The chain scanner starts at exp(2), but input1(1) is before it
    // and not fusible, so it won't be part of the chain scan.
    // Since add(3) references both exp(2) and input1(1), and exp(2) is
    // the previous chain element, the chain extends to add.
    // Then relu(4) references add(3), extending further.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        unary_node(2, "exp_0", TraceOp::Exp, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 2, 1, &[4]),
        unary_node(4, "relu_0", TraceOp::Relu, 3, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 5);
    // Chain: exp→add→relu fused into 1 dispatch.
    // Steps: Identity(in0), Identity(in1), Identity(exp placeholder),
    //        Identity(add placeholder), Dispatch(fused)
    assert_eq!(count_dispatches(&steps), 1);

    // The fused kernel should have 2 external inputs.
    match &steps[4] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(
                count_def_inputs(kernel.def()),
                2,
                "fused exp→add→relu should have 2 external inputs"
            );
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

// -- Tests: newly fusible ops (Silu, GeluErf, Maximum, Minimum) ---------------

#[test]
fn test_fuse_silu_then_mul() {
    // silu → mul: silu is a composite (sigmoid+mul) but fusible in chains.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        unary_node(2, "silu_0", TraceOp::Silu, 0, &[4]),
        binary_node(3, "mul_0", TraceOp::Mul, 2, 1, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    // silu→mul fused into 1 dispatch.
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_fuse_exp_then_silu() {
    // exp → silu on same shape should fuse into one dispatch.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[8]),
        unary_node(2, "silu_0", TraceOp::Silu, 1, &[8]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_fuse_maximum_then_relu() {
    // maximum(a, b) → relu: binary min/max fusible with unary activations.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "max_0", TraceOp::Maximum, 0, 1, &[4]),
        unary_node(3, "relu_0", TraceOp::Relu, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_fuse_minimum_then_sigmoid() {
    // minimum(a, b) → sigmoid: binary min/max fusible with unary activations.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "min_0", TraceOp::Minimum, 0, 1, &[4]),
        unary_node(3, "sigmoid_0", TraceOp::Sigmoid, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_gelu_erf_fuses_with_add() {
    // GeluErf decomposes to A&S 7.1.26 polynomial in scalar IR.
    // gelu_erf → add: fused into 1 dispatch.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        unary_node(2, "gelu_erf_0", TraceOp::GeluErf, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 2, 1, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 1); // gelu_erf + add fused
}

// -- Tests: parameterized activations in fusion chains -----------------------

#[test]
fn test_fuse_leaky_relu_then_exp() {
    // leaky_relu(slope=0.2) → exp: parameterized activation fusible in chains.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "leaky_0", TraceOp::LeakyRelu { slope: 0.2 }, 0, &[4]),
        unary_node(2, "exp_0", TraceOp::Exp, 1, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_fuse_elu_then_relu() {
    // elu(alpha=1.0) → relu: ELU is fusible with parameterized alpha.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "elu_0", TraceOp::Elu { alpha: 1.0 }, 0, &[8]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[8]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_fuse_exp_then_clamp() {
    // exp → clamp(min=0, max=6): clamp is fusible with literal bounds.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(
            2,
            "clamp_0",
            TraceOp::Clamp {
                min: Some(0.0),
                max: Some(6.0),
            },
            1,
            &[4],
        ),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_fuse_clamp_min_only() {
    // clamp(min=0, max=None) → relu: partial clamp fusible.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(
            1,
            "clamp_0",
            TraceOp::Clamp {
                min: Some(0.0),
                max: None,
            },
            0,
            &[4],
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_fuse_powf_then_add() {
    // powf(2.0) → add: power function fusible via exp(exponent * log(x)).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        unary_node(2, "pow_0", TraceOp::Powf { exponent: 2.0 }, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 2, 1, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_fuse_atan2_then_abs() {
    // atan2(a, b) → abs: binary trig fusible with unary math.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "atan2_0", TraceOp::Atan2, 0, 1, &[4]),
        unary_node(3, "abs_0", TraceOp::Abs, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_fuse_leaky_relu_mul_clamp_chain() {
    // leaky_relu → mul → clamp: 3-op chain with mixed parameterized ops.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        unary_node(2, "leaky_0", TraceOp::LeakyRelu { slope: 0.1 }, 0, &[4]),
        binary_node(3, "mul_0", TraceOp::Mul, 2, 1, &[4]),
        unary_node(
            4,
            "clamp_0",
            TraceOp::Clamp {
                min: Some(-1.0),
                max: Some(1.0),
            },
            3,
            &[4],
        ),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 5);
    // Chain: leaky_relu → mul → clamp = 1 fused dispatch.
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_all_new_fusible_ops_compile() {
    // Verify each new fusible op can appear in a chain without error.
    let new_unary_ops: Vec<TraceOp> = vec![
        TraceOp::LeakyRelu { slope: 0.01 },
        TraceOp::Elu { alpha: 1.0 },
        TraceOp::Clamp {
            min: Some(0.0),
            max: Some(1.0),
        },
        TraceOp::Clamp {
            min: Some(0.0),
            max: None,
        },
        TraceOp::Clamp {
            min: None,
            max: Some(1.0),
        },
        TraceOp::Powf { exponent: 0.5 },
    ];

    for (i, op) in new_unary_ops.into_iter().enumerate() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, &format!("op_{i}"), op, 0, &[4]),
            unary_node(2, "relu_tail", TraceOp::Relu, 1, &[4]),
        ]);
        let result = compile_trace_with_fusion(&graph);
        assert!(
            result.is_ok(),
            "fusion failed for new unary op index {i}: {:?}",
            result.err()
        );
        assert_eq!(
            count_dispatches(&result.unwrap()),
            1,
            "expected 1 fused dispatch for new unary op index {i}"
        );
    }

    // Atan2 (binary)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "atan2_0", TraceOp::Atan2, 0, 1, &[4]),
        unary_node(3, "relu_tail", TraceOp::Relu, 2, &[4]),
    ]);
    let result = compile_trace_with_fusion(&graph);
    assert!(
        result.is_ok(),
        "fusion failed for atan2: {:?}",
        result.err()
    );
    assert_eq!(count_dispatches(&result.unwrap()), 1);
}

// -- Tests: non-consecutive chain detection -----------------------------------

fn constant_weight_node(id: u64, shape: &[usize]) -> TraceNode {
    use nn_core::dyn_tensor::trace::WeightRef;
    let data = vec![1.0_f32; shape.iter().product::<usize>()];
    let weight = WeightRef::new(data, shape.to_vec()).unwrap();
    TraceNode::new(
        id,
        format!("weight_{id}"),
        TraceOp::ConstantWeight { weight },
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

#[test]
fn test_nonconsecutive_weight_between_fusible_ops() {
    // input(0) → exp(1) → [weight(2)] → mul(3, inputs: exp+weight) → relu(4)
    // ConstantWeight at index 2 sits between exp and mul in topological order.
    // Non-consecutive detection should fuse exp→mul→relu into 1 dispatch.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        constant_weight_node(2, &[4]),
        binary_node(3, "mul_0", TraceOp::Mul, 1, 2, &[4]),
        unary_node(4, "relu_0", TraceOp::Relu, 3, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 5);
    // Chain: exp(1)→mul(3)→relu(4), skipping weight(2).
    // Steps: Identity(in0), Identity(exp placeholder), NativeOp(weight),
    //        Identity(mul placeholder), Dispatch(fused)
    assert_eq!(
        count_dispatches(&steps),
        1,
        "exp→mul→relu should fuse even with weight between exp and mul"
    );
}

#[test]
fn test_nonconsecutive_two_weights_in_chain() {
    // input(0) → exp(1) → [weight_a(2)] → mul(3) → [weight_b(4)] → add(5) → relu(6)
    // Two weights interleaved. Chain: exp→mul→add→relu (4 ops fused).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        constant_weight_node(2, &[4]),
        binary_node(3, "mul_0", TraceOp::Mul, 1, 2, &[4]),
        constant_weight_node(4, &[4]),
        binary_node(5, "add_0", TraceOp::Add, 3, 4, &[4]),
        unary_node(6, "relu_0", TraceOp::Relu, 5, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 7);
    assert_eq!(
        count_dispatches(&steps),
        1,
        "exp→mul→add→relu should fuse with interleaved weights"
    );
    // Verify the fused kernel name contains x4 (4 ops).
    match &steps[6] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert!(
                kernel.name().contains("x4"),
                "fused dispatch should have 4-op chain: {}",
                kernel.name()
            );
        }
        other => panic!("expected Dispatch at last position, got {other:?}"),
    }
}

#[test]
fn test_nonconsecutive_input_between_fusible_ops() {
    // input(0) → exp(1) → [input(2)] → add(3, inputs: exp+input2) → relu(4)
    // An extra Input node sits between exp and add.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        input_node(2, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[4]),
        unary_node(4, "relu_0", TraceOp::Relu, 3, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 5);
    assert_eq!(
        count_dispatches(&steps),
        1,
        "exp→add→relu should fuse with input between exp and add"
    );
}

#[test]
fn test_nonconsecutive_weight_node_compiled_normally() {
    // Verify the weight node between chain members is compiled as NativeOp::ConstantWeight
    // (not IdentityPassthrough or Dispatch). ConstantWeight carries pre-computed data
    // that is uploaded to GPU at execution time (changed from InputForward in 0f8a5ef57).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        constant_weight_node(2, &[4]),
        binary_node(3, "mul_0", TraceOp::Mul, 1, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    // steps[2] is the weight node — should be NativeOp::ConstantWeight.
    assert!(
        matches!(&steps[2], CompiledStep::NativeOp { op, .. }
            if matches!(op, crate::NativeOpKind::ConstantWeight { .. })),
        "weight node between chain members should be NativeOp::ConstantWeight, got {:?}",
        steps[2]
    );
}

// -- Tests: constant inlining (D2) -------------------------------------------

#[test]
fn test_constant_inlined_as_literal() {
    // sigmoid(x) * 0.5: sigmoid → Constant(0.5) → Mul
    // The Constant should be inlined as a Literal IR node, NOT a Param.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
        constant_node(2, 0.5, &[]),
        binary_node(3, "mul_0", TraceOp::Mul, 1, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 1);

    // The fused TensorKernelDef should have 1 Input (x), not 2.
    match &steps[3] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(
                count_def_inputs(kernel.def()),
                1,
                "sigmoid * 0.5 should have 1 external input (x), constant inlined"
            );
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }

    // The scalar KernelDef should contain Literal(0.5) and only 1 Param.
    let scalar_kernel = extract_scalar_kernel(&steps[3]).expect("should have Elementwise node");
    assert_eq!(
        count_kernel_params(scalar_kernel),
        1,
        "only 1 Param (for x); constant 0.5 should be inlined as Literal"
    );
    // Sigmoid alone creates several Literals (0.0, 1.0). Adding 0.5 for the
    // Constant inlining means total Literals > 2. Just verify at least one
    // Literal has value 0.5.
    let has_half = scalar_kernel
        .nodes
        .iter()
        .any(|n| matches!(n.kind, IRNodeKind::Literal(v) if (v - 0.5).abs() < 1e-12));
    assert!(has_half, "scalar kernel should contain Literal(0.5)");
}

#[test]
fn test_multi_constant_chain_inlined() {
    // x * 2.0 + 1.0: Mul(x, Constant(2.0)) → Add(result, Constant(1.0))
    // Both constants should be inlined. Only 1 Param (for x).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        constant_node(1, 2.0, &[]),
        binary_node(2, "mul_0", TraceOp::Mul, 0, 1, &[4]),
        constant_node(3, 1.0, &[]),
        binary_node(4, "add_0", TraceOp::Add, 2, 3, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 5);
    assert_eq!(count_dispatches(&steps), 1);

    match &steps[4] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(
                count_def_inputs(kernel.def()),
                1,
                "x*2+1 should have 1 external input (x)"
            );
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }

    let scalar_kernel = extract_scalar_kernel(&steps[4]).expect("should have Elementwise node");
    assert_eq!(
        count_kernel_params(scalar_kernel),
        1,
        "only 1 Param (x); both 2.0 and 1.0 inlined as Literals"
    );
    let has_two = scalar_kernel
        .nodes
        .iter()
        .any(|n| matches!(n.kind, IRNodeKind::Literal(v) if (v - 2.0).abs() < 1e-12));
    let has_one = scalar_kernel
        .nodes
        .iter()
        .any(|n| matches!(n.kind, IRNodeKind::Literal(v) if (v - 1.0).abs() < 1e-12));
    assert!(has_two, "scalar kernel should contain Literal(2.0)");
    assert!(has_one, "scalar kernel should contain Literal(1.0)");
}

#[test]
#[allow(clippy::approx_constant)]
fn test_scalar_constant_weight_inlined() {
    // exp(x) * scalar_weight: scalar ConstantWeight (1 element) should be
    // inlined as Literal, not kept as Param.
    use nn_core::dyn_tensor::trace::WeightRef;
    let weight = WeightRef::new(vec![3.14_f32], vec![]).unwrap();
    let scalar_weight_node = TraceNode::new(
        2,
        "scalar_weight".into(),
        TraceOp::ConstantWeight { weight },
        vec![],
        vec![],
        DType::F32,
    );

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        scalar_weight_node,
        binary_node(3, "mul_0", TraceOp::Mul, 1, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 1);

    let scalar_kernel = extract_scalar_kernel(&steps[3]).expect("should have Elementwise node");
    assert_eq!(
        count_kernel_params(scalar_kernel),
        1,
        "scalar ConstantWeight should be inlined as Literal, not Param"
    );
    let has_pi = scalar_kernel
        .nodes
        .iter()
        .any(|n| matches!(n.kind, IRNodeKind::Literal(v) if (v - 3.14).abs() < 0.01));
    assert!(has_pi, "scalar kernel should contain Literal(~3.14)");
}

#[test]
fn test_non_scalar_constant_weight_remains_param() {
    // exp(x) * multi_element_weight: non-scalar ConstantWeight should NOT be
    // inlined — it stays as a Param (GPU buffer binding).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        constant_weight_node(2, &[4]), // 4-element weight, NOT scalar
        binary_node(3, "mul_0", TraceOp::Mul, 1, 2, &[4]),
    ]);
    let steps = compile_trace_with_fusion(&graph).unwrap();
    assert_eq!(steps.len(), 4);
    assert_eq!(count_dispatches(&steps), 1);

    let scalar_kernel = extract_scalar_kernel(&steps[3]).expect("should have Elementwise node");
    assert_eq!(
        count_kernel_params(scalar_kernel),
        2,
        "non-scalar ConstantWeight should remain as Param (x + weight = 2 Params)"
    );
}

// -- D5: Kokoro-pattern dispatch reduction measurement (#1815) ----------------

/// Kokoro step_regulate pattern: sigmoid → Constant(1/speed) → mul → clamp.
///
/// Without D1 (Constant skip): sigmoid is standalone, Constant breaks chain,
/// mul is standalone, clamp is standalone = 3 dispatches.
/// With D1+D2: sigmoid → mul → clamp fuses into 1 dispatch, Constant inlined.
#[test]
fn test_kokoro_step_regulate_pattern_fuses() {
    // sigmoid(x) → Constant(speed) → mul(sigmoid, const) → clamp(1, max_dur)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4]),
        unary_node(1, "sigmoid", TraceOp::Sigmoid, 0, &[1, 4]),
        constant_node(2, 0.5, &[1, 4]), // 1/speed scalar
        binary_node(3, "mul", TraceOp::Mul, 1, 2, &[1, 4]),
        // clamp(1.0, 20.0) — Kokoro duration clamping
        TraceNode::new(
            4,
            "clamp".into(),
            TraceOp::Clamp {
                min: Some(1.0),
                max: Some(20.0),
            },
            vec![3],
            vec![1, 4],
            DType::F32,
        ),
    ]);

    // With fusion: sigmoid → mul → clamp = 1 dispatch
    let fused_steps = compile_trace_with_fusion(&graph).unwrap();
    let fused_dispatches = count_dispatches(&fused_steps);

    // Without fusion: sigmoid, mul, clamp = 3 dispatches
    let unfused_steps = crate::trace_compile::compile_trace(&graph).unwrap();
    let unfused_dispatches = count_dispatches(&unfused_steps);

    assert_eq!(
        fused_dispatches, 1,
        "step_regulate pattern should fuse to 1 dispatch"
    );
    assert_eq!(
        unfused_dispatches, 3,
        "unfused should be 3 separate dispatches"
    );

    // Verify Constant is inlined as Literal (not Param).
    let scalar_kernel =
        extract_scalar_kernel(&fused_steps[4]).expect("last step should have Elementwise node");
    assert_eq!(
        count_kernel_params(scalar_kernel),
        1,
        "Constant(0.5) should be inlined as Literal, leaving only 1 Param (x)"
    );
}

/// Kokoro affine transform pattern: x * scale + offset (mul_scalar + add_scalar).
///
/// Without D1: x→Constant(scale)→mul→Constant(offset)→add = 2 dispatches (mul, add standalone).
/// With D1+D2: mul→add fuses into 1 dispatch, both Constants inlined.
#[test]
fn test_kokoro_affine_transform_pattern_fuses() {
    // x → Constant(scale) → mul(x, scale) → Constant(offset) → add(mul, offset)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8]),
        constant_node(1, 2.5, &[1, 8]), // scale
        binary_node(2, "mul", TraceOp::Mul, 0, 1, &[1, 8]),
        constant_node(3, 0.1, &[1, 8]), // offset
        binary_node(4, "add", TraceOp::Add, 2, 3, &[1, 8]),
    ]);

    let fused_steps = compile_trace_with_fusion(&graph).unwrap();
    let fused_dispatches = count_dispatches(&fused_steps);

    let unfused_steps = crate::trace_compile::compile_trace(&graph).unwrap();
    let unfused_dispatches = count_dispatches(&unfused_steps);

    assert_eq!(
        fused_dispatches, 1,
        "affine x*s+b should fuse to 1 dispatch"
    );
    assert_eq!(unfused_dispatches, 2, "unfused: mul + add = 2 dispatches");

    // Both constants should be inlined as Literals.
    let scalar_kernel =
        extract_scalar_kernel(&fused_steps[4]).expect("last step should have Elementwise node");
    assert_eq!(
        count_kernel_params(scalar_kernel),
        1,
        "both Constants should be inlined, leaving 1 Param (x)"
    );
}

/// Kokoro residual scale: add(x, h) → Constant(1/√2) → mul.
///
/// Without D1: add is standalone, mul is standalone = 2 dispatches.
/// With D1+D2: add → mul fuses into 1 dispatch, Constant inlined.
#[test]
fn test_kokoro_residual_scale_pattern_not_fused() {
    // add(x, h) → Constant(1/√2) → mul(add, const)
    // This [Add, Mul(scalar Constant)] pattern is intentionally NOT fused
    // so the resblock peephole pass (Pass 2) can see the standalone `add`
    // and fuse it into FusedResBlock with residual_scale absorption.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        input_node(1, &[1, 4, 16]),
        binary_node(2, "add", TraceOp::Add, 0, 1, &[1, 4, 16]),
        constant_node(3, std::f64::consts::FRAC_1_SQRT_2, &[1, 4, 16]),
        binary_node(4, "mul", TraceOp::Mul, 2, 3, &[1, 4, 16]),
    ]);

    let fused_steps = compile_trace_with_fusion(&graph).unwrap();
    let fused_dispatches = count_dispatches(&fused_steps);

    let unfused_steps = crate::trace_compile::compile_trace(&graph).unwrap();
    let unfused_dispatches = count_dispatches(&unfused_steps);

    assert_eq!(
        fused_dispatches, 2,
        "add + mul(scalar Constant) should NOT fuse — kept separate for resblock peephole"
    );
    assert_eq!(unfused_dispatches, 2, "unfused: add + mul = 2 dispatches");
}

/// FusionStats correctly reports chain count and dispatch savings.
#[test]
fn test_fusion_stats_kokoro_pattern() {
    use crate::trace_compile::compile_trace_to_plan_with_fusion;
    // 2 fusible chains: sigmoid→mul (2-op), exp→mul→add (3-op)
    let shape = &[1, 4];
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, shape),
        unary_node(1, "sigmoid", TraceOp::Sigmoid, 0, shape),
        constant_node(2, 0.5, shape),
        binary_node(3, "mul_c1", TraceOp::Mul, 1, 2, shape),
        // Gap
        input_node(4, shape),
        unary_node(5, "exp", TraceOp::Exp, 4, shape),
        constant_node(6, 2.0, shape),
        binary_node(7, "mul_c2", TraceOp::Mul, 5, 6, shape),
        constant_node(8, 1.0, shape),
        binary_node(9, "add_c2", TraceOp::Add, 7, 8, shape),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).unwrap();
    let stats = plan.fusion_stats();
    assert_eq!(stats.fused_chains, 2, "should detect 2 fused chains");
    assert_eq!(stats.fused_ops, 5, "chain1=2 + chain2=3 = 5 fused ops");
    assert_eq!(
        stats.dispatches_saved, 3,
        "5 fused ops - 2 chains = 3 saved"
    );
}

/// Composite Kokoro pipeline: multiple fusible segments separated by non-fusible ops.
///
/// Models a realistic pattern: fusible_chain_1 → conv1d → fusible_chain_2 → lstm → fusible_chain_3.
/// Each chain benefits from Constant skipping. Measures total dispatch reduction.
#[test]
fn test_kokoro_composite_dispatch_reduction() {
    // Chain 1: sigmoid → Constant(0.5) → mul → clamp  (Kokoro step_regulate)
    // [non-fusible]: Conv1d placeholder
    // Chain 2: exp → Constant(2.0) → mul → Constant(1.0) → add  (affine)
    // Chain 3: relu → Constant(0.1) → mul → add_with_input  (leaky relu approximation + residual)
    let shape = &[1, 4];
    let graph = ComputationGraph::from_nodes(vec![
        // Chain 1: sigmoid → mul_scalar(0.5) → clamp
        input_node(0, shape),
        unary_node(1, "sigmoid", TraceOp::Sigmoid, 0, shape),
        constant_node(2, 0.5, shape),
        binary_node(3, "mul_c1", TraceOp::Mul, 1, 2, shape),
        TraceNode::new(
            4,
            "clamp".into(),
            TraceOp::Clamp {
                min: Some(1.0),
                max: Some(20.0),
            },
            vec![3],
            shape.to_vec(),
            DType::F32,
        ),
        // Non-fusible gap (simulated as a separate input standing in for conv1d output)
        input_node(5, shape),
        // Chain 2: exp → mul_scalar(2.0) → add_scalar(1.0)
        unary_node(6, "exp", TraceOp::Exp, 5, shape),
        constant_node(7, 2.0, shape),
        binary_node(8, "mul_c2", TraceOp::Mul, 6, 7, shape),
        constant_node(9, 1.0, shape),
        binary_node(10, "add_c2", TraceOp::Add, 8, 9, shape),
        // Non-fusible gap
        input_node(11, shape),
        // Chain 3: tanh → mul_scalar(0.1) → add(with input)
        unary_node(12, "tanh", TraceOp::Tanh, 11, shape),
        constant_node(13, 0.1, shape),
        binary_node(14, "mul_c3", TraceOp::Mul, 12, 13, shape),
    ]);

    let fused_steps = compile_trace_with_fusion(&graph).unwrap();
    let fused_dispatches = count_dispatches(&fused_steps);

    let unfused_steps = crate::trace_compile::compile_trace(&graph).unwrap();
    let unfused_dispatches = count_dispatches(&unfused_steps);

    // Chain 1: 3 ops → 1 dispatch (saved 2)
    // Chain 2: 3 ops → 1 dispatch (saved 2)
    // Chain 3: 2 ops → 1 dispatch (saved 1)
    // Total saved: 5 dispatches (8 → 3)
    assert_eq!(
        fused_dispatches, 3,
        "3 chains should produce 3 fused dispatches"
    );
    assert_eq!(
        unfused_dispatches, 8,
        "unfused: 8 individual elementwise dispatches"
    );
    assert_eq!(
        unfused_dispatches - fused_dispatches,
        5,
        "fusion should save 5 dispatches across 3 Kokoro-like chains"
    );
}

// -- Performance proofs: detect_fusible_chains scaling (#3243) -----------------
//
// The nested loop structure of detect_fusible_chains() is theoretically O(n²)
// for all-fusible-same-shape graphs. These tests prove that:
// 1. Realistic heterogeneous graphs scale linearly.
// 2. Even the worst-case (all fusible, same shape) exhibits sub-quadratic
//    behavior because the `in_chain` HashSet and `break` on fan-out > 1
//    effectively linearize the inner loop.

/// Build a heterogeneous graph with `n` nodes: alternating fusible and
/// non-fusible ops. This is the realistic case for production models.
fn build_heterogeneous_graph(n: usize) -> ComputationGraph {
    let mut nodes = Vec::with_capacity(n);
    let shape = &[1, 64];

    // First node is always an input.
    nodes.push(input_node(0, shape));

    for i in 1..n {
        let prev_id = (i - 1) as u64;
        let id = i as u64;
        if i % 3 == 0 {
            // Non-fusible op every 3rd node (reshape or reduce).
            nodes.push(TraceNode::new(
                id,
                format!("reshape_{i}"),
                TraceOp::Reshape {
                    target_shape: shape.to_vec(),
                },
                vec![prev_id],
                shape.to_vec(),
                DType::F32,
            ));
        } else if i % 3 == 1 {
            nodes.push(unary_node(
                id,
                &format!("exp_{i}"),
                TraceOp::Exp,
                prev_id,
                shape,
            ));
        } else {
            nodes.push(unary_node(
                id,
                &format!("relu_{i}"),
                TraceOp::Relu,
                prev_id,
                shape,
            ));
        }
    }
    ComputationGraph::from_nodes(nodes)
}

/// Build a worst-case graph: all nodes are fusible elementwise with same
/// shape, forming one long chain. This maximizes inner loop iterations
/// before the `in_chain` set catches them.
fn build_all_fusible_graph(n: usize) -> ComputationGraph {
    let mut nodes = Vec::with_capacity(n);
    let shape = &[1, 64];

    nodes.push(input_node(0, shape));
    for i in 1..n {
        let prev_id = (i - 1) as u64;
        let id = i as u64;
        // Alternate Exp and Relu to keep it realistic but all fusible.
        let op = if i % 2 == 0 {
            TraceOp::Exp
        } else {
            TraceOp::Relu
        };
        nodes.push(unary_node(id, &format!("op_{i}"), op, prev_id, shape));
    }
    ComputationGraph::from_nodes(nodes)
}

#[test]
fn proof_fusion_chain_detection_quadratic_heterogeneous() {
    // FINDING: Heterogeneous graphs exhibit O(n²) scaling in detect_fusible_chains.
    //
    // Root cause: When a short chain (e.g., exp→relu) ends because the next
    // consumer is non-fusible (reshape), the inner loop does NOT break — the
    // chain tail (relu) has use_count == 1, so the `break` guard doesn't fire.
    // The inner loop scans all remaining nodes looking for another fusible
    // consumer of the chain tail, which doesn't exist.
    //
    // For ~n/3 chain starts each scanning ~n remaining nodes: O(n²/3).
    //
    // Fix: break when no fusible node consumes `cur` within the forward scan.
    // Currently the inner loop only breaks on use_count != 1, not on
    // "consumer is non-fusible". Adding a check for the consumer's fusibility
    // (or a `found_extension` flag) would linearize this case.
    //
    // This is compile-time-only. For production model sizes (100-500 nodes),
    // execution is < 50ms — acceptable for one-time model compilation.
    let sizes = [200, 1000, 5000];
    let mut timings = Vec::new();

    for &n in &sizes {
        let graph = build_heterogeneous_graph(n);
        let nodes = graph.nodes();
        let use_counts = super::build_use_counts(&graph);

        let start = std::time::Instant::now();
        let chains = super::detect_fusible_chains(nodes, &use_counts);
        let elapsed = start.elapsed();
        timings.push(elapsed.as_nanos() as f64);

        // Structural: chains should exist and each should be length 2.
        assert!(!chains.is_empty(), "n={n}: should detect fusion chains");
        for chain in &chains {
            assert_eq!(chain.len(), 2, "n={n}: heterogeneous chains are length 2");
        }
    }

    // Document the O(n²) scaling: 25x size increase gives ~625x time increase.
    // Assert it stays below 1500x (generous bound for CI variance on small n).
    let ratio = timings[2] / timings[0].max(1.0);
    assert!(
        ratio < 1500.0,
        "heterogeneous graph scaling ratio {ratio:.1}x exceeds 1500x \
         (expected ~625x for O(n²) on 25x size). \
         Timings: {:.0}ns, {:.0}ns, {:.0}ns",
        timings[0],
        timings[1],
        timings[2],
    );

    // Production safety: 500 nodes (typical model) must complete under 50ms.
    let graph_500 = build_heterogeneous_graph(500);
    let nodes_500 = graph_500.nodes();
    let uc_500 = super::build_use_counts(&graph_500);
    let start = std::time::Instant::now();
    let _ = super::detect_fusible_chains(nodes_500, &uc_500);
    let elapsed_ms = start.elapsed().as_millis();
    assert!(
        elapsed_ms < 50,
        "fusion chain detection for 500-node graph took {elapsed_ms}ms (> 50ms limit)"
    );
}

#[test]
fn proof_fusion_chain_detection_subquadratic_all_fusible() {
    // Worst case: all fusible same-shape. The entire graph is one chain.
    // After the first node starts a chain and consumes all subsequent nodes
    // via the inner loop, `in_chain` prevents re-scanning them.
    // Effective: O(n) because each node is visited once by the inner loop
    // (added to in_chain) and skipped by the outer loop.
    let sizes = [200, 1000, 5000];
    let mut timings = Vec::new();

    for &n in &sizes {
        let graph = build_all_fusible_graph(n);
        let nodes = graph.nodes();
        let use_counts = super::build_use_counts(&graph);

        let start = std::time::Instant::now();
        let chains = super::detect_fusible_chains(nodes, &use_counts);
        let elapsed = start.elapsed();
        timings.push(elapsed.as_nanos() as f64);

        // Structural: exactly 1 chain containing all fusible nodes.
        assert_eq!(
            chains.len(),
            1,
            "n={n}: all-fusible graph should produce 1 chain"
        );
        assert_eq!(
            chains[0].len(),
            n - 1,
            "n={n}: chain should contain all fusible nodes (n-1, excluding input)"
        );
    }

    // Sub-quadratic: 25x size increase should give < 100x time increase.
    // Quadratic = 625x, linear = 25x. 100x allows for cache/allocation overhead.
    let ratio = timings[2] / timings[0].max(1.0);
    assert!(
        ratio < 100.0,
        "all-fusible graph scaling ratio {ratio:.1}x exceeds 100x for 25x size increase \
         (timings: {:.0}ns, {:.0}ns, {:.0}ns). \
         If this fails, the in_chain HashSet optimization may be broken.",
        timings[0],
        timings[1],
        timings[2],
    );
}

#[test]
fn proof_fusion_chain_count_bounded_by_graph_size() {
    // Invariant: total fused ops across all chains ≤ n (each node appears
    // in at most one chain). This proves no double-counting or unbounded
    // chain growth.
    for n in [50, 200, 1000] {
        let graph = build_heterogeneous_graph(n);
        let nodes = graph.nodes();
        let use_counts = super::build_use_counts(&graph);
        let chains = super::detect_fusible_chains(nodes, &use_counts);

        let total_fused: usize = chains.iter().map(Vec::len).sum();
        assert!(
            total_fused <= n,
            "n={n}: total fused ops ({total_fused}) exceeds graph size"
        );

        // No duplicates across chains.
        let mut seen = std::collections::HashSet::new();
        for chain in &chains {
            for &idx in chain {
                assert!(
                    seen.insert(idx),
                    "n={n}: node index {idx} appears in multiple chains"
                );
            }
        }
    }
}

#[test]
fn proof_use_count_map_linear_in_graph_size() {
    // build_use_counts is O(n * avg_fan_in). For elementwise ops, fan_in
    // is 1-2, so this is O(n). Verify the map size is bounded.
    for n in [100, 500, 2000] {
        let graph = build_heterogeneous_graph(n);
        let use_counts = super::build_use_counts(&graph);
        // At most n entries (one per node that is referenced as input).
        assert!(
            use_counts.len() <= n,
            "n={n}: use_count map size ({}) exceeds graph size",
            use_counts.len()
        );
    }
}

// -- Tests: add + scalar mul filter for resblock peephole (#F0) ---------------

#[test]
fn test_add_scalar_mul_not_fused() {
    // Reproduces the F0 AdainResBlk1d pattern: add(h, shortcut) + mul_scalar(1/√2).
    // The [Add, Mul] chain where Mul's other input is a scalar ConstantWeight
    // should NOT be fused, so the resblock peephole pass can see the `add`.
    use nn_core::dyn_tensor::trace::WeightRef;
    let inv_sqrt2 = 1.0_f32 / std::f32::consts::SQRT_2;
    let scalar_cw = TraceNode::new(
        2,
        "inv_sqrt2".into(),
        TraceOp::ConstantWeight {
            weight: WeightRef::new(vec![inv_sqrt2], vec![]).unwrap(),
        },
        vec![],
        vec![],
        DType::F32,
    );
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        input_node(1, &[1, 4, 8]),
        binary_node(10, "add_res", TraceOp::Add, 0, 1, &[1, 4, 8]),
        scalar_cw,
        binary_node(11, "mul_scale", TraceOp::Mul, 10, 2, &[1, 4, 8]),
    ]);

    let nodes = graph.nodes();
    let use_counts = super::build_use_counts(&graph);
    let chains = super::detect_fusible_chains(nodes, &use_counts);

    // The [Add, Mul(scalar)] chain should NOT be detected.
    assert!(
        chains.is_empty(),
        "add + scalar_mul chain should NOT be fused (needed by resblock peephole), \
         but got {} chain(s): {:?}",
        chains.len(),
        chains
    );

    // Also verify compilation: add and mul should be separate Dispatch steps.
    let steps = compile_trace_with_fusion(&graph).unwrap();
    let dispatch_count = count_dispatches(&steps);
    assert_eq!(
        dispatch_count, 2,
        "add and mul should be 2 separate dispatches, got {dispatch_count}"
    );
}

#[test]
fn test_add_nonscalar_mul_still_fuses() {
    // When the Mul's other input is a non-scalar ConstantWeight (e.g., [4]),
    // the chain should still be fused normally.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(10, "add_res", TraceOp::Add, 0, 1, &[4]),
        constant_weight_node(2, &[4]), // 4-element weight, NOT scalar
        binary_node(11, "mul_scale", TraceOp::Mul, 10, 2, &[4]),
    ]);

    let nodes = graph.nodes();
    let use_counts = super::build_use_counts(&graph);
    let chains = super::detect_fusible_chains(nodes, &use_counts);

    // Non-scalar weight: should still fuse [Add, Mul].
    assert_eq!(
        chains.len(),
        1,
        "add + non-scalar_mul should be fused, got {} chains",
        chains.len()
    );
}

#[test]
fn test_add_constant_mul_not_fused() {
    // Reproduces the ACTUAL F0 pipeline pattern: mul_scalar creates
    // TraceOp::Constant (via DynTensor::full(&[], val)), not ConstantWeight.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        input_node(1, &[1, 4, 8]),
        binary_node(10, "add_res", TraceOp::Add, 0, 1, &[1, 4, 8]),
        TraceNode::new(
            2,
            "inv_sqrt2".into(),
            TraceOp::Constant {
                value: (1.0_f64 / std::f64::consts::SQRT_2),
            },
            vec![],
            vec![],
            DType::F32,
        ),
        binary_node(11, "mul_scale", TraceOp::Mul, 10, 2, &[1, 4, 8]),
    ]);

    let nodes = graph.nodes();
    let use_counts = super::build_use_counts(&graph);
    let chains = super::detect_fusible_chains(nodes, &use_counts);

    // TraceOp::Constant scalar: should NOT be fused, same as ConstantWeight.
    assert!(
        chains.is_empty(),
        "add + Constant(scalar) mul should NOT be fused, got {chains:?}"
    );
}

// -- Tests: chain truncation for production rb_per_stage=3 (#3513) ------------

#[test]
fn test_long_chain_ending_add_scalar_mul_truncated() {
    // Production Kokoro rb_per_stage=3 averaging pattern:
    //   [add(internal), add(avg1), add(avg2), mul(1/3)]
    // The trailing [Add, Mul(scalar)] must be truncated so FusedResBlock's
    // detect_post_add_scale can find the pattern. The leading adds should
    // still be fused together.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]), // rb0 output
        input_node(1, &[1, 4, 8]), // rb1 output
        input_node(3, &[1, 4, 8]), // rb2 output
        // add(rb0, rb1) — internal residual add
        binary_node(10, "add_rb01", TraceOp::Add, 0, 1, &[1, 4, 8]),
        // add(sum01, rb2) — averaging add
        binary_node(11, "add_avg", TraceOp::Add, 10, 3, &[1, 4, 8]),
        // mul(sum012, 1/3) — averaging scale
        TraceNode::new(
            4,
            "one_third".into(),
            TraceOp::Constant { value: 1.0 / 3.0 },
            vec![],
            vec![],
            DType::F32,
        ),
        binary_node(12, "mul_avg", TraceOp::Mul, 11, 4, &[1, 4, 8]),
    ]);

    let nodes = graph.nodes();
    let use_counts = super::build_use_counts(&graph);
    let chains = super::detect_fusible_chains(nodes, &use_counts);

    // The chain [Add, Add, Mul(scalar)] should be truncated to [Add] (just
    // the first add). Since length 1 < 2, no chain is emitted. The trailing
    // Add+Mul(scalar) stays separate for FusedResBlock peephole.
    assert!(
        chains.is_empty(),
        "chain ending with [Add, Mul(scalar)] should be truncated below min length, \
         got {} chain(s): {:?}",
        chains.len(),
        chains
    );

    // Compilation: all 3 ops should be separate dispatches.
    let steps = compile_trace_with_fusion(&graph).unwrap();
    let dispatch_count = count_dispatches(&steps);
    assert_eq!(
        dispatch_count, 3,
        "add_rb01 + add_avg + mul_avg should be 3 separate dispatches, got {dispatch_count}"
    );
}

#[test]
fn test_four_add_chain_ending_scalar_mul_truncated_to_fusible_pair() {
    // Longer chain: [Add, Add, Add, Add, Mul(scalar)]
    // Truncation removes trailing [Add, Mul(scalar)] → [Add, Add, Add] fused.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        input_node(1, &[1, 4, 8]),
        input_node(3, &[1, 4, 8]),
        input_node(5, &[1, 4, 8]),
        input_node(7, &[1, 4, 8]),
        binary_node(10, "add_0", TraceOp::Add, 0, 1, &[1, 4, 8]),
        binary_node(11, "add_1", TraceOp::Add, 10, 3, &[1, 4, 8]),
        binary_node(12, "add_2", TraceOp::Add, 11, 5, &[1, 4, 8]),
        binary_node(13, "add_3", TraceOp::Add, 12, 7, &[1, 4, 8]),
        TraceNode::new(
            8,
            "scalar".into(),
            TraceOp::Constant { value: 0.25 },
            vec![],
            vec![],
            DType::F32,
        ),
        binary_node(14, "mul_scale", TraceOp::Mul, 13, 8, &[1, 4, 8]),
    ]);

    let nodes = graph.nodes();
    let use_counts = super::build_use_counts(&graph);
    let chains = super::detect_fusible_chains(nodes, &use_counts);

    // Truncation removes trailing [Add, Mul(scalar)] → [Add, Add, Add] remains.
    assert_eq!(
        chains.len(),
        1,
        "should have 1 fused chain of 3 adds, got {} chains: {:?}",
        chains.len(),
        chains
    );
    assert_eq!(
        chains[0].len(),
        3,
        "fused chain should be 3 ops (the first 3 adds), got {}",
        chains[0].len()
    );
}

#[test]
fn test_exp_add_scalar_mul_truncated_to_nothing() {
    // Chain: [Exp, Add, Mul(scalar)]
    // Truncation removes trailing [Add, Mul(scalar)] → [Exp] alone, length < 2.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        input_node(1, &[1, 4, 8]),
        unary_node(10, "exp", TraceOp::Exp, 0, &[1, 4, 8]),
        binary_node(11, "add_res", TraceOp::Add, 10, 1, &[1, 4, 8]),
        TraceNode::new(
            2,
            "scalar".into(),
            TraceOp::Constant { value: 0.5 },
            vec![],
            vec![],
            DType::F32,
        ),
        binary_node(12, "mul_scale", TraceOp::Mul, 11, 2, &[1, 4, 8]),
    ]);

    let nodes = graph.nodes();
    let use_counts = super::build_use_counts(&graph);
    let chains = super::detect_fusible_chains(nodes, &use_counts);

    // [Exp] is too short to fuse.
    assert!(
        chains.is_empty(),
        "[Exp, Add, Mul(scalar)] should truncate to [Exp] which is < 2, got {chains:?}"
    );
}

#[test]
fn test_chain_ending_add_nonscalar_mul_not_truncated() {
    // Chain: [Add, Add, Mul(non-scalar)] should NOT be truncated.
    // Only scalar Mul triggers truncation.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        input_node(1, &[1, 4, 8]),
        input_node(3, &[1, 4, 8]),
        binary_node(10, "add_0", TraceOp::Add, 0, 1, &[1, 4, 8]),
        binary_node(11, "add_1", TraceOp::Add, 10, 3, &[1, 4, 8]),
        // Non-scalar weight (shape [1,4,8]), not a scalar constant
        constant_weight_node(5, &[1, 4, 8]),
        binary_node(12, "mul_w", TraceOp::Mul, 11, 5, &[1, 4, 8]),
    ]);

    let nodes = graph.nodes();
    let use_counts = super::build_use_counts(&graph);
    let chains = super::detect_fusible_chains(nodes, &use_counts);

    // Full chain [Add, Add, Mul] should be fused — no truncation.
    assert_eq!(
        chains.len(),
        1,
        "chain with non-scalar Mul should fuse normally, got {} chains",
        chains.len()
    );
    assert_eq!(
        chains[0].len(),
        3,
        "full chain [Add, Add, Mul(non-scalar)] should be length 3, got {}",
        chains[0].len()
    );
}

#[test]
fn test_chain_ending_mul_without_preceding_add_not_truncated() {
    // Chain: [Exp, Mul(scalar)] — no Add before Mul, no truncation.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 8]),
        unary_node(10, "exp", TraceOp::Exp, 0, &[1, 4, 8]),
        TraceNode::new(
            2,
            "scalar".into(),
            TraceOp::Constant { value: 2.0 },
            vec![],
            vec![],
            DType::F32,
        ),
        binary_node(11, "mul_scale", TraceOp::Mul, 10, 2, &[1, 4, 8]),
    ]);

    let nodes = graph.nodes();
    let use_counts = super::build_use_counts(&graph);
    let chains = super::detect_fusible_chains(nodes, &use_counts);

    // [Exp, Mul(scalar)] should fuse — no Add before Mul, so no truncation.
    assert_eq!(
        chains.len(),
        1,
        "[Exp, Mul(scalar)] should fuse (no preceding Add), got {} chains",
        chains.len()
    );
    assert_eq!(chains[0].len(), 2);
}
