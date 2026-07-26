// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op coverage tests for `nn-import`.
//!
//! Verifies that each supported aten op mapping produces a valid `ImportedGraph`
//! via synthetic graph JSON + in-memory weights. No PyTorch, no Metal, no external
//! model files required.
//!
//! Organized by op category:
//!   - Unary element-wise (relu, gelu, sigmoid, tanh, neg, abs, exp, log, sqrt, sin, cos, silu)
//!   - Binary element-wise (add, sub, mul, div, maximum, minimum)
//!   - Reduction (sum, mean, amax, amin)
//!   - Shape (reshape, permute, transpose, unsqueeze, squeeze, cat, slice, expand, flip)
//!   - Linear (matmul + bias via linear.default)
//!   - Matrix multiply (mm, bmm, matmul)
//!   - Softmax / log_softmax
//!   - Identity / clone / contiguous

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;
use nn_import::{build_graph, ResolvedWeight};

// ---------------------------------------------------------------------------
// Helpers: build minimal graph JSON and parse it
// ---------------------------------------------------------------------------

/// Build a minimal ExportedProgram JSON string for a single-op graph:
///   input x:[1,4] -> op -> output:[out_shape]
///
/// `nodes_json` is the raw JSON array for the "nodes" field.
/// `extra_inputs` adds additional graph-level input tensors (e.g., for binary ops).
/// `extra_tensor_values` adds additional tensor_values entries.
fn single_op_graph_json(
    nodes_json: &str,
    output_name: &str,
    output_shape: &[usize],
    extra_inputs: &str,
    extra_tensor_values: &str,
) -> String {
    let out_sizes: String = output_shape
        .iter()
        .map(|s| format!("{{\"as_int\": {s}}}"))
        .collect::<Vec<_>>()
        .join(", ");
    let out_strides = compute_strides(output_shape);
    let out_strides_json: String = out_strides
        .iter()
        .map(|s| format!("{{\"as_int\": {s}}}"))
        .collect::<Vec<_>>()
        .join(", ");

    let extra_tv = if extra_tensor_values.is_empty() {
        String::new()
    } else {
        format!(", {extra_tensor_values}")
    };
    let extra_in = if extra_inputs.is_empty() {
        String::new()
    } else {
        format!(", {extra_inputs}")
    };

    format!(
        r#"{{
    "graph_module": {{
        "graph": {{
            "inputs": [
                {{"as_tensor": {{"name": "x"}}}}{extra_in}
            ],
            "outputs": [{{"as_tensor": {{"name": "{output_name}"}}}}],
            "nodes": [{nodes_json}],
            "tensor_values": {{
                "x": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 4}}], "requires_grad": false, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "{output_name}": {{"dtype": 7, "sizes": [{out_sizes}], "requires_grad": false, "strides": [{out_strides_json}]}}{extra_tv}
            }},
            "is_single_tensor_return": true
        }},
        "signature": {{
            "input_specs": [
                {{"user_input": {{"arg": {{"as_tensor": {{"name": "x"}}}}}}}}
            ],
            "output_specs": [
                {{"user_output": {{"arg": {{"as_tensor": {{"name": "{output_name}"}}}}}}}}
            ]
        }},
        "module_call_graph": []
    }},
    "schema_version": {{"major": 8, "minor": 15}},
    "opset_version": {{"aten": 10}},
    "range_constraints": {{}}
}}"#
    )
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Build a graph JSON for a binary op with two user inputs: x:[1,4] and y:[1,4].
fn binary_op_graph_json(target: &str, output_name: &str, rhs_arg_name: &str) -> String {
    let node = format!(
        r#"{{
            "target": "{target}",
            "inputs": [
                {{"name": "self", "arg": {{"as_tensor": {{"name": "x"}}}}, "kind": 1}},
                {{"name": "{rhs_arg_name}", "arg": {{"as_tensor": {{"name": "y"}}}}, "kind": 1}}
            ],
            "outputs": [{{"as_tensor": {{"name": "{output_name}"}}}}],
            "metadata": {{}}
        }}"#
    );
    two_input_graph_json(&node, output_name, &[1, 4], &[1, 4], &[1, 4])
}

/// Build a graph JSON with two user inputs x and y.
/// Both are registered in the signature as user_inputs.
fn two_input_graph_json(
    nodes_json: &str,
    output_name: &str,
    x_shape: &[usize],
    y_shape: &[usize],
    output_shape: &[usize],
) -> String {
    let x_sizes = shape_to_json(x_shape);
    let x_strides = strides_to_json(x_shape);
    let y_sizes = shape_to_json(y_shape);
    let y_strides = strides_to_json(y_shape);
    let out_sizes = shape_to_json(output_shape);
    let out_strides = strides_to_json(output_shape);

    format!(
        r#"{{
    "graph_module": {{
        "graph": {{
            "inputs": [
                {{"as_tensor": {{"name": "x"}}}},
                {{"as_tensor": {{"name": "y"}}}}
            ],
            "outputs": [{{"as_tensor": {{"name": "{output_name}"}}}}],
            "nodes": [{nodes_json}],
            "tensor_values": {{
                "x": {{"dtype": 7, "sizes": [{x_sizes}], "requires_grad": false, "strides": [{x_strides}]}},
                "y": {{"dtype": 7, "sizes": [{y_sizes}], "requires_grad": false, "strides": [{y_strides}]}},
                "{output_name}": {{"dtype": 7, "sizes": [{out_sizes}], "requires_grad": false, "strides": [{out_strides}]}}
            }},
            "is_single_tensor_return": true
        }},
        "signature": {{
            "input_specs": [
                {{"user_input": {{"arg": {{"as_tensor": {{"name": "x"}}}}}}}},
                {{"user_input": {{"arg": {{"as_tensor": {{"name": "y"}}}}}}}}
            ],
            "output_specs": [
                {{"user_output": {{"arg": {{"as_tensor": {{"name": "{output_name}"}}}}}}}}
            ]
        }},
        "module_call_graph": []
    }},
    "schema_version": {{"major": 8, "minor": 15}},
    "opset_version": {{"aten": 10}},
    "range_constraints": {{}}
}}"#
    )
}

fn shape_to_json(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|s| format!("{{\"as_int\": {s}}}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn strides_to_json(shape: &[usize]) -> String {
    let strides = compute_strides(shape);
    strides
        .iter()
        .map(|s| format!("{{\"as_int\": {s}}}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Build a graph JSON for a unary op: x:[1,4] -> op -> output:[1,4].
fn unary_op_graph_json(target: &str, output_name: &str) -> String {
    let node = format!(
        r#"{{
            "target": "{target}",
            "inputs": [
                {{"name": "input", "arg": {{"as_tensor": {{"name": "x"}}}}, "kind": 1}}
            ],
            "outputs": [{{"as_tensor": {{"name": "{output_name}"}}}}],
            "metadata": {{}}
        }}"#
    );
    single_op_graph_json(&node, output_name, &[1, 4], "", "")
}

/// Parse graph JSON and build an ImportedGraph with empty weights.
fn import_from_json(json: &str) -> nn_import::ImportedGraph {
    let program =
        nn_import::parse_exported_program(json.as_bytes()).expect("graph JSON should parse");
    let weights: HashMap<String, ResolvedWeight> = HashMap::new();
    build_graph(&program, &weights).expect("build_graph should succeed")
}

/// Parse graph JSON and build an ImportedGraph with provided weights.
fn import_from_json_with_weights(
    json: &str,
    weights: HashMap<String, ResolvedWeight>,
) -> nn_import::ImportedGraph {
    let program =
        nn_import::parse_exported_program(json.as_bytes()).expect("graph JSON should parse");
    build_graph(&program, &weights).expect("build_graph should succeed")
}

/// Assert that the output node of the graph matches the expected TraceOp variant.
fn assert_output_op(
    graph: &nn_import::ImportedGraph,
    check: impl FnOnce(&TraceOp) -> bool,
    msg: &str,
) {
    let output = graph.graph.output_node().expect("graph should have output");
    assert!(check(output.op()), "{msg}: got {:?}", output.op());
}

// ===========================================================================
// Unary element-wise ops
// ===========================================================================

#[test]
fn test_op_coverage_relu() {
    let json = unary_op_graph_json("torch.ops.aten.relu.default", "relu_out");
    let imported = import_from_json(&json);
    assert_eq!(imported.num_user_inputs, 1);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Relu), "relu");
}

#[test]
fn test_op_coverage_gelu_tanh() {
    let node = r#"{
        "target": "torch.ops.aten.gelu.default",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "approximate", "arg": {"as_string": "tanh"}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "gelu_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "gelu_out", &[1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Gelu), "gelu tanh");
}

#[test]
fn test_op_coverage_gelu_erf() {
    let node = r#"{
        "target": "torch.ops.aten.gelu.default",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "approximate", "arg": {"as_string": "none"}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "gelu_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "gelu_out", &[1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::GeluErf), "gelu erf");
}

#[test]
fn test_op_coverage_sigmoid() {
    let json = unary_op_graph_json("torch.ops.aten.sigmoid.default", "sigmoid_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Sigmoid), "sigmoid");
}

#[test]
fn test_op_coverage_tanh() {
    let json = unary_op_graph_json("torch.ops.aten.tanh.default", "tanh_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Tanh), "tanh");
}

#[test]
fn test_op_coverage_neg() {
    let json = unary_op_graph_json("torch.ops.aten.neg.default", "neg_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Neg), "neg");
}

#[test]
fn test_op_coverage_abs() {
    let json = unary_op_graph_json("torch.ops.aten.abs.default", "abs_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Abs), "abs");
}

#[test]
fn test_op_coverage_exp() {
    let json = unary_op_graph_json("torch.ops.aten.exp.default", "exp_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Exp), "exp");
}

#[test]
fn test_op_coverage_log() {
    let json = unary_op_graph_json("torch.ops.aten.log.default", "log_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Log), "log");
}

#[test]
fn test_op_coverage_sqrt() {
    let json = unary_op_graph_json("torch.ops.aten.sqrt.default", "sqrt_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Sqrt), "sqrt");
}

#[test]
fn test_op_coverage_sin() {
    let json = unary_op_graph_json("torch.ops.aten.sin.default", "sin_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Sin), "sin");
}

#[test]
fn test_op_coverage_cos() {
    let json = unary_op_graph_json("torch.ops.aten.cos.default", "cos_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Cos), "cos");
}

#[test]
fn test_op_coverage_silu() {
    let json = unary_op_graph_json("torch.ops.aten.silu.default", "silu_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Silu), "silu");
}

#[test]
fn test_op_coverage_reciprocal() {
    let json = unary_op_graph_json("torch.ops.aten.reciprocal.default", "recip_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Recip), "reciprocal");
}

#[test]
fn test_op_coverage_floor() {
    let json = unary_op_graph_json("torch.ops.aten.floor.default", "floor_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Floor), "floor");
}

#[test]
fn test_op_coverage_round() {
    let json = unary_op_graph_json("torch.ops.aten.round.default", "round_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Round), "round");
}

#[test]
fn test_op_coverage_dropout() {
    let json = unary_op_graph_json("torch.ops.aten.dropout.default", "dropout_out");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Dropout), "dropout");
}

// ===========================================================================
// Binary element-wise ops
// ===========================================================================

#[test]
fn test_op_coverage_add() {
    let json = binary_op_graph_json("torch.ops.aten.add.Tensor", "add_out", "other");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Add), "add");
}

#[test]
fn test_op_coverage_sub() {
    let json = binary_op_graph_json("torch.ops.aten.sub.Tensor", "sub_out", "other");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Sub), "sub");
}

#[test]
fn test_op_coverage_mul() {
    let json = binary_op_graph_json("torch.ops.aten.mul.Tensor", "mul_out", "other");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Mul), "mul");
}

#[test]
fn test_op_coverage_div() {
    let json = binary_op_graph_json("torch.ops.aten.div.Tensor", "div_out", "other");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Div), "div");
}

#[test]
fn test_op_coverage_maximum() {
    let json = binary_op_graph_json("torch.ops.aten.maximum.default", "max_out", "other");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Maximum), "maximum");
}

#[test]
fn test_op_coverage_minimum() {
    let json = binary_op_graph_json("torch.ops.aten.minimum.default", "min_out", "other");
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::Minimum), "minimum");
}

// ===========================================================================
// Matrix multiply ops
// ===========================================================================

#[test]
fn test_op_coverage_mm() {
    // mm: [1,4] x [4,3] -> [1,3]
    let node = r#"{
        "target": "torch.ops.aten.mm.default",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "mat2", "arg": {"as_tensor": {"name": "y"}}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "mm_out"}}],
        "metadata": {}
    }"#;
    let json = two_input_graph_json(node, "mm_out", &[1, 4], &[4, 3], &[1, 3]);
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::MatMul), "mm");
}

#[test]
fn test_op_coverage_bmm() {
    // bmm: [2,4,3] x [2,3,5] -> [2,4,5]
    let node = r#"{
        "target": "torch.ops.aten.bmm.default",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "mat2", "arg": {"as_tensor": {"name": "y"}}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "bmm_out"}}],
        "metadata": {}
    }"#;
    let json = two_input_graph_json(node, "bmm_out", &[2, 4, 3], &[2, 3, 5], &[2, 4, 5]);
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::MatMul), "bmm");
}

#[test]
fn test_op_coverage_matmul() {
    // matmul uses "other" for rhs, same as binary ops.
    let node = r#"{
        "target": "torch.ops.aten.matmul.default",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "other", "arg": {"as_tensor": {"name": "y"}}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "matmul_out"}}],
        "metadata": {}
    }"#;
    let json = two_input_graph_json(node, "matmul_out", &[1, 4], &[4, 3], &[1, 3]);
    let imported = import_from_json(&json);
    assert_output_op(&imported, |op| matches!(op, TraceOp::MatMul), "matmul");
}

// ===========================================================================
// Reduction ops
// ===========================================================================

#[test]
fn test_op_coverage_reduce_sum() {
    let node = r#"{
        "target": "torch.ops.aten.sum.dim_IntList",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_ints": [1]}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "sum_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "sum_out", &[1, 1], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| {
            matches!(
                op,
                TraceOp::ReduceSum {
                    dim: 1,
                    keepdim: false
                }
            )
        },
        "reduce_sum",
    );
}

#[test]
fn test_op_coverage_reduce_sum_keepdim() {
    let node = r#"{
        "target": "torch.ops.aten.sum.dim_IntList",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_ints": [1]}, "kind": 1},
            {"name": "keepdim", "arg": {"as_bool": true}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "sum_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "sum_out", &[1, 1], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| {
            matches!(
                op,
                TraceOp::ReduceSum {
                    dim: 1,
                    keepdim: true
                }
            )
        },
        "reduce_sum keepdim",
    );
}

#[test]
fn test_op_coverage_reduce_mean() {
    let node = r#"{
        "target": "torch.ops.aten.mean.dim",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_ints": [1]}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "mean_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "mean_out", &[1, 1], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| {
            matches!(
                op,
                TraceOp::ReduceMean {
                    dim: 1,
                    keepdim: false
                }
            )
        },
        "reduce_mean",
    );
}

#[test]
fn test_op_coverage_reduce_max() {
    let node = r#"{
        "target": "torch.ops.aten.amax.default",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_ints": [0]}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "max_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "max_out", &[1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| {
            matches!(
                op,
                TraceOp::ReduceMax {
                    dim: 0,
                    keepdim: false
                }
            )
        },
        "reduce_max",
    );
}

#[test]
fn test_op_coverage_reduce_min() {
    let node = r#"{
        "target": "torch.ops.aten.amin.default",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_ints": [0]}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "min_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "min_out", &[1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| {
            matches!(
                op,
                TraceOp::ReduceMin {
                    dim: 0,
                    keepdim: false
                }
            )
        },
        "reduce_min",
    );
}

// ===========================================================================
// Shape ops
// ===========================================================================

#[test]
fn test_op_coverage_reshape_view() {
    let node = r#"{
        "target": "torch.ops.aten.view.default",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "size", "arg": {"as_ints": [2, 2]}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "view_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "view_out", &[2, 2], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Reshape { target_shape } if *target_shape == vec![2, 2]),
        "reshape/view",
    );
}

#[test]
fn test_op_coverage_reshape_default() {
    let node = r#"{
        "target": "torch.ops.aten.reshape.default",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "size", "arg": {"as_ints": [4, 1]}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "reshape_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "reshape_out", &[4, 1], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Reshape { target_shape } if *target_shape == vec![4, 1]),
        "reshape.default",
    );
}

#[test]
fn test_op_coverage_unsafe_view() {
    let node = r#"{
        "target": "torch.ops.aten._unsafe_view.default",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "size", "arg": {"as_ints": [1, 4]}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "uv_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "uv_out", &[1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Reshape { .. }),
        "unsafe_view",
    );
}

#[test]
fn test_op_coverage_permute() {
    let node = r#"{
        "target": "torch.ops.aten.permute.default",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dims", "arg": {"as_ints": [1, 0]}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "permute_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "permute_out", &[4, 1], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Permute { axes } if *axes == vec![1, 0]),
        "permute",
    );
}

#[test]
fn test_op_coverage_transpose() {
    let node = r#"{
        "target": "torch.ops.aten.transpose.int",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim0", "arg": {"as_int": 0}, "kind": 1},
            {"name": "dim1", "arg": {"as_int": 1}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "transpose_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "transpose_out", &[4, 1], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Transpose { dim0: 0, dim1: 1 }),
        "transpose",
    );
}

#[test]
fn test_op_coverage_unsqueeze() {
    let node = r#"{
        "target": "torch.ops.aten.unsqueeze.default",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_int": 0}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "unsqueeze_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "unsqueeze_out", &[1, 1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Unsqueeze { dim: 0 }),
        "unsqueeze",
    );
}

#[test]
fn test_op_coverage_squeeze_dim() {
    // Input needs a size-1 dim. Use [1,1,4] -> squeeze dim=0 -> [1,4].
    let node = r#"{
        "target": "torch.ops.aten.squeeze.dim",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_int": 0}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "squeeze_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "squeeze_out", &[4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Squeeze { dim: 0 }),
        "squeeze.dim",
    );
}

#[test]
fn test_op_coverage_cat() {
    // cat two inputs along dim=1: x:[1,4] + y:[1,4] -> [1,8]
    let node = r#"{
        "target": "torch.ops.aten.cat.default",
        "inputs": [
            {"name": "tensors", "arg": {"as_tensors": [{"name": "x"}, {"name": "y"}]}, "kind": 1},
            {"name": "dim", "arg": {"as_int": 1}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "cat_out"}}],
        "metadata": {}
    }"#;
    let json = two_input_graph_json(node, "cat_out", &[1, 4], &[1, 4], &[1, 8]);
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| {
            matches!(
                op,
                TraceOp::Cat {
                    dim: 1,
                    num_inputs: 2
                }
            )
        },
        "cat",
    );
}

#[test]
fn test_op_coverage_slice() {
    let node = r#"{
        "target": "torch.ops.aten.slice.Tensor",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_int": 1}, "kind": 1},
            {"name": "start", "arg": {"as_int": 0}, "kind": 1},
            {"name": "end", "arg": {"as_int": 2}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "slice_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "slice_out", &[1, 2], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| {
            matches!(
                op,
                TraceOp::Narrow {
                    dim: 1,
                    start: 0,
                    length: 2
                }
            )
        },
        "slice",
    );
}

#[test]
fn test_op_coverage_expand() {
    let node = r#"{
        "target": "torch.ops.aten.expand.default",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "size", "arg": {"as_ints": [3, 4]}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "expand_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "expand_out", &[3, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Expand { target_shape } if *target_shape == vec![3, 4]),
        "expand",
    );
}

#[test]
fn test_op_coverage_flip() {
    let node = r#"{
        "target": "torch.ops.aten.flip.default",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dims", "arg": {"as_ints": [1]}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "flip_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "flip_out", &[1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Flip { dim: 1 }),
        "flip",
    );
}

// ===========================================================================
// Linear (matmul + bias)
// ===========================================================================

#[test]
fn test_op_coverage_linear() {
    // linear: x:[1,4] * weight:[8,4]^T + bias:[8] -> [1,8]
    let node = r#"{
        "target": "torch.ops.aten.linear.default",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "weight", "arg": {"as_tensor": {"name": "p_weight"}}, "kind": 1},
            {"name": "bias", "arg": {"as_tensor": {"name": "p_bias"}}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "linear_out"}}],
        "metadata": {}
    }"#;
    // Build graph JSON with parameter input_specs.
    let json = format!(
        r#"{{
    "graph_module": {{
        "graph": {{
            "inputs": [
                {{"as_tensor": {{"name": "p_weight"}}}},
                {{"as_tensor": {{"name": "p_bias"}}}},
                {{"as_tensor": {{"name": "x"}}}}
            ],
            "outputs": [{{"as_tensor": {{"name": "linear_out"}}}}],
            "nodes": [{node}],
            "tensor_values": {{
                "x": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 4}}], "requires_grad": false, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "p_weight": {{"dtype": 7, "sizes": [{{"as_int": 8}}, {{"as_int": 4}}], "requires_grad": true, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "p_bias": {{"dtype": 7, "sizes": [{{"as_int": 8}}], "requires_grad": true, "strides": [{{"as_int": 1}}]}},
                "linear_out": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 8}}], "requires_grad": false, "strides": [{{"as_int": 8}}, {{"as_int": 1}}]}}
            }},
            "is_single_tensor_return": true
        }},
        "signature": {{
            "input_specs": [
                {{"parameter": {{"arg": {{"name": "p_weight"}}, "parameter_name": "fc.weight"}}}},
                {{"parameter": {{"arg": {{"name": "p_bias"}}, "parameter_name": "fc.bias"}}}},
                {{"user_input": {{"arg": {{"as_tensor": {{"name": "x"}}}}}}}}
            ],
            "output_specs": [
                {{"user_output": {{"arg": {{"as_tensor": {{"name": "linear_out"}}}}}}}}
            ]
        }},
        "module_call_graph": []
    }},
    "schema_version": {{"major": 8, "minor": 15}},
    "opset_version": {{"aten": 10}},
    "range_constraints": {{}}
}}"#
    );

    // Provide synthetic weight data.
    let mut weights = HashMap::new();
    weights.insert(
        "p_weight".to_string(),
        ResolvedWeight::new(vec![0.01; 32], vec![8, 4]),
    );
    weights.insert(
        "p_bias".to_string(),
        ResolvedWeight::new(vec![0.0; 8], vec![8]),
    );

    let imported = import_from_json_with_weights(&json, weights);
    assert_eq!(imported.num_user_inputs, 1);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Linear { .. }),
        "linear",
    );
}

#[test]
fn test_op_coverage_linear_no_bias() {
    // linear without bias: x:[1,4] * weight:[8,4]^T -> [1,8]
    let node = r#"{
        "target": "torch.ops.aten.linear.default",
        "inputs": [
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "weight", "arg": {"as_tensor": {"name": "p_weight"}}, "kind": 1},
            {"name": "bias", "arg": {"as_none": true}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "linear_out"}}],
        "metadata": {}
    }"#;

    let json = format!(
        r#"{{
    "graph_module": {{
        "graph": {{
            "inputs": [
                {{"as_tensor": {{"name": "p_weight"}}}},
                {{"as_tensor": {{"name": "x"}}}}
            ],
            "outputs": [{{"as_tensor": {{"name": "linear_out"}}}}],
            "nodes": [{node}],
            "tensor_values": {{
                "x": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 4}}], "requires_grad": false, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "p_weight": {{"dtype": 7, "sizes": [{{"as_int": 8}}, {{"as_int": 4}}], "requires_grad": true, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "linear_out": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 8}}], "requires_grad": false, "strides": [{{"as_int": 8}}, {{"as_int": 1}}]}}
            }},
            "is_single_tensor_return": true
        }},
        "signature": {{
            "input_specs": [
                {{"parameter": {{"arg": {{"name": "p_weight"}}, "parameter_name": "fc.weight"}}}},
                {{"user_input": {{"arg": {{"as_tensor": {{"name": "x"}}}}}}}}
            ],
            "output_specs": [
                {{"user_output": {{"arg": {{"as_tensor": {{"name": "linear_out"}}}}}}}}
            ]
        }},
        "module_call_graph": []
    }},
    "schema_version": {{"major": 8, "minor": 15}},
    "opset_version": {{"aten": 10}},
    "range_constraints": {{}}
}}"#
    );

    let mut weights = HashMap::new();
    weights.insert(
        "p_weight".to_string(),
        ResolvedWeight::new(vec![0.01; 32], vec![8, 4]),
    );

    let imported = import_from_json_with_weights(&json, weights);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Linear { bias, .. } if bias.is_none()),
        "linear no bias",
    );
}

// ===========================================================================
// Softmax / log_softmax
// ===========================================================================

#[test]
fn test_op_coverage_softmax() {
    let node = r#"{
        "target": "torch.ops.aten.softmax.int",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_int": 1}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "softmax_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "softmax_out", &[1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Softmax { dim: 1 }),
        "softmax",
    );
}

#[test]
fn test_op_coverage_softmax_internal() {
    let node = r#"{
        "target": "torch.ops.aten._softmax.default",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_int": 0}, "kind": 1},
            {"name": "half_to_float", "arg": {"as_bool": false}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "softmax_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "softmax_out", &[1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Softmax { dim: 0 }),
        "_softmax",
    );
}

#[test]
fn test_op_coverage_log_softmax() {
    let node = r#"{
        "target": "torch.ops.aten.log_softmax.int",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_int": 1}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "logsoftmax_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "logsoftmax_out", &[1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::LogSoftmax { dim: 1 }),
        "log_softmax",
    );
}

#[test]
fn test_op_coverage_log_softmax_internal() {
    let node = r#"{
        "target": "torch.ops.aten._log_softmax.default",
        "inputs": [
            {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_int": 0}, "kind": 1},
            {"name": "half_to_float", "arg": {"as_bool": false}, "kind": 1}
        ],
        "outputs": [{"as_tensor": {"name": "logsoftmax_out"}}],
        "metadata": {}
    }"#;
    let json = single_op_graph_json(node, "logsoftmax_out", &[1, 4], "", "");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::LogSoftmax { dim: 0 }),
        "_log_softmax",
    );
}

// ===========================================================================
// Identity / memory layout ops
// ===========================================================================

#[test]
fn test_op_coverage_contiguous() {
    let json = unary_op_graph_json("torch.ops.aten.contiguous.default", "contig_out");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Reshape { target_shape } if target_shape.is_empty()),
        "contiguous (identity reshape)",
    );
}

#[test]
fn test_op_coverage_clone() {
    let json = unary_op_graph_json("torch.ops.aten.clone.default", "clone_out");
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Reshape { target_shape } if target_shape.is_empty()),
        "clone (identity reshape)",
    );
}

// ===========================================================================
// Supported ops list completeness
// ===========================================================================

/// Verify that supported_ops() returns a non-empty, sorted, deduplicated list
/// and that known categories are represented.
#[test]
fn test_supported_ops_list_properties() {
    let ops = nn_import::supported_ops();
    assert!(!ops.is_empty(), "supported_ops should be non-empty");

    // Verify sorted.
    for pair in ops.windows(2) {
        assert!(
            pair[0] <= pair[1],
            "supported_ops not sorted: {:?} > {:?}",
            pair[0],
            pair[1]
        );
    }

    // Verify no duplicates.
    let unique: std::collections::HashSet<&&str> = ops.iter().collect();
    assert_eq!(unique.len(), ops.len(), "supported_ops contains duplicates");

    // Check that key categories are represented.
    let has = |prefix: &str| ops.iter().any(|o| o.starts_with(prefix));
    assert!(has("aten::relu"), "missing unary ops");
    assert!(has("aten::add"), "missing binary ops");
    assert!(
        has("aten::sum") || has("aten::mean"),
        "missing reduction ops"
    );
    assert!(
        has("aten::reshape") || has("aten::view"),
        "missing shape ops"
    );
    assert!(has("aten::linear"), "missing linear op");
    assert!(has("aten::softmax"), "missing softmax");
    assert!(has("aten::conv"), "missing convolution ops");
    assert!(has("aten::lstm"), "missing recurrent ops");
    assert!(has("aten::embedding"), "missing embedding");
}

/// Verify the total count of supported ops is at least the number from the
/// SUPPORTED_ATEN_OPS table (currently 83+ ops). This catches accidental
/// removal of ops from the dispatch table.
#[test]
fn test_supported_ops_minimum_count() {
    let ops = nn_import::supported_ops();
    assert!(
        ops.len() >= 80,
        "expected at least 80 supported ops, got {}",
        ops.len()
    );
}

// ===========================================================================
// Multi-op pipeline test: unary chain
// ===========================================================================

/// Test a two-op pipeline: x -> relu -> neg -> output.
/// Verifies that multi-node graphs with dependencies are imported correctly.
#[test]
fn test_op_coverage_unary_chain_relu_neg() {
    let nodes = r#"
        {
            "target": "torch.ops.aten.relu.default",
            "inputs": [
                {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}
            ],
            "outputs": [{"as_tensor": {"name": "relu_mid"}}],
            "metadata": {}
        },
        {
            "target": "torch.ops.aten.neg.default",
            "inputs": [
                {"name": "input", "arg": {"as_tensor": {"name": "relu_mid"}}, "kind": 1}
            ],
            "outputs": [{"as_tensor": {"name": "neg_out"}}],
            "metadata": {}
        }
    "#;
    let extra_tv = r#""relu_mid": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]}"#;
    let json = single_op_graph_json(nodes, "neg_out", &[1, 4], "", extra_tv);
    let imported = import_from_json(&json);

    // Output should be Neg (the last op in the chain).
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Neg),
        "chain output neg",
    );

    // Graph should have 3 nodes: input + relu + neg.
    assert_eq!(imported.graph.len(), 3, "expected 3 nodes in chain graph");
}

// ===========================================================================
// Multi-op pipeline test: binary + unary
// ===========================================================================

/// Test: x + y -> sigmoid -> output.
/// Verifies binary op feeding into unary op.
#[test]
fn test_op_coverage_binary_then_unary() {
    let nodes = r#"
        {
            "target": "torch.ops.aten.add.Tensor",
            "inputs": [
                {"name": "self", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
                {"name": "other", "arg": {"as_tensor": {"name": "y"}}, "kind": 1}
            ],
            "outputs": [{"as_tensor": {"name": "add_mid"}}],
            "metadata": {}
        },
        {
            "target": "torch.ops.aten.sigmoid.default",
            "inputs": [
                {"name": "input", "arg": {"as_tensor": {"name": "add_mid"}}, "kind": 1}
            ],
            "outputs": [{"as_tensor": {"name": "sig_out"}}],
            "metadata": {}
        }
    "#;
    // Build a two-input graph with an extra intermediate tensor_values entry.
    let json = format!(
        r#"{{
    "graph_module": {{
        "graph": {{
            "inputs": [
                {{"as_tensor": {{"name": "x"}}}},
                {{"as_tensor": {{"name": "y"}}}}
            ],
            "outputs": [{{"as_tensor": {{"name": "sig_out"}}}}],
            "nodes": [{nodes}],
            "tensor_values": {{
                "x": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 4}}], "requires_grad": false, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "y": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 4}}], "requires_grad": false, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "add_mid": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 4}}], "requires_grad": false, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}},
                "sig_out": {{"dtype": 7, "sizes": [{{"as_int": 1}}, {{"as_int": 4}}], "requires_grad": false, "strides": [{{"as_int": 4}}, {{"as_int": 1}}]}}
            }},
            "is_single_tensor_return": true
        }},
        "signature": {{
            "input_specs": [
                {{"user_input": {{"arg": {{"as_tensor": {{"name": "x"}}}}}}}},
                {{"user_input": {{"arg": {{"as_tensor": {{"name": "y"}}}}}}}}
            ],
            "output_specs": [
                {{"user_output": {{"arg": {{"as_tensor": {{"name": "sig_out"}}}}}}}}
            ]
        }},
        "module_call_graph": []
    }},
    "schema_version": {{"major": 8, "minor": 15}},
    "opset_version": {{"aten": 10}},
    "range_constraints": {{}}
}}"#
    );
    let imported = import_from_json(&json);
    assert_output_op(
        &imported,
        |op| matches!(op, TraceOp::Sigmoid),
        "binary+unary chain output sigmoid",
    );
    // 2 inputs (x, y) + 2 ops (add, sigmoid) = 4 nodes.
    assert_eq!(imported.graph.len(), 4, "expected 4 nodes");
}
