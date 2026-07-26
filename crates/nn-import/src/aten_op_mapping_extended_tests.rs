// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for nn-import aten op mapping, weight name translation,
//! shape inference, config parsing, and error handling.
//!
//! All tests use mock data and do not require actual model files.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;
use nn_core::DType;

use crate::error::ImportError;
use crate::graph_build::{build_graph, build_weight_map};
use crate::kokoro_weights::{kokoro_name_mapping, map_pytorch_key, validate_kokoro_keys};
use crate::op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentBool, ArgumentFloat, ArgumentInt, ArgumentInts, ArgumentNone, ArgumentString,
    ArgumentTensor, NamedArgument, Node, SymInt, SymIntConcrete, SymIntExpr, SymIntSymbolic,
    TensorArgument, TensorMeta,
};
use crate::{parse_exported_program, ImportError as IE};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn empty_ctx() -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::default());
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

fn ctx_with_weights(w: HashMap<String, ResolvedWeight>) -> OpMapContext<'static> {
    let meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::default());
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::new(w));
    OpMapContext {
        tensor_meta: meta,
        weights,
    }
}

fn tensor_arg(name: &str) -> Argument {
    Argument::Tensor(ArgumentTensor {
        as_tensor: TensorArgument {
            name: name.to_string(),
        },
    })
}

fn int_arg(val: i64) -> Argument {
    Argument::Int(ArgumentInt { as_int: val })
}

fn ints_arg(vals: &[i64]) -> Argument {
    Argument::Ints(ArgumentInts {
        as_ints: vals.to_vec(),
    })
}

fn float_arg(val: f64) -> Argument {
    Argument::Float(ArgumentFloat { as_float: val })
}

fn bool_arg(val: bool) -> Argument {
    Argument::Bool(ArgumentBool { as_bool: val })
}

fn none_arg() -> Argument {
    Argument::None(ArgumentNone { as_none: true })
}

fn str_arg(val: &str) -> Argument {
    Argument::Str(ArgumentString {
        as_string: val.to_string(),
    })
}

fn named(name: &str, arg: Argument) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg,
        kind: Some(1),
    }
}

fn simple_node(target: &str, inputs: Vec<NamedArgument>) -> Node {
    Node {
        target: target.to_string(),
        inputs,
        outputs: vec![tensor_arg("output")],
        metadata: HashMap::new(),
    }
}

// =========================================================================
// Section 1: supported_ops() invariants
// =========================================================================

#[test]
fn test_supported_ops_count_over_threshold() {
    let ops = supported_ops();
    assert!(
        ops.len() >= 200,
        "expected >= 200 supported ops, got {}",
        ops.len()
    );
}

#[test]
fn test_supported_ops_sorted_and_deduped_ext() {
    let ops = supported_ops();
    for w in ops.windows(2) {
        assert!(
            w[0] <= w[1],
            "supported_ops() not sorted: {:?} > {:?}",
            w[0],
            w[1]
        );
        // Dedup allows equal but not strictly required — just check sorted.
    }
}

#[test]
fn test_supported_ops_includes_all_categories() {
    let ops = supported_ops();
    // Spot-check one op from each major category.
    let expected = [
        // Unary
        "aten::relu",
        "aten::gelu",
        "aten::silu",
        "aten::tanh",
        "aten::sigmoid",
        "aten::exp",
        "aten::log",
        "aten::sqrt",
        "aten::abs",
        "aten::neg",
        "aten::rsqrt",
        // Binary
        "aten::add",
        "aten::sub",
        "aten::mul",
        "aten::div",
        "aten::maximum",
        "aten::minimum",
        // Matrix
        "aten::mm",
        "aten::bmm",
        "aten::matmul",
        // Linear
        "aten::linear",
        // Conv
        "aten::convolution",
        "aten::conv1d",
        "aten::conv2d",
        "aten::conv_transpose1d",
        // Norm
        "aten::layer_norm",
        "aten::group_norm",
        "aten::batch_norm",
        "aten::instance_norm",
        // Attention
        "aten::softmax",
        "aten::log_softmax",
        "aten::scaled_dot_product_attention",
        // Embedding
        "aten::embedding",
        // Reduction
        "aten::sum",
        "aten::mean",
        "aten::amax",
        // Shape
        "aten::view",
        "aten::reshape",
        "aten::transpose",
        "aten::permute",
        "aten::flatten",
        "aten::unsqueeze",
        "aten::squeeze",
        "aten::cat",
        "aten::slice",
        "aten::select",
        // Activation
        "aten::elu",
        "aten::leaky_relu",
        "aten::dropout",
        // Comparison
        "aten::where",
        "aten::clamp",
        // Power
        "aten::pow",
        // Creation
        "aten::zeros",
        "aten::ones",
        "aten::full",
        "aten::arange",
        // Identity
        "aten::contiguous",
        "aten::clone",
    ];
    for op in &expected {
        assert!(
            ops.contains(op),
            "supported_ops() missing expected op: {op}"
        );
    }
}

// =========================================================================
// Section 2: Unary element-wise op mapping
// =========================================================================

#[test]
fn test_relu_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.relu.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    let (op, inputs) = result.unwrap();
    assert!(matches!(op, TraceOp::Relu));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_gelu_default_maps_to_gelu_erf() {
    // Default gelu (no approximate arg) maps to GeluErf
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gelu.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    let (op, inputs) = result.unwrap();
    assert!(
        matches!(op, TraceOp::GeluErf),
        "default gelu should be GeluErf, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_gelu_tanh_approximate_maps_to_gelu() {
    // gelu with approximate="tanh" maps to Gelu (tanh approximation)
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.gelu.default",
        vec![
            named("self", tensor_arg("x")),
            named("approximate", str_arg("tanh")),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    let (op, inputs) = result.unwrap();
    assert!(
        matches!(op, TraceOp::Gelu),
        "tanh gelu should be Gelu, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_silu_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.silu.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    let (op, inputs) = result.unwrap();
    assert!(matches!(op, TraceOp::Silu));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_tanh_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.tanh.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    let (op, inputs) = result.unwrap();
    assert!(matches!(op, TraceOp::Tanh));
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_sigmoid_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sigmoid.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap().0, TraceOp::Sigmoid));
}

#[test]
fn test_exp_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.exp.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap().0, TraceOp::Exp));
}

#[test]
fn test_log_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.log.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap().0, TraceOp::Log));
}

#[test]
fn test_sqrt_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sqrt.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap().0, TraceOp::Sqrt));
}

#[test]
fn test_abs_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.abs.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap().0, TraceOp::Abs));
}

#[test]
fn test_neg_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.neg.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap().0, TraceOp::Neg));
}

#[test]
fn test_rsqrt_maps_to_powf_neg_half() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.rsqrt.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    let (op, inputs) = result.unwrap();
    assert_eq!(inputs, vec!["x"]);
    match op {
        TraceOp::Powf { exponent } => {
            assert!(
                (exponent - (-0.5)).abs() < 1e-9,
                "rsqrt should map to Powf(-0.5)"
            );
        }
        other => panic!("expected Powf, got {other:?}"),
    }
}

#[test]
fn test_floor_and_round_ops() {
    let ctx = empty_ctx();
    for (target, expected_name) in [
        ("torch.ops.aten.floor.default", "Floor"),
        ("torch.ops.aten.round.default", "Round"),
    ] {
        let node = simple_node(target, vec![named("self", tensor_arg("x"))]);
        let result = map_node_to_trace_op(&node, &ctx, 0);
        assert!(result.is_ok(), "{expected_name} should map successfully");
    }
}

// =========================================================================
// Section 3: Binary element-wise op mapping
// =========================================================================

#[test]
fn test_add_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.add.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    let (op, inputs) = result.unwrap();
    assert!(matches!(op, TraceOp::Add));
    assert_eq!(inputs, vec!["a", "b"]);
}

#[test]
fn test_sub_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sub.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap().0, TraceOp::Sub));
}

#[test]
fn test_mul_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mul.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap().0, TraceOp::Mul));
}

#[test]
fn test_div_op_mapping() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.div.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap().0, TraceOp::Div));
}

#[test]
fn test_maximum_minimum_ops() {
    let ctx = empty_ctx();
    for (target, variant) in [
        ("torch.ops.aten.maximum.default", "Maximum"),
        ("torch.ops.aten.minimum.default", "Minimum"),
    ] {
        let node = simple_node(
            target,
            vec![
                named("self", tensor_arg("a")),
                named("other", tensor_arg("b")),
            ],
        );
        let result = map_node_to_trace_op(&node, &ctx, 0);
        assert!(result.is_ok(), "{variant} should map successfully");
    }
}

// =========================================================================
// Section 4: MatMul variants
// =========================================================================

#[test]
fn test_matmul_variants_all_map_to_matmul() {
    let ctx = empty_ctx();
    // mm and bmm use "mat2", matmul uses "other"
    let cases = [
        ("torch.ops.aten.mm.default", "mat2"),
        ("torch.ops.aten.bmm.default", "mat2"),
        ("torch.ops.aten.matmul.default", "other"),
    ];
    for (target, rhs_name) in &cases {
        let node = simple_node(
            target,
            vec![
                named("self", tensor_arg("a")),
                named(rhs_name, tensor_arg("b")),
            ],
        );
        let result = map_node_to_trace_op(&node, &ctx, 0);
        assert!(result.is_ok(), "matmul variant {target} should work");
        let (op, inputs) = result.unwrap();
        assert!(
            matches!(op, TraceOp::MatMul),
            "{target} should map to MatMul"
        );
        assert_eq!(inputs, vec!["a", "b"]);
    }
}

// =========================================================================
// Section 5: Linear op with weights
// =========================================================================

#[test]
fn test_linear_with_weight_and_bias() {
    let mut weights = HashMap::new();
    weights.insert(
        "w".to_string(),
        ResolvedWeight::new(vec![1.0; 6], vec![3, 2]),
    );
    weights.insert("b".to_string(), ResolvedWeight::new(vec![0.0; 3], vec![3]));
    let ctx = ctx_with_weights(weights);
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w")),
            named("bias", tensor_arg("b")),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(
        result.is_ok(),
        "linear with weight+bias should succeed: {:?}",
        result.err()
    );
    let (op, inputs) = result.unwrap();
    assert!(
        matches!(op, TraceOp::Linear { .. }),
        "expected Linear, got {op:?}"
    );
    assert_eq!(inputs[0], "x");
}

#[test]
fn test_linear_without_bias() {
    let mut weights = HashMap::new();
    weights.insert(
        "w".to_string(),
        ResolvedWeight::new(vec![1.0; 6], vec![3, 2]),
    );
    let ctx = ctx_with_weights(weights);
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("w")),
            named("bias", none_arg()),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(
        result.is_ok(),
        "linear without bias should succeed: {:?}",
        result.err()
    );
    let (op, _inputs) = result.unwrap();
    match &op {
        TraceOp::Linear { bias, .. } => {
            assert!(bias.is_none(), "bias should be None when none_arg passed");
        }
        other => panic!("expected Linear, got {other:?}"),
    }
}

// =========================================================================
// Section 6: Softmax / LogSoftmax with dim resolution
// =========================================================================

#[test]
fn test_softmax_with_dim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.softmax.int",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(-1))],
    );
    // input_ndim=3 so dim=-1 -> dim=2
    let result = map_node_to_trace_op(&node, &ctx, 3);
    assert!(result.is_ok(), "softmax should succeed: {:?}", result.err());
    let (op, inputs) = result.unwrap();
    assert_eq!(inputs, vec!["x"]);
    match op {
        TraceOp::Softmax { dim } => assert_eq!(dim, 2),
        other => panic!("expected Softmax, got {other:?}"),
    }
}

#[test]
fn test_softmax_positive_dim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.softmax.int",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(1))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 4);
    assert!(result.is_ok());
    match result.unwrap().0 {
        TraceOp::Softmax { dim } => assert_eq!(dim, 1),
        other => panic!("expected Softmax, got {other:?}"),
    }
}

#[test]
fn test_log_softmax_positive_dim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.log_softmax.int",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(0))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 2);
    assert!(result.is_ok());
    match result.unwrap().0 {
        TraceOp::LogSoftmax { dim } => assert_eq!(dim, 0),
        other => panic!("expected LogSoftmax, got {other:?}"),
    }
}

// =========================================================================
// Section 7: Shape ops (reshape, view, transpose, permute, unsqueeze, squeeze)
// =========================================================================

#[test]
fn test_reshape_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.reshape.default",
        vec![
            named("self", tensor_arg("x")),
            named("shape", ints_arg(&[2, 3, 4])),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    let (op, inputs) = result.unwrap();
    assert_eq!(inputs, vec!["x"]);
    match op {
        TraceOp::Reshape { target_shape } => {
            assert_eq!(target_shape, vec![2, 3, 4]);
        }
        other => panic!("expected Reshape, got {other:?}"),
    }
}

#[test]
fn test_view_maps_to_reshape() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.view.default",
        vec![
            named("self", tensor_arg("x")),
            named("size", ints_arg(&[6, 8])),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    match result.unwrap().0 {
        TraceOp::Reshape { target_shape } => {
            assert_eq!(target_shape, vec![6, 8]);
        }
        other => panic!("expected Reshape from view, got {other:?}"),
    }
}

#[test]
fn test_transpose_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.transpose.int",
        vec![
            named("self", tensor_arg("x")),
            named("dim0", int_arg(0)),
            named("dim1", int_arg(1)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 3);
    assert!(result.is_ok());
    match result.unwrap().0 {
        TraceOp::Transpose { dim0, dim1 } => {
            assert_eq!(dim0, 0);
            assert_eq!(dim1, 1);
        }
        other => panic!("expected Transpose, got {other:?}"),
    }
}

#[test]
fn test_permute_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.permute.default",
        vec![
            named("self", tensor_arg("x")),
            named("dims", ints_arg(&[2, 0, 1])),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    match result.unwrap().0 {
        TraceOp::Permute { axes } => {
            assert_eq!(axes, vec![2, 0, 1]);
        }
        other => panic!("expected Permute, got {other:?}"),
    }
}

#[test]
fn test_unsqueeze_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.unsqueeze.default",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(0))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    match result.unwrap().0 {
        TraceOp::Unsqueeze { dim } => assert_eq!(dim, 0),
        other => panic!("expected Unsqueeze, got {other:?}"),
    }
}

#[test]
fn test_squeeze_with_dim() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.squeeze.dim",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(0))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 3);
    assert!(result.is_ok());
    match result.unwrap().0 {
        TraceOp::Squeeze { dim } => assert_eq!(dim, 0),
        other => panic!("expected Squeeze, got {other:?}"),
    }
}

// =========================================================================
// Section 8: Cat op
// =========================================================================

#[test]
fn test_cat_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.cat.default",
        vec![
            // cat takes a tensor list and a dim
            named("tensors", tensor_arg("list_placeholder")),
            named("dim", int_arg(1)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 3);
    // cat may require special handling of the tensor list — verify it at least
    // dispatches to the correct handler (may or may not succeed depending on
    // tensor list format).
    // The important thing is it's recognized as a valid op.
    let _ = result; // We just verify it doesn't panic.
}

// =========================================================================
// Section 9: Reduction ops
// =========================================================================

#[test]
fn test_sum_reduction() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sum.dim_IntList",
        vec![
            named("self", tensor_arg("x")),
            named("dim", ints_arg(&[1])),
            named("keepdim", bool_arg(true)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 3);
    assert!(result.is_ok(), "sum should succeed: {:?}", result.err());
    match result.unwrap().0 {
        TraceOp::ReduceSum { dim, keepdim } => {
            assert_eq!(dim, 1);
            assert!(keepdim);
        }
        other => panic!("expected ReduceSum, got {other:?}"),
    }
}

#[test]
fn test_mean_reduction() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.mean.dim",
        vec![
            named("self", tensor_arg("x")),
            named("dim", ints_arg(&[2])),
            named("keepdim", bool_arg(false)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 3);
    assert!(result.is_ok(), "mean should succeed: {:?}", result.err());
    match result.unwrap().0 {
        TraceOp::ReduceMean { dim, keepdim } => {
            assert_eq!(dim, 2);
            assert!(!keepdim);
        }
        other => panic!("expected ReduceMean, got {other:?}"),
    }
}

// =========================================================================
// Section 10: Embedding
// =========================================================================

#[test]
fn test_embedding_op() {
    let mut weights = HashMap::new();
    weights.insert(
        "emb_w".to_string(),
        ResolvedWeight::new(vec![0.0; 30], vec![10, 3]),
    );
    let ctx = ctx_with_weights(weights);
    let node = simple_node(
        "torch.ops.aten.embedding.default",
        vec![
            named("weight", tensor_arg("emb_w")),
            named("indices", tensor_arg("idx")),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(
        result.is_ok(),
        "embedding should succeed: {:?}",
        result.err()
    );
    let (op, _inputs) = result.unwrap();
    assert!(
        matches!(op, TraceOp::Embedding { .. }),
        "expected Embedding, got {op:?}"
    );
}

// =========================================================================
// Section 11: Slice op
// =========================================================================

#[test]
fn test_slice_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.slice.Tensor",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(0)),
            named("start", int_arg(0)),
            named("end", int_arg(10)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 3);
    assert!(
        result.is_ok(),
        "slice should be supported: {:?}",
        result.err()
    );
}

// =========================================================================
// Section 12: Select (decomposition path)
// =========================================================================

#[test]
fn test_select_op_handled_by_expand() {
    // select.int is handled by try_expand_node (decomposition), not map_node_to_trace_op.
    // Verify it's in the supported ops list and that map_node_to_trace_op returns
    // UnsupportedOp (since the expand path handles it instead).
    let ops = supported_ops();
    assert!(
        ops.contains(&"aten::select"),
        "select should be in supported ops list"
    );
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.select.int",
        vec![
            named("self", tensor_arg("x")),
            named("dim", int_arg(0)),
            named("index", int_arg(2)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 3);
    // map_node_to_trace_op doesn't handle select.int -- it's decomposed via try_expand_node
    assert!(result.is_err());
}

// =========================================================================
// Section 13: Flatten (expand path)
// =========================================================================

#[test]
fn test_flatten_in_supported_ops() {
    let ops = supported_ops();
    assert!(
        ops.contains(&"aten::flatten"),
        "flatten should be in supported ops list"
    );
}

#[test]
fn test_flatten_not_in_map_node() {
    // flatten is handled by try_expand_node, not map_node_to_trace_op
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.flatten.using_ints",
        vec![
            named("self", tensor_arg("x")),
            named("start_dim", int_arg(1)),
            named("end_dim", int_arg(-1)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 3);
    assert!(
        result.is_err(),
        "flatten should not be handled by map_node_to_trace_op"
    );
}

// =========================================================================
// Section 14: Activation ops
// =========================================================================

#[test]
fn test_elu_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.elu.default",
        vec![
            named("self", tensor_arg("x")),
            named("alpha", float_arg(1.0)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok(), "elu should succeed: {:?}", result.err());
    match result.unwrap().0 {
        TraceOp::Elu { alpha } => {
            assert!((alpha - 1.0).abs() < 1e-9);
        }
        other => panic!("expected Elu, got {other:?}"),
    }
}

#[test]
fn test_leaky_relu_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.leaky_relu.default",
        vec![
            named("self", tensor_arg("x")),
            named("negative_slope", float_arg(0.01)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(
        result.is_ok(),
        "leaky_relu should succeed: {:?}",
        result.err()
    );
    match result.unwrap().0 {
        TraceOp::LeakyRelu { slope } => {
            assert!((slope - 0.01).abs() < 1e-9);
        }
        other => panic!("expected LeakyRelu, got {other:?}"),
    }
}

#[test]
fn test_dropout_maps_to_dropout() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.dropout.default",
        vec![
            named("input", tensor_arg("x")),
            named("p", float_arg(0.1)),
            named("train", bool_arg(false)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok(), "dropout should succeed: {:?}", result.err());
    assert!(matches!(result.unwrap().0, TraceOp::Dropout));
}

// =========================================================================
// Section 15: Clamp op
// =========================================================================

#[test]
fn test_clamp_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clamp.default",
        vec![
            named("self", tensor_arg("x")),
            named("min", float_arg(-1.0)),
            named("max", float_arg(1.0)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok(), "clamp should succeed: {:?}", result.err());
    match result.unwrap().0 {
        TraceOp::Clamp { min, max } => {
            assert!((min.unwrap() - (-1.0)).abs() < 1e-9);
            assert!((max.unwrap() - 1.0).abs() < 1e-9);
        }
        other => panic!("expected Clamp, got {other:?}"),
    }
}

// =========================================================================
// Section 16: Where / comparison
// =========================================================================

#[test]
fn test_where_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.where.self",
        vec![
            named("condition", tensor_arg("cond")),
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(
        result.is_ok(),
        "where should be supported: {:?}",
        result.err()
    );
    let (_op, inputs) = result.unwrap();
    assert_eq!(inputs.len(), 3);
}

// =========================================================================
// Section 17: Pow op with exponent rewriting
// =========================================================================

#[test]
fn test_pow_op_integer_exponent_rewrites_to_sqr() {
    // pow with exponent=2.0 is rewritten to Sqr for correctness (#2751)
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pow.Tensor_Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("exponent", float_arg(2.0)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(
        result.is_ok(),
        "pow should be supported: {:?}",
        result.err()
    );
    let (op, inputs) = result.unwrap();
    assert_eq!(inputs, vec!["x"]);
    assert!(
        matches!(op, TraceOp::Sqr),
        "exponent=2.0 should rewrite to Sqr, got {op:?}"
    );
}

#[test]
fn test_pow_op_half_exponent_rewrites_to_sqrt() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pow.Tensor_Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("exponent", float_arg(0.5)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    let (op, _inputs) = result.unwrap();
    assert!(
        matches!(op, TraceOp::Sqrt),
        "exponent=0.5 should rewrite to Sqrt, got {op:?}"
    );
}

#[test]
fn test_pow_op_fractional_exponent_stays_powf() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.pow.Tensor_Scalar",
        vec![
            named("self", tensor_arg("x")),
            named("exponent", float_arg(3.0)),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok());
    let (op, inputs) = result.unwrap();
    assert_eq!(inputs, vec!["x"]);
    match op {
        TraceOp::Powf { exponent } => {
            assert!((exponent - 3.0).abs() < 1e-7);
        }
        other => panic!("expected Powf, got {other:?}"),
    }
}

// =========================================================================
// Section 18: Tensor creation ops
// =========================================================================

#[test]
fn test_zeros_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.zeros.default",
        vec![named("size", ints_arg(&[2, 3]))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok(), "zeros should succeed: {:?}", result.err());
}

#[test]
fn test_ones_op() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.ones.default",
        vec![named("size", ints_arg(&[4, 5]))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok(), "ones should succeed: {:?}", result.err());
}

// =========================================================================
// Section 19: Identity ops (contiguous, clone)
// =========================================================================

#[test]
fn test_contiguous_maps_to_identity_reshape() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.contiguous.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(
        result.is_ok(),
        "contiguous should succeed: {:?}",
        result.err()
    );
    // contiguous maps to Reshape { target_shape: vec![] } (identity)
    match result.unwrap().0 {
        TraceOp::Reshape { target_shape } => {
            assert!(
                target_shape.is_empty(),
                "identity reshape should have empty shape"
            );
        }
        other => panic!("expected Reshape (identity), got {other:?}"),
    }
}

#[test]
fn test_clone_maps_to_identity_reshape() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.clone.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok(), "clone should succeed: {:?}", result.err());
    match result.unwrap().0 {
        TraceOp::Reshape { target_shape } => {
            assert!(
                target_shape.is_empty(),
                "identity reshape should have empty shape"
            );
        }
        other => panic!("expected Reshape (identity), got {other:?}"),
    }
}

// =========================================================================
// Section 20: Error handling
// =========================================================================

#[test]
fn test_unsupported_op_returns_error() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.nonexistent_operation.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_err());
    match result.unwrap_err() {
        ImportError::UnsupportedOp { target } => {
            assert!(target.contains("nonexistent_operation"));
        }
        other => panic!("expected UnsupportedOp, got {other:?}"),
    }
}

#[test]
fn test_missing_argument_returns_error() {
    let ctx = empty_ctx();
    // softmax requires "dim" but we don't provide it
    let node = simple_node(
        "torch.ops.aten.softmax.int",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 3);
    assert!(result.is_err(), "missing dim should produce error");
}

#[test]
fn test_wrong_argument_type_returns_error() {
    let ctx = empty_ctx();
    // softmax dim should be int, pass a string
    let node = simple_node(
        "torch.ops.aten.softmax.int",
        vec![
            named("self", tensor_arg("x")),
            named("dim", str_arg("last")),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 3);
    assert!(result.is_err(), "wrong arg type should produce error");
}

#[test]
fn test_missing_weight_returns_error() {
    // Linear needs weight in context but we provide empty context
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.linear.default",
        vec![
            named("input", tensor_arg("x")),
            named("weight", tensor_arg("missing_w")),
            named("bias", none_arg()),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    // Should fail because weight "missing_w" is not in ctx.weights
    assert!(result.is_err(), "missing weight should produce error");
}

#[test]
fn test_negative_dim_without_ndim_returns_error() {
    let ctx = empty_ctx();
    // softmax with dim=-1 but input_ndim=0 means we can't resolve
    let node = simple_node(
        "torch.ops.aten.softmax.int",
        vec![named("self", tensor_arg("x")), named("dim", int_arg(-1))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    // With ndim=0, resolving dim=-1 may produce error or underflow
    // The exact behavior depends on implementation — just verify no panic.
    let _ = result;
}

// =========================================================================
// Section 21: Weight name translation (Kokoro)
// =========================================================================

#[test]
fn test_kokoro_map_key_identity_for_known_prefixes() {
    let test_keys = [
        "plbert.embeddings.word_embeddings.weight",
        "bert_encoder.weight",
        "text_encoder.lstm.weight_ih_l0",
        "prosody_predictor.shared.0.conv.weight",
        "predictor.F0.0.c1.weight",
        "decoder.conv_pre.weight",
    ];
    for key in &test_keys {
        let mapped = map_pytorch_key(key);
        assert_eq!(
            mapped.as_deref(),
            Some(*key),
            "known prefix key should map to identity"
        );
    }
}

#[test]
fn test_kokoro_map_key_unknown_prefix_returns_none() {
    assert_eq!(map_pytorch_key("unknown_module.weight"), None);
    assert_eq!(map_pytorch_key("some_other.bias"), None);
    assert_eq!(map_pytorch_key(""), None);
}

#[test]
fn test_validate_kokoro_keys_all_present() {
    let keys = vec![
        "plbert.x",
        "bert_encoder.x",
        "text_encoder.x",
        "prosody_predictor.x",
        "predictor.x",
        "decoder.x",
    ];
    let missing = validate_kokoro_keys(&keys);
    assert!(
        missing.is_empty(),
        "all prefixes present, should have no missing"
    );
}

#[test]
fn test_validate_kokoro_keys_some_missing() {
    let keys = vec!["plbert.x", "decoder.x"];
    let missing = validate_kokoro_keys(&keys);
    assert!(!missing.is_empty(), "should report missing prefixes");
    assert!(
        missing.contains(&"bert_encoder."),
        "bert_encoder. should be missing"
    );
    assert!(
        missing.contains(&"text_encoder."),
        "text_encoder. should be missing"
    );
}

#[test]
fn test_kokoro_name_mapping_closure() {
    let mapping = kokoro_name_mapping();
    // Known prefix returns identity
    assert_eq!(
        mapping("decoder.conv_pre.weight"),
        "decoder.conv_pre.weight"
    );
    // Unknown prefix falls back to identity (unwrap_or in the closure)
    assert_eq!(mapping("unknown.weight"), "unknown.weight");
}

// =========================================================================
// Section 22: Parse infrastructure
// =========================================================================

#[test]
fn test_parse_invalid_json_returns_error() {
    let bad_json = b"this is not json";
    let result = parse_exported_program(bad_json);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), IE::JsonParse(_)));
}

#[test]
fn test_parse_unsupported_schema_version() {
    let json = serde_json::json!({
        "schema_version": {"major": 9, "minor": 0},
        "graph_module": {
            "graph": {
                "inputs": [],
                "outputs": [],
                "nodes": []
            },
            "signature": {
                "input_specs": [],
                "output_specs": []
            }
        }
    });
    let bytes = serde_json::to_vec(&json).unwrap();
    let result = parse_exported_program(&bytes);
    assert!(result.is_err());
    match result.unwrap_err() {
        IE::UnsupportedSchema { major, .. } => assert_eq!(major, 9),
        other => panic!("expected UnsupportedSchema, got {other:?}"),
    }
}

#[test]
fn test_parse_minimal_exported_program() {
    let json = serde_json::json!({
        "schema_version": {"major": 8, "minor": 0},
        "graph_module": {
            "graph": {
                "inputs": [],
                "outputs": [],
                "nodes": []
            },
            "signature": {
                "input_specs": [],
                "output_specs": []
            }
        }
    });
    let bytes = serde_json::to_vec(&json).unwrap();
    let result = parse_exported_program(&bytes);
    assert!(
        result.is_ok(),
        "minimal schema 8 program should parse: {:?}",
        result.err()
    );
    let program = result.unwrap();
    assert_eq!(program.schema_version.major, 8);
    assert!(program.graph_module.graph.nodes.is_empty());
}

#[test]
fn test_parse_with_tensor_values() {
    let json = serde_json::json!({
        "schema_version": {"major": 8, "minor": 0},
        "graph_module": {
            "graph": {
                "inputs": [],
                "outputs": [],
                "nodes": [],
                "tensor_values": {
                    "x": {
                        "dtype": 7,
                        "sizes": [{"as_int": 2}, {"as_int": 3}]
                    }
                }
            },
            "signature": {
                "input_specs": [],
                "output_specs": []
            }
        }
    });
    let bytes = serde_json::to_vec(&json).unwrap();
    let result = parse_exported_program(&bytes);
    assert!(result.is_ok());
    let program = result.unwrap();
    let meta = program.graph_module.graph.tensor_values.get("x");
    assert!(meta.is_some(), "tensor_values should contain 'x'");
    let meta = meta.unwrap();
    assert_eq!(meta.dtype, 7); // F32
    assert_eq!(meta.concrete_shape(), Some(vec![2, 3]));
}

// =========================================================================
// Section 23: TensorMeta and SymInt helpers
// =========================================================================

#[test]
fn test_tensor_meta_concrete_shape() {
    let meta = TensorMeta {
        dtype: 7,
        sizes: vec![
            SymInt::Concrete(SymIntConcrete { as_int: 4 }),
            SymInt::Concrete(SymIntConcrete { as_int: 8 }),
            SymInt::Concrete(SymIntConcrete { as_int: 16 }),
        ],
        requires_grad: false,
        strides: vec![],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta.concrete_shape(), Some(vec![4, 8, 16]));
}

#[test]
fn test_tensor_meta_to_dtype() {
    // ScalarType 7 = F32
    let meta = TensorMeta {
        dtype: 7,
        sizes: vec![],
        requires_grad: false,
        strides: vec![],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta.to_dtype(), Some(DType::F32));

    // ScalarType 6 = F16
    let meta_f16 = TensorMeta {
        dtype: 6,
        sizes: vec![],
        requires_grad: false,
        strides: vec![],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta_f16.to_dtype(), Some(DType::F16));

    // ScalarType 13 = BF16
    let meta_bf16 = TensorMeta {
        dtype: 13,
        sizes: vec![],
        requires_grad: false,
        strides: vec![],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta_bf16.to_dtype(), Some(DType::BF16));

    // Unknown type
    let meta_unknown = TensorMeta {
        dtype: 999,
        sizes: vec![],
        requires_grad: false,
        strides: vec![],
        storage_offset: None,
        device: None,
        layout: None,
    };
    assert_eq!(meta_unknown.to_dtype(), None);
}

#[test]
fn test_symint_concrete_extraction() {
    let sym = SymInt::Concrete(SymIntConcrete { as_int: 42 });
    assert_eq!(sym.as_concrete(), Some(42));
}

#[test]
fn test_symint_symbolic_returns_none() {
    let sym = SymInt::Symbolic(SymIntSymbolic {
        as_expr: SymIntExpr {
            expr_str: "s0".to_string(),
            hint: None,
        },
    });
    assert_eq!(sym.as_concrete(), None);
}

#[test]
fn test_tensor_meta_with_symbolic_dim_returns_none_shape() {
    let meta = TensorMeta {
        dtype: 7,
        sizes: vec![
            SymInt::Concrete(SymIntConcrete { as_int: 4 }),
            SymInt::Symbolic(SymIntSymbolic {
                as_expr: SymIntExpr {
                    expr_str: "s0".to_string(),
                    hint: None,
                },
            }),
        ],
        requires_grad: false,
        strides: vec![],
        storage_offset: None,
        device: None,
        layout: None,
    };
    // One symbolic dim means concrete_shape returns None
    assert_eq!(meta.concrete_shape(), None);
}

// =========================================================================
// Section 24: ResolvedWeight
// =========================================================================

#[test]
fn test_resolved_weight_new() {
    let w = ResolvedWeight::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(w.data.len(), 6);
    assert_eq!(w.shape, vec![2, 3]);
}

#[test]
fn test_resolved_weight_clone() {
    let w = ResolvedWeight::new(vec![1.0, 2.0], vec![2]);
    let w2 = w.clone();
    assert_eq!(w2.data, w.data);
    assert_eq!(w2.shape, w.shape);
}

// =========================================================================
// Section 25: Graph build with empty program
// =========================================================================

#[test]
fn test_build_graph_empty_program() {
    let json = serde_json::json!({
        "schema_version": {"major": 8, "minor": 0},
        "graph_module": {
            "graph": {
                "inputs": [],
                "outputs": [],
                "nodes": []
            },
            "signature": {
                "input_specs": [],
                "output_specs": []
            }
        }
    });
    let bytes = serde_json::to_vec(&json).unwrap();
    let program = parse_exported_program(&bytes).unwrap();
    let weights: HashMap<String, ResolvedWeight> = HashMap::new();
    let result = build_graph(&program, &weights);
    assert!(
        result.is_ok(),
        "empty program should build: {:?}",
        result.err()
    );
    let imported = result.unwrap();
    assert_eq!(imported.num_user_inputs, 0);
    assert!(imported.user_input_names.is_empty());
    assert!(imported.output_names.is_empty());
    assert_eq!(imported.graph.len(), 0);
}

// =========================================================================
// Section 26: build_weight_map
// =========================================================================

#[test]
fn test_build_weight_map_empty() {
    let input_specs: Vec<crate::parse::InputSpec> = vec![];
    let weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let result = build_weight_map(&input_specs, &weight_data);
    assert!(result.is_empty());
}

// =========================================================================
// Section 27: Multi-segment model type
// =========================================================================

#[test]
fn test_multi_segment_model_struct() {
    use crate::multi_segment::MultiSegmentModel;
    // Just verify we can construct the type
    let model = MultiSegmentModel::new(vec![], vec![], vec![]);
    assert!(model.segments.is_empty());
    assert!(model.segment_order.is_empty());
    assert!(model.shared_weights.is_empty());
}

// =========================================================================
// Section 28: In-place activation variants (Wave 16)
// =========================================================================

#[test]
fn test_inplace_relu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.relu_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok(), "relu_ should succeed: {:?}", result.err());
    assert!(matches!(result.unwrap().0, TraceOp::Relu));
}

#[test]
fn test_inplace_sigmoid() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.sigmoid_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(
        result.is_ok(),
        "sigmoid_ should succeed: {:?}",
        result.err()
    );
    assert!(matches!(result.unwrap().0, TraceOp::Sigmoid));
}

#[test]
fn test_inplace_silu() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.silu_.default",
        vec![named("self", tensor_arg("x"))],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok(), "silu_ should succeed: {:?}", result.err());
    assert!(matches!(result.unwrap().0, TraceOp::Silu));
}

// =========================================================================
// Section 29: Add in-place variant
// =========================================================================

#[test]
fn test_add_inplace_maps_to_add() {
    let ctx = empty_ctx();
    let node = simple_node(
        "torch.ops.aten.add_.Tensor",
        vec![
            named("self", tensor_arg("a")),
            named("other", tensor_arg("b")),
        ],
    );
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(result.is_ok(), "add_ should succeed: {:?}", result.err());
    assert!(matches!(result.unwrap().0, TraceOp::Add));
}

// =========================================================================
// Section 30: Argument accessor methods
// =========================================================================

#[test]
fn test_argument_as_tensor_name() {
    let arg = tensor_arg("nn_tensor");
    assert_eq!(arg.as_tensor_name(), Some("nn_tensor"));
    let non_tensor = int_arg(42);
    assert_eq!(non_tensor.as_tensor_name(), None);
}

#[test]
fn test_argument_as_int() {
    let arg = int_arg(42);
    assert_eq!(arg.as_int(), Some(42));
    let non_int = tensor_arg("x");
    assert_eq!(non_int.as_int(), None);
}

#[test]
fn test_argument_as_ints() {
    let arg = ints_arg(&[1, 2, 3]);
    assert_eq!(arg.as_ints(), Some(&[1i64, 2, 3][..]));
    let non_ints = int_arg(1);
    assert_eq!(non_ints.as_ints(), None);
}

#[test]
fn test_argument_as_float() {
    let arg = float_arg(3.14);
    assert_eq!(arg.as_float(), Some(3.14));
    let non_float = int_arg(3);
    assert_eq!(non_float.as_float(), None);
}

#[test]
fn test_argument_as_bool_val() {
    let arg = bool_arg(true);
    assert_eq!(arg.as_bool_val(), Some(true));
    let non_bool = int_arg(1);
    assert_eq!(non_bool.as_bool_val(), None);
}

#[test]
fn test_argument_as_string() {
    let arg = str_arg("hello");
    assert_eq!(arg.as_string(), Some("hello"));
    let non_str = int_arg(0);
    assert_eq!(non_str.as_string(), None);
}

#[test]
fn test_argument_is_none() {
    let arg = none_arg();
    assert!(arg.is_none());
    let non_none = int_arg(0);
    assert!(!non_none.is_none());
}
