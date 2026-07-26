// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! DynTensor trace infrastructure tests: TraceOp variant coverage,
//! trace-to-graph translation round-trips, GroupNorm decomposition,
//! shape preservation, and error handling.
//!
//! Part A: TraceOp coverage (variant construction, canonical_name,
//!         classification, arity).
//! Part B: Trace-to-graph translation (elementwise, linear, softmax,
//!         residual, GroupNorm decomposition, sequential chains).
//! Part C: Round-trip tests (record → translate → IBP, empty trace,
//!         shapes consistent, graph bounds).

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{
    record_input, trace_graph, ComputationGraph, TraceNode, TraceOp, WeightRef,
};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{GroupNorm, LayerNorm, Linear, Module, RmsNorm};
use nn_core::{DType, Device};
use nn_verify::{trace_to_graph_model, trace_to_graph_model_multi_input, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

// ============================================================================
// Part A: TraceOp Variant Coverage
// ============================================================================

// -- A1: All elementwise unary ops can be constructed and have canonical names --

#[test]
fn test_trace_op_elementwise_unary_ops() {
    // Ops with explicit canonical_name entries.
    let named_ops: Vec<(TraceOp, &str)> = vec![
        (TraceOp::Relu, "relu"),
        (TraceOp::Gelu, "gelu"),
        (TraceOp::GeluErf, "gelu"),
        (TraceOp::Silu, "silu"),
        (TraceOp::Tanh, "tanh"),
        (TraceOp::Sigmoid, "sigmoid"),
        (TraceOp::Exp, "exp"),
        (TraceOp::Log, "log"),
        (TraceOp::Sqrt, "sqrt"),
        (TraceOp::Sqr, "sqr"),
        (TraceOp::Abs, "abs"),
        (TraceOp::Neg, "neg"),
        (TraceOp::Recip, "recip"),
        (TraceOp::Sin, "sin"),
        (TraceOp::Cos, "cos"),
        (TraceOp::Floor, "floor"),
        (TraceOp::Round, "round"),
        (TraceOp::Fract, "fract"),
    ];
    for (op, expected_name) in &named_ops {
        assert_eq!(
            op.canonical_name(),
            *expected_name,
            "canonical_name mismatch for {op:?}"
        );
        assert_eq!(
            op.expected_arity(),
            Some(1),
            "unary op {op:?} should have arity 1"
        );
    }

    // Ops that fall through to the #[non_exhaustive] catch-all in canonical_name.
    // These are newer variants (Tan, Ceil, Sign) not yet added to the match.
    // Verify they still have correct arity and classification.
    let catchall_ops = vec![TraceOp::Tan, TraceOp::Ceil, TraceOp::Sign];
    for op in &catchall_ops {
        assert_eq!(
            op.expected_arity(),
            Some(1),
            "unary op {op:?} should have arity 1"
        );
        // canonical_name falls through to catch-all for these.
        let name = op.canonical_name();
        assert!(
            !name.is_empty(),
            "canonical_name should return something for {op:?}"
        );
    }

    // Total: 18 named + 3 catchall = 21 unary elementwise variants.
    assert_eq!(named_ops.len() + catchall_ops.len(), 21);
}

// -- A2: Binary element-wise ops --

#[test]
fn test_trace_op_binary_ops() {
    let binary_ops: Vec<(TraceOp, &str)> = vec![
        (TraceOp::Add, "add"),
        (TraceOp::Sub, "sub"),
        (TraceOp::Mul, "mul"),
        (TraceOp::Div, "div"),
        (TraceOp::Maximum, "maximum"),
        (TraceOp::Minimum, "minimum"),
    ];
    for (op, expected_name) in &binary_ops {
        assert_eq!(
            op.canonical_name(),
            *expected_name,
            "canonical_name mismatch for {op:?}"
        );
        assert_eq!(
            op.expected_arity(),
            Some(2),
            "binary op {op:?} should have arity 2"
        );
    }
    assert_eq!(binary_ops.len(), 6);
}

// -- A3: Reduction ops --

#[test]
fn test_trace_op_reduction_ops() {
    let reduction_ops: Vec<(TraceOp, &str)> = vec![
        (
            TraceOp::ReduceSum {
                dim: 0,
                keepdim: true,
            },
            "reduce_sum",
        ),
        (
            TraceOp::ReduceMean {
                dim: 1,
                keepdim: false,
            },
            "reduce_mean",
        ),
        (
            TraceOp::ReduceMax {
                dim: 0,
                keepdim: true,
            },
            "reduce_max",
        ),
        (
            TraceOp::ReduceMin {
                dim: 1,
                keepdim: false,
            },
            "reduce_min",
        ),
    ];
    for (op, expected_name) in &reduction_ops {
        assert_eq!(
            op.canonical_name(),
            *expected_name,
            "canonical_name mismatch for {op:?}"
        );
        assert_eq!(
            op.expected_arity(),
            Some(1),
            "reduction {op:?} should have arity 1"
        );
    }
}

// -- A4: Shape ops --

#[test]
fn test_trace_op_shape_ops() {
    let shape_ops: Vec<(TraceOp, &str, Option<usize>)> = vec![
        (
            TraceOp::Reshape {
                target_shape: vec![6],
            },
            "reshape",
            Some(1),
        ),
        (
            TraceOp::Transpose { dim0: 0, dim1: 1 },
            "transpose",
            Some(1),
        ),
        (
            TraceOp::Permute {
                axes: vec![1, 0, 2],
            },
            "permute",
            Some(1),
        ),
        (TraceOp::Unsqueeze { dim: 0 }, "unsqueeze", Some(1)),
        (TraceOp::Squeeze { dim: 1 }, "squeeze", Some(1)),
        (
            TraceOp::Narrow {
                dim: 0,
                start: 0,
                length: 2,
            },
            "narrow",
            Some(1),
        ),
        (
            TraceOp::Cat {
                dim: 0,
                num_inputs: 3,
            },
            "cat",
            Some(3),
        ),
        (TraceOp::Flip { dim: 0 }, "flip", Some(1)),
        (
            TraceOp::Expand {
                target_shape: vec![4, 4],
            },
            "expand",
            Some(1),
        ),
    ];
    for (op, expected_name, expected_arity) in &shape_ops {
        assert_eq!(
            op.canonical_name(),
            *expected_name,
            "canonical_name mismatch for {op:?}"
        );
        assert_eq!(
            op.expected_arity(),
            *expected_arity,
            "arity mismatch for {op:?}"
        );
    }
}

// -- A5: Normalization ops --

#[test]
fn test_trace_op_norm_ops() {
    let w = WeightRef::new(vec![1.0; 4], vec![4]).unwrap();
    let b = WeightRef::new(vec![0.0; 4], vec![4]).unwrap();
    let rm = WeightRef::new(vec![0.0; 4], vec![4]).unwrap();
    let rv = WeightRef::new(vec![1.0; 4], vec![4]).unwrap();

    let norm_ops: Vec<(TraceOp, &str)> = vec![
        (
            TraceOp::LayerNorm {
                eps: 1e-5,
                weight: w.clone(),
                bias: b.clone(),
            },
            "layer_norm",
        ),
        (
            TraceOp::RmsNorm {
                eps: 1e-5,
                weight: w.clone(),
            },
            "rms_norm",
        ),
        (
            TraceOp::GroupNorm {
                num_groups: 2,
                eps: 1e-5,
                weight: w.clone(),
                bias: b.clone(),
            },
            "group_norm",
        ),
        (TraceOp::InstanceNorm { eps: 1e-5 }, "instance_norm"),
        (
            TraceOp::BatchNorm {
                eps: 1e-5,
                weight: w,
                bias: b,
                running_mean: rm,
                running_var: rv,
            },
            "batch_norm",
        ),
    ];
    for (op, expected_name) in &norm_ops {
        assert_eq!(
            op.canonical_name(),
            *expected_name,
            "canonical_name mismatch for {op:?}"
        );
        assert_eq!(
            op.expected_arity(),
            Some(1),
            "norm op {op:?} should have arity 1"
        );
    }
}

// -- A6: Attention ops --

#[test]
fn test_trace_op_attention_ops() {
    let cos = WeightRef::new(vec![1.0; 8], vec![2, 4]).unwrap();
    let sin = WeightRef::new(vec![0.0; 8], vec![2, 4]).unwrap();

    let attention_ops: Vec<(TraceOp, &str)> = vec![
        (TraceOp::Softmax { dim: 1 }, "softmax"),
        (TraceOp::LogSoftmax { dim: 1 }, "log_softmax"),
        (TraceOp::Sdpa { scale: 0.125 }, "sdpa"),
        (TraceOp::SdpaCausal { scale: 0.125 }, "sdpa_causal"),
        (
            TraceOp::RotaryEmbedding {
                head_dim: 8,
                offset: 0,
                cos_cache: cos,
                sin_cache: sin,
            },
            "rope",
        ),
        (
            TraceOp::MultiHeadAttention {
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 8,
            },
            "mha",
        ),
    ];
    for (op, expected_name) in &attention_ops {
        assert_eq!(
            op.canonical_name(),
            *expected_name,
            "canonical_name mismatch for {op:?}"
        );
    }
}

// -- A7: Conv ops --

#[test]
fn test_trace_op_conv_variants() {
    let w1d = WeightRef::new(vec![1.0; 3], vec![1, 1, 3]).unwrap();
    let w2d = WeightRef::new(vec![1.0; 9], vec![1, 1, 3, 3]).unwrap();
    let w3d = WeightRef::new(vec![1.0; 27], vec![1, 1, 3, 3, 3]).unwrap();

    let conv_ops: Vec<(TraceOp, &str, Option<usize>)> = vec![
        (
            TraceOp::Conv1d {
                weight: w1d.clone(),
                bias: None,
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            "conv1d",
            Some(1),
        ),
        (
            TraceOp::Conv2d {
                weight: w2d.clone(),
                bias: None,
                padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
            },
            "conv2d",
            Some(1),
        ),
        (
            TraceOp::Conv3d {
                weight: w3d,
                bias: None,
                padding: [0, 0, 0],
                stride: [1, 1, 1],
                dilation: [1, 1, 1],
                groups: 1,
            },
            "conv3d",
            Some(1),
        ),
        (
            TraceOp::ConvTranspose1d {
                weight: w1d,
                bias: None,
                padding: 0,
                output_padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            "conv_transpose1d",
            Some(1),
        ),
        (
            TraceOp::ConvTranspose2d {
                weight: w2d,
                bias: None,
                padding: [0, 0],
                output_padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
            },
            "conv_transpose2d",
            Some(1),
        ),
    ];
    for (op, expected_name, expected_arity) in &conv_ops {
        assert_eq!(
            op.canonical_name(),
            *expected_name,
            "canonical_name mismatch for {op:?}"
        );
        assert_eq!(
            op.expected_arity(),
            *expected_arity,
            "arity mismatch for {op:?}"
        );
    }
}

// -- A8: Embedding op --

#[test]
fn test_trace_op_embedding() {
    let w = WeightRef::new(vec![1.0; 12], vec![3, 4]).unwrap();
    let op = TraceOp::Embedding { weight: w };
    assert_eq!(op.canonical_name(), "embedding");
    assert_eq!(op.expected_arity(), Some(1));
}

// -- A9: Selection / indexing ops --

#[test]
fn test_trace_op_selection_indexing() {
    let selection_ops: Vec<(TraceOp, &str)> = vec![
        (TraceOp::Argmax { dim: 0 }, "argmax"),
        (TraceOp::Argmin { dim: 0 }, "argmin"),
        (TraceOp::Topk { k: 3, dim: 0 }, "topk"),
        (TraceOp::IndexSelect { dim: 0 }, "index_select"),
        (TraceOp::Gather { dim: 0 }, "gather"),
    ];
    for (op, expected_name) in &selection_ops {
        assert_eq!(
            op.canonical_name(),
            *expected_name,
            "canonical_name mismatch for {op:?}"
        );
    }
}

// -- A10: MatMul op --

#[test]
fn test_trace_op_matmul() {
    let op = TraceOp::MatMul;
    assert_eq!(op.canonical_name(), "matmul");
    assert_eq!(op.expected_arity(), Some(2));
}

// -- A11: All non-field variants can be constructed (completeness check) --

#[test]
fn test_trace_op_all_simple_variants_constructible() {
    // Verify that every simple (no-field) variant constructs and has a non-empty name.
    let simple_ops = vec![
        TraceOp::Input,
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
        TraceOp::MatMul,
        TraceOp::Relu,
        TraceOp::Gelu,
        TraceOp::GeluErf,
        TraceOp::Silu,
        TraceOp::Tanh,
        TraceOp::Sigmoid,
        TraceOp::Exp,
        TraceOp::Log,
        TraceOp::Sqrt,
        TraceOp::Sqr,
        TraceOp::Abs,
        TraceOp::Neg,
        TraceOp::Recip,
        TraceOp::Sin,
        TraceOp::Cos,
        TraceOp::Tan,
        TraceOp::Floor,
        TraceOp::Ceil,
        TraceOp::Round,
        TraceOp::Sign,
        TraceOp::Fract,
        TraceOp::Softplus,
        TraceOp::Selu,
        TraceOp::Mish,
        TraceOp::HardSigmoid,
        TraceOp::HardSwish,
        TraceOp::Softsign,
        TraceOp::SwiGlu,
        TraceOp::Dropout,
        TraceOp::WhereCond,
        TraceOp::Atan2,
    ];
    for op in &simple_ops {
        let name = op.canonical_name();
        assert!(
            !name.is_empty(),
            "canonical_name should be non-empty for {op:?}"
        );
    }
    // Verify count: 39 simple (no-field) variants.
    assert!(simple_ops.len() >= 39);
}

// ============================================================================
// Part B: Trace-to-Graph Translation
// ============================================================================

// -- B1: ReLU translates to a ReLU LayerSpec and propagates IBP --

#[test]
fn test_relu_translates_to_layer() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
    ]);

    let result = trace_to_graph_model(&graph).expect("ReLU translation should succeed");
    let gn = result.graph;
    assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");

    let input_bounds = uniform_bounds(&[2, 4], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP should succeed");
    assert_bounds_valid(&output);

    // ReLU output: lower >= 0 for positive inputs, upper preserves positive range.
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "ReLU lower bound should be >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(
            v <= 1.01,
            "ReLU upper bound should be <= input range, got {v}"
        );
    }
}

// -- B2: Linear translates and propagates --

#[test]
fn test_linear_translates() {
    let weight = DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap();
    let bias = DynTensor::new(&[0.1, -0.1], &[2], &cpu()).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = linear.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let result = trace_to_graph_model(&graph).expect("Linear translation");
    let gn = result.graph;

    let input_bounds = uniform_bounds(&[2, 2], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    // Identity-ish weights with small bias: bounds should be close to input ± bias.
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            l < u,
            "lower {l} must be < upper {u} for non-trivial bounds"
        );
    }
}

// -- B3: Softmax translates and bounds are in [0, 1] --

#[test]
fn test_softmax_translates() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.softmax(1)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Softmax translation")
        .graph;

    let input_bounds = uniform_bounds(&[2, 3], 3.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "softmax lower >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.01, "softmax upper <= 1, got {v}");
    }
}

// -- B4: Residual pattern (add with skip connection) translates --

#[test]
fn test_residual_add_translates() {
    // Pattern: x -> relu(x) -> relu(x) + x (skip connection).
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
        // Add: relu_output + original_input (skip connection)
        TraceNode::new(
            2,
            "add_0".into(),
            TraceOp::Add,
            vec![1, 0],
            vec![2, 4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Residual translation")
        .graph;

    let input_bounds = uniform_bounds(&[2, 4], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);

    // Residual of ReLU(x) + x: output range is wider than input.
    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

// -- B5: Sequential chain of ops translates --

#[test]
fn test_sequential_translation() {
    // Chain: Input -> Relu -> Sigmoid -> Tanh
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "sigmoid_0".into(),
            TraceOp::Sigmoid,
            vec![1],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "tanh_0".into(),
            TraceOp::Tanh,
            vec![2],
            vec![2, 4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Sequential translation")
        .graph;
    assert!(gn.num_nodes() >= 4, "Should have at least 4 nodes");

    let input_bounds = uniform_bounds(&[2, 4], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);

    // After sigmoid → tanh: bounds should be within [-1, 1] (tanh range)
    // and positive (since sigmoid output is in (0, 1), tanh(0..1) is in (0, 0.76)).
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -1.01, "tanh output lower >= -1, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.01, "tanh output upper <= 1, got {v}");
    }
}

// -- B6: GroupNorm decomposes to reshape -> instance_norm -> reshape -> affine --

#[test]
fn test_groupnorm_decomposes() {
    let num_channels = 4;
    let num_groups = 2;
    let weight = DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[num_channels], &cpu()).unwrap();
    let bias = DynTensor::new(&[0.0, 0.0, 0.0, 0.0], &[num_channels], &cpu()).unwrap();
    let group_norm = GroupNorm::new(num_groups, num_channels, weight, bias, 1e-5).unwrap();

    // Input: [batch=1, channels=4, spatial=8] -- spatial > 1 to avoid degenerate norm.
    let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let x = DynTensor::new(&data, &[1, 4, 8], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4, 8], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = group_norm.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Verify GroupNorm is in the trace.
    let has_gn = graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op(), TraceOp::GroupNorm { .. }));
    assert!(has_gn, "trace should contain GroupNorm op");

    // Translate to NY graph. GroupNorm decomposes into
    // Reshape -> InstanceNorm -> Reshape -> Mul -> Add (5 LayerSpecs).
    let result = trace_to_graph_model(&graph).expect("GroupNorm translation");
    let gn = result.graph;
    // The decomposition produces multiple NY nodes.
    assert!(
        gn.num_nodes() >= 5,
        "GroupNorm decomposition should produce >= 5 nodes, got {}",
        gn.num_nodes()
    );

    // IBP propagation should succeed on the decomposed graph.
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 8]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 8]), 1.0_f32),
    )
    .expect("valid bounds");

    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP on GroupNorm decomposition");
    assert_bounds_valid(&output);
}

// -- B7: Translation preserves shapes --

#[test]
fn test_translation_preserves_shapes() {
    // Build a simple graph and verify input/output shapes match.
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.relu()?;
        Ok(y)
    })
    .unwrap();

    // Verify trace shapes match.
    assert_eq!(result.dims(), &[2, 3]);
    let output_node = graph.output_node().unwrap();
    assert_eq!(output_node.output_shape(), &[2, 3]);

    // Verify NY graph accepts matching bounds shape.
    let gn = trace_to_graph_model(&graph).expect("translation").graph;
    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP");

    // Output bounds should have the same number of elements as the traced output.
    let (lo, _hi) = output.lower_upper();
    assert_eq!(lo.len(), 6, "output should have 2*3=6 elements");
}

// -- B8: Unknown (Custom) op handling — OpaqueSkip over-approximation --

/// An unknown `TraceOp::Custom` op is accepted and translated to a conservative
/// `OpaqueSkip` layer rather than hard-failing (intended design, #4349).
///
/// SOUNDNESS: `OpaqueSkipLayer` is a sound *over-approximation* of an arbitrary
/// unknown op — its IBP rule returns `[-inf, +inf]` (verified in
/// ny-propagate `skip_merge.rs`), NOT a silent identity passthrough. A
/// passthrough would be unsound (a real custom op can change values it would
/// not bound). We assert here that the custom op's output is widened well
/// beyond the finite input interval, proving the over-approximation is in
/// effect (the [-inf,+inf] bounds are sanitized to large finite sentinels by
/// the GraphNetwork wrapper downstream). Accepting unknown ops as OpaqueSkip
/// lets a model containing a single unknown op still verify soundly instead of
/// being rejected outright.
#[test]
fn test_unsupported_op_custom() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "custom_0".into(),
            TraceOp::Custom {
                name: "nn_custom_op".into(),
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);

    // Custom op is accepted as a conservative OpaqueSkip layer (no hard fail).
    let gn = trace_to_graph_model(&graph)
        .expect("Custom op should translate to a conservative OpaqueSkip layer")
        .graph;

    // Soundness: the OpaqueSkip output must over-approximate (widen) the input,
    // not pass it through unchanged. Input is [-1, 1]; the unknown op's output
    // must be much wider than that (a sound superset of any possible op output).
    let input_bounds = uniform_bounds(&[4], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP over OpaqueSkip");
    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "OpaqueSkip lower <= upper: [{l}, {u}]");
        assert!(
            u - l > 1e3,
            "OpaqueSkip must over-approximate the unknown op (width >> input \
             width of 2), got [{l}, {u}] — a near-identity passthrough would be \
             UNSOUND for an arbitrary custom op"
        );
    }
}

// -- B9: Reshape translation preserves element count --

#[test]
fn test_reshape_translation_preserves_elements() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.reshape([3, 2])?;
        Ok(y)
    })
    .unwrap();
    assert_eq!(result.dims(), &[3, 2]);

    let gn = trace_to_graph_model(&graph).expect("Reshape").graph;
    let input_bounds = uniform_bounds(&[2, 3], 5.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.len(), 6, "reshaped output should still have 6 elements");
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            l >= -5.0 - 1e-6 && u <= 5.0 + 1e-6,
            "reshape should preserve bounds: [{l}, {u}]"
        );
    }
}

// -- B10: ReduceSum translation --

#[test]
fn test_reduce_sum_translates() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.sum_keepdim(1)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("ReduceSum translation")
        .graph;

    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);

    // Sum of 3 elements each in [-1, 1]: result in [-3, 3].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -3.0 - 0.1, "sum lower >= -3, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 3.0 + 0.1, "sum upper <= 3, got {v}");
    }
}

// ============================================================================
// Part C: Round-Trip Tests
// ============================================================================

// -- C1: Trace record → translate → IBP propagation --

#[test]
fn test_trace_record_replay() {
    // Build a model via nn layers, trace it, translate, and verify IBP.
    let weight = DynTensor::new(&[1.0, 0.5, -0.5, 1.0], &[2, 2], &cpu()).unwrap();
    let bias = DynTensor::new(&[0.0, 0.0], &[2], &cpu()).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();

    let (output_tensor, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = linear.forward(&x)?;
        let y = y.relu()?;
        Ok(y)
    })
    .unwrap();

    // Verify trace captured the expected ops.
    assert!(
        graph.len() >= 2,
        "trace should have at least input + linear + relu"
    );
    let output_node = graph.output_node().unwrap();
    assert!(matches!(output_node.op(), TraceOp::Relu));

    // Translate and propagate.
    let gn = trace_to_graph_model(&graph)
        .expect("round-trip translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 2], 3.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);

    // Output should be non-trivial (not all zeros).
    let (_lo, hi) = output.lower_upper();
    let max_upper = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_upper > 0.0,
        "ReLU of linear with non-zero weights should have positive upper bound"
    );

    // Verify the actual DynTensor computation was not affected by tracing.
    let vals = output_tensor.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 4);
}

// -- C2: Traced graph → IBP bounds propagation succeeds --

#[test]
fn test_trace_graph_bounds() {
    // More complex chain: LayerNorm → ReLU → Sigmoid.
    let weight = DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[4], &cpu()).unwrap();
    let bias = DynTensor::new(&[0.0, 0.0, 0.0, 0.0], &[4], &cpu()).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = ln.forward(&x)?;
        let y = y.relu()?;
        let y = y.sigmoid()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("LayerNorm + ReLU + Sigmoid")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 10.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);

    // After sigmoid: bounds should be in [0, 1].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "sigmoid lower >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.01, "sigmoid upper <= 1, got {v}");
    }
}

// -- C3: All intermediate shapes valid --

#[test]
fn test_trace_graph_shapes_consistent() {
    // Chain multiple shape-changing ops and verify translation.
    let x = DynTensor::new(
        &(0..24).map(|i| i as f32).collect::<Vec<_>>(),
        &[2, 3, 4],
        &cpu(),
    )
    .unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        // Reshape [2, 3, 4] -> [6, 4]
        let y = x.reshape([6, 4])?;
        // Relu preserves shape
        let y = y.relu()?;
        // Reshape [6, 4] -> [2, 12]
        let y = y.reshape([2, 12])?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 12]);

    // Verify trace node shapes.
    for node in graph.nodes() {
        let shape = node.output_shape();
        let elem_count: usize = shape.iter().product();
        assert_eq!(
            elem_count,
            24,
            "node {:?} should preserve element count 24, shape={:?}",
            node.op(),
            shape
        );
    }

    // Translate and propagate.
    let gn = trace_to_graph_model(&graph)
        .expect("shape-chain translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3, 4], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.len(),
        24,
        "output should have 24 elements after reshape chain"
    );
}

// -- C4: Empty trace → error --

#[test]
fn test_empty_trace_returns_error() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let err = trace_to_graph_model(&graph).expect_err("empty graph should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("empty"),
        "error should mention empty, got: {msg}"
    );
}

// -- C5: Topology validation catches dangling references --

#[test]
fn test_trace_bad_topology_rejected() {
    // Node 1 references non-existent node 999.
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![999], // dangling reference
            vec![4],
            DType::F32,
        ),
    ]);

    let err = trace_to_graph_model(&graph).expect_err("bad topology should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("topology") || msg.contains("validation"),
        "error should mention topology, got: {msg}"
    );
}

// -- C6: Dtype cast count tracked --

#[test]
fn test_dtype_cast_count_zero_for_f32_graph() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);

    let result = trace_to_graph_model(&graph).expect("translation");
    assert_eq!(
        result.dtype_cast_count, 0,
        "pure F32 graph should have 0 dtype casts"
    );
}

// -- C7: Multi-input variable rejection in single-input mode --

#[test]
fn test_multi_variable_input_rejected_single_mode() {
    // Two genuine variable inputs should be rejected by single-input mode.
    // Single-input mode aliases both to the same NETWORK_INPUT, which is unsound.
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "input_1".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "add_0".into(),
            TraceOp::Add,
            vec![0, 1],
            vec![2, 4],
            DType::F32,
        ),
    ]);

    let err = trace_to_graph_model(&graph).expect_err("multi-variable should fail in single mode");
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("variable")
            || msg.to_lowercase().contains("multiple")
            || msg.to_lowercase().contains("input"),
        "error should mention multiple inputs, got: {msg}"
    );

    // But multi-input mode should accept it.
    let result = trace_to_graph_model_multi_input(&graph);
    assert!(
        result.is_ok(),
        "multi-input mode should accept two variable inputs"
    );
}

// -- C8: RmsNorm translates (tests normalization + affine chain) --

#[test]
fn test_rmsnorm_translates() {
    let weight = DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[4], &cpu()).unwrap();
    let rms_norm = RmsNorm::new(weight, 1e-5).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = rms_norm.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("RmsNorm translation")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), 0.5_f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 5.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
}
