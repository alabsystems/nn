// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Totality test for the nn-verify → ny-trace-bridge schema converter.
//!
//! `nn_verify::to_schema` must be **total** over `nn-core`'s trace IR: every
//! `TraceOp` variant (all 123), every support enum, and `SegmentedGraph`
//! convert without panicking, mapping 1:1 onto the NY schema mirror.
//!
//! Guard structure:
//!
//! * [`all_nn_trace_ops`] holds one witness per `nn-core` variant, in
//!   declaration order, with *distinctive* payload values (no defaults) so a
//!   transposed or dropped field cannot cancel out.
//! * [`schema_discriminant`] is a **wildcard-free** match over the schema
//!   `TraceOp` (which, unlike the `#[non_exhaustive]` source enum, can be
//!   matched exhaustively cross-crate). When the NY schema mirrors a new
//!   op, this file stops compiling until the converter maps it — the
//!   build-breaking half of the mirror-sync guard. The runtime half is the
//!   converter's fail-closed panic arms for unmirrored source variants.
//! * Per-family tests pin tricky payloads (conv attrs, norm attrs, LSTM node
//!   IDs, Kokoro fused blocks, compare ops, dtype table, placeholders,
//!   segment boundaries) by exact `assert_eq!` against a hand-built expected
//!   schema op.
//!
//! The whole file is gated on `feature = "ny"` (which carries the
//! ny-trace-bridge dependency); with the feature off it compiles to nothing.
#![cfg(feature = "ny")]

use nn_core::dyn_tensor::trace::{
    record_input, trace_graph, ComputationGraph, KokoroFusedOp, ResBlockActivation,
    TraceActivation, TraceNode, TraceOp, TraceUpsampleMode, WeightRef,
};
use nn_core::dyn_tensor::{CompareOp, DynTensor, GridSamplePaddingMode};
use nn_core::layers::{Linear, Module};
use nn_core::{DType, Device};
use nn_verify::{segmented_to_schema, to_schema};
use ny_trace_bridge::schema;

/// Shorthand: a real (non-placeholder) `nn-core` weight.
fn wr(data: &[f32], shape: &[usize]) -> WeightRef {
    WeightRef::new(data.to_vec(), shape.to_vec()).expect("witness weight data matches shape")
}

/// Shorthand: the schema payload [`wr`] must map to.
fn wp(data: &[f32], shape: &[usize]) -> schema::WeightPayload {
    schema::WeightPayload::f32(data.to_vec(), shape.to_vec())
}

/// One witness value per `nn-core` `TraceOp` variant, in declaration order
/// (so index == [`schema_discriminant`] of the converted op). Payloads use
/// distinctive values on purpose — a field swap must show, not cancel.
fn all_nn_trace_ops() -> Vec<TraceOp> {
    let w = || wr(&[1.5, -2.5], &[2]);
    let ow = || Some(wr(&[0.25], &[1]));
    vec![
        TraceOp::Input,
        TraceOp::ConstantWeight { weight: w() },
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
        TraceOp::ReduceSum {
            dim: 2,
            keepdim: true,
        },
        TraceOp::ReduceMean {
            dim: 3,
            keepdim: false,
        },
        TraceOp::ReduceMax {
            dim: 1,
            keepdim: true,
        },
        TraceOp::ReduceMin {
            dim: 4,
            keepdim: false,
        },
        TraceOp::Reshape {
            target_shape: vec![2, 3, 5],
        },
        TraceOp::Transpose { dim0: 1, dim1: 3 },
        TraceOp::Narrow {
            dim: 2,
            start: 5,
            length: 7,
        },
        TraceOp::Unsqueeze { dim: 3 },
        TraceOp::Squeeze { dim: 2 },
        TraceOp::Permute {
            axes: vec![2, 0, 1],
        },
        TraceOp::Cat {
            dim: 1,
            num_inputs: 3,
        },
        TraceOp::LayerNorm {
            eps: 1e-5,
            weight: w(),
            bias: ow().unwrap(),
        },
        TraceOp::RmsNorm {
            eps: 1e-6,
            weight: w(),
        },
        TraceOp::GroupNorm {
            num_groups: 4,
            eps: 1e-4,
            weight: w(),
            bias: ow().unwrap(),
        },
        TraceOp::InstanceNorm { eps: 1e-3 },
        TraceOp::BatchNorm {
            eps: 1e-5,
            weight: w(),
            bias: ow().unwrap(),
            running_mean: wr(&[0.5], &[1]),
            running_var: wr(&[1.25], &[1]),
        },
        TraceOp::Linear {
            weight: w(),
            bias: ow(),
        },
        TraceOp::Conv1d {
            weight: w(),
            bias: ow(),
            padding: 1,
            stride: 2,
            dilation: 3,
            groups: 4,
        },
        TraceOp::Conv2d {
            weight: w(),
            bias: ow(),
            padding: [1, 2],
            stride: [3, 4],
            dilation: [5, 6],
            groups: 7,
        },
        TraceOp::Conv3d {
            weight: w(),
            bias: ow(),
            padding: [1, 2, 3],
            stride: [4, 5, 6],
            dilation: [7, 8, 9],
            groups: 2,
        },
        TraceOp::ConvTranspose1d {
            weight: w(),
            bias: ow(),
            padding: 1,
            output_padding: 2,
            stride: 3,
            dilation: 4,
            groups: 5,
        },
        TraceOp::ConvTranspose2d {
            weight: w(),
            bias: ow(),
            padding: [1, 2],
            output_padding: [3, 4],
            stride: [5, 6],
            dilation: [7, 8],
            groups: 9,
        },
        TraceOp::Softmax { dim: 2 },
        TraceOp::LogSoftmax { dim: 3 },
        TraceOp::Sdpa { scale: 0.125 },
        TraceOp::SdpaCausal { scale: 0.25 },
        TraceOp::RotaryEmbedding {
            head_dim: 8,
            offset: 16,
            cos_cache: w(),
            sin_cache: wr(&[0.75, -0.75], &[2]),
        },
        TraceOp::MultiHeadAttention {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
        },
        TraceOp::Embedding { weight: w() },
        TraceOp::Lstm {
            weight_ih: w(),
            weight_hh: wr(&[0.5, 0.5], &[2]),
            bias_ih: ow(),
            bias_hh: None,
            hidden_size: 32,
            initial_hidden: Some(41),
            initial_cell: None,
        },
        TraceOp::MaxPool1d {
            kernel_size: 3,
            stride: 2,
            padding: 1,
        },
        TraceOp::AvgPool2d {
            kernel_size: [2, 3],
            stride: [4, 5],
            padding: [0, 1],
        },
        TraceOp::MaxPool2d {
            kernel_size: [3, 2],
            stride: [1, 2],
            padding: [1, 0],
        },
        TraceOp::AdaptiveAvgPool2d {
            output_size: [7, 5],
        },
        TraceOp::AvgPool1d {
            kernel_size: 4,
            stride: 3,
            padding: 2,
        },
        TraceOp::AdaptiveAvgPool1d { output_size: 9 },
        TraceOp::AdaptiveMaxPool2d {
            output_size: [3, 4],
        },
        TraceOp::Activation {
            kind: TraceActivation::GeluErf,
        },
        TraceOp::Elu { alpha: 0.9 },
        TraceOp::LeakyRelu { slope: 0.2 },
        TraceOp::Softplus,
        TraceOp::Selu,
        TraceOp::Celu { alpha: 1.1 },
        TraceOp::Mish,
        TraceOp::HardSigmoid,
        TraceOp::HardSwish,
        TraceOp::Softsign,
        TraceOp::PRelu { slope: w() },
        TraceOp::KokoroFused(KokoroFusedOp::SnakeTensor { alpha: w() }),
        TraceOp::SwiGlu,
        TraceOp::Dropout,
        TraceOp::PixelShuffle { upscale_factor: 3 },
        TraceOp::PixelUnshuffle {
            downscale_factor: 2,
        },
        TraceOp::Upsample1d { factor: 4 },
        TraceOp::Upsample2d {
            mode: TraceUpsampleMode::Bicubic,
            scale_h: 1.5,
            scale_w: 2.5,
        },
        TraceOp::ResizeBilinear {
            target_h: 24,
            target_w: 32,
        },
        TraceOp::Triu { diagonal: -1 },
        TraceOp::Tril { diagonal: 2 },
        TraceOp::GridSample {
            padding_mode: GridSamplePaddingMode::Border,
            align_corners: true,
        },
        TraceOp::QLinear {
            weight: w(),
            bias: ow(),
        },
        TraceOp::Topk { k: 5, dim: 1 },
        TraceOp::Argmax { dim: 2 },
        TraceOp::Argmin { dim: 3 },
        TraceOp::ArgSort {
            dim: 1,
            descending: true,
        },
        TraceOp::Sort {
            dim: 2,
            descending: false,
        },
        TraceOp::IndexSelect { dim: 1 },
        TraceOp::Gather { dim: 2 },
        TraceOp::WhereCond,
        TraceOp::Expand {
            target_shape: vec![4, 5, 6],
        },
        TraceOp::Compare {
            op: CompareOp::Gt,
            value: 0.5,
        },
        TraceOp::CompareTensor { op: CompareOp::Le },
        TraceOp::Cumsum { dim: 1 },
        TraceOp::RepeatInterleave { dim: 2 },
        TraceOp::Powf { exponent: 2.5 },
        TraceOp::ToDtype {
            target_dtype: DType::BF16,
        },
        TraceOp::Flip { dim: 3 },
        TraceOp::Roll {
            shifts: vec![2, -3],
            dims: vec![0, 1],
        },
        TraceOp::Unfold {
            dim: 1,
            size: 400,
            step: 160,
        },
        TraceOp::SliceSet { dim: 2, start: 8 },
        TraceOp::Scatter { dim: 1 },
        TraceOp::ScatterAdd { dim: 2 },
        TraceOp::IndexAdd { dim: 3 },
        TraceOp::IndexPut { dim: 1 },
        TraceOp::Clamp {
            min: Some(-1.5),
            max: None,
        },
        TraceOp::Constant { value: 3.25 },
        TraceOp::ReflectionPad1d {
            pad_left: 2,
            pad_right: 3,
        },
        TraceOp::ReflectionPad2d {
            pad_left: 1,
            pad_right: 2,
            pad_top: 3,
            pad_bottom: 4,
        },
        TraceOp::ConstantPadNd {
            padding: vec![1, 2, 3, 4],
            value: -0.5,
        },
        TraceOp::Atan2,
        TraceOp::Arange {
            start: 1.0,
            end: 9.0,
            step: 0.5,
        },
        TraceOp::SegmentBoundary {
            reason: "length_regulate".into(),
            input_bounds: Some((-2.5, 4.5)),
        },
        TraceOp::MoeGating {
            num_experts: 128,
            top_k: 4,
        },
        TraceOp::Custom {
            name: "external_op".into(),
        },
    ]
}

/// Exhaustive, **wildcard-free** discriminant over the schema `TraceOp`.
///
/// This is the compile-time mirror-sync guard: the schema enum is not
/// `#[non_exhaustive]`, so this match must name every variant. When the NY
/// schema grows an op (mirroring a new `nn-core` variant), this stops
/// compiling until the converter — and this test — cover it.
fn schema_discriminant(op: &schema::TraceOp) -> usize {
    match op {
        schema::TraceOp::Input => 0,
        schema::TraceOp::ConstantWeight { .. } => 1,
        schema::TraceOp::Add => 2,
        schema::TraceOp::Sub => 3,
        schema::TraceOp::Mul => 4,
        schema::TraceOp::Div => 5,
        schema::TraceOp::Maximum => 6,
        schema::TraceOp::Minimum => 7,
        schema::TraceOp::MatMul => 8,
        schema::TraceOp::Relu => 9,
        schema::TraceOp::Gelu => 10,
        schema::TraceOp::GeluErf => 11,
        schema::TraceOp::Silu => 12,
        schema::TraceOp::Tanh => 13,
        schema::TraceOp::Sigmoid => 14,
        schema::TraceOp::Exp => 15,
        schema::TraceOp::Log => 16,
        schema::TraceOp::Sqrt => 17,
        schema::TraceOp::Sqr => 18,
        schema::TraceOp::Abs => 19,
        schema::TraceOp::Neg => 20,
        schema::TraceOp::Recip => 21,
        schema::TraceOp::Sin => 22,
        schema::TraceOp::Cos => 23,
        schema::TraceOp::Tan => 24,
        schema::TraceOp::Floor => 25,
        schema::TraceOp::Ceil => 26,
        schema::TraceOp::Round => 27,
        schema::TraceOp::Sign => 28,
        schema::TraceOp::Fract => 29,
        schema::TraceOp::ReduceSum { .. } => 30,
        schema::TraceOp::ReduceMean { .. } => 31,
        schema::TraceOp::ReduceMax { .. } => 32,
        schema::TraceOp::ReduceMin { .. } => 33,
        schema::TraceOp::Reshape { .. } => 34,
        schema::TraceOp::Transpose { .. } => 35,
        schema::TraceOp::Narrow { .. } => 36,
        schema::TraceOp::Unsqueeze { .. } => 37,
        schema::TraceOp::Squeeze { .. } => 38,
        schema::TraceOp::Permute { .. } => 39,
        schema::TraceOp::Cat { .. } => 40,
        schema::TraceOp::LayerNorm { .. } => 41,
        schema::TraceOp::RmsNorm { .. } => 42,
        schema::TraceOp::GroupNorm { .. } => 43,
        schema::TraceOp::InstanceNorm { .. } => 44,
        schema::TraceOp::BatchNorm { .. } => 45,
        schema::TraceOp::Linear { .. } => 46,
        schema::TraceOp::Conv1d { .. } => 47,
        schema::TraceOp::Conv2d { .. } => 48,
        schema::TraceOp::Conv3d { .. } => 49,
        schema::TraceOp::ConvTranspose1d { .. } => 50,
        schema::TraceOp::ConvTranspose2d { .. } => 51,
        schema::TraceOp::Softmax { .. } => 52,
        schema::TraceOp::LogSoftmax { .. } => 53,
        schema::TraceOp::Sdpa { .. } => 54,
        schema::TraceOp::SdpaCausal { .. } => 55,
        schema::TraceOp::RotaryEmbedding { .. } => 56,
        schema::TraceOp::MultiHeadAttention { .. } => 57,
        schema::TraceOp::Embedding { .. } => 58,
        schema::TraceOp::Lstm { .. } => 59,
        schema::TraceOp::MaxPool1d { .. } => 60,
        schema::TraceOp::AvgPool2d { .. } => 61,
        schema::TraceOp::MaxPool2d { .. } => 62,
        schema::TraceOp::AdaptiveAvgPool2d { .. } => 63,
        schema::TraceOp::AvgPool1d { .. } => 64,
        schema::TraceOp::AdaptiveAvgPool1d { .. } => 65,
        schema::TraceOp::AdaptiveMaxPool2d { .. } => 66,
        schema::TraceOp::Activation { .. } => 67,
        schema::TraceOp::Elu { .. } => 68,
        schema::TraceOp::LeakyRelu { .. } => 69,
        schema::TraceOp::Softplus => 70,
        schema::TraceOp::Selu => 71,
        schema::TraceOp::Celu { .. } => 72,
        schema::TraceOp::Mish => 73,
        schema::TraceOp::HardSigmoid => 74,
        schema::TraceOp::HardSwish => 75,
        schema::TraceOp::Softsign => 76,
        schema::TraceOp::PRelu { .. } => 77,
        schema::TraceOp::KokoroFused(_) => 78,
        schema::TraceOp::SwiGlu => 79,
        schema::TraceOp::Dropout => 80,
        schema::TraceOp::PixelShuffle { .. } => 81,
        schema::TraceOp::PixelUnshuffle { .. } => 82,
        schema::TraceOp::Upsample1d { .. } => 83,
        schema::TraceOp::Upsample2d { .. } => 84,
        schema::TraceOp::ResizeBilinear { .. } => 85,
        schema::TraceOp::Triu { .. } => 86,
        schema::TraceOp::Tril { .. } => 87,
        schema::TraceOp::GridSample { .. } => 88,
        schema::TraceOp::QLinear { .. } => 89,
        schema::TraceOp::Topk { .. } => 90,
        schema::TraceOp::Argmax { .. } => 91,
        schema::TraceOp::Argmin { .. } => 92,
        schema::TraceOp::ArgSort { .. } => 93,
        schema::TraceOp::Sort { .. } => 94,
        schema::TraceOp::IndexSelect { .. } => 95,
        schema::TraceOp::Gather { .. } => 96,
        schema::TraceOp::WhereCond => 97,
        schema::TraceOp::Expand { .. } => 98,
        schema::TraceOp::Compare { .. } => 99,
        schema::TraceOp::CompareTensor { .. } => 100,
        schema::TraceOp::Cumsum { .. } => 101,
        schema::TraceOp::RepeatInterleave { .. } => 102,
        schema::TraceOp::Powf { .. } => 103,
        schema::TraceOp::ToDtype { .. } => 104,
        schema::TraceOp::Flip { .. } => 105,
        schema::TraceOp::Roll { .. } => 106,
        schema::TraceOp::Unfold { .. } => 107,
        schema::TraceOp::SliceSet { .. } => 108,
        schema::TraceOp::Scatter { .. } => 109,
        schema::TraceOp::ScatterAdd { .. } => 110,
        schema::TraceOp::IndexAdd { .. } => 111,
        schema::TraceOp::IndexPut { .. } => 112,
        schema::TraceOp::Clamp { .. } => 113,
        schema::TraceOp::Constant { .. } => 114,
        schema::TraceOp::ReflectionPad1d { .. } => 115,
        schema::TraceOp::ReflectionPad2d { .. } => 116,
        schema::TraceOp::ConstantPadNd { .. } => 117,
        schema::TraceOp::Atan2 => 118,
        schema::TraceOp::Arange { .. } => 119,
        schema::TraceOp::SegmentBoundary { .. } => 120,
        schema::TraceOp::MoeGating { .. } => 121,
        schema::TraceOp::Custom { .. } => 122,
    }
}

/// Number of `TraceOp` variants; must stay in lock-step with the NY schema
/// mirror's own pin.
const EXPECTED_TRACE_OP_VARIANTS: usize = 123;

/// Build a graph whose node `i` carries witness op `i` (node 0 is the
/// `Input` witness; every later node consumes it), convert, and return both.
fn convert_witness_graph() -> (ComputationGraph, schema::ComputationGraph) {
    let ops = all_nn_trace_ops();
    let input_id = 10;
    let nodes: Vec<TraceNode> = ops
        .into_iter()
        .enumerate()
        .map(|(i, op)| {
            let inputs = if i == 0 { vec![] } else { vec![input_id] };
            TraceNode::new(
                input_id + i as u64,
                format!("op_{i}"),
                op,
                inputs,
                vec![1, i + 1],
                DType::F32,
            )
        })
        .collect();
    let graph = ComputationGraph::from_nodes(nodes);
    let converted = to_schema(&graph);
    (graph, converted)
}

/// Convert a single op through the public `to_schema` surface and return its
/// schema image (nodes: `[Input, op]`).
fn convert_one(op: TraceOp) -> schema::TraceOp {
    let nodes = vec![
        TraceNode::new(
            1,
            "x".into(),
            TraceOp::Input,
            vec![],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(2, "n".into(), op, vec![1], vec![1, 4], DType::F32),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let mut converted = to_schema(&graph);
    converted.nodes.pop().expect("two-node graph").op
}

/// The core totality assertion: all 123 variants convert (no panic), land on
/// the right schema variant, and node metadata round-trips exactly.
#[test]
fn converter_is_total_over_all_trace_op_variants() {
    let (graph, converted) = convert_witness_graph();
    assert_eq!(
        graph.nodes().len(),
        EXPECTED_TRACE_OP_VARIANTS,
        "witness list must cover every TraceOp variant"
    );
    assert_eq!(converted.nodes.len(), EXPECTED_TRACE_OP_VARIANTS);

    for (i, (src, dst)) in graph.nodes().iter().zip(&converted.nodes).enumerate() {
        // Op lands on the mirror variant at the same declaration position.
        assert_eq!(
            schema_discriminant(&dst.op),
            i,
            "witness {i} ({:?}) mapped to the wrong schema variant {:?}",
            src.op(),
            dst.op
        );
        // Node metadata round-trips exactly.
        assert_eq!(dst.id, schema::NodeId(src.id()), "id of witness {i}");
        assert_eq!(dst.name, src.name(), "name of witness {i}");
        assert_eq!(
            dst.inputs,
            src.inputs()
                .iter()
                .map(|&id| schema::NodeId(id))
                .collect::<Vec<_>>(),
            "inputs of witness {i}"
        );
        assert_eq!(dst.output_shape, src.output_shape(), "shape of witness {i}");
        assert_eq!(dst.output_dtype, schema::DType::F32, "dtype of witness {i}");
    }

    // Output marking survives (from_nodes: last node is the sole output).
    assert_eq!(
        converted.output_nodes,
        vec![schema::NodeId(graph.output_node().expect("non-empty").id())]
    );
    // The converted graph is well-formed by the schema's own validator.
    converted
        .validate_topology()
        .expect("converted witness graph is topologically ordered");
}

/// A recorder-built trace (the same public tracing API real models use)
/// round-trips ids, names, shapes, dtypes, inputs, and output marking.
#[test]
fn recorder_built_trace_round_trips_metadata() {
    let cpu = Device::Cpu;
    let w1 = DynTensor::new(&[1.0, -1.0, 0.5, 0.5], &[2, 2], &cpu).unwrap();
    let b1 = DynTensor::new(&[0.1, -0.2], &[2], &cpu).unwrap();
    let layer = Linear::new(w1, Some(b1)).unwrap();
    let x = DynTensor::new(&[0.5, -0.5], &[1, 2], &cpu).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let h = layer.forward(&x)?;
        h.relu()
    })
    .unwrap();

    let converted = to_schema(&graph);
    assert_eq!(converted.nodes.len(), graph.nodes().len());
    for (src, dst) in graph.nodes().iter().zip(&converted.nodes) {
        assert_eq!(dst.id, schema::NodeId(src.id()));
        assert_eq!(dst.name, src.name());
        assert_eq!(
            dst.inputs,
            src.inputs()
                .iter()
                .map(|&id| schema::NodeId(id))
                .collect::<Vec<_>>()
        );
        assert_eq!(dst.output_shape, src.output_shape());
    }
    let discs: Vec<usize> = converted
        .nodes
        .iter()
        .map(|n| schema_discriminant(&n.op))
        .collect();
    assert_eq!(
        discs,
        vec![0, 46, 9], // Input, Linear, Relu
        "recorder trace must map onto the expected schema ops"
    );
    assert_eq!(
        converted.output_nodes,
        graph
            .output_nodes()
            .iter()
            .map(|n| schema::NodeId(n.id()))
            .collect::<Vec<_>>()
    );
}

/// Conv-family attrs (padding/stride/dilation/groups/output_padding) map
/// field-for-field — a swap of any two attrs fails here.
#[test]
fn conv_attrs_map_field_for_field() {
    let got = convert_one(TraceOp::Conv2d {
        weight: wr(&[1.5, -2.5], &[2]),
        bias: Some(wr(&[0.25], &[1])),
        padding: [1, 2],
        stride: [3, 4],
        dilation: [5, 6],
        groups: 7,
    });
    let want = schema::TraceOp::Conv2d {
        weight: wp(&[1.5, -2.5], &[2]),
        bias: Some(wp(&[0.25], &[1])),
        padding: [1, 2],
        stride: [3, 4],
        dilation: [5, 6],
        groups: 7,
    };
    assert_eq!(got, want);

    let got = convert_one(TraceOp::ConvTranspose2d {
        weight: wr(&[1.5, -2.5], &[2]),
        bias: None,
        padding: [1, 2],
        output_padding: [3, 4],
        stride: [5, 6],
        dilation: [7, 8],
        groups: 9,
    });
    let want = schema::TraceOp::ConvTranspose2d {
        weight: wp(&[1.5, -2.5], &[2]),
        bias: None,
        padding: [1, 2],
        output_padding: [3, 4],
        stride: [5, 6],
        dilation: [7, 8],
        groups: 9,
    };
    assert_eq!(got, want);
}

/// Norm-family payloads (eps, num_groups, running stats) map exactly.
#[test]
fn norm_attrs_map_field_for_field() {
    let got = convert_one(TraceOp::BatchNorm {
        eps: 1e-5,
        weight: wr(&[2.0], &[1]),
        bias: wr(&[-3.0], &[1]),
        running_mean: wr(&[0.5], &[1]),
        running_var: wr(&[1.25], &[1]),
    });
    let want = schema::TraceOp::BatchNorm {
        eps: 1e-5,
        weight: wp(&[2.0], &[1]),
        bias: wp(&[-3.0], &[1]),
        running_mean: wp(&[0.5], &[1]),
        running_var: wp(&[1.25], &[1]),
    };
    assert_eq!(got, want);

    let got = convert_one(TraceOp::GroupNorm {
        num_groups: 4,
        eps: 1e-4,
        weight: wr(&[1.0], &[1]),
        bias: wr(&[0.0], &[1]),
    });
    let want = schema::TraceOp::GroupNorm {
        num_groups: 4,
        eps: 1e-4,
        weight: wp(&[1.0], &[1]),
        bias: wp(&[0.0], &[1]),
    };
    assert_eq!(got, want);
}

/// LSTM: optional biases and the `Option<NodeId>` initial states map exactly
/// (node IDs go through the schema's newtype).
#[test]
fn lstm_payload_maps_node_ids_and_weights() {
    let got = convert_one(TraceOp::Lstm {
        weight_ih: wr(&[1.5, -2.5], &[2]),
        weight_hh: wr(&[0.5, 0.5], &[2]),
        bias_ih: Some(wr(&[0.25], &[1])),
        bias_hh: None,
        hidden_size: 32,
        initial_hidden: Some(41),
        initial_cell: None,
    });
    let want = schema::TraceOp::Lstm {
        weight_ih: wp(&[1.5, -2.5], &[2]),
        weight_hh: wp(&[0.5, 0.5], &[2]),
        bias_ih: Some(wp(&[0.25], &[1])),
        bias_hh: None,
        hidden_size: 32,
        initial_hidden: Some(schema::NodeId(41)),
        initial_cell: None,
    };
    assert_eq!(got, want);
}

/// Every Kokoro fused variant maps field-for-field, including the 8-weight
/// `FusedAdainResBlock` under both activation variants.
#[test]
fn kokoro_fused_payloads_map_field_for_field() {
    let got = convert_one(TraceOp::KokoroFused(KokoroFusedOp::AdainSnake {
        alpha: wr(&[0.9], &[1]),
        eps: 1e-5,
    }));
    let want = schema::TraceOp::KokoroFused(schema::KokoroFusedOp::AdainSnake {
        alpha: wp(&[0.9], &[1]),
        eps: 1e-5,
    });
    assert_eq!(got, want);

    let got = convert_one(TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu {
        eps: 1e-4,
        slope: 0.2,
    }));
    let want = schema::TraceOp::KokoroFused(schema::KokoroFusedOp::AdainLeakyRelu {
        eps: 1e-4,
        slope: 0.2,
    });
    assert_eq!(got, want);

    let got = convert_one(TraceOp::KokoroFused(KokoroFusedOp::AdaLayerNorm {
        norm_weight: wr(&[1.0], &[1]),
        norm_bias: wr(&[0.5], &[1]),
        eps: 1e-6,
    }));
    let want = schema::TraceOp::KokoroFused(schema::KokoroFusedOp::AdaLayerNorm {
        norm_weight: wp(&[1.0], &[1]),
        norm_bias: wp(&[0.5], &[1]),
        eps: 1e-6,
    });
    assert_eq!(got, want);

    // Snake-activation ResBlock: all 8 weights + 2 alphas + attrs.
    let got = convert_one(TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
        activation: ResBlockActivation::Snake {
            alpha1: wr(&[0.1], &[1]),
            alpha2: wr(&[0.2], &[1]),
        },
        adain1_weight: wr(&[1.0], &[1]),
        adain1_bias: wr(&[2.0], &[1]),
        adain2_weight: wr(&[3.0], &[1]),
        adain2_bias: wr(&[4.0], &[1]),
        conv1_weight: wr(&[5.0], &[1]),
        conv1_bias: wr(&[6.0], &[1]),
        conv1_dilation: 3,
        conv1_padding: 2,
        conv2_weight: wr(&[7.0], &[1]),
        conv2_bias: wr(&[8.0], &[1]),
        conv2_padding: 1,
        eps: 1e-5,
        residual_scale: 0.5,
    }));
    let want = schema::TraceOp::KokoroFused(schema::KokoroFusedOp::FusedAdainResBlock {
        activation: schema::ResBlockActivation::Snake {
            alpha1: wp(&[0.1], &[1]),
            alpha2: wp(&[0.2], &[1]),
        },
        adain1_weight: wp(&[1.0], &[1]),
        adain1_bias: wp(&[2.0], &[1]),
        adain2_weight: wp(&[3.0], &[1]),
        adain2_bias: wp(&[4.0], &[1]),
        conv1_weight: wp(&[5.0], &[1]),
        conv1_bias: wp(&[6.0], &[1]),
        conv1_dilation: 3,
        conv1_padding: 2,
        conv2_weight: wp(&[7.0], &[1]),
        conv2_bias: wp(&[8.0], &[1]),
        conv2_padding: 1,
        eps: 1e-5,
        residual_scale: 0.5,
    });
    assert_eq!(got, want);

    // LeakyRelu-activation ResBlock: the other ResBlockActivation variant.
    let got = convert_one(TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
        activation: ResBlockActivation::LeakyRelu { slope: 0.2 },
        adain1_weight: wr(&[1.0], &[1]),
        adain1_bias: wr(&[2.0], &[1]),
        adain2_weight: wr(&[3.0], &[1]),
        adain2_bias: wr(&[4.0], &[1]),
        conv1_weight: wr(&[5.0], &[1]),
        conv1_bias: wr(&[6.0], &[1]),
        conv1_dilation: 1,
        conv1_padding: 1,
        conv2_weight: wr(&[7.0], &[1]),
        conv2_bias: wr(&[8.0], &[1]),
        conv2_padding: 1,
        eps: 1e-5,
        residual_scale: std::f64::consts::FRAC_1_SQRT_2,
    }));
    match got {
        schema::TraceOp::KokoroFused(schema::KokoroFusedOp::FusedAdainResBlock {
            activation: schema::ResBlockActivation::LeakyRelu { slope },
            ..
        }) => assert_eq!(slope, 0.2),
        other => panic!("expected LeakyRelu FusedAdainResBlock, got {other:?}"),
    }
}

/// All six comparison discriminants map by name through both `Compare` and
/// `CompareTensor`.
#[test]
fn compare_ops_map_one_to_one() {
    let pairs = [
        (CompareOp::Eq, schema::CompareOp::Eq),
        (CompareOp::Ne, schema::CompareOp::Ne),
        (CompareOp::Ge, schema::CompareOp::Ge),
        (CompareOp::Gt, schema::CompareOp::Gt),
        (CompareOp::Lt, schema::CompareOp::Lt),
        (CompareOp::Le, schema::CompareOp::Le),
    ];
    for (nn_op, schema_op) in pairs {
        let got = convert_one(TraceOp::Compare {
            op: nn_op,
            value: 0.5,
        });
        assert_eq!(
            got,
            schema::TraceOp::Compare {
                op: schema_op,
                value: 0.5
            }
        );
        let got = convert_one(TraceOp::CompareTensor { op: nn_op });
        assert_eq!(got, schema::TraceOp::CompareTensor { op: schema_op });
    }
}

/// All eleven named activations map by name.
#[test]
fn activation_kinds_map_one_to_one() {
    let pairs = [
        (TraceActivation::Relu, schema::TraceActivation::Relu),
        (TraceActivation::Gelu, schema::TraceActivation::Gelu),
        (TraceActivation::GeluErf, schema::TraceActivation::GeluErf),
        (TraceActivation::Silu, schema::TraceActivation::Silu),
        (TraceActivation::Sigmoid, schema::TraceActivation::Sigmoid),
        (TraceActivation::Tanh, schema::TraceActivation::Tanh),
        (TraceActivation::Exp, schema::TraceActivation::Exp),
        (TraceActivation::Log, schema::TraceActivation::Log),
        (TraceActivation::Elu, schema::TraceActivation::Elu),
        (
            TraceActivation::LeakyRelu,
            schema::TraceActivation::LeakyRelu,
        ),
        (TraceActivation::Mish, schema::TraceActivation::Mish),
    ];
    for (kind, want) in pairs {
        assert_eq!(
            convert_one(TraceOp::Activation { kind }),
            schema::TraceOp::Activation { kind: want }
        );
    }
}

/// Upsample modes and grid-sample padding modes map by name; `align_corners`
/// survives.
#[test]
fn upsample_and_grid_sample_modes_map_one_to_one() {
    let modes = [
        (
            TraceUpsampleMode::Nearest,
            schema::TraceUpsampleMode::Nearest,
        ),
        (
            TraceUpsampleMode::Bilinear,
            schema::TraceUpsampleMode::Bilinear,
        ),
        (
            TraceUpsampleMode::Bicubic,
            schema::TraceUpsampleMode::Bicubic,
        ),
    ];
    for (mode, want) in modes {
        assert_eq!(
            convert_one(TraceOp::Upsample2d {
                mode,
                scale_h: 1.5,
                scale_w: 2.5,
            }),
            schema::TraceOp::Upsample2d {
                mode: want,
                scale_h: 1.5,
                scale_w: 2.5,
            }
        );
    }
    let paddings = [
        (
            GridSamplePaddingMode::Zeros,
            schema::GridSamplePaddingMode::Zeros,
        ),
        (
            GridSamplePaddingMode::Border,
            schema::GridSamplePaddingMode::Border,
        ),
    ];
    for (mode, want) in paddings {
        for align_corners in [false, true] {
            assert_eq!(
                convert_one(TraceOp::GridSample {
                    padding_mode: mode,
                    align_corners,
                }),
                schema::TraceOp::GridSample {
                    padding_mode: want,
                    align_corners,
                }
            );
        }
    }
}

/// The full dtype table maps 1:1 (nn-core `BF16` spells `Bf16` on the wire).
#[test]
fn to_dtype_maps_all_dtypes() {
    let pairs = [
        (DType::F32, schema::DType::F32),
        (DType::F16, schema::DType::F16),
        (DType::BF16, schema::DType::Bf16),
        (DType::F64, schema::DType::F64),
        (DType::I32, schema::DType::I32),
        (DType::I64, schema::DType::I64),
        (DType::U32, schema::DType::U32),
        (DType::U8, schema::DType::U8),
        (DType::Bool, schema::DType::Bool),
    ];
    for (dtype, want) in pairs {
        assert_eq!(
            convert_one(TraceOp::ToDtype {
                target_dtype: dtype
            }),
            schema::TraceOp::ToDtype { target_dtype: want }
        );
    }
}

/// Shape-only weight refs (`WeightRef::from_shape`, the source's capture-
/// failure fallback) become schema placeholders — the "no data" fact must
/// survive the wire so the bridge can refuse them explicitly.
#[test]
fn weight_placeholder_survives_the_wire() {
    let got = convert_one(TraceOp::ConstantWeight {
        weight: WeightRef::from_shape(&[2, 3]),
    });
    match got {
        schema::TraceOp::ConstantWeight { weight } => {
            assert!(weight.is_placeholder());
            assert_eq!(weight.shape, vec![2, 3]);
            assert_eq!(weight.data, schema::WeightData::Placeholder);
        }
        other => panic!("expected ConstantWeight, got {other:?}"),
    }
    // A real weight stays a real f32 payload.
    let got = convert_one(TraceOp::ConstantWeight {
        weight: wr(&[1.5, -2.5], &[2]),
    });
    match got {
        schema::TraceOp::ConstantWeight { weight } => {
            assert!(!weight.is_placeholder());
            assert_eq!(weight, wp(&[1.5, -2.5], &[2]));
        }
        other => panic!("expected ConstantWeight, got {other:?}"),
    }
}

/// `SegmentBoundary` payloads convert, and `segmented_to_schema` agrees with
/// splitting on the schema side: convert-then-split == split-then-convert.
#[test]
fn segmented_graph_split_commutes_with_conversion() {
    let nodes = vec![
        TraceNode::new(
            1,
            "x".into(),
            TraceOp::Input,
            vec![],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "r".into(),
            TraceOp::Relu,
            vec![1],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "boundary".into(),
            TraceOp::SegmentBoundary {
                reason: "length_regulate".into(),
                input_bounds: Some((-2.5, 4.5)),
            },
            vec![2],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(4, "a".into(), TraceOp::Abs, vec![3], vec![1, 4], DType::F32),
    ];
    let graph = ComputationGraph::from_nodes(nodes);

    // The boundary op itself converts with its payload intact.
    let converted = to_schema(&graph);
    assert_eq!(
        converted.nodes[2].op,
        schema::TraceOp::SegmentBoundary {
            reason: "length_regulate".into(),
            input_bounds: Some((-2.5, 4.5)),
        }
    );

    // Split-then-convert equals convert-then-split: nn-core's splitter and
    // the schema's mirrored splitter must agree through the converter.
    let split_then_convert = segmented_to_schema(&graph.split_at_segment_boundaries());
    let convert_then_split = converted.split_at_segment_boundaries();
    assert_eq!(split_then_convert, convert_then_split);

    // And the split metadata itself is what nn-core attached.
    assert_eq!(split_then_convert.segments.len(), 2);
    assert_eq!(
        split_then_convert.segments[0].boundary_reason.as_deref(),
        Some("length_regulate")
    );
    assert_eq!(
        split_then_convert.segments[0].boundary_bounds,
        Some((-2.5, 4.5))
    );
    assert_eq!(split_then_convert.segments[1].boundary_reason, None);
    // The boundary marker node is dropped from both segments.
    let total: usize = split_then_convert
        .segments
        .iter()
        .map(|s| s.graph.nodes.len())
        .sum();
    assert_eq!(total, 3);
}
