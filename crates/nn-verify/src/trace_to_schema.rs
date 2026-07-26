// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Serialize an `nn-core` `ComputationGraph` into the NY-owned
//! `ny_trace_bridge::schema::ComputationGraph` wire format.
//!
//! The converter is **total** over the traced IR: every `TraceOp` variant
//! (all 123 as of the mirror), every support enum (`CompareOp`,
//! `GridSamplePaddingMode`, `TraceActivation`, `TraceUpsampleMode`,
//! `KokoroFusedOp`, `ResBlockActivation`), and [`SegmentedGraph`] map 1:1
//! onto their schema mirrors. Totality is deliberate: the schema is the wire
//! format, so a captured trace must always be *expressible* in it. Whether an
//! op can then be *translated* into an `ny_build::GraphModel` is decided
//! downstream by `ny_trace_bridge::translate`, which refuses unported ops
//! explicitly — sound by construction. Serialization must never be the layer
//! that drops or rewrites an op.
//!
//! ## Mirror-sync guards
//!
//! `nn-core`'s `TraceOp`, `DType`, and support enums are `#[non_exhaustive]`,
//! so Rust *requires* a wildcard arm in every cross-crate match here. Each of
//! those mandated arms panics: a variant added upstream fails loudly (fail
//! closed) instead of being silently mis-serialized. Every *current* variant
//! has its own explicit arm — nothing known is routed through a catch-all.
//! The compile-time guard runs in the other direction:
//! `tests/trace_to_schema_totality.rs` matches the schema `TraceOp` (which is
//! exhaustive) wildcard-free and pins the variant count, so when the NY
//! schema mirrors a new op the test build breaks until this converter maps
//! it, and the totality test exercises all 123 conversions end-to-end.

use nn_core::dyn_tensor::trace::{
    ComputationGraph, GraphSegment, KokoroFusedOp, ResBlockActivation, SegmentedGraph,
    TraceActivation, TraceNode, TraceOp, TraceUpsampleMode, WeightRef,
};
use nn_core::dyn_tensor::{CompareOp, GridSamplePaddingMode};
use nn_core::DType;

use ny_trace_bridge::schema;

/// Convert an `nn-core` traced computation graph into the NY bridge schema.
///
/// Reads every node through `nn-core`'s public accessors (its `TraceNode`
/// fields are `pub(super)`) and mirrors them 1:1 onto the `pub`-field schema
/// types. Node IDs and dtypes map directly; weights (`nn-core`'s flat-f32
/// `WeightRef`) become [`schema::WeightPayload`] — f32 data when captured,
/// shape-only placeholder when the source itself only captured a shape.
///
/// # Panics
///
/// Panics only if `nn-core` grows a `TraceOp`, `DType`, or support-enum
/// variant that is not yet mirrored in `ny_trace_bridge::schema` (the source
/// enums are `#[non_exhaustive]`). A new upstream variant must fail loudly
/// rather than serialize incorrectly.
#[must_use]
pub fn to_schema(g: &ComputationGraph) -> schema::ComputationGraph {
    let nodes = g.nodes().iter().map(node_to_schema).collect();
    let output_nodes = g
        .output_nodes()
        .iter()
        .map(|n| schema::NodeId(n.id()))
        .collect();
    schema::ComputationGraph {
        nodes,
        output_nodes,
    }
}

/// Convert an `nn-core` segmented graph (split at `SegmentBoundary` markers)
/// into the NY bridge schema, segment by segment.
///
/// Boundary metadata (`boundary_reason`, `boundary_bounds`) carries over
/// verbatim; each segment's sub-graph goes through [`to_schema`].
///
/// # Panics
///
/// Same contract as [`to_schema`]: only on an unmirrored future `nn-core`
/// variant.
#[must_use]
pub fn segmented_to_schema(g: &SegmentedGraph) -> schema::SegmentedGraph {
    schema::SegmentedGraph {
        segments: g.segments.iter().map(segment_to_schema).collect(),
    }
}

/// Mirror a single `nn-core` `GraphSegment` onto the schema `GraphSegment`.
fn segment_to_schema(segment: &GraphSegment) -> schema::GraphSegment {
    schema::GraphSegment {
        graph: to_schema(&segment.graph),
        boundary_reason: segment.boundary_reason.clone(),
        boundary_bounds: segment.boundary_bounds,
    }
}

/// Mirror a single `nn-core` `TraceNode` onto the schema `TraceNode`.
fn node_to_schema(node: &TraceNode) -> schema::TraceNode {
    schema::TraceNode {
        id: schema::NodeId(node.id()),
        name: node.name().to_string(),
        op: op_to_schema(node.op()),
        inputs: node.inputs().iter().map(|&id| schema::NodeId(id)).collect(),
        output_shape: node.output_shape().to_vec(),
        output_dtype: dtype_to_schema(node.output_dtype()),
    }
}

/// Map every `nn-core` `TraceOp` onto the schema `TraceOp` mirror, 1:1 by
/// name and field. Total over all 123 current variants; the trailing arm is
/// the `#[non_exhaustive]`-mandated wildcard and fires only for a future
/// variant this mirror does not know (fail closed).
fn op_to_schema(op: &TraceOp) -> schema::TraceOp {
    match op {
        TraceOp::Input => schema::TraceOp::Input,
        TraceOp::ConstantWeight { weight } => schema::TraceOp::ConstantWeight {
            weight: weight_to_payload(weight),
        },
        TraceOp::Add => schema::TraceOp::Add,
        TraceOp::Sub => schema::TraceOp::Sub,
        TraceOp::Mul => schema::TraceOp::Mul,
        TraceOp::Div => schema::TraceOp::Div,
        TraceOp::Maximum => schema::TraceOp::Maximum,
        TraceOp::Minimum => schema::TraceOp::Minimum,
        TraceOp::MatMul => schema::TraceOp::MatMul,
        TraceOp::Relu => schema::TraceOp::Relu,
        TraceOp::Gelu => schema::TraceOp::Gelu,
        TraceOp::GeluErf => schema::TraceOp::GeluErf,
        TraceOp::Silu => schema::TraceOp::Silu,
        TraceOp::Tanh => schema::TraceOp::Tanh,
        TraceOp::Sigmoid => schema::TraceOp::Sigmoid,
        TraceOp::Exp => schema::TraceOp::Exp,
        TraceOp::Log => schema::TraceOp::Log,
        TraceOp::Sqrt => schema::TraceOp::Sqrt,
        TraceOp::Sqr => schema::TraceOp::Sqr,
        TraceOp::Abs => schema::TraceOp::Abs,
        TraceOp::Neg => schema::TraceOp::Neg,
        TraceOp::Recip => schema::TraceOp::Recip,
        TraceOp::Sin => schema::TraceOp::Sin,
        TraceOp::Cos => schema::TraceOp::Cos,
        TraceOp::Tan => schema::TraceOp::Tan,
        TraceOp::Floor => schema::TraceOp::Floor,
        TraceOp::Ceil => schema::TraceOp::Ceil,
        TraceOp::Round => schema::TraceOp::Round,
        TraceOp::Sign => schema::TraceOp::Sign,
        TraceOp::Fract => schema::TraceOp::Fract,
        TraceOp::ReduceSum { dim, keepdim } => schema::TraceOp::ReduceSum {
            dim: *dim,
            keepdim: *keepdim,
        },
        TraceOp::ReduceMean { dim, keepdim } => schema::TraceOp::ReduceMean {
            dim: *dim,
            keepdim: *keepdim,
        },
        TraceOp::ReduceMax { dim, keepdim } => schema::TraceOp::ReduceMax {
            dim: *dim,
            keepdim: *keepdim,
        },
        TraceOp::ReduceMin { dim, keepdim } => schema::TraceOp::ReduceMin {
            dim: *dim,
            keepdim: *keepdim,
        },
        TraceOp::Reshape { target_shape } => schema::TraceOp::Reshape {
            target_shape: target_shape.clone(),
        },
        TraceOp::Transpose { dim0, dim1 } => schema::TraceOp::Transpose {
            dim0: *dim0,
            dim1: *dim1,
        },
        TraceOp::Narrow { dim, start, length } => schema::TraceOp::Narrow {
            dim: *dim,
            start: *start,
            length: *length,
        },
        TraceOp::Unsqueeze { dim } => schema::TraceOp::Unsqueeze { dim: *dim },
        TraceOp::Squeeze { dim } => schema::TraceOp::Squeeze { dim: *dim },
        TraceOp::Permute { axes } => schema::TraceOp::Permute { axes: axes.clone() },
        TraceOp::Cat { dim, num_inputs } => schema::TraceOp::Cat {
            dim: *dim,
            num_inputs: *num_inputs,
        },
        TraceOp::LayerNorm { eps, weight, bias } => schema::TraceOp::LayerNorm {
            eps: *eps,
            weight: weight_to_payload(weight),
            bias: weight_to_payload(bias),
        },
        TraceOp::RmsNorm { eps, weight } => schema::TraceOp::RmsNorm {
            eps: *eps,
            weight: weight_to_payload(weight),
        },
        TraceOp::GroupNorm {
            num_groups,
            eps,
            weight,
            bias,
        } => schema::TraceOp::GroupNorm {
            num_groups: *num_groups,
            eps: *eps,
            weight: weight_to_payload(weight),
            bias: weight_to_payload(bias),
        },
        TraceOp::InstanceNorm { eps } => schema::TraceOp::InstanceNorm { eps: *eps },
        TraceOp::BatchNorm {
            eps,
            weight,
            bias,
            running_mean,
            running_var,
        } => schema::TraceOp::BatchNorm {
            eps: *eps,
            weight: weight_to_payload(weight),
            bias: weight_to_payload(bias),
            running_mean: weight_to_payload(running_mean),
            running_var: weight_to_payload(running_var),
        },
        TraceOp::Linear { weight, bias } => schema::TraceOp::Linear {
            weight: weight_to_payload(weight),
            bias: bias.as_ref().map(weight_to_payload),
        },
        TraceOp::Conv1d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => schema::TraceOp::Conv1d {
            weight: weight_to_payload(weight),
            bias: bias.as_ref().map(weight_to_payload),
            padding: *padding,
            stride: *stride,
            dilation: *dilation,
            groups: *groups,
        },
        TraceOp::Conv2d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => schema::TraceOp::Conv2d {
            weight: weight_to_payload(weight),
            bias: bias.as_ref().map(weight_to_payload),
            padding: *padding,
            stride: *stride,
            dilation: *dilation,
            groups: *groups,
        },
        TraceOp::Conv3d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => schema::TraceOp::Conv3d {
            weight: weight_to_payload(weight),
            bias: bias.as_ref().map(weight_to_payload),
            padding: *padding,
            stride: *stride,
            dilation: *dilation,
            groups: *groups,
        },
        TraceOp::ConvTranspose1d {
            weight,
            bias,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        } => schema::TraceOp::ConvTranspose1d {
            weight: weight_to_payload(weight),
            bias: bias.as_ref().map(weight_to_payload),
            padding: *padding,
            output_padding: *output_padding,
            stride: *stride,
            dilation: *dilation,
            groups: *groups,
        },
        TraceOp::ConvTranspose2d {
            weight,
            bias,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        } => schema::TraceOp::ConvTranspose2d {
            weight: weight_to_payload(weight),
            bias: bias.as_ref().map(weight_to_payload),
            padding: *padding,
            output_padding: *output_padding,
            stride: *stride,
            dilation: *dilation,
            groups: *groups,
        },
        TraceOp::Softmax { dim } => schema::TraceOp::Softmax { dim: *dim },
        TraceOp::LogSoftmax { dim } => schema::TraceOp::LogSoftmax { dim: *dim },
        TraceOp::Sdpa { scale } => schema::TraceOp::Sdpa { scale: *scale },
        TraceOp::SdpaCausal { scale } => schema::TraceOp::SdpaCausal { scale: *scale },
        TraceOp::RotaryEmbedding {
            head_dim,
            offset,
            cos_cache,
            sin_cache,
        } => schema::TraceOp::RotaryEmbedding {
            head_dim: *head_dim,
            offset: *offset,
            cos_cache: weight_to_payload(cos_cache),
            sin_cache: weight_to_payload(sin_cache),
        },
        TraceOp::MultiHeadAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        } => schema::TraceOp::MultiHeadAttention {
            num_heads: *num_heads,
            num_kv_heads: *num_kv_heads,
            head_dim: *head_dim,
        },
        TraceOp::Embedding { weight } => schema::TraceOp::Embedding {
            weight: weight_to_payload(weight),
        },
        TraceOp::Lstm {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
            initial_hidden,
            initial_cell,
        } => schema::TraceOp::Lstm {
            weight_ih: weight_to_payload(weight_ih),
            weight_hh: weight_to_payload(weight_hh),
            bias_ih: bias_ih.as_ref().map(weight_to_payload),
            bias_hh: bias_hh.as_ref().map(weight_to_payload),
            hidden_size: *hidden_size,
            initial_hidden: initial_hidden.map(schema::NodeId),
            initial_cell: initial_cell.map(schema::NodeId),
        },
        TraceOp::MaxPool1d {
            kernel_size,
            stride,
            padding,
        } => schema::TraceOp::MaxPool1d {
            kernel_size: *kernel_size,
            stride: *stride,
            padding: *padding,
        },
        TraceOp::AvgPool2d {
            kernel_size,
            stride,
            padding,
        } => schema::TraceOp::AvgPool2d {
            kernel_size: *kernel_size,
            stride: *stride,
            padding: *padding,
        },
        TraceOp::MaxPool2d {
            kernel_size,
            stride,
            padding,
        } => schema::TraceOp::MaxPool2d {
            kernel_size: *kernel_size,
            stride: *stride,
            padding: *padding,
        },
        TraceOp::AdaptiveAvgPool2d { output_size } => schema::TraceOp::AdaptiveAvgPool2d {
            output_size: *output_size,
        },
        TraceOp::AvgPool1d {
            kernel_size,
            stride,
            padding,
        } => schema::TraceOp::AvgPool1d {
            kernel_size: *kernel_size,
            stride: *stride,
            padding: *padding,
        },
        TraceOp::AdaptiveAvgPool1d { output_size } => schema::TraceOp::AdaptiveAvgPool1d {
            output_size: *output_size,
        },
        TraceOp::AdaptiveMaxPool2d { output_size } => schema::TraceOp::AdaptiveMaxPool2d {
            output_size: *output_size,
        },
        TraceOp::Activation { kind } => schema::TraceOp::Activation {
            kind: activation_to_schema(*kind),
        },
        TraceOp::Elu { alpha } => schema::TraceOp::Elu { alpha: *alpha },
        TraceOp::LeakyRelu { slope } => schema::TraceOp::LeakyRelu { slope: *slope },
        TraceOp::Softplus => schema::TraceOp::Softplus,
        TraceOp::Selu => schema::TraceOp::Selu,
        TraceOp::Celu { alpha } => schema::TraceOp::Celu { alpha: *alpha },
        TraceOp::Mish => schema::TraceOp::Mish,
        TraceOp::HardSigmoid => schema::TraceOp::HardSigmoid,
        TraceOp::HardSwish => schema::TraceOp::HardSwish,
        TraceOp::Softsign => schema::TraceOp::Softsign,
        TraceOp::PRelu { slope } => schema::TraceOp::PRelu {
            slope: weight_to_payload(slope),
        },
        TraceOp::KokoroFused(fused) => schema::TraceOp::KokoroFused(kokoro_to_schema(fused)),
        TraceOp::SwiGlu => schema::TraceOp::SwiGlu,
        TraceOp::Dropout => schema::TraceOp::Dropout,
        TraceOp::PixelShuffle { upscale_factor } => schema::TraceOp::PixelShuffle {
            upscale_factor: *upscale_factor,
        },
        TraceOp::PixelUnshuffle { downscale_factor } => schema::TraceOp::PixelUnshuffle {
            downscale_factor: *downscale_factor,
        },
        TraceOp::Upsample1d { factor } => schema::TraceOp::Upsample1d { factor: *factor },
        TraceOp::Upsample2d {
            mode,
            scale_h,
            scale_w,
        } => schema::TraceOp::Upsample2d {
            mode: upsample_mode_to_schema(*mode),
            scale_h: *scale_h,
            scale_w: *scale_w,
        },
        TraceOp::ResizeBilinear { target_h, target_w } => schema::TraceOp::ResizeBilinear {
            target_h: *target_h,
            target_w: *target_w,
        },
        TraceOp::Triu { diagonal } => schema::TraceOp::Triu {
            diagonal: *diagonal,
        },
        TraceOp::Tril { diagonal } => schema::TraceOp::Tril {
            diagonal: *diagonal,
        },
        TraceOp::GridSample {
            padding_mode,
            align_corners,
        } => schema::TraceOp::GridSample {
            padding_mode: grid_padding_to_schema(*padding_mode),
            align_corners: *align_corners,
        },
        TraceOp::QLinear { weight, bias } => schema::TraceOp::QLinear {
            weight: weight_to_payload(weight),
            bias: bias.as_ref().map(weight_to_payload),
        },
        TraceOp::Topk { k, dim } => schema::TraceOp::Topk { k: *k, dim: *dim },
        TraceOp::Argmax { dim } => schema::TraceOp::Argmax { dim: *dim },
        TraceOp::Argmin { dim } => schema::TraceOp::Argmin { dim: *dim },
        TraceOp::ArgSort { dim, descending } => schema::TraceOp::ArgSort {
            dim: *dim,
            descending: *descending,
        },
        TraceOp::Sort { dim, descending } => schema::TraceOp::Sort {
            dim: *dim,
            descending: *descending,
        },
        TraceOp::IndexSelect { dim } => schema::TraceOp::IndexSelect { dim: *dim },
        TraceOp::Gather { dim } => schema::TraceOp::Gather { dim: *dim },
        TraceOp::WhereCond => schema::TraceOp::WhereCond,
        TraceOp::Expand { target_shape } => schema::TraceOp::Expand {
            target_shape: target_shape.clone(),
        },
        TraceOp::Compare { op, value } => schema::TraceOp::Compare {
            op: compare_op_to_schema(*op),
            value: *value,
        },
        TraceOp::CompareTensor { op } => schema::TraceOp::CompareTensor {
            op: compare_op_to_schema(*op),
        },
        TraceOp::Cumsum { dim } => schema::TraceOp::Cumsum { dim: *dim },
        TraceOp::RepeatInterleave { dim } => schema::TraceOp::RepeatInterleave { dim: *dim },
        TraceOp::Powf { exponent } => schema::TraceOp::Powf {
            exponent: *exponent,
        },
        TraceOp::ToDtype { target_dtype } => schema::TraceOp::ToDtype {
            target_dtype: dtype_to_schema(*target_dtype),
        },
        TraceOp::Flip { dim } => schema::TraceOp::Flip { dim: *dim },
        TraceOp::Roll { shifts, dims } => schema::TraceOp::Roll {
            shifts: shifts.clone(),
            dims: dims.clone(),
        },
        TraceOp::Unfold { dim, size, step } => schema::TraceOp::Unfold {
            dim: *dim,
            size: *size,
            step: *step,
        },
        TraceOp::SliceSet { dim, start } => schema::TraceOp::SliceSet {
            dim: *dim,
            start: *start,
        },
        TraceOp::Scatter { dim } => schema::TraceOp::Scatter { dim: *dim },
        TraceOp::ScatterAdd { dim } => schema::TraceOp::ScatterAdd { dim: *dim },
        TraceOp::IndexAdd { dim } => schema::TraceOp::IndexAdd { dim: *dim },
        TraceOp::IndexPut { dim } => schema::TraceOp::IndexPut { dim: *dim },
        TraceOp::Clamp { min, max } => schema::TraceOp::Clamp {
            min: *min,
            max: *max,
        },
        TraceOp::Constant { value } => schema::TraceOp::Constant { value: *value },
        TraceOp::ReflectionPad1d {
            pad_left,
            pad_right,
        } => schema::TraceOp::ReflectionPad1d {
            pad_left: *pad_left,
            pad_right: *pad_right,
        },
        TraceOp::ReflectionPad2d {
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        } => schema::TraceOp::ReflectionPad2d {
            pad_left: *pad_left,
            pad_right: *pad_right,
            pad_top: *pad_top,
            pad_bottom: *pad_bottom,
        },
        TraceOp::ConstantPadNd { padding, value } => schema::TraceOp::ConstantPadNd {
            padding: padding.clone(),
            value: *value,
        },
        TraceOp::Atan2 => schema::TraceOp::Atan2,
        TraceOp::Arange { start, end, step } => schema::TraceOp::Arange {
            start: *start,
            end: *end,
            step: *step,
        },
        TraceOp::SegmentBoundary {
            reason,
            input_bounds,
        } => schema::TraceOp::SegmentBoundary {
            reason: reason.clone(),
            input_bounds: *input_bounds,
        },
        TraceOp::MoeGating {
            num_experts,
            top_k,
        } => schema::TraceOp::MoeGating {
            num_experts: *num_experts,
            top_k: *top_k,
        },
        TraceOp::Custom { name } => schema::TraceOp::Custom { name: name.clone() },
        // `nn_core` `TraceOp` is `#[non_exhaustive]`: this arm is mandated by
        // the compiler (E0004 without it) and fires only for a future variant
        // the schema mirror does not know yet. Fail closed — never serialize
        // an unknown op.
        other => panic!("nn-core TraceOp variant {other:?} is not mirrored in the NY bridge schema; extend ny_trace_bridge::schema and this converter"),
    }
}

/// Map the Kokoro fused-op payload onto its schema mirror, 1:1 by field.
fn kokoro_to_schema(op: &KokoroFusedOp) -> schema::KokoroFusedOp {
    match op {
        KokoroFusedOp::SnakeTensor { alpha } => schema::KokoroFusedOp::SnakeTensor {
            alpha: weight_to_payload(alpha),
        },
        KokoroFusedOp::AdainSnake { alpha, eps } => schema::KokoroFusedOp::AdainSnake {
            alpha: weight_to_payload(alpha),
            eps: *eps,
        },
        KokoroFusedOp::AdainLeakyRelu { eps, slope } => schema::KokoroFusedOp::AdainLeakyRelu {
            eps: *eps,
            slope: *slope,
        },
        KokoroFusedOp::AdaLayerNorm {
            norm_weight,
            norm_bias,
            eps,
        } => schema::KokoroFusedOp::AdaLayerNorm {
            norm_weight: weight_to_payload(norm_weight),
            norm_bias: weight_to_payload(norm_bias),
            eps: *eps,
        },
        KokoroFusedOp::FusedAdainResBlock {
            activation,
            adain1_weight,
            adain1_bias,
            adain2_weight,
            adain2_bias,
            conv1_weight,
            conv1_bias,
            conv1_dilation,
            conv1_padding,
            conv2_weight,
            conv2_bias,
            conv2_padding,
            eps,
            residual_scale,
        } => schema::KokoroFusedOp::FusedAdainResBlock {
            activation: resblock_activation_to_schema(activation),
            adain1_weight: weight_to_payload(adain1_weight),
            adain1_bias: weight_to_payload(adain1_bias),
            adain2_weight: weight_to_payload(adain2_weight),
            adain2_bias: weight_to_payload(adain2_bias),
            conv1_weight: weight_to_payload(conv1_weight),
            conv1_bias: weight_to_payload(conv1_bias),
            conv1_dilation: *conv1_dilation,
            conv1_padding: *conv1_padding,
            conv2_weight: weight_to_payload(conv2_weight),
            conv2_bias: weight_to_payload(conv2_bias),
            conv2_padding: *conv2_padding,
            eps: *eps,
            residual_scale: *residual_scale,
        },
        // `#[non_exhaustive]`-mandated arm: future fused ops fail closed.
        other => panic!("nn-core KokoroFusedOp variant {other:?} is not mirrored in the NY bridge schema; extend ny_trace_bridge::schema and this converter"),
    }
}

/// Map a fused-ResBlock activation onto its schema mirror.
///
/// `ResBlockActivation` is the one trace enum `nn-core` does *not* mark
/// `#[non_exhaustive]`, so this match is genuinely wildcard-free: a new
/// source variant is a compile error here.
fn resblock_activation_to_schema(activation: &ResBlockActivation) -> schema::ResBlockActivation {
    match activation {
        ResBlockActivation::Snake { alpha1, alpha2 } => schema::ResBlockActivation::Snake {
            alpha1: weight_to_payload(alpha1),
            alpha2: weight_to_payload(alpha2),
        },
        ResBlockActivation::LeakyRelu { slope } => {
            schema::ResBlockActivation::LeakyRelu { slope: *slope }
        }
    }
}

/// Map the named-activation discriminant onto its schema mirror, 1:1 by name.
fn activation_to_schema(kind: TraceActivation) -> schema::TraceActivation {
    match kind {
        TraceActivation::Relu => schema::TraceActivation::Relu,
        TraceActivation::Gelu => schema::TraceActivation::Gelu,
        TraceActivation::GeluErf => schema::TraceActivation::GeluErf,
        TraceActivation::Silu => schema::TraceActivation::Silu,
        TraceActivation::Sigmoid => schema::TraceActivation::Sigmoid,
        TraceActivation::Tanh => schema::TraceActivation::Tanh,
        TraceActivation::Exp => schema::TraceActivation::Exp,
        TraceActivation::Log => schema::TraceActivation::Log,
        TraceActivation::Elu => schema::TraceActivation::Elu,
        TraceActivation::LeakyRelu => schema::TraceActivation::LeakyRelu,
        TraceActivation::Mish => schema::TraceActivation::Mish,
        // `#[non_exhaustive]`-mandated arm: future activations fail closed.
        other => panic!("nn-core TraceActivation variant {other:?} is not mirrored in the NY bridge schema; extend ny_trace_bridge::schema and this converter"),
    }
}

/// Map the 2-D upsample mode onto its schema mirror, 1:1 by name.
fn upsample_mode_to_schema(mode: TraceUpsampleMode) -> schema::TraceUpsampleMode {
    match mode {
        TraceUpsampleMode::Nearest => schema::TraceUpsampleMode::Nearest,
        TraceUpsampleMode::Bilinear => schema::TraceUpsampleMode::Bilinear,
        TraceUpsampleMode::Bicubic => schema::TraceUpsampleMode::Bicubic,
        // `#[non_exhaustive]`-mandated arm: future modes fail closed.
        other => panic!("nn-core TraceUpsampleMode variant {other:?} is not mirrored in the NY bridge schema; extend ny_trace_bridge::schema and this converter"),
    }
}

/// Map the comparison discriminant onto its schema mirror, 1:1 by name.
fn compare_op_to_schema(op: CompareOp) -> schema::CompareOp {
    match op {
        CompareOp::Eq => schema::CompareOp::Eq,
        CompareOp::Ne => schema::CompareOp::Ne,
        CompareOp::Ge => schema::CompareOp::Ge,
        CompareOp::Gt => schema::CompareOp::Gt,
        CompareOp::Lt => schema::CompareOp::Lt,
        CompareOp::Le => schema::CompareOp::Le,
        // `#[non_exhaustive]`-mandated arm: future comparisons fail closed.
        other => panic!("nn-core CompareOp variant {other:?} is not mirrored in the NY bridge schema; extend ny_trace_bridge::schema and this converter"),
    }
}

/// Map the grid-sample padding mode onto its schema mirror, 1:1 by name.
fn grid_padding_to_schema(mode: GridSamplePaddingMode) -> schema::GridSamplePaddingMode {
    match mode {
        GridSamplePaddingMode::Zeros => schema::GridSamplePaddingMode::Zeros,
        GridSamplePaddingMode::Border => schema::GridSamplePaddingMode::Border,
        // `#[non_exhaustive]`-mandated arm: future modes fail closed.
        other => panic!("nn-core GridSamplePaddingMode variant {other:?} is not mirrored in the NY bridge schema; extend ny_trace_bridge::schema and this converter"),
    }
}

/// Convert an `nn-core` `WeightRef` (flat f32 + shape) to a schema
/// `WeightPayload`.
///
/// Captured weights become `WeightPayload::f32`, preserving the exact
/// representation NN emits today. Shape-only refs (`WeightRef::from_shape`,
/// the source's last-resort fallback when data extraction fails) become the
/// schema's `Placeholder` so the "data was never captured" fact survives the
/// wire — the bridge can then refuse them explicitly instead of reading an
/// empty tensor as real weights.
fn weight_to_payload(weight: &WeightRef) -> schema::WeightPayload {
    if weight.is_placeholder() {
        schema::WeightPayload::placeholder(weight.shape().to_vec())
    } else {
        schema::WeightPayload::f32(weight.data().to_vec(), weight.shape().to_vec())
    }
}

/// Map `nn-core`'s `DType` onto the schema `DType` mirror, 1:1.
fn dtype_to_schema(dtype: DType) -> schema::DType {
    match dtype {
        DType::F32 => schema::DType::F32,
        DType::F16 => schema::DType::F16,
        DType::BF16 => schema::DType::Bf16,
        DType::F64 => schema::DType::F64,
        DType::I32 => schema::DType::I32,
        DType::I64 => schema::DType::I64,
        DType::U32 => schema::DType::U32,
        DType::U8 => schema::DType::U8,
        DType::Bool => schema::DType::Bool,
        // `nn_core::DType` is `#[non_exhaustive]`; a new dtype must fail
        // loudly rather than be silently misclassified.
        other => panic!("nn-core DType {other:?} is not mirrored in the NY bridge schema; extend ny_trace_bridge::schema and this converter"),
    }
}
