// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convolution compilation helpers for `trace_compile`.
//!
//! Extracted from `trace_compile_ops.rs` to keep files under 500 lines.
//! These functions lower Conv1d, Conv2d, and ConvTranspose1d `TraceOp`
//! variants into `TensorKernelDef` dispatch plans.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, WeightRef};

use crate::tensor_ir::TensorIRError;

use super::super::{resolve_input_shape, CompiledKernel, CompiledStep};
use super::{add_weight, build_op_with_weights};

/// Minimum total GEMM FLOPs (M*K*N) to justify im2col overhead in the
/// compiled model path. Matches `MIN_GEMM_FLOPS` in
/// `dyn_tensor_metal_conv_gemm.rs`. Part of #3390.
const MIN_CONV1D_GEMM_FLOPS: usize = 2_000_000;

/// Returns true if a Conv1d should use the im2col + GEMM NativeOp path
/// in the compiled model. Same threshold as the DynTensor-level routing
/// in `MetalDynBackend::should_use_conv1d_gemm`.
fn should_use_conv1d_gemm_native(
    input_shape: &[usize],
    weight_shape: &[usize],
    out_len: usize,
    groups: usize,
) -> bool {
    // Only standard (non-grouped) convolutions. The GEMM path does not
    // support grouped conv (would need per-group loops).
    if groups != 1 {
        return false;
    }
    if input_shape.len() != 3 || weight_shape.len() != 3 {
        return false;
    }
    let c_out = weight_shape[0];
    let c_in_k = weight_shape[1] * weight_shape[2]; // C_in_per_group * K
    c_out * c_in_k * out_len >= MIN_CONV1D_GEMM_FLOPS
}

/// Returns true if a Conv1d is depthwise (groups == in_channels) or more
/// generally a grouped convolution that should route to the `Conv1dGemm`
/// NativeOp path.
///
/// Depthwise convolutions (groups == in_channels, weight shape `[C, 1, K]`)
/// are common in Kokoro's generator and decoder. Routing them to the NativeOp
/// path avoids the IR decomposition overhead and delegates to the Metal
/// backend's optimized `gpu_conv1d` dispatch. Part of #3538.
fn should_use_conv1d_depthwise_native(input_shape: &[usize], groups: usize) -> bool {
    if groups <= 1 {
        return false;
    }
    if input_shape.len() != 3 {
        return false;
    }
    let in_channels = input_shape[1];
    // Depthwise: groups == in_channels. Also handle any grouped conv where
    // groups > 1 and divides in_channels evenly.
    groups == in_channels || (in_channels > 0 && in_channels.is_multiple_of(groups))
}

// -- Convolutions -------------------------------------------------------------

pub(in crate::trace_compile) fn compile_conv1d(
    node: &TraceNode,
    graph: &ComputationGraph,
    weight: &WeightRef,
    bias: &Option<WeightRef>,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let weight_shape = weight.shape();
    let output_shape = node.output_shape();

    // Compute output length for GEMM routing check.
    let out_len = output_shape.last().copied().unwrap_or(0);

    // Route qualifying Conv1d shapes to NativeOp:
    // 1. Large standard convolutions → im2col + GEMM path (#3390)
    // 2. Depthwise/grouped convolutions → generic gpu_conv1d path (#3538)
    let use_native = should_use_conv1d_gemm_native(input_shape, weight_shape, out_len, groups)
        || should_use_conv1d_depthwise_native(input_shape, groups);
    if use_native {
        let mut weight_data = std::collections::HashMap::new();
        weight_data.insert("weight".to_string(), weight.clone());
        let has_bias = bias.is_some();
        if let Some(bw) = bias {
            weight_data.insert("bias".to_string(), bw.clone());
        }
        let k_size = weight_shape[2];
        let out_ch = weight_shape[0];
        return Ok(CompiledStep::NativeOp {
            op: crate::NativeOpKind::Conv1dGemm {
                input_shape: input_shape.to_vec(),
                out_channels: out_ch,
                kernel_size: k_size,
                stride,
                padding,
                dilation,
                groups,
                has_bias,
            },
            weight_data,
        });
    }

    let (def, weight_data) = build_op_with_weights("conv1d", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);
        let w = add_weight(b, wd, "weight", weight);
        let bi = bias.as_ref().map(|bw| add_weight(b, wd, "bias", bw));
        b.add_conv1d_full(
            input,
            w,
            bi,
            stride,
            padding,
            dilation,
            groups,
            node.output_shape(),
        )
    })?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

pub(in crate::trace_compile) fn compile_conv2d(
    node: &TraceNode,
    graph: &ComputationGraph,
    weight: &WeightRef,
    bias: &Option<WeightRef>,
    padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let (def, weight_data) = build_op_with_weights("conv2d", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);
        let w = add_weight(b, wd, "weight", weight);
        let bi = bias.as_ref().map(|bw| add_weight(b, wd, "bias", bw));
        b.add_conv2d_full(
            input,
            w,
            bi,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            groups,
            node.output_shape(),
        )
    })?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

pub(in crate::trace_compile) fn compile_conv_transpose2d(
    _node: &TraceNode,
    _graph: &ComputationGraph,
    _weight: &WeightRef,
    _bias: &Option<WeightRef>,
    _padding: [usize; 2],
    _output_padding: [usize; 2],
    _stride: [usize; 2],
    _dilation: [usize; 2],
    _groups: usize,
) -> Result<CompiledStep, TensorIRError> {
    // ConvTranspose2d IR construction works, but MSL codegen is not yet
    // implemented (codegen_msl_tensor_dispatch.rs returns UnsupportedOp).
    // Fail early here instead of producing a CompiledModel that errors at
    // execution time.
    Err(TensorIRError::UnsupportedTraceOp {
        name: "ConvTranspose2d (MSL codegen not yet implemented)".into(),
    })
}

// -- Pooling ------------------------------------------------------------------

pub(in crate::trace_compile) fn compile_avg_pool2d(
    node: &TraceNode,
    graph: &ComputationGraph,
    kernel_size: &[usize; 2],
    stride: &[usize; 2],
    padding: &[usize; 2],
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let mut b = crate::tensor_block_builder::TensorBlockBuilder::new("avg_pool2d");
    let input = b.add_input("input_0", input_shape);
    let output = b.add_avg_pool_2d(
        input,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        node.output_shape(),
    );
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: std::collections::HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

pub(in crate::trace_compile) fn compile_max_pool2d(
    node: &TraceNode,
    graph: &ComputationGraph,
    kernel_size: &[usize; 2],
    stride: &[usize; 2],
    padding: &[usize; 2],
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let mut b = crate::tensor_block_builder::TensorBlockBuilder::new("max_pool2d");
    let input = b.add_input("input_0", input_shape);
    let output = b.add_max_pool_2d(
        input,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        node.output_shape(),
    );
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: std::collections::HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

/// Compile a `TraceOp::MaxPool1d` into a `NativeOp` step.
///
/// Delegates to `DynTensor::max_pool1d()` at execution time via the
/// native bridge. No MSL codegen needed — uses the existing CPU-roundtrip
/// pooling implementation on GPU tensors. Part of #2295 (PyanNet).
pub(in crate::trace_compile) fn compile_max_pool1d(
    node: &TraceNode,
    graph: &ComputationGraph,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    Ok(CompiledStep::NativeOp {
        op: crate::NativeOpKind::MaxPool1d {
            kernel_size,
            stride,
            padding,
            input_shape: input_shape.to_vec(),
        },
        weight_data: std::collections::HashMap::new(),
    })
}

pub(in crate::trace_compile) fn compile_conv_transpose1d(
    node: &TraceNode,
    graph: &ComputationGraph,
    weight: &WeightRef,
    bias: &Option<WeightRef>,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let (def, weight_data) = build_op_with_weights("conv_transpose1d", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);
        let w = add_weight(b, wd, "weight", weight);
        let bi = bias.as_ref().map(|bw| add_weight(b, wd, "bias", bw));
        b.add_conv_transpose_1d(
            input,
            w,
            bi,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
            node.output_shape(),
        )
    })?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}
