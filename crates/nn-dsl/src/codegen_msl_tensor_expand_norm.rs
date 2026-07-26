// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Norm-specific op expansion helpers for MSL codegen.
//!
//! Contains the decomposition logic for InstanceNorm, RmsNorm, LayerNorm,
//! and AdaIN affine transforms. Extracted from `codegen_msl_tensor_expand.rs`
//! to stay under the 500-line limit (#341).

use crate::ir::{BinOpKind, UnaryFnKind};
use crate::tensor_builders::{
    binop_kernel, broadcast_node, elementwise_node, reduce_node, square_kernel, unary_kernel,
};
use crate::tensor_ir::{BroadcastAlignment, ReduceOp, TensorNode, TensorNodeId, TensorOpKind};

use super::ExpandState;

/// Emit the InstanceNorm core: `(x - mean) * rsqrt(var + eps)`.
///
/// Returns the node ID of the final normalized output.
pub(super) fn emit_instance_norm_core(
    st: &mut ExpandState,
    input: usize,
    eps: usize,
    axis: usize,
    full: &[usize],
    reduced: &[usize],
) -> usize {
    let mean = st.alloc();
    st.push(reduce_node(mean, ReduceOp::Mean, input, axis, reduced));

    let bcast_mean = st.alloc();
    st.push(broadcast_node(
        bcast_mean,
        mean,
        full,
        BroadcastAlignment::Left,
    ));

    let centered = st.alloc();
    st.push(elementwise_node(
        centered,
        binop_kernel("sub", BinOpKind::Sub),
        &[input, bcast_mean],
        full,
    ));

    let sq = st.alloc();
    st.push(elementwise_node(sq, square_kernel(), &[centered], full));

    let var = st.alloc();
    st.push(reduce_node(var, ReduceOp::Mean, sq, axis, reduced));

    let bcast_var = st.alloc();
    st.push(broadcast_node(
        bcast_var,
        var,
        full,
        BroadcastAlignment::Left,
    ));

    let bcast_eps = st.alloc();
    st.push(broadcast_node(
        bcast_eps,
        eps,
        full,
        BroadcastAlignment::Left,
    ));

    let var_eps = st.alloc();
    st.push(elementwise_node(
        var_eps,
        binop_kernel("add", BinOpKind::Add),
        &[bcast_var, bcast_eps],
        full,
    ));

    let rsqrt = st.alloc();
    st.push(elementwise_node(
        rsqrt,
        unary_kernel("rsqrt", UnaryFnKind::Rsqrt),
        &[var_eps],
        full,
    ));

    let normed = st.alloc();
    st.push(elementwise_node(
        normed,
        binop_kernel("mul", BinOpKind::Mul),
        &[centered, rsqrt],
        full,
    ));
    normed
}

/// Emit the RmsNorm decomposition: `x * rsqrt(mean(x^2) + eps) * weight`.
///
/// Returns the node ID of the final weighted output.
pub(super) fn emit_rms_norm_core(
    st: &mut ExpandState,
    input: usize,
    eps: usize,
    weight: usize,
    axis: usize,
    full: &[usize],
    reduced: &[usize],
) -> usize {
    let sq = st.alloc();
    st.push(elementwise_node(sq, square_kernel(), &[input], full));

    let mean_sq = st.alloc();
    st.push(reduce_node(mean_sq, ReduceOp::Mean, sq, axis, reduced));

    let bcast_ms = st.alloc();
    st.push(broadcast_node(
        bcast_ms,
        mean_sq,
        full,
        BroadcastAlignment::Left,
    ));

    let bcast_e = st.alloc();
    st.push(broadcast_node(bcast_e, eps, full, BroadcastAlignment::Left));

    let sum = st.alloc();
    st.push(elementwise_node(
        sum,
        binop_kernel("add", BinOpKind::Add),
        &[bcast_ms, bcast_e],
        full,
    ));

    let rsq = st.alloc();
    st.push(elementwise_node(
        rsq,
        unary_kernel("rsqrt", UnaryFnKind::Rsqrt),
        &[sum],
        full,
    ));

    let normed = st.alloc();
    st.push(elementwise_node(
        normed,
        binop_kernel("mul", BinOpKind::Mul),
        &[input, rsq],
        full,
    ));

    // Right-aligned broadcast for weight [hidden] -> [N, hidden]
    let bcast_w = st.alloc();
    st.push(broadcast_node(
        bcast_w,
        weight,
        full,
        BroadcastAlignment::Right,
    ));

    let out = st.alloc();
    st.push(elementwise_node(
        out,
        binop_kernel("mul", BinOpKind::Mul),
        &[normed, bcast_w],
        full,
    ));
    out
}

/// Emit affine transform: `gamma * x + beta` with channel reshape + broadcast.
///
/// Gamma/beta are `[C]`-shaped and need reshape to channel_3d then broadcast.
/// Returns the node ID of the final output.
pub(super) fn emit_affine_transform(
    st: &mut ExpandState,
    normed: usize,
    gamma: usize,
    beta: Option<usize>,
    full: &[usize],
    norm_axis: usize,
) -> usize {
    let channel_3d = make_channel_shape(full, norm_axis);

    // reshape gamma [C] -> channel_3d
    let gr = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(gr),
        TensorOpKind::Reshape {
            input: TensorNodeId::new(gamma),
            target_shape: channel_3d.clone(),
        },
        channel_3d.clone(),
    ));

    // broadcast gamma -> full
    let gb = st.alloc();
    st.push(broadcast_node(gb, gr, full, BroadcastAlignment::Left));

    // gamma * normed
    let scaled = st.alloc();
    st.push(elementwise_node(
        scaled,
        binop_kernel("mul", BinOpKind::Mul),
        &[normed, gb],
        full,
    ));

    if let Some(b) = beta {
        let br = st.alloc();
        st.push(TensorNode::new(
            TensorNodeId::new(br),
            TensorOpKind::Reshape {
                input: TensorNodeId::new(b),
                target_shape: channel_3d.clone(),
            },
            channel_3d,
        ));

        let bb = st.alloc();
        st.push(broadcast_node(bb, br, full, BroadcastAlignment::Left));

        let result = st.alloc();
        st.push(elementwise_node(
            result,
            binop_kernel("add", BinOpKind::Add),
            &[scaled, bb],
            full,
        ));
        result
    } else {
        scaled
    }
}

/// Build a channel-broadcast shape for affine parameters.
///
/// For shape `[B, C, T]` with axis=2, returns `[B, C, 1]` (1 at norm axis).
fn make_channel_shape(full: &[usize], norm_axis: usize) -> Vec<usize> {
    full.iter()
        .enumerate()
        .map(|(i, &d)| if i == norm_axis { 1 } else { d })
        .collect()
}
