// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Standalone backward functions for Conv1d and Scaled Dot-Product Attention.
//!
//! These are public utilities for manual gradient computation outside the
//! automatic differentiation tape. The tape-based backward dispatch uses
//! `backward_rules_conv.rs` (via `backward_rules.rs`); this module provides
//! the same math as callable functions.
//!
//! Conv1d backward:
//!   grad_input  = conv_transpose1d(grad_output, weight)
//!   grad_weight = cross_correlate(input, grad_output) via im2col + GEMM
//!
//! Scaled dot-product attention backward:
//!   Given Q, K, V with scores = softmax(Q @ K^T / sqrt(d_k)) @ V
//!   grad_Q = (dS @ K) / sqrt(d_k)
//!   grad_K = (dS^T @ Q) / sqrt(d_k)
//!   grad_V = attn_weights^T @ grad_output
//!   where dS = (grad_output @ V^T) * attn_weights - attn_weights * rowsum(...)

use nn_core::dyn_tensor::DynTensor;

use crate::error::{AutodiffError, Result};

/// Backward rule for Conv1d.
///
/// Given forward: `output = conv1d(input, weight, stride, padding)`
///   - `grad_input = conv_transpose1d(grad_output, weight)`
///   - `grad_weight = cross_correlate(input, grad_output)` via im2col + GEMM
///
/// Supports groups=1, dilation=1. For grouped/dilated convolutions, use the
/// tape-based backward via `TrackedTensor::conv1d`.
///
/// # Arguments
/// - `grad_output`: gradient of the loss w.r.t. conv1d output, shape `[B, out_ch, L_out]`
/// - `input`: the original input tensor, shape `[B, in_ch, L_in]`
/// - `weight`: the convolution kernel, shape `[out_ch, in_ch, K]`
/// - `stride`: convolution stride
/// - `padding`: zero-padding on each side
///
/// # Returns
/// `(grad_input, grad_weight)` where:
/// - `grad_input` has the same shape as `input`: `[B, in_ch, L_in]`
/// - `grad_weight` has the same shape as `weight`: `[out_ch, in_ch, K]`
pub fn conv1d_backward(
    grad_output: &DynTensor,
    input: &DynTensor,
    weight: &DynTensor,
    stride: usize,
    padding: usize,
) -> Result<(DynTensor, DynTensor)> {
    // Validate ranks.
    if input.rank() != 3 {
        return Err(AutodiffError::WrongInputRank {
            op: "conv1d_backward",
            expected: 3,
            actual: input.rank(),
        });
    }
    if weight.rank() != 3 {
        return Err(AutodiffError::WrongInputRank {
            op: "conv1d_backward(weight)",
            expected: 3,
            actual: weight.rank(),
        });
    }
    if grad_output.rank() != 3 {
        return Err(AutodiffError::WrongInputRank {
            op: "conv1d_backward(grad_output)",
            expected: 3,
            actual: grad_output.rank(),
        });
    }

    let dilation = 1;
    let groups = 1;

    // --- grad_input via conv_transpose1d ---
    // Compute output_padding to reconstruct original input length.
    let in_len = input.dims()[2];
    let k_size = weight.dims()[2];
    let base = in_len + 2 * padding;
    let effective_k = dilation * (k_size - 1) + 1;
    let output_padding = if base >= effective_k {
        (base - effective_k) % stride
    } else {
        0
    };

    let grad_input =
        grad_output.conv_transpose1d(weight, padding, output_padding, stride, dilation, groups)?;

    // --- grad_weight via im2col + GEMM ---
    let grad_weight = conv1d_kernel_grad(
        input,
        weight,
        grad_output,
        padding,
        stride,
        dilation,
        groups,
    )?;

    Ok((grad_input, grad_weight))
}

/// Cross-correlation of input with grad_output to compute kernel gradient (1D).
///
/// GEMM-based: im2col(input) + matmul(columns^T, grad).
fn conv1d_kernel_grad(
    in_data: &DynTensor,
    kernel_data: &DynTensor,
    grad: &DynTensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<DynTensor> {
    let batch = in_data.dims()[0];
    let in_ch = in_data.dims()[1];
    let out_ch = kernel_data.dims()[0];
    let k_size = kernel_data.dims()[2];
    let out_len = grad.dims()[2];
    let ch_per_group = in_ch / groups;
    let out_ch_per_group = out_ch / groups;

    let mut group_grads = Vec::with_capacity(groups);
    for g in 0..groups {
        // Extract group slices along channel dimension.
        let input_g = in_data.narrow(1, g * ch_per_group, ch_per_group)?;
        let grad_g = grad.narrow(1, g * out_ch_per_group, out_ch_per_group)?;

        // im2col: [B, ch_per_group * K, L_out]
        let columns = input_g.im2col_1d(k_size, stride, padding, dilation)?;

        // Reshape to merge batch into spatial dimension for 2D matmul:
        // columns: [B, ch/G*K, L_out] -> transpose(1,2) -> [B, L_out, ch/G*K]
        //        -> reshape [B*L_out, ch/G*K]
        let col_2d = columns
            .transpose(1, 2)?
            .reshape([batch * out_len, ch_per_group * k_size])?;

        // grad_g: [B, oc/G, L_out] -> transpose(1,2) -> [B, L_out, oc/G]
        //       -> reshape [B*L_out, oc/G]
        let grad_2d = grad_g
            .transpose(1, 2)?
            .reshape([batch * out_len, out_ch_per_group])?;

        // GEMM: [ch/G*K, B*L_out] @ [B*L_out, oc/G] = [ch/G*K, oc/G]
        let dw = col_2d.t()?.matmul(&grad_2d)?;

        // Reshape to kernel layout: [ch/G, K, oc/G] -> permute -> [oc/G, ch/G, K]
        let dw = dw
            .reshape([ch_per_group, k_size, out_ch_per_group])?
            .permute([2, 0, 1])?;

        group_grads.push(dw);
    }

    // Concatenate groups along out_ch dimension: [out_ch, ch/G, K]
    DynTensor::cat(&group_grads, 0).map_err(Into::into)
}

/// Backward rule for Scaled Dot-Product Attention (SDPA).
///
/// Given forward:
///   `scores = Q @ K^T / sqrt(d_k)`
///   `attn_weights = softmax(scores, dim=-1)`
///   `output = attn_weights @ V`
///
/// Computes:
///   - `grad_q`: gradient w.r.t. query tensor
///   - `grad_k`: gradient w.r.t. key tensor
///   - `grad_v`: gradient w.r.t. value tensor
///
/// # Arguments
/// - `grad_output`: gradient of loss w.r.t. attention output, shape `[B, H, S_q, d_k]`
/// - `query`: query tensor, shape `[B, H, S_q, d_k]`
/// - `key`: key tensor, shape `[B, H, S_kv, d_k]`
/// - `value`: value tensor, shape `[B, H, S_kv, d_k]`
///
/// # Returns
/// `(grad_q, grad_k, grad_v)` with shapes matching their respective inputs.
pub fn scaled_dot_product_attention_backward(
    grad_output: &DynTensor,
    query: &DynTensor,
    key: &DynTensor,
    value: &DynTensor,
) -> Result<(DynTensor, DynTensor, DynTensor)> {
    // Validate ranks: all must be 4D [B, H, S, d_k].
    for (_name, t) in [
        ("grad_output", grad_output),
        ("query", query),
        ("key", key),
        ("value", value),
    ] {
        if t.rank() != 4 {
            return Err(AutodiffError::WrongInputRank {
                op: "sdpa_backward",
                expected: 4,
                actual: t.rank(),
            });
        }
    }

    let d_k = query.dims()[3];
    let scale = 1.0 / (d_k as f64).sqrt();

    // Recompute forward attention weights.
    // scores = Q @ K^T / sqrt(d_k)  shape: [B, H, S_q, S_kv]
    let k_t = key.transpose(2, 3)?;
    let scores = query.matmul(&k_t)?.mul_scalar(scale)?;

    // attn_weights = softmax(scores, dim=-1)  shape: [B, H, S_q, S_kv]
    let attn_weights = scores.softmax(3)?;

    // --- grad_v = attn_weights^T @ grad_output ---
    // attn_weights^T: [B, H, S_kv, S_q]
    // grad_output: [B, H, S_q, d_k]
    // grad_v: [B, H, S_kv, d_k]
    let attn_t = attn_weights.transpose(2, 3)?;
    let grad_v = attn_t.matmul(grad_output)?;

    // --- Backprop through attn_weights @ V ---
    // d_attn = grad_output @ V^T  shape: [B, H, S_q, S_kv]
    let v_t = value.transpose(2, 3)?;
    let d_attn = grad_output.matmul(&v_t)?;

    // --- Backprop through softmax ---
    // d_scores = attn_weights * (d_attn - sum(d_attn * attn_weights, dim=-1, keepdim=True))
    let dot = d_attn.mul(&attn_weights)?.sum_keepdim(3)?;
    let d_scores = attn_weights.mul(&d_attn.sub(&dot.expand(d_attn.dims())?)?)?;

    // --- Backprop through scaling ---
    // d_scores_scaled = d_scores / sqrt(d_k)
    // But we already applied scale in forward, so d_scores needs to be scaled.
    let d_scores_scaled = d_scores.mul_scalar(scale)?;

    // --- grad_q = d_scores_scaled @ K ---
    // d_scores_scaled: [B, H, S_q, S_kv]
    // K: [B, H, S_kv, d_k]
    // grad_q: [B, H, S_q, d_k]
    let grad_q = d_scores_scaled.matmul(key)?;

    // --- grad_k = d_scores_scaled^T @ Q ---
    // d_scores_scaled^T: [B, H, S_kv, S_q]
    // Q: [B, H, S_q, d_k]
    // grad_k: [B, H, S_kv, d_k]
    let d_scores_t = d_scores_scaled.transpose(2, 3)?;
    let grad_k = d_scores_t.matmul(query)?;

    Ok((grad_q, grad_k, grad_v))
}

#[cfg(test)]
#[path = "backward_rules_conv_attn_tests.rs"]
mod tests;
