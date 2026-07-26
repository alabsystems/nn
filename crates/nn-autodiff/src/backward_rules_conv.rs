// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward rules for 1-D convolution (Conv1d).
//!
//! Conv2d rules are in `backward_rules_conv2d.rs`.
//! ConvTranspose1d rules are in `backward_rules_conv_transpose.rs`.

use nn_core::dyn_tensor::DynTensor;

use crate::error::Result;
use crate::grad::GradStore;
use crate::op::Op;

use super::accumulate;

/// Backward rule for 1-D convolution.
///
/// grad_input = conv_transpose1d(grad_output, kernel)
/// grad_kernel = cross-correlation(input, grad_output)
pub(crate) fn backward_conv1d(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::Conv1d {
            input,
            kernel,
            padding,
            stride,
            dilation,
            groups,
        } => {
            // Compute output_padding so conv_transpose1d reconstructs the
            // original input length. When stride > 1, integer division in
            // forward conv1d loses the remainder; output_padding restores it.
            let in_len = input.tensor().dims()[2];
            let k_size = kernel.tensor().dims()[2];
            let base = in_len + 2 * padding;
            let effective_k = dilation * (k_size - 1) + 1;
            let output_padding = if base >= effective_k {
                (base - effective_k) % stride
            } else {
                0
            };
            let grad_input = grad.conv_transpose1d(
                kernel.tensor(),
                *padding,
                output_padding,
                *stride,
                *dilation,
                *groups,
            )?;
            accumulate(input, &grad_input, grads)?;

            let gk = conv1d_kernel_grad(
                input.tensor(),
                kernel.tensor(),
                grad,
                *padding,
                *stride,
                *dilation,
                *groups,
            )?;
            accumulate(kernel, &gk, grads)
        }
        other => Err(super::unsupported(other)),
    }
}

/// Cross-correlation of input with grad_output to compute kernel gradient (1D).
///
/// GEMM-based implementation: im2col(input) + matmul(columns^T, grad).
/// Device-agnostic — works on both CPU and GPU without explicit transfers.
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
        // Extract group slices along channel dimension
        let input_g = in_data.narrow(1, g * ch_per_group, ch_per_group)?;
        let grad_g = grad.narrow(1, g * out_ch_per_group, out_ch_per_group)?;

        // im2col: [B, ch_per_group * K, L_out]
        let columns = input_g.im2col_1d(k_size, stride, padding, dilation)?;

        // Reshape to merge batch into spatial dimension for 2D matmul:
        // columns: [B, ch/G·K, L_out] → transpose(1,2) → [B, L_out, ch/G·K]
        //        → reshape [B·L_out, ch/G·K]
        let col_2d = columns
            .transpose(1, 2)?
            .reshape([batch * out_len, ch_per_group * k_size])?;

        // grad_g: [B, oc/G, L_out] → transpose(1,2) → [B, L_out, oc/G]
        //       → reshape [B·L_out, oc/G]
        let grad_2d = grad_g
            .transpose(1, 2)?
            .reshape([batch * out_len, out_ch_per_group])?;

        // GEMM: [ch/G·K, B·L_out] @ [B·L_out, oc/G] = [ch/G·K, oc/G]
        let dw = col_2d.t()?.matmul(&grad_2d)?;

        // Reshape to kernel layout: [ch/G, K, oc/G] → permute → [oc/G, ch/G, K]
        let dw = dw
            .reshape([ch_per_group, k_size, out_ch_per_group])?
            .permute([2, 0, 1])?;

        group_grads.push(dw);
    }

    // Concatenate groups along out_ch dimension: [out_ch, ch/G, K]
    DynTensor::cat(&group_grads, 0).map_err(Into::into)
}
