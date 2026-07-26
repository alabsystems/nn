// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward rules for 2-D convolution (Conv2d).
//!
//! Extracted from `backward_rules_conv.rs` for 500-line compliance.
//! Conv1d rules remain in `backward_rules_conv.rs`.
//! ConvTranspose1d rules are in `backward_rules_conv_transpose.rs`.

use nn_core::dyn_tensor::DynTensor;

use crate::error::Result;
use crate::grad::GradStore;
use crate::op::Op;

use super::accumulate;

/// Backward rule for 2-D convolution.
///
/// grad_input = conv_transpose2d(grad_output, kernel)
/// grad_kernel = cross-correlation(input, grad_output)
pub(crate) fn backward_conv2d(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::Conv2d {
            input,
            kernel,
            padding,
            stride,
            dilation,
            groups,
        } => {
            // Compute output_padding so conv_transpose2d reconstructs the
            // original spatial dimensions. When stride > 1, integer division
            // in forward conv2d loses the remainder; output_padding restores it.
            let in_h = input.tensor().dims()[2];
            let in_w = input.tensor().dims()[3];
            let k_h = kernel.tensor().dims()[2];
            let k_w = kernel.tensor().dims()[3];

            let output_padding_h =
                compute_conv_output_padding(in_h, k_h, *padding, *stride, *dilation);
            let output_padding_w =
                compute_conv_output_padding(in_w, k_w, *padding, *stride, *dilation);

            // For conv_transpose2d we need symmetric output_padding; use max.
            // In practice H and W output_padding are computed independently, but
            // the DynTensor conv_transpose2d API takes a single output_padding
            // applied to both spatial dims. We compute per-dim and verify they
            // are equal (which they are when H/W use the same padding/stride/dilation).
            // If they differ, fall back to the kernel-grad-only path with manual
            // input gradient computation.
            let pad2 = [*padding, *padding];
            let stride2 = [*stride, *stride];
            let dilation2 = [*dilation, *dilation];
            if output_padding_h == output_padding_w {
                let grad_input = grad.conv_transpose2d(
                    kernel.tensor(),
                    pad2,
                    [output_padding_h, output_padding_w],
                    stride2,
                    dilation2,
                    *groups,
                )?;
                accumulate(input, &grad_input, grads)?;
            } else {
                // Asymmetric output padding: use max(op_h, op_w) then narrow.
                let grad_input = conv2d_input_grad_asymmetric(
                    input.tensor(),
                    kernel.tensor(),
                    grad,
                    pad2,
                    output_padding_h,
                    output_padding_w,
                    stride2,
                    dilation2,
                    *groups,
                )?;
                accumulate(input, &grad_input, grads)?;
            }

            let gk = conv2d_kernel_grad(
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

/// Compute output_padding for a single spatial dimension.
fn compute_conv_output_padding(
    input_len: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    let base = input_len + 2 * padding;
    let effective_k = dilation * (kernel_size - 1) + 1;
    if base >= effective_k {
        (base - effective_k) % stride
    } else {
        0
    }
}

/// Cross-correlation of input with grad_output to compute kernel gradient (2D).
///
/// GEMM-based implementation: im2col_2d(input) + matmul(columns^T, grad).
/// Device-agnostic — works on both CPU and GPU without explicit transfers.
fn conv2d_kernel_grad(
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
    let k_dims = kernel_data.dims();
    let out_ch = k_dims[0];
    let k_in_ch = k_dims[1];
    let k_h = k_dims[2];
    let k_w = k_dims[3];
    let (out_h, out_w) = (grad.dims()[2], grad.dims()[3]);
    let ch_per_group = in_ch / groups;
    let out_ch_per_group = out_ch / groups;
    let spatial_out = out_h * out_w;

    let mut group_grads = Vec::with_capacity(groups);
    for g in 0..groups {
        let input_g = in_data.narrow(1, g * ch_per_group, ch_per_group)?;
        let grad_g = grad.narrow(1, g * out_ch_per_group, out_ch_per_group)?;

        // im2col_2d: [B, ch/G * kH * kW, H_out * W_out]
        let columns = input_g.im2col_2d(k_h, k_w, stride, padding, dilation)?;

        // Merge batch into spatial: [B·H_out·W_out, ch/G·kH·kW]
        let col_2d = columns
            .transpose(1, 2)?
            .reshape([batch * spatial_out, ch_per_group * k_h * k_w])?;

        // grad_g: [B, oc/G, H_out, W_out] → [B, oc/G, H_out*W_out]
        //       → transpose → [B·H_out·W_out, oc/G]
        let grad_flat = grad_g
            .reshape([batch, out_ch_per_group, spatial_out])?
            .transpose(1, 2)?
            .reshape([batch * spatial_out, out_ch_per_group])?;

        // GEMM: [ch/G·kH·kW, B·spatial_out] @ [B·spatial_out, oc/G]
        let dw = col_2d.t()?.matmul(&grad_flat)?;

        // Reshape: [ch/G·kH·kW, oc/G] → [ch/G, kH, kW, oc/G] → permute → [oc/G, ch/G, kH, kW]
        let dw = dw
            .reshape([ch_per_group, k_h * k_w, out_ch_per_group])?
            .permute([2, 0, 1])?
            .reshape([out_ch_per_group, ch_per_group, k_h, k_w])?;

        group_grads.push(dw);
    }

    let _ = k_in_ch; // Used for documentation: k_in_ch == ch_per_group
    DynTensor::cat(&group_grads, 0).map_err(Into::into)
}

/// Device-agnostic input gradient for Conv2d with asymmetric output_padding.
///
/// Uses conv_transpose2d with the larger output_padding (producing an
/// oversized result), then narrows each spatial dimension to the original
/// input size. No as_cpu_f32 — fully device-agnostic.
fn conv2d_input_grad_asymmetric(
    in_data: &DynTensor,
    kernel_data: &DynTensor,
    grad: &DynTensor,
    padding: [usize; 2],
    output_padding_h: usize,
    output_padding_w: usize,
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Result<DynTensor> {
    let in_h = in_data.dims()[2];
    let in_w = in_data.dims()[3];

    // Use max output_padding so both spatial dims are >= target.
    let op_max = output_padding_h.max(output_padding_w);
    let oversized = grad.conv_transpose2d(
        kernel_data,
        padding,
        [op_max, op_max],
        stride,
        dilation,
        groups,
    )?;

    // Narrow to exact target dimensions.
    let result = oversized.narrow(2, 0, in_h)?.narrow(3, 0, in_w)?;
    Ok(result)
}
