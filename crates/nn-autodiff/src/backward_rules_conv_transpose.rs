// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward rules for transposed convolution operations (ConvTranspose1d).
//!
//! Extracted from `backward_rules_conv.rs` for 500-line compliance.

use nn_core::dyn_tensor::DynTensor;

use crate::error::Result;
use crate::grad::GradStore;
use crate::op::Op;

use super::accumulate;

/// Backward rule for 1-D transposed convolution.
///
/// ConvTranspose1d backward: grad_input = conv1d(grad_output, kernel),
/// grad_kernel = im2col(grad)^T @ input (GEMM-based, device-agnostic).
/// This is the dual of Conv1d: the backward of ConvTranspose1d is Conv1d.
pub(crate) fn backward_conv_transpose1d(
    op: &Op,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    match op {
        Op::ConvTranspose1d {
            input,
            kernel,
            padding,
            stride,
            dilation,
            groups,
            output_padding,
        } => {
            let _ = output_padding;
            let grad_input = grad.conv1d(kernel.tensor(), *padding, *stride, *dilation, *groups)?;
            accumulate(input, &grad_input, grads)?;

            // grad_kernel: GEMM-based cross-correlation.
            // ConvTranspose1d kernel layout: [in_ch, oc/G, K]
            //
            // im2col_1d(grad_g, K, stride, padding, dilation) produces [B, oc/G*K, in_len]
            // because the ConvTranspose1d output length maps back to in_len through
            // the conv1d_out_len formula applied to grad's spatial dimension.
            //
            // dW = input_2d^T @ grad_col_2d → [in_ch/G, oc/G*K] → reshape [in_ch/G, oc/G, K]
            let in_data = input.tensor();
            let k_data = kernel.tensor();
            let k_dims = k_data.dims();

            let batch = in_data.dims()[0];
            let in_ch = k_dims[0];
            let ch_per_group = k_dims[1]; // oc_per_group in ConvTranspose1d kernel
            let k_size = k_dims[2];
            let in_ch_per_group = in_ch / *groups;
            let in_len = in_data.dims()[2];
            let out_ch_total = grad.dims()[1];
            let out_ch_per_group = out_ch_total / *groups;

            let mut group_grads = Vec::with_capacity(*groups);
            for g in 0..*groups {
                // input_g: [B, in_ch/G, in_len]
                let input_g = in_data.narrow(1, g * in_ch_per_group, in_ch_per_group)?;
                // grad_g: [B, oc/G, out_len]
                let grad_g = grad.narrow(1, g * out_ch_per_group, out_ch_per_group)?;

                // im2col on grad: [B, oc/G * K, in_len]
                let grad_columns = grad_g.im2col_1d(k_size, *stride, *padding, *dilation)?;

                // Merge batch+spatial for 2D matmul:
                // grad_col: [B, oc/G*K, in_len] → transpose → [B, in_len, oc/G*K]
                //         → reshape → [B*in_len, oc/G*K]
                let grad_col_2d = grad_columns
                    .transpose(1, 2)?
                    .reshape([batch * in_len, ch_per_group * k_size])?;

                // input_g: [B, in_ch/G, in_len] → transpose → [B, in_len, in_ch/G]
                //        → reshape → [B*in_len, in_ch/G]
                let input_2d = input_g
                    .transpose(1, 2)?
                    .reshape([batch * in_len, in_ch_per_group])?;

                // GEMM: [in_ch/G, B*in_len] @ [B*in_len, oc/G*K] = [in_ch/G, oc/G*K]
                let dw = input_2d.t()?.matmul(&grad_col_2d)?;

                // Reshape to ConvTranspose1d kernel layout: [in_ch/G, oc/G, K]
                let dw = dw.reshape([in_ch_per_group, ch_per_group, k_size])?;
                group_grads.push(dw);
            }

            // Concatenate groups along in_ch dimension: [in_ch, oc/G, K]
            let gk = DynTensor::cat(&group_grads, 0)?;
            accumulate(kernel, &gk, grads)?;
            Ok(())
        }
        other => Err(super::unsupported(other)),
    }
}
