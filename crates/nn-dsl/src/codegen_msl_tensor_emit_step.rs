// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-step MSL dispatch: matches a single `DispatchStep` and returns its MSL.
//!
//! Extracted from the `emit_tensor_msl_with_contract` match body in
//! `codegen_msl_tensor_emit.rs` to keep that file under the 500-line limit.

use crate::codegen_msl;
use crate::codegen_msl_structural;
use crate::codegen_msl_tensor::{DispatchStep, TensorMSLCodegenError};
use crate::precision::PrecisionContract;
use crate::tensor_ir::TensorKernelDef;

use super::complex::{emit_embedding_kernel, emit_softmax_kernel};
use super::emit_reduce_kernel;
use super::index::{emit_f32_to_u32_kernel, emit_gather_kernel, emit_index_select_kernel};
use super::ops::{emit_binary_add_kernel, emit_binary_mul_kernel, emit_zero_pad_1d_kernel};

#[path = "codegen_msl_tensor_emit_step_activation.rs"]
mod activation;
#[path = "codegen_msl_tensor_emit_step_gemm.rs"]
mod gemm_step;

/// Emit MSL source for a single dispatch step.
///
/// Returns `Ok(Some(msl))` for steps that produce MSL, `Ok(None)` for no-op
/// steps (e.g., `Reshape`), or an error if codegen fails.
pub(super) fn emit_step_msl(
    step: &DispatchStep,
    kernel: &TensorKernelDef,
    contract: PrecisionContract,
) -> Result<Option<String>, TensorMSLCodegenError> {
    match step {
        DispatchStep::Reduce {
            kernel_name,
            op,
            dtype,
            ..
        } => Ok(Some(emit_reduce_kernel(kernel_name, *op, *dtype, contract))),
        DispatchStep::Elementwise { scalar_kernel, .. } => {
            if scalar_kernel.params.len() > codegen_msl::MAX_DIRECT_BINDING_INPUTS {
                // Packed variant: reads all params from a single buffer via offsets.
                // Part of #1649.
                Ok(Some(codegen_msl::emit_msl_packed_with_contract(
                    scalar_kernel,
                    contract,
                )?))
            } else {
                Ok(Some(codegen_msl::emit_msl_with_contract(
                    scalar_kernel,
                    contract,
                )?))
            }
        }
        DispatchStep::Broadcast {
            kernel_name,
            dtype,
            input_shape,
            output_shape,
            alignment,
            ..
        } => Ok(Some(super::emit_broadcast_kernel(
            kernel_name,
            *dtype,
            input_shape,
            output_shape,
            *alignment,
        )?)),
        DispatchStep::Reshape { .. } => Ok(None),
        DispatchStep::AxisSelect {
            kernel_name,
            dtype,
            input_shape,
            axis,
            index,
            ..
        } => Ok(Some(codegen_msl_structural::emit_axis_select_kernel(
            kernel_name,
            *dtype,
            input_shape,
            *axis,
            *index,
        )?)),
        DispatchStep::Stack {
            kernel_name,
            dtype,
            inputs,
            input_shape,
            axis,
            ..
        } => Ok(Some(codegen_msl_structural::emit_stack_kernel(
            kernel_name,
            *dtype,
            input_shape,
            inputs.len(),
            *axis,
        )?)),
        DispatchStep::Conv1d(ref p) => Ok(Some(
            crate::codegen_msl_structural_conv::emit_conv1d_kernel(
                &p.kernel_name,
                p.dtype,
                p.in_channels,
                p.out_channels,
                p.kernel_size,
                p.in_length,
                p.stride,
                p.padding,
                p.dilation,
                p.groups,
                p.bias.is_some(),
            )?,
        )),
        DispatchStep::Conv2d(ref p) => Ok(Some(
            crate::codegen_msl_structural_conv::emit_conv2d_kernel(
                &p.kernel_name,
                p.dtype,
                p.in_channels,
                p.out_channels,
                p.kernel_h,
                p.kernel_w,
                p.in_height,
                p.in_width,
                p.stride_h,
                p.stride_w,
                p.padding_h,
                p.padding_w,
                p.dilation_h,
                p.dilation_w,
                p.groups,
                p.bias.is_some(),
            )?,
        )),
        DispatchStep::ConvTranspose1d(ref p) => Ok(Some(
            crate::codegen_msl_structural_conv::emit_conv_transpose_1d_kernel(
                &p.kernel_name,
                p.dtype,
                p.in_channels,
                p.out_channels,
                p.kernel_size,
                p.in_length,
                p.stride,
                p.padding,
                p.dilation,
                p.groups,
                p.output_padding,
                p.bias.is_some(),
            )?,
        )),
        DispatchStep::BinaryAdd {
            kernel_name,
            dtype,
            total_elements,
            broadcast,
            ..
        } => Ok(Some(emit_binary_add_kernel(
            kernel_name,
            *dtype,
            *total_elements,
            broadcast.as_ref(),
        )?)),
        DispatchStep::BinaryMul {
            kernel_name,
            dtype,
            total_elements,
            broadcast,
            ..
        } => Ok(Some(emit_binary_mul_kernel(
            kernel_name,
            *dtype,
            *total_elements,
            broadcast.as_ref(),
        )?)),
        // Activations (9 variants) — delegated to submodule.
        DispatchStep::Sigmoid { .. }
        | DispatchStep::Gelu { .. }
        | DispatchStep::GeluErf { .. }
        | DispatchStep::Relu { .. }
        | DispatchStep::Tanh { .. }
        | DispatchStep::LeakyRelu { .. }
        | DispatchStep::Elu { .. }
        | DispatchStep::Exp { .. }
        | DispatchStep::Softplus { .. } => Ok(Some(activation::emit_activation_msl(step)?)),
        DispatchStep::Narrow {
            kernel_name,
            dtype,
            input_shape,
            axis,
            start,
            length,
            ..
        } => Ok(Some(codegen_msl_structural::emit_narrow_kernel(
            kernel_name,
            *dtype,
            input_shape,
            *axis,
            *start,
            *length,
        )?)),
        DispatchStep::ZeroPad1d {
            kernel_name,
            dtype,
            channels,
            in_length,
            pad_left,
            out_length,
            ..
        } => Ok(Some(emit_zero_pad_1d_kernel(
            kernel_name,
            *dtype,
            *channels,
            *in_length,
            *pad_left,
            *out_length,
        )?)),
        DispatchStep::Softmax {
            kernel_name,
            dtype,
            output,
            axis,
            axis_size,
            outer_size,
            ..
        } => {
            let rank = kernel.nodes[output.index()].shape.len();
            if *axis + 1 != rank {
                return Err(TensorMSLCodegenError::NonLastAxisSoftmax {
                    node_id: *output,
                    axis: *axis,
                    shape: kernel.nodes[output.index()].shape.clone(),
                });
            }
            let _ = (*axis_size, *outer_size);
            Ok(Some(emit_softmax_kernel(kernel_name, *dtype)))
        }
        // GEMM (6 variants) — delegated to submodule.
        DispatchStep::Linear { .. }
        | DispatchStep::MatMul { .. }
        | DispatchStep::SimdgroupLinear(..)
        | DispatchStep::SimdgroupMatMul(..)
        | DispatchStep::TiledLinear(..)
        | DispatchStep::TiledMatMul(..) => Ok(Some(gemm_step::emit_gemm_msl(step)?)),
        DispatchStep::Embedding {
            kernel_name,
            dtype,
            embedding_dim,
            ..
        } => Ok(Some(emit_embedding_kernel(
            kernel_name,
            *dtype,
            *embedding_dim,
        )?)),
        DispatchStep::Transpose {
            kernel_name,
            dtype,
            input_shape,
            axes,
            total_elements,
            ..
        } => {
            if step.tiled_transpose_params().is_some() {
                Ok(Some(
                    codegen_msl_structural::emit_tiled_transpose_2d_kernel(kernel_name, *dtype),
                ))
            } else {
                Ok(Some(codegen_msl_structural::emit_transpose_kernel(
                    kernel_name,
                    *dtype,
                    input_shape,
                    axes,
                    *total_elements,
                )?))
            }
        }
        DispatchStep::Concat {
            kernel_name,
            dtype,
            first_input_shape,
            input_axis_sizes,
            axis,
            ..
        } => Ok(Some(codegen_msl_structural::emit_concat_kernel(
            kernel_name,
            *dtype,
            first_input_shape,
            input_axis_sizes,
            *axis,
        )?)),
        DispatchStep::IndexSelect {
            kernel_name,
            dtype,
            input_shape,
            dim,
            ..
        } => {
            let main = emit_index_select_kernel(kernel_name, *dtype, input_shape, *dim)?;
            let (_, conv) = emit_f32_to_u32_kernel(kernel_name);
            Ok(Some(format!("{conv}\n\n{main}")))
        }
        DispatchStep::Gather {
            kernel_name,
            dtype,
            input_shape,
            dim,
            ..
        } => {
            let main = emit_gather_kernel(kernel_name, *dtype, input_shape, *dim)?;
            let (_, conv) = emit_f32_to_u32_kernel(kernel_name);
            Ok(Some(format!("{conv}\n\n{main}")))
        }
    }
}
