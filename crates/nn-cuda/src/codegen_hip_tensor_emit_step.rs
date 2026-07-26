// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-step HIP dispatch: matches a `DispatchStep` and returns HIP C++ source.
//!
//! Parallel to `nn-dsl::codegen_msl_tensor_emit_step` — routes each dispatch
//! step variant to the appropriate HIP emitter function.

use crate::codegen_hip_tensor_emit_complex::{
    emit_embedding_kernel, emit_linear_kernel, emit_matmul_kernel, emit_softmax_kernel,
};
use crate::codegen_hip_tensor_emit_conv::{
    emit_conv1d_kernel, emit_conv2d_kernel, emit_conv_transpose1d_kernel,
};
use crate::codegen_hip_tensor_emit_elementwise::emit_elementwise_hip;
use crate::codegen_hip_tensor_emit_index::{
    emit_f32_to_u32_hip, emit_gather_hip, emit_index_select_hip, emit_zero_pad_1d_hip,
};
use crate::codegen_hip_tensor_emit_ops::{
    emit_binary_add_kernel, emit_binary_mul_kernel, emit_gelu_erf_kernel, emit_gelu_kernel,
    emit_relu_kernel, emit_sigmoid_kernel, emit_tanh_kernel,
};
use crate::codegen_hip_tensor_emit_select::{emit_axis_select_kernel, emit_stack_kernel};
use crate::codegen_hip_tensor_emit_structural::{
    emit_broadcast_kernel, emit_concat_kernel, emit_narrow_kernel, emit_reduce_kernel,
    emit_transpose_kernel,
};
use crate::HipCodegenError;
use nn_dsl::{DispatchStep, TensorKernelDef};

/// Emit HIP C++ source for a single dispatch step.
///
/// Returns `Ok(Some(hip_source))` for steps that produce HIP kernels,
/// `Ok(None)` for no-op steps (e.g., `Reshape`), or an error for
/// unsupported steps.
pub fn emit_step_hip(
    step: &DispatchStep,
    kernel: &TensorKernelDef,
) -> Result<Option<String>, HipCodegenError> {
    match step {
        DispatchStep::Reshape { .. } => Ok(None),

        DispatchStep::BinaryAdd {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => Ok(Some(emit_binary_add_kernel(
            kernel_name,
            *dtype,
            *total_elements,
        )?)),

        DispatchStep::BinaryMul {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => Ok(Some(emit_binary_mul_kernel(
            kernel_name,
            *dtype,
            *total_elements,
        )?)),

        DispatchStep::Sigmoid {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => Ok(Some(emit_sigmoid_kernel(
            kernel_name,
            *dtype,
            *total_elements,
        )?)),

        DispatchStep::Gelu {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => Ok(Some(emit_gelu_kernel(
            kernel_name,
            *dtype,
            *total_elements,
        )?)),

        DispatchStep::GeluErf {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => Ok(Some(emit_gelu_erf_kernel(
            kernel_name,
            *dtype,
            *total_elements,
        )?)),

        DispatchStep::Relu {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => Ok(Some(emit_relu_kernel(
            kernel_name,
            *dtype,
            *total_elements,
        )?)),

        DispatchStep::Tanh {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => Ok(Some(emit_tanh_kernel(
            kernel_name,
            *dtype,
            *total_elements,
        )?)),

        DispatchStep::Linear {
            kernel_name,
            dtype,
            in_features,
            out_features,
            bias,
            ..
        } => Ok(Some(emit_linear_kernel(
            kernel_name,
            *dtype,
            *in_features,
            *out_features,
            bias.is_some(),
        )?)),

        DispatchStep::MatMul {
            kernel_name,
            dtype,
            m,
            k,
            n,
            transpose_right,
            broadcast_right,
            scale,
            ..
        } => Ok(Some(emit_matmul_kernel(
            kernel_name,
            *dtype,
            *m,
            *k,
            *n,
            *transpose_right,
            *broadcast_right,
            *scale,
        )?)),

        DispatchStep::Softmax {
            kernel_name,
            dtype,
            output,
            axis,
            ..
        } => {
            let rank = kernel.nodes[output.index()].shape.len();
            if *axis + 1 != rank {
                return Err(HipCodegenError::NonLastAxisSoftmax {
                    node_id: *output,
                    axis: *axis,
                    shape: kernel.nodes[output.index()].shape.clone(),
                });
            }
            Ok(Some(emit_softmax_kernel(kernel_name, *dtype)?))
        }

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

        // Simdgroup variants: use rocWMMA tiled GEMM when dimensions are
        // aligned, fall back to naive kernels otherwise.
        DispatchStep::SimdgroupLinear(ref p) => {
            if crate::codegen_hip_tensor_emit_gemm::should_use_rocwmma(
                p.batch_size,
                p.in_features,
                p.out_features,
            ) {
                Ok(Some(
                    crate::codegen_hip_tensor_emit_gemm::emit_rocwmma_linear_kernel(
                        &p.kernel_name,
                        p.dtype,
                        p.in_features,
                        p.out_features,
                        p.batch_size,
                        p.bias.is_some(),
                    )?,
                ))
            } else {
                Ok(Some(emit_linear_kernel(
                    &p.kernel_name,
                    p.dtype,
                    p.in_features,
                    p.out_features,
                    p.bias.is_some(),
                )?))
            }
        }

        DispatchStep::SimdgroupMatMul(ref p) => {
            if crate::codegen_hip_tensor_emit_gemm::should_use_rocwmma(p.m, p.k, p.n) {
                Ok(Some(
                    crate::codegen_hip_tensor_emit_gemm::emit_rocwmma_matmul_kernel(
                        &p.kernel_name,
                        p.dtype,
                        p.m,
                        p.k,
                        p.n,
                        p.batch_size,
                        p.transpose_right,
                        p.broadcast_right,
                        p.scale,
                    )?,
                ))
            } else {
                Ok(Some(emit_matmul_kernel(
                    &p.kernel_name,
                    p.dtype,
                    p.m,
                    p.k,
                    p.n,
                    p.transpose_right,
                    p.broadcast_right,
                    p.scale,
                )?))
            }
        }

        // --- Reduce ---
        DispatchStep::Reduce {
            kernel_name,
            op,
            dtype,
            ..
        } => {
            // HIP PoC: only last-axis reductions (same as MSL fast path).
            // Multi-axis reduction decomposes to multiple last-axis steps
            // in the dispatch planner.
            Ok(Some(emit_reduce_kernel(kernel_name, *op, *dtype)?))
        }

        // --- Broadcast ---
        DispatchStep::Broadcast {
            kernel_name,
            dtype,
            input_shape,
            output_shape,
            alignment,
            ..
        } => Ok(Some(emit_broadcast_kernel(
            kernel_name,
            *dtype,
            input_shape,
            output_shape,
            *alignment,
        )?)),

        // --- Narrow ---
        DispatchStep::Narrow {
            kernel_name,
            dtype,
            input_shape,
            axis,
            start,
            length,
            ..
        } => Ok(Some(emit_narrow_kernel(
            kernel_name,
            *dtype,
            input_shape,
            *axis,
            *start,
            *length,
        )?)),

        // --- Transpose ---
        DispatchStep::Transpose {
            kernel_name,
            dtype,
            input_shape,
            axes,
            ..
        } => Ok(Some(emit_transpose_kernel(
            kernel_name,
            *dtype,
            input_shape,
            axes,
        )?)),

        // --- Concat ---
        DispatchStep::Concat {
            kernel_name,
            dtype,
            first_input_shape,
            input_axis_sizes,
            axis,
            ..
        } => Ok(Some(emit_concat_kernel(
            kernel_name,
            *dtype,
            first_input_shape,
            input_axis_sizes,
            *axis,
        )?)),

        // --- Conv1d ---
        DispatchStep::Conv1d(ref p) => Ok(Some(emit_conv1d_kernel(
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
        )?)),

        // --- Conv2d ---
        DispatchStep::Conv2d(ref p) => Ok(Some(emit_conv2d_kernel(
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
        )?)),

        // --- ConvTranspose1d ---
        DispatchStep::ConvTranspose1d(ref p) => Ok(Some(emit_conv_transpose1d_kernel(
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
        )?)),

        // --- AxisSelect ---
        DispatchStep::AxisSelect {
            kernel_name,
            dtype,
            input_shape,
            axis,
            index,
            ..
        } => Ok(Some(emit_axis_select_kernel(
            kernel_name,
            *dtype,
            input_shape,
            *axis,
            *index,
        )?)),

        // --- Stack ---
        DispatchStep::Stack {
            kernel_name,
            dtype,
            inputs,
            input_shape,
            axis,
            ..
        } => Ok(Some(emit_stack_kernel(
            kernel_name,
            *dtype,
            input_shape,
            inputs.len(),
            *axis,
        )?)),

        // --- Elementwise (composed KernelDef IR) ---
        DispatchStep::Elementwise { scalar_kernel, .. } => {
            Ok(Some(emit_elementwise_hip(scalar_kernel)?))
        }

        // --- ZeroPad1d ---
        DispatchStep::ZeroPad1d {
            kernel_name,
            dtype,
            channels,
            in_length,
            pad_left,
            out_length,
            ..
        } => Ok(Some(emit_zero_pad_1d_hip(
            kernel_name,
            *dtype,
            *channels,
            *in_length,
            *pad_left,
            *out_length,
        )?)),

        // --- IndexSelect ---
        DispatchStep::IndexSelect {
            kernel_name,
            dtype,
            input_shape,
            dim,
            ..
        } => {
            let main = emit_index_select_hip(kernel_name, *dtype, input_shape, *dim)?;
            let (_, conv) = emit_f32_to_u32_hip(kernel_name);
            Ok(Some(format!("{conv}\n\n{main}")))
        }

        // --- Gather ---
        DispatchStep::Gather {
            kernel_name,
            dtype,
            input_shape,
            dim,
            ..
        } => {
            let main = emit_gather_hip(kernel_name, *dtype, input_shape, *dim)?;
            let (_, conv) = emit_f32_to_u32_hip(kernel_name);
            Ok(Some(format!("{conv}\n\n{main}")))
        }
        // Catch-all for future DispatchStep variants (#[non_exhaustive]).
        _ => Err(HipCodegenError::UnsupportedStep {
            step_name: "unknown",
        }),
    }
}
