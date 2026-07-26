// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL emission for unary activation dispatch steps.
//!
//! Extracted from `codegen_msl_tensor_emit_step.rs` to keep that file under
//! the 450-line limit. All functions are `pub(super)` — called from the
//! `emit_step_msl` match in the parent `step` module.

use crate::codegen_msl_tensor::{DispatchStep, TensorMSLCodegenError};

use super::super::ops::{
    emit_elu_kernel, emit_exp_kernel, emit_gelu_erf_kernel, emit_gelu_kernel,
    emit_leaky_relu_kernel, emit_relu_kernel, emit_sigmoid_kernel, emit_softplus_kernel,
    emit_tanh_kernel,
};

/// Emit MSL for one of the 9 activation dispatch steps.
///
/// Caller matches the grouped activation variants and delegates here.
pub(super) fn emit_activation_msl(step: &DispatchStep) -> Result<String, TensorMSLCodegenError> {
    match step {
        DispatchStep::Sigmoid {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => emit_sigmoid_kernel(kernel_name, *dtype, *total_elements),
        DispatchStep::Gelu {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => emit_gelu_kernel(kernel_name, *dtype, *total_elements),
        DispatchStep::GeluErf {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => emit_gelu_erf_kernel(kernel_name, *dtype, *total_elements),
        DispatchStep::Relu {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => emit_relu_kernel(kernel_name, *dtype, *total_elements),
        DispatchStep::Tanh {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => emit_tanh_kernel(kernel_name, *dtype, *total_elements),
        DispatchStep::LeakyRelu {
            kernel_name,
            dtype,
            total_elements,
            negative_slope,
            ..
        } => emit_leaky_relu_kernel(kernel_name, *dtype, *total_elements, *negative_slope),
        DispatchStep::Elu {
            kernel_name,
            dtype,
            total_elements,
            alpha,
            ..
        } => emit_elu_kernel(kernel_name, *dtype, *total_elements, *alpha),
        DispatchStep::Exp {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => emit_exp_kernel(kernel_name, *dtype, *total_elements),
        DispatchStep::Softplus {
            kernel_name,
            dtype,
            total_elements,
            ..
        } => emit_softplus_kernel(kernel_name, *dtype, *total_elements),
        _ => unreachable!("emit_activation_msl called with non-activation step"),
    }
}
