// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Output buffer size estimation for individual dispatch steps.
//!
//! Part of the peak memory model (#1739 Phase 19).

use nn_dsl::DispatchStep;

const F32_BYTES: u64 = 4;

/// Compute the output buffer size in bytes for a single dispatch step.
///
/// Unlike [`super::step_memory_bytes`] which counts total traffic (reads + writes),
/// this computes the size of the output tensor that must remain allocated
/// until consumed by a downstream step. Used for peak memory estimation.
///
/// All sizes assume fp32 (4 bytes per element).
pub fn step_output_bytes(step: &DispatchStep) -> u64 {
    match step {
        DispatchStep::Linear { total_elements, .. }
        | DispatchStep::MatMul { total_elements, .. }
        | DispatchStep::Sigmoid { total_elements, .. }
        | DispatchStep::Gelu { total_elements, .. }
        | DispatchStep::Relu { total_elements, .. }
        | DispatchStep::Tanh { total_elements, .. }
        | DispatchStep::BinaryAdd { total_elements, .. }
        | DispatchStep::BinaryMul { total_elements, .. }
        | DispatchStep::Elementwise { total_elements, .. }
        | DispatchStep::Broadcast { total_elements, .. }
        | DispatchStep::Transpose { total_elements, .. } => (*total_elements as u64) * F32_BYTES,

        DispatchStep::Conv1d(p) => (p.total_elements as u64) * F32_BYTES,
        DispatchStep::Conv2d(p) => (p.total_elements as u64) * F32_BYTES,
        DispatchStep::ConvTranspose1d(p) => (p.total_elements as u64) * F32_BYTES,

        DispatchStep::Softmax {
            axis_size,
            outer_size,
            ..
        } => (*axis_size as u64) * (*outer_size as u64) * F32_BYTES,

        DispatchStep::Reduce { outer_size, .. } => (*outer_size as u64) * F32_BYTES,

        DispatchStep::Embedding { total_elements, .. } => (*total_elements as u64) * F32_BYTES,

        DispatchStep::Reshape { .. } => 0, // No new allocation

        DispatchStep::AxisSelect {
            input_shape, axis, ..
        } => {
            if *axis < input_shape.len() && input_shape[*axis] > 0 {
                let total: u64 = input_shape.iter().map(|&d| d as u64).product();
                (total / (input_shape[*axis] as u64)) * F32_BYTES
            } else {
                0
            }
        }

        DispatchStep::Stack {
            input_shape,
            inputs,
            ..
        } => {
            let per_input: u64 = input_shape.iter().map(|&d| d as u64).product();
            (inputs.len() as u64) * per_input * F32_BYTES
        }

        DispatchStep::Narrow {
            input_shape,
            length,
            axis,
            ..
        } => {
            if *axis < input_shape.len() && input_shape[*axis] > 0 {
                let total: u64 = input_shape.iter().map(|&d| d as u64).product();
                (total / (input_shape[*axis] as u64)) * (*length as u64) * F32_BYTES
            } else {
                0
            }
        }

        DispatchStep::ZeroPad1d {
            channels,
            out_length,
            ..
        } => (*channels as u64) * (*out_length as u64) * F32_BYTES,

        DispatchStep::Concat {
            first_input_shape,
            input_axis_sizes,
            axis,
            ..
        } => {
            let non_axis_product: u64 = first_input_shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != *axis)
                .map(|(_, &d)| d as u64)
                .product();
            let total_axis: u64 = input_axis_sizes.iter().map(|&s| s as u64).sum();
            non_axis_product * total_axis * F32_BYTES
        }

        // Unknown future variants: conservative estimate of 0.
        // Always warn — silent 0 in release builds hides missing coverage.
        _ => {
            eprintln!("[cost_model] step_output_bytes: unhandled DispatchStep variant: {step:?}");
            0
        }
    }
}
