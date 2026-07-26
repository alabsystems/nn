// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Memory traffic estimation for individual dispatch steps.
//!
//! Extracted from `cost_model_ops.rs` to keep files under the 500-line limit.
//! Part of the computational cost model (#1739).

use nn_dsl::{Conv1dParams, Conv2dParams, ConvTranspose1dParams, DispatchStep};

use super::F32_BYTES;

/// Compute estimated memory traffic in bytes for a single dispatch step.
///
/// Counts read bytes + write bytes for fp32 (4 bytes per element).
/// For weight-based ops (Linear, Conv), includes weight reads.
pub fn step_memory_bytes(step: &DispatchStep) -> u64 {
    match step {
        DispatchStep::Linear {
            in_features,
            out_features,
            batch_size,
            total_elements,
            bias,
            ..
        } => linear_memory(
            *in_features,
            *out_features,
            *batch_size,
            *total_elements,
            bias.is_some(),
        ),
        DispatchStep::MatMul {
            m,
            k,
            n,
            batch_size,
            total_elements,
            ..
        } => matmul_memory(*m, *k, *n, *batch_size, *total_elements),
        DispatchStep::Conv1d(p) => conv1d_memory(p),
        DispatchStep::Conv2d(p) => conv2d_memory(p),
        DispatchStep::ConvTranspose1d(p) => conv_transpose1d_memory(p),
        _ => step_memory_bytes_other(step),
    }
}

/// Memory traffic for non-weight-based dispatch steps.
fn step_memory_bytes_other(step: &DispatchStep) -> u64 {
    match step {
        DispatchStep::Softmax {
            axis_size,
            outer_size,
            ..
        } => 2 * (*axis_size as u64) * (*outer_size as u64) * F32_BYTES,
        DispatchStep::Reduce {
            reduce_dim,
            outer_size,
            ..
        } => {
            (*outer_size as u64) * (*reduce_dim as u64) * F32_BYTES
                + (*outer_size as u64) * F32_BYTES
        }
        // Unary element-wise: read + write
        DispatchStep::Sigmoid { total_elements, .. }
        | DispatchStep::Gelu { total_elements, .. }
        | DispatchStep::Relu { total_elements, .. }
        | DispatchStep::Tanh { total_elements, .. } => 2 * (*total_elements as u64) * F32_BYTES,
        // Binary element-wise: 2 reads + 1 write
        DispatchStep::BinaryAdd { total_elements, .. }
        | DispatchStep::BinaryMul { total_elements, .. } => {
            3 * (*total_elements as u64) * F32_BYTES
        }
        DispatchStep::Elementwise {
            total_elements,
            inputs,
            ..
        } => ((inputs.len() as u64) + 1) * (*total_elements as u64) * F32_BYTES,
        DispatchStep::Broadcast { total_elements, .. }
        | DispatchStep::Transpose { total_elements, .. } => {
            2 * (*total_elements as u64) * F32_BYTES
        }
        _ => step_memory_bytes_data_movement(step),
    }
}

/// Memory traffic for data movement / indexing dispatch steps.
fn step_memory_bytes_data_movement(step: &DispatchStep) -> u64 {
    match step {
        DispatchStep::Reshape { .. } => 0,
        DispatchStep::AxisSelect { input_shape, .. } => axis_select_memory(input_shape),
        DispatchStep::Stack {
            input_shape,
            inputs,
            ..
        } => stack_memory(input_shape, inputs.len()),
        DispatchStep::Narrow {
            input_shape,
            length,
            axis,
            ..
        } => narrow_memory(input_shape, *length, *axis),
        DispatchStep::ZeroPad1d {
            channels,
            in_length,
            out_length,
            ..
        } => zero_pad_memory(*channels, *in_length, *out_length),
        DispatchStep::Embedding {
            total_elements,
            num_indices,
            ..
        } => embedding_memory(*total_elements, *num_indices),
        DispatchStep::Concat {
            first_input_shape,
            input_axis_sizes,
            axis,
            ..
        } => concat_memory(first_input_shape, input_axis_sizes, *axis),
        // Unknown future variants: conservative 0 memory bytes.
        // Always warn — silent 0 in release builds hides missing coverage.
        _ => {
            eprintln!(
                "[cost_model] step_memory_bytes_data_movement: unhandled DispatchStep variant: {step:?}"
            );
            0
        }
    }
}

fn matmul_memory(m: usize, k: usize, n: usize, batch: usize, total: usize) -> u64 {
    let left = (batch as u64) * (m as u64) * (k as u64) * F32_BYTES;
    let right = (batch as u64) * (k as u64) * (n as u64) * F32_BYTES;
    let output = (total as u64) * F32_BYTES;
    left + right + output
}

fn stack_memory(input_shape: &[usize], num_inputs: usize) -> u64 {
    let per_input: u64 = input_shape.iter().map(|&d| d as u64).product();
    let n = num_inputs as u64;
    2 * n * per_input * F32_BYTES
}

fn zero_pad_memory(channels: usize, in_length: usize, out_length: usize) -> u64 {
    let input = (channels as u64) * (in_length as u64) * F32_BYTES;
    let output = (channels as u64) * (out_length as u64) * F32_BYTES;
    input + output
}

fn embedding_memory(total_elements: usize, num_indices: usize) -> u64 {
    let index_bytes = (num_indices as u64) * 4;
    let read = (total_elements as u64) * F32_BYTES;
    let write = (total_elements as u64) * F32_BYTES;
    index_bytes + read + write
}

fn linear_memory(in_f: usize, out_f: usize, batch: usize, total: usize, has_bias: bool) -> u64 {
    let input = (batch as u64) * (in_f as u64) * F32_BYTES;
    let weight = (out_f as u64) * (in_f as u64) * F32_BYTES;
    let output = (total as u64) * F32_BYTES;
    let bias_b = if has_bias {
        (out_f as u64) * F32_BYTES
    } else {
        0
    };
    input + weight + bias_b + output
}

fn conv1d_memory(p: &Conv1dParams) -> u64 {
    let input = (p.in_channels as u64) * (p.in_length as u64) * F32_BYTES;
    let cpg = p
        .in_channels
        .checked_div(p.groups)
        .unwrap_or(p.in_channels);
    let weight = (p.out_channels as u64) * (cpg as u64) * (p.kernel_size as u64) * F32_BYTES;
    let output = (p.total_elements as u64) * F32_BYTES;
    let bias_b = if p.bias.is_some() {
        (p.out_channels as u64) * F32_BYTES
    } else {
        0
    };
    input + weight + bias_b + output
}

fn conv2d_memory(p: &Conv2dParams) -> u64 {
    let input = (p.in_channels as u64) * (p.in_height as u64) * (p.in_width as u64) * F32_BYTES;
    let cpg = p
        .in_channels
        .checked_div(p.groups)
        .unwrap_or(p.in_channels);
    let weight = (p.out_channels as u64)
        * (cpg as u64)
        * (p.kernel_h as u64)
        * (p.kernel_w as u64)
        * F32_BYTES;
    let output = (p.total_elements as u64) * F32_BYTES;
    let bias_b = if p.bias.is_some() {
        (p.out_channels as u64) * F32_BYTES
    } else {
        0
    };
    input + weight + bias_b + output
}

fn conv_transpose1d_memory(p: &ConvTranspose1dParams) -> u64 {
    let input = (p.in_channels as u64) * (p.in_length as u64) * F32_BYTES;
    let cpg = p
        .in_channels
        .checked_div(p.groups)
        .unwrap_or(p.in_channels);
    let weight = (p.out_channels as u64) * (cpg as u64) * (p.kernel_size as u64) * F32_BYTES;
    let output = (p.total_elements as u64) * F32_BYTES;
    let bias_b = if p.bias.is_some() {
        (p.out_channels as u64) * F32_BYTES
    } else {
        0
    };
    input + weight + bias_b + output
}

fn axis_select_memory(input_shape: &[usize]) -> u64 {
    let input_elements: u64 = input_shape.iter().map(|&d| d as u64).product();
    let output_elements = if input_shape.is_empty() {
        0
    } else {
        input_elements / input_shape.iter().max().copied().unwrap_or(1) as u64
    };
    (input_elements + output_elements) * F32_BYTES
}

fn narrow_memory(input_shape: &[usize], length: usize, axis: usize) -> u64 {
    let input_elements: u64 = input_shape.iter().map(|&d| d as u64).product();
    let output_elements = if axis < input_shape.len() && input_shape[axis] > 0 {
        input_elements / (input_shape[axis] as u64) * (length as u64)
    } else {
        0
    };
    (input_elements + output_elements) * F32_BYTES
}

fn concat_memory(first_input_shape: &[usize], input_axis_sizes: &[usize], axis: usize) -> u64 {
    let non_axis_product: u64 = first_input_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &d)| d as u64)
        .product();
    let total_axis: u64 = input_axis_sizes.iter().map(|&s| s as u64).sum();
    let total_elements = non_axis_product * total_axis;
    2 * total_elements * F32_BYTES
}
