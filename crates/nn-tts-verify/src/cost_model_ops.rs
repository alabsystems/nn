// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! FLOP estimation for individual dispatch steps.
//!
//! Output buffer sizing is in `cost_model_ops_output.rs`.
//! Weight sizing is in `cost_model_ops_weight.rs`.
//! Memory traffic estimation is in `cost_model_ops_memory.rs`.
//! Part of the computational cost model (#1739).

use nn_dsl::{Conv1dParams, Conv2dParams, ConvTranspose1dParams, DispatchStep};

#[path = "cost_model_ops_memory.rs"]
mod memory;
pub use memory::step_memory_bytes;

#[path = "cost_model_ops_output.rs"]
mod output;
pub use output::step_output_bytes;

#[path = "cost_model_ops_weight.rs"]
mod weight;
pub use weight::step_weight_bytes;

/// Bytes per f32 element; shared with `cost_model_ops_memory.rs` via `super::F32_BYTES`.
const F32_BYTES: u64 = 4;

/// Compute theoretical FLOPs for a single dispatch step.
///
/// FLOP definitions follow standard conventions:
/// - MatMul [M,K]×[K,N]: 2×M×K×N (multiply-accumulate = 2 FLOPs)
/// - Conv1d: 2 × out_len × out_channels × (in_channels/groups) × kernel_size
/// - Element-wise: 1 FLOP per element (conservative for multi-op kernels)
/// - Reduce: elements in the reduction (1 FLOP per accumulation)
pub fn step_flops(step: &DispatchStep) -> u64 {
    match step {
        DispatchStep::Linear {
            in_features,
            out_features,
            batch_size,
            bias,
            ..
        } => {
            let matmul = 2 * (*batch_size as u64) * (*in_features as u64) * (*out_features as u64);
            let bias_add = if bias.is_some() {
                (*batch_size as u64) * (*out_features as u64)
            } else {
                0
            };
            matmul + bias_add
        }

        DispatchStep::MatMul {
            m,
            k,
            n,
            batch_size,
            ..
        } => 2 * (*batch_size as u64) * (*m as u64) * (*k as u64) * (*n as u64),

        DispatchStep::Conv1d(params) => conv1d_flops(params),
        DispatchStep::Conv2d(params) => conv2d_flops(params),
        DispatchStep::ConvTranspose1d(params) => conv_transpose1d_flops(params),

        DispatchStep::Softmax {
            axis_size,
            outer_size,
            ..
        } => 5 * (*outer_size as u64) * (*axis_size as u64),

        DispatchStep::Reduce {
            reduce_dim,
            outer_size,
            ..
        } => (*outer_size as u64) * (*reduce_dim as u64),

        // Element-wise unary ops: 1 FLOP per element
        DispatchStep::Sigmoid { total_elements, .. }
        | DispatchStep::Gelu { total_elements, .. }
        | DispatchStep::Relu { total_elements, .. }
        | DispatchStep::Tanh { total_elements, .. } => *total_elements as u64,

        // Element-wise binary ops: 1 FLOP per element
        DispatchStep::BinaryAdd { total_elements, .. }
        | DispatchStep::BinaryMul { total_elements, .. } => *total_elements as u64,

        // Custom kernel: 1 FLOP per element (conservative)
        DispatchStep::Elementwise { total_elements, .. } => *total_elements as u64,

        // Data movement ops: 0 FLOPs
        DispatchStep::Broadcast { .. }
        | DispatchStep::Reshape { .. }
        | DispatchStep::AxisSelect { .. }
        | DispatchStep::Stack { .. }
        | DispatchStep::Narrow { .. }
        | DispatchStep::ZeroPad1d { .. }
        | DispatchStep::Transpose { .. }
        | DispatchStep::Concat { .. }
        | DispatchStep::Embedding { .. } => 0,

        // Unknown future variants: conservative 0 FLOPs.
        // Always warn — silent 0 in release builds hides missing coverage.
        _ => {
            eprintln!("[cost_model] step_flops: unhandled DispatchStep variant: {step:?}");
            0
        }
    }
}

fn conv1d_flops(p: &Conv1dParams) -> u64 {
    let out_len = p.total_elements.checked_div(p.out_channels).unwrap_or(0);
    let cpg = p
        .in_channels
        .checked_div(p.groups)
        .unwrap_or(p.in_channels);
    let conv =
        2 * (out_len as u64) * (p.out_channels as u64) * (cpg as u64) * (p.kernel_size as u64);
    let bias_add = if p.bias.is_some() {
        p.total_elements as u64
    } else {
        0
    };
    conv + bias_add
}

fn conv2d_flops(p: &Conv2dParams) -> u64 {
    let out_spatial = p.total_elements.checked_div(p.out_channels).unwrap_or(0);
    let cpg = p
        .in_channels
        .checked_div(p.groups)
        .unwrap_or(p.in_channels);
    let conv = 2
        * (out_spatial as u64)
        * (p.out_channels as u64)
        * (cpg as u64)
        * (p.kernel_h as u64)
        * (p.kernel_w as u64);
    let bias_add = if p.bias.is_some() {
        p.total_elements as u64
    } else {
        0
    };
    conv + bias_add
}

fn conv_transpose1d_flops(p: &ConvTranspose1dParams) -> u64 {
    let out_len = p.total_elements.checked_div(p.out_channels).unwrap_or(0);
    let cpg = p
        .in_channels
        .checked_div(p.groups)
        .unwrap_or(p.in_channels);
    let conv =
        2 * (out_len as u64) * (p.out_channels as u64) * (cpg as u64) * (p.kernel_size as u64);
    let bias_add = if p.bias.is_some() {
        p.total_elements as u64
    } else {
        0
    };
    conv + bias_add
}
