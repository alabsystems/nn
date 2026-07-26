// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight buffer size estimation for individual dispatch steps.
//!
//! Part of the peak memory model (#1739 Phase 19).

use nn_dsl::DispatchStep;

const F32_BYTES: u64 = 4;

/// Compute the weight buffer size in bytes for dispatch steps that use weights.
///
/// Returns 0 for element-wise and data-movement ops. This represents
/// memory that must be allocated for the lifetime of the model (not freed
/// between layers).
pub fn step_weight_bytes(step: &DispatchStep) -> u64 {
    match step {
        DispatchStep::Linear {
            in_features,
            out_features,
            bias,
            ..
        } => {
            let w = (*in_features as u64) * (*out_features as u64) * F32_BYTES;
            let b = if bias.is_some() {
                (*out_features as u64) * F32_BYTES
            } else {
                0
            };
            w + b
        }
        DispatchStep::Conv1d(p) => conv_weight_bytes(
            p.out_channels,
            p.in_channels,
            p.groups,
            p.kernel_size,
            p.bias.is_some(),
        ),
        DispatchStep::Conv2d(p) => {
            let cpg = p
                .in_channels
                .checked_div(p.groups)
                .unwrap_or(p.in_channels);
            let w = (p.out_channels as u64)
                * (cpg as u64)
                * (p.kernel_h as u64)
                * (p.kernel_w as u64)
                * F32_BYTES;
            let b = if p.bias.is_some() {
                (p.out_channels as u64) * F32_BYTES
            } else {
                0
            };
            w + b
        }
        DispatchStep::ConvTranspose1d(p) => conv_weight_bytes(
            p.out_channels,
            p.in_channels,
            p.groups,
            p.kernel_size,
            p.bias.is_some(),
        ),
        DispatchStep::Embedding {
            total_elements,
            num_indices,
            ..
        } => {
            // Embedding table size approximation: output size as conservative lower bound.
            if *num_indices > 0 {
                ((*total_elements as u64) / (*num_indices as u64))
                    * (*num_indices as u64)
                    * F32_BYTES
            } else {
                0
            }
        }
        // Unknown future variants: conservative 0 weight bytes.
        // Always warn — silent 0 in release builds hides missing coverage.
        _ => {
            eprintln!("[cost_model] step_weight_bytes: unhandled DispatchStep variant: {step:?}");
            0
        }
    }
}

/// Shared weight size computation for 1D convolution variants.
fn conv_weight_bytes(
    out_channels: usize,
    in_channels: usize,
    groups: usize,
    kernel_size: usize,
    has_bias: bool,
) -> u64 {
    let cpg = in_channels.checked_div(groups).unwrap_or(in_channels);
    let w = (out_channels as u64) * (cpg as u64) * (kernel_size as u64) * F32_BYTES;
    let b = if has_bias {
        (out_channels as u64) * F32_BYTES
    } else {
        0
    };
    w + b
}
