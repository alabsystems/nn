// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass 8: AddLayerNorm + Linear → AddNormLinear.
//!
//! Fuses adjacent `NativeOp{AddLayerNorm}` and `Dispatch{linear}` into a
//! single `NativeOp{AddNormLinear}`. Executes residual-add, LayerNorm, and
//! GEMM in one Metal dispatch using threadgroup memory.
//!
//! This pass runs AFTER AddLayerNorm (pass 6) and NormLinear (pass 7) to
//! catch the `Add + LayerNorm + Linear` patterns that those passes create.
//! In PlBert transformer layers, pass 6 fuses `Add + LayerNorm` → `AddLayerNorm`,
//! blocking pass 7 (NormLinear) from seeing the LayerNorm. This pass recovers
//! the fusion opportunity by combining `AddLayerNorm + Linear` → `AddNormLinear`.
//!
//! Part of #3351 T2.1 (dispatch reduction).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::WeightRef;

use super::super::{CompiledStep, NativeOpKind};

/// Scan for `NativeOp{AddLayerNorm}` + `Dispatch{linear}` pairs and fuse them.
pub(super) fn fuse_add_norm_linear(steps: &mut [CompiledStep], use_counts: &[usize]) {
    let len = steps.len();
    if len < 2 {
        return;
    }
    let mut i = 0;
    while i + 1 < len {
        if try_fuse(steps, i, use_counts) {
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i] (AddLayerNorm) with steps[i+1] (Linear).
fn try_fuse(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
    // Step[i] must be NativeOp{AddLayerNorm}.
    let (eps, input_shape, hidden_dim, ln_weight_data) = match &steps[i] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::AddLayerNorm {
                    eps,
                    input_shape,
                    hidden_dim,
                },
            weight_data,
        } => (*eps, input_shape.clone(), *hidden_dim, weight_data.clone()),
        _ => return false,
    };

    // Fan-out: AddLayerNorm output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // Step[i+1] must be Dispatch with name "linear".
    let linear_info = match &steps[i + 1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "linear" => super::extract_linear_params(kernel, weight_data),
        _ => None,
    };
    let linear_info = match linear_info {
        Some(info) => info,
        None => return false,
    };

    // Hidden dim must match linear in_features.
    if hidden_dim != linear_info.in_features {
        return false;
    }

    // Threadgroup memory limit: hidden_dim * 4 + 2048 (reduction scratch) <= 32768.
    // Max safe hidden_dim = (32768 - 2048) / 4 = 7680.
    if hidden_dim > 7680 {
        return false;
    }

    // Rename norm weights to avoid collision with linear "weight"/"bias".
    let mut merged: HashMap<String, WeightRef> = HashMap::new();
    let mut norm_wd = ln_weight_data;
    if let Some(w) = norm_wd.remove("weight") {
        merged.insert("norm_weight".to_string(), w);
    }
    if let Some(b) = norm_wd.remove("bias") {
        merged.insert("norm_bias".to_string(), b);
    }
    // Linear weights keep original names ("weight", optional "bias").
    merged.extend(linear_info.weight_data);

    // Place AddNormLinear at step[i] (AddLayerNorm position) — preserves the
    // 2-input edge_map from AddLayerNorm (inputs: a, b).
    steps[i] = CompiledStep::NativeOp {
        op: NativeOpKind::AddNormLinear {
            eps,
            input_shape,
            hidden_dim,
            out_features: linear_info.out_features,
            has_bias: linear_info.has_bias,
        },
        weight_data: merged,
    };
    steps[i + 1] = CompiledStep::IdentityPassthrough;
    true
}

#[cfg(test)]
#[path = "trace_compile_peephole_add_norm_linear_tests.rs"]
mod tests;
