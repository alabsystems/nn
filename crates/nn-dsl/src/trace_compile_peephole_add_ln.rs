// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: BinaryAdd + LayerNorm → AddLayerNorm.
//!
//! Fuses adjacent `Dispatch{add}` + `NativeOp{LayerNorm}` into a single
//! `AddLayerNorm` NativeOp. Reads both inputs and normalizes in one Metal
//! dispatch without materializing the intermediate sum. Part of #1815 Tier 5 D2.

use super::super::{CompiledStep, NativeOpKind};

/// Scan for `Dispatch{add}` + `NativeOp{LayerNorm}` pairs and fuse them.
///
/// Matches:
/// - `steps[i]` is `Dispatch` with `kernel.name() == "add"`
/// - `steps[i+1]` is `NativeOp{LayerNorm { eps, input_shape, hidden_dim }}`
/// - `use_counts[i] == 1` (add output consumed only by LayerNorm)
pub(super) fn fuse_add_layer_norm(steps: &mut [CompiledStep], use_counts: &[usize]) {
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

/// Try to fuse steps[i] (add) with steps[i+1] (LayerNorm).
fn try_fuse(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
    // Step[i] must be Dispatch with name "add".
    let is_add = matches!(
        &steps[i],
        CompiledStep::Dispatch { kernel, .. } if kernel.name() == "add"
    );
    if !is_add {
        return false;
    }

    // Fan-out: add output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // Step[i+1] must be NativeOp{LayerNorm}.
    let (eps, input_shape, hidden_dim, ln_weight_data) = match &steps[i + 1] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::LayerNorm {
                    eps,
                    input_shape,
                    hidden_dim,
                },
            weight_data,
        } => (*eps, input_shape.clone(), *hidden_dim, weight_data.clone()),
        _ => return false,
    };

    // Fuse: replace add step with AddLayerNorm, replace LN step with passthrough.
    // The add step's edge_map (2 inputs: a, b) is preserved as AddLayerNorm's inputs.
    // The LayerNorm's weight_data (weight, bias) moves to the fused step.
    steps[i] = CompiledStep::NativeOp {
        op: NativeOpKind::AddLayerNorm {
            eps,
            input_shape,
            hidden_dim,
        },
        weight_data: ln_weight_data,
    };
    steps[i + 1] = CompiledStep::IdentityPassthrough;
    true
}

#[cfg(test)]
#[path = "trace_compile_peephole_add_ln_tests.rs"]
mod tests;
