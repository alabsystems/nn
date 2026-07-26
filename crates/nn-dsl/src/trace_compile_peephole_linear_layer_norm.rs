// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse Linear + LayerNorm → FusedLinearLayerNorm.
//!
//! Detects consecutive step pairs:
//!   Step i:   Dispatch { kernel: "linear" }
//!   Step i+1: NativeOp { LayerNorm { eps, input_shape, hidden_dim } }
//!
//! The Linear output must be single-consumer (use_counts == 1).
//!
//! Replaces the pair with:
//! - steps[i]   → NativeOp { FusedLinearLayerNorm { ... } }
//! - steps[i+1] → IdentityPassthrough
//!
//! Saves 1 dispatch per pair. The reverse of NormLinear (pass 7). In PlBert
//! transformer layers, attention output projections and FFN output projections
//! feed directly into LayerNorm before residual addition. Fusing eliminates
//! the intermediate buffer.
//!
//! Must run AFTER AddLayerNorm (pass 6) and NormLinear (pass 7) which consume
//! LayerNorm from the other direction.
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::WeightRef;

use super::super::{CompiledStep, NativeOpKind};
use super::linear_activation::extract_linear_params;

/// Scan for Linear + LayerNorm pairs and fuse them.
pub(super) fn fuse_linear_layer_norm(steps: &mut [CompiledStep], use_counts: &[usize]) {
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

/// Try to fuse steps[i] (Linear) with steps[i+1] (LayerNorm).
fn try_fuse(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
    // ---- Step i: Dispatch with kernel name "linear" ----
    let (linear_info, linear_weight_data) = match &steps[i] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "linear" => match extract_linear_params(kernel, weight_data) {
            Some(info) => (info, weight_data.clone()),
            None => return false,
        },
        _ => return false,
    };

    // Fan-out: linear output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+1: NativeOp { LayerNorm } ----
    let (eps, norm_hidden_dim) = match &steps[i + 1] {
        CompiledStep::NativeOp {
            op: NativeOpKind::LayerNorm {
                eps, hidden_dim, ..
            },
            weight_data,
            ..
        } => {
            // Verify the LayerNorm has weights.
            if !weight_data.contains_key("weight") {
                return false;
            }
            (*eps, *hidden_dim)
        }
        _ => return false,
    };

    // The linear output features must match the LayerNorm hidden_dim.
    if linear_info.out_features != norm_hidden_dim {
        return false;
    }

    // Extract LayerNorm weights.
    let norm_weight_data = match &steps[i + 1] {
        CompiledStep::NativeOp { weight_data, .. } => weight_data.clone(),
        _ => return false,
    };

    // Build merged weight_data.
    let mut merged_weight_data: HashMap<String, WeightRef> = HashMap::new();

    // Linear weights.
    if let Some(w) = linear_weight_data.get("weight") {
        merged_weight_data.insert("weight".to_string(), w.clone());
    }
    if let Some(b) = linear_weight_data.get("bias") {
        merged_weight_data.insert("bias".to_string(), b.clone());
    }

    // LayerNorm weights (renamed with norm_ prefix).
    if let Some(w) = norm_weight_data.get("weight") {
        merged_weight_data.insert("norm_weight".to_string(), w.clone());
    }
    if let Some(b) = norm_weight_data.get("bias") {
        merged_weight_data.insert("norm_bias".to_string(), b.clone());
    }

    let fused_op = NativeOpKind::FusedLinearLayerNorm {
        in_features: linear_info.in_features,
        out_features: linear_info.out_features,
        has_bias: linear_info.has_bias,
        eps,
        input_shape: linear_info.input_shape,
    };

    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weight_data,
    };
    steps[i + 1] = CompiledStep::IdentityPassthrough;

    true
}
