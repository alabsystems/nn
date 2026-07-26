// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass 7: Norm + Linear → NormLinear / AddNormLinear.
//!
//! Fuses adjacent normalization and Linear Dispatch into NativeOps.
//! Part of #3089, #4252.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::WeightRef;

use crate::tensor_ir::TensorOpKind;

use super::super::{CompiledStep, FusedNormKind, NativeOpKind};

/// Scan for Norm + Linear pairs and fuse them.
///
/// Matches:
/// - `NativeOp{LayerNorm}` + `Dispatch{linear}` → NormLinear
/// - `Dispatch{rms_norm}` + `Dispatch{linear}` → NormLinear
/// - `NativeOp{AddLayerNorm}` + `IdentityPassthrough` + `Dispatch{linear}` → AddNormLinear
pub(super) fn fuse_norm_linear(steps: &mut [CompiledStep], use_counts: &[usize]) {
    let len = steps.len();
    if len < 2 {
        return;
    }
    let mut i = 0;
    while i < len {
        if let Some(advance) = try_fuse_add_norm_linear(steps, i, use_counts) {
            i += advance;
            continue;
        }
        if i + 1 < len && try_fuse(steps, i, use_counts) {
            i += 2;
        } else {
            i += 1;
        }
    }
}

struct NormInfo {
    kind: FusedNormKind,
    eps: f32,
    input_shape: Vec<usize>,
    hidden_dim: usize,
    weight_data: HashMap<String, WeightRef>,
}

fn extract_norm_info(step: &CompiledStep) -> Option<NormInfo> {
    match step {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::LayerNorm {
                    eps,
                    input_shape,
                    hidden_dim,
                },
            weight_data,
        } => Some(NormInfo {
            kind: FusedNormKind::LayerNorm,
            eps: *eps,
            input_shape: input_shape.clone(),
            hidden_dim: *hidden_dim,
            weight_data: weight_data.clone(),
        }),
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "rms_norm" => extract_rms_norm_params(kernel, weight_data),
        _ => None,
    }
}

fn extract_rms_norm_params(
    kernel: &super::super::CompiledKernel,
    weight_data: &HashMap<String, WeightRef>,
) -> Option<NormInfo> {
    let eps_ref = weight_data.get("eps")?;
    let eps_data = eps_ref.data();
    if eps_data.len() != 1 {
        return None;
    }
    let eps = eps_data[0];
    if !eps.is_finite() || eps <= 0.0 {
        return None;
    }
    let weight_ref = weight_data.get("weight")?;
    let weight_shape = weight_ref.shape();
    if weight_shape.len() != 1 {
        return None;
    }
    let hidden_dim = weight_shape[0];
    let def = kernel.def();
    let input_shape = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::Input { .. }))
        .map(|n| n.shape.clone())?;
    let mut norm_wd = HashMap::new();
    norm_wd.insert("weight".to_string(), weight_ref.clone());
    Some(NormInfo {
        kind: FusedNormKind::RmsNorm,
        eps,
        input_shape,
        hidden_dim,
        weight_data: norm_wd,
    })
}

struct AddNormInfo {
    eps: f32,
    input_shape: Vec<usize>,
    hidden_dim: usize,
    weight_data: HashMap<String, WeightRef>,
}

fn extract_add_norm_info(step: &CompiledStep) -> Option<AddNormInfo> {
    match step {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::AddLayerNorm {
                    eps,
                    input_shape,
                    hidden_dim,
                },
            weight_data,
        } => Some(AddNormInfo {
            eps: *eps,
            input_shape: input_shape.clone(),
            hidden_dim: *hidden_dim,
            weight_data: weight_data.clone(),
        }),
        _ => None,
    }
}

fn validate_and_merge(
    hidden_dim: usize,
    linear_info: &super::LinearInfo,
    norm_wd: &mut HashMap<String, WeightRef>,
) -> Option<HashMap<String, WeightRef>> {
    if hidden_dim != linear_info.in_features {
        return None;
    }
    if hidden_dim > 7680 {
        return None;
    }
    let mut merged: HashMap<String, WeightRef> = HashMap::new();
    if let Some(w) = norm_wd.remove("weight") {
        merged.insert("norm_weight".to_string(), w);
    }
    if let Some(b) = norm_wd.remove("bias") {
        merged.insert("norm_bias".to_string(), b);
    }
    merged.extend(linear_info.weight_data.clone());
    Some(merged)
}

#[cfg(test)]
#[path = "trace_compile_peephole_norm_linear_tests.rs"]
mod tests;

fn try_fuse(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
    let norm = match extract_norm_info(&steps[i]) {
        Some(info) => info,
        None => return false,
    };
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }
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
    let mut norm_wd = norm.weight_data;
    let merged = match validate_and_merge(norm.hidden_dim, &linear_info, &mut norm_wd) {
        Some(m) => m,
        None => return false,
    };
    // Use FusedLayerNormLinear for LayerNorm NativeOp sources (PlBert encoder),
    // NormLinear for RmsNorm Dispatch sources (Qwen3/GLM5). Both delegate to the
    // same executor but have distinct registry entries for stage tracking (#4252).
    let fused_op = match norm.kind {
        FusedNormKind::LayerNorm => NativeOpKind::FusedLayerNormLinear {
            eps: norm.eps,
            input_shape: norm.input_shape,
            hidden_dim: norm.hidden_dim,
            out_features: linear_info.out_features,
            has_bias: linear_info.has_bias,
        },
        FusedNormKind::RmsNorm => NativeOpKind::NormLinear {
            norm_kind: norm.kind,
            eps: norm.eps,
            input_shape: norm.input_shape,
            hidden_dim: norm.hidden_dim,
            out_features: linear_info.out_features,
            has_bias: linear_info.has_bias,
        },
    };
    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged,
    };
    steps[i + 1] = CompiledStep::IdentityPassthrough;
    true
}

/// Fuse AddLayerNorm + [IdentityPassthrough] + Linear → AddNormLinear.
/// Part of #4252.
fn try_fuse_add_norm_linear(
    steps: &mut [CompiledStep],
    i: usize,
    use_counts: &[usize],
) -> Option<usize> {
    let len = steps.len();
    let add_norm = extract_add_norm_info(&steps[i])?;
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return None;
    }
    let (linear_idx, advance) = if i + 1 < len {
        if matches!(&steps[i + 1], CompiledStep::IdentityPassthrough) && i + 2 < len {
            if use_counts.get(i + 1).copied().unwrap_or(0) != 1 {
                return None;
            }
            (i + 2, 3)
        } else {
            (i + 1, 2)
        }
    } else {
        return None;
    };
    let linear_info = match &steps[linear_idx] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "linear" => super::extract_linear_params(kernel, weight_data),
        _ => None,
    }?;
    let mut add_norm_wd = add_norm.weight_data;
    let merged = validate_and_merge(add_norm.hidden_dim, &linear_info, &mut add_norm_wd)?;
    steps[i] = CompiledStep::NativeOp {
        op: NativeOpKind::AddNormLinear {
            eps: add_norm.eps,
            input_shape: add_norm.input_shape,
            hidden_dim: add_norm.hidden_dim,
            out_features: linear_info.out_features,
            has_bias: linear_info.has_bias,
        },
        weight_data: merged,
    };
    for step in &mut steps[(i + 1)..=linear_idx] {
        *step = CompiledStep::IdentityPassthrough;
    }
    Some(advance)
}
