// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GEMM eligibility detection for F16 simdgroup dispatch.
//!
//! Extracted from `compiled_model_build.rs` — scans compiled steps
//! for Linear/MatMul ops that qualify for simdgroup-tiled F16 GEMM.
//! Part of #3085, #2981.

use nn_dsl::trace_compile::{CompiledStep, NativeOpKind};
use nn_dsl::TensorOpKind;

/// Extract `MixedGemmInfo` for simdgroup-eligible GEMM steps. Part of #3085, #2981.
pub(in crate::compiled_model) fn extract_mixed_gemm_infos(
    steps: &[CompiledStep],
) -> Vec<Option<super::super::MixedGemmInfo>> {
    steps
        .iter()
        .map(|step| {
            match step {
                CompiledStep::NativeOp { op, .. } => {
                    let NativeOpKind::LinearActivation {
                        activation,
                        in_features,
                        out_features,
                        has_bias,
                        input_shape,
                    } = op
                    else {
                        return None;
                    };
                    if input_shape.is_empty() {
                        return None;
                    }
                    let m = input_shape.iter().rev().skip(1).product::<usize>().max(1);
                    if !crate::dyn_tensor_metal::should_use_f16_simdgroup(
                        m,
                        *in_features,
                        *out_features,
                        1,
                    ) {
                        return None;
                    }
                    return Some(super::super::MixedGemmInfo {
                        m,
                        k: *in_features,
                        n: *out_features,
                        batch_count: 1,
                        transpose_b: true,
                        broadcast_b: false,
                        has_bias: *has_bias,
                        activation: Some(*activation),
                    });
                }
                CompiledStep::Dispatch { .. } => {}
                _ => return None,
            }
            let CompiledStep::Dispatch { kernel, .. } = step else {
                return None;
            };
            let def = kernel.def();
            let output_idx = def.output.index();
            let output_node = def.nodes.get(output_idx)?;
            match &output_node.kind {
                TensorOpKind::Linear {
                    input,
                    weight,
                    bias,
                } => {
                    let weight_node = def.nodes.get(weight.index())?;
                    if weight_node.shape.len() != 2 {
                        return None;
                    }
                    let out_features = weight_node.shape[0];
                    let in_features = weight_node.shape[1];
                    let input_node = def.nodes.get(input.index())?;
                    let m: usize = input_node.shape.iter().rev().skip(1).product();
                    if !crate::dyn_tensor_metal::should_use_f16_simdgroup(
                        m,
                        in_features,
                        out_features,
                        1,
                    ) {
                        return None;
                    }
                    Some(super::super::MixedGemmInfo {
                        m,
                        k: in_features,
                        n: out_features,
                        batch_count: 1,
                        transpose_b: true,
                        broadcast_b: false,
                        has_bias: bias.is_some(),
                        activation: None,
                    })
                }
                TensorOpKind::MatMul {
                    left,
                    right,
                    transpose_right,
                    ..
                } => {
                    let left_node = def.nodes.get(left.index())?;
                    let right_node = def.nodes.get(right.index())?;
                    let rank = left_node.shape.len();
                    if rank < 2 {
                        return None;
                    }
                    let m = left_node.shape[rank - 2];
                    let k = left_node.shape[rank - 1];
                    let n = if *transpose_right {
                        let idx = right_node.shape.len().checked_sub(2)?;
                        *right_node.shape.get(idx)?
                    } else {
                        *right_node.shape.last()?
                    };
                    let batch: usize = left_node.shape.iter().rev().skip(2).product();
                    let batch_count = batch.max(1);
                    let broadcast_b = right_node.shape.len() < left_node.shape.len()
                        || right_node.shape.iter().rev().skip(2).product::<usize>() < batch_count;
                    if !crate::dyn_tensor_metal::should_use_f16_simdgroup(m, k, n, batch_count) {
                        return None;
                    }
                    Some(super::super::MixedGemmInfo {
                        m,
                        k,
                        n,
                        batch_count,
                        transpose_b: *transpose_right,
                        broadcast_b,
                        has_bias: false,
                        activation: None,
                    })
                }
                _ => None,
            }
        })
        .collect()
}
