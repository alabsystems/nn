// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! AddLayerNorm executor for `CompiledModel`.
//!
//! Fused residual-add + LayerNorm: `LN(a + b, weight, bias)` in a single
//! Metal dispatch. Part of #1815 Tier 5 D2.

use nn_core::Result;

use crate::gpu_slice::GpuSlice;

use super::CompiledModel;
use super::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};

/// Execute a `NativeOpKind::AddLayerNorm` step.
///
/// Resolves two graph inputs (residual a, new value b) and pre-uploaded
/// weight/bias, calls `gpu_add_layer_norm_fused` (single Metal dispatch).
pub(super) fn execute_native_add_layer_norm(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    eps: f32,
    input_shape: &[usize],
    hidden_dim: usize,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);

    // Input 0: residual (a), Input 1: new value (b).
    let a_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let b_slice = model.resolve_input_slice(step_idx, 1, buffers)?;

    let weights = &model.def.weight_buffers[step_idx];

    let output = crate::dyn_tensor_metal::native_add_layer_norm(
        &slice_to_dyn(&a_slice, input_shape, dtype)?,
        &slice_to_dyn(&b_slice, input_shape, dtype)?,
        &weight_to_dyn(
            weights,
            "weight",
            &[hidden_dim],
            dtype,
            step_idx,
            "AddLayerNorm",
        )?,
        &weight_to_dyn(
            weights,
            "bias",
            &[hidden_dim],
            dtype,
            step_idx,
            "AddLayerNorm",
        )?,
        f64::from(eps),
    )
    .map_err(|e| native_dispatch_err(step_idx, format!("NativeOp AddLayerNorm: {e}")))?;

    dyn_to_slice(&output, step_idx, "AddLayerNorm")
}
