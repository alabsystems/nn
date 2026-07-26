// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gated DeltaNet cell decomposition into primitive tensor ops.
//!
//! Decomposes a single-timestep Gated DeltaNet recurrence into primitives:
//! MatMul, BinaryMul, BinaryAdd, and Reshape. Every primitive has a working
//! Metal dispatch path, so the decomposed DeltaNet runs on GPU without a
//! monolithic kernel.
//!
//! The Gated DeltaNet recurrence (arXiv 2412.06464, Qwen3.5):
//!
//! ```text
//! decayed = gate * state                              // [H, K, V]
//! v_retrieved = k^T @ decayed                         // [H, V]
//! new_state = decayed + outer(k, beta*v) - outer(k, beta*v_retrieved)
//! output = scale * q @ new_state                      // [H, V]
//! ```
//!
//! The subtraction is handled via MatMul's scale parameter (scale=-1.0) to
//! avoid needing a BinarySub op.
//!
//! Part of #834 — Gated DeltaNet for Qwen3.5 model support.

use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::{TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNodeId};

/// Result of a Gated DeltaNet cell decomposition: output and updated state.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetOutputs {
    /// Output vector o_t = scale * q_t @ S_t. Shape: [num_heads, value_dim].
    pub output: TensorNodeId,
    /// Updated recurrent state S_t. Shape: [num_heads, key_dim, value_dim].
    pub new_state: TensorNodeId,
}

fn validate_dimensions(
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<(), TensorIRError> {
    if num_heads == 0 {
        return Err(TensorIRLayerError::GatedDeltaNetZeroDimension { param: "num_heads" }.into());
    }
    if key_dim == 0 {
        return Err(TensorIRLayerError::GatedDeltaNetZeroDimension { param: "key_dim" }.into());
    }
    if value_dim == 0 {
        return Err(TensorIRLayerError::GatedDeltaNetZeroDimension { param: "value_dim" }.into());
    }
    Ok(())
}

/// Decompose a Gated DeltaNet cell into primitive ops within an existing builder.
///
/// # Shapes
///
/// - `q`: `[H, K]` — query vector per head (L2-normalized)
/// - `k`: `[H, K]` — key vector per head (L2-normalized)
/// - `v`: `[H, V]` — value vector per head
/// - `state`: `[H, K, V]` — recurrent state matrix per head
/// - `gate`: `[H, 1, 1]` — decay factor `exp(g_t)` in `(0, 1)`, broadcast to `[H, K, V]`
/// - `beta`: `[H, 1]` — write strength in `(0, 1)`, broadcast to `[H, V]`
pub fn decompose_gated_delta_net(
    builder: &mut TensorBlockBuilder,
    q: TensorNodeId,
    k: TensorNodeId,
    v: TensorNodeId,
    state: TensorNodeId,
    gate: TensorNodeId,
    beta: TensorNodeId,
    scale: f32,
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> GatedDeltaNetOutputs {
    let state_shape = [num_heads, key_dim, value_dim];
    let hv_shape = [num_heads, value_dim];

    // Broadcast gate [H, 1, 1] -> [H, K, V] and beta [H, 1] -> [H, V]
    let gate_bc = builder.add_broadcast(gate, &state_shape);
    let beta_bc = builder.add_broadcast(beta, &hv_shape);

    // Step 1: Decay — gate * state  [H, K, V]
    let decayed = builder.add_binary_mul(gate_bc, state, &state_shape);

    // Step 2: Retrieval — v_retrieved = k^T @ decayed_state
    // Reshape k to [H, 1, K], matmul with decayed [H, K, V] -> [H, 1, V], reshape to [H, V]
    let k_row = builder.add_reshape(k, &[num_heads, 1, key_dim]);
    let vr_3d = builder.add_matmul(k_row, decayed, false, None, &[num_heads, 1, value_dim]);
    let v_retrieved = builder.add_reshape(vr_3d, &hv_shape);

    // Step 3: Scaled write and cancel terms
    // beta * v (what to write), beta * v_retrieved (what to cancel)
    let beta_v = builder.add_binary_mul(beta_bc, v, &hv_shape);
    let beta_vr = builder.add_binary_mul(beta_bc, v_retrieved, &hv_shape);

    // Step 4: Rank-1 state updates via outer products
    // outer(k, x) = k_col [H, K, 1] @ x_row [H, 1, V] -> [H, K, V]
    let k_col = builder.add_reshape(k, &[num_heads, key_dim, 1]);
    let bv_row = builder.add_reshape(beta_v, &[num_heads, 1, value_dim]);
    let bvr_row = builder.add_reshape(beta_vr, &[num_heads, 1, value_dim]);

    let pos_update = builder.add_matmul(k_col, bv_row, false, None, &state_shape);
    // Negative update uses MatMul scale=-1.0 to avoid needing BinarySub
    let neg_update = builder.add_matmul(k_col, bvr_row, false, Some(-1.0), &state_shape);

    // Step 5: State update — new_state = decayed + pos_update + neg_update
    let tmp = builder.add_binary_add(decayed, pos_update, &state_shape);
    let new_state = builder.add_binary_add(tmp, neg_update, &state_shape);

    // Step 6: Output — o = scale * q @ new_state
    // q_row [H, 1, K] @ new_state [H, K, V] -> [H, 1, V], reshape to [H, V]
    let q_row = builder.add_reshape(q, &[num_heads, 1, key_dim]);
    let o_3d = builder.add_matmul(
        q_row,
        new_state,
        false,
        Some(scale),
        &[num_heads, 1, value_dim],
    );
    let output = builder.add_reshape(o_3d, &hv_shape);

    GatedDeltaNetOutputs { output, new_state }
}

/// Build a decomposed Gated DeltaNet cell as a standalone `TensorKernelDef`.
///
/// Returns the output vector. For both output and updated state, use
/// `build_gated_delta_net_decomposed_dual`.
///
/// # Errors
///
/// Returns error if any dimension is zero or scale is invalid.
pub fn build_gated_delta_net_decomposed(
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
    scale: f32,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_dimensions(num_heads, key_dim, value_dim)?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(TensorIRLayerError::GatedDeltaNetScaleInvalid { value: scale }.into());
    }

    let mut builder = TensorBlockBuilder::new("gated_delta_net_decomposed");
    let q = builder.add_input("q", &[num_heads, key_dim]);
    let k = builder.add_input("k", &[num_heads, key_dim]);
    let v = builder.add_input("v", &[num_heads, value_dim]);
    let state = builder.add_input("state", &[num_heads, key_dim, value_dim]);
    let gate = builder.add_input("gate", &[num_heads, 1, 1]);
    let beta = builder.add_input("beta", &[num_heads, 1]);

    let outputs = decompose_gated_delta_net(
        &mut builder,
        q,
        k,
        v,
        state,
        gate,
        beta,
        scale,
        num_heads,
        key_dim,
        value_dim,
    );
    builder.build(outputs.output)
}

/// Build a decomposed Gated DeltaNet cell that outputs both output and new_state.
///
/// Concatenates output `[H, 1, V]` and new_state `[H, K, V]` along axis 1,
/// producing shape `[H, 1+K, V]`. Caller splits at `[..1..]` (output) and
/// `[1:..]` (new state). Follows the LSTM dual-output stacking pattern.
///
/// # Errors
///
/// Returns error if any dimension is zero or scale is invalid.
#[allow(dead_code)] // Called from #[cfg(test)] and #[cfg(kani)] only
pub(crate) fn build_gated_delta_net_decomposed_dual(
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
    scale: f32,
) -> Result<TensorKernelDef, TensorIRError> {
    validate_dimensions(num_heads, key_dim, value_dim)?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(TensorIRLayerError::GatedDeltaNetScaleInvalid { value: scale }.into());
    }

    let mut builder = TensorBlockBuilder::new("gated_delta_net_decomposed_dual");
    let q = builder.add_input("q", &[num_heads, key_dim]);
    let k = builder.add_input("k", &[num_heads, key_dim]);
    let v = builder.add_input("v", &[num_heads, value_dim]);
    let state = builder.add_input("state", &[num_heads, key_dim, value_dim]);
    let gate = builder.add_input("gate", &[num_heads, 1, 1]);
    let beta = builder.add_input("beta", &[num_heads, 1]);

    let outputs = decompose_gated_delta_net(
        &mut builder,
        q,
        k,
        v,
        state,
        gate,
        beta,
        scale,
        num_heads,
        key_dim,
        value_dim,
    );

    // Reshape output [H, V] -> [H, 1, V] for concat with state [H, K, V]
    let output_3d = builder.add_reshape(outputs.output, &[num_heads, 1, value_dim]);
    let combined = builder.add_concat(
        &[output_3d, outputs.new_state],
        1,
        &[num_heads, 1 + key_dim, value_dim],
    );
    builder.build(combined)
}

#[cfg(kani)]
#[path = "gated_delta_net_kani_builder_tests.rs"]
mod kani_builder_proofs;

#[cfg(test)]
#[path = "gated_delta_net_tests.rs"]
mod tests;
