// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gated DeltaNet linear attention layer for [`DynTensor`].
//!
//! Provides [`GatedDeltaNet`] as a drop-in replacement for
//! [`MultiHeadAttention`](crate::layers::MultiHeadAttention) in transformer models.
//! Used by Qwen3.5 models (arXiv 2412.06464).
//!
//! The Gated DeltaNet recurrence (per head, per timestep):
//!
//! ```text
//! decayed = gate * state                                    [H, K, V]
//! v_retrieved = k^T @ decayed                               [H, V]
//! new_state = decayed + outer(k, beta*v - beta*v_retrieved) [H, K, V]
//! output = scale * q @ new_state                            [H, V]
//! ```
//!
//! O(n) per token vs O(n²) for standard attention.
//!
//! Part of #834 — Gated DeltaNet for Qwen3.5 model support.

use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, validate_heads, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{DType, Device, Result, TensorError};

/// Recurrent state for Gated DeltaNet: one matrix per head.
///
/// Shape: `[batch, num_heads, key_dim, value_dim]`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GatedDeltaNetState {
    /// Recurrent state matrix, shape `[B, H, K, V]`.
    pub state: DynTensor,
}

impl GatedDeltaNetState {
    /// Create a new state from a tensor.
    ///
    /// Returns an error if the tensor does not have rank 4 (`[B, H, K, V]`).
    pub fn new(state: DynTensor) -> Result<Self> {
        if state.rank() != 4 {
            return Err(TensorError::InvalidShape(format!(
                "GatedDeltaNetState requires rank 4 [B, H, K, V], got rank {}",
                state.rank(),
            )));
        }
        Ok(Self { state })
    }

    /// Create a zero-initialized state.
    pub fn zeros(
        batch: usize,
        num_heads: usize,
        key_dim: usize,
        value_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let state = DynTensor::zeros(&[batch, num_heads, key_dim, value_dim], DType::F32, device)?;
        Self::new(state)
    }
}

/// Gated DeltaNet linear attention module.
///
/// Drop-in replacement for [`MultiHeadAttention`](crate::layers::MultiHeadAttention)
/// in transformer models. Maintains a recurrent state instead of materializing
/// the full attention matrix, giving O(n) per-token complexity.
///
/// # Weight projections
///
/// - `q_proj`: `[D, H * K]` — query projection
/// - `k_proj`: `[D, H * K]` — key projection
/// - `v_proj`: `[D, H * V]` — value projection
/// - `gate_proj`: `[D, H]` — decay gate (sigmoid → (0,1))
/// - `beta_proj`: `[D, H]` — write strength (sigmoid → (0,1))
/// - `out_proj`: `[H * V, D]` — output projection
#[derive(Clone)]
pub struct GatedDeltaNet {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    gate_proj: Linear,
    beta_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
    scale: f64,
}

impl std::fmt::Debug for GatedDeltaNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatedDeltaNet")
            .field("num_heads", &self.num_heads)
            .field("key_dim", &self.key_dim)
            .field("value_dim", &self.value_dim)
            .finish_non_exhaustive()
    }
}

impl GatedDeltaNet {
    /// Create from pre-loaded projection weights.
    ///
    /// - `key_dim`: per-head key dimension (K)
    /// - `value_dim`: per-head value dimension (V)
    ///
    /// Scale defaults to `1.0 / sqrt(key_dim)`.
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        gate_proj: Linear,
        beta_proj: Linear,
        out_proj: Linear,
        num_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self> {
        validate_heads(num_heads, "GatedDeltaNet")?;
        if key_dim == 0 {
            return Err(TensorError::InvalidShape(
                "GatedDeltaNet: key_dim must be > 0".into(),
            ));
        }
        if value_dim == 0 {
            return Err(TensorError::InvalidShape(
                "GatedDeltaNet: value_dim must be > 0".into(),
            ));
        }
        let scale = 1.0 / (key_dim as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            gate_proj,
            beta_proj,
            out_proj,
            num_heads,
            key_dim,
            value_dim,
            scale,
        })
    }

    /// Load from a [`VarBuilder`] with standard weight names.
    ///
    /// Loads: `q_proj.weight`, `k_proj.weight`, `v_proj.weight`,
    /// `gate_proj.weight`, `beta_proj.weight`, `out_proj.weight`
    /// (and optional `.bias` for each if `bias` is true).
    ///
    /// - `dim`: model dimension (D)
    /// - `num_heads`: number of attention heads (H)
    /// - `key_dim`: per-head key dimension (K). Typically `dim / num_heads`.
    /// - `value_dim`: per-head value dimension (V). Typically `dim / num_heads`.
    /// - `bias`: whether to load bias parameters
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        dim: usize,
        num_heads: usize,
        key_dim: usize,
        value_dim: usize,
        bias: bool,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        // new() validates all three, but we need num_heads > 0 for
        // weight shape computation below. key_dim/value_dim validated by new().
        validate_heads(num_heads, "GatedDeltaNet::load")?;

        let qk_total = num_heads * key_dim;
        let v_total = num_heads * value_dim;

        let load_linear = |prefix: &str, out_features: usize| -> Result<Linear> {
            let sub = vb.pp(prefix);
            let w = sub.get(&[out_features, dim], "weight")?;
            let b = if bias {
                Some(sub.get(&[out_features], "bias")?)
            } else {
                None
            };
            Linear::new(w, b)
        };

        let q_proj = load_linear("q_proj", qk_total)?;
        let k_proj = load_linear("k_proj", qk_total)?;
        let v_proj = load_linear("v_proj", v_total)?;
        let gate_proj = load_linear("gate_proj", num_heads)?;
        let beta_proj = load_linear("beta_proj", num_heads)?;
        let out_proj = load_linear("out_proj", dim)?;

        Self::new(
            q_proj, k_proj, v_proj, gate_proj, beta_proj, out_proj, num_heads, key_dim, value_dim,
        )
    }

    /// Run one forward step (single timestep or full sequence).
    ///
    /// # Arguments
    /// - `x`: input hidden states `[B, S, D]`
    /// - `state`: optional recurrent state `[B, H, K, V]`.
    ///   If `None`, zero-initialized.
    ///
    /// # Returns
    /// `(output, new_state)` where:
    /// - `output`: `[B, S, D]` — projected output
    /// - `new_state`: updated recurrent state (from last timestep)
    pub fn forward(
        &self,
        x: &DynTensor,
        state: Option<&GatedDeltaNetState>,
    ) -> Result<(DynTensor, GatedDeltaNetState)> {
        let (b, s, _d) = x.dims3().map_err(|_| TensorError::RankMismatch {
            expected: 3,
            actual: x.rank(),
        })?;

        let device = x.device();
        let h = self.num_heads;
        let k_dim = self.key_dim;
        let v_dim = self.value_dim;

        // Initialize state if not provided
        let mut current_state = match state {
            Some(s) => s.state.clone(),
            None => DynTensor::zeros(&[b, h, k_dim, v_dim], DType::F32, &device)?,
        };

        // Project all timesteps at once: [B, S, D] -> [B, S, ...]
        let q_all = self.q_proj.forward(x)?; // [B, S, H*K]
        let k_all = self.k_proj.forward(x)?; // [B, S, H*K]
        let v_all = self.v_proj.forward(x)?; // [B, S, H*V]
        let gate_all = self.gate_proj.forward(x)?.sigmoid()?; // [B, S, H]
        let beta_all = self.beta_proj.forward(x)?.sigmoid()?; // [B, S, H]

        // Reshape projections to per-head: [B, S, H, dim]
        let q_all = q_all.reshape([b, s, h, k_dim])?;
        let k_all = k_all.reshape([b, s, h, k_dim])?;
        let v_all = v_all.reshape([b, s, h, v_dim])?;

        let mut output_steps = Vec::with_capacity(s);

        for t in 0..s {
            // Extract timestep t: [B, H, dim]
            let q = q_all.narrow(1, t, 1)?.squeeze(1)?; // [B, H, K]
            let k = k_all.narrow(1, t, 1)?.squeeze(1)?; // [B, H, K]
            let v = v_all.narrow(1, t, 1)?.squeeze(1)?; // [B, H, V]
            let gate = gate_all.narrow(1, t, 1)?.squeeze(1)?; // [B, H]
            let beta = beta_all.narrow(1, t, 1)?.squeeze(1)?; // [B, H]

            let (out_t, new_state) = self.step(&q, &k, &v, &gate, &beta, &current_state)?;

            output_steps.push(out_t.unsqueeze(1)?); // [B, 1, H*V]
            current_state = new_state;
        }

        // Stack outputs: [B, S, H*V]
        let output_refs: Vec<&DynTensor> = output_steps.iter().collect();
        let output = DynTensor::cat(&output_refs, 1)?;

        // Project output: [B, S, H*V] -> [B, S, D]
        let output = self.out_proj.forward(&output)?;

        // Tier 1 finiteness check (#1209): multi-step recurrence can amplify NaN.
        check_output_finite(&output, "GatedDeltaNet")?;

        Ok((output, GatedDeltaNetState::new(current_state)?))
    }

    /// Single-timestep recurrence step (internal).
    ///
    /// # Arguments
    /// - `q`: `[B, H, K]`
    /// - `k`: `[B, H, K]`
    /// - `v`: `[B, H, V]`
    /// - `gate`: `[B, H]` — decay gate in (0, 1)
    /// - `beta`: `[B, H]` — write strength in (0, 1)
    /// - `state`: `[B, H, K, V]`
    ///
    /// # Returns
    /// `(output_hv, new_state)` where output is `[B, H*V]` (flattened heads).
    fn step(
        &self,
        q: &DynTensor,
        k: &DynTensor,
        v: &DynTensor,
        gate: &DynTensor,
        beta: &DynTensor,
        state: &DynTensor,
    ) -> Result<(DynTensor, DynTensor)> {
        let dims = state.dims();
        let b = dims[0];
        let h = dims[1];
        let v_dim = dims[3];

        // Step 1: Decay — gate * state
        // gate [B, H] -> [B, H, 1, 1] for broadcasting with state [B, H, K, V]
        let gate_4d = gate.unsqueeze(2)?.unsqueeze(3)?; // [B, H, 1, 1]
        let decayed = state.broadcast_mul(&gate_4d)?; // [B, H, K, V]

        // Step 2: Retrieval — v_retrieved = k^T @ decayed
        // k [B, H, K] -> [B, H, 1, K]
        let k_row = k.unsqueeze(2)?; // [B, H, 1, K]
                                     // k_row [B, H, 1, K] @ decayed [B, H, K, V] -> [B, H, 1, V]
        let v_retrieved_4d = k_row.matmul(&decayed)?; // [B, H, 1, V]
        let v_retrieved = v_retrieved_4d.squeeze(2)?; // [B, H, V]

        // Step 3: Scaled write and cancel
        // beta [B, H] -> [B, H, 1] for broadcasting with v [B, H, V]
        let beta_3d = beta.unsqueeze(2)?; // [B, H, 1]
        let beta_v = v.broadcast_mul(&beta_3d)?; // [B, H, V]
        let beta_vr = v_retrieved.broadcast_mul(&beta_3d)?; // [B, H, V]

        // Step 4: Write term — outer(k, beta*v - beta*v_retrieved)
        // Factor: outer(k, beta_v - beta_vr) uses 1 matmul instead of 2 (F9, #1241).
        let k_col = k.unsqueeze(3)?; // [B, H, K, 1]
        let bv_diff = beta_v.sub(&beta_vr)?; // [B, H, V]
        let bv_diff_row = bv_diff.unsqueeze(2)?; // [B, H, 1, V]
        let update = k_col.matmul(&bv_diff_row)?; // [B, H, K, V]

        // Step 5: State update — new_state = decayed + update
        let new_state = decayed.add(&update)?; // [B, H, K, V]

        // Step 6: Output — o = scale * q @ new_state
        // q_row [B, H, 1, K] @ new_state [B, H, K, V] -> [B, H, 1, V]
        let q_row = q.unsqueeze(2)?; // [B, H, 1, K]
        let output_4d = q_row.matmul(&new_state)?; // [B, H, 1, V]
        let output_3d = output_4d.squeeze(2)?; // [B, H, V]
        let output_scaled = output_3d.mul_scalar(self.scale)?; // [B, H, V]

        // Flatten heads: [B, H, V] -> [B, H*V]
        let output = output_scaled.reshape([b, h * v_dim])?;

        Ok((output, new_state))
    }

    /// Number of attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Per-head key dimension.
    #[must_use]
    pub fn key_dim(&self) -> usize {
        self.key_dim
    }

    /// Per-head value dimension.
    #[must_use]
    pub fn value_dim(&self) -> usize {
        self.value_dim
    }
}

#[cfg(kani)]
#[path = "kani_gated_delta_net_proofs.rs"]
mod kani_gated_delta_net_proofs;

#[cfg(test)]
#[path = "gated_delta_net_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "gated_delta_net_tests_extended.rs"]
mod tests_extended;
