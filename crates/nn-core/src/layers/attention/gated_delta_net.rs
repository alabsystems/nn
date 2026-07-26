// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gated DeltaNet linear attention (Yang et al., 2024; arXiv:2412.06464).
//!
//! A linear attention variant used by Qwen3.5 that replaces quadratic softmax
//! attention with a per-head recurrence. Architecture:
//!
//! 1. **Input projection:** `Linear(hidden_size, num_heads * head_dim * 3 + num_heads)`
//!    produces Q, K, V (each `[B, T, H, D]`) and beta (`[B, T, H]`).
//! 2. **Short depthwise Conv1d** on Q, K, V for local context (default kernel_size=4).
//! 3. **Delta rule recurrence** per head:
//!    ```text
//!    S_t = S_{t-1} + beta_t * (v_t (x) k_t - diag(k_t @ k_t^T) * S_{t-1})
//!    ```
//! 4. **Output:** `o_t = S_t @ q_t`, then output projection back to `hidden_size`.
//!
//! O(n) per token vs O(n^2) for standard attention.
//!
//! Part of #834 — Gated DeltaNet for Qwen3.5 model support.

use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, validate_heads, Conv1d, Conv1dConfig, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{DType, Device, Result, TensorError};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`GatedDeltaNet`].
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetConfig {
    /// Model hidden dimension (input/output).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head dimension for Q, K, V.
    pub head_dim: usize,
    /// Kernel size for the short depthwise Conv1d on Q/K/V (default 4).
    pub conv_kernel_size: usize,
}

impl GatedDeltaNetConfig {
    /// Validate the configuration, returning an error for invalid values.
    pub fn validate(&self) -> Result<()> {
        validate_heads(self.num_heads, "GatedDeltaNetConfig")?;
        if self.hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "GatedDeltaNetConfig: hidden_size must be > 0",
            });
        }
        if self.head_dim == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "GatedDeltaNetConfig: head_dim must be > 0",
            });
        }
        if self.conv_kernel_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "GatedDeltaNetConfig: conv_kernel_size must be > 0",
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Recurrent state
// ---------------------------------------------------------------------------

/// Recurrent state for Gated DeltaNet: one `[head_dim, head_dim]` matrix per
/// batch element per head.
///
/// Shape: `[batch, num_heads, head_dim, head_dim]`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GatedDeltaNetState {
    /// Recurrent state matrix S, shape `[B, H, D, D]`.
    pub state: DynTensor,
}

impl GatedDeltaNetState {
    /// Create a new state from a tensor.
    ///
    /// Returns an error if the tensor is not rank 4.
    pub fn new(state: DynTensor) -> Result<Self> {
        if state.rank() != 4 {
            return Err(TensorError::InvalidShape(format!(
                "GatedDeltaNetState requires rank 4 [B, H, D, D], got rank {}",
                state.rank(),
            )));
        }
        Ok(Self { state })
    }

    /// Create a zero-initialized state.
    pub fn zeros(batch: usize, num_heads: usize, head_dim: usize, device: &Device) -> Result<Self> {
        let state = DynTensor::zeros(&[batch, num_heads, head_dim, head_dim], DType::F32, device)?;
        Self::new(state)
    }
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// Gated DeltaNet linear attention layer.
///
/// Implements the full paper architecture (Yang et al., 2024):
/// - Single input projection producing Q, K, V, beta
/// - Short depthwise Conv1d for local context on Q, K, V
/// - Delta rule recurrence per head
/// - Output projection
#[derive(Clone)]
pub struct GatedDeltaNet {
    /// Input projection: hidden_size -> (3 * num_heads * head_dim + num_heads).
    in_proj: Linear,
    /// Short depthwise Conv1d on Q channel dimension.
    q_conv: Conv1d,
    /// Short depthwise Conv1d on K channel dimension.
    k_conv: Conv1d,
    /// Short depthwise Conv1d on V channel dimension.
    v_conv: Conv1d,
    /// Output projection: num_heads * head_dim -> hidden_size.
    out_proj: Linear,
    cfg: GatedDeltaNetConfig,
    /// Precomputed 1/sqrt(head_dim).
    scale: f64,
}

impl std::fmt::Debug for GatedDeltaNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatedDeltaNet")
            .field("hidden_size", &self.cfg.hidden_size)
            .field("num_heads", &self.cfg.num_heads)
            .field("head_dim", &self.cfg.head_dim)
            .field("conv_kernel_size", &self.cfg.conv_kernel_size)
            .finish_non_exhaustive()
    }
}

impl GatedDeltaNet {
    /// Construct from pre-loaded weights.
    ///
    /// Caller is responsible for ensuring weight shapes match the config.
    pub fn new(
        in_proj: Linear,
        q_conv: Conv1d,
        k_conv: Conv1d,
        v_conv: Conv1d,
        out_proj: Linear,
        cfg: GatedDeltaNetConfig,
    ) -> Result<Self> {
        cfg.validate()?;
        let scale = 1.0 / (cfg.head_dim as f64).sqrt();
        Ok(Self {
            in_proj,
            q_conv,
            k_conv,
            v_conv,
            out_proj,
            cfg,
            scale,
        })
    }

    /// Load from a [`VarBuilder`] with standard weight names.
    ///
    /// Expected weight names under the `vb` prefix:
    /// - `in_proj.weight` (and optional `.bias`)
    /// - `q_conv.weight`, `q_conv.bias`
    /// - `k_conv.weight`, `k_conv.bias`
    /// - `v_conv.weight`, `v_conv.bias`
    /// - `out_proj.weight` (and optional `.bias`)
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: GatedDeltaNetConfig) -> Result<Self> {
        cfg.validate()?;
        let vb = vb.as_ref();

        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let hd = h * d;
        let in_proj_out = 3 * hd + h; // Q + K + V + beta

        // Input projection
        let in_w = vb
            .pp("in_proj")
            .get(&[in_proj_out, cfg.hidden_size], "weight")?;
        let in_b = vb.pp("in_proj").get(&[in_proj_out], "bias").ok();
        let in_proj = Linear::new(in_w, in_b)?;

        // Depthwise Conv1d for Q, K, V: groups = channels (depthwise)
        let conv_cfg = Conv1dConfig {
            padding: cfg.conv_kernel_size - 1, // causal padding
            stride: 1,
            dilation: 1,
            groups: hd, // depthwise
        };

        let load_conv = |prefix: &str| -> Result<Conv1d> {
            let sub = vb.pp(prefix);
            let w = sub.get(&[hd, 1, cfg.conv_kernel_size], "weight")?;
            let b = sub.get(&[hd], "bias").ok();
            Conv1d::new(w, b, conv_cfg)
        };

        let q_conv = load_conv("q_conv")?;
        let k_conv = load_conv("k_conv")?;
        let v_conv = load_conv("v_conv")?;

        // Output projection
        let out_w = vb.pp("out_proj").get(&[cfg.hidden_size, hd], "weight")?;
        let out_b = vb.pp("out_proj").get(&[cfg.hidden_size], "bias").ok();
        let out_proj = Linear::new(out_w, out_b)?;

        Self::new(in_proj, q_conv, k_conv, v_conv, out_proj, cfg)
    }

    /// Full-sequence forward pass.
    ///
    /// Processes the entire sequence, returning output and the final recurrent
    /// state (from the last timestep).
    ///
    /// # Arguments
    /// - `x`: input hidden states `[B, T, hidden_size]`
    /// - `state`: optional initial recurrent state `[B, H, D, D]`
    ///
    /// # Returns
    /// `(output, final_state)` where output is `[B, T, hidden_size]`.
    pub fn forward(
        &self,
        x: &DynTensor,
        state: Option<&GatedDeltaNetState>,
    ) -> Result<(DynTensor, GatedDeltaNetState)> {
        let (b, t, _hidden) = x.dims3().map_err(|_| TensorError::RankMismatch {
            expected: 3,
            actual: x.rank(),
        })?;

        let h = self.cfg.num_heads;
        let d = self.cfg.head_dim;
        let hd = h * d;

        // --- Input projection: [B, T, hidden] -> [B, T, 3*H*D + H] ---
        let projected = self.in_proj.forward(x)?;

        // Split into Q, K, V, beta
        let qkv = projected.narrow(2, 0, 3 * hd)?;
        let beta_raw = projected.narrow(2, 3 * hd, h)?; // [B, T, H]

        let q_raw = qkv.narrow(2, 0, hd)?; // [B, T, H*D]
        let k_raw = qkv.narrow(2, hd, hd)?; // [B, T, H*D]
        let v_raw = qkv.narrow(2, 2 * hd, hd)?; // [B, T, H*D]

        // --- Short depthwise Conv1d for local context ---
        // Conv1d expects [B, C, T], so transpose [B, T, C] -> [B, C, T]
        let q_conv_in = q_raw.transpose(1, 2)?; // [B, H*D, T]
        let k_conv_in = k_raw.transpose(1, 2)?;
        let v_conv_in = v_raw.transpose(1, 2)?;

        let q_conv_out = self.q_conv.forward(&q_conv_in)?; // [B, H*D, T + pad]
        let k_conv_out = self.k_conv.forward(&k_conv_in)?;
        let v_conv_out = self.v_conv.forward(&v_conv_in)?;

        // Causal trim: keep only the first T elements (causal = left-padded)
        let q_conv_out = q_conv_out.narrow(2, 0, t)?;
        let k_conv_out = k_conv_out.narrow(2, 0, t)?;
        let v_conv_out = v_conv_out.narrow(2, 0, t)?;

        // Apply SiLU activation after conv (standard in Gated DeltaNet)
        let q_post = q_conv_out.silu()?;
        let k_post = k_conv_out.silu()?;
        let v_post = v_conv_out.silu()?;

        // Transpose back: [B, H*D, T] -> [B, T, H*D]
        let q_post = q_post.transpose(1, 2)?;
        let k_post = k_post.transpose(1, 2)?;
        let v_post = v_post.transpose(1, 2)?;

        // Reshape to per-head: [B, T, H, D]
        let q_all = q_post.reshape([b, t, h, d])?;
        let k_all = k_post.reshape([b, t, h, d])?;
        let v_all = v_post.reshape([b, t, h, d])?;

        // Beta: sigmoid -> (0, 1), shape [B, T, H]
        let beta_all = beta_raw.sigmoid()?;

        // --- Delta rule recurrence ---
        let device = x.device();
        let mut current_state = match state {
            Some(s) => s.state.clone(),
            None => DynTensor::zeros(&[b, h, d, d], DType::F32, &device)?,
        };

        let mut output_steps = Vec::with_capacity(t);

        for step in 0..t {
            let q = q_all.narrow(1, step, 1)?.squeeze(1)?; // [B, H, D]
            let k = k_all.narrow(1, step, 1)?.squeeze(1)?; // [B, H, D]
            let v = v_all.narrow(1, step, 1)?.squeeze(1)?; // [B, H, D]
            let beta = beta_all.narrow(1, step, 1)?.squeeze(1)?; // [B, H]

            let (out_t, new_state) = delta_step(&q, &k, &v, &beta, &current_state, self.scale)?;

            output_steps.push(out_t.unsqueeze(1)?); // [B, 1, H*D]
            current_state = new_state;
        }

        // Concatenate timesteps: [B, T, H*D]
        let output_refs: Vec<&DynTensor> = output_steps.iter().collect();
        let output = DynTensor::cat(&output_refs, 1)?;

        // Output projection: [B, T, H*D] -> [B, T, hidden_size]
        let output = self.out_proj.forward(&output)?;

        check_output_finite(&output, "GatedDeltaNet")?;

        Ok((output, GatedDeltaNetState::new(current_state)?))
    }

    /// Single-step recurrent forward (for autoregressive inference).
    ///
    /// Processes one token at a time. The caller manages the Conv1d history
    /// externally (pass the token through projection + conv before calling,
    /// or use this simplified version that skips conv for single-step).
    ///
    /// # Arguments
    /// - `x`: input for a single timestep `[B, 1, hidden_size]`
    /// - `state`: recurrent state from previous step `[B, H, D, D]`
    ///
    /// # Returns
    /// `(output, new_state)` where output is `[B, 1, hidden_size]`.
    pub fn forward_recurrent(
        &self,
        x: &DynTensor,
        state: &GatedDeltaNetState,
    ) -> Result<(DynTensor, GatedDeltaNetState)> {
        // Delegate to the full forward path with T=1.
        // Conv1d with causal padding on T=1 still works correctly (uses only
        // the current token; prior context comes from the recurrent state).
        self.forward(x, Some(state))
    }

    /// Number of attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.cfg.num_heads
    }

    /// Per-head dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.cfg.head_dim
    }

    /// Configuration used to create this layer.
    #[must_use]
    pub fn config(&self) -> &GatedDeltaNetConfig {
        &self.cfg
    }
}

// ---------------------------------------------------------------------------
// Delta rule recurrence step (standalone for reuse)
// ---------------------------------------------------------------------------

/// Single-timestep delta-rule recurrence.
///
/// Implements:
/// ```text
/// S_t = S_{t-1} + beta_t * (v_t (x) k_t - diag(k_t @ k_t^T) * S_{t-1})
/// o_t = scale * S_t @ q_t
/// ```
///
/// # Arguments
/// - `q`: `[B, H, D]`
/// - `k`: `[B, H, D]`
/// - `v`: `[B, H, D]`
/// - `beta`: `[B, H]` — write strength in (0, 1)
/// - `state`: `[B, H, D, D]` — recurrent state S_{t-1}
/// - `scale`: typically `1/sqrt(head_dim)`
///
/// # Returns
/// `(output, new_state)` where output is `[B, H*D]` (flattened heads)
/// and new_state is `[B, H, D, D]`.
fn delta_step(
    q: &DynTensor,
    k: &DynTensor,
    v: &DynTensor,
    beta: &DynTensor,
    state: &DynTensor,
    scale: f64,
) -> Result<(DynTensor, DynTensor)> {
    let dims = state.dims();
    let b = dims[0];
    let h = dims[1];
    let d = dims[3]; // head_dim (K == V == D for this variant)

    // --- Outer product: v_t (x) k_t -> [B, H, D, D] ---
    // v: [B, H, D] -> [B, H, D, 1]
    // k: [B, H, D] -> [B, H, 1, D]
    let v_col = v.unsqueeze(3)?; // [B, H, D, 1]
    let k_row = k.unsqueeze(2)?; // [B, H, 1, D]
    let outer_vk = v_col.matmul(&k_row)?; // [B, H, D, D]

    // --- Erase term: diag(k_t @ k_t^T) * S_{t-1} ---
    // diag(k k^T) = element-wise k^2 along D, applied as scaling per row of S.
    // k^2: [B, H, D] -> [B, H, D, 1] for broadcast with S [B, H, D, D]
    let k_sq = k.mul(k)?; // [B, H, D]
    let k_sq_col = k_sq.unsqueeze(3)?; // [B, H, D, 1]
                                       // erase_term[i, j] = k_sq[i] * S[i, j]  (broadcast over last dim j)
                                       // Actually: diag(k k^T) * S means row-i of S is scaled by (k^T k)_ii = k_i^2
                                       // So we broadcast k_sq over the V dimension (columns of S).
                                       // k_row [B, H, 1, D] broadcasts across first D dim of S.
                                       // But diag(k @ k^T) is a diagonal matrix: entry (i,i) = k_i^2.
                                       // diag(K) * S means row i of result = k_i^2 * row i of S.
                                       // So we need k_sq as [B, H, D, 1] to broadcast with S [B, H, D, D].
    let erase = state.broadcast_mul(&k_sq_col)?; // [B, H, D, D]

    // --- Delta update: beta * (outer_vk - erase) ---
    let delta = outer_vk.sub(&erase)?; // [B, H, D, D]
    let beta_4d = beta.unsqueeze(2)?.unsqueeze(3)?; // [B, H, 1, 1]
    let update = delta.broadcast_mul(&beta_4d)?; // [B, H, D, D]

    // --- State update: S_t = S_{t-1} + update ---
    let new_state = state.add(&update)?; // [B, H, D, D]

    // --- Output: o_t = scale * S_t @ q_t ---
    // q: [B, H, D] -> [B, H, D, 1]
    let q_col = q.unsqueeze(3)?; // [B, H, D, 1]
    let output_4d = new_state.matmul(&q_col)?; // [B, H, D, 1]
    let output_3d = output_4d.squeeze(3)?; // [B, H, D]
    let output_scaled = output_3d.mul_scalar(scale)?; // [B, H, D]

    // Flatten heads: [B, H, D] -> [B, H*D]
    let output = output_scaled.reshape([b, h * d])?;

    Ok((output, new_state))
}

#[cfg(test)]
#[path = "gated_delta_net_tests.rs"]
mod tests;
