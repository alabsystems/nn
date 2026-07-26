// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SwiGLU Feed-Forward Network (Shazeer 2020).
//!
//! Gated FFN used in DiT/transformer TTS models:
//!
//! ```text
//! gate = silu(x @ w_gate)
//! up   = x @ w_up
//! out  = (gate * up) @ w_down
//! ```
//!
//! Three dvoice models use this pattern:
//! - CosyVoice3: `Linear` (with bias), `ff_dim = dim * 4`
//! - Irodori-TTS: `linear_no_bias`, `ff_dim = dim * 4`
//! - Ming-omni: `linear_no_bias`, `ff_dim = dim * 4`

use super::{check_output_finite, Linear, Module};
use crate::dyn_tensor::DynTensor;
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// SwiGLU gated feedforward network.
///
/// Computes `w_down(silu(w_gate(x)) * w_up(x))`.
/// The SiLU-gated linear unit typically expands to `ff_dim = dim * 4`,
/// then projects back down to `dim`.
#[derive(Debug, Clone)]
pub struct SwiGlu {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,
}

impl SwiGlu {
    /// Create from pre-built Linear projections.
    ///
    /// - `w_gate`: Linear from `dim` to `ff_dim` (gating path, SiLU applied)
    /// - `w_up`: Linear from `dim` to `ff_dim` (value path)
    /// - `w_down`: Linear from `ff_dim` to `dim` (output projection)
    ///
    /// Returns an error if `w_gate` and `w_up` output dimensions differ,
    /// or if `w_down` input dimension does not match the gate/up output.
    pub fn new(w_gate: Linear, w_up: Linear, w_down: Linear) -> Result<Self> {
        let gate_out = w_gate.weight().dim(0)?;
        let up_out = w_up.weight().dim(0)?;
        if gate_out != up_out {
            return Err(TensorError::shape_mismatch(vec![gate_out], vec![up_out]));
        }
        let down_in = w_down.weight().dim(1)?;
        if down_in != gate_out {
            return Err(TensorError::shape_mismatch(vec![gate_out], vec![down_in]));
        }
        Ok(Self {
            w_gate,
            w_up,
            w_down,
        })
    }

    /// Load from a [`VarBuilder`] using PyTorch-style weight names.
    ///
    /// Loads `w_gate`, `w_up`, and `w_down` sub-modules (each a Linear).
    /// Weight names: `w_gate.weight`, `w_up.weight`, `w_down.weight`,
    /// plus optional `*.bias` tensors.
    ///
    /// - `dim`: Input dimension (and output dimension of `w_down`).
    /// - `hidden_dim`: Intermediate FFN dimension (`ff_dim`, typically `dim * 4`).
    pub fn load(vb: impl AsRef<VarBuilder>, dim: usize, hidden_dim: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let w_gate = Linear::load(vb.pp("w_gate"), dim, hidden_dim)?;
        let w_up = Linear::load(vb.pp("w_up"), dim, hidden_dim)?;
        let w_down = Linear::load(vb.pp("w_down"), hidden_dim, dim)?;
        Self::new(w_gate, w_up, w_down)
    }

    /// Access the gate projection.
    #[must_use]
    pub fn w_gate(&self) -> &Linear {
        &self.w_gate
    }

    /// Access the up projection.
    #[must_use]
    pub fn w_up(&self) -> &Linear {
        &self.w_up
    }

    /// Access the down projection.
    #[must_use]
    pub fn w_down(&self) -> &Linear {
        &self.w_down
    }
}

impl Module for SwiGlu {
    /// Forward pass decomposes into Linear + SiLU + Mul + Linear.
    ///
    /// Inner ops record their own trace nodes so the compiled model can
    /// dispatch each primitive directly — no composite SwiGlu compile support
    /// needed. This also enables NY verification of the subgraph.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let gate = self.w_gate.forward(x)?.silu()?;
        let up = self.w_up.forward(x)?;
        let h = gate.broadcast_mul(&up)?;
        let output = self.w_down.forward(&h)?;
        check_output_finite(&output, "SwiGlu")?;
        Ok(output)
    }
}

#[cfg(kani)]
#[path = "kani_swiglu_act_embed_proofs.rs"]
mod kani_proofs;

#[cfg(test)]
#[path = "swiglu_tests.rs"]
mod tests;
