// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expert FFN implementations for Mixture-of-Experts layers.
//!
//! Two expert types:
//! - [`ExpertFFN`]: SwiGLU gated MLP (3 projections: gate + up + down with SiLU gating)
//! - [`ExpertMlp`]: Standard MLP (2 projections: up + down with configurable activation)

use super::{Activation, Linear, Module};
use crate::dyn_tensor::DynTensor;
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

// -- ExpertFFN ----------------------------------------------------------------

/// SwiGLU-style expert FFN: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
///
/// Each expert in a Mixture-of-Experts layer is a standard SwiGLU MLP
/// with three linear projections and no bias (Qwen3 convention).
///
/// This is the per-expert computation unit. For the full MoE layer with
/// routing, see [`super::MoeLayer`].
#[derive(Debug, Clone)]
pub struct ExpertFFN {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl ExpertFFN {
    /// Create from pre-built linear projections.
    ///
    /// - `gate_proj`: `[hidden_size, expert_intermediate_size]` (gating path, SiLU applied)
    /// - `up_proj`: `[hidden_size, expert_intermediate_size]` (value path)
    /// - `down_proj`: `[expert_intermediate_size, hidden_size]` (output projection)
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Result<Self> {
        let gate_out = gate_proj.weight().dim(0)?;
        let up_out = up_proj.weight().dim(0)?;
        if gate_out != up_out {
            return Err(TensorError::shape_mismatch(vec![gate_out], vec![up_out]));
        }
        let down_in = down_proj.weight().dim(1)?;
        if down_in != gate_out {
            return Err(TensorError::shape_mismatch(vec![gate_out], vec![down_in]));
        }
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Load expert weights from a [`VarBuilder`].
    ///
    /// Expects keys: `gate_proj.weight`, `up_proj.weight`, `down_proj.weight`
    /// under the given prefix.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        hidden_size: usize,
        expert_intermediate_size: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let gate_proj = Linear::load(vb.pp("gate_proj"), hidden_size, expert_intermediate_size)?;
        let up_proj = Linear::load(vb.pp("up_proj"), hidden_size, expert_intermediate_size)?;
        let down_proj = Linear::load(vb.pp("down_proj"), expert_intermediate_size, hidden_size)?;
        Self::new(gate_proj, up_proj, down_proj)
    }

    /// Access the gate projection.
    #[must_use]
    pub fn gate_proj(&self) -> &Linear {
        &self.gate_proj
    }

    /// Access the up projection.
    #[must_use]
    pub fn up_proj(&self) -> &Linear {
        &self.up_proj
    }

    /// Access the down projection.
    #[must_use]
    pub fn down_proj(&self) -> &Linear {
        &self.down_proj
    }
}

impl Module for ExpertFFN {
    /// Forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let h = gate.broadcast_mul(&up)?;
        self.down_proj.forward(&h)
    }
}

// -- ExpertMlp ----------------------------------------------------------------

/// Generic two-layer expert FFN: `down_proj(activation(up_proj(x)))`.
///
/// Unlike [`ExpertFFN`] which uses the SwiGLU gated pattern (3 projections),
/// `ExpertMlp` is the standard MLP pattern used in Mixtral and many other
/// MoE architectures: a single up-projection, an element-wise activation,
/// and a down-projection back to model dimension.
///
/// The activation is configurable via the [`Activation`] enum (ReLU, GELU,
/// SiLU, etc.).
#[derive(Debug, Clone)]
pub struct ExpertMlp {
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl ExpertMlp {
    /// Create from pre-built linear projections and an activation function.
    ///
    /// - `up_proj`: `[hidden_size, intermediate_size]` (expands to FFN dimension)
    /// - `down_proj`: `[intermediate_size, hidden_size]` (projects back)
    /// - `activation`: element-wise activation between projections
    pub fn new(up_proj: Linear, down_proj: Linear, activation: Activation) -> Result<Self> {
        let up_out = up_proj.weight().dim(0)?;
        let down_in = down_proj.weight().dim(1)?;
        if up_out != down_in {
            return Err(TensorError::shape_mismatch(vec![up_out], vec![down_in]));
        }
        Ok(Self {
            up_proj,
            down_proj,
            activation,
        })
    }

    /// Load expert weights from a [`VarBuilder`].
    ///
    /// Expects keys: `up_proj.weight`, `down_proj.weight` under the given
    /// prefix. Optional bias tensors are loaded if present.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        hidden_size: usize,
        intermediate_size: usize,
        activation: Activation,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let up_proj = Linear::load(vb.pp("up_proj"), hidden_size, intermediate_size)?;
        let down_proj = Linear::load(vb.pp("down_proj"), intermediate_size, hidden_size)?;
        Self::new(up_proj, down_proj, activation)
    }

    /// Access the up projection.
    #[must_use]
    pub fn up_proj(&self) -> &Linear {
        &self.up_proj
    }

    /// Access the down projection.
    #[must_use]
    pub fn down_proj(&self) -> &Linear {
        &self.down_proj
    }

    /// Access the activation function.
    #[must_use]
    pub fn activation(&self) -> Activation {
        self.activation
    }
}

impl Module for ExpertMlp {
    /// Forward: `down_proj(activation(up_proj(x)))`.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let h = self.up_proj.forward(x)?;
        let h = self.activation.forward(&h)?;
        self.down_proj.forward(&h)
    }
}
