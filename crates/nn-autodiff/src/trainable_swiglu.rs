// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trainable SwiGLU feed-forward with [`Var`] weights.
//!
//! SwiGLU = SiLU(W_gate @ x) * (W_up @ x), then W_down projects back.
//! Used in Qwen3 and LLaMA transformer FFN blocks.

use crate::error::Result;
use crate::tracked::TrackedTensor;
use crate::trainable::{TrainableLinear, TrainableModule};
use crate::var::Var;
use std::sync::Arc;

/// Trainable SwiGLU feed-forward layer.
///
/// Three trainable linear projections:
/// - `w_gate`: `[dim, hidden_dim]` — gating path through SiLU
/// - `w_up`: `[dim, hidden_dim]` — up-projection
/// - `w_down`: `[hidden_dim, dim]` — down-projection
///
/// Forward: `output = W_down(SiLU(W_gate(x)) * W_up(x))`
///
/// Required for Qwen3 and LLaMA MLP fine-tuning.
#[derive(Debug, Clone)]
pub struct TrainableSwiGlu {
    w_gate: TrainableLinear,
    w_up: TrainableLinear,
    w_down: TrainableLinear,
}

impl TrainableSwiGlu {
    /// Create from existing [`TrainableLinear`] projections.
    pub fn new(w_gate: TrainableLinear, w_up: TrainableLinear, w_down: TrainableLinear) -> Self {
        Self {
            w_gate,
            w_up,
            w_down,
        }
    }

    /// Create with zero-initialized weights.
    ///
    /// `dim` = input/output dimension, `hidden_dim` = intermediate dimension.
    pub fn zeros(dim: usize, hidden_dim: usize, bias: bool) -> Result<Self> {
        let w_gate = TrainableLinear::new(dim, hidden_dim, bias)?;
        let w_up = TrainableLinear::new(dim, hidden_dim, bias)?;
        let w_down = TrainableLinear::new(hidden_dim, dim, bias)?;
        Ok(Self::new(w_gate, w_up, w_down))
    }

    /// Reference to the gate projection weights.
    #[must_use]
    pub fn w_gate(&self) -> &TrainableLinear {
        &self.w_gate
    }

    /// Reference to the up-projection weights.
    #[must_use]
    pub fn w_up(&self) -> &TrainableLinear {
        &self.w_up
    }

    /// Reference to the down-projection weights.
    #[must_use]
    pub fn w_down(&self) -> &TrainableLinear {
        &self.w_down
    }
}

impl TrainableModule for TrainableSwiGlu {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        // gate = SiLU(W_gate @ x)
        let gate = self.w_gate.forward(x)?.silu()?;
        // up = W_up @ x
        let up = self.w_up.forward(x)?;
        // h = gate * up (element-wise)
        let h = gate.mul(&up)?;
        // output = W_down @ h
        self.w_down.forward(&h)
    }

    fn vars(&self) -> Vec<&Var> {
        let mut v = self.w_gate.vars();
        v.extend(self.w_up.vars());
        v.extend(self.w_down.vars());
        v
    }
}
