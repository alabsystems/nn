// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-Token Prediction (MTP) head for decoder models.
//!
//! MTP predicts N future tokens simultaneously from the same hidden state,
//! enabling speculative decoding verification and improved training signal.
//!
//! Architecture (GLM-OCR / DeepSeek-V3 pattern):
//!
//! ```text
//! hidden_states [B, T, D]
//!       |
//!   ┌───┴───┬───────┬───────┐
//!   │       │       │       │
//! head_0  head_1  head_2  head_3    (N parallel projections)
//!   │       │       │       │
//! [B,T,V] [B,T,V] [B,T,V] [B,T,V]  (per-head logits)
//!   │       │       │       │
//!   └───┬───┴───────┴───────┘
//!       |
//! [B, T, N, V]                       (stacked output)
//! ```
//!
//! Each prediction head is a [`Linear`] projection from `hidden_dim` to
//! `vocab_size`. Head 0 predicts the next token (t+1), head 1 predicts t+2,
//! and so on. During inference, all N predictions can be verified in parallel
//! via speculative decoding.
//!
//! # Variants
//!
//! - **Independent heads** (default): Each head is a separate `Linear(D → V)`.
//!   Simple, no parameter sharing. Used by GLM-OCR.
//!
//! - **Shared trunk + per-head projection**: A shared `Linear(D → D)` + per-head
//!   `Linear(D → V)`. Reduces parameters when `vocab_size >> hidden_dim`.
//!   Enabled via [`MtpHeadConfig::shared_trunk`].
//!
//! # Weight loading
//!
//! ```text
//! mtp.heads.0.weight          [V, D]
//! mtp.heads.0.bias            [V]     (optional)
//! mtp.heads.1.weight          [V, D]
//! ...
//! mtp.shared_trunk.weight     [D, D]  (only if shared_trunk = true)
//! mtp.shared_trunk.bias       [D]     (optional, only if shared_trunk = true)
//! ```

use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, Linear, Module, RmsNorm};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`MtpHead`].
#[derive(Debug, Clone)]
pub struct MtpHeadConfig {
    /// Number of future tokens to predict simultaneously.
    /// Head 0 predicts t+1, head 1 predicts t+2, etc.
    pub num_predict_tokens: usize,

    /// Hidden dimension of the input (model backbone output).
    pub hidden_size: usize,

    /// Vocabulary size (output dimension of each prediction head).
    pub vocab_size: usize,

    /// If `true`, a shared `Linear(D -> D)` trunk is applied before the
    /// per-head projections. Reduces total parameters when `V >> D`.
    pub shared_trunk: bool,

    /// If `true`, apply per-head RmsNorm before the linear projection.
    /// Common in DeepSeek-V3 MTP architecture.
    pub per_head_norm: bool,

    /// Epsilon for per-head RmsNorm (only used when `per_head_norm` is true).
    pub norm_eps: f64,
}

impl MtpHeadConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.num_predict_tokens == 0 {
            return Err(TensorError::InvalidShape(
                "MtpHeadConfig: num_predict_tokens must be > 0".into(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(TensorError::InvalidShape(
                "MtpHeadConfig: hidden_size must be > 0".into(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(TensorError::InvalidShape(
                "MtpHeadConfig: vocab_size must be > 0".into(),
            ));
        }
        if self.per_head_norm && !self.norm_eps.is_finite() {
            return Err(TensorError::InvalidShape(
                "MtpHeadConfig: norm_eps must be finite".into(),
            ));
        }
        Ok(())
    }
}

impl Default for MtpHeadConfig {
    fn default() -> Self {
        Self {
            num_predict_tokens: 4,
            hidden_size: 256,
            vocab_size: 1000,
            shared_trunk: false,
            per_head_norm: false,
            norm_eps: 1e-5,
        }
    }
}

// ---------------------------------------------------------------------------
// MtpHead
// ---------------------------------------------------------------------------

/// Multi-Token Prediction head.
///
/// Predicts `num_predict_tokens` future tokens in parallel from the same
/// hidden state. Used by GLM-OCR and DeepSeek-V3 for speculative decoding
/// and improved training signal.
///
/// See [module documentation](self) for architecture details.
#[derive(Debug, Clone)]
pub struct MtpHead {
    /// Per-head linear projections: `hidden_size -> vocab_size`.
    heads: Vec<Linear>,

    /// Optional shared trunk applied before per-head projections.
    shared_trunk: Option<Linear>,

    /// Optional per-head RmsNorm (applied before linear projection).
    head_norms: Vec<RmsNorm>,

    /// Configuration.
    cfg: MtpHeadConfig,
}

impl MtpHead {
    /// Create an MTP head from pre-built components.
    ///
    /// - `heads`: One `Linear(D -> V)` per prediction token.
    /// - `shared_trunk`: Optional `Linear(D -> D)` applied before per-head projections.
    /// - `head_norms`: Per-head RmsNorm layers (empty if not using per-head norm).
    /// - `cfg`: Configuration (must match component dimensions).
    pub fn new(
        heads: Vec<Linear>,
        shared_trunk: Option<Linear>,
        head_norms: Vec<RmsNorm>,
        cfg: MtpHeadConfig,
    ) -> Result<Self> {
        cfg.validate()?;

        if heads.len() != cfg.num_predict_tokens {
            return Err(TensorError::InvalidShape(format!(
                "MtpHead: expected {} heads, got {}",
                cfg.num_predict_tokens,
                heads.len()
            )));
        }

        // Validate head dimensions.
        for (i, head) in heads.iter().enumerate() {
            if head.out_features() != cfg.vocab_size {
                return Err(TensorError::InvalidShape(format!(
                    "MtpHead: head {i} out_features ({}) != vocab_size ({})",
                    head.out_features(),
                    cfg.vocab_size
                )));
            }
            if head.in_features() != cfg.hidden_size {
                return Err(TensorError::InvalidShape(format!(
                    "MtpHead: head {i} in_features ({}) != hidden_size ({})",
                    head.in_features(),
                    cfg.hidden_size
                )));
            }
        }

        // Validate shared trunk dimensions if present.
        if let Some(ref trunk) = shared_trunk {
            if !cfg.shared_trunk {
                return Err(TensorError::InvalidShape(
                    "MtpHead: shared_trunk provided but config.shared_trunk is false".into(),
                ));
            }
            if trunk.in_features() != cfg.hidden_size || trunk.out_features() != cfg.hidden_size {
                return Err(TensorError::InvalidShape(format!(
                    "MtpHead: shared_trunk must be [{D}, {D}], got [{}, {}]",
                    trunk.out_features(),
                    trunk.in_features(),
                    D = cfg.hidden_size
                )));
            }
        } else if cfg.shared_trunk {
            return Err(TensorError::InvalidShape(
                "MtpHead: config.shared_trunk is true but no trunk provided".into(),
            ));
        }

        // Validate per-head norms.
        if cfg.per_head_norm {
            if head_norms.len() != cfg.num_predict_tokens {
                return Err(TensorError::InvalidShape(format!(
                    "MtpHead: expected {} head_norms, got {}",
                    cfg.num_predict_tokens,
                    head_norms.len()
                )));
            }
        } else if !head_norms.is_empty() {
            return Err(TensorError::InvalidShape(
                "MtpHead: head_norms provided but per_head_norm is false".into(),
            ));
        }

        Ok(Self {
            heads,
            shared_trunk,
            head_norms,
            cfg,
        })
    }

    /// Load an MTP head from a [`VarBuilder`].
    ///
    /// Weight names:
    /// - `heads.{i}.weight` `[vocab_size, hidden_size]`
    /// - `heads.{i}.bias` `[vocab_size]` (optional, auto-detected)
    /// - `shared_trunk.weight` `[hidden_size, hidden_size]` (if `shared_trunk`)
    /// - `shared_trunk.bias` `[hidden_size]` (optional)
    /// - `head_norms.{i}.weight` `[hidden_size]` (if `per_head_norm`)
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: MtpHeadConfig) -> Result<Self> {
        let vb = vb.as_ref();
        cfg.validate()?;

        let mut heads = Vec::with_capacity(cfg.num_predict_tokens);
        for i in 0..cfg.num_predict_tokens {
            let head_vb = vb.pp(format!("heads.{i}"));
            heads.push(Linear::load(&head_vb, cfg.hidden_size, cfg.vocab_size)?);
        }

        let shared_trunk = if cfg.shared_trunk {
            let trunk_vb = vb.pp("shared_trunk");
            Some(Linear::load(&trunk_vb, cfg.hidden_size, cfg.hidden_size)?)
        } else {
            None
        };

        let mut head_norms = Vec::new();
        if cfg.per_head_norm {
            for i in 0..cfg.num_predict_tokens {
                let norm_vb = vb.pp(format!("head_norms.{i}"));
                let weight = norm_vb.get(&[cfg.hidden_size], "weight")?;
                head_norms.push(RmsNorm::new(weight, cfg.norm_eps)?);
            }
        }

        Self::new(heads, shared_trunk, head_norms, cfg)
    }

    /// Forward pass: hidden states to multi-token logits.
    ///
    /// - Input: `hidden_states` of shape `[B, T, D]`
    /// - Output: logits of shape `[B, T, N, V]`
    ///
    /// where `N = num_predict_tokens`, `D = hidden_size`, `V = vocab_size`.
    ///
    /// Each head `i` produces logits for position `t + i + 1`:
    /// - Head 0: next-token prediction (same as standard LM head)
    /// - Head 1: two-tokens-ahead prediction
    /// - etc.
    pub fn forward(&self, hidden_states: &DynTensor) -> Result<DynTensor> {
        let (batch, seq_len, hidden_dim) = hidden_states.dims3()?;
        if hidden_dim != self.cfg.hidden_size {
            return Err(TensorError::InvalidShape(format!(
                "MtpHead: input hidden_dim ({hidden_dim}) != config hidden_size ({})",
                self.cfg.hidden_size
            )));
        }

        // Optional shared trunk transform.
        let trunk_out = match &self.shared_trunk {
            Some(trunk) => trunk.forward(hidden_states)?,
            None => hidden_states.clone(),
        };

        // Compute per-head logits and stack.
        let mut head_logits = Vec::with_capacity(self.cfg.num_predict_tokens);
        for (i, head) in self.heads.iter().enumerate() {
            let h = if self.cfg.per_head_norm {
                self.head_norms[i].forward(&trunk_out)?
            } else {
                trunk_out.clone()
            };
            let logits_i = head.forward(&h)?; // [B, T, V]
            head_logits.push(logits_i);
        }

        // Stack along a new dimension: [B, T, V] x N -> [B, T, N, V].
        // We reshape each [B, T, V] to [B, T, 1, V], then cat along dim 2.
        let mut reshaped = Vec::with_capacity(head_logits.len());
        for logits_i in &head_logits {
            reshaped.push(logits_i.reshape([batch, seq_len, 1, self.cfg.vocab_size])?);
        }
        let output = DynTensor::cat(&reshaped, 2)?;
        check_output_finite(&output, "MtpHead")?;
        Ok(output)
    }

    /// Forward pass returning per-head logits as separate tensors.
    ///
    /// Returns a `Vec<DynTensor>` of length `num_predict_tokens`, where each
    /// element has shape `[B, T, V]`. Useful when heads need to be consumed
    /// independently (e.g., speculative decoding verification).
    pub fn forward_per_head(&self, hidden_states: &DynTensor) -> Result<Vec<DynTensor>> {
        let (_batch, _seq_len, hidden_dim) = hidden_states.dims3()?;
        if hidden_dim != self.cfg.hidden_size {
            return Err(TensorError::InvalidShape(format!(
                "MtpHead: input hidden_dim ({hidden_dim}) != config hidden_size ({})",
                self.cfg.hidden_size
            )));
        }

        let trunk_out = match &self.shared_trunk {
            Some(trunk) => trunk.forward(hidden_states)?,
            None => hidden_states.clone(),
        };

        let mut head_logits = Vec::with_capacity(self.cfg.num_predict_tokens);
        for (i, head) in self.heads.iter().enumerate() {
            let h = if self.cfg.per_head_norm {
                self.head_norms[i].forward(&trunk_out)?
            } else {
                trunk_out.clone()
            };
            let logits_i = head.forward(&h)?;
            check_output_finite(&logits_i, &format!("MtpHead.head_{i}"))?;
            head_logits.push(logits_i);
        }

        Ok(head_logits)
    }

    /// Number of prediction heads (future tokens predicted).
    #[must_use]
    pub fn num_predict_tokens(&self) -> usize {
        self.cfg.num_predict_tokens
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &MtpHeadConfig {
        &self.cfg
    }

    /// Access a specific prediction head.
    #[must_use]
    pub fn head(&self, index: usize) -> Option<&Linear> {
        self.heads.get(index)
    }

    /// Access the shared trunk (if present).
    #[must_use]
    pub fn shared_trunk(&self) -> Option<&Linear> {
        self.shared_trunk.as_ref()
    }
}

#[cfg(kani)]
#[path = "kani_mtp_head.rs"]
mod kani_mtp_head;

#[cfg(test)]
#[path = "mtp_head_tests.rs"]
mod tests;
