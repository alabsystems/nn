// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Diffusion Transformer (DiT) block composites.
//!
//! Two variants matching dvoice model architectures:
//!
//! - [`DiTBlock`]: 3-param AdaLN (Ming-omni, Irodori) — two separate AdaLN modules
//!   for attention and FFN sub-blocks.
//! - [`DiTBlockDual`]: 6-param AdaLN (CosyVoice3) — single `AdaLnZeroDual` produces
//!   modulation for both sub-blocks from one projection.
//!
//! ## Forward pattern
//!
//! ```text
//! (mod_attn, gate_attn) = adaln_attn(x, cond)
//! x = x + gate_attn * attn(mod_attn)
//! (mod_ffn, gate_ffn) = adaln_ffn(x, cond)
//! x = x + gate_ffn * ffn(mod_ffn)
//! ```

use super::{check_output_finite, Module};
use crate::dyn_tensor::DynTensor;
use crate::Result;

use super::{apply_adaln_modulation, AdaLnZero, AdaLnZeroDual};

// -- DiTBlock (3-param variant) -----------------------------------------------

/// DiT block with two separate 3-param AdaLN modules (Ming-omni, Irodori pattern).
///
/// Each sub-block (attention, FFN) has its own AdaLN modulation. The conditioning
/// signal is shared between both AdaLN modules.
///
/// **No `load()` method:** DiTBlock is a composite type whose sub-blocks
/// (attention, FFN) are trait objects — the concrete types vary per model.
/// Construct manually via [`DiTBlock::new()`] after loading sub-components
/// with their own `load()` methods (e.g., [`AdaLnZero::load()`],
/// [`SwiGlu::load()`], [`JointAttention::load()`]).
pub struct DiTBlock {
    adaln_attn: AdaLnZero,
    attn: Box<dyn Module + Send + Sync>,
    adaln_ffn: AdaLnZero,
    ffn: Box<dyn Module + Send + Sync>,
}

impl std::fmt::Debug for DiTBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiTBlock")
            .field("adaln_attn", &self.adaln_attn)
            .field("adaln_ffn", &self.adaln_ffn)
            .finish_non_exhaustive()
    }
}

impl DiTBlock {
    /// Create from pre-built components.
    ///
    /// - `adaln_attn`: AdaLN for the attention sub-block
    /// - `attn`: Self-attention or cross-attention module
    /// - `adaln_ffn`: AdaLN for the FFN sub-block
    /// - `ffn`: Feed-forward network (typically SwiGlu)
    pub fn new(
        adaln_attn: AdaLnZero,
        attn: Box<dyn Module + Send + Sync>,
        adaln_ffn: AdaLnZero,
        ffn: Box<dyn Module + Send + Sync>,
    ) -> Result<Self> {
        Ok(Self {
            adaln_attn,
            attn,
            adaln_ffn,
            ffn,
        })
    }

    /// Forward: adaln → attn → gated residual → adaln → ffn → gated residual.
    ///
    /// - `x`: input tensor `[B, S, dim]`
    /// - `cond`: conditioning signal `[B, cond_dim]` or `[B, S, cond_dim]`
    pub fn forward(&self, x: &DynTensor, cond: &DynTensor) -> Result<DynTensor> {
        // Attention sub-block
        let (modulated, gate_attn) = self.adaln_attn.forward(x, cond)?;
        let attn_out = self.attn.forward(&modulated)?;
        let x = (x + &gate_attn.broadcast_mul(&attn_out)?)?;

        // FFN sub-block
        let (modulated, gate_ffn) = self.adaln_ffn.forward(&x, cond)?;
        let ffn_out = self.ffn.forward(&modulated)?;
        let output = (&x + &gate_ffn.broadcast_mul(&ffn_out)?)?;
        check_output_finite(&output, "DiTBlock")?;
        Ok(output)
    }
}

// -- DiTBlockDual (6-param CosyVoice3 variant) --------------------------------

/// DiT block with 6-param AdaLN (CosyVoice3 pattern).
///
/// A single `AdaLnZeroDual` produces (scale1, shift1, gate1, scale2, shift2, gate2)
/// from the timestep embedding. Stream 1 modulates attention, stream 2 modulates FFN.
///
/// This variant also takes explicit normalization modules for each sub-block,
/// since `AdaLnZeroDual` doesn't include normalization (it only projects
/// the conditioning).
///
/// **No `load()` method:** DiTBlockDual is a composite type whose sub-blocks
/// are trait objects. Construct manually via [`DiTBlockDual::new()`] after
/// loading sub-components with their own `load()` methods
/// (e.g., [`AdaLnZeroDual::load()`]).
pub struct DiTBlockDual {
    adaln: AdaLnZeroDual,
    norm_attn: Box<dyn Module + Send + Sync>,
    attn: Box<dyn Module + Send + Sync>,
    norm_ffn: Box<dyn Module + Send + Sync>,
    ffn: Box<dyn Module + Send + Sync>,
}

impl std::fmt::Debug for DiTBlockDual {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiTBlockDual")
            .field("adaln", &self.adaln)
            .finish_non_exhaustive()
    }
}

impl DiTBlockDual {
    /// Create from pre-built components.
    ///
    /// - `adaln`: 6-param AdaLN modulation
    /// - `norm_attn`: Normalization for attention input (LayerNorm no-affine)
    /// - `attn`: Self-attention module
    /// - `norm_ffn`: Normalization for FFN input (LayerNorm no-affine)
    /// - `ffn`: Feed-forward network (typically SwiGlu)
    pub fn new(
        adaln: AdaLnZeroDual,
        norm_attn: Box<dyn Module + Send + Sync>,
        attn: Box<dyn Module + Send + Sync>,
        norm_ffn: Box<dyn Module + Send + Sync>,
        ffn: Box<dyn Module + Send + Sync>,
    ) -> Result<Self> {
        Ok(Self {
            adaln,
            norm_attn,
            attn,
            norm_ffn,
            ffn,
        })
    }

    /// Forward: project cond → modulate attn → gated residual → modulate ffn → gated residual.
    ///
    /// - `x`: input tensor `[B, S, dim]`
    /// - `t_emb`: timestep embedding `[B, dim]`
    pub fn forward(&self, x: &DynTensor, t_emb: &DynTensor) -> Result<DynTensor> {
        let params = self.adaln.forward(t_emb)?;

        // Unsqueeze params from [B, dim] to [B, 1, dim] for sequence broadcast.
        // AdaLnZeroDual produces [B, dim]; x is [B, S, dim].
        let scale1 = params.scale1.unsqueeze(1)?;
        let shift1 = params.shift1.unsqueeze(1)?;
        let gate1 = params.gate1.unsqueeze(1)?;
        let scale2 = params.scale2.unsqueeze(1)?;
        let shift2 = params.shift2.unsqueeze(1)?;
        let gate2 = params.gate2.unsqueeze(1)?;

        // Attention sub-block: norm → modulate → attn → gated residual
        let normed = self.norm_attn.forward(x)?;
        let modulated = apply_adaln_modulation(&normed, &scale1, &shift1)?;
        let attn_out = self.attn.forward(&modulated)?;
        let x = (x + &gate1.broadcast_mul(&attn_out)?)?;

        // FFN sub-block: norm → modulate → ffn → gated residual
        let normed = self.norm_ffn.forward(&x)?;
        let modulated = apply_adaln_modulation(&normed, &scale2, &shift2)?;
        let ffn_out = self.ffn.forward(&modulated)?;
        let output = (&x + &gate2.broadcast_mul(&ffn_out)?)?;
        check_output_finite(&output, "DiTBlockDual")?;
        Ok(output)
    }
}

#[cfg(test)]
#[path = "dit_block_tests.rs"]
mod tests;
