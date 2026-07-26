// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Adaptive Layer Normalization with zero-initialized modulation (AdaLN-Zero).
//!
//! Three variants matching dvoice DiT model architectures:
//!
//! - [`AdaLnZero`]: 3-param (scale, shift, gate) — Ming-omni, Irodori base
//! - [`AdaLnZeroDual`]: 6-param (attn + FFN modulation) — CosyVoice3
//! - [`LowRankAdaLn`]: Rank-bottleneck 3-param — Irodori low-rank variant
//!
//! All decompose to existing nn primitives: Linear, LayerNorm/RmsNorm,
//! narrow, broadcast arithmetic. No new TensorOpKind needed.
//!
//! ## AdaLN-Zero pattern
//!
//! ```text
//! 1. Project conditioning: params = Linear(cond)
//! 2. Split: (scale, shift, gate) = narrow(params, 3 chunks)
//! 3. Normalize: normed = Norm(x)
//! 4. Modulate: out = normed * (1 + scale) + shift
//! 5. Gated residual: x + gate * sub_block(out)
//! ```
//!
//! The "+1" offset means zero-initialized scale preserves identity at init.

use super::{check_output_finite, Linear, Module};
use crate::dyn_tensor::DynTensor;
use crate::var_builder::VarBuilder;
use crate::Result;
use crate::TensorError;

// -- AdaLnParams (6-param return type for CosyVoice3) -------------------------

/// Six modulation parameters for dual-stream DiT blocks (CosyVoice3).
///
/// Split order matches PyTorch convention: (scale, shift, gate) per stream.
/// Attention stream uses (scale1, shift1, gate1).
/// FFN stream uses (scale2, shift2, gate2).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdaLnParams {
    pub scale1: DynTensor,
    pub shift1: DynTensor,
    pub gate1: DynTensor,
    pub scale2: DynTensor,
    pub shift2: DynTensor,
    pub gate2: DynTensor,
}

impl AdaLnParams {
    /// Create adaptive layer-norm parameters for attention and FFN streams.
    pub fn new(
        scale1: DynTensor,
        shift1: DynTensor,
        gate1: DynTensor,
        scale2: DynTensor,
        shift2: DynTensor,
        gate2: DynTensor,
    ) -> Self {
        Self {
            scale1,
            shift1,
            gate1,
            scale2,
            shift2,
            gate2,
        }
    }
}

// -- Helper: apply modulation -------------------------------------------------

/// Apply adaptive normalization modulation: `normed * (1 + scale) + shift`.
///
/// The `(1 + scale)` offset means zero-initialized scale produces identity.
pub fn apply_adaln_modulation(
    normed: &DynTensor,
    scale: &DynTensor,
    shift: &DynTensor,
) -> Result<DynTensor> {
    let scale_plus_one = (scale + 1.0)?;
    normed.broadcast_mul(&scale_plus_one)?.broadcast_add(shift)
}

// -- AdaLnZero (3-param) -----------------------------------------------------

/// Adaptive Layer Normalization with zero-initialized modulation (3-param).
///
/// Produces (scale, shift, gate) from a conditioning signal, then applies
/// `normed * (1 + scale) + shift`. The gate output is returned separately
/// for the caller to apply as gated residual: `x + gate * sub_block(modulated)`.
///
/// Used by Ming-omni (RMSNorm) and as base for Irodori (low-rank variant).
pub struct AdaLnZero {
    proj: Linear,
    norm: Box<dyn Module + Send + Sync>,
    dim: usize,
}

impl std::fmt::Debug for AdaLnZero {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaLnZero")
            .field("proj", &self.proj)
            .field("norm", &format_args!("<dyn Module>"))
            .field("dim", &self.dim)
            .finish()
    }
}

impl AdaLnZero {
    /// Create from pre-built components.
    ///
    /// - `proj`: Linear projection from `cond_dim` to `3 * dim`
    /// - `norm`: Normalization layer (RmsNorm, LayerNorm, etc.)
    /// - `dim`: Hidden dimension (projection output is split into 3 chunks of this size)
    ///
    /// Returns `Err(TensorError::InvalidShape)` if `dim == 0`.
    pub fn new(proj: Linear, norm: Box<dyn Module + Send + Sync>, dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::InvalidShape(
                "AdaLnZero: dim must be > 0".into(),
            ));
        }
        Ok(Self { proj, norm, dim })
    }

    /// Load the projection Linear from a [`VarBuilder`], using a caller-provided norm.
    ///
    /// The norm layer (RmsNorm, LayerNorm, etc.) cannot be auto-loaded because
    /// its concrete type is not known at compile time. The caller constructs the
    /// norm separately and passes it in.
    ///
    /// Loads `proj.weight` and optional `proj.bias` from the VarBuilder.
    ///
    /// - `cond_dim`: Conditioning signal dimension.
    /// - `dim`: Hidden dimension (projection output is `3 * dim`).
    /// - `norm`: Pre-built normalization layer.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        cond_dim: usize,
        dim: usize,
        norm: Box<dyn Module + Send + Sync>,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let proj = Linear::load(vb.pp("proj"), cond_dim, 3 * dim)?;
        Self::new(proj, norm, dim)
    }

    /// Hidden dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Forward: returns (modulated_x, gate).
    ///
    /// - `x`: input tensor `[B, S, dim]` or `[B, dim]`
    /// - `cond`: conditioning signal `[B, cond_dim]` or `[B, S, cond_dim]`
    ///
    /// Returns `(modulated, gate)` where:
    /// - `modulated = norm(x) * (1 + scale) + shift`
    /// - `gate` is for gated residual: `x + gate * sub_block(modulated)`
    pub fn forward(&self, x: &DynTensor, cond: &DynTensor) -> Result<(DynTensor, DynTensor)> {
        let params = self.proj.forward(cond)?;
        let last_dim = params
            .rank()
            .checked_sub(1)
            .ok_or(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            })?;

        let scale = params.narrow(last_dim, 0, self.dim)?;
        let shift = params.narrow(last_dim, self.dim, self.dim)?;
        let gate = params.narrow(last_dim, 2 * self.dim, self.dim)?;

        let normed = self.norm.forward(x)?;
        let modulated = apply_adaln_modulation(&normed, &scale, &shift)?;
        check_output_finite(&modulated, "AdaLnZero")?;
        Ok((modulated, gate))
    }
}

// -- AdaLnZeroDual (6-param, CosyVoice3) -------------------------------------

/// CosyVoice3 variant: produces 6 modulation parameters from one projection.
///
/// A single `Linear(dim, 6 * dim)` projects the timestep embedding to produce
/// (scale1, shift1, gate1, scale2, shift2, gate2) — modulation for both the
/// attention and FFN sub-blocks in a DiT block.
///
/// Note: CosyVoice3 applies SiLU to the conditioning before projection.
/// Conditioning is `[B, dim]` (not `[B, S, dim]`) — caller must unsqueeze
/// for broadcast over sequence dim.
#[derive(Debug, Clone)]
pub struct AdaLnZeroDual {
    modulation: Linear,
    dim: usize,
}

impl AdaLnZeroDual {
    /// Create from pre-built Linear projection.
    ///
    /// - `modulation`: Linear from `dim` to `6 * dim`
    /// - `dim`: Hidden dimension
    ///
    /// Returns `Err(TensorError::InvalidShape)` if `dim == 0`.
    pub fn new(modulation: Linear, dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::InvalidShape(
                "AdaLnZeroDual: dim must be > 0".into(),
            ));
        }
        Ok(Self { modulation, dim })
    }

    /// Load from a [`VarBuilder`] using PyTorch-style weight names.
    ///
    /// Loads the `modulation` Linear projecting `[dim]` to `[6 * dim]`.
    /// Weight name: `modulation.weight` plus optional `modulation.bias`.
    pub fn load(vb: impl AsRef<VarBuilder>, dim: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let modulation = Linear::load(vb.pp("modulation"), dim, 6 * dim)?;
        Self::new(modulation, dim)
    }

    /// Hidden dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Forward: apply SiLU to conditioning, project, split into 6 params.
    ///
    /// - `t_emb`: timestep embedding `[B, dim]`
    ///
    /// Returns [`AdaLnParams`] with scale/shift/gate for both sub-blocks.
    pub fn forward(&self, t_emb: &DynTensor) -> Result<AdaLnParams> {
        let silu = t_emb.silu()?;
        let params = self.modulation.forward(&silu)?;
        check_output_finite(&params, "AdaLnZeroDual")?;
        let last_dim = params
            .rank()
            .checked_sub(1)
            .ok_or(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            })?;
        let d = self.dim;
        Ok(AdaLnParams {
            scale1: params.narrow(last_dim, 0, d)?,
            shift1: params.narrow(last_dim, d, d)?,
            gate1: params.narrow(last_dim, 2 * d, d)?,
            scale2: params.narrow(last_dim, 3 * d, d)?,
            shift2: params.narrow(last_dim, 4 * d, d)?,
            gate2: params.narrow(last_dim, 5 * d, d)?,
        })
    }
}

// -- LowRankAdaLn (Irodori variant) -------------------------------------------

/// Low-rank variant: `down -> SiLU -> up` reduces parameter count.
///
/// Instead of `Linear(cond_dim, 3 * dim)`, uses a bottleneck:
/// `Linear(cond_dim, rank) -> SiLU -> Linear(rank, 3 * dim)`.
/// With rank = dim/4, this reduces parameters by ~43%.
///
/// Used by Irodori-TTS DiT blocks.
pub struct LowRankAdaLn {
    down: Linear,
    up: Linear,
    norm: Box<dyn Module + Send + Sync>,
    dim: usize,
}

impl std::fmt::Debug for LowRankAdaLn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LowRankAdaLn")
            .field("down", &self.down)
            .field("up", &self.up)
            .field("norm", &format_args!("<dyn Module>"))
            .field("dim", &self.dim)
            .finish()
    }
}

impl LowRankAdaLn {
    /// Create from pre-built components.
    ///
    /// - `down`: Linear from `cond_dim` to `rank`
    /// - `up`: Linear from `rank` to `3 * dim`
    /// - `norm`: Normalization layer (typically RmsNorm)
    /// - `dim`: Hidden dimension
    ///
    /// Returns `Err(TensorError::InvalidShape)` if `dim == 0`.
    pub fn new(
        down: Linear,
        up: Linear,
        norm: Box<dyn Module + Send + Sync>,
        dim: usize,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::InvalidShape(
                "LowRankAdaLn: dim must be > 0".into(),
            ));
        }
        Ok(Self {
            down,
            up,
            norm,
            dim,
        })
    }

    /// Load the bottleneck Linear layers from a [`VarBuilder`], using a caller-provided norm.
    ///
    /// The norm layer cannot be auto-loaded because its concrete type is not known
    /// at compile time. The caller constructs the norm separately.
    ///
    /// Loads `down.weight`, `up.weight`, and optional bias tensors.
    ///
    /// - `cond_dim`: Conditioning signal dimension.
    /// - `rank`: Bottleneck dimension.
    /// - `dim`: Hidden dimension (up-projection output is `3 * dim`).
    /// - `norm`: Pre-built normalization layer.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        cond_dim: usize,
        rank: usize,
        dim: usize,
        norm: Box<dyn Module + Send + Sync>,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let down = Linear::load(vb.pp("down"), cond_dim, rank)?;
        let up = Linear::load(vb.pp("up"), rank, 3 * dim)?;
        Self::new(down, up, norm, dim)
    }

    /// Hidden dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Forward: bottleneck projection -> split -> modulate.
    ///
    /// Same return type as [`AdaLnZero::forward`]: `(modulated, gate)`.
    pub fn forward(&self, x: &DynTensor, cond: &DynTensor) -> Result<(DynTensor, DynTensor)> {
        let h = self.down.forward(cond)?;
        let h = h.silu()?;
        let params = self.up.forward(&h)?;
        let last_dim = params
            .rank()
            .checked_sub(1)
            .ok_or(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            })?;

        let scale = params.narrow(last_dim, 0, self.dim)?;
        let shift = params.narrow(last_dim, self.dim, self.dim)?;
        let gate = params.narrow(last_dim, 2 * self.dim, self.dim)?;

        let normed = self.norm.forward(x)?;
        let modulated = apply_adaln_modulation(&normed, &scale, &shift)?;
        check_output_finite(&modulated, "LowRankAdaLn")?;
        Ok((modulated, gate))
    }
}

#[cfg(test)]
#[path = "adaln_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kani_adaln_proofs.rs"]
mod kani_adaln_proofs;
