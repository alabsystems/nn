// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Residual Vector Quantization (RVQ) for neural audio codecs.
//!
//! Provides [`VqCodebook`] (single-level VQ) and [`Rvq`] (multi-level residual VQ)
//! matching the patterns used by Qwen3-TTS SpeechTokenizer and MOSS CAT Tokenizer.
//!
//! All operations decompose to existing [`DynTensor`] primitives — no new kernel ops.

use crate::dyn_tensor::DynTensor;
use crate::layers::Module;
use crate::{DType, Device, Embedding, Result, TensorError, VarBuilder};

// -- VqCodebook ---------------------------------------------------------------

/// Single-level Vector Quantization codebook.
///
/// Wraps an [`Embedding`] table of shape `[codebook_size, dim]`.
/// - **Decode:** `indices → embedding lookup → continuous vectors`
/// - **Quantize:** `features → L2 nearest-neighbor → (quantized, indices)`
#[derive(Debug, Clone)]
pub struct VqCodebook {
    embedding: Embedding,
}

impl VqCodebook {
    /// Create a codebook from an embedding weight matrix.
    ///
    /// - `weight`: shape `[codebook_size, dim]`
    pub fn new(weight: DynTensor) -> Result<Self> {
        let (_, _) = weight.dims2()?;
        Ok(Self {
            embedding: Embedding::new(weight)?,
        })
    }

    /// Load codebook weights from a [`VarBuilder`].
    pub fn load(vb: impl AsRef<VarBuilder>, codebook_size: usize, dim: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let weight = vb.get(&[codebook_size, dim], "weight")?;
        Self::new(weight)
    }

    /// Load with usage-normalized embeddings (Qwen3-TTS pattern).
    ///
    /// Loads `embedding_sum` `[codebook_size, dim]` and `cluster_usage` `[codebook_size]`,
    /// then computes `weight = sum / max(usage, 1e-5)` per row.
    pub fn load_normalized(
        vb: impl AsRef<VarBuilder>,
        codebook_size: usize,
        dim: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let sum = vb.get(&[codebook_size, dim], "embedding_sum")?;
        let usage = vb.get(&[codebook_size], "cluster_usage")?;
        // Clamp usage to avoid division by zero, then broadcast-divide
        let usage_clamped = usage.clamp_min(1e-5)?.unsqueeze(1)?; // [codebook_size, 1]
        let weight = sum.broadcast_div(&usage_clamped)?;
        Self::new(weight)
    }

    /// Create a zero-initialized codebook (for testing).
    pub fn zeros(codebook_size: usize, dim: usize, device: &Device) -> Result<Self> {
        let weight = DynTensor::zeros(&[codebook_size, dim], DType::F32, device)?;
        Self::new(weight)
    }

    /// Codebook size (number of entries).
    #[must_use]
    pub fn codebook_size(&self) -> usize {
        self.embedding.weight().dims()[0]
    }

    /// Embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.embedding.weight().dims()[1]
    }

    /// Embedding weight reference.
    #[must_use]
    pub fn weight(&self) -> &DynTensor {
        self.embedding.weight()
    }

    /// Decode: token indices → continuous vectors via embedding lookup.
    ///
    /// Input: `[seq]` or `[batch, seq]` indices (U32 preferred; F32 legacy accepted).
    /// Output: `[seq, dim]` or `[batch, seq, dim]`.
    pub fn decode(&self, indices: &DynTensor) -> Result<DynTensor> {
        self.embedding.forward(indices)
    }

    /// Quantize: continuous vectors → nearest codebook entry (L2 distance).
    ///
    /// Computes `||x - e||² = ||x||² - 2·x·eᵀ + ||e||²` then takes argmin.
    ///
    /// Input: `[seq, dim]` or `[batch, seq, dim]`.
    /// Returns: `(quantized, indices)` where indices has the last dim removed.
    pub fn quantize(&self, x: &DynTensor) -> Result<(DynTensor, DynTensor)> {
        let x_ndim = x.rank();
        if x_ndim < 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }
        let input_dim = x.dims()[x_ndim - 1];
        if input_dim != self.dim() {
            return Err(TensorError::shape_mismatch(
                vec![self.dim()],
                vec![input_dim],
            ));
        }

        let embeddings = self.embedding.weight(); // [codebook_size, dim]

        // ||x||² along last axis → [..., 1]
        let x_sq = x.sqr()?.sum_keepdim(x_ndim - 1)?;

        // ||e||² along last axis → [codebook_size]
        let e_sq = embeddings.sqr()?.sum(1)?;

        // x·eᵀ → [..., codebook_size]
        let embeddings_t = embeddings.transpose(0, 1)?; // [dim, codebook_size]
        let cross = x.matmul(&embeddings_t)?;

        // dist = ||x||² - 2·x·eᵀ + ||e||²
        // Use broadcast ops since shapes differ: x_sq [..., 1], cross [..., codebook_size]
        let dist = x_sq
            .broadcast_sub(&cross.mul_scalar(2.0)?)?
            .broadcast_add(&e_sq.unsqueeze(0)?)?;

        // argmin → [...] indices
        let indices_dim = dist.rank() - 1;
        let indices = dist.argmin(indices_dim)?;

        // Look up the quantized vectors
        let quantized = self.decode(&indices)?;

        Ok((quantized, indices))
    }
}

// -- Rvq ----------------------------------------------------------------------

/// Multi-level Residual Vector Quantization.
///
/// Each level quantizes the residual from the previous level.
/// Used by neural audio codecs (EnCodec, SpeechTokenizer, MOSS CAT).
#[derive(Debug, Clone)]
pub struct Rvq {
    codebooks: Vec<VqCodebook>,
}

impl Rvq {
    /// Create an RVQ from a list of codebooks.
    ///
    /// All codebooks must have the same embedding dimension.
    pub fn new(codebooks: Vec<VqCodebook>) -> Result<Self> {
        if codebooks.is_empty() {
            return Err(TensorError::ValueOutOfRange {
                description: "RVQ requires at least one codebook",
            });
        }
        let dim = codebooks[0].dim();
        for (_i, cb) in codebooks.iter().enumerate().skip(1) {
            if cb.dim() != dim {
                return Err(TensorError::shape_mismatch(vec![dim], vec![cb.dim()]));
            }
        }
        Ok(Self { codebooks })
    }

    /// Load RVQ codebook weights from a [`VarBuilder`].
    ///
    /// Loads `{prefix}.{i}.weight` for i in `0..n_levels`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        n_levels: usize,
        codebook_size: usize,
        dim: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let mut codebooks = Vec::with_capacity(n_levels);
        for i in 0..n_levels {
            let cb = VqCodebook::load(vb.pp(i), codebook_size, dim)?;
            codebooks.push(cb);
        }
        Self::new(codebooks)
    }

    /// Number of quantization levels (codebooks).
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.codebooks.len()
    }

    /// Embedding dimension (same across all codebooks).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.codebooks[0].dim()
    }

    /// Reference to codebooks.
    #[must_use]
    pub fn codebooks(&self) -> &[VqCodebook] {
        &self.codebooks
    }

    /// Encode: continuous features → multi-level codebook indices.
    ///
    /// Input: `[seq, dim]` features.
    /// Output: `[n_levels, seq]` indices tensor.
    ///
    /// `n_levels` controls how many quantization levels to use (capped at
    /// the number of codebooks).
    pub fn encode(&self, features: &DynTensor, n_levels: usize) -> Result<DynTensor> {
        let levels = n_levels.min(self.codebooks.len());
        if levels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RVQ encode requires at least 1 level",
            });
        }

        let mut residual = features.clone();
        let mut all_indices = Vec::with_capacity(levels);

        for cb in self.codebooks.iter().take(levels) {
            let (quantized, indices) = cb.quantize(&residual)?;
            // argmin returns U32; keep as U32 (Embedding::forward handles U32 natively).
            all_indices.push(indices);
            residual = residual.sub(&quantized)?;
        }

        // Stack into [n_levels, seq]
        let refs: Vec<&DynTensor> = all_indices.iter().collect();
        DynTensor::stack(&refs, 0)
    }

    /// Decode: multi-level codebook indices → continuous features.
    ///
    /// Input: `[n_levels, seq]` indices tensor (U32 preferred; F32 legacy accepted).
    /// Output: `[seq, dim]` summed embeddings.
    pub fn decode(&self, codes: &DynTensor) -> Result<DynTensor> {
        let code_ndim = codes.rank();
        if code_ndim < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: code_ndim,
            });
        }
        let n_levels = codes.dims()[0];
        if n_levels > self.codebooks.len() {
            return Err(TensorError::ValueOutOfRange {
                description: "RVQ decode: codes has more levels than codebooks",
            });
        }

        let mut sum: Option<DynTensor> = None;
        for i in 0..n_levels {
            // Select level i: narrow + squeeze to remove leading dim
            let level_codes = codes.narrow(0, i, 1)?.squeeze(0)?;
            let embedded = self.codebooks[i].decode(&level_codes)?;
            sum = Some(match sum {
                Some(acc) => acc.add(&embedded)?,
                None => embedded,
            });
        }

        sum.ok_or(TensorError::ValueOutOfRange {
            description: "RVQ decode: no codebook levels to decode",
        })
    }
}
