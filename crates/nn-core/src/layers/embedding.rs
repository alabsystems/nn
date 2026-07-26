// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`Embedding`] lookup table — extracted from `layers.rs` for file-size compliance (#1377).

use super::Module;
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Result, TensorError};

/// Embedding lookup table.
///
/// Matches candle-nn `Embedding`. Maps integer indices to dense vectors.
#[derive(Debug, Clone)]
pub struct Embedding {
    weight: DynTensor,
}

impl Embedding {
    /// Create an Embedding from a weight matrix.
    ///
    /// - `weight`: shape `[vocab_size, embedding_dim]` (must be 2D)
    ///
    /// Returns an error if `weight` is not 2D.
    pub fn new(weight: DynTensor) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: weight.rank(),
            });
        }
        Ok(Self { weight })
    }

    /// Weight matrix reference.
    #[must_use]
    pub fn weight(&self) -> &DynTensor {
        &self.weight
    }

    /// Embedding weight matrix reference (candle compat).
    ///
    /// Alias for [`weight()`](Self::weight) — matches candle-nn's
    /// `Embedding::embeddings()` API used by dvoice models.
    #[must_use]
    pub fn embeddings(&self) -> &DynTensor {
        &self.weight
    }

    /// Look up embeddings for the given indices.
    ///
    /// `ids` should contain integer indices. Each index selects a row from the
    /// weight matrix. When the weight is on GPU, dispatches via `index_select`
    /// (Embedding kernel) without transferring the weight table to CPU.
    pub fn forward_ids(&self, ids: &[usize]) -> Result<DynTensor> {
        let (vocab_size, _embed_dim) = self.weight.dims2()?;
        for &id in ids {
            if id >= vocab_size {
                return Err(TensorError::EmbeddingIndexOutOfRange {
                    index: id,
                    vocab_size,
                });
            }
        }
        // Build a U32 index tensor and delegate to index_select(dim=0).
        // Created on CPU — index_select handles GPU transfer internally.
        let ids_u32: Vec<u32> = ids
            .iter()
            .map(|&i| {
                u32::try_from(i).map_err(|_| TensorError::EmbeddingIndexOutOfRange {
                    index: i,
                    vocab_size: self.weight.dims()[0],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let ids_tensor = DynTensor::from_vec_u32(ids_u32, &[ids.len()], &crate::Device::Cpu)?;
        self.weight.index_select(&ids_tensor, 0)
    }
}

impl Module for Embedding {
    /// Forward: look up embeddings by integer indices.
    ///
    /// Accepts U32 tensors (from `argmax`, `topk`, etc.), I64 tensors
    /// (candle's default for token IDs), and F32 tensors with integer-valued
    /// elements (legacy path). Output shape is `[..input_shape, embedding_dim]`.
    ///
    /// For multi-dimensional inputs (e.g., `[B, S]`), the output preserves all
    /// input dimensions and appends `embedding_dim` (e.g., `[B, S, D]`),
    /// matching PyTorch's `nn.Embedding` semantics.
    ///
    /// When both weight and input are on GPU with integer dtype (U32 or I64),
    /// uses native GPU `index_select` — no GPU→CPU round-trip.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let input_dims = x.dims().to_vec();
        let embed_dim = self.weight.dims().last().copied().unwrap_or(0);

        super::traced_forward(
            &[x],
            || {
                Ok(TraceOp::Embedding {
                    weight: self.weight.to_weight_ref()?,
                })
            },
            || {
                // Fast path: integer index tensors use index_select directly,
                // which dispatches natively on GPU without extracting IDs to CPU.
                if x.dtype() == DType::U32 || x.dtype() == DType::I64 {
                    let flat_ids = x.reshape([x.elem_count()])?;
                    let flat_result = self.weight.index_select(&flat_ids, 0)?;
                    let mut out_shape = input_dims.clone();
                    out_shape.push(embed_dim);
                    flat_result.reshape(&out_shape)
                } else {
                    let ids = self.extract_ids(x)?;
                    let flat_result = self.forward_ids(&ids)?;
                    let mut out_shape = input_dims.clone();
                    out_shape.push(embed_dim);
                    flat_result.reshape(&out_shape)
                }
            },
        )
    }
}

impl Embedding {
    /// Extract integer indices from an input tensor of any dtype.
    ///
    /// Returns a flat `Vec<usize>` of indices, validating each element.
    fn extract_ids(&self, x: &DynTensor) -> Result<Vec<usize>> {
        if x.dtype() == DType::U32 {
            let owned_cpu;
            let cpu_ref = if x.device().is_gpu() {
                owned_cpu = x.to_device(&crate::Device::Cpu)?;
                &owned_cpu
            } else {
                x
            };
            let u32_data = cpu_ref.as_cpu_u32()?;
            return Ok(u32_data.iter().map(|&v| v as usize).collect());
        }
        if x.dtype() == DType::I64 {
            let owned_cpu;
            let cpu_ref = if x.device().is_gpu() {
                owned_cpu = x.to_device(&crate::Device::Cpu)?;
                &owned_cpu
            } else {
                x
            };
            let i64_data = cpu_ref.as_cpu_i64()?;
            let mut ids = Vec::with_capacity(i64_data.len());
            for &v in i64_data.iter() {
                if v < 0 {
                    return Err(TensorError::ValueOutOfRange {
                        description: "embedding index must be non-negative",
                    });
                }
                let idx = usize::try_from(v).map_err(|_| TensorError::ValueOutOfRange {
                    description: "embedding index exceeds usize::MAX",
                })?;
                ids.push(idx);
            }
            return Ok(ids);
        }
        // Legacy F32 path: interpret float values as indices
        let (vocab_size, _) = self.weight.dims2()?;
        let cpu_x = x.to_device(&crate::Device::Cpu)?;
        let arr = cpu_x.to_f32_array()?;
        let flat: Vec<f32> = arr.iter().copied().collect();
        let mut ids = Vec::with_capacity(flat.len());
        for &v in &flat {
            if !v.is_finite() || v < 0.0 || v != v.trunc() {
                return Err(TensorError::ValueOutOfRange {
                    description: "embedding index must be a non-negative finite integer",
                });
            }
            let idx = v as usize;
            if idx >= vocab_size {
                return Err(TensorError::EmbeddingIndexOutOfRange {
                    index: idx,
                    vocab_size,
                });
            }
            ids.push(idx);
        }
        Ok(ids)
    }
}
