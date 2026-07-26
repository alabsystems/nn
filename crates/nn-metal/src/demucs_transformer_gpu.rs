// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU weight management and builder helpers for [`DemucsTransformer`].
//!
//! Extracted from `demucs_transformer.rs` to keep the parent module under
//! the 500-line limit. Part of #1410 Direction 4.

use std::collections::HashMap;

use nn_dsl::tensor_ir::TensorKernelDef;

use crate::buffer::MetalBuffer;
use crate::element::MetalElement;
use crate::PipelineCache;

use super::{DemucsTransformer, DemucsTransformerError};

// ---------------------------------------------------------------------------
// Persistent GPU weight cache
// ---------------------------------------------------------------------------

/// Holds all weight tensors as persistent Metal GPU buffers.
/// Lazily initialized on first `forward_gpu()` call via `OnceLock`.
pub(super) struct TransformerGpuWeights {
    pub(super) upsample_t: HashMap<String, MetalBuffer>,
    pub(super) upsample_s: HashMap<String, MetalBuffer>,
    pub(super) norm_in_t: HashMap<String, MetalBuffer>,
    pub(super) norm_in_s: HashMap<String, MetalBuffer>,
    pub(super) temporal_layers: Vec<HashMap<String, MetalBuffer>>,
    pub(super) spectral_layers: Vec<HashMap<String, MetalBuffer>>,
    pub(super) downsample_t: HashMap<String, MetalBuffer>,
    pub(super) downsample_s: HashMap<String, MetalBuffer>,
    pub(super) sinusoidal_t: HashMap<String, MetalBuffer>,
    pub(super) sinusoidal_s: HashMap<String, MetalBuffer>,
}

// ---------------------------------------------------------------------------
// GPU weight upload + builder helpers
// ---------------------------------------------------------------------------

impl DemucsTransformer {
    /// Lazily upload all CPU weight data to persistent GPU buffers.
    ///
    /// Called on the first `forward_gpu()` invocation. Subsequent calls return
    /// the cached buffers (via `OnceLock`). Each weight `Vec<f32>` is uploaded
    /// once via `f32::create_buffer`, and the resulting `MetalBuffer` is reused
    /// for all future dispatches as `DispatchInput::Gpu` (zero-copy alias).
    pub(super) fn ensure_gpu_weights(
        &self,
        cache: &PipelineCache,
    ) -> Result<crate::GpuWeightRef<'_, TransformerGpuWeights>, DemucsTransformerError> {
        self.gpu_weights.get_or_init_with(
            || {
                let ctx = cache.context();
                let upload = |wmap: &HashMap<String, Vec<f32>>| -> Result<HashMap<String, MetalBuffer>, String> {
                    wmap.iter()
                        .map(|(name, data)| {
                            let buf = f32::create_buffer(ctx, data)
                                .map_err(|e| format!("{name}: {e}"))?;
                            Ok((name.clone(), buf))
                        })
                        .collect()
                };
                Ok(TransformerGpuWeights {
                    upsample_t: upload(&self.upsample_t_weights)?,
                    upsample_s: upload(&self.upsample_s_weights)?,
                    norm_in_t: upload(&self.norm_in_t_weights)?,
                    norm_in_s: upload(&self.norm_in_s_weights)?,
                    temporal_layers: self
                        .temporal_layer_weights
                        .iter()
                        .map(upload)
                        .collect::<Result<Vec<_>, _>>()?,
                    spectral_layers: self
                        .spectral_layer_weights
                        .iter()
                        .map(upload)
                        .collect::<Result<Vec<_>, _>>()?,
                    downsample_t: upload(&self.downsample_t_weights)?,
                    downsample_s: upload(&self.downsample_s_weights)?,
                    sinusoidal_t: upload(&self.sinusoidal_t_weights)?,
                    sinusoidal_s: upload(&self.sinusoidal_s_weights)?,
                })
            },
            DemucsTransformerError::GpuBufferAlloc,
        )
    }

    /// Build a `TensorKernelDef` for a 2D transpose [rows, cols] → [cols, rows].
    pub(super) fn build_transpose_def(
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<TensorKernelDef, DemucsTransformerError> {
        use nn_dsl::tensor_block_builder::TensorBlockBuilder;
        let mut b = TensorBlockBuilder::new(name);
        let data = b.add_input(nn_dsl::input_names::DATA, &[rows, cols]);
        let out = b.add_transpose(data, &[1, 0], &[cols, rows]);
        // TensorBlockBuilder::build returns TensorIRError; From<TensorIRError>
        // is implemented for DemucsTransformerError via TransformerBuildError.
        Ok(b.build(out)?)
    }

    /// Build a `TensorKernelDef` for adding a precomputed sinusoidal table.
    ///
    /// The table is a weight tensor "sinusoidal" of shape `[seq_len, dim]`.
    /// The op is element-wise: `output = data + sinusoidal`.
    pub(super) fn build_sinusoidal_add_def(
        name: &str,
        seq_len: usize,
        dim: usize,
    ) -> Result<TensorKernelDef, DemucsTransformerError> {
        use nn_dsl::tensor_block_builder::TensorBlockBuilder;
        let mut b = TensorBlockBuilder::new(name);
        let data = b.add_input(nn_dsl::input_names::DATA, &[seq_len, dim]);
        let table = b.add_input("sinusoidal", &[seq_len, dim]);
        let out = b.add_binary_add(data, table, &[seq_len, dim]);
        Ok(b.build(out)?)
    }
}
