// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight loading for HTDemucs from safetensors via `WeightMap` or direct file.
//!
//! Maps flat safetensors tensor names (produced by
//! `convert_demucs_safetensors.py`) to the structured `HTDemucsWeights`
//! hierarchy. The naming convention:
//!
//! - Encoder:    `enc{b}_conv_weight`, `enc{b}_dc{d}_compress_weight`, ...
//! - Decoder:    `dec{b}_rewrite_weight`, `dec{b}_dc{d}_compress_weight`, ...
//! - Transformer: `xformer_self{i}_q_weight`, `xformer_cross{i}_norm1_weight`, ...
//!
//! Part of #779 — Milestone 1 weight converter (Rust side).

use std::path::Path;

use crate::demucs_shared::{channels_at_depth, DCONV_DEPTH};
use crate::demucs_temporal_decoder::{
    DConvSubLayerWeights, DecoderBlockWeights, DemucsTemporalDecoderWeights,
};
use crate::demucs_temporal_encoder::{DemucsTemporalEncoderWeights, EncoderBlockWeights};
use crate::demucs_transformer::LayerNormWeights;
use crate::safetensors::WeightMap;
use crate::HTDemucsWeights;

// Architecture constants (must match Python converter).
const DEPTH: usize = 4;
const TRANSFORMER_DIM: usize = 512;
const FFN_DIM: usize = 2048;
const BOTTLENECK_DIM: usize = 384;
const NUM_LAYERS: usize = 5;

/// Errors from weight loading.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WeightLoadError {
    /// Tensor not found in safetensors file.
    #[error("tensor '{name}' not found in weight map")]
    MissingTensor { name: String },
    /// Byte length not aligned to f32.
    #[error("tensor '{name}' byte length {len} not aligned to f32")]
    ByteAlignment { name: String, len: usize },
    /// WeightMap read error.
    #[error("weight map error: {0}")]
    WeightMap(#[from] crate::safetensors::WeightError),
    /// IO error reading safetensors file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Safetensors format/parse error.
    #[error("safetensors format: {0}")]
    SafetensorsFormat(String),
    /// Weight tensor contains NaN or Inf values.
    #[error("tensor '{name}' has {count} non-finite value(s)")]
    NonFiniteWeight { name: String, count: usize },
    /// Tensor byte range exceeds buffer length.
    #[error("tensor '{name}' byte range {offset}..{end} exceeds buffer length {buf_len}")]
    OutOfBounds {
        name: String,
        offset: usize,
        end: usize,
        buf_len: usize,
    },
}

// ---------------------------------------------------------------------------
// TensorSource abstraction
// ---------------------------------------------------------------------------

/// Trait for types that can supply raw tensor byte data by name.
///
/// Implemented for [`WeightMap`] (mmap-backed) and [`ParsedSafetensors`]
/// (heap-backed) so that all `load_*` functions work with either source.
pub(crate) trait TensorSource {
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], WeightLoadError>;
}

impl TensorSource for WeightMap {
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], WeightLoadError> {
        self.tensor_data(name)
            .map_err(|_| WeightLoadError::MissingTensor {
                name: name.to_string(),
            })
    }
}

/// Pre-parsed safetensors data for safe (non-mmap) loading.
///
/// Holds the owned `Vec<u8>` alongside offset/length metadata for each tensor.
/// Avoids re-parsing the safetensors header on every `tensor_bytes` call.
pub(crate) struct ParsedSafetensors {
    data: Vec<u8>,
    /// (offset, length) into `data` for each tensor name.
    tensors: std::collections::HashMap<String, (usize, usize)>,
}

impl ParsedSafetensors {
    /// Read and parse a safetensors file into memory.
    fn from_file(path: impl AsRef<Path>) -> Result<Self, WeightLoadError> {
        let data = std::fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| WeightLoadError::SafetensorsFormat(e.to_string()))?;
        let mut tensors = std::collections::HashMap::new();
        for (name, view) in st.tensors() {
            let bytes = view.data();
            // Compute the offset of this tensor's data within the owned `data` buffer.
            // view.data() returns a subslice of the deserialized `data`.
            let offset = bytes.as_ptr() as usize - data.as_ptr() as usize;
            tensors.insert(name, (offset, bytes.len()));
        }
        Ok(Self { data, tensors })
    }
}

impl TensorSource for ParsedSafetensors {
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], WeightLoadError> {
        let &(offset, len) =
            self.tensors
                .get(name)
                .ok_or_else(|| WeightLoadError::MissingTensor {
                    name: name.to_string(),
                })?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| WeightLoadError::OutOfBounds {
                name: name.to_string(),
                offset,
                end: usize::MAX,
                buf_len: self.data.len(),
            })?;
        self.data
            .get(offset..end)
            .ok_or_else(|| WeightLoadError::OutOfBounds {
                name: name.to_string(),
                offset,
                end,
                buf_len: self.data.len(),
            })
    }
}

// ---------------------------------------------------------------------------
// Extraction helpers (generic over TensorSource)
// ---------------------------------------------------------------------------

/// Extract f32 data from a tensor source by name.
///
/// Uses `f32::from_le_bytes` via `chunks_exact(4)` instead of `bytemuck::cast_slice`
/// to avoid panics from misaligned byte slices. Safetensors data offsets are not
/// guaranteed to be 4-byte aligned (the JSON header can have any length).
fn extract(src: &impl TensorSource, name: &str) -> Result<Vec<f32>, WeightLoadError> {
    let bytes = src.tensor_bytes(name)?;
    if bytes.len() % 4 != 0 {
        return Err(WeightLoadError::ByteAlignment {
            name: name.to_string(),
            len: bytes.len(),
        });
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let count = floats.iter().filter(|v| !v.is_finite()).count();
    if count > 0 {
        return Err(WeightLoadError::NonFiniteWeight {
            name: name.to_string(),
            count,
        });
    }
    Ok(floats)
}

fn load_dconv(
    src: &impl TensorSource,
    prefix: &str,
    channels: usize,
) -> Result<Vec<DConvSubLayerWeights>, WeightLoadError> {
    let mut layers = Vec::with_capacity(DCONV_DEPTH);
    for d in 0..DCONV_DEPTH {
        let dp = format!("{prefix}_dc{d}");
        layers.push(DConvSubLayerWeights {
            conv_compress_weight: extract(src, &format!("{dp}_compress_weight"))?,
            conv_compress_bias: extract(src, &format!("{dp}_compress_bias"))?,
            norm_compress_gamma: extract(src, &format!("{dp}_norm_compress_gamma"))?,
            norm_compress_beta: extract(src, &format!("{dp}_norm_compress_beta"))?,
            conv_expand_weight: extract(src, &format!("{dp}_expand_weight"))?,
            conv_expand_bias: extract(src, &format!("{dp}_expand_bias"))?,
            norm_expand_gamma: extract(src, &format!("{dp}_norm_expand_gamma"))?,
            norm_expand_beta: extract(src, &format!("{dp}_norm_expand_beta"))?,
            layer_scale: extract(src, &format!("{dp}_layer_scale"))?,
        });
    }
    let _ = channels; // Used for documentation; sizes validated by constructor.
    Ok(layers)
}

fn load_norm(src: &impl TensorSource, prefix: &str) -> Result<LayerNormWeights, WeightLoadError> {
    Ok(LayerNormWeights {
        weight: extract(src, &format!("{prefix}_weight"))?,
        bias: extract(src, &format!("{prefix}_bias"))?,
    })
}

fn load_encoder(src: &impl TensorSource) -> Result<DemucsTemporalEncoderWeights, WeightLoadError> {
    let mut blocks = Vec::with_capacity(DEPTH);
    for b in 0..DEPTH {
        let out_ch = channels_at_depth(b);
        let prefix = format!("enc{b}");
        blocks.push(EncoderBlockWeights {
            conv_weight: extract(src, &format!("{prefix}_conv_weight"))?,
            conv_bias: extract(src, &format!("{prefix}_conv_bias"))?,
            dconv: load_dconv(src, &prefix, out_ch)?,
            rewrite_weight: extract(src, &format!("{prefix}_rewrite_weight"))?,
            rewrite_bias: extract(src, &format!("{prefix}_rewrite_bias"))?,
        });
    }
    Ok(DemucsTemporalEncoderWeights { blocks })
}

fn load_decoder(src: &impl TensorSource) -> Result<DemucsTemporalDecoderWeights, WeightLoadError> {
    let mut blocks = Vec::with_capacity(DEPTH);
    for b in 0..DEPTH {
        let enc_depth = DEPTH - 1 - b;
        let in_ch = channels_at_depth(enc_depth);
        let prefix = format!("dec{b}");
        blocks.push(DecoderBlockWeights {
            rewrite_weight: extract(src, &format!("{prefix}_rewrite_weight"))?,
            rewrite_bias: extract(src, &format!("{prefix}_rewrite_bias"))?,
            dconv: load_dconv(src, &prefix, in_ch)?,
            conv_tr_weight: extract(src, &format!("{prefix}_conv_tr_weight"))?,
            conv_tr_bias: extract(src, &format!("{prefix}_conv_tr_bias"))?,
        });
    }
    Ok(DemucsTemporalDecoderWeights { blocks })
}

#[path = "htdemucs_weights_transformer.rs"]
mod transformer_weights;
use transformer_weights::load_transformer;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl HTDemucsWeights {
    /// Load temporal branch weights from a safetensors `WeightMap`.
    ///
    /// Expects the naming convention produced by `convert_demucs_safetensors.py`.
    /// Spectral branch weights are set to `None` (temporal-only mode).
    ///
    /// # Errors
    ///
    /// Returns `WeightLoadError` if any expected tensor is missing or
    /// has incorrect byte alignment.
    pub fn from_weight_map(wm: &WeightMap) -> Result<Self, WeightLoadError> {
        Self::from_source(wm)
    }

    /// Load weights from a safetensors file without mmap (fully safe).
    ///
    /// Reads the entire file into memory, parses the safetensors header, and
    /// extracts tensor data as owned `Vec<f32>`. Unlike [`from_weight_map`](Self::from_weight_map),
    /// this does not require mmap, `unsafe`, or a Metal context — suitable for
    /// consumers that want a safe loading path.
    ///
    /// For zero-copy loading with shared Metal buffers, use
    /// [`from_weight_map`](Self::from_weight_map) with a [`WeightMap`] instead.
    ///
    /// # Errors
    ///
    /// Returns `WeightLoadError::Io` on read failures or
    /// `WeightLoadError::SafetensorsFormat` on parse/extraction errors.
    pub fn from_safetensors_file(path: impl AsRef<Path>) -> Result<Self, WeightLoadError> {
        let parsed = ParsedSafetensors::from_file(path)?;
        Self::from_source(&parsed)
    }

    /// Shared loading logic: build weights from any `TensorSource`.
    fn from_source(src: &impl TensorSource) -> Result<Self, WeightLoadError> {
        Ok(Self {
            encoder: load_encoder(src)?,
            transformer: load_transformer(src)?,
            decoder: load_decoder(src)?,
            spectral_encoder: None,
            spectral_decoder: None,
        })
    }
}

#[cfg(test)]
#[path = "htdemucs_weights_tests.rs"]
mod tests;
