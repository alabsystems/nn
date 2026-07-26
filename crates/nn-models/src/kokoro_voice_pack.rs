// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro voice pack loading — safetensors-based style embeddings.
//!
//! A voice pack is a safetensors file where each key is a voice name
//! (e.g., `"af_heart"`, `"am_adam"`) mapping to a `[256]` or `[1, 256]`
//! style embedding tensor. The loader validates shapes, normalizes to
//! `[1, 2*style_dim]`, and provides name-based lookup.
//!
//! # Pipeline Position
//!
//! ```text
//! Text → preprocess → espeak → remap → tokenize → [VoicePack.get("af_heart")] → synthesize
//!                                                        ↑ style embedding
//! ```
//!
//! # File Format
//!
//! The safetensors file should contain named tensors with shape `[256]` or `[1, 256]`
//! (for `style_dim=128`). The voice name is the tensor key. Example keys:
//! `af_heart`, `af_bella`, `am_adam`, `bf_emma`.

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::DynTensor;

use crate::kokoro_error::KokoroError;

/// A collection of named voice style embeddings loaded from a safetensors file.
///
/// Each voice is a `[1, 2*style_dim]` tensor (default: `[1, 256]`).
/// Used by [`KokoroTextPipeline`](crate::kokoro_pipeline::KokoroTextPipeline)
/// and [`CompiledKokoro`] to select voice identity at synthesis time.
#[derive(Debug, Clone)]
pub struct VoicePack {
    voices: HashMap<String, DynTensor>,
    style_dim: usize,
}

impl VoicePack {
    /// Load a voice pack from a safetensors file.
    ///
    /// Each tensor in the file is treated as a voice style embedding.
    /// Tensors must have shape `[2*style_dim]` or `[1, 2*style_dim]`.
    /// The default `style_dim` is 128 (total embedding size = 256).
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::Tensor` if the file cannot be read or parsed.
    /// Returns `KokoroError::InvalidInput` if any tensor has the wrong shape.
    pub fn load(path: impl AsRef<Path>, style_dim: usize) -> Result<Self, KokoroError> {
        let tensors = nn_core::dyn_tensor::load_safetensors(path)?;
        Self::from_tensors(tensors, style_dim)
    }

    /// Load a voice pack from in-memory safetensors bytes.
    pub fn load_from_bytes(bytes: &[u8], style_dim: usize) -> Result<Self, KokoroError> {
        let tensors = nn_core::dyn_tensor::load_safetensors_from_bytes(bytes)?;
        Self::from_tensors(tensors, style_dim)
    }

    /// Build a voice pack from pre-loaded tensors.
    ///
    /// Validates and normalizes each tensor to `[1, 2*style_dim]`.
    pub fn from_tensors(
        tensors: HashMap<String, DynTensor>,
        style_dim: usize,
    ) -> Result<Self, KokoroError> {
        let expected_len = 2 * style_dim;
        let mut voices = HashMap::with_capacity(tensors.len());

        for (name, tensor) in tensors {
            let normalized = normalize_style_shape(&name, &tensor, expected_len)?;
            voices.insert(name, normalized);
        }

        Ok(Self { voices, style_dim })
    }

    /// Create an empty voice pack with the given style dimension.
    #[must_use]
    pub fn empty(style_dim: usize) -> Self {
        Self {
            voices: HashMap::new(),
            style_dim,
        }
    }

    /// Add a voice to the pack.
    ///
    /// The tensor must have shape `[2*style_dim]` or `[1, 2*style_dim]`.
    pub fn add_voice(
        &mut self,
        name: impl Into<String>,
        style: &DynTensor,
    ) -> Result<(), KokoroError> {
        let name = name.into();
        let expected_len = 2 * self.style_dim;
        let normalized = normalize_style_shape(&name, style, expected_len)?;
        self.voices.insert(name, normalized);
        Ok(())
    }

    /// Get a voice style embedding by name.
    ///
    /// Returns `None` if the voice is not in the pack.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&DynTensor> {
        self.voices.get(name)
    }

    /// Get a voice style embedding by name, or return an error.
    pub fn get_or_err(&self, name: &str) -> Result<&DynTensor, KokoroError> {
        self.voices.get(name).ok_or_else(|| {
            KokoroError::InvalidInput(format!(
                "voice '{name}' not found in voice pack (available: {})",
                self.voice_names().collect::<Vec<_>>().join(", ")
            ))
        })
    }

    /// Number of voices in the pack.
    #[must_use]
    pub fn len(&self) -> usize {
        self.voices.len()
    }

    /// Whether the pack is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.voices.is_empty()
    }

    /// Iterator over voice names.
    pub fn voice_names(&self) -> impl Iterator<Item = &str> {
        self.voices.keys().map(String::as_str)
    }

    /// The style dimension (half of the total embedding size).
    #[must_use]
    pub fn style_dim(&self) -> usize {
        self.style_dim
    }

    /// Get all voice names sorted alphabetically.
    #[must_use]
    pub fn sorted_voice_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.voice_names().collect();
        names.sort_unstable();
        names
    }
}

/// Normalize a style tensor to `[1, expected_len]`.
///
/// Accepts `[expected_len]` (adds batch dim) or `[1, expected_len]` (pass-through).
fn normalize_style_shape(
    name: &str,
    tensor: &DynTensor,
    expected_len: usize,
) -> Result<DynTensor, KokoroError> {
    let dims = tensor.dims();
    match dims.len() {
        1 => {
            if dims[0] != expected_len {
                return Err(KokoroError::InvalidInput(format!(
                    "voice '{name}': expected [{expected_len}], got [{}]",
                    dims[0]
                )));
            }
            // Reshape [N] → [1, N]
            tensor
                .reshape([1, expected_len])
                .map_err(KokoroError::Tensor)
        }
        2 => {
            if dims[0] != 1 || dims[1] != expected_len {
                return Err(KokoroError::InvalidInput(format!(
                    "voice '{name}': expected [1, {expected_len}], got [{}, {}]",
                    dims[0], dims[1]
                )));
            }
            Ok(tensor.clone())
        }
        _ => Err(KokoroError::InvalidInput(format!(
            "voice '{name}': expected 1D or 2D tensor, got {}D",
            dims.len()
        ))),
    }
}

#[cfg(test)]
#[path = "kokoro_voice_pack_tests.rs"]
mod tests;
