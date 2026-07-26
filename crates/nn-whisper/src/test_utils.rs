// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-crate test helpers for nn-whisper.
//!
//! Provides a shared `tiny_config()` used by all whisper test files.
//! Eliminates 4 duplicate definitions across decode_tests, whisper_tests,
//! safetensors_load, and decode_integration.

use crate::config::WhisperConfig;
use crate::WhisperModel;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

/// Tiny Whisper config for fast testing: 2 heads, 1 layer, small dims.
pub fn tiny_config() -> WhisperConfig {
    WhisperConfig {
        num_mel_bins: 4,
        max_source_positions: 8,
        d_model: 16,
        encoder_attention_heads: 2,
        encoder_layers: 1,
        encoder_ffn_dim: 32,
        vocab_size: 32,
        max_target_positions: 16,
        decoder_attention_heads: 2,
        decoder_layers: 1,
        decoder_ffn_dim: 32,
    }
}

/// Tiny WhisperModel with zero weights for fast testing.
pub fn tiny_model() -> WhisperModel {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    WhisperModel::load(&vb, config).expect("invariant: zero-weight model loads")
}

/// Tiny encoder output tensor `[1, 8, d_model]` for decode tests.
pub fn tiny_encoder_output() -> DynTensor {
    let config = tiny_config();
    DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).expect("invariant: zeros tensor")
}
