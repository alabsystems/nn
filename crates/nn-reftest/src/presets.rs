// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tolerance presets for common model types.
//!
//! Different model architectures have different numerical characteristics.
//! Transformer attention with softmax is less tolerant than simple MLPs.
//! Audio models with STFT/iSTFT have phase sensitivity. Quantized models
//! sacrifice precision for throughput.
//!
//! # Example
//!
//! ```rust
//! use nn_reftest::presets::TolerancePreset;
//! use nn_reftest::ComparisonConfig;
//!
//! // Get a config tuned for transformer models.
//! let config: ComparisonConfig = TolerancePreset::TRANSFORMER.to_config();
//! assert_eq!(config.abs_tolerance, 1e-4);
//! assert_eq!(config.rel_tolerance, 1e-3);
//!
//! // Or use the preset name for diagnostics.
//! assert_eq!(TolerancePreset::TRANSFORMER.name, "transformer");
//! ```

use crate::compare::ComparisonConfig;

/// A named set of tolerance thresholds tuned for a particular model class.
///
/// Each preset defines absolute, relative, and cosine similarity thresholds
/// chosen to match the typical numerical behavior of that model category.
/// Use [`to_config`](Self::to_config) to convert into a [`ComparisonConfig`]
/// suitable for [`compare_tensors`](crate::compare_tensors) or
/// [`compare_traces`](crate::compare_traces).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TolerancePreset {
    /// Human-readable identifier (e.g., `"transformer"`, `"audio"`).
    pub name: &'static str,
    /// Maximum allowed absolute difference per element.
    pub abs_threshold: f64,
    /// Maximum allowed relative difference per element.
    pub rel_threshold: f64,
    /// Minimum required cosine similarity (1.0 = identical direction).
    pub cos_threshold: f64,
    /// What this preset is intended for.
    pub description: &'static str,
}

impl TolerancePreset {
    /// Strict: for simple element-wise ops (relu, add, mul) where results
    /// should match to near-f32 precision. Matches [`ComparisonConfig::strict()`].
    ///
    /// abs=1e-6, rel=1e-5, cos=0.999999
    pub const STRICT: Self = Self {
        name: "strict",
        abs_threshold: 1e-6,
        rel_threshold: 1e-5,
        cos_threshold: 0.999_999,
        description: "Simple element-wise ops (relu, add, mul) with near-f32 precision",
    };

    /// Standard: for most model layers (linear, conv, normalization) in f32.
    ///
    /// abs=1e-5, rel=1e-4, cos=0.9999
    pub const STANDARD: Self = Self {
        name: "standard",
        abs_threshold: 1e-5,
        rel_threshold: 1e-4,
        cos_threshold: 0.9999,
        description: "Most model layers (linear, conv, normalization) in f32",
    };

    /// Transformer: for attention/softmax layers where exponential operations
    /// amplify small input differences.
    ///
    /// abs=1e-4, rel=1e-3, cos=0.999
    pub const TRANSFORMER: Self = Self {
        name: "transformer",
        abs_threshold: 1e-4,
        rel_threshold: 1e-3,
        cos_threshold: 0.999,
        description: "Attention/softmax layers with exponential amplification",
    };

    /// Audio: for STFT/iSTFT pipelines where phase sensitivity and
    /// windowed transforms introduce larger numerical differences.
    ///
    /// abs=1e-3, rel=1e-2, cos=0.99
    pub const AUDIO: Self = Self {
        name: "audio",
        abs_threshold: 1e-3,
        rel_threshold: 1e-2,
        cos_threshold: 0.99,
        description: "STFT/iSTFT pipelines with phase sensitivity",
    };

    /// Quantized: for int8/bf16 inference where reduced precision is expected.
    ///
    /// abs=1e-2, rel=5e-2, cos=0.95
    pub const QUANTIZED: Self = Self {
        name: "quantized",
        abs_threshold: 1e-2,
        rel_threshold: 5e-2,
        cos_threshold: 0.95,
        description: "int8/bf16 inference with reduced precision",
    };

    /// TTS: for text-to-speech models (e.g., Kokoro) where the full pipeline
    /// chains vocoder, prosody, and signal generation stages.
    ///
    /// abs=5e-3, rel=1e-2, cos=0.995
    pub const TTS: Self = Self {
        name: "tts",
        abs_threshold: 5e-3,
        rel_threshold: 1e-2,
        cos_threshold: 0.995,
        description: "Text-to-speech pipelines (vocoder + prosody + signal generation)",
    };

    /// All built-in presets, for iteration and lookup.
    pub const ALL: &'static [Self] = &[
        Self::STRICT,
        Self::STANDARD,
        Self::TRANSFORMER,
        Self::AUDIO,
        Self::QUANTIZED,
        Self::TTS,
    ];

    /// Convert this preset into a [`ComparisonConfig`].
    ///
    /// The resulting config uses the preset's absolute, relative, and cosine
    /// thresholds. Optional gates (RMS, peak amplitude, spectral) are left
    /// disabled; use the builder methods on [`ComparisonConfig`] to add them.
    #[must_use]
    pub fn to_config(&self) -> ComparisonConfig {
        ComparisonConfig::new(
            self.abs_threshold as f32,
            self.rel_threshold as f32,
            self.cos_threshold as f32,
        )
    }

    /// Look up a preset by name (case-insensitive).
    ///
    /// Returns `None` if no built-in preset matches.
    #[must_use]
    pub fn by_name(name: &str) -> Option<Self> {
        let lower = name.to_ascii_lowercase();
        Self::ALL.iter().find(|p| p.name == lower).copied()
    }
}

#[cfg(test)]
#[path = "presets_tests.rs"]
mod tests;
