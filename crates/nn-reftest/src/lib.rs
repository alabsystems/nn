// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reference tensor comparison tooling for nn.
//!
//! `nn-reftest` provides a first-class framework for comparing intermediate
//! tensors between a reference implementation (typically PyTorch) and a Rust/GPU
//! implementation. This catches the bug class that formal verification misses:
//! porting errors, wiring errors, convention mismatches, and missing data.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use nn_reftest::{ReferenceTrace, compare_traces, ComparisonConfig};
//!
//! // Load reference tensors exported from PyTorch (safetensors format).
//! let reference = nn_reftest::load_safetensors("traces/reference.safetensors")
//!     .expect("failed to load reference");
//!
//! // Capture Rust model outputs.
//! let mut candidate = ReferenceTrace::new();
//! candidate.checkpoint("layer1", &[1.0, 2.0, 3.0], &[3]).expect("valid checkpoint");
//! candidate.checkpoint("layer2", &[4.0, 5.0], &[2]).expect("valid checkpoint");
//!
//! // Compare layer-by-layer.
//! let report = compare_traces(&reference, &candidate, &ComparisonConfig::default())
//!     .expect("comparison failed");
//!
//! assert!(report.all_passed, "{}", report.summary());
//! ```
//!
//! # Test Assertions
//!
//! Use the [`assert_traces_match!`] macro in tests:
//!
//! ```rust
//! # use nn_reftest::{ReferenceTrace, assert_traces_match};
//! # let mut candidate = ReferenceTrace::new();
//! # candidate.checkpoint("layer1", &[1.0, 2.0, 3.0], &[3]).expect("valid checkpoint");
//! # let mut reference = ReferenceTrace::new();
//! # reference.checkpoint("layer1", &[1.0, 2.0, 3.0], &[3]).expect("valid checkpoint");
//! assert_traces_match!(candidate, reference);
//! assert_traces_match!(candidate, reference, abs = 1e-4, rel = 1e-3);
//! ```
//!
//! # Features
//!
//! - **`spectral`** — Enables STFT-based audio comparison via
//!   `compare_spectral()`, `assert_spectral_match!`, and
//!   `SpectralConfig`. Metrics include log-spectral distance,
//!   spectral convergence, and phase coherence. Required for audio
//!   model parity testing (Demucs, Kokoro, Whisper).

pub mod compare;
pub mod error;
pub mod load;
pub mod npy;
pub mod presets;
#[cfg(feature = "spectral")]
pub mod spectral;
pub mod tolerance;
pub mod trace;

#[cfg(kani)]
mod kani_compare;

#[cfg(kani)]
mod kani_compare_extended;

#[cfg(kani)]
mod kani_npy_convert;

#[cfg(kani)]
mod kani_npy_load_proofs;

#[cfg(kani)]
mod kani_load;

#[cfg(kani)]
mod kani_compare_basic;
#[cfg(kani)]
mod kani_load_safetensors;
#[cfg(kani)]
mod kani_npy_format;

#[cfg(kani)]
mod kani_trace_config;

#[cfg(kani)]
mod kani_reftest_wave11;

#[cfg(all(kani, feature = "spectral"))]
mod kani_spectral;

// Re-exports for convenience.
pub use compare::{
    compare_tensors, compare_traces, ComparisonConfig, DivergenceReport, LayerComparison,
};
pub use error::ReftestError;
pub use load::{load_safetensors, load_safetensors_from_bytes};
pub use npy::{load_npy, load_npy_dir, load_npy_from_bytes, NpyDType, NpyError, NpyTensor};
pub use presets::TolerancePreset;
pub use trace::{NamedTensor, ReferenceTrace};

#[cfg(feature = "spectral")]
pub use spectral::{
    compare_spectral, stft_full, stft_magnitude, SpectralComparison, SpectralConfig, StftConfig,
    StftResult, WindowFn,
};

/// Assert that two traces match within tolerance.
///
/// Compares all checkpoints and panics with a detailed message on the first
/// layer that exceeds the configured thresholds.
///
/// # Examples
///
/// ```rust
/// # use nn_reftest::{ReferenceTrace, assert_traces_match};
/// # let mut candidate = ReferenceTrace::new();
/// # candidate.checkpoint("layer1", &[1.0, 2.0, 3.0], &[3]).expect("valid checkpoint");
/// # let mut reference = ReferenceTrace::new();
/// # reference.checkpoint("layer1", &[1.0, 2.0, 3.0], &[3]).expect("valid checkpoint");
/// // Default tolerance (abs=1e-5, rel=1e-4, cos=0.9999).
/// assert_traces_match!(candidate, reference);
///
/// // Unified epsilon budget for abs+rel thresholds.
/// assert_traces_match!(candidate, reference, epsilon = 1e-5);
///
/// // Custom absolute and relative tolerance.
/// assert_traces_match!(candidate, reference, abs = 1e-4, rel = 1e-3);
///
/// // Full custom config.
/// assert_traces_match!(candidate, reference, abs = 1e-4, rel = 1e-3, cos = 0.999);
/// ```
#[macro_export]
macro_rules! assert_traces_match {
    ($candidate:expr, $reference:expr) => {
        $crate::assert_traces_match!($candidate, $reference, abs = 1e-5, rel = 1e-4, cos = 0.9999)
    };
    ($candidate:expr, $reference:expr, epsilon = $epsilon:expr) => {
        $crate::assert_traces_match!(
            $candidate,
            $reference,
            abs = $epsilon,
            rel = $epsilon,
            cos = 0.9999
        )
    };
    ($candidate:expr, $reference:expr, epsilon = $epsilon:expr, cos = $cos:expr) => {
        $crate::assert_traces_match!(
            $candidate,
            $reference,
            abs = $epsilon,
            rel = $epsilon,
            cos = $cos
        )
    };
    ($candidate:expr, $reference:expr, abs = $abs:expr, rel = $rel:expr) => {
        $crate::assert_traces_match!($candidate, $reference, abs = $abs, rel = $rel, cos = 0.9999)
    };
    ($candidate:expr, $reference:expr, abs = $abs:expr, rel = $rel:expr, cos = $cos:expr) => {{
        let config = $crate::ComparisonConfig::new($abs, $rel, $cos);
        let report = $crate::compare_traces(&$reference, &$candidate, &config)
            .expect("trace comparison failed");
        if !report.all_passed {
            let idx = report
                .first_failure
                .expect("first_failure set when !all_passed");
            let layer = &report.layers[idx];
            panic!(
                "Tensor mismatch at layer '{}' (index {}):\n  \
                 max_abs_diff = {:.6e}\n  \
                 mean_abs_diff = {:.6e}\n  \
                 cosine_similarity = {:.8}\n  \
                 max_rel_diff = {:.6e}\n  \
                 threshold: abs={:.1e}, rel={:.1e}, cos={:.6}\n  \
                 shape: {:?}, elements: {}",
                layer.name,
                idx,
                layer.max_abs_diff,
                layer.mean_abs_diff,
                layer.cosine_similarity,
                layer.max_rel_diff,
                $abs,
                $rel,
                $cos,
                layer.shape,
                layer.num_elements,
            );
        }
    }};
}

/// Assert that two traces match using a named [`TolerancePreset`].
///
/// This is a convenience wrapper around [`assert_traces_match!`] that accepts
/// a [`TolerancePreset`] constant instead of raw numeric thresholds.
///
/// # Examples
///
/// ```rust
/// # use nn_reftest::{ReferenceTrace, assert_traces_match_preset, TolerancePreset};
/// # let mut candidate = ReferenceTrace::new();
/// # candidate.checkpoint("layer1", &[1.0, 2.0, 3.0], &[3]).expect("valid checkpoint");
/// # let mut reference = ReferenceTrace::new();
/// # reference.checkpoint("layer1", &[1.0, 2.0, 3.0], &[3]).expect("valid checkpoint");
/// // Use a model-specific preset.
/// assert_traces_match_preset!(candidate, reference, TolerancePreset::TRANSFORMER);
///
/// // Or the default standard preset.
/// assert_traces_match_preset!(candidate, reference, TolerancePreset::STANDARD);
/// ```
#[macro_export]
macro_rules! assert_traces_match_preset {
    ($candidate:expr, $reference:expr, $preset:expr) => {{
        let preset: $crate::TolerancePreset = $preset;
        let config = preset.to_config();
        let report = $crate::compare_traces(&$reference, &$candidate, &config)
            .expect("trace comparison failed");
        if !report.all_passed {
            let idx = report
                .first_failure
                .expect("first_failure set when !all_passed");
            let layer = &report.layers[idx];
            panic!(
                "Tensor mismatch at layer '{}' (index {}) using preset '{}':\n  \
                 max_abs_diff = {:.6e}\n  \
                 mean_abs_diff = {:.6e}\n  \
                 cosine_similarity = {:.8}\n  \
                 max_rel_diff = {:.6e}\n  \
                 threshold: abs={:.1e}, rel={:.1e}, cos={:.6}\n  \
                 preset: {}\n  \
                 shape: {:?}, elements: {}",
                layer.name,
                idx,
                preset.name,
                layer.max_abs_diff,
                layer.mean_abs_diff,
                layer.cosine_similarity,
                layer.max_rel_diff,
                preset.abs_threshold,
                preset.rel_threshold,
                preset.cos_threshold,
                preset.description,
                layer.shape,
                layer.num_elements,
            );
        }
    }};
}

/// Assert spectral match between two 1-D audio signals.
///
/// Compares signals in the frequency domain using STFT-based metrics:
/// log-spectral distance, spectral convergence, and phase coherence.
///
/// # Examples
///
/// ```rust,no_run
/// # use nn_reftest::assert_spectral_match;
/// # let candidate = vec![0.0f32; 1024];
/// # let reference = vec![0.0f32; 1024];
/// // Default thresholds (LSD < 1.0 dB, SC < 0.01, phase > 0.95).
/// assert_spectral_match!(candidate, reference);
///
/// // Custom thresholds.
/// assert_spectral_match!(candidate, reference, lsd_db = 2.0, sc = 0.05);
/// ```
#[cfg(feature = "spectral")]
#[macro_export]
macro_rules! assert_spectral_match {
    ($candidate:expr, $reference:expr) => {
        $crate::assert_spectral_match!(
            $candidate,
            $reference,
            lsd_db = 1.0,
            sc = 0.01,
            phase = 0.95
        )
    };
    ($candidate:expr, $reference:expr, lsd_db = $lsd:expr, sc = $sc:expr) => {
        $crate::assert_spectral_match!(
            $candidate,
            $reference,
            lsd_db = $lsd,
            sc = $sc,
            phase = 0.95
        )
    };
    ($candidate:expr, $reference:expr, lsd_db = $lsd:expr, sc = $sc:expr, phase = $phase:expr) => {{
        let config = $crate::SpectralConfig::new($lsd, $sc, $phase);
        let result = $crate::compare_spectral(&$reference[..], &$candidate[..], &config)
            .expect("spectral comparison failed");
        if !result.passed {
            panic!(
                "Spectral comparison failed:\n  \
                 log_spectral_distance = {:.4} dB (max: {:.4})\n  \
                 spectral_convergence = {:.6} (max: {:.6})\n  \
                 phase_coherence = {:.4} (min: {:.4})\n  \
                 max_magnitude_diff = {:.4} dB\n  \
                 mean_magnitude_diff = {:.4} dB",
                result.log_spectral_distance_db,
                $lsd,
                result.spectral_convergence,
                $sc,
                result.phase_coherence,
                $phase,
                result.max_magnitude_diff_db,
                result.mean_magnitude_diff_db,
            );
        }
    }};
}

#[cfg(test)]
#[path = "reftest_expanded_tests.rs"]
mod reftest_expanded_tests;

#[cfg(test)]
#[path = "comparison_tests.rs"]
mod comparison_tests;

#[cfg(test)]
#[path = "trace_comparison_tests.rs"]
mod trace_comparison_tests;

#[cfg(test)]
#[path = "reftest_extended_tests.rs"]
mod reftest_extended_tests;

#[cfg(test)]
#[path = "reftest_comparison_extended_tests.rs"]
mod reftest_comparison_extended_tests;

#[cfg(test)]
#[path = "tolerance_extended_tests.rs"]
mod tolerance_extended_tests;

#[cfg(test)]
#[path = "reftest_tolerance_comparison_tests.rs"]
mod reftest_tolerance_comparison_tests;

#[cfg(test)]
#[path = "reftest_pipeline_extended_tests.rs"]
mod reftest_pipeline_extended_tests;
