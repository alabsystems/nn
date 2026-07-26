// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Configuration types and result structs for tensor comparison.
//!
//! Extracted from `compare.rs` to stay under the 500-line limit (Part of #1575).

/// Tolerance configuration for tensor comparison.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ComparisonConfig {
    /// Maximum allowed absolute difference per element.
    pub abs_tolerance: f32,
    /// Maximum allowed relative difference per element.
    pub rel_tolerance: f32,
    /// Minimum required cosine similarity (1.0 = identical direction).
    pub cosine_threshold: f32,
    /// Maximum allowed RMS (root-mean-square) difference. `None` = gate disabled.
    pub rms_tolerance: Option<f32>,
    /// Maximum allowed absolute value in the candidate tensor. `None` = gate disabled.
    pub peak_amplitude_limit: Option<f32>,
    /// Spectral comparison config for audio tensors. `None` = spectral gate disabled.
    #[cfg(feature = "spectral")]
    pub spectral: Option<crate::spectral::SpectralConfig>,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            abs_tolerance: 1e-5,
            rel_tolerance: 1e-4,
            cosine_threshold: 0.9999,
            rms_tolerance: None,
            peak_amplitude_limit: None,
            #[cfg(feature = "spectral")]
            spectral: None,
        }
    }
}

impl ComparisonConfig {
    /// Create a comparison config with custom tolerances.
    ///
    /// RMS and peak amplitude gates default to disabled (`None`).
    /// Use builder methods to enable them:
    /// ```
    /// # use nn_reftest::ComparisonConfig;
    /// let config = ComparisonConfig::new(1e-5, 1e-4, 0.9999)
    ///     .with_rms_tolerance(1e-4)
    ///     .with_peak_amplitude_limit(1e3);
    /// ```
    #[must_use]
    pub fn new(abs_tolerance: f32, rel_tolerance: f32, cosine_threshold: f32) -> Self {
        Self {
            abs_tolerance,
            rel_tolerance,
            cosine_threshold,
            rms_tolerance: None,
            peak_amplitude_limit: None,
            #[cfg(feature = "spectral")]
            spectral: None,
        }
    }

    /// Enable the RMS (root-mean-square) difference gate.
    #[must_use]
    pub fn with_rms_tolerance(mut self, tolerance: f32) -> Self {
        self.rms_tolerance = Some(tolerance);
        self
    }

    /// Enable the peak amplitude gate.
    #[must_use]
    pub fn with_peak_amplitude_limit(mut self, limit: f32) -> Self {
        self.peak_amplitude_limit = Some(limit);
        self
    }

    /// Strict comparison: tight tolerances suitable for f32 bit-level equivalence.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            abs_tolerance: 1e-6,
            rel_tolerance: 1e-5,
            cosine_threshold: 0.999_999,
            rms_tolerance: None,
            peak_amplitude_limit: None,
            #[cfg(feature = "spectral")]
            spectral: None,
        }
    }

    /// Relaxed comparison: suitable for f16 or cross-device comparison.
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            abs_tolerance: 1e-2,
            rel_tolerance: 1e-1,
            cosine_threshold: 0.999,
            rms_tolerance: None,
            peak_amplitude_limit: None,
            #[cfg(feature = "spectral")]
            spectral: None,
        }
    }

    /// Enable spectral comparison for audio tensors.
    ///
    /// When set, `compare_tensors()` will additionally compute spectral metrics
    /// for 1-D tensors and include the result in `LayerComparison`.
    #[cfg(feature = "spectral")]
    #[must_use]
    pub fn with_spectral(mut self, spectral: crate::spectral::SpectralConfig) -> Self {
        self.spectral = Some(spectral);
        self
    }
}

/// Per-layer comparison result between a reference and candidate tensor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LayerComparison {
    /// Layer/checkpoint name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Largest absolute element-wise difference.
    pub max_abs_diff: f32,
    /// Mean absolute element-wise difference.
    pub mean_abs_diff: f32,
    /// Cosine similarity between the two tensors (1.0 = identical).
    pub cosine_similarity: f32,
    /// Largest relative element-wise difference.
    pub max_rel_diff: f32,
    /// Total number of elements compared.
    pub num_elements: usize,
    /// Root-mean-square difference between reference and candidate.
    pub rms_diff: f32,
    /// Maximum absolute value in the candidate tensor.
    pub peak_amplitude: f32,
    /// Whether all metrics are within the configured tolerances.
    pub passed: bool,
    /// Spectral comparison result (populated when spectral config is set and tensor is 1-D).
    #[cfg(feature = "spectral")]
    pub spectral: Option<crate::spectral::SpectralComparison>,
}

impl std::fmt::Display for LayerComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        write!(
            f,
            "[{status}] {name} ({shape:?}): max_abs={max_abs:.2e}, mean_abs={mean_abs:.2e}, \
             rms={rms:.2e}, cos={cos:.6}, max_rel={max_rel:.2e}, peak={peak:.2e}",
            name = self.name,
            shape = self.shape,
            max_abs = self.max_abs_diff,
            mean_abs = self.mean_abs_diff,
            rms = self.rms_diff,
            cos = self.cosine_similarity,
            max_rel = self.max_rel_diff,
            peak = self.peak_amplitude,
        )
    }
}

/// Full comparison report across all checkpoints.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DivergenceReport {
    /// Per-layer comparison results, one per checkpoint pair.
    pub layers: Vec<LayerComparison>,
    /// Index of the first layer exceeding tolerance, if any.
    pub first_failure: Option<usize>,
    /// `true` if every layer passed all tolerance checks.
    pub all_passed: bool,
}

impl DivergenceReport {
    /// Render a human-readable summary of the comparison.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut out = String::new();
        for layer in &self.layers {
            out.push_str(&format!("{layer}\n"));
        }
        if self.all_passed {
            out.push_str(&format!("\nAll {} layers passed.\n", self.layers.len()));
        } else if let Some(idx) = self.first_failure {
            out.push_str(&format!(
                "\nFirst failure at layer {} ('{}').\n",
                idx, self.layers[idx].name,
            ));
        }
        out
    }
}
