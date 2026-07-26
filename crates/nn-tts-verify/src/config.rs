// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Configuration types for TTS audio verification.

use crate::bounds::SpectralCoverageConfig;
use crate::error::{validate_finite, validate_finite_positive, InvalidConfigKind, TtsVerifyError};
use crate::multi_res_stft::MultiResStftConfig;

/// Policy for how the verifier handles hard bound failures.
///
/// Controls whether a failing hard bound check causes the verification to
/// return an error, logs a warning but continues, or attempts remediation
/// (e.g., clamping audio to fix clipping).
///
/// Part of #3780, #3758, #3760.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum RejectionPolicy {
    /// Return an error when any hard bound fails (default).
    ///
    /// The `Certificate.overall_passed` will be `false` and the caller
    /// decides how to handle it. This is the existing behavior.
    #[default]
    Reject,
    /// Log the failure but do not mark the certificate as failed.
    ///
    /// Individual `HardBound.passed` fields still reflect the true result,
    /// but `Certificate.overall_passed` ignores hard bound failures.
    /// Useful for monitoring/dashboards where you want to observe violations
    /// without blocking synthesis output.
    Warn,
    /// Attempt to fix the violation before final output.
    ///
    /// Currently a no-op (falls back to `Warn` behavior) — reserved for
    /// future remediation logic (e.g., DC offset removal, peak normalization).
    Remediate,
}

/// Per-check threshold override for a single hard bound.
///
/// When `Some`, the override replaces the corresponding field in
/// [`HardBoundsConfig`] for that specific check. When `None`, the
/// default from `HardBoundsConfig` is used.
///
/// Part of #3780, #3758, #3760.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CheckOverrides {
    /// Override minimum RMS for non-silence check.
    pub min_rms: Option<f64>,
    /// Override maximum amplitude for clipping check.
    pub max_amplitude: Option<f64>,
    /// Override maximum DC offset.
    pub max_dc_offset: Option<f64>,
    /// Override maximum sample-to-sample difference for click check.
    pub max_click_diff: Option<f64>,
    /// Override minimum duration in seconds.
    pub min_duration_sec: Option<f64>,
    /// Override maximum duration in seconds.
    pub max_duration_sec: Option<f64>,
    /// Override maximum tail-to-body energy ratio.
    pub max_tail_energy_ratio: Option<f64>,
}

impl CheckOverrides {
    /// Create empty overrides (all `None` — use defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate that any present override values are finite and sensible.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        if let Some(v) = self.min_rms {
            validate_finite_positive(v, "override:min_rms")?;
        }
        if let Some(v) = self.max_amplitude {
            validate_finite_positive(v, "override:max_amplitude")?;
        }
        if let Some(v) = self.max_dc_offset {
            validate_finite(v, "override:max_dc_offset")?;
        }
        if let Some(v) = self.max_click_diff {
            validate_finite_positive(v, "override:max_click_diff")?;
        }
        if let Some(v) = self.min_duration_sec {
            validate_finite(v, "override:min_duration_sec")?;
        }
        if let Some(v) = self.max_duration_sec {
            validate_finite_positive(v, "override:max_duration_sec")?;
        }
        if let Some(v) = self.max_tail_energy_ratio {
            validate_finite_positive(v, "override:max_tail_energy_ratio")?;
        }
        // Cross-field: if both duration overrides are present, min < max.
        if let (Some(min_d), Some(max_d)) = (self.min_duration_sec, self.max_duration_sec) {
            if min_d >= max_d {
                return Err(TtsVerifyError::InvalidConfig(
                    InvalidConfigKind::RangeInverted {
                        param: "override:min_duration_sec / max_duration_sec",
                    },
                ));
            }
        }
        Ok(())
    }
}

/// Configuration for hard bound checks.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HardBoundsConfig {
    /// Minimum RMS for non-silence check. Default: 0.01.
    pub min_rms: f64,
    /// Maximum amplitude for clipping check. Default: 1.0.
    pub max_amplitude: f64,
    /// Maximum DC offset. Default: 0.05.
    pub max_dc_offset: f64,
    /// Maximum sample-to-sample difference for click check. Default: 0.5.
    pub max_click_diff: f64,
    /// Minimum duration in seconds. Default: 0.1.
    pub min_duration_sec: f64,
    /// Maximum duration in seconds. Default: 300.0.
    pub max_duration_sec: f64,
    /// Length of tail region in milliseconds for tail energy check. Default: 50.0.
    pub tail_ms: f64,
    /// Length of body region in milliseconds for tail energy check. Default: 500.0.
    pub body_ms: f64,
    /// Maximum tail-to-body RMS energy ratio. Default: 3.0.
    pub max_tail_energy_ratio: f64,
    /// Spectral coverage configuration.
    pub spectral: SpectralCoverageConfig,
    /// Policy for handling hard bound failures. Default: [`RejectionPolicy::Reject`].
    ///
    /// Part of #3780, #3758, #3760.
    pub rejection_policy: RejectionPolicy,
    /// Per-check threshold overrides. Default: no overrides.
    ///
    /// When set, individual thresholds in overrides take precedence over the
    /// top-level fields (e.g., `overrides.min_rms` overrides `self.min_rms`).
    ///
    /// Part of #3780, #3758, #3760.
    pub overrides: CheckOverrides,
}

impl HardBoundsConfig {
    /// Validate that all f64 fields are finite and within sensible ranges.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        validate_finite_positive(self.min_rms, "min_rms")?;
        validate_finite_positive(self.max_amplitude, "max_amplitude")?;
        validate_finite(self.max_dc_offset, "max_dc_offset")?;
        validate_finite_positive(self.max_click_diff, "max_click_diff")?;
        validate_finite(self.min_duration_sec, "min_duration_sec")?;
        validate_finite_positive(self.max_duration_sec, "max_duration_sec")?;
        if self.min_duration_sec >= self.max_duration_sec {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::RangeInverted {
                    param: "min_duration_sec / max_duration_sec",
                },
            ));
        }
        validate_finite_positive(self.tail_ms, "tail_ms")?;
        validate_finite_positive(self.body_ms, "body_ms")?;
        validate_finite_positive(self.max_tail_energy_ratio, "max_tail_energy_ratio")?;
        self.spectral.validate()?;
        self.overrides.validate()?;
        Ok(())
    }

    /// Return the effective minimum RMS threshold (override or default).
    #[must_use]
    pub fn effective_min_rms(&self) -> f64 {
        self.overrides.min_rms.unwrap_or(self.min_rms)
    }

    /// Return the effective maximum amplitude threshold (override or default).
    #[must_use]
    pub fn effective_max_amplitude(&self) -> f64 {
        self.overrides.max_amplitude.unwrap_or(self.max_amplitude)
    }

    /// Return the effective maximum DC offset threshold (override or default).
    #[must_use]
    pub fn effective_max_dc_offset(&self) -> f64 {
        self.overrides.max_dc_offset.unwrap_or(self.max_dc_offset)
    }

    /// Return the effective maximum click diff threshold (override or default).
    #[must_use]
    pub fn effective_max_click_diff(&self) -> f64 {
        self.overrides.max_click_diff.unwrap_or(self.max_click_diff)
    }

    /// Return the effective minimum duration in seconds (override or default).
    #[must_use]
    pub fn effective_min_duration_sec(&self) -> f64 {
        self.overrides
            .min_duration_sec
            .unwrap_or(self.min_duration_sec)
    }

    /// Return the effective maximum duration in seconds (override or default).
    #[must_use]
    pub fn effective_max_duration_sec(&self) -> f64 {
        self.overrides
            .max_duration_sec
            .unwrap_or(self.max_duration_sec)
    }

    /// Return the effective maximum tail-to-body energy ratio (override or default).
    #[must_use]
    pub fn effective_max_tail_energy_ratio(&self) -> f64 {
        self.overrides
            .max_tail_energy_ratio
            .unwrap_or(self.max_tail_energy_ratio)
    }
}

impl Default for HardBoundsConfig {
    fn default() -> Self {
        Self {
            min_rms: 0.01,
            max_amplitude: 1.0,
            max_dc_offset: 0.05,
            max_click_diff: 0.5,
            min_duration_sec: 0.1,
            max_duration_sec: 300.0,
            tail_ms: 50.0,
            body_ms: 500.0,
            max_tail_energy_ratio: 3.0,
            spectral: SpectralCoverageConfig::default(),
            rejection_policy: RejectionPolicy::default(),
            overrides: CheckOverrides::default(),
        }
    }
}

/// Configuration for quality metrics.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct QualityConfig {
    /// Maximum MCD in dB. Default: 6.0 (Kubichek 1993).
    pub max_mcd_db: f64,
    /// Minimum HNR in dB. Default: 15.0 (Boersma 1993).
    pub min_hnr_db: f64,
    /// F0 range in Hz. Default: (80.0, 400.0) (Titze 1994).
    pub f0_range: (f64, f64),
    /// Spectral tilt range in dB/octave. Default: (-12.0, -3.0) (Fant 1960).
    pub spectral_tilt: (f64, f64),
    /// Minimum cosine similarity for voice matching. Default: 0.85.
    pub min_cosine_similarity: f64,
    /// Minimum SNR in dB. Default: 10.0.
    pub min_snr_db: f64,
    /// Minimum SDR in dB. Default: 5.0 (Vincent et al. 2006).
    pub min_sdr_db: f64,
    /// Multi-resolution STFT loss config. Default: None (disabled).
    pub multi_res_stft: Option<MultiResStftConfig>,
    /// Minimum F0 contour correlation. Default: None (disabled).
    pub min_f0_contour_correlation: Option<f64>,
    /// Minimum STOI score. Default: None (disabled).
    pub min_stoi: Option<f64>,
    /// Minimum PESQ MOS-LQO score. Default: None (disabled).
    pub min_pesq: Option<f64>,
}

impl QualityConfig {
    /// Validate that all f64 fields are finite.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        validate_finite_positive(self.max_mcd_db, "max_mcd_db")?;
        validate_finite(self.min_hnr_db, "min_hnr_db")?;
        validate_finite(self.f0_range.0, "f0_range.0")?;
        validate_finite(self.f0_range.1, "f0_range.1")?;
        if self.f0_range.0 >= self.f0_range.1 {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::RangeInverted { param: "f0_range" },
            ));
        }
        validate_finite(self.spectral_tilt.0, "spectral_tilt.0")?;
        validate_finite(self.spectral_tilt.1, "spectral_tilt.1")?;
        validate_finite(self.min_cosine_similarity, "min_cosine_similarity")?;
        validate_finite(self.min_snr_db, "min_snr_db")?;
        validate_finite(self.min_sdr_db, "min_sdr_db")?;
        if let Some(ref stft) = self.multi_res_stft {
            stft.validate()?;
        }
        if let Some(v) = self.min_f0_contour_correlation {
            validate_finite(v, "min_f0_contour_correlation")?;
        }
        if let Some(v) = self.min_stoi {
            validate_finite(v, "min_stoi")?;
        }
        if let Some(v) = self.min_pesq {
            validate_finite(v, "min_pesq")?;
        }
        Ok(())
    }
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            max_mcd_db: 6.0,
            min_hnr_db: 15.0,
            f0_range: (80.0, 400.0),
            spectral_tilt: (-12.0, -3.0),
            min_cosine_similarity: 0.85,
            min_snr_db: 10.0,
            min_sdr_db: 5.0,
            multi_res_stft: None,
            min_f0_contour_correlation: None,
            min_stoi: None,
            min_pesq: None,
        }
    }
}
