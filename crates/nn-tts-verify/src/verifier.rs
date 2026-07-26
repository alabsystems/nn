// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`TtsVerifier`] builder and runner — extracted from `lib.rs` for file-size
//! compliance (#1804).

use crate::bounds;
use crate::certificate::Certificate;
use crate::config::{HardBoundsConfig, QualityConfig, RejectionPolicy};
use crate::deterministic;
use crate::error::TtsVerifyError;
use crate::quality;
use crate::{compute_f0_contour_correlation, compute_multi_res_stft, compute_pesq, compute_stoi};

/// TTS audio quality verifier.
///
/// Runs hard bounds and (optionally) quality metrics on PCM audio.
/// Construct via [`TtsVerifier::builder()`].
#[derive(Debug, Clone)]
pub struct TtsVerifier {
    pub(crate) sample_rate: u32,
    pub(crate) hard_bounds: HardBoundsConfig,
    pub(crate) quality: Option<QualityConfig>,
}

/// Builder for [`TtsVerifier`].
#[derive(Debug, Clone)]
pub struct TtsVerifierBuilder {
    sample_rate: u32,
    hard_bounds: HardBoundsConfig,
    quality: Option<QualityConfig>,
}

impl TtsVerifier {
    /// Create a builder with default configuration.
    #[must_use]
    pub fn builder() -> TtsVerifierBuilder {
        TtsVerifierBuilder {
            sample_rate: 24000,
            hard_bounds: HardBoundsConfig::default(),
            quality: None,
        }
    }

    /// Verify standalone audio output (hard bounds only).
    ///
    /// Does not require a reference signal. Checks non-silence, clipping,
    /// DC offset, clicks, duration, spectral coverage, and Nyquist.
    ///
    /// Uses effective thresholds (per-check overrides take precedence over
    /// defaults) and applies the configured [`RejectionPolicy`]:
    /// - `Reject` (default): `overall_passed` reflects actual hard bound results.
    /// - `Warn`: `overall_passed` ignores hard bound failures (quality still applies).
    /// - `Remediate`: same as `Warn` (remediation is reserved for future use).
    pub fn verify(&self, samples: &[f32]) -> Result<Certificate, TtsVerifyError> {
        self.validate_input(samples)?;
        let hb = &self.hard_bounds;

        let hard_results = vec![
            bounds::check_non_silence(samples, hb.effective_min_rms()),
            bounds::check_no_clipping(samples, hb.effective_max_amplitude()),
            bounds::check_no_dc_offset(samples, hb.effective_max_dc_offset()),
            bounds::check_no_clicks(samples, hb.effective_max_click_diff()),
            bounds::check_duration(
                samples,
                self.sample_rate,
                hb.effective_min_duration_sec(),
                hb.effective_max_duration_sec(),
            ),
            bounds::check_tail_energy(
                samples,
                self.sample_rate,
                hb.tail_ms,
                hb.body_ms,
                hb.effective_max_tail_energy_ratio(),
            ),
            bounds::check_spectral_coverage(samples, self.sample_rate, &hb.spectral)?,
            bounds::check_nyquist(samples, self.sample_rate)?,
        ];

        // Quality metrics without reference: HNR, F0, spectral tilt (no MCD).
        let quality_results = if let Some(ref qc) = self.quality {
            let mut metrics = Vec::with_capacity(3);
            metrics.push(quality::compute_hnr(
                samples,
                self.sample_rate,
                qc.min_hnr_db,
            )?);
            let f0 = quality::extract_f0(samples, self.sample_rate)?;
            metrics.push(quality::check_f0_range(&f0, qc.f0_range.0, qc.f0_range.1));
            metrics.push(quality::compute_spectral_tilt(
                samples,
                self.sample_rate,
                qc.spectral_tilt,
            )?);
            metrics
        } else {
            Vec::new()
        };

        let hard_pass = hard_results.iter().all(|b| b.passed);
        let quality_pass = quality_results.iter().all(|m| m.passed);

        // Apply rejection policy: Warn/Remediate treat hard bound failures as
        // non-blocking (individual HardBound.passed fields still reflect truth).
        let effective_hard_pass = match hb.rejection_policy {
            RejectionPolicy::Reject => hard_pass,
            RejectionPolicy::Warn | RejectionPolicy::Remediate => true,
        };

        let cert = Certificate {
            hard_bounds: hard_results,
            quality_metrics: quality_results,
            phoneme_results: None,
            overall_passed: effective_hard_pass && quality_pass,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: Some(deterministic::pcm_sha256(samples)),
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        // Reject policy: return Err when verification fails.
        if hb.rejection_policy == RejectionPolicy::Reject && !cert.overall_passed {
            return Err(TtsVerifyError::VerificationRejected {
                cert: Box::new(cert),
            });
        }

        Ok(cert)
    }

    /// Verify audio with a reference signal (hard bounds + quality metrics including MCD).
    ///
    /// Runs all hard bound checks plus MCD, HNR, F0 range, and spectral tilt.
    /// Uses effective thresholds and applies the configured [`RejectionPolicy`].
    pub fn verify_with_reference(
        &self,
        candidate: &[f32],
        reference: &[f32],
    ) -> Result<Certificate, TtsVerifyError> {
        self.validate_input(candidate)?;
        if reference.is_empty() {
            return Err(TtsVerifyError::EmptyInput);
        }
        if candidate.len() != reference.len() {
            return Err(TtsVerifyError::LengthMismatch {
                candidate: candidate.len(),
                reference: reference.len(),
            });
        }
        let hb = &self.hard_bounds;

        // Hard bounds on the candidate (using effective thresholds).
        let hard_results = vec![
            bounds::check_non_silence(candidate, hb.effective_min_rms()),
            bounds::check_no_clipping(candidate, hb.effective_max_amplitude()),
            bounds::check_no_dc_offset(candidate, hb.effective_max_dc_offset()),
            bounds::check_no_clicks(candidate, hb.effective_max_click_diff()),
            bounds::check_duration(
                candidate,
                self.sample_rate,
                hb.effective_min_duration_sec(),
                hb.effective_max_duration_sec(),
            ),
            bounds::check_tail_energy(
                candidate,
                self.sample_rate,
                hb.tail_ms,
                hb.body_ms,
                hb.effective_max_tail_energy_ratio(),
            ),
            bounds::check_spectral_coverage(candidate, self.sample_rate, &hb.spectral)?,
            bounds::check_nyquist(candidate, self.sample_rate)?,
        ];

        // Quality metrics with reference.
        let qc = self.quality.clone().unwrap_or_default();
        let mut quality_results = Vec::with_capacity(10);
        quality_results.push(quality::compute_mcd(
            candidate,
            reference,
            self.sample_rate,
            qc.max_mcd_db,
        )?);
        quality_results.push(quality::compute_hnr(
            candidate,
            self.sample_rate,
            qc.min_hnr_db,
        )?);
        let f0 = quality::extract_f0(candidate, self.sample_rate)?;
        quality_results.push(quality::check_f0_range(&f0, qc.f0_range.0, qc.f0_range.1));
        quality_results.push(quality::compute_spectral_tilt(
            candidate,
            self.sample_rate,
            qc.spectral_tilt,
        )?);
        quality_results.push(quality::compute_cosine_similarity(
            candidate,
            reference,
            qc.min_cosine_similarity,
        )?);
        quality_results.push(quality::compute_snr(candidate, reference, qc.min_snr_db)?);
        quality_results.push(quality::compute_sdr(candidate, reference, qc.min_sdr_db)?);

        // Multi-resolution STFT loss (if configured).
        if let Some(ref stft_config) = qc.multi_res_stft {
            quality_results.push(compute_multi_res_stft(
                candidate,
                reference,
                self.sample_rate,
                stft_config,
            )?);
        }

        // F0 contour correlation (if configured).
        if let Some(min_corr) = qc.min_f0_contour_correlation {
            quality_results.push(compute_f0_contour_correlation(
                candidate,
                reference,
                self.sample_rate,
                min_corr,
            )?);
        }

        // STOI (if configured).
        if let Some(min_stoi) = qc.min_stoi {
            quality_results.push(compute_stoi(
                reference,
                candidate,
                self.sample_rate,
                min_stoi,
            )?);
        }

        // PESQ (if configured).
        if let Some(min_pesq) = qc.min_pesq {
            quality_results.push(compute_pesq(
                reference,
                candidate,
                self.sample_rate,
                min_pesq,
            )?);
        }

        let hard_pass = hard_results.iter().all(|b| b.passed);
        let quality_pass = quality_results.iter().all(|m| m.passed);

        // Apply rejection policy: Warn/Remediate treat hard bound failures as
        // non-blocking (individual HardBound.passed fields still reflect truth).
        let effective_hard_pass = match hb.rejection_policy {
            RejectionPolicy::Reject => hard_pass,
            RejectionPolicy::Warn | RejectionPolicy::Remediate => true,
        };

        let cert = Certificate {
            hard_bounds: hard_results,
            quality_metrics: quality_results,
            phoneme_results: None,
            overall_passed: effective_hard_pass && quality_pass,
            crown_evidence: None,
            junction_summary: None,
            deterministic_hash: Some(deterministic::pcm_sha256(candidate)),
            #[cfg(feature = "ny")]
            dead_neuron_eq_proof: None,
        };

        // Reject policy: return Err when verification fails.
        if hb.rejection_policy == RejectionPolicy::Reject && !cert.overall_passed {
            return Err(TtsVerifyError::VerificationRejected {
                cert: Box::new(cert),
            });
        }

        Ok(cert)
    }

    fn validate_input(&self, samples: &[f32]) -> Result<(), TtsVerifyError> {
        if samples.is_empty() {
            return Err(TtsVerifyError::EmptyInput);
        }
        if self.sample_rate == 0 {
            return Err(TtsVerifyError::InvalidSampleRate(0));
        }
        let non_finite = samples.iter().filter(|x| !x.is_finite()).count();
        if non_finite > 0 {
            return Err(TtsVerifyError::NonFiniteInput { count: non_finite });
        }
        Ok(())
    }
}

impl TtsVerifierBuilder {
    /// Set the audio sample rate in Hz. Default: 24000.
    #[must_use]
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set custom hard bounds configuration.
    #[must_use]
    pub fn hard_bounds(mut self, config: HardBoundsConfig) -> Self {
        self.hard_bounds = config;
        self
    }

    /// Enable quality metrics with the given configuration.
    #[must_use]
    pub fn quality(mut self, config: QualityConfig) -> Self {
        self.quality = Some(config);
        self
    }

    /// Enable quality metrics with default thresholds.
    #[must_use]
    pub fn with_quality(mut self) -> Self {
        self.quality = Some(QualityConfig::default());
        self
    }

    /// Set the maximum sample-to-sample difference threshold for the click
    /// detection hard bound. Default: 0.5.
    ///
    /// Kokoro-82M speech output can produce click_diff values of 0.59-0.81
    /// due to normal plosive energy, not click artifacts. Callers can raise
    /// this threshold to avoid false rejections.
    ///
    /// This is a convenience setter that modifies the builder's internal
    /// [`HardBoundsConfig::max_click_diff`] field. It can also be set via
    /// [`HardBoundsConfig`] directly or via [`CheckOverrides`].
    #[must_use]
    pub fn max_click_diff(mut self, threshold: f64) -> Self {
        self.hard_bounds.max_click_diff = threshold;
        self
    }

    /// Build the verifier.
    ///
    /// Returns `Err` if the sample rate is zero.
    pub fn build(self) -> Result<TtsVerifier, TtsVerifyError> {
        if self.sample_rate == 0 {
            return Err(TtsVerifyError::InvalidSampleRate(0));
        }
        Ok(TtsVerifier {
            sample_rate: self.sample_rate,
            hard_bounds: self.hard_bounds,
            quality: self.quality,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HardBoundsConfig, RejectionPolicy};

    /// Generate silent audio (all zeros) — fails the non-silence hard bound.
    fn silent_audio(sample_rate: u32, duration_sec: f64) -> Vec<f32> {
        vec![0.0; (f64::from(sample_rate) * duration_sec) as usize]
    }

    /// Generate valid audio — passes all hard bounds at default thresholds.
    ///
    /// Uses a rich harmonic series (fundamental + overtones) to satisfy the
    /// spectral coverage check, which requires energy across multiple bands.
    fn valid_audio(sample_rate: u32, duration_sec: f64) -> Vec<f32> {
        let len = (f64::from(sample_rate) * duration_sec) as usize;
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let pi2 = 2.0 * std::f32::consts::PI;
                // Rich harmonic series: fundamental 200 Hz + overtones up to Nyquist.
                // This provides energy across many spectral bands.
                let mut s = 0.0_f32;
                for k in 1..=50 {
                    let freq = 200.0 * k as f32;
                    if freq > sample_rate as f32 / 2.0 {
                        break;
                    }
                    s += (1.0 / k as f32) * (pi2 * freq * t).sin();
                }
                // Normalize to keep amplitude within [-0.9, 0.9].
                s * 0.15
            })
            .collect()
    }

    #[test]
    fn test_verify_reject_policy_returns_err_for_bad_audio() {
        let hb = HardBoundsConfig {
            rejection_policy: RejectionPolicy::Reject,
            ..HardBoundsConfig::default()
        };
        let verifier = TtsVerifier::builder()
            .sample_rate(24000)
            .hard_bounds(hb)
            .build()
            .expect("builder should succeed");

        // Silent audio fails the non-silence check.
        let samples = silent_audio(24000, 1.0);
        let result = verifier.verify(&samples);
        assert!(
            result.is_err(),
            "Reject policy must return Err for failing audio"
        );
        match result.unwrap_err() {
            TtsVerifyError::VerificationRejected { cert } => {
                assert!(!cert.overall_passed, "cert.overall_passed must be false");
                assert!(
                    !cert.passes_hard_bounds(),
                    "at least one hard bound must fail"
                );
            }
            other => panic!("expected VerificationRejected, got: {other:?}"),
        }
    }

    #[test]
    fn test_verify_reject_policy_returns_ok_for_good_audio() {
        let hb = HardBoundsConfig {
            rejection_policy: RejectionPolicy::Reject,
            ..HardBoundsConfig::default()
        };
        let verifier = TtsVerifier::builder()
            .sample_rate(24000)
            .hard_bounds(hb)
            .build()
            .expect("builder should succeed");

        let samples = valid_audio(24000, 1.0);
        let cert = verifier
            .verify(&samples)
            .expect("Reject policy must return Ok for passing audio");
        assert!(cert.overall_passed, "cert.overall_passed must be true");
    }

    #[test]
    fn test_verify_warn_policy_returns_ok_for_bad_audio() {
        let hb = HardBoundsConfig {
            rejection_policy: RejectionPolicy::Warn,
            ..HardBoundsConfig::default()
        };
        let verifier = TtsVerifier::builder()
            .sample_rate(24000)
            .hard_bounds(hb)
            .build()
            .expect("builder should succeed");

        // Silent audio fails the non-silence check, but Warn returns Ok.
        let samples = silent_audio(24000, 1.0);
        let cert = verifier
            .verify(&samples)
            .expect("Warn policy must return Ok even for failing audio");
        // Warn masks hard bound failures in overall_passed.
        assert!(
            cert.overall_passed,
            "Warn policy should mask hard bound failures"
        );
        // Individual hard bounds still reflect truth.
        assert!(
            !cert.passes_hard_bounds(),
            "individual hard bounds should still fail"
        );
    }

    #[test]
    fn test_builder_max_click_diff_default() {
        let verifier = TtsVerifier::builder()
            .sample_rate(24000)
            .build()
            .expect("builder should succeed");
        assert!(
            (verifier.hard_bounds.max_click_diff - 0.5).abs() < f64::EPSILON,
            "default max_click_diff should be 0.5"
        );
    }

    #[test]
    fn test_builder_max_click_diff_custom() {
        let verifier = TtsVerifier::builder()
            .sample_rate(24000)
            .max_click_diff(1.0)
            .build()
            .expect("builder should succeed");
        assert!(
            (verifier.hard_bounds.max_click_diff - 1.0).abs() < f64::EPSILON,
            "max_click_diff should be 1.0 after builder override"
        );
        assert!(
            (verifier.hard_bounds.effective_max_click_diff() - 1.0).abs() < f64::EPSILON,
            "effective threshold should also be 1.0"
        );
    }

    #[test]
    fn test_builder_max_click_diff_passes_audio_with_plosives() {
        // Audio with a sharp transient that exceeds 0.5 but is under 1.0.
        let mut samples = valid_audio(24000, 1.0);
        // Insert a sharp transient: adjacent samples with diff ~0.7.
        let mid = samples.len() / 2;
        samples[mid] = 0.0;
        samples[mid + 1] = 0.7;

        // Default threshold (0.5) should reject (default policy is Reject,
        // so verify() returns Err(VerificationRejected) containing the cert).
        let verifier_strict = TtsVerifier::builder()
            .sample_rate(24000)
            .build()
            .expect("builder should succeed");
        let result_strict = verifier_strict.verify(&samples);
        let cert_strict = match result_strict {
            Err(TtsVerifyError::VerificationRejected { cert }) => cert,
            other => panic!(
                "expected VerificationRejected for diff ~0.7 with threshold 0.5, got: {other:?}"
            ),
        };
        let click_bound = cert_strict
            .hard_bounds
            .iter()
            .find(|b| b.name == "no_clicks")
            .expect("no_clicks bound should exist");
        assert!(
            !click_bound.passed,
            "default threshold 0.5 should reject diff ~0.7"
        );

        // Relaxed threshold (1.0) should pass.
        let verifier_relaxed = TtsVerifier::builder()
            .sample_rate(24000)
            .max_click_diff(1.0)
            .build()
            .expect("builder should succeed");
        let cert_relaxed = verifier_relaxed
            .verify(&samples)
            .expect("relaxed verify should return Ok");
        let click_bound_relaxed = cert_relaxed
            .hard_bounds
            .iter()
            .find(|b| b.name == "no_clicks")
            .expect("no_clicks bound should exist");
        assert!(
            click_bound_relaxed.passed,
            "relaxed threshold 1.0 should pass diff ~0.7"
        );
    }

    #[test]
    fn test_verify_remediate_policy_returns_ok_for_bad_audio() {
        let hb = HardBoundsConfig {
            rejection_policy: RejectionPolicy::Remediate,
            ..HardBoundsConfig::default()
        };
        let verifier = TtsVerifier::builder()
            .sample_rate(24000)
            .hard_bounds(hb)
            .build()
            .expect("builder should succeed");

        let samples = silent_audio(24000, 1.0);
        let cert = verifier
            .verify(&samples)
            .expect("Remediate policy must return Ok (placeholder, same as Warn)");
        assert!(
            cert.overall_passed,
            "Remediate should mask hard bound failures like Warn"
        );
    }
}
