// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Audio-domain empirical disentanglement measurement.
//!
//! Complements the CROWN-based formal disentanglement in [`crate::disentanglement`]
//! with empirical measurements on synthesized audio pairs. Given a baseline
//! utterance and a perturbed utterance that differs in one control dimension,
//! computes how much each acoustic property changed.
//!
//! If changing `prosody_style` shifts F0 but maintains high spectral similarity
//! (low MCD), that is empirical evidence of disentanglement in the audio domain.
//!
//! Part of #1738: Compositional Verification of Prosody Controls.

use crate::error::{validate_finite, validate_finite_positive, TtsVerifyError};
use crate::f0_contour;
use crate::quality;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Result of audio-domain disentanglement measurement between a baseline
/// and perturbed utterance pair.
#[derive(Debug, Clone)]
pub struct AudioDisentanglementResult {
    /// F0 contour Pearson correlation between baseline and perturbed.
    /// High (near 1.0) = F0 unchanged. Low (near 0.0) = F0 shifted.
    pub f0_correlation: f64,

    /// Mel Cepstral Distortion between baseline and perturbed (dB).
    /// Low (< 4 dB) = spectrally similar. High (> 8 dB) = spectrally different.
    pub mcd: f64,

    /// Duration ratio: perturbed_duration / baseline_duration.
    /// 1.0 = unchanged. < 1.0 = shorter. > 1.0 = longer.
    pub duration_ratio: f64,

    /// Waveform cosine similarity between baseline and perturbed.
    /// Placeholder for speaker embedding similarity (ECAPA-TDNN from #1736).
    /// Higher = more similar overall waveform shape.
    pub waveform_similarity: f64,
}

/// Summary of which acoustic properties changed and which were preserved.
#[derive(Debug, Clone)]
pub struct DisentanglementEvidence {
    /// The audio measurement result.
    pub result: AudioDisentanglementResult,

    /// Whether F0 was preserved (correlation > threshold).
    pub f0_preserved: bool,

    /// Whether spectral envelope was preserved (MCD < threshold).
    pub spectral_preserved: bool,

    /// Whether duration was preserved (ratio within tolerance).
    pub duration_preserved: bool,

    /// Whether overall waveform shape was preserved (similarity > threshold).
    pub waveform_preserved: bool,
}

/// Thresholds for disentanglement evidence classification.
#[derive(Debug, Clone)]
pub struct DisentanglementThresholds {
    /// Minimum F0 correlation to consider F0 "preserved". Default: 0.85.
    pub f0_correlation_min: f64,
    /// Maximum MCD (dB) to consider spectral envelope "preserved". Default: 6.0.
    pub mcd_max: f64,
    /// Maximum duration ratio deviation from 1.0. Default: 0.1 (±10%).
    pub duration_ratio_tolerance: f64,
    /// Minimum waveform cosine similarity to consider "preserved". Default: 0.8.
    pub waveform_similarity_min: f64,
}

impl DisentanglementThresholds {
    /// Validate that all f64 fields are finite and within sensible ranges.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        validate_finite(self.f0_correlation_min, "f0_correlation_min")?;
        validate_finite_positive(self.mcd_max, "mcd_max")?;
        validate_finite_positive(self.duration_ratio_tolerance, "duration_ratio_tolerance")?;
        validate_finite(self.waveform_similarity_min, "waveform_similarity_min")?;
        Ok(())
    }
}

impl Default for DisentanglementThresholds {
    fn default() -> Self {
        Self {
            f0_correlation_min: 0.85,
            mcd_max: 6.0,
            duration_ratio_tolerance: 0.1,
            waveform_similarity_min: 0.8,
        }
    }
}

// ---------------------------------------------------------------------------
// Core measurement
// ---------------------------------------------------------------------------

/// Measure audio-domain disentanglement between a baseline and perturbed
/// utterance that differ in one control dimension.
///
/// Computes four acoustic properties:
/// - **F0 correlation**: Pearson correlation of F0 contours (via YIN)
/// - **MCD**: Mel Cepstral Distortion (spectral distance)
/// - **Duration ratio**: length ratio (perturbed / baseline)
/// - **Waveform similarity**: cosine similarity of raw waveforms
///
/// The waveform similarity is a placeholder for speaker embedding cosine
/// similarity. When ECAPA-TDNN (#1736) is available, this should be replaced
/// with embedding-space similarity.
///
/// # Errors
///
/// Returns `TtsVerifyError` if inputs are empty, sample rate is zero, or
/// if underlying DSP computations fail (e.g., audio too short for MFCC).
pub fn measure_audio_disentanglement(
    baseline: &[f32],
    perturbed: &[f32],
    sample_rate: u32,
) -> Result<AudioDisentanglementResult, TtsVerifyError> {
    if baseline.is_empty() || perturbed.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(sample_rate));
    }

    // Duration ratio (does not require equal lengths).
    let duration_ratio = perturbed.len() as f64 / baseline.len() as f64;

    // For F0 and MCD, we need to handle potentially different lengths.
    // Truncate to the shorter signal for pairwise comparison.
    let min_len = baseline.len().min(perturbed.len());
    let base_trunc = &baseline[..min_len];
    let pert_trunc = &perturbed[..min_len];

    // F0 contour correlation via YIN.
    let base_f0 = quality::extract_f0(base_trunc, sample_rate)?;
    let pert_f0 = quality::extract_f0(pert_trunc, sample_rate)?;
    let f0_correlation = f0_contour::f0_pearson_correlation(&base_f0, &pert_f0)?;

    // MCD between truncated signals.
    let mcd_result = quality::compute_mcd(base_trunc, pert_trunc, sample_rate, f64::INFINITY)?;
    let mcd = mcd_result.value;

    // Waveform cosine similarity (placeholder for speaker embedding).
    let waveform_result =
        quality::compute_cosine_similarity(base_trunc, pert_trunc, f64::NEG_INFINITY)?;
    let waveform_similarity = waveform_result.value;

    Ok(AudioDisentanglementResult {
        f0_correlation,
        mcd,
        duration_ratio,
        waveform_similarity,
    })
}

/// Classify which acoustic properties were preserved or changed,
/// producing disentanglement evidence.
///
/// Given a measurement result and thresholds, determines which properties
/// were "preserved" (within threshold) and which "changed" (outside threshold).
///
/// Disentanglement evidence: if changing one control dimension (e.g., pitch)
/// shifts one property (F0) but preserves others (spectral, duration, waveform),
/// then that control is disentangled from those other properties.
pub fn classify_disentanglement(
    result: AudioDisentanglementResult,
    thresholds: &DisentanglementThresholds,
) -> DisentanglementEvidence {
    let f0_preserved = result.f0_correlation >= thresholds.f0_correlation_min;
    let spectral_preserved = result.mcd <= thresholds.mcd_max;
    let duration_preserved =
        (result.duration_ratio - 1.0).abs() <= thresholds.duration_ratio_tolerance;
    let waveform_preserved = result.waveform_similarity >= thresholds.waveform_similarity_min;

    DisentanglementEvidence {
        result,
        f0_preserved,
        spectral_preserved,
        duration_preserved,
        waveform_preserved,
    }
}

#[cfg(test)]
#[path = "audio_disentanglement_tests.rs"]
mod tests;
