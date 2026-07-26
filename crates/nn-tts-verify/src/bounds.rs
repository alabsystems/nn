// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Hard bounds for TTS audio verification.
//!
//! Each check returns a `HardBound` with pass/fail and the measured value.
//! If any hard bound fails, the audio output is definitively broken.

use crate::dsp;
use crate::error::{validate_finite, DspErrorKind, InvalidConfigKind, TtsVerifyError};

/// Result of a single hard bound check.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HardBound {
    /// Human-readable name of the check.
    pub name: &'static str,
    /// Whether the check passed.
    pub passed: bool,
    /// Measured value from the audio.
    pub value: f64,
    /// Threshold used for comparison.
    pub threshold: f64,
}

/// Check that audio is not silence (RMS above minimum threshold).
///
/// A silent output indicates a catastrophic model failure (zero activations,
/// dead neurons, or empty decoder output).
pub fn check_non_silence(samples: &[f32], min_rms: f64) -> HardBound {
    let value = dsp::rms(samples);
    HardBound {
        name: "non_silence",
        passed: value >= min_rms,
        value,
        threshold: min_rms,
    }
}

/// Check that no samples exceed the clipping threshold.
///
/// Clipping produces audible distortion. Well-formed PCM should stay within
/// [-1.0, 1.0] (or the specified `max_amplitude`).
pub fn check_no_clipping(samples: &[f32], max_amplitude: f64) -> HardBound {
    let peak =
        crate::stats::fold_max_propagate_nan(samples.iter().map(|&x| f64::from(x).abs()), 0.0_f64);
    HardBound {
        name: "no_clipping",
        passed: peak <= max_amplitude,
        value: peak,
        threshold: max_amplitude,
    }
}

/// Check that DC offset is within bounds.
///
/// Large DC offset indicates a bias bug in the vocoder or normalization failure.
pub fn check_no_dc_offset(samples: &[f32], max_offset: f64) -> HardBound {
    let value = dsp::dc_offset(samples).abs();
    HardBound {
        name: "no_dc_offset",
        passed: value <= max_offset,
        value,
        threshold: max_offset,
    }
}

/// Check for clicks (large sample-to-sample jumps).
///
/// Clicks indicate discontinuities from iSTFT overlap-add errors,
/// buffer boundary misalignment, or decoder glitches.
pub fn check_no_clicks(samples: &[f32], max_diff: f64) -> HardBound {
    let value = dsp::max_sample_diff(samples);
    HardBound {
        name: "no_clicks",
        passed: value <= max_diff,
        value,
        threshold: max_diff,
    }
}

/// Check that audio duration is within expected range.
///
/// Too short = truncated output. Too long = infinite decoder loop.
/// Citation: empirical bounds from TTS deployment.
pub fn check_duration(samples: &[f32], sample_rate: u32, min_sec: f64, max_sec: f64) -> HardBound {
    let duration = if sample_rate > 0 {
        samples.len() as f64 / f64::from(sample_rate)
    } else {
        0.0
    };
    HardBound {
        name: "duration",
        passed: duration >= min_sec && duration <= max_sec,
        value: duration,
        threshold: min_sec, // Reports the lower bound as threshold.
    }
}

/// Configuration for spectral coverage check.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpectralCoverageConfig {
    /// Number of frequency bands to check.
    pub n_bands: usize,
    /// Minimum energy per band in dB.
    pub min_energy_db: f64,
    /// Minimum fraction of bands that must exceed the threshold.
    pub min_coverage: f64,
}

impl SpectralCoverageConfig {
    /// Validate that all f64 fields are finite.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        validate_finite(self.min_energy_db, "min_energy_db")?;
        validate_finite(self.min_coverage, "min_coverage")?;
        if self.n_bands == 0 {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::NonPositive { param: "n_bands" },
            ));
        }
        Ok(())
    }
}

impl Default for SpectralCoverageConfig {
    fn default() -> Self {
        Self {
            n_bands: 8,
            min_energy_db: -60.0,
            min_coverage: 0.5,
        }
    }
}

/// Check that the audio has energy across the frequency spectrum.
///
/// Speech should have energy in multiple frequency bands. Audio with energy
/// concentrated in a single band suggests a stuck oscillator or monotone output.
pub fn check_spectral_coverage(
    samples: &[f32],
    sample_rate: u32,
    config: &SpectralCoverageConfig,
) -> Result<HardBound, TtsVerifyError> {
    if config.n_bands == 0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "n_bands must be > 0",
        }));
    }
    let band_energy = dsp::spectral_band_energy(samples, sample_rate, config.n_bands)?;
    let bands_above = band_energy
        .iter()
        .filter(|&&e| e >= config.min_energy_db)
        .count();
    let coverage = bands_above as f64 / config.n_bands as f64;
    Ok(HardBound {
        name: "spectral_coverage",
        passed: coverage >= config.min_coverage,
        value: coverage,
        threshold: config.min_coverage,
    })
}

/// Check for energy spikes in the tail of the utterance.
///
/// Computes the RMS energy over the last `tail_ms` of audio and compares
/// it to the RMS energy of the body (from 20% to 80% of the audio).
/// If `tail_rms / body_rms > max_ratio`, the bound fails — indicating
/// an energy spike at the utterance boundary (truncation artifact, decoder
/// blowup, or overlap-add error at the final frame).
pub fn check_tail_energy(
    samples: &[f32],
    sample_rate: u32,
    tail_ms: f64,
    body_ms: f64,
    max_ratio: f64,
) -> HardBound {
    let n = samples.len();
    let sr = f64::from(sample_rate);

    // Compute tail region: last `tail_ms` milliseconds.
    let tail_samples = ((tail_ms / 1000.0) * sr).ceil() as usize;
    let tail_start = n.saturating_sub(tail_samples);
    let tail_rms = if tail_start < n {
        dsp::rms(&samples[tail_start..])
    } else {
        0.0
    };

    // Compute body region: from 20% to 80% of audio, capped at body_ms.
    let body_start_idx = n / 5; // 20%
    let body_end_max = 4 * n / 5; // 80%
    let body_max_samples = ((body_ms / 1000.0) * sr).ceil() as usize;
    let body_end_idx = body_end_max.min(body_start_idx + body_max_samples);
    let body_rms = if body_start_idx < body_end_idx {
        dsp::rms(&samples[body_start_idx..body_end_idx])
    } else {
        0.0
    };

    // Ratio: tail_rms / body_rms. If body is silent, ratio is 0 (no spike).
    let ratio = if body_rms > 1e-12 {
        tail_rms / body_rms
    } else {
        0.0
    };

    HardBound {
        name: "tail_energy",
        passed: ratio <= max_ratio,
        value: ratio,
        threshold: max_ratio,
    }
}

/// Check that audio energy above Nyquist is negligible.
///
/// Energy near Nyquist (sample_rate / 2) indicates aliasing from incorrect
/// upsampling or insufficient anti-aliasing filtering.
pub fn check_nyquist(samples: &[f32], sample_rate: u32) -> Result<HardBound, TtsVerifyError> {
    // Check top 5% of spectrum for excessive energy relative to total.
    let band_energy = dsp::spectral_band_energy(samples, sample_rate, 20)?;
    let total_energy: f64 = band_energy.iter().map(|&e| 10.0_f64.powf(e / 10.0)).sum();
    let nyquist_energy: f64 = band_energy[19]; // Last band (top 5%).
    let nyquist_linear = 10.0_f64.powf(nyquist_energy / 10.0);
    let ratio = if total_energy > 0.0 {
        nyquist_linear / total_energy
    } else {
        0.0
    };
    // Nyquist band should have < 10% of total energy for clean speech.
    let threshold = 0.1;
    Ok(HardBound {
        name: "nyquist",
        passed: ratio <= threshold,
        value: ratio,
        threshold,
    })
}

#[cfg(test)]
#[path = "bounds_tests.rs"]
mod tests;
